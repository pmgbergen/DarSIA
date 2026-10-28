"""Serializable optical phase heads shared by FluidFlower experiment series.

The aqueous branch remains the run-native BTB/Titration signal.  This module
provides a second, parameter-independent image signal for free gas.  Its output
is an optical probability, not a saturation; the calibration layer maps it to
``Sg`` and couples aqueous mass through the remaining pore volume ``1 - Sg``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darsia.fluidflower.phase_features import (
    OpticalFeatureContract,
    optical_feature_matrix,
    optical_feature_samples,
    prepare_optical_feature_workspace,
)

MODEL_SCHEMA = 1
MODEL_ALGORITHM = "ffac-dual-phase-logistic-v1"
# Schema 1 remains loadable so completed campaigns retain an exact, inspectable
# model contract. Schema 2 makes the generic/chemistry split explicit.
TRANSFERABLE_MODEL_SCHEMA = 2
TRANSFERABLE_MODEL_ALGORITHM = "ffac-transferable-phase-logistic-v2"
FEATURE_NAMES = (
    "delta_r",
    "delta_g",
    "delta_b",
    "delta_intensity",
    "delta_red_fraction",
    "delta_green_fraction",
    "delta_blue_fraction",
    "acidification",
    "delta_acidification",
    "texture_highpass",
    "rgb_change_norm",
)

_DARSIA_CRITICAL_SOURCES = {
    "darsia_heterogeneous_color_to_mass_analysis": (
        "presets/workflows/heterogeneous_color_to_mass_analysis.py"
    ),
    "darsia_flash": "multiphase/flash.py",
    "darsia_rig": "presets/workflows/rig.py",
    "darsia_colorcorrection": "corrections/color/colorcorrection.py",
    "darsia_imread": "image/imread.py",
    "darsia_color_path_interpolation": ("signals/models/color_path_interpolation.py"),
    "darsia_calibration_config": "presets/workflows/config/calibration.py",
}


def _source_digest(source: Path) -> str:
    """Hash Python source independently of platform line endings."""

    return hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _source_tree_digest(package_root: Path) -> str:
    """Hash the complete importable DarSIA Python source tree."""

    digest = hashlib.sha256()
    sources = sorted(
        path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts
    )
    if not sources:
        raise FileNotFoundError(f"No DarSIA Python sources found below {package_root}")
    for source in sources:
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        content = source.read_bytes().replace(b"\r\n", b"\n")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def darsia_runtime_fingerprints() -> dict[str, str]:
    """Hash DarSIA sources that define dual-signal preprocessing and phase use."""

    import darsia

    package_root = Path(darsia.__file__).resolve().parent
    fingerprints: dict[str, str] = {}
    for name, relative_path in _DARSIA_CRITICAL_SOURCES.items():
        source = package_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Required DarSIA source is missing: {source}")
        fingerprints[name] = _source_digest(source)
    fingerprints["darsia_source_tree"] = _source_tree_digest(package_root)
    return fingerprints


def _as_rgb_float(image: Any) -> np.ndarray:
    from darsia.fluidflower.phase_features import _as_rgb_float as _canonical_as_rgb_float
    return _canonical_as_rgb_float(image)


def _acidification(rgb: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    yellow = np.float32(0.5) * (rgb[..., 0] + rgb[..., 1])
    blue = rgb[..., 2]
    return (yellow - blue) / (yellow + blue + np.float32(epsilon))


def _feature_workspace(
    image_rgb: Any,
    baseline_rgb: Any,
    *,
    texture_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = _as_rgb_float(image_rgb)
    baseline = _as_rgb_float(baseline_rgb)
    if image.shape != baseline.shape:
        raise ValueError(
            f"Image/baseline shape mismatch: {image.shape} != {baseline.shape}"
        )
    gray = np.mean(image, axis=2, dtype=np.float32)
    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(0.1, float(texture_sigma)),
        sigmaY=max(0.1, float(texture_sigma)),
    )
    return image, baseline, gray, smooth


def _feature_matrix(
    image: np.ndarray,
    baseline: np.ndarray,
    gray: np.ndarray,
    smooth: np.ndarray,
    selection: Any,
) -> np.ndarray:
    current = np.asarray(image[selection], dtype=np.float32)
    reference = np.asarray(baseline[selection], dtype=np.float32)
    delta = current - reference
    current_sum = np.sum(current, axis=-1, keepdims=True) + np.float32(1e-4)
    reference_sum = np.sum(reference, axis=-1, keepdims=True) + np.float32(1e-4)
    chroma_delta = current / current_sum - reference / reference_sum
    acid = _acidification(current)
    acid_reference = _acidification(reference)
    intensity_delta = np.mean(delta, axis=-1, dtype=np.float32)
    texture = np.abs(
        np.asarray(gray[selection], dtype=np.float32)
        - np.asarray(smooth[selection], dtype=np.float32)
    )
    change_norm = np.linalg.norm(delta, axis=-1)
    features = np.column_stack(
        (
            delta.reshape(-1, 3),
            intensity_delta.reshape(-1),
            chroma_delta.reshape(-1, 3),
            acid.reshape(-1),
            (acid - acid_reference).reshape(-1),
            texture.reshape(-1),
            change_norm.reshape(-1),
        )
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature contract mismatch: {features.shape[1]} != {len(FEATURE_NAMES)}"
        )
    return np.asarray(features, dtype=np.float32)


def feature_samples(
    image_rgb: Any,
    baseline_rgb: Any,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    texture_sigma: float = 2.0,
    previous_rgb: Any | None = None,
    clearwater_rgb: Any | None = None,
    feature_contract: OpticalFeatureContract | None = None,
) -> np.ndarray:
    """Extract feature rows at selected native-resolution pixels."""

    if feature_contract is not None:
        return optical_feature_samples(
            image_rgb,
            baseline_rgb,
            rows,
            columns,
            previous_rgb=previous_rgb,
            clearwater_rgb=clearwater_rgb,
            texture_sigma=texture_sigma,
            contract=feature_contract,
        )

    image, baseline, gray, smooth = _feature_workspace(
        image_rgb,
        baseline_rgb,
        texture_sigma=texture_sigma,
    )
    row_index = np.asarray(rows, dtype=np.intp)
    column_index = np.asarray(columns, dtype=np.intp)
    if row_index.shape != column_index.shape:
        raise ValueError("Row and column index arrays must have the same shape")
    if row_index.size and (
        np.min(row_index) < 0
        or np.max(row_index) >= image.shape[0]
        or np.min(column_index) < 0
        or np.max(column_index) >= image.shape[1]
    ):
        raise IndexError("Feature sample coordinates fall outside the image")
    return _feature_matrix(
        image,
        baseline,
        gray,
        smooth,
        (row_index, column_index),
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "model_sha256"}
    encoded = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DualPhaseSignalModel:
    """Regularized binary model returning optical free-gas probability."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    texture_sigma: float = 2.0
    training_provenance: Mapping[str, Any] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_contract: Mapping[str, Any] | None = None
    schema: int = MODEL_SCHEMA
    algorithm: str = MODEL_ALGORITHM

    def __post_init__(self) -> None:
        if self.schema == MODEL_SCHEMA:
            if self.algorithm != MODEL_ALGORITHM:
                raise ValueError(
                    f"Unsupported dual-signal algorithm {self.algorithm!r}"
                )
            if tuple(self.feature_names) != FEATURE_NAMES:
                raise ValueError("Legacy dual-signal feature-name contract differs")
            if self.feature_contract not in (None, {}):
                raise ValueError(
                    "Legacy dual-signal models cannot declare a v2 contract"
                )
        elif self.schema == TRANSFERABLE_MODEL_SCHEMA:
            if self.algorithm != TRANSFERABLE_MODEL_ALGORITHM:
                raise ValueError(
                    f"Unsupported transferable algorithm {self.algorithm!r}"
                )
            if not isinstance(self.feature_contract, Mapping):
                raise ValueError(
                    "Transferable models require an optical feature contract"
                )
            contract = OpticalFeatureContract.from_dict(self.feature_contract)
            if tuple(self.feature_names) != contract.feature_names:
                raise ValueError("Transferable model feature-name contract differs")
        else:
            raise ValueError(f"Unsupported dual-signal schema {self.schema}")
        expected = len(self.feature_names)
        lengths = (
            len(self.feature_mean),
            len(self.feature_scale),
            len(self.coefficients),
        )
        if any(length != expected for length in lengths):
            raise ValueError(f"Dual-signal feature lengths {lengths} != {expected}")
        if any(not np.isfinite(value) for value in self.feature_mean):
            raise ValueError("Dual-signal feature means must be finite")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.feature_scale):
            raise ValueError("Dual-signal feature scales must be finite and positive")
        if any(not np.isfinite(value) for value in self.coefficients):
            raise ValueError("Dual-signal coefficients must be finite")
        if not np.isfinite(self.intercept):
            raise ValueError("Dual-signal intercept must be finite")
        if not np.isfinite(self.texture_sigma) or self.texture_sigma <= 0.0:
            raise ValueError("Dual-signal texture sigma must be positive")

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """Predict gas probability for an already extracted feature matrix."""

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                "Prediction feature matrix differs from the model contract: "
                f"{matrix.shape} versus (*, {len(self.feature_names)})"
            )
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        scale = np.asarray(self.feature_scale, dtype=np.float32)
        coefficients = np.asarray(self.coefficients, dtype=np.float32)
        standardized = (matrix - mean) / scale
        logits = standardized @ coefficients + np.float32(self.intercept)
        logits = np.clip(logits, -30.0, 30.0)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)

    def predict_score(
        self,
        image_rgb: Any,
        baseline_rgb: Any,
        *,
        previous_rgb: Any | None = None,
        clearwater_rgb: Any | None = None,
        chunk_rows: int = 64,
    ) -> np.ndarray:
        """Predict a full-resolution optical score with bounded temporary memory."""

        if self.schema == TRANSFERABLE_MODEL_SCHEMA:
            contract = OpticalFeatureContract.from_dict(self.feature_contract or {})
            workspace = prepare_optical_feature_workspace(
                image_rgb,
                baseline_rgb,
                previous_rgb=previous_rgb,
                clearwater_rgb=clearwater_rgb,
                texture_sigma=self.texture_sigma,
                contract=contract,
            )
            image_shape = workspace.current.shape
        else:
            image, baseline, gray, smooth = _feature_workspace(
                image_rgb,
                baseline_rgb,
                texture_sigma=self.texture_sigma,
            )
            image_shape = image.shape
        output = np.empty(image_shape[:2], dtype=np.float32)
        step = max(1, int(chunk_rows))
        for start in range(0, image_shape[0], step):
            stop = min(image_shape[0], start + step)
            if self.schema == TRANSFERABLE_MODEL_SCHEMA:
                features = optical_feature_matrix(
                    workspace,
                    np.s_[start:stop, :],
                )
            else:
                features = _feature_matrix(
                    image,
                    baseline,
                    gray,
                    smooth,
                    np.s_[start:stop, :],
                )
            output[start:stop] = self.predict_features(features).reshape(
                stop - start,
                image_shape[1],
            )
        return output

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": int(self.schema),
            "algorithm": self.algorithm,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "texture_sigma": float(self.texture_sigma),
            "training_provenance": dict(self.training_provenance),
        }
        if self.feature_contract is not None:
            payload["feature_contract"] = dict(self.feature_contract)
        payload["model_sha256"] = _canonical_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DualPhaseSignalModel:
        schema = int(payload.get("schema", -1))
        algorithm = str(payload.get("algorithm"))
        feature_names = tuple(
            str(value) for value in payload.get("feature_names") or ()
        )
        feature_contract: Mapping[str, Any] | None = None
        if schema == MODEL_SCHEMA:
            if algorithm != MODEL_ALGORITHM:
                raise ValueError(f"Unsupported dual-signal algorithm {algorithm!r}")
            if feature_names != FEATURE_NAMES:
                raise ValueError("Dual-signal feature-name contract differs")
        elif schema == TRANSFERABLE_MODEL_SCHEMA:
            if algorithm != TRANSFERABLE_MODEL_ALGORITHM:
                raise ValueError(f"Unsupported transferable algorithm {algorithm!r}")
            raw_contract = payload.get("feature_contract")
            if not isinstance(raw_contract, Mapping):
                raise ValueError("Transferable model lacks an optical feature contract")
            contract = OpticalFeatureContract.from_dict(raw_contract)
            if feature_names != contract.feature_names:
                raise ValueError("Transferable model feature-name contract differs")
            feature_contract = dict(raw_contract)
        else:
            raise ValueError(f"Unsupported dual-signal schema {schema!r}")
        expected_hash = str(payload.get("model_sha256", ""))
        actual_hash = _canonical_hash(payload)
        if not expected_hash or expected_hash != actual_hash:
            raise ValueError("Dual-signal model hash mismatch")
        return cls(
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            texture_sigma=float(payload.get("texture_sigma", 2.0)),
            training_provenance=dict(payload.get("training_provenance") or {}),
            feature_names=feature_names,
            feature_contract=feature_contract,
            schema=schema,
            algorithm=algorithm,
        )

    def save(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: Path) -> DualPhaseSignalModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls.from_dict(payload)


def fit_dual_phase_signal_model(
    features: np.ndarray,
    gas_labels: Sequence[int | bool],
    *,
    sample_weight: Sequence[float] | None = None,
    class_weight: str | Mapping[int, float] | None = "balanced",
    regularization_c: float = 0.5,
    texture_sigma: float = 2.0,
    training_provenance: Mapping[str, Any] | None = None,
    feature_contract: OpticalFeatureContract | None = None,
) -> DualPhaseSignalModel:
    """Fit an interpretable gas-vs-non-gas model from reviewed pixels."""

    from sklearn.linear_model import LogisticRegression

    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(gas_labels, dtype=np.int8)
    feature_names = (
        FEATURE_NAMES if feature_contract is None else feature_contract.feature_names
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Training features must have shape (n, {len(feature_names)}), "
            f"got {matrix.shape}"
        )
    if labels.shape != (matrix.shape[0],):
        raise ValueError("Training labels do not match feature rows")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Training data must contain both gas and non-gas samples")
    finite = np.all(np.isfinite(matrix), axis=1)
    if not np.all(finite):
        matrix = matrix[finite]
        labels = labels[finite]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)[finite]
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.maximum(scale, 1e-6)
    standardized = (matrix - mean) / scale
    weights = (
        None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    classifier = LogisticRegression(
        C=max(1e-6, float(regularization_c)),
        class_weight=class_weight,
        max_iter=2000,
        solver="lbfgs",
        random_state=0,
    )
    classifier.fit(standardized, labels, sample_weight=weights)
    provenance = dict(training_provenance or {})
    if feature_contract is not None:
        provenance.setdefault("optical_feature_contract", feature_contract.to_dict())
    return DualPhaseSignalModel(
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        texture_sigma=float(texture_sigma),
        training_provenance=provenance,
        feature_names=tuple(feature_names),
        feature_contract=(
            None if feature_contract is None else feature_contract.to_dict()
        ),
        schema=(
            MODEL_SCHEMA if feature_contract is None else TRANSFERABLE_MODEL_SCHEMA
        ),
        algorithm=(
            MODEL_ALGORITHM
            if feature_contract is None
            else TRANSFERABLE_MODEL_ALGORITHM
        ),
    )
