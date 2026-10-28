"""Versioned optical features for transferable FluidFlower phase models.

The generic feature block deliberately contains no assumption about a blue to
yellow, green to red, or any other indicator-specific colour direction.  A
small chemistry adapter may append such features when they are justified by
the indicator system used in a campaign.

The baseline and dynamic change channels intentionally retain DarSIA's core
strength: detecting optical change relative to a registered reference.  The
classifier then learns whether that change is aqueous CO2, free gas, water, or
an artifact.  It does not infer concentration, depth, saturation, or mass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

FEATURE_CONTRACT_SCHEMA = 1
BASELINE_RELATIVE_FEATURE_VERSION = "baseline-relative-texture-dynamic-v3"
GENERIC_FEATURE_VERSION = "baseline-aware-bright-grain-v4"
RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION = "baseline-relative-bright-grain-v5"
INPUT_COLOR_SPACE = "radiometrically-corrected-rgb-0-1"
SPATIAL_CONTEXT_FRACTION = 0.0015
DARSIA_CHANGE_DETECTOR = "concentration-analysis-plain-background-difference-v1"
NO_CHANGE_DETECTOR = "none"
DARSIA_CHANGE_FEATURE_NAMES = (
    "delta_r",
    "delta_g",
    "delta_b",
    "rgb_change_norm",
    "local_rgb_change_mean",
    "local_rgb_change_std",
    "local_rgb_change_coherence",
    "dynamic_delta_r",
    "dynamic_delta_g",
    "dynamic_delta_b",
    "dynamic_rgb_change_norm",
    "local_dynamic_rgb_change_mean",
)


@dataclass(frozen=True)
class ChangeEvidence:
    """Registered optical-change arrays supplied to the phase classifier."""

    baseline_rgb_difference: np.ndarray
    baseline_change_norm: np.ndarray
    local_baseline_change_mean: np.ndarray
    local_baseline_change_std: np.ndarray
    dynamic_rgb_difference: np.ndarray
    dynamic_change_norm: np.ndarray
    local_dynamic_change_mean: np.ndarray


class ChangeEvidenceProvider(Protocol):
    """Provider contract for interchangeable DarSIA change detectors."""

    detector: str
    feature_names: tuple[str, ...]

    def evaluate(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        previous: np.ndarray,
        *,
        spatial_sigma: float,
    ) -> ChangeEvidence:
        """Return registered baseline and previous-frame change evidence."""


GENERIC_STATIC_FEATURE_NAMES_V2 = (
    "current_r",
    "current_g",
    "current_b",
    "delta_r",
    "delta_g",
    "delta_b",
    "current_intensity",
    "delta_intensity",
    "current_red_fraction",
    "current_green_fraction",
    "current_blue_fraction",
    "delta_red_fraction",
    "delta_green_fraction",
    "delta_blue_fraction",
    "log_r_over_g",
    "log_g_over_b",
    "log_r_over_b",
    "delta_log_r_over_g",
    "delta_log_g_over_b",
    "delta_log_r_over_b",
    "texture_highpass",
    "delta_texture_highpass",
    "rgb_change_norm",
    "local_rgb_change_mean",
    "local_rgb_change_std",
    "local_rgb_change_coherence",
)

GENERIC_DYNAMIC_FEATURE_NAMES_V2 = (
    "dynamic_available",
    "dynamic_delta_r",
    "dynamic_delta_g",
    "dynamic_delta_b",
    "dynamic_delta_intensity",
    "dynamic_texture_delta",
    "dynamic_rgb_change_norm",
    "local_dynamic_rgb_change_mean",
)

GENERIC_STATIC_FEATURE_NAMES_V1 = GENERIC_STATIC_FEATURE_NAMES_V2[:-3]
GENERIC_DYNAMIC_FEATURE_NAMES_V1 = GENERIC_DYNAMIC_FEATURE_NAMES_V2[:-1]
GENERIC_STATIC_FEATURE_NAMES_V3 = (
    "delta_r",
    "delta_g",
    "delta_b",
    "delta_intensity",
    "delta_red_fraction",
    "delta_green_fraction",
    "delta_blue_fraction",
    "delta_log_r_over_g",
    "delta_log_g_over_b",
    "delta_log_r_over_b",
    "delta_texture_highpass",
    "abs_delta_texture_highpass",
    "rgb_change_norm",
    "local_rgb_change_mean",
    "local_rgb_change_std",
    "local_rgb_change_coherence",
)
BASELINE_BRIGHT_GRAIN_FEATURE_NAMES = (
    "baseline_intensity",
    "baseline_signed_local_contrast",
    "current_signed_local_contrast",
    "delta_signed_local_contrast",
    "persistent_bright_contrast",
    "new_bright_contrast",
)
GENERIC_STATIC_FEATURE_NAMES = (
    *GENERIC_STATIC_FEATURE_NAMES_V3,
    *BASELINE_BRIGHT_GRAIN_FEATURE_NAMES,
)
RELATIVE_BRIGHT_GRAIN_FEATURE_NAMES = (
    "baseline_relative_local_contrast",
    "current_relative_local_contrast",
    "delta_relative_local_contrast",
    "persistent_relative_bright_contrast",
    "new_relative_bright_contrast",
    "relative_bright_contrast_agreement",
)
GENERIC_DYNAMIC_FEATURE_NAMES = GENERIC_DYNAMIC_FEATURE_NAMES_V2
GENERIC_FEATURE_NAMES_BY_VERSION: Mapping[
    str, tuple[tuple[str, ...], tuple[str, ...]]
] = {
    "corrected-rgb-texture-dynamic-v1": (
        GENERIC_STATIC_FEATURE_NAMES_V1,
        GENERIC_DYNAMIC_FEATURE_NAMES_V1,
    ),
    "corrected-rgb-texture-dynamic-v2": (
        GENERIC_STATIC_FEATURE_NAMES_V2,
        GENERIC_DYNAMIC_FEATURE_NAMES_V2,
    ),
    BASELINE_RELATIVE_FEATURE_VERSION: (
        GENERIC_STATIC_FEATURE_NAMES_V3,
        GENERIC_DYNAMIC_FEATURE_NAMES,
    ),
    GENERIC_FEATURE_VERSION: (
        GENERIC_STATIC_FEATURE_NAMES,
        GENERIC_DYNAMIC_FEATURE_NAMES,
    ),
    RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION: (
        (*GENERIC_STATIC_FEATURE_NAMES, *RELATIVE_BRIGHT_GRAIN_FEATURE_NAMES),
        GENERIC_DYNAMIC_FEATURE_NAMES,
    ),
}

CHEMISTRY_ADAPTER_FEATURES: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "btb-ratiometric-v1": (
        "btb_acidification",
        "delta_btb_acidification",
    ),
    "btb-ratiometric-relative-v2": ("delta_btb_acidification",),
}

REFERENCE_ADAPTER_FEATURES: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "clearwater-projection-v1": (
        "clearwater_reference_reliability",
        "clearwater_projection",
        "clearwater_orthogonal_distance",
        "clearwater_alignment",
        "clearwater_projection_score",
        "clearwater_distance_from_baseline",
        "clearwater_distance_to_endpoint",
        "clearwater_texture_difference",
    ),
}


def _as_rgb_float(image: Any) -> np.ndarray:
    array = np.asarray(getattr(image, "img", image))
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image, got {array.shape}.")
    array = np.asarray(array[..., :3], dtype=np.float32)
    if array.size and float(np.nanmax(array)) > 1.5:
        array = array / np.float32(255.0)
    return np.clip(array, 0.0, 1.0)


@dataclass(frozen=True)
class OpticalFeatureContract:
    """Serializable contract for one optical feature matrix."""

    chemistry_adapter: str = "none"
    reference_adapter: str = "none"
    include_dynamic: bool = True
    generic_version: str = GENERIC_FEATURE_VERSION
    input_color_space: str = INPUT_COLOR_SPACE
    change_detector: str = DARSIA_CHANGE_DETECTOR
    schema: int = FEATURE_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FEATURE_CONTRACT_SCHEMA:
            raise ValueError(f"Unsupported optical feature schema {self.schema}")
        if self.generic_version not in GENERIC_FEATURE_NAMES_BY_VERSION:
            raise ValueError(
                f"Unsupported generic feature version {self.generic_version!r}"
            )
        if self.input_color_space != INPUT_COLOR_SPACE:
            raise ValueError(
                f"Unsupported optical input colour space {self.input_color_space!r}"
            )
        if self.change_detector not in {
            DARSIA_CHANGE_DETECTOR,
            NO_CHANGE_DETECTOR,
        }:
            raise ValueError(
                f"Unsupported DarSIA change detector {self.change_detector!r}"
            )
        if self.chemistry_adapter not in CHEMISTRY_ADAPTER_FEATURES:
            raise ValueError(
                f"Unsupported chemistry adapter {self.chemistry_adapter!r}"
            )
        if self.reference_adapter not in REFERENCE_ADAPTER_FEATURES:
            raise ValueError(
                f"Unsupported optical reference adapter {self.reference_adapter!r}"
            )

    @property
    def full_feature_names(self) -> tuple[str, ...]:
        """Return all features before applying a change-evidence ablation."""

        static, dynamic_names = GENERIC_FEATURE_NAMES_BY_VERSION[self.generic_version]
        dynamic = dynamic_names if self.include_dynamic else ()
        chemistry = CHEMISTRY_ADAPTER_FEATURES[self.chemistry_adapter]
        reference = REFERENCE_ADAPTER_FEATURES[self.reference_adapter]
        return (*static, *dynamic, *chemistry, *reference)

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = self.full_feature_names
        if self.change_detector == NO_CHANGE_DETECTOR:
            return tuple(
                name for name in names if name not in DARSIA_CHANGE_FEATURE_NAMES
            )
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": int(self.schema),
            "generic_version": self.generic_version,
            "input_color_space": self.input_color_space,
            "change_detector": self.change_detector,
            "chemistry_adapter": self.chemistry_adapter,
            "reference_adapter": self.reference_adapter,
            "include_dynamic": bool(self.include_dynamic),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpticalFeatureContract:
        contract = cls(
            schema=int(payload.get("schema", -1)),
            generic_version=str(payload.get("generic_version") or ""),
            input_color_space=str(payload.get("input_color_space") or ""),
            change_detector=str(
                payload.get("change_detector") or DARSIA_CHANGE_DETECTOR
            ),
            chemistry_adapter=str(payload.get("chemistry_adapter") or "none"),
            reference_adapter=str(payload.get("reference_adapter") or "none"),
            include_dynamic=bool(payload.get("include_dynamic", True)),
        )
        declared = tuple(payload.get("feature_names") or ())
        if declared and declared != contract.feature_names:
            raise ValueError("Optical feature-name contract differs")
        return contract


@dataclass(frozen=True)
class OpticalFeatureWorkspace:
    current: np.ndarray
    baseline: np.ndarray
    previous: np.ndarray
    clearwater: np.ndarray
    current_gray: np.ndarray
    baseline_gray: np.ndarray
    previous_gray: np.ndarray
    clearwater_gray: np.ndarray
    current_smooth: np.ndarray
    baseline_smooth: np.ndarray
    previous_smooth: np.ndarray
    clearwater_smooth: np.ndarray
    rgb_change_norm: np.ndarray
    local_rgb_change_mean: np.ndarray
    local_rgb_change_std: np.ndarray
    dynamic_rgb_change_norm: np.ndarray
    local_dynamic_rgb_change_mean: np.ndarray
    clearwater_span_floor: np.float32
    spatial_context_sigma: np.float32
    dynamic_available: bool
    change_evidence: ChangeEvidence
    contract: OpticalFeatureContract


def _gray_and_smooth(
    rgb: np.ndarray, texture_sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    gray = np.mean(rgb, axis=2, dtype=np.float32)
    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(0.1, float(texture_sigma)),
        sigmaY=max(0.1, float(texture_sigma)),
    )
    return gray, np.asarray(smooth, dtype=np.float32)


def relative_bright_grain_columns(baseline_gray, baseline_smooth,
                                 current_gray, current_smooth, *, xp=np):
    """Learnable registered grain evidence, shared by CPU and CUDA features.

    Local contrast divided by local brightness is invariant to an unclipped
    multiplicative intensity change above the exposure floor. This keeps a
    persistent shiny grain recognizable when dye attenuation changes. The
    bounded floor and range limit dark-pixel noise. New contrast remains a
    separate feature even at an originally bright grain; no gas veto, temporal
    label, or wash-image assumption is imposed.
    """
    def relative(gray, smooth):
        return xp.clip((gray - smooth) / xp.maximum(smooth, xp.float32(.05)),
                       xp.float32(-1), xp.float32(4))

    baseline = relative(baseline_gray, baseline_smooth)
    current = relative(current_gray, current_smooth)
    baseline_bright = xp.maximum(baseline, xp.float32(0))
    current_bright = xp.maximum(current, xp.float32(0))
    persistent = xp.minimum(baseline_bright, current_bright)
    new = xp.maximum(current_bright - baseline_bright, xp.float32(0))
    agreement = persistent * persistent / (
        xp.maximum(baseline_bright, current_bright) + xp.float32(.05))
    return baseline, current, current - baseline, persistent, new, agreement


def _local_mean(values: np.ndarray, sigma: float) -> np.ndarray:
    return np.asarray(
        cv2.GaussianBlur(
            np.asarray(values, dtype=np.float32),
            (0, 0),
            sigmaX=max(0.1, float(sigma)),
            sigmaY=max(0.1, float(sigma)),
        ),
        dtype=np.float32,
    )


class PlainDarSIAChangeEvidenceProvider:
    """DarSIA's signed registered-reference difference plus local context."""

    detector = DARSIA_CHANGE_DETECTOR
    feature_names = DARSIA_CHANGE_FEATURE_NAMES

    def evaluate(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        previous: np.ndarray,
        *,
        spatial_sigma: float,
    ) -> ChangeEvidence:
        baseline_delta = np.asarray(current - baseline, dtype=np.float32)
        dynamic_delta = np.asarray(current - previous, dtype=np.float32)
        baseline_norm = np.linalg.norm(baseline_delta, axis=-1).astype(
            np.float32,
            copy=False,
        )
        dynamic_norm = np.linalg.norm(dynamic_delta, axis=-1).astype(
            np.float32,
            copy=False,
        )
        local_mean = _local_mean(baseline_norm, spatial_sigma)
        local_squared = _local_mean(np.square(baseline_norm), spatial_sigma)
        local_std = np.sqrt(
            np.maximum(
                local_squared - np.square(local_mean),
                np.float32(0.0),
            )
        ).astype(np.float32, copy=False)
        return ChangeEvidence(
            baseline_rgb_difference=baseline_delta,
            baseline_change_norm=baseline_norm,
            local_baseline_change_mean=local_mean,
            local_baseline_change_std=local_std,
            dynamic_rgb_difference=dynamic_delta,
            dynamic_change_norm=dynamic_norm,
            local_dynamic_change_mean=_local_mean(dynamic_norm, spatial_sigma),
        )


class NoChangeEvidenceProvider:
    """Ablation provider that contributes no DarSIA change-evidence channels."""

    detector = NO_CHANGE_DETECTOR
    feature_names: tuple[str, ...] = ()

    def evaluate(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        previous: np.ndarray,
        *,
        spatial_sigma: float,
    ) -> ChangeEvidence:
        del baseline, previous, spatial_sigma
        spatial = np.zeros(current.shape[:2], dtype=np.float32)
        vector = np.zeros_like(current, dtype=np.float32)
        return ChangeEvidence(
            baseline_rgb_difference=vector,
            baseline_change_norm=spatial,
            local_baseline_change_mean=spatial.copy(),
            local_baseline_change_std=spatial.copy(),
            dynamic_rgb_difference=vector.copy(),
            dynamic_change_norm=spatial.copy(),
            local_dynamic_change_mean=spatial.copy(),
        )


CHANGE_EVIDENCE_PROVIDERS: Mapping[str, ChangeEvidenceProvider] = {
    DARSIA_CHANGE_DETECTOR: PlainDarSIAChangeEvidenceProvider(),
    NO_CHANGE_DETECTOR: NoChangeEvidenceProvider(),
}


def change_evidence_provider(detector: str) -> ChangeEvidenceProvider:
    """Resolve a versioned change detector without importing Qt or training code."""

    try:
        return CHANGE_EVIDENCE_PROVIDERS[str(detector)]
    except KeyError as exc:
        raise ValueError(f"Unsupported DarSIA change detector {detector!r}") from exc


def prepare_optical_feature_workspace(
    image_rgb: Any,
    baseline_rgb: Any,
    *,
    previous_rgb: Any | None = None,
    clearwater_rgb: Any | None = None,
    texture_sigma: float = 2.0,
    spatial_context_sigma: float | None = None,
    contract: OpticalFeatureContract | None = None,
) -> OpticalFeatureWorkspace:
    """Prepare reusable full-frame arrays before row-wise feature extraction."""

    resolved_contract = contract or OpticalFeatureContract()
    current = _as_rgb_float(image_rgb)
    baseline = _as_rgb_float(baseline_rgb)
    if current.shape != baseline.shape:
        raise ValueError(
            f"Image/baseline shape mismatch: {current.shape} != {baseline.shape}"
        )
    dynamic_available = previous_rgb is not None
    previous = baseline if previous_rgb is None else _as_rgb_float(previous_rgb)
    if previous.shape != current.shape:
        raise ValueError(
            f"Image/previous shape mismatch: {current.shape} != {previous.shape}"
        )
    uses_clearwater_reference = resolved_contract.reference_adapter != "none"
    if uses_clearwater_reference and clearwater_rgb is None:
        raise ValueError(
            f"Optical reference adapter {resolved_contract.reference_adapter!r} "
            "requires a corrected clear-water reference"
        )
    # A clear-water image is only an input to the optional reference adapter.
    # Do not convert, smooth, or analyse it for the default ``none`` adapter.
    # Keeping the baseline as the inert workspace value preserves the feature
    # layout while avoiding another full-frame float conversion and Gaussian
    # blur for every tile/frame.
    if uses_clearwater_reference:
        clearwater = _as_rgb_float(clearwater_rgb)
        if clearwater.shape != current.shape:
            raise ValueError(
                f"Image/clear-water shape mismatch: {current.shape} != {clearwater.shape}"
            )
    else:
        if clearwater_rgb is not None and np.asarray(clearwater_rgb).shape != current.shape:
            raise ValueError(
                f"Image/clear-water shape mismatch: {current.shape} != {np.asarray(clearwater_rgb).shape}"
            )
        clearwater = baseline
    current_gray, current_smooth = _gray_and_smooth(current, texture_sigma)
    baseline_gray, baseline_smooth = _gray_and_smooth(baseline, texture_sigma)
    if dynamic_available:
        previous_gray, previous_smooth = _gray_and_smooth(previous, texture_sigma)
    else:
        previous_gray = baseline_gray
        previous_smooth = baseline_smooth
    if uses_clearwater_reference:
        clearwater_gray, clearwater_smooth = _gray_and_smooth(clearwater, texture_sigma)
    else:
        # The reference adapter is disabled, so these arrays are never read.
        # Alias the already-computed baseline arrays instead of allocating a
        # second set of full-frame arrays.
        clearwater_gray, clearwater_smooth = baseline_gray, baseline_smooth
    resolved_spatial_sigma = np.float32(
        max(
            float(texture_sigma),
            float(spatial_context_sigma)
            if spatial_context_sigma is not None
            else SPATIAL_CONTEXT_FRACTION * float(min(current.shape[:2])),
        )
    )
    change_evidence = change_evidence_provider(
        resolved_contract.change_detector
    ).evaluate(
        current,
        baseline,
        previous,
        spatial_sigma=float(resolved_spatial_sigma),
    )
    rgb_change_norm = change_evidence.baseline_change_norm
    local_rgb_change_mean = change_evidence.local_baseline_change_mean
    local_rgb_change_std = change_evidence.local_baseline_change_std
    dynamic_rgb_change_norm = change_evidence.dynamic_change_norm
    local_dynamic_rgb_change_mean = change_evidence.local_dynamic_change_mean
    if uses_clearwater_reference:
        clearwater_span = np.linalg.norm(clearwater - baseline, axis=-1)
        finite_clearwater_span = clearwater_span[np.isfinite(clearwater_span)]
        if finite_clearwater_span.size == 0:
            raise ValueError("Clear-water reference has no finite RGB span")
        clearwater_span_floor = np.float32(
            max(0.01, float(np.quantile(finite_clearwater_span, 0.25)))
        )
    else:
        # This value is not read unless the adapter is enabled.
        clearwater_span_floor = np.float32(0.01)
    return OpticalFeatureWorkspace(
        current=current,
        baseline=baseline,
        previous=previous,
        clearwater=clearwater,
        current_gray=current_gray,
        baseline_gray=baseline_gray,
        previous_gray=previous_gray,
        clearwater_gray=clearwater_gray,
        current_smooth=current_smooth,
        baseline_smooth=baseline_smooth,
        previous_smooth=previous_smooth,
        clearwater_smooth=clearwater_smooth,
        rgb_change_norm=rgb_change_norm,
        local_rgb_change_mean=local_rgb_change_mean,
        local_rgb_change_std=local_rgb_change_std,
        dynamic_rgb_change_norm=dynamic_rgb_change_norm,
        local_dynamic_rgb_change_mean=local_dynamic_rgb_change_mean,
        clearwater_span_floor=clearwater_span_floor,
        spatial_context_sigma=resolved_spatial_sigma,
        dynamic_available=dynamic_available,
        change_evidence=change_evidence,
        contract=resolved_contract,
    )


def _fractions(rgb: np.ndarray, epsilon: np.float32) -> np.ndarray:
    return rgb / (np.sum(rgb, axis=-1, keepdims=True) + epsilon)


def _log_ratios(rgb: np.ndarray, epsilon: np.float32) -> np.ndarray:
    safe = np.maximum(rgb, epsilon)
    return np.column_stack(
        (
            np.log(safe[..., 0] / safe[..., 1]).reshape(-1),
            np.log(safe[..., 1] / safe[..., 2]).reshape(-1),
            np.log(safe[..., 0] / safe[..., 2]).reshape(-1),
        )
    ).astype(np.float32, copy=False)


def _btb_acidification(rgb: np.ndarray, epsilon: np.float32) -> np.ndarray:
    yellow = np.float32(0.5) * (rgb[..., 0] + rgb[..., 1])
    blue = rgb[..., 2]
    return (yellow - blue) / (yellow + blue + epsilon)


def optical_feature_matrix(
    workspace: OpticalFeatureWorkspace,
    selection: Any,
) -> np.ndarray:
    """Extract generic features plus the explicitly selected chemistry adapter."""

    epsilon = np.float32(1e-4)
    current = np.asarray(workspace.current[selection], dtype=np.float32)
    baseline = np.asarray(workspace.baseline[selection], dtype=np.float32)
    previous = np.asarray(workspace.previous[selection], dtype=np.float32)
    evidence = workspace.change_evidence
    delta = np.asarray(evidence.baseline_rgb_difference[selection], dtype=np.float32)
    dynamic_delta = np.asarray(
        evidence.dynamic_rgb_difference[selection], dtype=np.float32
    )
    current_fraction = _fractions(current, epsilon)
    baseline_fraction = _fractions(baseline, epsilon)
    current_ratios = _log_ratios(current, epsilon)
    baseline_ratios = _log_ratios(baseline, epsilon)
    current_intensity = np.mean(current, axis=-1, dtype=np.float32)
    baseline_intensity = np.mean(baseline, axis=-1, dtype=np.float32)
    previous_intensity = np.mean(previous, axis=-1, dtype=np.float32)
    current_texture = np.abs(
        np.asarray(workspace.current_gray[selection], dtype=np.float32)
        - np.asarray(workspace.current_smooth[selection], dtype=np.float32)
    )
    baseline_texture = np.abs(
        np.asarray(workspace.baseline_gray[selection], dtype=np.float32)
        - np.asarray(workspace.baseline_smooth[selection], dtype=np.float32)
    )
    previous_texture = np.abs(
        np.asarray(workspace.previous_gray[selection], dtype=np.float32)
        - np.asarray(workspace.previous_smooth[selection], dtype=np.float32)
    )
    rows = current.reshape(-1, 3).shape[0]
    texture_delta = current_texture - baseline_texture
    if workspace.contract.generic_version in {
        "corrected-rgb-texture-dynamic-v1",
        "corrected-rgb-texture-dynamic-v2",
    }:
        blocks: list[np.ndarray] = [
            current.reshape(-1, 3),
            delta.reshape(-1, 3),
            current_intensity.reshape(-1, 1),
            (current_intensity - baseline_intensity).reshape(-1, 1),
            current_fraction.reshape(-1, 3),
            (current_fraction - baseline_fraction).reshape(-1, 3),
            current_ratios,
            current_ratios - baseline_ratios,
            current_texture.reshape(-1, 1),
            texture_delta.reshape(-1, 1),
            np.asarray(workspace.rgb_change_norm[selection], dtype=np.float32).reshape(
                -1, 1
            ),
        ]
    else:
        blocks = [
            delta.reshape(-1, 3),
            (current_intensity - baseline_intensity).reshape(-1, 1),
            (current_fraction - baseline_fraction).reshape(-1, 3),
            current_ratios - baseline_ratios,
            texture_delta.reshape(-1, 1),
            np.abs(texture_delta).reshape(-1, 1),
            np.asarray(workspace.rgb_change_norm[selection], dtype=np.float32).reshape(
                -1, 1
            ),
        ]
    if workspace.contract.generic_version in {
        "corrected-rgb-texture-dynamic-v2",
        BASELINE_RELATIVE_FEATURE_VERSION,
        GENERIC_FEATURE_VERSION,
        RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION,
    }:
        local_mean = np.asarray(
            workspace.local_rgb_change_mean[selection], dtype=np.float32
        )
        local_std = np.asarray(
            workspace.local_rgb_change_std[selection], dtype=np.float32
        )
        blocks.extend(
            (
                local_mean.reshape(-1, 1),
                local_std.reshape(-1, 1),
                (local_mean / (local_std + np.float32(1e-4))).reshape(-1, 1),
            )
        )
    if workspace.contract.generic_version in {GENERIC_FEATURE_VERSION, RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION}:
        # Preserve the sign discarded by the legacy absolute high-pass texture.
        # A registered preinjection bright grain can persist into aqueous CO2;
        # its presence is evidence for the learned heads, never a gas veto.
        baseline_contrast = (
            np.asarray(workspace.baseline_gray[selection], dtype=np.float32)
            - np.asarray(workspace.baseline_smooth[selection], dtype=np.float32)
        )
        current_contrast = (
            np.asarray(workspace.current_gray[selection], dtype=np.float32)
            - np.asarray(workspace.current_smooth[selection], dtype=np.float32)
        )
        baseline_bright = np.maximum(baseline_contrast, np.float32(0.0))
        current_bright = np.maximum(current_contrast, np.float32(0.0))
        blocks.extend(
            (
                baseline_intensity.reshape(-1, 1),
                baseline_contrast.reshape(-1, 1),
                current_contrast.reshape(-1, 1),
                (current_contrast - baseline_contrast).reshape(-1, 1),
                np.minimum(baseline_bright, current_bright).reshape(-1, 1),
                np.maximum(
                    current_bright - baseline_bright, np.float32(0.0)
                ).reshape(-1, 1),
            )
        )
    if workspace.contract.generic_version == RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION:
        blocks.extend(value.reshape(-1, 1) for value in relative_bright_grain_columns(
            workspace.baseline_gray[selection], workspace.baseline_smooth[selection],
            workspace.current_gray[selection], workspace.current_smooth[selection]))
    if workspace.contract.include_dynamic:
        if workspace.dynamic_available:
            available = np.ones((rows, 1), dtype=np.float32)
            dynamic_rgb = dynamic_delta.reshape(-1, 3)
            dynamic_intensity = (current_intensity - previous_intensity).reshape(-1, 1)
            dynamic_texture = (current_texture - previous_texture).reshape(-1, 1)
            dynamic_norm = np.asarray(
                workspace.dynamic_rgb_change_norm[selection], dtype=np.float32
            ).reshape(-1, 1)
            local_dynamic_norm = np.asarray(
                workspace.local_dynamic_rgb_change_mean[selection],
                dtype=np.float32,
            ).reshape(-1, 1)
        else:
            available = np.zeros((rows, 1), dtype=np.float32)
            dynamic_rgb = np.zeros((rows, 3), dtype=np.float32)
            dynamic_intensity = np.zeros((rows, 1), dtype=np.float32)
            dynamic_texture = np.zeros((rows, 1), dtype=np.float32)
            dynamic_norm = np.zeros((rows, 1), dtype=np.float32)
            local_dynamic_norm = np.zeros((rows, 1), dtype=np.float32)
        dynamic_blocks: list[np.ndarray] = [
            available,
            dynamic_rgb,
            dynamic_intensity,
            dynamic_texture,
            dynamic_norm,
        ]
        if workspace.contract.generic_version in {
            "corrected-rgb-texture-dynamic-v2",
            BASELINE_RELATIVE_FEATURE_VERSION,
            GENERIC_FEATURE_VERSION,
            RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION,
        }:
            dynamic_blocks.append(local_dynamic_norm)
        blocks.extend(dynamic_blocks)
    if workspace.contract.chemistry_adapter == "btb-ratiometric-v1":
        acid = _btb_acidification(current, epsilon)
        acid_reference = _btb_acidification(baseline, epsilon)
        blocks.extend(
            (
                acid.reshape(-1, 1),
                (acid - acid_reference).reshape(-1, 1),
            )
        )
    elif workspace.contract.chemistry_adapter == "btb-ratiometric-relative-v2":
        acid = _btb_acidification(current, epsilon)
        acid_reference = _btb_acidification(baseline, epsilon)
        blocks.append((acid - acid_reference).reshape(-1, 1))
    if workspace.contract.reference_adapter == "clearwater-projection-v1":
        clearwater = np.asarray(workspace.clearwater[selection], dtype=np.float32)
        direction = clearwater - baseline
        span_squared = np.sum(direction * direction, axis=-1)
        span = np.sqrt(np.maximum(span_squared, np.float32(0.0)))
        span_floor = workspace.clearwater_span_floor
        safe_span = np.maximum(span, span_floor)
        projection = np.sum(delta * direction, axis=-1) / np.maximum(
            span_squared,
            np.float32(1e-8),
        )
        projected_delta = projection[..., None] * direction
        orthogonal = (
            np.linalg.norm(
                delta - projected_delta,
                axis=-1,
            )
            / safe_span
        )
        alignment = np.exp(np.float32(-0.5) * np.square(orthogonal / np.float32(0.20)))
        projection_score = np.clip(projection, 0.0, 1.0) * alignment
        distance_from_baseline = np.linalg.norm(delta, axis=-1) / safe_span
        distance_to_endpoint = (
            np.linalg.norm(
                current - clearwater,
                axis=-1,
            )
            / safe_span
        )
        clearwater_texture = np.abs(
            np.asarray(workspace.clearwater_gray[selection], dtype=np.float32)
            - np.asarray(workspace.clearwater_smooth[selection], dtype=np.float32)
        )
        reliability = np.clip(
            span / (np.float32(2.0) * span_floor),
            0.0,
            1.0,
        )
        blocks.extend(
            (
                reliability.reshape(-1, 1),
                np.clip(projection, -2.0, 3.0).reshape(-1, 1),
                np.clip(orthogonal, 0.0, 5.0).reshape(-1, 1),
                alignment.reshape(-1, 1),
                projection_score.reshape(-1, 1),
                np.clip(distance_from_baseline, 0.0, 5.0).reshape(-1, 1),
                np.clip(distance_to_endpoint, 0.0, 5.0).reshape(-1, 1),
                np.abs(current_texture - clearwater_texture).reshape(-1, 1),
            )
        )
    features = np.column_stack(blocks).astype(np.float32, copy=False)
    produced_names = workspace.contract.full_feature_names
    selected_names = workspace.contract.feature_names
    if selected_names != produced_names:
        selected_columns = [produced_names.index(name) for name in selected_names]
        features = np.asarray(features[:, selected_columns], dtype=np.float32)
    expected = len(workspace.contract.feature_names)
    if features.shape != (rows, expected):
        raise RuntimeError(
            f"Optical feature contract mismatch: {features.shape} != {(rows, expected)}"
        )
    return features


def darsia_change_signal(
    workspace: OpticalFeatureWorkspace,
    selection: Any = (...,),
) -> dict[str, np.ndarray | bool | str]:
    """Expose the DarSIA reference-change evidence used by the classifier.

    The RGB difference follows ``ConcentrationAnalysis`` with ``diff option =
    plain``.  Local statistics and the previous-frame difference add context,
    while leaving phase classification to the learned heads.
    """

    evidence = workspace.change_evidence
    return {
        "detector": workspace.contract.change_detector,
        "baseline_rgb_difference": np.asarray(
            evidence.baseline_rgb_difference[selection], dtype=np.float32
        ),
        "baseline_change_norm": np.asarray(
            evidence.baseline_change_norm[selection], dtype=np.float32
        ),
        "local_baseline_change_mean": np.asarray(
            evidence.local_baseline_change_mean[selection], dtype=np.float32
        ),
        "local_baseline_change_std": np.asarray(
            evidence.local_baseline_change_std[selection], dtype=np.float32
        ),
        "dynamic_available": bool(workspace.dynamic_available),
        "dynamic_rgb_difference": np.asarray(
            evidence.dynamic_rgb_difference[selection], dtype=np.float32
        ),
        "dynamic_change_norm": np.asarray(
            evidence.dynamic_change_norm[selection], dtype=np.float32
        ),
        "local_dynamic_change_mean": np.asarray(
            evidence.local_dynamic_change_mean[selection], dtype=np.float32
        ),
    }


def optical_feature_samples(
    image_rgb: Any,
    baseline_rgb: Any,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    previous_rgb: Any | None = None,
    clearwater_rgb: Any | None = None,
    texture_sigma: float = 2.0,
    contract: OpticalFeatureContract | None = None,
) -> np.ndarray:
    """Extract contract-defined features at native-resolution pixels."""

    workspace = prepare_optical_feature_workspace(
        image_rgb,
        baseline_rgb,
        previous_rgb=previous_rgb,
        clearwater_rgb=clearwater_rgb,
        texture_sigma=texture_sigma,
        contract=contract,
    )
    row_index = np.asarray(rows, dtype=np.intp)
    column_index = np.asarray(columns, dtype=np.intp)
    if row_index.shape != column_index.shape:
        raise ValueError("Row and column index arrays must have the same shape")
    if row_index.size and (
        np.min(row_index) < 0
        or np.max(row_index) >= workspace.current.shape[0]
        or np.min(column_index) < 0
        or np.max(column_index) >= workspace.current.shape[1]
    ):
        raise IndexError("Feature sample coordinates fall outside the image")
    return optical_feature_matrix(workspace, (row_index, column_index))


__all__ = [
    "BASELINE_BRIGHT_GRAIN_FEATURE_NAMES",
    "BASELINE_RELATIVE_FEATURE_VERSION",
    "CHANGE_EVIDENCE_PROVIDERS",
    "CHEMISTRY_ADAPTER_FEATURES",
    "DARSIA_CHANGE_DETECTOR",
    "DARSIA_CHANGE_FEATURE_NAMES",
    "FEATURE_CONTRACT_SCHEMA",
    "GENERIC_FEATURE_VERSION",
    "INPUT_COLOR_SPACE",
    "NO_CHANGE_DETECTOR",
    "REFERENCE_ADAPTER_FEATURES",
    "RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION",
    "RELATIVE_BRIGHT_GRAIN_FEATURE_NAMES",
    "ChangeEvidence",
    "ChangeEvidenceProvider",
    "NoChangeEvidenceProvider",
    "OpticalFeatureContract",
    "OpticalFeatureWorkspace",
    "PlainDarSIAChangeEvidenceProvider",
    "change_evidence_provider",
    "darsia_change_signal",
    "optical_feature_matrix",
    "optical_feature_samples",
    "prepare_optical_feature_workspace",
    "relative_bright_grain_columns",
]
