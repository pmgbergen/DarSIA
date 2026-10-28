"""Reference-derived observation gate for weak aqueous colour-path signals.

The gate is deliberately defined in the parameter-independent colour-path
coordinate, before the optimised signal model and Flash chemistry.  It is
therefore identical for SimpleFlash and TitrationFlash and cannot be tuned by
Optuna to hide a mass-balance error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GATE_SCHEMA = 1


def normalise_aqueous_noise_gate_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower().replace("_", "-")
    aliases = {
        "0": "off",
        "false": "off",
        "none": "off",
        "1": "baseline-soft",
        "true": "baseline-soft",
        "on": "baseline-soft",
        "baseline": "baseline-soft",
        "soft": "baseline-soft",
        "causal": "baseline-causal",
        "causal-soft": "baseline-causal",
        "baseline-causal-soft": "baseline-causal",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"off", "baseline-soft", "baseline-causal"}:
        raise ValueError(
            f"Unknown aqueous noise gate {value!r}; expected off, baseline-soft, "
            "or baseline-causal."
        )
    return mode


def file_sha256(path: Path) -> str:
    from ..project import _sha256_file_uncached
    return _sha256_file_uncached(Path(path))


def smoothstep_confidence(
    values: Any,
    low: float,
    high: float,
    *,
    xp: Any = np,
) -> Any:
    """Return a C1-continuous confidence ramp from zero to one."""

    width = max(float(high) - float(low), np.finfo(float).eps)
    unit = xp.clip((values - float(low)) / width, 0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


@dataclass(frozen=True)
class LabelNoiseThreshold:
    label: int
    low: float
    high: float
    sample_count: int
    median: float
    mad: float
    q95: float
    q99: float
    q995: float
    q999: float
    q9995: float
    maximum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": int(self.label),
            "low": float(self.low),
            "high": float(self.high),
            "sample_count": int(self.sample_count),
            "median": float(self.median),
            "mad": float(self.mad),
            "q95": float(self.q95),
            "q99": float(self.q99),
            "q995": float(self.q995),
            "q999": float(self.q999),
            "q9995": float(self.q9995),
            "maximum": float(self.maximum),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabelNoiseThreshold":
        return cls(
            label=int(payload["label"]),
            low=float(payload["low"]),
            high=float(payload["high"]),
            sample_count=int(payload.get("sample_count", 0)),
            median=float(payload.get("median", 0.0)),
            mad=float(payload.get("mad", 0.0)),
            q95=float(payload.get("q95", 0.0)),
            q99=float(payload.get("q99", 0.0)),
            q995=float(payload.get("q995", payload.get("low", 0.0))),
            q999=float(payload.get("q999", payload.get("high", 0.0))),
            q9995=float(payload.get("q9995", payload.get("high", 0.0))),
            maximum=float(payload.get("maximum", 0.0)),
        )


@dataclass(frozen=True)
class AqueousNoiseGate:
    """Per-label soft gate estimated solely from no-CO2 reference frames."""

    mode: str
    quantile_low: float
    quantile_high: float
    spatial_stride: int
    thresholds: Mapping[int, LabelNoiseThreshold]
    source_paths: tuple[str, ...]
    source_sha256: tuple[str, ...]
    coordinate: str = "raw_color_path"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def threshold_for(self, label: int) -> LabelNoiseThreshold | None:
        return self.thresholds.get(int(label))

    def apply_array(
        self,
        coordinate: Any,
        labels: Any,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        values = np.asarray(coordinate)
        label_array = np.asarray(labels)
        if values.shape[:2] != label_array.shape[:2]:
            raise ValueError(
                "Aqueous noise-gate coordinate/label shapes differ: "
                f"{values.shape[:2]} != {label_array.shape[:2]}"
            )
        if values.ndim == 3 and values.shape[-1] == 1:
            scalar = values[..., 0]
        elif values.ndim == 2:
            scalar = values
        else:
            raise ValueError(
                "Aqueous noise gate expects a scalar colour-path coordinate; "
                f"got shape {values.shape}."
            )
        output = np.array(scalar, copy=copy)
        if not self.enabled:
            return output
        for label, threshold in self.thresholds.items():
            mask = label_array == int(label)
            if not np.any(mask):
                continue
            selected = output[mask]
            confidence = smoothstep_confidence(
                selected,
                threshold.low,
                threshold.high,
            )
            output[mask] = selected * confidence
        return output

    def confidence_array(self, coordinate: Any, labels: Any) -> np.ndarray:
        """Return the parameter-independent per-pixel observation confidence."""

        values = np.asarray(coordinate)
        label_array = np.asarray(labels)
        if values.shape[:2] != label_array.shape[:2]:
            raise ValueError(
                "Aqueous noise-gate coordinate/label shapes differ: "
                f"{values.shape[:2]} != {label_array.shape[:2]}"
            )
        if values.ndim == 3 and values.shape[-1] == 1:
            scalar = values[..., 0]
        elif values.ndim == 2:
            scalar = values
        else:
            raise ValueError(
                "Aqueous noise gate expects a scalar colour-path coordinate; "
                f"got shape {values.shape}."
            )
        confidence = np.zeros(scalar.shape, dtype=np.float32)
        if not self.enabled:
            return confidence
        for label, threshold in self.thresholds.items():
            mask = label_array == int(label)
            if not np.any(mask):
                continue
            confidence[mask] = np.asarray(
                smoothstep_confidence(
                    np.asarray(scalar[mask], dtype=np.float32),
                    threshold.low,
                    threshold.high,
                ),
                dtype=np.float32,
            )
        return confidence

    def apply_image(self, image: Any, labels: Any, *, copy: bool = False) -> Any:
        target = image.copy() if copy else image
        target.img = self.apply_array(target.img, labels, copy=True)
        return target

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_SCHEMA,
            "mode": self.mode,
            "coordinate": self.coordinate,
            "quantile_low": float(self.quantile_low),
            "quantile_high": float(self.quantile_high),
            "spatial_stride": int(self.spatial_stride),
            "source_paths": list(self.source_paths),
            "source_sha256": list(self.source_sha256),
            "thresholds": {
                str(label): threshold.to_dict()
                for label, threshold in sorted(self.thresholds.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AqueousNoiseGate":
        mode = normalise_aqueous_noise_gate_mode(str(payload.get("mode", "off")))
        raw_thresholds = payload.get("thresholds") or {}
        thresholds = {
            int(label): LabelNoiseThreshold.from_dict(value)
            for label, value in raw_thresholds.items()
        }
        return cls(
            mode=mode,
            coordinate=str(payload.get("coordinate", "raw_color_path")),
            quantile_low=float(payload.get("quantile_low", 0.99)),
            quantile_high=float(payload.get("quantile_high", 0.999)),
            spatial_stride=int(payload.get("spatial_stride", 1)),
            thresholds=thresholds,
            source_paths=tuple(str(path) for path in payload.get("source_paths", [])),
            source_sha256=tuple(
                str(value) for value in payload.get("source_sha256", [])
            ),
        )


def _deterministic_cap(values: np.ndarray, maximum: int) -> np.ndarray:
    if values.size <= maximum:
        return values
    indices = np.linspace(0, values.size - 1, maximum, dtype=np.int64)
    return values[indices]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray([0.50, 0.95, 0.99, 0.995, 0.999, 0.9995, 1.0])
    result = np.quantile(values, probabilities)
    median = float(result[0])
    mad = float(np.median(np.abs(values - median)))
    return {
        "median": median,
        "mad": mad,
        "q95": float(result[1]),
        "q99": float(result[2]),
        "q995": float(result[3]),
        "q999": float(result[4]),
        "q9995": float(result[5]),
        "maximum": float(result[6]),
    }


def estimate_aqueous_noise_gate(
    prepared_references: Sequence[Any],
    labels: Any,
    active_labels: Iterable[int],
    *,
    mode: str = "baseline-soft",
    source_paths: Sequence[Path] = (),
    quantile_low: float = 0.99,
    quantile_high: float = 0.999,
    spatial_stride: int = 4,
    max_samples_per_label: int = 2_000_000,
    minimum_width: float = 1e-4,
) -> AqueousNoiseGate:
    """Estimate deterministic false-positive limits from corrected references."""

    if not 0.5 <= float(quantile_low) < float(quantile_high) < 1.0:
        raise ValueError("Noise-gate quantiles must satisfy 0.5 <= low < high < 1.")
    if not prepared_references:
        raise ValueError("At least one prepared no-CO2 reference is required.")
    stride = max(1, int(spatial_stride))
    label_array = np.asarray(labels)[::stride, ::stride]
    sampled_frames: list[np.ndarray] = []
    for image in prepared_references:
        array = np.asarray(getattr(image, "img", image))
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(
                f"Prepared no-CO2 reference is not scalar: shape={array.shape}"
            )
        if array.shape != np.asarray(labels).shape[:2]:
            raise ValueError(
                "Prepared reference/label shapes differ: "
                f"{array.shape} != {np.asarray(labels).shape[:2]}"
            )
        sampled_frames.append(np.asarray(array[::stride, ::stride], dtype=np.float64))

    thresholds: dict[int, LabelNoiseThreshold] = {}
    for label in sorted({int(value) for value in active_labels}):
        mask = label_array == label
        samples = np.concatenate(
            [frame[mask].reshape(-1) for frame in sampled_frames]
        )
        samples = samples[np.isfinite(samples)]
        samples = np.maximum(samples, 0.0)
        if not samples.size:
            raise ValueError(f"No finite no-CO2 samples found for label {label}.")
        samples = _deterministic_cap(samples, int(max_samples_per_label))
        stats = _quantiles(samples)
        low = max(0.0, float(np.quantile(samples, quantile_low)))
        high = max(
            float(np.quantile(samples, quantile_high)),
            low + max(float(minimum_width), abs(low) * 0.05),
        )
        thresholds[label] = LabelNoiseThreshold(
            label=label,
            low=low,
            high=high,
            sample_count=int(samples.size),
            **stats,
        )

    paths = tuple(str(Path(path)) for path in source_paths)
    hashes = tuple(file_sha256(Path(path)) for path in source_paths)
    return AqueousNoiseGate(
        mode=normalise_aqueous_noise_gate_mode(mode),
        quantile_low=float(quantile_low),
        quantile_high=float(quantile_high),
        spatial_stride=stride,
        thresholds=thresholds,
        source_paths=paths,
        source_sha256=hashes,
    )
