"""Parameter-independent optical support derived from no-CO2 reference frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from darsia.fluidflower.tools.aqueous_noise_gate import smoothstep_confidence


def btb_acidification_coordinate(image: Any, *, epsilon: float = 1e-4) -> np.ndarray:
    """Return an illumination-resistant blue-to-yellow BTB coordinate.

    Acidified BTB loses blue relative to its red/green response. The bounded
    normalized contrast is invariant to a common RGB gain and does not depend
    on the run-specific DarSIA colour path or any Optuna parameter.
    """

    array = np.asarray(getattr(image, "img", image), dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image, got {array.shape}.")
    red = np.maximum(array[..., 0], 0.0)
    green = np.maximum(array[..., 1], 0.0)
    blue = np.maximum(array[..., 2], 0.0)
    yellow = 0.5 * (red + green)
    return np.asarray(
        (yellow - blue) / (yellow + blue + float(epsilon)),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ResidualThreshold:
    label: int
    low: float
    high: float
    sample_count: int


@dataclass(frozen=True)
class SpatialAqueousSupportModel:
    """Pixel-centred colour-path support with per-label residual thresholds."""

    baseline_median: np.ndarray
    thresholds: Mapping[int, ResidualThreshold]
    drift_quantile: float = 0.10

    def confidence(
        self,
        coordinate: Any,
        labels: Any,
        *,
        remove_frame_offset: bool = True,
    ) -> tuple[np.ndarray, dict[int, float]]:
        values = np.asarray(coordinate, dtype=np.float32)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        label_array = np.asarray(labels)
        if values.shape != self.baseline_median.shape:
            raise ValueError(
                f"Coordinate shape {values.shape} != baseline {self.baseline_median.shape}."
            )
        if label_array.shape != values.shape:
            raise ValueError(f"Label shape {label_array.shape} != {values.shape}.")

        excess = values - self.baseline_median
        output = np.zeros(values.shape, dtype=np.float32)
        offsets: dict[int, float] = {}
        for label, threshold in self.thresholds.items():
            mask = label_array == int(label)
            selected = excess[mask]
            if not selected.size:
                continue
            offset = 0.0
            if remove_frame_offset:
                finite = selected[np.isfinite(selected)]
                if finite.size:
                    offset = max(
                        0.0,
                        float(np.quantile(finite, float(self.drift_quantile))),
                    )
            offsets[int(label)] = offset
            output[mask] = smoothstep_confidence(
                selected - offset,
                threshold.low,
                threshold.high,
            ).astype(np.float32)
        return output, offsets


def _scalar_array(value: Any) -> np.ndarray:
    array = np.asarray(getattr(value, "img", value), dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected scalar observation coordinate, got {array.shape}.")
    return array


def estimate_spatial_aqueous_support(
    prepared_references: Sequence[Any],
    labels: Any,
    active_labels: Iterable[int],
    *,
    quantile_low: float = 0.99,
    quantile_high: float = 0.999,
    minimum_width: float = 1e-4,
    drift_quantile: float = 0.10,
) -> SpatialAqueousSupportModel:
    """Estimate local baselines and global residual limits from no-CO2 images."""

    if len(prepared_references) < 2:
        raise ValueError("At least two no-CO2 references are required.")
    if not 0.5 <= quantile_low < quantile_high < 1.0:
        raise ValueError("Residual quantiles must satisfy 0.5 <= low < high < 1.")
    arrays = [_scalar_array(value) for value in prepared_references]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Reference shapes differ: {sorted(shapes)}")
    label_array = np.asarray(labels)
    if label_array.shape != arrays[0].shape:
        raise ValueError(
            f"Label shape {label_array.shape} != reference shape {arrays[0].shape}."
        )

    stack = np.stack(arrays, axis=0)
    baseline = np.nanmedian(stack, axis=0).astype(np.float32)
    residual = stack - baseline[None, ...]
    thresholds: dict[int, ResidualThreshold] = {}
    for label in sorted({int(value) for value in active_labels}):
        mask = label_array == label
        samples = residual[:, mask].reshape(-1)
        samples = samples[np.isfinite(samples)]
        samples = np.maximum(samples, 0.0)
        if not samples.size:
            raise ValueError(f"No finite residual samples for label {label}.")
        low = max(0.0, float(np.quantile(samples, quantile_low)))
        high = max(
            float(np.quantile(samples, quantile_high)),
            low + max(float(minimum_width), abs(low) * 0.05),
        )
        thresholds[label] = ResidualThreshold(
            label=label,
            low=low,
            high=high,
            sample_count=int(samples.size),
        )
    return SpatialAqueousSupportModel(
        baseline_median=baseline,
        thresholds=thresholds,
        drift_quantile=float(drift_quantile),
    )
