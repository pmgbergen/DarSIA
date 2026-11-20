"""Kernel evaluation: the signal is not clipped, unless asked to."""

import numpy as np

import darsia

# Key colors (relative to the baseline) of a path with values 0, 0.5, 1.
RELATIVE_COLORS = [np.zeros(3), np.array([-0.4, -0.05, 0.05]), np.array([-0.8, -0.2, 0.1])]
VALUES = [0.0, 0.5, 1.0]
# Colors on the line through the first and last key color, where the Gaussian
# interpolation undershoots (before the first) and overshoots (close to the last).
PROBE = np.array(
    [RELATIVE_COLORS[0] + t * (RELATIVE_COLORS[2] - RELATIVE_COLORS[0]) for t in (-0.2, 0.95)]
)


def signal_function(evaluation: darsia.KernelEvaluation):
    color_path = darsia.ColorPath(relative_colors=RELATIVE_COLORS, base_color=np.ones(3))
    supports = color_path.equidistant_distances
    return evaluation.signal_function(
        color_path, darsia.ColorMode.RELATIVE, supports, VALUES
    )


def test_signal_is_clipped_on_request() -> None:
    signal = signal_function(darsia.KernelEvaluation(clip=True))(PROBE)
    assert np.all((signal >= 0.0) & (signal <= 1.0))


def test_by_default_the_interpolation_is_returned_as_is() -> None:
    unclipped = signal_function(darsia.KernelEvaluation())
    clipped = signal_function(darsia.KernelEvaluation(clip=True))
    raw = unclipped._interpolation(PROBE)
    assert np.array_equal(unclipped(PROBE), raw)
    assert np.any((raw < 0.0) | (raw > 1.0))  # the probe leaves the value range
    assert np.allclose(clipped(PROBE), np.clip(raw, 0.0, 1.0))
    # At the key colors both interpolate the values.
    keys = np.array(RELATIVE_COLORS)
    assert np.allclose(unclipped(keys), VALUES, atol=1e-6)


def test_clip_is_stored_with_the_mode(tmp_path) -> None:
    darsia.KernelEvaluation(clip=True).save(tmp_path)
    loaded = darsia.EvaluationMode.load(tmp_path)
    assert isinstance(loaded, darsia.KernelEvaluation)
    assert loaded.clip is True
    assert loaded.same_as(darsia.KernelEvaluation(clip=True))
    assert not loaded.same_as(darsia.KernelEvaluation())


def test_modes_stored_without_clip_are_not_clipped() -> None:
    mode = darsia.KernelEvaluation.from_parameters(
        {"kernel": "gaussian", "gamma": 10.0, "regularization": 1e-8}
    )
    assert mode.clip is False
