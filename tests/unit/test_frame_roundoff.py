"""Equivalent camera registrations versus an actual metric displacement."""
import numpy as np
import pytest

from darsia.fluidflower import painted_analysis_exclusions as subject
from darsia.fluidflower import painted_adoption_table, photo_edge_continuation, layer_painting


@pytest.mark.parametrize('displacement', [0., 1e-5])
def test_frame_comparison_distinguishes_roundoff_from_physical_motion(monkeypatch, displacement):
    metric = np.array([[0., 490.], [899., 490.], [899., 0.], [0., 0.]])
    source_h = np.array([[6., .01, 7.], [.02, -6., 3085.], [0., 0., 1.]])
    target_h = np.array([[5.8, .03, 8.], [-.02, -6.1, 3090.], [0., 0., 1.]])
    def project(points, matrix):
        out = np.c_[points, np.ones(4)] @ matrix.T
        return out[:, :2] / out[:, 2:]
    shifted = metric.copy()
    shifted[:, 0] += displacement
    source, target = project(metric, source_h), project(shifted, target_h)
    monkeypatch.setattr(painted_adoption_table, '_frame', lambda value: value)
    monkeypatch.setattr(layer_painting, 'frame_polygon', lambda value, _scale: value)
    monkeypatch.setattr(photo_edge_continuation, 'validate_contract',
        lambda _contract, **_options: (source_h, None, None, None, target, None))
    result = subject._frame_comparison({'frame': source},
        {'target_model': {'metric_to_grid': target_h.tolist()}})
    assert result['max_edge_distance_mm'] > 0
    assert result['mismatch'] is bool(displacement)
    assert result['numerical_roundoff_mm'] < 1e-9
