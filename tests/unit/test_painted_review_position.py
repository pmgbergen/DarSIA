"""Analytic metric-registration comparisons with explicit shared source bindings."""

from copy import deepcopy
import math

import pytest

from darsia.fluidflower.painted_review_position import position_readout


def _model(matrix=None):
    """Supply literal model metadata without a source file or photograph."""
    return dict(metric_to_camera_u=deepcopy(matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        metric_units="mm", source_profile=dict(format="registration-profile", version=1),
        camera_u_context=dict(frame="shared-camera-u", shape_hw=[10, 20]),
        provenance=dict(binding="synthetic-model"))


def _read(models, *, run="target", order=None, anchor=(0, 0), **choices):
    """Call the public API with one explicit reference and declared order."""
    return position_readout(models, run, reference_run="reference",
        declared_order=order or ["reference", "target"], anchor_metric_mm=anchor, **choices)


def test_direct_reference_translation_is_separate_from_immediate_previous():
    """Every reference comparison is direct, independently of the previous run."""
    models = dict(reference=_model([[1, 0, 3], [0, 1, 4], [0, 0, 1]]),
        middle=_model([[1, 0, 5], [0, 1, 1], [0, 0, 1]]),
        target=_model([[1, 0, 10], [0, 1, 2], [0, 0, 1]]))
    for run_id, model in models.items():
        model["provenance"]["binding"] = run_id
    result = _read(models, order=["reference", "middle", "target"], anchor=(100, 200))
    assert result["method"] == "camera registration (map placement)"
    assert result["scope"] == "local-projective-x-tangent" and result["advisory_only"]
    assert result["previous_run_id"] == "middle" and not result["unavailable_reasons"]
    for key, dx, dy in (("reference", 7, -2), ("previous", 5, 1)):
        row = result[key]
        assert row["anchor_metric_mm"] == [100.0, 200.0]
        assert row["dx_mm"] == pytest.approx(dx) and row["dy_mm"] == pytest.approx(dy)
        assert row["shift_mm"] == pytest.approx(math.hypot(dx, dy))
        assert row["rotation_degrees"] == pytest.approx(0)
        assert row["shift_warning"] and not row["rotation_warning"]
        assert row["provenance"] == dict(target=models["target"]["provenance"],
            comparison=models[row["comparison_run_id"]]["provenance"])


def test_rotation_uses_the_local_metric_x_tangent():
    """A thirty-degree rigid registration has zero anchor shift but an angle warning."""
    cosine, sine = math.sqrt(3) / 2, .5
    result = _read(dict(reference=_model(),
        target=_model([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])))
    row = result["reference"]
    assert row["shift_mm"] == 0 and not row["shift_warning"]
    assert row["rotation_degrees"] == pytest.approx(30)
    assert row["rotation_warning"] and row["warning"]
    assert row["rotation_scope"] == "local-projective-x-tangent"


def test_projective_position_and_tangent_match_the_anchor_analytic_derivative():
    """The quotient map at (2,3) has tangent proportional to (1,-0.3)."""
    result = _read(dict(reference=_model(),
        target=_model([[1, 0, 0], [0, 1, 0], [.1, 0, 1]])), anchor=(2, 3),
        shift_tolerance_mm=1, angle_tolerance_degrees=20)
    row = result["reference"]
    assert row["dx_mm"] == pytest.approx(-1 / 3)
    assert row["dy_mm"] == pytest.approx(-.5)
    assert row["shift_mm"] == pytest.approx(math.sqrt(13) / 6)
    assert row["rotation_degrees"] == pytest.approx(math.degrees(math.atan2(-.3, 1)))
    assert not row["warning"]


@pytest.mark.parametrize("failure,reason", [("missing", "missing"), ("malformed", "metric_units"),
    ("context", "camera_u_context")])
def test_unavailable_immediate_previous_is_not_skipped(failure, reason):
    """An unavailable neighbour never becomes an earlier valid model."""
    models = dict(reference=_model(), previous=_model(), target=_model())
    if failure == "missing":
        del models["previous"]
    elif failure == "malformed":
        models["previous"]["metric_units"] = "m"
    else:
        models["previous"]["camera_u_context"]["frame"] = "another-camera-u"
    result = _read(models, order=["reference", "previous", "target"])
    assert result["reference"] is not None
    assert result["previous_run_id"] == "previous" and result["previous"] is None
    assert reason in result["unavailable_reasons"]["previous"]


@pytest.mark.parametrize("field,value", [
    ("source_profile", dict(format="different-profile", version=1)),
    ("camera_u_context", dict(frame="different-camera-u", shape_hw=[10, 20])),
    ("source_profile", dict(format="registration-profile", version=True)),
])
def test_equal_shapes_do_not_replace_exact_shared_context_and_profile(field, value):
    """Literal source identity, including JSON value types, must agree."""
    models = dict(reference=_model(), target=_model())
    models["target"][field] = value
    result = _read(models)
    assert result["reference"] is None and result["previous"] is None
    assert field in result["unavailable_reasons"]["reference"]


@pytest.mark.parametrize("field,value,reason", [
    ("metric_units", "m", "metric_units"),
    ("metric_units", None, "metric_units"),
    ("metric_to_camera_u", [[1, 0], [0, 1]], "3x3"),
    ("metric_to_camera_u", [[True, 0, 0], [0, 1, 0], [0, 0, 1]], "finite number"),
    ("metric_to_camera_u", [[1, 0, 0], [0, 0, 0], [0, 0, 1]], "singular"),
    ("metric_to_camera_u", [[1, 0, math.nan], [0, 1, 0], [0, 0, 1]], "finite number"),
    ("source_profile", {}, "nonempty"),
    ("camera_u_context", {}, "nonempty"),
    ("provenance", {}, "provenance"),
])
def test_malformed_needed_models_are_explicitly_unavailable(field, value, reason):
    """Bad model fields do not become zero displacement or inferred metric units."""
    models = dict(reference=_model(), target=_model())
    models["target"][field] = value
    result = _read(models)
    assert result["reference"] is None and result["previous"] is None
    assert reason in result["unavailable_reasons"]["reference"]


def test_anchor_at_relative_projective_horizon_is_unavailable():
    """An invertible homography still cannot map a point on its horizon."""
    result = _read(dict(reference=_model(),
        target=_model([[1, 0, 0], [0, 1, 0], [1, 0, -2]])), anchor=(2, 1))
    assert result["reference"] is None
    assert "horizon" in result["unavailable_reasons"]["reference"]


def test_validated_self_comparison_is_literal_zero_and_snapshots_do_not_alias_inputs():
    """Self comparison cannot warn through inverse roundoff or mutate source metadata."""
    models = dict(reference=_model([[1.2, .3, 7], [.2, 1.1, 5], [.001, .002, 1]]))
    original = deepcopy(models)
    result = _read(models, run="reference", order=["reference"], anchor=(2, 3),
        shift_tolerance_mm=0, angle_tolerance_degrees=0)
    row = result["reference"]
    assert [row[key] for key in ("dx_mm", "dy_mm", "shift_mm", "rotation_degrees")] == [0.0] * 4
    assert not row["warning"] and result["previous"] is None and result["previous_run_id"] is None
    row["provenance"]["target"]["binding"] = "changed result only"
    row["camera_u_context"]["shape_hw"][0] = 99
    row["source_profile"]["version"] = 3
    assert models == original


@pytest.mark.parametrize("choices", [
    dict(shift_tolerance_mm=True), dict(shift_tolerance_mm=-1),
    dict(shift_tolerance_mm=math.inf), dict(angle_tolerance_degrees=False),
    dict(angle_tolerance_degrees=-.1), dict(angle_tolerance_degrees=math.nan),
    dict(anchor=(True, 0)), dict(anchor=(0, math.inf)),
    dict(order=["reference", "reference", "target"]),
])
def test_invalid_explicit_anchor_order_or_tolerances_are_rejected(choices):
    """Bad caller parameters cannot silently select another point, order or tolerance."""
    with pytest.raises(ValueError):
        _read(dict(reference=_model(), target=_model()), **choices)
