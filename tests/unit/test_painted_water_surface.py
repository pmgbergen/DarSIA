"""Display-only lines cannot acquire physical authority from an absent ruler."""
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower.painted_water_surface import (
    make_water_surface_action, move_water_surface_to_photo_point,
    propose_water_surface, water_surface_findings, water_surface_measurements,
)


def basis(run="ac17"):
    return dict(format="darsia-water-surface-basis-v1", run_id=run,
        photo=dict(path="C:/bound/clearwater.npy", sha256="a" * 64, size_bytes=123),
        photo_shape=[601, 901], photo_to_metric_h=np.eye(3).tolist(),
        own_geometry_origin_metric_mm=[20., 550.], own_geometry_up_metric_xy=[0., -1.],
        left_bolts_photo_xy=[[20., 550.], [20., 355.], [20., 160.]],
        bolt_heights_mm=[0, 195, 390], bottom_edge_photo_xy=[[0., 550.], [900., 550.]],
        own_top_photo_xy=[[0., 40.], [900., 40.]],
        transported_top_photo_xy=[[0., 30.], [900., 30.]],
        port_photo_xy=None, provenance=dict(fixture=True, production_geometry_unchanged=True))


def action(binding, y=100, x2=900):
    return make_water_surface_action(binding, [[0., y], [x2, y]],
        actor="Human fixture", confirmed_utc="2026-10-22T08:00:00Z")


def test_observed_bolts_height_and_two_distinct_computed_tops():
    b = basis()
    row = water_surface_measurements(action(b), expected_basis=b)
    assert row["height_mm"] == pytest.approx(450)
    assert row["bolt_height_mm"] == pytest.approx(450)
    assert row["height_basis"] == "existing-own-geometry"
    assert row["own_top_minus_water_mm"] == pytest.approx(60)
    assert row["transported_top_minus_water_mm"] == pytest.approx(70)
    assert row["angle_degrees"] == 0
    assert row["ruler_extrapolated"] is True
    assert row["production_geometry_changed"] is False


def test_missing_actual_bolt_observations_never_become_frame_ruler():
    b = basis()
    b["left_bolts_photo_xy"] = None
    row = water_surface_measurements(action(b), expected_basis=b)
    assert row["height_mm"] == pytest.approx(450) and row["angle_degrees"] == 0
    assert row["bolt_height_mm"] is None
    assert row["own_top_minus_water_mm"] == pytest.approx(60)
    assert "observed-0-195-390-mm-left-bolts" in row["unavailable"]


def test_absent_existing_metric_binding_is_explicit_and_forbids_move():
    b = basis()
    b["photo_to_metric_h"] = None
    row = water_surface_measurements(action(b), expected_basis=b)
    assert row["height_mm"] is None and row["angle_degrees"] is None
    with pytest.raises(ValueError, match="unavailable"):
        move_water_surface_to_photo_point([[0., 100.], [900., 100.]], [100., 120.], basis=b)


def test_projective_observed_bolt_ruler_is_not_assumed_linear():
    b = basis()
    b["left_bolts_photo_xy"] = [[20., 550.], [20., 420.], [20., 160.]]
    row = water_surface_measurements(action(b, y=420), expected_basis=b)
    assert row["bolt_height_mm"] == pytest.approx(195)
    assert row["own_geometry_height_mm"] == pytest.approx(130)
    row = water_surface_measurements(action(b, y=160), expected_basis=b)
    assert row["bolt_height_mm"] == pytest.approx(390)


def test_click_move_preserves_existing_metric_angle_and_passes_clicked_point():
    b = basis()
    b["photo_to_metric_h"] = [[1., .03, 0.], [0., 1., 0.], [.0004, .0001, 1.]]
    old = [[0., 100.], [900., 180.]]
    click = [400., 310.]
    moved = move_water_surface_to_photo_point(old, click, basis=b)
    a = make_water_surface_action(b, old, actor="Human", confirmed_utc="2026-10-22T08:00:00Z")
    c = make_water_surface_action(b, moved, actor="Human", confirmed_utc="2026-10-22T08:00:00Z")
    left = water_surface_measurements(a, expected_basis=b)
    right = water_surface_measurements(c, expected_basis=b)
    assert right["angle_degrees"] == pytest.approx(left["angle_degrees"])
    line = np.cross(np.r_[moved[0], 1.], np.r_[moved[1], 1.])
    assert np.dot(line, np.r_[click, 1.]) == pytest.approx(0, abs=1e-7)
    assert b["photo_to_metric_h"][2] == [.0004, .0001, 1.]


def test_known_port_column_reports_local_height_not_left_column():
    b = basis()
    b["port_photo_xy"] = [800., 500.]
    a = make_water_surface_action(b, [[0., 100.], [900., 190.]],
        actor="Human", confirmed_utc="2026-10-22T08:00:00Z")
    row = water_surface_measurements(a, expected_basis=b)
    assert row["height_location"] == "bound-port-column"
    assert row["height_mm"] == pytest.approx(370)


def test_confirmation_is_exactly_bound_to_photo_and_observations():
    b = basis()
    a = action(b)
    for changed in ("photo", "left_bolts_photo_xy", "own_top_photo_xy", "photo_to_metric_h"):
        other = deepcopy(b)
        if changed == "photo":
            other[changed]["sha256"] = "b" * 64
        else:
            other[changed][0][0] += .25
        with pytest.raises(ValueError, match="basis"):
            water_surface_measurements(a, expected_basis=other)


def test_confirmation_needs_explicit_actor_utc_and_in_photo_points():
    b = basis()
    for actor, when, points in (("", "2026-10-22T08:00:00Z", [[0, 10], [900, 10]]),
        ("Human", "2026-10-22T08:00:00", [[0, 10], [900, 10]]),
        ("Human", "2026-10-22T08:00:00Z", [[0, 10], [901, 10]]),
        ("Human", "2026-10-22T08:00:00Z", [[False, 10], [900, 10]])):
        with pytest.raises(ValueError):
            make_water_surface_action(b, points, actor=actor, confirmed_utc=when)


def test_proposal_is_unconfirmed_and_flat_photo_has_no_invented_line():
    b = basis()
    photo = np.ones((601, 901, 3), dtype=np.float32)
    assert propose_water_surface(photo, basis=b)["line_photo_xy"] is None
    photo[100:] *= .5
    proposal = propose_water_surface(photo, basis=b)
    assert proposal["line_photo_xy"] == [[0., 99.5], [900., 99.5]]
    assert proposal["diagnostics"]["human_confirmation_required"] is True
    assert proposal["diagnostics"]["physical_classification"] is False


def test_reversed_line_endpoints_do_not_create_false_tilt():
    b = basis()
    a = make_water_surface_action(b, [[900., 100.], [0., 100.]],
        actor="Human", confirmed_utc="2026-10-22T08:00:00Z")
    assert water_surface_measurements(a, expected_basis=b)["angle_degrees"] == 0


def test_confirmed_series_median_warning_has_no_physical_verdict():
    bases = {run: basis(run) for run in ("ac17", "ac18", "ac19")}
    actions = {run: action(bases[run], y) for run, y in zip(bases, (100, 101, 111))}
    rows = water_surface_findings(actions, bases)
    assert [row["severity"] for row in rows] == ["information", "information", "warning"]
    assert rows[2]["height_delta_from_confirmed_median_mm"] == pytest.approx(-10)
    assert all(row["physical_verdict"] is False for row in rows)
    assert all(row["severity"] == "information" for row in
               water_surface_findings(actions, bases, height_tolerance_mm=12))


def test_one_confirmation_has_no_series_height_comparison():
    b = basis()
    row = water_surface_findings({"ac17": action(b)}, {"ac17": b})[0]
    assert row["confirmed_series_median_mm"] is None
    assert row["height_delta_from_confirmed_median_mm"] is None
    with pytest.raises(ValueError, match="tolerances"):
        water_surface_findings({"ac17": action(b)}, {"ac17": b}, height_tolerance_mm=True)


def test_bolt_ruler_cannot_extrapolate_across_its_projective_pole():
    b = basis()
    b["left_bolts_photo_xy"] = [[20., 550.], [20., 400.], [20., 350.]]
    row = water_surface_measurements(action(b), expected_basis=b)
    assert row["own_geometry_height_mm"] == pytest.approx(450)
    assert row["bolt_height_mm"] is None
    assert "observed-bolt-ruler-projective-horizon" in row["unavailable"]


def test_own_height_needs_explicit_existing_origin_and_up_convention():
    b = basis()
    b["own_geometry_origin_metric_mm"] = None
    row = water_surface_measurements(action(b), expected_basis=b)
    assert row["height_mm"] is None
    assert row["bolt_height_mm"] == pytest.approx(450)
    assert "existing-own-geometry-origin-or-up-convention" in row["unavailable"]
