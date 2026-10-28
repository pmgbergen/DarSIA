"""Drawn fixture roles, bound versions and unchanged old-role exports."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import geometry_import, layer_painting as lp
from darsia.fluidflower import layer_painting_fixtures as fixtures
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower.geometry_raster_sources import CANONICAL_EDIT
from darsia.fluidflower.published_dataset import simulator_geometry as simulator
from darsia.fluidflower.series_geometry import RigFrame

from .test_map_adoption import _native_index

SHAPE, SIZE = (40, 60), (60, 40)
FRAME = RigFrame(corners={"top_left": (-.5, -.5), "top_right": (-.5, 59.5),
    "bottom_right": (39.5, 59.5), "bottom_left": (39.5, -.5)}, width_m=.06, height_m=.04)


def _state(role="colorchecker", *, complete=False):
    session = lp.PaintingSession(run_id="run1", image_size=SIZE)
    session.add_layer("Upper", "porous_medium")
    session.add_layer("Lower", "porous_medium")
    session.add_layer("ColorChecker", role)
    for entry in session.facies.values():
        if entry.role == "porous_medium":
            entry.porosity, entry.permeability = .4, 10.0
    session.snap_mm = 0.0
    session.edge_fill_mm = 0.0
    session.polygons = [
        lp.Polygon(1, [(0, 0), (0, 60 if complete else 19),
                       (20, 60 if complete else 19), (20, 0)]),
        lp.Polygon(2, [(20, 0 if complete else 31), (20, 60),
                       (40, 60), (40, 0 if complete else 31)]),
        lp.Polygon(3, [(15, 20), (15, 30), (25, 30), (25, 20)]),
        lp.Polygon(2, [(16, 23), (16, 25), (18, 25), (18, 23)]),
    ]
    classes = np.ones(SHAPE, np.uint8)
    classes[20:] = 2
    classes[15:26, 20:31] = 3
    proposal = lp.LayerProposal(classes, np.full(SHAPE, 200, np.uint8), SIZE,
        lp.proposal_digest(session, FRAME), frame=FRAME.to_dict(),
        layers=[dict(layer_id=i, pixels=int((classes == i).sum())) for i in (1, 2, 3)])
    return session, proposal


def _version(tmp_path, *, kind="proposal", complete=False):
    session, proposal = _state(complete=complete)
    project = SimpleNamespace(root=tmp_path)
    if kind == "manual":
        saved = manual.manual_map(session, SIZE, frame=FRAME)
        record = manual.save_manual_version(project, session, saved, name="Map")
    else:
        record = lp.save_version(project, session, proposal, [], name="Map")
    record["approval"] = {"sha256": record["sha256"]}
    return record, fixtures.version_fixture_underlay(record, frame=FRAME)


@pytest.mark.parametrize("role", ["porous_medium", "free_water", "excluded"])
@pytest.mark.parametrize("corrected", [False, True])
def test_all_old_role_exports_equal_the_frozen_original_byte_for_byte(role, corrected):
    session, proposal = _state(role)
    correction = [lp.Correction("brush", 2, [(8, 8)], radius_px=2)] if corrected else []
    source = (Path(__file__).parent / "fixtures/painted_label_map_55cb9e93_frozen.txt").read_text()
    namespace = vars(lp).copy()
    exec(compile(source, "frozen_old_export", "exec"), namespace)
    old_labels, old_regions = namespace["painted_label_map"](session, proposal, frame=FRAME,
                                                           corrections=correction)
    labels, regions = lp.painted_label_map(session, proposal, frame=FRAME, corrections=correction)
    assert labels.dtype == old_labels.dtype and labels.tobytes() == old_labels.tobytes()
    assert json.dumps(regions, sort_keys=True).encode() == json.dumps(old_regions, sort_keys=True).encode()
    assert fixtures.fixture_underlay(session, proposal.classes, frame=FRAME) is None


def test_colorchecker_is_explicit_zero_porosity_and_disabled_not_a_name_guess():
    session, proposal = _state()
    loaded = lp.PaintingSession.from_dict(json.loads(json.dumps(session.to_dict())))
    assert loaded.facies_of(3).role == "colorchecker"
    assert loaded.facies_of(3).effective_porosity() == 0.0
    _labels, regions = lp.painted_label_map(loaded, proposal, frame=FRAME)
    own = [row for row in regions if row["analysis_role"] == "colorchecker"]
    assert own and all(row["porosity"] == 0.0 and not row["enabled"] for row in own)
    source = {"facies": {"9": {"labels": [3]}}}
    properties = {9: {"name": "ColorChecker", "porosity": .7, "permeability": 99}}
    row = geometry_import._region_rows(source, properties, [3], {9: "colorchecker"})[0]
    assert row["analysis_role"] == "colorchecker" and row["porosity"] == 0 and not row["enabled"]
    assert row["permeability"] is None and properties[9]["porosity"] == .7
    old = geometry_import._region_rows(source, properties, [3], {9: "excluded"})[0]
    assert old["analysis_role"] == "excluded" and old["porosity"] == .7


def test_drawn_underlay_preserves_explicit_top_priority_and_all_vertices():
    session, proposal = _state()
    session.polygons.append(lp.Polygon(1, [(14, 19), (14, 31), (26, 31), (26, 19)]))
    ordinary = fixtures.fixture_underlay(session, proposal.classes, frame=FRAME)
    assert ordinary.drawn_layers[17, 24] == 2
    session.layers[0] = replace(session.layers[0], always_on_top=True)
    before = deepcopy(session.to_dict())
    priority = fixtures.fixture_underlay(session, proposal.classes, frame=FRAME)
    assert priority.drawn_layers[17, 24] == 1
    assert priority.analysis_layers[17, 24] == 3
    assert session.to_dict() == before


def test_only_polygons_count_and_fixture_footprint_survives_smaller_sand_polygon():
    session, proposal = _state()
    before = deepcopy(session.to_dict())
    underlay = fixtures.fixture_underlay(session, proposal.classes, frame=FRAME)
    assert underlay.analysis_layers[17, 24] == 3
    assert underlay.drawn_layers[17, 24] == 2
    assert underlay.fixture_mask[22, 25] and underlay.drawn_layers[22, 25] == 0
    # The classifier calls this lower sand; it is still not a drawing.
    changed = proposal.classes.copy()
    changed[22, 25] = 2
    assert fixtures.fixture_underlay(session, changed, frame=FRAME).drawn_layers[22, 25] == 0
    session.strokes.append(lp.Stroke(1, 5, [(22, 25)]))
    assert fixtures.fixture_underlay(session, changed, frame=FRAME).drawn_layers[22, 25] == 0
    session.strokes.clear()
    assert session.to_dict() == before


def test_smaller_drawn_sand_polygon_wins_without_using_the_fixture_or_classifier():
    session, proposal = _state()
    session.polygons.append(lp.Polygon(1, [(16, 23), (16, 24), (17, 24), (17, 23)]))
    underlay = fixtures.fixture_underlay(session, proposal.classes, frame=FRAME)
    assert underlay.drawn_layers[16, 23] == 1 and underlay.analysis_layers[16, 23] == 3


def test_continuation_never_replaces_drawn_sand_and_flags_partition_the_fixture():
    session, proposal = _state()
    underlay = fixtures.fixture_underlay(session, proposal.classes, frame=FRAME)
    _x, y, _area = simulator.canvas_coordinates(SHAPE, .06, .04)
    result = fixtures.continue_fixture_underlay(underlay.analysis_layers,
        {1: "porous_medium", 2: "porous_medium", 3: "colorchecker"},
        underlay.fixture_mask, underlay.drawn_layers, y)
    assert result["layers"][17, 24] == 2 and result["drawn_behind"][17, 24]
    assert not np.any(result["drawn_behind"] & result["fixture_filled"])
    assert np.array_equal(result["drawn_behind"] | result["fixture_filled"] | result["unfilled"],
                          underlay.fixture_mask)
    assert sum(result["counts"].values()) == int(underlay.fixture_mask.sum())


def test_no_donor_remains_unfilled_and_colorchecker_cannot_be_its_own_underlay():
    labels = np.full((4, 6), 3, np.uint8)
    hidden = np.ones(labels.shape, bool)
    drawn = np.zeros_like(labels)
    heights = np.broadcast_to(np.arange(4, 0, -1)[:, None], labels.shape)
    result = fixtures.continue_fixture_underlay(labels, {3: "colorchecker"}, hidden, drawn, heights)
    assert result["unfilled"].all() and not result["fixture_filled"].any()
    drawn[1, 1] = 3
    with pytest.raises(ValueError, match="explicit non-ColorChecker role"):
        fixtures.continue_fixture_underlay(labels, {3: "colorchecker"}, hidden, drawn, heights)


@pytest.mark.parametrize("role", ["excluded", "free_water"])
def test_every_explicit_underlying_role_counts_but_nonporous_drawing_never_donates_sand(role):
    session, proposal = _state()
    layer = session.add_layer("Behind fixture", role)
    session.polygons.append(lp.Polygon(layer.layer_id, [(16, 23), (16, 25), (18, 25), (18, 23)]))
    underlay = fixtures.fixture_underlay(session, proposal.classes, frame=FRAME)
    assert underlay.drawn_layers[17, 24] == layer.layer_id
    assert underlay.analysis_layers[17, 24] == 3  # The visible fixture is always on top.
    labels = np.full(SHAPE, 3, np.uint8)  # No porous donor anywhere.
    _x, y, _area = simulator.canvas_coordinates(SHAPE, .06, .04)
    result = fixtures.continue_fixture_underlay(labels,
        {3: "colorchecker", layer.layer_id: role}, underlay.fixture_mask, underlay.drawn_layers, y)
    assert result["layers"][17, 24] == layer.layer_id
    assert result["drawn_behind"][17, 24] and not result["fixture_filled"].any()
    assert np.array_equal(result["unfilled"], underlay.fixture_mask & ~result["drawn_behind"])


def test_fixture_polygon_wins_over_a_smaller_excluded_polygon_and_outside_is_not_a_gap():
    session, proposal = _state()
    layer = session.add_layer("Silicone", "excluded")
    session.polygons.append(lp.Polygon(layer.layer_id, [(16, 23), (16, 24), (17, 24), (17, 23)]))
    smaller = replace(FRAME, corners={"top_left": (1, 1), "top_right": (1, 58),
        "bottom_right": (38, 58), "bottom_left": (38, 1)})
    underlay = fixtures.fixture_underlay(session, proposal.classes, frame=smaller)
    assert underlay.analysis_layers[16, 23] == 3 and underlay.drawn_layers[16, 23] == layer.layer_id
    assert underlay.analysis_layers[0, 0] == lp.OUTSIDE
    assert not underlay.fixture_mask[0, 0] and underlay.drawn_layers[0, 0] == 0


@pytest.mark.parametrize("kind", ["proposal", "manual"])
def test_both_version_kinds_bind_the_real_files_frame_and_drawn_arrays(tmp_path, kind):
    record, underlay = _version(tmp_path, kind=kind, complete=True)
    assert fixtures.checked_for_adoption(underlay, underlay.analysis_layers, record,
                                         frame=FRAME).digest() == underlay.digest()
    with pytest.raises(ValueError, match="independently supplied frame"):
        fixtures.checked_for_adoption(underlay, underlay.analysis_layers, record,
                                       frame=replace(FRAME, width_m=.12))
    underlay.drawn_layers[17, 24] = 1
    with pytest.raises(ValueError, match="replay"):
        fixtures.checked_for_adoption(underlay, underlay.analysis_layers, record, frame=FRAME)


def test_a_free_preview_and_changed_frozen_session_cannot_be_adopted(tmp_path):
    session, proposal = _state()
    preview = fixtures.fixture_underlay(session, proposal.classes, frame=FRAME)
    with pytest.raises(ValueError, match="exact approved"):
        fixtures.checked_for_adoption(preview, preview.analysis_layers,
            {"sha256": "a" * 64, "approval": {"sha256": "a" * 64}}, frame=FRAME)
    record, _underlay = _version(tmp_path)
    path = Path(record["folder"]) / "session.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed since"):
        fixtures.version_fixture_underlay(record, frame=FRAME)


def test_fixture_adoption_needs_a_complete_independent_frame_and_own_approval(tmp_path):
    record, underlay = _version(tmp_path)
    for frame in (None, replace(FRAME, corners={}), replace(FRAME, width_m=None)):
        with pytest.raises(ValueError, match="independently supplied full rig frame"):
            fixtures.checked_for_adoption(underlay, underlay.analysis_layers, record, frame=frame)
    unapproved = dict(record, approval={})
    with pytest.raises(ValueError, match="approval of its own SHA"):
        fixtures.checked_for_adoption(underlay, underlay.analysis_layers, unapproved, frame=FRAME)


def test_drawn_polygons_cannot_silently_be_interpreted_on_a_reduced_grid():
    session, proposal = _state()
    with pytest.raises(ValueError, match="full reference photograph"):
        fixtures.fixture_underlay(session, proposal.classes[::2, ::2], frame=FRAME)


def test_preparation_projects_and_binds_every_run_without_changing_analysis_semantics(tmp_path):
    record, underlay = _version(tmp_path / "versions")
    prepared = adoption.prepare_adoption(tmp_path / "adoption", underlay.analysis_layers,
        reference_run="run1", geometry_index=_native_index(tmp_path / "geometry"), runs=["run1", "run2"],
        roles={1: "porous_medium", 2: "porous_medium", 3: "colorchecker"},
        version=record, fixture_underlay=underlay, fixture_frame=FRAME)
    folder = Path(prepared["folder"])
    assert prepared["source_fixture_underlay"]["sha256"]
    assert all(row["fixture_underlay"]["sha256"] for row in prepared["runs"])
    table = tmp_path / "facies.csv"
    table.write_text("id,name,porosity,permeability\n1,Upper,.4,10\n2,Lower,.3,20\n3,Fixture,.7,90\n")
    current = {"configuration": {"source_mode": CANONICAL_EDIT,
        "canonical_import_parent": {"revision_id": "parent"},
        "facies_properties_source_path": str(table),
        "frame_binding": {"display_reference_run": "run1", "runs": {
            run: {"fields": {"shape_corrected_baseline": {}}} for run in ("run1", "run2")}}}}
    config, _summary = adoption.adoption_candidate(current, folder, {1: 1, 2: 2, 3: 3})
    fixture = next(row for row in config["regions"] if row["label_id"] == 3)
    assert fixture["analysis_role"] == "colorchecker" and fixture["porosity"] == 0 and not fixture["enabled"]
    assert set(config["painted_fixture_underlays"]) == {"run1", "run2"}
    for row in prepared["runs"]:
        with np.load(folder / row["labels"]["path"], allow_pickle=True) as saved:
            result = fixtures.adopted_underlay(config, row["run_id"], saved["array"])
        assert result["fixture_mask"].any() and result["drawn_facies"].any()
    with np.load(folder / prepared["runs"][0]["labels"]["path"], allow_pickle=True) as saved:
        labels = saved["array"].copy()
    lost = deepcopy(config)
    lost.pop("painted_fixture_underlays")
    with pytest.raises(ValueError, match="lost its prepared fixture underlay"):
        fixtures.adopted_underlay(lost, "run1", labels)
    changed_role = deepcopy(config)
    next(row for row in changed_role["regions"] if row["label_id"] == 3)["analysis_role"] = "excluded"
    with pytest.raises(ValueError, match="unchanged explicit role"):
        fixtures.adopted_underlay(changed_role, "run1", labels)
    per_layer = deepcopy(config)
    per_layer["grouping_mode"] = "per_layer"
    for region in per_layer["regions"]:
        if region["label_id"] in (1, 2):
            region["facies_id"] = 3 - region["label_id"]
    for bound in per_layer["painted_fixture_underlays"].values():
        bound["layer_to_facies"].update({"1": 2, "2": 1})
    per_layer_result = fixtures.adopted_underlay(per_layer, "run1", labels)
    assert per_layer_result["drawn_facies"][12, 24] == 2
    per_layer["grouping_mode"] = "shared_facies"
    assert fixtures.adopted_underlay(per_layer, "run1", labels)["drawn_facies"][12, 24] == 1
    with pytest.raises(ValueError, match="cannot share one facies"):
        adoption.adoption_candidate(current, folder, {1: 1, 2: 2, 3: 1})
    path = folder / prepared["runs"][0]["fixture_underlay"]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with np.load(folder / prepared["runs"][0]["labels"]["path"], allow_pickle=True) as saved:
        with pytest.raises(ValueError, match="underlay changed"):
            fixtures.adopted_underlay(config, "run1", saved["array"])


def test_optional_simulator_underlay_keeps_drawn_pixels_and_default_flags_unchanged():
    labels = np.ones(SHAPE, np.int32)
    labels[20:] = 2
    labels[15:25, 20:30] = 3
    roles = {1: "porous_medium", 2: "porous_medium", 3: "colorchecker"}
    args = dict(width_m=.06, height_m=.04, inner_frame="canvas", sand_surface="highest_sand")
    old = simulator.complete_geometry(labels, roles, **args)
    drawn = np.zeros(SHAPE, np.int32)
    drawn[17, 24] = 2
    new = simulator.complete_geometry(labels, roles, drawn_underlay={
        "fixture_mask": labels == 3, "drawn_facies": drawn}, **args)
    assert old.drawn_behind is None and "drawn_behind_cells" not in old.counts
    assert new.facies[17, 24] == 2 and new.drawn_behind[17, 24]
    assert not new.fixture_filled[17, 24] and not new.unfilled.any()
    assert np.array_equal(new.facies[labels != 3], old.facies[labels != 3])
    assert new.counts["drawn_behind_cells"] == 1


def test_default_simulator_export_equals_frozen_original_bytes():
    from .published_dataset.test_simulator_geometry import layered, ROLES, TABLE, FLAT_DEPTH, DECLARED
    frozen = (Path(__file__).parent / "fixtures/simulator_geometry_55cb9e93_frozen.txt").read_text()
    namespace = vars(simulator).copy()
    exec(compile(frozen, "frozen_simulator_default", "exec"), namespace)
    args = dict(width_m=.06, height_m=.04, depth_points=FLAT_DEPTH, resolution_mm=2.0,
                fixture_polygons_m={}, **DECLARED)
    expected = namespace["export_geometry"](layered(), ROLES, TABLE, **args)
    actual = simulator.export_geometry(layered(), ROLES, TABLE, **args)
    assert actual == expected


def test_simulator_export_reports_new_disjoint_flags_without_changing_analysis_labels():
    import io
    from .published_dataset.test_simulator_geometry import TABLE, FLAT_DEPTH
    labels = np.ones(SHAPE, np.int32)
    labels[20:] = 2
    labels[15:25, 20:30] = 4
    before = labels.copy()
    drawn = np.zeros(SHAPE, np.int32)
    drawn[17, 24] = 2
    data, metadata, counts, _warnings = simulator.export_geometry(labels,
        {1: "porous_medium", 2: "porous_medium", 4: "colorchecker"}, TABLE,
        width_m=.06, height_m=.04, depth_points=FLAT_DEPTH, resolution_mm=1.0,
        inner_frame="canvas", sand_surface="highest_sand",
        drawn_underlay={"fixture_mask": labels == 4, "drawn_facies": drawn})
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        assert saved["drawn_behind"].any()
        assert not np.any(saved["drawn_behind"] & saved["fixture_filled"])
    assert counts["drawn_behind_cells"] == 1
    assert json.loads(metadata)["fixture_boundary_agreement"]["status"] == "unavailable"
    assert np.array_equal(labels, before)


def test_simulator_retains_drawn_excluded_and_water_without_porous_volume_or_continuation():
    from .published_dataset.test_simulator_geometry import TABLE, FLAT_DEPTH
    labels = np.ones(SHAPE, np.int32)
    labels[20:] = 2
    labels[15:25, 20:30] = 4
    roles = {1: "porous_medium", 2: "porous_medium", 4: "colorchecker",
             5: "excluded", 6: "free_water"}
    drawn = np.zeros(SHAPE, np.int32)
    drawn[17, 24], drawn[18, 24], drawn[17, 25] = 5, 6, 2
    completed = simulator.complete_geometry(labels, roles, width_m=.06, height_m=.04,
        inner_frame="canvas", sand_surface="highest_sand",
        drawn_underlay={"fixture_mask": labels == 4, "drawn_facies": drawn})
    assert completed.facies[17, 24] == 5 and completed.facies[18, 24] == 6
    assert not completed.domain[17, 24] and not completed.domain[18, 24]
    assert completed.facies[17, 25] == 2 and completed.domain[17, 25]
    assert completed.counts["drawn_behind_cells"] == 3
    assert not np.any(completed.drawn_behind & completed.fixture_filled)
    assert not np.any(completed.drawn_behind & completed.unfilled)
    arrays, _metadata = simulator.grid_products(completed, width_m=.06, height_m=.04,
        depth_points=FLAT_DEPTH, table=TABLE, resolution_mm=1.0, precision="float64")
    assert arrays["drawn_behind_facies_ids"].tolist() == [2, 5, 6]
    assert arrays["drawn_behind_area_fraction"].sum() == pytest.approx(3)
    assert arrays["domain_area_m2"].sum() == pytest.approx((40 * 60 - 2) * 1e-6)


def test_polygon_agreement_is_report_only_and_absence_is_unavailable():
    polygons = {"fixture": [(.02, .015), (.03, .015), (.03, .025), (.02, .025)]}
    mask = simulator.fixture_mask(SHAPE, .06, .04, polygons)
    exact = fixtures.fixture_boundary_agreement(mask, polygons, width_m=.06, height_m=.04)
    assert exact["maximum_boundary_distance_mm"] == 0 and exact["report_only"]
    before = mask.copy()
    absent = fixtures.fixture_boundary_agreement(mask, None, width_m=.06, height_m=.04)
    assert absent["status"] == "unavailable" and absent["maximum_boundary_distance_mm"] is None
    assert np.array_equal(mask, before)


def test_measured_fixture_polygons_never_enlarge_the_frozen_role_footprint():
    labels = np.ones(SHAPE, np.int32)
    labels[20:] = 2
    labels[15:25, 20:30] = 3
    drawn = np.zeros(SHAPE, np.int32)
    drawn[17, 24] = 2
    args = dict(width_m=.06, height_m=.04, inner_frame="canvas", sand_surface="highest_sand",
        drawn_underlay={"fixture_mask": labels == 3, "drawn_facies": drawn})
    roles = {1: "porous_medium", 2: "porous_medium", 3: "colorchecker"}
    reference = simulator.complete_geometry(labels, roles, fixture_polygons_m=None, **args)
    measured = {"measured fixture": [(.001, .001), (.059, .001), (.059, .039), (.001, .039)]}
    compared = simulator.complete_geometry(labels, roles, fixture_polygons_m=measured, **args)
    for name in ("facies", "domain", "original", "drawn_behind", "fixture_filled", "edge_filled", "unfilled"):
        assert np.array_equal(getattr(compared, name), getattr(reference, name))
    agreement = fixtures.fixture_boundary_agreement(labels == 3, measured, width_m=.06, height_m=.04)
    assert agreement["report_only"] and agreement["maximum_boundary_distance_mm"] > 0
