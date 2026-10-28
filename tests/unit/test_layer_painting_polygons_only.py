"""Synthetic native-grid examples only; no experiment files or worker jobs."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import layer_painting_polygons_only as only
from darsia.fluidflower import layer_painting_version_archive as versions
from darsia.fluidflower.series_geometry import RigFrame


def state(*, full=False, inset=False):
    session = lp.PaintingSession(run_id="sample", image_path="sample.png",
                                 image_stamp="sample-stamp", image_size=(20, 12))
    session.layers = [lp.PaintedLayer(1, "Sand", "#888888", "Sand"),
                      lp.PaintedLayer(2, "Water", "#4444ff", "Water")]
    session.facies = {"Sand": lp.FaciesEntry("porous_medium", .35, 10.),
                      "Water": lp.FaciesEntry("free_water")}
    frame = RigFrame(corners={"top_left": (2., 2.) if inset else (0., 0.),
        "top_right": (2., 17.) if inset else (0., 19.),
        "bottom_right": (9., 17.) if inset else (11., 19.),
        "bottom_left": (9., 2.) if inset else (11., 0.)},
        width_m=.2, height_m=.12, run_id="sample", image="sample.png",
        image_size=(20, 12))
    vertices = [(0., 0.), (0., 19.), (11., 19.), (11., 0.)] if full else [
        (3.012345678901, 4.012345678901), (3., 10.), (8., 10.), (8., 4.)]
    session.polygons = [lp.Polygon(1, vertices)]
    return session, frame


def fill_gap(proposal, layer=1):
    row, col = np.argwhere(proposal.inside & (proposal.classes == 0))[0]
    return lp.Correction("fill", layer, [(float(row), float(col))],
                         replaced_layer=0, proposal_digest=proposal.digest)


def saved(tmp_path, session, proposal, corrections=()):
    return versions.save_map_version(SimpleNamespace(root=tmp_path), session,
        proposal, corrections, name="First", method=only.METHOD,
        proposal_writer=only.save_proposal)


def test_native_polygons_smallest_wins_without_classifier_strokes_or_edge_fill(monkeypatch):
    session, frame = state()
    session.polygons += [lp.Polygon(2, [(4., 5.), (4., 7.), (6., 7.), (6., 5.)])]
    session.strokes = [lp.Stroke(2, 500., [(0., 0.)])]
    session.edge_fill_mm = 1000.
    session.snap_mm = 1000.
    session.min_region_cm2 = 1000.
    def forbidden(*args, **kwargs):
        raise AssertionError("Classifier/manual processing must not run")
    for name in ("propose_layers", "rasterize_strokes", "merge_small_regions", "upsample_classes"):
        monkeypatch.setattr(lp, name, forbidden)
    monkeypatch.setattr(manual, "manual_map", forbidden)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    expected = lp.rasterize_polygons(session.polygons, (12, 20), (1., 1.))
    np.testing.assert_array_equal(proposal.classes, expected)
    assert proposal.classes[5, 6] == 2
    assert proposal.classes[0, 0] == 0
    assert proposal.classes.shape == (12, 20)
    assert proposal.uncovered_pixels() > 0
    assert proposal.stroke_count == proposal.merged_regions == proposal.snap_px == 0


def test_digest_and_currentness_bind_unrounded_vertices_full_frame_and_source_only():
    session, frame = state()
    proposal = only.make_proposal((20, 12), session, frame=frame)
    assert only.proposal_current(proposal, session, frame)
    session.strokes.append(lp.Stroke(2, 50., [(5., 5.)]))
    session.snap_mm = 99.
    session.edge_fill_mm = 99.
    session.smoothing = "high"
    assert only.proposal_current(proposal, session, frame)
    session.polygons[0].points[0] = (3.012345678902, 4.012345678901)
    assert not only.proposal_current(proposal, session, frame)
    other = only.make_proposal((20, 12), session, frame=frame)
    assert proposal.digest != other.digest
    frame.source += " changed"
    assert not only.proposal_current(other, session, frame)
    session.image_stamp += " changed"
    assert not only.proposal_current(proposal, session, frame)


def test_gap_fill_is_clipped_to_full_inside_and_restore_can_reopen_gap():
    session, frame = state(inset=True)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    before = proposal.classes.copy()
    fill = fill_gap(proposal, 2)
    classes, decided, skipped = only.corrected_classes(proposal, [fill], layer_ids={1, 2})
    assert not skipped and np.any(decided)
    assert not np.any(classes[~proposal.inside])
    assert only.gap_count(proposal, [fill], {1, 2}) == 0
    row, col = fill.points[0]
    restore = lp.Correction("restore", 0, [(row, col)], radius_px=2.)
    assert only.gap_count(proposal, [fill, restore], {1, 2}) > 0
    np.testing.assert_array_equal(proposal.classes, before)


def test_brush_and_lasso_can_assign_inside_zero_pixels_and_never_exterior():
    session, frame = state(inset=True)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    brush = lp.Correction("brush", 2, [(2., 3.)], radius_px=4.)
    lasso = lp.Correction("lasso", 2, [(0., 0.), (0., 19.), (11., 19.), (11., 0.)])
    for correction in (brush, lasso):
        classes, _decided, skipped = only.corrected_classes(proposal, [correction])
        assert not skipped and np.any((proposal.classes == 0) & (classes == 2))
        assert not np.any(classes[~proposal.inside])
    assert only.gap_count(proposal, [lasso]) == 0


def test_roundtrip_keeps_exact_vertices_and_legacy_loader_refuses(tmp_path):
    session, frame = state(inset=True)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    path = tmp_path / "proposal.npz"
    only.save_proposal(proposal, path)
    restored = only.load_proposal(path)
    assert restored.source == proposal.source
    assert restored.source["polygons"][0][1][0][0] == 3.012345678901
    np.testing.assert_array_equal(restored.inside, proposal.inside)
    np.testing.assert_array_equal(restored.classes, proposal.classes)
    with pytest.raises(ValueError, match="Unsupported"):
        lp.LayerProposal.load(path)


@pytest.mark.parametrize("change", ["dtype", "shape", "inside", "classes", "method", "schema", "digest", "polygon", "frame"])
def test_tampered_archive_is_refused_without_casting_or_accepting_forged_inside(tmp_path, change):
    session, frame = state(inset=True)
    path = tmp_path / "proposal.npz"
    only.save_proposal(only.make_proposal((20, 12), session, frame=frame), path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    meta = json.loads(str(values["meta"]))
    if change == "dtype":
        values["classes"] = values["classes"].astype(np.int16)
    elif change == "shape":
        values["classes"] = values["classes"][:1]
    elif change == "inside":
        values["inside"][0, 0] = True
        meta["digest"] = only._digest(meta["source"], values["classes"], values["confidence"], values["inside"])
    elif change == "classes":
        values["classes"][4, 5] = 2
        meta["digest"] = only._digest(meta["source"], values["classes"], values["confidence"], values["inside"])
    elif change == "polygon":
        meta["source"]["polygons"][0][0] = 255
    elif change == "frame":
        meta["source"]["frame"]["corners"]["top_left"][0] += .0000001
    else:
        meta[change] = 2 if change == "schema" else "wrong"
    values["meta"] = np.asarray(json.dumps(meta))
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises((ValueError, KeyError)):
        only.load_proposal(path)


def test_gaps_block_exact_export_approval_and_revision_before_writes(tmp_path):
    session, frame = state()
    proposal = only.make_proposal((20, 12), session, frame=frame)
    record = saved(tmp_path, session, proposal)
    with pytest.raises(ValueError, match="unassigned"):
        only.painted_label_map(session, proposal, frame=frame)
    with pytest.raises(ValueError, match="unassigned"):
        only.approve_version(record)
    reference = SimpleNamespace(baseline_image=tmp_path / "sample.png", width_m=.2, height_m=.12)
    assert any("unassigned" in issue for issue in only.check_ready_for_revision(session, proposal, frame, reference))
    with pytest.raises(ValueError, match="unassigned"):
        only.save_painted_revision(SimpleNamespace(root=tmp_path), session, proposal, reference, frame=frame)
    assert not (tmp_path / "overrides").exists()
    assert not (Path(record["folder"]).parent / "approvals.jsonl").exists()


def test_corrected_saved_version_is_exact_current_and_approvable(tmp_path, monkeypatch):
    session, frame = state(inset=True)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    correction = fill_gap(proposal, 2)
    record = saved(tmp_path, session, proposal, [correction])
    restored_session, restored_proposal, corrections = only.load_version(record)
    assert only.proposal_current(restored_proposal, restored_session, frame)
    assert restored_session.polygons[0].points == session.polygons[0].points
    expected = only.corrected_classes(proposal, [correction])[0]
    def forbidden(*args, **kwargs):
        raise AssertionError("No old classifier export route")
    monkeypatch.setattr(lp, "painted_label_map", forbidden)
    monkeypatch.setattr(lp, "upsample_classes", forbidden)
    monkeypatch.setattr(lp, "merge_small_regions", forbidden)
    loaded_session, classes = only.version_layers(record, frame=frame)
    np.testing.assert_array_equal(classes, expected)
    labels, rows = only.painted_label_map(loaded_session, restored_proposal,
                                         frame=frame, corrections=corrections)
    table = np.zeros(int(labels.max()) + 1, np.uint8)
    for row in rows:
        table[row["label_id"]] = row["layer_id"]
    np.testing.assert_array_equal(table[labels], expected)
    assert only.approve_version(record)["sha256"] == record["sha256"]
    wrong_frame = deepcopy(frame)
    wrong_frame.width_m += 1e-12
    with pytest.raises(ValueError, match="frame changed"):
        only.version_layers(record, frame=wrong_frame)


@pytest.mark.parametrize("bad_history", [
    b'{broken\n', b'\xff\n', b'[]\n', b'{"format":"other"}\n',
    ('{"format":"%s","format":"%s"}\n'
     % (lp.LAYER_APPROVAL_FORMAT, lp.LAYER_APPROVAL_FORMAT)).encode("utf8"),
    ('{"format":"%s","value":NaN}\n' % lp.LAYER_APPROVAL_FORMAT).encode("utf8"),
])
def test_direct_approval_refuses_corrupt_history_without_appending(
        tmp_path, monkeypatch, bad_history):
    session, frame = state(full=True)
    record = saved(tmp_path, session, only.make_proposal((20, 12), session, frame=frame))
    history = Path(record["folder"]).parent / "approvals.jsonl"
    history.write_bytes(bad_history)
    def forbidden(*args, **kwargs):
        raise AssertionError("Corrupt history must fail before map validation")
    monkeypatch.setattr(only, "version_layers", forbidden)
    with pytest.raises(ValueError, match="approvals history"):
        only.approve_version(record)
    assert history.read_bytes() == bad_history


def test_direct_approval_bounds_history_before_appending(tmp_path, monkeypatch):
    session, frame = state(full=True)
    record = saved(tmp_path, session, only.make_proposal((20, 12), session, frame=frame))
    history = Path(record["folder"]).parent / "approvals.jsonl"
    raw = b" " * 32
    history.write_bytes(raw)
    monkeypatch.setattr(only, "_MAX_APPROVAL_HISTORY", len(raw) - 1)
    with pytest.raises(ValueError, match="history exceeds"):
        only.approve_version(record)
    assert history.read_bytes() == raw


def test_direct_approval_preserves_compatible_events_and_final_line(tmp_path):
    session, frame = state(full=True)
    record = saved(tmp_path, session, only.make_proposal((20, 12), session, frame=frame))
    history = Path(record["folder"]).parent / "approvals.jsonl"
    event = dict(format=lp.LAYER_APPROVAL_FORMAT, event="note", run_id=session.run_id,
                 version_sha256=record["sha256"], note="Retain this event")
    raw = json.dumps(event).encode("utf8")  # A valid last line need not end in LF.
    history.write_bytes(raw)
    approval = only.approve_version(record)
    assert history.read_bytes().startswith(raw + b"\n")
    rows = lp.read_map_approvals(SimpleNamespace(root=tmp_path), session.run_id)
    assert rows == [event, approval]
    assert lp.map_approval(rows[:1], record["sha256"]) is None
    assert lp.map_approval(rows, record["sha256"]) == approval


def test_direct_approval_revalidates_history_on_transient_retry(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import job_queue

    session, frame = state(full=True)
    record = saved(tmp_path, session, only.make_proposal((20, 12), session, frame=frame))
    history = Path(record["folder"]).parent / "approvals.jsonl"
    corrupt = b'{interrupted-history\n'
    calls = []
    def interrupted_flush(descriptor):
        calls.append(descriptor)
        # Model history damaged during a transient failure. A retry may not
        # append another approval after that damage.
        history.write_bytes(corrupt)
        raise PermissionError("Synthetic transient history failure")
    retry = job_queue.retry_transient
    monkeypatch.setattr(job_queue, "retry_transient",
                        lambda action: retry(action, delays=(0,), wait=lambda delay: False))
    monkeypatch.setattr(only.os, "fsync", interrupted_flush)
    with pytest.raises(ValueError, match="approvals history"):
        only.approve_version(record)
    assert len(calls) == 1
    assert history.read_bytes() == corrupt


def test_invalid_fill_blocks_export_even_when_base_polygons_cover_frame():
    session, frame = state(full=True)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    wrong = lp.Correction("fill", 2, [(5., 5.)], replaced_layer=2)
    assert only.corrected_classes(proposal, [wrong])[2] == [0]
    with pytest.raises(ValueError, match="no longer apply"):
        only.painted_layers(session, proposal, frame=frame, corrections=[wrong])


@pytest.mark.parametrize("change", ["file", "selection", "map_digest", "marker"])
def test_saved_version_checks_file_selection_content_and_method(tmp_path, change):
    session, frame = state(full=True)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    record = saved(tmp_path, session, proposal)
    if change == "file":
        path = Path(record["folder"]) / "session.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "selection":
        record["name"] = "Different"
    elif change == "map_digest":
        record["sha256"] = "0" * 64
        path = Path(record["folder"]) / "version.json"
        value = json.loads(path.read_text())
        value["sha256"] = record["sha256"]
        path.write_text(json.dumps(value))
    else:
        record.pop("proposal_method")
    with pytest.raises(ValueError):
        only.load_version(record)


def test_old_methods_delegate_to_original_functions(monkeypatch, tmp_path):
    session, frame = state(full=True)
    old = lp.LayerProposal(np.ones((12, 20), np.uint8), np.full((12, 20), 255, np.uint8),
                            (20, 12), "old")
    token = object()
    for name in ("corrected_classes", "painted_label_map", "check_ready_for_revision",
                 "load_version", "approve_version", "save_painted_revision"):
        monkeypatch.setattr(lp, name, lambda *a, **kw: token)
    monkeypatch.setattr(manual, "painted_layers", lambda *a, **kw: token)
    monkeypatch.setattr(manual, "version_layers", lambda *a, **kw: token)
    assert only.corrected_classes(old, []) is token
    assert only.painted_label_map(session, old, frame=frame) is token
    assert only.painted_layers(session, old, frame=frame) is token
    assert only.check_ready_for_revision(session, old, frame, None) is token
    assert only.load_version({}) is token
    assert only.approve_version({}) is token
    assert only.version_layers({}, frame=frame) is token
    assert only.save_painted_revision(None, session, old, None, frame=frame) is token
    monkeypatch.setattr(lp, "proposal_digest", lambda *a: "old")
    assert only.proposal_current(old, session, frame)
    path = tmp_path / "legacy.npz"
    old.save(path)
    assert not only.is_polygons_only(only.load_proposal(path))


def test_readiness_preserves_physical_properties_and_native_region_gate(monkeypatch):
    session, frame = state(full=True)
    proposal = only.make_proposal((20, 12), session, frame=frame)
    reference = SimpleNamespace(baseline_image=Path("sample.png"), width_m=.2, height_m=.12)
    assert only.check_ready_for_revision(session, proposal, frame, reference) == []
    session.facies["Sand"].permeability = None
    assert any("permeability" in row for row in only.check_ready_for_revision(session, proposal, frame, reference))
    monkeypatch.setattr(lp, "NATIVE_MAX_REGIONS", 0)
    assert any("region limit" in row for row in only.check_ready_for_revision(session, proposal, frame, reference))


def test_new_revision_passes_exact_native_regions_to_existing_store(tmp_path, monkeypatch):
    from darsia.fluidflower import geometry_facies as geometry

    session, frame = state(full=True)
    # A deliberate tiny polygon remains a region, regardless of the ordinary
    # classifier's island threshold or smoothing settings.
    session.polygons.append(lp.Polygon(2, [(5.1, 5.1), (5.1, 5.2), (5.2, 5.1)]))
    session.min_region_cm2 = 1000.
    proposal = only.make_proposal((20, 12), session, frame=frame)
    assert 0 < np.count_nonzero(proposal.classes == 2) < 5
    captured = {}
    monkeypatch.setattr(lp, "photo_size", lambda *a, **kw: (20, 12))
    monkeypatch.setattr(lp, "reference_camera_u", lambda *a: None)
    monkeypatch.setattr(lp, "revision_injectors", lambda *a: ([], []))
    monkeypatch.setattr(geometry, "_save_darsia_field",
                        lambda path, value, **kw: captured.update({Path(path).name: value.copy()}))
    monkeypatch.setattr(geometry, "default_geometry_config", lambda project: {})
    monkeypatch.setattr(geometry, "validate_geometry_config", lambda config: {"issues": []})
    class Store:
        def __init__(self, project):
            pass
        def save_revision(self, config, **kwargs):
            captured["config"] = config
            return {"saved": True}
    monkeypatch.setattr(geometry, "GeometryRevisionStore", Store)
    def forbidden(*a, **kw):
        raise AssertionError("Old resampling/merge path must not run")
    for name in ("painted_label_map", "upsample_classes", "merge_small_regions"):
        monkeypatch.setattr(lp, name, forbidden)
    reference = SimpleNamespace(baseline_image=tmp_path / "sample.png", run_id="sample",
                                 width_m=.2, height_m=.12)
    assert only.save_painted_revision(SimpleNamespace(root=tmp_path), session, proposal,
                                     reference, frame=frame) == {"saved": True}
    config = captured["config"]
    labels = captured["labels.npz"]
    table = np.zeros(int(labels.max()) + 1, np.uint8)
    for row in config["regions"]:
        table[row["label_id"]] = row["layer_id"]
    np.testing.assert_array_equal(table[labels], proposal.classes)
    assert config["facies_generation"]["method"] == only.METHOD
    assert config["facies_generation"]["automatic_edge_fill"] is False
