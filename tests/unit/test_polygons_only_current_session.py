"""Exact current drafts and immutable versions, in temporary packages only."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_polygons_only as polygons
from darsia.fluidflower import layer_painting_version_archive as versions


def draft():
    session = lp.PaintingSession.default("sample", "sample.png", (12, 8))
    session.image_stamp = "unchanged"
    session.polygons = [lp.Polygon(2, [(1.123456789, 1.987654321),
                                      (1., 5.), (6., 5.), (6., 1.)])]
    return session


def test_current_roundtrip_keeps_exact_vertices_without_adopting_different_inputs(tmp_path):
    project = SimpleNamespace(root=tmp_path)
    session = draft()
    proposal = polygons.make_proposal((12, 8), session, frame=None)
    path = polygons.save_current_session(project, session)
    loaded = lp.PaintingSession.from_dict(json.loads(path.read_text(encoding="utf8")))
    restored = polygons.checked_current_session(proposal, loaded, None)
    assert restored.polygons == session.polygons
    assert restored.polygons[0].points[0] == (1.123456789, 1.987654321)
    assert restored is not loaded
    stale = deepcopy(loaded)
    stale.polygons[0].points[0] = (1.123456790, 1.987654321)
    with pytest.raises(ValueError, match="current draft differs"):
        polygons.checked_current_session(proposal, stale, None)
    stale = deepcopy(loaded)
    stale.image_stamp = "another acquisition"
    with pytest.raises(ValueError, match="current draft differs"):
        polygons.checked_current_session(proposal, stale, None)


def test_legacy_rounded_draft_is_not_silently_replaced_with_a_proposal_source():
    session = draft()
    proposal = polygons.make_proposal((12, 8), session, frame=None)
    rounded = lp.PaintingSession.from_dict(session.to_dict())
    assert not polygons.proposal_current(proposal, rounded, None)
    with pytest.raises(ValueError):
        polygons.checked_current_session(proposal, rounded, None)
    assert rounded.polygons[0].points[0] == (1.1, 2.0)


def test_version_metadata_reports_actual_gap_count_and_strict_export_remains_default(tmp_path):
    project = SimpleNamespace(root=tmp_path / "new-package")
    session = draft()
    proposal = polygons.make_proposal((12, 8), session, frame=None)
    record = versions.save_map_version(project, session, proposal, [], name="Unfinished",
                                       method=polygons.METHOD, proposal_writer=polygons.save_proposal)
    assert record["uncovered_pixels"] == polygons.gap_count(proposal) > 0
    assert polygons.painted_layers(session, proposal, frame=None, require_complete=False).shape == (8, 12)
    with pytest.raises(ValueError, match="unassigned"):
        polygons.painted_layers(session, proposal, frame=None)
    with pytest.raises(ValueError, match="unassigned"):
        polygons.approve_version(record)


def test_bad_history_keeps_verified_versions_unknown_and_refuses_append_or_move(tmp_path):
    project = SimpleNamespace(root=tmp_path)
    session = draft()
    record = versions.save_drawing_version(project, session, name="Preserved drawing")
    log = lp.approvals_path(project, session.run_id)
    log.write_bytes(b"not json\n")
    rows = versions.list_version_metadata(project, session.run_id)
    assert len(rows) == 1 and rows[0]["sha256"] == record["sha256"]
    assert rows[0]["folder"] == record["folder"]
    assert rows[0]["history_error"] and rows[0]["approved"] is None
    assert rows[0]["approval_known"] is False
    from pathlib import Path
    folder = Path(record["folder"])
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    for action in (lambda: versions.rename_version(project, session.run_id, record, "Changed"),
                   lambda: versions.set_version_note(project, session.run_id, record, "A note"),
                   lambda: versions.archive_version(project, session.run_id, record)):
        with pytest.raises(ValueError):
            action()
        assert log.read_bytes() == b"not json\n"
        assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
