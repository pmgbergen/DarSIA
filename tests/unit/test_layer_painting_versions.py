"""Saved versions of a painted map: append-only and identified by their SHA-256.

A version holds the session, the proposal and the corrections in its own
folder. Saving never overwrites an earlier version; restoring writes the run's
current files from one. Temporary folders and synthetic data only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import layer_painting as lp


class _Project:
    def __init__(self, root):
        self.root = root


def _state():
    session = lp.PaintingSession(run_id="ac17", image_size=(30, 20))
    session.layers.append(lp.PaintedLayer(2, "L2", "#888888", "F2"))
    session.layers.append(lp.PaintedLayer(3, "L3", "#999999", "F3"))
    session.polygons.append(lp.Polygon(2, [(2.0, 2.0), (2.0, 10.0), (10.0, 10.0)]))
    classes = np.full((20, 30), 2, dtype=np.uint8)
    classes[:, 15:] = 3
    confidence = np.full(classes.shape, 200, dtype=np.uint8)
    return session, lp.LayerProposal(classes, confidence, (30, 20), "inputs")


def _files(folder):
    return {path.name: path.read_bytes() for path in Path(folder).iterdir()}


def test_a_version_keeps_its_files_and_identifies_the_map(tmp_path):
    project = _Project(tmp_path)
    session, proposal = _state()
    record = lp.save_version(project, session, proposal, [], name="  First  ")
    folder = Path(record["folder"])
    assert folder.parent == lp.versions_folder(project, "ac17")
    assert sorted(_files(folder)) == sorted(lp.VERSION_FILES + ("version.json",))
    assert record["name"] == "First" and record["kind"] == "proposal"
    assert record["correction_count"] == 0
    assert record["sha256"] == lp.map_digest(session, proposal, [])
    assert folder.name.endswith(record["sha256"][:12])
    saved = json.loads((folder / "version.json").read_text(encoding="utf-8"))
    assert saved["format"] == lp.LAYER_VERSION_FORMAT and "folder" not in saved
    proposal_bytes = (folder / "proposal.npz").read_bytes()
    assert saved["files"]["proposal.npz"] == hashlib.sha256(proposal_bytes).hexdigest()
    # The same map saved later has the same SHA-256: save times do not count.
    session.updated_utc, session.updated_by = "later", "someone"
    assert lp.map_digest(session, proposal, []) == record["sha256"]
    with pytest.raises(ValueError, match="needs a name"):
        lp.save_version(project, session, proposal, [], name="  ")


def test_versions_are_never_overwritten_and_listed_oldest_first(tmp_path):
    project = _Project(tmp_path)
    session, proposal = _state()
    first = lp.save_version(project, session, proposal, [], name="First")
    before = _files(first["folder"])
    brush = lp.Correction("brush", 3, [(5.0, 5.0)], radius_px=2.0)
    second = lp.save_version(project, session, proposal, [brush], name="Second")
    again = lp.save_version(project, session, proposal, [], name="First again")
    assert len({first["folder"], second["folder"], again["folder"]}) == 3
    assert _files(first["folder"]) == before
    names = [row["name"] for row in lp.list_versions(project, "ac17")]
    assert names == ["First", "Second", "First again"]
    assert second["kind"] == "edited map" and second["sha256"] != first["sha256"]
    assert again["sha256"] == first["sha256"]
    assert lp.list_versions(project, "ac18") == []


def test_restoring_a_version_writes_the_current_files_after_checking_them(tmp_path):
    project = _Project(tmp_path)
    session, proposal = _state()
    brush = lp.Correction("brush", 3, [(5.0, 5.0)], radius_px=2.0)
    record = lp.save_version(project, session, proposal, [brush], name="Edited")
    loaded_session, loaded_proposal, loaded_corrections = lp.load_version(record)
    assert loaded_session.polygons == session.polygons
    assert np.array_equal(loaded_proposal.classes, proposal.classes)
    assert loaded_corrections == [brush]
    lp.restore_version(project, record)
    assert lp.load_corrections(project, "ac17") == [brush]
    assert np.array_equal(lp.load_proposal(project, "ac17").classes, proposal.classes)
    text = lp.session_path(project, "ac17").read_text(encoding="utf-8")
    assert lp.PaintingSession.from_dict(json.loads(text)).polygons == session.polygons
    # A version whose files changed after saving is refused.
    changed = Path(record["folder"]) / "corrections.json"
    changed.write_text(changed.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since it was saved"):
        lp.load_version(record)


def test_restoring_a_proposal_writes_no_correction_file(tmp_path):
    session, proposal = _state()
    record = lp.save_version(_Project(tmp_path / "a"), session, proposal, [], name="P")
    target = _Project(tmp_path / "b")
    lp.restore_version(target, record)
    assert lp.load_proposal(target, "ac17") is not None
    assert not lp.corrections_path(target, "ac17").exists()
