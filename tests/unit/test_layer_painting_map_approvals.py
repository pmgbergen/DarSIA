"""Approving a saved map version: append-only records bound to the map's SHA-256.

An approval names the version, the map's SHA-256 and the SHA-256 of each of
its files, the reviewer from the series' reviewer setting (or nobody) and the
time. It changes no map, session, proposal or version. Temporary folders and
synthetic data only.
"""

import getpass
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
    session = lp.PaintingSession(run_id="run1", image_size=(30, 20))
    session.layers.append(lp.PaintedLayer(2, "L2", "#888888", "F2"))
    session.layers.append(lp.PaintedLayer(3, "L3", "#999999", "F3"))
    session.polygons.append(lp.Polygon(2, [(2.0, 2.0), (2.0, 10.0), (10.0, 10.0)]))
    classes = np.full((20, 30), 2, dtype=np.uint8)
    classes[:, 15:] = 3
    confidence = np.full(classes.shape, 200, dtype=np.uint8)
    return session, lp.LayerProposal(classes, confidence, (30, 20), "inputs")


def _digests(folder):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(folder).iterdir()
    }


def test_an_approval_names_the_version_its_map_and_the_reviewer(tmp_path):
    project = _Project(tmp_path)
    session, proposal = _state()
    record = lp.save_version(project, session, proposal, [], name="Map")
    before = _digests(record["folder"])
    approval = lp.approve_version(record, reviewer=" Test reviewer ", now="2026-10-21T01:00:00Z")
    assert approval == {
        "format": lp.LAYER_APPROVAL_FORMAT,
        "run_id": "run1",
        "version": Path(record["folder"]).name,
        "name": "Map",
        "kind": "proposal",
        "sha256": record["sha256"],
        "files": record["files"],
        "reviewer": "Test reviewer",
        "approved_utc": "2026-10-21T01:00:00Z",
    }
    path = lp.approvals_path(project, "run1")
    assert path.parent == Path(record["folder"]).parent
    line = path.read_text(encoding="utf-8").splitlines()
    assert len(line) == 1 and json.loads(line[0]) == approval
    # Nothing of the version changes, and the list of versions ignores the file.
    assert _digests(record["folder"]) == before
    assert [row["folder"] for row in lp.list_versions(project, "run1")] == [record["folder"]]
    assert lp.read_map_approvals(project, "run1") == [approval]
    assert lp.map_approval([approval], record["sha256"]) == approval
    assert lp.map_approval([approval], "0" * 64) is None
    assert lp.map_approval([approval], "") is None


def test_approvals_are_appended_and_the_latest_counts(tmp_path):
    project = _Project(tmp_path)
    session, proposal = _state()
    first = lp.save_version(project, session, proposal, [], name="First")
    session.polygons.append(lp.Polygon(3, [(12.0, 2.0), (12.0, 10.0), (18.0, 10.0)]))
    second = lp.save_version(project, session, proposal, [], name="Second")
    one = lp.approve_version(first, now="2026-10-21T01:00:00Z")
    two = lp.approve_version(second, now="2026-10-21T02:00:00Z")
    again = lp.approve_version(first, reviewer="Test reviewer", now="2026-10-21T03:00:00Z")
    assert lp.read_map_approvals(project, "run1") == [one, two, again]
    assert one["reviewer"] is None  # Without a reviewer setting nobody is named.
    assert lp.map_approval(lp.read_map_approvals(project, "run1"), first["sha256"]) == again
    # Without a reviewer the operating-system login is never recorded.
    login = getpass.getuser().strip().lower()
    if login:
        assert login not in lp.approvals_path(project, "run1").read_text(
            encoding="utf-8"
        ).lower()


def test_a_changed_version_is_not_approved(tmp_path):
    project = _Project(tmp_path)
    session, proposal = _state()
    record = lp.save_version(project, session, proposal, [], name="Map")
    (Path(record["folder"]) / "session.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="has changed since it was saved"):
        lp.approve_version(record)
    assert not lp.approvals_path(project, "run1").exists()


def test_a_damaged_approval_file_is_never_read_as_not_approved(tmp_path):
    project = _Project(tmp_path)
    session, proposal = _state()
    record = lp.save_version(project, session, proposal, [], name="Map")
    lp.approve_version(record)
    path = lp.approvals_path(project, "run1")
    path.write_text(path.read_text(encoding="utf-8") + '{"format": "other"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Line 2"):
        lp.read_map_approvals(project, "run1")
    assert lp.read_map_approvals(project, "other_run") == []
