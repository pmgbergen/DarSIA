"""Optional converter Protocols visibility; saved status stays non-numerical."""
from pathlib import Path

import pytest

from darsia.fluidflower.workflow_row_status import WorkflowStatusContext, _protocols


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def context(tmp_path):
    root = tmp_path / "series"
    folder = root / "raw/one"
    folder.mkdir(parents=True)
    imaging = write(root / "protocols/one/imaging_protocol.csv", b"timestamp\n2026-01-01\n")
    return WorkflowStatusContext(root, run_id="one", config=dict(
        data=dict(folder=str(folder)), protocols=dict(imaging={str(folder): str(imaging)})),
        manifest=dict(paths=dict(protocols="protocols")))


def test_protocol_blacklist_and_unknown_files_are_visible_but_not_completion_inputs(tmp_path):
    current = context(tmp_path)
    original = _protocols(current)
    assert original.state == "done"
    blacklist = write(current.series_root / "protocols/one/blacklist_protocol.csv",
                      b"path\nimage1.JPG\nimage2.JPG\n")
    report = write(current.series_root / "protocols/one/EXPORT_REPORT.json", b"{}")
    before = {path: path.read_bytes() for path in (blacklist, report)}
    actual = _protocols(current)
    assert actual.state == original.state
    assert actual.artifacts == original.artifacts
    assert "blacklist_protocol.csv (2 image blacklist rows)" in actual.summary
    assert "EXPORT_REPORT.json" in actual.summary
    assert "ignored for completion" in actual.summary
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_unreadable_extra_metadata_does_not_make_valid_protocols_missing(tmp_path):
    current = context(tmp_path)
    write(current.series_root / "protocols/one/blacklist_protocol.csv", b"\xff")
    actual = _protocols(current)
    assert actual.state == "done"
    assert "Extra protocol metadata not inspected" in actual.summary


def test_missing_imaging_mapping_still_reports_extra_metadata_without_changing_state(tmp_path):
    from dataclasses import replace
    from darsia.fluidflower.workflow_row_status import _configured_protocols

    current = context(tmp_path)
    current.config["protocols"] = {}
    write(current.series_root / "protocols/one/blacklist_protocol.csv", b"path\nimage1.JPG\n")
    original = _configured_protocols(current)
    actual = _protocols(current)
    assert original.state == actual.state == "missing"
    assert "blacklist_protocol.csv (1 image blacklist rows)" in actual.summary
    assert "ignored for completion" in actual.summary
    # Only the explanatory summary changes; all status/evidence fields survive.
    assert replace(actual, summary=original.summary) == original
