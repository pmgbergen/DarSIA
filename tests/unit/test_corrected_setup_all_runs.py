"""Apply the current corrected revision to Setup for all runs, with one receipt.

The synthetic two-run corrected layout of the canonical-layout tests: AC17 is
the display reference run and AC29 the other. Each run goes through the
existing per-run export and apply; the display run goes first and the first
run that cannot be applied stops the sequence. No real series is used.
"""

import json
from pathlib import Path

import pytest

from darsia.fluidflower import corrected_setup as subject
from darsia.fluidflower import corrected_setup_all_runs as all_runs

from .test_corrected_setup_canonical_layout import inventory, layout  # noqa: F401


def _expected(layout, row):
    replacements = json.loads((Path(row["export"]) / "export.json").read_text())["binding"][
        "replacements"
    ]
    return subject.corrected_config_bytes(layout["original"][row["run_id"]], replacements)


def test_every_run_is_exported_and_applied_with_one_receipt(layout):
    references = [layout["reference"]("ac29"), layout["reference"]("ac17")]
    messages = []
    path, receipt = all_runs.apply_geometry_to_all_runs(
        layout["project"], references, actor="synthetic", progress=messages.append
    )
    assert [row["run_id"] for row in receipt["runs"]] == ["ac17", "ac29"]
    assert [row["status"] for row in receipt["runs"]] == ["applied", "applied"]
    assert receipt["complete"] and receipt["not_attempted"] == []
    assert receipt["Build_run"] is False and receipt["revision_id"] == "a" * 32
    assert path.parent == layout["project"].root / "exports/native_setup" / ("a" * 32)
    assert json.loads(path.read_text(encoding="utf-8")) == receipt
    assert messages[0] == "Applying the geometry to AC17 (1 of 2)…"
    for row in receipt["runs"]:
        assert Path(row["apply_receipt"]).is_file()
        for kind in ("runs", "merged"):
            config = layout["series"] / "config" / kind / f"{row['run_id']}.toml"
            assert config.read_bytes() == _expected(layout, row)
    # Again: every run's completed export and apply are reused; no configuration is written.
    before = inventory(layout["series"] / "config")
    again, second = all_runs.apply_geometry_to_all_runs(
        layout["project"], references, actor="synthetic"
    )
    assert again != path and second["complete"]
    assert [row["export_reused"] for row in second["runs"]] == [True, True]
    assert inventory(layout["series"] / "config") == before


def test_the_first_run_that_cannot_be_applied_stops_the_sequence(layout):
    references = [layout["reference"](run) for run in ("ac17", "ac99", "ac29")]
    path, receipt = all_runs.apply_geometry_to_all_runs(
        layout["project"], references, actor="synthetic"
    )
    assert [(row["run_id"], row["status"]) for row in receipt["runs"]] == [
        ("ac17", "applied"),
        ("ac99", "stopped"),
    ]
    assert "No corrected rig maps are bound for selected run ac99" in receipt["runs"][1]["error"]
    assert receipt["not_attempted"] == ["ac29"] and not receipt["complete"]
    assert (layout["series"] / "config/merged/ac29.toml").read_bytes() == layout["original"]["ac29"]
    assert json.loads(path.read_text(encoding="utf-8"))["not_attempted"] == ["ac29"]


def test_only_a_corrected_revision_with_its_display_run_is_applied(layout):
    with pytest.raises(ValueError, match="display reference run ac17 is not among"):
        all_runs.apply_order(["ac29"], "ac17")
    assert all_runs.apply_order(["ac29", "ac17", "ac30"], "ac17") == ["ac17", "ac29", "ac30"]
    current = json.loads(layout["current"].read_text())
    current["configuration"]["frame"] = "raw"
    layout["current"].write_text(json.dumps(current))
    with pytest.raises(ValueError, match="corrected-frame geometry revision"):
        all_runs.apply_geometry_to_all_runs(
            layout["project"], [layout["reference"]("ac17")], actor="synthetic"
        )
    assert not (layout["project"].root / "exports").exists()
