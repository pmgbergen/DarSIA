"""Manifest checks, README, proof records, and the ``darsia dataset build`` command line."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from darsia.fluidflower.published_dataset import builder, cli, manifest, proof, scan
from darsia.fluidflower.published_dataset.readme import render_readme
from tests.unit.published_dataset.synthetic import make_series


@pytest.fixture(autouse=True)
def fictional_machine(monkeypatch):
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),))


@pytest.fixture
def built(tmp_path):
    series = make_series(tmp_path)
    report = builder.build_record(builder.BuildRequest(series.series, series.job, "R", "S50", tmp_path / "record",
                                                       deny=(), fingerprint=False))
    assert report.built, report.text()
    return series, report.manifest


def test_manifest_checks_refuse_broken_documents(built):
    _series, document = built
    assert manifest.validate_manifest(document) is document
    for change, message in (
            (lambda d: d.update(format="other"), "format"),
            (lambda d: d["identity"].update(record="Q"), "record"),
            (lambda d: d["identity"].update(tier="S70"), "tier"),
            (lambda d: d["code"].update(published_commit="d" * 40), "needs a proof"),
            (lambda d: d["runs"].append(deepcopy(d["runs"][0])), "listed twice"),
            (lambda d: d["runs"][0]["frames"][0]["raw"].update(sha256="x"), "raw SHA-256"),
            (lambda d: d["expected"]["runs"]["r1"]["frames"].pop("r1_f1"), "expected results and frames differ"),
            (lambda d: d["reproducibility"]["other_machines"].update(same_machine="close"), "bit-identical"),
            (lambda d: d["reproducibility"]["other_machines"]["masses"].update(relative=-1), "nonnegative"),
            (lambda d: d.update(quick_check_valid="yes"), "quick_check_valid"),
            (lambda d: d["size"].update(files=100), "fewer than 100"),
            (lambda d: d.update(simulator={"geometry": {"runs": {}}}), "only an S record")):
        broken = deepcopy(document)
        change(broken)
        with pytest.raises(ValueError, match=message):
            manifest.validate_manifest(broken)
    assert manifest.rule_of(document) == manifest.DEFAULT_RULE


def test_manifest_round_trip(built, tmp_path):
    _series, document = built
    assert manifest.is_dataset(tmp_path / "record") and not manifest.is_dataset(tmp_path)
    assert manifest.load_manifest(tmp_path / "record") == document
    (tmp_path / "copy").mkdir()
    path = manifest.write_manifest(tmp_path / "copy", document)
    assert path.read_text(encoding="utf-8") == manifest.manifest_text(document)
    with pytest.raises(FileExistsError):
        manifest.write_manifest(tmp_path / "copy", document)
    with pytest.raises(FileNotFoundError):
        manifest.load_manifest(tmp_path / "missing")


def test_readme_follows_the_manifest(built):
    _series, document = built
    text = render_readme(document)
    assert text.startswith("# Synthetic series") and "mass maps **1 mm**: 2 runs, 6 frames" in text
    assert "0.1 % of the reference CO2 area" in text and "Quick check (the first frames of one run): valid" in text
    variation = deepcopy(document)
    variation["known_variation"] = [{"summary": "reference against a second machine: within the limits"}]
    assert "- reference against a second machine: within the limits" in render_readme(variation)
    simulator = deepcopy(document)
    simulator["identity"]["record"] = "S"
    assert "simulator_<run>.zip" in render_readme(simulator)
    assert "series.zip" not in render_readme(simulator)


def test_proof_records_bind_both_sides():
    rows = [("series", "a.json", "1" * 64), ("raw", "x/IMG.ARW", "2" * 64)]
    assert proof.inputs_digest(rows) == proof.inputs_digest(list(reversed(rows)))
    good = {"format": proof.PROOF_FORMAT, "expected": {"commit": "a" * 40, "plan_sha256": "b" * 64},
            "published": {"commit": "c" * 40, "plan_sha256": "d" * 64, "inputs_sha256": "e" * 64},
            "machine": "reference machine", "frames": [{"verdict": "IDENTICAL"}], "verdict": "IDENTICAL"}
    binding = dict(expected_commit="a" * 40, expected_plan_sha256="b" * 64, published_commit="c" * 40,
                   published_plan_sha256="d" * 64, published_inputs_sha256="e" * 64)
    assert proof.proof_problems(good, **binding) == []
    assert proof.proof_problems({"format": "x"}, **binding) == [f"The proof must be a {proof.PROOF_FORMAT} record"]
    bad = dict(good, machine="second machine", frames=[], verdict="OUTSIDE")
    assert len(proof.proof_problems(bad, **dict(binding, published_inputs_sha256="f" * 64))) == 4


def test_command_line_builds_checks_and_refuses(tmp_path, capsys):
    series = make_series(tmp_path)
    arguments = ["build", "--series", str(series.series), "--job", str(series.job), "--record", "R",
                 "--tier", "S50", "--no-fingerprint"]
    assert cli.main([*arguments, "--output", str(tmp_path / "checked"), "--check"]) == 0
    assert "Checked: nothing refuses the record." in capsys.readouterr().out
    assert not (tmp_path / "checked").exists()
    report = tmp_path / "report.json"
    assert cli.main([*arguments, "--output", str(tmp_path / "record"), "--report", str(report),
                     "--peak-memory-bytes", "1000", "--link", "article=10.1/x", "--root",
                     f"raw={series.camera}"]) == 0
    assert json.loads(report.read_text())["built"] is True
    document = manifest.load_manifest(tmp_path / "record")
    assert document["machine"]["peak_memory_bytes"] == 1000
    assert document["identity"]["links"]["article"] == "10.1/x"
    assert cli.main([*arguments, "--output", str(tmp_path / "record")]) == 1
    assert "new or empty" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["build", "--link", "no-equals"])


def test_request_from_arguments_and_the_gui_entry_point_dispatch(tmp_path, monkeypatch):
    current = cli.request_from(cli.parser().parse_args(["build", "--series", "s", "--job", "j",
        "--record", "R", "--output", "o", "--mass-map-resolution-mm", "5"]))
    assert current.tier is None and current.expected_resolution_mm is None
    assert current.software_mode == "references" and current.mass_map_resolution_mm == 5
    parsed = cli.parser().parse_args(["build", "--series", "s", "--job", "j", "--record", "S", "--tier", "S100",
                                      "--output", "o", "--simulator-resolution-mm", "1"])
    request = cli.request_from(parsed)
    assert (request.record, request.tier, request.simulator_resolution_mm, request.roots) == ("S", "S100", 1.0, None)
    assert request.simulator_geometry is None
    parsed = cli.parser().parse_args(["build", "--series", "s", "--job", "j", "--record", "S", "--tier", "S100",
                                      "--output", "o", "--simulator-resolution-mm", "1",
                                      "--simulator-geometry", "request.json"])
    assert cli.request_from(parsed).simulator_geometry == Path("request.json")
    from darsia.gui import __main__ as gui_main

    seen = []
    monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv) or 7)
    monkeypatch.setattr(sys, "argv", ["darsia", "dataset", "build", "--check"])
    with pytest.raises(SystemExit) as exit_info:
        gui_main.main()
    assert exit_info.value.code == 7 and seen == [["build", "--check"]]


def test_module_entry_point_runs_the_command_line():
    import os
    from pathlib import Path

    import darsia

    environment = dict(os.environ, PYTHONPATH=str(Path(darsia.__file__).resolve().parents[1]))
    completed = subprocess.run([sys.executable, "-B", "-m", "darsia.fluidflower.published_dataset", "--help"],
                               capture_output=True, text=True, timeout=120, env=environment)
    assert completed.returncode == 0 and "build" in completed.stdout
