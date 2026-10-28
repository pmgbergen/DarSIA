"""The launcher's steps with a stand-in runner: checks, refusals, scope, verdicts and report."""

import json
import shutil
import zipfile
from argparse import Namespace

import pytest

from darsia.fluidflower.published_dataset import builder, dataset_check, reproduce_cli, scan
from darsia.fluidflower.published_dataset.comparison import IDENTICAL, OUTSIDE, load_layer
from darsia.fluidflower.published_dataset.layout import inside
from darsia.fluidflower.published_dataset.reproduce import (Reproduction, ReproductionRefused, Settings,
                                                           choose_scope, execute, plan_differences, scope_frames)
from tests.unit.published_dataset.synthetic import COMMIT, make_series

INSTALLED = {"commit": COMMIT, "clean": True, "source": "Git checkout"}
OTHER = {"commit": "e" * 40, "clean": True, "source": "Git checkout"}


class FakeRunner:
    """Returns the expected results, optionally changed, instead of running the job layer."""

    def __init__(self, change=None):
        self.change, self.ran, self.check = change, [], None

    def run(self, check, results, scope, settings, *, progress=None):
        self.check = check
        self.ran.append((scope.kind, list(scope.runs), dict(scope.frames), settings.workers))
        if progress is not None:
            progress({run: "done" for run in scope.runs})

    def reproduced(self, run, case):
        entry = self.check.manifest["expected"]["runs"][run]["frames"][case]
        masses, layer = dict(entry["masses_g"]), load_layer(inside(self.check.root, entry["layer"]["path"]))
        if self.change is not None:
            self.change(run, case, masses, layer)
        return masses, layer


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),))
    series = make_series(tmp_path / "producer-site")
    report = builder.build_record(builder.BuildRequest(series.series, series.job, "R", "S50", tmp_path / "record",
                                                       deny=(), fingerprint=False, peak_memory_bytes=2 * 1024**3))
    assert report.built, report.text()
    folder = tmp_path / "dataset"
    shutil.copytree(tmp_path / "record", folder)
    for name in ("series.zip", "expected.zip"):
        with zipfile.ZipFile(folder / name) as archive:
            archive.extractall(folder)
    return folder


def opened(dataset, results, runner=None, installed=INSTALLED):
    reproduction = Reproduction(dataset, results, runner=runner or FakeRunner())
    reproduction.open(installed=installed, fingerprint={})
    return reproduction


def test_dataset_checks_find_unpacked_parts_sizes_and_changed_files(dataset, tmp_path):
    reproduction = opened(dataset, tmp_path / "results")
    assert reproduction.check.problems == [] and reproduction.verify() == []
    assert reproduction.check.runs == ["r1", "r2"] and len(reproduction.check.frames("r1")) == 3
    (dataset / "calibration" / "frozen.json").write_text('{"gain": 9}\n')
    assert reproduction.verify() == ["Changed: calibration/frozen.json"]
    with (dataset / "raw_r1.zip").open("ab") as stream:
        stream.write(b"x")
    assert "Size differs from the manifest: raw_r1.zip" in dataset_check.open_dataset(
        dataset, installed=INSTALLED, fingerprint={}).problems
    (dataset / "expected" / "r1" / "r1_f1" / "storage_stored_frame.npz").unlink()
    problems = dataset_check.open_dataset(dataset, installed=INSTALLED, fingerprint={}).problems
    assert any(item.startswith("Not unpacked: the expected results") for item in problems)


def test_another_commit_is_refused_unless_overridden_and_then_not_a_reproduction(dataset, tmp_path):
    reproduction = opened(dataset, tmp_path / "results", installed=OTHER)
    scope = choose_scope(reproduction.check.manifest, "all")
    with pytest.raises(ReproductionRefused, match="not a reproduction"):
        reproduction.run(scope, Settings(workers=1, threads=2))
    result = execute(reproduction, scope, Settings(workers=1, threads=2, override_commit=True), machine={})
    report = json.loads(result["files"]["json"].read_text(encoding="utf-8"))
    assert report["reproduction"] is False and report["note"].startswith("Not a reproduction")
    assert "Not a reproduction" in result["files"]["html"].read_text(encoding="utf-8")
    assert dataset_check.compare_code({"code": {"published_commit": COMMIT}},
                                      {"commit": COMMIT, "clean": False})["same"] is False


def test_results_folder_must_be_new_empty_outside_and_large_enough(dataset, tmp_path):
    reproduction = opened(dataset, dataset / "inside")
    scope = choose_scope(reproduction.check.manifest, "all")
    assert "The results folder must lie outside the dataset folder" in reproduction.preflight(
        scope, Settings(workers=1, threads=2))
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    assert dataset_check.results_folder_problems(full, dataset, 0)[0].startswith("The results folder must be new")
    assert "Too little free space" in dataset_check.results_folder_problems(tmp_path / "new", dataset, 10**18)[0]
    assert reproduction.preflight(scope, Settings(workers=0, threads=2))[0] == "Choose at least one worker"


def test_scope_quick_only_when_valid_one_run_or_all(dataset):
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert choose_scope(manifest, "all").runs == ["r1", "r2"]
    assert choose_scope(manifest, "run", runs=["r2", "r1"]).runs == ["r1", "r2"]
    quick = choose_scope(manifest, "quick", run="r2", quick_frames=10)
    assert quick.frames == {"r2": 3} and [frame["case_id"] for _run, frame in scope_frames(manifest, quick)] == [
        "r2_f1", "r2_f2", "r2_f3"]
    assert choose_scope(manifest, "quick").frames == {"r1": 3}
    with pytest.raises(ReproductionRefused, match="no run"):
        choose_scope(manifest, "run", run="r9")
    with pytest.raises(ReproductionRefused):
        choose_scope(manifest, "some")
    manifest["quick_check_valid"] = False
    with pytest.raises(ReproductionRefused, match="not valid"):
        choose_scope(manifest, "quick")


def test_estimates_use_the_reference_machines_measurements(dataset):
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    estimate = dataset_check.estimates(manifest, ["r1", "r2"], workers=2)
    assert estimate["seconds"] == pytest.approx(6.0 * 6 / 2)
    assert estimate["seconds_basis"] == "measured on the reference machine"
    assert estimate["output_bytes"] == manifest["machine"]["output_bytes_per_frame"] * 6
    assert estimate["scratch_bytes"] == max(row["archive"]["size_bytes"] for row in manifest["runs"])
    manifest["machine"]["output_bytes_per_frame"] = None
    assert dataset_check.estimates(manifest, ["r1"])["output_basis"].startswith("ESTIMATE")


def test_an_outside_frame_is_reported_with_its_first_divergence(dataset, tmp_path):
    def change(run, case, masses, layer):
        if case == "r2_f2":
            masses["gas"] = masses["gas"] + 1.0
    reproduction = opened(dataset, tmp_path / "results", FakeRunner(change))
    scope = choose_scope(reproduction.check.manifest, "all")
    result = execute(reproduction, scope, Settings(workers=1, threads=2), machine={})
    summary = result["summary"]
    assert summary["dataset"] == OUTSIDE and summary["runs"] == {"r1": IDENTICAL, "r2": OUTSIDE}
    assert summary["first_divergence"]["case_id"] == "r2_f2" and summary["first_divergence"]["quantity"] == "gas mass"
    html = result["files"]["html"].read_text(encoding="utf-8")
    assert "First divergence" in html and "r2_f2" in html


def test_command_line_refusals_and_usage(dataset, tmp_path, monkeypatch):
    arguments = reproduce_cli.parser().parse_args(["--dataset", str(dataset), "--results", str(tmp_path / "x")])
    assert reproduce_cli.run(arguments, runner=FakeRunner(), machine={}, installed=OTHER) == 1
    quick = reproduce_cli.parser().parse_args(["--dataset", str(dataset), "--results", str(tmp_path / "q"),
                                              "--quick", "--quick-frames", "1", "--workers", "1", "--full-check"])
    runner = FakeRunner()
    assert reproduce_cli.run(quick, runner=runner, machine={}, installed=INSTALLED) == 0
    assert runner.ran == [("quick", ["r1"], {"r1": 1}, 1)]
    with pytest.raises(SystemExit):
        reproduce_cli.main(["--dataset", str(dataset)])
    opened_with = []
    monkeypatch.setattr("darsia.fluidflower.published_dataset.reproduce_window.main",
                        lambda arguments: opened_with.append(arguments) or 0)
    assert reproduce_cli.main([]) == 0 and len(opened_with) == 1
    assert isinstance(opened_with[0], Namespace)


def test_settings_that_differ_from_the_published_plan_are_named(dataset, tmp_path):
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    machine, same = manifest["machine"], {"same": True}
    published = Settings(workers=1, threads=machine["threads"], backend=machine["backend"])
    assert plan_differences(manifest, choose_scope(manifest, "all"), published, same) == []
    changed = Settings(workers=1, threads=machine["threads"] + 1, backend="cuda")
    lines = plan_differences(manifest, choose_scope(manifest, "run", run="r2"), changed,
                             {"same": False, "installed": "e" * 40, "published": COMMIT})
    assert lines == ["Scope: run(s) r2 of the published 2",
                     f"Threads per worker: {machine['threads'] + 1} (published: {machine['threads']})",
                     f"Backend: cuda (published: {machine['backend']})",
                     f"DarSIA commit: {'e' * 40} (published: {COMMIT}); not a reproduction"]
    quick = plan_differences(manifest, choose_scope(manifest, "quick", quick_frames=2), published, same)
    assert quick == ["Scope: the quick check, the first 2 frame(s) of run r1 (the published plan has 2 run(s))"]
    reproduction = opened(dataset, tmp_path / "results")
    scope = choose_scope(reproduction.check.manifest, "run", run="r1")
    result = execute(reproduction, scope, Settings(workers=1, threads=machine["threads"]), machine={})
    report = json.loads(result["files"]["json"].read_text(encoding="utf-8"))
    assert report["differences_from_published_plan"] == ["Scope: run(s) r1 of the published 2"]
    html = result["files"]["html"].read_text(encoding="utf-8")
    assert "Settings that differ from the published plan" in html and "run(s) r1 of the published 2" in html
