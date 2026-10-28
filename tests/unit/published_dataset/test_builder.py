"""The builder writes a flat, stored, name-free record, and refuses whatever the spec refuses."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import pytest

from darsia.fluidflower.published_dataset import builder, scan
from darsia.fluidflower.published_dataset.builder import BuildRequest, build_record, check_build
from darsia.fluidflower.published_dataset.checksums import verify_checksums
from darsia.fluidflower.published_dataset.job_reader import FinishedJob
from darsia.fluidflower.published_dataset.manifest import load_manifest
from darsia.fluidflower.published_dataset.proof import PROOF_FORMAT, inputs_digest
from tests.unit.published_dataset.synthetic import COMMIT, PRODUCER, as_json, make_series, sha, tiff


@pytest.fixture(autouse=True)
def fictional_machine(monkeypatch):
    """Fictional build-machine names, so real host or user names never enter a test."""
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),
        scan.ScanName("sample_builder", "build machine user name")))


def request(series, **changes):
    values = dict(series_root=series.series, job_root=series.job, record="R", tier="S50",
                  output=series.root / "record", deny=("Fictional Person",), fingerprint=False)
    values.update(changes)
    return BuildRequest(**values)


def snapshot(*folders):
    rows = {}
    for folder in folders:
        for path in Path(folder).rglob("*"):
            info = path.stat()
            rows[str(path)] = (path.is_dir(), info.st_size if path.is_file() else 0, info.st_mtime_ns)
    return rows


def test_r_record_is_flat_stored_and_complete(tmp_path):
    series = make_series(tmp_path, reviewer="Fictional Person")
    before = snapshot(series.series, series.job, series.camera, series.results)
    report = build_record(request(series))
    assert report.built, report.text()
    assert snapshot(series.series, series.job, series.camera, series.results) == before
    record = series.root / "record"
    assert sorted(os.listdir(record)) == ["CHECKSUMS.sha256", "README.md", "dataset_manifest.json",
                                          "expected.zip", "raw_r1.zip", "raw_r2.zip", "series.zip"]
    assert verify_checksums(record) == []
    manifest = load_manifest(record)
    assert manifest["identity"] == {"series_id": "SYN", "record": "R", "tier": "S50", "title": "Synthetic series",
                                    "links": {"other_record": None, "code": None, "article": None}}
    assert manifest["code"]["published_commit"] == manifest["code"]["expected_results_commit"] == COMMIT
    assert manifest["quick_check_valid"] is True and manifest["size"]["files"] == 7
    assert manifest["roots"] == {"series": ".", "raw": "raw"}
    frames = manifest["runs"][0]["frames"]
    assert [frame["raw"]["path"] for frame in frames] == [f"raw/r1/IMG_000{i}.ARW" for i in (1, 2, 3)]
    assert manifest["runs"][0]["references"][0]["path"] == "raw/r1/BASE_0000.ARW"
    expected = manifest["expected"]["runs"]["r1"]["frames"]["r1_f2"]
    assert expected["layer"]["path"] == "expected/r1/r1_f2/storage_stored_frame.npz"
    assert set(expected["masses_g"]) == {"total", "aqueous", "gas"}
    for name in ("series.zip", "expected.zip", "raw_r1.zip"):
        with zipfile.ZipFile(record / name) as archive:
            assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
    with zipfile.ZipFile(record / "series.zip") as archive:
        names = set(archive.namelist())
        assert {"series.toml", "plan/adopted_plan.toml", "plan/execution.toml", "dataset_aliases.toml",
                "provenance/r1.json", "config/runs/r1.json", "calibration/frozen.json"} <= names
        assert archive.read("plan/adopted_plan.toml") == (series.series / "plan" / "execution.toml").read_bytes()
        provenance = json.loads(archive.read("provenance/r1.json"))
        assert provenance["machine"] == "reference machine" and provenance["threads"] == 2
        # Each run's prepared task, exact bytes, for the launcher's replay.
        for run in ("r1", "r2"):
            task = (series.job / "results" / run / "task.json").read_bytes()
            assert archive.read(f"plan/tasks/{run}.json") == task
            assert manifest["plan"]["tasks"][run] == {"path": f"plan/tasks/{run}.json", "sha256": sha(task),
                                                      "size_bytes": len(task)}
    texts = b"".join((record / name).read_bytes() for name in os.listdir(record))
    with zipfile.ZipFile(record / "series.zip") as archive:
        texts += b"".join(archive.read(name) for name in archive.namelist())
    assert PRODUCER.encode() not in texts and b"example-build-host" not in texts
    readme = (record / "README.md").read_text(encoding="utf-8")
    assert COMMIT in readme and "darsia-reproduce" in readme
    allowed = [hit for hit in report.hits if hit.allowed]
    assert [hit.where for hit in allowed] == ["key /reviewer_name"]
    assert "Allowed names in validation records" in report.text()


def test_moved_record_unpacks_into_a_working_series_folder(tmp_path):
    series = make_series(tmp_path)
    assert build_record(request(series)).built
    downloaded = tmp_path / "elsewhere" / "download"
    shutil.copytree(series.root / "record", downloaded)
    shutil.rmtree(series.root / "record")
    for name in ("series.zip", "expected.zip"):
        with zipfile.ZipFile(downloaded / name) as archive:
            archive.extractall(downloaded)
    assert verify_checksums(downloaded) == []
    manifest = load_manifest(downloaded)
    assert (downloaded / "series.toml").is_file()
    with zipfile.ZipFile(downloaded / "raw_r2.zip") as archive:
        archive.extractall(downloaded)
    for frame in manifest["runs"][1]["frames"]:
        path = downloaded / frame["raw"]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["raw"]["sha256"]
    for run in manifest["expected"]["runs"].values():
        for frame in run["frames"].values():
            data = (downloaded / frame["layer"]["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == frame["layer"]["sha256"]


def test_check_writes_nothing_and_the_output_must_be_new_or_empty(tmp_path):
    series = make_series(tmp_path)
    plan, report = check_build(request(series))
    assert plan is not None and not report.refusals and not (series.root / "record").exists()
    (series.root / "record").mkdir()
    (series.root / "record" / "other.txt").write_text("keep")
    report = build_record(request(series))
    assert not report.built and "new or empty" in report.refusals[0]
    assert (series.root / "record" / "other.txt").read_text() == "keep"
    report = build_record(request(series, output=series.series / "inside"))
    assert any("outside the series folder" in item for item in report.refusals)


@pytest.mark.parametrize("case, message", [
    ("budget", "above the S50 budget"),
    ("files", "it must have fewer than"),
])
def test_size_and_file_limits_refuse(tmp_path, monkeypatch, case, message):
    series = make_series(tmp_path)
    if case == "budget":
        monkeypatch.setitem(builder.TIER_BUDGETS, "S50", 10_000)
    else:
        monkeypatch.setattr(builder, "FILE_LIMIT", 6)
    report = build_record(request(series))
    assert not report.built and any(message in item for item in report.refusals)


def test_tier_frames_must_be_the_jobs_and_tiers_must_nest(tmp_path):
    fewer = {"S50": {"runs": {"r1": ["IMG_0001.ARW", "IMG_0002.ARW"], "r2": ["IMG_0001.ARW", "IMG_0002.ARW",
                                                                            "IMG_0003.ARW"]}}}
    report = build_record(request(make_series(tmp_path / "a", tiers=fewer)))
    assert any("the expected results need exactly the job's frames" in item for item in report.refusals)
    all_frames = ["IMG_0001.ARW", "IMG_0002.ARW", "IMG_0003.ARW"]
    loose = {"S50": {"runs": {"r1": all_frames, "r2": all_frames}},
             "S100": {"runs": {"r1": all_frames[:2], "r2": all_frames}}}
    report = build_record(request(make_series(tmp_path / "b", tiers=loose)))
    assert any(item.startswith("The tiers are not nested") for item in report.refusals)
    report = build_record(request(make_series(tmp_path / "c"), tier="S100"))
    assert report.refusals == ["The tier file does not declare S100"]


def test_the_expected_layer_must_exist_at_one_millimetre(tmp_path):
    report = build_record(request(make_series(tmp_path / "a", resolution_mm=2.0)))
    assert any("the job has no 1 mm layer" in item for item in report.refusals)
    series = make_series(tmp_path / "b")
    job = FinishedJob(series.job)
    Path(job.layer("r2", "r2_f3").local_path).unlink()
    report = build_record(request(series))
    assert any(item.startswith("Run r2, r2_f3: an expected result is missing") for item in report.refusals)


def test_inputs_outside_the_dataset_or_changed_since_the_job_refuse(tmp_path):
    series = make_series(tmp_path)
    task_path = series.job / "results" / "r1" / "task.json"
    task = json.loads(task_path.read_text())
    task["input_catalog"]["inputs"].append({"reference": {"root_alias": "legacy", "relative_path": "x/table.json",
                                                          "sha256": "f" * 64, "bytes": 2, "role": "configuration"},
                                            "uses": ["mass.legacy"]})
    task_path.write_text(json.dumps(task))
    (series.series / "calibration" / "frozen.json").write_text('{"gain": 3.0}\n')
    report = build_record(request(series))
    assert any("lies outside the dataset folder: root 'legacy'" in item for item in report.refusals)
    assert any(item.startswith("Changed since the job read or wrote it: calibration/frozen.json")
               for item in report.refusals)


def test_a_task_that_cannot_be_replayed_is_refused(tmp_path):
    series = make_series(tmp_path)
    path = series.job / "results" / "r1" / "task.json"
    path.write_bytes(as_json(dict(json.loads(path.read_bytes()), pipeline="n1")))
    report = build_record(request(series))
    assert "Run r1: the launcher cannot replay the pipeline 'n1'" in report.refusals


def test_a_published_revision_needs_a_matching_proof(tmp_path):
    series = make_series(tmp_path)
    published = "d" * 40
    report = build_record(request(series, published_commit=published))
    assert "A published revision has no proof" in report.refusals[0]
    job = FinishedJob(series.job)
    files = builder.series_inputs({run: job.inputs(run) for run in job.runs}, builder.BuildReport())
    plan_sha = sha((series.series / "plan" / "execution.toml").read_bytes())
    proof = {"format": PROOF_FORMAT, "expected": {"commit": COMMIT, "plan_sha256": plan_sha},
             "published": {"commit": published, "plan_sha256": plan_sha,
                           "inputs_sha256": inputs_digest((item.root_alias, item.relative_path, item.sha256)
                                                          for item in files.values())},
             "machine": "reference machine", "verdict": "IDENTICAL",
             "frames": [{"run": "r1", "case_id": "r1_f1", "verdict": "IDENTICAL"}]}
    wrong = dict(proof, frames=[{"run": "r1", "case_id": "r1_f1", "verdict": "WITHIN EXPECTED VARIATION"}])
    (tmp_path / "wrong.json").write_bytes(as_json(wrong))
    report = build_record(request(series, published_commit=published, proof=tmp_path / "wrong.json"))
    assert "Proof: A frame of the proof is not bit-identical" in report.refusals
    (tmp_path / "proof.json").write_bytes(as_json(proof))
    report = build_record(request(series, published_commit=published, proof=tmp_path / "proof.json"))
    assert report.refusals == ["A published revision needs the finished job of the run that proved it "
                               "(its prepared tasks are published)"]
    # The proof run's job ran the published commit; its prepared tasks are the ones published.
    proof_job = tmp_path / "proof-job"
    shutil.copytree(series.job, proof_job)
    for path in [proof_job / "job.json", *proof_job.glob("results/*/task.json")]:
        path.write_bytes(as_json(dict(json.loads(path.read_bytes()), commit=published)))
    report = build_record(request(series, published_commit=published, proof=tmp_path / "proof.json",
                                  proof_job=series.job, output=tmp_path / "out-wrong"))
    assert "The proof job ran another commit or plan than the published revision" in report.refusals
    task_path = proof_job / "results" / "r2" / "task.json"
    task = json.loads(task_path.read_bytes())
    short = dict(task, input_catalog=dict(task["input_catalog"], inputs=task["input_catalog"]["inputs"][1:]))
    task_path.write_bytes(as_json(short))
    report = build_record(request(series, published_commit=published, proof=tmp_path / "proof.json",
                                  proof_job=proof_job))
    assert "Run r2: its prepared task reads other inputs than the record publishes" in report.refusals
    task_path.write_bytes(as_json(task))
    report = build_record(request(series, published_commit=published, proof=tmp_path / "proof.json",
                                  proof_job=proof_job))
    assert report.built, report.text()
    assert report.manifest["code"]["published_commit"] == published
    assert report.manifest["code"]["proof"]["verdict"] == "IDENTICAL"
    task = (proof_job / "results" / "r1" / "task.json").read_bytes()
    assert report.manifest["plan"]["tasks"]["r1"] == {"path": "plan/tasks/r1.json", "sha256": sha(task),
                                                      "size_bytes": len(task)}


@pytest.mark.parametrize("name, source", [
    ("example-build-host", "build machine host name"),
    ("sample_builder", "build machine user name"),
    (PRODUCER, "production receipt"),
    ("Fictional Person", "deny-list"),
])
def test_each_name_is_refused_in_a_config_a_receipt_a_file_name_and_an_archive_member(tmp_path, name, source):
    extra = {"config/notes.json": as_json({"note": f"checked by {name}"}),
             "data/receipt.json": as_json({"host": name, "user": "x"}),
             f"config/{name}.json": as_json({})}
    camera = {f"r1/{name}/REF_0001.ARW": tiff(b"reference")}
    report = build_record(request(make_series(tmp_path, extra=extra, extra_camera=camera)))
    assert not report.built
    found = [(hit.file, hit.where, hit.excerpt) for hit in scan.refused(report.hits) if hit.source == source]
    places = {(file, where) for file, where, _excerpt in found}
    assert {("series.zip:config/notes.json", "key /note"), ("series.zip:data/receipt.json", "key /host")} <= places
    assert any(file == "series.zip" and where == "member name" and f"config/{name}.json" in excerpt
               for file, where, excerpt in found)
    assert any(file == "raw_r1.zip" and where == "member name" and f"raw/r1/{name}/REF_0001.ARW" in excerpt
               for file, where, excerpt in found)


def test_absolute_paths_are_refused_wherever_they_are(tmp_path):
    extra = {"config/paths.json": as_json({"root": "C:\\Data\\series", "share": "\\\\server\\share\\x",
                                           "home": "/home/someone/x"})}
    report = build_record(request(make_series(tmp_path, extra=extra, series_toml_extra='raw_note = "D:/x"\n')))
    sources = sorted({(hit.file, hit.source) for hit in scan.refused(report.hits) if hit.kind == "absolute path"})
    assert sources == [("series.zip:config/paths.json", "UNC path"), ("series.zip:config/paths.json", "drive letter"),
                       ("series.zip:config/paths.json", "home folder"), ("series.zip:series.toml", "drive letter")]


def test_a_validator_name_is_allowed_only_in_reviewer_fields(tmp_path):
    allowed = build_record(request(make_series(tmp_path / "a", reviewer="Fictional Person")))
    assert allowed.built and [hit.source for hit in allowed.hits if hit.allowed] == ["deny-list"]
    extra = {"config/approval.json": as_json({"approved_by": "Fictional Person"})}
    refused = build_record(request(make_series(tmp_path / "b", reviewer="Fictional Person", extra=extra)))
    assert not refused.built
    assert [(hit.file, hit.where) for hit in scan.refused(refused.hits)] == [
        ("series.zip:config/approval.json", "key /approved_by")]


def test_raw_exif_names_are_reported_never_refused_or_edited(tmp_path):
    series = make_series(tmp_path, artist="Fictional Person")
    raw = series.camera / "r1" / "BASE_0000.ARW"
    before = raw.read_bytes()
    report = build_record(request(series))
    assert report.built and raw.read_bytes() == before
    assert {"file": "raw_r1.zip:raw/r1/BASE_0000.ARW", "field": "Artist", "value": "Fictional Person"} in report.exif


def test_simulator_record_needs_an_explicit_resolution(tmp_path):
    series = make_series(tmp_path)
    assert "explicit simulator resolution" in check_build(request(series, record="S"))[1].refusals[0]
    other = check_build(request(series, record="S", simulator_resolution_mm=2.0))[1]
    assert any("not available yet" in item for item in other.refusals)
    report = build_record(request(series, record="S", simulator_resolution_mm=1.0))
    assert report.built, report.text()
    record = series.root / "record"
    assert sorted(os.listdir(record)) == ["CHECKSUMS.sha256", "README.md", "dataset_manifest.json",
                                          "simulator_r1.zip", "simulator_r2.zip"]
    with zipfile.ZipFile(record / "simulator_r1.zip") as archive:
        assert set(archive.namelist()) == {"products/r1/mass_table.csv",
                                           "products/r1/timeseries/protocols/injection.csv",
                                           *(f"products/r1/r1_f{i}/storage_stored_frame.npz" for i in (1, 2, 3))}
    assert load_manifest(record)["simulator"]["resolution_mm"] == 1.0
    assert "geometry" not in load_manifest(record)["simulator"]  # Without a request: as before.


def test_a_simulator_record_carries_the_completed_geometry_on_request(tmp_path):
    from tests.unit.published_dataset.test_simulator_geometry import FIXTURE, FLAT_DEPTH, layered, write_revision

    series = make_series(tmp_path)
    write_revision(series.series / "geometry", layered())
    wanted = tmp_path / "geometry_request.json"
    base = {"format": "darsia-simulator-geometry-request-v1", "geometry": "geometry/geometry.json",
            "inner_frame": "canvas", "sand_surface": "free_water", "depth_points": FLAT_DEPTH}
    wanted.write_text(json.dumps(dict(base, runs={"r2": {"fixtures_m": {}}})), encoding="utf-8")
    undeclared = check_build(request(series, record="S", simulator_resolution_mm=1.0, simulator_geometry=wanted))[1]
    assert any(item.startswith("Run r1: the simulator geometry could not be made (No fixtures are declared")
               for item in undeclared.refusals)
    assert not any(item.startswith("Run r2") for item in undeclared.refusals)  # An explicit empty mapping.
    wanted.write_text(json.dumps(dict(base, fixtures_m=FIXTURE)), encoding="utf-8")
    refused = check_build(request(series, simulator_geometry=wanted))[1]
    assert "A simulator geometry belongs to an S record" in refused.refusals
    report = build_record(request(series, record="S", simulator_resolution_mm=1.0, simulator_geometry=wanted))
    assert report.built, report.text()
    record = series.root / "record"
    manifest = load_manifest(record)
    section = manifest["simulator"]["geometry"]
    assert section["format"] == "darsia-simulator-geometry-v1" and section["fixture_roles"] == ["colorchecker"]
    assert section["request_sha256"] == sha(wanted.read_bytes()) and set(section["runs"]) == {"r1", "r2"}
    with zipfile.ZipFile(record / "simulator_r1.zip") as archive:
        npz = archive.read("products/r1/geometry/simulator_geometry.npz")
        document = json.loads(archive.read("products/r1/geometry/simulator_geometry.json"))
    row = section["runs"]["r1"]
    assert row["npz"] == {"path": "products/r1/geometry/simulator_geometry.npz", "sha256": sha(npz),
                          "size_bytes": len(npz)}
    assert row["counts"]["fixture_filled_cells"] > 0 and row["counts"]["unfilled_cells"] == 0
    assert document["grid"]["resolution_mm"] == 1.0 and document["source"]["labels"] == "labels.npz"
    assert document["method"]["inner_frame"] == "canvas" and document["method"]["sand_surface"] == "free_water"
    assert {item["permeability_unit"] for item in document["facies_table"]
            if item["analysis_role"] == "porous_medium"} == {"mD"}  # The series table states its unit.
    assert any(line.startswith("Run r1: Facies 2 has no swi, grain_size, source") for line in report.warnings)
    assert "simulator_geometry.npz" in (record / "README.md").read_text(encoding="utf-8")
    assert str(tmp_path) not in json.dumps(document)
    wanted.write_text(json.dumps(dict(base, geometry="geometry/missing.json")), encoding="utf-8")
    missing = check_build(request(series, record="S", simulator_resolution_mm=1.0, simulator_geometry=wanted,
                                  output=tmp_path / "other"))[1]
    assert any(item.startswith("The simulator geometry request is not usable") for item in missing.refusals)


def test_another_series_or_an_unfinished_job_is_refused(tmp_path):
    series = make_series(tmp_path)
    (series.series / "series.toml").write_text('schema_version = 1\nseries_id = "OTHER"\n[roots]\nseries = "."\n')
    assert "The job belongs to another series" in build_record(request(series)).refusals
    report = build_record(request(series, job_root=tmp_path / "jobs" / "missing"))
    assert report.refusals[0].startswith("Not a finished production job")
    assert build_record(request(series, record="X")).refusals == ["Record type must be one of R, S: 'X'"]
