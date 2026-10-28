"""Machine and environment facts without names; reading a finished production job."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower.published_dataset import environment, job_reader
from tests.unit.published_dataset.synthetic import PRODUCER, TOKEN, make_series


class _File(str):
    """A distribution's file entry: its relative path, with a RECORD hash."""

    @property
    def hash(self):
        return SimpleNamespace(value="recorded-" + self)


class _Distribution:
    def __init__(self, name, version, files, root):
        self.metadata = {"Name": name}
        self.version, self.files, self._root = version, [_File(item) for item in files], root

    def locate_file(self, item):
        return self._root / str(item)


def test_fingerprint_covers_each_distributions_installed_files(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_bytes(b"a")
    (tmp_path / "pkg" / "a.cpython-313.pyc").write_bytes(b"compiled")
    (tmp_path / "pkg-1.dist-info").mkdir()
    (tmp_path / "pkg-1.dist-info" / "RECORD").write_bytes(b"x")
    files = ["pkg/a.py", "pkg/a.cpython-313.pyc", "pkg-1.dist-info/RECORD", "pkg/missing.py"]
    first = environment.distribution_fingerprint(distributions=[_Distribution("Pkg_Name", "1.0", files, tmp_path)])
    assert first["format"] == environment.FINGERPRINT_FORMAT
    assert first["distributions"][0]["name"] == "pkg-name" and first["distributions"][0]["files"] == 2
    (tmp_path / "pkg" / "a.py").write_bytes(b"changed")
    second = environment.distribution_fingerprint(distributions=[_Distribution("pkg-name", "1.0", files, tmp_path)])
    assert first["sha256"] != second["sha256"]
    assert environment.compare_fingerprints(first, second) == [
        "pkg-name: same version 1.0, other installed files (another build)"]
    third = dict(second, distributions=[dict(second["distributions"][0], version="2.0")], python="3.12")
    assert environment.compare_fingerprints(first, third)[1] == "pkg-name: version 1.0 published, 2.0 here"
    assert "not installed here" in environment.compare_fingerprints(first, dict(second, distributions=[]))[0]
    recorded = environment.distribution_fingerprint(distributions=[_Distribution("pkg", "1", files, tmp_path)],
                                                    hash_files=False)
    assert recorded["file_digests"] == "recorded"


def test_machine_facts_carry_no_names(monkeypatch):
    facts = environment.machine_facts()
    assert set(facts) == {"platform", "cpu", "memory"} and "host" not in json.dumps(facts).casefold()
    assert isinstance(environment.cpu_model(), str) and environment.cpu_model()
    sets = environment.instruction_sets()
    assert sets["summary"] and isinstance(sets["features"], list)
    assert set(environment.platform_facts()) == {"system", "release", "machine", "python"}


def test_lock_summary(tmp_path):
    path = tmp_path / "requirements-worker.lock"
    path.write_bytes(b"# Python 3.13. Exact.\nnumpy==2.4.4\nscipy==1.0\n# note==1\n")
    summary = environment.lock_summary(path)
    assert summary["pins"] == 2 and summary["python"] == "3.13" and summary["hashes"] is False
    assert summary["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_finished_job_reads_runs_inputs_frames_and_products(tmp_path):
    series = make_series(tmp_path)
    job = job_reader.FinishedJob(series.job)
    assert job.runs == ["r1", "r2"] and job.series_id == "SYN"
    assert job.frames("r1") == ["r1_f1", "r1_f2", "r1_f3"]
    inputs = job.inputs("r1")
    cameras = [item for item in inputs if item.camera_file]
    assert [item.relative_path for item in cameras] == [
        "r1/BASE_0000.ARW", "r1/IMG_0001.ARW", "r1/IMG_0002.ARW", "r1/IMG_0003.ARW"]
    assert all(not item.camera_file for item in inputs if item.root_alias == "series")
    result = job.frame_result("r1", "r1_f2")
    assert result["case_id"] == "r1_f2" and result["bindings"]["source_sha256"] == series.frames["r1"][1]["sha256"]
    layer = job.layer("r1", "r1_f2")
    assert layer.product_id == f"attempts/{TOKEN}/preparation/r1_f2/storage_stored_frame.npz"
    assert job.product_source(layer).is_file() and job.layer("r1", "r9") is None
    assert job.product_source(job.mass_table("r1")).parent.name == "metadata"
    assert job.plan_reference()["relative_path"] == "plan/execution.toml"
    assert job.series_manifest_reference()["relative_path"] == "series.toml"
    names = {item.text for item in job.receipt_names()}
    assert {PRODUCER, PRODUCER + "-cpu"} <= names
    assert job.measured("r1") == {"backend": "cpu", "precision": "float32", "threads": 2, "frames": 3,
                                  "wall_seconds": 18.0, "seconds_per_frame": 6.0, "pipeline": "native-phase-n1"}
    assert job_reader.producer_is_this_machine(job, [PRODUCER.upper()])
    assert not job_reader.producer_is_this_machine(job, ["another-host"])
    with pytest.raises(job_reader.JobError, match="no product"):
        job.product("r1", "missing")


def test_finished_job_refuses_unfinished_or_changed_jobs(tmp_path):
    series = make_series(tmp_path)
    with pytest.raises(job_reader.JobError, match="No job.json"):
        job_reader.FinishedJob(tmp_path)
    completed = series.job / "results" / "r1" / "completed.json"
    completed.write_bytes(completed.read_bytes() + b" ")
    with pytest.raises(job_reader.JobError, match="differs from its record"):
        job_reader.FinishedJob(series.job)
    (series.job / "done" / "r2.json").unlink()
    document = json.loads((series.job / "job.json").read_text())
    document["expected_runs"] = ["r2"]
    (series.job / "job.json").write_text(json.dumps(document))
    with pytest.raises(job_reader.JobError, match="not finished"):
        job_reader.FinishedJob(series.job)
    document["format"] = "another-job"
    (series.job / "job.json").write_text(json.dumps(document))
    with pytest.raises(job_reader.JobError, match="alias job"):
        job_reader.FinishedJob(series.job)


def test_producer_only_products_need_the_producing_machine(tmp_path):
    series = make_series(tmp_path)
    job = job_reader.FinishedJob(series.job)
    layer = job.layer("r1", "r1_f1")
    job_reader.Path(layer.local_path).unlink()
    with pytest.raises(job_reader.JobError, match="producing machine"):
        job.product_source(layer)
