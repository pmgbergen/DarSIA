"""Job preparation: a lattice request gives its task the lattice pipeline.

A run configuration without the request prepares byte-identical job files,
exactly as before the lattice hook existed. Synthetic metadata only.
"""
import json

import pytest

from darsia.fluidflower import storage_outputs
from darsia.fluidflower.pipeline import alias_job, lattice_runner, series

from .gui.test_alias_job_metadata import assembly_fixture

STORAGE = dict(format=storage_outputs.REQUEST_FORMAT, gas_score_encoding=storage_outputs.LA_ENCODING,
               resolution_mm=1.0, extent_m=[0.0, 0.145, 0.0, 0.183])


def prepare(tmp_path, monkeypatch, *, storage=None, names=("job",)):
    _root, manifest, source, config, _value = assembly_fixture(tmp_path, monkeypatch)
    if storage is not None:
        config["storage"] = storage
    destinations = []
    for name in names:
        destination = tmp_path / name
        alias_job.prepare_alias_job(source["path"], destination, expected_sha256=source["sha256"],
                                    series_manifest=manifest)
        destinations.append(destination)
        # The second preparation runs without the lattice hook.
        monkeypatch.setattr(lattice_runner, "lattice_task", lambda task, config: task)
    return destinations if len(names) > 1 else destinations[0]


def documents(destination):
    """The prepared job files, without each queue copy's fresh random attempt token."""
    result = {}
    for path in ("job.json", "queue/ac52.json", "results/ac52/task.json"):
        value = json.loads((destination / path).read_bytes())
        value.pop("attempt_id", None)
        result[path] = value
    return result


def test_a_lattice_request_prepares_a_lattice_task_with_its_resolution(tmp_path, monkeypatch):
    destination = prepare(tmp_path, monkeypatch, storage=dict(STORAGE, computing_resolution_mm=1.0))
    task = json.loads((destination / "results/ac52/task.json").read_bytes())
    assert task["pipeline"] == lattice_runner.PIPELINE and task["computing_resolution_mm"] == 1.0
    job = json.loads((destination / "job.json").read_bytes())
    assert "computing_resolution_mm" not in job and "pipeline" not in job


@pytest.mark.parametrize("storage", [None, STORAGE])
def test_a_configuration_without_the_request_prepares_unchanged_files(tmp_path, monkeypatch, storage):
    with_hook, without_hook = prepare(tmp_path, monkeypatch, storage=storage, names=("hook", "plain"))
    assert (with_hook / "job.json").read_bytes() == (without_hook / "job.json").read_bytes()
    task = documents(with_hook)["results/ac52/task.json"]
    assert task["pipeline"] == series.PIPELINE and "computing_resolution_mm" not in task
    assert documents(with_hook) == documents(without_hook)


def test_an_invalid_lattice_request_fails_before_any_job_folder(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="cannot be finer"):
        prepare(tmp_path, monkeypatch, storage=dict(STORAGE, resolution_mm=0.5, computing_resolution_mm=1.0),
                names=("never",))
    assert not (tmp_path / "never").exists()
