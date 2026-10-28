"""Original published P1 aliases reach the unchanged full completion verifier.

The fixture uses original queue/publication methods with declared synthetic
ownership and tiny saved bytes. It runs no numerical producer and makes no
campaign, source-processing, filesystem-qualification or retirement claim.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import calibration_check_calculation as checks, job_filesystem
from darsia.fluidflower import published_jobs_snapshot
from darsia.fluidflower.mass.n1_statistics import CONTRACT
from darsia.fluidflower.pipeline import job_queue, phase1_protocol as protocol, publication_runtime
from darsia.fluidflower.pipeline import products_publication as publication
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput
from darsia.fluidflower.store_map import input_reference
from . import test_staged_data as closure_fixture
from .test_phase1_worker_protocol import metadata, scientific_fixture, SyntheticServerClock


@pytest.fixture
def published_phase1(tmp_path, monkeypatch):
    series = tmp_path / "series"
    series.mkdir()
    manifest = series / "series.toml"
    manifest.write_bytes(b'schema_version=1\nseries_id="Synthetic"\n[roots]\nseries="."\n')
    data, raw, job, task = metadata()
    config_path = series / "phase1.json"
    config_path.write_bytes(raw)
    roots = {"series": str(series)}

    def reference(path):
        bound = job_queue.descriptor(path)
        return input_reference(str(path), roots, sha256=bound["sha256"],
                               size_bytes=bound["size_bytes"], role="configuration")

    configuration, series_manifest = reference(config_path), reference(manifest)
    task.update(configuration=configuration, series=deepcopy(configuration),
                series_manifest=deepcopy(series_manifest))
    task["input_catalog"].update(configuration=configuration, inputs=[
        dict(reference=deepcopy(configuration), uses=["run.configuration", "run.execution_series"]),
        dict(reference=deepcopy(series_manifest), uses=["run.series_manifest"])])
    job["series_manifest"] = deepcopy(series_manifest)
    job, tasks = protocol.bind_phase1_identity(job, [task])
    task = tasks[0]
    queue = job_queue.JobQueue(tmp_path / "shared", machine_id="main", server_clock=SyntheticServerClock())
    job_queue.write_immutable(queue.root / "job.json", job)
    queue.enqueue(task)

    def availability(value):
        return tuple(ResolvedInput(deepcopy(row["reference"]),
            series.joinpath(*row["reference"]["relative_path"].split("/")), tuple(row["uses"]))
            for row in value["input_catalog"]["inputs"])

    def output_factory(value, token):
        local = tmp_path / "local-results"
        output = local / value["job_id"] / value["run"] / ".attempts" / token
        admission = job_filesystem.LocalFileAdmission(True, "synthetic-local", "unit fixture",
            "No production filesystem qualification", str(tmp_path))
        return job_queue.LocalOutputBinding("main", local, output, admission)

    def no_prefix(_attempt):
        raise AssertionError("Fresh phase one cannot borrow another producer")

    publisher = publication_runtime.make_publication_factory(queue.root, job, "main",
        producer_origin_for_attempt=no_prefix)
    owner = dict(machine="main", pid=123, process_create_time="synthetic-indexed-check-input")
    claim = queue.claim("main", "cpu", owner, availability=availability,
        output_factory=output_factory, publication_factory=publisher)
    spec = data["frames"][0]
    statistics = dict(contract=CONTRACT, run=data["run"], frame_index=spec["frame_index"],
        elapsed_h=spec["elapsed_h"], selected_target_h=spec["mass_spec"]["selected_target_h"],
        source_binding=deepcopy(spec["source"]["source_binding"]), fixture_only=True)
    original_write = closure_fixture._write

    def fixture_bytes(path, value=None, raw=None):
        # Supply valid tiny statistics/QA bytes at the fixture's writing seam.
        # No publication, source identity or verification predicate is replaced.
        if Path(path).name == "native_statistics.json.gz":
            raw = gzip.compress(json.dumps(statistics, allow_nan=False).encode(), mtime=0)
        elif raw is not None and Path(path).suffix == ".json":
            raw = b'{"synthetic_unit_payload":true}\n'
        return original_write(path, value=value, raw=raw)

    with monkeypatch.context() as patch:
        patch.setattr(closure_fixture, "_write", fixture_bytes)
        closure, base = scientific_fixture(claim, data)
    result = protocol.worker_result(base, task=claim.task, job=job,
        commit=task["commit"], owner=owner, token=claim.token)
    queue.complete(claim, result)
    done = job_queue.read_json(queue.root / "done" / (task["run"] + ".json"))
    index = job_queue.descriptor(job_queue._own_asset(queue.root, done["products_index"]))
    return SimpleNamespace(series=series, root=queue.root, run=task["run"], claim=claim,
        completion=result, closure=closure, statistics=statistics, index=index)


def _rows(inputs, fixture):
    # The held-out path needs no plan fitting inputs; the original completion
    # verifier, statistics reader and per-acquisition checks run unchanged.
    return checks._run_rows(inputs, fixture.root, fixture.run, {}, SimpleNamespace(fit_runs=()))


@pytest.mark.parametrize("forbid_producer_metadata", [False, True])
def test_original_published_metadata_and_gzip_feed_full_verifier(
        published_phase1, monkeypatch, forbid_producer_metadata):
    fixture = published_phase1
    producer = fixture.claim.output.resolve()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if forbid_producer_metadata:
            assert not (path.resolve().is_relative_to(producer)
                        and path.name.endswith((".json", ".json.gz"))), "Producer-local metadata was opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    inputs = checks._Inputs(fixture.series)
    factory = _rows(inputs, fixture)
    assert list(factory()) == [fixture.statistics]
    assert list(factory()) == [fixture.statistics]  # The original repeated-read guard remains active.
    bindings = inputs.finish()
    shared = publication.published_metadata_descriptor(
        fixture.index, "phase1/case-one/native_statistics.json.gz")
    normalized = dict(path=shared["path"], sha256=shared["sha256"], size_bytes=shared["bytes"])
    assert normalized in bindings
    assert Path(shared["path"]).read_bytes().startswith(b"\x1f\x8b")
    assert fixture.closure["record"]["statistics"]["path"].startswith(str(fixture.claim.output))
    assert not any(Path(bound["path"]).is_relative_to(producer)
                   and Path(bound["path"]).name.endswith((".json", ".json.gz")) for bound in bindings)


@pytest.mark.parametrize("leaf", ["phase1_completed.json", "phase1/case-one/native_statistics.json.gz"])
def test_shared_metadata_tamper_is_rejected_by_final_recheck(published_phase1, leaf):
    fixture = published_phase1
    inputs = checks._Inputs(fixture.series)
    assert list(_rows(inputs, fixture)()) == [fixture.statistics]
    inputs.finish()
    bound = publication.published_metadata_descriptor(fixture.index, leaf)
    path = Path(bound["path"])
    path.write_bytes(path.read_bytes() + b" ")  # Intentional corruption of this test's own copy only.
    with pytest.raises(ValueError, match="changed|differs"):
        inputs.finish()


def test_shared_statistics_tamper_before_read_is_rejected(published_phase1):
    fixture = published_phase1
    inputs = checks._Inputs(fixture.series)
    factory = _rows(inputs, fixture)
    bound = publication.published_metadata_descriptor(fixture.index, "phase1/case-one/native_statistics.json.gz")
    path = Path(bound["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|differs"):
        list(factory())


def test_alias_projection_reuses_one_original_metadata_snapshot(published_phase1, monkeypatch):
    fixture = published_phase1
    original = publication._read_index
    calls = []

    def tracked(bound):
        calls.append(deepcopy(bound))
        return original(bound)

    monkeypatch.setattr(publication, "_read_index", tracked)
    monkeypatch.setattr(published_jobs_snapshot, "_read_index", tracked)
    inputs = checks._Inputs(fixture.series)
    assert list(_rows(inputs, fixture)()) == [fixture.statistics]
    # One read in the original terminal, one for alias projection. Every
    # metadata member still goes through _SnapshotMetadata.read/finish.
    assert len(calls) == 2
    inputs.finish()
