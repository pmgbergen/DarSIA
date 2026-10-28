"""Operational location admission; synthetic files only, no numerical producer."""
from pathlib import Path
import hashlib
import json

import pytest

from darsia.fluidflower.store_map import parse_store_map
from darsia.fluidflower.pipeline.store_runtime import (
    ResolvedInput, RunUnavailable, InputHashMismatch,
    resolve_catalog, verify_inputs,
)


def reference(path, payload, role="configuration"):
    return dict(root_alias="series", relative_path=path,
                sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), role=role)


def task_fixture(tmp_path):
    payload = b'{"run":"ac52"}\n'
    (tmp_path / "run.json").write_bytes(payload)
    (tmp_path / "execution.toml").write_bytes(b"job_id='gate'\n")
    (tmp_path / "series.toml").write_bytes(b"series_id='FF_AC'\n")
    config = reference("run.json", payload)
    task = dict(run="ac52", configuration=config,
                series=reference("execution.toml", b"job_id='gate'\n"),
                series_manifest=reference("series.toml", b"series_id='FF_AC'\n"))
    task["input_catalog"] = dict(format="darsia-role-input-catalog-v1", complete=True,
                                 run="ac52", configuration=config,
                                 inputs=[dict(reference=task["series"], uses=["run.execution_series"]),
                                         dict(reference=config, uses=["run.configuration"]),
                                         dict(reference=task["series_manifest"], uses=["run.series_manifest"])])
    text = ("[machine]\nid='test-main'\n[stores.FF_AC]\nseries=" +
            json.dumps(str(tmp_path)) + "\nresults=" +
            json.dumps(str(tmp_path / "local-results")) + "\n")
    return task, parse_store_map(text.encode()), payload


def test_availability_only_stats_and_never_reads_payload(tmp_path, monkeypatch):
    task, stores, _ = task_fixture(tmp_path)
    # Load shared location settings once before the finite input-stat pass.
    from darsia.fluidflower.storage_paths import resolve_storage_path
    resolve_storage_path(tmp_path / "run.json", series_roots=(tmp_path,))
    def forbidden(*args, **kwargs):
        raise AssertionError("Preclaim admission must not open a payload")
    monkeypatch.setattr(Path, "open", forbidden)
    rows = resolve_catalog(task, stores, "FF_AC")
    assert len(rows) == 3


def test_missing_input_cannot_claim_or_create_results(tmp_path):
    task, stores, _ = task_fixture(tmp_path)
    task["input_catalog"]["configuration"]["relative_path"] = "missing.json"
    before = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(RunUnavailable, match="run.configuration"):
        resolve_catalog(task, stores, "FF_AC")
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_wrong_byte_count_stays_unavailable_before_claim(tmp_path):
    task, stores, _ = task_fixture(tmp_path)
    task["configuration"]["bytes"] += 1
    with pytest.raises(RunUnavailable, match="byte count"):
        resolve_catalog(task, stores, "FF_AC")


def test_wrong_hash_fails_before_any_metadata_or_frame(tmp_path):
    task, stores, _ = task_fixture(tmp_path)
    task["configuration"]["sha256"] = "0" * 64
    rows = resolve_catalog(task, stores, "FF_AC")
    with pytest.raises(InputHashMismatch, match="SHA/bytes differ"):
        verify_inputs(task, rows)


def test_same_physical_input_is_hashed_once_across_roles(tmp_path, monkeypatch):
    task, stores, _ = task_fixture(tmp_path)
    rows = resolve_catalog(task, stores, "FF_AC")
    other = ResolvedInput(dict(rows[0].reference, role="model"), rows[0].path,
                          ("model.same_bytes",))
    opens = []
    original_open = Path.open
    def counted(path, *args, **kwargs):
        opens.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted)
    bound = verify_inputs(task, (*rows, other))
    assert opens.count(str(rows[0].path)) == 1
    assert bound.for_use("model.same_bytes").reference["role"] == "model"


def test_metadata_read_refuses_changed_input_after_admission(tmp_path):
    task, stores, _ = task_fixture(tmp_path)
    bound = verify_inputs(task, resolve_catalog(task, stores, "FF_AC"))
    (tmp_path / "run.json").write_bytes(b'{"run":"ac52","changed":true}\n')
    with pytest.raises(InputHashMismatch, match="changed before metadata"):
        bound.metadata("run.configuration")


def test_metadata_uses_original_bound_bytes(tmp_path):
    task, stores, payload = task_fixture(tmp_path)
    bound = verify_inputs(task, resolve_catalog(task, stores, "FF_AC"))
    assert bound.metadata("run.configuration") == {"run": "ac52"}
    assert (tmp_path / "run.json").read_bytes() == payload
