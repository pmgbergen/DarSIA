"""Shared storage relocation preserves frozen bytes and bounded input admission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os

import pytest
import toml

from darsia.fluidflower import storage_paths
from darsia.fluidflower.store_map import parse_store_map
from darsia.fluidflower.pipeline import store_runtime
from darsia.fluidflower.pipeline.job_inputs import catalog_order_key


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def moved(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DARSIA_") and name.endswith("_PROJECT_ROOT"):
            monkeypatch.delenv(name)
    registry = tmp_path / "registry.toml"
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    project = tmp_path / "project"
    root, results = project / "DarSIA", project / "Results"
    root.mkdir(parents=True)
    results.mkdir()
    old = root / "results"
    document = {"schema_version": 1, "series_id": "Study",
        "paths": {"results": "../Results"},
        "storage": {"path_aliases": {str(old): str(results)}}}
    _write(root / "series.toml", toml.dumps(document).encode())
    storage_paths.refresh_storage_paths()
    payload = b'{"run":"r1","gain":2.5}\n'
    actual = _write(results / "config.json", payload)
    source = _write(root / "frozen" / "execution.toml", b'job_id="stored"\n')
    manifest = _write(root / "frozen" / "series.toml", b'schema_version=1\nseries_id="Study"\n')
    def reference(relative, data):
        return dict(root_alias="series", relative_path=relative,
                    sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), role="configuration")
    config = reference("results/config.json", payload)
    task = dict(run="r1", configuration=config,
        series=reference("frozen/execution.toml", source.read_bytes()),
        series_manifest=reference("frozen/series.toml", manifest.read_bytes()))
    inputs = [dict(reference=task[key], uses=[use]) for key, use in (
        ("configuration", "run.configuration"), ("series", "run.execution_series"),
        ("series_manifest", "run.series_manifest"))]
    task["input_catalog"] = dict(format="darsia-role-input-catalog-v1", complete=True,
        run="r1", configuration=config, inputs=sorted(inputs, key=lambda row: catalog_order_key(row["reference"])))
    stores = parse_store_map(toml.dumps({"machine": {"id": "local"}, "stores": {"Study": {
        "series": str(root), "results": str(results), "project": str(project)}}}).encode())
    return dict(root=root, project=project, old=old, results=results, actual=actual,
                document=document, registry=registry, task=task, stores=stores, payload=payload)


def test_worker_seed_resolves_old_inert_reference_and_verifies_original_hash(moved):
    task, stores = moved["task"], moved["stores"]
    before = deepcopy(task)
    rows = store_runtime.resolve_catalog(task, stores, "Study")
    verified = store_runtime.verify_inputs(task, rows)
    assert verified.for_use("run.configuration").path == moved["actual"]
    assert verified.metadata("run.configuration") == {"run": "r1", "gain": 2.5}
    assert task == before and moved["actual"].read_bytes() == moved["payload"]
    assert not moved["old"].exists()
    task["configuration"]["sha256"] = "0" * 64
    with pytest.raises(store_runtime.InputHashMismatch, match="SHA/bytes"):
        store_runtime.verify_inputs(task, store_runtime.resolve_catalog(task, stores, "Study"))


def test_native_frozen_config_uses_shared_alias_and_keeps_authored_bytes(moved):
    from darsia.presets.workflows.config.data import DataConfig
    from darsia.presets.workflows.config.rig import RigConfig

    moved["registry"].write_text(toml.dumps({"schema_version": 1, "series": [
        {"id": "Study", "root": str(moved["root"])}]}))
    folder = moved["project"] / "photographs"
    _write(folder / "baseline.jpg", b"not decoded")
    original = toml.dumps({"data": {"folders": [str(folder)], "baseline": "baseline.jpg",
        "format": "JPG", "results": str(moved["old"] / "r1")},
        "rig": {"width": 1.0, "height": 1.0, "dim": 2, "path": str(moved["old"] / "r1/setup/rig")}}).encode()
    config = _write(moved["root"] / "config/runs/r1.toml", original)
    data = DataConfig().load(config, require_data=False, require_results=False)
    rig = RigConfig().load(config, results=data.results)
    assert data.results == moved["results"] / "r1"
    assert rig.path == moved["results"] / "r1/setup/rig"
    assert config.read_bytes() == original


def test_shared_aliases_are_cached_until_explicit_refresh_and_merge_local(moved, monkeypatch):
    moved["registry"].write_text(toml.dumps({"schema_version": 1, "series": [
        {"id": "Study", "root": str(moved["root"])}]}))
    opened = []
    original_open = Path.open
    def counted(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == moved["root"] / "series.toml" and "r" in mode:
            opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted)
    for _ in range(10):
        assert storage_paths.resolve_storage_path(moved["old"] / "config.json") == moved["actual"]
    assert len(opened) == 1
    document = deepcopy(moved["document"])
    newer = moved["project"] / "newer results"
    document["storage"]["path_aliases"][str(moved["old"])] = str(newer)
    (moved["root"] / "series.toml").write_text(toml.dumps(document))
    assert storage_paths.resolve_storage_path(moved["old"] / "config.json") == moved["actual"]
    storage_paths.refresh_storage_paths(moved["root"])
    assert storage_paths.resolve_storage_path(moved["old"] / "config.json") == newer / "config.json"
    assert len(opened) == 2
    local = moved["project"] / "explicit local mirror"
    moved["registry"].write_text(toml.dumps({"schema_version": 1,
        "series": [{"id": "Study", "root": str(moved["root"])}],
        "storage": {"path_aliases": {str(moved["old"]): str(local)}}}))
    assert storage_paths.resolve_storage_path(moved["old"] / "config.json") == local / "config.json"


def test_alias_outside_explicit_roots_is_rejected_before_input_use(moved, tmp_path):
    outside = tmp_path / "outside"
    _write(outside / "config.json", moved["payload"])
    document = deepcopy(moved["document"])
    document["storage"]["path_aliases"][str(moved["old"])] = str(outside)
    (moved["root"] / "series.toml").write_text(toml.dumps(document))
    storage_paths.refresh_storage_paths()
    with pytest.raises(store_runtime.RunUnavailable, match="escapes.*results and project"):
        store_runtime.resolve_catalog(moved["task"], moved["stores"], "Study")


def test_explicit_project_root_allows_its_relocated_input_and_capacity_reader(moved):
    from darsia.fluidflower.pipeline.capacity import configuration_for_task

    archived = moved["project"] / "archived results"
    _write(archived / "config.json", moved["payload"])
    document = deepcopy(moved["document"])
    document["storage"]["path_aliases"][str(moved["old"])] = str(archived)
    (moved["root"] / "series.toml").write_text(toml.dumps(document))
    storage_paths.refresh_storage_paths()
    assert json.loads(moved["payload"]) == configuration_for_task(moved["stores"], "Study", moved["task"])


def test_remap_cannot_escape_through_a_link(moved, tmp_path):
    outside = tmp_path / "outside"
    _write(outside / "config.json", moved["payload"])
    try:
        (moved["results"] / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Fixture account cannot create a directory link")
    moved["task"]["configuration"]["relative_path"] = "results/link/config.json"
    with pytest.raises(store_runtime.RunUnavailable, match="through a link"):
        store_runtime.resolve_catalog(moved["task"], moved["stores"], "Study")
