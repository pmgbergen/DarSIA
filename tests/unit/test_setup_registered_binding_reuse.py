"""Retained Setup bindings keep original source checks and operation boundaries."""
from copy import deepcopy
import hashlib
import os

import numpy as np
import pytest

from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import setup_source_reads as reads


@pytest.fixture
def registered(tmp_path):
    path = tmp_path / "labels.bin"
    raw = b"original"
    path.write_bytes(raw)
    bound = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    config = dict(source_mode="canonical-from-import")
    current = dict(revision_id="a" * 32, configuration=config)
    binding = dict(run_id="sample", guarded_sources=[bound], files=dict(labels=bound),
                   labels_shape=[2, 4], registration=dict(kind="historical"))
    calls = []

    def reader(project_root, revision, configuration, run):
        calls.append((project_root, deepcopy(revision), deepcopy(configuration), run))
        sources.verify_registered_sources(binding)
        return deepcopy(binding)

    def read(*, revision=None, configuration=None, run="sample", root=tmp_path):
        return reads.reuse_registered_binding(root, current if revision is None else revision,
            config if configuration is None else configuration, run, reader=reader)

    return dict(path=path, current=current, config=config, binding=binding, calls=calls,
                read=read, root=tmp_path)


def test_one_original_reader_with_reverified_isolated_metadata(registered, monkeypatch):
    case = registered
    verified = sources.verify_registered_sources
    checks = []

    def verify(binding):
        checks.append(deepcopy(binding))
        return verified(binding)

    monkeypatch.setattr(sources, "verify_registered_sources", verify)
    with reads.reuse_setup_source_reads():
        retained = reads._REGISTERED.get()
        first = case["read"]()
        first["labels_shape"].append(9)
        second = case["read"]()
        second["files"]["labels"]["sha256"] = "0" * 64
        assert case["read"]() == case["binding"]
        assert len(case["calls"]) == 1
        assert len(checks) == 3
        assert len(retained["entries"]) == 1
    assert not retained["entries"] and reads._REGISTERED.get() is None


def test_changed_consumed_source_is_rejected_on_reuse(registered):
    case = registered
    with reads.reuse_setup_source_reads():
        case["read"]()
        path = case["path"]
        before = path.stat()
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(b"modified")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, path)
        with pytest.raises(ValueError, match="changed"):
            case["read"]()
        assert len(case["calls"]) == 1
        assert not reads._REGISTERED.get()["entries"]


def test_original_forced_fresh_scope_rehashes_even_a_colliding_stamp(registered, monkeypatch):
    from darsia.fluidflower.painted_adoption_table import _FILES
    case = registered
    with reads.reuse_setup_source_reads() as cache:
        case["read"]()
        stamp = cache.stamp(case["path"])
        case["path"].write_bytes(b"modified")
        monkeypatch.setattr(cache, "stamp", lambda path: stamp)
        with _FILES.fresh(), pytest.raises(ValueError, match="changed"):
            case["read"]()
        assert len(case["calls"]) == 2
        assert not reads._REGISTERED.get()["entries"]


def test_default_new_nested_and_forced_fresh_contexts_use_original_reader(registered):
    case = registered
    case["read"]()
    case["read"]()
    assert len(case["calls"]) == 2
    with reads.reuse_setup_source_reads() as cache:
        case["read"]()
        with reads.reuse_setup_source_reads():
            case["read"]()
            case["read"]()
        case["read"]()
        assert len(case["calls"]) == 4
        with cache.fresh():
            case["read"]()
            case["read"]()
        case["read"]()
        case["read"]()
        assert len(case["calls"]) == 7
    with reads.reuse_setup_source_reads():
        case["read"]()
    assert len(case["calls"]) == 8


@pytest.mark.parametrize("changed", ["project", "revision", "configuration", "run"])
def test_exact_project_revision_configuration_and_run_key(registered, changed):
    case = registered
    kwargs = {}
    if changed == "project":
        kwargs["root"] = case["root"] / "another"
    elif changed == "revision":
        kwargs["revision"] = dict(case["current"], revision_id="b" * 32)
    elif changed == "configuration":
        kwargs["configuration"] = dict(case["config"], changed=True)
    else:
        kwargs["run"] = "another"
    with reads.reuse_setup_source_reads():
        case["read"]()
        case["read"](**kwargs)
        assert len(case["calls"]) == 2


@pytest.mark.parametrize("configuration", [dict(source_mode="imported-rasters"),
    dict(source_mode="canonical-from-import", manual_label_edit={})])
def test_legacy_and_private_manual_bindings_bypass_reuse(registered, configuration):
    with reads.reuse_setup_source_reads():
        for _ in range(2):
            registered["read"](configuration=configuration)
        assert len(registered["calls"]) == 2
        assert not reads._REGISTERED.get()["entries"]


@pytest.mark.parametrize("limit", ["entries", "bytes", "array"])
def test_retention_is_bounded_and_never_keeps_arrays(registered, monkeypatch, limit):
    case = registered
    if limit == "entries":
        monkeypatch.setattr(reads, "_MAXIMUM_REGISTERED_BINDINGS", 1)
    elif limit == "bytes":
        monkeypatch.setattr(reads, "_MAXIMUM_REGISTERED_METADATA_BYTES", 1)
    else:
        case["binding"]["unexpected_array"] = np.zeros((2, 2))
    with reads.reuse_setup_source_reads():
        case["read"]()
        case["read"](run="another")
        case["read"]()
        assert len(case["calls"]) == 3
        entries = reads._REGISTERED.get()["entries"]
        assert len(entries) == (1 if limit == "entries" else 0)
