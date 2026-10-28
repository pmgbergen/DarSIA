"""Original ROI provenance uses only a private project's declared native files."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from darsia.fluidflower import metric_analysis_domains as domains
from darsia.fluidflower import painting_analysis_box_sources as sources


_FIRST = [[.9000000000000001, .49], [.1234567890123456, .02]]
_SECOND = [[.25, .30], [.75, .10]]


def _roi(name, corners, *, label=None):
    text = (f'[[roi]]\nname = "{name}"\n'
            f'corner_1 = {corners[0]!r}\ncorner_2 = {corners[1]!r}\n')
    return text + (f"label = {label}\n" if label is not None else "")


def _write_series(root):
    common = root / "config/shared.toml"
    run = root / "config/experiments/first.toml"
    other = root / "config/experiments/second.toml"
    run.parent.mkdir(parents=True)
    common.write_text(_roi("box1", _FIRST), encoding="utf8")
    run.write_text(_roi("box2", _SECOND), encoding="utf8")
    other.write_text("# Another explicitly declared private experiment.\n", encoding="utf8")
    manifest = root / "series.toml"
    manifest.write_text('schema_version = 1\nseries_id = "testseries"\n'
        '[paths]\ncommon_config = "config/shared.toml"\nconfig = "config"\n'
        '[runs]\nids = ["run_a", "run_b"]\n[runs.config_files]\n'
        'run_a = "config/experiments/first.toml"\nrun_b = "config/experiments/second.toml"\n',
        encoding="utf8")
    return SimpleNamespace(root=root, common=common, run=run, other=other, manifest=manifest)


@pytest.fixture
def context(tmp_path, monkeypatch):
    private = _write_series(tmp_path / "selected-series")
    registry = tmp_path / "private-registry.toml"
    registry.write_text("schema_version = 1\n", encoding="utf8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    monkeypatch.setenv("DARSIA_TESTSERIES_PROJECT_ROOT", str(private.root))
    private.project = SimpleNamespace(root=private.root / "geometry-package", manifest={
        "series_id": "testseries", "tools": {"native_series": {
            "series_id": "testseries", "series_root": str(private.root)}}})
    return private


def _read(context, **kwargs):
    return sources.load_legacy_box_source(context.project, "run_a",
        reference_config_path=kwargs.pop("reference_config_path", context.run), **kwargs)


def _bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_actual_ordered_reader_exact_values_and_sha_bound_source_are_read_only(context, monkeypatch):
    files_before, project_before = _bytes(context.root), deepcopy(context.project.manifest)
    called = []
    original = sources.RoiRegistry.load
    def observed(self, paths):
        called.append(tuple(paths))
        return original(self, paths)
    monkeypatch.setattr(sources.RoiRegistry, "load", observed)
    result = _read(context)
    assert called == [(context.common.resolve(), context.run.resolve())]
    assert result["values"] == {"box1": _FIRST, "box2": _SECOND}
    assert result["values"]["box1"][0][0].hex() == _FIRST[0][0].hex()
    source = result["source"]
    assert source["run_id"] == "run_a"
    assert source["series_root"] == str(context.root.resolve())
    assert source["reference_config_path"] == str(context.run.resolve())
    assert source["series_manifest"]["path"] == str(context.manifest.resolve())
    assert source["color_state"] is None
    assert [row["path"] for row in source["configurations"]] == [str(context.common.resolve()), str(context.run.resolve())]
    for bound in [source["series_manifest"], *source["configurations"]]:
        data = Path(bound["path"]).read_bytes()
        assert bound["sha256"] == hashlib.sha256(data).hexdigest()
        assert bound["size_bytes"] == len(data)
    assert _bytes(context.root) == files_before
    assert context.project.manifest == project_before


def test_values_remain_pending_legacy_frame_and_original_corner_order_survives_review(context):
    result = _read(context)
    assert result["source"]["frame"] == "legacy_nominal_0.9x0.49_m"
    assert result["source"]["decision"] == "pending"
    rows = domains.import_legacy_boxes(result["values"], source=result["source"])
    assert rows[0]["bounds_mm"][1] == _FIRST[0][0] * 1000
    assert rows[0]["legacy_import"]["original_value"] == _FIRST
    assert all(row["legacy_import"]["decision"] == "pending" for row in rows)
    with pytest.raises(ValueError, match="Review imported"):
        domains.mass_boxes(dict(domains=rows))
    for row in rows:
        row["legacy_import"]["decision"] = "use_coordinates_in_measured_frame"
    assert domains.mass_boxes(dict(domains=rows)) == {"box1": _FIRST, "box2": _SECOND}


def test_absent_pair_returns_none_without_searching_for_other_inputs(context):
    context.common.write_text(_roi("other", [[0., 0.], [1., 1.]]), encoding="utf8")
    context.run.write_text("# No original box pair in this declared file.\n", encoding="utf8")
    # A convincing unselected file must not be discovered or imported.
    (context.root / "latest-job.toml").write_text(_roi("box1", _FIRST) + _roi("box2", _SECOND), encoding="utf8")
    before = _bytes(context.root)
    assert _read(context) is None
    assert _bytes(context.root) == before


def test_existing_measured_box_conflict_is_not_read_or_overwritten(context):
    saved = context.project.root / "overrides/analysis_domains/measured_frame/current.json"
    saved.parent.mkdir(parents=True)
    existing = dict(domains=[dict(role="box1", bounds_mm=[10., 20., 30., 40.])],
                    note="An existing draft is owned by its editor")
    saved.write_text(json.dumps(existing), encoding="utf8")
    before = saved.read_bytes()
    result = _read(context)
    assert result["values"] == {"box1": _FIRST, "box2": _SECOND}
    assert saved.read_bytes() == before
    assert result["source"]["decision"] == "pending"


def test_partial_pair_is_refused(context):
    context.run.write_text("# No box2.\n", encoding="utf8")
    with pytest.raises(ValueError, match="Both original box1 and box2"):
        _read(context)


@pytest.mark.parametrize("restricted", ["box1", "box2"])
def test_label_restricted_box_is_never_silently_dropped(context, restricted):
    path, value = (context.common, _FIRST) if restricted == "box1" else (context.run, _SECOND)
    path.write_text(_roi(restricted, value, label=3), encoding="utf8")
    with pytest.raises(ValueError, match="unrestricted"):
        _read(context)


@pytest.mark.parametrize("duplicate", ["box1", "unrelated"])
def test_duplicate_registry_names_are_refused_even_when_values_agree(context, duplicate):
    if duplicate == "box1":
        context.run.write_text(context.run.read_text(encoding="utf8") + _roi("box1", _FIRST), encoding="utf8")
    else:
        for path in (context.common, context.run):
            path.write_text(path.read_text(encoding="utf8") + _roi("unrelated", [[0., 0.], [1., 1.]]), encoding="utf8")
    with pytest.raises(ValueError, match="duplicated"):
        _read(context)


def test_wrong_reference_configuration_does_not_fall_back_to_selected_run(context):
    with pytest.raises(ValueError, match="captured reference configuration differs"):
        _read(context, reference_config_path=context.other)


def test_unknown_run_and_missing_native_link_are_refused(context):
    with pytest.raises(ValueError, match="uniquely declared"):
        sources.load_legacy_box_source(context.project, "unlisted", reference_config_path=context.run)
    context.project.manifest["tools"] = {}
    with pytest.raises(ValueError, match="explicit native-series link"):
        _read(context)


def test_link_series_id_must_match_its_manifest(context):
    context.project.manifest["tools"]["native_series"]["series_id"] = "differentseries"
    with pytest.raises(ValueError, match="ID and selected manifest differ"):
        _read(context)


def test_run_ids_that_collide_after_normalization_are_not_a_unique_reference(context):
    text = context.manifest.read_text(encoding="utf8")
    context.manifest.write_text(text.replace('["run_a", "run_b"]', '["run_a", "RUN_A", "run_b"]'), encoding="utf8")
    with pytest.raises(ValueError, match="uniquely declared"):
        _read(context)


def test_registered_root_must_be_the_loaded_project_root(context, tmp_path, monkeypatch):
    other = _write_series(tmp_path / "other-series")
    monkeypatch.setenv("DARSIA_TESTSERIES_PROJECT_ROOT", str(other.root))
    with pytest.raises(ValueError, match="registered series root differs"):
        _read(context)


@pytest.mark.parametrize("invalid_chain", ["reversed", "duplicate", "unlisted_overlay"])
def test_configuration_order_and_declared_paths_are_verified(context, monkeypatch, invalid_chain):
    if invalid_chain == "reversed":
        paths = [context.run, context.common]
    elif invalid_chain == "duplicate":
        paths = [context.common, context.common]
    else:
        extra = context.root / "other.toml"
        extra.write_text("# Explicitly unselected input.\n", encoding="utf8")
        paths = [context.common, context.run, extra]
    monkeypatch.setattr(sources.series_paths, "config_files", lambda *args: paths)
    with pytest.raises(ValueError, match="configuration chain"):
        _read(context)


def test_missing_declared_file_is_not_silently_skipped_by_registry(context):
    context.run.rename(context.run.with_suffix(".unavailable"))
    with pytest.raises(FileNotFoundError):
        _read(context)


def test_active_color_overlay_uses_original_native_order_and_binds_selector(context):
    state = context.run.parent / ".color_state/run_a.txt"
    state.parent.mkdir()
    state.write_text("on\n", encoding="utf8")
    overlay = context.root / "config/coloron.toml"
    overlay.write_text(_roi("another-region", [[0., 0.], [1., 1.]]), encoding="utf8")
    result = _read(context)
    assert result["values"] == {"box1": _FIRST, "box2": _SECOND}
    assert [row["path"] for row in result["source"]["configurations"]] == [
        str(path.resolve()) for path in (context.common, context.run, overlay)]
    assert result["source"]["color_state"]["sha256"] == hashlib.sha256(state.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed", ["common", "run", "manifest", "selector"])
def test_changed_bytes_or_selector_after_registry_load_refuse_result(context, monkeypatch, changed):
    original = sources.RoiRegistry.load
    def racing(self, paths):
        result = original(self, paths)
        if changed == "selector":
            path = context.run.parent / ".color_state/run_a.txt"
            path.parent.mkdir()
            path.write_text("off\n", encoding="utf8")
        else:
            path = getattr(context, changed)
            path.write_bytes(path.read_bytes() + b"\n# A concurrent metadata edit.\n")
        return result
    monkeypatch.setattr(sources.RoiRegistry, "load", racing)
    with pytest.raises(ValueError, match="source changed"):
        _read(context)


def test_changed_active_chain_after_load_is_refused(context, monkeypatch):
    state = context.run.parent / ".color_state/run_a.txt"
    state.parent.mkdir()
    state.write_text("on\n", encoding="utf8")
    (context.root / "config/coloron.toml").write_text("# Active native overlay.\n", encoding="utf8")
    original = sources.RoiRegistry.load
    def racing(self, paths):
        result = original(self, paths)
        state.write_text("off\n", encoding="utf8")
        return result
    monkeypatch.setattr(sources.RoiRegistry, "load", racing)
    with pytest.raises(ValueError, match="source changed"):
        _read(context)


def test_reader_can_run_in_metadata_worker_without_gui_reference_getter(context, monkeypatch):
    calls = []
    original = sources.RoiRegistry.load
    def observed(self, paths):
        calls.append(threading.get_ident())
        return original(self, paths)
    monkeypatch.setattr(sources.RoiRegistry, "load", observed)
    main = threading.get_ident()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_read, context).result(timeout=5.)
    assert result["values"] == {"box1": _FIRST, "box2": _SECOND}
    assert len(calls) == 1 and calls[0] != main
