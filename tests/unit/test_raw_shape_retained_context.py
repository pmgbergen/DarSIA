"""Explicit historical byte copies remain local to one transfer worker."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from darsia.fluidflower import raw_shape_basis_reader as reader
from darsia.fluidflower import painted_adoption_geometry as geometry
from .test_raw_shape_configuration_lineage import synthetic_lineage, _check  # noqa: F401


def _bound(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def retained(tmp_path):
    source = tmp_path / "series.toml"
    source.write_bytes(b"[data]\nwidth = 899.0\n")
    original = _bound(source)
    folder = tmp_path / "retained"
    folder.mkdir()
    copy = folder / "series.toml"
    copy.write_bytes(source.read_bytes())
    alias = dict(original=original, copy=dict(_bound(copy), path=copy.name))
    source.write_bytes(b"[data]\nwidth = 123.0\n")
    return source, original, folder, copy, alias


def test_historical_context_is_scoped_and_not_inherited_by_another_worker(retained):
    source, original, folder, copy, alias = retained
    with pytest.raises(ValueError, match="changed"):
        reader._Pins().add(original, copyable=True)
    with reader.retained_source_context([alias], folder=folder):
        pins = reader._Pins()
        pins.add(original, copyable=True)
        assert pins.rows[reader._key(source)]["physical"] == _bound(copy)
        with ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(lambda: reader._Pins().add(original, copyable=True))
            with pytest.raises(ValueError, match="changed"):
                future.result()
    assert not reader.has_retained_source_context()
    with pytest.raises(ValueError, match="changed"):
        reader._Pins().add(original, copyable=True)


def test_context_does_not_admit_physical_only_source_roles(retained):
    _source, original, folder, _copy, alias = retained
    with reader.retained_source_context([alias], folder=folder):
        with pytest.raises(ValueError, match="role or original identity"):
            reader._Pins().add(original)


def test_context_rechecks_retained_bytes_before_success_and_restores_state(retained):
    _source, original, folder, copy, alias = retained
    with pytest.raises(ValueError, match="changed"):
        with reader.retained_source_context([alias], folder=folder):
            reader._Pins().add(original, copyable=True)
            copy.write_bytes(b"# changed copy\n")
    assert not reader.has_retained_source_context()
    altered = deepcopy(alias)
    altered["copy"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changes historical bytes"):
        with reader.retained_source_context([altered], folder=folder):
            pass


def test_explicit_preparation_copies_take_precedence_over_context(retained, tmp_path):
    _source, original, folder, _copy, alias = retained
    second = tmp_path / "preparation"
    second.mkdir()
    copied = second / "source.bin"
    copied.write_bytes((folder / "series.toml").read_bytes())
    explicit = dict(original=original, copy=dict(_bound(copied), path=copied.name))
    with reader.retained_source_context([alias], folder=folder):
        pins = reader._Pins([explicit], folder=second)
        pins.add(original, copyable=True)
        assert pins.rows[reader._key(original["path"])]["physical"] == _bound(copied)


def test_preparation_retains_selected_historical_bytes_with_original_identity(retained, tmp_path, monkeypatch):
    source, original, folder, copy, alias = retained
    model = dict(run_id="sample", baseline_native_grid=dict(raw_shape_basis_authority={}))
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    model_bound = _bound(model_path)
    proof = dict(raw_shape_basis={}, models={"sample": model_bound})

    def checked_model(value):
        assert value == model
        pins = reader._Pins()
        pins.add(original, copyable=True)
        return reader.ReadShapeBasis({}, {}, pins, compatibility={})

    monkeypatch.setattr(reader, "read_model_basis", checked_model)
    monkeypatch.setattr(reader.ReadShapeBasis, "revalidate", lambda self: self.guards.final())
    with reader.retained_source_context([alias], folder=folder):
        rows, checks = geometry.received_source_rows(proof)
        assert rows == {(original["path"], original["sha256"], original["size_bytes"]): _bound(copy)}
        assert len(checks) == 1
        assert checks[0].copy_source_rows()[0]["original"] == original
        assert source.read_bytes() != copy.read_bytes()
    assert geometry.received_source_rows(proof) == ({}, [])


@pytest.mark.parametrize("kind", ["_PlacementWorker", "_AdoptionWorker"])
def test_default_worker_route_uses_explicit_project_selection_before_emitting_success(retained, kind):
    from darsia.fluidflower import series_layer_view
    _source, original, folder, copy, alias = retained
    project = SimpleNamespace(manifest=dict(tools=dict(native_series=dict(
        retained_shape_basis_sources=dict(format=reader.RETAINED_SOURCE_FORMAT,
                                         folder=str(folder), aliases=[alias])))))
    worker = getattr(series_layer_view, kind)({}, None, dict(project=project))
    method = "_place" if kind == "_PlacementWorker" else "_prepare"
    completed, failed = [], []
    worker.finished.connect(completed.append)
    worker.failed.connect(failed.append)

    def action():
        assert reader.has_retained_source_context()
        reader._Pins().add(original, copyable=True)
        return {"checked": True}

    setattr(worker, method, action)
    worker.run()
    assert completed == [{"checked": True}] and failed == []
    assert not reader.has_retained_source_context()
    completed.clear()

    def changed_action():
        action()
        copy.write_bytes(b"# mutated before publication\n")
        return {"checked": True}

    setattr(worker, method, changed_action)
    worker.run()
    assert completed == [] and len(failed) == 1 and "changed" in failed[0]
    assert not reader.has_retained_source_context()


def test_absent_project_selection_is_unchanged_and_malformed_selection_is_refused(retained):
    _source, _original, folder, _copy, alias = retained
    project = SimpleNamespace(manifest={})
    with reader.project_retained_source_context(project):
        assert not reader.has_retained_source_context()
    project.manifest = dict(tools=dict(native_series=dict(retained_shape_basis_sources=dict(
        format=reader.RETAINED_SOURCE_FORMAT, folder="relative", aliases=[alias]))))
    with pytest.raises(ValueError, match="explicit folder"):
        with reader.project_retained_source_context(project):
            pass


def test_typed_metadata_reader_uses_historical_bytes_only_in_selected_worker(retained):
    from darsia.fluidflower.raw_shape_configuration_lineage import _historical_metadata_loader
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    source, original, folder, _copy, alias = retained
    read = lambda: _get_section_from_toml(source, "data")["width"]
    assert read() == 123.0
    with reader.retained_source_context([alias], folder=folder):
        pins = reader._Pins()
        pins.add(original, copyable=True)
        with _historical_metadata_loader(pins):
            assert read() == 899.0
            with ThreadPoolExecutor(max_workers=1) as worker:
                assert worker.submit(read).result() == 123.0
    assert read() == 123.0


@pytest.mark.parametrize("change_before_return", [False, True])
def test_findings_dispatch_selects_retained_metadata_in_its_own_reader(
        retained, tmp_path, monkeypatch, change_before_return):
    from darsia.fluidflower import painted_transfer_findings as findings
    from darsia.fluidflower.painting_guided_review_bindings import dispatch_findings
    from darsia.fluidflower.raw_shape_configuration_lineage import _historical_metadata_loader
    from darsia.presets.workflows.config.utils import _get_section_from_toml

    source, original, folder, copied, alias = retained
    project = SimpleNamespace(root=tmp_path / "workspace", manifest=dict(tools=dict(native_series=dict(
        retained_shape_basis_sources=dict(format=reader.RETAINED_SOURCE_FORMAT,
                                         folder=str(folder), aliases=[alias])))))
    current = dict(revision_id="synthetic-immutable-revision")
    current_path = project.root / "overrides/geometry_facies/current.json"
    revision_path = project.root / "results/geometry" / current["revision_id"] / "revision.json"
    for path in (current_path, revision_path):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(current), encoding="utf8")
    scheduled = []
    dialog = SimpleNamespace(_closing=False, busy=lambda: False,
        _start=lambda key, job: scheduled.append((key, job)))
    controller = SimpleNamespace(gap_dialog=dialog, view=SimpleNamespace(project=project))
    request = dict(generation=7, review=dict(preparation_context={"explicit": "original"}),
        input_bindings=dict(review={"exact": "review"}, target_bindings=[], analysis_exclusions=None,
                            source_choices=[], display_preferences={}))
    consumed = []

    def collect(revision, review, **kwargs):
        assert reader.has_retained_source_context()
        assert revision == _bound(revision_path) and review == request["input_bindings"]["review"]
        assert kwargs["source_context"] == {"explicit": "original"}
        pins = reader._Pins()
        assert pins.add(original, copyable=True) == original
        with _historical_metadata_loader(pins):
            consumed.append(_get_section_from_toml(source, "data")["width"])
        if change_before_return:
            copied.write_bytes(b"# changed after scientific reader returned\n")
        return dict(report=dict(source_bindings={}))

    monkeypatch.setattr(findings, "collect_transfer_findings", collect)
    dispatch_findings(controller, request)
    assert len(scheduled) == 1 and scheduled[0][0] == ("findings-catalogue", 7)
    assert not reader.has_retained_source_context()
    # Exercise the independent worker, rather than inheriting a caller's scope.
    with ThreadPoolExecutor(max_workers=1) as worker:
        result, active_after = worker.submit(lambda: (
            scheduled[0][1](), reader.has_retained_source_context())).result()
    assert consumed == [899.0] and not active_after, result
    assert result["generation"] == 7
    if change_before_return:
        assert "changed" in result["error"] and "report" not in result
    else:
        assert result["report"]["source_bindings"]["current_workspace"] == _bound(current_path)
    assert not reader.has_retained_source_context()
    assert _get_section_from_toml(source, "data")["width"] == 123.0


def test_historical_metadata_does_not_hide_changed_unselected_geometry(retained, tmp_path):
    from darsia.fluidflower.raw_shape_configuration_lineage import _historical_metadata_loader
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    _source, _original, folder, _copy, alias = retained
    physical = tmp_path / "geometry.toml"
    physical.write_bytes(b"[data]\nwidth = 899.0\n")
    binding = _bound(physical)
    with reader.retained_source_context([alias], folder=folder):
        pins = reader._Pins()
        pins.add(binding, copyable=True)
        physical.write_bytes(b"[data]\nwidth = 111.0\n")
        with pytest.raises(ValueError, match="changed|conflicting"):
            with _historical_metadata_loader(pins):
                _get_section_from_toml(physical, "data")


def test_original_lineage_and_typed_consumption_replay_exact_selected_metadata(synthetic_lineage, tmp_path):
    case = synthetic_lineage
    expected = _check(case)
    pool = tmp_path / "retained-lineage-inputs"
    pool.mkdir()
    aliases = []
    for number, binding in enumerate([case["request"]["series_context"]["manifest"],
                                      case["request"]["configuration_files"][0]]):
        path = pool / f"{number}.toml"
        path.write_bytes(Path(binding["path"]).read_bytes())
        aliases.append(dict(original=deepcopy(binding), copy=dict(_bound(path), path=path.name)))
        Path(binding["path"]).write_bytes(b"# current independent configuration\n")
    with pytest.raises(ValueError):
        _check(case)
    with reader.retained_source_context(aliases, folder=pool):
        pins = reader._Pins()
        assert _check(case, pins) == expected
        pins.final()
        assert pins.rows[reader._key(case["actual_path"])]["physical"]["path"] != str(case["actual_path"])


def test_same_byte_raw_relocation_keeps_original_identity_and_final_file_guards(tmp_path):
    from darsia.fluidflower.rig_point_review import verify_photo_relocation
    original = tmp_path / "previous" / "capture.ARW"
    current = tmp_path / "current" / "capture.ARW"
    current.parent.mkdir()
    current.write_bytes(b"exact original RAW bytes")
    binding = dict(_bound(current), path=str(original))
    relocation = verify_photo_relocation(binding, current)
    with reader.reviewed_photo_context([relocation]):
        pins = reader._Pins()
        assert pins.add(binding) == binding
        assert pins.rows[reader._key(original)]["original"] == binding
        assert pins.rows[reader._key(original)]["physical"] == relocation["current"]
        assert pins.copyable == {}
        with pytest.raises(ValueError, match="another source role"):
            reader._Pins().add(binding, copyable=True)
    pins.final()
    current.write_bytes(b"changed physical RAW data")
    with pytest.raises(ValueError, match="relocat|changed"):
        pins.final()
    with pytest.raises(ValueError, match="match|count"):
        with reader.reviewed_photo_context([relocation]):
            pass


def test_output_guard_protects_both_old_and_current_raw_folder(tmp_path):
    from darsia.fluidflower.rig_point_review import verify_photo_relocation
    original = tmp_path / "previous" / "capture.ARW"
    current = tmp_path / "current" / "capture.ARW"
    current.parent.mkdir()
    current.write_bytes(b"exact original RAW bytes")
    binding = dict(_bound(current), path=str(original))
    relocation = verify_photo_relocation(binding, current)
    basis = tmp_path / "basis"
    basis.mkdir()
    model = dict(reference_image=binding, baseline_native_grid=dict(
        raw_shape_basis_authority={}, renderer=dict(path=str(basis / "renderer.json")),
        parent_grid_compatibility=dict(fields={})))
    with pytest.raises(FileNotFoundError):
        reader.guard_output_root(tmp_path / "new-review", [model])
    with reader.reviewed_photo_context([relocation]):
        reader.guard_output_root(tmp_path / "new-review", [model])
        for protected in (original.parent, current.parent, basis):
            with pytest.raises(ValueError, match="outside"):
                reader.guard_output_root(protected / "new-review", [model])
