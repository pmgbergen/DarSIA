from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from darsia.fluidflower import phase_model_library as library

from darsia.fluidflower import model_index_names as ui


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture
def model(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    (series / "series.toml").write_text('[series]\nid="ffac"\n', encoding="utf-8")
    root = series / "results" / "phase-model-fixture"
    root.mkdir(parents=True)
    request = dict(
        model_id=root.name, output_root=str(root), settings={"series": ["ffac"]}
    )
    write(root / "training_request.json", request)
    write(
        root / "summary.json",
        dict(model_id=root.name, inventory={"datasets": ["ffac"]}),
    )
    name = dict(
        format=ui.NAME_FORMAT,
        model_id=root.name,
        training_request_sha256=hashlib.sha256(
            (root / "training_request.json").read_bytes()
        ).hexdigest(),
        display_name="Testmodell \u00e6\u00f8\u00e5",
        recorded_by="researcher",
        recorded_utc="2026-10-16T00:00:00Z",
    )
    write(root / "model_ui_name.json", name)
    return series, root, name


def entry_for(pointer, root):
    document = json.loads(pointer.read_bytes())
    return dict(
        root=str(root),
        model_id=root.name,
        library_reference=str(pointer),
        library_source_identity=document["source_identity"],
    )


def test_original_indexer_and_science_bytes_unchanged(model):
    series, root, name = model
    source = {p.name: p.read_bytes() for p in root.iterdir()}
    pointer = library.index_model(series, root)
    doc = json.loads(pointer.read_bytes())
    doc["future_pointer"] = {"keep": [1, 2, 3]}
    doc["metadata"]["future_metadata"] = {"nested": "preserved"}
    write(pointer, doc)
    raw = pointer.read_bytes()
    ui.write_index_ui_name(pointer, name, expected_bytes=raw)
    updated = json.loads(pointer.read_bytes())
    assert updated["source_identity"] == doc["source_identity"]
    assert updated["target"] == doc["target"]
    assert updated["future_pointer"] == doc["future_pointer"]
    assert updated["metadata"]["future_metadata"] == doc["metadata"]["future_metadata"]
    assert updated["metadata"]["display_name"] == name["display_name"]
    assert all(p.read_bytes() == source[p.name] for p in root.iterdir())


def test_explicit_reindex_reapplies_original_sidecar(model):
    series, root, name = model
    pointer = ui.index_model_with_ui_name(series, root)
    assert ui.model_ui_display_name(entry_for(pointer, root)) == name["display_name"]
    library.index_model(
        series, root
    )  # Original producer still owns its original semantics.
    assert "ui_display_name" not in json.loads(pointer.read_bytes())["metadata"]
    assert ui.model_ui_display_name(entry_for(pointer, root)) == name["display_name"]
    ui.index_model_with_ui_name(series, root)
    assert (
        json.loads(pointer.read_bytes())["metadata"]["ui_display_name"]["model_id"]
        == root.name
    )


def test_index_only_binding_survives_explicit_reindex(model):
    series, root, name = model
    pointer = ui.index_model_with_ui_name(series, root)
    (root / "model_ui_name.json").unlink()
    ui.index_model_with_ui_name(series, root)
    assert ui.model_ui_display_name(entry_for(pointer, root)) == name["display_name"]


def test_wrong_id_or_request_does_not_write(model):
    series, root, name = model
    pointer = library.index_model(series, root)
    raw = pointer.read_bytes()
    for update in ({"model_id": "other"}, {"training_request_sha256": "a" * 64}):
        with pytest.raises(ValueError, match="does not match"):
            ui.write_index_ui_name(pointer, dict(name, **update), expected_bytes=raw)
        assert pointer.read_bytes() == raw
    with pytest.raises(ValueError, match="Completed model ID differs"):
        ui.index_model_with_ui_name(series, root, expected_model_id="other")
    assert pointer.read_bytes() == raw


def test_expected_bytes_cas_rejects_external_edit(model):
    series, root, name = model
    pointer = library.index_model(series, root)
    raw = pointer.read_bytes()
    doc = json.loads(raw)
    doc["external"] = True
    write(pointer, doc)
    external = pointer.read_bytes()
    with pytest.raises(RuntimeError, match="since it was loaded"):
        ui.write_index_ui_name(pointer, name, expected_bytes=raw)
    assert pointer.read_bytes() == external


def test_save_time_external_edit_is_not_overwritten(model, monkeypatch):
    series, root, name = model
    pointer = library.index_model(series, root)
    original = Path.read_bytes
    count = 0
    external = b""

    def read(path):
        nonlocal count, external
        if path == pointer:
            count += 1
            if count == 2:
                doc = json.loads(original(path))
                doc["external"] = "changed during save"
                write(path, doc)
                external = original(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(RuntimeError, match="while saving"):
        ui.write_index_ui_name(pointer, name)
    assert original(pointer) == external
    assert not list(pointer.parent.glob("*.tmp"))


def test_conflicting_bound_name_is_preserved(model):
    series, root, name = model
    pointer = ui.index_model_with_ui_name(series, root)
    raw = pointer.read_bytes()
    with pytest.raises(ValueError, match="already present"):
        ui.write_index_ui_name(pointer, dict(name, display_name="another name"))
    assert pointer.read_bytes() == raw


def test_passive_index_label_reads_metadata_only_and_never_writes(model, monkeypatch):
    series, root, name = model
    pointer = ui.index_model_with_ui_name(series, root)
    entry = entry_for(pointer, root)

    def forbidden(*a, **k):
        raise AssertionError("No navigation writes, scans or source reads")

    monkeypatch.setattr(ui, "saved_ui_name", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(ui.os, "replace", forbidden)
    assert ui.model_ui_display_name(entry) == name["display_name"]
    assert (
        ui.model_reference_label(series, root.name)
        == name["display_name"] + " \u00b7 " + root.name
    )
    assert (
        ui.model_reference_label(series, str(pointer))
        == name["display_name"] + " \u00b7 " + root.name
    )
    assert ui.model_reference_label(series, "not-indexed") == "not-indexed"


def test_stale_index_identity_is_not_used(model):
    series, root, name = model
    pointer = ui.index_model_with_ui_name(series, root)
    entry = entry_for(pointer, root)
    entry["library_source_identity"] = dict(
        entry["library_source_identity"], model_id="wrong"
    )
    (root / "model_ui_name.json").unlink()
    assert ui.model_ui_display_name(entry) == ""


def test_corrupt_sidecar_or_changed_request_is_not_shown(model):
    _series, root, name = model
    write(root / "model_ui_name.json", dict(name, model_id="other"))
    assert ui.saved_ui_name(root, root.name) is None
    write(root / "model_ui_name.json", name)
    request = json.loads((root / "training_request.json").read_bytes())
    request["new_field"] = "external"
    write(root / "training_request.json", request)
    assert ui.saved_ui_name(root, root.name) is None


def test_reference_outside_current_series_is_not_adopted(model, tmp_path):
    series, root, _name = model
    pointer = ui.index_model_with_ui_name(series, root)
    other = tmp_path / "other"
    other.mkdir()
    assert ui.model_reference_label(other, str(pointer)) == str(pointer)


def test_absent_or_malformed_names_leave_original_text(model):
    series, root, _name = model
    (root / "model_ui_name.json").write_text("{bad", encoding="utf-8")
    pointer = library.index_model(series, root)
    assert ui.model_ui_display_name(entry_for(pointer, root)) == ""
    assert ui.model_reference_label(series, root.name) == root.name


@pytest.fixture(scope="module")
def native_module():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from darsia.fluidflower import native_phase_steps as native

    # Resolve only the already installed package's repository test fixture.
    import darsia.fluidflower.native_phase_steps as original_native

    tests = (
        Path(original_native.__file__).resolve().parents[3]
        / "tests"
        / "unit"
        / "gui"
        / "test_native_phase_steps.py"
    )
    fixture_spec = importlib.util.spec_from_file_location(
        "_retained_native_steps_fixture", tests
    )
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    yield native, fixture, app


def test_native_completion_indexes_real_model_and_keeps_original_callback(
    model, native_module
):
    series, root, name = model
    native, fixture, _app = native_module
    studio = fixture.FakeStudio()
    calls = []
    studio._record_completed_model = lambda model_id, folder: calls.append(
        (model_id, folder)
    )
    studio._model_version_text = lambda entry: entry["model_id"]
    steps = native.NativePhaseSteps(studio)
    steps._named_model_context[root.name] = (series, root.resolve())
    studio._record_completed_model(root.name, root)
    pointer = library.library_root(series) / (root.name + ".json")
    assert pointer.is_file()
    assert (
        json.loads(pointer.read_bytes())["metadata"]["display_name"]
        == name["display_name"]
    )
    assert calls == [(root.name, root)]
    (root / "model_ui_name.json").unlink()  # Labels now demonstrably use the index.
    entry = entry_for(pointer, root)
    assert name["display_name"] in studio._validation_model_label(entry)
    assert name["display_name"] in studio._model_version_text(entry)
    from darsia.fluidflower.ui.review_version_controls import model_choice_arguments

    args = model_choice_arguments(
        [dict(entry, display_name=studio._validation_model_label(entry))], root
    )
    choice = json.loads(args[args.index("--review-model-choice") + 1])
    assert name["display_name"] in choice["label"]
    assert choice["root"] == str(root)
    steps.deleteLater()
    studio.deleteLater()


def test_native_completion_error_preserves_completed_callback(
    model, native_module, tmp_path
):
    series, root, _name = model
    native, fixture, _app = native_module
    studio = fixture.FakeStudio()
    calls = []
    studio._record_completed_model = lambda model_id, folder: calls.append(
        (model_id, folder)
    )
    steps = native.NativePhaseSteps(studio)
    steps._named_model_context[root.name] = (
        series,
        (tmp_path / "another-root").resolve(),
    )
    studio._record_completed_model(root.name, root)
    assert calls == [(root.name, root)]
    assert "Model completed;" in studio.status.text()
    assert "folder differs" in studio.status.text()
    assert not library.library_root(series).exists()
    steps.deleteLater()
    studio.deleteLater()


def test_explicit_named_launch_captures_own_series_for_later_completion(
    model, native_module
):
    series, root, _name = model
    native, fixture, _app = native_module
    (root / "model_ui_name.json").unlink()
    studio = fixture.FakeStudio()
    studio._record_completed_model = lambda *args: None
    steps = native.NativePhaseSteps(studio)
    steps.series_root = series
    steps._ready = True
    steps.model_name.setText("new display name")
    launched = []
    steps._original_launch = lambda *args: launched.append(args) or "original-result"
    request = dict(
        model_id=root.name,
        output_root=str(root),
        request_path=str(root / "training_request.json"),
    )
    result = steps._launch_named_job(
        request, ["fixed-original-command"], root, "Original training status"
    )
    assert result == "original-result" and len(launched) == 1
    assert steps._named_model_context[root.name] == (series, root.resolve())
    steps.series_root = series / "other-selected-series"
    studio._record_completed_model(root.name, root)
    pointer = library.library_root(series) / (root.name + ".json")
    assert (
        json.loads(pointer.read_bytes())["metadata"]["display_name"]
        == "new display name"
    )
    assert not library.library_root(steps.series_root).exists()
    steps.deleteLater()
    studio.deleteLater()


@pytest.mark.parametrize("field,value", [("metadata", [1]), ("source_identity", [1])])
def test_malformed_pointer_stays_unlabelled_in_passive_ui(model, field, value):
    series, root, _name = model
    pointer = ui.index_model_with_ui_name(series, root)
    entry = entry_for(pointer, root)
    (root / "model_ui_name.json").unlink()
    document = json.loads(pointer.read_bytes())
    document[field] = value
    write(pointer, document)
    assert ui.model_ui_display_name(entry) == ""
    assert ui.model_reference_label(series, root.name) == root.name
