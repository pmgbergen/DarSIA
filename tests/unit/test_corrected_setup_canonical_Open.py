"""Compile-only metadata Open-routing cases; no Qt window or actual geometry.

Synthetic opaque files are used only to exercise receipt/path/stat routing.
They do not qualify archive content, Store equivalence or scientific admission.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower import corrected_setup_ui as subject
from darsia.fluidflower.corrected_setup import _replacements
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT


def descriptor(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


@pytest.fixture
def opened(tmp_path, monkeypatch):
    series = tmp_path / "private_series"
    package = series / "geometry_facies"
    package.mkdir(parents=True)
    manifest = series / "series.toml"
    manifest.write_text('[runs]\nids=["reference","other"]\n')
    runs, bundles = {}, {}
    for run in ("reference", "other"):
        source = series / "original" / run
        source.mkdir(parents=True)
        archive = source / "opaque.npz"
        archive.write_bytes(b"synthetic opaque archive identity; never decoded")
        config = source / "configuration.toml"
        config.write_text('marker="original"\n')
        props = source / "original.csv"
        props.write_bytes(b"id,porosity,permeability\n0,.3,10\n")
        runs[run] = {"configuration": descriptor(config), "facies_properties": descriptor(props),
                     "configured_depth_map": descriptor(archive),
                     "fields": {name: descriptor(archive) for name in
                                ("shape_corrected_baseline", "labels", "facies", "depth")}}
        bundles[run] = [str(config)]
    config = {"frame": "rig-corrected", "frame_binding": {
        "display_reference_run": "reference", "source_reference_run": "source",
        "runs": runs}}
    current = package / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({"schema": 1, "format": GEOMETRY_REVISION_FORMAT,
        "revision_id": "a" * 32, "configuration": config}))
    root = package / "exports/native_setup" / ("a" * 32)
    root.mkdir(parents=True)

    def make_export(run):
        destination = root if run == "reference" else root / "runs" / run
        destination.mkdir(parents=True, exist_ok=True)
        replacements = _replacements(destination)
        for name in subject.EXPORT_FILES:
            data = toml.dumps(replacements).encode() if name == "native_setup.toml" else ("opaque " + name).encode()
            (destination / name).write_bytes(data)
        own = runs[run]
        binding = {"configuration": own["configuration"], "properties": own["facies_properties"],
                   "configured_depth_source": own["configured_depth_map"], "sources": own["fields"],
                   "replacements": replacements, "series_root": str(series),
                   "series_manifest": descriptor(manifest), "configuration_bundles": bundles}
        receipt = {"schema": 1, "format": subject.FORMAT, "revision_id": "a" * 32,
                   "run_id": run, "reference_run": "reference", "frame": "rig-corrected",
                   "current_sha256": descriptor(current)["sha256"], "binding": binding,
                   "files": {name: descriptor(destination / name) for name in subject.EXPORT_FILES},
                   "root_export": None if run == "reference" else descriptor(root / "export.json")}
        if run == "reference":
            receipt["runs"] = {"other": "runs/other/export.json"}
        (destination / "export.json").write_text(json.dumps(receipt))
        return destination

    make_export("reference")
    selected = {"run": "reference"}
    dialogs, messages = [], []
    workspace = SimpleNamespace(project=SimpleNamespace(root=package),
        reference=lambda: SimpleNamespace(run_id=selected["run"]), _show_export_dialog=dialogs.append)
    window = SimpleNamespace(geometry_workspace=workspace, print_log=messages.append)
    monkeypatch.setattr(subject, "native_series_link", lambda _project: {"series_root": str(series)})
    return dict(series=series, package=package, root=root, current=current, selected=selected,
                window=window, dialogs=dialogs, messages=messages, make_export=make_export)


def test_reference_Open_uses_ordinary_root_and_reads_native_setup_text(opened):
    assert subject.open_current_corrected_export(opened["window"]) is True
    export = opened["dialogs"][0]
    assert export.destination == opened["root"] and export.run_id == "reference"
    assert export.toml_text == (opened["root"] / "native_setup.toml").read_text()
    assert export.reused is True and not opened["messages"]


def test_selected_other_run_Open_uses_only_its_exact_child(opened):
    destination = opened["make_export"]("other")
    opened["selected"]["run"] = "other"
    assert subject.open_current_corrected_export(opened["window"]) is True
    assert opened["dialogs"][0].destination == destination
    assert opened["dialogs"][0].run_id == "other"


def test_missing_child_never_falls_back_to_reference_or_old_namespace(opened):
    opened["selected"]["run"] = "other"
    old = opened["package"] / "exports/corrected_setup" / ("a" * 32) / "other"
    old.mkdir(parents=True)
    (old / "export.json").write_text('{"fixture":"old namespace is not a fallback"}')
    before = {str(p): p.read_bytes() for p in opened["package"].rglob("*") if p.is_file()}
    assert subject.open_current_corrected_export(opened["window"]) is True
    assert not opened["dialogs"]
    assert "No saved corrected export is available for run other" in opened["messages"][0]
    assert before == {str(p): p.read_bytes() for p in opened["package"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["foreign_run", "changed_root", "missing_root", "foreign_sources", "foreign_bundle"])
def test_foreign_or_unbound_child_is_refused_without_dialog_or_write(opened, problem):
    destination = opened["make_export"]("other")
    opened["selected"]["run"] = "other"
    path = destination / "export.json"
    data = json.loads(path.read_bytes())
    if problem == "foreign_run":
        data["run_id"] = "reference"
    elif problem == "changed_root":
        root = opened["root"] / "export.json"
        root.write_bytes(root.read_bytes() + b"\n")
    elif problem == "missing_root":
        (opened["root"] / "export.json").unlink()
    elif problem == "foreign_sources":
        data["binding"]["sources"]["labels"]["path"] = "foreign.npz"
    else:
        data["binding"]["configuration_bundles"]["reference"] = data["binding"]["configuration_bundles"]["other"]
    path.write_text(json.dumps(data))
    before = {str(p): p.read_bytes() for p in opened["package"].rglob("*") if p.is_file()}
    assert subject.open_current_corrected_export(opened["window"]) is True
    assert not opened["dialogs"] and "Could not open this run's corrected export" in opened["messages"][0]
    assert before == {str(p): p.read_bytes() for p in opened["package"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["missing_file", "changed_toml", "foreign_file_path"])
def test_incomplete_or_changed_named_file_is_not_opened(opened, problem):
    path = opened["root"] / "export.json"
    if problem == "missing_file":
        (opened["root"] / "native_labels.npz").unlink()
    elif problem == "changed_toml":
        (opened["root"] / "native_setup.toml").write_bytes(b"changed setup text")
    else:
        data = json.loads(path.read_bytes())
        data["files"]["native_facies.npz"]["path"] = "foreign.npz"
        path.write_text(json.dumps(data))
    assert subject.open_current_corrected_export(opened["window"]) is True
    assert not opened["dialogs"] and opened["messages"]


def test_Open_reads_small_metadata_and_setup_text_never_array_payloads(opened, monkeypatch):
    original = subject._small
    seen = []
    def metadata_only(path):
        path = Path(path)
        seen.append(path)
        assert path.suffix in {".json", ".toml"}
        return original(path)
    monkeypatch.setattr(subject, "_small", metadata_only)
    assert subject.open_current_corrected_export(opened["window"]) is True
    assert opened["dialogs"] and seen


def test_raw_Open_route_stays_unhandled_and_creates_nothing(opened):
    data = json.loads(opened["current"].read_bytes())
    data["configuration"]["frame"] = "raw"
    opened["current"].write_text(json.dumps(data))
    before = {str(p): p.read_bytes() for p in opened["package"].rglob("*") if p.is_file()}
    assert subject.open_current_corrected_export(opened["window"]) is False
    assert not opened["dialogs"] and not opened["messages"]
    assert before == {str(p): p.read_bytes() for p in opened["package"].rglob("*") if p.is_file()}
