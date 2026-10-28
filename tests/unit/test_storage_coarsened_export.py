"""Standalone exports retain the verified source closure and missing data."""
from copy import deepcopy
import json
from pathlib import Path
import weakref

import numpy as np
import pytest

from darsia.fluidflower import storage_coarsened_export as export
from darsia.fluidflower import storage_coarsening as coarsening
from darsia.fluidflower import storage_extensive as extensive
from darsia.fluidflower import storage_extensive_io as extensive_io
from darsia.fluidflower import storage_outputs as output
from darsia.fluidflower.pipeline import products_publication as publication
from .test_storage_indexed_products import _published, _collect
from .test_storage_n1_saved_products import (tiny_config, phase1_data,
    _inputs as n1_inputs, _run_with_extensive, _index as n1_index)
from tests.fluidflower.test_storage_coarsening import make_payload


def _indexed(tmp_path, monkeypatch, *, pending=False, aligned=True):
    # The existing fixture leaves a third empty row/column by default. An
    # explicit 2 mm extent makes this real tiny 1 mm producer coarsenable.
    original = output.parse_request
    def request(config, precision):
        config = deepcopy(config)
        config["storage"]["extent_m"] = [0., .002, 0., .002]
        return original(config, precision)
    with monkeypatch.context() as context:
        if aligned:
            context.setattr(output, "parse_request", request)
        state, index, _ = _published(tmp_path, pending=pending, extensive_inputs=True)
    _root, document, _rows = publication._read_index(index)
    return state, index, document["publishing_machine_id"]


def _options(machine, target=2):
    return dict(producer_kind="native-phase-n1", qualification=extensive_io.QUALIFICATION,
        current_machine_id=machine, target_resolution_mm=target)


def _read(index, options):
    return extensive_io.read_indexed_extensive_run(index,
        **{key: value for key, value in options.items() if key != "target_resolution_mm"})


def _assert_false(document):
    for key in export._FALSE_FLAGS:
        assert document[key] is False
    assert document["native_to_1mm_reexport_gate"] == "NOT_RUN"


@pytest.mark.parametrize("target", [1, 2])
@pytest.mark.parametrize("pending", [False, True])
def test_real_verified_index_reopens_exact_arrays_and_unmodified_metadata(tmp_path, monkeypatch, target, pending):
    state, index, machine = _indexed(tmp_path / "source", monkeypatch, pending=pending)
    options = _options(machine, target)
    source = next(_read(index, options)["frames"])
    expected = coarsening.coarsen_verified_extensive(source, target_resolution_mm=target)
    original_index = Path(index["path"]).read_bytes()
    original_archive = Path(source["loaded_assets"][-1]["path"]).read_bytes()
    result = export.export_coarsened_extensive(index, tmp_path / "export", **options)
    assert result["complete"] is True
    assert result["ordered_case_ids"] == [source["original_frame_identity"]["case_id"]]
    _assert_false(result)
    frame = result["frames"][0]
    metadata = json.loads(Path(frame["metadata"]["path"]).read_bytes())
    assert metadata == expected["metadata"]
    assert metadata["source_binding"]["original_frame_identity"] == source["original_frame_identity"]
    assert metadata["source_binding"]["loaded_assets"] == source["loaded_assets"]
    assert all(row == {"status": extensive.PENDING} for row in metadata["channels"].values())
    with np.load(frame["arrays"]["path"], allow_pickle=False) as reopened:
        assert set(reopened.files) == set(expected["arrays"])
        assert not any("cell_" + key + "_g" in reopened for key in extensive.CHANNELS)
        for key, array in expected["arrays"].items():
            assert extensive._identity(reopened[key]) == extensive._identity(array)
    receipt = json.loads(Path(frame["receipt"]["path"]).read_bytes())
    _assert_false(receipt)
    assert receipt["arrays"] == output._asset(Path(frame["arrays"]["path"]))
    assert json.loads(Path(result["receipt"]["path"]).read_bytes()) == {
        key: value for key, value in result.items() if key != "receipt"}
    assert Path(index["path"]).read_bytes() == original_index
    assert Path(source["loaded_assets"][-1]["path"]).read_bytes() == original_archive
    assert not (state["directory"] / "products.npz").exists()


def test_real_mass_only_index_keeps_exact_multi_frame_order_and_provenance(
        tiny_config, phase1_data, tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    config = n1_inputs(source_root, tiny_config, phase1_data)
    config["storage"]["resolution_mm"] = 1
    completion = _run_with_extensive(source_root, config, monkeypatch)
    index = n1_index(source_root, completion)
    _root, document, _rows = publication._read_index(index)
    options = dict(_options(document["publishing_machine_id"]), producer_kind="n1-mass-only")
    result = export.export_coarsened_extensive(index, tmp_path / "export", **options)
    assert result["ordered_case_ids"] == completion["ordered_case_ids"]
    assert [row["case_id"] for row in result["frames"]] == completion["ordered_case_ids"]
    assert len(result["frames"]) > 1
    for number, frame in enumerate(result["frames"]):
        assert Path(frame["arrays"]["path"]).parent.name == "frame_%06d" % number
        metadata = json.loads(Path(frame["metadata"]["path"]).read_bytes())
        assert metadata["source_binding"]["original_frame_identity"]["case_id"] == frame["case_id"]
        assert metadata["source_binding"]["state"] == "synthetic_io_only"
        assert metadata["physical_binding_admitted"] is False
    _assert_false(result)


def _synthetic_stream(monkeypatch, index, machine, *, changed_second=False, check_release=False):
    _root, document, _rows = publication._read_index(index)
    calls, refs = [], []
    def read(*args, **kwargs):
        pass_number = len(calls)
        calls.append(kwargs)
        def frames():
            for number, case in enumerate(("sample_first", "sample_second")):
                if check_release:
                    assert all(ref() is None for ref in refs)
                payload = make_payload()
                payload.update(state="synthetic_io_only", loaded_assets=[],
                    original_frame_identity=dict(case_id=case, run=document["run"],
                        source={"version": 2 if changed_second and pass_number == 1 and number == 1 else 1}))
                refs[:] = [weakref.ref(value) for value in payload["arrays"].values()]
                yield payload
                del payload
        return dict(ordered_case_ids=["sample_first", "sample_second"], frames=frames())
    monkeypatch.setattr(extensive_io, "read_indexed_extensive_run", read)
    return calls


@pytest.mark.parametrize("target", [5, 10])
def test_bounded_stream_releases_arrays_and_supports_larger_explicit_targets(tmp_path, monkeypatch, target):
    _state, index, machine = _indexed(tmp_path / "source", monkeypatch)
    calls = _synthetic_stream(monkeypatch, index, machine, check_release=True)
    result = export.export_coarsened_extensive(index, tmp_path / "export", **_options(machine, target))
    assert len(calls) == 2
    assert [row["case_id"] for row in result["frames"]] == ["sample_first", "sample_second"]
    for frame in result["frames"]:
        with np.load(frame["arrays"]["path"], allow_pickle=False) as arrays:
            assert arrays["cell_mass_total_g"].shape == (20 // target, 20 // target)
            assert arrays["cell_mass_total_g"].sum() == 10.


def test_changed_late_frame_preserves_completed_frame_and_explicit_failure(tmp_path, monkeypatch):
    _state, index, machine = _indexed(tmp_path / "source", monkeypatch)
    _synthetic_stream(monkeypatch, index, machine, changed_second=True)
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match="changed between export passes"):
        export.export_coarsened_extensive(index, destination, **_options(machine))
    failure = json.loads((destination / "export_failure.json").read_bytes())
    assert failure["complete"] is False and failure["partial_files_preserved"] is True
    assert [row["case_id"] for row in failure["frames"]] == ["sample_first"]
    assert (destination / "frames/frame_000000/arrays.npz").is_file()
    assert not (destination / "export.json").exists()
    _assert_false(failure)


def test_partial_array_write_is_kept_with_incomplete_failure_receipt(tmp_path, monkeypatch):
    _state, index, machine = _indexed(tmp_path / "source", monkeypatch)
    def fail(path, arrays):
        with path.open("xb") as stream:
            stream.write(b"partial")
        raise OSError("synthetic storage failure")
    monkeypatch.setattr(output, "_write_npz", fail)
    destination = tmp_path / "export"
    with pytest.raises(OSError, match="synthetic storage failure"):
        export.export_coarsened_extensive(index, destination, **_options(machine))
    assert (destination / "frames/frame_000000/arrays.npz").read_bytes() == b"partial"
    failure = json.loads((destination / "export_failure.json").read_bytes())
    assert failure["active_frame"]["number"] == 0 and failure["frames"] == []
    assert failure["state"] == "incomplete_failed"


@pytest.mark.parametrize("bad", ["index", "archive", "qualification", "producer", "alignment"])
def test_invalid_source_is_refused_before_any_output_write(tmp_path, monkeypatch, bad):
    state, index, machine = _indexed(tmp_path / "source", monkeypatch, aligned=bad != "alignment")
    options = _options(machine)
    if bad == "index":
        path = Path(index["path"])
        path.write_bytes(path.read_bytes() + b" ")
    elif bad == "archive":
        path = Path(state["document"]["extensive"]["archive"]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
    elif bad == "qualification":
        options["qualification"] = extensive_io.GENUINE_QUALIFICATION
    elif bad == "producer":
        options["producer_kind"] = "unspecified"
    destination = tmp_path / "export"
    with pytest.raises(ValueError):
        export.export_coarsened_extensive(index, destination, **options)
    assert not destination.exists()


@pytest.mark.parametrize("where", ["index", "producer", "alias", "collection"])
def test_protected_directories_are_rejected_before_creating_any_output(tmp_path, monkeypatch, where):
    state, index, machine = _indexed(tmp_path / "source", monkeypatch)
    options = _options(machine)
    if where == "index":
        destination = Path(index["path"]).parent / "new_export"
    elif where == "producer":
        destination = state["root"] / "new_export"
    elif where == "alias":
        alias = tmp_path / "late_alias" / "products.npz"
        options["source_paths"] = {"native/products.npz": str(alias)}
        destination = alias.parent / "new_export"
    else:
        collection = _collect(tmp_path / "source", index)
        options["collections"] = [collection]
        destination = Path(collection["path"]).parent / "new_export"
    with pytest.raises(ValueError, match="inside an index, collection or producer"):
        export.export_coarsened_extensive(index, destination, **options)
    assert not destination.exists()


def test_existing_destination_is_never_overwritten(tmp_path, monkeypatch):
    _state, index, machine = _indexed(tmp_path / "source", monkeypatch)
    destination = tmp_path / "export"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        export.export_coarsened_extensive(index, destination, **_options(machine))
    assert list(destination.iterdir()) == [sentinel] and sentinel.read_bytes() == b"keep"


def test_accessible_canonical_drive_alias_cannot_enter_producer_folder(tmp_path, monkeypatch):
    state, index, machine = _indexed(tmp_path / "source", monkeypatch)
    destination = tmp_path / "drive_alias" / "export"
    canonical = export._canonical
    def mapped(path):
        if Path(path) == destination:
            return state["root"] / "export"
        return canonical(path)
    monkeypatch.setattr(export, "_canonical", mapped)
    with pytest.raises(ValueError, match="inside an index, collection or producer"):
        export.export_coarsened_extensive(index, destination, **_options(machine))
    assert not destination.parent.exists()
    assert not (state["root"] / "export").exists()


def test_foreign_original_paths_are_never_probed_after_explicit_collection(tmp_path, monkeypatch):
    state, index, _machine = _indexed(tmp_path / "source", monkeypatch)
    collection = _collect(tmp_path / "source", index)
    original_stat, original_lstat = Path.stat, Path.lstat
    def guard(function):
        def checked(path, *args, **kwargs):
            assert not str(path).startswith(str(state["root"])), "Foreign original path was probed"
            return function(path, *args, **kwargs)
        return checked
    monkeypatch.setattr(Path, "stat", guard(original_stat))
    monkeypatch.setattr(Path, "lstat", guard(original_lstat))
    result = export.export_coarsened_extensive(index, tmp_path / "export",
        **dict(_options("read_host"), collections=[collection]))
    assert result["complete"] is True


def test_cli_dispatch_requires_explicit_source_and_grid_bindings(tmp_path, monkeypatch, capsys):
    calls = []
    def run(index, destination, **kwargs):
        calls.append((index, destination, kwargs))
        return {"complete": True}
    monkeypatch.setattr(export, "export_coarsened_extensive", run)
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps({"collections": [], "source_paths": {}}), encoding="utf8")
    args = ["--index", str(tmp_path / "index.json"), "--index-sha256", "a" * 64,
        "--index-size-bytes", "123", "--machine", "read_host", "--output", str(tmp_path / "export"),
        "--producer-kind", "n1-mass-only", "--qualification", extensive_io.GENUINE_QUALIFICATION,
        "--target-resolution-mm", "10", "--bindings", str(bindings)]
    assert export.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"complete": True}
    assert calls[0][0] == dict(path=str(tmp_path / "index.json"), sha256="a" * 64, size_bytes=123)
    assert calls[0][2] == dict(producer_kind="n1-mass-only", qualification=extensive_io.GENUINE_QUALIFICATION,
        current_machine_id="read_host", target_resolution_mm=10, collections=[], source_paths={})
    with pytest.raises(SystemExit):
        export.main(args[:args.index("--qualification")])
