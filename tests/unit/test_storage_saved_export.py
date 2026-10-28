"""Tiny saved-only export wiring; no camera input, model, queue, or movie encoder."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import importlib.machinery
import json
from pathlib import Path
import types
import sys
import weakref

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_saved_export as export
from darsia.fluidflower import storage_original_photo as photo
from darsia.fluidflower import storage_outputs as output
from darsia.fluidflower import storage_saved_products as saved
# Frozen original reader is a validation-only fixture, never a DarSIA product module.
_ORACLE_NAME = "ffac_test_fixture_storage_default_reader"
_ORACLE_PATH = Path(__file__).parents[1] / "fixtures" / "storage_saved_products_default_oracle.py.fixture"
_ORACLE_RAW = _ORACLE_PATH.read_bytes()
if (hashlib.sha256(_ORACLE_RAW).hexdigest() != "4a0eef912849ef32bca560326817483db37e53fb88e817094664f305eb2682d0"
        or len(_ORACLE_RAW) != 21213):
    raise RuntimeError("Frozen original default-reader test fixture changed")
_ORACLE_SPEC = importlib.util.spec_from_file_location(_ORACLE_NAME, _ORACLE_PATH, loader=importlib.machinery.SourceFileLoader(_ORACLE_NAME, str(_ORACLE_PATH)))
original_reader = importlib.util.module_from_spec(_ORACLE_SPEC)
original_reader.__package__ = "darsia.fluidflower"
sys.modules[_ORACLE_NAME] = original_reader
_ORACLE_SPEC.loader.exec_module(original_reader)
from darsia.fluidflower.pipeline import products_publication as publication
from .test_storage_indexed_products import _published, _write
from .test_storage_saved_products import _fixture, _reseal


def _photo(tmp_path, state):
    tmp_path.mkdir()
    rgb = np.array([[[0, 0, 0], [220, 30, 80]], [[20, 180, 50], [40, 60, 240]]], np.uint8)
    jpeg = tmp_path / "already_registered_pre_light_review.jpg"
    Image.fromarray(rgb).save(jpeg, quality=100, subsampling=0)
    review = output._asset(jpeg)
    camera = dict(path=str(tmp_path / "camera_NEVER_READ.jpg"), sha256="d" * 64)
    certificate = dict(run="ac52", frame_index=28,
        camera_jpeg_path=camera["path"], camera_jpeg_sha256=camera["sha256"],
        native_review_jpeg_path=review["path"], native_review_jpeg_sha256=review["sha256"],
        native_shape_hw=[2, 2], content_box_xywh=[0, 0, 2, 2],
        H_ac28_to_frame_native=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        binding=dict(geometry_module_sha256="f" * 64, input_sources=dict(camera_jpeg=camera),
            content_box_xywh=[0, 0, 2, 2], native_shape_hw=[2, 2],
            original_shape_configs=[], original_shape_corrections=[]))
    certificate_asset = _write(tmp_path / "certificate.json", certificate)
    native = dict(certificate, geometry_certificate_path=certificate_asset["path"],
                  geometry_certificate_sha256=certificate_asset["sha256"])
    native_asset = _write(tmp_path / "native.json", native)
    selected_index = state.get("selected_frame_index", 1)
    current = dict(run="ac52", frame_index=selected_index, source_sha256="d" * 64,
        source_photo=camera, source_geometry_receipt={k: certificate_asset[k] for k in ("path", "sha256")},
        source_content_box_xywh=[0, 0, 2, 2], source_native_shape_hw=[2, 2],
        native_origin_xy=[0, 0], phase_shape_hw=[2, 2], photograph_canvas_wh=[2, 2],
        H_ac28_to_frame_native=deepcopy(certificate["H_ac28_to_frame_native"]))
    current_asset = _write(tmp_path / "current.json", current)
    return photo.publish_registered_original_png(tmp_path / "original.png",
        task=dict(run="ac52", precision="float32"),
        frame_spec=dict(case_id=state["record"]["frame_id"], frame_index=selected_index,
                        source=deepcopy(state["document"]["source"])),
        native_geometry=native_asset, geometry_certificate=certificate_asset,
        current_geometry=current_asset, saved_native_review=review,
        geometry_producer_sha256="f" * 64, resolve_asset=lambda bound: Path(bound["path"]))


def _republish(tmp_path, index, *, photograph=None, completion_change=None):
    _root, doc, rows = publication._read_index(index)
    original = rows.pop("completed.json")
    completed = dict(path=original["local_path"], sha256=original["sha256"], size_bytes=original["bytes"])
    if completion_change:
        value = json.loads(Path(completed["path"]).read_bytes())
        value.update(completion_change)
        completed = _write(tmp_path / "new_completion.json", value)
    products = list(rows.values())
    metadata = [key for key, row in rows.items() if "metadata_copy" in row]
    if photograph:
        for key, bound, kind in [("photos/original.png", photograph["photograph"], "map"),
                                 ("photos/receipt.json", photograph["receipt"], "operational_receipt")]:
            products.append(publication.product(key, machine_id="Main", local_path=bound["path"],
                sha256=bound["sha256"], bytes=bound["size_bytes"], kind=kind))
        metadata.append("photos/receipt.json")
    return publication.publish_run_metadata(tmp_path / "new_shared", job=doc["job"], run="ac52",
        attempt_id="a" * 32, machine_id="Main", completed=completed, products=products,
        metadata_product_ids=metadata, file_writers_closed=True, all_final_products_listed=True)


def _with_photo(tmp_path):
    state, old_index, _ = _published(tmp_path)
    photograph = _photo(tmp_path / "photo", state)
    index = _republish(tmp_path, old_index, photograph=photograph)
    return state, index, photograph


def _film(index, destination, photograph=None, **kwargs):
    return export.export_indexed_storage(index, destination, current_machine_id="Main",
        film_field="saturation", fps=12., geometry_producer_sha256="f" * 64,
        photographs={} if photograph is None else {"ac52_f1": photograph["receipt"]}, **kwargs)


def test_default_no_export_is_an_IO_free_noop(tmp_path):
    result = export.export_indexed_storage(object(), object(), current_machine_id=object())
    assert result == dict(state="not_requested", outputs=[], default_native_unchanged=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fps", [None, 0, -1, True, float("nan")])
def test_film_needs_explicit_positive_finite_fps_before_any_input_read(tmp_path, fps):
    with pytest.raises(ValueError, match="explicit finite positive fps"):
        export.export_indexed_storage(object(), tmp_path / "out", current_machine_id="Main",
                                      film_field="saturation", fps=fps)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field,limit", [("concentration", None), ("concentration", 0), ("saturation", 1.)])
def test_fixed_display_scale_is_explicit_and_cannot_rescale_other_fields(tmp_path, field, limit):
    with pytest.raises(ValueError):
        export.export_indexed_storage(object(), tmp_path / "out", current_machine_id="Main",
                                      figure_field=field, concentration_limit_g_l=limit)
    assert not (tmp_path / "out").exists()


def test_csv_uses_real_saved_cell_sums_per_acquisition_without_native_archive(tmp_path):
    state, index, _ = _published(tmp_path)
    before = Path(index["path"]).read_bytes()
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main", table=True)
    with Path(result["table"]["table"]["path"]).open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["case_id"] == "ac52_f1"
    assert float(rows[0]["cell_mass_total_sum_g"]) == 9.
    assert float(rows[0]["cell_mass_aqueous_sum_g"]) == 3.
    assert float(rows[0]["cell_mass_gas_sum_g"]) == 3.
    assert Path(index["path"]).read_bytes() == before
    assert not (state["directory"] / "products.npz").exists()


def test_missing_photo_blocks_film_without_blocking_explicit_csv_or_using_fallback(tmp_path):
    _state, index, _ = _published(tmp_path)
    result = _film(index, tmp_path / "out", table=True)
    sequence = result["film_frames"]["sequence"]
    assert result["table"]["row_count"] == 1
    assert sequence["state"] == "not_film_ready" and not sequence["frames"]
    assert sequence["blocked"]["ac52_f1"]["reason"] == "indexed_registered_pre_light_photo_not_bound"
    assert not sequence["movie_encoded"] and not (tmp_path / "out/frames").exists()
    assert result["complete"] is False and result["state"] == "not_film_ready"


def test_existing_coarse_preview_writer_stays_truthful_about_missing_photo(tmp_path):
    _state, index, _ = _published(tmp_path)
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main",
        figure_field="concentration", concentration_limit_g_l=2.)
    assert len(result["figures"]) == 1 and result["film_frames"] is None
    receipt = json.loads(Path(result["figures"][0]["receipt"]["path"]).read_bytes())
    assert receipt["fixed_limits"] == [0., 2.]
    assert receipt["photograph"]["state"] == "missing" and not receipt["final_original_left_film_ready"]


def test_pending_before_field_cannot_become_film_ready_or_follow_a_photo(tmp_path):
    _state, index, _ = _published(tmp_path, pending=True)
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main",
        film_field="before_rules_coarse", fps=12., geometry_producer_sha256="f" * 64,
        photographs={"ac52_f1": {"not_a_descriptor": "must never be opened"}})
    sequence = result["film_frames"]["sequence"]
    assert not sequence["frames"] and sequence["blocked"]["ac52_f1"]["reason"] == "saved_before_rules_pending"


def test_real_photo_publisher_index_and_field_figure_compose_literal_original_left(tmp_path, monkeypatch):
    _state, index, photograph = _with_photo(tmp_path)
    with Image.open(photograph["photograph"]["path"]) as image:
        expected_rgba = np.asarray(image).copy()
    seen = {}
    original = export.rendering._figure
    def observe(*args):
        figure, mesh = original(*args)
        save = figure.savefig
        def inspect(*a, **kw):
            seen["rgba"] = np.asarray(figure.axes[0].images[0].get_array()).copy()
            seen["background"] = figure.axes[0].get_facecolor()
            seen["coordinates"] = mesh.get_coordinates().copy()
            seen["limits"] = mesh.get_clim()
            return save(*a, **kw)
        figure.savefig = inspect
        return figure, mesh
    monkeypatch.setattr(export.rendering, "_figure", observe)
    result = _film(index, tmp_path / "out", photograph)
    sequence = result["film_frames"]["sequence"]
    np.testing.assert_array_equal(seen["rgba"], expected_rgba)
    assert seen["background"] == (1., 1., 1., 1.) and seen["limits"] == (0., 1.)
    np.testing.assert_array_equal(seen["coordinates"][0, :, 0], [0., .001, .002, .003])
    assert sequence["original_left_frame_sequence_ready"] and sequence["fps"] == 12.
    assert len(sequence["frames"]) == 1 and not sequence["movie_encoded"]
    assert sequence["state"] == "frames_ready_encoding_pending"
    assert not (tmp_path / "photo/camera_NEVER_READ.jpg").exists()


@pytest.mark.parametrize("mutation", ["run", "case_id", "source", "precision", "prelight", "canvas", "producer"])
def test_forged_photo_metadata_is_rejected_before_any_export(tmp_path, mutation):
    state, old_index, _ = _published(tmp_path)
    photograph = _photo(tmp_path / "photo", state)
    doc = json.loads(Path(photograph["receipt"]["path"]).read_bytes())
    if mutation == "source":
        doc["source"]["source_sha256"] = "a" * 64
    elif mutation == "prelight":
        doc["colour_light_or_v5_changes"] = True
    elif mutation == "canvas":
        doc["target_canvas_hw"] = [3, 2]
    elif mutation == "producer":
        doc["geometry_producer_sha256"] = "a" * 64
    else:
        doc[mutation] = "different"
    photograph["receipt"] = _write(tmp_path / "different_photo_receipt.json", doc)
    index = _republish(tmp_path, old_index, photograph=photograph)
    with pytest.raises(ValueError, match="Saved original photograph"):
        _film(index, tmp_path / "out", photograph)
    assert (tmp_path / "out").is_dir()
    assert not (tmp_path / "out/frames").exists()
    assert not (tmp_path / "out/export.json").exists()


def test_changed_saved_photo_bytes_cannot_be_replaced_by_a_corrected_picture(tmp_path):
    _state, index, photograph = _with_photo(tmp_path)
    png = Path(photograph["photograph"]["path"])
    png.write_bytes(png.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        _film(index, tmp_path / "out", photograph)
    assert (tmp_path / "out").is_dir()
    assert not (tmp_path / "out/frames").exists()
    assert not (tmp_path / "out/export.json").exists()


def test_other_machine_reports_explicit_photo_collect_required_without_implicit_collect(tmp_path, monkeypatch):
    _state, index, photograph = _with_photo(tmp_path)
    _root, _doc, rows = publication._read_index(index)
    storage_maps = [key for key, row in rows.items() if key.startswith("storage/") and row["kind"] == "map"]
    collection = publication.collect_products(index, storage_maps, tmp_path / "collected",
        current_machine_id="Main", file_writers_closed=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Exporter must not collect implicitly")
    monkeypatch.setattr(publication, "collect_products", forbidden)
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Second",
        collections=[collection], film_field="saturation", fps=12.,
        geometry_producer_sha256="f" * 64, photographs={"ac52_f1": photograph["receipt"]})
    assert result["film_frames"]["sequence"]["blocked"]["ac52_f1"]["reason"] == "explicit_photo_collect_required"


def test_phase2_N1_completion_is_not_silently_treated_as_native_phase(tmp_path):
    _state, index, _ = _published(tmp_path)
    index = _republish(tmp_path, index, completion_change=dict(pipeline="phase2-n1"))
    with pytest.raises(ValueError, match="complete saved native-phase"):
        export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main", table=True)
    assert not (tmp_path / "out").exists()


def test_unknown_photo_case_does_not_expand_the_run_or_search_for_inputs(tmp_path):
    _state, index, _ = _published(tmp_path)
    with pytest.raises(ValueError, match="exact case IDs"):
        export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main",
            film_field="saturation", fps=12., geometry_producer_sha256="f" * 64,
            photographs={"unrequested": {}})
    assert not (tmp_path / "out").exists()


def test_existing_output_is_preserved_before_index_loading(tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    marker = destination / "keep"
    marker.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        export.export_indexed_storage(object(), destination, current_machine_id="Main", table=True)
    assert marker.read_bytes() == b"unchanged"


def test_cli_preserves_explicit_index_collection_options_scales_and_fps(tmp_path, monkeypatch, capsys):
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps(dict(collections=[], source_paths={}, photographs={})), encoding="utf8")
    called = {}
    def observe(index, destination, **kwargs):
        called.update(index=index, destination=destination, **kwargs)
        return dict(state="synthetic CLI dispatch only")
    monkeypatch.setattr(export, "export_indexed_storage", observe)
    assert export.main(["--index", "index.json", "--index-sha256", "a" * 64,
        "--index-size-bytes", "123", "--machine", "Second", "--output", "fresh-output",
        "--bindings", str(bindings), "--table", "--film-field", "concentration",
        "--concentration-limit-g-l", "2.5", "--fps", "12",
        "--geometry-producer-sha256", "f" * 64]) == 0
    assert called["index"] == dict(path="index.json", sha256="a" * 64, size_bytes=123)
    assert called["current_machine_id"] == "Second" and called["fps"] == 12.
    assert called["concentration_limit_g_l"] == 2.5 and called["table"] is True
    assert "synthetic CLI dispatch only" in capsys.readouterr().out


def test_pending_before_does_not_block_complete_saturation_film_CSV_or_mass_figure(tmp_path):
    state, original_index, _ = _published(tmp_path, pending=True)
    photograph = _photo(tmp_path / "photo", state)
    index = _republish(tmp_path, original_index, photograph=photograph)
    result = _film(index, tmp_path / "out", photograph, table=True,
                   figure_field="concentration", concentration_limit_g_l=2.)
    sequence = result["film_frames"]["sequence"]
    assert result["complete"] is True and result["table"]["row_count"] == 1
    assert len(result["figures"]) == 1 and len(sequence["frames"]) == 1
    assert sequence["original_left_frame_sequence_ready"] and not sequence["blocked"]
    assert not sequence["movie_encoded"]
    assert not any(Path(row["path"]).name == "storage_gas_probability_before_rules.png"
                   for row in state["entry"]["files"])


def _two_frames(tmp_path, *, with_states=False):
    first = _fixture(tmp_path / "first")
    second = _fixture(tmp_path / "second", case="ac52_f2")
    manifest = deepcopy(first["manifest"])
    manifest["frames"].update(second["manifest"]["frames"])
    bound = _write(tmp_path / "both_manifest.json", manifest)
    for state in (first, second):
        state["record"]["manifest_sha256"] = bound["sha256"]
        _reseal(state)
    completion = dict(complete=True, worker_references_closed=True,
        pipeline="native-phase-n1", run="ac52", precision="float32",
        frames=2, case_count=2, records=[first["record"], second["record"]],
        ordered_case_ids=["ac52_f1", "ac52_f2"], manifest=bound,
        configuration=manifest["configuration"], calibration=dict(sha256="c" * 64))
    return (completion, [first, second]) if with_states else completion


def _two_frame_index(tmp_path, *, with_photos=False):
    completion, states = _two_frames(tmp_path, with_states=True)
    complete_asset = _write(tmp_path / "completed.json", completion)
    job = _write(tmp_path / "job.json", dict(format="synthetic-two-frame-metadata-job"))
    rows, metadata = [], []
    def add(key, bound, kind):
        rows.append(publication.product(key, machine_id="Main", local_path=bound["path"],
            sha256=bound["sha256"], bytes=bound["size_bytes"], kind=kind))
        if kind != "map":
            metadata.append(key)
    add("manifest.json", completion["manifest"], "manifest")
    photographs = {}
    for number, state in enumerate(states):
        case = state["record"]["frame_id"]
        for name, kind in [("commit.json", "commit_receipt"), ("result.json", "frame_scalar")]:
            add(case + "/" + name, output._asset(state["directory"] / name), kind)
        add(case + "/storage_receipt.json", state["entry"]["receipt"], "operational_receipt")
        for bound in state["entry"]["files"]:
            add(case + "/" + Path(bound["path"]).name, bound, "map")
        add(case + "/products.npz", dict(path=str(state["directory"] / "products.npz"),
            sha256="e" * 64, size_bytes=999), "map")
        if with_photos:
            state["selected_frame_index"] = number + 1
            photograph = _photo(tmp_path / ("photo" + str(number)), state)
            add(case + "/original.png", photograph["photograph"], "map")
            add(case + "/photo_receipt.json", photograph["receipt"], "operational_receipt")
            photographs[case] = photograph["receipt"]
    index = publication.publish_run_metadata(tmp_path / "shared", job=job, run="ac52",
        attempt_id="a" * 32, machine_id="Main", completed=complete_asset, products=rows,
        metadata_product_ids=metadata, file_writers_closed=True, all_final_products_listed=True)
    return (index, photographs) if with_photos else index


def _assert_exact_nested(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype
        assert left.flags.writeable == right.flags.writeable
        assert np.array_equal(left, right, equal_nan=True)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and list(left) == list(right)
        for key in left:
            _assert_exact_nested(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_exact_nested(a, b)
    else:
        assert type(left) is type(right) and left == right


@pytest.mark.parametrize("load_before", [False, True])
def test_default_run_reader_exactly_matches_the_unmodified_original_oracle(tmp_path, load_before):
    completion = _two_frames(tmp_path)
    resolve = lambda bound: Path(bound["path"])
    original = original_reader.read_stored_run(completion, resolve_asset=resolve, load_before=load_before)
    actual = saved.read_stored_run(completion, resolve_asset=resolve, load_before=load_before)
    assert isinstance(actual["frames"], list)
    _assert_exact_nested(original, actual)


def test_stream_header_is_eagerly_validated_but_arrays_are_lazy_and_released(tmp_path, monkeypatch):
    completion = _two_frames(tmp_path)
    references = []
    original = saved.read_stored_frame
    def observe(*args, **kwargs):
        assert all(reference() is None for reference in references), "Previous frame arrays remained live"
        frame = original(*args, **kwargs)
        references.append(weakref.ref(frame["arrays"]["concentration_aqueous"]))
        return frame
    monkeypatch.setattr(saved, "read_stored_frame", observe)
    run = saved.read_stored_run(completion, resolve_asset=lambda bound: Path(bound["path"]), stream=True)
    assert isinstance(run["frames"], types.GeneratorType) and not references
    assert run["ordered_case_ids"] == ["ac52_f1", "ac52_f2"]
    first = next(run["frames"])
    assert first["case_id"] == "ac52_f1"
    view = export._scalar_view(first)
    assert export.tables._row(first, 0) == export.tables._row(view, 0)
    del first
    assert references[0]() is None
    second = next(run["frames"])
    assert second["case_id"] == "ac52_f2"
    del second
    assert references[1]() is None
    with pytest.raises(StopIteration):
        next(run["frames"])


def test_stream_snapshots_input_metadata_and_keeps_per_frame_manifest_guard(tmp_path):
    completion = _two_frames(tmp_path)
    run = saved.read_stored_run(completion, resolve_asset=lambda bound: Path(bound["path"]), stream=True)
    completion["records"][0]["manifest_sha256"] = "0" * 64
    assert next(run["frames"])["case_id"] == "ac52_f1"
    malformed = deepcopy(run["original_completion"])
    malformed["records"][0]["manifest_sha256"] = "0" * 64
    guarded = saved.read_stored_run(malformed, resolve_asset=lambda bound: Path(bound["path"]), stream=True)
    with pytest.raises(ValueError, match="another run manifest"):
        next(guarded["frames"])


def test_export_releases_previous_arrays_and_keeps_only_scalar_views_for_CSV(tmp_path, monkeypatch):
    index = _two_frame_index(tmp_path)
    references = []
    original = saved.read_stored_frame
    def observe(*args, **kwargs):
        assert all(reference() is None for reference in references), "Exporter retained a previous coarse frame"
        frame = original(*args, **kwargs)
        references.append(weakref.ref(frame["arrays"]["cell_mass_total_g"]))
        return frame
    monkeypatch.setattr(saved, "read_stored_frame", observe)
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main", table=True)
    assert len(references) == 2 and all(reference() is None for reference in references)
    assert result["table"]["row_count"] == 2
    with Path(result["table"]["table"]["path"]).open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["case_id"] for row in rows] == ["ac52_f1", "ac52_f2"]
    assert [float(row["cell_mass_total_sum_g"]) for row in rows] == [9., 9.]


def test_original_left_sequence_releases_each_photo_before_next_acquisition(tmp_path, monkeypatch):
    index, photographs = _two_frame_index(tmp_path, with_photos=True)
    photo_references = []
    original_photo = export._photograph
    original_read = saved.read_stored_frame
    def read(*args, **kwargs):
        assert all(ref() is None for ref in photo_references), "Previous original RGBA stayed live"
        return original_read(*args, **kwargs)
    def photograph(*args, **kwargs):
        assert all(ref() is None for ref in photo_references)
        rgba, provenance = original_photo(*args, **kwargs)
        photo_references.append(weakref.ref(rgba))
        return rgba, provenance
    monkeypatch.setattr(saved, "read_stored_frame", read)
    monkeypatch.setattr(export, "_photograph", photograph)
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main",
        film_field="saturation", fps=12., geometry_producer_sha256="f" * 64,
        photographs=photographs)
    assert len(photo_references) == 2 and all(ref() is None for ref in photo_references)
    assert len(result["film_frames"]["sequence"]["frames"]) == 2


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_streaming_control_is_explicit_and_does_not_accept_implicit_modes(flag):
    with pytest.raises(ValueError, match="explicit boolean"):
        saved.read_stored_run(object(), resolve_asset=object(), stream=flag)


@pytest.mark.parametrize("raw", ['{"collections": [], "collections": []}',
                                  '{"source_paths": {"x": NaN}}',
                                  '{"source_paths": {"x": Infinity}}'])
def test_CLI_uses_original_strict_JSON_metadata_parser(tmp_path, raw):
    bindings = tmp_path / "bindings.json"
    bindings.write_text(raw, encoding="utf8")
    with pytest.raises(ValueError, match="Duplicate|Non-finite"):
        export.main(["--index", "must_not_be_read.json", "--index-sha256", "a" * 64,
            "--index-size-bytes", "1", "--machine", "Main", "--output", str(tmp_path / "out"),
            "--bindings", str(bindings), "--table"])
    assert not (tmp_path / "out").exists()


def test_final_export_receipt_is_fsynced_before_it_is_reported(tmp_path, monkeypatch):
    _state, index, _ = _published(tmp_path)
    calls = []
    original = export.os.fsync
    def observe(fd):
        calls.append(fd)
        return original(fd)
    monkeypatch.setattr(export.os, "fsync", observe)
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main", table=True)
    assert calls and Path(result["receipt"]["path"]).is_file()


def test_actual_Agg_canvas_and_renderer_are_released_across_two_frame_exports(tmp_path, monkeypatch):
    index, photographs = _two_frame_index(tmp_path, with_photos=True)
    canvases, renderers = [], []
    original = export.rendering._figure
    def observe(*args, **kwargs):
        assert all(ref() is None for ref in canvases), "An earlier Agg canvas stayed live"
        assert all(ref() is None for ref in renderers), "An earlier Agg renderer/buffer stayed live"
        figure, mesh = original(*args, **kwargs)
        save = figure.savefig
        def record(*a, **kw):
            result = save(*a, **kw)
            canvas = figure.canvas
            renderer = canvas.renderer
            canvases.append(weakref.ref(canvas))
            renderers.append(weakref.ref(renderer))
            return result
        figure.savefig = record
        return figure, mesh
    monkeypatch.setattr(export.rendering, "_figure", observe)
    result = export.export_indexed_storage(index, tmp_path / "out", current_machine_id="Main",
        table=True, figure_field="saturation", film_field="saturation", fps=12.,
        geometry_producer_sha256="f" * 64, photographs=photographs)
    assert len(result["figures"]) == 2 and len(result["film_frames"]["sequence"]["frames"]) == 2
    assert len(canvases) == len(renderers) == 4
    assert all(ref() is None for ref in canvases + renderers)


def test_Agg_disposal_detaches_buffer_and_both_backreferences_even_with_live_figure():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(1, 1))
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    canvas_ref, renderer_ref = weakref.ref(canvas), weakref.ref(canvas.renderer)
    export.rendering.release_stored_figure(figure)
    assert figure.canvas is None and canvas.figure is None and canvas.renderer is None
    assert renderer_ref() is None
    del canvas
    assert canvas_ref() is None
