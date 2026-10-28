"""Original native-phase storage writer/publication/read joins; authored UNRUN.

The tiny fields and completion/Jobs envelope are synthetic. Stored maps and
photographs use the existing original writers and indexed publication, and all
positive run/frame/photo reads use the unchanged original consumers. These
tests do not qualify a native/phase solver, worker, queue or real campaign.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower.calibration_review_native import NativeSavedReview
from darsia.fluidflower.calibration_review_native_bindings import photo_geometry
from darsia.fluidflower import storage_n1_saved_products as n1
from darsia.fluidflower import storage_saved_products as saved
from darsia.fluidflower import storage_saved_export as export
from darsia.fluidflower.pipeline import products_publication as publication
from darsia.fluidflower.pipeline.checkpoint import run_identity
from .test_storage_indexed_products import _published, _write
from .test_storage_saved_products import _json, _reseal
from .test_storage_saved_export import _photo
from .test_calibration_review_native import native_attachment, phase1_data, tiny_config


def _forbidden(*args, **kwargs):
    raise AssertionError("Wrong original protocol reader or forbidden producer read")


def _native_manifest(state):
    """Use the original runner's literal manifest identity and frame schema."""
    manifest = state["manifest"]
    manifest["task_identity"] = run_identity(dict(run=manifest["run"],
        configuration=manifest["configuration"], commit=manifest["code"]["commit"],
        pipeline="native-phase-n1"), "cpu", "float32", "c" * 64)
    manifest["resume_checkpoint"] = _write(state["root"] / "resume.json", dict(
        complete=True, source=None, prefix_case_ids=[], frame_producers=[], predecessor_input=None))
    spec = manifest["frames"][state["record"]["frame_id"]]
    spec.update(frame_index=1, elapsed_h=.25)
    state["result"].update(frame_index=1, elapsed_h=.25, injected_mass_g=10.)
    _json(state["manifest_path"], manifest)
    state["record"]["manifest_sha256"] = saved.output._asset(state["manifest_path"])["sha256"]
    _reseal(state)


def _republish(directory, original_index, completion, *, photograph=None):
    directory.mkdir()
    _root, document, rows = publication._read_index(original_index)
    rows.pop("completed.json")
    products = [publication.product(key, **{field: row[field] for field in
        ("machine_id", "local_path", "sha256", "bytes", "kind")}) for key, row in rows.items()]
    metadata = [key for key, row in rows.items() if "metadata_copy" in row]
    if photograph is not None:
        for key, bound, kind in (("photos/original.png", photograph["photograph"], "map"),
                                ("photos/receipt.json", photograph["receipt"], "operational_receipt")):
            products.append(publication.product(key, machine_id="Main", local_path=bound["path"],
                sha256=bound["sha256"], bytes=bound["size_bytes"], kind=kind))
            if kind == "operational_receipt":
                metadata.append(key)
    return publication.publish_run_metadata(directory / "shared", job=document["job"],
        run=completion["run"], attempt_id=document["attempt_id"], machine_id="Main",
        completed=_write(directory / "completed.json", completion), products=products,
        metadata_product_ids=metadata, file_writers_closed=True, all_final_products_listed=True)


@pytest.fixture
def native_phase_attachment(tmp_path):
    state, original_index, completion = _published(tmp_path / "native-phase", alter_state=_native_manifest)
    # The actual runner emits this original-manifest association. Operational
    # producer/owner qualification remains the separate Jobs snapshot boundary.
    completion["frame_producers"] = [dict(case_id=completion["ordered_case_ids"][0],
        manifest=deepcopy(completion["manifest"]))]
    completion.update(reused_case_ids=[], newly_computed_frames=completion["frames"])
    photograph = _photo(tmp_path / "photo", state)
    index = _republish(tmp_path / "native-phase-publication", original_index, completion,
                       photograph=photograph)
    return dict(state=state, completion=completion, index=index,
                photographs={completion["ordered_case_ids"][0]: photograph["receipt"]})


def _review(value, **options):
    return NativeSavedReview(value["index"],
        current_machine_id=options.pop("current_machine_id", "Main"),
        expected_run=value["completion"]["run"],
        expected_calibration_sha256=value["completion"]["calibration"]["sha256"], **options)


def test_native_phase_admission_calls_original_run_reader_metadata_only(native_phase_attachment, monkeypatch):
    value = native_phase_attachment
    original, calls = saved.read_stored_run, []
    def tracked(completion, **options):
        calls.append((deepcopy(completion), dict(options)))
        return original(completion, **options)
    monkeypatch.setattr(saved, "read_stored_run", tracked)
    monkeypatch.setattr(saved, "read_stored_frame", _forbidden)
    monkeypatch.setattr(n1, "read_n1_stored_run", _forbidden)
    review = _review(value, expected_pipeline="native-phase-n1", current_machine_id="Second")
    assert review.pipeline == "native-phase-n1"
    assert review.case_ids == tuple(value["completion"]["ordered_case_ids"])
    assert review.frame_metadata(0) == value["state"]["manifest"]["frames"][review.case_ids[0]]
    assert calls[0][0] == value["completion"] and len(calls) == 1
    assert calls[0][1]["stream"] is True and calls[0][1]["load_before"] is False


def test_native_phase_fresh_attempt_retains_unindexed_checkpoint_without_probing_it(native_phase_attachment, monkeypatch):
    value = native_phase_attachment
    checkpoint = value["state"]["manifest"]["resume_checkpoint"]
    _root, _document, products = publication._read_index(value["index"])
    assert all(row["local_path"] != checkpoint["path"] for row in products.values())
    original_stat, original_open = Path.stat, Path.open
    def stat(path, *args, **kwargs):
        if str(path) == checkpoint["path"]:
            return _forbidden()
        return original_stat(path, *args, **kwargs)
    def open_file(path, *args, **kwargs):
        if str(path) == checkpoint["path"]:
            return _forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "open", open_file)
    review = _review(value)
    assert review.pipeline == "native-phase-n1"
    assert review.read_frame(0)["case_id"] == review.case_ids[0]
    assert value["state"]["manifest"]["resume_checkpoint"] == checkpoint


@pytest.mark.parametrize("fault", ["reused", "count", "boolean_count", "missing_count", "mixed_manifest"])
def test_native_phase_resumed_or_ambiguous_frame_origins_are_refused(native_phase_attachment, tmp_path, monkeypatch, fault):
    value = native_phase_attachment
    completion = deepcopy(value["completion"])
    if fault == "reused":
        completion["reused_case_ids"] = completion["ordered_case_ids"][:]
    elif fault == "count":
        completion["newly_computed_frames"] -= 1
    elif fault == "boolean_count":
        completion["newly_computed_frames"] = True
    elif fault == "missing_count":
        completion.pop("newly_computed_frames")
    else:
        completion["frame_producers"][0]["manifest"]["sha256"] = "0" * 64
    index = _republish(tmp_path / "resumed", value["index"], completion)
    monkeypatch.setattr(saved, "read_stored_frame", _forbidden)
    with pytest.raises(ValueError, match="[Rr]esumed"):
        _review(dict(value, completion=completion, index=index))


def test_native_phase_maps_use_original_expected_guards_and_display_contract(native_phase_attachment, monkeypatch):
    from darsia.fluidflower.calibration_review_mass_map import _display_inputs
    value = native_phase_attachment
    review = _review(value)
    original, calls = saved.read_stored_frame, []
    def tracked(record, **options):
        calls.append((deepcopy(record), deepcopy(options["expected"])))
        return original(record, **options)
    monkeypatch.setattr(saved, "read_stored_frame", tracked)
    monkeypatch.setattr(n1, "read_n1_stored_frame", _forbidden)
    stored = review.read_frame(0)
    expected = dict(job_id=value["state"]["manifest"]["job_id"], run=review.expected_run,
        case_id=review.case_ids[0], precision="float32", code_commit="b" * 40,
        configuration=value["completion"]["configuration"], calibration_sha256="c" * 64,
        source=value["state"]["manifest"]["frames"][review.case_ids[0]]["source"])
    assert calls == [(value["completion"]["records"][0], expected)]
    assert "producer_kind" not in stored and "photograph" not in stored
    assert stored["storage_document"]["format"] == "darsia-storage-frame-v1"
    assert stored["native_scalar_reference"]["elapsed_h"] == review.frame_metadata(0)["elapsed_h"]
    for component in ("total", "aqueous", "gas"):
        field, *_rest = _display_inputs(stored, component, 5.)
        assert field is stored["arrays"]["cell_mass_" + component + "_g"]
        assert not field.flags.writeable
    assert all(Path(bound["path"]).name != "products.npz" for bound in stored["loaded_assets"])


def test_native_phase_foreign_collection_and_original_photo_reader(native_phase_attachment, tmp_path, monkeypatch):
    value = native_phase_attachment
    _root, _document, rows = publication._read_index(value["index"])
    keys = [key for key, row in rows.items() if row["kind"] == "map" and key != "native/products.npz"]
    collection = publication.collect_products(value["index"], keys, tmp_path / "collected",
        current_machine_id="Second", source_paths={key: rows[key]["local_path"] for key in keys},
        file_writers_closed=True)
    photos = deepcopy(value["photographs"])
    review = _review(value, current_machine_id="Second", collections=[collection],
                     photographs=photos, geometry_producer_sha256="f" * 64)
    photos[review.case_ids[0]]["sha256"] = "0" * 64
    assert review.photographs == value["photographs"]
    detached = review.photographs
    detached.clear()
    assert review.photographs == value["photographs"]
    original_read, original_photo, calls = Path.read_bytes, export._photograph, []
    def guarded(path):
        if path.is_relative_to(value["state"]["root"]) or path.is_relative_to(tmp_path / "photo"):
            return _forbidden()
        return original_read(path)
    def tracked(bound, stored, **options):
        calls.append(deepcopy(bound))
        return original_photo(bound, stored, **options)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    monkeypatch.setattr(export, "_photograph", tracked)
    monkeypatch.setattr(n1, "read_n1_original_photo", _forbidden)
    stored = review.read_frame(0)
    rgba, evidence = review.read_photo(stored)
    assert calls == [value["photographs"][review.case_ids[0]]]
    assert rgba.shape == (2, 2, 4) and not rgba.flags.writeable
    assert evidence["geometry_producer_sha256"] == "f" * 64


def test_native_phase_photo_needs_explicit_original_binding_and_geometry(native_phase_attachment):
    value = native_phase_attachment
    stored = _review(value).read_frame(0)
    with pytest.raises(FileNotFoundError, match="explicit indexed receipt"):
        _review(value).read_photo(stored)
    with pytest.raises(ValueError, match="explicit original geometry"):
        _review(value, photographs=value["photographs"]).read_photo(stored)
    with pytest.raises(ValueError, match="identity/pre-light"):
        _review(value, photographs=value["photographs"], geometry_producer_sha256="0" * 64).read_photo(stored)
    with pytest.raises(ValueError, match="exact cases"):
        _review(value, photographs={"different_case": next(iter(value["photographs"].values()))})
    original = _review(value, photographs=value["photographs"], geometry_producer_sha256="f" * 64)
    wrong = deepcopy(stored)
    wrong["storage_document"]["source"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original native-phase storage receipt"):
        original.read_photo(wrong)
    wrong = deepcopy(stored)
    wrong["storage_receipt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original native-phase storage receipt"):
        original.read_photo(wrong)
    wrong = deepcopy(stored)
    wrong["grid"]["native_shape"] = [1, 1]
    with pytest.raises(ValueError, match="admitted native-phase frame identity"):
        original.read_photo(wrong)


def test_native_phase_photo_is_still_inside_the_index(native_phase_attachment, tmp_path):
    value = native_phase_attachment
    receipt = next(iter(value["photographs"].values()))
    unindexed = tmp_path / "unindexed-copy.json"
    unindexed.write_bytes(Path(receipt["path"]).read_bytes())
    review = _review(value, photographs={value["completion"]["ordered_case_ids"][0]:
        dict(receipt, path=str(unindexed))}, geometry_producer_sha256="f" * 64)
    with pytest.raises(ValueError, match="outside the exact products-index"):
        review.read_photo(review.read_frame(0))


def test_both_original_protocols_refuse_an_authenticated_opposite_expectation(native_attachment, native_phase_attachment):
    _config, completion, index = native_attachment
    with pytest.raises(ValueError, match="authenticated job/task"):
        NativeSavedReview(index, current_machine_id="Main", expected_run=completion["run"],
            expected_calibration_sha256=completion["calibration"]["sha256"], expected_pipeline="native-phase-n1")
    with pytest.raises(ValueError, match="authenticated job/task"):
        _review(native_phase_attachment, expected_pipeline="n1")


@pytest.mark.parametrize("protocol", ["n1", "native-phase-n1"])
def test_completion_cannot_relabel_an_original_other_protocol_manifest(
        native_attachment, native_phase_attachment, tmp_path, protocol):
    _config, n1_completion, n1_index = native_attachment
    if protocol == "n1":
        completion, index = deepcopy(native_phase_attachment["completion"]), native_phase_attachment["index"]
        completion.pop("pipeline")
    else:
        completion, index = deepcopy(n1_completion), n1_index
        completion["pipeline"] = "native-phase-n1"
        completion.update(reused_case_ids=[], newly_computed_frames=completion["frames"])
    altered = _republish(tmp_path / "relabelled", index, completion)
    with pytest.raises(ValueError, match="manifest"):
        NativeSavedReview(altered, current_machine_id="Main", expected_run=completion["run"],
            expected_calibration_sha256=completion["calibration"]["sha256"], expected_pipeline=protocol)


@pytest.mark.parametrize("declared", ["n1", "lattice-phase-n1", None])
def test_completion_protocol_is_not_normalized_or_removed(native_phase_attachment, tmp_path, declared):
    value = native_phase_attachment
    completion = dict(value["completion"], pipeline=declared)
    index = _republish(tmp_path / "unsupported", value["index"], completion)
    with pytest.raises(ValueError, match="Unsupported original native saved completion protocol"):
        _review(dict(value, completion=completion, index=index))


def test_protocol_specific_photo_helper_never_relabels_frames(native_phase_attachment, native_attachment):
    stored = _review(native_phase_attachment).read_frame(0)
    with pytest.raises(ValueError, match="original N1 stored frame"):
        photo_geometry(stored, expected="f" * 64)
    _config, completion, index = native_attachment
    other = NativeSavedReview(index, current_machine_id="Main", expected_run=completion["run"],
        expected_calibration_sha256=completion["calibration"]["sha256"]).read_frame(0)
    with pytest.raises(ValueError, match="original native-phase stored frame"):
        photo_geometry(other, expected="f" * 64, pipeline="native-phase-n1")
