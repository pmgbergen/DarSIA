"""Source-authored saved-reader regressions; unrun during the B2 source phase."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import phase_review_raw_contradictions as subject


def asset(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return asset(path)


def array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    return asset(path)


def archive(path, fields):
    np.savez(path, **fields)
    return asset(path)


@pytest.fixture
def raw_frame(tmp_path, monkeypatch):
    from darsia.fluidflower import raw_production_review
    from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT, RAW_SOURCE_CONTRACT
    root = tmp_path / "review"
    folder = root / "native_frames/ac01"
    folder.mkdir(parents=True)
    model = tmp_path / "model"
    identity = ("model", "b" * 64)
    for name in ("summary.json", "models/generic/co2_presence_signal.json", "models/generic/gas_within_co2_signal.json"):
        document(model / name, {})
    configuration = document(tmp_path / "configuration.json", {})
    binding = document(tmp_path / "raw_binding.json", {})
    camera = tmp_path / "frame.ARW"
    camera.write_bytes(b"original acquisition bytes")
    acquisition = asset(camera)
    full_shape, shape, box = (5, 6), (2, 3), (2, 1, 3, 2)
    log = array(tmp_path / "log.npy", np.ones((3, *shape), np.float32))
    valid = np.ones(shape, bool)
    valid[1, 2] = False
    valid_ref = array(tmp_path / "valid.npy", valid)
    audit = document(tmp_path / "audit.json", dict(contract=RAW_SOURCE_CONTRACT,
        jpeg_inputs=False, sources=dict(camera_arw=acquisition),
        source_identity=dict(kind="raw-acquisition", raw_sha256=acquisition["sha256"]),
        processor=dict(source_binding=binding),
        renderer=dict(source_kind="raw", native_shape_hw=list(shape),
            native_to_camera_u_h=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])))
    full = dict(complete=True, source_sha256=acquisition["sha256"], phase_shape_hw=list(full_shape),
        native_origin_xy=[-2, -1], native_centre=dict(native_shape_hw=list(shape),
            log_linear_rgb=log, valid_mask=valid_ref))
    native_record = document(tmp_path / "native.json", dict(complete=True, processor_kind="raw_rig",
        source_sha256=acquisition["sha256"], audit=audit, full=full))
    fields = {name: np.zeros(full_shape, bool) for name in subject.CLASS_FIELDS}
    fields["rig_domain"][1:3, 2:5] = True
    fields["modelled"][:] = fields["rig_domain"]
    for field, index in (("filled_observed", (1, 3)), ("filled_unobserved", (2, 2))):
        fields["modelled"][index] = False
        fields[field][index] = True
    classes_ref = archive(tmp_path / "classes.npz", fields)
    labels = np.full(full_shape, 255, np.uint8)
    labels[1:3, 2:5] = [[0, 1, 2], [2, 1, 0]]
    label_path = tmp_path / "labels.png"
    Image.fromarray(labels).save(label_path)
    frame = dict(frame_index=0, case_id="ac01_f0", elapsed_h=1.5,
        acquisition=acquisition, native_record=native_record, audit=audit, rig_classes=classes_ref,
        native_geometry=document(tmp_path / "native_geometry.json", {}),
        rig_geometry=document(tmp_path / "rig_geometry.json", {}))
    native = dict(format=RAW_REVIEW_FORMAT, source_kind="raw", run="ac01", frame_count=1,
        reference={}, frames=[frame], native_configuration=configuration, raw_binding=binding,
        phase_shape_hw=list(full_shape), native_shape_hw=list(shape), native_origin_xy=[-2, -1],
        content_box_xywh=list(box), canvas_wh=[6, 5])
    native_ref = document(folder / "manifest.json", native)
    timeline = tmp_path / "timeline.csv"
    timeline.write_text("frame_index,elapsed_h\n0,1.5\n", encoding="utf-8")
    run = dict(run="ac01", native=native_ref, labels=[asset(label_path)], times=[1.5],
        shape=list(full_shape), timeline=asset(timeline))
    record = dict(source_model_root=str(model), source_model_id=identity[0],
        source_model_sha256=identity[1], pins=[native_ref, configuration, binding])
    recipe = document(tmp_path / "production_review.json", record)
    job = dict(run="ac01", dataset="ffac", candidate_folder=str(model),
        review_output_dir=str(root), phase_map_output_dir=str(tmp_path / "maps"),
        quality_flags=dict(raw_production_review=True), production_review=recipe)
    monkeypatch.setattr(raw_production_review, "validated_job", lambda selected: (record, run, native))
    reader = subject._Pins()
    reader.path(native_ref)
    return SimpleNamespace(root=root, model=model, identity=identity, native=native, native_ref=native_ref,
        native_path=Path(native_ref["path"]), job=job, frame=frame, run=run, record=record,
        labels=labels, label_path=label_path, fields=fields, classes_ref=classes_ref, valid=valid,
        valid_ref=valid_ref, reader=reader)


def load_raw(data):
    return subject._read_raw_production(data.model.resolve(), data.job, data.native_path,
                                        data.native, 0, data.reader)


def test_exact_crop_original_validity_and_saved_fill_populations(raw_frame):
    data = raw_frame
    before = {path: path.read_bytes() for path in data.root.parent.rglob("*") if path.is_file()}
    result = load_raw(data)
    assert result.native == data.native
    assert result.native is not data.native
    assert np.array_equal(result.labels, [[0, 1, 2], [2, 1, 0]])
    assert np.array_equal(result.domain, np.ones((2, 3), bool))
    assert np.array_equal(result.support, [[True, False, True], [False, True, False]])
    assert result.provenance["unsupported_domain_pixels"] == 3
    assert result.provenance["filled_observed_pixels"] == result.provenance["filled_unobserved_pixels"] == 1
    assert result.provenance["content_box_xywh"] == [2, 1, 3, 2]
    assert result.provenance["full_phase_shape_hw"] == [5, 6]
    assert result.provenance["source_model_identity"] == list(data.identity)
    assert result.pins[data.valid_ref["path"]] == data.valid_ref["sha256"]
    assert not any(value.flags.writeable for value in (result.labels, result.domain, result.support))
    assert all(path.read_bytes() == content for path, content in before.items())
    data.reader.verify()


def test_zero_background_outside_saved_domain_is_not_an_admitted_producer_encoding(raw_frame):
    data = raw_frame
    data.labels[~data.fields["rig_domain"]] = 0
    Image.fromarray(data.labels).save(data.label_path)
    data.run["labels"][0] = asset(data.label_path)
    with pytest.raises(ValueError, match="categorical labels disagree with their original saved rig domain"):
        load_raw(data)


def test_missing_final_classes_never_uses_photographic_support(raw_frame):
    data = raw_frame
    data.frame.pop("rig_classes")
    data.frame["photo_support"] = data.valid_ref
    with pytest.raises(subject.UnsupportedSavedPhaseFrame, match="photo_support cannot replace"):
        load_raw(data)


def test_changed_scientific_mask_fails_instead_of_becoming_unscored(raw_frame):
    data = raw_frame
    np.save(data.valid_ref["path"], ~data.valid)
    with pytest.raises(ValueError, match="changed|SHA|differs"):
        load_raw(data)


@pytest.mark.parametrize("corrupt", ["overlap", "dtype", "shape"])
def test_final_saved_classes_are_strict(raw_frame, corrupt):
    data = raw_frame
    if corrupt == "overlap":
        data.fields["modelled"][1, 3] = True
    elif corrupt == "dtype":
        data.fields["rig_domain"] = data.fields["rig_domain"].astype(np.uint8)
    else:
        data.fields["modelled"] = data.fields["modelled"][:-1]
    data.frame["rig_classes"] = archive(Path(data.classes_ref["path"]), data.fields)
    with pytest.raises(ValueError, match="partition|dtype or phase grid"):
        load_raw(data)


@pytest.mark.parametrize("corrupt", ["rgba", "unknown_code", "unresolved_in_domain", "shape"])
def test_categorical_encoding_is_not_coerced_or_reconstructed(raw_frame, corrupt):
    data = raw_frame
    if corrupt == "rgba":
        values = np.zeros((*data.labels.shape, 4), np.uint8)
    elif corrupt == "shape":
        values = data.labels[:-1]
    else:
        values = data.labels.copy()
        values[1, 2] = 19 if corrupt == "unknown_code" else 255
    Image.fromarray(values).save(data.label_path)
    data.run["labels"][0] = asset(data.label_path)
    with pytest.raises(ValueError, match="categorical|native grid"):
        load_raw(data)


def test_model_root_and_native_geometry_are_never_inferred(raw_frame):
    data = raw_frame
    with pytest.raises(ValueError, match="another selected model root"):
        subject._read_raw_production(data.model.parent, data.job, data.native_path, data.native, 0, data.reader)
    data.native["native_origin_xy"] = [-1, -1]
    with pytest.raises(ValueError, match="native origin differs"):
        load_raw(data)


def admit_public_reader(data, monkeypatch):
    from darsia.fluidflower import phase_evidence_audit, phase_validation, phase_saved_runtime_views
    from darsia.fluidflower import phase_blind_review_policy
    document(data.root / "postflight_manifest.json", dict(jobs=[data.job]))
    document(data.root / "validation_binding.json", dict(combined_root=str(data.root)))
    monkeypatch.setattr(phase_evidence_audit, "verified_review_binding",
        lambda *args, **kwargs: dict(combined_root=str(data.root)))
    monkeypatch.setattr(phase_validation, "_model_identity", lambda model: data.identity)
    monkeypatch.setattr(phase_validation, "_source_jobs", lambda root: [data.job])
    monkeypatch.setattr(phase_saved_runtime_views, "native_source",
        lambda job: (data.root, data.native_path, data.native))
    gates = []
    monkeypatch.setattr(phase_blind_review_policy, "require_predictions_visible",
                        lambda root, run: gates.append((Path(root).resolve(), run)))
    return gates


def test_public_reader_preserves_selected_model_and_all_publication_gates(raw_frame, monkeypatch):
    data = raw_frame
    gates = admit_public_reader(data, monkeypatch)
    result = subject.load_saved_phase_frame(data.model, data.root, data.job, 0,
        expected_native_sha256=data.native_ref["sha256"])
    assert result.provenance["phase_review_basis"] == subject.SAVED_NATIVE_BASIS
    assert gates.count((Path(data.job["phase_map_output_dir"]).resolve(), "ac01")) == 2
    assert gates.count((data.model.resolve(), "ac01")) >= 2
    assert str((data.root / "postflight_manifest.json").resolve()) in result.pins


def test_publication_hold_blocks_before_any_prediction_read(raw_frame, monkeypatch):
    from darsia.fluidflower import phase_blind_review_policy
    data = raw_frame
    admit_public_reader(data, monkeypatch)
    def held(*args):
        raise PermissionError("independent model publication hold")
    def forbidden(*args):
        pytest.fail("prediction bytes must not be read under a publication hold")
    monkeypatch.setattr(phase_blind_review_policy, "require_predictions_visible", held)
    monkeypatch.setattr(subject, "_categorical", forbidden)
    with pytest.raises(PermissionError, match="independent model publication hold"):
        subject.load_saved_phase_frame(data.model, data.root, data.job, 0)


def test_stale_native_identity_and_unselected_job_fail(raw_frame, monkeypatch):
    data = raw_frame
    admit_public_reader(data, monkeypatch)
    with pytest.raises(ValueError, match="native SHA differs"):
        subject.load_saved_phase_frame(data.model, data.root, data.job, 0, expected_native_sha256="c" * 64)
    different = deepcopy(data.job)
    different["candidate_folder"] = str(data.model.parent)
    with pytest.raises(ValueError, match="not the original selected"):
        subject.load_saved_phase_frame(data.model, data.root, different, 0)


def test_unregistered_raw_optical_map_does_not_borrow_predecessor_classes(raw_frame, monkeypatch):
    data = raw_frame
    data.job["quality_flags"] = {}
    admit_public_reader(data, monkeypatch)
    with pytest.raises(subject.UnsupportedSavedPhaseFrame, match="predecessor-model support"):
        subject.load_saved_phase_frame(data.model, data.root, data.job, 0)


def test_campaign_profile_binding_must_still_match_after_array_reads(raw_frame, monkeypatch):
    from darsia.fluidflower import phase_evidence_audit
    data = raw_frame
    admit_public_reader(data, monkeypatch)
    calls = []
    def binding(*args, **kwargs):
        calls.append(None)
        return dict(combined_root=str(data.root), physics_profile={"revision": len(calls)})
    monkeypatch.setattr(phase_evidence_audit, "verified_review_binding", binding)
    with pytest.raises(ValueError, match="campaign/profile binding changed"):
        subject.load_saved_phase_frame(data.model, data.root, data.job, 0)


@pytest.fixture
def v5_frame(tmp_path):
    from darsia.fluidflower.phase_training.log_linear_input import SOURCE_FORMAT, EXACT_VALID_MASK_POLICY
    folder = tmp_path / "native_frames/ac01"
    folder.mkdir(parents=True)
    photo = folder / "frame_000.jpg"
    photo.write_bytes(b"display-only JPEG bytes: the validity reader does not decode this")
    values = np.ones((3, 2, 3), np.float32)
    values[:, 1, 2] = np.nan
    valid = np.isfinite(values).all(axis=0)
    log_ref = array(photo.with_suffix(".log_linear_rgb.npy"), values)
    valid_ref = array(photo.with_suffix(".log_linear_valid_mask.npy"), valid)
    frame = dict(frame_index=0, elapsed_h=1.5, source_sha256="a" * 64,
        native_jpeg_sha256=asset(photo)["sha256"],
        v5_log_linear_rgb_path=log_ref["path"], v5_log_linear_rgb_sha256=log_ref["sha256"],
        v5_log_linear_valid_mask_path=valid_ref["path"], v5_log_linear_valid_mask_sha256=valid_ref["sha256"])
    native = dict(run="ac01", frame_count=1, frame_sources=[frame], canvas_wh=[6, 5],
        content_box_xywh=[2, 1, 3, 2], analysis_image_shape_hw=[2, 3],
        v5_photometry=dict(format=SOURCE_FORMAT, unit_denominator=65535., display_jpeg_is_training_input=False),
        v5_float_support_policy=EXACT_VALID_MASK_POLICY)
    bound = document(folder / "manifest.json", native)
    return SimpleNamespace(native=native, bound=bound, valid=valid, valid_ref=valid_ref, photo=photo)


def test_v5_uses_original_validity_reader_without_decoding_or_feature_recomputation(v5_frame, monkeypatch):
    import builtins
    import importlib
    from darsia.fluidflower.phase_training.log_linear_input import RunV5LinearInput

    def forbidden(*args, **kwargs):
        pytest.fail("source qualification must not decode RGB or reconstruct feature support")

    original_import = builtins.__import__
    original_import_module = importlib.import_module
    original_validity = RunV5LinearInput.load_validity
    validity_calls = []

    def require_no_frame_context(name, fromlist=()):
        if (name == "frame_context" or name.endswith(".frame_context")
                or "frame_context" in (fromlist or ())):
            forbidden()

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        require_no_frame_context(name, fromlist)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        require_no_frame_context(name)
        return original_import_module(name, package)

    def recorded_validity(self, photo_path, frame_id, *, extended=False):
        validity_calls.append((Path(photo_path).resolve(), frame_id, extended))
        return original_validity(self, photo_path, frame_id, extended=extended)

    # The feature-context module itself requires campaign DATA at import time.
    # Block that dependency without importing it merely to patch a method.
    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "__import__", guarded_import)
        guarded.setattr(importlib, "import_module", guarded_import_module)
        guarded.setattr(RunV5LinearInput, "load", forbidden)
        guarded.setattr(RunV5LinearInput, "load_validity", recorded_validity)
        result = subject.read_v5_source_validity(v5_frame.bound["path"], 0,
            expected_native_sha256=v5_frame.bound["sha256"])
    assert validity_calls == [(v5_frame.photo.resolve(), "frame_000", False)]
    assert np.array_equal(result.valid, v5_frame.valid)
    assert not result.valid.flags.writeable
    assert result.native == v5_frame.native
    assert result.provenance["scope"] == "source_validity_only"
    assert result.provenance["prediction_support_qualified"] is False
    assert "digest" in result.provenance["limitation"]
    assert not hasattr(result, "labels") and not hasattr(result, "support")


def test_v5_changed_mask_is_corruption_not_missing_coverage(v5_frame):
    np.save(v5_frame.valid_ref["path"], ~v5_frame.valid)
    with pytest.raises(ValueError, match="SHA|changed|differs"):
        subject.read_v5_source_validity(v5_frame.bound["path"], 0,
            expected_native_sha256=v5_frame.bound["sha256"])


def test_v5_without_saved_validity_has_no_synthetic_all_valid_fallback(v5_frame):
    v5_frame.native.pop("v5_float_support_policy")
    bound = document(Path(v5_frame.bound["path"]), v5_frame.native)
    with pytest.raises(subject.UnsupportedSavedPhaseFrame, match="no exact saved per-frame"):
        subject.read_v5_source_validity(bound["path"], 0, expected_native_sha256=bound["sha256"])


def test_v5_wrong_manifest_identity_is_rejected_before_reader(v5_frame):
    with pytest.raises(ValueError, match="SHA|changed|differs"):
        subject.read_v5_source_validity(v5_frame.bound["path"], 0, expected_native_sha256="c" * 64)
