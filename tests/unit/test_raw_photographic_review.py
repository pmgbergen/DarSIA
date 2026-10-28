"""RAW-derived lossless photographic review roots (package D2b).

Synthetic records on a 10 x 16 native grid inside a 14 x 24 phase canvas. The
camera-U renderer and the configured display modes are stubbed: this file
tests the review root's wiring, placement, identity and strictness, not the
shared colour science, which keeps its own tests.
"""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower import raw_photographic_review as review
from darsia.fluidflower.native import display_modes, raw_photograph
from darsia.fluidflower.phase_ffac_rig_camera_metadata import EXTENDED_SCIENTIFIC_CONTRACT as EXTENDED_CONTRACT

NATIVE_HW = (10, 16)
PHASE_HW = (14, 24)
ORIGIN_XY = (-5, -3)  # The native grid starts at canvas (5, 3).
FILL = [19, 22, 27]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode("utf-8")
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _write_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _record(tmp_path, name, *, raw_sha, profile_id="late", binding=None, jpeg_inputs=False, shift_x=0):
    folder = tmp_path / "records" / name
    log = np.empty((3, *NATIVE_HW), dtype=np.float32)
    log[:] = np.log(np.linspace(2.0, 60000.0, NATIVE_HW[1], dtype=np.float32))[None, None, :]
    log[:, 0, 0] = np.nan
    valid = np.ones(NATIVE_HW, dtype=bool)
    valid[0, 0] = False
    camera = dict(path=str(folder / f"{name}.ARW"), sha256=raw_sha)
    (folder).mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.ARW").write_bytes(b"synthetic RAW " + name.encode())
    camera["sha256"] = hashlib.sha256((folder / f"{name}.ARW").read_bytes()).hexdigest()
    audit = _write_json(folder / "native_audit.json", dict(
        contract=review.RAW_SOURCE_CONTRACT, jpeg_inputs=jpeg_inputs,
        sources=dict(camera_arw=camera),
        source_identity=dict(kind="raw-acquisition", raw_sha256=camera["sha256"]),
        processor=dict(source_binding=binding or {}),
        H_ac28_to_frame_native=[[1.0, 0.0, 1.0 + shift_x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        H_v5_B_to_frame_native=[[1.0, 0.0, float(shift_x)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        renderer=dict(source_kind="raw", profile_id=profile_id, camera_u_wh=[20, 12],
                      native_shape_hw=list(NATIVE_HW),
                      native_to_camera_u_h=[[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])))
    full = dict(complete=True, source_sha256=camera["sha256"], phase_shape_hw=list(PHASE_HW),
                native_origin_xy=list(ORIGIN_XY), contract=EXTENDED_CONTRACT, ring_windows=[],
                storage="exact-native-centre-plus-disjoint-frozen-raw-ring-windows",
                native_centre=dict(native_shape_hw=list(NATIVE_HW),
                                   log_linear_rgb=_write_npy(folder / "native_log.npy", log),
                                   valid_mask=_write_npy(folder / "native_valid.npy", valid)))
    return dict(audit=audit, full=full, camera=camera, valid=valid, folder=folder)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    configuration = _write_json(tmp_path / "native_configuration.json", dict(
        registration=dict(common=dict(profiles=dict(late={}))),
        display_modes=dict(modes=["natural", "lift1", "lift2"], tile_rows=4, unfilled_rgb8=FILL)))
    binding = _write_json(tmp_path / "raw_source_binding.json", dict(contract="binding"))
    reference = _record(tmp_path, "baseline", raw_sha="", binding={"sha256": "other"})
    native_row = _write_json(tmp_path / "records/baseline/raw_native_row.json",
                             dict(raw_producer_audit=reference["audit"]))
    acquisitions = _write_json(tmp_path / "raw_acquisitions.json", dict(records=[dict(
        source=dict(run="ac52"), full=reference["full"], native_row=native_row)]))
    frames = []
    for index, case_id in enumerate(("ac52_f1", "ac52_f2")):
        record = _record(tmp_path, case_id, raw_sha="", binding=binding, shift_x=index)
        native = _write_json(record["folder"] / "native.json", dict(
            complete=True, processor_kind="raw_rig", audit=record["audit"], full=record["full"],
            source_sha256=record["camera"]["sha256"]))
        frames.append(dict(case_id=case_id, frame_index=index, elapsed_h=0.5 * index,
                           native_record=native,
                           native_geometry=_write_json(record["folder"] / "native_geometry.json", {}),
                           rig_geometry=_write_json(record["folder"] / "rig_geometry.json", {}),
                           photo_support=record["valid"]))
    renders = []

    def camera_u(path, *, profile_path, profile_sha256, profile_id, expected_sha256=None, **kwargs):
        renders.append((path.name, profile_id))
        assert profile_sha256 == configuration["sha256"]
        image = np.zeros((12, 20, 3), dtype=np.uint8)
        image[..., 0] = np.arange(20, dtype=np.uint8)[None, :]
        image[..., 1] = np.arange(12, dtype=np.uint8)[:, None]
        return image

    def modes(log_rgb, support, *, tile_rows, parameters, selected_modes, **kwargs):
        assert log_rgb.dtype == np.float32 and support.dtype == np.bool_
        return ({name: np.full((*NATIVE_HW, 3), (index + 1) / 255, dtype=np.float32)
                 for index, name in enumerate(selected_modes)}, {})

    monkeypatch.setattr(raw_photograph, "load_camera_u_rgb8", camera_u)
    monkeypatch.setattr(display_modes, "render_display_modes", modes)
    return dict(configuration=configuration, binding=binding, frames=frames, renders=renders,
                reference=dict(acquisition_manifest=acquisitions, record_index=0,
                               photo_support=reference["valid"]))


def _write(tmp_path, inputs, root="review"):
    return review.write_raw_review_root(
        tmp_path / root, run="AC52", native_configuration=inputs["configuration"],
        raw_binding=inputs["binding"], reference=inputs["reference"], frames=inputs["frames"])


def test_writer_makes_one_lossless_root_with_the_required_manifest(tmp_path, inputs):
    path = _write(tmp_path, inputs)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "review/native_frames/ac52/manifest.json"
    assert manifest["format"] == review.RAW_REVIEW_FORMAT and manifest["source_kind"] == "raw"
    assert manifest["run"] == "ac52"
    assert manifest["modes"] == ["original", "natural", "lift1", "lift2", "raw_detail"]
    assert manifest["native_configuration"] == inputs["configuration"]
    assert manifest["raw_binding"] == inputs["binding"]
    assert manifest["content_box_xywh"] == [5, 3, 16, 10] and manifest["canvas_wh"] == [24, 14]
    assert [frame["case_id"] for frame in manifest["frames"]] == ["ac52_f1", "ac52_f2"]
    frame = manifest["frames"][1]
    assert frame["frame_index"] == 1 and frame["elapsed_h"] == 0.5
    assert set(frame["assets"]) == {"original", "natural", "lift1", "lift2", "raw_detail"}
    assert frame["assets"]["natural"]["path"] == "frames/ac52_f2/natural.png"
    reference = manifest["reference"]
    assert reference["record_index"] == 0 and set(reference["assets"]) == set(frame["assets"])
    assert inputs["renders"] == [("baseline.ARW", "late"), ("ac52_f1.ARW", "late"), ("ac52_f2.ARW", "late")]
    for asset in [*frame["assets"].values(), frame["photo_support"]]:
        file = path.parent / asset["path"]
        assert file.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert hashlib.sha256(file.read_bytes()).hexdigest() == asset["sha256"]


def test_original_is_the_camera_raw_pulled_onto_the_canvas(tmp_path, inputs):
    _write(tmp_path, inputs)
    image = np.asarray(Image.open(review.resolve_raw_photograph(tmp_path / "review", "ac52", 0, mode="original")))
    assert image.shape == (*PHASE_HW, 3)
    # Canvas (x, y) is native (x - 5, y - 3), which the audit maps to U (x - 3, y - 2).
    assert tuple(image[4, 9, :2]) == (6, 2) and tuple(image[13, 21, :2]) == (18, 11)
    assert tuple(image[0, 0]) == tuple(FILL)  # Outside the camera view.


def test_display_modes_sit_on_the_native_box_with_the_configured_fill(tmp_path, inputs):
    _write(tmp_path, inputs)
    natural = np.asarray(Image.open(review.resolve_raw_photograph(tmp_path / "review", "ac52", 1)))
    lift = np.asarray(Image.open(review.resolve_raw_photograph(tmp_path / "review", "ac52", 1, mode="lift1")))
    assert natural.shape == (*PHASE_HW, 3)
    assert tuple(natural[3, 5]) == (1, 1, 1) and tuple(lift[12, 20]) == (2, 2, 2)
    assert tuple(natural[2, 5]) == tuple(FILL) and tuple(natural[3, 4]) == tuple(FILL)


def test_raw_detail_is_the_pinned_composer_and_transfer_of_the_extended_raw(tmp_path, inputs):
    from darsia.fluidflower.phase_rig_raw_display import BACKGROUND, render_raw_detail
    _write(tmp_path, inputs)
    detail = np.asarray(Image.open(review.resolve_raw_photograph(tmp_path / "review", "ac52", 0, mode="raw_detail")))
    log = np.load(inputs["frames"][0]["native_record"]["path"].replace("native.json", "native_log.npy"))
    signal = np.full((3, *PHASE_HW), np.nan, dtype=np.float32)
    signal[:, 3:13, 5:21] = log
    support = np.zeros(PHASE_HW, dtype=bool)
    support[3:13, 5:21] = np.isfinite(log).all(0)
    signal[:, ~support] = np.nan
    assert np.array_equal(detail, render_raw_detail(signal, support, tile_rows=4))
    assert tuple(detail[3, 5]) == BACKGROUND and tuple(detail[0, 0]) == BACKGROUND
    assert detail[5, 20].sum() > detail[5, 6].sum()  # Brighter RAW to the right.


def test_pinned_record_restates_the_raw_record_without_new_values(tmp_path):
    record = _record(tmp_path, "pinned", raw_sha="")
    sources = review.raw_sources(record["full"], record["audit"])
    pinned, manifest = review.pinned_raw_record(sources)
    centre = pinned["full_extended_scientific"]["native_centre"]
    assert pinned["source_sha256"] == centre["source_sha256"] == record["full"]["source_sha256"]
    assert pinned["full_native_scientific"]["log_linear_rgb"] == record["full"]["native_centre"]["log_linear_rgb"]
    assert pinned["rig_domain_revision"] == dict(canvas_wh=[24, 14], phase_shape_hw=[14, 24], native_origin_xy=[-5, -3])
    assert manifest == dict(content_box_xywh=[5, 3, 16, 10], analysis_image_shape_hw=[10, 16])
    assert "source_sha256" not in record["full"]["native_centre"]  # The record itself is not changed.


def test_studio_rows_are_recognized_as_raw_review_rows(tmp_path, inputs):
    manifest = json.loads(_write(tmp_path, inputs).read_text(encoding="utf-8"))
    assert all(review.raw_review_row(row) for row in (manifest["reference"], *manifest["frames"]))
    assert not review.raw_review_row({"v5_display_modes": {}}) and not review.raw_review_row(None)


def test_photo_support_round_trips_on_canvas_and_native_grid(tmp_path, inputs):
    _write(tmp_path, inputs)
    canvas = review.load_raw_photo_support(tmp_path / "review", "ac52", 0)
    native = review.load_raw_photo_support(tmp_path / "review", "ac52", 0, canvas=False)
    assert canvas.shape == PHASE_HW and native.shape == NATIVE_HW
    assert np.array_equal(native, inputs["frames"][0]["photo_support"])
    assert not canvas[:3].any() and canvas[3, 6] and not canvas[3, 5]
    reference = review.load_raw_photo_support(tmp_path / "review", "ac52", reference=True, canvas=False)
    assert np.array_equal(reference, inputs["reference"]["photo_support"])


def test_a_root_is_never_written_twice_and_assets_are_sha_checked(tmp_path, inputs):
    path = _write(tmp_path, inputs)
    with pytest.raises(FileExistsError):
        _write(tmp_path, inputs)
    target = path.parent / "frames/ac52_f1/natural.png"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA"):
        review.resolve_raw_photograph(tmp_path / "review", "ac52", 0, mode="natural",
                                      reader=review.VerifiedAssetReader())
    with pytest.raises(ValueError, match="no 'change' photograph"):
        review.resolve_raw_photograph(tmp_path / "review", "ac52", 0, mode="change")


def test_reader_returns_none_for_other_formats_and_rejects_a_broken_raw_manifest(tmp_path, inputs):
    assert review.read_raw_review_manifest(tmp_path / "missing", "ac52") is None
    v5 = tmp_path / "v5/native_frames/ac52/manifest.json"
    v5.parent.mkdir(parents=True)
    v5.write_text(json.dumps({"run": "ac52", "frame_sources": []}), encoding="utf-8")
    assert review.read_raw_review_manifest(tmp_path / "v5", "ac52") is None
    path = _write(tmp_path, inputs)
    manifest = review.read_raw_review_manifest(tmp_path / "review", "AC52")
    assert manifest["frame_count"] == 2
    broken = dict(manifest, frames=list(reversed(manifest["frames"])))
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="out of order"):
        review.read_raw_review_manifest(tmp_path / "review", "ac52")


def test_records_and_modes_are_looked_up_without_file_io(tmp_path, inputs):
    _write(tmp_path, inputs)
    manifest = review.read_raw_review_manifest(tmp_path / "review", "ac52")
    assert review.raw_review_record(manifest, 1)["case_id"] == "ac52_f2"
    assert review.raw_review_record(manifest, reference=True)["record_index"] == 0
    with pytest.raises(ValueError, match="outside"):
        review.raw_review_record(manifest, 2)
    assert review.raw_modes_available(manifest, ("natural", "lift1"))
    assert review.raw_modes_available(manifest, ("original",), reference=True)
    assert not review.raw_modes_available(manifest, ("change",))


def test_sources_must_be_a_raw_acquisition_without_jpeg_inputs(tmp_path):
    record = _record(tmp_path, "jpeg_fed", raw_sha="", jpeg_inputs=True)
    with pytest.raises(ValueError, match="not a RAW-acquisition"):
        review.raw_sources(record["full"], record["audit"])
    clean = _record(tmp_path, "clean", raw_sha="")
    sources = review.raw_sources(clean["full"], clean["audit"])
    assert sources["content_box_xywh"] == (5, 3, 16, 10) and sources["profile_id"] == "late"
    with pytest.raises(ValueError, match="different acquisitions"):
        review.raw_sources(dict(clean["full"], source_sha256="0" * 64), clean["audit"])


def test_frames_must_share_one_binding_and_count_from_zero(tmp_path, inputs):
    frames = [dict(inputs["frames"][0], frame_index=1)]
    with pytest.raises(ValueError, match="count 0, 1, 2"):
        review.write_raw_review_root(
            tmp_path / "a", run="ac52", native_configuration=inputs["configuration"],
            raw_binding=inputs["binding"], reference=inputs["reference"], frames=frames)
    other = dict(inputs["binding"], sha256="f" * 64)
    with pytest.raises(ValueError, match="another RAW source binding"):
        review.write_raw_review_root(
            tmp_path / "b", run="ac52", native_configuration=inputs["configuration"],
            raw_binding=other, reference=inputs["reference"], frames=inputs["frames"])
    with pytest.raises(ValueError, match="Unknown RAW review modes"):
        review.write_raw_review_root(
            tmp_path / "c", run="ac52", native_configuration=inputs["configuration"],
            raw_binding=inputs["binding"], reference=inputs["reference"], frames=inputs["frames"],
            modes=("original", "sepia"))


def test_asset_paths_stay_inside_the_run_folder(tmp_path):
    good = dict(path="frames/ac52_f1/natural.png", sha256="a" * 64, size_bytes=3)
    expected = dict(good, path=str(tmp_path / "frames" / "ac52_f1" / "natural.png"))
    assert review.raw_asset_descriptor(tmp_path, good) == expected
    for bad in ("../x.png", "frames\\..\\..\\x.png", "C:/x.png", "/x.png", "C:x.png"):
        with pytest.raises(ValueError, match="escapes"):
            review.raw_asset_descriptor(tmp_path, dict(good, path=bad))


def test_a_parsed_manifest_is_checked_without_its_path(tmp_path, inputs):
    manifest = json.loads(_write(tmp_path, inputs).read_text(encoding="utf-8"))
    assert review.validate_raw_review_manifest(manifest, "AC52") == manifest
    for broken in (dict(manifest, format="other"), dict(manifest, run="ac53"), dict(manifest, frame_count=3)):
        with pytest.raises(ValueError, match="incomplete or out of order"):
            review.validate_raw_review_manifest(broken, "ac52")


def test_canvas_placement_helpers():
    assert review.canvas_content_box((-5, -3), PHASE_HW, NATIVE_HW) == (5, 3, 16, 10)
    with pytest.raises(ValueError, match="inside its phase canvas"):
        review.canvas_content_box((-10, -3), PHASE_HW, NATIVE_HW)
    sources = dict(content_box_xywh=(5, 3, 16, 10), phase_shape_hw=PHASE_HW)
    placed = review.place_on_canvas(np.ones(NATIVE_HW, dtype=np.uint8), sources, fill_rgb8=0)
    assert placed.shape == PHASE_HW and placed.sum() == 160 and placed[3, 5] == 1


def test_every_public_function_has_a_docstring():
    for name in review.__all__:
        item = getattr(review, name)
        if callable(item):
            assert item.__doc__ and item.__doc__.strip(), name
