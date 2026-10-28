"""Original-photo display-product gates; only small synthetic saved JPEGs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_original_photo as photo
from darsia.fluidflower.storage_saved_products import StoredProductCollectionRequired


def _asset(path):
    return photo._asset(path)


def _json(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode())
    return _asset(path)


def _fixture(tmp_path):
    yy, xx = np.indices((5, 7))
    rgb = np.stack((yy * 31 + xx * 11, xx * 29, yy * 47), axis=-1).astype(np.uint8)
    path = tmp_path / "saved_native_review.jpg"
    Image.fromarray(rgb).save(path, quality=95, subsampling=0)
    review = _asset(path)
    camera = dict(path=str(tmp_path / "original_camera_NEVER_READ.jpg"), sha256="d" * 64)
    cert = dict(run="synthetic", frame_index=28, camera_jpeg_path=camera["path"], camera_jpeg_sha256=camera["sha256"],
        native_review_jpeg_path=review["path"], native_review_jpeg_sha256=review["sha256"],
        native_shape_hw=[2, 3], content_box_xywh=[1, 1, 3, 2], H_ac28_to_frame_native=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        binding=dict(geometry_module_sha256="f" * 64, input_sources=dict(camera_jpeg=camera),
                     content_box_xywh=[1, 1, 3, 2], native_shape_hw=[2, 3],
                     original_shape_configs=[], original_shape_corrections=[]))
    cert_asset = _json(tmp_path / "certificate.json", cert)
    native = dict(cert, geometry_certificate_path=cert_asset["path"], geometry_certificate_sha256=cert_asset["sha256"])
    native_asset = _json(tmp_path / "native_geometry.json", native)
    current = dict(run="synthetic", frame_index=1, source_sha256=camera["sha256"], source_photo=camera,
        source_geometry_receipt={k: cert_asset[k] for k in ("path", "sha256")},
        source_content_box_xywh=[1, 1, 3, 2], source_native_shape_hw=[2, 3],
        native_origin_xy=[-1, -1], phase_shape_hw=[7, 7], photograph_canvas_wh=[7, 5],
        H_ac28_to_frame_native=deepcopy(cert["H_ac28_to_frame_native"]))
    current_asset = _json(tmp_path / "current_geometry.json", current)
    args = dict(task=dict(run="synthetic", precision="float32"), frame_spec=dict(case_id="selected_f1", frame_index=1, source=dict(source_sha256="d" * 64)),
        native_geometry=native_asset, geometry_certificate=cert_asset, current_geometry=current_asset,
        saved_native_review=review, geometry_producer_sha256="f" * 64)
    calls = []
    allowed = {f["path"] for f in [native_asset, cert_asset, current_asset, review]}
    def resolve(bound):
        calls.append(bound["path"])
        assert bound["path"] in allowed, "No camera/RAW/correction/model/manifest follow"
        return Path(bound["path"])
    return dict(args=args, calls=calls, resolve=resolve, current=current)


def test_asymmetric_known_ROI_copy_preserves_black_pixels_without_validity_mask():
    rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    rgb[1, 1] = 0
    before = rgb.copy()
    rgba = photo._place_known_roi(rgb, box=[1, 1, 3, 2], target_shape=[7, 7], native_origin=[-1, -1])
    np.testing.assert_array_equal(rgba[1:3, 1:4, :3], rgb[1:3, 1:4])
    assert rgba[1, 1, 3] == 255 and not rgba[1, 1, :3].any()
    coverage = np.zeros((7, 7), np.uint8)
    coverage[1:3, 1:4] = 255
    np.testing.assert_array_equal(rgba[..., 3], coverage)
    assert not rgba[coverage == 0, :3].any() and not rgba[5:].any()
    np.testing.assert_array_equal(rgb, before)


def test_producer_uses_only_saved_JPEG_and_exact_metadata_then_consumers_can_read_PNG(tmp_path):
    state = _fixture(tmp_path)
    originals = {p: Path(p).read_bytes() for p in [state["args"][k]["path"] for k in ("native_geometry", "geometry_certificate", "current_geometry", "saved_native_review")]}
    with Image.open(state["args"]["saved_native_review"]["path"]) as image:
        expected_roi = np.asarray(image).copy()[1:3, 1:4]
    result = photo.publish_registered_original_png(tmp_path / "photo.png", **state["args"], resolve_asset=state["resolve"])
    with Image.open(result["photograph"]["path"]) as image:
        assert image.mode == "RGBA"
        rgba = np.asarray(image).copy()
    np.testing.assert_array_equal(rgba[1:3, 1:4, :3], expected_roi)
    assert rgba.shape == (7, 7, 4) and not rgba[5:].any()
    receipt = json.loads(Path(result["receipt"]["path"]).read_bytes())
    assert receipt["known_ROI_pixel_identity"] == receipt["copied_ROI_pixel_identity"]
    assert receipt["alpha_inside_ROI"] == 255 and receipt["alpha_outside_ROI"] == 0
    assert "not per-pixel camera validity" in receipt["alpha_meaning"]
    assert receipt["camera_timeline_index"] == 28 and receipt["current_selected_frame_index"] == 1
    assert not receipt["full_canvas_photograph_coverage"] and not receipt["colour_light_or_v5_changes"]
    assert not receipt["camera_RAW_or_correction_chain_executed"] and len(state["calls"]) == 4
    assert all(Path(p).read_bytes() == b for p, b in originals.items())


@pytest.mark.parametrize("bad", ["source", "producer", "case_index", "homography", "origin"])
def test_changed_source_or_current_position_contract_is_rejected_before_output(tmp_path, bad):
    state = _fixture(tmp_path)
    if bad == "source": state["args"]["frame_spec"]["source"]["source_sha256"] = "e" * 64
    elif bad == "producer": state["args"]["geometry_producer_sha256"] = "e" * 64
    elif bad == "case_index": state["args"]["frame_spec"]["frame_index"] = 2
    else:
        current = state["current"]
        if bad == "homography": current["H_ac28_to_frame_native"][0][2] = 1
        else: current["native_origin_xy"] = [0, 0]
        state["args"]["current_geometry"] = _json(tmp_path / "current_geometry.json", current)
    with pytest.raises(ValueError, match="identity|placement"):
        photo.publish_registered_original_png(tmp_path / "not_created/photo.png", **state["args"], resolve_asset=state["resolve"])
    assert not (tmp_path / "not_created").exists()


def test_changed_certificate_cannot_be_hidden_in_a_resealed_native_geometry(tmp_path):
    state = _fixture(tmp_path)
    cert = json.loads(Path(state["args"]["geometry_certificate"]["path"]).read_bytes())
    cert["camera_jpeg_sha256"] = "e" * 64
    state["args"]["geometry_certificate"] = _json(tmp_path / "certificate.json", cert)
    with pytest.raises(ValueError, match="certificate"):
        photo.publish_registered_original_png(tmp_path / "not_created/photo.png", **state["args"], resolve_asset=state["resolve"])
    assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize("bad", ["float_rgb", "outside_source", "outside_target"])
def test_copy_refuses_conversion_or_out_of_bounds_ROI(bad):
    rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    box, target = [1, 1, 3, 2], [7, 7]
    if bad == "float_rgb": rgb = rgb.astype(np.float32) / 255
    elif bad == "outside_source": box = [6, 4, 3, 2]
    else: target = [2, 3]
    with pytest.raises(ValueError, match="uint8|placement"):
        photo._place_known_roi(rgb, box=box, target_shape=target, native_origin=[-box[0], -box[1]])


def test_missing_exact_saved_review_is_collect_not_camera_fallback(tmp_path):
    state = _fixture(tmp_path)
    def missing(bound):
        if bound == state["args"]["saved_native_review"]:
            raise FileNotFoundError(bound["path"])
        return state["resolve"](bound)
    with pytest.raises(StoredProductCollectionRequired) as failure:
        photo.publish_registered_original_png(tmp_path / "not_created/photo.png", **state["args"], resolve_asset=missing)
    assert failure.value.descriptor == state["args"]["saved_native_review"]
    assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize("occupied", ["photo", "receipt"])
def test_exclusive_outputs_are_refused_before_any_source_read(tmp_path, occupied):
    state = _fixture(tmp_path)
    destination = tmp_path / "photo.png"
    path = destination if occupied == "photo" else tmp_path / "photo.png.receipt.json"
    path.write_bytes(b"keep original")
    with pytest.raises(FileExistsError, match="immutable"):
        photo.publish_registered_original_png(destination, **state["args"], resolve_asset=state["resolve"])
    assert path.read_bytes() == b"keep original" and not state["calls"]
