"""Historical UM/BM region identity and nonlinear full-frame mask transfer."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.phase import um_bm_review_regions as regions
from darsia.fluidflower.phase_training.review_regions import annotation_mask


def _raw(value):
    return json.dumps(value).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _manifest(series="ffum", *, padded=False):
    return dict(dataset=series, run="run10" if series == "ffum" else "c1",
        canvas_wh=[14, 13] if padded else [10, 10], analysis_image_shape_hw=[10, 10],
        content_box_xywh=[2, 3, 10, 10] if padded else [0, 0, 10, 10],
        full_analysis_resolution=True, additional_spatial_resampling=False,
        frame_sources=[dict(frame_index=1, source_name="original.JPG" if series == "ffum" else "original.TIF",
                            source_sha256="a" * 64)])


def _frame(manifest):
    raw = _raw(manifest)
    return regions.bind_review_frame(raw, expected_sha256=_sha(raw), manifest_path="original/manifest.json",
        series_id=manifest["dataset"], run=manifest["run"], frame_index=1,
        source_name=manifest["frame_sources"][0]["source_name"])


def _record(manifest, *, polygon=False):
    payload = dict(id="original-id", frame_index=0, original_frame_index=0, shape="rectangle",
        coordinate_space="normalized_native_review_canvas_16x9", points=[],
        x=.2, y=.2, width=.2, height=.4, category="confirmed_no_co2")
    if polygon:
        payload.update(shape="freehand", points=[[.2,.3],[.6,.3],[.6,.45],[.4,.45],[.4,.7],[.2,.7]])
    return dict(region_id="run:manual:original-id", run=manifest["run"], dataset=manifest["dataset"],
        frame_index=0, phase_class="water", split="validation", payload=payload)


@pytest.mark.parametrize("series", ["ffum", "ffbm"])
def test_captured_evidence_keeps_original_identity_labels_split_and_aligned_frame(tmp_path, series):
    manifest = _manifest(series, padded=series == "ffum")
    record = _record(manifest, polygon=series == "ffum")
    raw = _raw(manifest)
    moved = tmp_path / "moved.json"
    moved.write_bytes(raw)
    extraction_record = dict(region_id=record["region_id"], status="no-unambiguous-supervision")
    extraction = dict(frames=[dict(native_manifest="old/location/manifest.json",
        inputs_sha256={"old/location/manifest.json": _sha(raw)}, run=manifest["run"],
        frame_index=1, source_name=manifest["frame_sources"][0]["source_name"], records=[extraction_record])])
    protocol = {}
    for name, document in [("evidence", dict(regions=[record])), ("extraction", extraction)]:
        raw = _raw(document)
        (tmp_path / f"{name}.json").write_bytes(raw)
        protocol[f"{name}_sha256"] = _sha(raw)
    (tmp_path / "regional_training_protocol.json").write_bytes(_raw(protocol))
    bound, = regions.load_model_regions(tmp_path, series_id=series, resolve_path=lambda path: moved)
    assert bound.record == record and bound.extraction_record == extraction_record
    assert bound.record["frame_index"] == 0 and bound.frame.frame_index == 1
    assert bound.frame.source["source_name"].endswith(".JPG" if series == "ffum" else ".TIF")
    # The moved path is accepted only with the same bound bytes.
    moved.write_bytes(moved.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum differs"):
        regions.load_model_regions(tmp_path, series_id=series, resolve_path=lambda path: moved)


def test_nonlinear_transfer_preserves_bent_mask_and_invalid_fullframe_support():
    manifest = _manifest()
    record = _record(manifest)
    region = regions.bind_review_region(record, _frame(manifest))
    old = annotation_mask(record["payload"], canvas_wh=[10,10], content_box=[0,0,10,10])
    support = np.ones((10, 14), dtype=bool)
    support[4, 3] = False

    def mapping(xy):
        result = xy.astype(float)
        result[:, 0] -= .3 * (result[:, 1] - 4) ** 2
        result[xy[:, 0] == 13] = np.nan
        return result

    transferred = regions.transfer_region(region, new_to_content_xy=mapping, valid_support=support, chunk_rows=3)
    expected = np.zeros_like(support)
    for y in range(10):
        for x in range(13):
            old_x = int(np.floor(x - .3 * (y - 4) ** 2 + .5))
            if 0 <= old_x < 10:
                expected[y, x] = old[y, old_x] and support[y, x]
    assert np.array_equal(transferred.mask, expected)
    ys, xs = np.nonzero(expected)
    assert expected.sum() < (ys.max()-ys.min()+1)*(xs.max()-xs.min()+1)
    assert transferred.region.record == record
    assert np.array_equal(regions.transfer_region(region, new_to_content_xy=mapping,
        valid_support=support, chunk_rows=10).mask, expected)


def test_freehand_and_canvas_offset_use_original_rasterization_without_a_box():
    manifest = _manifest(padded=True)
    record = _record(manifest, polygon=True)
    bound = regions.bind_review_region(record, _frame(manifest))
    expected = annotation_mask(record["payload"], canvas_wh=[14,13], content_box=[2,3,10,10])
    result = regions.transfer_region(bound, new_to_content_xy=lambda xy: xy, valid_support=np.ones((10,10), bool))
    assert np.array_equal(result.mask, expected)
    assert result.region.record == record
    # The retained polygon has pixels absent from its enclosing rectangle.
    ys, xs = np.nonzero(expected)
    assert expected.sum() < (ys.max()-ys.min()+1)*(xs.max()-xs.min()+1)


def test_raw_validation_box_keeps_id_and_explicit_split_without_training_conversion():
    frame = _frame(_manifest("ffbm"))
    annotation = _record(frame.manifest)["payload"]
    annotation["split"] = "held-out"
    result = regions.bind_review_region(annotation, frame)
    assert result.record == annotation and result.extraction_record is None
    assert "phase_class" not in result.record


def test_historical_quiz_polygon_keeps_proposal_id_and_analysis_evidence():
    manifest = _manifest()
    record = _record(manifest, polygon=True)
    record["source"] = "quiz"
    record["payload"]["proposal_id"] = record["payload"].pop("id")
    record["payload"]["analysis_polygon_px"] = [[2,3],[6,3],[6,5],[2,5]]
    result = regions.bind_review_region(record, _frame(manifest))
    assert result.record == record
    assert "id" not in result.record["payload"]


@pytest.mark.parametrize("change", ["series", "resampled", "source"])
def test_unbound_or_different_geometry_is_rejected(change):
    manifest = _manifest()
    if change == "series":
        manifest["dataset"] = "ffac"
    elif change == "resampled":
        manifest["additional_spatial_resampling"] = True
    else:
        manifest["frame_sources"][0]["source_name"] = "different.JPG"
    raw = _raw(manifest)
    with pytest.raises(ValueError):
        regions.bind_review_frame(raw, expected_sha256=_sha(raw), manifest_path="manifest.json",
            series_id="ffum", run="run10", frame_index=1, source_name="original.JPG")


def test_unknown_polygon_and_missing_coordinate_map_are_not_treated_as_boxes():
    manifest = _manifest()
    record = _record(manifest)
    record["payload"]["shape"] = "freehand"
    with pytest.raises(ValueError, match="retain its polygon"):
        regions.bind_review_region(record, _frame(manifest))
    bound = regions.bind_review_region(_record(manifest), _frame(manifest))
    with pytest.raises(ValueError, match="explicit complete coordinate map"):
        regions.transfer_region(bound, new_to_content_xy=None, valid_support=np.ones((10,10), bool))


def test_existing_drift_corrector_supplies_its_actual_matrix_and_restores_hook(monkeypatch):
    import cv2
    import darsia

    image = np.zeros((12, 16, 3), np.float32)
    image[4, 7] = 1
    correction = darsia.DriftCorrection(image, config={"active": True})
    matrix = np.array([[1.,0.,2.],[0.,1.,-1.]])
    estimator = correction.translation_estimator
    original = lambda *args, **kwargs: (matrix, True)
    monkeypatch.setattr(estimator, "find_effective_translation", original)
    registered, captured = regions.capture_legacy_drift(correction, image)
    assert np.array_equal(captured, matrix)
    assert np.array_equal(registered, cv2.warpAffine(image, matrix, (16,12)))
    assert estimator.find_effective_translation is original
    failure = lambda *args, **kwargs: (None, False)
    monkeypatch.setattr(estimator, "find_effective_translation", failure)
    with pytest.raises(ValueError, match="cannot be aligned"):
        regions.capture_legacy_drift(correction, image)
    assert estimator.find_effective_translation is failure


def test_saved_nonlinear_grid_is_composed_with_inverse_drift_and_half_pixel_resize():
    y, x = np.mgrid[:5,:6]
    # Actual cache ordering is row, column, flattened in C order.
    cache = dict(shape=(5,6), grid=np.array([(y + .2*x*x).ravel(), (x + .1*y*y).ravel()]))
    matrix = np.array([[1.,0.,2.],[0.,1.,-3.]])
    points = np.array([[2.,1.],[4.,3.],[1.5,2.],[-1.,2.]])
    result = regions.legacy_content_to_photo_xy(points, curvature_cache=cache,
        drift_matrix=matrix, photo_wh=(12,10), resized_wh=(6,5))
    registered = np.array([[2.1,1.8],[4.9,6.2],[1.9,2.5]])
    expected = (registered - [2.,-3.] + .5)*2 - .5
    assert np.allclose(result[:3], expected) and np.isnan(result[3]).all()
    with pytest.raises(ValueError, match="actual frame drift"):
        regions.legacy_content_to_photo_xy(points, curvature_cache=cache,
            drift_matrix=None, photo_wh=(12,10), resized_wh=(6,5))


def test_record_is_copied_without_modifying_original_annotation():
    manifest = _manifest()
    record = _record(manifest)
    saved = deepcopy(record)
    bound = regions.bind_review_region(record, _frame(manifest))
    bound.record["payload"]["category"] = "unchanged-original"
    assert record == saved


def test_shared_drive_paths_use_the_configured_data_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("DARSIA_FLUIDFLOWER_DATA_ROOT", raising=False)
    assert regions.shared_drive_path(tmp_path / "local") == tmp_path / "local"
    with pytest.raises(RuntimeError, match="DARSIA_FLUIDFLOWER_DATA_ROOT"):
        regions.shared_drive_path("Z:\\FF_UM\\rig")
    with pytest.raises(RuntimeError, match="DARSIA_FLUIDFLOWER_DATA_ROOT"):
        regions.shared_data_root()
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_DATA_ROOT", str(tmp_path))
    assert regions.shared_data_root() == Path(tmp_path)
    assert regions.shared_drive_path("z:\\FF_UM\\rig") == Path(tmp_path) / "FF_UM\\rig"
