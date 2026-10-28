"""Original direct RAW area sampler used by saved normalisation fits.

The operation order follows ``ffac_raw_extended_reference.py`` SHA-256
8553de98f0c6a999b1a3add780892e815cd17bc513728f8b698ff3cbd8b4f097.
An area mean in this producer uses float32 reduction; it must not be replaced
by the distinct full-native frame-light reduction.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from . import registration
from .io import sha256_file


def registered_edge_aware_raw_area_grid(
    arw_path, camera_jpeg_path, review_native_jpeg_path, *, expected_source_hashes,
    source_certificate_path, source_certificate_sha256, h_target_to_review_native,
    origin_cell_xy, shape_cells_hw, source_id, profile_id, content_box,
    parameters, area_parameters,
):
    """Decode exact RAW sources and retain the saved direct sampling arithmetic."""
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("RAW area rendering requires CPU rawpy/LibRaw") from error
    registration.check_demosaic_order(parameters=parameters)
    sensor = parameters["common"]["sensor"]
    arw, camera, review = map(Path, (arw_path, camera_jpeg_path, review_native_jpeg_path))
    if arw.stem.casefold() != camera.stem.casefold():
        raise ValueError("Original ARW/camera JPEG basename mismatch")
    for key, path in (("arw_sha256", arw), ("camera_jpeg_sha256", camera),
                      ("review_native_jpeg_sha256", review)):
        expected = expected_source_hashes.get(key)
        if not isinstance(expected, str) or sha256_file(path) != expected.lower():
            raise ValueError(f"Missing or changed original source: {key}")
    cert_path = Path(source_certificate_path)
    if sha256_file(cert_path) != source_certificate_sha256.lower():
        raise ValueError("Original RAW/review geometry certificate changed")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    if cert["sources"] != dict(expected_source_hashes):
        raise ValueError("Original RAW/review certificate source mismatch")
    if list(cert["content_box_xywh"]) != list(content_box):
        raise ValueError("Original review content box changed")
    h_review_camera = np.asarray(cert["review_to_camera_h"], np.float64)
    h_target_review = np.asarray(h_target_to_review_native, np.float64)
    for h in (h_review_camera, h_target_review):
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < area_parameters["homography_min_determinant"]:
            raise ValueError("Invalid source-bound target/review homography")
    h_target_camera = h_review_camera @ h_target_review
    h_target_camera /= h_target_camera[2, 2]
    if cert["run"].casefold() != source_id.casefold() or cert["profile"] != profile_id:
        raise ValueError("Original lens profile/run mismatch")
    distortion, ca_red, ca_blue = registration.read_lens_tables(arw, parameters=parameters["common"]["lens"])
    if (list(distortion) != cert["sony_distortion_table"] or
            list(ca_red) != cert["sony_ca_red_table"] or
            list(ca_blue) != cert["sony_ca_blue_table"]):
        raise ValueError("Original Sony lens/CA tags changed")
    with rawpy.imread(str(arw)) as raw:
        if (raw.raw_image_visible.shape != (sensor["raw_wh"][1], sensor["raw_wh"][0]) or
                not np.array_equal(raw.raw_pattern, sensor["cfa_pattern"]) or
                raw.color_desc != sensor["color_description"].encode("ascii") or
                tuple(int(x) for x in raw.black_level_per_channel) != tuple(sensor["black_levels"])):
            raise ValueError("Unexpected original RAW sensor/CFA/black level")
        bayer = np.maximum(raw.raw_image_visible.astype(np.int32) - sensor["black_level"], 0).astype(np.uint16)
    demosaiced = cv2.demosaicing(bayer, getattr(cv2, parameters["edge_aware"]["demosaic_code"]))
    channels = [demosaiced[..., channel].astype(np.float32) for channel in range(3)]
    return _sample_area_grid(channels, h_target_review, h_target_camera, distortion,
        ca_red, ca_blue, origin_cell_xy, shape_cells_hw, profile_id,
        parameters=parameters, area_parameters=area_parameters,
        source_certificate_path=cert_path, source_certificate_sha256=source_certificate_sha256,
        expected_source_hashes=expected_source_hashes)


def _sample_area_grid(channels, h_target_review, h_target_camera, distortion,
                      ca_red, ca_blue, origin_cell_xy, shape_cells_hw, profile_id, *,
                      parameters, area_parameters, source_certificate_path,
                      source_certificate_sha256, expected_source_hashes):
    """Keep direct sampling separable for small original-producer parity fixtures."""
    x_cell, y_cell = map(int, origin_cell_xy)
    rows, cols = map(int, shape_cells_hw)
    if not 1 <= rows <= area_parameters["maximum_rows"] or not 1 <= cols <= area_parameters["maximum_columns"]:
        raise ValueError("Unexpected target coarse-grid shape")
    out = np.empty((3, rows, cols), np.float32)
    finite_cell_count = 0
    params = parameters["common"]["profiles"][profile_id]["coefficients"]
    scale = area_parameters["cell_scale"]
    for cell_y0 in range(0, rows, area_parameters["tile_cell_rows"]):
        cell_y1 = min(cell_y0 + area_parameters["tile_cell_rows"], rows)
        yy, xx = np.mgrid[scale*(y_cell+cell_y0):scale*(y_cell+cell_y1),
                          scale*x_cell:scale*(x_cell+cols)].astype(np.float32)
        tile = np.empty((3, scale*(cell_y1-cell_y0), scale*cols), np.float32)
        for channel, ca in enumerate((ca_red, None, ca_blue)):
            sensor_x, sensor_y = registration.raw_coordinates(xx, yy, h_target_camera,
                params, distortion, ca, parameters=parameters["common"]["lens"])
            tile[channel] = cv2.remap(
                channels[channel], sensor_x.astype(np.float32),
                sensor_y.astype(np.float32), cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        area = tile.reshape(3, cell_y1-cell_y0, scale, cols, scale).mean((2, 4))
        out[:, cell_y0:cell_y1] = np.log(np.maximum(area, area_parameters["signal_floor"])).astype(np.float32)
        finite_cell_count += int(np.isfinite(out[:, cell_y0:cell_y1]).all(axis=0).sum())
    result_cert = dict(
        contract=area_parameters["contract"],
        original_source_certificate_path=str(source_certificate_path),
        original_source_certificate_sha256=source_certificate_sha256.lower(),
        original_source_sha256=dict(expected_source_hashes),
        target_to_review_native_h=h_target_review.tolist(),
        target_to_camera_jpeg_h=h_target_camera.tolist(),
        target_cell_origin_xy=[x_cell, y_cell], target_grid_shape_chw=list(out.shape),
        finite_rgb_cell_count=finite_cell_count, total_cells=rows*cols,
        aggregation=area_parameters["aggregation"],
        colour_basis=area_parameters["colour_basis"],
        out_of_sensor="NaN; no nearest/reflect fill",
        model_training_eligible=False,
    )
    return out, result_cert
