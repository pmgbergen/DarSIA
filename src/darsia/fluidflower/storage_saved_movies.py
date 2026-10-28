"""Optional movie encoding of SHA-bound saved original-left PNG sequences only."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

import cv2

from .post_materialization import _write_movie
from .storage_original_photo import _ALPHA


def _asset(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def _descriptor(bound, *, receipt_kind=False):
    keys = {"path", "sha256", "size_bytes"}
    if receipt_kind and isinstance(bound, dict) and bound.get("kind") == "operational_receipt":
        keys.add("kind")
    if (not isinstance(bound, dict) or set(bound) != keys
            or not isinstance(bound["path"], str) or not Path(bound["path"]).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", str(bound["sha256"])) is None
            or type(bound["size_bytes"]) is not int or bound["size_bytes"] < 1):
        raise ValueError("Saved movie input requires an exact finite file descriptor")
    return Path(bound["path"])


def _read(bound, loaded, *, receipt_kind=False):
    path = _descriptor(bound, receipt_kind=receipt_kind)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (len(data) != bound["size_bytes"] or hashlib.sha256(data).hexdigest() != bound["sha256"]
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ValueError("Saved movie input SHA/size/version differs: " + str(path))
    loaded.append(deepcopy(bound))
    return data


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate saved movie JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite saved movie JSON constant: " + value)
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Saved movie metadata must be a JSON object")
    return value


def _write_json(path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False,
                                 indent=2, sort_keys=True) + "\n").encode("utf8"))
        stream.flush()
        os.fsync(stream.fileno())
    return _asset(path)


def _exception(error, ancestors=()):
    """Preserve failure locations/chains without source-line filesystem reads."""
    if id(error) in ancestors:
        return dict(exception_cycle=True)
    seen = ancestors + (id(error),)
    frames, tb = [], error.__traceback__
    while tb is not None:
        frames.append(dict(filename=tb.tb_frame.f_code.co_filename,
                           function=tb.tb_frame.f_code.co_name, lineno=tb.tb_lineno))
        tb = tb.tb_next
    result = dict(type=type(error).__qualname__, message=str(error), frames=frames,
                  notes=list(getattr(error, "__notes__", ())),
                  suppress_context=error.__suppress_context__)
    result["cause"] = None if error.__cause__ is None else _exception(error.__cause__, seen)
    result["context"] = None if error.__context__ is None else _exception(error.__context__, seen)
    if isinstance(error, BaseExceptionGroup):
        result["group_members"] = [_exception(item, seen) for item in error.exceptions]
    return result


def _frame(item, *, case, run, precision, field, limits, geometry_producer_sha256, loaded):
    receipt = _json(_read(item["receipt"], loaded))
    photo = receipt.get("photograph", {})
    if (item.get("case_id") != case or receipt.get("format") != "darsia-saved-original-left-frame-v1"
            or receipt.get("complete") is not True or receipt.get("case_id") != case
            or receipt.get("run") != run or receipt.get("precision") != precision
            or receipt.get("field") != field or receipt.get("fixed_limits") != limits
            or receipt.get("frame") != item["frame"]
            or receipt.get("original_photo_RGB_or_alpha_changed") is not False
            or receipt.get("light_correction_or_registration_executed") is not False
            or receipt.get("field_array_transposed_or_regridded") is not False
            or photo.get("geometry_producer_sha256") != geometry_producer_sha256
            or photo.get("alpha_meaning") != _ALPHA):
        raise ValueError("Saved original-left movie frame identity/provenance differs: " + case)
    _descriptor(receipt["storage_receipt"], receipt_kind=True)  # Metadata provenance only; no stored scientific payload read.
    _descriptor(photo["photograph"])
    original = _json(_read(photo["receipt"], loaded, receipt_kind=True))
    if (original.get("format") != "darsia-stored-registered-original-photo-v1"
            or original.get("complete") is not True or original.get("run") != run
            or original.get("case_id") != case or original.get("precision") != precision
            or original.get("source") != receipt.get("source")
            or original.get("camera_source", {}).get("sha256") != receipt["source"]["source_sha256"]
            or original.get("photograph") != photo.get("photograph")
            or original.get("geometry_producer_sha256") != geometry_producer_sha256
            or original.get("alpha_meaning") != _ALPHA
            or original.get("alpha_inside_ROI") != 255 or original.get("alpha_outside_ROI") != 0
            or original.get("display_dtype") != "uint8"
            or original.get("colour_light_or_v5_changes") is not False
            or original.get("camera_RAW_or_correction_chain_executed") is not False
            or original.get("current_phase_products_changed") is not False):
        raise ValueError("Saved photograph pre-light provenance differs: " + case)
    raw = _read(item["frame"], loaded)
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Movie input is not a saved PNG: " + case)
    image = cv2.imread(item["frame"]["path"], cv2.IMREAD_COLOR)
    shape = (item.get("height_px"), item.get("width_px"))
    if (any(type(value) is not int or value < 1 for value in shape)
            or image is None or image.shape != (*shape, 3)
            or receipt.get("height_px") != shape[0] or receipt.get("width_px") != shape[1]):
        raise ValueError("Saved movie PNG dimensions/decode differ: " + case)
    del image, raw
    return shape, dict(case_id=case, source=deepcopy(receipt["source"]),
        frame=deepcopy(item["frame"]), render_receipt=deepcopy(item["receipt"]),
        photograph=deepcopy(photo), storage_receipt=deepcopy(receipt.get("storage_receipt")))


def _decoded_movie(path, *, expected_count, shape, fps):
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError("Encoded saved movie could not be opened")
        reported_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        reported_fps = video.get(cv2.CAP_PROP_FPS)
        if (not math.isfinite(reported_count) or reported_count != expected_count
                or not math.isfinite(reported_fps) or reported_fps != fps):
            raise ValueError("Encoded saved movie frame count or explicit fps differs")
        decoded = 0
        while True:
            ok, image = video.read()
            if not ok:
                break
            if image is None or image.shape != (*shape, 3):
                raise ValueError("Encoded saved movie dimensions differ")
            decoded += 1
            if decoded > expected_count:
                raise ValueError("Encoded saved movie contains extra frames")
        if decoded != expected_count:
            raise ValueError("Encoded saved movie did not decode every expected frame")
        return dict(decoded_frames=decoded, reported_frames=int(reported_count),
                    reported_fps=reported_fps, height_px=shape[0], width_px=shape[1],
                    duration_s=decoded / reported_fps, all_frames_sequentially_decoded=True)
    finally:
        video.release()


def encode_saved_sequence(export, sequence, destination, *, fps, geometry_producer_sha256):
    """Encode one complete immutable sequence; no rendering, collect, or replay.

    Existing export/sequence receipts remain unchanged. The movie lives only in
    this fresh private directory until its own completion receipt is written.
    A failure preserves every partial and raises; it is never automatically retried.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Saved movies require one fresh output directory")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(fps) or fps <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(geometry_producer_sha256)) is None):
        raise ValueError("Saved movies require explicit positive fps and geometry producer SHA")
    loaded = []
    product, frames = _json(_read(export, loaded)), _json(_read(sequence, loaded))
    kind = "n1-mass-only" if product.get("producer_kind") == "n1-mass-only" else "native-phase-n1"
    expected_format = "darsia-saved-n1-storage-export-v1" if kind == "n1-mass-only" else "darsia-saved-storage-export-v1"
    ordered = product.get("ordered_case_ids")
    if (product.get("format") != expected_format or product.get("complete") is not True
            or product.get("state") != "requested_exports_complete"
            or product.get("phase2_N1_contract_admitted") is not (kind == "n1-mass-only")
            or product.get("film_frames", {}).get("receipt") != sequence
            or product.get("film_frames", {}).get("sequence") != frames
            or frames.get("format") != "darsia-saved-original-left-frame-sequence-v1"
            or frames.get("state") != "frames_ready_encoding_pending"
            or frames.get("original_left_frame_sequence_ready") is not True
            or frames.get("blocked") != {} or frames.get("movie_encoded") is not False
            or frames.get("fps") != fps or isinstance(frames.get("fps"), bool)
            or not isinstance(ordered, list) or not ordered
            or any(not isinstance(case, str) or not case for case in ordered)
            or len(set(ordered)) != len(ordered)
            or frames.get("ordered_case_ids") != ordered
            or not isinstance(frames.get("frames"), list) or len(frames["frames"]) != len(ordered)
            or product.get("precision") not in {"float32", "float64"}
            or not isinstance(product.get("run"), str)
            or (kind == "n1-mass-only" and frames.get("producer_kind") != kind)
            or (kind == "native-phase-n1" and ("producer_kind" in frames or "producer_kind" in product))):
        raise ValueError("Saved movie needs one complete exact ordered export/sequence")
    paths, provenance, shape = [], [], None
    for case, item in zip(ordered, frames["frames"]):
        actual_shape, evidence = _frame(item, case=case, run=product["run"],
            precision=product["precision"], field=frames["field"], limits=frames["fixed_limits"],
            geometry_producer_sha256=geometry_producer_sha256, loaded=loaded)
        if shape is not None and shape != actual_shape:
            raise ValueError("Saved movie frame dimensions changed within sequence")
        shape = actual_shape
        paths.append(Path(item["frame"]["path"]))
        provenance.append(evidence)
    destination.mkdir(parents=True, exist_ok=False)
    movie = destination / "movie.mp4"
    try:
        if movie.exists():
            raise FileExistsError("Saved movie target already exists")
        codec = _write_movie(paths, movie, float(fps))
        decoded = _decoded_movie(movie, expected_count=len(ordered), shape=shape, fps=float(fps))
        for bound in loaded:
            if _asset(bound["path"]) != {key: bound[key] for key in ("path", "sha256", "size_bytes")}:
                raise ValueError("Saved movie input changed during encoding: " + bound["path"])
        result = dict(format="darsia-saved-original-left-movie-v1", complete=True,
            run=product["run"], producer_kind=kind, precision=product["precision"],
            export=deepcopy(export), sequence=deepcopy(sequence), fps_requested=float(fps),
            ordered_case_ids=deepcopy(ordered), field=frames["field"], fixed_limits=deepcopy(frames["fixed_limits"]),
            timing="one rendered PNG per acquisition at explicit fps; no frame repetition or interpolation",
            geometry_producer_sha256=geometry_producer_sha256, original_left_provenance=provenance,
            loaded_assets=loaded, codec=codec, decoded=decoded, movie=_asset(movie),
            encoded_video_is_lossy_display=True, PNG_pixel_equality_not_claimed=True,
            no_render_collect_or_scientific_producer=True, input_artifacts_unchanged=True)
        receipt = _write_json(destination / "completed.json", result)
        return dict(result, receipt=receipt)
    except BaseException as error:
        _write_json(destination / "STOP.json", dict(utc=datetime.now(timezone.utc).isoformat(),
            complete=False, error=_exception(error), partials_preserved=True, automatic_retry=False,
            export=deepcopy(export), sequence=deepcopy(sequence)))
        raise
