"""Two real tiny exporter-to-movie ABIs; original341 helpers are validation-only."""
import hashlib
import json
from pathlib import Path

import cv2
import pytest

from darsia.fluidflower import storage_saved_export as native_export
from darsia.fluidflower import storage_n1_saved_export as n1_export
from darsia.fluidflower import storage_saved_movies as movies
from . import test_storage_saved_export as native_helper
from . import test_storage_n1_saved_products as n1_helper
from .test_storage_n1_saved_products import tiny_config, phase1_data


def _asset(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _original_photo_assets(receipts):
    """Only the already-created tiny PNGs/receipts, never camera source links."""
    assets = []
    for bound in receipts:
        assets.append(_asset(bound["path"]))
        doc = json.loads(Path(bound["path"]).read_bytes())
        assets.append(_asset(doc["photograph"]["path"]))
    return assets


def _finish_export(value, destination, *, fps, geometry, monkeypatch):
    sequence = value["film_frames"]["sequence"]
    assert value["complete"] and sequence["original_left_frame_sequence_ready"]
    assert sequence["fps"] == fps and sequence["blocked"] == {}
    assert len(sequence["frames"]) == 2
    assert [item["case_id"] for item in sequence["frames"]] == value["ordered_case_ids"]
    bound_sequence = value["film_frames"]["receipt"]
    assert json.loads(Path(bound_sequence["path"]).read_bytes()) == sequence
    # Do not repair/reseal/projection-copy any original exporter receipt.
    before = [_asset(value["receipt"]["path"]), _asset(bound_sequence["path"])]
    for item in sequence["frames"]:
        before.extend((_asset(item["frame"]["path"]), _asset(item["receipt"]["path"])))
        receipt = json.loads(Path(item["receipt"]["path"]).read_bytes())
        before.extend(_original_photo_assets([receipt["photograph"]["receipt"]]))
    real_encoder, calls = movies._write_movie, []
    def observe(paths, path, actual_fps):
        calls.append(dict(frames=[_asset(p) for p in paths], target=str(path), fps=actual_fps))
        return real_encoder(paths, path, actual_fps)
    monkeypatch.setattr(movies, "_write_movie", observe)
    try:
        result = movies.encode_saved_sequence(value["receipt"], bound_sequence, destination,
            fps=fps, geometry_producer_sha256=geometry)
    except RuntimeError as error:
        if str(error) == "No compatible MP4 video encoder is available":
            assert (destination / "STOP.json").is_file()
            pytest.skip("Original avc1/mp4v unavailable; exporter-to-movie encoding not qualified")
        raise
    assert len(calls) == 1 and calls[0]["fps"] == fps
    assert calls[0]["frames"] == [item["frame"] for item in sequence["frames"]]
    assert calls[0]["target"] == str(destination / "movie.mp4")
    assert result["complete"] and result["precision"] == value["precision"] == "float32"
    assert result["ordered_case_ids"] == value["ordered_case_ids"]
    assert result["fps_requested"] == result["decoded"]["reported_fps"] == fps
    first = sequence["frames"][0]
    assert (result["decoded"]["height_px"], result["decoded"]["width_px"]) == (first["height_px"], first["width_px"])
    assert result["decoded"]["decoded_frames"] == len(sequence["frames"]) == 2
    assert result["movie"] == _asset(destination / "movie.mp4")
    assert all(_asset(asset["path"]) == asset for asset in before)
    # Independent complete sequential decoding; no scientific/MP4 pixel equality claim.
    video = cv2.VideoCapture(result["movie"]["path"])
    shapes = []
    try:
        assert video.get(cv2.CAP_PROP_FPS) == fps
        while True:
            ok, frame = video.read()
            if not ok:
                break
            shapes.append(frame.shape)
    finally:
        video.release()
    assert shapes == [(first["height_px"], first["width_px"], 3)] * 2
    return result


def test_original_native_phase_exporter_to_original_encoder_real_tiny_ABI(tmp_path, monkeypatch):
    index, photographs = native_helper._two_frame_index(tmp_path / "inputs", with_photos=True)
    originals = _original_photo_assets(photographs.values())
    value = native_export.export_indexed_storage(index, tmp_path / "export", current_machine_id="Main",
        film_field="saturation", fps=6., geometry_producer_sha256="f" * 64, photographs=photographs)
    result = _finish_export(value, tmp_path / "movie", fps=6., geometry="f" * 64, monkeypatch=monkeypatch)
    assert result["producer_kind"] == "native-phase-n1"
    assert all(_asset(asset["path"]) == asset for asset in originals)


def test_genuine_N1_exporter_to_original_encoder_real_tiny_ABI(tiny_config, phase1_data, tmp_path, monkeypatch):
    config = n1_helper._inputs(tmp_path / "inputs", tiny_config, phase1_data, before=True, photo=True)
    originals = _original_photo_assets(item["photograph"]["receipt"] for item in config["phase1"]["frames"].values())
    completion = n1_helper._run(tmp_path / "run", config)
    index = n1_helper._index(tmp_path / "publication", completion)
    value = n1_export.export_indexed_n1_storage(index, tmp_path / "export", current_machine_id="Main",
        film_field="saturation", fps=7.5, geometry_producer_sha256=n1_helper.GEOMETRY)
    result = _finish_export(value, tmp_path / "movie", fps=7.5, geometry=n1_helper.GEOMETRY, monkeypatch=monkeypatch)
    assert result["producer_kind"] == "n1-mass-only"
    assert any(asset.get("kind") == "operational_receipt" for asset in result["loaded_assets"])
    assert all(item["storage_receipt"]["kind"] == "operational_receipt" for item in result["original_left_provenance"])
    assert all(_asset(asset["path"]) == asset for asset in originals)
