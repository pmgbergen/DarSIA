"""Tiny saved PNG/receipt contracts; never scientific products or campaign inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_saved_movies as movies

GEOMETRY = "b" * 64
FPS = 3.5


def _asset(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _json(path, value):
    path.write_bytes((json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode())
    return _asset(path)


def _seal(fixture):
    items = fixture["sequence_value"]["frames"]
    for item, receipt, original, photo_path in zip(items, fixture["receipts"], fixture["originals"], fixture["photo_paths"]):
        receipt["photograph"]["receipt"] = _json(photo_path, original)
        receipt["frame"] = _asset(Path(item["frame"]["path"]))
        item["frame"] = deepcopy(receipt["frame"])
        item["receipt"] = _json(Path(item["receipt"]["path"]), receipt)
    fixture["sequence"] = _json(fixture["root"] / "film_sequence.json", fixture["sequence_value"])
    fixture["export_value"]["film_frames"] = dict(receipt=deepcopy(fixture["sequence"]), sequence=deepcopy(fixture["sequence_value"]))
    fixture["export"] = _json(fixture["root"] / "export.json", fixture["export_value"])
    return fixture


def _fixture(tmp_path, kind="native-phase-n1"):
    root = tmp_path / "saved"
    root.mkdir()
    cases = ["ac52_f1", "ac52_f2", "ac52_f3"]
    items, receipts, originals, photo_paths = [], [], [], []
    for number, case in enumerate(cases):
        rgb = np.zeros((16, 24, 3), np.uint8)
        rgb[..., number] = 230
        rgb[1:5, 2:8] = [31, 41, 59]  # asymmetric fixed feature
        frame = root / (case + ".png")
        Image.fromarray(rgb).save(frame)
        source = dict(source_sha256=hashlib.sha256(case.encode()).hexdigest(), frame_index=number + 1)
        photograph = dict(path=str(root / ("DO_NOT_READ_original_" + case + ".png")), sha256="a" * 64, size_bytes=123)
        original = dict(format="darsia-stored-registered-original-photo-v1", complete=True,
            run="ac52", case_id=case, precision="float32", source=deepcopy(source),
            camera_source=dict(sha256=source["source_sha256"]), geometry_producer_sha256=GEOMETRY,
            photograph=deepcopy(photograph), alpha_meaning=movies._ALPHA,
            alpha_inside_ROI=255, alpha_outside_ROI=0, display_dtype="uint8",
            colour_light_or_v5_changes=False, camera_RAW_or_correction_chain_executed=False,
            current_phase_products_changed=False)
        photo_path = root / (case + ".original.receipt.json")
        receipt = dict(format="darsia-saved-original-left-frame-v1", complete=True,
            run="ac52", case_id=case, precision="float32", source=deepcopy(source),
            field="concentration", fixed_limits=[0., 4.], width_px=24, height_px=16,
            original_photo_RGB_or_alpha_changed=False, light_correction_or_registration_executed=False,
            field_array_transposed_or_regridded=False,
            storage_receipt=dict(path=str(root / ("DO_NOT_READ_storage_" + case + ".json")), sha256="c" * 64, size_bytes=101),
            photograph=dict(receipt={}, photograph=photograph, geometry_producer_sha256=GEOMETRY,
                            alpha_meaning=movies._ALPHA, loaded_assets=[]), frame=_asset(frame))
        items.append(dict(case_id=case, frame=_asset(frame), receipt=dict(path=str(root / (case + ".receipt.json"))), width_px=24, height_px=16))
        receipts.append(receipt); originals.append(original); photo_paths.append(photo_path)
    sequence = dict(format="darsia-saved-original-left-frame-sequence-v1", state="frames_ready_encoding_pending",
        fps=FPS, timing="one frame per acquisition at explicit fps; acquisition intervals not inferred",
        field="concentration", fixed_limits=[0., 4.], ordered_case_ids=cases, frames=items,
        blocked={}, original_left_frame_sequence_ready=True, movie_encoded=False,
        movie_encoding="pending: no qualified explicit-fps encoder connected")
    export = dict(format="darsia-saved-storage-export-v1", complete=True, state="requested_exports_complete",
        run="ac52", precision="float32", ordered_case_ids=cases, phase2_N1_contract_admitted=False)
    if kind == "n1-mass-only":
        sequence["producer_kind"] = kind
        export.update(format="darsia-saved-n1-storage-export-v1", producer_kind=kind, phase2_N1_contract_admitted=True)
    return _seal(dict(root=root, sequence_value=sequence, export_value=export, receipts=receipts,
                      originals=originals, photo_paths=photo_paths))


def _call(fixture, destination, **options):
    return movies.encode_saved_sequence(fixture["export"], fixture["sequence"], destination,
        fps=options.pop("fps", FPS), geometry_producer_sha256=options.pop("geometry", GEOMETRY), **options)


class _Capture:
    def __init__(self, *, count=3, reported_count=3, fps=FPS, shape=(16, 24), opened=True):
        self.count, self.reported_count, self.fps, self.shape = count, reported_count, fps, shape
        self.opened, self.number, self.released = opened, 0, False
    def isOpened(self):
        return self.opened
    def get(self, prop):
        return self.reported_count if prop == cv2.CAP_PROP_FRAME_COUNT else self.fps
    def read(self):
        if self.number >= self.count:
            return False, None
        self.number += 1
        return True, np.zeros((*self.shape, 3), np.uint8)
    def release(self):
        self.released = True


def _fake_encoder(monkeypatch, capture=None, action=None):
    called = []
    cap = capture or _Capture()
    def encode(paths, target, fps):
        called.append((list(paths), target, fps))
        target.write_bytes(b"owned synthetic encoded candidate")
        if action:
            action()
        return "mp4v"
    monkeypatch.setattr(movies, "_write_movie", encode)
    monkeypatch.setattr(movies.cv2, "VideoCapture", lambda _path: cap)
    return called, cap


@pytest.mark.parametrize("kind", ["native-phase-n1", "n1-mass-only"])
def test_real_tiny_png_movie_explicit_fps_full_decode_and_order(tmp_path, kind):
    fixture = _fixture(tmp_path, kind)
    destination = tmp_path / "movie"
    before = {path: _asset(path) for path in fixture["root"].iterdir() if path.is_file()}
    try:
        result = _call(fixture, destination)
    except RuntimeError as error:
        if str(error) == "No compatible MP4 video encoder is available":
            assert (destination / "STOP.json").is_file()
            pytest.skip("No original avc1/mp4v encoder available; feature codec remains unqualified")
        raise
    assert result["complete"] and result["decoded"]["decoded_frames"] == 3
    assert result["fps_requested"] == result["decoded"]["reported_fps"] == FPS
    assert result["precision"] == "float32" and result["producer_kind"] == kind
    assert result["ordered_case_ids"] == ["ac52_f1", "ac52_f2", "ac52_f3"]
    assert result["codec"] in {"avc1", "mp4v"}
    assert result["movie"] == _asset(destination / "movie.mp4")
    assert result["receipt"] == _asset(destination / "completed.json")
    assert all(_asset(path) == bound for path, bound in before.items())
    assert not list(fixture["root"].glob("DO_NOT_READ*"))
    video = cv2.VideoCapture(result["movie"]["path"])
    dominant = []
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            dominant.append(int(np.argmax(frame.mean(axis=(0, 1)))))
    finally:
        video.release()
    assert dominant == [2, 1, 0]  # OpenCV BGR order; no pixel-equality claim for lossy MP4.


@pytest.mark.parametrize("problem", ["order", "duplicate", "missing_frame", "pending_photo", "pending_before",
    "precision", "source", "light_corrected", "photo_geometry", "limits", "unknown_kind", "not_png", "png_dimensions"])
def test_bad_saved_contract_refused_before_encoder(tmp_path, monkeypatch, problem):
    f = _fixture(tmp_path)
    seq, exp = f["sequence_value"], f["export_value"]
    if problem == "order":
        seq["frames"].reverse()
    elif problem == "duplicate":
        seq["ordered_case_ids"][1] = seq["ordered_case_ids"][0]
    elif problem == "missing_frame":
        seq["frames"].pop()
    elif problem in {"pending_photo", "pending_before"}:
        seq.update(state="not_film_ready", original_left_frame_sequence_ready=False,
                   blocked={"ac52_f2": dict(reason="saved_before_rules_pending" if problem == "pending_before" else "indexed_registered_pre_light_photo_not_bound")})
    elif problem == "precision":
        f["receipts"][1]["precision"] = "float64"
    elif problem == "source":
        f["originals"][1]["source"]["source_sha256"] = "f" * 64
    elif problem == "light_corrected":
        f["originals"][1]["colour_light_or_v5_changes"] = True
    elif problem == "photo_geometry":
        f["originals"][1]["geometry_producer_sha256"] = "d" * 64
    elif problem == "limits":
        f["receipts"][1]["fixed_limits"] = [0., 8.]
    elif problem == "unknown_kind":
        exp["producer_kind"] = "invented"
    elif problem == "not_png":
        Path(seq["frames"][1]["frame"]["path"]).write_bytes(b"not a PNG")
    elif problem == "png_dimensions":
        Image.fromarray(np.zeros((18, 24, 3), np.uint8)).save(seq["frames"][1]["frame"]["path"])
    _seal(f)
    called, _cap = _fake_encoder(monkeypatch)
    with pytest.raises(ValueError):
        _call(f, tmp_path / "movie")
    assert called == [] and not (tmp_path / "movie").exists()


def test_changed_png_SHA_refused(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    frame = Path(f["sequence_value"]["frames"][1]["frame"]["path"])
    frame.write_bytes(frame.read_bytes() + b"changed")
    called, _cap = _fake_encoder(monkeypatch)
    with pytest.raises(ValueError, match="SHA/size/version"):
        _call(f, tmp_path / "movie")
    assert called == []


@pytest.mark.parametrize("fps", [True, 0, -1, float("nan"), float("inf"), 5.])
def test_explicit_fps_required_and_must_match_sequence(tmp_path, monkeypatch, fps):
    f = _fixture(tmp_path)
    called, _cap = _fake_encoder(monkeypatch)
    with pytest.raises(ValueError):
        _call(f, tmp_path / "movie", fps=fps)
    assert called == []


def test_existing_directory_refused_without_overwrite(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    destination = tmp_path / "movie"
    destination.mkdir()
    sentinel = destination / "movie.mp4"
    sentinel.write_bytes(b"preserve existing")
    called, _cap = _fake_encoder(monkeypatch)
    with pytest.raises(FileExistsError):
        _call(f, destination)
    assert sentinel.read_bytes() == b"preserve existing" and called == []


@pytest.mark.parametrize("bad", ["reported_count", "early_EOF", "extra_frame", "fps", "dimensions", "not_opened"])
def test_full_decoding_guards_and_released_capture_preserve_partial(tmp_path, monkeypatch, bad):
    f = _fixture(tmp_path)
    options = {"reported_count": dict(reported_count=2), "early_EOF": dict(count=2),
               "extra_frame": dict(count=4), "fps": dict(fps=4.),
               "dimensions": dict(shape=(16, 22)), "not_opened": dict(opened=False)}
    cap = _Capture(**options[bad])
    called, cap = _fake_encoder(monkeypatch, cap)
    with pytest.raises(ValueError):
        _call(f, tmp_path / "movie")
    assert len(called) == 1 and cap.released
    assert (tmp_path / "movie/movie.mp4").read_bytes() == b"owned synthetic encoded candidate"
    stop = json.loads((tmp_path / "movie/STOP.json").read_bytes())
    assert stop["partials_preserved"] and not stop["automatic_retry"]
    assert not (tmp_path / "movie/completed.json").exists()


def test_encoder_failure_preserves_partial_without_second_attempt(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    calls = []
    def fail(paths, path, fps):
        calls.append((paths, path, fps))
        path.write_bytes(b"owned unfinished MP4")
        raise RuntimeError("synthetic original encoder failure")
    monkeypatch.setattr(movies, "_write_movie", fail)
    with pytest.raises(RuntimeError, match="synthetic original encoder failure"):
        _call(f, tmp_path / "movie")
    assert len(calls) == 1 and (tmp_path / "movie/movie.mp4").read_bytes() == b"owned unfinished MP4"
    stop = json.loads((tmp_path / "movie/STOP.json").read_bytes())
    assert stop["error"]["type"] == "RuntimeError" and stop["error"]["frames"]
    assert not (tmp_path / "movie/completed.json").exists()


def test_input_change_during_encoding_never_completes(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    frame = Path(f["sequence_value"]["frames"][0]["frame"]["path"])
    calls, _cap = _fake_encoder(monkeypatch, action=lambda: frame.write_bytes(b"changed owned fixture"))
    with pytest.raises(ValueError, match="input changed during encoding"):
        _call(f, tmp_path / "movie")
    assert len(calls) == 1 and (tmp_path / "movie/STOP.json").exists()
    assert not (tmp_path / "movie/completed.json").exists()
