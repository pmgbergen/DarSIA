"""Stored original photographs of RAW runs (package D2b).

The producer copies a RAW photographic review's lossless original inside the
native content box, under the receipt contract of saved JPEG originals, so the
existing saved-product readers take it unchanged. Synthetic review root only.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_original_photo as photo
from darsia.fluidflower import storage_saved_export as export
from .test_raw_photographic_review import PHASE_HW, _write, inputs  # noqa: F401

PRODUCER = "f" * 64


@pytest.fixture
def state(tmp_path, inputs):
    manifest_path = _write(tmp_path, inputs)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = manifest["frames"][1]
    calls = []

    def resolve(bound):
        calls.append(Path(bound["path"]).name)
        return Path(bound["path"])

    args = dict(task=dict(run="ac52", precision="float32"),
                frame_spec=dict(case_id="ac52_f2", frame_index=1,
                                source=dict(source_sha256=frame["acquisition"]["sha256"])),
                raw_review=photo._asset(manifest_path), geometry_producer_sha256=PRODUCER)
    return dict(args=args, calls=calls, resolve=resolve, frame=frame, folder=manifest_path.parent)


def test_raw_original_is_copied_inside_the_native_box_with_raw_identities(tmp_path, state):
    result = photo.publish_raw_original_png(tmp_path / "out/photo.png", **state["args"],
                                            resolve_asset=state["resolve"])
    assert state["calls"] == ["manifest.json", "original.png"]  # No RAW, record or model is read.
    rgba = np.asarray(Image.open(tmp_path / "out/photo.png"))
    original = np.asarray(Image.open(state["folder"] / "frames/ac52_f2/original.png"))
    inside = np.zeros(PHASE_HW, dtype=bool)
    inside[3:13, 5:21] = True
    assert rgba.shape == (*PHASE_HW, 4)
    assert np.array_equal(rgba[..., 3], inside.astype(np.uint8) * 255)
    assert np.array_equal(rgba[inside, :3], original[inside]) and not rgba[~inside, :3].any()
    receipt = json.loads((tmp_path / "out/photo.png.receipt.json").read_text(encoding="utf-8"))
    frame = state["frame"]
    assert receipt["format"] == "darsia-stored-registered-original-photo-v1" and receipt["source_kind"] == "raw"
    assert receipt["camera_source"]["sha256"] == frame["acquisition"]["sha256"]
    assert receipt["native_record"] == frame["native_record"] and receipt["raw_audit"] == frame["audit"]
    assert receipt["profile_id"] == "late" and receipt["raw_review"] == state["args"]["raw_review"]
    assert receipt["stored_photograph_ROI_xywh"] == [5, 3, 16, 10] and receipt["photograph"] == result["photograph"]


def test_existing_saved_readers_take_the_raw_original_unchanged(tmp_path, state):
    result = photo.publish_raw_original_png(tmp_path / "photo.png", **state["args"], resolve_asset=state["resolve"])
    stored = dict(run="ac52", case_id="ac52_f2", precision="float32",
                  source=state["args"]["frame_spec"]["source"], grid=dict(native_shape=list(PHASE_HW)))
    rgba, evidence = export._photograph(result["receipt"], stored, resolve_asset=state["resolve"],
                                        geometry_producer_sha256=PRODUCER)
    assert rgba.shape == (*PHASE_HW, 4) and evidence["photograph"] == result["photograph"]


def test_raw_original_needs_the_same_acquisition_and_a_fresh_destination(tmp_path, state):
    other = dict(state["args"], frame_spec=dict(state["args"]["frame_spec"], source=dict(source_sha256="0" * 64)))
    with pytest.raises(ValueError, match="camera RAW"):
        photo.publish_raw_original_png(tmp_path / "a.png", **other, resolve_asset=state["resolve"])
    tampered = dict(state["args"], raw_review=dict(state["args"]["raw_review"], sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA"):
        photo.publish_raw_original_png(tmp_path / "b.png", **tampered, resolve_asset=state["resolve"])
    (tmp_path / "c.png").write_bytes(b"")
    with pytest.raises(FileExistsError):
        photo.publish_raw_original_png(tmp_path / "c.png", **state["args"], resolve_asset=state["resolve"])
    assert not (tmp_path / "a.png").exists() and not (tmp_path / "b.png").exists()
