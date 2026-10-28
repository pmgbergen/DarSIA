"""Studio's photographic readers on a RAW photographic review root (package D2b).

The same public functions that read historical V5/JPEG review roots resolve
the lossless PNG assets of a RAW root: availability checks without file I/O,
SHA-verified paths, canvas images, the camera-RAW original and the photo
support. Synthetic records with stubbed rendering; no real series is read.
"""
import json

import numpy as np
import pytest
from PIL import Image

from PySide6.QtWidgets import QApplication

from darsia.fluidflower import phase_photographic_assets as photos
from darsia.fluidflower import raw_photographic_review as review
from ..test_raw_photographic_review import NATIVE_HW, PHASE_HW, inputs  # noqa: F401


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def root(tmp_path, inputs):
    review.write_raw_review_root(
        tmp_path / "review", run="ac52", native_configuration=inputs["configuration"],
        raw_binding=inputs["binding"], reference=inputs["reference"], frames=inputs["frames"])
    return tmp_path / "review"


def test_availability_reads_raw_rows_without_file_io(root):
    manifest = review.read_raw_review_manifest(root, "ac52")
    for rows in (manifest, {"frame_sources": manifest["frames"]}):
        assert photos.display_modes_available(rows)
        assert photos.original_display_available(rows)
        assert photos.raw_detail_available(rows)
    assert photos.display_modes_available(manifest, reference=True)
    assert not photos.change_display_available(root, "ac52")  # No change photograph is written.


def test_paths_and_images_come_from_the_raw_root(root, app):
    path = photos.resolve_display_mode(root, "ac52", 1, "lift1")
    assert path == (root / "native_frames/ac52/frames/ac52_f2/lift1.png").resolve()
    assert photos.resolve_raw_detail(root, "ac52", 0).name == "raw_detail.png"
    for mode in ("natural", "lift2", "original", "raw_detail"):
        image = photos.load_display_image(root, "ac52", 0, mode=mode)
        assert (image.width(), image.height()) == (PHASE_HW[1], PHASE_HW[0])
    original = photos.load_original_display_rgb(root, "ac52", 0)
    stored = np.asarray(Image.open(root / "native_frames/ac52/frames/ac52_f1/original.png"))
    assert np.array_equal(original, stored)
    with pytest.raises(ValueError, match="change"):
        photos.resolve_change_display(root, "ac52", 0)


def test_photo_support_replaces_the_historical_sand_mask(root, inputs):
    native = photos.resolve_photo_sand_domain(root, "ac52", 0, canvas=False)
    canvas = photos.resolve_photo_sand_domain(root, "ac52", 0)
    assert native.shape == NATIVE_HW and canvas.shape == PHASE_HW
    assert np.array_equal(native, inputs["frames"][0]["photo_support"])


def test_a_broken_raw_manifest_is_never_read_as_a_historical_root(root):
    path = root / "native_frames/ac52/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(dict(manifest, frame_count=5)), encoding="utf-8")
    with pytest.raises(ValueError, match="RAW review manifest"):
        photos.resolve_display_mode(root, "ac52", 0, "natural")
