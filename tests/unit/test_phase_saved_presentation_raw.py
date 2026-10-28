"""Saved presentations of RAW runs (package D2b).

A RAW photographic review gives the deferred presentation its timeline and
the prediction video its natural photographs; historical roots are unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import phase_saved_presentation as presentation
from darsia.fluidflower.native.config import configure_parameters, reset_parameters
from darsia.fluidflower.project import sha256_file
from .test_raw_photographic_review import _write, inputs  # noqa: F401

FORMAT = "synthetic-deferred-presentation"


@pytest.fixture
def root(tmp_path, inputs):
    _write(tmp_path, inputs)
    token = configure_parameters({"saved_review_metadata": {"presentation": {"deferred_format": FORMAT}}})
    yield tmp_path / "review"
    reset_parameters(token)


def _validate(tmp_path, root, elapsed):
    manifest = root / "native_frames/ac52/manifest.json"
    csv = tmp_path / "per_frame.csv"
    csv.write_text("elapsed_h\n" + "".join(f"{value}\n" for value in elapsed), encoding="utf-8")
    summary = dict(presentation_status="deferred", calibration_review_movie=dict(calibration_frame_count=2),
                   deferred_presentation=dict(
                       format=FORMAT, run="ac52", candidate_id="c1", params_sha256="p" * 64,
                       native_manifest=str(manifest), native_manifest_sha256=sha256_file(manifest),
                       phase_map_output_dir=str(tmp_path / "maps"), frame_count=2))
    return presentation.validate_deferred_presentation(
        summary, run="ac52", candidate_id="c1", params_sha256="p" * 64, review_output_dir=root,
        phase_map_output_dir=tmp_path / "maps", per_frame_csv=csv)


def test_raw_timeline_and_video_photographs_come_from_the_raw_review(tmp_path, root):
    _descriptor, native, times = _validate(tmp_path, root, (0.0, 0.5))
    assert times == [0.0, 0.5] and native["format"] == "darsia-raw-photographic-review-v1"
    candidate = SimpleNamespace(review_output_dir=root, run="ac52")
    expected = root / "native_frames/ac52/frames/ac52_f2/natural.png"
    assert Path(presentation._presentation_photo(candidate, native, 1)).resolve() == expected.resolve()
    with pytest.raises(ValueError, match="timeline"):
        _validate(tmp_path, root, (0.0, 0.7))


def test_historical_roots_keep_their_rows_and_frame_images(tmp_path):
    folder = tmp_path / "native_frames/ac52"
    folder.mkdir(parents=True)
    candidate = SimpleNamespace(review_output_dir=tmp_path, run="ac52")
    assert presentation.timeline_rows({"frame_sources": [{"elapsed_h": 1}]}, "ac52") == [{"elapsed_h": 1}]
    assert presentation._presentation_photo(candidate, {}, 3) == folder / "frame_003.jpg"
    (folder / "frame_003.png").write_bytes(b"")
    assert presentation._presentation_photo(candidate, {}, 3) == folder / "frame_003.png"
