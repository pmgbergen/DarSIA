"""The trainer's RAW photographic review route (package D2c).

A synthetic RAW review run is staged and one reviewed region is extracted
through the RAW frame-geometry cohort, as the training entry routes do. Code
path only: no model is fitted. Frame k is shifted k pixels right of B.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from darsia.fluidflower import raw_photographic_review as review
from darsia.fluidflower.native.config import configure_parameters, reset_parameters
from darsia.fluidflower.phase_saved_assets import asset
from .test_raw_frame_context import CONFIGURATION, EXPORTS
from .test_raw_photographic_review import NATIVE_HW, PHASE_HW, inputs  # noqa: F401

# The training alignment and optical-feature parameters are configuration, never
# code defaults; this fixture holds a configured series' method parameters.
FEATURES = json.loads((Path(__file__).parents[1] / "fixtures/training_temporal_alignment_features.json")
                      .read_text(encoding="utf-8"))
TRAINING_CONFIGURATION = dict(CONFIGURATION, saved_review_metadata=dict(
    CONFIGURATION["saved_review_metadata"],
    training_frame_context=dict(CONFIGURATION["saved_review_metadata"]["training_frame_context"], exports=dict(
        EXPORTS, PHOTO_SAFE_CORE_CONTRACT=dict(value="transported_safe_B_sand_core_only", python_type="str"))),
    training_review_sampling=dict(policy="synthetic-review-sampling"),
    training_temporal_alignment=dict(features=FEATURES)))


@pytest.fixture
def trainer(tmp_path, inputs):
    names = ("darsia.fluidflower.phase_training.frame_context", "darsia.fluidflower.phase_training.trainer")
    loaded = {name: name in sys.modules for name in names}
    token = configure_parameters(TRAINING_CONFIGURATION)
    from darsia.fluidflower.phase_training import trainer
    certified = np.zeros(NATIVE_HW, dtype=bool)
    certified[1:-1, 1:-1] = True
    photo = tmp_path / "photo_sand.npz"
    np.savez(photo, photo_certified_sand_domain=certified, photo_boundary_uncertain_domain=~certified)
    # As on the real RAW run, the AC28 context lacks the physical/safe fields:
    # training asks only for the observable domain and the photo-sand masks.
    np.savez(tmp_path / "ac28_partial.npz", domain=certified, facies=certified.astype(np.uint8),
             permeability=np.ones(NATIVE_HW, np.float32), x_m=np.zeros(NATIVE_HW[1]), y_m=np.zeros(NATIVE_HW[0]))
    context = asset(tmp_path / "ac28_partial.npz")
    training_context = dict(
        reference_context_path=context["path"], reference_context_sha256=context["sha256"],
        photo_sand_qa_pass=True, photo_sand_qa_scope="certified-photo-sand-mask",
        photo_certified_sand_mask_path=str(photo), photo_certified_sand_mask_sha256=asset(photo)["sha256"])
    manifest = review.write_raw_review_root(
        tmp_path / "review", run="ac52", native_configuration=inputs["configuration"],
        raw_binding=inputs["binding"], reference=inputs["reference"], frames=inputs["frames"],
        training_context=training_context)
    yield trainer, manifest
    reset_parameters(token)
    for name in names:  # Import-time defaults came from this synthetic configuration.
        if not loaded[name]:
            sys.modules.pop(name, None)
            vars(sys.modules["darsia.fluidflower.phase_training"]).pop(name.rsplit(".", 1)[1], None)


def _source(trainer, tmp_path, manifest):
    return trainer.RunSource(run="ac52", dataset="ffac", review_root=tmp_path / "review",
                             asset_manifest_path=manifest, decisions_path=tmp_path / "decisions.json",
                             combined_manifest_path=tmp_path / "combined.json")


def _region(trainer):
    from darsia.fluidflower.phase_training.logistic import ReviewedRegion
    # Native rows 4-7 and columns 6-12, on the 24 x 14 canvas with the box at (5, 3).
    payload = dict(id="water-1", x=11 / 24, y=7 / 14, width=6 / 24, height=3 / 14, elapsed_h=0.5)
    return ReviewedRegion(region_id="ffac:ac52:water-1", run="ac52", dataset="ffac", frame_index=1,
                          elapsed_h=0.5, phase_class="water", confidence="high", source="manual",
                          reviewer="synthetic", payload=payload)


def test_a_raw_run_is_staged_from_its_sha_bound_records(tmp_path, trainer):
    trainer, manifest = trainer
    staged = trainer._stage_run(_source(trainer, tmp_path, manifest), tmp_path / "cache")
    assert staged.baseline_path.name == "pre_injection.log_linear_rgb.npy"
    assert [path.name for path in staged.frame_paths] == ["frame_000.log_linear_rgb.npy",
                                                         "frame_001.log_linear_rgb.npy"]
    assert staged.frame_valid_mask_paths[1].name == "frame_001.log_linear_valid_mask.npy"
    record = json.loads(manifest.read_text(encoding="utf-8"))["frames"][1]["native_record"]
    original = json.loads(open(record["path"], encoding="utf-8").read())["full"]["native_centre"]
    assert asset(staged.frame_paths[1])["sha256"] == original["log_linear_rgb"]["sha256"]
    receipt = json.loads((staged.root / "staged_assets.json").read_text(encoding="utf-8"))
    assert receipt["source_kind"] == "raw" and len(receipt["files"]) == 7
    again = trainer._stage_run(_source(trainer, tmp_path, manifest), tmp_path / "cache")
    assert again == staged  # A complete staging is reused unchanged.


def test_a_raw_region_is_extracted_through_the_raw_cohort(tmp_path, trainer):
    trainer, manifest = trainer
    staged = trainer._stage_run(_source(trainer, tmp_path, manifest), tmp_path / "cache")
    regions = (_region(trainer),)
    key = trainer._feature_cache_key(staged, regions, pixels_per_region=50, texture_sigma=2.0)
    task = trainer.ExtractionTask(staged=staged, regions=regions, output_path=tmp_path / "f/samples.npz",
                                  metadata_path=tmp_path / "f/metadata.json", cache_key=key,
                                  pixels_per_region=50, texture_sigma=2.0, opencv_threads=1)
    report = trainer._extract_run_features(task)
    with np.load(task.output_path) as archive:
        features, owners = archive["features"], archive["region_indices"]
    assert features.dtype == np.float32 and features.shape[0] == owners.size == 28  # 7 x 4 box pixels.
    frame = json.loads(task.metadata_path.read_text(encoding="utf-8"))["frames"][0]
    receipts = [row["audit"]["sha256"] for row in json.loads(manifest.read_text(encoding="utf-8"))["frames"]]
    assert frame["raw_frame_geometry_receipt_sha256"] == receipts[1]
    assert frame["raw_previous_geometry_receipt_sha256"] == receipts[0]
    assert frame["raw_temporal_reference_contract"] == "darsia-raw-native-centre-source-supported-v1"
    assert not any(name.startswith("v5_") for name in frame)
    assert not report.get("cache_hit")
    again = trainer._extract_run_features(task)
    assert again["cache_hit"] is True  # The RAW feature cache is reused unchanged.


def test_raw_cache_keys_bind_the_raw_identities(tmp_path, trainer):
    trainer, manifest = trainer
    staged = trainer._stage_run(_source(trainer, tmp_path, manifest), tmp_path / "cache")
    regions = (_region(trainer),)
    key = trainer._feature_cache_key(staged, regions, pixels_per_region=50, texture_sigma=2.0)
    assert key == trainer._feature_cache_key(staged, regions, pixels_per_region=50, texture_sigma=2.0)
    identity = trainer._raw_geometry_identity(staged, regions)
    assert [row["index"] for row in identity["frame_receipts"]] == [0, 1]
    assert identity["training_context"]["photo_sand_qa_pass"] is True
    assert trainer._raw_review_manifest(json.loads(manifest.read_text(encoding="utf-8")))
    assert not trainer._raw_review_manifest({"frame_sources": []})
    for item in (trainer._raw_review_manifest, trainer._stage_raw_review_run, trainer._raw_geometry_identity):
        assert item.__doc__ and item.__doc__.strip()


def test_raw_reviews_are_one_kind_of_training_input(tmp_path, trainer):
    trainer, manifest = trainer
    review_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert review.timeline_rows(review_manifest) == review_manifest["frames"]
    assert review.review_content_shape(review_manifest) == NATIVE_HW
    assert review.timeline_rows({"frame_sources": [1]}) == [1]
    assert review.review_content_shape({"analysis_image_shape_hw": [2, 3]}) == (2, 3)
