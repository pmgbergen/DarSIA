"""RAW frame geometry in the training feature/model ABI (package D2b).

Synthetic RAW review: frame k is shifted k pixels right of baseline B, and
the AC28 reference one more. No real context, model or training is used.
"""
import sys

import numpy as np
import pytest

from darsia.fluidflower.native.config import configure_parameters, reset_parameters
from darsia.fluidflower.phase_training import raw_frame_context as raw_context
from darsia.fluidflower.phase_saved_assets import asset
from .test_raw_photographic_review import NATIVE_HW, _write, inputs  # noqa: F401

RASTER = ["domain", "physics_supported_domain", "physical_sand_domain", "observable_phase_domain",
          "facies", "facies_safe_interior", "permeability", "x_m_native", "y_m_native", "pixel_area_m2"]
EXPORTS = dict(
    SHAPE=dict(value=list(NATIVE_HW), python_type="tuple"),
    RASTER_FIELDS=dict(value=RASTER, python_type="tuple"),
    PHOTO_FIELDS=dict(value=["photo_certified_sand_domain", "photo_boundary_uncertain_domain"], python_type="tuple"),
    REFERENCE_SAFE_KEY=dict(value="facies_safe_interior", python_type="str"),
    PHOTO_SAFE_CORE_GUARD_PX=dict(value=1, python_type="int"),
    PHOTO_SAFE_CORE_CONTRACT=dict(value="transported_safe_B_sand_core_only", python_type="str"))
CONFIGURATION = dict(
    support=dict(homography_epsilon=1e-12, coverage_tolerance=1e-6, texture_footprint_hw=[1, 1]),
    saved_review_metadata=dict(training_frame_context=dict(
        exports=EXPORTS, reference_footprint_epsilon=1e-6, area_row_block_size=4, area_horizon_floor=1e-9)))


def _reference_context(path):
    domain = np.zeros(NATIVE_HW, dtype=bool)
    domain[3:7, 4:11] = True
    grid = np.ones(NATIVE_HW, dtype=np.float32)
    np.savez(path, domain=domain, physics_supported_domain=domain, physical_sand_domain=domain,
             observable_phase_domain=np.ones(NATIVE_HW, dtype=bool), facies=domain.astype(np.uint8) * 2,
             facies_safe_interior=domain, permeability=grid, x_m_native=grid, y_m_native=grid,
             pixel_area_m2=grid, x_m=np.linspace(0.0, 1.0, NATIVE_HW[1]), y_m=np.linspace(0.0, 1.0, NATIVE_HW[0]))
    return asset(path)


@pytest.fixture
def cohort(tmp_path, inputs):
    _write(tmp_path, inputs)
    context = _reference_context(tmp_path / "ac28_context.npz")
    name = "darsia.fluidflower.phase_training.frame_context"
    loaded = name in sys.modules
    token = configure_parameters(CONFIGURATION)
    yield raw_context.RawFrameGeometryCohort(tmp_path / "review", "AC52", context_binding=dict(
        reference_context_path=context["path"], reference_context_sha256=context["sha256"]))
    reset_parameters(token)
    if not loaded:  # Its import-time defaults came from this synthetic configuration.
        sys.modules.pop(name, None)
        vars(sys.modules["darsia.fluidflower.phase_training"]).pop("frame_context", None)


def test_frames_bind_their_raw_audit_geometry_and_identities(cohort):
    binding = cohort.frame(1)
    row = cohort.rows[1]
    assert binding["contract"] == raw_context.CONTRACT and binding["case_id"] == "ac52_f2"
    assert binding["receipt_sha256"] == row["audit"]["sha256"]
    assert binding["acquisition_sha256"] == row["acquisition"]["sha256"]
    assert binding["native_record_sha256"] == row["native_record"]["sha256"] and binding["profile_id"] == "late"
    assert binding["validation_only"] is True and binding["mass_eligible"] is False
    assert np.allclose(binding["H_ac28_to_frame_native"], [[1, 0, 2], [0, 1, 0], [0, 0, 1]])
    assert cohort.shape_hw == NATIVE_HW and cohort.source_manifest_sha256 == cohort.native_manifest_sha256
    with pytest.raises(ValueError, match="outside"):
        cohort.frame(2)
    with pytest.raises(ValueError, match="Unknown RAW frame context"):
        raw_context.RawFrameGeometryCohort(cohort.native_manifest_path.parents[2], "ac52",
                                           context_binding=dict(model="x"))


def test_transforms_and_context_use_the_v5_derivations(cohort):
    assert np.allclose(cohort.H_v5_B_to_frame(1), [[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(cohort.H_previous_to_frame(0, 1), [[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    context, binding = cohort.context(0, fields=("domain", "facies"))
    assert binding["frame_index"] == 0
    assert context["domain"][3, 5] and not context["domain"][3, 4] and context["facies"][6, 11] == 2


def _photo_only_root(tmp_path, inputs):
    # Photo-sand masks, but an AC28 context without its physical/safe fields, as on the real run.
    from darsia.fluidflower import raw_photographic_review as review
    certified = np.zeros(NATIVE_HW, dtype=bool)
    certified[2:8, 3:13] = True
    np.savez(tmp_path / "photo_sand.npz", photo_certified_sand_domain=certified,
             photo_boundary_uncertain_domain=~certified)
    np.savez(tmp_path / "ac28_partial.npz", domain=certified, facies=certified.astype(np.uint8),
             permeability=np.ones(NATIVE_HW, np.float32), x_m=np.zeros(NATIVE_HW[1]), y_m=np.zeros(NATIVE_HW[0]))
    partial, photo = asset(tmp_path / "ac28_partial.npz"), asset(tmp_path / "photo_sand.npz")
    review.write_raw_review_root(
        tmp_path / "photo_review", run="ac52", native_configuration=inputs["configuration"],
        raw_binding=inputs["binding"], reference=inputs["reference"], frames=inputs["frames"],
        training_context=dict(reference_context_path=partial["path"], reference_context_sha256=partial["sha256"],
                              photo_sand_qa_pass=True, photo_sand_qa_scope="certified-photo-sand-mask",
                              photo_certified_sand_mask_path=photo["path"],
                              photo_certified_sand_mask_sha256=photo["sha256"]))
    return tmp_path / "photo_review", certified


def test_training_fields_need_no_physical_reference_context(tmp_path, inputs, cohort):
    root, certified = _photo_only_root(tmp_path, inputs)
    photo_cohort = raw_context.RawFrameGeometryCohort(root, "ac52")
    fields = ("observable_phase_domain", "photo_certified_sand_domain", "photo_boundary_uncertain_domain")
    context, binding = photo_cohort.context(1, fields=iter(fields))
    assert set(context) == set(fields) and binding["frame_index"] == 1
    assert context["observable_phase_domain"].all() and context["observable_phase_domain"].shape == NATIVE_HW
    assert np.array_equal(context["photo_certified_sand_domain"], certified)
    assert np.array_equal(context["photo_boundary_uncertain_domain"], ~certified)
    assert photo_cohort.context(0, fields=("observable_phase_domain",))[0]["observable_phase_domain"].all()
    for physical in (("observable_phase_domain", "domain"), ("facies",), None):
        with pytest.raises(ValueError, match="missing physical/safe fields"):
            photo_cohort.context(1, fields=physical)


def test_photo_masks_still_need_their_qa(tmp_path, inputs, cohort):
    root, _certified = _photo_only_root(tmp_path, inputs)
    bound = raw_context.RawFrameGeometryCohort(root, "ac52").context_binding
    unchecked = raw_context.RawFrameGeometryCohort(root, "ac52", context_binding=dict(bound, photo_sand_qa_pass=False))
    with pytest.raises(ValueError, match="independent QA"):
        unchecked.context(1, fields=("photo_certified_sand_domain",))


def test_feature_support_uses_the_raw_records_valid_masks(cohort):
    safe, receipt = cohort.feature_support(1, 0)
    assert safe.shape == NATIVE_HW and not safe[:, 0].any() and not safe[0, 1] and safe[5, 5]
    assert receipt["contract"] == raw_context.SUPPORT_CONTRACT and receipt["previous_frame_index"] == 0
    assert receipt["frame_geometry_receipt_sha256"] == cohort.rows[1]["audit"]["sha256"]
    assert receipt["feature_supported_pixels"] == int(safe.sum())
    with pytest.raises(ValueError, match="earlier"):
        cohort.feature_support(0, 1)
