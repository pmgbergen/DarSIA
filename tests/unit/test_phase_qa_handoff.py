"""Actual phase-writer DATA handoff; no exclusion approval or eligibility fixture.

The tiny arrays exercise the original product writer, not phase inference or a
scientific run. These tests have not been executed for this source increment.
"""
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.phase.inputs import TemporalNativeInput
from darsia.fluidflower.pipeline import runner, series


@pytest.fixture
def written_phase(tmp_path):
    shape = (2, 3)
    source = dict(source_sha256="a" * 64, source_binding=dict(source_sha256="a" * 64))
    spec = dict(case_id="handoff-only", frame_index=1, elapsed_h=1.0,
                source=source, mass_spec=dict(selected_target_h=1.0))
    full = dict(scope="TESTONLY source handoff; no native qualification")
    native = dict(source_sha256=source["source_sha256"], full=full)
    temporal = TemporalNativeInput(np.zeros((3, *shape), np.float32),
        np.ones(shape, bool), (0, 0), dict(scope="TESTONLY writer input"))
    frame = dict(pixel_area_mm2=np.ones(shape, np.float32), temporal_input=temporal,
                 current_h=np.eye(3), scientific=dict(full=full))
    inferred = dict(labels=np.zeros(shape, np.uint8), pixel_area_mm2=np.ones(shape, np.float32),
        classes=dict(rig_domain=np.ones(shape, bool), modelled=np.ones(shape, bool),
                     filled_observed=np.zeros(shape, bool), filled_unobserved=np.zeros(shape, bool)),
        qa=dict(scope="TESTONLY phase QA DATA; no approval", retained_diagnostic="unaltered"),
        seam_masks={}, post_masks={})
    phase = runner._phase_outputs(tmp_path, spec, frame, inferred)
    return spec, native, phase, inferred["qa"]


def test_actual_phase_writer_qa_descriptor_is_carried_without_reinterpretation(written_phase):
    spec, native, phase, original_qa = written_phase
    original_spec, original_phase = deepcopy(spec), deepcopy(phase)
    bound = series.bind_mass_frame(spec, native, phase, baseline={})
    assert bound["phase_qa"] == phase["qa"]
    assert bound["phase_qa"] is not phase["qa"]
    assert DataReader().json(bound["phase_qa"]) == original_qa
    assert spec == original_spec and phase == original_phase
    assert "approved_analysis_gap" not in bound
    assert "scientific_eligibility" not in bound
    bound["phase_qa"]["sha256"] = "0" * 64
    assert phase == original_phase


def test_legacy_phase_without_qa_adds_no_new_mass_field(written_phase):
    spec, native, phase, _ = written_phase
    legacy = {key: value for key, value in phase.items() if key != "qa"}
    bound = series.bind_mass_frame(spec, native, legacy, baseline={})
    assert "phase_qa" not in bound
    assert bound["labels"] == phase["labels"] and bound["classes"] == phase["classes"]


def test_mass_recipe_cannot_supply_its_own_phase_qa(written_phase):
    spec, native, phase, _ = written_phase
    declared = deepcopy(spec)
    declared["mass_spec"]["phase_qa"] = deepcopy(phase["qa"])
    with pytest.raises(ValueError, match="fresh phase stage"):
        series.bind_mass_frame(declared, native, phase, baseline={})


def test_different_acquisition_cannot_carry_phase_qa(written_phase):
    spec, native, phase, _ = written_phase
    with pytest.raises(ValueError, match="another acquisition"):
        series.bind_mass_frame(spec, native, dict(phase, source_sha256="b" * 64), baseline={})
