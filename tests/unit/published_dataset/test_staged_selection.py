"""Metadata-only selection guards; no RAW data or scientific qualification."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from darsia.fluidflower.published_dataset.builder import BuildReport, job_frames
from darsia.fluidflower.published_dataset.staged_package import StagedInput


def _selection():
    cases = ["first", "second"]
    digest = "a" * 64
    inputs = [StagedInput("camera", case + ".ARW", digest, 12, "image",
                         ("frame." + case + ".native.acquisition",), True) for case in cases]
    results = {case: dict(frame_index=index, elapsed_h=float(index), bindings=dict(source_sha256=digest),
                         total_mass_g=1.0, aqueous_mass_g=0.5, gas_mass_g=0.5)
               for index, case in enumerate(cases)}
    job = SimpleNamespace(runs=["r"], frames=lambda run: cases,
                          frame_result=lambda run, case: results[case])
    return job, inputs, results


def test_staged_frames_keep_exact_named_acquisitions_with_identical_hashes():
    job, inputs, _ = _selection()
    report = BuildReport()
    frames, _, _ = job_frames(job, {"r": list(reversed(inputs))}, report, exact_acquisition=True)
    assert not report.refusals
    assert [row["raw"]["path"] for row in frames["r"]] == ["camera/first.ARW", "camera/second.ARW"]
    assert frames["r"][0]["raw"]["sha256"] == frames["r"][1]["raw"]["sha256"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "role", "processed", "hash", "not_camera"])
def test_staged_selection_refuses_missing_or_changed_original_named_raw_consumer(change):
    job, inputs, _ = _selection()
    if change == "missing":
        inputs[0] = replace(inputs[0], uses=("frame.other.native.acquisition",))
    elif change == "duplicate":
        inputs.append(replace(inputs[0], relative_path="duplicate.ARW"))
    elif change == "role":
        inputs[0] = replace(inputs[0], role="rig_model")
    elif change == "processed":
        inputs[0] = replace(inputs[0], relative_path="first.png")
    elif change == "hash":
        inputs[0] = replace(inputs[0], sha256="b" * 64)
    else:
        inputs[0] = replace(inputs[0], camera=False)
    report = BuildReport()
    frames, _, _ = job_frames(job, {"r": inputs}, report, exact_acquisition=True)
    assert len(report.refusals) == 1 and "original P1 RAW acquisition consumer" in report.refusals[0]
    assert [row["case_id"] for row in frames["r"]] == ["second"]


def test_legacy_frame_lookup_retains_its_existing_hash_selection():
    job, inputs, _ = _selection()
    report = BuildReport()
    frames, _, _ = job_frames(job, {"r": inputs}, report)
    assert not report.refusals
    assert [row["raw"]["path"] for row in frames["r"]] == ["camera/first.ARW", "camera/first.ARW"]
