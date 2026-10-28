"""Synthetic final-verdict unit cases, never RAW or worker qualification.

Authored source only. These exercise comparison/report inputs without invoking
scientific execution, replacing guards or constructing stage completion proof.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.published_dataset.comparison import IDENTICAL, OUTSIDE, WITHIN, compare_frame
from darsia.fluidflower.published_dataset.manifest import DEFAULT_RULE
from darsia.fluidflower.published_dataset.reproduce import Reproduction, Scope, _staged_bit_verdict


def _frame(expected, actual):
    layer = {"value": np.asarray([1.0], dtype=np.float32)}
    return compare_frame("r", "case", expected, actual, layer, deepcopy(layer), DEFAULT_RULE)


@pytest.mark.parametrize("left,right", [(0.0, -0.0), (1, 1.0), (1.0, True), (None, None), (float("nan"), float("nan"))])
def test_staged_mass_bits_refuse_scalar_value_equality_without_exact_encoding(left, right):
    expected = dict(total=left, aqueous=1.0, gas=0.5)
    actual = dict(expected, total=right)
    legacy = _frame(expected, actual)
    assert legacy.verdict == IDENTICAL  # The established comparison remains unchanged.
    strict = _staged_bit_verdict(legacy, expected, actual)
    assert strict.verdict == OUTSIDE
    assert strict.first_failure["quantity"] == "total mass bits"
    assert strict.as_dict()["checks"][-3]["passed"] is False


def test_staged_mass_bits_accept_matching_types_and_ieee_bytes():
    masses = dict(total=1.5, aqueous=1.0, gas=-0.0)
    strict = _staged_bit_verdict(_frame(masses, dict(masses)), masses, dict(masses))
    assert strict.verdict == IDENTICAL and strict.first_failure is None
    assert all(check.passed for check in strict.checks)


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("change", ["signed_zero_mass", "number_type", "small_mass_change", "signed_zero_array"])
def test_final_frame_counts_and_report_rows_follow_staged_bit_requirement(tmp_path, staged, change):
    expected = dict(total=0.0, aqueous=1.0, gas=0.5)
    actual = dict(expected)
    old = {"value": np.asarray([0.0, 1.0], dtype=np.float32)}
    new = deepcopy(old)
    if change == "signed_zero_mass":
        actual["total"] = -0.0
    elif change == "number_type":
        actual["total"] = 0
    elif change == "small_mass_change":
        actual["gas"] += 1e-8
    else:
        new["value"][0] = -0.0
    np.savez(tmp_path / "expected.npz", **old)
    manifest = dict(runs=[dict(run="r", frames=[dict(case_id="case")])],
        reproducibility=dict(other_machines=deepcopy(DEFAULT_RULE)),
        expected=dict(runs={"r": dict(frames={"case": dict(masses_g=expected, layer=dict(path="expected.npz"))})}))
    if staged:
        manifest["staged_reproduction"] = {"synthetic_verdict_only": True}
    reproduction = Reproduction(tmp_path, tmp_path / "results",
        runner=SimpleNamespace(reproduced=lambda run, case: (actual, new)))
    reproduction.check = SimpleNamespace(manifest=manifest)
    summary = reproduction.compare(Scope("all", ["r"]))
    if staged:
        assert summary["reproduced"] is False and summary["bit_identical_required"] is True
        assert summary["dataset"] == OUTSIDE and summary["runs"] == {"r": OUTSIDE}
        assert summary["frames"] == {IDENTICAL: 0, WITHIN: 0, OUTSIDE: 1}
        assert summary["first_divergence"]["case_id"] == "case"
        report_frame = reproduction.comparisons[0].as_dict()
        assert report_frame["verdict"] == OUTSIDE and report_frame["first_failure"] is not None
    else:
        assert summary["reproduced"] is True and "bit_identical_required" not in summary
        assert summary["dataset"] == (IDENTICAL if change in {"signed_zero_mass", "number_type"} else WITHIN)
