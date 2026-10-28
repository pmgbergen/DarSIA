"""Full P1 DATA admission on a synthetic closed profile, without running science.

The native/phase layout JSON is really written, hashed and read by the normal
capacity reader. The profile measurement and source-writer identities remain
unit fixtures, not a real qualification, RAW decode or filesystem admission.
The mandatory native BEFORE request and full P1 validator are never bypassed.
"""
from copy import deepcopy

import pytest

from darsia.fluidflower.pipeline import capacity, option_capacity, phase1_capacity, staged_data
from .test_common_capacity_policy import _phase1_layout
from .test_staged_data import phase1_data  # noqa: F401
from .gui.test_capacity import bound_output_profile_TESTONLY


def test_bound_phase1_profile_admits_the_full_mandatory_native_before_request(
    phase1_data, tmp_path, monkeypatch
):
    """Exercise the real selector, full DATA validator, metadata reader and estimate."""
    config = deepcopy(phase1_data)
    rows = _phase1_layout(tmp_path, config)
    # A complete P1 record retains native evidence; omitting storage would hide
    # the original dispatch defect and is deliberately not a fixture option.
    staged_data.validate_phase1_run(config)
    assert config["storage"]["resolution_mm"] == "native"
    qualified, _unused_config, profile = bound_output_profile_TESTONLY()
    task = dict(qualified, pipeline=phase1_capacity.PIPELINE, run=config["run"],
                ordered_case_ids=list(config["ordered_case_ids"]))
    layout = capacity._native_layout_for_task(None, "sample", task, config, rows)
    profile["result_profile"]["measured_native_layout"] = deepcopy(layout)
    # Only the synthetic measured profile's source identities are substituted.
    # Neither request validation nor native metadata reads/shape checks are.
    pinned = []
    monkeypatch.setattr(capacity, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(capacity.subprocess, "check_output", lambda args: ("unit " + args[-1]).encode())
    monkeypatch.setattr(capacity, "_writer_identity", lambda repository, commit, row: pinned.append(row))
    before = deepcopy(config)
    result = capacity._bound_result_estimate(None, "sample", task, config, profile, {},
                                             repository=tmp_path, metadata_rows=rows)
    assert isinstance(result, capacity.OptionSpaceEstimate)
    assert result.options["route"] == phase1_capacity.PIPELINE
    assert result.case_count == len(config["ordered_case_ids"])
    assert result.measured_native_bytes_per_frame == 101
    height, width = layout["phase_shape_hw"]
    extra = phase1_capacity.png_bound(height, width, 2) + phase1_capacity.RECEIPTS_BYTES
    assert result.products_per_frame_estimated_bytes == 101 + extra
    assert [row["relative_path"] for row in pinned] == list(phase1_capacity.WRITERS)
    assert result.options["declared"] == []
    assert config == before


def test_the_generic_full_route_still_refuses_phase1_native_storage(phase1_data):
    """The fix selects the proper P1 route rather than weakening the generic one."""
    before = deepcopy(phase1_data)
    staged_data.validate_phase1_run(phase1_data)
    with pytest.raises(ValueError, match="Optional output storage"):
        option_capacity.validate_requests(phase1_data)
    assert phase1_data == before
