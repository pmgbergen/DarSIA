"""Small source-support examples independent of a series or filesystem share."""

import numpy as np
import pytest

from darsia.fluidflower.native import support
from darsia.fluidflower.native.io import sha256_file


@pytest.fixture
def parameters():
    return dict(
        keep_policy="keep",
        fill_policy="fill",
        keep_contract="kept",
        fill_contract="filled",
        hole_max_area_px=4,
        connectivity=8,
        guard_px=1,
        vertical_gap_max_px=2,
        vertical_gap_rule="bounded",
        grid_shape_hw=[9, 9],
        grid_archive_key="sand",
        cell_to_reference_native=np.eye(3).tolist(),
        homography_epsilon=1e-9,
        coverage_tolerance=1e-6,
        texture_footprint_hw=[3, 3],
    )


def test_enclosed_holes_preserve_support_and_never_fill_a_border(parameters):
    core = np.ones((9, 9), bool)
    core[4, 4] = core[0, 0] = False
    saved = core.copy()
    filled, metadata, masks = support.apply_photo_safe_core_enclosed_holes(
        core, enclosed_hole_policy="fill", parameters=parameters, return_rule_masks=True
    )
    assert np.array_equal(core, saved)
    assert filled[4, 4] and not filled[0, 0]
    assert np.all(filled[core])
    assert masks["enclosed_rule1"][4, 4]
    assert metadata["enclosed_hole_added_component_count"] == 1
    assert (
        support.photo_safe_core_policy_metadata("keep", parameters=parameters)[
            "photo_sand_support_contract"
        ]
        == "kept"
    )
    with pytest.raises(ValueError, match="policy"):
        support.photo_safe_core_policy_metadata("unknown", parameters=parameters)


def test_vertical_fill_needs_support_on_both_sides(parameters):
    core = np.zeros((9, 5), bool)
    core[1:3, 2] = core[4:6, 2] = True
    filled, metadata = support.apply_photo_safe_core_enclosed_holes(
        core, enclosed_hole_policy="fill", parameters=parameters
    )
    assert filled[3, 2]
    assert not filled[0, 2] and not filled[6:, 2].any()
    assert metadata["vertical_gap_added_pixel_count"] == 1


def test_projected_core_erodes_edges_and_rejects_another_grid(parameters):
    grid = np.ones((9, 9), bool)
    actual = support.project_source_photo_safe_core(
        grid,
        np.eye(3),
        shape_hw=(9, 9),
        guard_px=1,
        enclosed_hole_policy="keep",
        parameters=parameters,
    )
    expected = np.zeros((9, 9), bool)
    expected[1:-1, 1:-1] = True
    assert np.array_equal(actual, expected)
    with pytest.raises(ValueError, match="grid"):
        support.project_source_photo_safe_core(
            grid[:-1],
            np.eye(3),
            shape_hw=(9, 9),
            guard_px=1,
            enclosed_hole_policy="keep",
            parameters=parameters,
        )


def test_feature_footprint_uses_previous_and_origin(parameters):
    current = np.ones((9, 9), bool)
    previous = current.copy()
    previous[4, 4] = False
    safe, raw = support.compute_feature_support(
        current_valid=current,
        baseline_valid=current,
        h_baseline_to_current=np.eye(3),
        previous_valid=previous,
        h_baseline_to_previous=np.eye(3),
        parameters=parameters,
    baseline_origin_xy=(0, 0), previous_origin_xy=(0, 0))
    assert not raw[4, 4]
    assert not safe[3:6, 3:6].any()
    _, shifted = support.compute_feature_support(
        current_valid=current,
        baseline_valid=current,
        h_baseline_to_current=np.eye(3),
        baseline_origin_xy=(1, 0),
        parameters=parameters,
    previous_origin_xy=(0, 0))
    assert not shifted[:, 0].any() and shifted[:, 1:].all()


def test_isolated_repair_preserves_finite_channels_and_excludes_border():
    params = dict(
        policy="fill",
        keep_policy="keep",
        max_area_px=3,
        connectivity=8,
        enclosure_footprint_hw=[3, 3],
        neighbour_radius_px=2,
        contract_metadata={"contract": "local-median"},
    )
    signal = np.full((3, 9, 9), 2, np.float32)
    signal[0, 4, 4] = np.nan
    signal[1, 4, 4] = np.float32(7)
    signal[:, 0, 0] = np.nan
    saved = signal.copy()
    result, mask, metadata = support.fill_isolated_invalid(
        signal, np.ones((9, 9), bool), parameters=params
    )
    assert np.array_equal(signal, saved, equal_nan=True)
    assert np.array_equal(result[np.isfinite(saved)], saved[np.isfinite(saved)])
    assert result[0, 4, 4] == 2 and result[1, 4, 4] == 7
    assert mask[4, 4] and not mask[0, 0]
    assert metadata["filled_pixel_count"] == 1
    assert support.invalid_policy_contract(parameters=params) == {
        "contract": "local-median"
    }
    assert support.validate_invalid_policy("keep", parameters=params) == "keep"
    with pytest.raises(ValueError, match="policy"):
        support.validate_invalid_policy("bad", parameters=params)


def test_bound_grid_is_read_only_and_changed_bytes_rejected(tmp_path, parameters):
    path = tmp_path / "support.npz"
    np.savez(path, sand=np.ones((9, 9), bool))
    digest = sha256_file(path)
    grid = support.source_photo_grid(path, digest, parameters=parameters)
    assert grid.dtype == bool and not grid.flags.writeable
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        support.source_photo_grid(path, digest, parameters=parameters)
