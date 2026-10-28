"""Direct findings parity on tiny synthetic inputs; no physical qualification."""

from copy import deepcopy
import json

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from tests.unit.test_geometry_imported_rasters import imported, snapshot


def test_depth_and_injector_issues_keep_the_full_original_order():
    """Depth findings precede each injector's original ordered diagnostics."""
    depth = np.array([[-1.0, 1.0, 1.0, 1.0, 6.0]], dtype=np.float32)
    before = depth.tobytes()
    rows = [
        {"name": "Port", "injector_id": "A", "x_m": 0.0, "y_m": 0.0},
        {
            "name": "Port", "injector_id": "a", "protocol_location_id": "B",
            "x_m": 2.0, "y_m": 0.0, "uncertainty_m": 0.5,
            "protocol_x_m": 0.0, "protocol_y_m": 0.0,
            "protocol_coordinate_candidates": [
                {"x_m": 0.0, "y_m": 0.0},
                {"x_m": 0.5, "y_m": 0.0},
                {"x_m": 1.0, "y_m": 0.0},
            ],
        },
        {"x_m": None, "y_m": "nan"},
    ]
    rows_before = deepcopy(rows)
    assert geometry.depth_and_injector_findings(
        depth, rows, width_m=1.0, height_m=1.0
    ) == {
        "depth": {"minimum_m": -1.0, "median_m": 1.0, "maximum_m": 6.0},
        "injector_count": 3,
        "issues": [
            {"severity": "error", "code": "nonpositive_depth",
             "message": "Depth map contains non-positive values", "details": {}},
            {"severity": "warning", "code": "depth_outliers",
             "message": "Depth maximum exceeds five times the median", "details": {}},
            {"severity": "warning", "code": "duplicate_injector_id",
             "message": "Injection-point identity 'a' is repeated", "details": {}},
            {"severity": "warning", "code": "injector_identity_differs_from_protocol",
             "message": "Port uses location ID 'a', but Excel uses 'B'", "details": {}},
            {"severity": "warning", "code": "duplicate_injector_name",
             "message": "Injection-point name 'Port' is repeated", "details": {}},
            {"severity": "warning", "code": "protocol_injector_position_conflict",
             "message": "Port has conflicting x/y positions in the protocol source",
             "details": {"injector_id": "a", "positions": [(0.0, 0.0), (1.0, 0.0)],
                         "tolerance_m": 0.5}},
            {"severity": "error", "code": "injector_outside_rig",
             "message": "Port lies outside the rig", "details": {}},
            {"severity": "warning", "code": "injector_position_differs_from_protocol",
             "message": "Port differs from its Excel protocol position",
             "details": {"injector_id": "a", "gui_position_m": [2.0, 0.0],
                         "protocol_position_m": [0.0, 0.0], "difference_m": 2.0,
                         "tolerance_m": 0.5}},
            {"severity": "error", "code": "invalid_injector_position",
             "message": "Injection Point 3 has no finite position; place it in the GUI or enter x/y",
             "details": {}},
        ],
    }
    assert depth.tobytes() == before and rows == rows_before


def test_depth_statistics_keep_finite_selection_and_strict_outlier_boundary():
    """Absent and empty maps differ, while finite support keeps its statistics."""
    rows = [{"x_m": 0.0, "y_m": 0.0}]
    assert geometry.depth_and_injector_findings(
        None, rows, width_m=1.0, height_m=1.0
    ) == {"depth": {}, "injector_count": 1, "issues": []}
    empty = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
    before = empty.tobytes()
    assert geometry.depth_and_injector_findings(
        empty, rows, width_m=1.0, height_m=1.0
    ) == {
        "depth": {}, "injector_count": 1,
        "issues": [{"severity": "error", "code": "empty_depth_map",
                    "message": "Depth map has no finite values", "details": {}}],
    }
    assert empty.tobytes() == before
    mixed = np.array([np.nan, 1.0, 1.0, 5.0, np.inf], dtype=np.float64)
    before = mixed.tobytes()
    assert geometry.depth_and_injector_findings(
        mixed, rows, width_m=1.0, height_m=1.0
    ) == {
        "depth": {"minimum_m": 1.0, "median_m": 1.0, "maximum_m": 5.0},
        "injector_count": 1, "issues": [],
    }
    assert mixed.tobytes() == before
    assert geometry.depth_and_injector_findings(
        np.array([0.0, 0.0, 5e-12]), rows, width_m=1.0, height_m=1.0
    )["issues"] == [
        {"severity": "error", "code": "nonpositive_depth",
         "message": "Depth map contains non-positive values", "details": {}},
    ]


def test_injector_defaults_numeric_parsing_and_input_rows_remain_literal():
    """Default names, folded IDs, literal names and parser failures stay distinct."""
    rows = [
        {},
        {"name": "port", "injector_id": "edge", "protocol_location_id": "EDGE",
         "x_m": "0,5", "y_m": "0,5", "uncertainty_m": "-1",
         "protocol_coordinate_candidates": [None, {"x_m": None, "y_m": 0.0}]},
        {"name": "Port", "injector_id": "EDGE", "x_m": 1.0, "y_m": 1.0},
    ]
    before = json.dumps(rows, sort_keys=True).encode("utf-8")
    assert geometry.depth_and_injector_findings(
        None, rows, width_m=1.0, height_m=1.0
    ) == {
        "depth": {}, "injector_count": 3,
        "issues": [
            {"severity": "error", "code": "invalid_injector_position",
             "message": "Injection Point 1 has no finite position; place it in the GUI or enter x/y",
             "details": {}},
            {"severity": "warning", "code": "duplicate_injector_id",
             "message": "Injection-point identity 'EDGE' is repeated", "details": {}},
        ],
    }
    assert json.dumps(rows, sort_keys=True).encode("utf-8") == before
    with pytest.raises(ValueError, match="could not convert string to float"):
        geometry.depth_and_injector_findings(
            None, [{"x_m": "not-a-number", "y_m": 0.0}], width_m=1.0, height_m=1.0
        )
    assert geometry.depth_and_injector_findings(
        None, None, width_m=None, height_m=None
    ) == {
        "depth": {}, "injector_count": 0,
        "issues": [{"severity": "warning", "code": "no_injectors",
                    "message": "No injection-point positions are defined", "details": {}}],
    }


def test_rig_bounds_remain_inclusive_and_require_both_dimensions():
    """Corners are admitted; an absent dimension preserves the original bound scope."""
    rows = [
        {"name": "Lower", "x_m": 0.0, "y_m": 0.0},
        {"name": "Upper", "x_m": 1.0, "y_m": 2.0},
        {"name": "Outside", "x_m": -0.25, "y_m": 1.0},
    ]
    expected = {"depth": {}, "injector_count": 3, "issues": [
        {"severity": "error", "code": "injector_outside_rig",
         "message": "Outside lies outside the rig", "details": {}},
    ]}
    assert geometry.depth_and_injector_findings(
        None, rows, width_m=1.0, height_m=2.0
    ) == expected
    for width, height in [(None, 2.0), (1.0, None)]:
        assert geometry.depth_and_injector_findings(
            None, rows, width_m=width, height_m=height
        ) == {"depth": {}, "injector_count": 3, "issues": []}


def test_protocol_positions_keep_greedy_order_and_inclusive_uncertainty():
    """Clustering is ordered and non-transitive, and equality is within tolerance."""
    row = {
        "name": "Port", "x_m": 0.5, "y_m": 0.0, "uncertainty_m": 0.5,
        "protocol_x_m": 0.0, "protocol_y_m": 0.0,
        "protocol_coordinate_candidates": [
            {"x_m": 0.0, "y_m": 0.0},
            {"x_m": 0.375, "y_m": 0.0},
            {"x_m": 0.75, "y_m": 0.0},
        ],
    }
    assert geometry.depth_and_injector_findings(
        None, [row], width_m=1.0, height_m=1.0
    ) == {
        "depth": {}, "injector_count": 1,
        "issues": [{"severity": "warning", "code": "protocol_injector_position_conflict",
                    "message": "Port has conflicting x/y positions in the protocol source",
                    "details": {"injector_id": "Port", "positions": [(0.0, 0.0), (0.75, 0.0)],
                                "tolerance_m": 0.5}}],
    }
    reordered = deepcopy(row)
    reordered["protocol_coordinate_candidates"] = [
        row["protocol_coordinate_candidates"][1],
        row["protocol_coordinate_candidates"][0],
        row["protocol_coordinate_candidates"][2],
    ]
    assert geometry.depth_and_injector_findings(
        None, [reordered], width_m=1.0, height_m=1.0
    ) == {"depth": {}, "injector_count": 1, "issues": []}
    boundary = {"x_m": 1e-4, "y_m": 0.0, "uncertainty_m": -1.0,
                "protocol_x_m": 0.0, "protocol_y_m": 0.0}
    assert geometry.depth_and_injector_findings(
        None, [boundary], width_m=1.0, height_m=1.0
    ) == {"depth": {}, "injector_count": 1, "issues": []}


def test_original_validator_inserts_findings_and_uses_actual_helper_summary(imported, monkeypatch):
    """The original tiny raster loader and validator surround the real helper call."""
    config = deepcopy(imported.config)
    config["format"] = "unsupported-synthetic-format"
    for region in config["regions"]:
        region["enabled"] = False
    config["injectors"] = [
        {"x_m": 0.1, "y_m": 0.2},
        {"name": "Missing", "x_m": 0.1, "y_m": None},
    ]
    config = sources.bind_imported_rasters(
        config, sources=imported.config["imported_rasters"]["sources"],
        diagnosis=imported.config["imported_rasters"]["diagnosis"],
    )
    config_before = deepcopy(config)
    sources_before = snapshot(imported.original)
    original = geometry.depth_and_injector_findings
    calls = []

    def observe(depth, rows, *, width_m, height_m):
        """Delegate all findings arithmetic and retain only call observations."""
        depth_before = depth.tobytes()
        rows_before = deepcopy(rows)
        result = original(depth, rows, width_m=width_m, height_m=height_m)
        assert depth.tobytes() == depth_before and rows == rows_before
        np.testing.assert_array_equal(depth, imported.arrays["depth"])
        assert depth.dtype == np.dtype("float64")
        assert (width_m, height_m) == (0.9, 0.49)
        calls.append(result)
        return result

    monkeypatch.setattr(geometry, "depth_and_injector_findings", observe)
    report = geometry.validate_geometry_config(config)
    assert len(calls) == 1
    assert report["issues"] == [
        {"severity": "error", "code": "unsupported_format",
         "message": "Unsupported geometry configuration", "details": {}},
        {"severity": "error", "code": "invalid_injector_position",
         "message": "Missing has no finite position; place it in the GUI or enter x/y",
         "details": {}},
        {"severity": "warning", "code": "no_active_porous_regions",
         "message": "No reviewed porous-medium or seal regions are enabled", "details": {}},
    ]
    assert report["issues"][1] == calls[0]["issues"][0]
    assert report["summary"]["depth"] == calls[0]["depth"]
    assert report["summary"]["depth"] == {
        "minimum_m": 0.01, "median_m": 0.01, "maximum_m": 0.01,
    }
    assert report["summary"]["injector_count"] == 2
    assert report["summary"]["source_mode"] == sources.IMPORTED
    assert report["summary"]["full_properties_ids"] == [1, 2, 3, 4, 6]
    assert report["summary"]["active_area_fraction"] is None
    assert report["summary"]["estimated_pore_volume_m3"] is None
    assert all(issue["code"] != "invalid_geometry_source_mode" for issue in report["issues"])
    assert config == config_before and snapshot(imported.original) == sources_before
