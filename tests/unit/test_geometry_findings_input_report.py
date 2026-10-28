"""Original geometry diagnostics remain separate from mandatory source admission."""

from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry

from .test_geometry_imported_rasters import imported


def test_public_validator_retains_imported_admission_after_internal_report(imported):
    """Only the full validator reports independently authoritative raster scope."""
    config = imported.config
    before = deepcopy(config)

    diagnostic = geometry._geometry_input_report(config)
    report = geometry.validate_geometry_config(config)

    assert report["issues"] == diagnostic["issues"]
    assert not any(issue["severity"] == "error" for issue in report["issues"])
    assert report["config_sha256"] == diagnostic["config_sha256"]
    assert "source_mode" not in diagnostic["summary"]
    assert diagnostic["summary"]["estimated_pore_volume_m3"] > 0
    assert report["summary"]["source_mode"] == "imported-rasters"
    assert report["summary"]["full_properties_ids"] == [1, 2, 3, 4, 6]
    assert report["summary"]["estimated_pore_volume_m3"] is None
    assert report["summary"]["active_area_fraction"] is None
    assert config == before


@pytest.mark.parametrize("pin", ["facies", "diagnosis"])
def test_public_validator_rejects_changed_source_pins_after_diagnostics(imported, pin):
    """An internal diagnostic report never replaces the original source checks."""
    config = deepcopy(imported.config)
    binding = (config["imported_rasters"]["diagnosis"] if pin == "diagnosis"
               else config["imported_rasters"]["sources"][pin])
    binding["sha256"] = "f" * 64
    before = deepcopy(config)

    diagnostic = geometry._geometry_input_report(config)
    report = geometry.validate_geometry_config(config)

    assert not any(issue["severity"] == "error" for issue in diagnostic["issues"])
    assert report["issues"][:-1] == diagnostic["issues"]
    assert report["issues"][-1]["code"] == "invalid_geometry_source_mode"
    assert report["issues"][-1]["severity"] == "error"
    assert report["summary"]["error_count"] == diagnostic["summary"]["error_count"] + 1
    assert config == before


def test_internal_report_keeps_property_depth_and_injector_issue_order(imported, tmp_path):
    """Keep the original predicates, numeric summaries and caller-owned inputs."""
    depth_path = tmp_path / "diagnostic-depth.npz"
    np.savez(depth_path, array=np.array([[0., 1., 1.], [1., 1., 10.]]))
    config = deepcopy(imported.config)
    config["depth_path"] = str(depth_path)
    config["width_m"] = -1.
    config["regions"][0]["porosity"] = 0.
    config["injectors"] = [dict(name="Point", injector_id="point", x_m="", y_m=.1)]
    before = deepcopy(config)

    report = geometry._geometry_input_report(config)

    assert [issue["code"] for issue in report["issues"]] == [
        "invalid_width", "invalid_porosity", "nonpositive_depth", "depth_outliers",
        "invalid_injector_position",
    ]
    assert report["summary"]["error_count"] == 4
    assert report["summary"]["warning_count"] == 1
    assert report["summary"]["depth"] == dict(minimum_m=0., median_m=1., maximum_m=10.)
    assert report["summary"]["injector_count"] == 1
    assert config == before
