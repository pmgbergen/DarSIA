"""Private producer DATA reuse preserves ordinary diagnostic predicates and admission."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower.painted_analysis_exclusions import DisplayFindingsChanged

from .test_geometry_imported_rasters import imported  # noqa: F401


def _data(imported, **options):
    return geometry._GeometryDiagnosticInputs(
        fields={"labels_path": imported.arrays["labels"], "depth_path": imported.arrays["depth"],
                "facies_source_path": imported.arrays["facies"]},
        baseline_shape=imported.arrays["baseline"].shape, **options)


def _without_locators(report):
    result = deepcopy(report)
    result.pop("validated_utc")
    result.pop("config_sha256")
    return result


def test_owned_arrays_match_original_file_diagnostics_and_do_not_mutate_inputs(imported):
    config = deepcopy(imported.config)
    before = deepcopy(config)
    originals = {key: value.copy() for key, value in imported.arrays.items()}
    expected = geometry._geometry_input_report(config)
    data = _data(imported)

    actual = geometry._geometry_input_report(config, inputs=data)

    assert _without_locators(actual) == _without_locators(expected)
    assert actual["config_sha256"] == expected["config_sha256"]
    assert config == before
    assert all(np.array_equal(value, originals[key]) for key, value in imported.arrays.items())
    imported.arrays["labels"][:] = 99
    assert np.array_equal(data.fields["labels_path"], originals["labels"])
    with pytest.raises(ValueError, match="read-only"):
        data.fields["labels_path"][0, 0] = 88
    with pytest.raises(TypeError):
        data.fields["labels_path"] = originals["labels"]


def test_post_target_arrays_use_original_shape_and_boundary_predicates_without_published_paths(imported, tmp_path):
    post_target = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32)
    run_target = np.ones((3, 3), dtype=np.int32)
    config = deepcopy(imported.config)
    config["run_geometry"]["runs"] = [dict(run_id="one", enabled=True, geometry_choice="run_override",
        acquisition_state="clearwater", source_image_path=config["baseline_path"])]
    oracle_labels, oracle_run = tmp_path / "oracle-labels.npz", tmp_path / "oracle-run.npz"
    np.savez(oracle_labels, array=post_target)
    np.savez(oracle_run, array=run_target)
    oracle_config = deepcopy(config)
    oracle_config["labels_path"] = str(oracle_labels)
    oracle_config["run_geometry"]["runs"][0]["labels_path"] = str(oracle_run)
    expected = geometry._geometry_input_report(oracle_config)
    data = geometry._GeometryDiagnosticInputs(fields={"labels_path": post_target},
        run_fields={"one": {"labels": run_target}})
    source_before = Path(config["labels_path"]).read_bytes()

    actual = geometry._geometry_input_report(config, inputs=data)

    assert _without_locators(actual) == _without_locators(expected)
    assert any(issue["code"] == "run_geometry_labels_shape" for issue in actual["issues"])
    assert any(issue["code"] == "unused_layer_definitions" for issue in actual["issues"])
    assert not any(issue["code"] == "run_geometry_override_missing" for issue in actual["issues"])
    assert actual["summary"]["selected_run_override_count"] == 1
    assert "labels_path" not in config["run_geometry"]["runs"][0]
    assert Path(config["labels_path"]).read_bytes() == source_before


@pytest.mark.parametrize("fault", ["label_dtype", "mask_dtype", "mask_shape", "mapped_mask", "unknown_run"])
def test_private_data_rejects_wrong_types_grid_masks_and_unrequested_runs(imported, fault):
    if fault == "label_dtype":
        with pytest.raises(ValueError, match="dtype"):
            geometry._GeometryDiagnosticInputs(fields={"labels_path": np.ones((2, 3), dtype=float)})
    elif fault == "mask_dtype":
        with pytest.raises(ValueError, match="boolean mask"):
            geometry._GeometryDiagnosticInputs(analysis_exclusion=np.zeros((2, 3), dtype=np.uint8))
    elif fault == "unknown_run":
        data = _data(imported, run_fields={"not-selected": {"labels": imported.arrays["labels"]}})
        with pytest.raises(ValueError, match="outside the actual geometry"):
            geometry._geometry_input_report(imported.config, inputs=data)
    else:
        mask = np.zeros((3, 3) if fault == "mask_shape" else (2, 3), dtype=bool)
        mask[0, 0] = True
        data = _data(imported, analysis_exclusion=mask)
        report = geometry._geometry_input_report(imported.config, inputs=data)
        issue = next(issue for issue in report["issues"] if issue["code"] == "painted_analysis_exclusions_changed")
        assert issue["severity"] == "error" and "mapped labels or another native grid" in issue["message"]


@pytest.mark.parametrize("failure", [False, True])
def test_shared_findings_or_original_failure_are_consumed_once_in_original_issue_position(imported, failure):
    issue = dict(severity="warning", code="original_review_issue", message="Original bound review issue")
    findings = (DisplayFindingsChanged([issue], "Original provider failure") if failure else dict(issues=[issue]))
    data = _data(imported, review_findings=findings)
    issue["message"] = "Changed caller metadata"

    report = geometry._geometry_input_report(imported.config, inputs=data)

    codes = [row["code"] for row in report["issues"]]
    assert codes.count("original_review_issue") == 1
    assert next(row for row in report["issues"] if row["code"] == "original_review_issue")["message"] == "Original bound review issue"
    expected_tail = ["original_review_issue", "painted_findings_changed"] if failure else ["original_review_issue"]
    assert codes[-len(expected_tail):] == expected_tail
    if failure:
        assert report["issues"][-1]["message"] == "Original provider failure"
    assert "source_mode" not in report["summary"]


def test_diagnostic_data_cannot_bypass_public_admission_after_actual_source_changes(imported):
    data = _data(imported)
    diagnostic = geometry._geometry_input_report(imported.config, inputs=data)
    assert not any(issue["severity"] == "error" for issue in diagnostic["issues"])
    source = Path(imported.config["facies_source_path"])
    source.write_bytes(source.read_bytes() + b"changed after diagnostics")

    report = geometry.validate_geometry_config(imported.config)

    assert any(issue["code"] == "invalid_geometry_source_mode" and issue["severity"] == "error"
               for issue in report["issues"])
    with pytest.raises(TypeError, match="inputs"):
        geometry.validate_geometry_config(imported.config, inputs=data)
