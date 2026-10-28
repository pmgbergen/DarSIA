"""Synthetic saved-view guard tests; no inference or Studio process."""
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from darsia.fluidflower import phase_saved_review_metadata as saved


def _parameters(section):
    return {"display_views": {"rig_contract": "rig-test", "floor_contact_contract": "floor-test"},
            "identity_report": {"contract": "identity-test"}}[section]


def test_saved_mask_partition_rejects_overlap_and_content_change(tmp_path):
    root = tmp_path / "classes.npz"
    modelled = np.array([[True, False, False], [False, True, False]])
    observed = np.array([[False, True, False], [False, False, False]])
    unobserved = np.array([[False, False, True], [False, False, True]])
    domain = modelled | observed | unobserved
    np.savez(root, modelled=modelled, filled_observed=observed,
             filled_unobserved=unobserved, rig_domain=domain)
    asset = {"path": str(root), "sha256": hashlib.sha256(root.read_bytes()).hexdigest()}
    revision = {"contract": "rig-test", "canvas_wh": [3, 2], "classes": asset,
                "native_origin_xy": [0, 0]}
    with patch.object(saved, "configured_parameters", side_effect=_parameters):
        masks, _ = saved.load_rig_classes(revision)
        assert np.array_equal(masks["rig_domain"], domain)
        assert all(not value.flags.writeable for value in masks.values())
        cropped, _ = saved.load_rig_classes(revision, shape_hw=(1, 2), origin_xy=(1, 0))
        assert np.array_equal(cropped["rig_domain"], domain[:1, 1:3])
        overlap = observed.copy(); overlap[0, 0] = True
        np.savez(root, modelled=modelled, filled_observed=overlap,
                 filled_unobserved=unobserved, rig_domain=domain)
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            saved.load_rig_classes(revision)
        revision["classes"]["sha256"] = hashlib.sha256(root.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="disjoint complete partition"):
            saved.load_rig_classes(revision)


def test_only_supported_bottom_extension_and_integer_crop():
    assert saved.rig_phase_shape({"canvas_wh": [3, 2], "phase_shape_hw": [3, 3]}) == (3, 3)
    with pytest.raises(ValueError):
        saved.rig_phase_shape({"canvas_wh": [3, 2], "phase_shape_hw": [3, 4]})
    with pytest.raises(ValueError):
        saved._crop_rig_grid(np.ones((2, 3), bool), (0, 0), (1, 1), (0.5, 0))


def test_saved_identity_guard_rejects_false_flag():
    audit = {"contract": "identity-test", "complete": True,
             "frozen_gas_no_production_reads": True,
             "legacy_reconstruction_unreachable": True,
             "frozen_gas_compare_and_report_only": True}
    with patch.object(saved, "configured_parameters", side_effect=_parameters):
        saved.require_frozen_gas_audit(audit)
        with pytest.raises(ValueError):
            saved.require_frozen_gas_audit(dict(audit, frozen_gas_no_production_reads=False))


def test_saved_result_reader_never_imports_calculator(tmp_path):
    path = tmp_path / "result.json"
    payload = {"complete": True, "selected_inputs": {"candidate": {"dataset": "series",
               "run": "example", "candidate_id": "model"}},
               "frames": [{"preview": "preview.png", "fields": "fields.npz"}],
               "method": {"settings": {"working_precision": "float32"}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert saved.result_paths(path) == [path.resolve()]
    value = saved.read_result(path)
    assert value["frames"][0]["preview"] == str(tmp_path / "preview.png")
    assert saved.inspect_result(path, value)["frames"] == 1
    assert not hasattr(saved, "run")
