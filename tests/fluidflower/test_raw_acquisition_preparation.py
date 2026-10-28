"""Prepared RAW baseline/history survives the unchanged phase input readers."""

import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.pipeline import raw_acquisition as raw, phase_inputs
from darsia.fluidflower.mass.n1_pipeline import asset, _write_json
from darsia.fluidflower.native import sidecar


@pytest.mark.parametrize("pad", [0, 1])
def test_prepared_masked_core_and_source_backed_temporal_ring_round_trip(tmp_path, monkeypatch, pad):
    params = dict(contract="fixture-input", kinds=dict(new_native="new", original_native="old",
        original_core="old-core", original_ring="old-ring"), source_sha_field="source_sha256",
        fields=dict(native=["log_path", "log_sha", "valid_path", "valid_sha"],
                    temporal=["ring_path", "ring_sha", "pad", "origin"]),
        policy_sources=dict(no_ring="core", ring="ring"))
    support = dict(grid_shape_hw=[2, 3], grid_archive_key="sand", guard_px=0,
        keep_policy="keep", fill_policy="fill", keep_contract="keep", fill_contract="fill",
        hole_max_area_px=0, connectivity=4, vertical_gap_max_px=1, vertical_gap_rule="none",
        homography_epsilon=1e-12, cell_to_reference_native=np.eye(3).tolist())
    ring = dict(contract="fixture-ring", maximum_center_delta_log=0,
                expected_keys=["top", "bottom", "left", "right", "pad_px", "contract"])
    phase = _write_json(tmp_path / "phase.json", dict(parameters=dict(inputs=params,
        geometry=dict(native_shape_hw=[2, 3], source_box_xywh=[1, 1, 3, 2]))))
    native_config = _write_json(tmp_path / "native.json", dict(source={}, support=support, extended_ring=ring))
    raw_file = tmp_path / "source.ARW"
    raw_file.write_bytes(b"RAW")
    acquisition = asset(raw_file)
    photo = tmp_path / "sand.npz"
    sand = np.ones((2, 3), bool)
    sand[0, 0] = False
    np.savez(photo, sand=sand)
    photo_asset = asset(photo)
    spec = dict(case_id="case", source=dict(run="fixture", source_sha256=acquisition["sha256"],
        source_binding=dict(source_sha256=acquisition["sha256"])), native_task=dict(kind="raw_rig"),
        phase_roles=dict(photo_sand_reference=photo_asset,
            photo_proof=dict(sha256=photo_asset["sha256"], phase_labels_used=False),
            reference_to_current_h=np.eye(3).tolist(), current_h=np.eye(3).tolist()))
    config = dict(format=raw.FORMAT, run="fixture", native=dict(configuration=native_config),
        phase=dict(configuration=phase), acquisitions=[spec], temporal_pad_px=pad)
    config_asset = _write_json(tmp_path / "preparation.json", config)
    full = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    log = full[:, 1:3, 1:4].copy()

    def native(owner, spec, stage):
        stage.mkdir(parents=True)
        audit = dict(contract="darsia-raw-rig-source-v1", jpeg_inputs=False, cpu_only=True,
            training=False, processor=dict(contract="darsia-raw-rig-source-v1"),
            sources=dict(camera_arw=acquisition))
        bound = _write_json(stage / "audit.json", audit)
        return dict(log=log, valid=np.isfinite(log).all(0), full_log=full, audit=audit,
            record=dict(audit=bound, full=dict(source_sha256=acquisition["sha256"],
                binding=dict(producer_audit=bound))))
    monkeypatch.setattr(raw.RawAcquisitionInputs, "native", native)
    records = raw.prepare(config_asset["path"], tmp_path / "output",
                          expected_sha256=config_asset["sha256"], unc_only=False)
    record = records[0]
    row = json.loads(Path(record["native_row"]["path"]).read_bytes())
    masked = phase_inputs.original_native_input(row, scientific_native_log=log, parameters=params)
    expected = log.copy()
    expected[:, 0, 0] = np.nan
    np.testing.assert_array_equal(masked.log.view(np.uint32), expected.view(np.uint32))
    temporal = phase_inputs.original_temporal_input(masked, row, parameters=params, ring_parameters=ring)
    if pad:
        expected_full = full.copy()
        expected_full[:, 1:3, 1:4] = expected
        np.testing.assert_array_equal(temporal.log.view(np.uint32), expected_full.view(np.uint32))
    else:
        np.testing.assert_array_equal(temporal.log.view(np.uint32), expected.view(np.uint32))
    assert row["raw_acquisition"] == acquisition and temporal.origin_xy == (-pad, -pad)
    with pytest.raises(FileExistsError):
        raw.prepare(config_asset["path"], tmp_path / "output",
                    expected_sha256=config_asset["sha256"], unc_only=False)
