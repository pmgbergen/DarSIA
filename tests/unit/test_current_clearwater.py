from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass.build_native_bridge import BuildNativeBridge, BuildNativeFrame
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json, _write_npz
from darsia.fluidflower.pipeline.current_clearwater import prepare_current_clearwater


def test_clearwater_reuses_current_bridge_basis_and_is_saved_once(tmp_path):
    reader = DataReader()
    raw_path = tmp_path / "selected-wash.ARW"
    raw_path.write_bytes(b"the already selected wash source")
    raw = asset(raw_path)
    baseline_path = tmp_path / "baseline.ARW"
    baseline_path.write_bytes(b"actual baseline")
    binding = _write_json(tmp_path / "raw-binding.json", dict(current_independent_basis=True))
    operation = _write_json(tmp_path / "operation.json", dict(current_completed_build=True))
    reference_h = np.array([[1., 0., 3.], [0., 1., 2.], [0., 0., 1.]])
    baseline_h = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
    current_h = np.array([[1., 0., 2.], [0., 1., 4.], [0., 0., 1.]])
    result_h = current_h @ np.linalg.inv(baseline_h) @ reference_h
    expected = np.full((3, 4, 3), .4, np.float32)
    expected[0, 0] = np.nan
    calls = []
    def frame(folder, acquisition, build_h, source_h, *, signal=None):
        folder.mkdir()
        geometry = _write_json(folder / "geometry.json", dict(run="sample", acquisition=acquisition,
            reference_to_native_h=source_h.tolist()))
        products = _write_npz(folder / "native.npz", dict(log_linear_rgb=np.moveaxis(
            np.log((expected if signal is None else signal) * np.float32(65535)), -1, 0),
            valid_mask=np.isfinite(expected).all(-1)))
        receipt = _write_json(folder / "frame.json", dict(raw_source_binding=binding,
            native_frame=geometry, build_to_frame_native_h=build_h.tolist()))
        return BuildNativeFrame(dict(geometry=geometry, processor=dict(raw_binding=binding)), products, receipt)
    baseline = frame(tmp_path / "baseline", asset(baseline_path), baseline_h, reference_h)
    class Bridge(BuildNativeBridge):
        def __init__(self):
            self.context = SimpleNamespace(run="sample", grid=dict(num_voxels=[3, 4]))
            self.binding, self.operation_result = binding, operation
            self.baseline_sha = asset(baseline_path)["sha256"]
        def frame(self, acquisition, output, *, source_purpose=None):
            assert acquisition == raw
            assert source_purpose == "clearwater"
            calls.append(acquisition)
            return frame(Path(output), acquisition, current_h, result_h)
    prepared = dict(bridge=Bridge(), produced=dict(baseline=baseline),
                    configuration=dict(reference_spec=dict(case_id="baseline")))
    source = dict(run="sample", raw=raw, selection_rule="saved middle wash acquisition")
    actual = prepare_current_clearwater(prepared, source, tmp_path / "clearwater")
    values = reader.array(actual["clearwater"]["rgb"])
    np.testing.assert_allclose(values, expected, atol=2e-7, equal_nan=True)
    np.testing.assert_array_equal(reader.array(actual["clearwater"]["valid"]), np.isfinite(expected).all(-1))
    np.testing.assert_allclose(actual["clearwater"]["H_reference_to_grid"], result_h)
    assert calls == [raw]
    assert reader.json(actual["receipt"])["old_clearwater_pixels_used"] is False
    assert prepare_current_clearwater(prepared, source, tmp_path / "clearwater") == actual
    assert calls == [raw]
    changed = deepcopy(source)
    changed["selection_rule"] = "another user choice"
    with pytest.raises(ValueError, match="another source or image basis"):
        prepare_current_clearwater(prepared, changed, tmp_path / "clearwater")
