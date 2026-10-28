"""Reproduce the selected clear-water RAW on the current prepared image basis."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..mass.build_native_bridge import BuildNativeBridge
from ..mass.n1_pipeline import _write_json, asset
from ..mass.production_precision import PixelMath
from .current_inputs import current_input_reader


def prepare_current_clearwater(prepared, source, output, *, reader=None):
    """Use the same live Build bridge as baseline/P1, preserving the source choice.

    No old clear-water pixels, colour matrices or donor fields are loaded. A
    completed derivative is reusable only for the exact source and RAW binding.
    """
    reader = current_input_reader(reader)
    bridge = prepared["bridge"]
    if not isinstance(bridge, BuildNativeBridge):
        raise TypeError("Clear-water preparation requires the current Build RAW bridge")
    run, raw = bridge.context.run, deepcopy(source["raw"])
    if source.get("run") != run or not source.get("selection_rule"):
        raise ValueError("The chosen clear-water acquisition belongs to another run")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "clearwater.json"
    identity = dict(run=run, source=deepcopy(source), raw_binding=deepcopy(bridge.binding),
                    operation_result=deepcopy(bridge.operation_result))
    if manifest.is_file():
        saved = reader.json(asset(manifest))
        if saved.get("input") != identity:
            raise ValueError("Prepared clear water belongs to another source or image basis")
        for pin in saved["clearwater"].values():
            if isinstance(pin, dict):
                reader.path(pin)
        return dict(clearwater=saved["clearwater"], raw=raw, receipt=asset(manifest))
    frames = list(prepared["produced"].values())
    same = [frame for frame in frames
            if reader.json(frame.native_task["geometry"])["acquisition"]["sha256"] == raw["sha256"]]
    if len(same) > 1:
        raise ValueError("The selected clear-water RAW has ambiguous prepared products")
    # An existing phase frame may have a per-frame intensity fit. Reuse only
    # the exact baseline (same field) or a prior explicit clear-water product.
    same = [frame for frame in same if raw["sha256"] == bridge.baseline_sha
            or reader.json(frame.receipt).get("source_purpose") == "clearwater"]
    frame = same[0] if same else bridge.frame(raw, folder / "source", source_purpose="clearwater")
    geometry, correction = reader.json(frame.native_task["geometry"]), reader.json(frame.receipt)
    if (geometry["run"] != run or geometry["acquisition"]["sha256"] != raw["sha256"]
            or frame.native_task["processor"]["raw_binding"] != bridge.binding
            or correction["raw_source_binding"] != bridge.binding
            or correction["native_frame"] != frame.native_task["geometry"]):
        raise ValueError("Current clear-water pixels differ from their RAW source/image basis")
    reference = prepared["configuration"]["reference_spec"]
    baseline = prepared["produced"][reference["case_id"]]
    baseline_geometry = reader.json(baseline.native_task["geometry"])
    baseline_receipt = reader.json(baseline.receipt)
    matrix = (np.asarray(correction["build_to_frame_native_h"], np.float64)
              @ np.linalg.inv(np.asarray(baseline_receipt["build_to_frame_native_h"], np.float64))
              @ np.asarray(baseline_geometry["reference_to_native_h"], np.float64))
    expected = np.asarray(geometry["reference_to_native_h"], np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12
            or not np.allclose(matrix / matrix[2, 2], expected / expected[2, 2], rtol=0, atol=1e-10)):
        raise ValueError("Current clear water and baseline have different reference coordinates")
    values = reader.archive(frame.products, ("log_linear_rgb", "valid_mask"))
    log, valid = values["log_linear_rgb"], values["valid_mask"]
    if (log.dtype != np.float32 or log.shape != (3, *bridge.context.grid["num_voxels"])
            or valid.dtype != bool or valid.shape != log.shape[1:]
            or not np.array_equal(valid, np.isfinite(log).all(0))):
        raise ValueError("Current clear-water products have another grid or numerical encoding")
    rgb = np.moveaxis(PixelMath("float32").decode_log_rgb(log), 0, -1)
    if not np.any(valid & np.isfinite(rgb).all(-1) & (rgb > 0).all(-1)):
        raise ValueError("The current clear-water source has no usable positive signal")
    from .io_ops import publish_file
    pins = {}
    for name, array in (("rgb", rgb), ("valid", valid)):
        path = folder / (name + ".npy")
        publish_file(path, lambda stream, array=array: np.save(stream, array, allow_pickle=False))
        pins[name] = asset(path)
    clearwater = dict(pins, H_reference_to_grid=(matrix / matrix[2, 2]).tolist())
    receipt = _write_json(manifest, dict(format="darsia-current-clearwater-v1", input=identity,
        clearwater=clearwater, source_frame=deepcopy(frame.receipt),
        encoding="float32-camera-linear-unit-rgb", old_clearwater_pixels_used=False,
        producer=asset(Path(__file__))))
    return dict(clearwater=clearwater, raw=raw, receipt=receipt)
