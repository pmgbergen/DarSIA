"""Per-run corrected depth identity, using synthetic local arrays only."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from darsia.fluidflower import geometry_workspace_frame as frame


def _descriptor(path, shape):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw), mtime_ns=path.stat().st_mtime_ns, shape=shape)


def test_ac29_retains_its_own_saved_depth_dtype_and_original_configured_pointer(tmp_path):
    rows = {}
    original = {}
    for run, value in (("ac17", .01), ("ac29", .015)):
        folder = tmp_path / run
        folder.mkdir()
        fields = {}
        for name in frame.FIELDS:
            path = folder / (name + ".npz")
            shape = (2, 3, 3) if name == "shape_corrected_baseline" else (2, 3)
            data = np.full(shape, value, dtype=np.float64) if name == "depth" else np.zeros(shape, np.uint8)
            np.savez(path, array=data)
            fields[name] = _descriptor(path, list(shape))
            original[str(path)] = path.read_bytes()
        rows[run] = dict(fields=fields, configured_depth_map=dict(path=str(folder / "measured-depth-500x1000.npz")))
    correction = tmp_path / "shape_correction.npz"
    correction.write_bytes(b"synthetic correction identity; not decoded")
    config = dict(frame="rig-corrected", frame_binding=dict(
        format=frame.FRAME_BINDING_FORMAT, source_reference_run="ac14", display_reference_run="ac17",
        reference_shape_corrections=[_descriptor(correction, [2, 3])], runs=rows))
    before = deepcopy(config)
    ac17 = frame.selected_fields(config, "ac17")
    ac29 = frame.selected_fields(config, "ac29")
    assert ac29["depth"] == rows["ac29"]["fields"]["depth"]
    assert ac29["depth"] != ac17["depth"]
    assert ac29["configured_depth_map"] == rows["ac29"]["configured_depth_map"]
    assert ac29["configured_depth_map"] != ac17["configured_depth_map"]
    with np.load(ac29["depth"]["path"]) as archive:
        assert archive["array"].dtype == np.dtype("float64")
        assert np.array_equal(archive["array"], np.full((2, 3), .015, dtype=np.float64))
    assert config == before
    assert all(Path(path).read_bytes() == data for path, data in original.items())
