"""GUI packet metadata for the explicit new-placement photo-edge policy."""
from dataclasses import replace

import numpy as np

from . import photo_edge_continuation as continuation
from .painting_preview_frame_domain import _statistics

KEY = "photo_edge_continuation"


def attach(result, source_layers, contract):
    """Keep packet buffers/provenance; add exact native counts/display masks."""
    import cv2
    if (result.source_run != contract["source_run"] or result.run_id != contract["target_run"]
            or contract["target_grid"] != "camera_u"):
        raise ValueError("The preview continuation belongs to another run or grid.")
    from .painting_run_preview import _same_path
    models = (result.geometry_binding or {}).get("models") or {}
    if (not _same_path(result.photo_path, contract["target_model"]["photograph"]["path"])
            or any((models.get(contract[side + "_run"]) or {}).get("model") != contract[side + "_model"]["descriptor"]
                   for side in ("source", "target"))):
        raise ValueError("The continuation preview belongs to another photograph/model binding.")
    source_h, target_h, _source_shape, shape, _frame, corners = continuation.validate_contract(contract, verify_files=True)
    if tuple(result.native_shape_hw or ()) != shape:
        raise ValueError("The continuation preview native grid differs from its own-frame model.")
    replay = continuation.apply_policy(source_layers, contract)
    display_shape = tuple(result.display_shape_hw or ())
    if display_shape != tuple(result.layers.shape) or display_shape != tuple(result.photo.shape[:2]):
        raise ValueError("The continuation preview display grids differ.")
    sy, sx = display_shape[0] / shape[0], display_shape[1] / shape[1]
    expected = np.asarray(((sx, 0., (sx - 1.)/2.), (0., sy, (sy - 1.)/2.), (0., 0., 1.)))
    if not np.array_equal(np.asarray(result.H_native_to_display), expected):
        raise ValueError("The continuation preview lacks its exact producer resize mapping.")
    def shown(array):
        if array.shape == display_shape:
            value = array.copy()
        else:
            value = cv2.resize(array.astype(np.uint8), display_shape[::-1], interpolation=cv2.INTER_NEAREST)
            if array.dtype == np.dtype(bool):
                value = value.astype(bool)
        value.setflags(write=False)
        return value
    if (not np.array_equal(shown(replay["labels"]), result.layers)
            or not np.array_equal(shown(replay["filled"]), result.uncovered)):
        raise ValueError("The continuation preview differs from its original categorical carrier.")
    from .painting_preview_frame_domain import _project
    target_to_mm = np.linalg.inv(target_h)
    statistics = _statistics(replay["gaps"], target_to_mm, _project(corners, target_to_mm))
    value = dict(format=continuation.FORMAT, policy=continuation.POLICY, adopted=False,
        authority=contract["frame_authority"]["kind"], frame_sha256=contract["frame_sha256"],
        source_run=contract["source_run"], target_run=contract["target_run"],
        display_inside=shown(replay["inside"]), display_continued=shown(replay["continued"]),
        display_gaps=shown(replay["gaps"]), **replay["counts"], **statistics)
    measurements = dict(result.measurements)
    measurements[KEY] = value
    return replace(result, measurements=measurements)
