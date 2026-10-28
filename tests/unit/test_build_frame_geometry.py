"""Small real saved-Build geometry seam, with explicit TESTONLY RAW receipts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass.n1_pipeline import _write_json, _write_npz
from darsia.fluidflower.pipeline.build_frame_geometry import prepare_build_frame_geometry
from darsia.fluidflower.pipeline.series import SeriesInputs
from .test_local_build_inputs import local_build_sources, local_build_case
from .test_staged_data import phase1_data


def test_actual_build_masks_and_area_reach_original_geometry_loader(local_build_case, tmp_path):
    case = local_build_case
    config = deepcopy(case.configuration)
    config["phase"]["configuration"] = _write_json(tmp_path / "phase.json", dict(
        geometry_parameters=dict(case.geometry_parameters, observation_horizon_floor=1e-10),
        phase_shape_hw=[3, 3], parameters=dict(geometry=dict(
            rig_origin_xy=[0, 0], source_box_xywh=[0, 0, 3, 3], native_shape_hw=[3, 3])),
        geometry_fields=dict(reference_to_frame="H_reference_to_frame", reference_to_metric="H_reference_to_metric")))
    build = case.make_adapter(config)
    receipt = deepcopy(case.receipt)
    receipt["correction"]["renderer"]["camera_u_wh"] = [3, 3]
    receipt_binding = _write_json(tmp_path / "frame.json", receipt)
    valid = np.ones((3, 3), bool)
    valid[1, 1] = False
    native = SimpleNamespace(native_task=case.spec["native_task"], receipt=receipt_binding,
        products=_write_npz(tmp_path / "native.npz", {"valid_mask": valid}))
    inside = np.ones((3, 3), bool)
    inside[2, 0] = False
    product = _write_npz(tmp_path / "map.npz", {"array": case.sources.labels, "inside": inside})
    produced = prepare_build_frame_geometry(build, native, map_product=product, output=tmp_path / "geometry")
    owner = SeriesInputs.__new__(SeriesInputs)
    owner.phase, owner.reader = build.phase, build.reader
    row = build.reader.json(produced)
    context, observed, top = owner.geometry(row, valid, receipt["correction"])
    expected = inside & (case.sources.labels != 4)
    np.testing.assert_array_equal(context.rig_domain, expected)
    np.testing.assert_array_equal(context.hardware_excluded, inside & (case.sources.labels == 4))
    np.testing.assert_array_equal(context.modelled, expected & valid)
    np.testing.assert_array_equal(context.filled_observed, expected & ~valid)
    np.testing.assert_array_equal(context.pixel_area_mm2, np.ones((3, 3), np.float32))
    np.testing.assert_array_equal(top, inside)
    assert observed.all()
    assert row["raw_source"] == case.sources.raw
    changed = deepcopy(receipt["correction"])
    changed["sources"]["camera_arw"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="RAW acquisition"):
        owner.geometry(row, valid, changed)
