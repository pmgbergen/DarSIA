"""Actual finite saved Build/source replay; no RAW decoder or scientific job."""
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower.pipeline.local_overflow_sources import SavedLocalBuildInputs
from darsia.fluidflower.mass.n1_pipeline import asset, _write_json, _write_npz
from darsia.fluidflower.mass.production_precision import PixelMath
from .test_local_build_inputs import local_build_sources, local_build_case, _prepared_case
from .test_staged_data import phase1_data
from ..fixtures.tiny_saved_model_fixture import tiny_config


def saved_case(case, tmp_path):
    live, config, frame, inferred, spec = _prepared_case(case)
    labels = tmp_path / "labels.png"
    Image.fromarray(inferred["labels"]).save(labels)
    record = dict(case_id=case.spec["case_id"], source=deepcopy(case.spec["source"]),
        native=deepcopy(case.native["record"]), phase=dict(labels=asset(labels),
            classes=_write_npz(tmp_path / "classes.npz", inferred["classes"]),
            qa=_write_json(tmp_path / "qa.json", inferred["qa"]),
            area=dict(_write_npz(tmp_path / "area.npz", dict(pixel_area_mm2=frame["pixel_area_mm2"])),
                      array_key="pixel_area_mm2")))
    replay = SavedLocalBuildInputs(config, operation_result=case.sources.operation_result,
        metric_model=case.sources.metric_model, frame_receipts=case.frame_receipts)
    mass = dict(config["statistics"], run=config["run"], frames=config["frames"], data_aliases=())
    replay.validate_mass_config(mass, statistics_only=True)
    return live, replay, record, spec


def test_saved_physical_replay_preserves_original_fields_and_coordinates(local_build_case, tmp_path):
    case = local_build_case
    live, replay, record, spec = saved_case(case, tmp_path)
    replay.restore(case.spec, record, spec)
    expected = live.physical_fields(spec, (3, 3), PixelMath("float32"))
    actual = replay.physical_fields(spec, (3, 3), PixelMath("float32"))
    for field in ("facies", "porosity", "depth_m", "swi", "physical_supported_mask"):
        np.testing.assert_array_equal(actual[field], expected[field])
    for actual_axis, expected_axis in zip(replay.coordinates(spec, (3, 3)), live.coordinates(spec, (3, 3))):
        np.testing.assert_array_equal(actual_axis, expected_axis)
    assert replay._pending["excluded_masks"].keys() == live._pending["excluded_masks"].keys()


@pytest.mark.parametrize("fault", ["source", "build-proof", "physical-bytes"])
def test_saved_replay_refuses_changed_original_source_or_physics(local_build_case, tmp_path, fault):
    case = local_build_case
    _, replay, record, spec = saved_case(case, tmp_path)
    if fault == "source":
        record["source"]["source_sha256"] = "f" * 64
    elif fault == "build-proof":
        value = replay.reader.json(record["phase"]["qa"])
        value["local_build"]["metric_model"]["sha256"] = "f" * 64
        record["phase"]["qa"] = _write_json(tmp_path / "wrong-qa.json", value)
    else:
        case.receipt_path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        replay.restore(case.spec, record, spec)
