"""Saved producer lineage and exact-bit gates; no RAW/scientific payload loading."""

import hashlib
import json

import numpy as np
import pytest

from darsia.fluidflower import normalisation_recipe as recipe


@pytest.fixture
def lineage(tmp_path):
    def write(name, data):
        path = tmp_path / name
        raw = json.dumps(data).encode()
        path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    pins = dict(raw="1"*64, fit="2"*64, old="3"*64, log="4"*64, valid="5"*64,
                frame_raw="a"*64, frame_log="b"*64, frame_valid="c"*64)
    sidecar = write("ac28_baseline.json", dict(run="ac28", npz_sha256=pins["fit"], source=dict(source_arw_sha256=pins["raw"])))
    producer = dict(contract="saved-source-producer", calibration_manifest_path="", calibration_manifest_sha256="")
    fit = write("fit.json", dict(contract="ffac-v4r5-interior-local-old-B-residual-calibration43-v1",
        results=[dict(run="ac28", baseline_path=str(tmp_path/"ac28_baseline.npz"), baseline_sha256=pins["fit"], sidecar_sha256=sidecar["sha256"]),
                 dict(run="virtual_reference", baseline_path=str(tmp_path/"reference.npz"), baseline_sha256="6"*64, sidecar_sha256="7"*64)],
        old_B_local_residual_fields=[dict(run="ac28", field_path=str(tmp_path/"old.npz"), field_sha256=pins["old"])],
        assets_path=str(tmp_path/"assets.npz"), assets_sha256="8"*64, fitter_sha256="9"*64))
    producer.update(calibration_manifest_path=fit["path"], calibration_manifest_sha256=fit["sha256"])
    frame_data = dict(run="ac28", frame_index=0, processor=producer, source=dict(camera_arw_sha256=pins["raw"]),
        certificate=dict(old_field_sha256=pins["old"]), log_linear_rgb_sha256=pins["log"],
        log_linear_rgb_shape=[3, 2, 3], log_linear_rgb_dtype="float32")
    frame = write("frame.json", frame_data)
    later_frame_data = dict(frame_data, frame_index=1,
        source=dict(camera_arw_sha256=pins["frame_raw"]), log_linear_rgb_sha256=pins["frame_log"])
    later_frame = write("later_frame.json", later_frame_data)
    output = write("output.json", dict(frame_sources=[dict(frame_index=0,
        v5_frame_receipt_path=frame["path"], v5_frame_receipt_sha256=frame["sha256"],
        v5_raw_path=str(tmp_path/"baseline.ARW"), v5_raw_sha256=pins["raw"],
        v5_log_linear_rgb_path=str(tmp_path/"output.npy"), v5_log_linear_rgb_sha256=pins["log"],
        v5_log_linear_valid_mask_path=str(tmp_path/"mask.npy"), v5_log_linear_valid_mask_sha256=pins["valid"]),
        dict(frame_index=1, v5_frame_receipt_path=later_frame["path"], v5_frame_receipt_sha256=later_frame["sha256"],
        v5_raw_path=str(tmp_path/"later.ARW"), v5_raw_sha256=pins["frame_raw"],
        v5_log_linear_rgb_path=str(tmp_path/"later.npy"), v5_log_linear_rgb_sha256=pins["frame_log"],
        v5_log_linear_valid_mask_path=str(tmp_path/"later_mask.npy"), v5_log_linear_valid_mask_sha256=pins["frame_valid"]) ]))
    batch = write("batch.json", dict(format="ffac-v5-native-materialization-batch-v1",
        revisions=[dict(run="ac28", source_group="current43", materialized_manifest_path=output["path"], materialized_manifest_sha256=output["sha256"])],
        processor_identities=dict(current43=producer), training_eligibility="validation-only"))
    training_data = dict(settings=dict(v5_raw_photometry=dict(format="ffac-v5-raw-full-native-training-v1",
        ac28_light_anchor=True, materialization_receipt_path=batch["path"], materialization_receipt_sha256=batch["sha256"])))
    training = write("training.json", training_data)
    return training, sidecar, pins


def test_joins_real_saved_producer_and_retains_unverified_scientific_pointers(lineage):
    training, _sidecar, pins = lineage
    plan = recipe.assemble_saved_method(training, "ac28")
    assert plan["producer_contract"] == "saved-source-producer"
    assert plan["baseline_raw"]["sha256"] == pins["raw"]
    assert plan["frame_index"] == 0 and plan["frame_raw"] == plan["baseline_raw"]
    assert plan["expected_log_rgb"]["sha256"] == pins["log"]
    assert plan["old_field"]["sha256"] == pins["old"]
    assert plan["apply_order"].index("old-B scalar residual after frame-light") > plan["apply_order"].index("original native correction")
    assert len(plan["verified_metadata"]) == 6
    assert not plan["ready"] and plan["pending_gates"]
    assert plan["numerical_status"] == "not_run"
    # None of the referenced RAW/NPZ/NPY files exists in this metadata fixture.


def test_later_frame_keeps_original_baseline_and_its_own_saved_output(lineage):
    training, _sidecar, pins = lineage
    plan = recipe.assemble_saved_method(training, "ac28", frame_index=1)
    assert plan["frame_index"] == 1
    assert plan["baseline_raw"]["sha256"] == pins["raw"]
    assert plan["expected_baseline"]["sha256"] == pins["fit"]
    assert plan["frame_raw"]["sha256"] == pins["frame_raw"]
    assert plan["expected_log_rgb"]["sha256"] == pins["frame_log"]
    assert plan["expected_valid_mask"]["sha256"] == pins["frame_valid"]
    assert len(plan["verified_metadata"]) == 7
    assert not plan["ready"] and plan["numerical_status"] == "not_run"


@pytest.mark.parametrize("frame_index", [-1, True, 1.5, "1", 2])
def test_invalid_or_missing_frame_is_not_replaced_by_baseline(lineage, frame_index):
    training, _sidecar, _pins = lineage
    with pytest.raises(ValueError, match="frame"):
        recipe.assemble_saved_method(training, "ac28", frame_index=frame_index)


def test_later_frame_receipt_is_checked_even_when_baseline_is_unchanged(lineage, tmp_path):
    training, _sidecar, _pins = lineage
    (tmp_path / "later_frame.json").write_text('{}')
    assert recipe.assemble_saved_method(training, "ac28")["frame_index"] == 0
    with pytest.raises(ValueError, match="differs from its exact binding"):
        recipe.assemble_saved_method(training, "ac28", frame_index=1)


def test_changed_source_metadata_is_not_silently_rebound(lineage):
    training, sidecar, _pins = lineage
    from pathlib import Path
    Path(sidecar["path"]).write_text('{}')
    with pytest.raises(ValueError, match="differs from its exact binding"):
        recipe.assemble_saved_method(training, "ac28")


@pytest.mark.parametrize("variant", ["value", "dtype", "shape", "signed_zero", "nan_payload"])
def test_exact_array_gate_rejects_every_bit_or_representation_change(variant):
    expected = np.array([0., 1., np.nan], np.float32)
    actual = expected.copy()
    if variant == "value":
        actual.view(np.uint32)[1] += 1
    elif variant == "dtype":
        actual = actual.astype(np.float64)
    elif variant == "shape":
        actual = actual.reshape(1, 3)
    elif variant == "signed_zero":
        actual[0] = -0.
    else:
        actual.view(np.uint32)[2] ^= 1
    with pytest.raises(ValueError, match="parity failed"):
        recipe.require_exact_array(actual, expected)
    recipe.require_exact_array(expected.copy(), expected)
