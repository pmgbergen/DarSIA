"""Actual saved-Build/geometry/selection joins with finite RAW producer doubles."""
from copy import deepcopy

import numpy as np
import pytest

from darsia.fluidflower.mass.build_native_bridge import BuildNativeBridge, BuildNativeFrame
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json, _write_npz
from darsia.fluidflower.pipeline import current_build_preparation as subject
from .test_local_build_inputs import _local_build_sources, local_build_case
from .test_native_optical_source import saved
from .test_staged_data import phase1_data
from .test_raw_photographic_review import inputs as review_inputs


@pytest.fixture
def local_build_sources(tmp_path):
    parameters = dict(source=dict(homography_match_atol=1e-8, homography_min_scale=1e-12,
        symbols=dict(reference_to_frame_h_key="H_reference_to_frame")), support={"fixture": "TESTONLY"})
    return _local_build_sources(saved.__wrapped__(tmp_path, native_parameters=parameters), tmp_path)


def _one_selected_frame(case, tmp_path, monkeypatch, *, unaligned=False):
    """Selection, map product and recorded calls for a baseline plus one selected frame."""
    baseline = case.sources.raw
    current = tmp_path / "current.ARW"
    current.write_bytes(b"TESTONLY current acquisition")
    current = asset(current)
    calls = []
    identity = np.eye(3).tolist()
    floor = _write_npz(tmp_path / "floor.npz", {"floor": np.zeros(3)})
    phase = dict(geometry_parameters=dict(case.geometry_parameters, observation_horizon_floor=1e-10),
        phase_shape_hw=[3, 3], parameters=dict(geometry=dict(rig_origin_xy=[0, 0],
            floor_source_origin_xy=[0, 0], source_box_xywh=[0, 0, 3, 3], native_shape_hw=[3, 3])),
        geometry_fields=dict(reference_to_frame="H_reference_to_frame", reference_to_metric="H_reference_to_metric"),
        chain_assets=dict(floor=floor))
    roles = dict(static=dict(components=floor, source_ids=[1], unchanged_source=False),
        retained_source_cache=None, encoding_contract=None, light_epoch_id="test", placement_metadata={},
        inherited_spacing_m=[.001, .001], probability_spacing_m=[.001, .001],
        photo_sand_reference=floor, photo_proof=dict(sha256=floor["sha256"], phase_labels_used=False))
    recipe = dict(phase=phase, phase_roles=roles, temporal_pad_px=0,
        native_configuration=case.configuration["native"]["configuration"], original_recipe_receipt=floor)
    monkeypatch.setattr(subject, "retained_phase_recipe", lambda *args, **kwargs: recipe)
    monkeypatch.setattr(subject, "build_photometry_binding", lambda *args, **kwargs: floor)

    class Bridge(BuildNativeBridge):
        def __init__(self, *args, **kwargs):
            self.context = case.sources.context
            self.reader = DataReader()
            self.processor = object()
        def frame(self, acquisition, output):
            calls.append(acquisition)
            if unaligned and acquisition == current:
                raise ValueError("ROIs cannot be aligned by translation.")
            output.mkdir()
            geometry = _write_json(output / "native.json", dict(run=self.context.run,
                acquisition=acquisition, reference_to_native_h=identity, baseline_to_native_h=identity))
            audit = deepcopy(case.receipt["correction"])
            audit["sources"]["camera_arw"] = acquisition
            audit["renderer"]["camera_u_wh"] = [3, 3]
            receipt = _write_json(output / "frame.json", dict(case.receipt, native_frame=geometry,
                physical_context=self.context.provenance, correction=audit))
            return BuildNativeFrame(dict(kind="raw_rig", geometry=geometry,
                processor=dict(raw_binding=floor)), _write_npz(output / "raw.npz",
                {"valid_mask": np.ones((3, 3), bool)}), receipt)
    monkeypatch.setattr(subject, "BuildNativeBridge", Bridge)
    prepared_order = []
    class Acquisition:
        def __init__(self, configuration, **kwargs):
            self.processors = {}
        def prepare(self, spec, stage):
            assert set(self.prepared_raw) == {spec["case_id"]}
            assert len(calls) == len(prepared_order) + 1
            assert spec["native_task"] == self.prepared_raw[spec["case_id"]].native_task
            prepared_order.append(spec["case_id"])
            stage.mkdir()
            _write_json(stage / "native.json", dict(source=spec["source"]))
            return dict(source=spec["source"])
    monkeypatch.setattr(subject, "RawAcquisitionInputs", Acquisition)
    series = _write_json(tmp_path / "series.json", {"test": True})
    row = lambda raw, index, elapsed, previous, reference: dict(raw_path=raw["path"],
        raw_sha256=raw["sha256"], recorded_raw_size_bytes=raw["size_bytes"],
        current_frame_index=index, elapsed_h=elapsed, previous_raw_sha256=previous, baseline=reference)
    selection = _write_json(tmp_path / "selection.json", dict(format="ffac-local-p1-selection-data-v1",
        decision=dict(actor="Reviewer", reason="Retain selected history"), history_policy="original history",
        input_authority=dict(current_series=series, phase_plan=floor), runs=[dict(run=case.sources.context.run,
            baseline=row(baseline, 0, -1., None, True), initial_previous_raw_sha256=baseline["sha256"],
            selected_frames=[row(current, 1, .125, baseline["sha256"], False)])]))
    product = _write_npz(tmp_path / "map.npz", dict(array=case.sources.labels, inside=np.ones((3, 3), bool)))
    return selection, product, calls, baseline, current, prepared_order


def test_current_build_produces_own_raw_geometry_and_preserves_early_history(local_build_case, tmp_path, monkeypatch):
    case = local_build_case
    selection, product, calls, baseline, current, prepared_order = _one_selected_frame(case, tmp_path, monkeypatch)
    actual = subject.prepare_current_build_acquisitions(case.sources.operation_result, case.sources.metric_model,
        selection, tmp_path / "prepared", map_product=product)
    data = actual["configuration"]
    assert actual["left_out"] == []
    assert calls == [baseline, current]
    assert list(actual["prepared_native"]) == prepared_order
    assert data["reference_spec"] == data["initial_previous_spec"]
    assert data["frames"][0]["elapsed_h"] == .125
    assert data["frames"][0]["source"]["previous_source_sha256"] == baseline["sha256"]
    assert data["frames"][0]["geometry"] != data["reference_spec"]["geometry"]
    geometry = DataReader().json(data["frames"][0]["geometry"])
    assert geometry["raw_source"] == current
    assert geometry["build_geometry"]["operation_result"] == case.sources.operation_result


def test_unaligned_selected_frame_stops_the_run_unless_left_out(local_build_case, tmp_path, monkeypatch):
    case = local_build_case
    selection, product, calls, baseline, current, prepared_order = _one_selected_frame(
        case, tmp_path, monkeypatch, unaligned=True)
    arguments = (case.sources.operation_result, case.sources.metric_model, selection)
    with pytest.raises(ValueError, match="ROIs cannot be aligned"):
        subject.prepare_current_build_acquisitions(*arguments, tmp_path / "default", map_product=product,
                                                   assemble_phase=False)
    calls.clear()
    prepared_order.clear()
    actual = subject.prepare_current_build_acquisitions(*arguments, tmp_path / "left-out", map_product=product,
        assemble_phase=False, leave_out_unaligned=True)
    assert calls == [baseline, current]
    assert [row["raw_sha256"] for row in actual["left_out"]] == [current["sha256"]]
    assert actual["left_out"][0]["reason"].startswith("ROIs cannot be aligned")
    assert actual["configuration"]["frames"] == [] and list(actual["acquisitions"]) == prepared_order


def test_training_materialization_uses_original_review_writer_and_keeps_source_products(tmp_path, review_inputs):
    reader = DataReader()
    inputs = review_inputs
    reference = reader.json(inputs["reference"]["acquisition_manifest"])["records"][0]
    records, rows, produced, natives, frames = {}, [], {}, {}, []
    raw_records = [reference, *[reader.json(row["native_record"]) for row in inputs["frames"]]]
    for index, record in enumerate(raw_records):
        full = record["full"]
        digest = full["source_sha256"]
        case = "ac52-" + digest[:20]
        if index:
            audit = record["audit"]
        else:
            audit = reader.json(record["native_row"])["raw_producer_audit"]
        valid = full["native_centre"]["valid_mask"]
        native_row = _write_json(tmp_path / (case + "_row.json"), dict(raw_producer_audit=audit,
            support_path=valid["path"], support_sha256=valid["sha256"]))
        acquisition = dict(full=full, native_row=native_row, source=dict(run="ac52"))
        records[case] = acquisition
        rows.append(dict(raw_sha256=digest, elapsed_h=index - 1., baseline=index == 0))
        geometry = _write_json(tmp_path / (case + "_geometry.json"),
            dict(run="ac52", acquisition=dict(sha256=digest)))
        products = _write_npz(tmp_path / (case + "_products.npz"), dict(value=np.ones((1, 1))))
        receipt = _write_json(tmp_path / (case + "_receipt.json"), dict(native_frame=geometry, products=products))
        produced[case] = BuildNativeFrame(dict(geometry=geometry, processor=dict(raw_binding=inputs["binding"])),
            products, receipt)
        natives[case] = _write_json(tmp_path / (case + "_native.json"), dict(complete=True,
            processor_kind="raw_rig", source_sha256=digest, audit=audit, full=full))
        if index:
            frames.append(dict(case_id=case, geometry=geometry))
    selection = _write_json(tmp_path / "selection.json", dict(runs=[dict(run="ac52", baseline=rows[0], selected_frames=rows[1:])]))
    operation = _write_json(tmp_path / "build.json", dict(real_source_fixture=True))
    metric = _write_json(tmp_path / "metric.json", dict(area_factor=1.))
    prepared = dict(configuration=dict(run="ac52", frames=frames),
        phase_recipe=dict(native_configuration=inputs["configuration"], original_photo={},
            phase=dict(parameters=dict(inputs=dict(fields=dict(native=["image", "sha", "support_path", "support_sha256"]))))),
        produced=produced, acquisitions=records, prepared_native=natives, configuration_binding=operation)
    old = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    output = tmp_path / "prepared"
    output.mkdir()
    result = subject.materialize_current_build_review(operation, metric, selection, output, tmp_path / "review",
        preparer=lambda *args, **kwargs: prepared)
    manifest = reader.json(result["manifest"])
    assert [frame["frame_index"] for frame in manifest["frames"]] == [0, 1]
    assert manifest["modes"] == ["raw_detail"]
    assert result["phase_inference_performed"] is False
    assert inputs["renders"] == []  # No camera-U decode just to make training metadata.
    basis = _write_json(tmp_path / "basis.json", dict(runs=[dict(run="ac52",
        source_manifest=result["manifest"], raw_binding=inputs["binding"])]))
    cache = subject._review_source_cache(dict(image_basis=basis), "ac52", operation, metric, reader)
    assert set(cache) == {row["raw_sha256"] for row in rows}
    other = _write_json(tmp_path / "other_build.json", dict(other_build=True))
    with pytest.raises(ValueError, match="another Build"):
        subject._review_source_cache(dict(image_basis=basis), "ac52", other, metric, reader)
    assert all(path.read_bytes() == data for path, data in old.items())


@pytest.mark.parametrize("source_only", [True, False])
def test_current_recipe_never_reads_the_old_classifier_or_donor_roles(tmp_path, source_only):
    """Physical source metadata stays bound; old model bundles are absent."""
    from darsia.fluidflower.pipeline.current_phase_recipe import retained_phase_recipe
    data = _write_npz(tmp_path / "physical.npz", dict(mask=np.ones((3, 3), bool)))
    photo = _write_json(tmp_path / "photo.json", dict(phase_labels_used_for_mask=False,
        photo_sand_reference_mask_path=data["path"], photo_sand_reference_mask_sha256=data["sha256"]))
    geometry = dict(rig_origin_xy=[0, 0], native_shape_hw=[3, 3], source_box_xywh=[0, 0, 3, 3])
    native = _write_json(tmp_path / "native.json", dict(source=dict(asset_aliases={},
        symbols=dict(reference_to_frame_h_key="reference_h")), parameters=dict(geometry=geometry),
        geometry_parameters=dict(fields=dict(reference_metric_field="metric_h")),
        barrier_parameters={}, chain_assets=dict(floor_curve=data, zones=data)))
    source = _write_json(tmp_path / "source.json", dict(run="sample",
        frame_sources=[dict(source_sha256="a" * 64, elapsed_h=1.)]))
    native_manifest = _write_json(tmp_path / "manifest.json", dict(v5_track1_photo_geometry=dict(
        frames=[dict(receipt_path=photo["path"], receipt_sha256=photo["sha256"])])))
    barrier = _write_json(tmp_path / "barrier.json", dict(source_component_ids=[1]))
    keys = ("contract", "spec_sha256", "run", "light_epoch_id", "light_epoch_table", "cpu_only",
        "fitting_performed", "training_labels_read", "validation_annotations_read", "floor_rule_applied",
        "frozen_co2_head_sha256", "co2_inference_performed")
    placement = {key: None for key in keys}
    placement["light_epoch_id"] = "original-source-epoch"
    folder = tmp_path / "selected_phase_frames" / "sample"
    folder.mkdir(parents=True)
    _write_json(folder / ("a" * 64 + ".json"), dict(run="sample", source_sha256="a" * 64,
        elapsed_h=1., qa=dict(p1=placement, native_inference=dict(
            historical_stage_conventions=dict(inherited_map=dict(spacing_m=[.001, .001]),
                g3_probability=dict(spacing_m=[.001, .001])),
            input_provenance=dict(baseline_temporal=dict(policy=dict(pad_px=0)))))))
    plan = _write_json(tmp_path / "plan.json", dict(runs=dict(sample=dict(source_manifest=source,
        parent_native_manifest=native_manifest)), frozen_barrier_reference=dict(components=data,
        geometry_manifest=barrier), light_epochs=dict(run_to_epoch=dict(sample="original-source-epoch")),
        phase_grid=dict(shape_hw=[3, 3], native_origin_xy=[0, 0],
            source_native_shape_hw=[3, 3], source_content_box_xywh=[0, 0, 3, 3])))
    selected = None if source_only else dict(model_root="actual-current-classifier")
    result = retained_phase_recipe(native, plan, "sample", source_only=source_only, selected_model=selected)
    expected = "source-only-no-inference" if source_only else "fresh-native-logistic"
    assert result["phase"]["model_bundles"]["family"] == expected
    assert result["phase_roles"]["photo_sand_reference"]["sha256"] == data["sha256"]
    assert "native" not in result["phase"]["model_bundles"]
