"""Real immutable metadata around explicit RAW/phase orchestration doubles.

These cases exercise source, registration and history joins. They do not decode
RAW, infer labels, qualify geometry, fit optics or establish numerical parity.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass import build_native_bridge as native_bridge
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json
from darsia.fluidflower.pipeline import build_raw_phase as collect


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    events, render_faults = [], {}
    raw = tmp_path / "raw"
    raw.mkdir()
    captures = []
    for label in ("baseline", "first", "second"):
        path = raw / (label + ".ARW")
        path.write_bytes(label.encode())
        captures.append(asset(path))
    native = _write_json(tmp_path / "native.json", {"original_native_parameters": True})
    absent = dict(path=str(tmp_path / "unconsumed_ancestry.json"), sha256="f" * 64)
    model = _write_json(tmp_path / "model.json", {"frozen_model_seam": True, "ancestry": absent})
    envelope_array = _write_json(tmp_path / "packed_envelope.json", {"packed_array_seam": True})
    envelopes = _write_json(tmp_path / "envelopes.json", dict(
        envelope1=envelope_array, envelope2=envelope_array, unconsumed_ancestry=absent))
    native_fields = ["masked_log_path", "masked_log_sha256", "masked_valid_path", "masked_valid_sha256"]
    temporal_fields = ["temporal_ring_path", "temporal_ring_sha256", "temporal_pad", "temporal_origin"]
    bundle = dict(documents=dict(co2=model, unconsumed_ancestry=absent),
                  requirements=dict(expected_sha256=dict(co2=model["sha256"])))
    phase = _write_json(tmp_path / "phase.json", dict(
        geometry_fields=dict(reference_to_frame="H_ac28_to_frame_native",
                             reference_to_metric="H_ac28_to_metric"),
        model_bundles=dict(native=bundle, edge=deepcopy(bundle),
            placement=dict(pixel_model=model, regional_model=model)),
        chain_assets=dict(envelopes=envelopes, floor=envelope_array, zones=envelope_array),
        parameters=dict(inputs=dict(fields=dict(native=native_fields, temporal=temporal_fields)),
            geometry=dict(rig_origin_xy=[-1, -1], native_shape_hw=[2, 3],
                          source_box_xywh=[1, 1, 3, 2]))))
    static = _write_json(tmp_path / "static.json", {"explicit_static_components_seam": True})
    geometries = [_write_json(tmp_path / ("geometry" + str(index) + ".json"), dict(
        run="sample", raw_source=capture, source_sha256=letter * 64,
        H_ac28_to_frame_native=np.eye(3).tolist(),
        area_parameters=dict(H_ac28_to_frame_native=np.eye(3).tolist(),
                             H_ac28_to_metric=[[2., 0., 5.], [0., 2., 7.], [0., 0., 1.]]),
        native_origin_xy=[-1, -1], source_native_shape_hw=[2, 3],
        source_content_box_xywh=[1, 1, 3, 2], phase_shape_hw=[4, 5],
        frame_photo_top_keep=envelope_array, unconsumed_ancestry=absent))
        for index, (capture, letter) in enumerate(zip(captures, "abc"))]
    photo = _write_json(tmp_path / "photo.json", {"explicit_photo_mask": True})
    identity = np.eye(3).tolist()
    monkeypatch.setattr(native_bridge, "inspect_local_file", lambda _path: SimpleNamespace(allowed=True))

    class Bridge:
        def __init__(self):
            self.output = tmp_path / "bridge_owner"
            self.output.mkdir()
            self.forbidden = [str(raw)]
            self.reader = DataReader()
            self.context = SimpleNamespace(run="sample", grid=dict(num_voxels=[2, 3]),
                provenance=dict(native_configuration=native, raw_acquisition=captures[0]))
            self.parameters = dict(source=dict(homography_min_scale=1e-12, homography_match_atol=1e-10),
                extension=dict(native_origin_xy=[-1, -1], native_shape_hw=[2, 3]))
            self.proof = _write_json(self.output / "bridge.json", {"explicit_bridge_seam": True})
            self.binding = _write_json(self.output / "binding.json", {"explicit_RAW_binding_seam": True})

        def _matrix(self, value):
            return collect.native_source.source_homography(value, parameters=self.parameters)

        def _revalidate(self):
            for descriptor in (self.proof, self.binding, native):
                self.reader.path(descriptor)

        def frame(self, acquisition, output):
            self.reader.path(acquisition)
            target = native_bridge._fresh(output, self.forbidden)
            events.append(("bridge", acquisition["sha256"]))
            geometry = _write_json(target / "geometry.json", dict(run="sample", acquisition=acquisition,
                baseline_to_native_h=identity, reference_to_native_h=identity))
            products = _write_json(target / "products.json", {"explicit_RAW_buffer_seam": True})
            receipt = _write_json(target / "frame.json", dict(bridge=self.proof, native_frame=geometry,
                products=products, build_to_frame_native_h=identity,
                correction=dict(renderer=dict(native_to_camera_u_h=identity, camera_u_wh=[3, 2]),
                    sources=dict(camera_arw=acquisition))))
            return native_bridge.BuildNativeFrame(dict(kind="raw_rig", geometry=geometry,
                processor=dict(raw_binding=self.binding)), products, receipt)

    monkeypatch.setattr(collect, "BuildNativeBridge", Bridge)
    bridge = Bridge()

    def native_record(spec, stage):
        stage.mkdir()
        geometry = bridge.reader.json(spec["native_task"]["geometry"])
        renderer = dict(native_to_camera_u_h=deepcopy(identity), camera_u_wh=[3, 2])
        fault = render_faults.get(stage.name)
        if fault == "homography":
            renderer["native_to_camera_u_h"][0][2] = 1.
        elif fault == "camera_grid":
            renderer["camera_u_wh"] = [4, 2]
        audit = _write_json(stage / "audit.json", dict(renderer=renderer,
            sources=dict(camera_arw=geometry["acquisition"])))
        shape = [3, 3] if fault == "native_grid" else [2, 3]
        np.save(stage / "native_log.npy", np.zeros((3, *shape), np.float32))
        np.save(stage / "native_valid.npy", np.ones(shape, bool))
        np.save(stage / "ring.npy", np.zeros((3, 1, 3), np.float32))
        return dict(source_sha256=spec["source"]["source_sha256"],
            full=dict(source_sha256=spec["source"]["source_sha256"],
                binding=dict(producer_audit=audit),
                native_centre=dict(native_shape_hw=shape,
                    log_linear_rgb=asset(stage / "native_log.npy"), valid_mask=asset(stage / "native_valid.npy")),
                ring_windows=[dict(canvas_xyxy=[0, 0, 3, 1], log_linear_rgb=asset(stage / "ring.npy"))]))

    class Acquisitions:
        def __init__(self, configuration, *, unc_only):
            assert not unc_only
            assert configuration["native"]["configuration"] == native
            assert configuration["phase"]["configuration"] == phase
            self.processors = {}

        def prepare(self, spec, stage):
            events.append(("acquisition", spec["case_id"]))
            record = native_record(spec, stage)
            np.save(stage / "masked_log.npy", np.zeros((3, 2, 3), np.float32))
            np.save(stage / "masked_valid.npy", np.ones((2, 3), bool))
            masked_log, masked_valid = asset(stage / "masked_log.npy"), asset(stage / "masked_valid.npy")
            values = {native_fields[0]: masked_log["path"], native_fields[1]: masked_log["sha256"],
                      native_fields[2]: masked_valid["path"], native_fields[3]: masked_valid["sha256"]}
            values.update(raw_acquisition=deepcopy(spec["raw_acquisition"]),
                          raw_producer_audit=record["full"]["binding"]["producer_audit"])
            row = _write_json(stage / "native_row.json", values)
            return dict(native_row=row, full=record["full"], source=deepcopy(spec["source"]),
                        current_h=deepcopy(spec["phase_roles"]["current_h"]))

    class Owner:
        def __init__(self, configuration, output, *, unc_only):
            assert not unc_only and "mass" not in configuration
            self.config, self.output, self.reader = configuration, output, DataReader()
            self.phase = self.reader.json(configuration["phase"]["configuration"])
            self.previous = deepcopy(configuration["phase"]["initial_previous"])
            self.parameters, self.model_bundle = {"original_phase_seam": True}, {"frozen_models_seam": True}
            events.append(("owner", configuration["run"]))

        def native(self, spec, stage):
            events.append(("native", spec["case_id"]))
            record = native_record(spec, stage)
            return dict(record=record, record_asset=_write_json(stage / "native.json", record))

        def frame(self, spec, native):
            events.append(("frame", spec["case_id"]))
            assert self.previous["source"]["source_sha256"] == spec["source"]["previous_source_sha256"]
            return dict(source=deepcopy(spec["source"]), case_id=spec["case_id"])

        def advance(self, frame):
            events.append(("advance", frame["case_id"]))
            self.previous = deepcopy(frame)

        def close(self):
            events.append(("close", "owner"))

    def infer(frame, previous, models, parameters):
        assert models == {"frozen_models_seam": True} and parameters == {"original_phase_seam": True}
        assert previous["source"]["source_sha256"] == frame["source"]["previous_source_sha256"]
        events.append(("infer", frame["case_id"]))
        return dict(source=deepcopy(frame["source"]), original_inference_seam=frame["case_id"])

    def phase_outputs(stage, spec, frame, inferred):
        assert inferred["original_inference_seam"] == frame["case_id"] == spec["case_id"]
        events.append(("phase_outputs", spec["case_id"]))
        product = _write_json(stage / "phase_product.json", inferred)
        return dict(complete=True, source_sha256=spec["source"]["source_sha256"], original_output_seam=product)

    monkeypatch.setattr(collect, "RawAcquisitionInputs", Acquisitions)
    monkeypatch.setattr(collect, "SeriesInputs", Owner)
    monkeypatch.setattr(collect, "infer_labels", infer)
    monkeypatch.setattr(collect, "_phase_outputs", phase_outputs)

    def specification(index, previous, *, selected):
        return dict(case_id="frame" + str(index), frame_index=index, elapsed_h=float(index),
            raw_acquisition=captures[index], geometry=geometries[index],
            source=dict(run="sample", source_sha256=captures[index]["sha256"],
                source_binding=dict(source_sha256=captures[index]["sha256"]),
                frame_index=index, elapsed_h=float(index), previous_source_sha256=previous,
                selected=selected, studio_anchor=False),
            phase_roles=dict(area=dict(role="selected", factor=1., revision={"source": "measured"}),
                current_h=deepcopy(identity), reference_to_current_h=deepcopy(identity),
                static=dict(components=static, source_ids=[1], unchanged_source=False),
                retained_source_cache=None, encoding_contract=None, light_epoch_id="saved",
                inherited_spacing_m=[.001, .001], probability_spacing_m=[.001, .001],
                placement_metadata={"explicit": True}, photo_sand_reference=photo,
                photo_proof=dict(sha256=photo["sha256"], phase_labels_used=False)))

    reference = specification(0, None, selected=False)
    config = dict(format=collect.FORMAT, run="sample", native=dict(configuration=native),
        phase=dict(configuration=phase), reference_spec=reference,
        initial_previous_spec=deepcopy(reference), temporal_pad_px=0,
        frames=[specification(1, captures[0]["sha256"], selected=True),
                specification(2, captures[1]["sha256"], selected=True)],
        history_policy="Selected RAW acquisitions in strict camera order", selection_index_field="frame_index")
    return SimpleNamespace(bridge=bridge, configuration=config, output=tmp_path / "phase_collection",
        events=events, render_faults=render_faults, captures=captures, model=model,
        raw_before={path: path.read_bytes() for path in raw.iterdir()})


def test_original_producer_order_and_exact_receipts_advance_only_after_acknowledgement(inputs):
    before = deepcopy(inputs.configuration)
    collector = collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    assert inputs.configuration == before
    assert not any(event[0] == "infer" for event in inputs.events)
    assert sum(event[0] == "acquisition" for event in inputs.events) == 1
    assert collector.reference_bridge_frame is collector.initial_previous_bridge_frame
    row = next(collector)
    assert row is collector.pending
    assert row["spec"]["raw_acquisition"] == inputs.captures[1]
    assert [event[0] for event in inputs.events[-4:]] == ["native", "frame", "infer", "phase_outputs"]
    receipt = collector.reader.json(row["phase_receipt"])
    assert receipt["complete"] and receipt["source"] == row["spec"]["source"]
    assert receipt["phase_configuration"] == inputs.configuration["phase"]["configuration"]
    assert receipt["bridge_frame"] == row["bridge_frame"].receipt
    assert receipt["products"] == row["phase_products"]
    assert "physical_context_verified" not in receipt
    with pytest.raises(RuntimeError, match="Acknowledge"):
        next(collector)
    with pytest.raises(ValueError, match="pending frame"):
        collector.commit(dict(row))
    assert not any(event[0] == "advance" for event in inputs.events)
    collector.commit(row)
    second = next(collector)
    collector.commit(second)
    with pytest.raises(StopIteration):
        next(collector)
    collector.close()
    collector.close()
    assert inputs.events.count(("close", "owner")) == 1
    assert {path: path.read_bytes() for path in inputs.raw_before} == inputs.raw_before


@pytest.mark.parametrize("change", ["native", "baseline", "source", "history", "time", "labels", "mass",
                                    "photo_sha", "predicted_photo"])
def test_incomplete_or_borrowed_inputs_are_refused_before_producer_work(inputs, change):
    config = inputs.configuration
    if change == "native":
        config["native"]["configuration"] = config["phase"]["configuration"]
    elif change == "baseline":
        config["reference_spec"] = deepcopy(config["frames"][0])
    elif change == "source":
        config["frames"][0]["source"]["source_binding"]["source_sha256"] = "f" * 64
    elif change == "history":
        config["frames"][1]["source"]["previous_source_sha256"] = "f" * 64
    elif change == "time":
        config["frames"][1]["elapsed_h"] = config["frames"][1]["source"]["elapsed_h"] = 1.
    elif change == "labels":
        config["frames"][0]["labels"] = config["phase"]["configuration"]
    elif change == "photo_sha":
        config["frames"][0]["phase_roles"]["photo_proof"]["sha256"] = "f" * 64
    elif change == "predicted_photo":
        config["frames"][0]["phase_roles"]["photo_proof"]["phase_labels_used"] = True
    else:
        config["mass"] = {"calibration": config["phase"]["configuration"]}
    with pytest.raises(ValueError):
        collect.BuildRawPhaseCollector(inputs.bridge, config, inputs.output)
    assert not inputs.events and not inputs.output.exists()


@pytest.mark.parametrize("role", ["current_h", "reference_to_current_h"])
def test_supplied_phase_homography_must_match_actual_bridge(inputs, role):
    inputs.configuration["frames"][0]["phase_roles"][role][0][2] = 1.
    with pytest.raises(ValueError, match="measured RAW bridge"):
        collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    assert not any(event[0] in {"infer", "advance", "owner"} for event in inputs.events)
    assert inputs.output.exists()


@pytest.mark.parametrize("changed", ["raw_source", "run"])
def test_geometry_must_name_its_exact_raw_acquisition_before_bridge_work(inputs, changed):
    spec = inputs.configuration["frames"][0]
    geometry = inputs.bridge.reader.json(spec["geometry"])
    assert geometry["source_sha256"] != spec["raw_acquisition"]["sha256"]
    if changed == "raw_source":
        geometry["raw_source"] = inputs.captures[2]
    else:
        geometry["run"] = "another_run"
    spec["geometry"] = _write_json(inputs.output.parent / "changed_geometry.json", geometry)
    with pytest.raises(ValueError, match="exact RAW acquisition or run"):
        collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    assert not inputs.events and not inputs.output.exists()


@pytest.mark.parametrize("fault", ["homography", "camera_grid", "native_grid"])
@pytest.mark.parametrize("stage", ["reference", "frame1"])
def test_second_original_render_must_retain_measured_bridge_registration(inputs, fault, stage):
    inputs.render_faults[stage] = fault
    if stage == "reference":
        with pytest.raises(ValueError, match="changed the bridge registration"):
            collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    else:
        collector = collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
        with pytest.raises(ValueError, match="changed the bridge registration"):
            next(collector)
        with pytest.raises(RuntimeError, match="failed"):
            next(collector)
        collector.close()
    assert not any(event[0] in {"infer", "advance"} for event in inputs.events)
    assert inputs.output.exists()


def test_changed_phase_input_is_not_reused_and_close_never_advances_pending_frame(inputs):
    collector = collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    path = Path(inputs.configuration["phase"]["configuration"]["path"])
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed|differs"):
        next(collector)
    collector.close()
    assert not any(event[0] in {"infer", "advance"} for event in inputs.events)


@pytest.mark.parametrize("changed", ["model", "masked_history", "scientific_history", "current_ring"])
def test_changed_consumed_model_or_history_file_refuses_commit_without_advancing(inputs, changed):
    if changed == "masked_history":
        inputs.configuration["initial_previous_spec"] = deepcopy(inputs.configuration["frames"][0])
        inputs.configuration["frames"] = inputs.configuration["frames"][1:]
    collector = collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    row = next(collector)
    if changed == "model":
        path = Path(inputs.model["path"])
    elif changed == "masked_history":
        assert collector.reference_bridge_frame is not collector.initial_previous_bridge_frame
        assert sum(event[0] == "acquisition" for event in inputs.events) == 2
        history = collector.reader.json(collector.configuration["phase"]["initial_previous"]["native_row"])
        path = Path(history["masked_log_path"])
    elif changed == "scientific_history":
        path = Path(collector.reference_record["full"]["native_centre"]["valid_mask"]["path"])
    else:
        path = Path(row["native"]["record"]["full"]["ring_windows"][0]["log_linear_rgb"]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed|differs"):
        collector.commit(row)
    assert collector.pending is row
    assert not any(event[0] == "advance" for event in inputs.events)
    collector.close()


def test_fresh_output_cannot_overlap_raw_or_existing_bridge_output(inputs):
    for output in (inputs.bridge.output / "nested", Path(inputs.captures[0]["path"]).parent / "nested"):
        with pytest.raises(ValueError, match="overlaps"):
            collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, output)
        assert not output.exists()
    inputs.output.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)


def test_consumed_geometry_matches_bridge_without_replacing_independent_wall_or_metric_maps(inputs):
    spec = inputs.configuration["frames"][0]
    original = inputs.bridge.reader.json(spec["geometry"])
    original["H_reference_rig_to_frame_native"] = [[1., 0., 19.], [0., 1., 23.], [0., 0., 1.]]
    spec["geometry"] = _write_json(inputs.output.parent / "independent-geometry.json", original)
    collector = collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    row = next(collector)
    actual = collector.reader.json(row["spec"]["geometry"])
    assert actual == original
    assert actual["H_reference_rig_to_frame_native"] != row["spec"]["phase_roles"]["reference_to_current_h"]
    collector.commit(row)
    assert ("advance", "frame1") in inputs.events
    collector.close()


@pytest.mark.parametrize("section", ["geometry", "area"])
def test_same_raw_stale_geometry_transform_refuses_before_any_raw_producer(inputs, section):
    spec = inputs.configuration["frames"][0]
    original = inputs.bridge.reader.json(spec["geometry"])
    target = original if section == "geometry" else original["area_parameters"]
    target["H_ac28_to_frame_native"][0][2] = 1.
    spec["geometry"] = _write_json(inputs.output.parent / "stale-geometry.json", original)
    with pytest.raises(ValueError, match="rig/area transform"):
        collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    assert not inputs.events and inputs.output.exists()


@pytest.mark.parametrize("field,value", [
    ("native_origin_xy", [-2, -1]),
    ("source_native_shape_hw", [2, 4]),
    ("source_content_box_xywh", [2, 1, 3, 2]),
])
def test_same_raw_geometry_keeps_original_native_origin_shape_and_content_box(inputs, field, value):
    spec = inputs.configuration["frames"][0]
    original = inputs.bridge.reader.json(spec["geometry"])
    original[field] = value
    spec["geometry"] = _write_json(inputs.output.parent / "different-grid.json", original)
    with pytest.raises(ValueError, match="rig origin or native shape"):
        collect.BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output)
    assert not inputs.events and inputs.output.exists()
