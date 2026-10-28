"""Post-run wiring with private RAW assets; no scientific worker is launched."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from darsia.fluidflower import jobs_panel as gui
from darsia.fluidflower import raw_review_postrun as post
from darsia.fluidflower.native import raw_rig_source
from darsia.fluidflower.pipeline.job_inputs import CATALOG_FORMAT, catalog_order_key
from darsia.fluidflower.published_dataset.job_reader import Product
from darsia.fluidflower.store_map import StoreMap, input_reference
from ..test_raw_photographic_review import inputs, _write_json, _write_npy
from ..test_raw_review_baseline_classes import source as baseline_source  # noqa: F401


def _npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    return post._pin(path)


@pytest.fixture
def finished_inputs(tmp_path, monkeypatch, inputs):
    """Real writer/support/RAW validators, injected terminal-reader and camera/display seams."""
    acquisitions = json.loads(Path(inputs["reference"]["acquisition_manifest"]["path"]).read_bytes())
    reference = acquisitions["records"][0]
    run = reference["source"]["run"]
    baseline_audit = json.loads(Path(reference["native_row"]["path"]).read_bytes())["raw_producer_audit"]
    baseline = json.loads(Path(baseline_audit["path"]).read_bytes())
    shape = tuple(reference["full"]["native_centre"]["native_shape_hw"])
    h = deepcopy(baseline["H_ac28_to_frame_native"])
    support = dict(grid_shape_hw=list(shape), grid_archive_key="source_grid",
        cell_to_reference_native=np.eye(3).tolist(), guard_px=0, fill_policy="fill",
        keep_policy="keep", keep_contract="kept", fill_contract="filled",
        hole_max_area_px=0, connectivity=8, vertical_gap_max_px=1,
        vertical_gap_rule="bounded", homography_epsilon=1e-12)
    native = json.loads(Path(inputs["configuration"]["path"]).read_bytes())
    native.update(support=support, source=dict(profiles={run: "late"}, source_id_pattern=r"[a-z][a-z0-9]*",
        native_shape_hw=list(shape), homography_min_scale=1e-12, homography_match_atol=1e-10,
        symbols=dict(reference_to_frame_h_key="reference_h")))
    next_configuration = _write_json(Path(inputs["configuration"]["path"]), native)
    inputs["configuration"].clear()
    inputs["configuration"].update(next_configuration)
    dummy = _write_npy(tmp_path / "model.npy", np.ones((1,), np.float32))
    sidecar = _write_json(tmp_path / "model.json", {})
    binding_value = dict(contract=raw_rig_source.CONTRACT, run=run, profile_id="late",
        native_shape_hw=list(shape), baseline_raw=baseline["sources"]["camera_arw"],
        baseline_reference_to_native_h=h, shape_assets=[dummy], image_metadata={},
        models=dict(fit=dict(data=dummy, sidecar=sidecar),
                    reference=dict(data=dummy, sidecar=sidecar), old_field=None),
        execution_identity=raw_rig_source.execution_identity())
    raw_binding = _write_json(Path(inputs["binding"]["path"]), binding_value)
    phase = _write_json(tmp_path / "phase.json", dict(parameters=dict(inputs=dict(source_sha_field="source_sha256"))))
    photo = _npz(tmp_path / "photo.npz", source_grid=np.ones(shape, bool))
    photo_roles = dict(current_h=np.eye(3).tolist(), reference_to_current_h=h,
        photo_sand_reference=photo, photo_proof=dict(phase_labels_used=False, sha256=photo["sha256"]))
    baseline.update(cpu_only=True, training=False, reference_h=h,
        processor=dict(contract=raw_rig_source.CONTRACT, source_binding=raw_binding))
    baseline_audit = _write_json(Path(baseline_audit["path"]), baseline)
    raw_sha = baseline["sources"]["camera_arw"]["sha256"]
    reference["source"] = dict(run=run, source_sha256=raw_sha)
    reference["full"]["binding"] = dict(producer_audit=baseline_audit)
    reference["native_row"] = _write_json(Path(reference["native_row"]["path"]),
        dict(contract="darsia-raw-phase-acquisition-v1", source_kind="raw",
            source_sha256=raw_sha, raw_acquisition=baseline["sources"]["camera_arw"],
            raw_producer_audit=baseline_audit))
    reference["current_h"] = np.eye(3).tolist()
    baseline_geometry = _write_json(tmp_path / "baseline_geometry.json",
        dict(contract=raw_rig_source.FRAME_CONTRACT, run=run,
             acquisition=baseline["sources"]["camera_arw"], reference_to_native_h=h,
             baseline_to_native_h=np.eye(3).tolist()))
    preparation_config = _write_json(tmp_path / "original_preparation.json",
        dict(native=dict(configuration=inputs["configuration"]), phase=dict(configuration=phase),
            acquisitions=[dict(source=reference["source"], native_task=dict(kind="raw_rig",
                processor=dict(raw_binding=raw_binding), geometry=baseline_geometry),
                phase_roles=photo_roles)], data_aliases=[]))
    acquisitions.update(format="darsia-raw-acquisition-preparation-v1", run=run,
        configuration=preparation_config, adopted=False)
    preparation = _write_json(Path(inputs["reference"]["acquisition_manifest"]["path"]), acquisitions)
    products, results, frames = {}, {}, []
    catalogue = {}
    def add(bound, use, role):
        ref = input_reference(bound["path"], {"series": str(tmp_path)},
            sha256=bound["sha256"], size_bytes=Path(bound["path"]).stat().st_size, role=role)
        key = tuple(ref.items())
        entry = catalogue.setdefault(key, dict(reference=ref, uses=[]))
        entry["uses"].append(use)
        return ref
    def product(bound):
        key = "product-" + str(len(products))
        products[key] = Product(key, "fixture-host", bound["path"], bound["sha256"],
            Path(bound["path"]).stat().st_size, "map", None)
    add(inputs["configuration"], "native.configuration", "configuration")
    add(phase, "phase.configuration", "configuration")
    add(reference["native_row"], "phase.reference.native_row", "configuration")
    add(baseline_audit, "phase.reference.raw_audit", "configuration")
    add(baseline["sources"]["camera_arw"], "phase.reference.raw_acquisition", "image")
    for leaf, role, use in (("log_linear_rgb", "image", "log"), ("valid_mask", "rig_model", "valid")):
        add(reference["full"]["native_centre"][leaf], "phase.reference.full.centre." + use, role)
    for index, original in enumerate(inputs["frames"]):
        case = original["case_id"]
        document = json.loads(Path(original["native_record"]["path"]).read_bytes())
        audit = json.loads(Path(document["audit"]["path"]).read_bytes())
        audit.update(reference_h=audit["H_ac28_to_frame_native"], cpu_only=True, training=False,
            processor=dict(contract=raw_rig_source.CONTRACT, source_binding=raw_binding))
        document["audit"] = _write_json(Path(document["audit"]["path"]), audit)
        native_record = _write_json(Path(original["native_record"]["path"]), document)
        geometry = _write_json(Path(original["native_geometry"]["path"]),
            dict(contract=raw_rig_source.FRAME_CONTRACT, run=run,
                acquisition=audit["sources"]["camera_arw"],
                reference_to_native_h=audit["reference_h"],
                baseline_to_native_h=audit["H_v5_B_to_frame_native"]))
        roles = dict(photo_roles, reference_to_current_h=audit["reference_h"])
        spec = dict(case_id=case, frame_index=10 + index * 3, elapsed_h=original["elapsed_h"],
            source=dict(source_sha256=document["source_sha256"], run=run),
            native_task=dict(kind="raw_rig", geometry=geometry, processor=dict(raw_binding=raw_binding)),
            geometry=original["rig_geometry"], phase_roles=roles)
        frames.append(spec)
        classes = _npz(Path(native_record["path"]).parent / "classes.npz",
            rig_domain=np.ones(document["full"]["phase_shape_hw"], bool),
            modelled=np.ones(document["full"]["phase_shape_hw"], bool),
            filled_observed=np.zeros(document["full"]["phase_shape_hw"], bool),
            filled_unobserved=np.zeros(document["full"]["phase_shape_hw"], bool))
        stage = dict(format="darsia-series-frame-v1", complete=True, case_id=case,
            source=spec["source"], native=document, phase=dict(classes=classes),
            files={"native.json": native_record})
        results[case] = dict(prediction=dict(pipeline=stage))
        for bound in [native_record, document["audit"], classes,
                      *document["full"]["native_centre"].values()]:
            if isinstance(bound, dict) and "path" in bound:
                product(bound)
        use = "frame." + case
        for bound, suffix, role in [(geometry, ".native_geometry", "rig_model"),
            (spec["geometry"], ".rig_geometry", "rig_model"), (raw_binding, ".native.raw_binding", "configuration"),
            (audit["sources"]["camera_arw"], ".native.acquisition", "image"),
            (photo, ".phase.photo_sand_reference", "rig_model")]:
            add(bound, use + suffix, role)
    config = dict(format="darsia-native-phase-n1-run-v1", run=run, frames=frames,
        native=dict(configuration=inputs["configuration"]),
        phase=dict(configuration=phase, reference=reference))
    configuration = _write_json(tmp_path / "run.json", config)
    configuration_ref = add(configuration, "run.configuration", "configuration")
    for result in results.values():
        result["prediction"]["pipeline"]["configuration_sha256"] = configuration["sha256"]
    task = dict(run=run, configuration=configuration_ref,
        ordered_case_ids=[frame["case_id"] for frame in frames],
        input_catalog=dict(format=CATALOG_FORMAT, complete=True, run=run,
            configuration=configuration_ref, inputs=sorted(catalogue.values(),
                key=lambda value: catalog_order_key(value["reference"]))))
    job_root = tmp_path / "job"
    _write_json(job_root / "job.json", dict(series_id="ffac", expected_runs=[run]))
    _write_json(job_root / "results" / run / "task.json", task)
    _write_json(job_root / "done" / (run + ".json"), dict(run=run, complete=True))
    job = NS(root=job_root, series_id="ffac", runs=[run], tasks={run: task},
        completed={run: dict(whole_run=True, worker_references_closed=True)},
        products={run: products}, frames=lambda _run: list(task["ordered_case_ids"]),
        frame_result=lambda _run, case: deepcopy(results[case]),
        product_source=lambda value: Path(value.local_path))
    stores = StoreMap("fixture-host", {"ffac": {"series": str(tmp_path)}}, {}, {})
    monkeypatch.setattr(post, "_finished", lambda _root: job)
    monkeypatch.setattr(post, "_stores", lambda _bound, _root: stores)
    store_file = tmp_path / "stores.json"
    _write_json(store_file, {"fixture": "no machine store file is used by this injected seam"})
    return NS(job=job, root=job_root, stores=stores, store_file=store_file,
        preparation=preparation, preparation_config=preparation_config,
        configuration=configuration, native=inputs["configuration"], results=results,
        photo=photo, inputs=inputs, output=tmp_path / "review", series_root=tmp_path)


def request(data):
    value = post.build_request(data.root, data.store_file, [data.preparation["path"]],
        data.output, series_root=data.series_root)
    return post.attach_request(value)


def test_real_writer_is_called_once_with_original_raw_source_h_and_sparse_order(finished_inputs):
    data = finished_inputs
    before = {str(path): path.read_bytes() for path in data.series_root.rglob("*") if path.is_file()}
    bound = request(data)
    result = post.run_request(bound)
    assert result["complete"] is True and result["adopted"] is False
    manifest = json.loads(Path(result["manifests"][0]["path"]).read_bytes())
    assert [row["case_id"] for row in manifest["frames"]] == data.job.frames(data.job.runs[0])
    assert [row["frame_index"] for row in manifest["frames"]] == list(range(len(manifest["frames"])))
    assert manifest["reference"]["acquisition_manifest"] == data.preparation
    assert "change" not in manifest["modes"]
    assert "no own rig geometry" in result["change_display"]
    assert result["baseline_classes"] == {}
    for path, raw in before.items():
        assert Path(path).read_bytes() == raw
    first_renders = list(data.inputs["renders"])
    reused = post.run_request(bound)
    assert reused == result
    assert data.inputs["renders"] == first_renders


@pytest.fixture
def baseline_inputs(finished_inputs, baseline_source, monkeypatch):
    """Add own source geometry at the prepared-input seam; use the real class helper/writer."""
    from darsia.fluidflower import raw_photographic_review as review
    data = finished_inputs
    template_record, phase = deepcopy(baseline_source)
    original_prepare = post.prepare_run
    prepared = original_prepare(data.job, data.job.runs[0], data.stores, data.preparation)
    record = deepcopy(prepared.reference_record)
    full = record["full"]
    shape, origin = full["phase_shape_hw"], full["native_origin_xy"]
    row = template_record["full"]["binding"]["geometry"]
    row.update(raw_source=dict(sha256=full["source_sha256"]), phase_shape_hw=shape,
        native_origin_xy=origin, frame_photo_top_keep=_write_npy(data.series_root / "own_top.npy", np.ones(shape, bool)),
        reference_rig=dict(polygon_reference_native=[origin, [origin[0]+shape[1]-1, origin[1]],
            [origin[0]+shape[1]-1, origin[1]+shape[0]-1], [origin[0], origin[1]+shape[0]-1]]))
    full["binding"]["geometry"] = row
    gp = phase["parameters"]["geometry"]
    gp.update(rig_origin_xy=origin, native_shape_hw=full["native_centre"]["native_shape_hw"],
        source_box_xywh=[-origin[0], -origin[1], *reversed(full["native_centre"]["native_shape_hw"])],
        floor_source_origin_xy=origin)
    phase["chain_assets"]["floor"] = _write_npy(data.series_root / "own_floor.npy", np.full(shape[1], shape[0]-2.))
    phase_bound = _write_json(data.series_root / "own_phase.json", phase)
    manifest = json.loads(Path(data.preparation["path"]).read_bytes())
    manifest["records"][prepared.reference_index] = record
    reference_bound = _write_json(data.series_root / "own_preparation.json", manifest)
    def prepare(*args):
        value = original_prepare(*args)
        for bound in (phase_bound, reference_bound, row["frame_photo_top_keep"], phase["chain_assets"]["floor"]):
            value.sources.original(bound)
        return replace(value, reference_record=record, reference_manifest=reference_bound,
            phase_configuration=phase_bound)
    monkeypatch.setattr(post, "prepare_run", prepare)
    calls = []
    def change(_sources, classes, **kwargs):
        calls.append((classes, kwargs["frame_index"]))
        return np.zeros((*shape, 3), np.uint8)
    monkeypatch.setattr(review, "render_change_asset", change)
    return data, calls


def test_own_baseline_classes_enable_change_and_reuse_without_writes(baseline_inputs):
    data, calls = baseline_inputs
    bound = request(data)
    result = post.run_request(bound)
    run = data.job.runs[0]
    manifest = json.loads(Path(result["manifests"][0]["path"]).read_bytes())
    assert manifest["modes"][-1] == "change"
    assert all("change" in row["assets"] for row in (manifest["reference"], *manifest["frames"]))
    classes = manifest["reference"]["rig_classes"]
    assert Path(classes["path"]) == data.output / "baseline_classes" / (run + ".npz")
    assert calls[0] == (classes, None) and len(calls) == 1 + len(manifest["frames"])
    before = {path: path.read_bytes() for path in data.output.rglob("*") if path.is_file()}
    assert post.run_request(bound) == result
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert len(calls) == 1 + len(manifest["frames"])


@pytest.mark.parametrize("tamper", ["arrays", "provenance", "path"])
def test_reuse_rejects_rebound_baseline_output_tampering(baseline_inputs, tamper):
    data, calls = baseline_inputs
    bound = request(data)
    result = post.run_request(bound)
    run = data.job.runs[0]
    provenance_bound = result["baseline_classes"][run]
    provenance_path = Path(provenance_bound["path"])
    provenance = json.loads(provenance_path.read_bytes())
    if tamper == "arrays":
        path = Path(provenance["classes"]["path"])
        with np.load(path) as arrays:
            values = {key: arrays[key].copy() for key in arrays.files}
        values["modelled"][0, 0] ^= True
        np.savez_compressed(path, **values)
        provenance["classes"] = post._pin(path)
    elif tamper == "provenance":
        provenance["record_index"] += 1
    else:
        path = data.series_root / "outside_classes.npz"
        path.write_bytes(Path(provenance["classes"]["path"]).read_bytes())
        provenance["classes"] = post._pin(path)
    result["baseline_classes"][run] = _write_json(provenance_path, provenance)
    _write_json(data.output / "postrun_completed.json", result)
    count = len(calls)
    with pytest.raises(ValueError, match="baseline class"):
        post.run_request(bound)
    assert len(calls) == count


def test_original_reference_current_h_is_not_used_as_photo_reference_h(finished_inputs):
    data = finished_inputs
    prepared = post.prepare_run(data.job, data.job.runs[0], data.stores, data.preparation)
    assert not np.array_equal(np.asarray(prepared.reference_record["current_h"]),
                              np.asarray(json.loads(Path(data.preparation_config["path"]).read_bytes())["acquisitions"][0]["phase_roles"]["reference_to_current_h"]))
    support = prepared.reference_photo()
    assert support.dtype == bool and not support[:, 0].any()


@pytest.mark.parametrize("drift", ["photo", "H", "source", "native_configuration"])
def test_changed_photo_h_source_or_configuration_refuses_before_output(finished_inputs, drift):
    data = finished_inputs
    if drift in {"photo", "H"}:
        value = json.loads(Path(data.preparation_config["path"]).read_bytes())
        roles = value["acquisitions"][0]["phase_roles"]
        if drift == "photo":
            roles["photo_proof"]["sha256"] = "0" * 64
        else:
            roles["reference_to_current_h"][0][2] += 1.0
        new_config = _write_json(Path(data.preparation_config["path"]), value)
        original = json.loads(Path(data.preparation["path"]).read_bytes())
        original["configuration"] = new_config
        data.preparation = _write_json(Path(data.preparation["path"]), original)
    elif drift == "native_configuration":
        Path(data.native["path"]).write_bytes(b"changed configuration")
    else:
        camera = json.loads(Path(data.preparation["path"]).read_bytes())["records"][0]["source"]["source_sha256"]
        assert camera
        audit = json.loads(Path(json.loads(Path(data.preparation["path"]).read_bytes())["records"][0]["native_row"]["path"]).read_bytes())["raw_producer_audit"]
        path = json.loads(Path(audit["path"]).read_bytes())["sources"]["camera_arw"]["path"]
        Path(path).write_bytes(b"changed RAW source")
    with pytest.raises((ValueError, OSError)):
        request(data)
    assert not data.output.exists()


def test_missing_original_preparation_is_not_inferred_from_nearby_files(finished_inputs):
    data = finished_inputs
    with pytest.raises(ValueError, match="missing"):
        post.build_request(data.root, data.store_file, [], data.output, series_root=data.series_root)
    assert not data.output.exists() and not post.request_path(data.root).exists()


def test_failed_writer_preserves_partial_output_and_never_calls_writer_again(finished_inputs, monkeypatch):
    data = finished_inputs
    bound = request(data)
    calls = []
    def fail(_prepared, root):
        calls.append(root)
        (Path(root) / "partial.png").write_bytes(b"partial")
        raise OSError("Display asset could not be written")
    monkeypatch.setattr(post, "render_run", fail)
    with pytest.raises(OSError):
        post.run_request(bound)
    assert (data.output / "postrun_error.json").is_file()
    with pytest.raises(ValueError, match="no automatic retry"):
        post.run_request(bound)
    assert len(calls) == 1 and (data.output / "partial.png").read_bytes() == b"partial"


@pytest.mark.parametrize("snapshot", [
    {"complete": False, "runs": [{"state": "Working"}], "errors": []},
    {"complete": True, "runs": [], "errors": []},
    {"complete": True, "runs": [{"state": "Complete"}], "errors": ["Source metadata changed"]},
    {"complete": True, "runs": [{"state": "Publishing"}], "errors": []},
])
def test_terminal_reader_refusal_prevents_any_product_reader_or_render(monkeypatch, snapshot):
    from darsia.fluidflower import jobs_metadata
    from darsia.fluidflower.published_dataset import job_reader
    calls = []
    monkeypatch.setattr(jobs_metadata, "JobSnapshotReader", lambda: lambda _row: snapshot)
    monkeypatch.setattr(job_reader, "FinishedJob", lambda _root: calls.append("products"))
    with pytest.raises(ValueError, match="genuinely completed"):
        post._finished("private-job")
    assert not calls


@pytest.mark.parametrize("result", [
    {"whole_run": False, "worker_references_closed": True},
    {"whole_run": True, "worker_references_closed": False},
])
def test_diagnostic_prefix_or_unclosed_worker_never_becomes_a_postrun_job(monkeypatch, result):
    from darsia.fluidflower import jobs_metadata
    from darsia.fluidflower.published_dataset import job_reader
    monkeypatch.setattr(jobs_metadata, "JobSnapshotReader", lambda: lambda _row:
        dict(complete=True, errors=[], runs=[dict(state="Complete")]))
    monkeypatch.setattr(job_reader, "FinishedJob", lambda _root:
        NS(series_id="ffac", completed={"experiment": result}))
    with pytest.raises(ValueError, match="whole-run closed"):
        post._finished("private-job")


def test_changed_original_job_refuses_the_saved_request(finished_inputs):
    data = finished_inputs
    bound = request(data)
    (data.root / "job.json").write_bytes(b"changed job")
    with pytest.raises(ValueError, match="production job changed"):
        post.run_request(bound)
    assert not data.output.exists()


def test_changed_final_publication_refuses_the_saved_request(finished_inputs):
    data = finished_inputs
    bound = request(data)
    (data.root / "done" / (data.job.runs[0] + ".json")).write_bytes(b"changed publication")
    with pytest.raises(ValueError, match="final publication changed"):
        post.run_request(bound)
    assert not data.output.exists()


def test_completed_review_is_not_reused_after_its_original_source_changes(finished_inputs):
    data = finished_inputs
    bound = request(data)
    post.run_request(bound)
    before = (data.output / "postrun_completed.json").read_bytes()
    Path(data.native["path"]).write_bytes(b"changed configuration")
    with pytest.raises((ValueError, OSError)):
        post.run_request(bound)
    assert (data.output / "postrun_completed.json").read_bytes() == before


def test_explicit_command_line_backfill_uses_the_same_real_writer_and_saved_request(finished_inputs, capsys):
    data = finished_inputs
    assert post.main(["--job-root", str(data.root), "--stores", str(data.store_file),
        "--series-root", str(data.series_root), "--preparation", data.preparation["path"],
        "--output-root", str(data.output)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["complete"] is True
    bound = post._pin(post.request_path(data.root))
    assert post.main(["--request", bound["path"], "--request-sha256", bound["sha256"]]) == 0
    assert json.loads(capsys.readouterr().out) == first


@pytest.mark.parametrize("corruption", ["omitted", "empty", "duplicate", "wrong_run", "external_path", "wrong_root", "adopted"])
def test_reuse_refuses_incomplete_foreign_or_untruthful_completion(finished_inputs, monkeypatch, corruption):
    data = finished_inputs
    bound = request(data)
    result = post.run_request(bound)
    completed_path = data.output / "postrun_completed.json"
    if corruption == "omitted":
        result.pop("manifests")
    elif corruption == "empty":
        result["manifests"] = []
    elif corruption == "duplicate":
        result["manifests"] *= 2
    elif corruption == "wrong_run":
        result["manifests"][0]["run"] = "another-experiment"
    elif corruption == "external_path":
        manifest = json.loads(Path(result["manifests"][0]["path"]).read_bytes())
        result["manifests"][0] = dict(_write_json(data.series_root / "external_manifest.json", manifest),
            run=data.job.runs[0])
    elif corruption == "wrong_root":
        result["review_root"] = str(data.series_root / "other_review")
    else:
        result["adopted"] = True
    _write_json(completed_path, result)
    calls = list(data.inputs["renders"])
    def forbidden(*_args):
        raise AssertionError("Refused completed reviews must never call the writer again")
    monkeypatch.setattr(post, "render_run", forbidden)
    with pytest.raises(ValueError, match="Existing RAW review"):
        post.run_request(bound)
    assert data.inputs["renders"] == calls


def test_reuse_refuses_valid_same_run_review_substituted_inside_expected_output(finished_inputs, monkeypatch):
    from darsia.fluidflower.raw_photographic_review import validate_raw_review_manifest
    data = finished_inputs
    bound = request(data)
    result = post.run_request(bound)
    run = data.job.runs[0]
    prepared = post.prepare_run(data.job, run, data.stores, data.preparation)
    other_config = data.series_root / "other_native_configuration.json"
    other_config.write_bytes(Path(prepared.native_configuration["path"]).read_bytes())
    other_bound = post._pin(other_config)
    prepared.sources.register(other_bound, other_config)
    foreign = replace(prepared, native_configuration=other_bound)
    foreign_manifest = post.render_run(foreign, data.series_root / "other_review")
    foreign_value = json.loads(foreign_manifest.read_bytes())
    assert validate_raw_review_manifest(foreign_value, run) == foreign_value
    expected_manifest = Path(result["manifests"][0]["path"])
    expected_manifest.write_bytes(foreign_manifest.read_bytes())
    result["manifests"][0] = dict(post._pin(expected_manifest), run=run)
    _write_json(data.output / "postrun_completed.json", result)
    calls = list(data.inputs["renders"])
    def forbidden(*_args):
        raise AssertionError("A foreign review must never trigger a fresh writer attempt")
    monkeypatch.setattr(post, "render_run", forbidden)
    with pytest.raises(ValueError, match="configuration"):
        post.run_request(bound)
    assert data.inputs["renders"] == calls


@pytest.mark.parametrize("when", ["after_complete", "after_render"])
def test_same_size_consumed_native_array_drift_refuses_reuse_or_final_completion(finished_inputs, monkeypatch, when):
    data = finished_inputs
    bound = request(data)
    reference = json.loads(Path(data.preparation["path"]).read_bytes())["records"][0]
    array_path = Path(reference["full"]["native_centre"]["log_linear_rgb"]["path"])
    def drift():
        raw = bytearray(array_path.read_bytes())
        old_size = len(raw)
        raw[-1] ^= 1
        array_path.write_bytes(raw)
        assert array_path.stat().st_size == old_size
    if when == "after_render":
        writer = post.render_run
        def render_then_drift(prepared, root):
            result = writer(prepared, root)
            drift()
            return result
        monkeypatch.setattr(post, "render_run", render_then_drift)
        with pytest.raises(ValueError, match="changed"):
            post.run_request(bound)
        assert not (data.output / "postrun_completed.json").exists()
        assert (data.output / "postrun_error.json").is_file()
    else:
        post.run_request(bound)
        before = (data.output / "postrun_completed.json").read_bytes()
        calls = list(data.inputs["renders"])
        drift()
        with pytest.raises(ValueError, match="changed"):
            post.run_request(bound)
        assert data.inputs["renders"] == calls
        assert (data.output / "postrun_completed.json").read_bytes() == before


def test_output_inside_original_raw_producer_folder_is_refused_before_any_write(finished_inputs):
    data = finished_inputs
    original = json.loads(Path(data.preparation["path"]).read_bytes())["records"][0]
    protected = Path(original["full"]["native_centre"]["log_linear_rgb"]["path"]).parent / "new_review"
    with pytest.raises(ValueError, match="original RAW producer/data folders"):
        post.build_request(data.root, data.store_file, [data.preparation["path"]],
            protected, series_root=data.series_root)
    assert not protected.exists() and not post.request_path(data.root).exists()


def test_output_inside_hash_verified_producer_directory_alias_is_refused_before_any_write(finished_inputs):
    data = finished_inputs
    original = json.loads(Path(data.preparation["path"]).read_bytes())["records"][0]
    literal_file = Path(original["full"]["native_centre"]["log_linear_rgb"]["path"])
    literal_folder = literal_file.parent
    physical_folder = data.series_root / "resolved_baseline_producer"
    # Move only this synthetic fixture's directory, preserving every saved
    # descriptor and file byte. Its old locator becomes a real directory alias.
    assert literal_folder.resolve().is_relative_to(data.series_root.resolve())
    assert physical_folder.resolve().is_relative_to(data.series_root.resolve())
    literal_folder.rename(physical_folder)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(physical_folder), str(literal_folder))
    else:
        literal_folder.symlink_to(physical_folder, target_is_directory=True)
    assert literal_folder.resolve() == physical_folder.resolve()
    assert post._pin(literal_file)["sha256"] == original["full"]["native_centre"]["log_linear_rgb"]["sha256"]
    output_alias = literal_folder / "new_review"
    output_resolved = output_alias.resolve()
    assert output_resolved.parent == physical_folder.resolve()
    assert post._key(output_resolved.parent) != post._key(literal_folder)
    with pytest.raises(ValueError, match="original RAW producer/data folders"):
        post.build_request(data.root, data.store_file, [data.preparation["path"]],
            output_alias, series_root=data.series_root)
    assert not output_alias.exists() and not output_resolved.exists()
    assert not post.request_path(data.root).exists()


def _snapshot(complete=False, root="fixture-job"):
    return dict(root=root, name="Fixture", state="Complete" if complete else "Working",
        series="Fixture", complete=complete, errors=[], runs=[], workers=[], machines=[], logs=[],
        job_facts=dict(format="darsia-alias-job-v1", series_id="campaign", pipeline="native-phase-n1",
                       expected_runs=["run-a"]),
        raw_review_preparations={"run-a": dict(path="saved-preparation.json", sha256="a" * 64, size_bytes=1)},
        master_machine="fixture-host", committed=1 if complete else 0, total_frames=1,
        remaining=0 if complete else 1)


def test_original_alias_jobs_without_a_pipeline_field_can_prepare_reviews(qtbot):
    widget = gui.JobsPanel()
    qtbot.addWidget(widget)
    value = _snapshot(True)
    del value["job_facts"]["pipeline"]
    widget._show_snapshot(value)
    assert widget.raw_review_button.isEnabled()
    value["job_facts"]["pipeline"] = "native-phase-statistics"
    widget._show_snapshot(value)
    assert not widget.raw_review_button.isEnabled()


def test_store_provider_failure_is_visible_and_releases_the_action(qtbot):
    def unavailable():
        raise OSError("Local store map is unavailable")
    widget = gui.JobsPanel(local_stores_path_provider=unavailable)
    qtbot.addWidget(widget)
    widget.current = _snapshot(True)
    widget.prepare_raw_review()
    assert "Local store map is unavailable" in widget.raw_review_status.text()
    assert widget.can_close() and widget.raw_review_button.isEnabled()


def test_jobs_observer_uses_actual_terminal_transition_once_and_runs_off_gui_thread(qtbot):
    calls = []
    def runner(root, **_kwargs):
        calls.append((root, threading.get_ident()))
        return {"complete": True}
    widget = gui.JobsPanel(raw_review_runner=runner)
    qtbot.addWidget(widget, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    main_thread = threading.get_ident()
    widget.snapshots = [_snapshot(True)]
    widget._observe_raw_review_completion()
    assert not calls
    widget.snapshots = [_snapshot(False)]
    widget._observe_raw_review_completion()
    widget.snapshots = [_snapshot(True)]
    widget._observe_raw_review_completion()
    qtbot.waitUntil(widget.can_close, timeout=10000)
    assert len(calls) == 1 and calls[0][1] != main_thread
    widget.snapshots = [_snapshot(False)]
    widget._observe_raw_review_completion()
    widget.snapshots = [_snapshot(True)]
    widget._observe_raw_review_completion()
    assert len(calls) == 1


def test_jobs_close_waits_for_valid_postrun_work_and_releases_owners(qtbot):
    entered, release = threading.Event(), threading.Event()
    def runner(_root, **_kwargs):
        entered.set()
        assert release.wait(5)
        return {"complete": True}
    widget = gui.JobsPanel(raw_review_runner=runner)
    def close_safely(value):
        release.set()
        qtbot.waitUntil(value.can_close, timeout=10000)
    qtbot.addWidget(widget, before_close_func=close_safely)
    widget.current = _snapshot(True)
    widget.prepare_raw_review()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        assert not widget.can_close()
        widget.close()
        assert not widget.can_close()
    finally:
        release.set()
    qtbot.waitUntil(widget.can_close, timeout=10000)
    assert widget._raw_review_thread is None and widget._raw_review_worker is None


def test_explicit_backfill_uses_declared_preparations_without_a_picker(qtbot, monkeypatch, tmp_path):
    calls = []
    def runner(root, **kwargs):
        calls.append((root, kwargs))
        return {"complete": True}
    def choose(*_args):
        raise AssertionError("Declared inputs must not open a file picker")
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileNames", choose)
    widget = gui.JobsPanel(raw_review_runner=runner)
    qtbot.addWidget(widget, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    widget.current, widget.series_root = _snapshot(True), tmp_path
    widget.prepare_raw_review()
    qtbot.waitUntil(lambda: widget.can_close() and len(calls) == 1, timeout=10000)
    assert calls[0][1]["preparations"] == tuple(widget.current["raw_review_preparations"].values())
    assert calls[0][1]["series_root"] == tmp_path


def test_automatic_missing_inputs_never_opens_a_picker_or_retries(qtbot, monkeypatch):
    calls = []
    def runner(root, **_kwargs):
        calls.append(root)
        raise post.ReviewInputsMissing("Choose original preparations")
    def forbidden(*_args):
        raise AssertionError("An automatic completion check must not open a file dialog")
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileNames", forbidden)
    widget = gui.JobsPanel(raw_review_runner=runner)
    qtbot.addWidget(widget, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    widget.snapshots = [_snapshot(False)]
    widget._observe_raw_review_completion()
    widget.snapshots = [_snapshot(True)]
    widget._observe_raw_review_completion()
    qtbot.waitUntil(widget.can_close, timeout=10000)
    widget.snapshots = [_snapshot(False)]
    widget._observe_raw_review_completion()
    widget.snapshots = [_snapshot(True)]
    widget._observe_raw_review_completion()
    assert len(calls) == 1


def test_preparation_declarations_are_series_scoped_relative_and_explicit(tmp_path):
    import toml
    descriptor = dict(path="inputs/preparation.json", sha256="a" * 64, size_bytes=12)
    source = dict(series=dict(id="campaign"), raw_review=dict(preparations={"run-a": descriptor}))
    (tmp_path / "series.toml").write_text(toml.dumps(source), encoding="utf-8")
    selected = post.declared_preparations(tmp_path, expected_series="campaign", runs=["run-a"])
    assert selected == {"run-a": dict(descriptor, path=str(tmp_path / descriptor["path"]))}
    with pytest.raises(ValueError, match="another series"):
        post.declared_preparations(tmp_path, expected_series="other", runs=["run-a"])
    with pytest.raises(post.ReviewInputsMissing, match="run-b"):
        post.declared_preparations(tmp_path, expected_series="campaign", runs=["run-a", "run-b"])


def test_missing_preparation_declarations_explain_the_disabled_action(qtbot):
    widget = gui.JobsPanel()
    qtbot.addWidget(widget)
    value = _snapshot(True)
    value.pop("raw_review_preparations")
    value["raw_review_reason"] = "Declare original RAW acquisition preparations for run-a"
    widget._show_snapshot(value)
    assert not widget.raw_review_button.isEnabled()
    assert widget.raw_review_status.text() == value["raw_review_reason"]
    assert widget.raw_review_button.toolTip() == value["raw_review_reason"]


def test_declared_preparation_byte_identity_is_checked_before_request_write(finished_inputs):
    data = finished_inputs
    declared = dict(data.preparation, sha256="0" * 64)
    with pytest.raises(ValueError, match="declared RAW acquisition preparation changed"):
        post.build_request(data.root, data.store_file, [declared], data.output, series_root=data.series_root)
    assert not data.output.exists() and not post.request_path(data.root).exists()


def test_attach_review_routes_to_basis_when_no_model_is_selected(qtbot):
    widget = gui.JobsPanel(raw_review_model_provider=lambda: None)
    qtbot.addWidget(widget)
    widget.current = _snapshot(True)
    with qtbot.waitSignal(widget.raw_model_requested, timeout=1000):
        widget.attach_raw_review()
    assert "Phase model" in widget.raw_review_status.text() and widget.can_close()


def test_attach_review_registers_the_explicit_model_off_the_gui_thread(finished_inputs, qtbot):
    data = finished_inputs
    calls, bindings = [], []
    def register(completion, model):
        calls.append((completion, model, threading.get_ident()))
        return dict(combined_root="registered-collection", selected_candidates={"run-a": "saved-predictions"})
    widget = gui.JobsPanel(raw_review_model_provider=lambda: {"root": "chosen-model"},
                           raw_review_registrar=register)
    qtbot.addWidget(widget, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    widget.series_root = data.series_root
    widget.current = _snapshot(True, root=str(data.root))
    widget.raw_review_registered.connect(bindings.append)
    main_thread = threading.get_ident()
    widget.attach_raw_review()
    qtbot.waitUntil(lambda: widget.can_close() and bool(bindings), timeout=10000)
    assert calls[0][0] == post.default_output(data.root, data.series_root) / "postrun_completed.json"
    assert calls[0][1] == "chosen-model" and calls[0][2] != main_thread
    assert bindings[0]["combined_root"] == "registered-collection"
    assert "scientific acceptance remains separate" in widget.raw_review_status.text()


def test_attach_review_origin_failure_is_visible_and_releases_the_thread(finished_inputs, qtbot):
    def register(*_args):
        raise ValueError("Production uses multiple model origins")
    widget = gui.JobsPanel(raw_review_model_provider=lambda: {"root": "chosen-model"},
                           raw_review_registrar=register)
    qtbot.addWidget(widget, before_close_func=lambda value: qtbot.waitUntil(value.can_close, timeout=10000))
    widget.series_root = finished_inputs.series_root
    widget.current = _snapshot(True, root=str(finished_inputs.root))
    widget.attach_raw_review()
    qtbot.waitUntil(widget.can_close, timeout=10000)
    assert "multiple model origins" in widget.raw_review_status.text()
    assert widget._raw_review_worker is None and widget.raw_review_attach_button.isEnabled()


def test_cli_uses_declared_preparations_when_no_picker_paths_are_given(finished_inputs, monkeypatch):
    import toml
    data = finished_inputs
    declaration = dict(series_id=data.job.series_id,
        raw_review=dict(preparations={data.job.runs[0]: data.preparation}))
    (data.series_root / "series.toml").write_text(toml.dumps(declaration), encoding="utf-8")
    calls = []
    monkeypatch.setattr(post, "prepare_and_run", lambda *args, **kwargs: calls.append((args, kwargs)) or {"complete": True})
    assert post.main(["--job-root", str(data.root), "--stores", str(data.store_file),
                      "--series-root", str(data.series_root)]) == 0
    assert calls[0][0][2] == [data.preparation]
