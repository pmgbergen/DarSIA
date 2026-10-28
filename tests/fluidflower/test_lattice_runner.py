"""The lattice runner on synthetic frames: products, provenance and resume.

Only ``series.SeriesInputs`` is replaced, by a fake that serves the synthetic
native frames of test_lattice_inputs. The runner, the lattice inputs, the
unchanged phase chain, the lattice N1 template and the unchanged frame
publisher all run. No NAS file is read.
"""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import darsia
import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import storage_outputs
from darsia.fluidflower.mass import n1_pipeline
from darsia.fluidflower.mass.n1_pipeline import asset
from darsia.fluidflower.pipeline import checkpoint, job_queue, lattice_runner as lr, series
from darsia.fluidflower.pipeline.job_queue import descriptor
from test_lattice_inputs import BOX, CANVAS, EXTENT, FIELDS, PLUME, reference_matrix, record, synthetic
from test_phase_rules import parameters as rule_parameters  # noqa: F401  (fixture)
from test_phase_domain import parameters as domain_parameters  # noqa: F401  (fixture)

STORAGE = dict(format=storage_outputs.REQUEST_FORMAT, gas_score_encoding=storage_outputs.LA_ENCODING,
               resolution_mm=1.0, extent_m=list(EXTENT), computing_resolution_mm=1.0)


def save(directory, name, value, kind):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        path.write_text(json.dumps(value), encoding="utf-8")
    elif kind == "npy":
        np.save(path, value, allow_pickle=False)
    elif kind == "npz":
        np.savez(path, **value)
    else:
        Image.fromarray(value).save(path)
    return asset(path)


def mass_template(directory):
    """A synthetic N1 template whose reference grid is the native centre of the canvas."""
    shape = (BOX[3], BOX[2])
    facies = np.ones(shape, np.uint8)
    facies[:, shape[1] // 2:] = 2
    colors, signals = {}, {}
    bounds = dict(min_value_aq=0.0, max_value_aq=1.0, min_value_g=0.0, max_value_g=1.0)
    for label in (0, 1, 2):  # facies 0 lies outside the physical reference
        base = Path(directory) / ("color_" + str(label))
        path = darsia.ColorPath(colors=[np.array([1.0, 1.0, 1.0]), np.array([0.6, 0.9, 1.3])])
        darsia.ColorPathInterpolation(path, darsia.ColorMode.RELATIVE, [0.0, 1.0]).save(base)
        colors[str(label)] = asset(base.with_suffix(".json"))
        base = Path(directory) / ("signal_" + str(label))
        darsia.PWTransformation([0.0, 1.0], [0.0, 1.0]).save(base)
        signals[str(label)] = asset(base.with_suffix(".csv"))
    flash = Path(directory) / "flash"
    darsia.SimpleFlash(**bounds).save(flash)
    return dict(
        format=n1_pipeline.CONFIG_FORMAT, geometry_fields=dict(FIELDS),
        shared=dict(physical_reference=save(directory, "physical.npz", dict(
                        facies=facies, domain=np.ones(shape, bool), porosity=np.full(shape, 0.3, np.float32),
                        depth_m=np.full(shape, 0.01, np.float32)), "npz"),
                    reference_matrix=reference_matrix().tolist(),
                    floor_curve=save(directory, "floor.npy", np.full(CANVAS[1], 712.0), "npy"),
                    floor_origin_xy=[-BOX[0], -BOX[1]],
                    depth_points=[[0.0, 0.0, 0.01], [0.2, 0.0, 0.01], [0.0, 0.2, 0.01], [0.2, 0.2, 0.01]],
                    porosity_by_facies={"1": 0.3, "2": 0.3}, swi_by_facies={"1": 0.2, "2": 0.25},
                    unquantified_facies=[], clearwater=dict(
                        rgb=save(directory, "clear.npy", np.ones((*shape, 3), np.float32), "npy"),
                        valid=save(directory, "clear_valid.npy", np.ones(shape, bool), "npy"),
                        H_reference_to_grid=np.eye(3).tolist()),
                    optics=dict(metadata=save(directory, "optics.json", dict(
                        label_ids=[0, 1, 2], ignore_labels=[0], basis="facies", flash_bounds=bounds), "json"),
                        color_paths=colors, signals=signals, flash=asset(flash.with_suffix(".json")))),
        calibration=save(directory, "calibration.json", dict(parameters=dict(candidate="N1", gain=1.2, beta=0.3)), "json"),
        physical_correction=save(directory, "correction.json", dict(correction_matrix=np.eye(3).tolist()), "json"),
        anchor=dict(elapsed_h=0.5, affected_pore_volume_m3=1e-6), recipe=dict(alkalinity_M=0.001, btb_M=0.0001),
        dimensions_m=[0.17, 0.14], boxes=dict(box1=[[0.02, 0.02], [0.08, 0.1]], box2=[[0.08, 0.02], [0.13, 0.1]]),
        summary=dict(baseline_gain=1.1, intervals=32768, max_column_distance_m=0.02, max_depth_m=0.0155,
                     max_porosity=0.45),
        bindings=dict(amendment_sha256="a" * 64, round4_spec_sha256="b" * 64),
        contracts=dict(result="saved-products", sandmap="saved-sandmap"))


def run_configuration(directory):
    """Two selected frames whose plume grows, with a lattice storage request."""
    template = mass_template(directory)
    shape = (BOX[3], BOX[2])
    mass_spec = dict(
        selected_target_h=1.0, light_epoch="measured", nominal_injected_mass_g=1.0,
        atmosphere=dict(pressure_bar=1.0, temperature_c=20.0, pressure_gradient_bar_per_m=0.0,
                        temperature_gradient_c_per_m=0.0),
        photo_top=dict(save(directory, "top.npz", dict(keep=np.ones(CANVAS, bool)), "npz"), array_key="keep"),
        original_domain=dict(parent_labels_map=save(directory, "parent.png", np.zeros(shape, np.uint8), "png"),
                             native_shape_hw=list(shape), frame_index=0,
                             **{FIELDS["original_domain_to_native"]: np.eye(3).tolist()}),
        inner_polygon_native=[[-30.0, -20.0], [509.0, -20.0], [509.0, 684.0], [-30.0, 684.0]])
    frames, previous = [], "b" * 64
    for index, digest in ((1, "c" * 64), (2, "d" * 64)):
        source = dict(source_sha256=digest, previous_source_sha256=previous, selected=True, studio_anchor=False,
                      parent_source=dict(source_changed=False), elapsed_h=float(index), run="synthetic",
                      source_binding=dict(source_sha256=digest))
        frames.append(dict(case_id="frame_" + str(index), frame_index=index, elapsed_h=float(index),
                           source=source, mass_spec=deepcopy(mass_spec)))
        previous = digest
    return dict(format=series.RUN_FORMAT, run="synthetic", frames=frames, storage=dict(STORAGE),
                table_fields=["case_id", "total_mass_g", "aqueous_mass_g", "gas_mass_g"],
                history_policy="ordered unique acquisitions",
                phase=dict(reference=dict(full=record(), source_sha256="r" * 64),
                           initial_previous=dict(full=record(), source=dict(source_sha256="b" * 64))),
                mass=dict(template=template, calibration=template["calibration"]))


class FakeInputs:
    """Serves the synthetic native frames in place of series.SeriesInputs."""

    fail_on = None

    def __init__(self, parameters, bundle, initial):
        self.parameters, self.model_bundle, self.previous = parameters, bundle, initial
        self.reader = n1_pipeline.DataReader()
        self.closed = False

    def native(self, spec, stage):
        if spec["case_id"] == FakeInputs.fail_on:
            raise RuntimeError("synthetic interruption")
        Path(stage).mkdir(parents=True)
        (Path(stage) / "native_log.npy").write_bytes(b"transient native intermediate")
        return dict(record=dict(full=record()), context=FakeInputs.context, case=spec["case_id"])

    def frame(self, spec, native):
        if self.previous["source"]["source_sha256"] != spec["source"]["previous_source_sha256"]:
            raise ValueError("Fresh frame has another acquisition predecessor")
        frame = dict(FakeInputs.frames[spec["case_id"]])
        frame["source"] = spec["source"]
        return frame

    def advance(self, frame):
        self.previous = {key: frame[key] for key in ("source", "full_log", "full_valid", "temporal_input")}

    def close(self):
        self.closed = True


@pytest.fixture
def lattice_world(tmp_path, monkeypatch, rule_parameters, domain_parameters):
    """A run configuration, its task and the fake native inputs, all synthetic."""
    config = run_configuration(tmp_path / "inputs")
    frames = {}
    for spec, rows in zip(config["frames"], (slice(420, 560), slice(420, 600))):
        _, inputs, native, frame, _ = synthetic(rule_parameters, domain_parameters)
        current = frame["full_log"].copy()
        current[:, PLUME[0], PLUME[1]] = frame["full_log"][:, 0:1, 0:1]
        current[:, rows, PLUME[1]] = np.log(np.array([200.0, 300.0, 420.0], np.float32))[:, None, None]
        frames[spec["case_id"]] = dict(frame, full_log=current, native_input=type(frame["native_input"])(
            np.ascontiguousarray(current[:, BOX[1]:BOX[1] + BOX[3], BOX[0]:BOX[0] + BOX[2]]),
            frame["native_input"].valid, frame["native_input"].provenance))
    FakeInputs.frames, FakeInputs.context, FakeInputs.fail_on = frames, native["context"], None
    created = []

    def make(config_value, output):
        fake = FakeInputs(inputs.parameters, inputs.model_bundle, dict(inputs.previous))
        created.append(fake)
        return fake
    monkeypatch.setattr(series, "SeriesInputs", make)
    monkeypatch.setattr(lr, "load_lattice_run", lambda task: deepcopy(config))
    monkeypatch.setattr(job_queue, "require_unc", lambda path: None)
    monkeypatch.setattr(n1_pipeline, "_is_unc", lambda path: True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    raw = json.dumps(config, sort_keys=True).encode()
    task = dict(run="synthetic", job_id="synthetic-job", pipeline=lr.PIPELINE, computing_resolution_mm=1.0,
                configuration=dict(path=str(tmp_path / "config.json"), sha256=hashlib.sha256(raw).hexdigest(),
                                   size_bytes=len(raw)),
                commit="e" * 40, calibration_sha256=config["mass"]["calibration"]["sha256"], precision="float64",
                threads=1, backend="cpu")
    return dict(config=config, task=task, created=created, scratch=scratch, root=tmp_path / "results")


def run(world, name, task=None):
    output = world["root"] / name
    return lr.run_lattice_series(task or world["task"], output, "cpu", "float64"), output


def test_a_lattice_run_publishes_lattice_frames_with_their_resolution_and_provenance(lattice_world):
    events = []
    output = lattice_world["root"] / "attempt"
    result = lr.run_lattice_series(lattice_world["task"], output, "cpu", "float64", events.append)
    assert result["complete"] and result["pipeline"] == lr.PIPELINE and result["frames"] == 2
    assert result["computing_resolution_mm"] == 1.0 and result["validation"]["state"] == "not_validated"
    assert result["worker_references_closed"] and lattice_world["created"][0].closed
    assert [event["state"] for event in events] == ["frame_started", "frame_completed"] * 2
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["pipeline"] == lr.PIPELINE and manifest["settings"]["computing_resolution_mm"] == 1.0
    assert manifest["lattice"]["lattice"]["shape"] == [183, 145]
    masses = []
    for spec in lattice_world["config"]["frames"]:
        case = spec["case_id"]
        final = output / "frames" / case
        document = json.loads((final / "result.json").read_bytes())
        stage = document["prediction"]["pipeline"]
        assert stage["format"] == lr.FRAME_FORMAT and stage["pipeline"] == lr.PIPELINE
        assert stage["computing_resolution_mm"] == 1.0 and stage["validation"]["state"] == "not_validated"
        assert stage["native"]["full"]["phase_shape_hw"] == list(CANVAS)
        with np.load(final / "products.npz") as products:
            assert products["phase_labels"].shape == (183, 145)
        lr.validate_lattice_frame(spec, document, final, config_of(lattice_world))
        masses.append(document["total_mass_g"])
    assert 0 < masses[0] < masses[1]  # the plume grows between the frames
    assert (output / "lattice" / "physical_units.json").is_file()
    units = json.loads((output / "lattice" / "physical_units.json").read_bytes())
    assert units["pixels_per_cell"] == pytest.approx(4.0)
    assert not list(output.rglob("native_log.npy"))  # native intermediates are not products
    assert not any(lattice_world["scratch"].iterdir())  # and their transient folder is gone
    table = (output / "mass_table.csv").read_text().splitlines()
    assert table[0] == "case_id,total_mass_g,aqueous_mass_g,gas_mass_g" and len(table) == 3


def config_of(world):
    return dict(world["config"], _configuration_sha256=world["task"]["configuration"]["sha256"])


def test_an_interrupted_run_resumes_from_its_committed_frame_exactly(lattice_world):
    reference, _ = run(lattice_world, "uninterrupted")
    FakeInputs.fail_on = "frame_2"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(lattice_world, "first")
    FakeInputs.fail_on = None
    first = lattice_world["root"] / "first"
    proof = checkpoint.retirement_document(lattice_world["task"], first, [dict(pid=1)], dict(kind="test"))
    (first / "retirement.json").write_text(json.dumps(proof), encoding="utf-8")
    task = dict(lattice_world["task"], resume_from=dict(attempt_directory=str(first),
                                                        retirement=descriptor(first / "retirement.json")))
    resumed, output = run(lattice_world, "second", task)
    assert resumed["reused_case_ids"] == ["frame_1"] and resumed["newly_computed_frames"] == 1
    expected = json.loads((lattice_world["root"] / "uninterrupted" / "frames" / "frame_2" / "result.json").read_bytes())
    actual = json.loads((output / "frames" / "frame_2" / "result.json").read_bytes())
    for key in ("total_mass_g", "aqueous_mass_g", "gas_mass_g", "affected_pore_volume_m3"):
        assert actual[key] == expected[key]
    with np.load(output / "frames" / "frame_2" / "products.npz") as left, \
            np.load(lattice_world["root"] / "uninterrupted" / "frames" / "frame_2" / "products.npz") as right:
        np.testing.assert_array_equal(left["phase_labels"], right["phase_labels"])


def test_the_request_needs_a_computing_resolution_no_finer_than_stored(lattice_world):
    config = lattice_world["config"]
    assert lr.lattice_request(config, "float32")["computing_resolution_mm"] == 1.0
    coarser = lr.lattice_request(dict(config, storage=dict(config["storage"], resolution_mm=2.0,
                                                           extent_m=[0.0, 0.146, 0.0, 0.184])), "float32")
    assert coarser["resolution_mm"] == 2.0 and coarser["computing_resolution_mm"] == 1.0
    for change, message in ((dict(resolution_mm=0.5), "cannot be finer"),
                            (dict(resolution_mm=2.0), "origin-zero metric lattice"),
                            (dict(resolution_mm="native"), "cannot be finer"),
                            (dict(computing_resolution_mm=0.25), "0.5 or 1 mm")):
        with pytest.raises(ValueError, match=message):
            lr.lattice_request(dict(config, storage=dict(config["storage"], **change)), "float32")
    with pytest.raises(ValueError, match="computing_resolution_mm"):
        lr.lattice_request(dict(config, storage={key: value for key, value in config["storage"].items()
                                                 if key != "computing_resolution_mm"}), "float32")


def test_the_native_runner_refuses_a_lattice_storage_request(lattice_world):
    with pytest.raises(ValueError, match="Unknown or missing storage output DATA"):
        storage_outputs.parse_request(lattice_world["config"], "float32")


def test_only_a_lattice_request_changes_a_prepared_task(lattice_world):
    task = dict(lattice_world["task"], pipeline=series.PIPELINE)
    task.pop("computing_resolution_mm")
    native = dict(lattice_world["config"])
    native.pop("storage")
    assert lr.lattice_task(task, native) is task
    plain = dict(native, storage={key: value for key, value in STORAGE.items() if key != "computing_resolution_mm"})
    assert lr.lattice_task(task, plain) is task
    lattice = lr.lattice_task(task, lattice_world["config"])
    assert lattice["pipeline"] == lr.PIPELINE and lattice["computing_resolution_mm"] == 1.0
    assert {key: value for key, value in lattice.items() if key not in {"pipeline", "computing_resolution_mm"}} == \
        {key: value for key, value in task.items() if key != "pipeline"}
    with pytest.raises(ValueError, match="native-phase-n1 route only"):
        lr.lattice_task(dict(task, pipeline="n1"), lattice_world["config"])


def test_a_task_of_another_pipeline_or_resolution_is_refused(lattice_world, monkeypatch):
    with pytest.raises(ValueError, match="computing resolutions differ"):
        lr.run_lattice_series(dict(lattice_world["task"], computing_resolution_mm=0.5),
                              lattice_world["root"] / "refused", "cpu", "float64")
    monkeypatch.undo()
    with pytest.raises(ValueError, match="explicit lattice-phase-n1 task"):
        lr.load_lattice_run(dict(lattice_world["task"], pipeline=series.PIPELINE))


def test_the_worker_dispatches_the_lattice_pipeline_to_its_runner():
    source = Path(lr.__file__).with_name("worker.py").read_text(encoding="utf-8")
    found = False
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == lr.PIPELINE):
            imports = [item for item in node.body if isinstance(item, ast.ImportFrom)]
            assert imports and imports[0].module == "lattice_runner"
            assert imports[0].names[0].name == "run_lattice_series"
            found = True
    assert found


def test_identity_keeps_content_hashes_of_transient_records():
    record_value = dict(full=dict(native_centre=dict(log_linear_rgb=dict(path="C:/tmp/x.npy", sha256="1" * 64,
                                                                         size_bytes=3)),
                                  ring_windows=[dict(canvas_xyxy=[0, 0, 1, 1],
                                                     log_linear_rgb=dict(path="C:/tmp/r.npy", sha256="2" * 64))]))
    identity = lr._identity(record_value)
    assert identity["full"]["native_centre"]["log_linear_rgb"] == dict(sha256="1" * 64, size_bytes=3)
    assert identity["full"]["ring_windows"][0] == dict(canvas_xyxy=[0, 0, 1, 1], log_linear_rgb=dict(sha256="2" * 64))
