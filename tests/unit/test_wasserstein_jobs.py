"""New Wasserstein queue contract/I/O tests. Source phase: intentionally unrun.

All inputs are local fixtures. The frozen-report fixture tests byte and task
binding; it is not evidence that any real scientific plan passed its gates.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from darsia.fluidflower import wasserstein_job_protocol as protocol
from darsia.fluidflower import wasserstein_job_inputs as inputs
from darsia.fluidflower import wasserstein_jobs as jobs


def bound(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return bound(path)


def reseal(plan):
    plan["task_contract"]["content_sha256"] = protocol.digest(
        {key: value for key, value in plan["task_contract"].items() if key != "content_sha256"})
    plan["validation"]["task_contract"] = deepcopy(plan["task_contract"])
    plan["validation"]["config_sha256"] = protocol.digest(plan["configuration"])
    plan["content_sha256"] = protocol.digest({key: value for key, value in plan.items() if key != "content_sha256"})
    return plan


def complete_solver_row(selected):
    """Complete metadata fixture, never evidence of an actual solver execution."""
    from math import sqrt
    from darsia._opencv_runtime import POLICY
    observed = dict(policy=POLICY, pid=1, thread_id=2, native_thread_id=3,
        opencv_version="fixture", use_ipp=False, use_ipp_not_exact=False,
        observation_scope="executing-thread-at-boundary")
    return dict(selected, status="success", solver_converged=True, solver_iterations=1,
        solver_warning_count=0, solver_warnings="", runtime_seconds=.1,
        source_mass_1_g=4.0, source_mass_2_g=4.0, pair_mean_detected_mass_g=4.0,
        absolute_mass_difference_g=0.0, relative_mass_difference=0.0,
        transport_cost_mass_m=1.0, mass_based_distance_m=.25,
        rig_based_normalized_distance=.25, rig_diagonal_normalized_distance=.25 / sqrt(2),
        plume_based_normalized_distance=.5, injected_mass_reference_g=4.0,
        reference_plume_radius_m=.5, rig_mean_dimension_m=1.0, rig_diagonal_m=sqrt(2),
        domain_width_m=1.0, domain_height_m=1.0, grid_rows=4, grid_columns=4,
        cell_width_m=.25, cell_height_m=.25,
        transport_mass_balancing="each_distribution_scaled_to_pair_mean",
        opencv_ipp=dict(policy=POLICY,
            before=dict(observed, stage="wasserstein-solve:before"),
            after=dict(observed, stage="wasserstein-solve:after"),
            saved_inputs="original provenance; no retroactive IPP observation"))


@pytest.fixture
def source(tmp_path):
    import numpy as np
    from darsia.fluidflower.mass_artifacts import write_field_frame, write_field_index
    from darsia.fluidflower.wasserstein import wasserstein_task_contract
    root = tmp_path / "series"
    root.mkdir()
    manifest = root / "series.toml"
    manifest.write_text('schema_version = 1\nseries_id = "TEST"\n[roots]\nseries = "."\n', encoding="utf-8")
    run_results, references = [], []
    arrays = {}
    for run in ("r1", "r2"):
        products = {}
        for product in inputs.PRODUCTS:
            directory = root / "mass" / run / product
            frames = []
            for index, time in enumerate((0.0, 1.0)):
                values = np.ones((4, 4), dtype=np.float32)
                if product != "transport_weight":
                    values[:] = 0
                    values[1:3, (1 if run == "r1" else 2)] = 2
                arrays[(run, product, index)] = values
                frames.append(write_field_frame(directory / f"frame-{index}.npz", values,
                    run_id=run, product=product, frame_id=f"f{index}", time_hours=time))
            index_bound = write_field_index(directory / "index.json", run_id=run, product=product,
                quantity=product, unit="1" if product == "transport_weight" else "g/pixel",
                frames=frames, grid=dict(width_m=1.0, height_m=1.0))
            products[product] = index_bound
            references.append(dict(bound(directory / "index.json"), run_id=run, product=product))
        run_results.append(dict(run_id=run, complete=True, target_times_hours=[0.0, 1.0],
            geometry=dict(width_m=1.0, height_m=1.0, native_cell_size_m=.25), products=products,
            injected_mass_g=[4.0, 4.0], reference_plume_radius_m=[.5, .5]))
    mass = save(root / "mass.json", dict(format="darsia-fluidflower-mass-analysis-result-v1",
        complete=True, mass_analysis_plan_id="mass-plan", mass_task_contract_sha256="a" * 64, run_results=run_results))
    mass["product_artifacts"] = references
    config = dict(mass_analysis_plan_id="mass-plan", mass_analysis_plan_sha256="b" * 64,
        mass_task_contract_sha256="a" * 64, run_ids=["r1", "r2"],
        comparison=dict(pairs=[dict(run_1="r1", run_2="r2")], domains=["whole_active_domain"],
                        mass_components=["mass_aqueous"]), spatial_discretization=dict(target_cell_size_m=.25),
        transport={}, outputs=dict(store_transport_diagnostics=False), execution_profile=dict(backend="cpu-float64"))
    selection = dict(selected_time_indices=[0, 1], selected_times_hours=[0.0, 1.0])
    contract = wasserstein_task_contract(config, mass, dict(domains=[]), selection)
    plan = reseal(dict(format=protocol.REVISION_FORMAT, plan_id="saved-plan", configuration=config,
        task_contract=contract, execution=dict(ready=True),
        validation=dict(valid=True, execution_ready=True, summary=dict(error_count=0, task_count=2))))
    plan_bound = save(root / "plan.json", plan)
    operations = [dict(format=protocol.OPERATION_FORMAT, run=protocol.operation_id(row["order"]),
                       plan=plan_bound, comparison=row) for row in protocol.comparisons(plan)]
    return dict(root=root, manifest=bound(manifest), plan=plan, plan_bound=plan_bound,
                operations=operations, arrays=arrays)


def test_preparation_revalidates_original_plan_and_never_changes_selection(source, monkeypatch):
    from darsia.fluidflower import wasserstein
    project = object()
    seen = []
    def validate(actual, config):
        seen.append((actual, deepcopy(config)))
        return deepcopy(source["plan"]["validation"])
    monkeypatch.setattr(wasserstein, "validate_wasserstein_plan", validate)
    assert jobs.build_operations(project, source["plan_bound"]) == source["operations"]
    assert seen == [(project, source["plan"]["configuration"])]
    monkeypatch.setattr(wasserstein, "validate_wasserstein_plan", lambda *_: dict(execution_ready=False))
    with pytest.raises(ValueError, match="Current original"):
        jobs.build_operations(project, source["plan_bound"])


@pytest.mark.parametrize("field", ["time_hours", "time_index", "mass_component", "run_2", "domain_id"])
def test_operation_cannot_change_a_saved_scientific_choice(source, field):
    changed = deepcopy(source["operations"][0])
    changed["comparison"][field] = 9 if field in {"time_hours", "time_index"} else "different"
    with pytest.raises(ValueError, match="frozen comparison"):
        protocol.validate_operation(changed, source["plan"])


def test_changed_revision_and_unsupported_transport_diagnostics_refuse(source):
    changed = deepcopy(source["plan"])
    changed["configuration"]["run_ids"].append("foreign")
    with pytest.raises(ValueError, match="identity differs"):
        protocol.validate_revision(changed)
    changed = deepcopy(source["plan"])
    changed["configuration"]["outputs"]["store_transport_diagnostics"] = True
    changed["task_contract"]["outputs"]["store_transport_diagnostics"] = True
    reseal(changed)
    with pytest.raises(ValueError, match="transport diagnostics"):
        protocol.validate_revision(changed)


def test_catalogue_covers_relative_fields_and_never_opens_arrays(source, monkeypatch):
    import numpy as np
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("Metadata preparation loaded a field array"))
    config = source["operations"][1]
    config_bound = save(source["root"] / "operation.json", config)
    catalog = InputCatalog({"series": source["root"]})
    result = inputs.collect_wasserstein_catalog(catalog, config_bound, config)
    uses = {use for row in result["inputs"] for use in row["uses"]}
    assert {"wasserstein.r1.mass_aqueous.frame.1", "wasserstein.r2.mass_aqueous.frame.1",
            "wasserstein.r1.transport_weight.frame.0", "wasserstein.r2.transport_weight.frame.0"} <= uses
    assert len([use for use in uses if ".frame." in use]) == 4
    assert "wasserstein.r1.mass_total.index" in uses
    assert not any(use.endswith("mass_total.frame.1") for use in uses)
    assert result["payload_hashes_verified"] is False
    assert result["numerical_calls_started"] is False


def test_changed_index_is_rejected_before_array_loading(source):
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    changed = source["root"] / "mass" / "r1" / "mass_total" / "index.json"
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        inputs.catalogue_wasserstein_inputs(InputCatalog({"series": source["root"]}), source["operations"][0])


def test_relative_descriptors_normalize_on_the_original_platform():
    payload = dict(path="products/../fields/frame.npz", sha256="a" * 64, size_bytes=12)
    assert inputs.relative_descriptor("C:\\series\\index.json", payload)["path"] == "C:\\series\\fields\\frame.npz"
    assert inputs.relative_descriptor("/series/index.json", payload)["path"] == "/series/fields/frame.npz"


def test_normalized_escape_is_refused_before_external_stat(source):
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    touched = []
    catalog = InputCatalog({"series": source["root"]}, file_stat=lambda path: touched.append(path))
    escaped = inputs.relative_descriptor(str(source["root"] / "index.json"),
        dict(path="../../outside.npz", sha256="a" * 64, size_bytes=0))
    with pytest.raises(ValueError):
        catalog.add(escaped, "image", "wasserstein.r1.mass_total.frame.0")
    assert touched == []


def test_job_metadata_writer_requires_full_coverage_before_any_directory(source):
    folder = source["root"] / "new-inputs"
    with pytest.raises(ValueError, match="whole saved plan"):
        jobs.write_wasserstein_inputs(folder, source["operations"][:1], job_id="w1-job", commit="c" * 40,
            environment_lock=dict(path="requirements-worker.lock", sha256="d" * 64), actor="Reviewer", reason="Compare saved mass")
    assert not folder.exists()


def execution(source):
    return jobs.write_wasserstein_inputs(source["root"] / "inputs", source["operations"],
        job_id="w1-job", commit="c" * 40,
        environment_lock=dict(path="requirements-worker.lock", sha256="d" * 64),
        actor="Reviewer", reason="Compare the selected saved mass results")


def test_shared_validator_rejection_happens_before_job_write(source, monkeypatch):
    from darsia.fluidflower.pipeline import alias_job, job_queue
    monkeypatch.setattr(alias_job, "shared_output_admission", lambda *_: None)
    seen = []
    def reject(task):
        seen.append(deepcopy(task))
        raise ValueError("shared runtime has not admitted Wasserstein")
    monkeypatch.setattr(job_queue, "validate_task", reject)
    root = source["root"] / "queue-job"
    with pytest.raises(ValueError, match="shared runtime"):
        jobs.prepare_wasserstein_job(execution(source), root, series_manifest=source["manifest"])
    assert not root.exists()
    assert seen[0]["precision"] == "float64" and seen[0]["backend"] == "cpu"
    assert "calibration_sha256" not in seen[0]
    assert source["arrays"][("r1", "mass_aqueous", 0)].dtype.name == "float32"


def test_unrooted_operation_is_refused_before_nested_metadata_read(source, monkeypatch):
    from darsia.fluidflower.pipeline import alias_job, job_inputs
    monkeypatch.setattr(alias_job, "shared_output_admission", lambda *_: None)
    prepared = execution(source)
    path = Path(prepared["path"])
    old = str(source["root"] / "inputs" / "w1-00000000.json")
    outside = str(source["root"].parent / "outside.json")
    text = path.read_text(encoding="utf-8").replace(json.dumps(old), json.dumps(outside))
    path.write_text(text, encoding="utf-8")
    actual_read = job_inputs.InputCatalog._read_metadata
    opened = []
    def read(candidate):
        opened.append(str(candidate))
        return actual_read(candidate)
    monkeypatch.setattr(job_inputs.InputCatalog, "_read_metadata", staticmethod(read))
    with pytest.raises(ValueError):
        jobs.prepare_wasserstein_job(bound(path), source["root"] / "job", series_manifest=source["manifest"])
    assert outside not in opened
    assert not (source["root"] / "job").exists()


@pytest.mark.parametrize("status,converged,cost", [("solver_not_converged", False, 1.0),
    ("single_zero_mass", False, None), ("success", True, float("nan")), ("success", True, -1.0)])
def test_unsolved_or_invalid_distances_cannot_complete(source, status, converged, cost):
    from darsia.fluidflower.wasserstein_job_runner import validate_solver_row
    selected = source["operations"][0]["comparison"]
    with pytest.raises(RuntimeError):
        validate_solver_row(dict(complete_solver_row(selected), status=status, solver_converged=converged,
                                 transport_cost_mass_m=cost), selected)


def test_sparse_and_inconsistent_original_rows_cannot_complete(source):
    from darsia.fluidflower.wasserstein_job_runner import validate_solver_row
    selected = source["operations"][0]["comparison"]
    row = complete_solver_row(selected)
    assert validate_solver_row(row, selected) is row
    with pytest.raises(ValueError, match="complete original row schema"):
        validate_solver_row(dict(selected, status="success", solver_converged=True, transport_cost_mass_m=1.0), selected)
    for change in (dict(source_mass_1_g=None), dict(grid_rows=True), dict(runtime_seconds=float("inf")),
                   dict(mass_based_distance_m=None), dict(plume_based_normalized_distance=float("nan")),
                   dict(pair_mean_detected_mass_g=5.0), dict(solver_warnings="unexpected"),
                   dict(status="both_zero_mass")):
        with pytest.raises(RuntimeError):
            validate_solver_row(dict(row, **change), selected)
    zeros = dict(row, status="both_zero_mass", solver_iterations=0, source_mass_1_g=0.0,
        source_mass_2_g=0.0, pair_mean_detected_mass_g=0.0, transport_cost_mass_m=0.0,
        mass_based_distance_m=0.0, rig_based_normalized_distance=0.0,
        rig_diagonal_normalized_distance=0.0, plume_based_normalized_distance=0.0)
    assert validate_solver_row(zeros, selected) is zeros
    with pytest.raises(RuntimeError, match="Both-zero"):
        validate_solver_row(dict(zeros, transport_cost_mass_m=1.0), selected)
    unavailable_reference = dict(row, injected_mass_reference_g=0.0,
        rig_based_normalized_distance=None, plume_based_normalized_distance=None)
    assert validate_solver_row(unavailable_reference, selected) is unavailable_reference


def test_executor_refuses_unbound_or_float32_execution_before_solver(source, monkeypatch, tmp_path):
    from darsia.fluidflower import wasserstein_job_runner as runner
    from darsia.fluidflower.pipeline import run_io
    monkeypatch.setattr(runner, "validate_wasserstein_task", lambda _: None)
    monkeypatch.setattr(runner, "_solve_operation", lambda *_: pytest.fail("Unadmitted execution reached science"))
    task = dict(precision="float64")
    with pytest.raises(ValueError, match="CPU-float64"):
        runner.run_wasserstein(task, tmp_path / "out", "cpu", "float32")
    monkeypatch.setattr(run_io, "for_task", lambda _: None)
    with pytest.raises(ValueError, match="I/O binding"):
        runner.run_wasserstein(task, tmp_path / "out", "cpu", "float64")
    assert not (tmp_path / "out").exists()


def test_original_thread_cap_restores_after_solver_failure(monkeypatch):
    import cv2
    import threadpoolctl
    from contextlib import contextmanager
    from darsia import _opencv_runtime
    from darsia.fluidflower import wasserstein_job_runner as runner
    state, limits = [7], []
    monkeypatch.setattr(cv2, "getNumThreads", lambda: state[-1])
    monkeypatch.setattr(cv2, "setNumThreads", state.append)
    monkeypatch.setattr(_opencv_runtime, "scientific_thread", lambda _: {})
    @contextmanager
    def capped(*, limits):
        assert limits == 1
        yield
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", capped)
    def fail(*_):
        assert state[-1] == 1
        raise RuntimeError("original solver failure")
    monkeypatch.setattr(runner, "_solve_operation", fail)
    with pytest.raises(RuntimeError, match="original solver failure"):
        runner._solve_capped(None, None, None)
    assert state == [7, 1, 7]


def queued_contract_fixture(source, monkeypatch):
    """Isolated typed hook only; this does not qualify the shared worker route."""
    from darsia.fluidflower.pipeline import alias_job, job_queue
    monkeypatch.setattr(alias_job, "shared_output_admission", lambda *_: None)
    monkeypatch.setattr(job_queue, "validate_task", protocol.validate_wasserstein_task)
    root = source["root"] / "queue-job"
    prepared = jobs.prepare_wasserstein_job(execution(source), root, series_manifest=source["manifest"])
    task = json.loads(Path(prepared["tasks"][0]["task"]["path"]).read_text(encoding="utf-8"))
    job = json.loads((root / "job.json").read_text(encoding="utf-8"))
    return task, job


def test_typed_task_and_job_cannot_substitute_contract_or_precision(source, monkeypatch):
    task, job = queued_contract_fixture(source, monkeypatch)
    assert protocol.validate_wasserstein_claim(task, job) is task
    for key, value in (("precision", "float32"), ("wasserstein_contract_sha256", "e" * 64),
                       ("comparison", dict(task["comparison"], time_hours=9.0))):
        with pytest.raises(ValueError):
            protocol.validate_wasserstein_claim(dict(task, **{key: value}), job)


@pytest.mark.parametrize("changed_format", ["csv", "xlsx"])
def test_actual_result_files_and_scalar_identity_are_verified(source, monkeypatch, changed_format):
    from darsia.fluidflower.wasserstein_job_results import validate_wasserstein_result
    from darsia.fluidflower.wasserstein_materialization import _write_csv, _write_xlsx
    task, job = queued_contract_fixture(source, monkeypatch)
    output = source["root"] / "owned-attempt"
    output.mkdir()
    row = complete_solver_row(task["comparison"])
    files = {"comparison.json": save(output / "comparison.json", row)}
    for filename, writer in (("wasserstein_distances.csv", _write_csv), ("wasserstein_distances.xlsx", _write_xlsx)):
        writer(output / filename, [row])
        files[filename] = bound(output / filename)
    result = dict(format=protocol.RESULT_FORMAT, complete=True, pipeline=protocol.PIPELINE,
        run=task["run"], backend="cpu", precision="float64", comparisons=1, frames=0,
        comparison=task["comparison"], wasserstein_plan=task["wasserstein_plan"],
        wasserstein_contract_sha256=task["wasserstein_contract_sha256"],
        worker_references_closed=True, validation=dict(state="not_scored"), files=files,
        solver_status="success", solver_converged=True)
    assert validate_wasserstein_result(result, task=task, job=job, output_root=output) is result
    with pytest.raises(ValueError, match="completed comparison"):
        validate_wasserstein_result(dict(result, frames=1), task=task, job=job, output_root=output)
    filename = "wasserstein_distances." + changed_format
    writer = _write_csv if changed_format == "csv" else _write_xlsx
    writer(output / filename, [dict(row, transport_cost_mass_m=2.0)])
    files[filename] = bound(output / filename)
    with pytest.raises(ValueError, match="differs from"):
        validate_wasserstein_result(result, task=task, job=job, output_root=output)
    writer(output / filename, [row])
    files[filename] = bound(output / filename)
    (output / "wasserstein_distances.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after computation"):
        validate_wasserstein_result(result, task=task, job=job, output_root=output)


@pytest.mark.parametrize("target_cell_size", [.25, .5])
def test_comparison_executor_matches_original_solver_on_original_fields(source, target_cell_size):
    """Finite numerical parity case for Root's later authorized science QA."""
    from darsia.fluidflower import wasserstein_materialization as original
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    from darsia.fluidflower.wasserstein_job_runner import _solve_operation
    plan, selected = deepcopy(source["plan"]), source["operations"][0]["comparison"]
    plan["configuration"]["spatial_discretization"]["target_cell_size_m"] = target_cell_size
    plan["task_contract"]["spatial_discretization"]["target_cell_size_m"] = target_cell_size
    reseal(plan)
    contract = plan["task_contract"]
    mass = jobs.read_bound(contract["mass_result"])
    refs = original._reference_map(contract)
    runs = [original._load_run_inputs(row, refs, inputs.MASS_PRODUCTS) for row in mass["run_results"]]
    grid = original._domain_grid("whole_active_domain", {}, runs[0], target_cell_size)
    reduced = [original._reduced_field(source["arrays"][(run, "mass_aqueous", 0)], grid,
               label=run + "/expected") for run in ("r1", "r2")]
    weight = original._reduced_weight(source["arrays"][("r1", "transport_weight", 0)], grid,
                                     label="r1/expected-weight")
    assert all(value.dtype.name == "float64" for value in (*reduced, weight))
    assert reduced[0].sum() == pytest.approx(source["arrays"][("r1", "mass_aqueous", 0)].sum())
    if target_cell_size == .5:
        assert grid.target_shape_rc == (2, 2)
        assert reduced[0].shape != source["arrays"][("r1", "mass_aqueous", 0)].shape
    expected = original._solve(original.SolveRequest(order=0, run_1=runs[0], run_2=runs[1], time_index=0,
        domain=grid, component="mass_aqueous", mass_1=reduced[0], mass_2=reduced[1], weight=weight))
    actual = _solve_operation(DataReader(), plan, selected)
    for column in original.RESULT_COLUMNS:
        if column == "runtime_seconds":
            continue
        if isinstance(expected[column], float):
            assert actual[column] == pytest.approx(expected[column], rel=1e-12, abs=1e-12)
        else:
            assert actual[column] == expected[column]
