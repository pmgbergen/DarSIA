"""TESTONLY: a qualified bound profile covers other runs on its head and route (N188 D2).

Results: the qualified bytes per frame times the task's frame count, plus its
largest frame in flight; another run's layout needs the measured kind, grid,
canvas and precision. Cache: the qualified service entitlement, no claim bytes.
Commit, backend, pipeline and precision stay exact. Synthetic metadata only; no
profile, service, ledger or claim is created.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity as cap
from darsia_cache import capacity as cache_capacity
from darsia_cache import control
from darsia_cache import first_qualification as first
from darsia_cache import worker_adapter as adapter_module
from darsia_cache.policy import CachePolicyError
from darsia_cache.worker_adapter import WorkerCacheAdapter
from .test_capacity import bound_output_profile_TESTONLY, native_layout


def other_run_TESTONLY(*, run="ac17", count=41):
    """The qualified 39-frame profile, and a task of another run with its own frames and configuration."""
    qualified, _, wrapped = bound_output_profile_TESTONLY()
    task = dict(qualified, run=run, attempt_id="2" * 32,
                ordered_case_ids=[run + "_f" + str(index) for index in range(1, count + 1)],
                configuration=dict(qualified["configuration"], sha256="e" * 64))
    config = dict(run=run, precision="float32", frames=[dict(case_id=case) for case in task["ordered_case_ids"]])
    return task, config, wrapped


def run_layout_TESTONLY(**changes):
    """Another run's own layout: its own configurations and native box place on the measured grid."""
    value = native_layout()
    value.update(native_configuration_sha256="c" * 64, phase_configuration_sha256="d" * 64,
                 source_box_xywh=[3, 2, 4, 2])
    value.update(changes)
    return value


def estimate_TESTONLY(task, config, wrapped):
    return cap._bound_result_estimate(None, "FF_AC", task, config, wrapped, {},
                                      repository="TESTONLY", metadata_rows=None)


def test_a_bound_profile_estimates_another_run_per_frame(monkeypatch):
    task, config, wrapped = other_run_TESTONLY(count=41)
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: run_layout_TESTONLY())
    value = estimate_TESTONLY(task, config, wrapped)
    per_frame = (wrapped["result_profile"]["measurement"]["total_logical_bytes"] + 38) // 39
    assert value.run == "ac17" and value.case_count == 41
    assert value.measured_native_bytes_per_frame == per_frame
    assert value.native_expected_bytes == per_frame * 41
    assert value.transient_estimated_bytes == 101 + 53  # The largest qualified frame, in flight.


def test_a_bound_profile_estimates_a_shorter_run_too(monkeypatch):
    task, config, wrapped = other_run_TESTONLY(run="ac18", count=12)
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: run_layout_TESTONLY())
    value = estimate_TESTONLY(task, config, wrapped)
    assert value.case_count == 12 and value.native_expected_bytes == value.measured_native_bytes_per_frame * 12


def test_the_qualified_task_itself_still_needs_its_exact_layout(monkeypatch):
    task, config, wrapped = bound_output_profile_TESTONLY()
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: run_layout_TESTONLY())
    with pytest.raises(ValueError, match="differs from measured profile"):
        estimate_TESTONLY(task, config, wrapped)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: native_layout())
    assert estimate_TESTONLY(task, config, wrapped).case_count == 39


@pytest.mark.parametrize("changes", [
    dict(native_shape_hw=[3, 4], native_shape_chw=[3, 3, 4], source_box_xywh=[1, 1, 4, 3]),  # Another grid.
    dict(phase_shape_hw=[4, 6]),  # Another canvas.
    dict(native_shape_chw=[4, 2, 4]),  # Other channels.
    dict(native_task_kind="raw_rig", raw_source_binding_sha256="e" * 64),  # Another kind.
    dict(precision="float64", feature_rgb_dtype="float64"),
])
def test_another_run_needs_the_measured_kind_grid_canvas_and_precision(monkeypatch, changes):
    task, config, wrapped = other_run_TESTONLY()
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: run_layout_TESTONLY(**changes))
    with pytest.raises(ValueError):
        estimate_TESTONLY(task, config, wrapped)


@pytest.mark.parametrize("key,value", [("commit", "f" * 40), ("backend", "cuda"),
                                       ("precision", "float64"), ("pipeline", "lattice-phase-n1")])
def test_a_bound_profile_never_estimates_another_head_or_route(monkeypatch, key, value):
    task, config, wrapped = other_run_TESTONLY()
    task[key] = value
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: pytest.fail("No writer read off its route"))
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: pytest.fail("No layout read off its route"))
    with pytest.raises(ValueError, match="qualified head and route"):
        estimate_TESTONLY(task, config, wrapped)


@pytest.mark.parametrize("option", [dict(storage={"TESTONLY": "not measured"}),
                                    dict(pixel_domain={"TESTONLY": "not measured"}),
                                    dict(gas_probability_storage=None),
                                    dict(probability_storage={"TESTONLY": "not measured"})])
def test_another_run_with_optional_outputs_needs_its_own_profile(monkeypatch, option):
    task, config, wrapped = other_run_TESTONLY()
    config.update(option)
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: run_layout_TESTONLY())
    with pytest.raises(ValueError, match="qualified head and route"):
        estimate_TESTONLY(task, config, wrapped)


def test_another_run_may_name_only_its_own_configurations_binding_and_box():
    measured = dict(native_layout(), native_task_kind="raw_rig", raw_source_binding_sha256="a" * 64)
    requested = dict(measured, native_configuration_sha256="c" * 64, phase_configuration_sha256="d" * 64,
                     raw_source_binding_sha256="e" * 64, source_box_xywh=[3, 2, 4, 2])
    cap._check_measured_native_layout({"measured_native_layout": measured}, requested, other_run=True)
    with pytest.raises(ValueError, match="differs from measured profile"):
        cap._check_measured_native_layout({"measured_native_layout": measured}, requested)
    with pytest.raises(ValueError, match="native processor kind differs"):
        cap._check_measured_native_layout({"measured_native_layout": measured}, native_layout(), other_run=True)


@pytest.fixture
def bound_cache_TESTONLY(tmp_path, monkeypatch):
    """One in-memory qualified service with a typed volume-bound job profile."""
    def pin(path):
        return dict(path=str(path), sha256="1" * 64, size_bytes=1)

    commit = "3" * 40
    qualified = dict(commit=commit, pipeline="native-phase-n1", backend="cpu", precision="float32",
                     run="ac52", ordered_case_ids=["ac52_f" + str(index) for index in range(1, 40)])
    repository = tmp_path / "repository"
    pipeline = repository / "src" / "darsia" / "fluidflower" / "pipeline"
    producers = [pin(path) for path in (Path(adapter_module.__file__), pipeline / "worker_io.py",
        pipeline / "capacity.py", pipeline / "runner.py", pipeline / "series.py",
        pipeline.parent / "native" / "sidecar.py", pipeline.parent / "mass" / "n1_pipeline.py")]
    runtime_sources = [pin(tmp_path / suffix) for suffix in ("numba/core/caching.py",
        "matplotlib/font_manager.py", "tempfile.py", "importlib/_bootstrap_external.py")]
    service, namespace, stores = pin(tmp_path / "provider.json"), pin(tmp_path / "namespace.json"), pin(tmp_path / "stores.toml")
    cap_document = dict(TESTONLY="fixed 500 GiB volume")
    profile = dict(format=first.BOUND_JOB_PROFILE, purpose="jobs", scope=qualified, provider=service,
        namespace_admission=namespace, stores_source=stores, writer_contract=first.BOUND_WRITER_CONTRACT,
        additional_claim_cache_bytes=0, allocation_model="shared-service-only-no-claim-files-v1",
        result_estimate_excludes_cache=True, actual_writer_IO_proven=False,
        sampled_highwater_is_lifetime_peak=False, operational_service_entitlement_bytes=200,
        capacity_bound=dict(volume_cap_observation=dict(binding=cap_document)),
        producer_source_closure=producers, runtime_source_closure=runtime_sources,
        measurement=pin(tmp_path / "measurement.json"), completion=pin(tmp_path / "completion.json"),
        qualification=dict(TESTONLY="qualification bindings"))
    documents = {profile["measurement"]["path"]: dict(complete=True, cache_operational_reserve_crossed=False,
                                                      first_stop=None),
                 profile["completion"]["path"]: dict(complete=True, passed=True, scope=dict(qualified))}
    volume_cap = SimpleNamespace(maximum_size_bytes=500 * 1024**3, document=lambda: cap_document)
    lease = SimpleNamespace(profile_binding=service, source=stores, maximum=200, package_file=__file__,
        document=dict(cache_bootstrap=dict(namespace_admission=namespace)),
        policy=SimpleNamespace(volume_cap=volume_cap), profile=dict(module_closure=producers + runtime_sources))
    adapter = WorkerCacheAdapter.__new__(WorkerCacheAdapter)
    adapter.runtime = SimpleNamespace(commit=commit, backend="cpu", repository=repository)
    adapter.lease, adapter.profiles, adapter.claims = lease, {}, {}
    resolved = []

    def resolve(binding, kind, read, *, stores_source=None, read_authorities=None):
        resolved.append((binding, kind, stores_source))
        return profile

    monkeypatch.setattr(first, "resolve_profile", resolve)
    monkeypatch.setattr(cache_capacity, "read_bound", lambda bound: (documents[bound["path"]], b""))
    task = dict(qualified, attempt_id="b" * 32)
    return SimpleNamespace(adapter=adapter, policy=SimpleNamespace(profiles=dict(jobs=service)),
                           profile=profile, documents=documents, task=task, qualified=qualified,
                           resolved=resolved, service=service)


def other_cache_task_TESTONLY(f, *, run="ac17", count=41):
    return dict(f.task, run=run, ordered_case_ids=[run + "_f" + str(index) for index in range(1, count + 1)])


def test_a_bound_cache_profile_covers_another_run_on_its_head_and_route(bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    task = other_cache_task_TESTONLY(f)
    maximum, binding, profile, _, _ = control.applicable_job_cache_profile(f.policy, f.adapter, task, None)
    assert maximum == 0 and binding == f.service and profile is f.profile
    assert f.adapter.profiles[task["attempt_id"]]["qualification_kind"] == "physical-volume-bound"
    assert f.resolved == [(f.service, "jobs", f.adapter.lease.source)]


def test_the_qualified_task_and_a_projection_are_covered_as_before(bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    assert control.applicable_job_cache_profile(f.policy, f.adapter, f.task, None)[0] == 0
    f.adapter.profiles.clear()
    control.applicable_job_cache_profile(f.policy, f.adapter, other_cache_task_TESTONLY(f), None, projection=True)
    assert f.adapter.profiles == {}, "A projection never records a claim proof"


@pytest.mark.parametrize("key,value", [("commit", "4" * 40), ("backend", "cuda"),
                                       ("precision", "float64"), ("pipeline", "lattice-phase-n1")])
def test_a_bound_cache_profile_never_covers_another_head_or_route(bound_cache_TESTONLY, key, value):
    f = bound_cache_TESTONLY
    task = other_cache_task_TESTONLY(f)
    task[key] = value
    with pytest.raises(CachePolicyError, match="qualified head and route"):
        control.applicable_job_cache_profile(f.policy, f.adapter, task, None)
    assert f.adapter.profiles == {}


def test_another_run_must_be_a_full_task_while_the_qualified_task_is_covered_as_before(bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    lean = dict(other_cache_task_TESTONLY(f), storage_profile="lean", format_version=2)
    with pytest.raises(CachePolicyError, match="qualified head and route"):
        control.applicable_job_cache_profile(f.policy, f.adapter, lean, None)
    full = dict(other_cache_task_TESTONLY(f), storage_profile="full", format_version=1)
    assert control.applicable_job_cache_profile(f.policy, f.adapter, full, None)[0] == 0
    qualified_lean = dict(f.task, storage_profile="lean", format_version=2)
    assert control.applicable_job_cache_profile(f.policy, f.adapter, qualified_lean, None)[0] == 0


@pytest.mark.parametrize("change", [
    lambda f: setattr(f.adapter.runtime, "commit", "4" * 40),  # The service runs another head.
    lambda f: setattr(f.adapter.runtime, "backend", "cuda"),
    lambda f: f.documents[f.profile["completion"]["path"]].update(scope=dict(f.qualified, run="ac53")),
    lambda f: f.profile.update(operational_service_entitlement_bytes=201),
])
def test_the_service_must_be_the_qualified_one_whichever_run_it_claims(bound_cache_TESTONLY, change):
    f = bound_cache_TESTONLY
    change(f)
    with pytest.raises(CachePolicyError, match="not applicable"):
        control.applicable_job_cache_profile(f.policy, f.adapter, other_cache_task_TESTONLY(f), None)
    assert f.adapter.profiles == {}
