"""Real refit statistics/catalogue inputs; no cache or science admission waiver."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity, refit_capacity, refit_runner, store_runtime
from darsia.fluidflower.store_map import parse_store_map
from test_refit_jobs import REPOSITORY, prepare, series  # noqa: F401 -- original fixture
from tests.unit.gui.test_bound_profile_other_runs import bound_cache_TESTONLY  # noqa: F401
from tests.unit.gui.test_capacity import ledger_fixture, owner
from darsia.fluidflower.pipeline.store_runtime import RunUnavailable
from darsia_cache import control
from darsia_cache.worker_adapter import _source_pins


@pytest.fixture
def refit_data(series):
    prepared, job_root = prepare(series)
    stores = parse_store_map(('[machine]\nid = "Main"\n\n[stores.TEST]\n'
        f"series = '{series['root']}'\nresults = '{series['tmp'] / 'out'}'\n").encode())
    task = json.loads(Path(prepared["tasks"][0]["task"]["path"]).read_bytes())
    rows = store_runtime.resolve_catalog(task, stores, "TEST")
    plan = capacity.configuration_for_task(stores, "TEST", task, rows)
    return SimpleNamespace(series=series, prepared=prepared, job_root=job_root,
                           stores=stores, task=task, rows=rows, plan=plan)


def _path_for(rows):
    def path_for(use, pin):
        selected = [row for row in rows if use in row.uses]
        assert len(selected) == 1
        assert selected[0].reference["sha256"] == pin["sha256"]
        assert selected[0].reference["bytes"] == pin["size_bytes"]
        return selected[0].path
    return path_for


def test_original_global_and_fold_budgets_retain_every_statistics_row(refit_data):
    data = refit_data
    before = {row.path: row.path.read_bytes() for row in data.rows}
    global_value = refit_capacity.budget(data.task, data.plan, path_for_use=_path_for(data.rows))
    fold_task = json.loads(Path(data.prepared["tasks"][1]["task"]["path"]).read_bytes())
    fold_rows = store_runtime.resolve_catalog(fold_task, data.stores, "TEST")
    fold_plan = capacity.configuration_for_task(data.stores, "TEST", fold_task, fold_rows)
    folded = refit_capacity.budget(fold_task, fold_plan, path_for_use=_path_for(fold_rows))
    assert len(global_value["statistics"]) == len(folded["statistics"]) == 46
    assert global_value["statistics"] == folded["statistics"]  # No LORO memory discount.
    assert global_value["decoded_statistics_bytes"] > sum(row["size_bytes"] for row in global_value["statistics"])
    assert global_value["workspace_estimated_bytes"] >= 64 * global_value["decoded_statistics_bytes"]
    assert global_value["transient_estimated_bytes"] >= global_value["products_estimated_bytes"]
    assert global_value["RAM_limit_enforced"] is False
    # Original cross-host locators preserve literal source identity; the
    # existing rooted consumer still supplies the actual same-byte input.
    aliased = deepcopy(data.plan)
    selected = aliased["statistics"]["r1"][0]
    physical = deepcopy(selected["statistics"])
    selected["statistics"]["path"] = str(data.series["tmp"] / "unavailable-producer/statistics.json.gz")
    aliased["data_aliases"] = [dict(logical=deepcopy(selected["statistics"]), physical=physical)]
    same_bytes = refit_capacity.budget(data.task, aliased, path_for_use=_path_for(data.rows))
    assert same_bytes["statistics"] == global_value["statistics"]
    aliased["data_aliases"][0]["physical"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="alias changes source SHA"):
        refit_capacity.budget(data.task, aliased, path_for_use=lambda *args: pytest.fail("No bad alias read"))
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (data.series["tmp"] / "out" / "refit-job1").exists()


def test_result_dispatch_binds_current_git_writers_without_a_native_estimate(refit_data, monkeypatch):
    data = refit_data
    # This tests code/data budgeting, not execution of the fixture's fake commit.
    commit = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    task = dict(data.task, commit=commit)
    monkeypatch.setattr(capacity, "settings", lambda *args: pytest.fail("No native measurement can estimate refit"))
    value = capacity.load_estimate(data.stores, "TEST", task, data.plan,
                                   repository=REPOSITORY, metadata_rows=data.rows)
    assert isinstance(value, refit_capacity.RefitEstimate)
    assert {row["relative_path"] for row in value.writer_identity} == set(refit_capacity.PRODUCT_WRITERS)
    assert all(row["relative_path"].startswith("src/darsia/fluidflower/") for row in value.writer_identity)
    assert set(refit_capacity.REQUIRED_SERVICE_SOURCES) - set(refit_capacity.PRODUCT_WRITERS) == {
        "src/darsia_cache/control.py", "src/darsia_cache/worker_adapter.py"}
    for relative in sorted(set(refit_capacity.REQUIRED_SERVICE_SOURCES) - set(refit_capacity.PRODUCT_WRITERS)):
        with pytest.raises(ValueError, match="repository-relative"):
            capacity._writer_identity(REPOSITORY, commit, dict(relative_path=relative, sha256="f" * 64))
    assert value.products_per_frame_estimated_bytes > 0
    assert value.products_estimated_bytes == value.case_count * value.products_per_frame_estimated_bytes
    assert value.needed(reserve_bytes=7) == value.products_estimated_bytes + value.transient_estimated_bytes + 7
    assert refit_capacity.require_estimate(task, value) is True
    with pytest.raises(ValueError, match="source-bound"):
        refit_capacity.require_estimate(dict(task, configuration=dict(task["configuration"], sha256="f" * 64)), value)


@pytest.mark.parametrize("fault", ["sha", "size", "missing_role"])
def test_refit_estimate_refuses_changed_or_unnamed_original_input(refit_data, fault):
    data = refit_data
    rows = list(data.rows)
    index = next(index for index, row in enumerate(rows) if any(use.endswith(".statistics") for use in row.uses))
    original = rows[index]
    reference = dict(original.reference)
    uses = list(original.uses)
    if fault == "sha":
        reference["sha256"] = "f" * 64
    elif fault == "size":
        reference["bytes"] += 1
    else:
        uses = [use for use in uses if not use.endswith(".statistics")]
    rows[index] = SimpleNamespace(path=original.path, reference=reference, uses=uses)
    with pytest.raises(ValueError, match="rooted original mass calibration input role"):
        refit_capacity.refit_estimate(data.stores, "TEST", data.task, data.plan,
                                       repository=REPOSITORY, metadata_rows=rows)


@pytest.mark.parametrize("fault", ["variant_f", "output_option", "case_order", "threads"])
def test_unbudgeted_refit_options_and_order_refuse_before_statistics_reads(refit_data, fault):
    data = refit_data
    task, plan = deepcopy(data.task), deepcopy(data.plan)
    if fault == "variant_f":
        plan["variant_f"]["enabled"] = True
    elif fault == "output_option":
        plan["gas_probability_storage"] = {}
    elif fault == "case_order":
        task["ordered_case_ids"].reverse()
    else:
        task["threads"] = 2
    with pytest.raises(ValueError):
        refit_capacity.budget(task, plan, path_for_use=lambda *args: pytest.fail("No input read on rejected route"))


def test_shared_decoder_retains_strict_json_pin_and_bounded_expansion(tmp_path, monkeypatch):
    path = tmp_path / "statistics.json.gz"
    def write(raw):
        payload = gzip.compress(raw)
        path.write_bytes(payload)
        return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    raw = b'{"value":3}'
    pin = write(raw)
    assert refit_capacity.read_statistics_bound(path, pin) == ({"value": 3}, len(raw))
    calls = []
    reader = SimpleNamespace(path=lambda descriptor: calls.append(descriptor) or path)
    assert refit_runner.read_statistics(reader, pin) == {"value": 3}
    assert calls == [pin, pin]  # Original before/after role binding remains.
    with pytest.raises(ValueError, match="changed"):
        refit_capacity.read_statistics_bound(path, dict(pin, sha256="f" * 64))
    pin = write(b'{"value":3,"value":4}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        refit_capacity.read_statistics_bound(path, pin)
    pin = write(b'{"value":NaN}')
    with pytest.raises(ValueError, match="Nonfinite JSON number"):
        refit_capacity.read_statistics_bound(path, pin)
    damaged = gzip.compress(raw)[:-5]
    path.write_bytes(damaged)
    pin = dict(path=str(path), sha256=hashlib.sha256(damaged).hexdigest(), size_bytes=len(damaged))
    with pytest.raises(ValueError, match="gzip is invalid"):
        refit_capacity.read_statistics_bound(path, pin)
    pin = write(b'{"value":"' + b'a' * 4096 + b'"}')
    monkeypatch.setattr(refit_capacity, "MAX_METADATA_BYTES", 128)  # Finite decoder boundary fixture only.
    with pytest.raises(ValueError, match="Decoded mass calibration statistics"):
        refit_capacity.read_statistics_bound(path, pin)


def _bound_service_fixture(f, task):
    """Extend only synthetic startup-source metadata, never qualify a service."""
    f.qualified["commit"] = task["commit"]
    f.adapter.runtime.commit = task["commit"]
    f.documents[f.profile["completion"]["path"]]["scope"]["commit"] = task["commit"]
    producers = f.profile["producer_source_closure"]
    existing = {row["path"] for row in producers}
    for relative in refit_capacity.REQUIRED_SERVICE_SOURCES:
        path = str(f.adapter.runtime.repository / relative)
        if path not in existing:
            producers.append(dict(path=path, sha256="1" * 64, size_bytes=1))
    f.profile["runtime_source_closure"].append(dict(
        path=str(f.adapter.runtime.repository / "runtime/scipy/optimize/_lsq/least_squares.py"),
        sha256="1" * 64, size_bytes=1))
    f.adapter.lease.profile["module_closure"] = producers + f.profile["runtime_source_closure"]


def _actual_estimate(data):
    commit = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    task = dict(data.task, commit=commit, attempt_id="b" * 32)
    value = refit_capacity.refit_estimate(data.stores, "TEST", task, data.plan,
        repository=REPOSITORY, metadata_rows=data.rows)
    return task, value


def test_refit_cache_route_requires_original_service_and_complete_refit_closure(refit_data, bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    task, value = _actual_estimate(refit_data)
    # Even a genuine budget is insufficient when the startup proof omits refit.
    f.qualified["commit"] = task["commit"]
    f.adapter.runtime.commit = task["commit"]
    f.documents[f.profile["completion"]["path"]]["scope"]["commit"] = task["commit"]
    with pytest.raises(ValueError, match="mass calibration source boundary"):
        control.applicable_job_cache_profile(f.policy, f.adapter, task, value)
    assert f.adapter.profiles == {}
    _bound_service_fixture(f, task)
    maximum, binding, _, _, _ = control.applicable_job_cache_profile(f.policy, f.adapter, task, value)
    assert maximum == 0 and binding == f.service  # Same service; no fabricated measured zero profile.
    assert f.adapter.profiles[task["attempt_id"]]["task"] == task
    f.adapter.profiles.clear()
    control.applicable_job_cache_profile(f.policy, f.adapter, task, value, projection=True)
    assert f.adapter.profiles == {}
    # The product namespace split must not drop either cache module from the
    # actual accepted service closure. Keep the original union exact and prove
    # the dedicated requirement as well as refusal by the original adapter.
    original_producers = list(f.profile["producer_source_closure"])
    for relative in ("src/darsia_cache/control.py", "src/darsia_cache/worker_adapter.py"):
        omitted = str(f.adapter.runtime.repository / relative)
        f.profile["producer_source_closure"] = [row for row in original_producers if row["path"] != omitted]
        f.adapter.lease.profile["module_closure"] = (
            f.profile["producer_source_closure"] + f.profile["runtime_source_closure"])
        with pytest.raises(ValueError, match="source boundary is outside the actual startup closure"):
            refit_capacity.require_service_sources(f.adapter.runtime.repository,
                _source_pins(f.profile["producer_source_closure"]),
                _source_pins(f.profile["runtime_source_closure"]))
        with pytest.raises(ValueError, match=(
                "source boundary is outside the actual startup closure|"
                "actual startup writer/source/runtime closure")):
            control.applicable_job_cache_profile(f.policy, f.adapter, task, value)
        assert f.adapter.profiles == {}
    f.profile["producer_source_closure"] = original_producers
    f.adapter.lease.profile["module_closure"] = original_producers + f.profile["runtime_source_closure"]
    # Startup module membership and the original actual source union stay exact.
    f.profile["producer_source_closure"] = [row for row in f.profile["producer_source_closure"]
        if not row["path"].replace("\\", "/").endswith("/pipeline/refit_capacity.py")]
    with pytest.raises(ValueError, match="actual startup"):
        control.applicable_job_cache_profile(f.policy, f.adapter, task, value)


@pytest.mark.parametrize("short", [False, True])
def test_original_claim_and_result_ledger_keep_whole_refit_allowance(
        refit_data, bound_cache_TESTONLY, tmp_path, monkeypatch, short):
    """Real result ledger/ClaimCapacity; finite service metadata boundary only.

    This does not bootstrap/qualify a cache service or run the fitter. The
    original source/cache applicability predicate runs at its claim boundary;
    the pre-existing physical cache ledger remains separately qualified.
    """
    data, f = refit_data, bound_cache_TESTONLY
    task, value = _actual_estimate(data)
    _bound_service_fixture(f, task)
    # The original finite admission binds checked_container to the helper's
    # existing fixture directory, just as its own pytest tmp_path callers do.
    ledger_root = tmp_path / "result-ledger"
    ledger_root.mkdir()
    ledger, _, state = ledger_fixture(ledger_root, monkeypatch)
    from darsia.fluidflower import job_filesystem
    proof = job_filesystem.inspect_results_root(ledger.results_root)
    assert job_filesystem.valid_results_admission(proof, ledger.results_root)
    assert Path(proof.checked_container).resolve(strict=True) == ledger_root.resolve(strict=True)
    state["free_bytes"] = value.needed(reserve_bytes=ledger.reserve) - int(short)
    runtime = SimpleNamespace(stores=data.stores, job=dict(series_id="TEST"), root=data.job_root,
        capacity_refusals={}, availability=lambda actual: data.rows, capacities={})
    boundary = capacity.ClaimCapacity(runtime, owner())
    boundary.ledger = ledger
    calls = []
    class FiniteServiceBoundary:
        def prepare(self, actual, estimate):
            calls.append("prepare")
            control.applicable_job_cache_profile(f.policy, f.adapter, actual, estimate)
        def acquire(self):
            calls.append("acquire")
        def check(self, actual, estimate, other):
            assert other == 0
            calls.append("cache_check")
            return dict(TESTONLY="separately-qualified physical-cache ledger boundary")
        def bind(self, claim):
            calls.append("cache_bind")
        def unlock(self):
            pass
    boundary.cache = FiniteServiceBoundary()
    try:
        if short:
            with pytest.raises(RunUnavailable, match="leave this whole run pending"):
                boundary.availability(task)
            assert not list(ledger.root.glob("r-*/reserved.json"))
        else:
            assert boundary.availability(task) == data.rows
            claim = SimpleNamespace(task=task, token=task["attempt_id"])
            boundary.bind(claim)
            _, reservation, _ = runtime.capacities[claim.token]
            retained = json.loads((reservation / "reserved.json").read_bytes())
            assert retained["estimate"]["workspace_estimated_bytes"] == value.workspace_estimated_bytes
            assert ledger.remaining() == value.products_estimated_bytes + value.transient_estimated_bytes
            ledger.progress(reservation, owner(), dict(state="refit_progress", run=task["run"], step="complete"))
            assert ledger.remaining() == value.products_estimated_bytes + value.transient_estimated_bytes
            assert calls[-1] == "cache_bind"
    finally:
        boundary._unlock()
    assert calls[:3] == ["prepare", "acquire", "cache_check"]
