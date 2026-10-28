"""Original private ledgers plus explicit Setup output ceilings.

Only filesystem/volume observations are synthetic. The tests use actual
CapacityLedger locks, immutable reservations, accounting and release records,
and the original full RAW-request reader over tiny private byte inputs. They
do not qualify a physical volume, RAW decoder, scientific grid or cache model.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_filesystem as filesystem
from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline import series_setup_admission as admission
from .test_series_setup_capacity import prepared_raw_request, _json
from .test_series_setup_protocol import _task, _operation


def _admission(tmp_path, monkeypatch, *, metadata_cap, raw_cap=0, shared=True,
               scratch=False, bind=True, metadata_file_cap=None, metadata_roots=None,
               task=None, configuration=None):
    """Return a real bound SetupAdmission for other bounded-writer tests.

    Every written byte remains under ``tmp_path``. The tiny RAW request has
    deliberately synthetic leaf files; its original reader still verifies
    their exact descriptors. No admission, reservation or release is mocked.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    operation = "build" if raw_cap else "apply"
    if task is None:
        task = _task(operation, run="sample")
        task.update(job_id="setup-job", attempt_id="1" * 32)
    else:
        task = deepcopy(task)
    config = deepcopy(configuration) if configuration is not None else _operation(operation, run="sample")
    config["metadata_output"] = dict(maximum_output_bytes=metadata_cap,
                                     maximum_file_bytes=metadata_cap if metadata_file_cap is None else metadata_file_cap)
    raw_root = None
    if raw_cap:
        scientific_inputs = tmp_path / "scientific-inputs"
        scientific_inputs.mkdir()
        _old_task, _old_config, request_path, request = prepared_raw_request.__wrapped__(scientific_inputs)
        request["output"].update(maximum_output_bytes=raw_cap, maximum_file_bytes=raw_cap)
        config["raw_build_request"] = _json(request_path, request)
        raw_root = Path(request["output"]["intended_root"])
    if metadata_roots is None:
        metadata_root = tmp_path / "publication"
        metadata_root.mkdir()
        metadata_roots = (metadata_root,)
    else:
        metadata_roots = tuple(Path(root).resolve() for root in metadata_roots)
        assert metadata_roots and all(root.is_dir() for root in metadata_roots)
        metadata_root = metadata_roots[0]
    scratch_root = tmp_path / "scratch"
    if scratch:
        scratch_root.mkdir()
    config["output_scope"] = dict(metadata_roots=list(map(str, metadata_roots)),
                                  scratch_roots=[str(scratch_root)] if scratch else [])
    roots = dict(series=str(tmp_path / "series"), results=str(metadata_root))
    profile = tmp_path / "unused-schema-profile.json"
    profile.write_bytes(b'{"unused":true}\n')
    profile_pin = dict(path=str(profile), sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), size_bytes=profile.stat().st_size)
    stores = SimpleNamespace(machine_id="fixture", roots=lambda _series: roots,
        document=dict(capacity=dict(state_directory=str(tmp_path / "capacity-state"),
                                   profiles={"sample-series": profile_pin}, reserve_bytes=7)))

    def local(path, **_kwargs):
        container = Path(path).resolve().parent
        while not container.exists():
            container = container.parent
        return filesystem.LocalFileAdmission(True, "fixture-fixed", "synthetic OS observation",
                                             "private test only", str(container))

    monkeypatch.setattr(filesystem, "inspect_local_file", local)
    states = {}

    def volume(path):
        path = Path(path).resolve()
        is_raw = raw_root is not None and path.is_relative_to(raw_root)
        is_scratch = scratch and path.is_relative_to(scratch_root)
        identity = "shared-volume" if shared else "raw-volume" if is_raw or is_scratch else "publication-volume"
        state = states.setdefault(identity, dict(volume_id=identity, volume_anchor=str(tmp_path),
                                                 free_bytes=10**9, checked_container=str(tmp_path)))
        return dict(state)

    context = dict(host="fixture-host", service_account_sid="S-1-5-21-100",
                   common_root=str(tmp_path / "machine-common"))

    def ledger_factory(selected, series_id, root):
        return capacity.CapacityLedger(selected, series_id, root, volume_provider=volume,
                                       context_provider=lambda: dict(context))

    actual = admission.prepare_setup_admission(stores, "sample-series", task, config,
        metadata_roots=metadata_roots, scratch_roots=(scratch_root,) if scratch else (),
        ledger_factory=ledger_factory)
    owner = dict(machine="fixture-cpu", pid=123, process_create_time="12345.125")
    claim = SimpleNamespace(task=deepcopy(task), token=task["attempt_id"])
    actual.test_context = SimpleNamespace(task=task, config=config, owner=owner, claim=claim,
        states=states, stores=stores, volume=volume, context=context, raw_root=raw_root,
        metadata_root=metadata_root, scratch_root=scratch_root if scratch else None,
        job_root=tmp_path / "job")
    if bind:
        with actual.locked():
            actual.check()
            actual.bind(actual.test_context.job_root, claim, owner)
    return actual


def test_two_actual_volumes_keep_raw_and_metadata_caps_separate(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False)
    ceilings = {row.volume_id: row.maximum_output_bytes for row in actual.volume_reservations}
    assert ceilings == {"raw-volume": 100, "publication-volume": 20}
    assert len(actual.bound_reservations) == 2
    for row in actual.bound_reservations:
        record = json.loads((Path(row["path"]) / "reserved.json").read_bytes())
        assert record["format"] == "darsia-local-capacity-reservation-v1"
        assert record["not_a_measured_profile"] is True
        assert record["maximum_output_bytes"] in {20, 100}
    assert actual.metadata_limits == dict(maximum_output_bytes=20, maximum_file_bytes=20)


def test_repeated_build_admission_never_reads_scientific_sources(tmp_path, monkeypatch):
    """The real locked admission and bind use request caps, not RAW verification."""
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, bind=False)
    config = actual.test_context.config
    request = json.loads(Path(config["raw_build_request"]["path"]).read_bytes())
    sources = {Path(pin["path"]) for pin in [request["baseline"], request["profile"],
        request["series_context"]["manifest"], request["shape_basis_certificate"],
        *request["configuration_files"], *(row["source"] for row in request["protocol_inputs"])]}
    original, opened = Path.open, []

    def guarded(path, *args, **kwargs):
        if path in sources:
            opened.append(path)
            raise AssertionError("An unclaimed Build admission read scientific input")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    for _ in range(20):
        with actual.locked():
            actual.check()
    with actual.locked():
        actual.check()
        actual.bind(actual.test_context.job_root, actual.test_context.claim,
                    actual.test_context.owner)
    assert not opened
    assert actual.bound_reservations


def test_claim_writes_reuse_transport_admission_but_final_bind_refreshes(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, bind=False)
    original, checks = actual._inspect_results, []
    refuse = False
    def inspect(path):
        checks.append(path)
        proof = original(path)
        return SimpleNamespace(**{**vars(proof), "allowed": False}) if refuse else proof
    actual._inspect_results = inspect
    with actual.locked():
        actual.check()
        initial = len(checks)
        for _ in range(20):
            actual._revalidate_domains(selected=actual.test_context.metadata_root / "claim.json")
        assert len(checks) == initial
        # This does not become a cached permission: final bind checks the
        # current filesystem/space even while the same lock is still held.
        refuse = True
        with pytest.raises(ValueError, match="lost positive filesystem"):
            actual.bind(actual.test_context.job_root, actual.test_context.claim,
                        actual.test_context.owner)
    assert len(checks) > initial and not actual.bound_reservations


def test_claim_transport_reuse_still_refuses_replaced_output_directory(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, bind=False)
    root = actual.test_context.metadata_root
    with actual.locked():
        actual.check()
        root.rename(root.with_name("original-publication"))
        root.mkdir()
        with pytest.raises(ValueError, match="directory identity"):
            actual._revalidate_domains(selected=root / "claim.json")


def test_same_volume_counts_one_global_metadata_allowance_even_with_scratch(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=True, scratch=True)
    assert [row.maximum_output_bytes for row in actual.volume_reservations] == [120]
    assert len(actual.bound_reservations) == 1
    row = actual.volume_reservations[0]
    assert row.raw_maximum_output_bytes == 100 and row.metadata_maximum_output_bytes == 20
    assert {root["role"] for root in row.roots} == {"raw", "metadata", "scratch"}
    assert actual.require_write(actual.test_context.metadata_root / "next.json")
    assert actual.require_write(actual.test_context.scratch_root / "partial.tmp")


def test_distinct_metadata_and_scratch_volumes_each_keep_full_shared_allowance(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, scratch=True)
    ceilings = {row.volume_id: row.maximum_output_bytes for row in actual.volume_reservations}
    assert ceilings == {"raw-volume": 120, "publication-volume": 20}
    # The bounded sink still has only one 20-byte global budget, not 20 per disk.
    assert actual.metadata_limits["maximum_output_bytes"] == 20


def test_existing_open_claims_are_included_before_any_new_bind(tmp_path, monkeypatch):
    first = _admission(tmp_path, monkeypatch, metadata_cap=100)
    data = first.test_context
    original = capacity.CapacityLedger(data.stores, "sample-series", data.metadata_root,
        volume_provider=data.volume, context_provider=lambda: dict(data.context))
    next_task = deepcopy(data.task)
    next_task["attempt_id"] = "2" * 32
    second = admission.prepare_setup_admission(data.stores, "sample-series", next_task, data.config,
        metadata_roots=(data.metadata_root,), ledger_factory=lambda *_args: original)
    data.states["shared-volume"]["free_bytes"] = 206  # 100 + open100 + reserve7 needs207.
    with second.locked():
        with pytest.raises(ValueError, match="existing open claims"):
            second.check()
    assert not second.bound_reservations
    with original.locked():
        assert original.remaining() == 100


def test_insufficient_second_volume_does_not_bind_first_volume(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, bind=False)
    actual.test_context.states["publication-volume"]["free_bytes"] = 26  # 20+7.
    with actual.locked():
        with pytest.raises(ValueError, match="exceed free space"):
            actual.check()
    assert not actual.bound_reservations
    assert not list((tmp_path / "capacity-state").glob("*/r-*/reserved.json"))


def test_locks_use_stable_original_scope_order_and_leave_reservations_live(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, bind=False)
    events = []
    original = capacity.CapacityLedger.locked

    @contextmanager
    def observed(ledger, *args, **kwargs):
        with original(ledger, *args, **kwargs):
            events.append(("enter", ledger.key))
            try:
                yield ledger
            finally:
                events.append(("leave", ledger.key))

    monkeypatch.setattr(capacity.CapacityLedger, "locked", observed)
    with actual.locked():
        actual.check()
        actual.bind(actual.test_context.job_root, actual.test_context.claim, actual.test_context.owner)
    expected = sorted(row.capacity_key for row in actual.volume_reservations)
    assert [key for action, key in events if action == "enter"] == expected
    assert [key for action, key in events if action == "leave"] == list(reversed(expected))
    assert all(not (Path(row["path"]) / "released.json").exists() for row in actual.bound_reservations)


def test_whole_cap_does_not_shrink_from_frame_progress_and_history_survives_release(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=100)
    data = actual.test_context
    ledger = capacity.CapacityLedger(data.stores, "sample-series", data.metadata_root,
        volume_provider=data.volume, context_provider=lambda: dict(data.context))
    reserved = Path(actual.bound_reservations[0]["path"])
    before = (reserved / "reserved.json").read_bytes()
    with ledger.locked():
        assert ledger.remaining() == 100
        with pytest.raises(ValueError, match="[Ww]hole|Setup|setup"):
            ledger.progress(reserved, data.owner,
                dict(state="frame_completed", complete=True, run="sample", case_id="apply"))
        assert ledger.remaining() == 100 and not list(reserved.glob("p-*.json"))
    actual.release(data.owner)
    with ledger.locked():
        assert ledger.remaining() == 0
    assert (reserved / "reserved.json").read_bytes() == before
    assert (reserved / "released.json").is_file()
    with pytest.raises(RuntimeError, match="active Setup reservation"):
        actual.require_write(data.metadata_root / "late.json")


def test_partial_bind_failure_preserves_truthful_prefix_without_automatic_release(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, shared=False, bind=False)
    original = capacity.CapacityLedger.bind
    calls = []

    def fail_second(ledger, *args):
        calls.append(ledger.key)
        if len(calls) == 2:
            raise OSError("synthetic second-volume IO failure")
        return original(ledger, *args)

    monkeypatch.setattr(capacity.CapacityLedger, "bind", fail_second)
    with actual.locked():
        actual.check()
        with pytest.raises(admission.SetupBindError, match="retained reservations") as failed:
            actual.bind(actual.test_context.job_root, actual.test_context.claim, actual.test_context.owner)
    assert failed.value.bound_reservations == actual.bound_reservations
    assert len(actual.bound_reservations) == 1
    path = Path(actual.bound_reservations[0]["path"])
    assert (path / "reserved.json").is_file() and not (path / "released.json").exists()
    with pytest.raises(RuntimeError, match="active Setup reservation"):
        actual.require_write(actual.test_context.metadata_root / "never.json")


def test_root_or_request_drift_between_plan_and_bind_is_refused(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, bind=False)
    request = Path(actual.configuration["raw_build_request"]["path"])
    old = request.read_bytes()
    request.write_bytes(b" " + old[1:])
    with actual.locked():
        with pytest.raises(ValueError, match="changed"):
            actual.check()
    assert not actual.bound_reservations


def test_unlisted_path_or_changed_active_reservation_cannot_authorize_a_write(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    with pytest.raises(ValueError, match="outside the explicit"):
        actual.require_write(tmp_path / "foreign.json")
    with pytest.raises(ValueError, match="outside the explicit"):
        actual.require_write(actual.test_context.raw_root / "bypass.json")
    path = Path(actual.bound_reservations[0]["path"]) / "reserved.json"
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="reservation.*changed"):
        actual.require_write(actual.test_context.metadata_root / "never.json")


def test_active_domain_volume_change_is_refused_before_write(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20)
    actual.test_context.states["shared-volume"]["volume_id"] = "changed-volume"
    with pytest.raises((ValueError, RuntimeError), match="volume"):
        actual.require_write(actual.test_context.metadata_root / "never.json")


def test_replaced_output_directory_is_not_the_captured_domain(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20)
    root = actual.test_context.metadata_root
    root.rename(root.with_name("retained-original-directory"))
    root.mkdir()
    with pytest.raises(ValueError, match="directory identity"):
        actual.require_write(root / "never.json")


def test_metadata_publication_checks_affected_roots_without_repeating_other_runs(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline.series_setup_sink import SetupMetadataSink
    from darsia.fluidflower.pipeline.job_queue import write_immutable
    roots = [tmp_path / name for name in ("job", "other-run", "config")]
    for root in roots:
        root.mkdir()
    nested = roots[-1] / "runs"
    nested.mkdir()
    roots.append(nested)
    actual = _admission(tmp_path / "metadata", monkeypatch, metadata_cap=4096, metadata_roots=roots)
    checked = []
    inspect = actual._inspect_results

    def observed(path):
        checked.append(path)
        return inspect(path)

    actual._inspect_results = observed
    sink = SetupMetadataSink(actual)
    with sink.activate():
        write_immutable(roots[0] / "progress.json", dict(state="original-progress"))
    assert checked and set(checked) == {roots[0]}
    assert sink.evidence()["actual_created_bytes"] > 0
    assert not sink.evidence()["task_retirement_claimed"]
    checked.clear()
    actual.require_write(nested / "new.toml")
    assert set(checked) == {roots[-2], nested}, "Every containing captured domain is still verified"
    # A changed unused root cannot slow or block another run. Its old inode
    # remains authoritative when a subsequent write actually targets it.
    roots[1].rename(roots[1].with_name("retained-other-run"))
    roots[1].mkdir()
    actual.require_write(roots[0] / "next.json")
    with pytest.raises(ValueError, match="directory identity"):
        actual.require_write(roots[1] / "never.json")


def test_new_admission_still_checks_every_output_root(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("job", "other-run")]
    for root in roots:
        root.mkdir()
    actual = _admission(tmp_path / "metadata", monkeypatch, metadata_cap=4096,
                        metadata_roots=roots, bind=False)
    roots[1].rename(roots[1].with_name("retained-other-run"))
    roots[1].mkdir()
    with actual.locked(), pytest.raises(ValueError, match="directory identity"):
        actual.check()
    assert not actual.bound_reservations


def test_build_admission_checks_only_declared_roots_for_its_actual_metadata(tmp_path, monkeypatch):
    """An old all-run plan need not serialize 40 unrelated root inspections."""
    roots = [tmp_path / "job", tmp_path / "own-attempts", *[
        tmp_path / ("other-" + str(index)) for index in range(40)]]
    for root in roots:
        root.mkdir()
    old = _admission(tmp_path / "source", monkeypatch, metadata_cap=4096, raw_cap=100,
        metadata_roots=roots, bind=False)
    data = old.test_context
    inspections, checks_under_lock = [], []
    def inspect(path):
        inspections.append(Path(path))
        return old._inspect_results(path)
    factory = lambda _stores, _series, path: capacity.CapacityLedger(
        data.stores, "sample-series", path, volume_provider=data.volume,
        context_provider=lambda: dict(data.context))
    actual = admission.prepare_setup_admission(data.stores, "sample-series", data.task, data.config,
        metadata_roots=roots, metadata_targets=(roots[0], roots[1]),
        ledger_factory=factory, inspect_results=inspect)
    assert actual.configuration == old.configuration  # No scientific/config-byte substitution.
    assert {domain.path for domain in actual._domains if domain.role == "metadata"} == set(roots[:2])
    with actual.locked():
        inspections.clear()
        actual.check()
        checks_under_lock.extend(inspections)
        actual.bind(data.job_root, data.claim, data.owner)
    assert set(checks_under_lock) == {roots[0], roots[1], data.raw_root}
    assert not set(roots[2:]) & set(inspections)
    assert actual.require_write(roots[1] / "result.json") == roots[1] / "result.json"
    with pytest.raises(ValueError, match="outside the explicit Setup output roots"):
        actual.require_write(roots[2] / "not-this-run.json")
    assert sum(row.maximum_output_bytes for row in actual.volume_reservations) == 4196


def test_build_metadata_narrowing_cannot_add_undeclared_writer_authority(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=4096, raw_cap=100, bind=False)
    data = actual.test_context
    with pytest.raises(ValueError, match="outside its declared output roots"):
        admission.prepare_setup_admission(data.stores, "sample-series", data.task, data.config,
            metadata_roots=data.config["output_scope"]["metadata_roots"],
            metadata_targets=(tmp_path / "foreign-output",))


def test_new_work_at_raw_destination_cannot_be_relaunched_after_planning(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100, bind=False)
    Path(actual.raw_authority["destination"]).mkdir()
    with actual.locked():
        with pytest.raises(ValueError, match="cannot be relaunched"):
            actual.check()
    assert not actual.bound_reservations


def test_retained_literal_caps_do_not_reopen_old_request_or_plan_sources(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    reserved = Path(actual.bound_reservations[0]["path"]) / "reserved.json"
    record = json.loads(reserved.read_bytes())
    request = Path(actual.configuration["raw_build_request"]["path"])
    old = request.read_bytes()
    original = Path.read_bytes

    def no_old_source_read(path):
        if path == request:
            raise AssertionError("Historical accounting must not reopen the original request")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", no_old_source_read)
    row = admission.validate_retained_reservation(record,
        capacity_key=actual.volume_reservations[0].capacity_key,
        volume_id=actual.volume_reservations[0].volume_id)
    assert row["maximum_output_bytes"] == 120
    assert original(request) == old


@pytest.mark.parametrize("field,value", [("maximum_output_bytes", 119), ("case_count", 2),
    ("frame_progress_release_allowed", True), ("source_commit", "wrong"),
    ("volume_id", "foreign"), ("capacity_key", "foreign")])
def test_retained_records_cannot_relax_whole_cap_or_scope(tmp_path, monkeypatch, field, value):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    path = Path(actual.bound_reservations[0]["path"]) / "reserved.json"
    record = json.loads(path.read_bytes())
    record["estimate"][field] = value
    original = actual.volume_reservations[0]
    with pytest.raises(ValueError):
        admission.validate_retained_reservation(record, capacity_key=original.capacity_key,
                                               volume_id=original.volume_id)


def test_plan_and_declared_reservation_are_defensive_copies(tmp_path, monkeypatch):
    actual = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100)
    changed = actual.configuration
    changed["metadata_output"]["maximum_output_bytes"] = 10**8
    changed = actual.raw_authority
    changed["maximum_output_bytes"] = 10**8
    assert actual.metadata_limits["maximum_output_bytes"] == 20
    assert actual.raw_authority["maximum_output_bytes"] == 100
    assert asdict(actual.volume_reservations[0])["maximum_output_bytes"] == 120
    changed = actual.volume_reservations[0]
    changed.setup_plan["sha256"] = "f" * 64
    changed.roots[0]["path"] = str(tmp_path / "foreign")
    assert actual.volume_reservations[0].setup_plan["sha256"] == "d" * 64
    assert actual.volume_reservations[0].roots[0]["path"] != str(tmp_path / "foreign")


def test_original_reader_still_counts_the_positive_setup_record(tmp_path, monkeypatch):
    """An existing worker sees a conservative whole cap, not an unknown format."""
    import ast
    actual = _admission(tmp_path, monkeypatch, metadata_cap=100)
    data = actual.test_context
    ledger = capacity.CapacityLedger(data.stores, "sample-series", data.metadata_root,
        volume_provider=data.volume, context_provider=lambda: dict(data.context))
    # Exact original method from the pinned pre-Setup source, with only CRLF
    # normalized for Git checkouts. No current reader or Git history is borrowed.
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "capacity_remaining_098465c1.py.txt"
    source = fixture.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == "9d3b3098efa41957eb0c2295666ddae6e4ee67e10398c2be3433ce5e5e2b9b33"
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "CapacityLedger")
    original = next(node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == "remaining")
    namespace = dict(vars(capacity))
    exec(compile(ast.Module(body=[original], type_ignores=[]), "<immutable-original-capacity>", "exec"), namespace)
    with ledger.locked():
        assert namespace["remaining"](ledger) == 100
        assert ledger.remaining() == 100


def test_raw_output_accepts_positive_shared_proof_and_rechecks_it(tmp_path, monkeypatch):
    """NAS native output keeps the same owned target, full caps and drift guards."""
    initial = _admission(tmp_path, monkeypatch, metadata_cap=20, raw_cap=100,
        shared=False, bind=False)
    data = initial.test_context
    local = filesystem.inspect_local_file
    connected = [True]

    def location(path, **kwargs):
        if Path(path).resolve().is_relative_to(data.raw_root):
            return filesystem.LocalFileAdmission(False, "remote", "NAS", "test", str(data.raw_root))
        return local(path, **kwargs)

    monkeypatch.setattr(filesystem, "inspect_local_file", location)
    monkeypatch.setattr(filesystem, "inspect_shared_folder", lambda *a, **k:
        filesystem.SharedFolderAdmission(connected[0], "smb", "test NAS", "synthetic OS proof"))
    factory = lambda selected, series_id, root: capacity.CapacityLedger(selected, series_id, root,
        volume_provider=data.volume, context_provider=lambda: dict(data.context))
    actual = admission.prepare_setup_admission(data.stores, "sample-series", data.task, data.config,
        metadata_roots=(data.metadata_root,), ledger_factory=factory)
    assert actual.raw_authority == initial.raw_authority
    assert {row.volume_id: row.maximum_output_bytes for row in actual.volume_reservations} == {
        "raw-volume": 100, "publication-volume": 20}
    with actual.locked():
        actual.check()
        actual.bind(data.job_root, data.claim, data.owner)
    assert actual.require_write(data.metadata_root / "next.json")
    connected[0] = False
    with pytest.raises(ValueError, match="output root lost positive filesystem admission"):
        actual.require_write(data.metadata_root / "unavailable.json")
    assert all(not (Path(row["path"]) / "released.json").exists()
        for row in actual.bound_reservations)


def test_scratch_output_stays_local_even_when_shared_storage_is_admitted(tmp_path, monkeypatch):
    """Shared native products do not extend authority over temporary scratch."""
    initial = _admission(tmp_path, monkeypatch, metadata_cap=20, scratch=True, bind=False)
    data = initial.test_context
    local = filesystem.inspect_local_file

    def location(path, **kwargs):
        if Path(path).resolve().is_relative_to(data.scratch_root):
            return filesystem.LocalFileAdmission(False, "remote", "NAS", "test", str(data.scratch_root))
        return local(path, **kwargs)

    monkeypatch.setattr(filesystem, "inspect_local_file", location)
    monkeypatch.setattr(filesystem, "inspect_shared_folder", lambda *a, **k:
        filesystem.SharedFolderAdmission(True, "smb", "test NAS", "synthetic OS proof"))
    with pytest.raises(ValueError, match="scratch roots must be positively local"):
        admission.prepare_setup_admission(data.stores, "sample-series", data.task, data.config,
            metadata_roots=(data.metadata_root,), scratch_roots=(data.scratch_root,))
    assert not initial.bound_reservations
