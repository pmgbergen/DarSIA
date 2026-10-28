"""TESTONLY service protocol tests using immutable in-memory control records.

No accepted policy/profile, native process, filesystem ownership, quota, writer,
cleanup or production retirement is qualified. Synthetic positive verifiers
below are local TESTONLY callbacks, never imported production pass providers.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Lock, get_ident
from types import SimpleNamespace

import pytest

import darsia_cache.bootstrap as bootstrap
from darsia_cache.control import _encoded
from darsia_cache.policy import CachePolicyError
from darsia_cache.service_lifecycle import ServiceLifecycle


PRIMARY = dict(pid=101, process_create_time="TESTONLY_PRIMARY_EPOCH", argv=["TESTONLY_SERVICE"])
OBSERVER = dict(pid=202, process_create_time="TESTONLY_OBSERVER_EPOCH", argv=["TESTONLY_OBSERVER"])
CHILD = dict(pid=303, process_create_time="TESTONLY_CHILD_EPOCH", argv=["TESTONLY_CHILD"])
TOKEN = "a" * 32
GENERATION = "b" * 32
GATE = "TESTONLY_EXACT_GATE"
FAILED_UTC = "2026-01-01T00:00:00Z"
AFTER_48H = "2026-01-03T00:00:00Z"


def descriptor_TESTONLY(label):
    return dict(path="TESTONLY_NOT_CREATED/" + label, sha256="0" * 64, size_bytes=1)


class ImmutableControl_TESTONLY:
    """Only Python bytes in a dict; methods assert the actual mutex boundary."""

    def __init__(self, ledger, initial=None):
        self.ledger = ledger
        self.records = dict(initial or {})

    def names(self):
        self.ledger.require_lock()
        return list(self.records)

    def read_once(self, name, maximum=1024**2):
        self.ledger.require_lock()
        raw = self.records[name]
        assert len(raw) <= maximum
        return raw, {"TESTONLY_IN_MEMORY_NOT_A_FILE_ID": True}

    def write_once(self, name, raw, maximum=1024**2):
        self.ledger.require_lock()
        assert isinstance(raw, bytes) and len(raw) <= maximum
        if name in self.records:
            raise FileExistsError("TESTONLY immutable record already exists")
        self.records[name] = raw


class OriginalBindingLedger_TESTONLY:
    def __init__(self):
        self.mutex = Lock()
        self.holder = None
        self.lock_entries = 0
        self.forbidden_actions = []
        self.bound = {"path": "TESTONLY_CONTROL_NOT_CREATED"}
        self.control = ImmutableControl_TESTONLY(self)
        self.reservation = dict(token=TOKEN, pool="TESTONLY_POOL", sha256="0" * 64)

    @contextmanager
    def locked(self):
        assert self.holder != get_ident(), "TESTONLY nested transaction refused"
        with self.mutex:
            assert self.holder is None
            self.holder = get_ident()
            self.lock_entries += 1
            try:
                yield self
            finally:
                self.holder = None

    def require_lock(self):
        assert self.holder == get_ident(), "Control access escaped the service mutex"

    def require_reservation(self, reservation, owner, needed):
        self.require_lock()
        assert reservation == self.reservation and owner == PRIMARY and needed == 1
        return {"TESTONLY_EXISTING_RESERVATION": True}, b"TESTONLY_NO_FILE"

    def forbidden(self, action):
        self.forbidden_actions.append(action)
        raise AssertionError("Lifecycle must never perform " + action)

    def reserve(self, *args, **kwargs):
        self.forbidden("new reservation")

    def close(self, *args, **kwargs):
        self.forbidden("budget close")

    def release(self, *args, **kwargs):
        self.forbidden("result release")

    def delete(self, *args, **kwargs):
        self.forbidden("deletion")


@pytest.fixture
def worlds(monkeypatch):
    # The sole production monkeypatch is the native current-owner observation.
    monkeypatch.setattr(bootstrap, "_owner", lambda: (deepcopy(OBSERVER), {"TESTONLY": True}))
    created = []

    def make(entry_kind="worker"):
        ledger = OriginalBindingLedger_TESTONLY()
        root = Path("TESTONLY_SERVICE_SCOPE_NOT_CREATED")
        manifest = dict(format="darsia-cache-external-import-scope-v1", token=TOKEN,
            owner=deepcopy(PRIMARY), root=str(root), reservation=deepcopy(ledger.reservation),
            entry_kind=entry_kind, role_identifier="TESTONLY_ROLE",
            purpose="tests" if entry_kind == "tests" else "jobs", maximum_bytes=1,
            external_writer_file_ownership_not_yet_frozen=True)
        lease = SimpleNamespace(owner=deepcopy(PRIMARY), root=root,
            reservation=deepcopy(ledger.reservation), manifest=manifest,
            source=descriptor_TESTONLY("stores"), profile_binding=descriptor_TESTONLY("profile"),
            document={"cache_bootstrap": {"namespace_admission": descriptor_TESTONLY("namespace")}},
            entry_kind=entry_kind, role_identifier="TESTONLY_ROLE", ledger=ledger,
            maximum=1, entry_limit=128,
            policy=SimpleNamespace(failed_retention_hours=48))
        service = ServiceLifecycle(lease)
        service.register_owner()
        world = SimpleNamespace(lease=lease, ledger=ledger, service=service, checks=[])
        created.append(world)
        return world

    yield make
    for world in created:
        assert world.ledger.forbidden_actions == []
        assert world.ledger.holder is None
        assert world.lease.manifest["owner"] == PRIMARY
        assert world.lease.reservation == world.ledger.reservation


def event_TESTONLY(kind):
    if kind.startswith("legacy_"):
        details = dict(worker_id="TESTONLY_LEGACY_WORKER")
    else:
        details = dict(slot=0, generation=GENERATION)
    if kind.endswith("observed"):
        details.update(original_native_identity=deepcopy(CHILD), epoch_unavailable=False)
    elif kind == "worker_children_retired":
        details.update(original_child_identities=[deepcopy(CHILD)], owned_launch_handle_retired=True,
                       service_parent_retirement_proven=False)
    return details


def launch_TESTONLY(world, kind):
    return world.service.record_launch(kind, event_TESTONLY(kind))


def seal_TESTONLY(world, outcome="success"):
    return world.service.seal(logical_gate=GATE, outcome=outcome, utc=FAILED_UTC,
        traceback_text="TESTONLY_LITERAL_FAILURE" if outcome == "failed" else None,
        publication=descriptor_TESTONLY("publication") if outcome == "success" else None)


def proofs_TESTONLY(world, **changes):
    def retirement(snapshot, bound):
        world.checks.append("TESTONLY_RETIREMENT")
        assert bound == descriptor_TESTONLY("retirement")
        assert snapshot["binding"]["manifest"]["owner"] == PRIMARY
        assert snapshot["child_coverage"]["complete_external_child_coverage_provider_required"] is True
        return True  # TESTONLY synthetic protocol; no actual process was inspected.

    def publication(snapshot, bound):
        world.checks.append("TESTONLY_PUBLICATION")
        assert bound == descriptor_TESTONLY("publication")
        return True  # TESTONLY synthetic protocol; no payload was inspected.

    result = dict(now_utc=AFTER_48H, retirement=descriptor_TESTONLY("retirement"),
                  verify_retirement=retirement, verify_publication=publication)
    result.update(changes)
    return result


def records_TESTONLY(world):
    with world.ledger.locked():
        return world.service._records()


@pytest.mark.parametrize("operation", ["owner", "worker", "legacy"])
def test_atomic_seal_refuses_new_owner_and_every_producer_intent(worlds, operation):
    world = worlds()
    seal_TESTONLY(world)
    before = dict(world.ledger.control.records)
    with pytest.raises(CachePolicyError, match="sealed"):
        if operation == "owner":
            child_lease = SimpleNamespace(**vars(world.lease))
            child_lease.owner = deepcopy(CHILD)
            ServiceLifecycle(child_lease).register_owner()
        else:
            launch_TESTONLY(world, operation + "_spawn_intent")
    assert world.ledger.control.records == before
    assert world.ledger.lock_entries >= 3


@pytest.mark.parametrize("family", ["worker", "legacy"])
def test_prior_intent_may_finish_observation_after_seal(worlds, family):
    world = worlds()
    launch_TESTONLY(world, family + "_spawn_intent")
    seal_TESTONLY(world)
    launch_TESTONLY(world, family + "_spawn_observed")
    if family == "worker":
        launch_TESTONLY(world, "worker_children_retired")
    kinds = {kind for kind, _ in records_TESTONLY(world)}
    assert ("observed" if family == "worker" else "legacy_observed") in kinds
    assert ("retired" in kinds) is (family == "worker")


@pytest.mark.parametrize("kind", ["worker_spawn_intent", "worker_spawn_observed",
    "worker_children_retired", "legacy_spawn_intent", "legacy_spawn_observed"])
def test_all_five_allowed_observations_enter_the_original_registry(worlds, kind):
    world = worlds()
    if kind in {"worker_spawn_observed", "worker_children_retired"}:
        launch_TESTONLY(world, "worker_spawn_intent")
    if kind == "worker_children_retired":
        launch_TESTONLY(world, "worker_spawn_observed")
    if kind == "legacy_spawn_observed":
        launch_TESTONLY(world, "legacy_spawn_intent")
    receipt = launch_TESTONLY(world, kind)
    mapping = dict(worker_spawn_intent="intent", worker_spawn_observed="observed",
        worker_children_retired="retired", legacy_spawn_intent="legacy_intent",
        legacy_spawn_observed="legacy_observed")
    rows = records_TESTONLY(world)
    assert any(stage == mapping[kind] and row["binding"] == receipt for (stage, _), row in rows.items())


@pytest.mark.parametrize("case", ["orphan_observation", "missing_observation", "unknown_epoch", "unresolved_intent"])
def test_missing_or_unknown_launch_proof_refuses(worlds, case):
    world = worlds()
    if case == "orphan_observation":
        with pytest.raises(CachePolicyError, match="No original owned"):
            launch_TESTONLY(world, "worker_spawn_observed")
        return
    launch_TESTONLY(world, "worker_spawn_intent")
    if case == "unknown_epoch":
        details = event_TESTONLY("worker_spawn_observed")
        details.update(original_native_identity=None, epoch_unavailable=True)
        world.service.record_launch("worker_spawn_observed", details)
    if case in {"missing_observation", "unknown_epoch"}:
        with pytest.raises(CachePolicyError, match="Unknown launch incarnation"):
            launch_TESTONLY(world, "worker_children_retired")
    else:
        seal_TESTONLY(world)
        with pytest.raises(CachePolicyError, match="Unresolved launch intent"):
            world.service.retention_eligibility(**proofs_TESTONLY(world))


def test_existing_orphan_record_is_preserved_and_refuses_eligibility(worlds):
    world = worlds()
    launch_TESTONLY(world, "worker_spawn_intent")
    launch_TESTONLY(world, "worker_spawn_observed")
    seal_TESTONLY(world)
    # Arrange a damaged prior namespace in a NEW in-memory control, never mutate
    # or overwrite an existing immutable record through the production API.
    initial = {name: raw for name, raw in world.ledger.control.records.items()
               if "-intent-" not in name}
    world.ledger.control = ImmutableControl_TESTONLY(world.ledger, initial)
    before = dict(initial)
    with pytest.raises(CachePolicyError, match="Orphan launch observation"):
        world.service.retention_eligibility(**proofs_TESTONLY(world))
    assert world.ledger.control.records == before


@pytest.mark.parametrize("observed", [False, True])
def test_legacy_scope_refuses_and_is_visible_in_rejected_snapshot(worlds, observed):
    world = worlds()
    launch_TESTONLY(world, "legacy_spawn_intent")
    if observed:
        launch_TESTONLY(world, "legacy_spawn_observed")
    seal_TESTONLY(world)
    with pytest.raises(CachePolicyError, match="legacy child lifecycle is UNQUALIFIED") as error:
        world.service.retention_eligibility(**proofs_TESTONLY(world))
    legacy = error.value.registry_snapshot["legacy_launches"]
    assert {row["document"]["kind"] for row in legacy} == (
        {"legacy_intent", "legacy_observed"} if observed else {"legacy_intent"})
    assert all(row["document"]["value"]["worker_id"] == "TESTONLY_LEGACY_WORKER" for row in legacy)
    assert world.checks == []


@pytest.mark.parametrize("entry_kind", ["runtime", "tests"])
def test_uninstrumented_runtime_and_tests_refuse_empty_registry(worlds, entry_kind):
    world = worlds(entry_kind)
    seal_TESTONLY(world)
    with pytest.raises(CachePolicyError, match="empty registry is not no-child proof") as error:
        world.service.retention_eligibility(**proofs_TESTONLY(world))
    assert error.value.registry_snapshot["child_coverage"]["entry_kind"] == entry_kind
    assert error.value.registry_snapshot["launches"] == []
    assert world.checks == []


def test_actual_external_observer_is_new_record_actor_not_dead_primary(worlds):
    world = worlds()
    seal_TESTONLY(world)
    result = world.service.retention_eligibility(**proofs_TESTONLY(world))
    rows = records_TESTONLY(world)
    row = next(value["document"] for (kind, _), value in rows.items() if kind == "eligibility")
    assert all(_encoded(value["document"]) == world.ledger.control.records[Path(value["binding"]["path"]).name]
               for value in rows.values())
    assert row["actor"] == row["value"]["observer"] == OBSERVER
    assert row["actor"] != PRIMARY
    assert world.lease.owner == world.lease.manifest["owner"] == PRIMARY
    assert result["cleanup_authorized"] is False and result["service_budget_closed"] is False
    assert result["external_writer_owned_inventory_and_final_cleanup_still_required"] is True
    assert world.checks == ["TESTONLY_RETIREMENT", "TESTONLY_PUBLICATION"]


def test_registered_live_owner_cannot_self_certify(worlds, monkeypatch):
    world = worlds()
    seal_TESTONLY(world)
    monkeypatch.setattr(bootstrap, "_owner", lambda: (deepcopy(PRIMARY), {"TESTONLY": True}))
    with pytest.raises(CachePolicyError, match="live owner cannot certify"):
        world.service.retention_eligibility(**proofs_TESTONLY(world))
    assert world.checks == []


@pytest.mark.parametrize("verifier", [None, "TESTONLY_NOT_CALLABLE", "TESTONLY_REFUSED"])
def test_missing_or_refused_external_retirement_verifier_refuses(worlds, verifier):
    world = worlds()
    seal_TESTONLY(world)
    function = (lambda *_: False) if verifier == "TESTONLY_REFUSED" else verifier
    with pytest.raises(CachePolicyError, match="retirement (verifier is required|not proven)"):
        world.service.retention_eligibility(**proofs_TESTONLY(world, verify_retirement=function))


@pytest.mark.parametrize("verifier", [None, False])
def test_missing_or_refused_original_publication_refuses(worlds, verifier):
    world = worlds()
    seal_TESTONLY(world)
    function = None if verifier is None else lambda *_: False
    with pytest.raises(CachePolicyError, match="publication remains unverified"):
        world.service.retention_eligibility(**proofs_TESTONLY(world, verify_publication=function))
    assert not any(kind == "eligibility" for kind, _ in records_TESTONLY(world))


@pytest.mark.parametrize("case", ["missing_verifier", "wrong_gate", "before_failure", "future_success"])
def test_unverified_or_wrong_next_success_refuses(worlds, case):
    world = worlds()
    seal_TESTONLY(world, "failed")
    success = dict(complete=True, logical_gate=GATE, utc="2026-01-01T01:00:00Z")
    if case == "wrong_gate":
        success["logical_gate"] = "TESTONLY_OTHER_GATE"
    elif case == "before_failure":
        success["utc"] = "2025-12-31T23:59:59Z"
    elif case == "future_success":
        success["utc"] = "2026-01-04T00:00:00Z"
    function = None if case == "missing_verifier" else lambda *_: deepcopy(success)
    with pytest.raises(CachePolicyError):
        world.service.retention_eligibility(**proofs_TESTONLY(world,
            same_gate_success=descriptor_TESTONLY("later_success"), verify_same_gate_success=function))


@pytest.mark.parametrize("now_utc,due", [("2026-01-02T23:59:59Z", False),
    (AFTER_48H, True), ("2026-01-03T00:00:01Z", True)])
def test_existing_configured_48_hour_rule_is_not_relaxed(worlds, now_utc, due):
    world = worlds()
    seal_TESTONLY(world, "failed")
    result = world.service.retention_eligibility(**proofs_TESTONLY(world, now_utc=now_utc))
    assert result["retention_due"] is due
    assert result["cleanup_authorized"] is False and result["service_budget_closed"] is False
    row = next(value["document"] for (kind, _), value in records_TESTONLY(world).items() if kind == "seal")
    assert row["value"]["traceback"] == "TESTONLY_LITERAL_FAILURE"


def test_verified_exact_next_success_can_end_failed_retention_before_48_hours(worlds):
    world = worlds()
    seal_TESTONLY(world, "failed")
    success = dict(complete=True, logical_gate=GATE, utc="2026-01-01T01:00:00Z")
    def verify_TESTONLY(snapshot, bound):
        assert bound == descriptor_TESTONLY("later_success")
        return deepcopy(success)
    result = world.service.retention_eligibility(**proofs_TESTONLY(world,
        now_utc="2026-01-01T02:00:00Z", same_gate_success=descriptor_TESTONLY("later_success"),
        verify_same_gate_success=verify_TESTONLY))
    assert result["retention_due"] is True and result["cleanup_authorized"] is False


@pytest.mark.parametrize("kind", ["owner", "intent", "seal"])
def test_immutable_conflicting_record_is_refused_without_overwrite(worlds, kind):
    world = worlds()
    if kind == "intent":
        launch_TESTONLY(world, "worker_spawn_intent")
    elif kind == "seal":
        seal_TESTONLY(world)
    before = dict(world.ledger.control.records)
    with pytest.raises(CachePolicyError, match="different lifecycle record"):
        if kind == "owner":
            original = world.lease.owner
            world.lease.owner = dict(PRIMARY, argv=["TESTONLY_CONFLICT"])
            try:
                world.service.register_owner()
            finally:
                world.lease.owner = original
        elif kind == "intent":
            details = event_TESTONLY("worker_spawn_intent")
            details["conflict"] = "TESTONLY_DIFFERENT"
            world.service.record_launch("worker_spawn_intent", details)
        else:
            world.service.seal(logical_gate=GATE, outcome="success", utc=AFTER_48H,
                               publication=descriptor_TESTONLY("publication"))
    assert world.ledger.control.records == before


def test_failed_seal_requires_literal_diagnostic_evidence(worlds):
    world = worlds()
    with pytest.raises(CachePolicyError, match="Literal failed-service evidence"):
        world.service.seal(logical_gate=GATE, outcome="failed", utc=FAILED_UTC)
    assert not any(kind == "seal" for kind, _ in records_TESTONLY(world))
