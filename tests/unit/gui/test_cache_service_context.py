"""Synthetic read-only original-service context contracts.

Only parsed metadata, immutable bytes and fake namespace/mutex handles are used.
The canonical modules have inert imports; every native/host/path boundary is
replaced before resolution. No accepted production policy, namespace, reader
retirement, cache writer, deletion or service cleanup is qualified here.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

import darsia_cache.bootstrap as bootstrap
import darsia_cache.capacity as capacity
import darsia_cache.control as control
import darsia_cache.service_context as resolver
from darsia_cache.control import _encoded
from darsia_cache.policy import CachePolicyError, cache_settings
from darsia_cache.service_lifecycle import ServiceLifecycle


TOKEN = "a" * 32
PRIMARY = dict(machine="testonly-host", pid=101, process_create_time="TESTONLY_PRIMARY_EPOCH")
OBSERVER = dict(machine="testonly-host", pid=202, process_create_time="TESTONLY_OBSERVER_EPOCH")
PACKAGE = "C:/TESTONLY_NOT_CREATED/source/darsia/__init__.py"


def path_key(path):
    return str(PureWindowsPath(path)).casefold()


def descriptor(label):
    # A synthetic parsed-metadata boundary, never a claimed real file digest.
    return dict(path="C:/TESTONLY_NOT_CREATED/metadata/" + label,
                sha256=hashlib.sha256(label.encode()).hexdigest(), size_bytes=1)


class World:
    """All constructors below allocate Python objects only."""

    def __init__(self):
        self.events, self.namespaces, self.ledgers = [], [], []
        self.lock_depth = 0
        self.writes = []
        self.close_failure_path = None
        self.read_failure_name = None
        self.original_owner = deepcopy(PRIMARY)
        self.observer = deepcopy(OBSERVER)
        self.actual = dict(host="testonly-host", service_account_sid="TESTONLY_SID",
                           common_root="C:/TESTONLY_NOT_CREATED/authority")
        self.profile_pin = descriptor("profile.json")
        self.completion_pin = descriptor("completion.json")
        self.namespace_pin = descriptor("namespace.json")
        self.policy_pin = descriptor("stores.toml")
        self.token = TOKEN
        self.receipts_path = "C:/TESTONLY_NOT_CREATED/receipts"
        self.receipt_pin = descriptor("unused")
        self.receipt_pin["path"] = str(Path(self.receipts_path) / (TOKEN + ".import-service.json"))
        self.document = dict(
            cache=dict(root="C:/TESTONLY_NOT_CREATED/cache", quota_bytes=4096,
                       failed_retention_hours=48,
                       profiles={"jobs": deepcopy(self.profile_pin),
                                 "tests": deepcopy(self.profile_pin),
                                 "display": deepcopy(self.profile_pin)}),
            cache_bootstrap=dict(
                entry_roles=dict(runtime="interactive", worker="worker", tests="tests"),
                roles=dict(
                    interactive=dict(purpose="display", profile=deepcopy(self.profile_pin)),
                    worker=dict(purpose="jobs", profile=deepcopy(self.profile_pin)),
                    tests=dict(purpose="tests", profile=deepcopy(self.profile_pin))),
                namespace_admission=deepcopy(self.namespace_pin), accounting_entry_limit=32))
        self.policy = cache_settings(self.document, required=True)
        self.root = self.policy.purpose("jobs") / TOKEN
        self.scope_identity = {"TESTONLY_ID": "original-scope"}
        self.receipt_identity = {"TESTONLY_ID": "original-receipts"}
        self.authority_identity = {"TESTONLY_ID": "original-authority"}
        self.cache_identity = {"TESTONLY_ID": "original-cache"}
        self.admission = dict(
            host=self.actual["host"], authorized_sids=["TESTONLY_SID"],
            cache_identity=deepcopy(self.cache_identity),
            host_ledger=dict(path=self.actual["common_root"], identity=deepcopy(self.authority_identity)),
            receipts=dict(path=self.receipts_path, identity=deepcopy(self.receipt_identity)),
            purposes={"jobs": dict(path=str(self.policy.purpose("jobs")),
                                   identity={"TESTONLY_ID": "jobs-parent"})})
        self.profile = dict(
            format="darsia-cache-import-profile-v1", purpose="jobs",
            role_identifier="worker", entry_kind="worker",
            writer_contract="cooperative-start-admission-v1", maximum_cache_bytes=128,
            completion=deepcopy(self.completion_pin))
        self.completion = dict(complete=True, passed=True)
        self.pool = hashlib.sha256((self.actual["host"] + "\0cache-host-pool-v2").encode()).hexdigest()
        self.reservation_row = dict(
            format="darsia-cache-budget-reservation-v2", pool=self.pool, utc="2026-01-01T00:00:00Z",
            owner=deepcopy(PRIMARY), token=TOKEN, maximum_bytes=128)
        self.reservation_raw = _encoded(self.reservation_row)
        self.reservation = dict(token=TOKEN, pool=self.pool,
                                sha256=hashlib.sha256(self.reservation_raw).hexdigest())
        self.manifest = dict(
            format="darsia-cache-external-import-scope-v1", token=TOKEN,
            owner=deepcopy(PRIMARY), purpose="jobs", root=str(self.root),
            root_identity=deepcopy(self.scope_identity), maximum_bytes=128,
            entry_kind="worker", role_identifier="worker", reservation=deepcopy(self.reservation))
        self.receipt = dict(
            format="darsia-cache-import-service-receipt-v1",
            policy=deepcopy(self.policy_pin), profile=deepcopy(self.profile_pin),
            namespace=deepcopy(self.namespace_pin), entry_kind="worker", role_identifier="worker",
            manifest=self.manifest, directories={"": deepcopy(self.scope_identity),
                                               "tmp": {"TESTONLY_ID": "original-tmp"}})
        self.policy_source = deepcopy(self.policy_pin)
        self.metadata = {
            path_key(self.receipt_pin["path"]): (self.receipt_pin, self.receipt),
            path_key(self.profile_pin["path"]): (self.profile_pin, self.profile),
            path_key(self.completion_pin["path"]): (self.completion_pin, self.completion),
            path_key(self.namespace_pin["path"]): (self.namespace_pin, self.admission)}
        self.host_record = dict(
            format="darsia-one-host-cache-binding-v2", host=self.actual["host"],
            authorized_sids=["TESTONLY_SID"], cache_root=self.policy.root,
            cache_identity=deepcopy(self.cache_identity), quota_bytes=4096)
        self.records = {
            "cache-" + self.pool + ".json": _encoded(self.host_record),
            "cache-r-" + TOKEN + ".json": self.reservation_raw,
            "cache-s-" + TOKEN + ".json": _encoded(self.manifest)}
        self.directory_ids = {
            path_key(self.receipts_path): deepcopy(self.receipt_identity),
            path_key(self.root): deepcopy(self.scope_identity),
            path_key(self.actual["common_root"]): deepcopy(self.authority_identity)}
        self.reservation_checks = []

    def outside(self, path, package_file):
        assert package_file == PACKAGE
        self.events.append(("outside", path_key(path)))
        return Path(path)

    def read_bound(self, binding):
        self.events.append(("metadata-read", path_key(binding["path"])))
        expected, value = self.metadata[path_key(binding["path"])]
        if binding != expected:
            raise CachePolicyError("TESTONLY immutable metadata descriptor differs")
        return deepcopy(value), b"TESTONLY_PARSED_BOUNDARY"

    def explicit_document(self, path, package_file):
        assert path_key(path) == path_key(self.policy_pin["path"]) and package_file == PACKAGE
        self.events.append(("policy-read", path_key(path)))
        return deepcopy(self.document), deepcopy(self.policy_source)

    def observe(self):
        self.events.append(("actual-observer",))
        return deepcopy(self.observer), deepcopy(self.actual)

    def sync_manifest_record(self):
        self.records["cache-s-" + TOKEN + ".json"] = _encoded(self.manifest)

    def replace_reservation_owner(self, owner):
        row = {**self.reservation_row, "owner": deepcopy(owner)}
        raw = _encoded(row)
        self.records["cache-r-" + TOKEN + ".json"] = raw
        self.manifest["reservation"] = dict(self.reservation, sha256=hashlib.sha256(raw).hexdigest())
        self.sync_manifest_record()

    def assert_no_mutation(self):
        assert self.writes == []
        assert not any(event[0] in {"producer", "ImportLease"} for event in self.events)

    def assert_closed(self):
        assert self.lock_depth == 0
        assert all(namespace.closed for namespace in self.namespaces)
        assert all(ledger.control is None and not ledger.lock_active for ledger in self.ledgers)


@pytest.fixture
def world(monkeypatch):
    w = World()

    class Namespace:
        def __init__(self, path, expected):
            self.path, self.closed, self.close_attempted = path_key(path), False, False
            w.events.append(("open-existing", self.path, deepcopy(expected)))
            if w.directory_ids.get(self.path) != expected:
                raise CachePolicyError("TESTONLY existing directory identity differs")
            w.namespaces.append(self)

        def names(self):
            assert not self.closed and w.lock_depth == 1
            w.events.append(("names", self.path))
            return list(w.records)

        def read_once(self, name, maximum=1024**2):
            assert not self.closed and w.lock_depth == 1
            w.events.append(("control-read", name, maximum))
            if name == w.read_failure_name:
                raise CachePolicyError("TESTONLY control read failure")
            raw = w.records[name]
            assert len(raw) <= maximum
            return raw, {"TESTONLY_NOT_A_REAL_FILE_ID": True}

        def write_once(self, name, raw, maximum=1024**2):
            assert not self.closed and w.lock_depth == 1
            assert isinstance(raw, bytes) and len(raw) <= maximum
            if name in w.records:
                raise FileExistsError("TESTONLY immutable record already exists")
            w.writes.append((name, raw, maximum))
            w.records[name] = raw

        def scope(self, *args, **kwargs):
            w.events.append(("producer", "scope/create"))
            raise AssertionError("Resolver must never create/open a producer scope")

        def close(self):
            self.close_attempted = True
            w.events.append(("close", self.path))
            if self.path == w.close_failure_path:
                raise CachePolicyError("TESTONLY reported close failure; actual closure unknown")
            self.closed = True

    @contextmanager
    def mutex(path, key):
        assert path_key(path) == path_key(w.actual["common_root"])
        assert key == w.pool and w.lock_depth == 0
        w.events.append(("mutex-enter",))
        w.lock_depth += 1
        try:
            yield
        finally:
            w.lock_depth -= 1
            w.events.append(("mutex-exit",))

    def forbidden_lease(*args, **kwargs):
        w.events.append(("ImportLease",))
        raise AssertionError("Resolver must not start/attach/register a producer")

    original_init = control.CacheBudgetLedger.__init__
    original_require = control.CacheBudgetLedger.require_reservation

    def ledger_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        w.ledgers.append(self)

    def require(self, reservation, owner, needed):
        w.reservation_checks.append((deepcopy(reservation), deepcopy(owner), needed))
        return original_require(self, reservation, owner, needed)

    monkeypatch.setattr(resolver, "_outside", w.outside)
    monkeypatch.setattr(resolver, "read_bound", w.read_bound)
    monkeypatch.setattr(resolver, "_explicit_document", w.explicit_document)
    monkeypatch.setattr(resolver, "_owner", w.observe)
    monkeypatch.setattr(resolver, "WindowsNamespace", Namespace)
    monkeypatch.setattr(control, "WindowsNamespace", Namespace)
    monkeypatch.setattr(control, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(control.CacheBudgetLedger, "__init__", ledger_init)
    monkeypatch.setattr(control.CacheBudgetLedger, "require_reservation", require)
    monkeypatch.setattr(capacity, "_capacity_host_context", lambda: deepcopy(w.actual))
    monkeypatch.setattr(capacity, "_LocalVolumeLock", mutex)
    monkeypatch.setattr(bootstrap, "ImportLease", forbidden_lease)
    return w


def resolve(world):
    return resolver.resolve_original_service_context(deepcopy(world.receipt_pin), package_file=PACKAGE)


def test_context_keeps_original_owner_and_uses_distinct_actual_observer_once(world):
    context = resolve(world)
    assert context.owner == OBSERVER and context.manifest["owner"] == PRIMARY
    assert context.reservation == world.reservation
    assert world.reservation_checks == [(world.reservation, PRIMARY, 128)]
    assert context.root == world.root and context.maximum == 128 and context.entry_limit == 32
    assert context.resolver_is_not_retirement_proof is True
    assert len(context.original_metadata) == 5
    reads = [event[1] for event in world.events if event[0] in {"metadata-read", "policy-read"}]
    assert len(reads) == len(set(reads)) == 5
    # Receipt/scope ID checks open existing handles but never enumerate payloads.
    assert [event[1] for event in world.events if event[0] == "names"] == [
        path_key(world.actual["common_root"]), path_key(world.actual["common_root"])]
    world.assert_no_mutation()
    world.assert_closed()
    context.owner["pid"] = 999
    context.manifest["owner"]["pid"] = 888
    assert world.observer == OBSERVER and world.manifest["owner"] == PRIMARY


@pytest.mark.parametrize("operation", ["read", "remember"])
def test_once_metadata_cache_refuses_conflicting_same_path_descriptor(world, operation):
    once = resolver._MetadataOnce(PACKAGE)
    first = once.read(world.profile_pin)
    assert once.read(deepcopy(world.profile_pin)) == first
    first["purpose"] = "changed returned copy"
    assert once.read(world.profile_pin)["purpose"] == "jobs"
    conflict = {**world.profile_pin, "sha256": "f" * 64}
    with pytest.raises(CachePolicyError):
        if operation == "read":
            once.read(conflict)
        else:
            once.remember(conflict, {})
    assert sum(event[0] == "metadata-read" for event in world.events) == 1
    world.assert_no_mutation()


@pytest.mark.parametrize("case", [
    "policy-source", "policy-quota", "profile-binding", "namespace-binding",
    "profile-purpose", "profile-completion", "original-host", "live-primary", "actual-account",
])
def test_wrong_original_source_policy_profile_namespace_or_actor_refuses(world, case):
    if case == "policy-source":
        world.policy_source["sha256"] = "f" * 64
    elif case == "policy-quota":
        world.document["cache"]["quota_bytes"] = 0
    elif case == "profile-binding":
        world.receipt["profile"] = descriptor("wrong-profile.json")
    elif case == "namespace-binding":
        world.receipt["namespace"] = descriptor("wrong-namespace.json")
    elif case == "profile-purpose":
        world.profile["purpose"] = "tests"
    elif case == "profile-completion":
        world.completion["passed"] = False
    elif case == "original-host":
        world.manifest["owner"]["machine"] = "other-host"
    elif case == "live-primary":
        world.observer = deepcopy(PRIMARY)
    elif case == "actual-account":
        world.actual["service_account_sid"] = "OTHER_TESTONLY_SID"
    with pytest.raises(CachePolicyError):
        resolve(world)
    world.assert_no_mutation()
    world.assert_closed()


@pytest.mark.parametrize("case", ["missing-host-binding", "wrong-host-binding", "reservation-owner", "manifest"])
def test_only_existing_exact_host_reservation_and_manifest_are_accepted(world, case):
    if case == "missing-host-binding":
        del world.records["cache-" + world.pool + ".json"]
    elif case == "wrong-host-binding":
        world.records["cache-" + world.pool + ".json"] = _encoded({**world.host_record, "quota_bytes": 1})
    elif case == "reservation-owner":
        world.replace_reservation_owner(OBSERVER)
    elif case == "manifest":
        world.records["cache-s-" + TOKEN + ".json"] = _encoded({**world.manifest, "maximum_bytes": 127})
    with pytest.raises(CachePolicyError):
        resolve(world)
    world.assert_no_mutation()
    world.assert_closed()
    assert any(event[0] == "mutex-exit" for event in world.events)


@pytest.mark.parametrize("case", ["scope-id", "receipt-path", "control-read"])
def test_resolution_failure_closes_all_fresh_handles_without_allocating(world, case):
    if case == "scope-id":
        world.directory_ids[path_key(world.root)] = {"TESTONLY_ID": "replaced-scope"}
    elif case == "receipt-path":
        world.receipt_pin["path"] = "C:/TESTONLY_NOT_CREATED/elsewhere/service.json"
        world.metadata[path_key(world.receipt_pin["path"])] = (world.receipt_pin, world.receipt)
    elif case == "control-read":
        world.read_failure_name = "cache-s-" + TOKEN + ".json"
    with pytest.raises(CachePolicyError):
        resolve(world)
    world.assert_no_mutation()
    world.assert_closed()


def test_reported_close_failure_propagates_without_claiming_handle_closure(world):
    world.close_failure_path = path_key(world.receipts_path)
    with pytest.raises(CachePolicyError, match="close failure"):
        resolve(world)
    assert len(world.namespaces) == 1 and world.namespaces[0].close_attempted
    assert not world.namespaces[0].closed  # Failure means closure unknown, never synthetic PASS.
    world.assert_no_mutation()
    assert world.lock_depth == 0


@pytest.mark.parametrize("operation", ["reserve", "admit", "bind_scope", "close", "measure"])
def test_observer_has_no_producer_allocation_or_budget_closure_capability(world, operation):
    context = resolve(world)
    with pytest.raises(CachePolicyError):
        getattr(context.ledger, operation)()
    assert not hasattr(context.ledger, "release")
    assert not hasattr(context, "start") and not hasattr(context, "attach") and not hasattr(context, "attach_owner")
    world.assert_no_mutation()
    world.assert_closed()


def test_observer_lock_refuses_initialization_before_open_and_restores_view_on_failure(world):
    context = resolve(world)
    before = len(world.namespaces)
    with pytest.raises(CachePolicyError):
        with context.ledger.locked(create_binding=True):
            pytest.fail("Read-only observer initialized a binding")
    assert len(world.namespaces) == before
    with pytest.raises(RuntimeError, match="TESTONLY caller failure"):
        with context.ledger.locked():
            assert isinstance(context.ledger.control, resolver._EligibilityMetadataOnly)
            raise RuntimeError("TESTONLY caller failure")
    world.assert_no_mutation()
    world.assert_closed()


def test_only_explicit_v11_eligibility_evidence_may_use_actual_observer_actor(world):
    context = resolve(world)
    lifecycle = ServiceLifecycle(context)
    original_records = deepcopy(world.records)
    value = dict(observer=deepcopy(OBSERVER), cleanup_authorized=False, service_budget_closed=False,
                 TESTONLY_NOT_RETIREMENT_OR_CLEANUP_PROOF=True)
    # Exercise the V11 writer API only; no retention provider or seal protocol is repeated.
    key = hashlib.sha256(_encoded(value)).hexdigest()
    with context.ledger.locked():
        record = lifecycle._put({}, "eligibility", key, value, actor=deepcopy(OBSERVER))
    assert len(world.writes) == 1
    name, raw, maximum = world.writes[0]
    assert name == "cache-l-" + TOKEN + "-eligibility-" + key + ".json"
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["size_bytes"] == len(raw) and maximum == 1024**2
    assert context.manifest["owner"] == PRIMARY
    assert all(world.records[name] == raw for name, raw in original_records.items())
    world.assert_closed()


@pytest.mark.parametrize("case", ["original-actor", "wrong-value-observer", "producer-record", "wrong-key"])
def test_observer_view_refuses_wrong_actor_and_non_eligibility_writes(world, case):
    context = resolve(world)
    lifecycle = ServiceLifecycle(context)
    value = dict(observer=deepcopy(OBSERVER), cleanup_authorized=False, service_budget_closed=False)
    actor, kind = deepcopy(OBSERVER), "eligibility"
    if case == "original-actor":
        actor = deepcopy(PRIMARY)
    elif case == "wrong-value-observer":
        value["observer"] = deepcopy(PRIMARY)
    elif case == "producer-record":
        kind = "owner"
    key = hashlib.sha256(_encoded(value)).hexdigest()
    if case == "wrong-key":
        key = "f" * 64
    with pytest.raises(CachePolicyError):
        with context.ledger.locked():
            lifecycle._put({}, kind, key, value, actor=actor)
    world.assert_no_mutation()
    world.assert_closed()




def test_typed_authorities_cache_uses_actual_reader_once_and_returns_copies(tmp_path,monkeypatch):
    path=tmp_path/"answers.json";raw=b'[{"id":"N43","decided_by":"owner"}]\n'
    path.write_bytes(raw)
    pin=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    once=resolver._MetadataOnce(str(tmp_path/"package"/"darsia"/"__init__.py"))
    original=resolver._exact_authority_list;calls=[]
    def actual(binding,package_file):
        calls.append(deepcopy(binding));return original(binding,package_file)
    monkeypatch.setattr(resolver,"_exact_authority_list",actual)
    value=once.read_authorities(pin);value[0]["decided_by"]="changed copy"
    assert once.read_authorities(pin)==[{"id":"N43","decided_by":"owner"}]
    assert calls==[pin]
    assert once.seen[path_key(path)]==(pin,[{"id":"N43","decided_by":"owner"}])
    with pytest.raises(CachePolicyError,match="descriptor changed"):
        once.read_authorities({**pin,"sha256":"f"*64})
    assert calls==[pin]


@pytest.mark.parametrize("document",[{},[None],[1]])
def test_typed_authorities_cache_cannot_relabel_an_object_or_invalid_rows(document):
    once=resolver._MetadataOnce(PACKAGE);pin=descriptor("answers.json")
    once.remember(pin,document)
    with pytest.raises(CachePolicyError,match="list of record objects"):
        once.read_authorities(pin)


@pytest.mark.parametrize("fault", [None,"unnamed_new_close","wrong_current_map"])
def test_Q134_two_request_time_maps_reconstruct_new_close_then_named_old_first_and_close(tmp_path,monkeypatch,fault):
    """Real request/Root/map bytes and constructors; external host ledger is synthetic.

    Capacity arithmetic and physical original-service resolution are separate
    tested boundaries. No authority reader or first/close constructor is mocked.
    """
    from tests.unit.gui.test_cache_bootstrap_environment import _historical_authority_case
    from tests.unit.gui.test_cache_first_qualification import provider
    from darsia_cache import first_qualification as first
    from darsia_cache.retained_qualification import RetainedColdScope
    case=_historical_authority_case(tmp_path)
    read=lambda binding:bootstrap._exact(binding,case.package)
    dummy=case.write("unused_metadata.json",{})
    declared=provider();provider_pin=case.write("provider.json",declared)
    task={key:value for key,value in declared["scope"].items() if key!="backend"}
    task.update(attempt_id="2"*32,job_id="TESTONLY-nested")
    task_pin=case.write("task.json",task)
    control_root=tmp_path/"synthetic_ledger";control_root.mkdir()
    contexts={};resolved=[]
    monkeypatch.setattr(first,"checked_required_capacity_bound",lambda value:{"TESTONLY":"separate capacity boundary"})
    def output_check(task,value,*,task_binding,read,read_authorities):
        return {"authority_records":read_authorities(value["answers_source"])}
    monkeypatch.setattr(first,"checked_output_entitlement",output_check)
    def make_first(name,*,cold=None,map_pin=None,answers=None):
        request=dict(format=first.REQUEST,execution_authorized=True,authority_ids=["Q97","Q99"],
            provider=provider_pin,stores_source=dummy,namespace_admission=declared["namespace_admission"],
            scope=declared["scope"],task=task_pin,cache_stat_period_seconds=30,
            public_command=["TESTONLY-python","-B","-m","darsia.fluidflower.worker"])
        if cold is not None:
            request.update(format=first.BOUND_REQUEST,
                authority_ids=["Q97","Q99","Q102","Q104","Q106","Q109","Q117","Q119"],
                capacity_bound_plan=dummy,q123_cold_scope=cold,
                output_entitlement_plan=case.write(name+"_output.json",dict(answers_source=answers)))
        release=dict(format="darsia-cache-first-qualification-root-release-v1",execution_authorized=True,
            provider=provider_pin,stores_source=dummy,scope=declared["scope"],request_payload_sha256=first._digest(request))
        if map_pin is not None:release["historical_authority_resolution"]=map_pin
        request["root_release"]=case.write(name+"_release.json",release)
        return case.write(name+"_request.json",request)
    def make_context(label,token):
        service=case.write(label+"_service.json",dict(TESTONLY=label))
        context=SimpleNamespace(entry_kind="worker",purpose="jobs",reservation=dict(token=token,pool="a"*64,sha256="b"*64),
            manifest=dict(owner=deepcopy(PRIMARY)),ledger=SimpleNamespace(bound=dict(path=str(control_root))),
            policy=SimpleNamespace(root=str(tmp_path/"synthetic_cache")))
        contexts[service["path"]]=context
        return service,context
    def make_close(name,service,original_request,context,*,map_pin=None,answers=None):
        request=dict(format="darsia-cache-retained-service-close-request-v1",authority_ids=["Q123"],
            original_service_receipt=service,original_request=original_request,original_native_retirement=dummy,
            original_lifecycle_eligibility=dummy,recorded_owner_retirement=dummy,retained_cache_inventory=dummy,
            answers_source=answers)
        release=dict(format="darsia-cache-retained-service-close-root-release-v1",execution_authorized=True,
            request_payload_sha256=first._digest(request))
        if map_pin is not None:release["historical_authority_resolution"]=map_pin
        request["root_release"]=case.write(name+"_release.json",release)
        request_pin=case.write(name+"_request.json",request)
        raw=b'{"TESTONLY":"closed metadata; no operation performed"}'
        path=control_root/("cache-c-"+context.reservation["token"]+".json");path.write_bytes(raw)
        close_pin=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
        result=dict(format="darsia-cache-retained-service-close-result-v1",request=request_pin,
            reservation=context.reservation,close_record=close_pin,original_service_receipt=service,
            original_lifecycle_eligibility=dummy,original_native_retirement=dummy,retained_cache_inventory=dummy,
            all_files_retained=True,retained_bytes_are_still_used=True,cleanup_authorized=False,rearm_authorized=False)
        return request_pin,case.write(name+"_result.json",result)
    def cold(name,result,answers):
        return case.write(name+"_cold.json",dict(format="darsia-q123-bound-cold-scope-v1",authority_ids=["Q123"],
            answers_source=answers,original_close_result=result,retained_cache_inventory=dummy,
            matplotlib_namespace=dict(path=str(tmp_path/name/"mpl"),identity=dict(TESTONLY="not admitted"))))
    base_first=make_first("old_base")
    old_service,old_context=make_context("old_base","a"*32)
    old_close,old_result=make_close("old_close",old_service,base_first,old_context,answers=case.old_pin)
    old_first=make_first("old_first",cold=cold("old_first",old_result,case.old_pin),answers=case.old_pin)
    map1=deepcopy(case.value);map1["preserved_requests"]=[base_first,old_close,old_first]
    map1_pin=case.write("map1_at_new_close_request_time.json",map1)
    new_service,new_context=make_context("failed_old_first","b"*32)
    new_close,new_result=make_close("new_close",new_service,old_first,new_context,
        map_pin=map1_pin,answers=case.current_pin)
    later=case.current+[{"id":"later_request_authority","decided_by":"coordinator"}]
    live_later=case.write("coordinator_answers.json",later)
    later_snapshot=case.write("later_answers_snapshot.json",later)
    live_status=case.write("coordinator_status.json",dict(seq=264))
    status_snapshot=case.write("later_status_snapshot.json",dict(seq=264))
    map2=deepcopy(map1)
    map2["current_authorities"]=dict(coordinator_answers=dict(source=live_later,snapshot=later_snapshot),
        coordinator_status=dict(source=live_status,snapshot=status_snapshot))
    if fault!="unnamed_new_close":map2["preserved_requests"].append(new_close)
    map2_pin=case.write("map2_at_later_first_request_time.json",map2)
    new_first=make_first("new_first",cold=cold("new_first",new_result,later_snapshot),
        map_pin=map1_pin if fault=="wrong_current_map" else map2_pin,answers=later_snapshot)
    def original_context(service,*,package_file,first_request=None,retained_close=None,historical_authorities=None):
        # Stand in only for original host/ledger metadata. Exercise the actual
        # nested first constructor, which recursively builds the actual old close.
        child=first.FirstQualification(first_request,provider_binding=provider_pin,stores_source=dummy,
            namespace=declared["namespace_admission"],read=read,
            read_authorities=historical_authorities.for_original(first_request),
            historical_authorities=historical_authorities)
        resolved.append(first_request)
        if first_request==old_first:
            assert isinstance(child.q123_cold_scope,RetainedColdScope)
        return contexts[service["path"]]
    monkeypatch.setattr(resolver,"resolve_original_service_context",original_context)
    monkeypatch.setattr(resolver.RetainedServiceCloseContext,"close_retained",
        lambda self:pytest.fail("Constructors must never perform a close"))
    if fault is not None:
        with pytest.raises(CachePolicyError):
            first.FirstQualification(new_first,provider_binding=provider_pin,stores_source=dummy,
                namespace=declared["namespace_admission"],read=read,read_authorities=lambda pin:pytest.fail("outer stale reader"))
        assert not resolved
    else:
        current=first.FirstQualification(new_first,provider_binding=provider_pin,stores_source=dummy,
            namespace=declared["namespace_admission"],read=read,read_authorities=lambda pin:pytest.fail("outer stale reader"))
        assert isinstance(current.q123_cold_scope.closed,resolver.RetainedServiceCloseContext)
        assert resolved==[base_first,old_first]
        assert current.historical_authorities.binding==map2_pin
        assert new_close not in map1["preserved_requests"] and new_close in map2["preserved_requests"]


@pytest.mark.parametrize("new_format", [
    "darsia-cache-volume-bound-import-profile-v1",
    "darsia-cache-volume-bound-job-profile-v1",
])
def test_direct_new_typed_profile_refuses_before_observer_or_ledger(world, monkeypatch, new_format):
    world.profile["format"] = new_format
    def forbidden_owner():
        raise AssertionError("Unresolved direct typed profile reached actual observer construction")
    monkeypatch.setattr(resolver, "_owner", forbidden_owner)
    with pytest.raises(CachePolicyError, match="original checked provider DAG"):
        resolver.resolve_original_service_context(world.receipt_pin, package_file=PACKAGE)
    assert world.ledgers == []
    assert not any(event[0] == "actual-observer" for event in world.events)
    world.assert_no_mutation()
    world.assert_closed()


# Q123 uses the unchanged original budget-close API through one explicit view.
# These fixtures never authorize a real retirement, file cleanup or rearm.
def retained_close_TESTONLY(world,monkeypatch):
    context=object.__new__(resolver.RetainedServiceCloseContext)
    context.original=resolve(world);context.binding=descriptor("Q123-request.json")
    context.request={key:descriptor(key+".json") for key in ("original_service_receipt",
        "original_lifecycle_eligibility","original_native_retirement","retained_cache_inventory")}
    calls=[]
    monkeypatch.setattr(context,"_retired_locked",lambda row:calls.append(deepcopy(row)) or True)
    return context,calls


def test_Q123_retained_close_uses_original_append_and_never_deletes_or_rearms(world,monkeypatch):
    context,calls=retained_close_TESTONLY(world,monkeypatch)
    before=deepcopy(world.records)
    actual=context.close_retained()
    assert len(calls)==1 and calls[0]==world.reservation_row
    name="cache-c-"+TOKEN+".json"
    assert set(world.records)-set(before)=={name}
    assert all(world.records[key]==raw for key,raw in before.items())
    row=capacity._json_bytes(world.records[name])
    assert row["owner"]==PRIMARY and row["token"]==TOKEN
    assert row["reservation_sha256"]==world.reservation["sha256"] and row["no_further_writes"] is True
    assert actual["close_record"]["sha256"]==hashlib.sha256(world.records[name]).hexdigest()
    assert actual["all_files_retained"] and actual["retained_bytes_are_still_used"]
    assert actual["cleanup_authorized"] is False and actual["rearm_authorized"] is False
    world.assert_closed()
    with pytest.raises(CachePolicyError,match="Closed, foreign"):
        context.close_retained()
    assert set(world.records)-set(before)=={name}


def test_Q123_retirement_refusal_preserves_all_records_and_original_observer_capability(world,monkeypatch):
    context,_=retained_close_TESTONLY(world,monkeypatch);before=deepcopy(world.records)
    def refuse(row):raise CachePolicyError("TESTONLY genuine owner still live")
    monkeypatch.setattr(context,"_retired_locked",refuse)
    with pytest.raises(CachePolicyError,match="still live"):context.close_retained()
    assert world.records==before and world.writes==[]
    with pytest.raises(CachePolicyError,match="Read-only observer"):
        context.original.ledger.close(world.reservation,owner=PRIMARY,verify_no_further_writes=lambda row:True)
    world.assert_closed()


@pytest.mark.parametrize("change",["token","owner","sha","cleanup","other_name"])
def test_Q123_close_view_refuses_any_other_token_owner_sha_or_write(change):
    writes=[]
    namespace=SimpleNamespace(write_once=lambda *args,**kwargs:writes.append(args))
    reservation=dict(token=TOKEN,sha256="a"*64)
    view=resolver._RetainedCloseMetadataOnly(namespace,reservation,PRIMARY)
    name="cache-c-"+TOKEN+".json"
    row=dict(owner=PRIMARY,token=TOKEN,utc="TESTONLY recorded close time",no_further_writes=True,reservation_sha256="a"*64)
    if change=="token":row["token"]="b"*32
    if change=="owner":row["owner"]=OBSERVER
    if change=="sha":row["reservation_sha256"]="b"*64
    if change=="cleanup":row["cleanup_authorized"]=True
    if change=="other_name":name="cache-r-"+TOKEN+".json"
    with pytest.raises(CachePolicyError,match="only the exact original"):
        view.write_once(name,_encoded(row))
    assert not writes


def test_Q123_retained_inventory_preserves_real_file_and_refuses_changed_identity(tmp_path):
    root=tmp_path/"TESTONLY-cache";root.mkdir();path=root/"fontlist.json";path.write_bytes(b"TESTONLY-font")
    info=path.lstat()
    inventory=dict(format="darsia-cache-retained-file-inventory-v1",root=str(root),no_cleanup=True,
        logical_bytes=info.st_size,files=[dict(relative_path="fontlist.json",dev=info.st_dev,ino=info.st_ino,
            size_bytes=info.st_size,mtime_ns=info.st_mtime_ns)])
    before=path.read_bytes()
    assert resolver.verify_retained_cache_inventory(inventory,str(root))==inventory
    assert path.read_bytes()==before
    changed=deepcopy(inventory);changed["files"][0]["ino"]+=1
    with pytest.raises(CachePolicyError,match="identity/version"):
        resolver.verify_retained_cache_inventory(changed,str(root))
    path.write_bytes(b"TESTONLY-modified-font")
    with pytest.raises(CachePolicyError,match="identity/version"):
        resolver.verify_retained_cache_inventory(inventory,str(root))


def test_Q123_close_request_without_actual_Root_release_never_resolves_service(monkeypatch):
    from darsia_cache import first_qualification as first
    roles=("original_service_receipt","original_request","original_native_retirement",
        "original_lifecycle_eligibility","recorded_owner_retirement","retained_cache_inventory","answers_source","root_release")
    request={role:descriptor(role+".json") for role in roles}
    request.update(format="darsia-cache-retained-service-close-request-v1",authority_ids=["Q123"])
    monkeypatch.setattr(resolver._MetadataOnce,"read",lambda self,binding:
        {} if binding==request["root_release"] else deepcopy(request))
    def forbidden(*args,**kwargs):raise AssertionError("No unreleased service resolution")
    monkeypatch.setattr(resolver,"resolve_original_service_context",forbidden)
    with pytest.raises(CachePolicyError,match="Actual Root release"):
        resolver.RetainedServiceCloseContext(descriptor("close-request.json"),package_file=PACKAGE)


def retained_retirement_TESTONLY(world,monkeypatch):
    import psutil
    from darsia_cache import service_lifecycle,first_qualification
    context=object.__new__(resolver.RetainedServiceCloseContext);context.original=resolve(world)
    context.request={role:descriptor(role+".json") for role in ("original_native_retirement",
        "original_service_receipt","recorded_owner_retirement","original_lifecycle_eligibility")}
    context.request["original_service_receipt"]=world.receipt_pin
    context._seal=first_qualification._digest(context.request)
    binding={"TESTONLY":"original lifecycle binding"}
    sealed=dict(binding=descriptor("original-seal.json"),document={"value":{"outcome":"failed"}})
    owner=dict(binding=descriptor("original-owner.json"),document={"value":{"owner":PRIMARY}})
    records={("seal","service"):sealed,("owner","TESTONLY-owner"):owner}
    snapshot=dict(binding=binding,seal=sealed,owners=[owner],launches=[],legacy_launches=[],
        child_coverage=dict(entry_kind="worker",registered_launches_are_not_exhaustive_proof=True,
            complete_external_child_coverage_provider_required=True))
    native=dict(format="darsia-native-owned-worker-retirement-v1",service_receipt=world.receipt_pin,
        primary=PRIMARY,whole_family_kernel_observation=True,no_process_kill_after_resume=True,
        terminal=dict(accounting=dict(active_processes=0,terminated_by_limit_processes=0),job_flags=0))
    eligibility=dict(binding=binding,kind="eligibility",key="TESTONLY-eligibility",
        value=dict(registry_sha256=service_lifecycle._digest(snapshot),
            retirement=context.request["original_native_retirement"],original_retirement_verifier_accepted=True,
            cleanup_authorized=False,service_budget_closed=False))
    records[("eligibility","TESTONLY-eligibility")]=dict(binding=context.request["original_lifecycle_eligibility"],document=eligibility)
    observed=dict(actual=dict(recorded_owned_epochs=[dict(pid=PRIMARY["pid"],create_time="TESTONLY_PRIMARY_EPOCH",recorded_epoch_absent=True)],
        original_whole_native_family_retirement=dict(context.request["original_native_retirement"],document=native)))
    docs={context.request["original_native_retirement"]["path"]:native,
        context.request["recorded_owner_retirement"]["path"]:observed,
        context.request["original_lifecycle_eligibility"]["path"]:eligibility}
    context.reads=SimpleNamespace(read=lambda pin:deepcopy(docs[pin["path"]]))
    monkeypatch.setattr(context,"_retained_files",lambda:{"TESTONLY":"separately verified real-file inventory"})
    monkeypatch.setattr(service_lifecycle,"ServiceLifecycle",lambda original:SimpleNamespace(binding=binding,_records=lambda:records))
    def absent(pid):raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil,"Process",absent)
    return context,native,eligibility,records,observed


def test_Q123_close_verifier_requires_exact_original_native_registry_and_eligibility(world,monkeypatch):
    context,_,_,_,_=retained_retirement_TESTONLY(world,monkeypatch)
    before=deepcopy(world.records)
    assert context._retired_locked(world.reservation_row) is True
    assert world.records==before and world.writes==[]


@pytest.mark.parametrize("kind",["active_native","kernel_limit","live_owner","unresolved_launch","registry","eligibility","wrong_owner","wrong_epoch"])
def test_Q123_close_verifier_refuses_live_or_incomplete_original_retirement(world,monkeypatch,kind):
    import psutil
    context,native,eligibility,records,observed=retained_retirement_TESTONLY(world,monkeypatch)
    row=deepcopy(world.reservation_row);before=deepcopy(world.records)
    if kind=="active_native":native["terminal"]["accounting"]["active_processes"]=1
    if kind=="kernel_limit":native["terminal"]["accounting"]["terminated_by_limit_processes"]=1
    if kind=="live_owner":monkeypatch.setattr(psutil,"Process",lambda pid:SimpleNamespace(create_time=lambda:"TESTONLY_PRIMARY_EPOCH"))
    if kind=="unresolved_launch":records[("intent","TESTONLY-unresolved")]=dict(document={})
    if kind=="registry":eligibility["value"]["registry_sha256"]="f"*64
    if kind=="eligibility":eligibility["value"]["cleanup_authorized"]=True
    if kind=="wrong_owner":row["owner"]=OBSERVER
    if kind=="wrong_epoch":observed["actual"]["recorded_owned_epochs"][0]["create_time"]="TESTONLY_OTHER_EPOCH"
    with pytest.raises(CachePolicyError):context._retired_locked(row)
    assert world.records==before and world.writes==[]


def test_Q123_closed_original_requires_exact_pinned_close(world,monkeypatch):
    context,_=retained_close_TESTONLY(world,monkeypatch)
    result=context.close_retained();before=deepcopy(world.records)
    with pytest.raises(CachePolicyError,match="Closed, foreign"):resolve(world)
    observed=resolver.resolve_original_service_context(world.receipt_pin,package_file=PACKAGE,
        retained_close=result["close_record"])
    assert observed.manifest==context.original.manifest and world.records==before
    wrong=deepcopy(result["close_record"]);wrong["sha256"]="f"*64
    with pytest.raises(CachePolicyError,match="reservation/close"):
        resolver.resolve_original_service_context(world.receipt_pin,package_file=PACKAGE,retained_close=wrong)
    assert world.records==before


def closed_registry_distinct_ledger_TESTONLY(world, monkeypatch):
    """Real ledger/lifecycle/close methods; only native and metadata I/O are synthetic."""
    from dataclasses import replace
    import psutil
    from darsia_cache.first_qualification import _digest

    original = resolve(world)
    current = control.CacheBudgetLedger(world.policy, SimpleNamespace(**world.actual),
        measurement_provider=lambda root: None, namespace_admission=world.admission,
        verify_host_admission=resolver._host_binding)
    assert current is not original.ledger
    producer = replace(original, owner=deepcopy(PRIMARY), ledger=current)
    lifecycle = ServiceLifecycle(producer)
    lifecycle.register_owner()
    generation = "b" * 32
    child = dict(machine=PRIMARY["machine"], pid=303, process_create_time="TESTONLY_CHILD_EPOCH")
    lifecycle.record_launch("worker_spawn_intent", dict(slot=0, generation=generation))
    lifecycle.record_launch("worker_spawn_observed", dict(slot=0, generation=generation,
        original_native_identity=child))
    lifecycle.record_launch("worker_children_retired", dict(slot=0, generation=generation,
        original_child_identities=[child], owned_launch_handle_retired=True))
    lifecycle.seal(logical_gate="TESTONLY_RETAINED_GATE", outcome="failed",
        utc="2026-01-01T00:00:00Z", traceback_text="TESTONLY original retained failure")

    context = object.__new__(resolver.RetainedServiceCloseContext)
    context.original = original
    context.binding = descriptor("TESTONLY-real-method-close-request.json")
    context.request = {role: descriptor("TESTONLY-" + role + ".json") for role in (
        "original_native_retirement", "recorded_owner_retirement", "retained_cache_inventory")}
    context.request["original_service_receipt"] = world.receipt_pin
    native = dict(format="darsia-native-owned-worker-retirement-v1", service_receipt=world.receipt_pin,
        primary=PRIMARY, whole_family_kernel_observation=True, no_process_kill_after_resume=True,
        terminal=dict(accounting=dict(active_processes=0, terminated_by_limit_processes=0), job_flags=0))
    monkeypatch.setattr(bootstrap, "_owner", world.observe)

    def synthetic_external_retirement(snapshot, pin):
        assert pin == context.request["original_native_retirement"]
        assert len(snapshot["owners"]) == 1 and len(snapshot["launches"]) == 1
        return True  # Synthetic external evidence; never a production provider.

    eligibility = lifecycle.retention_eligibility(now_utc="2026-01-03T00:00:00Z",
        retirement=context.request["original_native_retirement"], verify_retirement=synthetic_external_retirement)
    pin = eligibility["receipt"]
    context.request["original_lifecycle_eligibility"] = pin
    eligibility_doc = capacity._json_bytes(world.records[PureWindowsPath(pin["path"]).name])
    observed = dict(actual=dict(recorded_owned_epochs=[dict(pid=row["pid"],
        create_time=row["process_create_time"], recorded_epoch_absent=True) for row in (PRIMARY, child)],
        original_whole_native_family_retirement=dict(context.request["original_native_retirement"], document=native)))
    inventory = dict(format="darsia-cache-retained-file-inventory-v1", root=world.policy.root,
        no_cleanup=True, files=[], logical_bytes=0)
    for key, document in (("original_native_retirement", native), ("recorded_owner_retirement", observed),
                          ("original_lifecycle_eligibility", eligibility_doc), ("retained_cache_inventory", inventory)):
        bound = context.request[key]
        world.metadata[path_key(bound["path"])] = (bound, document)
    context.reads = resolver._MetadataOnce(PACKAGE)
    context._seal = _digest(context.request)

    def absent(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", absent)
    # This genuinely runs the original open-reservation close/retirement path.
    result = context.close_retained()
    assert "cache-c-" + TOKEN + ".json" in world.records
    assert not original.ledger.lock_active and original.ledger.control is None
    return context, current, result["close_record"], observed


def test_Q123_postclose_real_registry_uses_distinct_current_ledger_without_reopening(world, monkeypatch):
    from dataclasses import replace
    context, current, pin, _ = closed_registry_distinct_ledger_TESTONLY(world, monkeypatch)
    before = deepcopy(world.records)
    writes = list(world.writes)
    with current.locked(create_binding=False):
        # The ordinary producer/lifecycle guard still refuses the closed token.
        with pytest.raises(CachePolicyError, match="Closed, foreign"):
            current.require_reservation(world.reservation, PRIMARY, 0)
        with pytest.raises(CachePolicyError, match="Closed, foreign"):
            ServiceLifecycle(replace(context.original, ledger=current))._records()
        assert context.retired_after_close_locked(current, pin) is True
        assert context.original.ledger.control is None and not context.original.ledger.lock_active
        view = resolver._RetainedRegistryLedger(context.original, current, pin)
        for operation in (view.reserve, view.admit, view.bind_scope, view.close, view.project,
                          view.locked, view.control.write_once):
            with pytest.raises(CachePolicyError, match="cannot write or allocate"):
                operation()
        with pytest.raises(CachePolicyError, match="only reads the original service registry"):
            view.control.read_once("cache-r-" + TOKEN + ".json")
    assert world.records == before and world.writes == writes
    world.assert_closed()


@pytest.mark.parametrize("change", ["pin", "unlocked", "canonical", "cache-root", "registry",
                                   "registry-digest", "live-epoch", "close-owner"])
def test_Q123_postclose_refuses_changed_pin_lock_identity_registry_or_owner(world, monkeypatch, change):
    import psutil
    context, current, pin, _ = closed_registry_distinct_ledger_TESTONLY(world, monkeypatch)
    if change == "unlocked":
        with pytest.raises(CachePolicyError, match="transaction"):
            context.retired_after_close_locked(current, pin)
        return
    with current.locked(create_binding=False):
        if change == "pin":
            pin = dict(pin, sha256="f" * 64)
        elif change == "canonical":
            current.bound = dict(current.bound, identity={"TESTONLY": "another canonical ledger"})
        elif change == "cache-root":
            current.policy = SimpleNamespace(root="C:/TESTONLY_NOT_CREATED/different-cache")
        elif change in {"registry", "registry-digest"}:
            name = next(n for n in world.records if n.startswith("cache-l-" + TOKEN + "-owner-"))
            row = capacity._json_bytes(world.records[name])
            if change == "registry":
                row["value"]["owner"]["pid"] += 1
            else:
                row["utc"] = "2026-01-01T01:00:00Z"
            world.records[name] = _encoded(row)
        elif change == "close-owner":
            name = "cache-c-" + TOKEN + ".json"
            row = capacity._json_bytes(world.records[name])
            row["owner"] = OBSERVER
            world.records[name] = _encoded(row)
            pin = dict(pin, sha256=hashlib.sha256(world.records[name]).hexdigest(),
                size_bytes=len(world.records[name]))
        elif change == "live-epoch":
            monkeypatch.setattr(psutil, "Process", lambda pid:
                SimpleNamespace(create_time=lambda: "TESTONLY_PRIMARY_EPOCH"))
        before, writes = deepcopy(world.records), list(world.writes)
        with pytest.raises(CachePolicyError):
            context.retired_after_close_locked(current, pin)
        assert world.records == before and world.writes == writes
    world.assert_closed()
