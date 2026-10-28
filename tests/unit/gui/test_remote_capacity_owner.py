"""Q49 metadata/evidence doubles only: no NAS or OS owner probe is executed."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pytest

from darsia.fluidflower import job_filesystem as fs
from darsia.fluidflower import store_map
from darsia.fluidflower.pipeline import capacity as cap, worker
from darsia.fluidflower.pipeline.store_runtime import RunUnavailable
from .test_capacity import ledger_fixture, claim_fixture, owner, estimate, settings_doc


OWNER_PATH = r"\\nas-a\share\FF_AC\DarSIA\jobs\NAS_OWNER.json"
ROOT = r"\\nas-a\share\results"


def configured():
    doc = dict(format="darsia-single-host-nas-authority-v1", canonical_owner_path=OWNER_PATH,
        aliases=[r"\\nas-a\share", r"\\nas-b\share"],
        owner=dict(machine_id="main", host="physical-fixture", service_account_sid="S-1-5-21-100"),
        single_computing_host=True)
    raw = json.dumps(doc).encode()
    bound = dict(path=OWNER_PATH, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    context = dict(host="physical-fixture", service_account_sid="S-1-5-21-100")
    return doc, bound, context


def test_aliases_are_one_conservative_authority_not_a_volume_proof():
    doc, bound, context = configured()
    a = cap._owner_scope(doc, bound, ROOT, context)
    b = cap._owner_scope(doc, bound, r"\\nas-b\share\another-results-root", context)
    assert a == b and a["owner_document"] == bound
    assert "no physical-volume equivalence" in a["accounting"]


@pytest.mark.parametrize("defect", ["host", "account", "nonstring_host"])
def test_N51_legacy_alias_provenance_does_not_elect_one_computing_host(defect):
    doc, bound, context = configured()
    original = cap._owner_scope(doc, bound, ROOT, context)
    if defect == "host": context["host"] = "second-physical-machine"
    elif defect == "account": context["service_account_sid"] = "S-1-5-21-200"
    else: context["host"] = None
    assert cap._owner_scope(doc, bound, ROOT, context) == original


@pytest.mark.parametrize("root", [r"\\not-listed\share\results", r"\\nas-a\another-share\results", r"Z:\results"])
def test_unlisted_server_share_or_mapped_drive_is_not_silently_equivalent(root):
    doc, bound, context = configured()
    with pytest.raises((ValueError, RunUnavailable)):
        cap._owner_scope(doc, bound, root, context)


@pytest.mark.parametrize("defect", ["schema", "not_true", "duplicate_alias", "unsorted_alias", "subdirectory_alias", "canonical_path", "sid"])
def test_exact_owner_schema_and_namespace_cannot_be_relaxed(defect):
    doc, bound, context = configured()
    if defect == "schema": doc["extra_owner"] = "other"
    elif defect == "not_true": doc["single_computing_host"] = 1
    elif defect == "duplicate_alias": doc["aliases"] += [r"\\NAS-A\share"]
    elif defect == "unsorted_alias": doc["aliases"].reverse()
    elif defect == "subdirectory_alias": doc["aliases"][0] += "\\subdirectory"
    elif defect == "canonical_path": bound["path"] = OWNER_PATH.replace("NAS_OWNER.json", "other.json")
    else: doc["owner"]["service_account_sid"] = "guessed-account-name"
    with pytest.raises((ValueError, RunUnavailable)):
        cap._owner_scope(doc, bound, ROOT, context)


def test_local_output_reads_no_owner_metadata_or_context(tmp_path, monkeypatch):
    proof = fs.LocalFileAdmission(True, "fixed", "synthetic local proof", "injected evidence", str(tmp_path))
    monkeypatch.setattr(fs, "valid_results_admission", lambda admission, root: admission is proof)
    monkeypatch.setattr(cap, "read_bound", lambda _: pytest.fail("Local output must not read a NAS owner"))
    monkeypatch.setattr(cap, "_capacity_host_context", lambda: pytest.fail("No OS owner query for local output"))
    assert cap.remote_results_owner(SimpleNamespace(), "FF_AC", tmp_path, admission=proof) is None


def test_remote_guard_reads_exact_config_only_and_never_self_elects(tmp_path, monkeypatch):
    doc, bound, context = configured()
    settings = settings_doc(tmp_path)
    settings["capacity"]["remote_owner"] = bound
    stores = SimpleNamespace(document=settings, machine_id="another-logical-alias")
    proof = fs.ResultsRootAdmission(True, "smb", False, "injected", "injected", ROOT, ROOT,
        fs.SharedFolderAdmission(True, "smb", "injected", "injected"))
    monkeypatch.setattr(fs, "valid_results_admission", lambda actual, root: actual is proof)
    seen = []
    def read(value):
        seen.append(deepcopy(value));return deepcopy(doc), b"already verified synthetic bytes"
    monkeypatch.setattr(cap, "read_bound", read)
    before = list(tmp_path.rglob("*"))
    result = cap.remote_results_owner(stores, "FF_AC", ROOT, admission=proof, context=context)
    assert seen == [bound] and result["owner"]["host"] == context["host"]
    assert list(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("defect", ["missing", "changed"])
def test_N51_no_owner_is_required_but_explicit_provenance_stays_SHA_bound(tmp_path, monkeypatch, defect):
    doc, bound, context = configured()
    settings = settings_doc(tmp_path)
    if defect == "changed": settings["capacity"]["remote_owner"] = bound
    proof = SimpleNamespace(local=False)
    monkeypatch.setattr(fs, "valid_results_admission", lambda *a: True)
    def fail(_): raise ValueError("literal owner SHA changed")
    monkeypatch.setattr(cap, "read_bound", fail)
    before = list(tmp_path.rglob("*"))
    if defect == "missing":
        assert cap.remote_results_owner(SimpleNamespace(document=settings), "FF_AC", ROOT,
                                        admission=proof, context=context) is None
    else:
        with pytest.raises(RunUnavailable, match="missing or changed"):
            cap.remote_results_owner(SimpleNamespace(document=settings), "FF_AC", ROOT,
                                     admission=proof, context=context)
    assert list(tmp_path.rglob("*")) == before


def test_wrong_owner_startup_precedes_even_the_original_queue_constructor(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "remote_results_owner", lambda *a, **k: (_ for _ in ()).throw(RunUnavailable("literal other actual host")))
    monkeypatch.setattr(worker, "JobQueue", lambda *a, **k: pytest.fail("No claim/queue after owner refusal"))
    stores = SimpleNamespace(machine_id="main", roots=lambda _: dict(results=tmp_path / "not-created"))
    job = dict(format="darsia-alias-job-v1", series_id="FF_AC")
    with pytest.raises(RunUnavailable, match="literal other actual host"):
        worker._run_tasks(str(tmp_path), job, "a" * 40, "main-cpu", "cpu", None, None,
                          once=True, exit_when_idle=False, stores=stores)
    assert not (tmp_path / "not-created").exists()


def test_all_aliases_share_one_ledger_even_when_client_volume_ids_differ(tmp_path, monkeypatch):
    local, stores, state = ledger_fixture(tmp_path, monkeypatch)
    doc, bound, context = configured()
    scope = cap._owner_scope(doc, bound, ROOT, context)
    monkeypatch.setattr(cap, "remote_results_owner", lambda *a, **k: deepcopy(scope))
    a = cap.CapacityLedger(stores, "FF_AC", tmp_path / "root-a",
        volume_provider=lambda _: dict(volume_id="client-A", free_bytes=10000), context_provider=local.context_provider)
    other_stores = SimpleNamespace(document=stores.document, machine_id="same-host-cuda-alias", roots=stores.roots)
    b = cap.CapacityLedger(other_stores, "FF_AC", tmp_path / "root-b",
        volume_provider=lambda _: dict(volume_id="client-B", free_bytes=10000), context_provider=local.context_provider)
    assert a.key == b.key and a.binding_path == b.binding_path and a.root == b.root
    with a.locked():
        first = a.bind(tmp_path / "job1", claim_fixture(), owner(), estimate())
    with b.locked():
        assert b.remaining() == 390
        b.bind(tmp_path / "job2", claim_fixture(job="two", token="2" * 32), owner(124), estimate())
    assert a.remaining() == 780
    saved = json.loads((first / "reserved.json").read_bytes())
    assert saved["observed_client_volume"]["volume_id"] == "client-A"
    assert saved["volume_identity_claim"].startswith("none;")
    assert "remote_scope" in json.loads(a.binding_path.read_bytes())


def test_alias_does_not_bypass_the_one_common_local_ledger_directory(tmp_path, monkeypatch):
    local, stores, state = ledger_fixture(tmp_path, monkeypatch)
    doc, bound, context = configured()
    scope = cap._owner_scope(doc, bound, ROOT, context)
    monkeypatch.setattr(cap, "remote_results_owner", lambda *a, **k: deepcopy(scope))
    a = cap.CapacityLedger(stores, "FF_AC", tmp_path / "root-a", volume_provider=lambda _: dict(state), context_provider=local.context_provider)
    with a.locked(): pass
    changed = deepcopy(stores.document);changed["capacity"]["state_directory"] = str(tmp_path / "wrong-ledger")
    b = cap.CapacityLedger(SimpleNamespace(document=changed, machine_id="main", roots=stores.roots), "FF_AC", tmp_path / "root-b",
        volume_provider=lambda _: dict(volume_id="another-client-id", free_bytes=1), context_provider=local.context_provider)
    with pytest.raises(RunUnavailable, match="another configured directory"):
        with b.locked(): pytest.fail("Cannot split the authority pool")
    assert not (tmp_path / "wrong-ledger").exists()


@pytest.mark.parametrize("defect", ["local_path", "bool_size", "too_large", "bad_sha"])
def test_local_map_requires_one_exact_bounded_canonical_owner_descriptor(tmp_path, defect):
    _, bound, _ = configured();settings = settings_doc(tmp_path)
    if defect == "local_path": bound["path"] = str(tmp_path / "self-elected.json")
    elif defect == "bool_size": bound["size_bytes"] = True
    elif defect == "too_large": bound["size_bytes"] = 16385
    else: bound["sha256"] = "A" * 64
    settings["capacity"]["remote_owner"] = bound
    with pytest.raises(store_map.StoreMapError): store_map.capacity_settings(settings)


def test_changed_metadata_size_is_refused_before_opening_payload(tmp_path, monkeypatch):
    path = tmp_path / "wrong-sized-owner.json";path.write_bytes(b"x" * 17000)
    original = Path.open
    def guarded(self, *a, **k):
        if self == path: pytest.fail("Oversized metadata must not be opened")
        return original(self, *a, **k)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="changed before read"):
        cap.read_bound(dict(path=str(path), sha256="0" * 64, size_bytes=2))


@pytest.mark.parametrize("host", ["reference-host", "second-host"])
def test_N51_both_hosts_can_use_admitted_NAS_without_owner_or_results_quota(tmp_path, monkeypatch, host):
    settings = settings_doc(tmp_path)
    assert "remote_owner" not in settings["capacity"]
    proof = SimpleNamespace(local=False)
    monkeypatch.setattr(fs, "valid_results_admission", lambda actual, root: actual is proof and root == ROOT)
    monkeypatch.setattr(cap, "read_bound", lambda _: pytest.fail("N51 does not require NAS_OWNER.json"))
    context = dict(host=host, service_account_sid="S-1-5-21-100")
    assert cap.remote_results_owner(SimpleNamespace(document=settings), "FF_AC", ROOT,
                                    admission=proof, context=context) is None


def test_N51_two_actual_hosts_keep_separate_local_bindings_for_same_NAS(tmp_path, monkeypatch):
    local, stores, state = ledger_fixture(tmp_path, monkeypatch)
    doc, bound, first = configured()
    stores.document["capacity"]["remote_owner"] = bound
    proof = SimpleNamespace(allowed=True, local=False)
    monkeypatch.setattr(fs, "inspect_results_root", lambda _: proof)
    monkeypatch.setattr(fs, "valid_results_admission", lambda actual, root: actual is proof)
    monkeypatch.setattr(cap, "read_bound", lambda _: (deepcopy(doc), b"verified synthetic metadata"))
    common = local.context_provider()
    a = cap.CapacityLedger(stores, "FF_AC", ROOT, volume_provider=lambda _: dict(state),
        context_provider=lambda: dict(common, host="reference-host"))
    b = cap.CapacityLedger(stores, "FF_AC", ROOT, volume_provider=lambda _: dict(state),
        context_provider=lambda: dict(common, host="second-host", service_account_sid="S-1-5-21-200"))
    assert a.remote_scope == b.remote_scope
    assert a.key != b.key and a.binding_path != b.binding_path and a.root != b.root
    with a.locked():
        a.bind(tmp_path / "main-job", claim_fixture(), owner(), estimate())
    with b.locked():
        b.bind(tmp_path / "second-job", claim_fixture(job="two", token="2" * 32), owner(124), estimate())
    assert a.remaining() == b.remaining() == 390
    assert a._common_binding() is True and b._common_binding() is True
