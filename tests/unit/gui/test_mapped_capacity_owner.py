"""Q53 metadata/OS-provider doubles only; no actual mappings or NAS qualification."""
from copy import deepcopy
from pathlib import PureWindowsPath
from types import SimpleNamespace
import ctypes
import hashlib
import json

import pytest

from darsia.fluidflower import job_filesystem as fs
from darsia.fluidflower.pipeline import capacity as cap
from darsia.fluidflower.pipeline.store_runtime import RunUnavailable
from .test_capacity import ledger_fixture, claim_fixture, owner, estimate, settings_doc
from .test_remote_capacity_owner import configured, ROOT, OWNER_PATH


MAPPED = r"X:\results"


def drive_configuration():
    document, _, context = configured()
    document["aliases"].append(dict(drive="X:", target_unc=r"\\nas-a\share"))
    raw = json.dumps(document).encode()
    binding = dict(path=OWNER_PATH, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    return document, binding, context


def evidence(monkeypatch, document, binding):
    """Inject explicit remote evidence, retaining original local admission."""
    source = fs.SharedFolderAdmission(True, "smb", "synthetic", "injected OS facts")
    proof = fs.ResultsRootAdmission(True, "smb", False, "synthetic", "injected OS facts",
                                   MAPPED, MAPPED, source)
    original_inspect, original_valid = fs.inspect_results_root, fs.valid_results_admission
    monkeypatch.setattr(fs, "inspect_results_root", lambda root:
        proof if PureWindowsPath(str(root)).drive in {"X:", r"\\nas-a\share"}
        else original_inspect(root))
    monkeypatch.setattr(fs, "valid_results_admission", lambda actual, root:
        (actual.allowed is True and actual.local is False and actual.protocol in {"smb", "nfs"})
        if actual is proof else original_valid(actual, root))
    reads = []

    def read(actual):
        reads.append(deepcopy(actual))
        raw = json.dumps(document).encode()
        if (actual != binding or hashlib.sha256(raw).hexdigest() != binding["sha256"]
                or len(raw) != binding["size_bytes"]):
            raise ValueError("synthetic owner SHA/size changed")
        return deepcopy(document), raw

    monkeypatch.setattr(cap, "read_bound", read)
    return proof, reads


def test_explicit_drive_and_unc_keep_one_static_scope_and_five_owner_keys():
    document, binding, context = drive_configuration()
    unc = cap._owner_scope(document, binding, ROOT, context)
    mapped = cap._owner_scope(document, binding, MAPPED, context)
    assert unc == mapped
    assert set(document) == {"format", "canonical_owner_path", "aliases", "owner", "single_computing_host"}
    assert mapped["owner_document"] == binding
    assert mapped["aliases"][-1] == dict(drive="X:", target_unc=r"\\nas-a\share" + "\\")
    assert "no physical-volume equivalence" in mapped["accounting"]


@pytest.mark.parametrize("defect", ["bare_drive", "lower_drive", "duplicate_drive", "unknown_key",
    "unapproved_target", "target_subfolder", "drive_first", "device_target", "unc_duplicate"])
def test_mapped_alias_contract_cannot_expand_authority(defect):
    document, binding, context = drive_configuration()
    if defect == "bare_drive": document["aliases"][-1] = "X:"
    elif defect == "lower_drive": document["aliases"][-1]["drive"] = "x:"
    elif defect == "duplicate_drive": document["aliases"].append(deepcopy(document["aliases"][-1]))
    elif defect == "unknown_key": document["aliases"][-1]["guess"] = True
    elif defect == "unapproved_target": document["aliases"][-1]["target_unc"] = r"\\not-approved\share"
    elif defect == "target_subfolder": document["aliases"][-1]["target_unc"] += r"\folder"
    elif defect == "drive_first": document["aliases"].insert(0, document["aliases"].pop())
    elif defect == "device_target": document["aliases"][-1]["target_unc"] = r"\\?\UNC\nas-a\share"
    else: document["aliases"].insert(1, r"\\NAS-A\share")
    with pytest.raises((ValueError, RunUnavailable)):
        cap._owner_scope(document, binding, MAPPED, context)


@pytest.mark.parametrize("root", [r"Y:\results", r"W:\results", r"X:relative", r"X:\..\escape", r"\\?\X:\results"])
def test_unknown_or_relative_drive_roots_refuse(root):
    document, binding, context = drive_configuration()
    with pytest.raises((ValueError, RunUnavailable)):
        cap._owner_scope(document, binding, root, context)


@pytest.mark.parametrize("target", [r"\\nas-b\share", r"\\other\share", r"\\nas-a\share\folder",
    r"\\?\UNC\nas-a\share", "", None])
def test_remap_to_even_another_approved_unc_refuses(tmp_path, monkeypatch, target):
    document, binding, context = drive_configuration()
    proof, _ = evidence(monkeypatch, document, binding)
    settings = settings_doc(tmp_path); settings["capacity"]["remote_owner"] = binding
    stores = SimpleNamespace(document=settings)
    with pytest.raises(RunUnavailable, match="mapped|Mapped"):
        cap.remote_results_owner(stores, "FF_AC", MAPPED, admission=proof,
                                 context=context, mapping_provider=lambda drive: target)


@pytest.mark.parametrize("field,value", [("host", "another-host"), ("service_account_sid", "S-1-5-21-999")])
def test_N51_second_host_keeps_explicit_mapped_target_validation(tmp_path, monkeypatch, field, value):
    document, binding, context = drive_configuration()
    proof, _ = evidence(monkeypatch, document, binding)
    settings = settings_doc(tmp_path); settings["capacity"]["remote_owner"] = binding
    context[field] = value
    calls = []
    def mapping(drive):
        calls.append(drive)
        return r"\\nas-a\share"
    result = cap.remote_results_owner(SimpleNamespace(document=settings), "FF_AC", MAPPED,
                                     admission=proof, context=context, mapping_provider=mapping)
    assert result["owner_document"] == binding and calls == ["X:"]


def test_positive_mapping_does_not_replace_positive_transport_admission(tmp_path, monkeypatch):
    document, binding, context = drive_configuration()
    proof, _ = evidence(monkeypatch, document, binding)
    monkeypatch.setattr(fs, "valid_results_admission", lambda *args: False)
    with pytest.raises(RunUnavailable, match="positive output admission"):
        cap.remote_results_owner(SimpleNamespace(document={}), "FF_AC", MAPPED,
            admission=proof, context=context,
            mapping_provider=lambda drive: pytest.fail("No mapping query after transport refusal"))


def test_query_failure_refuses_without_writing(tmp_path, monkeypatch):
    document, binding, context = drive_configuration()
    proof, _ = evidence(monkeypatch, document, binding)
    settings = settings_doc(tmp_path); settings["capacity"]["remote_owner"] = binding
    before = list(tmp_path.rglob("*"))

    def unavailable(drive):
        assert drive == "X:"
        raise OSError(2250, "drive not connected")

    with pytest.raises(RunUnavailable, match="unavailable"):
        cap.remote_results_owner(SimpleNamespace(document=settings), "FF_AC", MAPPED,
                                 admission=proof, context=context, mapping_provider=unavailable)
    assert list(tmp_path.rglob("*")) == before


def mapped_ledger(tmp_path, monkeypatch):
    local, stores, volume = ledger_fixture(tmp_path, monkeypatch)
    document, binding, _ = drive_configuration()
    evidence(monkeypatch, document, binding)
    stores.document["capacity"]["remote_owner"] = binding
    mapping = dict(target=r"\\nas-a\share")
    calls = []

    def query(drive):
        calls.append(drive)
        return mapping["target"]

    ledger = cap.CapacityLedger(stores, "FF_AC", MAPPED,
        volume_provider=lambda root: dict(volume), context_provider=local.context_provider,
        mapping_provider=query)
    return ledger, stores, volume, document, mapping, calls


def test_fresh_volume_rereads_exact_mapping_without_forking_static_pool(tmp_path, monkeypatch):
    ledger, _, volume, _, mapping, calls = mapped_ledger(tmp_path, monkeypatch)
    scope, key = deepcopy(ledger.remote_scope), ledger.key
    assert calls == ["X:"]
    first = ledger.fresh_volume()
    assert calls == ["X:", "X:"] and first["free_bytes"] == volume["free_bytes"]
    assert first["mapped_drive_observation"]["target_unc"] == mapping["target"]
    assert first["mapped_drive_observation"]["provider"] == "injected mapping provider"
    mapping["target"] = r"\\nas-b\share"  # Separately approved, but not X:'s approved target.
    with pytest.raises(RunUnavailable, match="changed its exact"):
        ledger.fresh_volume()
    assert calls == ["X:", "X:", "X:"]
    assert ledger.remote_scope == scope and ledger.key == key


def test_N51_fresh_volume_keeps_pinned_document_and_mapping_without_host_exclusion(tmp_path, monkeypatch):
    ledger, _, _, document, _, calls = mapped_ledger(tmp_path, monkeypatch)
    original_context = ledger.context_provider
    ledger.context_provider = lambda: dict(original_context(), host="changed-host")
    ledger.fresh_volume()
    assert calls == ["X:", "X:"]
    document["owner"]["machine_id"] = "changed"
    with pytest.raises(RunUnavailable, match="missing or changed"):
        ledger.fresh_volume()
    assert calls == ["X:", "X:"]


def test_unc_and_drive_use_one_capacity_reservation_pool(tmp_path, monkeypatch):
    mapped, stores, volume, _, _, calls = mapped_ledger(tmp_path, monkeypatch)
    unc = cap.CapacityLedger(stores, "FF_AC", ROOT,
        volume_provider=lambda root: dict(volume, volume_id="different-client-volume-id"),
        context_provider=mapped.context_provider,
        mapping_provider=lambda drive: pytest.fail("UNC branch must not query drives"))
    assert mapped.remote_scope == unc.remote_scope
    assert mapped.key == unc.key and mapped.root == unc.root and mapped.binding_path == unc.binding_path
    with mapped.locked():
        mapped.bind(tmp_path / "first-job", claim_fixture(), owner(), estimate())
    with unc.locked():
        assert unc.remaining() == 390
        unc.bind(tmp_path / "other-job", claim_fixture(job="two", token="2" * 32), owner(124), estimate())
    assert mapped.remaining() == 780 and calls == ["X:"]


def test_actual_preclaim_availability_refuses_remap_and_releases_local_lock(tmp_path, monkeypatch):
    ledger, stores, _, _, mapping, _ = mapped_ledger(tmp_path, monkeypatch)
    mapping["target"] = r"\\nas-b\share"
    runtime = SimpleNamespace(stores=stores, job=dict(series_id="FF_AC"),
        capacity_refusals={}, availability=lambda task: (), capacities={})
    check = cap.ClaimCapacity(runtime, owner()); check.ledger = ledger
    # The result-only remap test enters no real cache profile or namespace.
    class CacheBoundary:
        def __init__(self):
            self.prepared = 0
            self.calls = []
        def prepare(self, task, value):
            assert task["run"] == "ac52"
            assert value == estimate()
            self.prepared += 1
            self.calls.append("prepare")
        def acquire(self):
            self.calls.append("acquire")
        def check(self, *args):
            self.calls.append("check")
            return dict(TEST_ONLY_SEPARATE_CACHE_BOUNDARY=True)
        def unlock(self):
            self.calls.append("unlock")
    check.cache = CacheBoundary()
    monkeypatch.setattr(cap, "configuration_for_task", lambda *args: {})
    monkeypatch.setattr(cap, "load_estimate", lambda *args, **kwargs: estimate())
    with pytest.raises(RunUnavailable, match="changed its exact"):
        check.availability(claim_fixture().task)
    assert check.lock is None and runtime.capacity_refusals
    assert check.cache.prepared == 1
    assert not list(tmp_path.rglob("reserved.json"))


class Query:
    def __init__(self, target, status=0):
        self.target, self.status, self.calls = target, status, []

    def __call__(self, drive, buffer, length):
        self.calls.append((drive, len(buffer)))
        if self.status == 0:
            buffer.value = self.target
            length._obj.value = len(self.target) + 1
        return self.status


def test_WNet_query_is_one_bounded_exact_drive_read_without_connection_calls(monkeypatch):
    query = Query(r"\\nas-a\share")
    libraries = []

    def library(name, **kwargs):
        libraries.append((name, kwargs))
        return SimpleNamespace(WNetGetConnectionW=query)

    monkeypatch.setattr(ctypes, "WinDLL", library, raising=False)
    assert fs.windows_mapped_drive_target("X:") == r"\\nas-a\share"
    assert query.calls == [("X:", 32768)]
    assert libraries == [("mpr", dict(use_last_error=True))]


@pytest.mark.parametrize("status,target", [(2250, ""), (234, ""), (0, "")])
def test_WNet_unmapped_buffer_or_empty_target_refuses(monkeypatch, status, target):
    query = Query(target, status)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs:
                        SimpleNamespace(WNetGetConnectionW=query), raising=False)
    with pytest.raises(OSError): fs.windows_mapped_drive_target("X:")
    assert query.calls == [("X:", 32768)]


@pytest.mark.parametrize("drive", ["x:", "X:\\", "XY:", "X", None])
def test_WNet_never_queries_an_implicit_or_noncanonical_drive(monkeypatch, drive):
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs:
                        pytest.fail("Invalid drive must not load an OS provider"), raising=False)
    with pytest.raises(ValueError): fs.windows_mapped_drive_target(drive)
