"""Memory-only pre-import application, never a host cache admission/profile."""
from types import SimpleNamespace
from copy import deepcopy
import hashlib
import json
import tempfile

import pytest

import darsia_cache.bootstrap as bootstrap
from darsia_cache.environment import derive_cache_environment, WRITER_ENVIRONMENT_KEYS
from darsia_cache.policy import CachePolicyError


def test_bootstrap_applies_all_child_writer_keys_and_preserves_parent(monkeypatch):
    token = "b" * 32
    source = dict(path="C:/TESTONLY/stores.toml", sha256="a" * 64, size_bytes=1)
    document = {"cache": dict(root="C:/TESTONLY_CACHE", quota_bytes=1,
                              profiles={"jobs": {"TESTONLY": "not a real profile"}})}
    parent = dict(CUPY_CACHE_DIR="old-cupy", TEMP="old-temp", HOME="durable-home",
                  LOCALAPPDATA="durable-state", OMP_NUM_THREADS="7",
                  UNRELATED_APPLICATION_KEY="preserved")
    original_parent = dict(parent)
    child = dict(parent)
    calls = []

    class Lease_TESTONLY:
        def __init__(self, actual_document, actual_source, package_file, *,
                     entry_kind, role_identifier):
            assert actual_document is document and actual_source is source
            assert package_file == "TESTONLY_PACKAGE"
            assert entry_kind == "worker" and role_identifier is None
            self.purpose = "jobs"
            self.reservation = dict(token=token)
            self.root = "C:/TESTONLY_CACHE/jobs/" + token
            self.receipt = dict(TESTONLY="not a real host/service receipt")
        def start(self):
            calls.append("start")
        def exit_intent(self):
            raise AssertionError("TESTONLY atexit registration must not execute")
        def close_pins(self):
            calls.append("close_pins")

    child_sys = SimpleNamespace(modules={}, pycache_prefix="old-prefix")
    monkeypatch.setattr(bootstrap, "_LEASE", None)
    monkeypatch.setattr(bootstrap, "os", SimpleNamespace(environ=child))
    monkeypatch.setattr(bootstrap, "sys", child_sys)
    monkeypatch.setattr(bootstrap, "_worker_arguments", lambda: ["TESTONLY"])
    monkeypatch.setattr(bootstrap, "_stores_argument", lambda: source["path"])
    monkeypatch.setattr(bootstrap, "_explicit_document", lambda *args: (document, source))
    monkeypatch.setattr(bootstrap, "_entry_kind", lambda: "worker")
    monkeypatch.setattr(bootstrap, "ImportLease", Lease_TESTONLY)
    monkeypatch.setattr(bootstrap, "atexit", SimpleNamespace(register=lambda fn: calls.append(fn)))
    monkeypatch.setattr(tempfile, "tempdir", "old-tempfile-choice")

    lease = bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    expected = derive_cache_environment(document, purpose="jobs", token=token,
        scope_root=lease.root, parent_environment=parent)
    assert all(child[key] == expected[key] for key in WRITER_ENVIRONMENT_KEYS)
    assert child["CUPY_CACHE_DIR"].endswith(token + "\\cupy")
    assert child_sys.pycache_prefix == expected["PYTHONPYCACHEPREFIX"]
    assert json.loads(child[bootstrap.RECEIPT_ENV]) == lease.receipt
    assert all(child[key] == parent[key] for key in
               ("HOME", "LOCALAPPDATA", "OMP_NUM_THREADS", "UNRELATED_APPLICATION_KEY"))
    assert parent == original_parent
    assert tempfile.tempdir is None
    assert calls == ["start", lease.exit_intent]
    assert bootstrap.current_import_lease() is lease


def test_first_only_redirects_all_inherited_cache_application_paths_inside_owned_scope():
    from pathlib import PureWindowsPath
    from darsia_cache.environment import first_qualification_environment, FIRST_WRITER_ENVIRONMENT_KEYS
    token="b"*32;root="Q:/TESTONLY_CACHE/jobs/"+token
    document={"cache":dict(root="Q:/TESTONLY_CACHE",quota_bytes=1,profiles={"jobs":{}})}
    parent=dict(HOME="C:/old-home",LOCALAPPDATA="C:/old-local",XDG_CACHE_HOME="C:/old-xdg",
                USERPROFILE="C:/old-user",APPDATA="C:/old-app",UNRELATED="preserved")
    original=dict(parent)
    ordinary=derive_cache_environment(document,purpose="jobs",token=token,scope_root=root,parent_environment=parent)
    assert ordinary["HOME"]==parent["HOME"] and ordinary["LOCALAPPDATA"]==parent["LOCALAPPDATA"]
    actual=first_qualification_environment(ordinary,scope_root=root)
    assert parent==original and actual["UNRELATED"]=="preserved"
    for key in FIRST_WRITER_ENVIRONMENT_KEYS:
        if key!="NUMBA_CACHE_LOCATOR_CLASSES":
            assert PureWindowsPath(actual[key]).is_relative_to(PureWindowsPath(root))
    assert actual["NUMBA_CACHE_LOCATOR_CLASSES"]=="UserProvidedCacheLocator"



def explicit_stores_path_TESTONLY(monkeypatch,*,changed=None,declared_size=None,raw=None):
    data=b'[machine]\nid="TESTONLY"\n' if raw is None else raw
    fields=dict(st_dev=123,st_ino=456,st_size=len(data),st_mtime_ns=789,
        st_atime=1.0,st_atime_ns=1_000_000_000)
    if declared_size is not None:fields['st_size']=declared_size
    before=SimpleNamespace(**fields);after=SimpleNamespace(**dict(fields,**(changed or {})))
    states=iter([before,after]);reads=[]
    class Path_TESTONLY:
        def stat(self):return next(states)
        def read_bytes(self):reads.append('read');return data
        def __str__(self):return 'C:/TESTONLY/stores.toml'
    path=Path_TESTONLY()
    monkeypatch.setattr(bootstrap,'_outside',lambda selected,package:path)
    return data,reads


def test_explicit_stores_read_ignores_access_time_only_with_exact_raw_descriptor(monkeypatch):
    import hashlib
    raw,reads=explicit_stores_path_TESTONLY(monkeypatch,changed=dict(
        st_atime=9.0,st_atime_ns=9_000_000_000))
    document,source=bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')
    assert document=={'machine':{'id':'TESTONLY'}} and reads==['read']
    assert source==dict(path='C:/TESTONLY/stores.toml',sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def test_explicit_stores_read_still_refuses_every_identity_or_content_version_change(monkeypatch):
    import pytest
    from darsia_cache.policy import CachePolicyError
    for field,value in [('st_dev',124),('st_ino',457),('st_size',999),('st_mtime_ns',790)]:
        explicit_stores_path_TESTONLY(monkeypatch,changed={field:value})
        with pytest.raises(CachePolicyError,match='changed while reading'):
            bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')
    raw,_=explicit_stores_path_TESTONLY(monkeypatch)
    explicit_stores_path_TESTONLY(monkeypatch,declared_size=len(raw)+1)
    with pytest.raises(CachePolicyError,match='changed while reading'):
        bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')


def test_explicit_stores_read_preserves_positive_one_mib_before_read_bound(monkeypatch):
    import pytest
    from darsia_cache.policy import CachePolicyError
    for size in [0,1024**2+1]:
        _,reads=explicit_stores_path_TESTONLY(monkeypatch,declared_size=size)
        with pytest.raises(CachePolicyError,match='one MiB'):
            bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')
        assert reads==[]


def _historical_authority_case(tmp_path):
    """Real temporary metadata; no patched SHA/parser/path/version boundary."""
    def write(name, value, *, raw=None):
        path = tmp_path / name
        raw = json.dumps(value, indent=1).encode() if raw is None else raw
        path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    old = [{"id": "N43", "decided_by": "owner"},
        {"id": "Q123-answer-option1-append-only-closeout-and-new-empty-exclusive-scope-retained-bytes-counted",
            "decided_by": "coordinator"}]
    old_raw = json.dumps(old, indent=1).encode()
    live_answers = write("coordinator_answers.json", old, raw=old_raw)
    historical_snapshot = write("historical_answers.json", old, raw=old_raw)
    current = old + [{"id": "Q134-answer-option1-historical-authority-resolver-and-immutable-authority-snapshots",
        "decided_by": "coordinator"}]
    live_answers_now = write("coordinator_answers.json", current)
    current_answers = write("current_answers_snapshot.json", current)
    live_status = write("coordinator_status.json", {"seq": 263, "state": "current"})
    current_status = write("current_status_snapshot.json", {"seq": 263, "state": "current"})
    request = write("preserved_original_request.json", {"answers_source": live_answers})
    value = dict(format="darsia-cache-historical-authority-resolution-v1", authority_ids=["Q134"],
        current_authorities=dict(coordinator_answers=dict(source=live_answers_now, snapshot=current_answers),
            coordinator_status=dict(source=live_status, snapshot=current_status)),
        preserved_requests=[request], historical_authorities=[dict(original=live_answers,
            snapshot=historical_snapshot, entry_count=len(old))])
    binding = write("root_bound_history_map.json", value)
    return SimpleNamespace(write=write, old=old, current=current, old_pin=live_answers,
        current_pin=current_answers, request=request, value=value, binding=binding,
        package=str(tmp_path/"not_created_package"/"darsia"/"__init__.py"))


def test_q134_exact_history_does_not_change_default_reader_or_old_or_live_bytes(tmp_path):
    case = _historical_authority_case(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(CachePolicyError, match="changed before read"):
        bootstrap._exact_authority_list(case.old_pin, case.package)
    reader = bootstrap._HistoricalAuthorityReader(case.binding, case.package)
    original = reader.for_original(case.request)
    assert original(case.old_pin) == case.old
    assert reader.current_authorities(case.current_pin) == case.current
    with pytest.raises(CachePolicyError, match="immutable full-current"):
        reader.current_authorities(case.old_pin)
    with pytest.raises(CachePolicyError, match="changed before read"):
        bootstrap._exact_authority_list(case.old_pin, case.package)
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize("change", ["missing_status", "status_snapshot_size", "answers_live_alias",
    "original_sha", "original_size", "original_path", "snapshot_prefix", "entry_count",
    "duplicate_request", "missing_history", "no_q134", "live_prefix"])
def test_q134_wrong_map_raw_pin_path_prefix_or_current_pair_fails_closed(tmp_path, change):
    case = _historical_authority_case(tmp_path)
    value = deepcopy(case.value)
    history = value["historical_authorities"][0]
    if change == "missing_status":
        del value["current_authorities"]["coordinator_status"]
    elif change == "status_snapshot_size":
        value["current_authorities"]["coordinator_status"]["snapshot"]["size_bytes"] += 1
    elif change == "answers_live_alias":
        value["current_authorities"]["coordinator_answers"]["snapshot"] = value["current_authorities"]["coordinator_answers"]["source"]
    elif change == "original_sha":
        history["original"]["sha256"] = "f" * 64
    elif change == "original_size":
        history["original"]["size_bytes"] += 1
    elif change == "original_path":
        history["original"]["path"] = str(tmp_path/"other_live_answers.json")
    elif change == "snapshot_prefix":
        history["snapshot"] = case.write("wrong_historical_snapshot.json", [{"id": "changed"}] + case.old[1:])
        history["original"].update({key:history["snapshot"][key] for key in ("sha256", "size_bytes")})
    elif change == "entry_count":
        history["entry_count"] = True
    elif change == "duplicate_request":
        value["preserved_requests"].append(deepcopy(case.request))
    elif change == "missing_history":
        value["historical_authorities"] = []
    elif change == "no_q134":
        snapshot = case.write("wrong_current_answers.json", case.old + [{"id": "other"}])
        value["current_authorities"]["coordinator_answers"]["snapshot"] = snapshot
        value["current_authorities"]["coordinator_answers"]["source"].update(
            {key:snapshot[key] for key in ("sha256", "size_bytes")})
    elif change == "live_prefix":
        case.write("coordinator_answers.json", [{"id": "removed_original"}] + case.current[1:])
    binding = case.write("wrong_map.json", value)
    with pytest.raises(CachePolicyError):
        bootstrap._HistoricalAuthorityReader(binding, case.package)


def test_q134_only_named_original_exact_descriptor_can_read_history(tmp_path):
    case = _historical_authority_case(tmp_path)
    reader = bootstrap._HistoricalAuthorityReader(case.binding, case.package)
    for unknown in (None, {**case.request, "sha256": "e" * 64},
                    {**case.request, "path": str(tmp_path/"new_request.json")}):
        with pytest.raises(CachePolicyError):
            reader.for_original(unknown)
    original = reader.for_original(case.request)
    with pytest.raises(CachePolicyError, match="descriptor differs"):
        original({**case.old_pin, "sha256": "d" * 64})
    historical_path = tmp_path/"historical_answers.json"
    historical_path.write_bytes(historical_path.read_bytes() + b" ")
    with pytest.raises(CachePolicyError):
        original(case.old_pin)


def test_q134_current_snapshot_survives_live_appends_and_status_updates_without_adopting_them(tmp_path):
    case = _historical_authority_case(tmp_path)
    case.write("coordinator_answers.json", case.current + [{"id": "later_answer", "decided_by": "coordinator"}])
    case.write("coordinator_status.json", {"seq": 264, "state": "updated later"})
    reader = bootstrap._HistoricalAuthorityReader(case.binding, case.package)
    assert reader.current_authorities(case.current_pin) == case.current
    assert reader.for_original(case.request)(case.old_pin) == case.old
    assert reader.current["coordinator_status"][2] == {"seq": 263, "state": "current"}
    assert all(row.get("id") != "later_answer" for row in reader.current_authorities(case.current_pin))


def test_q134_metadata_once_uses_explicit_original_reader_and_preserves_copy_guards(tmp_path):
    from darsia_cache.service_context import _MetadataOnce
    case = _historical_authority_case(tmp_path)
    reader = bootstrap._HistoricalAuthorityReader(case.binding, case.package)
    once = _MetadataOnce(case.package, authority_reader=reader.for_original(case.request))
    value = once.read_authorities(case.old_pin)
    value[0]["id"] = "changed copy"
    assert once.read_authorities(case.old_pin) == case.old
    with pytest.raises(CachePolicyError, match="descriptor changed"):
        once.read_authorities({**case.old_pin, "sha256": "a" * 64})


def test_q134_live_append_after_stable_capture_is_not_reopened_or_adopted(monkeypatch,tmp_path):
    from pathlib import Path
    case = _historical_authority_case(tmp_path)
    reader = bootstrap._HistoricalAuthorityReader(case.binding, case.package)
    live = Path(case.value["current_authorities"]["coordinator_answers"]["source"]["path"])
    original_open, original_decode = Path.open, bootstrap._authority_records
    reads=[]
    def counted_open(path, mode="r", *args, **kwargs):
        if path == live and "r" in mode:
            reads.append(str(path))
        return original_open(path, mode, *args, **kwargs)
    def append_before_parse(raw):
        case.write("coordinator_answers.json", case.current + [{"id": "appended_after_stable_capture"}])
        return original_decode(raw)
    monkeypatch.setattr(Path,"open",counted_open)
    monkeypatch.setattr(bootstrap,"_authority_records",append_before_parse)
    reader._verify_actual_prefix(case.current)
    assert reads == [str(live)]
    monkeypatch.setattr(bootstrap,"_authority_records",original_decode)
    assert reader.current_authorities(case.current_pin) == case.current


@pytest.mark.parametrize("raw", [b'[{"id":"a","id":"b"}]', b'[{"x":NaN}]', b'[null]', b'{}'])
def test_q134_captured_prefix_decoder_keeps_duplicate_nonfinite_and_list_guards(raw):
    with pytest.raises(ValueError):
        bootstrap._authority_records(raw)
