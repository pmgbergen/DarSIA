"""Memory-only default classification; no DarSIA import or host config read."""
import json
from types import SimpleNamespace

import pytest

import darsia_cache.bootstrap as bootstrap
from darsia_cache.policy import CachePolicyError


LEXICAL = "C:/TESTONLY_VIRTUAL/Local/DarSIA/stores.toml"
PHYSICAL = "C:/TESTONLY_MSIX/LocalCache/Local/DarSIA/stores.toml"
LEGACY = b'machine="TESTONLY"\n[stores.FF_AC]\nresults="TESTONLY_NOT_OPENED"\n'


class MemoryPath_TESTONLY:
    def __init__(self, value, state):
        self.value, self.state = str(value), state

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return isinstance(other, MemoryPath_TESTONLY) and self.value == other.value

    def expanduser(self):
        return self

    def is_absolute(self):
        return self.value.startswith(("C:/", "//"))

    def resolve(self, *, strict):
        assert strict is True
        self.state.resolves.append(self.value)
        if self.state.missing or (self.state.alias_missing and len(self.state.resolves) > 1):
            raise FileNotFoundError("TESTONLY missing default")
        target = self.state.retarget if self.state.retarget and len(self.state.resolves) > 1 else self.state.physical
        return MemoryPath_TESTONLY(target, self.state)

    def stat(self):
        assert self.value == self.state.physical or self.value in self.state.additional_documents
        self.state.stats += 1
        if self.state.vanished and self.state.stats > 1:
            raise FileNotFoundError("TESTONLY vanished during read")
        raw = self.state.raw if self.value == self.state.physical else self.state.additional_documents[self.value]
        fields = dict(st_dev=123, st_ino=456,
            st_mode=0o100644 if self.state.regular else 0o040755, st_nlink=1,
            st_size=len(raw) if self.state.declared_size is None else self.state.declared_size,
            st_mtime_ns=self.state.stats if self.state.changed else 1,
            st_ctime_ns=2, st_birthtime_ns=3, st_file_attributes=0x20, st_reparse_tag=0,
            st_atime=4.0, st_atime_ns=4_000_000_000)
        if self.state.stats > 1:
            fields.update(self.state.after_changes)
        return SimpleNamespace(**fields)

    def open(self, mode):
        assert mode == "rb"
        assert self.value == self.state.physical or self.value in self.state.additional_documents
        self.state.opens.append(self.value)
        state = self.state
        raw = state.raw if self.value == state.physical else state.additional_documents[self.value]
        class Stream_TESTONLY:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                state.closes += 1
            def read(self, maximum):
                assert maximum == 1024**2 + 1
                state.read_limits.append(maximum)
                return raw[:maximum]
        return Stream_TESTONLY()


@pytest.fixture
def default_world(monkeypatch):
    state = SimpleNamespace(raw=LEGACY, physical=PHYSICAL, path=LEXICAL,
        declared_size=None, changed=False, regular=True, missing=False, vanished=False,
        retarget=None, alias_missing=False, additional_documents={}, after_changes={},
        resolves=[], stats=0, opens=[], closes=0, read_limits=[], strict_calls=[],
        worker=False, direct=None)
    environment = {}
    monkeypatch.setattr(bootstrap, "_LEASE", None)
    monkeypatch.setattr(bootstrap, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(bootstrap, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(bootstrap, "Path", lambda value: MemoryPath_TESTONLY(value, state))
    monkeypatch.setattr(bootstrap, "_default_store_map_path", lambda: state.path)
    monkeypatch.setattr(bootstrap, "_worker_arguments", lambda: ["TESTONLY_WORKER"] if state.worker else None)
    monkeypatch.setattr(bootstrap, "_stores_argument", lambda: state.direct)
    def strict_original_path_TESTONLY(path, package_file):
        state.strict_calls.append(str(path))
        assert str(path) == LEXICAL and package_file == "TESTONLY_PACKAGE"
        raise CachePolicyError("TESTONLY_ORIGINAL_MANAGED_ALIAS_REFUSAL")
    monkeypatch.setattr(bootstrap, "_explicit_document", strict_original_path_TESTONLY)
    monkeypatch.setattr(bootstrap, "_exact", lambda *args: {"policy": {"path": LEXICAL}})
    yield SimpleNamespace(state=state, environment=environment)
    assert state.closes == len(state.opens)


def test_unconfigured_virtualized_default_returns_unmanaged(default_world):
    world = default_world
    assert bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE") is None
    assert world.state.resolves == [LEXICAL, LEXICAL]
    assert world.state.opens == [PHYSICAL] and world.state.strict_calls == []
    assert world.state.read_limits == [1024**2 + 1]
    assert bootstrap._LEASE is None and world.environment == {}


@pytest.mark.parametrize("raw", [b"cache={}\n", b"cache_bootstrap={}\n",
    b"[cache]\nroot='TESTONLY'\n", b"[cache_bootstrap]\n"])
def test_any_cache_key_even_empty_uses_original_lexical_strict_guard(default_world, raw):
    world = default_world
    world.state.raw = raw
    with pytest.raises(CachePolicyError, match="ORIGINAL_MANAGED_ALIAS_REFUSAL"):
        bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    assert world.state.opens == [PHYSICAL]
    assert world.state.strict_calls == [LEXICAL]


@pytest.mark.parametrize("authority", ["policy", "receipt", "direct", "worker", "expected_stores"])
def test_explicit_or_worker_binding_never_takes_classifier_path(default_world, authority):
    world = default_world
    if authority == "policy":
        world.environment[bootstrap.POLICY_ENV] = LEXICAL
    elif authority == "receipt":
        world.environment[bootstrap.RECEIPT_ENV] = json.dumps({"TESTONLY_BOUND_RECEIPT": True})
    elif authority == "direct":
        world.state.worker, world.state.direct = True, LEXICAL
    elif authority == "worker":
        world.state.worker = True
    else:
        world.environment[bootstrap.EXPECTED_STORES_ENV] = json.dumps({"TESTONLY_EXPECTED_STORES": True})
    with pytest.raises(CachePolicyError, match="ORIGINAL_MANAGED_ALIAS_REFUSAL"):
        bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    assert world.state.strict_calls == [LEXICAL]
    assert world.state.resolves == world.state.opens == []


@pytest.mark.parametrize("case", ["changed", "vanished", "initial_large", "grew", "malformed", "empty", "directory"])
def test_changed_large_or_malformed_default_is_never_silently_unmanaged(default_world, case):
    world = default_world
    if case == "changed":
        world.state.changed = True
    elif case == "vanished":
        world.state.vanished = True
    elif case == "initial_large":
        world.state.declared_size = 1024**2 + 1
    elif case == "grew":
        world.state.declared_size, world.state.raw = 1, b"x" * (1024**2 + 1)
    elif case == "malformed":
        world.state.raw = b"cache = ["
    elif case == "empty":
        world.state.raw = b""
    else:
        world.state.regular = False
    expected = bootstrap.tomllib.TOMLDecodeError if case == "malformed" else CachePolicyError
    with pytest.raises(expected):
        bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    assert world.state.strict_calls == [] and bootstrap._LEASE is None


@pytest.mark.parametrize("case", ["relative", "network_alias"])
def test_default_classifier_keeps_local_bounded_metadata_boundary(default_world, case):
    world = default_world
    if case == "relative":
        world.state.path = "TESTONLY_RELATIVE/stores.toml"
    else:
        world.state.physical = "//TESTONLY_NETWORK/stores.toml"
    with pytest.raises(CachePolicyError):
        bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    assert world.state.opens == [] and world.state.strict_calls == []


def test_missing_ordinary_default_remains_unmanaged(default_world):
    world = default_world
    world.state.missing = True
    assert bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE") is None
    assert world.state.opens == world.state.strict_calls == []


def test_empty_explicit_policy_still_refuses_before_default_classification(default_world):
    world = default_world
    world.environment[bootstrap.POLICY_ENV] = ""
    with pytest.raises(CachePolicyError, match="must be nonempty"):
        bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    assert world.state.resolves == world.state.opens == world.state.strict_calls == []


@pytest.mark.parametrize("case", ["retarget_to_managed", "alias_disappears"])
def test_final_lexical_alias_change_refuses_unmanaged_classification(default_world, case):
    world = default_world
    # First read is the old legacy document. The second lexical resolution
    # changes to a different managed target, which must never be opened here.
    if case == "retarget_to_managed":
        world.state.retarget = "C:/TESTONLY_NEW_MANAGED/stores.toml"
        world.state.additional_documents[world.state.retarget] = b"cache={}\n"
    else:
        world.state.alias_missing = True
    with pytest.raises(CachePolicyError, match="retargeted|disappeared"):
        bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    assert world.state.resolves == [LEXICAL, LEXICAL]
    assert world.state.opens == [PHYSICAL]
    assert world.state.strict_calls == [] and bootstrap._LEASE is None


def test_access_time_only_change_during_read_remains_unmanaged(default_world):
    world = default_world
    world.state.after_changes = {"st_atime": 9.0, "st_atime_ns": 9_000_000_000}
    assert bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE") is None
    assert world.state.stats == 2 and world.state.resolves == [LEXICAL, LEXICAL]
    assert world.state.opens == [PHYSICAL] and world.state.strict_calls == []
    assert bootstrap._LEASE is None and world.environment == {}


@pytest.mark.parametrize("field,replacement", [
    ("st_dev", 124), ("st_ino", 457), ("st_mode", 0o040755), ("st_nlink", 2),
    ("st_size", 1), ("st_mtime_ns", 99), ("st_ctime_ns", 99),
    ("st_birthtime_ns", 99), ("st_file_attributes", 0x420),
    ("st_reparse_tag", 0xA000000C),
])
def test_each_stable_identity_version_or_reparse_link_change_refuses(default_world, field, replacement):
    world = default_world
    world.state.after_changes = {field: replacement}
    with pytest.raises(CachePolicyError, match="changed during unmanaged classification"):
        bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    assert world.state.stats == 2 and world.state.resolves == [LEXICAL, LEXICAL]
    assert world.state.opens == [PHYSICAL] and world.state.strict_calls == []
    assert bootstrap._LEASE is None and world.environment == {}
