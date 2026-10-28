"""TESTONLY: R3, the RAW39 reference, is admitted only by its exact release entry.

Like N43, R3 is pinned in REFERENCE_RELEASES: its answer records with their
deciders and its accepted plan's SHA-256. After the RAW drift fix (Q159, N152)
the plan was re-pinned and N154 names it; the entry also pins the four
retained font files R3 may find in the cache, and admission counts exactly
those verified bytes instead of an empty cache. The entries of R3's stopped
attempts stay so their requests can be rebuilt, but they cannot start. Any
other authority set, record, plan or inventory is refused. Synthetic data only.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES
from .test_n43_reference_administration import Ledger_TESTONLY, administration_TESTONLY, pin_TESTONLY

OLD_PLAN = "3ea1caa8ff432f5c17d2904f7f43080e23aeac039a447d8667a1586749e0426e"
R3_PLAN = "a88d0950617b445a6ec93fae826d2b359a930857cc97bfe24d58b5bc746e2301"
R3_GO = ("N134-OWNER-r3-inputs-into-a-new-ff-ac-folder-reference-build-once-fresh-cache-released-"
         "after-the-code-change")
OLD_PLAN_RECORD = "N136-COORDINATOR-r3-plan-3ea1caa8-entitlement-65728995330B-pinned-for-the-exact-code-entry"
R3_RETAINED = ("N139-OWNER-r3-starts-with-the-verified-117590-byte-retained-font-file-counted-file-and-old-"
               "approval-untouched")
R3_SECOND_FONT = "N142-OWNER-r3-close-failed-service-95c-count-both-font-files-235180-bytes-one-new-start"
R3_THIRD_FONT = "N145-OWNER-yes-q157-paired-model-staging-close-5229-count-three-font-files-one-new-start"
R3_FOURTH_FONT = "N152-OWNER-yes-raw-drift-fix-first-close-7fdd-count-four-fonts-r3-on-fixed-code-standing-yes"
R3_PLAN_RECORD = "N154-COORDINATOR-r3-plan-a88d0950-drift-fix-repins-entitlement-65728995330B-four-fonts-pinned"
FONT_BYTES = 117590
R3_INVENTORY = "5ab92bb37ce086192baeba8ffe0b86ef0a0777d3af22d1bf9f1bda89a6c74b51"
R3_BYTES = 4 * FONT_BYTES
R3_AUTHORITIES = ("N134", "N139", "N142", "N145", "N152", "N154")


def owner_answer(record):
    return dict(id=record, decided_by="owner")


def agent_answer(record):
    return dict(id=record, decided_by="coordinator")


R3_ANSWERS = [owner_answer(R3_GO), owner_answer(R3_RETAINED), owner_answer(R3_SECOND_FONT), owner_answer(R3_THIRD_FONT), owner_answer(R3_FOURTH_FONT),
              agent_answer(R3_PLAN_RECORD)]
# Earlier attempts' entries: authority ids -> (pinned inventory, retained files, plan, records).
EARLIER = {
    ("N134", "N136", "N139"): (
        "be5c8bf204e6a5021f79bd10125542bcfd87adaf80c2401e66cb7ce9463852a2", 1, OLD_PLAN,
        [owner_answer(R3_GO), agent_answer(OLD_PLAN_RECORD), owner_answer(R3_RETAINED)]),
    ("N134", "N136", "N139", "N142"): (
        "668f556a8d20c7d82484a9449521e46522b4d22b1bf2a0290e9f77d323c90a35", 2, OLD_PLAN,
        [owner_answer(R3_GO), agent_answer(OLD_PLAN_RECORD), owner_answer(R3_RETAINED), owner_answer(R3_SECOND_FONT)]),
    ("N134", "N136", "N139", "N142", "N145"): (
        "96986702430169ac39fe43ea62195db66d36229cd287ddac6fe57f5e46890fce", 3, OLD_PLAN,
        [owner_answer(R3_GO), agent_answer(OLD_PLAN_RECORD), owner_answer(R3_RETAINED), owner_answer(R3_SECOND_FONT), owner_answer(R3_THIRD_FONT)]),
}


def r3_TESTONLY(answers=None, authority_ids=R3_AUTHORITIES, plan=R3_PLAN, inventory=None,
                inventory_sha256=R3_INVENTORY, logical_bytes=R3_BYTES):
    state, value, _read = administration_TESTONLY()
    cache = next(row["path"] for row in value["owned_roots"] if row["role"] == "cache")
    inventory = inventory if inventory is not None else dict(format=first.RETAINED_INVENTORY, root=cache,
        no_cleanup=True, files=[{"TESTONLY": "retained font files"}], logical_bytes=logical_bytes)
    value.update(authority_ids=list(authority_ids),
                 accepted_plan=dict(path="C:/TESTONLY/R3_accepted_plan.json", sha256=plan, size_bytes=1),
                 retained_cache_inventory=dict(path="C:/TESTONLY/retained_inventory.json",
                                               sha256=inventory_sha256, size_bytes=1127))
    answers = deepcopy(R3_ANSWERS) if answers is None else answers

    def read(binding):
        if binding == value["answers_source"]:
            return deepcopy(answers)
        if binding == value["retained_cache_inventory"]:
            return deepcopy(inventory)
        return {"TESTONLY": "read exact plan"}

    return state, value, read


def without(record):
    return [row for row in R3_ANSWERS if row["id"] != record]


def decided_by(record, decider):
    return [dict(row, decided_by=decider) if row["id"] == record else dict(row) for row in R3_ANSWERS]


def test_r3_is_admitted_by_its_exact_release_entry():
    state, value, read = r3_TESTONLY()
    actual = first.checked_reference_administration(value, state, read=read)
    assert actual["authority_ids"] == list(R3_AUTHORITIES)
    assert actual["aggregate_entitlement_bytes"] == first.REFERENCE_ENTITLEMENT_BYTES
    assert actual["retained_cache_inventory"]["sha256"] == R3_INVENTORY
    assert actual["accepted_plan"]["sha256"] == R3_PLAN


@pytest.mark.parametrize("kwargs", [
    dict(plan=R3_PLAN[:8] + "0" * 56),  # Only the short prefix matches.
    dict(plan=OLD_PLAN),  # The plan before the drift fix.
    dict(authority_ids=("N134", "N136")),  # The empty-cache R3 entry is gone.
    *[dict(authority_ids=ids) for ids in EARLIER],  # Earlier entries pin fewer files and the old plan.
    dict(authority_ids=R3_AUTHORITIES + ("N999",)),
    dict(authority_ids=tuple(reversed(R3_AUTHORITIES))),  # Order is part of the entry.
    *[dict(answers=without(row["id"])) for row in R3_ANSWERS],
    *[dict(answers=decided_by(row["id"], "coordinator" if row["decided_by"] == "owner" else "owner"))
      for row in R3_ANSWERS],
    dict(answers=[]),
])
def test_r3_refuses_any_other_plan_authority_set_or_record(kwargs):
    state, value, read = r3_TESTONLY(**kwargs)
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


@pytest.mark.parametrize("change", ["missing", "pin", "earlier_pin", "bytes", "float_bytes", "format", "cleanup",
                                    "root"])
def test_r3_requires_exactly_the_pinned_retained_inventory_of_its_cache_root(change):
    pins = dict(pin="0" * 64, earlier_pin=EARLIER[("N134", "N136", "N139", "N142", "N145")][0])
    state, value, read = r3_TESTONLY(inventory_sha256=pins.get(change, R3_INVENTORY))
    inventory = read(value["retained_cache_inventory"])
    edits = dict(bytes=dict(logical_bytes=R3_BYTES + 1), float_bytes=dict(logical_bytes=float(R3_BYTES)),
                 format=dict(format="darsia-cache-retained-file-inventory-v0"), cleanup=dict(no_cleanup=False),
                 root=dict(root="Q:/TESTONLY/another-cache"))
    if change in edits:
        inventory.update(edits[change])
        state, value, read = r3_TESTONLY(inventory=inventory)
    if change == "missing":
        value.pop("retained_cache_inventory")
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


@pytest.mark.parametrize("ids", list(EARLIER))
def test_earlier_attempts_requests_can_still_be_rebuilt_for_their_close(ids):
    sha, files, plan, answers = EARLIER[ids]
    state, value, read = r3_TESTONLY(authority_ids=ids, answers=deepcopy(answers), plan=plan,
                                     inventory_sha256=sha, logical_bytes=files * FONT_BYTES)
    assert first.checked_reference_administration(value, state, read=read)["authority_ids"] == list(ids)


def test_n43_names_no_retained_inventory():
    state, value, read = administration_TESTONLY()
    value["retained_cache_inventory"] = pin_TESTONLY("retained_inventory.json")
    with pytest.raises(CachePolicyError, match="Exact N43 reference administration"):
        first.checked_reference_administration(value, state, read=read)


def test_r3_keeps_the_n43_entitlement():
    state, value, read = r3_TESTONLY()
    value["aggregate_entitlement_bytes"] = first.REFERENCE_ENTITLEMENT_BYTES + 1
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


def test_released_reference_names_only_exact_entries():
    assert first.released_reference(["N43", "Q109", "Q106"]) is first.REFERENCE_RELEASES[("N43", "Q109", "Q106")]
    r3 = first.released_reference(list(R3_AUTHORITIES))
    assert r3["plan_sha256"] == R3_PLAN
    assert r3["retained_cache"] == dict(inventory_sha256=R3_INVENTORY, logical_bytes=R3_BYTES)
    for ids, (sha, files, plan, _answers) in EARLIER.items():
        entry = first.released_reference(list(ids))
        assert entry["retained_cache"] == dict(inventory_sha256=sha, logical_bytes=files * FONT_BYTES)
        assert entry["plan_sha256"] == plan
    assert "retained_cache" not in first.REFERENCE_RELEASES[("N43", "Q109", "Q106")]
    assert first.released_reference(["N134", "N136"]) is None
    assert first.released_reference(R3_AUTHORITIES) is None  # A list, as stored in JSON.
    assert first.released_reference([["N134"], *R3_AUTHORITIES[1:]]) is None
    assert first.released_reference(None) is None


def retained_state_TESTONLY(tmp_path, *, retained=True, listed=4):
    """Four real retained files and an exact inventory of the first `listed`.

    No actual cache is read. With listed < 4 the inventory is an earlier
    attempt's pin while all four files are on disk.
    """
    root = tmp_path / "cache"
    rows, paths = [], []
    for token in ("e" * 32, "f" * 32, "d" * 32, "7" * 32):
        relative = "jobs/" + token + "/matplotlib/fontlist-v390.json"
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True)
        path.write_bytes(b"f" * FONT_BYTES)
        info = path.lstat()
        rows.append(dict(relative_path=relative, dev=info.st_dev, ino=info.st_ino,
                         size_bytes=info.st_size, mtime_ns=info.st_mtime_ns))
        paths.append(path)
    rows = rows[:listed]
    inventory = dict(format=first.RETAINED_INVENTORY, root=str(root), no_cleanup=True, files=rows,
                     logical_bytes=sum(row["size_bytes"] for row in rows))
    state = first.FirstQualification.__new__(first.FirstQualification)
    state.reference_build = True
    state.bound_mode = False
    state.binding = pin_TESTONLY("request.json")
    state.reference_administration = dict(host_policy="main-q87-fixed500")
    state.reference_retained_inventory = None
    if retained:
        state.reference_retained_inventory = inventory
        state._reference_retained_seal = first._digest(inventory)
    ledger = Ledger_TESTONLY()
    for token in ("a" * 32, "b" * 32, "c" * 32, "8" * 32):  # Closed earlier services: 35517, 95c, 5229, 7fdd.
        ledger.documents.update({"cache-r-" + token + ".json": b"{}", "cache-c-" + token + ".json": b"{}"})
    lease = SimpleNamespace(policy=SimpleNamespace(root=str(root)), ledger=ledger)
    return state, lease, paths


def test_r3_counts_exactly_its_verified_retained_files(tmp_path):
    state, lease, _ = retained_state_TESTONLY(tmp_path)
    assert state.initial_cache_bytes(lease) == R3_BYTES


def test_without_a_retained_release_a_cold_start_stays_empty(tmp_path):
    state, lease, _ = retained_state_TESTONLY(tmp_path, retained=False)
    assert state.initial_cache_bytes(lease) == 0
    assert first.FirstQualification.__new__(first.FirstQualification).initial_cache_bytes(lease) == 0


@pytest.mark.parametrize("change", ["grown", "missing", "open_reservation", "unknown_reservation", "memory"])
def test_r3_retained_count_refuses_any_changed_file_open_reservation_or_inventory(tmp_path, change):
    state, lease, paths = retained_state_TESTONLY(tmp_path)
    if change == "grown":
        paths[3].write_bytes(b"f" * (FONT_BYTES + 1))
    elif change == "missing":
        paths[3].unlink()
    elif change == "open_reservation":
        lease.ledger.documents["cache-r-" + "9" * 32 + ".json"] = b"TESTONLY fully written live reservation"
    elif change == "unknown_reservation":
        lease.ledger.documents["cache-r-TESTONLY.json"] = b"{}"
    else:
        state.reference_retained_inventory["logical_bytes"] = 0
    with pytest.raises((CachePolicyError, OSError)):
        state.initial_cache_bytes(lease)


def admission_TESTONLY(monkeypatch, state):
    """Reference admission with only its namespace and length scan replaced."""
    monkeypatch.setattr(state, "_namespace", lambda lease: True, raising=False)
    snapshot = dict(observed_scan_anomalies=[], roots=[dict(role="results", logical_bytes=0)],
                    logical_bytes=R3_BYTES, volumes=[])
    monkeypatch.setattr(first, "reference_scope_snapshot", lambda *args: snapshot)
    monkeypatch.setattr(first, "require_reference_not_stopped", lambda *args: None)


def report_TESTONLY(**changes):
    value = dict(reserve_bytes=PHYSICAL_RESERVE_BYTES, hard_filesystem_quota_verified=True,
                 current_cache_logical_bytes=R3_BYTES, active_cache_remaining_bytes=0)
    value.update(changes)
    return value


@pytest.mark.parametrize("listed", [1, 2, 3])
def test_earlier_attempts_entries_cannot_start_with_four_font_files_on_disk(tmp_path, monkeypatch, listed):
    state, lease, _ = retained_state_TESTONLY(tmp_path, listed=listed)
    admission_TESTONLY(monkeypatch, state)
    assert state.initial_cache_bytes(lease) == listed * FONT_BYTES
    with lease.ledger.locked():
        with pytest.raises(CachePolicyError, match="fresh exclusive cache"):
            state.reference_admission(lease, report=report_TESTONLY(), initial=True)


@pytest.mark.parametrize("retained,changes,admitted", [
    (True, {}, True),
    (True, dict(current_cache_logical_bytes=0), False),
    (True, dict(current_cache_logical_bytes=3 * FONT_BYTES), False),  # Only the first three font files.
    (True, dict(current_cache_logical_bytes=R3_BYTES + 1), False),
    (True, dict(active_cache_remaining_bytes=1), False),
    (False, dict(current_cache_logical_bytes=0), True),  # N43 is unchanged.
    (False, {}, False),
])
def test_reference_admission_counts_exactly_the_released_retained_bytes(tmp_path, monkeypatch, retained,
                                                                         changes, admitted):
    state, lease, _ = retained_state_TESTONLY(tmp_path, retained=retained)
    admission_TESTONLY(monkeypatch, state)
    report = report_TESTONLY(**changes)
    with lease.ledger.locked():
        if admitted:
            assert state.reference_admission(lease, report=report, initial=True)["sufficient"] is True
        else:
            with pytest.raises(CachePolicyError, match="fresh exclusive cache"):
                state.reference_admission(lease, report=report, initial=True)


class ProjectingLedger_TESTONLY(Ledger_TESTONLY):
    def __init__(self, report):
        super().__init__()
        self.report = report

    def project(self, needed):
        assert self.lock_active and needed == 0
        return deepcopy(self.report)


@pytest.mark.parametrize("retained,current,admitted", [
    (True, R3_BYTES, True), (True, 3 * FONT_BYTES, False), (True, 0, False),
    (False, 0, True), (False, R3_BYTES, False)])
def test_consume_admits_r3_with_exactly_its_retained_bytes(tmp_path, monkeypatch, retained, current, admitted):
    state, lease, _ = retained_state_TESTONLY(tmp_path, retained=retained)
    lease.ledger = ProjectingLedger_TESTONLY(report_TESTONLY(current_cache_logical_bytes=current))
    calls = []
    monkeypatch.setattr(first, "_outputs_absent", lambda provider: None)
    monkeypatch.setattr(state, "_namespace", lambda lease: True, raising=False)
    monkeypatch.setattr(state, "_once_key", lambda: "c" * 64, raising=False)
    monkeypatch.setattr(state, "reference_admission", lambda lease, **kwargs: calls.append(kwargs), raising=False)
    state.provider = {"scope": {"TESTONLY": "scope"}}
    state.provider_binding, state.stores_source = pin_TESTONLY("provider.json"), pin_TESTONLY("stores.toml")
    lease.owner = {"TESTONLY": "owner"}
    if admitted:
        state.consume(lease)
        assert state.started is True and [call["initial"] for call in calls] == [True]
    else:
        with pytest.raises(CachePolicyError, match="empty exclusive capped cache"):
            state.consume(lease)
        assert calls == [] and lease.ledger.documents == {}
