"""Approve run records (N141 spec 2.5, Part 2): append-only, bound to a calibration.

Local temporary series folders only; no result, calibration or job is touched.
"""

import datetime as dt
import hashlib
import json

import pytest

from darsia.fluidflower import calibration_approvals as approvals


def _series(tmp_path):
    root = tmp_path / "series"
    (root / "fits").mkdir(parents=True)
    (root / "fits/frozen.json").write_text('{"frozen": true}', encoding="utf-8")
    return root


def test_calibration_identity_names_the_reference_and_hashes_its_bytes(tmp_path):
    root = _series(tmp_path)
    identity = approvals.calibration_identity(root, "fits/frozen.json")
    assert identity == {
        "reference": "fits/frozen.json",
        "sha256": hashlib.sha256(b'{"frozen": true}').hexdigest(),
    }
    with pytest.raises(ValueError, match="No mass calibration"):
        approvals.calibration_identity(root, "")


def test_approvals_are_appended_and_never_rewritten(tmp_path):
    root = _series(tmp_path)
    calibration = approvals.calibration_identity(root, "fits/frozen.json")
    retried = []

    def retry(action):
        retried.append(action)
        return action()

    first = approvals.approve_run(root, "ac28", calibration, reviewer="reviewer-a",
                                  now=dt.datetime(2026, 10, 20, 8, 0, tzinfo=dt.timezone.utc), retry=retry)
    path = root / approvals.APPROVALS_PATH
    before = path.read_bytes()
    approvals.approve_run(root, "AC47", calibration, result="results/ac47.json",
                          reviewer="reviewer-b", retry=retry)
    assert path.read_bytes().startswith(before)  # The first record is untouched.
    assert len(retried) == 2
    records = approvals.read_run_approvals(root)
    assert [record["run"] for record in records] == ["ac28", "AC47"]
    assert records[0] == first
    assert first["format"] == approvals.APPROVAL_FORMAT
    assert first["recorded_utc"] == "2026-10-20T08:00:00+00:00" and first["result"] is None
    assert records[1]["result"] == "results/ac47.json" and records[1]["reviewer"] == "reviewer-b"


def test_latest_approval_of_a_run_for_one_calibration(tmp_path):
    root = _series(tmp_path)
    calibration = approvals.calibration_identity(root, "fits/frozen.json")
    other = dict(calibration, sha256="0" * 64)
    approvals.approve_run(root, "ac28", other, reviewer="old", retry=lambda action: action())
    assert approvals.run_approval(approvals.read_run_approvals(root), "ac28", calibration["sha256"]) is None
    approvals.approve_run(root, "ac28", calibration, reviewer="first", retry=lambda action: action())
    approvals.approve_run(root, "AC28", calibration, reviewer="second", retry=lambda action: action())
    latest = approvals.run_approval(approvals.read_run_approvals(root), "Ac28", calibration["sha256"])
    assert latest["reviewer"] == "second"


def test_a_damaged_record_is_reported_not_read_as_unapproved(tmp_path):
    root = _series(tmp_path)
    assert approvals.read_run_approvals(root) == []
    path = root / approvals.APPROVALS_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"format": "something-else"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a run approval record"):
        approvals.read_run_approvals(root)


def test_without_a_reviewer_setting_the_record_names_nobody(tmp_path, monkeypatch):
    import getpass

    monkeypatch.setattr(getpass, "getuser", lambda: pytest.fail("The OS login is never recorded"))
    root = _series(tmp_path)
    calibration = approvals.calibration_identity(root, "fits/frozen.json")
    for reviewer in (None, "", "   "):
        record = approvals.approve_run(root, "ac28", calibration, reviewer=reviewer, retry=lambda action: action())
        assert record["reviewer"] is None
    assert approvals.approve_run(root, "ac28", calibration, reviewer=" Reviewer A ",
                                 retry=lambda action: action())["reviewer"] == "Reviewer A"


def test_the_append_uses_the_job_queue_transient_retry_by_default(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import job_queue

    root = _series(tmp_path)
    calls = []
    real = job_queue.retry_transient
    monkeypatch.setattr(job_queue, "retry_transient", lambda action, **kw: calls.append(action) or real(action, delays=()))
    approvals.approve_run(root, "ac28", approvals.calibration_identity(root, "fits/frozen.json"), reviewer="r")
    assert len(calls) == 1 and len(approvals.read_run_approvals(root)) == 1
