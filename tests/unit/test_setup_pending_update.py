"""Repeated Pending updates retain the already completed Apply contract."""
from pathlib import Path

from darsia.fluidflower.pipeline import job_queue
from .test_setup_job_reprepare import (
    failed_before_apply, unstarted, setup_job, controls, _completed_apply_case,
    _complete_apply_result, _retry, _bytes,
)


def test_repeated_pending_code_update_keeps_all_original_apply_and_build_controls(failed_before_apply, monkeypatch):
    case = failed_before_apply
    directory, _ = _completed_apply_case(case, monkeypatch)
    _complete_apply_result(case)
    expected_exports = _bytes(directory)
    expected_controls = _bytes(case.job_root / "setup-controls")
    first = _retry(case)
    assert (case.job_root / "queue/setup-apply.json").is_file()
    assert not list((case.job_root / "failed").glob("*.json"))

    for new_commit in ("6" * 40, "6" * 40):  # Another version, then the same version.
        already_current = job_queue.read_json(case.job_root / "job.json")["commit"] == new_commit
        before = _bytes(case.job_root)
        case.identity = dict(case.identity, commit=new_commit)
        updated = _retry(case)
        assert updated["root"] == first["root"]
        assert _bytes(directory) == expected_exports
        assert _bytes(case.job_root / "setup-controls") == expected_controls
        assert len(case.registry.jobs(case.series, "synthetic")) == 1
        if already_current:
            durable = lambda files: {path: data for path, data in files.items() if not path.endswith("/admission.lock")}
            assert updated["unchanged"] and durable(_bytes(case.job_root)) == durable(before)
        else:
            receipt = job_queue.read_json(Path(updated["history"]) / "receipt.json")
            assert receipt["retained_apply"]["coordinator"] == job_queue.descriptor(
                case.job_root / "setup-controls/apply-complete.json")
        queued = job_queue.read_json(case.job_root / "queue/setup-apply.json")
        assert queued["operation"] == "apply" and queued["commit"] == new_commit
