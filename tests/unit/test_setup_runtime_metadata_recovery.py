"""New job-owned runtime metadata is retained without a directory whitelist."""
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import job_queue
from .test_setup_failed_build import failed_build, unstarted, setup_job, controls
from .test_setup_job_reprepare import _retry, _bytes


def _runtime_inventory(case):
    root = case.job_root
    task = job_queue.read_json(root / "claimed/private-host-cpu/sample.json")
    observation = root / "claim_observations" / ("sample." + task["attempt_id"] + ".json")
    job_queue.write_immutable(observation, dict(run="sample", attempt_id=task["attempt_id"],
        claimed_relative_path="claimed/private-host-cpu/sample.json", state="ownerless_claim_first_observed"))
    # Existing and future runtime namespaces, including nested metadata and a
    # standalone root file. None grants execution or scientific authority.
    for relative in ("clock_samples/runtime.json", "future_runtime/nested/status.json", ".future_runtime/receipt.json", "runtime.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"runtime_only": true}')
    return observation


def test_full_runtime_inventory_is_archived_with_claims_and_retirement(failed_build):
    case = failed_build
    observation = _runtime_inventory(case)
    earlier = case.job_root / "history/older-generation.json"
    earlier.parent.mkdir(exist_ok=True)
    earlier.write_bytes(b"prior immutable history")
    # Same root inventory as the actual interrupted job, plus future metadata.
    assert {"claim_observations", "clock_samples", "claimed", "leases", "decisions",
        "done", "failed", "queue", "results", "logs", "exports", "setup-controls", "ui", ".capacity",
        "history", "publishing", "reclaimed", "job.json"} <= {
        path.name for path in case.job_root.iterdir()}
    before = _bytes(case.job_root)
    result = _retry(case)
    archive = Path(result["history"]) / "previous"
    for relative in (observation.relative_to(case.job_root), Path("clock_samples/runtime.json"),
                     Path("future_runtime/nested/status.json"), Path(".future_runtime/receipt.json"), Path("runtime.json")):
        assert (archive / relative).read_bytes() == before[relative.as_posix()]
        assert not (case.job_root / relative).exists()
    assert (case.job_root / "setup-controls/apply-complete.json").is_file()
    assert (archive / "done/setup-apply.json").is_file()
    assert earlier.read_bytes() == b"prior immutable history"


def test_foreign_known_claim_observation_does_not_weaken_ownership(failed_build):
    case = failed_build
    path = _runtime_inventory(case)
    value = job_queue.read_json(path)
    value["claimed_relative_path"] = "claimed/foreign/sample.json"
    path.write_text(__import__('json').dumps(value))
    old = (case.job_root / "job.json").read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        _retry(case)
    assert (case.job_root / "job.json").read_bytes() == old


def test_declared_scientific_source_namespace_is_never_archived(unstarted):
    from darsia.fluidflower.setup_job_reprepare import _preserved_entries
    case = unstarted
    path = case.job_root / "model_sources/original.json"
    path.parent.mkdir()
    path.write_text('{"original":true}')
    plan = dict(case.plan, native_parameters=job_queue.descriptor(path))
    assert "model_sources" in _preserved_entries(case.job_root,
        job_queue.read_json(case.job_root / "job.json"), plan, case.stores)
