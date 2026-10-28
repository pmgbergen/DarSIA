"""A real RunIO binding admits only its exact indexed own shared completion."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.pipeline.products_publication import publish_run_metadata
from darsia.fluidflower.pipeline.store_runtime import InputHashMismatch
from .test_run_io_binding import fixture, doc, put


def own_publication(tmp_path, *, other_attempt=False):
    task, _, _, _, binding = fixture(tmp_path)
    job_root = tmp_path / task["job_id"]
    job = doc(job_root / "job.json", dict(commit=task["commit"], expected_runs=[task["run"]]))
    output = binding.output if not other_attempt else binding.output.parent / ("2" * 32)
    candidate = doc(output / "candidate_completed.json", dict(complete=True,
        worker_references_closed=True, run=task["run"], task=task,
        attempt_id=task["attempt_id"], commit=task["commit"]))
    index = publish_run_metadata(job_root / "lease" / "artifacts", job=job,
        run=task["run"], attempt_id=task["attempt_id"], machine_id=binding.machine_id,
        completed=candidate, products=[], metadata_product_ids=[],
        file_writers_closed=True, all_final_products_listed=True)
    public = put(job_root / "results" / task["run"] / "completed.json", Path(candidate["path"]).read_bytes())
    return binding, job, candidate, index, public


def test_actual_run_io_uses_only_the_own_indexed_candidate_for_shared_c_completion(tmp_path):
    binding, job, candidate, index, public = own_publication(tmp_path)
    original_inputs, original_versions = deepcopy(binding.verified.rows), deepcopy(binding.verified.versions)
    with binding.activate():
        with pytest.raises(InputHashMismatch, match="Unlisted"):
            DataReader().json(public)
        with pytest.raises(InputHashMismatch, match="released"):
            binding.admit_common_completion(public, job=job, index=index)
        binding.common_closure_released = True
        binding.common_job_binding = job
        assert binding.admit_common_completion(public, job=job, index=index) == candidate
        reader = DataReader()
        assert reader.path(public) == Path(candidate["path"])
        assert reader.json(public)["task"] == binding.task
    assert binding.verified.rows == original_inputs and binding.verified.versions == original_versions
    assert public["path"] != candidate["path"]


@pytest.mark.parametrize("mutation", ("foreign_attempt", "foreign_publication", "wrong_sha", "changed_after_admission"))
def test_foreign_or_changed_own_publication_never_becomes_an_input_fallback(tmp_path, mutation):
    binding, job, _, index, public = own_publication(tmp_path, other_attempt=mutation == "foreign_attempt")
    binding.common_closure_released = True
    binding.common_job_binding = job
    if mutation == "foreign_publication":
        public = put(tmp_path / "foreign" / "results" / binding.task["run"] / "completed.json",
                     Path(public["path"]).read_bytes())
    elif mutation == "wrong_sha":
        public = dict(public, sha256="f" * 64)
    if mutation == "changed_after_admission":
        binding.admit_common_completion(public, job=job, index=index)
        Path(public["path"]).write_bytes(b"changed publication")
        with binding.activate(), pytest.raises(InputHashMismatch, match="changed"):
            DataReader().json(public)
    else:
        with pytest.raises((ValueError, InputHashMismatch)):
            binding.admit_common_completion(public, job=job, index=index)
    assert all(row.path != Path(public["path"]) for row in binding.verified.rows)
