"""Original publication verifier on synthetic owned metadata; no RAW/science.

The fixture executes the actual queue publication and original terminal reader
when these tests are scheduled. Its explicitly synthetic payloads and ownership
are not scientific evidence, host admission, or a coupled RAW acceptance test.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import reproduction_lineage as lineage
from darsia.fluidflower.pipeline.job_queue import descriptor
from .test_phase1_enqueue_alias_master import owned_publication


def _arguments(data):
    return dict(configuration=data.result["configuration"],
                completion=data.result["phase1_completion"], expected_commit=data.job["commit"])


def _origin(data):
    return lineage.bind_phase1_origin(descriptor(data.root / "job.json"),
                                     data.task["run"], **_arguments(data))


def test_original_terminal_binding_returns_exact_catalogue_and_shared_proof_files(tmp_path):
    data = owned_publication(tmp_path, complete=True)
    origin = _origin(data)
    verified = lineage.verify_phase1_origin(origin, **_arguments(data),
        ordered_case_ids=data.task["ordered_case_ids"])
    assert verified["origin"] == origin
    assert verified["task"]["input_catalog"] == data.task["input_catalog"]
    assert verified["completed"] == data.result
    assert verified["scientific_inputs_verified"] is False
    assert verified["worker_admission_granted"] is False
    files = verified["proof_files"]
    assert len({row["relative_path"] for row in files}) == len(files)
    assert any(row["relative_path"].endswith("native_statistics.json.gz") for row in files)
    assert {"job.json", "results/run-one/task.json", "done/run-one.json"} <= {
        row["relative_path"] for row in files}
    for row in files:
        assert Path(row["path"]).resolve() == (data.root / row["relative_path"]).resolve()
        assert descriptor(Path(row["path"])) == {key: row[key] for key in ("path", "sha256", "size_bytes")}


def test_complete_named_shared_proof_tree_reopens_without_producer_local_payloads(tmp_path):
    data = owned_publication(tmp_path / "source", complete=True)
    origin = _origin(data)
    original = lineage.verify_phase1_origin(origin, **_arguments(data))
    copied = tmp_path / "portable-proof"
    for row in original["proof_files"]:
        target = copied / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(row["path"]).read_bytes())
    moved = deepcopy(origin)
    for role in ("job", "task", "done", "products_index"):
        relative = Path(origin[role]["path"]).resolve().relative_to(data.root.resolve())
        moved[role] = dict(origin[role], path=str(copied / relative))
    verified = lineage.verify_phase1_origin(moved, **_arguments(data))
    assert verified["task"] == original["task"]
    assert verified["completed"] == original["completed"]
    assert verified["products_index"] == original["products_index"]
    assert verified["scientific_inputs_verified"] is False


@pytest.mark.parametrize("role", ("job", "task", "done", "products_index"))
def test_changed_origin_metadata_refuses_even_when_source_completion_is_unchanged(tmp_path, role):
    data = owned_publication(tmp_path, complete=True)
    origin = _origin(data)
    target = Path(origin[role]["path"])
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        lineage.verify_phase1_origin(origin, **_arguments(data))


def test_reanchored_task_digest_cannot_replace_original_publication_selection(tmp_path):
    data = owned_publication(tmp_path, complete=True)
    origin = _origin(data)
    path = Path(origin["task"]["path"])
    task = json.loads(path.read_bytes())
    task["ordered_case_ids"] = ["other-case"]
    path.write_bytes(json.dumps(task).encode())
    origin["task"] = descriptor(path)
    with pytest.raises(ValueError):
        lineage.verify_phase1_origin(origin, **_arguments(data))


@pytest.mark.parametrize("change", ("missing", "hash_only", "wrong_run", "wrong_selection", "wrong_source"))
def test_full_origin_needs_actual_publication_and_exact_sealed_source(tmp_path, change):
    data = owned_publication(tmp_path, complete=True)
    origin, arguments = _origin(data), _arguments(data)
    if change == "missing":
        origin.pop("job")
    elif change == "hash_only":
        origin["job"] = dict(sha256=origin["job"]["sha256"])
    elif change == "wrong_run":
        origin["run"] = "another-run"
    elif change == "wrong_selection":
        arguments["ordered_case_ids"] = ["another-case"]
    else:
        arguments["completion"] = dict(arguments["completion"], path=str(tmp_path / "other.json"))
    with pytest.raises(ValueError):
        lineage.verify_phase1_origin(origin, **arguments)


def test_tampered_shared_original_frame_refuses_beyond_top_level_origin_hashes(tmp_path):
    data = owned_publication(tmp_path, complete=True)
    origin = _origin(data)
    verified = lineage.verify_phase1_origin(origin, **_arguments(data))
    row = next(row for row in verified["proof_files"]
               if row["relative_path"].endswith("metadata/phase1/case-one/phase1_commit.json"))
    target = Path(row["path"])
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        lineage.verify_phase1_origin(origin, **_arguments(data))


def test_sealer_persists_origin_without_changing_scientific_data_and_legacy_omits_it(tmp_path):
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    from darsia.fluidflower.pipeline.staged_data import seal_phase2_configuration, validate_phase2_run
    data = owned_publication(tmp_path / "fixture", complete=True)
    origin = _origin(data)
    arguments = dict(completion_bound=data.result["phase1_completion"],
        configuration_bound=data.result["configuration"], expected_phase1_commit=data.job["commit"],
        fitted_calibration=descriptor(data.claim.output / "new_fitted_calibration.json"),
        fitted_anchor=dict(elapsed_h=1., affected_pore_volume_m3=.001), reader=DataReader(unc_only=False))
    original = seal_phase2_configuration(tmp_path / "legacy.json", **arguments)
    added = seal_phase2_configuration(tmp_path / "bound.json", phase1_origin=origin, **arguments)
    baseline = json.loads(Path(original["configuration"]["path"]).read_bytes())
    bound = json.loads(Path(added["configuration"]["path"]).read_bytes())
    assert "origin" not in baseline["phase1"]
    assert bound["phase1"].pop("origin") == origin
    assert bound == baseline
    bound["phase1"]["origin"] = deepcopy(origin)
    validate_phase2_run(bound)
    bound["phase1"]["origin"]["run"] = "another-run"
    with pytest.raises(ValueError, match="another run"):
        validate_phase2_run(bound)


@pytest.mark.parametrize("root", (
    "C:/recorded-producer/job", r"\\original-host\jobs\job", "/recorded-producer/job",
    r"C:\recorded-producer\job",
))
def test_passive_origin_schema_preserves_foreign_absolute_producer_spelling(root):
    origin = dict(format=lineage.FORMAT, run="one", **{
        role: dict(path=root + "/" + relative, sha256="a" * 64, size_bytes=1)
        for role, relative in (("job", "job.json"), ("task", "results/one/task.json"),
                               ("done", "done/one.json"), ("products_index", "results/one/products.json"))})
    original = deepcopy(origin)
    assert lineage.validate_phase1_origin(origin) is origin
    assert origin == original


@pytest.mark.parametrize("role", ("job", "task", "done", "products_index"))
@pytest.mark.parametrize("path", (
    "", "recorded-producer/job.json", r"C:recorded-producer\job.json",
    r"\recorded-producer\job.json", "file:///recorded-producer/job.json", None,
))
def test_passive_origin_schema_rejects_nonabsolute_role_path(role, path):
    """The lexical check applies to every role without opening producer files."""
    origin = dict(format=lineage.FORMAT, run="one", **{
        key: dict(path="C:/recorded-producer/job/" + relative, sha256="a" * 64, size_bytes=1)
        for key, relative in (("job", "job.json"), ("task", "results/one/task.json"),
                              ("done", "done/one.json"), ("products_index", "results/one/products.json"))})
    origin[role]["path"] = path
    original = deepcopy(origin)
    with pytest.raises(ValueError, match="Original phase-one " + role + " needs exact bounded metadata bytes"):
        lineage.validate_phase1_origin(origin)
    assert origin == original
