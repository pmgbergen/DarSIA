"""Real isolated Git identity tests; no scientific executor, job claim or NAS."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import worker
from darsia.fluidflower.pipeline.run_io import InputHashMismatch
from darsia.fluidflower.native.io import sha256_file
from darsia.fluidflower.native.source import verify_execution_identity


SOURCE = "src/darsia/fluidflower/native/source.py"
DISPLAY = "src/darsia/fluidflower/native/display_modes.py"
LOCK = "requirements-worker.lock"
SOURCE_BYTES = b"# Authored operational fixture; never executed.\nVALUE = 1\n"
LOCK_BYTES = b"# Authored environment fixture.\npytest==8.0.0\n"


def _git(repo, *args, binary=False):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=not binary,
        stderr=subprocess.PIPE,
    )


def _put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _origin(tmp_path, *, display=False):
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "--object-format=sha1")
    _git(repo, "config", "user.name", "Q40 isolated test")
    _git(repo, "config", "user.email", "q40@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    _put(repo / ".gitattributes", b"*.py text\n*.lock text\n")
    _put(repo / SOURCE, SOURCE_BYTES)
    _put(repo / LOCK, LOCK_BYTES)
    if display:
        from darsia.fluidflower.native import display_modes
        _put(repo / DISPLAY, Path(display_modes.__file__).read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Authored source and lock fixture")
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def _checkout(origin, target, head, *, crlf):
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", str(origin), str(target)],
        stderr=subprocess.PIPE,
    )
    _git(target, "config", "core.autocrlf", "true" if crlf else "false")
    _git(target, "config", "core.eol", "crlf" if crlf else "lf")
    _git(target, "checkout", "--detach", head)
    assert _git(target, "status", "--porcelain") == ""
    assert _git(target, "rev-parse", "HEAD").strip() == head
    expected = SOURCE_BYTES.replace(b"\n", b"\r\n") if crlf else SOURCE_BYTES
    assert (target / SOURCE).read_bytes() == expected
    return target


def _existing_fixture():
    """Reuse the original admission/owned-output fixture from its sibling test module."""
    path = Path(__file__).with_name("test_run_io_binding.py")
    name = "_q40_original_run_io_test_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.fixture


def _binding(tmp_path, repo, head):
    tmp_path.mkdir()
    task, config, bound, host, binding = _existing_fixture()(tmp_path)
    binding.repository = repo.resolve()
    binding.commit = head
    binding.task["commit"] = head
    return task, config, bound, host, binding


def _admit(binding):
    digest = hashlib.sha256(SOURCE_BYTES).hexdigest()
    binding._admit_code(SOURCE, digest, "processor_module_sha256")
    return digest


def _startup(tmp_path, repo, head, monkeypatch, *, lock_digest=None, job_commit=None):
    job_root = tmp_path / "metadata-only-startup"
    job_root.mkdir()
    job = dict(commit=job_commit or head,
               environment_lock=dict(path=LOCK, sha256=lock_digest or hashlib.sha256(LOCK_BYTES).hexdigest()),
               precision="float32", calibration_sha256="b" * 64, expected_runs=["ac52"])
    (job_root / "job.json").write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr(worker, "__file__", str(repo / "src/darsia/fluidflower/pipeline/worker.py"))
    monkeypatch.setattr(worker, "require_unc", lambda _path: None)
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path:
        job_filesystem.classify_shared_folder(path, job_filesystem.FilesystemFacts("smb", "injected local test transport")))
    context = {}
    result = worker.verify_startup(job_root, startup_context=context)
    return result, context, job


def test_one_commit_clean_LF_and_CRLF_share_canonical_identity_and_distinct_raw_provenance(tmp_path):
    origin, head = _origin(tmp_path)
    records = []
    for crlf in (False, True):
        label = "crlf" if crlf else "lf"
        repo = _checkout(origin, tmp_path / label, head, crlf=crlf)
        _task, _config, _bound, _host, binding = _binding(tmp_path / (label + "-binding"), repo, head)
        expected = _admit(binding)
        raw = (repo / SOURCE).read_bytes()
        with binding.activate():
            actual = sha256_file(repo / SOURCE)
            verify_execution_identity(dict(processor_module_sha256=actual),
                dict(processor_module_sha256=expected), ["processor_module_sha256"])
            assert actual == expected
        record = binding.code_audit[-1]
        assert record["identity_algorithm"] == "sha256-git-blob-content"
        assert record["sha256"] == expected
        assert record["working_tree_sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["working_tree_size_bytes"] == len(raw)
        assert record["size_bytes"] == len(SOURCE_BYTES)
        assert record["EOL_only_representation"] is crlf
        records.append(record)
        assert _git(repo, "status", "--porcelain") == ""
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["working_tree_sha256"] != records[1]["working_tree_sha256"]


@pytest.mark.parametrize("crlf", [False, True])
def test_original_native_hash_outside_binding_still_means_exact_raw_bytes(tmp_path, crlf):
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=crlf)
    assert sha256_file(repo / SOURCE) == hashlib.sha256((repo / SOURCE).read_bytes()).hexdigest()
    assert (sha256_file(repo / SOURCE) == hashlib.sha256(SOURCE_BYTES).hexdigest()) is (not crlf)


def test_original_display_contract_uses_canonical_source_without_rendering(tmp_path):
    origin, head = _origin(tmp_path, display=True)
    repo = _checkout(origin, tmp_path / "crlf", head, crlf=True)
    _task, _config, _bound, _host, binding = _binding(tmp_path / "binding", repo, head)
    canonical = _git(repo, "show", head + ":" + DISPLAY, binary=True)
    digest = hashlib.sha256(canonical).hexdigest()
    binding._admit_code(DISPLAY, digest, "display_contract_renderer_code_sha256")
    name = "darsia.fluidflower.native._q40_original_display_contract"
    spec = importlib.util.spec_from_file_location(name, repo / DISPLAY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        with binding.activate():
            result = module.display_contract(parameters=dict(contract_metadata=dict(display_only=True)))
        assert result == dict(display_only=True, renderer_code_sha256=digest)
        assert (repo / DISPLAY).read_bytes() != canonical
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("change", ["canonical_digest", "substantive_before_admission", "untracked_path"])
def test_source_admission_rejects_nonrepresentation_changes(tmp_path, change):
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=False)
    _task, _config, _bound, _host, binding = _binding(tmp_path / "binding", repo, head)
    relative, digest = SOURCE, hashlib.sha256(SOURCE_BYTES).hexdigest()
    if change == "canonical_digest":
        digest = "0" * 64
    elif change == "substantive_before_admission":
        (repo / SOURCE).write_bytes(SOURCE_BYTES.replace(b"VALUE = 1", b"VALUE = 2"))
    else:
        relative = "src/darsia/fluidflower/native/untracked.py"
        _put(repo / relative, SOURCE_BYTES)
    with pytest.raises(InputHashMismatch):
        binding._admit_code(relative, digest, "processor_module_sha256")


@pytest.mark.parametrize("eol_only", [False, True])
@pytest.mark.parametrize("consumer", ["asset_path", "native_hash"])
def test_source_mutation_after_admission_is_refused_even_if_EOL_only(tmp_path, eol_only, consumer):
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=False)
    _task, _config, _bound, _host, binding = _binding(tmp_path / "binding", repo, head)
    _admit(binding)
    changed = SOURCE_BYTES.replace(b"\n", b"\r\n") if eol_only else SOURCE_BYTES + b"# later mutation\n"
    (repo / SOURCE).write_bytes(changed)
    with binding.activate(), pytest.raises(InputHashMismatch, match="Admitted current code changed"):
        binding.asset_path(repo / SOURCE) if consumer == "asset_path" else sha256_file(repo / SOURCE)


def test_repeated_source_role_cannot_replace_the_admitted_raw_snapshot(tmp_path):
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=False)
    _task, _config, _bound, _host, binding = _binding(tmp_path / "binding", repo, head)
    _admit(binding)
    (repo / SOURCE).write_bytes(SOURCE_BYTES.replace(b"\n", b"\r\n"))
    with pytest.raises(InputHashMismatch, match="Conflicting or changed"):
        _admit(binding)


def test_data_EOL_mutation_still_refused_and_owned_outputs_still_raw(tmp_path):
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=True)
    _task, config, _bound, _host, binding = _binding(tmp_path / "binding", repo, head)
    _admit(binding)
    output = binding.output / "own.txt"
    _put(output, b"own\r\nresult\r\n")
    with binding.activate():
        assert sha256_file(output) == hashlib.sha256(output.read_bytes()).hexdigest()
        reader = DataReader(unc_only=True)
        actual = reader.path(config["calibration"])
        original = actual.read_bytes()
        actual.write_bytes(original + b"\r\n")
        with pytest.raises((ValueError, InputHashMismatch)):
            reader.json(config["calibration"])


@pytest.mark.parametrize("crlf", [False, True])
def test_original_worker_startup_accepts_clean_same_commit_lock_with_explicit_raw_provenance(tmp_path, monkeypatch, crlf):
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=crlf)
    result, context, job = _startup(tmp_path, repo, head, monkeypatch)
    assert result == (job, head)
    identity = context["environment_lock_identity"]
    assert identity["sha256"] == hashlib.sha256(LOCK_BYTES).hexdigest()
    assert identity["size_bytes"] == len(LOCK_BYTES)
    assert identity["working_tree_sha256"] == hashlib.sha256((repo / LOCK).read_bytes()).hexdigest()
    assert identity["EOL_only_representation"] is crlf
    assert _git(repo, "status", "--porcelain") == ""


@pytest.mark.parametrize("problem", ["wrong_commit", "wrong_canonical_lock_SHA", "dirty_source", "dirty_lock"])
def test_original_worker_startup_refuses_wrong_commit_dirty_or_wrong_lock(tmp_path, monkeypatch, problem):
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=False)
    options = {}
    if problem == "wrong_commit":
        options["job_commit"] = "a" * 40
    elif problem == "wrong_canonical_lock_SHA":
        options["lock_digest"] = "0" * 64
    elif problem == "dirty_source":
        (repo / SOURCE).write_bytes(SOURCE_BYTES + b"# changed\n")
    else:
        (repo / LOCK).write_bytes(LOCK_BYTES + b"numpy==9.0\n")
    with pytest.raises(RuntimeError):
        _startup(tmp_path, repo, head, monkeypatch, **options)


def test_wrong_repository_native_identity_is_not_accepted_as_new_current_binding(tmp_path):
    origin, head = _origin(tmp_path)
    repo = _checkout(origin, tmp_path / "checkout", head, crlf=True)
    _task, _config, _bound, _host, binding = _binding(tmp_path / "binding", repo, head)
    raw_digest = hashlib.sha256((repo / SOURCE).read_bytes()).hexdigest()
    assert raw_digest != hashlib.sha256(SOURCE_BYTES).hexdigest()
    with pytest.raises(InputHashMismatch, match="exact execution binding"):
        binding._admit_code(SOURCE, raw_digest, "processor_module_sha256")
