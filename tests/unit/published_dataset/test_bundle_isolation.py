"""Real Git-object fixtures for offline bundle verification isolation.

These tests create only disposable repositories; they do not execute scientific
code, install software, or contact a remote. Authored fixtures are not a claim
that the tests have been run.
"""
import os
import subprocess
from types import SimpleNamespace

import pytest

from darsia.fluidflower.published_dataset.export_software import _bundle


def _environment():
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="")
    return environment


def _git(repository, *args, input=None, environment=None, check=True):
    return subprocess.run(["git", "-C", str(repository),
        "-c", "core.hooksPath=" + str(repository / "empty-hooks"),
        "-c", "commit.gpgsign=false", "-c", "user.name=Bundle fixture",
        "-c", "user.email=bundle-fixture@example.org", *args],
        input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        env=_environment() if environment is None else environment, check=check)


def _snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def bundle_case(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    template = tmp_path / "empty-template"
    template.mkdir()
    _git(repository, "init", "--template=" + str(template))
    (repository / "source.txt").write_bytes(b"An actual reachable source blob.\n")
    _git(repository, "add", "source.txt")
    _git(repository, "commit", "-m", "Bundle object fixture")
    commit = _git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    blob = _git(repository, "rev-parse", "HEAD:source.txt").stdout.decode().strip()
    complete = tmp_path / "complete.bundle"
    _git(repository, "bundle", "create", str(complete), "HEAD")

    # pack-objects without --revs packs only the explicitly named objects. The
    # commit and tree are genuine, but their reachable source blob is omitted.
    objects = [line.split()[0] for line in
               _git(repository, "rev-list", "--objects", commit).stdout.decode().splitlines()]
    assert blob in objects and commit in objects
    packed = [oid for oid in objects if oid != blob]
    pack = _git(repository, "pack-objects", "--stdout",
                input=("\n".join(packed) + "\n").encode()).stdout
    incomplete = tmp_path / "missing-blob.bundle"
    incomplete.write_bytes(("# v2 git bundle\n" + commit + " HEAD\n\n").encode() + pack)
    return SimpleNamespace(repository=repository, template=template, commit=commit,
                           blob=blob, complete=complete, incomplete=incomplete)


def test_missing_bundle_object_is_not_supplied_by_inherited_alternates(tmp_path, monkeypatch, bundle_case):
    case = bundle_case
    proof = tmp_path / "alternate-proof.git"
    _git(tmp_path, "init", "--bare", "--template=" + str(case.template), str(proof))
    alternate = str(case.repository / ".git" / "objects")
    environment = _environment()
    environment["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = alternate
    _git(proof, "bundle", "unbundle", str(case.incomplete), environment=environment)
    # Independent Git checks establish the failure trigger: this pack has no
    # blob, while the same object lookup/fsck succeeds with the external store.
    assert _git(proof, "cat-file", "-e", case.blob, check=False).returncode != 0
    _git(proof, "cat-file", "-e", case.blob, environment=environment)
    _git(proof, "fsck", "--full", "--strict", "--no-reflogs", case.commit,
         environment=environment)
    before = _snapshot(case.repository)
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", alternate)
    with pytest.raises(ValueError, match="complete, readable Git object closure"):
        _bundle(case.incomplete, case.commit)
    assert _snapshot(case.repository) == before


@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY"])
def test_repository_or_object_redirection_cannot_supply_missing_bundle_objects(monkeypatch, bundle_case, variable):
    case = bundle_case
    selected = case.repository / ".git"
    if variable == "GIT_OBJECT_DIRECTORY":
        selected /= "objects"
    # Each variable is tested independently, with the omitted object actually
    # present in the caller-selected repository/object store.
    _git(case.repository, "cat-file", "-e", case.blob)
    before = _snapshot(case.repository)
    monkeypatch.setenv(variable, str(selected))
    with pytest.raises(ValueError, match="complete, readable Git object closure"):
        _bundle(case.incomplete, case.commit)
    assert _snapshot(case.repository) == before


@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE",
                                      "GIT_OBJECT_DIRECTORY", "GIT_INDEX_FILE"])
def test_complete_bundle_does_not_write_to_inherited_git_locations(tmp_path, monkeypatch, bundle_case, variable):
    outside = tmp_path / "outside-verification"
    selected = outside / "redirected"
    before = _snapshot(bundle_case.repository)
    monkeypatch.setenv(variable, str(selected))
    _bundle(bundle_case.complete, bundle_case.commit)
    assert not outside.exists()
    assert _snapshot(bundle_case.repository) == before


def test_complete_bundle_ignores_inherited_git_config_injection(monkeypatch, bundle_case):
    # An invalid command-scope config count makes Git reject the environment if
    # caller-controlled configuration reaches any verifier subprocess.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "not-an-integer")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.repositoryformatversion")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "9999")
    _bundle(bundle_case.complete, bundle_case.commit)


def test_isolated_bundle_keeps_exact_head_and_prerequisite_refusals(tmp_path, bundle_case):
    case = bundle_case
    with pytest.raises(ValueError, match="exact published commit"):
        _bundle(case.complete, "0" * 40)
    raw = case.complete.read_bytes()
    header, pack = raw.split(b"\n\n", 1)
    prerequisite = tmp_path / "requires-original-repository.bundle"
    lines = header.splitlines(keepends=True)
    prerequisite.write_bytes(lines[0] + ("-" + case.commit + " prerequisite\n").encode()
                             + b"".join(lines[1:]) + b"\n\n" + pack)
    with pytest.raises(ValueError, match="unavailable prerequisite commits"):
        _bundle(prerequisite, case.commit)
