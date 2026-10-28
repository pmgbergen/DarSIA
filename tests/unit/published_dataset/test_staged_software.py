"""Upstream reproduction software uses each original job's exact identities.

These fixtures create real Git bundles and internally complete synthetic wheels;
they do not execute an analysis, install a library, or access external services.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from darsia.fluidflower.published_dataset.archives import write_stored_archive
from darsia.fluidflower.published_dataset.job_reader import FinishedJob
from darsia.fluidflower.published_dataset.staged_software import FORMAT, staged_software_members
from .synthetic import as_json, make_series, sha, write
from .test_export_options import _wheel_bytes, software_inventory


def _included_assets(root, inventory, stage):
    """Produce a real self-contained Git pack and a complete wheel fixture."""
    repository = root / "repository"
    repository.mkdir()
    template = root / "empty-template"
    template.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository),
            "-c", "core.hooksPath=" + str(template), *args], text=True).strip()

    git("init", "--template=" + str(template))
    (repository / "source.txt").write_text("Exact upstream source fixture: " + stage + "\n")
    git("add", "source.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false",
        "commit", "-m", "Upstream source fixture")
    commit = git("rev-parse", "HEAD")
    bundle = root / "source.bundle"
    git("bundle", "create", str(bundle), "HEAD")
    inventory["code"].update(commit=commit, bundle=dict(path=bundle.name,
        sha256=sha(bundle.read_bytes()), size_bytes=bundle.stat().st_size))
    wheel = root / "numpy-2.4.4-py3-none-any.whl"
    wheel.write_bytes(_wheel_bytes())
    inventory["libraries"][0].update(path=wheel.name, sha256=sha(wheel.read_bytes()),
                                     size_bytes=wheel.stat().st_size)
    return commit


def _case(tmp_path, *, stages=("phase1_r1",), mode="references"):
    selection_path = tmp_path / "inventories" / "stages.json"
    jobs, inventories, selected, sources = {}, {}, {}, {}
    for stage in stages:
        series = make_series(tmp_path / "jobs" / stage)
        path, inventory = software_inventory(series, selection_path.parent / stage)
        if mode == "include":
            commit = _included_assets(path.parent, inventory, stage)
            job_path = series.job / "job.json"
            job = json.loads(job_path.read_bytes())
            job["commit"] = commit
            write(job_path, as_json(job))
            write(path, as_json(inventory))
        jobs[stage] = FinishedJob(series.job).job
        inventories[stage] = path
        selected[stage] = path.relative_to(selection_path.parent).as_posix()
        sources[stage] = inventory
    write(selection_path, as_json(dict(format=FORMAT, jobs=selected)))
    request = SimpleNamespace(record="R", software_mode=mode, software_stage_inventory=selection_path)
    return SimpleNamespace(request=request, jobs=jobs, inventories=inventories,
                           selected=selected, sources=sources)


def test_each_upstream_stage_preserves_its_inventory_and_exact_software_references(tmp_path):
    fixture = _case(tmp_path, stages=("phase1_r2", "phase1_r1"))
    original_jobs = deepcopy(fixture.jobs)
    original_sources = {stage: path.read_bytes() for stage, path in fixture.inventories.items()}
    members, sections = staged_software_members(fixture.request, fixture.jobs)
    by_name = {member.name: b"".join(member.chunks()) for member in members}
    assert list(sections) == ["phase1_r1", "phase1_r2"]
    assert fixture.jobs == original_jobs
    assert len(members) == len(by_name) == 6
    for stage, section in sections.items():
        prefix = "software/stages/" + stage + "/"
        assert section["mode"] == "references" and section["python"] == "3.13.9"
        assert section["code"]["commit"] == fixture.jobs[stage]["commit"]
        lock = section["environment_lock"]
        assert lock["path"] == prefix + "requirements-worker.lock"
        assert sha(by_name[lock["path"]]) == fixture.jobs[stage]["environment_lock"]["sha256"]
        source = section["source_inventory"]
        assert by_name[source["path"]] == original_sources[stage]
        assert sha(by_name[source["path"]]) == source["sha256"]
        assert len(by_name[source["path"]]) == source["size_bytes"]
        versions = section["inventory"]
        assert sha(by_name[versions["path"]]) == versions["sha256"]
        assert len(by_name[versions["path"]]) == versions["size_bytes"]
        assert json.loads(by_name[versions["path"]]) == {
            key: value for key, value in section.items() if key != "inventory"}
    # Original relative inventories remain snapshots, not silently rewritten files.
    assert {stage: path.read_bytes() for stage, path in fixture.inventories.items()} == original_sources


def test_included_stage_software_uses_genuine_bundles_and_complete_wheels(tmp_path):
    fixture = _case(tmp_path, stages=("phase1_r1", "phase1_r2"), mode="include")
    members, sections = staged_software_members(fixture.request, fixture.jobs)
    output = tmp_path / "staged-software.zip"
    archive_record = write_stored_archive(output, members)
    assert len(archive_record["members"]) == 10
    assert sections["phase1_r1"]["code"]["commit"] != sections["phase1_r2"]["code"]["commit"]
    with zipfile.ZipFile(output) as archive:
        for stage, section in sections.items():
            root = fixture.inventories[stage].parent
            assert section["mode"] == "include"
            assert archive.read(section["code"]["path"]) == (root / "source.bundle").read_bytes()
            wheel = section["libraries"][0]
            assert archive.read(wheel["path"]) == (root / Path(wheel["path"]).name).read_bytes()
            assert wheel["path"].startswith("software/stages/" + stage + "/wheels/")
            assert sha(archive.read(section["inventory"]["path"])) == section["inventory"]["sha256"]


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_stage_inventory_requires_exact_original_job_set(tmp_path, change):
    fixture = _case(tmp_path)
    selected = dict(fixture.selected)
    if change == "missing":
        selected.clear()
    elif change == "extra":
        selected["unbound_stage"] = next(iter(selected.values()))
    else:
        fixture.request.software_stage_inventory.write_text('{"format": "' + FORMAT
            + '", "jobs": {"phase1_r1": "phase1_r1/software.json",'
            + ' "phase1_r1": "phase1_r1/software.json"}}')
    if change != "duplicate":
        write(fixture.request.software_stage_inventory, as_json(dict(format=FORMAT, jobs=selected)))
    with pytest.raises(ValueError, match="exactly once|duplicate keys"):
        staged_software_members(fixture.request, fixture.jobs)


@pytest.mark.parametrize("change, message", [
    ("no_selection", "explicit software inventory"),
    ("no_mode", "exact software references"),
    ("wrong_record", "exact software references"),
    ("commit", "commit differs"),
    ("lock", "lock differs"),
    ("lock_bytes", "differs from its explicit"),
    ("library", "each locked library"),
    ("version", "Library version differs"),
    ("python", "exact Python"),
    ("unbound_job_commit", "exact original code commit"),
    ("unbound_job_lock", "original environment lock binding"),
])
def test_upstream_software_never_substitutes_current_or_incomplete_metadata(tmp_path, change, message):
    fixture = _case(tmp_path)
    stage = "phase1_r1"
    inventory = fixture.sources[stage]
    if change == "no_selection":
        fixture.request.software_stage_inventory = None
    elif change == "no_mode":
        fixture.request.software_mode = None
    elif change == "wrong_record":
        fixture.request.record = "S"
    elif change == "commit":
        inventory["code"]["commit"] = "f" * 40
    elif change == "lock":
        inventory["environment_lock"]["sha256"] = "f" * 64
    elif change == "lock_bytes":
        (fixture.inventories[stage].parent / "requirements-worker.lock").write_bytes(b"changed lock")
    elif change == "library":
        inventory["libraries"] = []
    elif change == "version":
        inventory["libraries"][0]["version"] = "0.0.0"
    elif change == "python":
        inventory["python"] = "3.12.9"
    elif change == "unbound_job_commit":
        fixture.jobs[stage].pop("commit")
    elif change == "unbound_job_lock":
        fixture.jobs[stage].pop("environment_lock")
    write(fixture.inventories[stage], as_json(inventory))
    with pytest.raises(ValueError, match=message):
        staged_software_members(fixture.request, fixture.jobs)


@pytest.mark.parametrize("change", ["bundle", "wheel", "bundle_matching_hash", "wheel_matching_hash"])
def test_upstream_include_refuses_tampering_even_with_a_redeclared_outer_hash(tmp_path, change):
    fixture = _case(tmp_path, mode="include")
    stage = "phase1_r1"
    inventory = fixture.sources[stage]
    bound = inventory["code"]["bundle"] if change.startswith("bundle") else inventory["libraries"][0]
    path = fixture.inventories[stage].parent / bound["path"]
    path.write_bytes(path.read_bytes()[:-20] if change.startswith("bundle")
                     else _wheel_bytes(missing_member=True))
    if change.endswith("matching_hash"):
        bound.update(sha256=sha(path.read_bytes()), size_bytes=path.stat().st_size)
    write(fixture.inventories[stage], as_json(inventory))
    with pytest.raises(ValueError, match="differs from its explicit|complete, readable Git object closure|complete contents"):
        staged_software_members(fixture.request, fixture.jobs)


def test_upstream_included_assets_are_checked_again_when_archived(tmp_path):
    fixture = _case(tmp_path, mode="include")
    members, sections = staged_software_members(fixture.request, fixture.jobs)
    bound = sections["phase1_r1"]["code"]
    bundle = next(member for member in members if member.name == bound["path"])
    raw = bundle.source.read_bytes()
    bundle.source.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="Source SHA-256 differs"):
        write_stored_archive(tmp_path / "tampered.zip", members)


@pytest.mark.parametrize("missing", ["inventory", "lock", "bundle", "wheel"])
def test_explicit_upstream_software_files_must_exist(tmp_path, missing):
    fixture = _case(tmp_path, mode="include")
    source = fixture.sources["phase1_r1"]
    inventory_path = fixture.inventories["phase1_r1"]
    paths = {"inventory": inventory_path,
             "lock": inventory_path.parent / source["environment_lock"]["path"],
             "bundle": inventory_path.parent / source["code"]["bundle"]["path"],
             "wheel": inventory_path.parent / source["libraries"][0]["path"]}
    # Renaming keeps the fixture bytes available, while the explicit binding is missing.
    paths[missing].rename(paths[missing].with_name(paths[missing].name + ".unavailable"))
    with pytest.raises(FileNotFoundError):
        staged_software_members(fixture.request, fixture.jobs)


@pytest.mark.parametrize("relative", ["../outside.json", "/absolute.json", "C:/outside.json", "folder\\inventory.json"])
def test_stage_inventory_paths_cannot_escape_the_explicit_inventory_root(tmp_path, relative):
    fixture = _case(tmp_path)
    write(fixture.request.software_stage_inventory,
          as_json(dict(format=FORMAT, jobs={"phase1_r1": relative})))
    with pytest.raises(ValueError, match="relative path"):
        staged_software_members(fixture.request, fixture.jobs)


@pytest.mark.parametrize("stages", [dict(a={}, A={}), {"../stage": {}}, {"CON": {}}, {"stage.": {}}])
def test_stage_ids_cannot_collide_or_escape_portable_archive_names(stages):
    with pytest.raises(ValueError, match="portable"):
        staged_software_members(SimpleNamespace(software_stage_inventory=None), stages)


def test_no_upstream_stages_need_no_extra_software_inventory():
    assert staged_software_members(SimpleNamespace(software_stage_inventory=None), {}) == ([], {})
