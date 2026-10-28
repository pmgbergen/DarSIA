"""Opt-in package/readback of the genuine fixture; no scientific executor.

No environment selection means an explicit skip, never qualification. See the
genuine fixture module for its independent manifest pin and exact original
software/job/environment requirements. All outputs stay in pytest's temp root.
"""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from .test_staged_raw_route import genuine_fixture


def test_genuine_original_staged_package_reopens_complete_proof_and_input_closure(genuine_fixture, tmp_path):
    from darsia.fluidflower.published_dataset import staged_package as package
    from darsia.fluidflower.published_dataset.staged_software import FORMAT, staged_software_members
    from darsia.fluidflower.published_dataset.layout import inside
    fixture = genuine_fixture
    run = fixture.manifest["run"]
    assert fixture.phase2.runs == [run], "Choose an original single-run finite fixture; never truncate the saved selection"
    plan_row, = [row for row in fixture.phase2_rows if "run.execution_series" in row.uses]
    published = SimpleNamespace(commit=fixture.phase2.job["commit"], plan_sha256=plan_row.reference["sha256"],
        plan_path=plan_row.path, configs=[fixture.phase2_config], proof=None)
    request = SimpleNamespace(record="R", retain_producer_metadata=True, software_mode="references")
    prepared = package.prepare_staged(request, fixture.phase2, published, fixture.manifest["roots"], None)
    inventories, selected = tmp_path / "inventories", {}
    for phase, prefix in (("phase1", "p1"), ("phase2", "p2")):
        bound = fixture.manifest[phase + "_software"]
        raw = Path(bound["path"]).read_bytes()
        source = json.loads(raw)
        stage_id = prefix + "-" + run
        copied = inventories / stage_id / "software.json"
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(raw)
        lock = source["environment_lock"]
        target = inside(copied.parent, lock["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(Path(bound["path"]).parent, lock["path"]), target)
        selected[stage_id] = copied.relative_to(inventories).as_posix()
    selection = inventories / "stages.json"
    selection.write_text(json.dumps(dict(format=FORMAT, jobs=selected)), encoding="utf-8")
    request.software_stage_inventory = selection
    software_members, software = staged_software_members(request, prepared.stage_jobs)
    package.attach_staged_software(prepared, software, software_members)
    portable = tmp_path / "portable"
    for member in [*prepared.members, *software_members]:
        target = inside(portable, member.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for chunk in member.chunks():
                stream.write(chunk)
    for relative, raw in prepared.tasks.items():
        target = inside(portable, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    plan = inside(portable, "plan/adopted_plan.toml")
    plan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan_row.path, plan)
    frames = []
    for spec in fixture.phase2_config["frames"]:
        use = "frame." + spec["case_id"] + ".native.acquisition"
        source, = [row["reference"] for row in fixture.phase1_task["input_catalog"]["inputs"] if use in row["uses"]]
        frames.append(dict(case_id=spec["case_id"], frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"],
            raw=dict(path=source["root_alias"] + "/" + source["relative_path"],
                     sha256=source["sha256"], size_bytes=source["bytes"])))
    manifest = dict(identity=dict(record="R", series_id=fixture.manifest["series_id"]),
        code=dict(published_commit=fixture.phase2.job["commit"]), runs=[dict(run=run, frames=frames)],
        staged_reproduction=prepared.section)
    actual = package.readback_staged(portable, manifest)
    assert actual["p1-" + run]["task"] == fixture.phase1_task
    assert actual["p2-" + run]["task"] == fixture.phase2_task
    assert set(actual) == {"p1-" + run, "p2-" + run}
    product = next(row for row in prepared.section["transport"]
        if row["logical"]["path"] == fixture.phase2_config["phase1"]["completion"]["path"])
    assert product["asset"]["sha256"] == fixture.phase2_config["phase1"]["completion"]["sha256"]
    # Missing exact indexed originals cannot be replaced by a declared digest.
    proof = prepared.section["stages"]["p1-" + run]["task"]
    target = inside(portable, proof["path"])
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA/size"):
        package.readback_staged(portable, manifest)
