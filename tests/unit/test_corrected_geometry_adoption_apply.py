"""Local synthetic lifecycle tests, not actual Store-facies/Build qualification.

This file is compile-only in the external delivery. No fixture runs on delivery.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest
import toml

from darsia.fluidflower import geometry_import as subject
from darsia.fluidflower.geometry_workspace_frame import FRAME_BINDING_FORMAT


def descriptor(path, shape=None):
    raw = path.read_bytes()
    result = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
              "size_bytes": len(raw), "mtime_ns": path.stat().st_mtime_ns}
    if shape is not None:
        result["shape"] = shape
    return result


def write_plan(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version=1\nseries_id="synthetic"\n[runs]\nids=["first","second"]\n')
    source_map = root / "map.png"
    source_photo = root / "historical.JPG"
    table = root / "original.csv"
    source_map.write_bytes(b"synthetic source map, never decoded")
    source_photo.write_bytes(b"synthetic source photograph, never decoded")
    table.write_bytes(b"id,name,porosity,permeability,swi CO2\n1,Sand,.4,10,.12\n")
    rows = {}
    for run in ("first", "second"):
        folder = root / run
        folder.mkdir()
        fields = {}
        for name in ("shape_corrected_baseline", "labels", "facies", "depth"):
            path = folder / (name + ".npz")
            path.write_bytes((run + ":" + name).encode())
            fields[name] = descriptor(path, [2, 3, 3] if name == "shape_corrected_baseline" else [2, 3])
        configured_depth = folder / "configured-original-depth.npz"
        configured_depth.write_bytes((run + ":configured-depth-distinct-from-saved-rig-depth").encode())
        config = {"rig": {"path": str(folder), "width": .9, "height": .49},
                  "depth": {"depth_map": str(configured_depth)},
                  "labeling": {"colored_image": str(source_map)},
                  "facies": {"props": str(table), "1": {"labels": [0]}}}
        config_path = root / (run + ".toml")
        config_path.write_text(toml.dumps(config))
        rows[run] = {"fields": fields, "configuration": descriptor(config_path),
                     "configured_depth_map": descriptor(configured_depth),
                     "facies_properties": descriptor(table)}
    correction = root / "shape_correction.npz"
    correction.write_bytes(b"synthetic saved correction, never decoded")
    binding = {"format": FRAME_BINDING_FORMAT, "source_reference_run": "historical",
               "display_reference_run": "first", "runs": rows,
               "reference_shape_corrections": [descriptor(correction)]}
    project = SimpleNamespace(root=root / "geometry_facies", manifest={})
    monkeypatch.setattr(subject, "native_series_link", lambda _p: {
        "series_root": str(root), "reference_run": "first"})
    calls = []

    class ContractStore:
        """No raster computation: only tests the wrapper's exact Store call."""
        def __init__(self, value):
            self.current_path = value.root / "overrides/geometry_facies/current.json"
            self.revisions_path = value.root / "overrides/geometry_facies/revisions.jsonl"
            self.artifact_root = value.root / "results/geometry"

        def revisions(self):
            return []

        def save_revision(self, config, *, actor, reason, preserve_failed_artifacts=False):
            # No validation/UUID override parameter exists in this contract stub.
            calls.append({"config": deepcopy(config), "actor": actor, "reason": reason,
                          "preserve_failed_artifacts": preserve_failed_artifacts})
            revision_id = uuid.uuid4().hex
            destination = self.artifact_root / revision_id
            destination.mkdir(parents=True)
            shutil.copyfile(config["depth_path"], destination / "depth.npz")
            stored = deepcopy(config)
            stored.update(depth_path=str(destination / "depth.npz"),
                          depth_artifact_path=str(destination / "depth.npz"),
                          depth_source_reference=descriptor(Path(config["depth_path"])))
            revision = {"schema": 1, "revision_id": revision_id,
                        "configuration": stored,
                        "artifacts": {"depth_path": stored["depth_path"],
                                      "facies_path": str(destination / "facies.npz")}}
            # This stub does not create a derived facies or claim equivalence.
            raw = json.dumps(revision).encode()
            (destination / "revision.json").write_bytes(raw)
            self.current_path.parent.mkdir(parents=True)
            self.current_path.write_bytes(raw)
            return revision

    monkeypatch.setattr(subject, "GeometryRevisionStore", ContractStore)
    monkeypatch.setattr(subject, "import_coloured_map", lambda *_a, **_k: pytest.fail("no palette import"))
    monkeypatch.setattr(subject.np, "load", lambda *_a, **_k: pytest.fail("wrapper must not decode arrays"))
    args = {"series_root": root, "configuration": root / "first.toml", "facies_table": table,
            "source_map": source_map, "source_image": source_photo, "frame_binding": binding,
            "facies_roles": {1: "porous_medium"}, "permeability_unit": "darcy", "actor": "preparer"}
    plan = subject.prepare_corrected_adoption(project, **args)
    plan_path = tmp_path / "prepared.json"
    digest = write_plan(plan_path, plan)
    return project, plan, plan_path, digest, calls


def apply(data):
    project, _plan, plan_path, digest, _calls = data
    return subject.apply_prepared_corrected_adoption(
        project, preparation=plan_path, preparation_sha256=digest, actor="explicit adopter")


def test_exact_reference_copies_single_real_store_contract_and_depth_separation(prepared):
    project, plan, _path, _digest, calls = prepared
    result = apply(prepared)
    assert len(calls) == 1 and calls[0]["preserve_failed_artifacts"] is True
    assert calls[0]["config"] == plan["candidate"]
    assert result["revision_id"] == result["revision"]["revision_id"]
    assert len(result["revision_id"]) == 32
    assert result["revision_name"] == "adopted-v1" != result["revision_id"]
    for row in plan["reference_copy_plan"]:
        assert Path(row["target"]).read_bytes() == Path(row["source"]["path"]).read_bytes()
    stored = result["revision"]["configuration"]
    assert stored["frame_binding"] == plan["candidate"]["frame_binding"]
    assert stored["provenance"] == plan["candidate"]["provenance"]
    receipt = json.loads(result["receipt"].read_bytes())
    assert receipt["Store_called_once"] and not receipt["Store_validation_overridden"]
    assert not receipt["Store_UUID_overridden"] and not receipt["workspace_depth_generated"]
    assert receipt["lossless_export_gate"].startswith("NOT RUN")
    assert receipt["rebuilt_rig_gate"].startswith("NOT RUN")
    assert receipt["original_configured_depth_pointer"] != receipt["reference_copies"]["depth"]
    assert (project.root / "annotations/geometry_import/initial_corrected_adoption_claim.json").is_file()


@pytest.mark.parametrize("kind", ["current", "history", "orphan"])
def test_preexisting_workspace_work_is_preserved_before_claim(prepared, kind):
    project, _plan, _path, _digest, calls = prepared
    store = subject.GeometryRevisionStore(project)
    target = {"current": store.current_path, "history": store.revisions_path,
              "orphan": store.artifact_root / ".old-uid.tmp/preserved.txt"}[kind]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original work")
    with pytest.raises(FileExistsError, match="preserved"):
        apply(prepared)
    assert not calls and target.read_bytes() == b"original work"
    assert not (project.root / "annotations/geometry_import/initial_corrected_adoption_claim.json").exists()


def test_any_other_run_source_change_blocks_before_copy_or_store(prepared):
    project, plan, _path, _digest, calls = prepared
    Path(plan["candidate"]["frame_binding"]["runs"]["second"]["fields"]["labels"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed|version changed|size changed"):
        apply(prepared)
    assert not calls and not project.root.exists()


def test_stale_preparation_digest_blocks_all_writes(prepared):
    project, plan, path, _digest, calls = prepared
    altered = deepcopy(plan)
    altered["revision_name"] = "unreviewed change"
    write_plan(path, altered)
    with pytest.raises(ValueError, match="prepared adoption document changed"):
        apply(prepared)
    assert not calls and not project.root.exists()


def test_existing_prepared_copy_directory_is_not_reused(prepared):
    project, plan, _path, _digest, calls = prepared
    work = Path(plan["reference_copy_plan"][0]["target"]).parent
    work.mkdir(parents=True)
    original = work / "partial.txt"
    original.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="fresh owned"):
        apply(prepared)
    assert not calls and original.read_bytes() == b"preserve"


def test_store_failure_preserves_copies_exact_traceback_and_partial_uid_no_retry(prepared, monkeypatch):
    project, plan, _path, _digest, calls = prepared
    base = subject.GeometryRevisionStore

    class FailingStore(base):
        def save_revision(self, config, *, actor, reason, preserve_failed_artifacts=False):
            assert preserve_failed_artifacts is True
            calls.append({"failed": True})
            stage = self.artifact_root / ".synthetic-uid.tmp"
            stage.mkdir(parents=True)
            (stage / "partial.txt").write_bytes(b"Store partial")
            raise ValueError("exact synthetic Store validation failure")

    monkeypatch.setattr(subject, "GeometryRevisionStore", FailingStore)
    with pytest.raises(ValueError, match="^exact synthetic Store validation failure$"):
        apply(prepared)
    work = Path(plan["reference_copy_plan"][0]["target"]).parent
    failure = json.loads((work / "failure.json").read_bytes())
    assert failure["Store_called"] is True and failure["actual_current_exists"] is False
    assert "ValueError: exact synthetic Store validation failure" in failure["traceback"]
    assert failure["Store_partial_paths"] == [str(base(project).artifact_root / ".synthetic-uid.tmp")]
    assert not failure["automatic_retry"] and not failure["automatic_rollback"] and not failure["deletion_requested"]
    assert all(Path(row["target"]).exists() for row in plan["reference_copy_plan"])
    with pytest.raises(FileExistsError):
        apply(prepared)
    assert len(calls) == 1
