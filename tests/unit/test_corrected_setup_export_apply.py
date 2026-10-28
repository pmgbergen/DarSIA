"""Synthetic explicit export/config transactions; no real run or Build."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib
import numpy as np

import pytest
import toml

from darsia.fluidflower import corrected_setup as subject
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.geometry_workspace_frame import FRAME_BINDING_FORMAT, adopted_content_digest


def _descriptor(path, shape=None):
    raw = path.read_bytes()
    row = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw), mtime_ns=path.stat().st_mtime_ns)
    if shape:
        row["shape"] = shape
    return row


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "series"
    (root / "config/runs").mkdir(parents=True)
    (root / "config/merged").mkdir()
    (root / "series.toml").write_text('series_id="Synthetic"\n[runs]\nids=["ac29","ac17"]\n')
    folder = root / "old/ac29/rig"
    folder.mkdir(parents=True)
    fields = {}
    for name in ("shape_corrected_baseline", "labels", "facies", "depth"):
        path = folder / (name + ".npz")
        path.write_bytes(("unmodified archive " + name).encode())
        fields[name] = _descriptor(path, [2, 3, 3] if name == "shape_corrected_baseline" else [2, 3])
    original_depth = folder.parent / "measured-depth.npz"
    original_depth.write_bytes(b"the original measured depth map; never substituted")
    props = root / "ac29-original.csv"
    props.write_bytes(b'id,name,porosity,permeability,swi CO2\n0,old,.37,10,.12\n')
    other_props = root / "canonical-other.csv"
    other_props.write_bytes(b'id,name,porosity,permeability\n0,wrong,.9,999\n')
    original = dict(data=dict(results=str(root / "results/ac29")),
                    depth=dict(depth_map=str(original_depth)),
                    labeling=dict(colored_image="original-map.png", labels="raw-labels.npz", unite_labels=[[1, 8]]),
                    facies={"path": "raw-facies.npz", "props": str(props), "0": {"labels": [1, 8]}},
                    corrections={"curvature": {"crop": {"top_left": [88, 101], "width": .9}}},
                    calibration=dict(marker="unchanged"))
    before = ("# original comment and layout\r\n" + toml.dumps(original).replace("\n", "\r\n")).encode()
    merged = root / "config/merged/ac29.toml"
    partial = root / "config/runs/ac29.toml"
    merged.write_bytes(before)
    partial.write_bytes(before)
    correction = root / "shape_correction.npz"
    correction.write_bytes(b"original correction identity")
    binding = dict(format=FRAME_BINDING_FORMAT, source_reference_run="ac14", display_reference_run="ac29",
                   reference_shape_corrections=[_descriptor(correction)],
                   runs={"ac29": dict(fields=fields, configuration=_descriptor(merged), configured_depth_map=_descriptor(original_depth), facies_properties=_descriptor(props))})
    config = dict(frame="rig-corrected", frame_binding=binding, facies_properties_source_path=str(other_props),
                  baseline_path=fields["shape_corrected_baseline"]["path"], labels_path=fields["labels"]["path"],
                  facies_source_path=fields["facies"]["path"], depth_path=fields["depth"]["path"], regions=[])
    config["provenance"] = dict(adopted_content_sha256=adopted_content_digest(config))
    project = SimpleNamespace(root=root / "geometry_facies")
    project.root.mkdir()
    current = project.root / "current.json"
    current.write_text(json.dumps({
        "schema": 1, "format": GEOMETRY_REVISION_FORMAT, "revision_id": "a" * 32,
        "configuration": config, "artifacts": {"facies_path": fields["facies"]["path"]},
    }))
    monkeypatch.setattr(subject, "GeometryRevisionStore", lambda project: SimpleNamespace(current_path=current))
    monkeypatch.setattr(subject, "native_series_link", lambda project: {"series_root": str(root)})
    import darsia
    monkeypatch.setattr(darsia, "imread", lambda path: object())
    monkeypatch.setattr(subject, "_load_array", lambda path: np.zeros((2, 3), np.uint8))
    grids = []
    monkeypatch.setattr(subject, "_require_exact_grid", lambda path, baseline, field: grids.append((path, field)))
    return dict(project=project, reference=SimpleNamespace(run_id="ac29"), root=root,
                current=current, fields=fields, original=original, before=before,
                merged=merged, partial=partial, properties=props, grids=grids)


def test_export_copies_exact_selected_archives_and_full_run_csv_without_relabelling(setup):
    result = subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    assert result.run_id == "ac29" and result.receipt["actual_Build_run"] is False
    for name, filename in subject.NAMES.items():
        assert (result.destination / filename).read_bytes() == Path(setup["fields"][name]["path"]).read_bytes()
    assert (result.destination / "facies_properties.csv").read_bytes() == setup["properties"].read_bytes()
    assert b"swi CO2" in (result.destination / "facies_properties.csv").read_bytes()
    assert [name for path, name in setup["grids"]] == ["labels", "facies", "depth"]
    assert result.receipt["lossless_rebuilt_gate"] == "NOT RUN"
    assert setup["merged"].read_bytes() == setup["partial"].read_bytes() == setup["before"]


def test_apply_changes_only_four_leaves_and_preserves_old_depth_crop_ids_and_comments(setup):
    result = subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    receipt = subject.apply_corrected_selected_run(setup["project"], result)
    expected = deepcopy(setup["original"])
    for section, values in result.receipt["binding"]["replacements"].items():
        expected[section].update(values)
    for path in (setup["merged"], setup["partial"]):
        assert tomllib.loads(path.read_bytes().decode()) == expected
        assert path.read_bytes().startswith(b"# original comment and layout\r\n")
        assert b"\r\n" in path.read_bytes()
    data = json.loads(receipt.read_bytes())
    assert data["state"] == "complete" and len(data["files"]) == 2
    assert data["crop_changed"] is data["depth_pointer_changed"] is data["registration_run"] is data["Build_run"] is False
    assert any(path.read_bytes() == setup["before"] for path in receipt.parent.glob("*-before.toml"))


def test_changed_workspace_refuses_before_config_write(setup):
    result = subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    setup["current"].write_bytes(b"external workspace edit")
    with pytest.raises(RuntimeError, match="workspace revision changed"):
        subject.apply_corrected_selected_run(setup["project"], result)
    assert setup["merged"].read_bytes() == setup["partial"].read_bytes() == setup["before"]


def test_changed_export_archive_refuses_before_config_write(setup):
    result = subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    (result.destination / "native_labels.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        subject.apply_corrected_selected_run(setup["project"], result)
    assert setup["merged"].read_bytes() == setup["partial"].read_bytes() == setup["before"]


def test_changed_original_csv_on_same_path_is_not_adopted_silently(setup):
    setup["properties"].write_bytes(b"new physical properties at same path")
    with pytest.raises(ValueError, match="identity mismatch"):
        subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    assert not (setup["project"].root / "exports").exists()


def test_real_grid_refusal_precedes_any_export_folder(setup, monkeypatch):
    def refuse(*args):
        raise ValueError("Corrected labels grid mismatch (shape)")
    monkeypatch.setattr(subject, "_require_exact_grid", refuse)
    with pytest.raises(ValueError, match="grid mismatch"):
        subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    assert not (setup["project"].root / "exports").exists()


def test_edited_corrected_revision_is_not_silently_replaced_by_old_maps(setup):
    current = json.loads(setup["current"].read_bytes())
    current["configuration"]["regions"] = [{"label_id": 77, "facies_id": 4}]
    setup["current"].write_text(json.dumps(current))
    with pytest.raises(ValueError, match="was edited"):
        subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    assert not (setup["project"].root / "exports").exists()


def test_second_target_failure_preserves_originals_and_reports_actual_partial_application(setup, monkeypatch):
    result = subject.export_corrected_setup(setup["project"], reference=setup["reference"])
    original_replace = subject.os.replace
    count = []
    def fail_second(source, target):
        count.append(target)
        if len(count) == 2:
            raise OSError("synthetic second configuration replacement failure")
        return original_replace(source, target)
    monkeypatch.setattr(subject.os, "replace", fail_second)
    with pytest.raises(subject.CorrectedApplyError, match="synthetic second") as error:
        subject.apply_corrected_selected_run(setup["project"], result)
    receipt = json.loads(error.value.receipt.read_bytes())
    assert receipt["state"] == "failed"
    assert [row["state"] for row in receipt["files"]] == ["applied", "staged"]
    assert setup["partial"].read_bytes() == setup["before"]
    assert len(list(error.value.receipt.parent.glob("*-before.toml"))) == 2


def test_text_patch_rejects_any_unrequested_leaf():
    changes = dict(labeling=dict(labels="new.npz", frame="rig-corrected"), facies=dict(path="new-facies.npz", props="original.csv", guessed=1))
    with pytest.raises(ValueError, match="Only the four"):
        subject.corrected_config_bytes(b'[depth]\ndepth_map="old.npz"\n', changes)
