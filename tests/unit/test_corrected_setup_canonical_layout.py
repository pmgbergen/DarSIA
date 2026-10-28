"""Compile-only Q17 layout/bundle tests, using synthetic local archives only.

These cases do not qualify the real Store mismatch, perform a Build, or change
the independent-raster authority held at Q18.
"""
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
from darsia.fluidflower import setup_rig as dispatch
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.geometry_workspace_frame import FRAME_BINDING_FORMAT, adopted_content_digest
from darsia.fluidflower.setup_display import completed_current_native_setup_export


def descriptor(path, shape=None):
    raw = path.read_bytes()
    row = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    if shape is not None:
        row["shape"] = shape
    return row


def inventory(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def layout(tmp_path, monkeypatch):
    series = tmp_path / "private_migration_target"
    (series / "config/runs").mkdir(parents=True)
    (series / "config/merged").mkdir()
    (series / "series.toml").write_text(toml.dumps({"series_id": "synthetic",
        "runs": {"ids": ["ac17", "ac29"]}}), encoding="utf-8")
    runs, originals = {}, {}
    for run in ("ac17", "ac29"):
        folder = series / "original" / run
        folder.mkdir(parents=True)
        fields = {}
        arrays = {"shape_corrected_baseline": np.zeros((2, 3, 3), dtype=np.float32),
                  "labels": np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8),
                  "facies": np.zeros((2, 3), dtype=np.uint8),
                  "depth": np.full((2, 3), .01)}
        for name, array in arrays.items():
            path = folder / (name + ".npz")
            np.savez(path, array=array)
            fields[name] = descriptor(path, list(array.shape))
        depth = folder / "configured-depth.npz"
        depth.write_bytes(b"separate original configured depth identity")
        props = folder / "full-original.csv"
        props.write_bytes(b"id,name,porosity,permeability,measured-extra\n0,original,.37,10,.12\n")
        original = {"data": {"results": str(series / "results" / run)},
                    "depth": {"depth_map": str(depth)},
                    "labeling": {"labels": "raw-labels.npz", "colored_image": "source.png"},
                    "facies": {"path": "raw-facies.npz", "props": str(props)},
                    "corrections": {"curvature": {"crop": {"top_left": [3, 8], "width": .9}}}}
        before = ("# exact synthetic original\r\n" + toml.dumps(original).replace("\n", "\r\n")).encode()
        for kind in ("runs", "merged"):
            (series / "config" / kind / (run + ".toml")).write_bytes(before)
        runs[run] = dict(fields=fields,
                        configuration=descriptor(series / "config/merged" / (run + ".toml")),
                        configured_depth_map=descriptor(depth), facies_properties=descriptor(props))
        originals[run] = before
    correction = series / "original-correction.npz"
    correction.write_bytes(b"original immutable correction identity")
    binding = dict(format=FRAME_BINDING_FORMAT, source_reference_run="ac14",
                   display_reference_run="ac17", reference_shape_corrections=[descriptor(correction)], runs=runs)
    fields = runs["ac17"]["fields"]
    config = dict(frame="rig-corrected", frame_binding=binding,
                  baseline_path=fields["shape_corrected_baseline"]["path"],
                  labels_path=fields["labels"]["path"], facies_source_path=fields["facies"]["path"],
                  depth_path=fields["depth"]["path"], regions=[])
    config["provenance"] = {"adopted_content_sha256": adopted_content_digest(config)}
    project = SimpleNamespace(root=series / "geometry_facies")
    current = project.root / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(schema=1, format=GEOMETRY_REVISION_FORMAT,
        revision_id="a" * 32, configuration=config, artifacts={"facies_path": fields["facies"]["path"]})))
    monkeypatch.setattr(subject, "GeometryRevisionStore", lambda _project: SimpleNamespace(current_path=current))
    monkeypatch.setattr(subject, "native_series_link", lambda _project: {"series_root": str(series)})
    import darsia
    monkeypatch.setattr(darsia, "imread", lambda _path: object())
    monkeypatch.setattr(subject, "_require_exact_grid", lambda *_args: None)
    return dict(series=series, project=project, current=current, runs=runs,
                original=originals, reference=lambda run: SimpleNamespace(run_id=run))


def export(data, run="ac17"):
    return subject.export_corrected_setup(data["project"], reference=data["reference"](run), actor="synthetic explicit export")


def test_reference_uses_ordinary_schema_and_existing_status_reader(layout):
    result = export(layout)
    expected = layout["project"].root / "exports/native_setup" / ("a" * 32)
    assert result.destination == expected
    assert result.receipt["schema"] == 1 and result.receipt["format"] == subject.NATIVE_SETUP_EXPORT_FORMAT
    assert result.receipt["run_id"] == result.receipt["reference_run"] == "ac17"
    assert result.receipt["frame"] == "rig-corrected"
    assert set(result.receipt["files"]) == {"native_labels.npz", "native_facies.npz", "facies_properties.csv", "native_setup.toml", "colored_labels.png"}
    assert completed_current_native_setup_export(layout["project"].root) == result.receipt
    for name, row in result.receipt["files"].items():
        assert row == descriptor(result.destination / name)
    for key, name in subject.NAMES.items():
        assert (result.destination / name).read_bytes() == Path(layout["runs"]["ac17"]["fields"][key]["path"]).read_bytes()
    assert result.receipt["lossless_rebuilt_gate"] == "NOT RUN"


def test_planned_child_pointer_never_claims_the_child_is_complete(layout):
    result = export(layout)
    assert result.receipt["runs"] == {"ac29": "runs/ac29/export.json"}
    assert result.receipt["runs_are_planned_paths_only"] is True
    assert result.receipt["child_completion_not_claimed"] is True
    assert not (result.destination / "runs/ac29/export.json").exists()


def test_child_requires_real_reference_export_and_does_not_create_it(layout):
    with pytest.raises(ValueError, match="display-reference run before"):
        export(layout, "ac29")
    assert not (layout["project"].root / "exports").exists()


def test_child_is_immutable_selected_run_and_reference_bytes_do_not_change(layout):
    first = export(layout)
    root_before = (first.destination / "export.json").read_bytes()
    child = export(layout, "ac29")
    assert child.destination == first.destination / "runs/ac29"
    assert child.receipt["run_id"] == "ac29" and child.receipt["reference_run"] == "ac17"
    assert child.receipt["root_export"] == descriptor(first.destination / "export.json")
    assert (first.destination / "export.json").read_bytes() == root_before
    assert child.receipt["binding"]["configuration"] == layout["runs"]["ac29"]["configuration"]


@pytest.mark.parametrize("run", ["ac17", "ac29"])
def test_exact_export_reuses_all_bytes_without_a_new_timestamp(layout, monkeypatch, run):
    export(layout)
    result = export(layout, run)
    before = inventory(layout["project"].root)
    monkeypatch.setattr(subject, "utc_now", lambda: pytest.fail("Reuse must not generate a timestamp"))
    again = export(layout, run)
    assert again.reused and again.receipt == result.receipt
    assert inventory(layout["project"].root) == before


@pytest.mark.parametrize("run", ["ac17", "ac29"])
def test_existing_mismatched_root_or_child_is_preserved(layout, run):
    export(layout)
    result = export(layout, run)
    path = result.destination / "export.json"
    changed = deepcopy(result.receipt)
    changed["run_id"] = "foreign"
    path.write_text(json.dumps(changed))
    before = inventory(result.destination)
    with pytest.raises(ValueError, match="different identity"):
        export(layout, run)
    assert inventory(result.destination) == before


def test_changed_original_property_binding_refuses_existing_export_without_overwrite(layout):
    result = export(layout)
    before = inventory(result.destination)
    Path(layout["runs"]["ac17"]["facies_properties"]["path"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="identity mismatch"):
        export(layout)
    assert inventory(result.destination) == before


def test_ambiguous_declared_configuration_bundle_is_rejected(layout):
    path = layout["series"] / "series.toml"
    data = tomllib.loads(path.read_text())
    data["runs"]["config_files"] = {"ac29": "config/runs/ac17.toml"}
    path.write_text(toml.dumps(data))
    with pytest.raises(ValueError, match="ambiguous between runs"):
        export(layout)
    assert not (layout["project"].root / "exports").exists()


def test_second_apply_and_export_reuse_exact_completed_bytes_without_writing(layout, monkeypatch):
    result = export(layout)
    original = deepcopy(layout["original"])
    applied = subject.apply_corrected_selected_run(layout["project"], result)
    before = inventory(layout["series"])
    monkeypatch.setattr(subject, "utc_now", lambda: pytest.fail("A second Apply must not create a timestamp"))
    monkeypatch.setattr(subject.uuid, "uuid4", lambda: pytest.fail("A second Apply must not create a UUID"))
    assert subject.apply_corrected_selected_run(layout["project"], result) == applied
    assert export(layout).reused
    assert inventory(layout["series"]) == before
    for kind in ("runs", "merged"):
        path = layout["series"] / "config" / kind / "ac17.toml"
        assert path.read_bytes() == subject.corrected_config_bytes(original["ac17"], result.receipt["binding"]["replacements"])
    assert (layout["series"] / "config/merged/ac29.toml").read_bytes() == original["ac29"]


def test_changed_config_after_apply_refuses_idempotent_reuse_and_preserves_records(layout):
    result = export(layout)
    subject.apply_corrected_selected_run(layout["project"], result)
    path = layout["series"] / "config/runs/ac17.toml"
    path.write_bytes(path.read_bytes() + b"# unrelated later edit\n")
    before = inventory(result.destination)
    with pytest.raises(ValueError, match="identity mismatch"):
        subject.apply_corrected_selected_run(layout["project"], result)
    assert inventory(result.destination) == before


def test_partial_failed_apply_is_preserved_and_cannot_be_retried(layout, monkeypatch):
    result = export(layout)
    replace = subject.os.replace
    calls = []
    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("synthetic second configuration replacement failure")
        return replace(source, target)
    monkeypatch.setattr(subject.os, "replace", fail_second)
    with pytest.raises(subject.CorrectedApplyError, match="synthetic second"):
        subject.apply_corrected_selected_run(layout["project"], result)
    assert (result.destination / "apply-claim.json").is_file()
    assert not (result.destination / "apply-complete.json").exists()
    before = inventory(result.destination)
    with pytest.raises(ValueError, match="incomplete"):
        subject.apply_corrected_selected_run(layout["project"], result)
    assert inventory(result.destination) == before


@pytest.mark.parametrize("as_list", [False, True])
def test_dispatcher_accepts_only_actual_completed_selected_bundle(layout, monkeypatch, as_list):
    result = export(layout)
    subject.apply_corrected_selected_run(layout["project"], result)
    path = layout["series"] / "config/merged/ac17.toml"
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    selected = [path] if as_list else path
    dispatch.setup_rig(dispatch.Rig, selected, False)
    assert len(calls) == 1 and calls[0][1] == selected
    assert calls[0][0].setup is dispatch.Rig.setup


def test_build_refuses_foreign_run_config_using_reference_export(layout, monkeypatch):
    result = export(layout)
    subject.apply_corrected_selected_run(layout["project"], result)
    foreign = layout["series"] / "config/merged/ac29.toml"
    foreign.write_bytes(subject.corrected_config_bytes(foreign.read_bytes(), result.receipt["binding"]["replacements"]))
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="different or ambiguous selected-run"):
        dispatch.setup_rig(dispatch.Rig, foreign, False)
    assert calls == []


def test_build_refuses_export_not_yet_applied(layout, monkeypatch):
    result = export(layout)
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    path = layout["series"] / "config/runs/ac17.toml"
    path.write_bytes(subject.corrected_config_bytes(path.read_bytes(), result.receipt["binding"]["replacements"]))
    with pytest.raises(ValueError, match="Apply this run"):
        dispatch.setup_rig(dispatch.Rig, path)
    assert calls == []


def test_existing_incomplete_completion_is_never_rewritten_as_success(layout):
    result = export(layout)
    subject.apply_corrected_selected_run(layout["project"], result)
    path = result.destination / "apply-complete.json"
    receipt = json.loads(path.read_bytes())
    receipt["complete"] = False
    path.write_text(json.dumps(receipt))
    before = inventory(result.destination)
    with pytest.raises(ValueError, match="completion has a different identity"):
        subject.apply_corrected_selected_run(layout["project"], result)
    assert inventory(result.destination) == before


def test_declared_child_bundle_build_uses_its_own_three_files(layout, monkeypatch):
    export(layout)
    child = export(layout, "ac29")
    subject.apply_corrected_selected_run(layout["project"], child)
    path = layout["series"] / "config/merged/ac29.toml"
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    dispatch.setup_rig(dispatch.Rig, path)
    assert len(calls) == 1 and calls[0][1] == path
    data = tomllib.loads(path.read_text())
    assert Path(data["labeling"]["labels"]).parent == child.destination
    assert Path(data["facies"]["path"]).parent == child.destination
    assert Path(data["facies"]["props"]).parent == child.destination


def test_original_raw_dispatch_remains_exact_and_needs_no_export(layout, monkeypatch):
    path = layout["series"] / "config/merged/ac29.toml"
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    dispatch.setup_rig(dispatch.Rig, path, True)
    assert calls == [(dispatch.Rig, path, True)]


def test_original_store_facies_mismatch_still_stops_before_any_export(layout):
    different = layout["series"] / "different-derived-facies.npz"
    np.savez(different, array=np.ones((2, 3), dtype=np.uint8))
    current = json.loads(layout["current"].read_bytes())
    current["artifacts"]["facies_path"] = str(different)
    layout["current"].write_text(json.dumps(current))
    with pytest.raises(ValueError, match="Store-derived facies differ from the original adopted facies; no corrected export was written"):
        export(layout)
    assert not (layout["project"].root / "exports").exists()


def test_external_edit_of_target_one_during_second_replace_never_completes(layout, monkeypatch):
    result = export(layout)
    replace = subject.os.replace
    targets = []
    external = b"external edit after the first target was already replaced\n"
    def mutate_first(source, target):
        targets.append(Path(target))
        outcome = replace(source, target)
        if len(targets) == 2:
            targets[0].write_bytes(external)
        return outcome
    monkeypatch.setattr(subject.os, "replace", mutate_first)
    with pytest.raises(subject.CorrectedApplyError, match="target differs from its expected-after bytes") as error:
        subject.apply_corrected_selected_run(layout["project"], result)
    record = json.loads(error.value.receipt.read_bytes())
    assert record["state"] == "failed"
    assert [row["state"] for row in record["files"]] == ["applied", "applied"]
    assert targets[0].read_bytes() == external
    assert len(list(error.value.receipt.parent.glob("*-before.toml"))) == 2
    assert (result.destination / "apply-claim.json").exists()
    assert not (result.destination / "apply-complete.json").exists()


@pytest.mark.parametrize("changed", ["source_properties", "current", "export_receipt", "common"])
def test_non_target_input_changed_during_apply_is_failed_without_completion(layout, monkeypatch, changed):
    common = layout["series"] / "config/common.toml"
    if changed == "common":
        common.write_text('marker="explicit original shared metadata"\n')
    result = export(layout)
    paths = {"source_properties": Path(layout["runs"]["ac17"]["facies_properties"]["path"]),
             "current": layout["current"], "export_receipt": result.destination / "export.json", "common": common}
    altered = paths[changed]
    replace, calls = subject.os.replace, []
    def alter_non_target(source, target):
        outcome = replace(source, target)
        calls.append(target)
        if len(calls) == 2:
            altered.write_bytes(altered.read_bytes() + b"\nexternal change\n")
        return outcome
    monkeypatch.setattr(subject.os, "replace", alter_non_target)
    with pytest.raises(subject.CorrectedApplyError) as error:
        subject.apply_corrected_selected_run(layout["project"], result)
    assert json.loads(error.value.receipt.read_bytes())["state"] == "failed"
    assert altered.read_bytes().endswith(b"external change\n")
    assert not (result.destination / "apply-complete.json").exists()
    assert len(list(error.value.receipt.parent.glob("*-before.toml"))) == 2


@pytest.mark.parametrize("changed", ["missing", "wrong_run", "root_bytes", "root_file", "root_source"])
def test_child_apply_rechecks_genuine_root_transition_before_any_write(layout, changed):
    reference = export(layout)
    child = export(layout, "ac29")
    root_json = reference.destination / "export.json"
    if changed == "missing":
        root_json.unlink()
    elif changed == "wrong_run":
        data = json.loads(root_json.read_bytes())
        data["run_id"] = "ac29"
        root_json.write_text(json.dumps(data))
    elif changed == "root_bytes":
        root_json.write_bytes(root_json.read_bytes() + b"\n")
    elif changed == "root_file":
        (reference.destination / "native_labels.npz").write_bytes(b"external changed root archive")
    else:
        Path(layout["runs"]["ac17"]["facies_properties"]["path"]).write_bytes(b"external changed root original properties")
    before = {(kind, run): (layout["series"] / "config" / kind / (run + ".toml")).read_bytes()
              for kind in ("runs", "merged") for run in ("ac17", "ac29")}
    child_before = inventory(child.destination)
    with pytest.raises((ValueError, OSError)):
        subject.apply_corrected_selected_run(layout["project"], child)
    assert child_before == inventory(child.destination)
    assert not (child.destination / "apply-claim.json").exists()
    assert all((layout["series"] / "config" / kind / (run + ".toml")).read_bytes() == raw
               for (kind, run), raw in before.items())


def test_child_root_file_changed_during_apply_is_retained_as_failed_partial(layout, monkeypatch):
    reference = export(layout)
    child = export(layout, "ac29")
    replace, calls = subject.os.replace, []
    def change_root_after_second_target(source, target):
        outcome = replace(source, target)
        calls.append(target)
        if len(calls) == 2:
            (reference.destination / "native_facies.npz").write_bytes(b"external edit of root while child Apply was running")
        return outcome
    monkeypatch.setattr(subject.os, "replace", change_root_after_second_target)
    with pytest.raises(subject.CorrectedApplyError, match="identity mismatch") as error:
        subject.apply_corrected_selected_run(layout["project"], child)
    record = json.loads(error.value.receipt.read_bytes())
    assert record["state"] == "failed" and len(record["files"]) == 2
    assert (child.destination / "apply-claim.json").is_file()
    assert not (child.destination / "apply-complete.json").exists()
    assert len(list(error.value.receipt.parent.glob("*-before.toml"))) == 2
