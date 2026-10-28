"""Q64 source/current identities and pre-dispatch guards, synthetic data only.

The original dispatcher is observed through a test double, never a real Build.
"""
from copy import deepcopy
import json
from pathlib import Path
import tomllib

import pytest
import toml

from darsia.fluidflower import corrected_setup as subject
from darsia.fluidflower import setup_rig as dispatch
from darsia.fluidflower.geometry_workspace_frame import adopted_content_digest
from .test_corrected_setup_canonical_layout import descriptor, export, inventory, layout


@pytest.fixture
def pointer(layout):
    data = layout
    series = data["series"]
    manifest_path = series / "series.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    manifest["runs"].update(config_files={}, merged_files={}, geometry_configuration_bindings={},
                            excluded={"ac29": "Synthetic excluded run remains excluded."})
    bindings, sources, old_bytes, before, pre_bytes = {}, {}, {}, {}, {}
    for run in ("ac17", "ac29"):
        source = series / "config/merged" / (run + ".toml")
        original = tomllib.loads(source.read_text())
        old_rig = series / "original_rig" / run
        old_rig.mkdir(parents=True)
        (old_rig / "meta_data.json").write_bytes(b"preserved original rig")
        original["rig"] = {"path": str(old_rig)}
        original["protocols"] = dict(imaging={f"raw/{run}": f"old/{run}/imaging.csv"},
                                     injection=f"old/{run}/injection.csv", imaging_mode="auto")
        input_folder = series / "input" / run
        (input_folder / "nested").mkdir(parents=True)
        (input_folder / "nested/baseline.jpg").write_bytes(b"not decoded by this metadata gate")
        results = Path(original["data"]["results"])
        (results / "cache").mkdir(parents=True)
        (results / "raw_cache/nested").mkdir(parents=True)
        (results / "raw_cache/nested/baseline.npz").write_bytes(b"existing cache; not decoded")
        original["data"].update(folder=str(input_folder), baseline="nested/baseline.jpg", use_cache=True)
        raw = ("# immutable physical source\r\n" + toml.dumps(original).replace("\n", "\r\n")).encode()
        source.write_bytes(raw)
        (series / "config/runs" / (run + ".toml")).write_bytes(raw)
        data["runs"][run]["configuration"] = descriptor(source)
        sources[run], old_bytes[run] = descriptor(source), raw
        current = deepcopy(original)
        mode = "unchanged-zero-leaves"
        if run == "ac17":
            current["protocols"] = dict(imaging={f"raw/{run}": f"gui/{run}/imaging.csv"},
                                        injection=f"gui/{run}/injection.csv", imaging_mode="authored")
            mode = "clock-migrated-three-leaves"
        previous = []
        for kind in ("runs", "merged"):
            path = (series / "config/gui_pre_pointer_operation" / kind / (run + ".toml")
                    if run == "ac17" else series / "config" / kind / (run + ".toml"))
            if run == "ac17":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(toml.dumps(current))
            previous.append(descriptor(path))
            pre_bytes[str(path)] = path.read_bytes()
        target = data["project"].root / "results/rig_models" / ("a" * 32) / run
        current["rig"]["path"] = str(target)
        current_raw = ("# new GUI pointer copy\r\n" + toml.dumps(current).replace("\n", "\r\n")).encode()
        targets = []
        for kind in ("runs", "merged"):
            path = series / "config/geometry_output_operation" / kind / (run + ".toml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(current_raw)
            targets.append(descriptor(path))
            before[str(path)] = current_raw
            manifest["runs"]["config_files" if kind == "runs" else "merged_files"][run] = path.relative_to(series).as_posix()
        changes = {} if run == "ac29" else {
            f"protocols.{key}": dict(before=original["protocols"][key], after=current["protocols"][key])
            for key in ("imaging", "injection", "imaging_mode")}
        bindings[run] = dict(format=subject.CONFIGURATION_BINDING_V2_FORMAT, run_id=run,
            source_configuration=sources[run], current_configuration=targets[-1],
            apply_configurations=targets, pre_pointer_configurations=previous,
            protocol_change_mode=mode, protocol_changes=changes,
            output_pointer_change=dict(field="rig.path", before=str(old_rig), after=str(target)),
            comparison_sha256="e" * 64)
        manifest["runs"]["geometry_configuration_bindings"][run] = bindings[run]
    manifest_path.write_text(toml.dumps(manifest), encoding="utf-8")
    revision = json.loads(data["current"].read_bytes())
    config = revision["configuration"]
    config["frame_binding"]["runs"] = deepcopy(data["runs"])
    config["provenance"]["adopted_content_sha256"] = adopted_content_digest(config)
    data["current"].write_text(json.dumps(revision))
    return dict(layout=data, bindings=bindings, sources=sources, old_bytes=old_bytes,
                before=before, pre_bytes=pre_bytes, manifest=manifest_path,
                revision_before=data["current"].read_bytes())


def _rewrite(pointer, change):
    manifest = tomllib.loads(pointer["manifest"].read_text())
    change(manifest["runs"]["geometry_configuration_bindings"]["ac17"])
    pointer["manifest"].write_text(toml.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("run", ["ac17", "ac29"])
def test_v2_export_apply_and_build_guard_use_only_new_current_copies(pointer, run):
    data = pointer["layout"]
    export(data)
    result = export(data, run)
    binding = pointer["bindings"][run]
    assert result.receipt["binding"]["dual_configuration_binding"] == binding
    assert result.receipt["binding"]["configuration"] == pointer["sources"][run]
    assert result.receipt["binding"]["series_manifest"] == descriptor(pointer["manifest"])
    receipt = subject.apply_corrected_selected_run(data["project"], result)
    assert len(json.loads(receipt.read_bytes())["files"]) == 2
    for row in binding["apply_configurations"]:
        path = Path(row["path"])
        before = pointer["before"][str(path)]
        assert path.read_bytes() == subject.corrected_config_bytes(before, result.receipt["binding"]["replacements"])
        old, new = tomllib.loads(before.decode()), tomllib.loads(path.read_text())
        assert new["rig"] == old["rig"] and new["protocols"] == old["protocols"]
        assert new["depth"] == old["depth"] and new["corrections"] == old["corrections"]
    merged = Path(binding["current_configuration"]["path"])
    assert subject.require_corrected_export_for_build(merged) == result.receipt
    assert not Path(binding["output_pointer_change"]["after"]).exists()
    assert Path(pointer["sources"][run]["path"]).read_bytes() == pointer["old_bytes"][run]
    assert all(Path(row["path"]).read_bytes() == pointer["pre_bytes"][row["path"]]
               for row in binding["pre_pointer_configurations"])
    assert data["current"].read_bytes() == pointer["revision_before"]
    assert tomllib.loads(pointer["manifest"].read_text())["runs"]["excluded"] == {
        "ac29": "Synthetic excluded run remains excluded."}


def test_original_dispatcher_receives_same_args_and_no_output_subclass(pointer, monkeypatch):
    data = pointer["layout"]
    result = export(data)
    subject.apply_corrected_selected_run(data["project"], result)
    merged = Path(pointer["bindings"]["ac17"]["current_configuration"]["path"])
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda cls, path, show: calls.append((cls, path, show)))
    monkeypatch.setattr(dispatch, "corrected_rig_class", lambda cls: cls)
    cls = type("ExistingRig", (), {})
    assert dispatch.setup_rig(cls, merged, True) is None
    assert calls == [(cls, merged, True)]
    assert not Path(pointer["bindings"]["ac17"]["output_pointer_change"]["after"]).exists()


@pytest.mark.parametrize("mutation", [
    "missing_mode", "unknown_mode", "extra_key", "missing_pointer", "extra_pointer_key", "wrong_leaf",
    "wrong_before", "null_after", "unchanged_after", "wrong_project", "wrong_revision", "wrong_run",
    "wrong_source_sha", "wrong_current_sha", "bool_size", "source_as_target", "current_not_merged",
    "bad_comparison_sha", "wrong_protocol_record", "three_as_zero", "missing_protocol_record", "wrong_format",
    "missing_pre", "swapped_pre", "pre_outside", "pre_as_target", "wrong_pre_sha", "pre_bool_size", "extra_pre",
])
def test_v2_closed_binding_rejects_incomplete_or_foreign_metadata(pointer, mutation):
    def change(row):
        output = row["output_pointer_change"]
        if mutation == "missing_mode": del row["protocol_change_mode"]
        elif mutation == "unknown_mode": row["protocol_change_mode"] = "automatic"
        elif mutation == "extra_key": row["fallback"] = True
        elif mutation == "missing_pointer": del row["output_pointer_change"]
        elif mutation == "extra_pointer_key": output["tolerance"] = 1
        elif mutation == "wrong_leaf": output["field"] = "data.results"
        elif mutation == "wrong_before": output["before"] += "-foreign"
        elif mutation == "null_after": output["after"] = None
        elif mutation == "unchanged_after": output["after"] = output["before"]
        elif mutation == "wrong_project": output["after"] = str(Path(output["after"]).parents[4] / "foreign/ac17")
        elif mutation == "wrong_revision": output["after"] = output["after"].replace("a" * 32, "b" * 32)
        elif mutation == "wrong_run": output["after"] = str(Path(output["after"]).with_name("ac29"))
        elif mutation == "wrong_source_sha": row["source_configuration"]["sha256"] = "b" * 64
        elif mutation == "wrong_current_sha":
            row["current_configuration"]["sha256"] = "b" * 64
            row["apply_configurations"][-1]["sha256"] = "b" * 64
        elif mutation == "bool_size": row["apply_configurations"][0]["size_bytes"] = True
        elif mutation == "source_as_target": row["apply_configurations"][0] = row["source_configuration"]
        elif mutation == "current_not_merged": row["current_configuration"] = row["apply_configurations"][0]
        elif mutation == "bad_comparison_sha": row["comparison_sha256"] = "E" * 64
        elif mutation == "wrong_protocol_record": row["protocol_changes"]["protocols.injection"]["after"] = "foreign.csv"
        elif mutation == "three_as_zero": row["protocol_change_mode"] = "unchanged-zero-leaves"
        elif mutation == "wrong_format": row["format"] = subject.CONFIGURATION_BINDING_FORMAT
        elif mutation == "missing_pre": del row["pre_pointer_configurations"]
        elif mutation == "swapped_pre": row["pre_pointer_configurations"].reverse()
        elif mutation == "pre_outside": row["pre_pointer_configurations"][0]["path"] = str(pointer["layout"]["series"].parent / "runs/ac17.toml")
        elif mutation == "pre_as_target": row["pre_pointer_configurations"][0] = row["apply_configurations"][0]
        elif mutation == "wrong_pre_sha": row["pre_pointer_configurations"][0]["sha256"] = "b" * 64
        elif mutation == "pre_bool_size": row["pre_pointer_configurations"][0]["size_bytes"] = True
        elif mutation == "extra_pre": row["pre_pointer_configurations"].append(row["pre_pointer_configurations"][0])
        else: del row["protocol_changes"]["protocols.imaging"]
    _rewrite(pointer, change)
    before = inventory(pointer["layout"]["series"])
    with pytest.raises(ValueError):
        export(pointer["layout"])
    assert inventory(pointer["layout"]["series"]) == before
    assert not (pointer["layout"]["project"].root / "exports").exists()


@pytest.mark.parametrize("mutation", ["unrelated_value", "toml_type", "partial_pointer", "missing_rig", "extra_protocol"])
def test_hash_valid_config_differences_are_not_hidden_by_new_target_hashes(pointer, mutation):
    selected = pointer["bindings"]["ac17"]
    index = 0 if mutation == "partial_pointer" else 1
    path = Path(selected["apply_configurations"][index]["path"])
    value = tomllib.loads(path.read_text())
    if mutation == "unrelated_value": value["data"]["results"] += "-other"
    elif mutation == "toml_type": value["corrections"]["curvature"]["crop"]["top_left"][0] = 3.0
    elif mutation == "partial_pointer": value["rig"]["path"] += "-other"
    elif mutation == "missing_rig": del value["rig"]
    else: value["protocols"]["extra_clock"] = "not-authorized.csv"
    path.write_text(toml.dumps(value))
    def change(row):
        row["apply_configurations"][index] = descriptor(path)
        if index == 1:
            row["current_configuration"] = descriptor(path)
    _rewrite(pointer, change)
    with pytest.raises(ValueError):
        export(pointer["layout"])
    assert not (pointer["layout"]["project"].root / "exports").exists()


@pytest.mark.parametrize("fact", ["project_missing", "project_foreign", "revision_missing", "revision_invalid"])
def test_v2_needs_actual_project_and_revision_even_with_hash_valid_binding(pointer, fact):
    data = pointer["layout"]
    kwargs = dict(project_root=data["project"].root, revision="a" * 32)
    if fact == "project_missing": kwargs["project_root"] = None
    elif fact == "project_foreign": kwargs["project_root"] = data["series"] / "foreign_project"
    elif fact == "revision_missing": kwargs["revision"] = None
    else: kwargs["revision"] = "PENDING"
    with pytest.raises(ValueError, match="actual series project and revision"):
        subject._configuration_bundle(data["series"], "ac17", pointer["sources"]["ac17"], **kwargs)


@pytest.mark.parametrize("missing", ["results", "cache", "raw_cache", "baseline_cache"])
def test_missing_original_cache_boundary_stops_before_dispatch_without_writes(pointer, monkeypatch, missing):
    data = pointer["layout"]
    result = export(data)
    subject.apply_corrected_selected_run(data["project"], result)
    merged = Path(pointer["bindings"]["ac17"]["current_configuration"]["path"])
    results = Path(tomllib.loads(merged.read_text())["data"]["results"])
    path = {"results": results, "cache": results / "cache", "raw_cache": results / "raw_cache",
            "baseline_cache": results / "raw_cache/nested/baseline.npz"}[missing]
    path.rename(path.with_name(path.name + "-preserved-missing"))
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *_args: pytest.fail("dispatcher reached after missing-cache guard"))
    before = inventory(data["series"])
    with pytest.raises(ValueError, match="before dispatch|raw-cache"):
        dispatch.setup_rig(type("ExistingRig", (), {}), merged)
    assert inventory(data["series"]) == before
    assert not path.exists()


@pytest.mark.parametrize("state", ["absent", "empty", "populated", "file"])
def test_fresh_build_target_is_only_absent_or_empty_and_never_cleared(pointer, state):
    data = pointer["layout"]
    result = export(data)
    subject.apply_corrected_selected_run(data["project"], result)
    selected = pointer["bindings"]["ac17"]
    target = Path(selected["output_pointer_change"]["after"])
    if state in ("empty", "populated"):
        target.mkdir(parents=True)
    if state == "populated": (target / "partial.txt").write_bytes(b"preserved partial")
    elif state == "file":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"foreign file")
    before = inventory(data["series"])
    merged = Path(selected["current_configuration"]["path"])
    if state in ("absent", "empty"):
        assert subject.require_corrected_export_for_build(merged) == result.receipt
    else:
        with pytest.raises(ValueError, match="not empty"):
            subject.require_corrected_export_for_build(merged)
    assert inventory(data["series"]) == before


def test_v2_requires_apply_and_refuses_historical_build_overlay(pointer):
    data = pointer["layout"]
    result = export(data)
    selected = pointer["bindings"]["ac17"]
    # Before Apply the original labels do not point to any canonical export.
    merged = Path(selected["current_configuration"]["path"])
    before = inventory(data["series"])
    with pytest.raises(FileNotFoundError): subject.require_corrected_export_for_build(merged)
    assert inventory(data["series"]) == before
    subject.apply_corrected_selected_run(data["project"], result)
    with pytest.raises(ValueError, match="different or ambiguous"):
        subject.require_corrected_export_for_build([Path(pointer["sources"]["ac17"]["path"]), merged])


def test_effective_build_pointer_override_does_not_redirect_to_old_model(pointer, tmp_path):
    selected = pointer["bindings"]["ac17"]
    override = tmp_path / "foreign_overlay.toml"
    override.write_text(toml.dumps({"rig": {"path": selected["output_pointer_change"]["before"]}}))
    before = inventory(pointer["layout"]["series"])
    with pytest.raises(ValueError, match="entire effective configuration"):
        subject._require_fresh_v2_build([Path(selected["current_configuration"]["path"]), override], selected)
    assert inventory(pointer["layout"]["series"]) == before


def test_effective_data_overlay_cannot_change_the_original_cache_boundary(pointer, tmp_path):
    selected = pointer["bindings"]["ac17"]
    override = tmp_path / "foreign_data_overlay.toml"
    override.write_text(toml.dumps({"data": {"results": str(tmp_path)}}))
    before = inventory(pointer["layout"]["series"])
    with pytest.raises(ValueError, match="entire effective configuration"):
        subject._require_fresh_v2_build([Path(selected["current_configuration"]["path"]), override], selected)
    assert inventory(pointer["layout"]["series"]) == before


def test_zero_protocol_binding_cannot_claim_three_changes(pointer):
    data = pointer["layout"]
    manifest = tomllib.loads(pointer["manifest"].read_text())
    manifest["runs"]["geometry_configuration_bindings"]["ac29"]["protocol_change_mode"] = "clock-migrated-three-leaves"
    pointer["manifest"].write_text(toml.dumps(manifest))
    export(data)
    with pytest.raises(ValueError, match="exactly the three protocol"):
        export(data, "ac29")


@pytest.mark.parametrize("before", ["same", "ancestor", "descendant"])
def test_output_overlap_with_preserved_source_folder_is_rejected(pointer, before):
    data = pointer["layout"]
    row = deepcopy(pointer["bindings"]["ac17"])
    target = Path(row["output_pointer_change"]["after"])
    old = target if before == "same" else target.parent if before == "ancestor" else target / "old"
    row["output_pointer_change"]["before"] = str(old)
    with pytest.raises(ValueError, match="overlap"):
        subject._dual_configuration_binding_v2(row, "ac17", pointer["sources"]["ac17"],
            [item["path"] for item in row["apply_configurations"]], series=data["series"],
            project_root=data["project"].root, revision="a" * 32, destination=None, receipt=None)


@pytest.mark.parametrize("mutation", ["depth", "correction", "type", "extra_leaf", "protocol"])
def test_hash_valid_partial_must_match_its_own_preserved_before_document(pointer, mutation):
    partial = Path(pointer["bindings"]["ac17"]["apply_configurations"][0]["path"])
    value = tomllib.loads(partial.read_text())
    if mutation == "depth": value["depth"]["depth_map"] += "-other"
    elif mutation == "correction": value["corrections"]["curvature"]["crop"]["top_left"][0] = 8
    elif mutation == "type": value["corrections"]["curvature"]["crop"]["top_left"][0] = 3.0
    elif mutation == "extra_leaf": value["corrections"]["unbound_leaf"] = True
    else: value["protocols"]["imaging_mode"] = "other"
    partial.write_text(toml.dumps(value))
    _rewrite(pointer, lambda row: row["apply_configurations"].__setitem__(0, descriptor(partial)))
    before = inventory(pointer["layout"]["series"])
    with pytest.raises(ValueError, match="own preserved rig.path migration"):
        export(pointer["layout"])
    assert inventory(pointer["layout"]["series"]) == before
    assert not (pointer["layout"]["project"].root / "exports").exists()


@pytest.mark.parametrize("order", ["common_partial", "partial_common_merged", "merged_common_partial"])
def test_genuine_smaller_partial_plus_common_has_exact_effective_build_config(pointer, order):
    data = pointer["layout"]
    selected = pointer["bindings"]["ac17"]
    partial = Path(selected["apply_configurations"][0]["path"])
    merged = Path(selected["current_configuration"]["path"])
    full = tomllib.loads(merged.read_text())
    common = data["series"] / "config/common.toml"
    common.write_text(toml.dumps(full))
    smaller = {"rig": deepcopy(full["rig"]), "data": {"results": full["data"]["results"]},
               "protocols": {"imaging_mode": full["protocols"]["imaging_mode"]}}
    partial.write_text(toml.dumps(smaller))
    before_partial = Path(selected["pre_pointer_configurations"][0]["path"])
    preserved_smaller = deepcopy(smaller)
    preserved_smaller["rig"]["path"] = selected["output_pointer_change"]["before"]
    before_partial.write_text(toml.dumps(preserved_smaller))
    manifest = tomllib.loads(pointer["manifest"].read_text())
    manifest.setdefault("paths", {})["common_config"] = common.relative_to(data["series"]).as_posix()
    manifest["runs"]["geometry_configuration_bindings"]["ac17"]["apply_configurations"][0] = descriptor(partial)
    manifest["runs"]["geometry_configuration_bindings"]["ac17"]["pre_pointer_configurations"][0] = descriptor(before_partial)
    pointer["manifest"].write_text(toml.dumps(manifest))
    assert set(smaller) < set(full)
    assert set(smaller["data"]) < set(full["data"])
    source_before = Path(pointer["sources"]["ac17"]["path"]).read_bytes()
    common_before = common.read_bytes()
    result = export(data)
    subject.apply_corrected_selected_run(data["project"], result)
    paths = {"common_partial": [common, partial],
             "partial_common_merged": [partial, common, merged],
             "merged_common_partial": [merged, common, partial]}[order]
    assert subject.require_corrected_export_for_build(paths) == result.receipt
    assert Path(pointer["sources"]["ac17"]["path"]).read_bytes() == source_before
    assert common.read_bytes() == common_before
    assert not Path(selected["output_pointer_change"]["after"]).exists()


@pytest.mark.parametrize("section", ["depth", "correction", "type", "extra"])
def test_declared_common_overlay_cannot_change_other_effective_sections_after_apply(pointer, section):
    data = pointer["layout"]
    selected = pointer["bindings"]["ac17"]
    common = data["series"] / "config/common.toml"
    if section == "depth": value = {"depth": {"depth_map": "foreign-depth.npz"}}
    elif section == "correction": value = {"corrections": {"curvature": {"crop": {"top_left": [8, 8]}}}}
    elif section == "type": value = {"corrections": {"curvature": {"crop": {"top_left": [3.0, 8.0]}}}}
    else: value = {"unbound_extra": {"value": True}}
    common.write_text(toml.dumps(value))
    manifest = tomllib.loads(pointer["manifest"].read_text())
    manifest.setdefault("paths", {})["common_config"] = common.relative_to(data["series"]).as_posix()
    pointer["manifest"].write_text(toml.dumps(manifest))
    result = export(data)
    subject.apply_corrected_selected_run(data["project"], result)
    before = inventory(data["series"])
    with pytest.raises(ValueError, match="entire effective configuration"):
        subject.require_corrected_export_for_build([Path(selected["current_configuration"]["path"]), common])
    assert inventory(data["series"]) == before


def test_historical_partial_selection_and_metadata_are_preserved_against_their_own_before(pointer):
    data = pointer["layout"]
    selected = pointer["bindings"]["ac17"]
    prior = Path(selected["pre_pointer_configurations"][0]["path"])
    target = Path(selected["apply_configurations"][0]["path"])
    original = tomllib.loads(prior.read_text())
    original["comparison_grid_v1"] = {"selection_manifest": "historical-metadata-only.json", "sha256": "c" * 64}
    original["data_interval"] = [{"name": "historical_partial", "start": "00:00:00", "num": 7}]
    original["data_time"] = [{"name": "historical_times", "times": ["00:01:00"]}]
    prior.write_text(toml.dumps(original))
    after_pointer = deepcopy(original)
    after_pointer["rig"]["path"] = selected["output_pointer_change"]["after"]
    target.write_text(toml.dumps(after_pointer))
    def change(row):
        row["pre_pointer_configurations"][0] = descriptor(prior)
        row["apply_configurations"][0] = descriptor(target)
    _rewrite(pointer, change)
    prior_bytes, target_bytes = prior.read_bytes(), target.read_bytes()
    result = export(data)
    subject.apply_corrected_selected_run(data["project"], result)
    assert prior.read_bytes() == prior_bytes
    assert target.read_bytes() == subject.corrected_config_bytes(target_bytes, result.receipt["binding"]["replacements"])
    actual = tomllib.loads(target.read_text())
    for key in ("comparison_grid_v1", "data_interval", "data_time"):
        assert actual[key] == original[key]
    merged = Path(selected["current_configuration"]["path"])
    assert subject.require_corrected_export_for_build(merged) == result.receipt
    with pytest.raises(ValueError, match="entire effective configuration"):
        subject.require_corrected_export_for_build([merged, target])


@pytest.mark.parametrize("change", ["depth", "type", "extra_protocol"])
def test_merged_before_keeps_independent_source_zero_or_three_protocol_proof(pointer, change):
    selected = pointer["bindings"]["ac17"]
    prior = Path(selected["pre_pointer_configurations"][1]["path"])
    target = Path(selected["current_configuration"]["path"])
    before = tomllib.loads(prior.read_text())
    if change == "depth": before["depth"]["depth_map"] += "-other"
    elif change == "type": before["corrections"]["curvature"]["crop"]["top_left"][0] = 3.0
    else: before["protocols"]["extra_clock"] = "foreign.csv"
    prior.write_text(toml.dumps(before))
    after = deepcopy(before)
    after["rig"]["path"] = selected["output_pointer_change"]["after"]
    target.write_text(toml.dumps(after))
    def rebind(row):
        row["pre_pointer_configurations"][1] = descriptor(prior)
        row["current_configuration"] = descriptor(target)
        row["apply_configurations"][1] = descriptor(target)
    _rewrite(pointer, rebind)
    with pytest.raises(ValueError, match="non-protocol value or TOML type"):
        export(pointer["layout"])
    assert not (pointer["layout"]["project"].root / "exports").exists()


@pytest.mark.parametrize("index", [0, 1])
def test_changed_preserved_before_file_after_export_blocks_apply_before_writes(pointer, index):
    data = pointer["layout"]
    result = export(data)
    prior = Path(pointer["bindings"]["ac17"]["pre_pointer_configurations"][index]["path"])
    prior.write_bytes(prior.read_bytes() + b"# external edit\n")
    before = inventory(data["series"])
    with pytest.raises(ValueError, match="identity mismatch"):
        subject.apply_corrected_selected_run(data["project"], result)
    assert inventory(data["series"]) == before
    assert not (result.destination / "apply-claim.json").exists()


@pytest.mark.parametrize("role", ["source", "partial_before", "merged_before"])
def test_v2_parses_only_the_bytes_whose_SHA_and_length_it_verifies(pointer, monkeypatch, role):
    data = pointer["layout"]
    selected = pointer["bindings"]["ac17"]
    chosen = (selected["source_configuration"] if role == "source" else
              selected["pre_pointer_configurations"][0 if role == "partial_before" else 1])
    path = Path(chosen["path"])
    original = path.read_bytes()
    replacement = original.replace(b" = ", b"=  ", 1)
    assert replacement != original and len(replacement) == len(original)
    subject._verified(chosen)  # A real earlier metadata admission saw the old bytes.
    read, changed = subject._small, []
    def change_between_admission_and_consumption(value):
        if subject._same(value, path) and not changed:
            changed.append(str(path))
            path.write_bytes(replacement)  # Same length and parsed values; a different SHA.
        return read(value)
    monkeypatch.setattr(subject, "_small", change_between_admission_and_consumption)
    with pytest.raises(ValueError, match="consumed configuration bytes.*SHA/size"):
        subject._dual_configuration_binding_v2(selected, "ac17", pointer["sources"]["ac17"],
            [row["path"] for row in selected["apply_configurations"]], series=data["series"],
            project_root=data["project"].root, revision="a" * 32, destination=None, receipt=None)
    assert changed == [str(path)]
    assert tomllib.loads(path.read_text()) == tomllib.loads(original.decode())
    assert not (data["project"].root / "exports").exists()


@pytest.fixture
def Q121_private_output_binding_TESTONLY(tmp_path,monkeypatch):
    """TOML/dispatch seam only. Real editor provenance is tested separately.

    The manual-source reader is an explicit unit double here; this fixture
    cannot qualify a real Store, GUI, corrected map or Build.
    """
    from darsia.fluidflower import geometry_raster_sources as sources
    series=tmp_path/"Q121-private-series";root=series/"geometry_facies"
    run="ac17";revision="b"*32
    source=tmp_path/"original-config/ac17.toml";source.parent.mkdir()
    raw=("# original bytes remain except TWO declared pointer lines\r\n"+toml.dumps(dict(
        rig=dict(path=str(tmp_path/"original-rig")),
        data=dict(results=str(tmp_path/"original-results"),use_cache=False),
        labeling=dict(labels="unchanged-original-labels"),
        calibration=dict(exact_original_float=.123))).replace("\n","\r\n")).encode()
    source.write_bytes(raw)
    changes={"rig.path":dict(field="rig.path",before=str(tmp_path/"original-rig"),
        after=str(root/"results/rig_models"/revision/run)),
        "data.results":dict(field="data.results",before=str(tmp_path/"original-results"),
        after=str(root/"results/native_data"/revision/run))}
    previous=[];targets=[]
    for role in ("runs","merged"):
        old=series/"config/pre-pointer"/role/(run+".toml");old.parent.mkdir(parents=True)
        old.write_bytes(raw);previous.append(descriptor(old))
        target=series/"config"/role/(run+".toml");target.parent.mkdir(parents=True)
        target.write_bytes(subject._private_manual_output_bytes(raw,changes));targets.append(descriptor(target))
    record=root/"results/geometry"/revision/"revision.json";record.parent.mkdir(parents=True)
    current=dict(TESTONLY_seam=True,revision_id=revision);record.write_text(json.dumps(current))
    calls=[]
    def actual_manual_source_reader(project_root,document,requested):
        assert project_root==root and document==current and requested==run
        calls.append((project_root,document,requested))
        return dict(format=sources.MANUAL_LABEL_EDIT,run_id=run,revision_id=revision)
    monkeypatch.setattr(sources,"private_manual_label_edit_sources",actual_manual_source_reader)
    value=dict(format=subject.PRIVATE_MANUAL_CONFIGURATION_FORMAT,run_id=run,
        source_configuration=descriptor(source),current_configuration=targets[-1],
        apply_configurations=targets,pre_pointer_configurations=previous,
        output_pointer_changes=changes,manual_revision=descriptor(record),comparison_sha256="c"*64)
    manifest=dict(runs=dict(ids=[run],geometry_configuration_bindings={run:value}))
    (series/"series.toml").write_text(toml.dumps(manifest))
    return dict(series=series,root=root,run=run,revision=revision,value=value,raw=raw,calls=calls)


def _Q121_checked_TESTONLY(data,value=None):
    return subject._private_manual_configuration_binding(value or data["value"],data["run"],
        data["value"]["source_configuration"],[row["path"] for row in data["value"]["apply_configurations"]],
        series=data["series"],project_root=data["root"],revision=data["revision"],destination=None,receipt=None)


def test_Q121_private_output_binding_routes_to_actual_manual_reader_and_keeps_original_bytes(Q121_private_output_binding_TESTONLY):
    data=Q121_private_output_binding_TESTONLY
    assert _Q121_checked_TESTONLY(data)==data["value"] and len(data["calls"])==1
    bundle=subject._configuration_bundle(data["series"],data["run"],data["value"]["source_configuration"],
        project_root=data["root"],revision=data["revision"])
    assert bundle["dual_configuration_binding"]==data["value"] and len(data["calls"])==2
    for row in data["value"]["pre_pointer_configurations"]:
        assert Path(row["path"]).read_bytes()==data["raw"]


@pytest.mark.parametrize("mutation",["foreign_output","extra_value","missing_manual","foreign_manual","changed_pre_pointer"])
def test_Q121_private_output_binding_refuses_foreign_or_unproven_changes(Q121_private_output_binding_TESTONLY,mutation):
    data=Q121_private_output_binding_TESTONLY;value=deepcopy(data["value"])
    if mutation=="foreign_output":value["output_pointer_changes"]["data.results"]["after"]=str(data["series"].parent/"foreign-results")
    elif mutation=="missing_manual":del value["manual_revision"]
    elif mutation=="foreign_manual":value["manual_revision"]["path"]=str(data["series"].parent/"foreign-revision.json")
    elif mutation=="changed_pre_pointer":
        path=Path(value["pre_pointer_configurations"][0]["path"])
        path.write_bytes(path.read_bytes()+b"# not original bytes\n")
        value["pre_pointer_configurations"][0]=descriptor(path)
    else:
        for index,row in enumerate(value["apply_configurations"]):
            path=Path(row["path"]);path.write_bytes(path.read_bytes().replace(b"0.123",b"0.124"))
            value["apply_configurations"][index]=descriptor(path)
        value["current_configuration"]=value["apply_configurations"][-1]
    with pytest.raises(ValueError):_Q121_checked_TESTONLY(data,value)


def test_Q121_private_binding_cannot_skip_failed_manual_provenance(Q121_private_output_binding_TESTONLY,monkeypatch):
    from darsia.fluidflower import geometry_raster_sources as sources
    data=Q121_private_output_binding_TESTONLY
    def reject(*_args):raise ValueError("TESTONLY actual manual source not proven")
    monkeypatch.setattr(sources,"private_manual_label_edit_sources",reject)
    with pytest.raises(ValueError,match="manual source not proven"):_Q121_checked_TESTONLY(data)


def test_Q121_two_output_leaves_still_fail_UNCHANGED_ordinary_dual_V2(Q121_private_output_binding_TESTONLY):
    data=Q121_private_output_binding_TESTONLY;private=data["value"]
    old={key:deepcopy(private[key]) for key in ("run_id","source_configuration","current_configuration",
        "apply_configurations","pre_pointer_configurations","comparison_sha256")}
    old.update(format=subject.CONFIGURATION_BINDING_V2_FORMAT,protocol_change_mode="unchanged-zero-leaves",
        protocol_changes={},output_pointer_change=private["output_pointer_changes"]["rig.path"])
    with pytest.raises(ValueError,match="beyond its own preserved rig.path"):
        subject._dual_configuration_binding_v2(old,data["run"],private["source_configuration"],
            [row["path"] for row in private["apply_configurations"]],series=data["series"],
            project_root=data["root"],revision=data["revision"],destination=None,receipt=None)


def test_Q121_private_build_reuses_original_fresh_effective_config_checks(Q121_private_output_binding_TESTONLY):
    data=Q121_private_output_binding_TESTONLY
    checked=_Q121_checked_TESTONLY(data)
    Path(checked["output_pointer_changes"]["data.results"]["after"]).mkdir(parents=True)
    subject._require_fresh_private_manual_build(Path(checked["current_configuration"]["path"]),checked)
    rig=Path(checked["output_pointer_changes"]["rig.path"]["after"]);rig.mkdir(parents=True)
    (rig/"partial-must-be-preserved").write_bytes(b"not an empty output")
    with pytest.raises(ValueError,match="not empty"):
        subject._require_fresh_private_manual_build(Path(checked["current_configuration"]["path"]),checked)
