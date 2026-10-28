"""Q60 metadata/transaction contracts with existing tiny synthetic geometry.

No campaign data, geometry registration or actual Rig Build is performed.
"""
from copy import deepcopy
import json
from pathlib import Path
import tomllib

import pytest
import toml

from darsia.fluidflower import corrected_setup as subject
from darsia.fluidflower.geometry_workspace_frame import adopted_content_digest
from .test_corrected_setup_canonical_layout import descriptor, export, inventory, layout


@pytest.fixture
def dual(layout):
    series = layout["series"]
    manifest_path = series / "series.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    manifest["runs"].update(config_files={}, merged_files={}, geometry_configuration_bindings={})
    old, current, before = {}, {}, {}
    for run in ("ac17", "ac29"):
        source = series / "config/merged" / (run + ".toml")
        original = tomllib.loads(source.read_text())
        original["protocols"] = dict(imaging={f"raw/{run}": f"old/{run}/imaging.csv"},
                                     injection=f"old/{run}/injection.csv", imaging_mode="auto")
        raw = ("# immutable physical source\r\n" + toml.dumps(original).replace("\n", "\r\n")).encode()
        source.write_bytes(raw)
        (series / "config/runs" / (run + ".toml")).write_bytes(raw)
        layout["runs"][run]["configuration"] = descriptor(source)
        old[run] = descriptor(source)
        expected = deepcopy(original)
        expected["protocols"] = dict(imaging={f"raw/{run}": f"gui/{run}/imaging.csv"},
                                     injection=f"gui/{run}/injection.csv", imaging_mode="authored")
        target_raw = ("# current GUI formatting retained\r\n" + toml.dumps(expected).replace("\n", "\r\n")).encode()
        targets = []
        for kind in ("runs", "merged"):
            target = series / "config/gui_clock" / kind / (run + ".toml")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(target_raw)
            targets.append(descriptor(target))
            before[str(target)] = target_raw
            manifest["runs"]["config_files" if kind == "runs" else "merged_files"][run] = target.relative_to(series).as_posix()
        changes = {f"protocols.{key}": dict(before=original["protocols"][key], after=expected["protocols"][key])
                   for key in ("imaging", "injection", "imaging_mode")}
        binding = dict(format=subject.CONFIGURATION_BINDING_FORMAT, run_id=run,
                       source_configuration=old[run], current_configuration=targets[-1],
                       apply_configurations=targets, protocol_changes=changes, comparison_sha256="e" * 64)
        manifest["runs"]["geometry_configuration_bindings"][run] = binding
        current[run] = binding
    manifest_path.write_text(toml.dumps(manifest), encoding="utf-8")
    revision = json.loads(layout["current"].read_bytes())
    config = revision["configuration"]
    config["frame_binding"]["runs"] = deepcopy(layout["runs"])
    config["provenance"]["adopted_content_sha256"] = adopted_content_digest(config)
    layout["current"].write_text(json.dumps(revision))
    return dict(layout=layout, source=old, selected=current, before=before,
                manifest=manifest_path, revision_before=layout["current"].read_bytes())


def _rewrite(dual, change):
    value = tomllib.loads(dual["manifest"].read_text())
    change(value)
    dual["manifest"].write_text(toml.dumps(value), encoding="utf-8")


def test_dual_export_keeps_source_and_binds_exact_current_bundle(dual):
    result = export(dual["layout"])
    binding = result.receipt["binding"]
    assert binding["configuration"] == dual["source"]["ac17"]
    assert binding["dual_configuration_binding"] == dual["selected"]["ac17"]
    assert binding["series_manifest"] == descriptor(dual["manifest"])
    assert dual["layout"]["current"].read_bytes() == dual["revision_before"]
    assert all(Path(row["path"]).read_bytes() == dual["before"][row["path"]]
               for row in dual["selected"]["ac17"]["apply_configurations"])


def test_apply_uses_current_bytes_preserving_protocols_comments_and_historical_source(dual):
    data = dual["layout"]
    sources = {key: Path(row["path"]).read_bytes() for key, row in dual["source"].items()}
    result = export(data)
    receipt = subject.apply_corrected_selected_run(data["project"], result)
    record = json.loads(receipt.read_bytes())
    assert len(record["files"]) == 2
    for row in dual["selected"]["ac17"]["apply_configurations"]:
        path = Path(row["path"])
        original = dual["before"][str(path)]
        expected = subject.corrected_config_bytes(original, result.receipt["binding"]["replacements"])
        assert path.read_bytes() == expected
        assert path.read_bytes().startswith(b"# current GUI formatting retained\r\n")
        assert tomllib.loads(expected.decode())["protocols"] == tomllib.loads(original.decode())["protocols"]
    assert all(Path(row["path"]).read_bytes() == sources[key] for key, row in dual["source"].items())
    assert data["current"].read_bytes() == dual["revision_before"]
    assert descriptor(dual["manifest"]) == result.receipt["binding"]["series_manifest"]


def test_current_bundle_build_and_idempotent_export_apply_use_genuine_after_receipts(dual):
    data = dual["layout"]
    result = export(data)
    first = subject.apply_corrected_selected_run(data["project"], result)
    merged = Path(dual["selected"]["ac17"]["current_configuration"]["path"])
    assert subject.require_corrected_export_for_build(merged) == result.receipt
    before = inventory(data["series"])
    assert subject.apply_corrected_selected_run(data["project"], result) == first
    assert export(data).reused
    assert inventory(data["series"]) == before


@pytest.mark.parametrize("mutation", ["missing", "wrong_source_sha", "wrong_current_sha", "wrong_partial_sha",
                                    "wrong_run", "source_as_target", "current_not_merged", "descriptor_bool_size",
                                    "wrong_protocol_record", "missing_protocol_record", "unknown_field", "proof_sha"])
def test_malformed_or_incomplete_dual_binding_refuses_before_export(dual, mutation):
    def change(manifest):
        table = manifest["runs"]["geometry_configuration_bindings"]
        row = table["ac17"]
        if mutation == "missing": del table["ac17"]
        elif mutation == "wrong_source_sha": row["source_configuration"]["sha256"] = "b" * 64
        elif mutation == "wrong_current_sha":
            row["current_configuration"]["sha256"] = "b" * 64
            row["apply_configurations"][-1]["sha256"] = "b" * 64
        elif mutation == "wrong_partial_sha": row["apply_configurations"][0]["sha256"] = "b" * 64
        elif mutation == "wrong_run": row["run_id"] = "ac29"
        elif mutation == "source_as_target": row["apply_configurations"][0] = row["source_configuration"]
        elif mutation == "current_not_merged": row["current_configuration"] = row["apply_configurations"][0]
        elif mutation == "descriptor_bool_size": row["apply_configurations"][0]["size_bytes"] = True
        elif mutation == "wrong_protocol_record": row["protocol_changes"]["protocols.imaging"]["after"] = "foreign.csv"
        elif mutation == "missing_protocol_record": del row["protocol_changes"]["protocols.imaging"]
        elif mutation == "unknown_field": row["override"] = True
        else: row["comparison_sha256"] = "E" * 64
    _rewrite(dual, change)
    before = inventory(dual["layout"]["series"])
    with pytest.raises(ValueError): export(dual["layout"])
    assert inventory(dual["layout"]["series"]) == before
    assert not (dual["layout"]["project"].root / "exports").exists()


@pytest.mark.parametrize("mutation", ["unrelated", "toml_type", "protocol_unchanged", "protocol_missing", "extra_protocol"])
def test_hash_valid_other_difference_cannot_be_disguised_by_rebinding_current_sha(dual, mutation):
    row = dual["selected"]["ac17"]
    path = Path(row["current_configuration"]["path"])
    config = tomllib.loads(path.read_text())
    if mutation == "unrelated": config["data"]["results"] = "silently-new-output"
    elif mutation == "toml_type": config["corrections"]["curvature"]["crop"]["top_left"][0] = 3.0
    elif mutation == "protocol_unchanged": config["protocols"]["imaging_mode"] = "auto"
    elif mutation == "protocol_missing": del config["protocols"]["imaging_mode"]
    else: config["protocols"]["extra_clock"] = "invented.csv"
    path.write_text(toml.dumps(config))
    def change(manifest):
        binding = manifest["runs"]["geometry_configuration_bindings"]["ac17"]
        binding["current_configuration"] = descriptor(path)
        binding["apply_configurations"][-1] = descriptor(path)
    _rewrite(dual, change)
    with pytest.raises(ValueError, match="protocol|TOML type"): export(dual["layout"])
    assert not (dual["layout"]["project"].root / "exports").exists()


@pytest.mark.parametrize("target", ["source", "partial", "manifest", "parent"])
def test_changed_identity_after_export_cannot_apply_any_config(dual, target):
    result = export(dual["layout"])
    path = {"source": Path(dual["source"]["ac17"]["path"]),
            "partial": Path(dual["selected"]["ac17"]["apply_configurations"][0]["path"]),
            "manifest": dual["manifest"], "parent": dual["layout"]["current"]}[target]
    path.write_bytes(path.read_bytes() + b"\n")
    before = inventory(dual["layout"]["series"])
    with pytest.raises((ValueError, RuntimeError)):
        subject.apply_corrected_selected_run(dual["layout"]["project"], result)
    assert inventory(dual["layout"]["series"]) == before
    assert not (result.destination / "apply-claim.json").exists()


@pytest.mark.parametrize("mixed", [False, True])
def test_historical_source_cannot_be_a_build_overlay_even_after_valid_apply(dual, mixed):
    result = export(dual["layout"])
    subject.apply_corrected_selected_run(dual["layout"]["project"], result)
    old = Path(dual["source"]["ac17"]["path"])
    current = Path(dual["selected"]["ac17"]["current_configuration"]["path"])
    # Keep a corrected selected config last so the guard reads its real export.
    with pytest.raises(ValueError, match="different or ambiguous selected-run"):
        subject.require_corrected_export_for_build([current, old, current] if mixed else [old, current])


def test_changed_target_after_apply_is_not_current_binding_or_new_expected_bytes(dual):
    result = export(dual["layout"])
    subject.apply_corrected_selected_run(dual["layout"]["project"], result)
    path = Path(dual["selected"]["ac17"]["current_configuration"]["path"])
    path.write_bytes(path.read_bytes() + b"# later edit\n")
    before = inventory(dual["layout"]["series"])
    with pytest.raises(ValueError, match="identity mismatch"):
        subject.require_corrected_export_for_build(path)
    assert inventory(dual["layout"]["series"]) == before


@pytest.mark.parametrize("kind", ["another_run", "common"])
def test_explicit_dual_metadata_does_not_waive_bundle_ambiguity(dual, kind):
    def change(manifest):
        target = manifest["runs"]["config_files"]["ac17"]
        if kind == "another_run": manifest["runs"]["config_files"]["ac29"] = target
        else: manifest.setdefault("paths", {})["common_config"] = target
    _rewrite(dual, change)
    with pytest.raises(ValueError, match="ambiguous"):
        export(dual["layout"])
    assert not (dual["layout"]["project"].root / "exports").exists()


def test_current_target_change_during_export_is_preserved_without_publishing(dual, monkeypatch):
    path = Path(dual["selected"]["ac17"]["current_configuration"]["path"])
    original = subject.shutil.copyfile
    seen = []
    def change_after_copy(source, target):
        value = original(source, target)
        if not seen:
            seen.append(True)
            path.write_bytes(path.read_bytes() + b"# concurrent GUI edit\n")
        return value
    monkeypatch.setattr(subject.shutil, "copyfile", change_after_copy)
    with pytest.raises(ValueError, match="identity mismatch"):
        export(dual["layout"])
    expected = dual["layout"]["project"].root / "exports/native_setup" / ("a" * 32)
    assert not expected.exists()
    assert list(expected.parent.glob(".ac17.*.tmp"))  # Failed staging is preserved.


def test_historical_source_change_during_apply_is_failed_with_preserved_partials(dual, monkeypatch):
    result = export(dual["layout"])
    path = Path(dual["source"]["ac17"]["path"])
    replace, calls = subject.os.replace, []
    def change_after_second(source, target):
        value = replace(source, target)
        calls.append(target)
        if len(calls) == 2:
            path.write_bytes(path.read_bytes() + b"# external source edit\n")
        return value
    monkeypatch.setattr(subject.os, "replace", change_after_second)
    with pytest.raises(subject.CorrectedApplyError, match="identity mismatch") as error:
        subject.apply_corrected_selected_run(dual["layout"]["project"], result)
    record = json.loads(error.value.receipt.read_bytes())
    assert record["state"] == "failed" and len(record["files"]) == 2
    assert len(list(error.value.receipt.parent.glob("*-before.toml"))) == 2
    assert (result.destination / "apply-claim.json").is_file()
    assert not (result.destination / "apply-complete.json").exists()


def test_reference_and_child_have_separate_current_targets_and_original_sources(dual):
    data = dual["layout"]
    first = export(data)
    child = export(data, "ac29")
    source_before = {run: Path(row["path"]).read_bytes() for run, row in dual["source"].items()}
    for result in (first, child):
        subject.apply_corrected_selected_run(data["project"], result)
        current = Path(dual["selected"][result.run_id]["current_configuration"]["path"])
        assert subject.require_corrected_export_for_build(current) == result.receipt
    assert child.receipt["root_export"] == descriptor(first.destination / "export.json")
    assert all(Path(row["path"]).read_bytes() == source_before[run] for run, row in dual["source"].items())
