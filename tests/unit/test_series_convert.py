"""Ordinary synthetic metadata conversion tests; no model fit or image IO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from darsia.fluidflower import series_convert as convert


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def fixture(tmp_path, *, existing_merge=False):
    root=tmp_path/"series"
    manifest=write(root/"series.toml", b'schema_version=1\nseries_id="Synthetic"\n'
        b'[runs]\nids=["one"]\n[runs.config_files]\none="config/runs/one.toml"\n'
        b'[paths]\ncommon_config="config/common.toml"\nmodels="models"\n'
        b'[unknown]\nkeep="unchanged"\n')
    write(root/"config/common.toml", b'[rig]\nwidth=1.0\nheight=0.5\n')
    write(root/"config/runs/one.toml", b'[data]\nformat="JPG"\n')
    if existing_merge:
        write(root/"config/merged/one.toml", b'[rig]\nwidth=1.0\nheight=0.5\n[data]\nformat="JPG"\n')
    choices=dict(schema_version=1, series_id="Synthetic", operation="fixture",
        expected_series_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        specification="Synthetic metadata fixture", model_sources=[])
    return root,choices


def test_original_named_registry_merge_retains_accumulation_and_duplicate_guard(tmp_path):
    common=write(tmp_path/"common.toml", b'[[data_time]]\nname="base"\n'
        b'[depth]\nmeasurements="original.csv"\n')
    partial=write(tmp_path/"partial.toml", b'[[data_time]]\nname="selected"\n'
        b'[depth]\ndepth_map="existing.npy"\n')
    actual=convert.native_merged([common,partial])
    assert [row["name"] for row in actual["data_time"]]==["base","selected"]
    assert actual["depth"]=={"measurements":"original.csv","depth_map":"existing.npy"}
    partial.write_bytes(b'[[data_time]]\nname="base"\n')
    with pytest.raises(AssertionError):
        convert.native_merged([common,partial])


def test_no_method_choice_means_no_method_or_native_is_invented(tmp_path):
    root,choices=fixture(tmp_path)
    before=(root/"series.toml").read_bytes()
    plan=convert.plan_series_conversion(root,choices)
    manifest=next(row for row in plan["writes"] if row["relative_path"]=="series.toml")
    actual=tomllib.loads(manifest["after_utf8"])
    assert "method" not in actual and "native" not in actual
    assert actual["unknown"]=={"keep":"unchanged"}
    assert not plan["complete"] and not plan["scientific_computation"]
    assert any("Depth" in p for p in plan["pending"])
    assert (root/"series.toml").read_bytes()==before
    assert not (root/"config/merged/one.toml").exists()


def test_explicit_method_does_not_choose_newest_model_or_run_setup(tmp_path,monkeypatch):
    root,choices=fixture(tmp_path)
    choices["active_method"]="phase_model"
    plan=convert.plan_series_conversion(root,choices)
    manifest=next(row for row in plan["writes"] if row["relative_path"]=="series.toml")
    actual=tomllib.loads(manifest["after_utf8"])
    assert actual["method"]=={"active":"phase_model"}
    assert any("needs_decision" in p for p in plan["pending"])
    assert not (root/"results").exists()
    assert not (root/"geometry_facies").exists()


def test_image_blacklist_reports_images_and_cannot_be_run_exclusion_source(tmp_path):
    root,choices=fixture(tmp_path)
    blacklist=write(root/"protocols/one/blacklist_protocol.csv",b"path\nphoto_1.JPG\nphoto_2.JPG\n")
    extras=convert.inspect_protocol_extras(blacklist.parent)
    assert extras[0]["row_count"]==2
    assert extras[0]["affects_run_exclusion"] is False
    assert extras[0]["ignored_for_protocol_completion"] is True
    plan=convert.plan_series_conversion(root,choices)
    manifest=next(row for row in plan["writes"] if row["relative_path"]=="series.toml")
    assert "excluded" not in tomllib.loads(manifest["after_utf8"])["runs"]
    choices.update(excluded_runs={"one":"not derived from image names"},
                   run_exclusion_sources=[convert._descriptor(blacklist)])
    with pytest.raises(ValueError,match="Image blacklist"):
        convert.plan_series_conversion(root,choices)


def test_existing_merged_drift_is_reported_and_not_rewritten(tmp_path):
    root,choices=fixture(tmp_path,existing_merge=True)
    merged=root/"config/merged/one.toml"
    merged.write_bytes(b'[rig]\nwidth=9.0\nheight=0.5\n[data]\nformat="JPG"\n')
    before=merged.read_bytes()
    plan=convert.plan_series_conversion(root,choices)
    assert plan["merged_configurations"][0]["semantic_exact"] is False
    assert plan["merged_configurations"][0]["exact_unified_diff"]
    assert not any(row["relative_path"]=="config/merged/one.toml" for row in plan["writes"])
    assert any("no overwrite" in item for item in plan["pending"])
    assert merged.read_bytes()==before


def model(root, name="phase-model-fixture", *, complete=True):
    source=root/"models"/name
    write(source/"summary.json",json.dumps(dict(model_id=name,
        inventory=dict(datasets=["Synthetic"]))).encode())
    write(source/"models/adapter/co2_presence_signal.json",b"{}")
    if complete:
        write(source/"models/adapter/gas_within_co2_signal.json",b"{}")
    return source


def test_original_model_indexer_only_used_for_valid_complete_explicit_sources(tmp_path,monkeypatch):
    root,choices=fixture(tmp_path)
    good=model(root)
    bad=model(root,"phase-model-incomplete",complete=False)
    choices["model_sources"]=[str(good.relative_to(root)),str(bad.relative_to(root))]
    original={str(p):p.read_bytes() for p in root.rglob("*.json")}
    plan=convert.plan_series_conversion(root,choices)
    assert len(plan["model_references"])==1
    assert len(plan["invalid_or_incomplete_models"])==1
    assert plan["invalid_or_incomplete_models"][0]["indexed"] is False
    assert not (root/"models/phase").exists()
    receipt=convert.apply_metadata_conversion(plan)
    pointer=root/plan["model_references"][0]["target_relative"]
    entry=json.loads(pointer.read_bytes())
    assert entry["format"]=="phase-model-library-reference-v1"
    assert entry["source_identity"]==plan["model_references"][0]["source_identity"]
    assert len(list((root/"models/phase").glob("*.json")))==1
    assert all(Path(path).read_bytes()==raw for path,raw in original.items())
    assert not receipt["complete"] and receipt["metadata_steps_applied"]


def test_explicit_apply_cas_rejects_external_edit_and_keeps_original(tmp_path):
    root,choices=fixture(tmp_path)
    plan=convert.plan_series_conversion(root,choices)
    current=(root/"series.toml").read_bytes()+b"\n# external edit\n"
    (root/"series.toml").write_bytes(current)
    with pytest.raises(ValueError,match="Conversion input changed"):
        convert.apply_metadata_conversion(plan)
    assert (root/"series.toml").read_bytes()==current
    assert not (root/"config/merged/one.toml").exists()


def test_repeated_metadata_conversion_has_no_replacement_writes(tmp_path):
    root,choices=fixture(tmp_path)
    plan=convert.plan_series_conversion(root,choices)
    before=(root/"series.toml").read_bytes()
    receipt=convert.apply_metadata_conversion(plan)
    assert (root/"config/merged/one.toml").is_file()
    assert list((root/"config/_previous").glob("*/series.toml"))[0].read_bytes()==before
    again=convert.plan_series_conversion(root,choices)
    assert again["writes"]==[]
    assert receipt["metadata_only"] and not receipt["scientific_computation"]


def test_historical_review_pointer_is_not_relocated_by_same_folder_name(tmp_path):
    root,choices=fixture(tmp_path)
    old=dict(source_model_id="phase-model-fixture",
        manifest=dict(path=str(tmp_path/"old-machine/collection/postflight_manifest.json"),sha256="a"*64))
    index=write(root/"phase_review_collections/index.json",
        json.dumps(dict(format="fluidflower-phase-review-collection-v1",collections=[old])).encode())
    write(root/"phase_review_collections/collection/postflight_manifest.json",b'{"local":"copy"}')
    before=index.read_bytes()
    report=convert.inspect_review_registration(root)
    assert report["entries"]==[old]
    assert report["pending"] and report["changed"] is False
    assert index.read_bytes()==before


def test_present_review_manifest_with_changed_SHA_is_not_reported_as_valid(tmp_path):
    root,_choices=fixture(tmp_path)
    manifest=write(tmp_path/"original_review/postflight_manifest.json",b'{"source_model_id":"m"}')
    index=write(root/"phase_review_collections/index.json",
        json.dumps(dict(format="fluidflower-phase-review-collection-v1",collections=[
            dict(source_model_id="m",source_model_sha256="b"*64,
                 source_physics_runtime="c"*64,
                 manifest=dict(path=str(manifest),sha256="a"*64))])).encode())
    original=index.read_bytes()
    result=convert.inspect_review_registration(root)
    assert "differs" in result["pending"][0]["reason"]
    assert index.read_bytes()==original


def test_explicit_foreign_model_choice_cannot_change_the_series_model(tmp_path):
    root,choices=fixture(tmp_path)
    source=model(root)
    (source/"summary.json").write_bytes(json.dumps(dict(model_id=source.name,
        inventory=dict(datasets=["Other"]))).encode())
    choices.update(active_method="phase_model",model_choice=dict(
        path=str(source.relative_to(root)),model_id=source.name,authority="synthetic explicit owner choice"))
    original=(root/"series.toml").read_bytes()
    with pytest.raises(ValueError,match="matching original library identity"):
        convert.plan_series_conversion(root,choices)
    assert (root/"series.toml").read_bytes()==original


def planned_manifest(plan):
    return tomllib.loads(next(row["after_utf8"] for row in plan["writes"]
                             if row["relative_path"] == "series.toml"))


def update_manifest(root, choices, suffix):
    path = root / "series.toml"
    path.write_bytes(path.read_bytes() + suffix)
    choices["expected_series_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_minimal_roots_do_not_invent_unused_raw_or_legacy_results(tmp_path):
    root, choices = fixture(tmp_path)
    plan = convert.plan_series_conversion(root, choices)
    assert planned_manifest(plan)["roots"] == {"series": "."}
    assert plan["roots"]["not_declared"] == ["raw", "legacy_results"]
    assert not (root / "config/_previous").exists()


def test_literal_legacy_roots_preserve_runtime_article_names_and_all_fields(tmp_path):
    root, choices = fixture(tmp_path)
    update_manifest(root, choices, b'[legacy]\nraw_inputs="../raw"\nruntime_inputs="../Analysis/V1"\n'
        b'article_inputs="../Dataset"\nraw_experiments=["../one","../two"]\n')
    plan = convert.plan_series_conversion(root, choices)
    actual = planned_manifest(plan)
    assert actual["roots"] == {"series": ".", "raw": "../raw",
        "runtime_inputs": "../Analysis/V1", "article_inputs": "../Dataset"}
    assert "legacy_results" not in actual["roots"]
    assert actual["legacy"]["raw_experiments"] == ["../one", "../two"]
    assert plan["roots"]["sources"]["raw"] == "series.toml.legacy.raw_inputs"
    assert actual["unknown"] == {"keep": "unchanged"}


def test_existing_roots_preserved_and_explicit_choices_are_bound(tmp_path):
    root, choices = fixture(tmp_path)
    update_manifest(root, choices, b'[legacy]\nraw_inputs="../old-raw"\nprevious_results="../old-results"\n'
        b'[roots]\nseries="."\nraw="../declared-raw"\nmodel_sources="../models"\n')
    original = (root / "series.toml").read_bytes()
    plan = convert.plan_series_conversion(root, choices)
    assert planned_manifest(plan)["roots"] == {"series": ".", "raw": "../declared-raw",
        "legacy_results": "../old-results", "model_sources": "../models"}
    assert plan["roots"]["sources"]["raw"] == "series.toml.roots.raw"
    choices["roots"] = {"raw": "../explicit-raw", "legacy_results": "../explicit-results"}
    plan = convert.plan_series_conversion(root, choices)
    actual = planned_manifest(plan)
    assert actual["roots"]["raw"] == "../explicit-raw"
    assert actual["roots"]["model_sources"] == "../models"
    assert actual["legacy"]["raw_inputs"] == "../old-raw"
    assert plan["choices"]["roots"] == choices["roots"]
    assert plan["roots"]["sources"]["legacy_results"] == "choices.roots.legacy_results"
    assert (root / "series.toml").read_bytes() == original


@pytest.mark.parametrize("roots", [None, {"series": ".."}, {"raw": ""},
    {"raw": "."}, {"INVALID": "../raw"}])
def test_invalid_or_ambiguous_roots_refuse_before_migration(tmp_path, roots):
    root, choices = fixture(tmp_path)
    original = (root / "series.toml").read_bytes()
    choices["roots"] = roots
    with pytest.raises(ValueError):
        convert.plan_series_conversion(root, choices)
    assert (root / "series.toml").read_bytes() == original
    assert not (root / "config/merged").exists()
    assert not (root / "config/_previous").exists()
    assert not (root / "models/phase").exists()


def test_original_resolver_refuses_unknown_consumed_input_before_migration(tmp_path):
    from darsia.fluidflower.store_map import manifest_roots, input_reference
    root, choices = fixture(tmp_path)
    update_manifest(root, choices, b'[legacy]\nraw_inputs="../raw"\n')
    plan = convert.plan_series_conversion(root, choices)
    document = planned_manifest(plan)
    roots = manifest_roots(root, document)
    known = input_reference(tmp_path / "raw" / "one.JPG", roots,
        sha256="a" * 64, size_bytes=17, role="image")
    assert known["root_alias"] == "raw" and known["relative_path"] == "one.JPG"
    with pytest.raises(ValueError, match="No declared root"):
        input_reference(tmp_path / "unknown" / "one.JPG", roots,
            sha256="a" * 64, size_bytes=17, role="image")
    assert "legacy_results" not in document["roots"]
    assert not (root / "config/_previous").exists()
    assert not (root / "config/merged").exists()


def test_apply_refuses_manifest_without_roots_before_any_backup_or_write(tmp_path):
    import toml
    root, choices = fixture(tmp_path)
    plan = convert.plan_series_conversion(root, choices)
    for row in plan["writes"]:
        if row["relative_path"] == "series.toml":
            value = tomllib.loads(row["after_utf8"])
            del value["roots"]
            row["after_utf8"] = toml.dumps(value)
    original = (root / "series.toml").read_bytes()
    with pytest.raises(ValueError, match=r"declare \[roots\]"):
        convert.apply_metadata_conversion(plan)
    assert (root / "series.toml").read_bytes() == original
    assert not (root / "config/_previous").exists()
    assert not (root / "config/merged").exists()


def test_existing_rig_directory_matches_original_status_reader_contract(tmp_path):
    from darsia.fluidflower.workflow_row_status import WorkflowStatusContext, _rig

    root, choices = fixture(tmp_path)
    baseline = write(root / "raw/baseline.JPG", b"metadata-only baseline; never decoded")
    rig = root / "results/setup/rig"
    for name in ("baseline.npz", "depth.npz", "labels.npz", "facies.npz", "porosity.npz"):
        write(rig / name, b"metadata-only NPZ placeholder; never loaded")
    write(rig / "meta_data.json", json.dumps(dict(baseline_path=str(baseline.resolve()))).encode())
    partial = root / "config/runs/one.toml"
    partial.write_bytes(b'[data]\nformat="JPG"\nbaseline="baseline.JPG"\nfolder="raw"\n'
                        b'[rig]\npath="results/setup/rig"\n')
    original = {str(path): path.read_bytes() for path in (partial, rig / "meta_data.json")}
    config = convert.native_merged([root / "config/common.toml", partial])
    status = _rig(WorkflowStatusContext(root, run_id="one", config=config))
    assert status.state == "done"
    plan = convert.plan_series_conversion(root, choices)
    evidence = plan["configured_fields_stat_only"][0]["rig_model"]
    assert evidence["exists"] is True
    assert evidence["payload_opened"] is False and evidence["SHA_verified"] is False
    assert all(Path(path).read_bytes() == value for path, value in original.items())

    # A file at rig.path cannot stand in for the original run-model folder.
    write(root / "results/not_a_rig.npz", b"single file is not a rig-model directory")
    partial.write_bytes(partial.read_bytes().replace(b'results/setup/rig', b'results/not_a_rig.npz'))
    plan = convert.plan_series_conversion(root, choices)
    assert plan["configured_fields_stat_only"][0]["rig_model"]["exists"] is False


def test_cas_failure_preserves_current_manifest_original_backup_and_temporary(tmp_path, monkeypatch):
    root, choices = fixture(tmp_path)
    manifest = root / "series.toml"
    original = manifest.read_bytes()
    plan = convert.plan_series_conversion(root, choices)
    planned = next(row["after_utf8"].encode("utf-8") for row in plan["writes"]
                   if row["relative_path"] == "series.toml")
    current = original + b"\n# concurrent metadata edit\n"
    original_bound = convert._bound_source

    def changed_after_temporary(bound):
        if Path(bound["path"]) == manifest and list(root.glob(".series.toml.*.tmp")):
            manifest.write_bytes(current)
        return original_bound(bound)

    monkeypatch.setattr(convert, "_bound_source", changed_after_temporary)
    with pytest.raises(ValueError, match="Conversion input changed"):
        convert.apply_metadata_conversion(plan)
    assert manifest.read_bytes() == current
    backups = list((root / "config/_previous").glob("*/series.toml"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    temporary = list(root.glob(".series.toml.*.tmp"))
    assert len(temporary) == 1 and temporary[0].read_bytes() == planned
    failure = json.loads((backups[0].parent / "conversion_failed.json").read_bytes())
    assert failure["failed"] and not failure["complete"]
    assert "Conversion input changed" in failure["traceback"]
    assert (root / "config/merged/one.toml").is_file()
