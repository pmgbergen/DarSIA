"""Synthetic metadata/stat tests only; no DarSIA, Qt, image or array import."""

from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from darsia.fluidflower import workflow_row_status as s


def put(path, content="saved artifact"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def doc(path, value):
    return put(path, json.dumps(value))


@pytest.fixture
def ctx(tmp_path):
    put(
        tmp_path / "series.toml",
        'schema_version = 1\nseries_id = "AC"\n[runs]\nids = ["ac28", "ac52"]\n',
    )
    config_path = put(tmp_path / "config/merged/ac28.toml", 'run = "ac28"\n')
    config = {
        "data": {"baseline": "BASE.JPG", "results": str(tmp_path / "results/ac28")}
    }
    return s.load_status_context(tmp_path, "ac28", config_path, config)


def assert_missing(status):
    assert status.state == "missing"
    assert status.missing
    assert status.producer
    assert status.next_step


def geometry(ctx, revision="a" * 32):
    package = ctx.series_root / "geometry_facies"
    labels = put(package / "saved/labels.npz")
    facies = put(package / "saved/facies.npz")
    props = put(package / "saved/props.csv")
    value = {
        "schema": 1,
        "format": s.GEOMETRY_FORMAT,
        "revision_id": revision,
        "created_utc": "2026-10-15T12:00:00Z",
        "configuration": {"labels_path": str(labels)},
        "artifacts": {"facies_path": str(facies), "facies_properties_path": str(props)},
    }
    doc(package / "overrides/geometry_facies/current.json", value)
    return package, value


def test_real_scope_context_can_supply_actual_identity(ctx):
    scope = SimpleNamespace(
        series_root=ctx.series_root,
        selected_run="ac28",
        loaded_path=ctx.config_path,
        loaded=ctx.config,
    )
    loaded = s.load_status_context(None, scope_context=scope)
    assert loaded.run_bound and loaded.run_id == "ac28"
    assert loaded.scope_context is scope


def test_foreign_open_config_does_not_borrow_selected_run(ctx):
    path = put(ctx.series_root / "foreign.toml", 'run = "external"\n')
    loaded = s.load_status_context(ctx.series_root, "ac28", path, ctx.config)
    assert not loaded.run_bound
    assert_missing(s.row_status(loaded, "setup", "rig"))


def test_missing_registered_config_is_not_bound_even_with_loaded_values(ctx):
    ctx.config_path.unlink()
    loaded = s.load_status_context(ctx.series_root, "ac28", ctx.config_path, ctx.config)
    assert not loaded.run_bound


def test_unknown_selected_run_is_missing_not_not_applicable(ctx):
    loaded = s.load_status_context(
        ctx.series_root, "foreign", ctx.config_path, ctx.config
    )
    assert_missing(s.row_status(loaded, "analysis", "mass"))


def test_bad_manifest_is_missing(ctx):
    put(ctx.series_root / "series.toml", "invalid = [\n")
    loaded = s.load_status_context(ctx.series_root, "ac28", ctx.config_path)
    assert loaded.issues and not loaded.run_bound


def test_oversized_metadata_is_bounded(ctx):
    path = put(ctx.series_root / "oversize.json", " " * (s.MAX_METADATA_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        ctx.document(path)


def test_metadata_cache_reads_each_bound_document_once(ctx):
    path = doc(ctx.series_root / "summary.json", {"first": True})
    assert ctx.document(path)["first"] is True
    doc(path, {"first": False})
    assert ctx.document(path)["first"] is True


def test_protocols_use_exact_config_paths_and_require_all(ctx):
    one = put(ctx.series_root / "custom/injection.csv")
    two = put(ctx.series_root / "other/pt.csv")
    three = put(ctx.series_root / "image-protocol.csv")
    ctx.config["data"]["folders"] = ["raw-folder"]
    ctx.config["protocols"] = {
        "injection": [str(one), "sheet"],
        "pressure_temperature": str(two),
        "imaging": {"raw-folder": str(three)},
    }
    assert s.row_status(ctx, "preprocessing", "protocols").state == "done"
    two.unlink()
    assert_missing(s.row_status(ctx, "preprocessing", "protocols"))


def test_empty_protocol_configuration_is_not_success(ctx):
    assert_missing(s.row_status(ctx, "preprocessing", "protocols"))


def test_measurements_do_not_mean_depth_map_done(ctx):
    csv = put(ctx.series_root / "depth.csv")
    ctx.config["depth"] = {"measurements": str(csv)}
    assert s.row_status(ctx, "preprocessing", "depth_measurements").state == "done"
    assert_missing(s.row_status(ctx, "setup", "depth"))


def test_explicit_depth_map_stat_without_array_open(ctx, monkeypatch):
    path = put(ctx.series_root / "custom/depth.npz")
    ctx.config["depth"] = {"depth_map": str(path)}
    original = Path.open

    def forbid_array(self, *args, **kwargs):
        if self.suffix in {".npz", ".png", ".jpg"}:
            raise AssertionError("Passive status opened image/array bytes")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_array)
    assert s.row_status(ctx, "setup", "depth").state == "done"


def test_inherited_crop_is_not_new_geometry(ctx):
    ctx.config["corrections"] = {
        "curvature": {
            "crop": {
                key: [0, 0]
                for key in ("top_left", "bottom_left", "bottom_right", "top_right")
            }
        }
    }
    assert s.row_status(ctx, "preprocessing", "crop").state == "done"
    assert_missing(s.row_status(ctx, "geometry_facies", "rig_frame"))


def test_geometry_requires_own_saved_frame_and_injection_points(ctx):
    package = ctx.series_root / "geometry_facies"
    link = {
        "reference_run": "ac28",
        "rig_frames": {
            "ac28": {
                "corners": {
                    key: [1, 2]
                    for key in ("top_left", "bottom_left", "bottom_right", "top_right")
                }
            }
        },
        "injection_points": [{"x_m": 0.1, "y_m": 0.2}],
    }
    doc(package / "series.json", {"tools": {"native_series": link}})
    assert s.row_status(ctx, "geometry_facies", "rig_frame").state == "done"


def test_geometry_with_incomplete_corners_is_missing(ctx):
    package = ctx.series_root / "geometry_facies"
    doc(
        package / "series.json",
        {
            "tools": {
                "native_series": {
                    "reference_run": "ac28",
                    "rig_frames": {"ac28": {"corners": {"top_left": [1, 2]}}},
                    "injection_points": [{"x_m": 1}],
                }
            }
        },
    )
    assert_missing(s.row_status(ctx, "geometry_facies", "rig_frame"))


def test_geometry_wrong_named_four_corners_is_missing(ctx):
    package = ctx.series_root / "geometry_facies"
    doc(
        package / "series.json",
        {
            "tools": {
                "native_series": {
                    "reference_run": "ac28",
                    "rig_frames": {
                        "ac28": {
                            "corners": {name: [1, 2] for name in ("a", "b", "c", "d")}
                        }
                    },
                    "injection_points": [{"x_m": 1}],
                }
            }
        },
    )
    assert_missing(s.row_status(ctx, "geometry_facies", "rig_frame"))


def test_layers_current_revision_and_bound_artifacts(ctx):
    geometry(ctx)
    value = s.row_status(ctx, "geometry_facies", "layers")
    assert value.state == "done" and len(value.artifacts) == 4
    assert value.date == "2026-10-15 14:00 CEST"
    assert value.date_source == "created_utc"


def test_geometry_rejects_revision_path_escape(ctx):
    geometry(ctx, "../" + "a" * 29)
    assert_missing(s.row_status(ctx, "geometry_facies", "layers"))


def test_current_export_stale_revision_is_missing(ctx):
    package, current = geometry(ctx)
    dest = package / "exports/native_setup" / current["revision_id"]
    doc(
        dest / "export.json",
        {"schema": 1, "format": s.EXPORT_FORMAT, "revision_id": "b" * 32},
    )
    assert_missing(s.row_status(ctx, "geometry_facies", "export"))


def test_not_yet_applied_geometry_has_a_next_step_instead_of_a_file_error(ctx):
    geometry(ctx)
    value = s.row_status(ctx, "geometry_facies", "export")
    assert value.state == "missing"
    assert value.summary == "Apply has not run for the saved geometry revision yet."
    assert value.missing == ()
    assert value.next_step == "Prepare the Apply and Build job, then start it in Jobs."
    assert value.artifacts == ()


def test_export_requires_every_descriptor_and_file(ctx):
    package, current = geometry(ctx)
    dest = package / "exports/native_setup" / current["revision_id"]
    names = (
        "colored_labels.png",
        "native_labels.npz",
        "facies_properties.csv",
        "native_setup.toml",
    )
    for name in names:
        put(dest / name)
    receipt = {
        "schema": 1,
        "format": s.EXPORT_FORMAT,
        "revision_id": current["revision_id"],
        "files": {name: {"sha256": "1" * 64} for name in names},
    }
    doc(dest / "export.json", receipt)
    assert s.row_status(ctx, "geometry_facies", "export").state == "done"
    (dest / names[0]).unlink()
    assert_missing(s.row_status(ctx, "geometry_facies", "export"))


def test_legacy_run_model_does_not_complete_new_geometry(ctx):
    root = ctx.series_root / "legacy-rig"
    ctx.config["rig"] = {"path": str(root)}
    ctx.config["data"]["baseline"] = str(Path("/raw/BASE.JPG").absolute())
    doc(
        root / "meta_data.json",
        {"baseline_path": str(Path("/raw/BASE.JPG").absolute())},
    )
    for name in (
        "baseline.npz",
        "depth.npz",
        "labels.npz",
        "facies.npz",
        "porosity.npz",
    ):
        put(root / name)
    status = s.row_status(ctx, "setup", "rig")
    assert status.state == "done" and "not mark a new geometry" in status.summary
    assert_missing(s.row_status(ctx, "geometry_facies", "export"))


def _run_model(root, baseline, padding=0):
    doc(root / "meta_data.json", {"baseline_path": str(baseline), "analysis": "x" * padding})
    for name in ("baseline.npz", "depth.npz", "labels.npz", "facies.npz", "porosity.npz"):
        put(root / name)


def test_run_model_metadata_above_general_bound_is_read(ctx):
    root = ctx.series_root / "built-rig"
    baseline = Path("/raw/BASE.JPG").absolute()
    ctx.config["rig"] = {"path": str(root)}
    ctx.config["data"]["baseline"] = str(baseline)
    _run_model(root, baseline, padding=s.MAX_METADATA_BYTES + 1)
    assert s.row_status(ctx, "setup", "rig").state == "done"


def test_relocated_rig_baseline_is_compared_through_storage_aliases(ctx, monkeypatch):
    from darsia.fluidflower import storage_paths
    old, new = ctx.series_root / "old-raw", ctx.series_root / "new-raw"
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(old).replace("\\", "/").casefold(), str(new).replace("\\", "/"))])
    root = ctx.series_root / "built-rig"
    ctx.config["rig"] = {"path": str(root)}
    # The configuration and the run model both name the photograph at its old place.
    ctx.config["data"]["baseline"] = str(old / "BASE.JPG")
    _run_model(root, old / "BASE.JPG")
    assert s.row_status(ctx, "setup", "rig").state == "done"
    ctx.config["data"].update(baseline="BASE.JPG", folders=[str(old)])
    assert s.row_status(ctx, "setup", "rig").state == "done"
    _run_model(root, old / "OTHER.JPG")
    ctx._metadata_cache.clear()
    assert_missing(s.row_status(ctx, "setup", "rig"))


def test_wrong_rig_baseline_is_missing(ctx):
    root = ctx.series_root / "legacy-rig"
    ctx.config["rig"] = {"path": str(root)}
    doc(root / "meta_data.json", {"baseline_path": "/raw/OTHER.JPG"})
    assert_missing(s.row_status(ctx, "setup", "rig"))


def test_unbound_method_never_infers_latest_history(ctx):
    doc(ctx.series_root / "models/latest/summary.json", {"model_id": "later"})
    doc(
        ctx.series_root / "results/latest-fit.json",
        {"parameters": {"gain": 1, "beta": 2}},
    )
    assert_missing(s.row_status(ctx, "phase_model", "training"))
    assert_missing(s.row_status(ctx, "phase_model", "mass_from_phase"))


def test_explicit_model_summary_and_failed_publication(ctx):
    root = ctx.series_root / "models/explicit"
    ctx.method_selection = {"model_version": str(root)}
    doc(
        root / "summary.json",
        {"model_id": "explicit", "created_utc": "2026-10-15T12:00:00Z"},
    )
    assert s.row_status(ctx, "phase_model", "training").state == "done"
    doc(root / "publication_receipt.json", {"qa_status": "fail"})
    assert_missing(s.row_status(ctx, "phase_model", "training"))


@pytest.mark.parametrize(
    "parameters",
    [{}, {"gain": 1}, {"gain": True, "beta": 2}, {"gain": 1, "beta": float("nan")}],
)
def test_invalid_recorded_phase_calibration_missing(ctx, parameters):
    path = doc(ctx.series_root / "fit.json", {"parameters": parameters})
    ctx.method_selection = {"mass_calibration": str(path)}
    assert_missing(s.row_status(ctx, "phase_model", "mass_from_phase"))


def test_explicit_saved_calibration_no_refit(ctx):
    path = doc(
        ctx.series_root / "fit.json", {"parameters": {"gain": 23.859, "beta": 0.6513}}
    )
    ctx.method_selection = SimpleNamespace(mass_calibration=str(path))
    value = s.row_status(ctx, "phase_model", "mass_from_phase")
    assert value.state == "done" and "no refit" in value.summary


def test_colour_path_has_metadata_and_named_csv_not_just_folder(ctx):
    root = ctx.series_root / "colour"
    ctx.config["calibration"] = {"color": {"embedding": "colour"}}
    ctx.config["color_path"] = [
        {"name": "colour", "root": str(root), "basis": "physical"}
    ]
    put(root / "unrelated.txt")
    assert_missing(s.row_status(ctx, "calibration", "color"))
    doc(root / "metadata.json", {"basis": "physical", "label_ids": [0, 1]})
    put(root / "color_paths.csv")
    assert s.row_status(ctx, "calibration", "color").state == "done"


def test_explicit_colour_mass_files_only_and_listing_bounded(ctx, monkeypatch):
    root = ctx.series_root / "colour-mass"
    ctx.method_selection = {"colour_calibration": str(root)}
    doc(
        root / "flash/flash.json",
        dict(min_value_aq=0, max_value_aq=1, min_value_g=0, max_value_g=1),
    )
    put(root / "signal_model/signal_model_1.csv")
    assert s.row_status(ctx, "calibration", "mass").state == "done"
    monkeypatch.setattr(s, "MAX_LIST_ENTRIES", 0)
    assert_missing(s.row_status(ctx, "calibration", "mass"))


def test_mass_hidden_alternatives_same_saved_reference(ctx):
    assert_missing(s.row_status(ctx, "calibration", "default_mass"))
    assert_missing(s.row_status(ctx, "calibration", "mass_optuna"))


def test_maintenance_is_not_applicable_but_does_not_disable_action(ctx):
    for item in ("reset", "delete"):
        value = s.row_status(ctx, "calibration", item)
        assert value.state == "not_applicable" and "does not disable" in value.next_step


def test_absent_optional_config_is_missing_not_not_applicable(ctx):
    assert_missing(s.row_status(ctx, "analysis", "volume"))
    assert_missing(s.row_status(ctx, "unknown", "unknown"))


def test_nonempty_analysis_dir_with_unrelated_file_is_not_done(ctx):
    folder = Path(ctx.config["data"]["results"]) / "mass"
    put(folder / "progress.log")
    assert_missing(s.row_status(ctx, "analysis", "mass"))


def test_analysis_fixed_scalar_table_and_explicit_override(ctx):
    result = Path(ctx.config["data"]["results"])
    put(result / "sparse_data/integrated_mass.csv")
    assert s.row_status(ctx, "analysis", "mass").state == "done"
    override = ctx.series_root / "explicit-output"
    ctx.config["analysis"] = {"cropping": {"folder": str(override)}}
    put(override / "png/BASE.png")
    assert s.row_status(ctx, "analysis", "cropping").state == "done"


def test_analysis_shallow_listing_has_global_entry_limit(ctx, monkeypatch):
    folder = Path(ctx.config["data"]["results"]) / "cropping"
    put(folder / "progress.log")
    monkeypatch.setattr(s, "MAX_LIST_ENTRIES", 0)
    assert "bounded" in s.row_status(ctx, "analysis", "cropping").summary


def test_explicit_delivery_identity_and_recorded_sha_required(ctx):
    result = put(ctx.series_root / "selected-result.json")
    receipt = doc(
        ctx.series_root / "delivery.json",
        {
            "format": s.DELIVERY_FORMAT,
            "candidate": {"dataset": "AC", "run": "ac28"},
            "result": str(result),
            "result_sha256": "1" * 64,
        },
    )
    ctx.result_receipt_reference = str(receipt)
    value = s.row_status(ctx, "mass_analysis", "frozen")
    assert value.state == "done" and "viewer verifies" in value.summary
    result.unlink()
    assert_missing(s.row_status(ctx, "mass_analysis", "frozen"))


def test_foreign_delivery_is_missing(ctx):
    path = doc(
        ctx.series_root / "delivery.json",
        {"format": s.DELIVERY_FORMAT, "candidate": {"dataset": "OTHER", "run": "ac28"}},
    )
    ctx.result_receipt_reference = str(path)
    assert_missing(s.row_status(ctx, "mass_analysis", "frozen"))


def test_series_scoped_delivery_can_be_another_registered_run(ctx):
    result = put(ctx.series_root / "ac52-result.json")
    receipt = doc(
        ctx.series_root / "ac52-delivery.json",
        {
            "format": s.DELIVERY_FORMAT,
            "candidate": {"dataset": "AC", "run": "ac52"},
            "result": str(result),
            "result_sha256": "1" * 64,
        },
    )
    ctx.result_receipt_reference = str(receipt)
    assert s.row_status(ctx, "mass_analysis", "frozen").state == "done"


def test_unregistered_delivery_run_is_missing(ctx):
    path = doc(
        ctx.series_root / "delivery.json",
        {"format": s.DELIVERY_FORMAT, "candidate": {"dataset": "AC", "run": "ac99"}},
    )
    ctx.result_receipt_reference = str(path)
    assert_missing(s.row_status(ctx, "mass_analysis", "frozen"))


def test_review_and_plan_require_explicit_selection(ctx):
    for action, item in (
        ("phase_model", "selection_plan"),
        ("mass_analysis", "review"),
    ):
        assert_missing(s.row_status(ctx, action, item))
    path = doc(
        ctx.series_root / "explicit-review.json", {"schema": 1, "review_id": "fixture"}
    )
    ctx.review_reference = str(path)
    assert_missing(s.row_status(ctx, "mass_analysis", "review"))


def test_oslo_dates_observe_dst_and_reject_unzoned_time():
    assert s.oslo_date("2026-10-15T12:00:00Z") == "2026-10-15 14:00 CEST"
    assert s.oslo_date("2026-12-15T12:00:00Z") == "2026-12-15 13:00 CET"
    assert s.oslo_date("2026-10-15T12:00:00") == ""
    assert s.oslo_date("bad") == ""


def test_passive_all_rows_do_not_write_or_use_rglob(ctx, monkeypatch):
    before = {str(p): p.read_bytes() for p in ctx.series_root.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("No recursive scan")
        ),
    )
    pairs = [
        ("preprocessing", key) for key in ("protocols", "depth_measurements", "crop")
    ]
    pairs += [
        ("setup", key) for key in ("depth", "rig", "segmentation", "facies", "all")
    ]
    pairs += [("geometry_facies", key) for key in ("rig_frame", "layers", "export")]
    pairs += [
        ("calibration", key)
        for key in ("color", "mass", "default_mass", "mass_optuna", "reset", "delete")
    ]
    pairs += [
        ("phase_model", key)
        for key in ("training", "mass_from_phase", "selection_plan")
    ]
    pairs += [
        ("analysis", key)
        for key in (
            "cropping",
            "mass",
            "segmentation",
            "fingers",
            "thresholding",
            "volume",
        )
    ]
    pairs += [("mass_analysis", key) for key in ("frozen", "review")]
    for action, item in pairs:
        value = s.row_status(ctx, action, item)
        assert value.state in {"done", "missing", "not_applicable"}
        assert isinstance(asdict(value), dict)
    assert all(Path(path).read_bytes() == raw for path, raw in before.items())


def test_saved_missing_metadata_stat_is_missing_not_exception():
    state = s._saved("Saved data", [None], "Metadata producer")
    assert_missing(state)
    assert "disappeared" in state.missing[0]


def test_saved_one_disappeared_file_never_reports_partial_done(tmp_path):
    path = put(tmp_path / "remaining.json")
    state = s._saved("Saved data", [s._artifact(path), None], "Metadata producer")
    assert_missing(state)
    assert not state.artifacts


def test_saved_empty_artifacts_is_missing_not_done():
    assert_missing(s._saved("Saved data", [], "Metadata producer"))


def test_metadata_disappears_between_document_and_stat(ctx, monkeypatch):
    path = doc(ctx.series_root / "calibration.json", {"gain": 23.859, "beta": 0.6513})
    ctx.method_selection = {"mass_calibration_reference": str(path)}
    actual = ctx.document

    def disappearing_document(p):
        value = actual(p)
        if p == path:
            path.unlink()
        return value

    monkeypatch.setattr(ctx, "document", disappearing_document)
    assert_missing(s.row_status(ctx, "phase_model", "mass_from_phase"))


def test_config_disappears_between_binding_and_status(ctx):
    ctx.config = {
        "corrections": {
            "curvature": {
                "crop": {
                    k: [0, 0]
                    for k in ("top_left", "bottom_left", "top_right", "bottom_right")
                }
            }
        }
    }
    ctx.config_path.unlink()
    assert_missing(s.row_status(ctx, "preprocessing", "crop"))


def test_metadata_open_race_is_missing(ctx, monkeypatch):
    geometry(ctx)

    def removed(_path):
        raise FileNotFoundError("removed during metadata inspection")

    monkeypatch.setattr(ctx, "document", removed)
    state = s.row_status(ctx, "geometry_facies", "layers")
    assert_missing(state)
    assert "removed during" in state.missing[0]


def test_rig_same_basename_different_folder_is_missing(ctx):
    root = ctx.series_root / "rig"
    ctx.config["rig"] = {"path": str(root)}
    ctx.config["data"]["baseline"] = str(ctx.series_root / "raw/ac28/BASE.JPG")
    doc(
        root / "meta_data.json",
        {"baseline_path": str(ctx.series_root / "raw/ac52/BASE.JPG")},
    )
    for name in (
        "baseline.npz",
        "depth.npz",
        "labels.npz",
        "facies.npz",
        "porosity.npz",
    ):
        put(root / name)
    assert_missing(s.row_status(ctx, "setup", "rig"))


def test_relative_rig_metadata_is_ambiguous_and_missing(ctx):
    root = ctx.series_root / "rig"
    ctx.config["rig"] = {"path": str(root)}
    ctx.config["data"]["folder"] = str(ctx.series_root / "raw/ac28")
    doc(root / "meta_data.json", {"baseline_path": "BASE.JPG"})
    for name in (
        "baseline.npz",
        "depth.npz",
        "labels.npz",
        "facies.npz",
        "porosity.npz",
    ):
        put(root / name)
    assert_missing(s.row_status(ctx, "setup", "rig"))


def test_rig_folder_search_follows_existing_selected_config_order(ctx):
    one = ctx.series_root / "raw/one"
    two = ctx.series_root / "raw/two"
    ctx.config["data"]["folder"] = str(one)
    ctx.config["data"]["folders"] = [str(two)]
    put(two / "BASE.JPG")
    assert s._configured_rig_baseline(ctx) == two / "BASE.JPG"
    put(one / "BASE.JPG")
    assert s._configured_rig_baseline(ctx) == one / "BASE.JPG"


def test_rig_cache_identity_is_exact_not_only_basename(ctx):
    folder = ctx.series_root / "raw/ac28"
    put(folder / "BASE.JPG")
    ctx.config["data"].update(folder=str(folder), use_cache=True)
    assert (
        s._configured_rig_baseline(ctx)
        == ctx.series_root / "results/ac28/raw_cache/BASE.npz"
    )


def test_relative_baseline_without_folder_is_not_identity(ctx):
    with pytest.raises(ValueError, match="no explicit configured data folder"):
        s._configured_rig_baseline(ctx)


@pytest.mark.parametrize(
    "point",
    [
        {"name": "port"},
        {"x_m": None, "y_m": None, "pixel": None},
        {"x_m": float("nan"), "y_m": 1},
        {"pixel": {"run_id": "ac52", "row": 1, "col": 2}},
    ],
)
def test_unlocated_injection_point_cannot_complete_geometry(ctx, point):
    package, _ = geometry(ctx)
    link = {
        "reference_run": "ac28",
        "rig_frames": {
            "ac28": {
                "corners": {
                    key: [1, 2]
                    for key in ("top_left", "bottom_left", "top_right", "bottom_right")
                }
            }
        },
        "injection_points": [point],
    }
    doc(package / "series.json", {"tools": {"native_series": link}})
    assert_missing(s.row_status(ctx, "geometry_facies", "rig_frame"))


def test_injection_pixel_must_be_bound_to_reference(ctx):
    package, _ = geometry(ctx)
    link = {
        "reference_run": "ac28",
        "rig_frames": {
            "ac28": {
                "corners": {
                    key: [1, 2]
                    for key in ("top_left", "bottom_left", "top_right", "bottom_right")
                }
            }
        },
        "injection_points": [{"pixel": {"run_id": "ac28", "row": 1, "col": 2}}],
    }
    doc(package / "series.json", {"tools": {"native_series": link}})
    assert s.row_status(ctx, "geometry_facies", "rig_frame").state == "done"


def test_plan_version_purposes_and_series_folder_binding(ctx):
    path = put(
        ctx.series_root / "config/plan.toml",
        'schema_version=1\n[purposes.production]\nmode="all"\n',
    )
    ctx.selection_plan_reference = str(path)
    assert s.row_status(ctx, "phase_model", "selection_plan").state == "done"
    other = put(
        ctx.series_root.parent / "other-series/config/plan.toml", path.read_text()
    )
    ctx.selection_plan_reference = str(other)
    assert_missing(s.row_status(ctx, "phase_model", "selection_plan"))


@pytest.mark.parametrize(
    "text",
    [
        "schema_version=1\n",
        'schema_version=1\n[purposes.unknown]\nmode="all"\n',
        'schema_version=1\n[purposes.review]\nmode="target_hours"\ntarget_hours=[]\n',
        'schema_version=1\nseries_id="other"\n[purposes.production]\nmode="all"\n',
    ],
)
def test_empty_or_unbound_plan_is_missing(ctx, text):
    ctx.selection_plan_reference = str(put(ctx.series_root / "config/plan.toml", text))
    assert_missing(s.row_status(ctx, "phase_model", "selection_plan"))


def test_review_collection_must_match_model_and_series(ctx):
    root = ctx.series_root / "models/model1"
    doc(root / "summary.json", {"model_id": "model1"})
    ctx.method_selection = {"model_version": str(root)}
    value = {
        "format": "fluidflower-phase-review-collection-v1",
        "source_model_id": "model1",
        "jobs": [{"dataset": "AC", "run": "ac28"}],
    }
    path = doc(ctx.series_root / "review.json", value)
    ctx.review_reference = str(path)
    assert s.row_status(ctx, "mass_analysis", "review").state == "done"
    value["source_model_id"] = "model2"
    doc(path, value)
    ctx._metadata_cache.clear()
    assert_missing(s.row_status(ctx, "mass_analysis", "review"))
    value["source_model_id"] = "model1"
    value["jobs"][0]["dataset"] = "other"
    doc(path, value)
    ctx._metadata_cache.clear()
    assert_missing(s.row_status(ctx, "mass_analysis", "review"))


def test_ui_saved_references_are_keyed_by_current_series(tmp_path):
    # Execute only these real adapter method bodies; no Qt app or UI is created.
    import ast
    from types import SimpleNamespace
    import inspect
    from darsia.fluidflower.workflow_status_ui import NativeWorkflowStatus

    code = inspect.getsource(NativeWorkflowStatus)
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef))
    methods = [
        n
        for n in node.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_series_key", "set_selected_reference"}
    ]
    fixture = ast.ClassDef(
        name="AdapterFixture", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    namespace = {"Path": Path}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[])),
            "actual_adapter_methods",
            "exec",
        ),
        namespace,
    )
    adapter = namespace["AdapterFixture"]()
    workspace = SimpleNamespace(
        series_id="a", series_roots={"a": tmp_path / "a", "b": tmp_path / "b"}
    )
    adapter.window = SimpleNamespace(phase_model_workspace=workspace)
    adapter.selected_references = {}
    adapter.refresh = lambda: None
    adapter.set_selected_reference("review", "A-review.json")
    key_a = adapter._series_key()
    assert adapter.selected_references[key_a]["review"] == "A-review.json"
    workspace.series_id = "b"
    assert adapter.selected_references.get(adapter._series_key(), {}) == {}
    adapter.set_selected_reference("review", "B-review.json")
    assert adapter.selected_references[key_a]["review"] == "A-review.json"


def test_directory_listing_is_streamed_and_stops_at_bound(ctx, monkeypatch):
    counter = []

    class Listing:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            for i in range(10000):
                counter.append(i)
                yield SimpleNamespace(
                    path=str(ctx.series_root / f"file_{i}.txt"),
                    is_dir=lambda **kw: False,
                )

    monkeypatch.setattr(s.os, "scandir", lambda path: Listing())
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda *a: (_ for _ in ()).throw(AssertionError("No eager listing")),
    )
    assert_missing(s.row_status(ctx, "analysis", "cropping"))
    assert len(counter) == s.MAX_LIST_ENTRIES + 1


def test_colour_listing_is_streamed_and_stops_at_bound(ctx, monkeypatch):
    root = ctx.series_root / "calibration"
    doc(
        root / "flash/flash.json",
        {
            key: 0
            for key in ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g")
        },
    )
    ctx.method_selection = {"colour_calibration": str(root)}
    counter = []

    class Listing:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            for i in range(10000):
                counter.append(i)
                yield SimpleNamespace(path=str(root / f"unrelated_{i}.txt"))

    monkeypatch.setattr(s.os, "scandir", lambda path: Listing())
    assert_missing(s.row_status(ctx, "calibration", "mass"))
    assert len(counter) == s.MAX_LIST_ENTRIES + 1


def test_imaging_only_protocol_is_legitimate_saved_output(ctx):
    image = put(ctx.series_root / "imaging.csv", "time,image\n")
    ctx.config["data"]["folders"] = ["raw"]
    ctx.config["protocols"] = {
        "imaging": {"raw": str(image)},
        "injection": " ",
        "pressure_temperature": None,
    }
    status = s.row_status(ctx, "preprocessing", "protocols")
    assert status.state == "done" and "does not validate" in status.summary


@pytest.mark.parametrize("imaging", [None, "imaging.csv", {}, {"wrong": "imaging.csv"}])
def test_injection_or_pt_alone_cannot_complete_protocols_without_bound_imaging(
    ctx, imaging
):
    put(ctx.series_root / "imaging.csv")
    inj = put(ctx.series_root / "injection.csv")
    pt = put(ctx.series_root / "pt.csv")
    ctx.config["data"]["folders"] = ["raw"]
    ctx.config["protocols"] = {
        "imaging": imaging,
        "injection": str(inj),
        "pressure_temperature": str(pt),
    }
    assert_missing(s.row_status(ctx, "preprocessing", "protocols"))


@pytest.mark.parametrize(
    "imaging",
    [
        {"raw": ""},
        {"raw": None},
        {"raw": []},
        {"raw": ["imaging.csv", "sheet", "extra"]},
        {"raw": ["imaging.csv"]},
    ],
)
def test_malformed_imaging_destination_is_missing(ctx, imaging):
    put(ctx.series_root / "imaging.csv")
    ctx.config["data"]["folders"] = ["raw"]
    ctx.config["protocols"] = {"imaging": imaging}
    assert_missing(s.row_status(ctx, "preprocessing", "protocols"))


def test_protocols_accept_legacy_singular_data_folder(ctx):
    image = put(ctx.series_root / "imaging.csv")
    ctx.config["data"]["folder"] = "raw"
    ctx.config["protocols"] = {"imaging": {"raw": str(image)}}
    assert s.row_status(ctx, "preprocessing", "protocols").state == "done"


def test_protocols_use_union_of_legacy_and_current_folders(ctx):
    first = put(ctx.series_root / "first.csv")
    second = put(ctx.series_root / "second.csv")
    ctx.config["data"].update(folder="raw-a", folders=["raw-a", "raw-b"])
    ctx.config["protocols"] = {"imaging": {"raw-a": str(first), "raw-b": str(second)}}
    assert s.row_status(ctx, "preprocessing", "protocols").state == "done"
    del ctx.config["protocols"]["imaging"]["raw-a"]
    assert_missing(s.row_status(ctx, "preprocessing", "protocols"))


def test_protocols_skip_original_singular_placeholder(ctx):
    image = put(ctx.series_root / "imaging.csv")
    ctx.config["data"].update(folder="No file chosen", folders=["raw"])
    ctx.config["protocols"] = {"imaging": {"raw": str(image)}}
    assert s.row_status(ctx, "preprocessing", "protocols").state == "done"


@pytest.mark.parametrize("registered", [True, False])
def test_jobs_overview_is_administrative_without_saved_output_lookup(ctx, monkeypatch, registered):
    context = ctx if registered else s.WorkflowStatusContext(None)
    before = {path: path.read_bytes() for path in ctx.series_root.rglob("*") if path.is_file()}

    def forbidden_document(*args, **kwargs):
        pytest.fail("The Jobs view has no independent output to inspect")

    monkeypatch.setattr(context, "document", forbidden_document)
    state = s.row_status(context, "jobs", "overview")

    assert state.state == "not_applicable"
    assert "selected job's progress" in state.summary
    assert "job list and panel" in state.summary
    assert not state.artifacts and not state.missing
    assert not state.producer and not state.next_step and not state.date_source
    assert {path: path.read_bytes() for path in ctx.series_root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("count", [None, 30, 20])
def test_calibration_image_status_describes_default_or_saved_choice_without_output_checks(ctx, monkeypatch, count):
    ctx.run_bound = False  # A series input choice needs no loaded run.
    if count is not None:
        ctx.manifest["calibration"] = {"mass": {"image_selection": {"max_images_per_run": count}}}
    monkeypatch.setattr(ctx, "document", lambda *a, **k: pytest.fail("Image choices need no saved output read"))
    status = s.row_status(ctx, "calibration", "images")
    assert status.state == ("default_choice" if count is None else "saved_choice")
    assert f"{30 if count is None else count} calibration target images" in status.summary
    assert status.counts == dict(max_images_per_run=30 if count is None else count, explicit=count is not None)
    assert not status.missing and not status.artifacts and not status.date
    assert "not a completed calibration" in status.summary


def test_invalid_calibration_image_count_is_a_choice_error_not_a_missing_output(ctx):
    ctx.manifest["calibration"] = {"mass": {"image_selection": {"max_images_per_run": 0}}}
    status = s.row_status(ctx, "calibration", "images")
    assert status.state == "choice_error"
    assert not status.missing and not status.producer


def test_saved_calibration_method_row_is_done(tmp_path):
    from types import SimpleNamespace
    from darsia.fluidflower import workflow_row_status as status
    (tmp_path / "series.toml").write_text("schema_version = 1\n", encoding="utf-8")
    context = SimpleNamespace(series_root=tmp_path, manifest={"schema_version": 1, "method": {"active": "phase_model"}},
                              method_selection=None)
    row = status.row_status(context, "method", "use_for_series")
    assert row.state == "done" and "Titration + dynamic phase maps" in row.summary
    context.manifest = {"schema_version": 1}
    assert status.row_status(context, "method", "use_for_series").state == "missing"


def test_original_full_setup_row_names_image_correction(tmp_path):
    (tmp_path / "series.toml").write_text("schema_version = 1\n", encoding="utf-8")
    context = SimpleNamespace(series_root=tmp_path, manifest={"schema_version": 1, "method": {"active": "phase_model"}},
                              method_selection=None)
    row = s.row_status(context, "setup", "all")
    assert row.state == "not_applicable"
    assert "image correction and colour paths" in row.summary
    assert "image preparation" not in row.summary.casefold()
