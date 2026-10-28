"""Round-2b method applicability/exclusions; only synthetic saved metadata.

External preparation compiles this text only; Root runs it after adoption.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import workflow_row_status as s


def put(path, value="saved file"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path



def document(path, value):
    return put(path, json.dumps(value))



@pytest.fixture
def production(tmp_path):
    put(tmp_path / "series.toml", 'schema_version=1\nseries_id="AC"\n[runs]\nids=["ac28","ac20"]\n')
    config_path = put(tmp_path / "config/merged/ac28.toml", 'fixture="ac28"\n')
    raw = tmp_path / "raw/ac28"
    baseline = put(raw / "BASE.JPG")
    rig = tmp_path / "production/ac28/setup/rig"
    document(rig / "meta_data.json", {"baseline_path": str(baseline)})
    for name in ("baseline.npz", "depth.npz", "labels.npz", "facies.npz", "porosity.npz"):
        put(rig / name, "stat-only fixture, intentionally not an array")
    labels = put(tmp_path / "data/segmented.png", "stat-only image fixture")
    props = put(tmp_path / "data/facies.csv", "facies,porosity\n0,0.4\n")
    injection = put(tmp_path / "protocols/ac28/injection.csv", "id,location_x,location_y\n1,0.45,0.03\n2,0.72,0.18\n")
    imaging = put(tmp_path / "protocols/ac28/imaging.csv", "image,time\nBASE.JPG,0\n")
    depth = put(tmp_path / "production/ac28/setup/depth/depth_map.npz")
    config = {"data": {"baseline": "BASE.JPG", "folders": [str(raw)], "results": str(tmp_path / "results/ac28")},
              "rig": {"path": str(rig)}, "depth": {"depth_map": str(depth)},
              "labeling": {"colored_image": str(labels)}, "facies": {"props": str(props)},
              "corrections": {"curvature": {"crop": {name: [0, 1] for name in ("top_left", "bottom_left", "bottom_right", "top_right")}}},
              "protocols": {"injection": str(injection), "imaging": {str(raw): str(imaging)}}}
    ctx = s.load_status_context(tmp_path, "ac28", config_path, config)
    return SimpleNamespace(context=ctx, rig=rig, injection=injection, props=props)



@pytest.mark.parametrize("action,item", [("calibration", "color"), ("calibration", "mass"),
                                           ("calibration", "default_mass"), ("calibration", "mass_optuna"),
                                           ("analysis", "cropping"), ("analysis", "mass"),
                                           ("analysis", "segmentation"), ("analysis", "fingers"),
                                           ("analysis", "thresholding"), ("analysis", "volume")])
def test_phase_method_makes_only_original_colour_rows_inapplicable(production, monkeypatch, action, item):
    ctx = production.context
    ctx.manifest["method"] = {"active": "phase_model"}
    monkeypatch.setattr(ctx, "document", lambda _path: pytest.fail("Inactive row read output metadata"))
    status = s.row_status(ctx, action, item)
    assert status.state == "not_applicable"
    assert "method phase_model" in status.summary



@pytest.mark.parametrize("action,item", [("phase_model", "training"), ("phase_model", "mass_from_phase"),
                                           ("phase_model", "selection_plan"), ("mass_analysis", "frozen"),
                                           ("mass_analysis", "review")])
def test_colour_method_greys_phase_rows_without_loading_them(production, monkeypatch, action, item):
    ctx = production.context
    ctx.manifest["method"] = {"active": "colour_path"}
    monkeypatch.setattr(ctx, "document", lambda _path: pytest.fail("Inactive phase row read metadata"))
    assert s.row_status(ctx, action, item).state == "not_applicable"



def test_unselected_method_does_not_invent_applicability(production):
    assert s.row_status(production.context, "analysis", "mass").state == "missing"
    production.context.manifest["method"] = {"active": "phase_model"}
    assert s.row_status(production.context, "calibration", "reset").summary.startswith("Maintenance action")
    assert "selected job" in s.row_status(production.context, "jobs", "overview").summary



def test_counts_use_only_explicit_exclusions_and_saved_identities(production):
    ctx = production.context
    ids = [f"ac{index}" for index in range(1, 44)]
    excluded = {run: "documented exclusion" for run in ids[-4:]}
    ctx.manifest["runs"] = {"ids": ids, "excluded": excluded}
    counts = s.analysis_run_counts(ctx, ids[:-4])
    assert counts["saved_run_count"] == 39 and counts["excluded_count"] == 4
    assert counts["missing_runs"] == [] and counts["eligible_count"] == 39
    counts = s.analysis_run_counts(ctx, ids[:-5])
    assert counts["missing_runs"] == [ids[-5]]
    assert s.analysis_run_counts(ctx, ids)["saved_run_count"] == 39



@pytest.mark.parametrize("excluded", [{"foreign": "reason"}, {"ac20": ""}, {"ac20": 1}, ["ac20"]])
def test_invalid_exclusions_do_not_silently_change_counts(production, excluded):
    production.context.manifest["runs"]["excluded"] = excluded
    with pytest.raises(ValueError):
        s.analysis_run_counts(production.context, ["ac28"])



def test_selected_exclusion_is_explicit_but_setup_is_still_available(production):
    ctx = production.context
    ctx.manifest["runs"]["excluded"] = {"ac28": "chemical outlier"}
    ctx.manifest["method"] = {"active": "colour_path"}
    status = s.row_status(ctx, "analysis", "mass")
    assert status.state == "not_applicable" and "excluded: chemical outlier" in status.summary
    assert s.row_status(ctx, "setup", "rig").state == "done"



def test_image_summary_excludes_absent_runs_using_real_ordinary_format(production):
    import hashlib
    from darsia.fluidflower.image_selection_status import saved_selection_summary
    from darsia.fluidflower.series_image_selection import FORMAT
    ctx = production.context
    ctx.manifest["runs"]["excluded"] = {"ac20": "chemical outlier"}
    protocol = Path(next(iter(ctx.config["protocols"]["imaging"].values())))
    protocol_bytes = protocol.read_bytes()
    raw = Path(ctx.config["data"]["folders"][0])
    second = put(raw / "SECOND.JPG", "stat-only fixture; no image decoding")
    ctx.manifest["image_selection"] = {
        "format": FORMAT, "schema_version": 1,
        "runs": {"ac28": {
            "selected": [
                {"source_path": str(raw / "BASE.JPG"), "image_id": "BASE.JPG", "frame_index": 0, "elapsed_h": -.1},
                {"source_path": str(second), "image_id": "SECOND.JPG", "frame_index": 1, "elapsed_h": .5}],
            "source_metadata": [{"path": str(protocol), "sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                                 "size_bytes": len(protocol_bytes)}],
            "saved_utc": "2026-10-16T20:00:00+00:00"}}}
    original = saved_selection_summary(ctx.manifest)
    assert original["missing_runs"] == ["ac20"] and not original["complete"]
    assert original["selected_frame_count"] == 2 and original["saved_run_count"] == 1
    before = (ctx.series_root / "series.toml").read_bytes()
    status = s.row_status(ctx, "image_selection", "plan")
    assert status.state == "done" and "2 images" in status.summary
    assert status.counts["saved_run_count"] == 1 and status.counts["excluded_count"] == 1
    assert status.counts["missing_runs"] == [] and "1 of 1 active runs" in status.summary
    assert "not a computed result" in status.summary
    assert status.date_source == "saved_utc"
    assert (ctx.series_root / "series.toml").read_bytes() == before
    # The passive count wrapper still rejects malformed ordinary source binding.
    ctx.manifest["image_selection"]["runs"]["ac28"]["source_metadata"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="malformed source metadata binding"):
        saved_selection_summary(ctx.manifest)
    rejected = s.row_status(ctx, "image_selection", "plan")
    assert rejected.state == "missing" and "malformed source metadata binding" in " ".join(rejected.missing)
    assert (ctx.series_root / "series.toml").read_bytes() == before


@pytest.mark.parametrize("item", ["rig_frame", "layers", "export"])
def test_pipeline_products_do_not_substitute_for_a_workspace_revision(production, monkeypatch, item):
    original = Path.open
    def no_payload(path, *args, **kwargs):
        if path.suffix.casefold() in {".npz", ".png", ".jpg"}:
            pytest.fail("A passive geometry row opened a numerical/image payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_payload)
    result = s.row_status(production.context, "geometry_facies", item)
    assert result.state == "missing"
    assert not (production.context.series_root / "geometry_facies").exists()


def test_original_geometry_reader_still_accepts_a_real_saved_registered_frame(production):
    ctx = production.context
    document(ctx.series_root / "geometry_facies/series.json", {"tools": {"native_series": {
        "reference_run": "ac28", "rig_frames": {"ac28": {"run_id": "ac28", "corners": {
            name: [0, 1] for name in ("top_left", "bottom_left", "bottom_right", "top_right")}}},
        "injection_points": [{"x_m": .1, "y_m": .2}]}}})
    result = s.row_status(ctx, "geometry_facies", "rig_frame")
    assert result.state == "done"
    assert [a.path for a in result.artifacts] == [str(ctx.series_root / "geometry_facies/series.json")]


def test_invalid_workspace_is_not_masked_by_complete_pipeline_products(production):
    put(production.context.series_root / "geometry_facies/overrides/geometry_facies/current.json", "{broken")
    for item in ("layers", "export"):
        result = s.row_status(production.context, "geometry_facies", item)
        assert result.state == "missing"
        assert "Saved geometry" in result.summary


def test_full_setup_still_requires_protocols_depth_and_each_original_rig_map(production):
    ctx = production.context
    assert s.row_status(ctx, "setup", "all").state == "done"
    assert s.row_status(ctx, "geometry_facies", "layers").state == "missing"
    (production.rig / "porosity.npz").unlink()
    assert s.row_status(ctx, "setup", "all").state == "missing"


def test_excluded_selected_run_remains_explicit_even_when_its_method_is_inactive(production):
    ctx = production.context
    ctx.manifest["method"] = {"active": "phase_model"}
    ctx.manifest["runs"]["excluded"] = {"ac28": "chemical outlier"}
    result = s.row_status(ctx, "analysis", "mass")
    assert result.state == "not_applicable"
    assert "method phase_model" in result.summary
    assert "excluded: chemical outlier" in result.selected_run_text


def test_inapplicable_rows_have_text_glyph_and_grey_style_without_a_new_state():
    from darsia.fluidflower import workflow_status_ui as ui
    result = s.RowStatus("not_applicable", "Not applicable for method phase_model.",
                         selected_run_text="Selected AC20: excluded: documented reason")
    rendered = ui.status_text(result)
    assert rendered.startswith("— Not applicable")
    assert "method phase_model" in rendered and "excluded: documented reason" in rendered
    assert "color:" in ui.state_style(result)
    assert set(ui.STATE_LABELS) == {"done", "missing", "not_applicable"}



@pytest.mark.parametrize("review_run", ["ac28", "AC_28", "Ac-28"])
def test_review_counts_use_original_identity_and_registered_exclusions(production, review_run):
    ctx = production.context
    model = ctx.series_root / "models/phase/test-version"
    document(model / "summary.json", {"model_id": "saved-model"})
    review = document(ctx.series_root / "reviews/collection.json", {
        "format": "fluidflower-phase-review-collection-v1", "source_model_id": "saved-model",
        "jobs": [{"dataset": "A_C", "run": review_run},
                 {"dataset": "AC", "run": "AC_20"},
                 {"dataset": "AC", "run": "AC99"},
                 {"dataset": "other-series", "run": "ac28"}]})
    ctx.review_reference = str(review)
    ctx.manifest["method"] = {"active": "phase_model", "phase_model": {"model_version": str(model)}}
    ctx.manifest["runs"]["excluded"] = {"ac20": "chemical outlier"}
    # Both the unchanged original reader and the count wrapper use _identity.
    assert s._saved_row_status(ctx, "mass_analysis", "review").state == "done"
    status = s.row_status(ctx, "mass_analysis", "review")
    assert status.state == "done"
    assert status.counts == {"run_count": 2, "eligible_count": 1, "saved_run_count": 1,
                             "excluded_count": 1, "missing_runs": [],
                             "excluded_runs": {"ac20": "chemical outlier"}}
    assert "1 done · 1 excluded" in status.summary
