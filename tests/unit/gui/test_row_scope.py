"""Synthetic scope/provenance checks; no Qt or actual series data."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from darsia.fluidflower import row_scope as scopes


@pytest.fixture
def series(tmp_path):
    root = tmp_path / "series"
    (root / "config" / "runs").mkdir(parents=True)
    manifest = root / "series.toml"
    manifest.write_bytes(
        b'schema_version = 1\nseries_id = "FF_AC"\n[runs]\nids = ["ac28", "ac52"]\nexpected_count = 999\n[runs.config_files]\nac28 = "config/runs/ac28.toml"\nac52 = "config/runs/ac52.toml"\n'
    )
    common = root / "config" / "common.toml"
    common.write_bytes(
        b'[calibration.color]\nembedding = "shared-colour"\n[calibration.mass]\nembedding = "shared-mass"\n[calibration.mass.options]\nthreshold = 0.2\n'
    )
    run = root / "config" / "runs" / "ac28.toml"
    run.write_bytes(b'[data]\nresults = "per-run-results"\n')
    loaded = {
        "calibration": {
            "color": {"embedding": "shared-colour"},
            "mass": {"embedding": "shared-mass", "options": {"threshold": 0.2}},
        },
        "data": {"results": "per-run-results"},
    }
    return root, manifest, common, run, loaded


COMMON_LINE = "Settings: whole series (common.toml) · Runs on: AC28"


def context(series):
    root, _manifest, _common, _run, loaded = series
    return scopes.load_scope_context(
        root,
        "ffac",
        "AC28",
        root / "config/merged/ac28.toml",
        loaded,
        series_label="AC",
    )


@pytest.mark.parametrize(
    "action,item_id",
    [
        ("phase_model", "normalisation"),
        ("phase_model", "training"),
        ("phase_model", "overflow"),
        ("phase_model", "mass_from_phase"),
        ("phase_model", "calibration_review"),
        ("phase_model", "selection_plan"),
        ("mass_analysis", "frozen"),
        ("mass_analysis", "review"),
        ("method", "use_for_series"),
    ],
)
def test_phase_rows_and_method_choice_are_series_not_selected_run(
    series, action, item_id
):
    value = scopes.row_scope(context(series), action, item_id)
    assert value.scope == "series"
    assert value.line == "Applies to: whole series AC (2 runs)"
    assert not value.run_selector_enabled
    assert "whole series" in value.note


@pytest.mark.parametrize("item_id", ["overview", "sources", "new_analysis", "wasserstein"])
def test_jobs_pages_are_series_pages_that_perform_no_calculation(series, item_id):
    value = scopes.row_scope(context(series), "jobs", item_id)
    assert value.scope == "series" and not value.run_selector_enabled
    assert value.line == "Applies to: whole series AC (2 runs)"
    assert "performs no calculation" in value.note


def test_scope_summary_names_the_whole_series_or_the_run_for_the_header_line(series, tmp_path):
    ctx = context(series)
    assert scopes.scope_summary(ctx, None) == ""
    assert scopes.scope_summary(ctx, scopes.series_scope(ctx)) == "Whole series (2 runs)"
    assert scopes.scope_summary(ctx, scopes.row_scope(ctx, "analysis", "cropping")) == "Run AC28 only"
    assert scopes.scope_summary(ctx, scopes.row_scope(ctx, "calibration", "color")) == COMMON_LINE
    unknown = scopes.load_scope_context(tmp_path / "missing", "x", series_label="X")
    assert scopes.scope_summary(unknown, scopes.series_scope(unknown)) == "Whole series (run count unavailable)"


@pytest.mark.parametrize("item_id", ["color", "mass", "default_mass", "mass_optuna"])
def test_common_origin_only_when_loaded_branch_is_inherited(series, item_id):
    value = scopes.row_scope(context(series), "calibration", item_id)
    # Common settings, but the action runs on the selected run (N141).
    assert value.scope == "run" and value.run_selector_enabled
    assert value.line == COMMON_LINE
    assert "Shared calibration settings from common.toml" in value.note
    assert "other runs may override" in value.note
    assert "Existing actions use the selected run" in value.note


def test_run_mass_override_does_not_change_independent_common_colour_scope(series):
    _root, _manifest, _common, run, loaded = series
    run.write_bytes(run.read_bytes() + b'[calibration.mass]\nembedding = "run-mass"\n')
    loaded["calibration"]["mass"]["embedding"] = "run-mass"
    current = context(series)
    assert scopes.row_scope(current, "calibration", "mass").scope == "run"
    assert scopes.row_scope(current, "calibration", "color").line == COMMON_LINE


def test_even_identical_explicit_run_override_is_run_provenance(series):
    _root, _manifest, _common, run, _loaded = series
    run.write_bytes(
        run.read_bytes() + b'[calibration.mass]\nembedding = "shared-mass"\n'
    )
    value = scopes.row_scope(context(series), "calibration", "mass")
    assert value.scope == "run"
    assert (
        "explicitly overrides"
        in scopes.common_calibration_origin(context(series), "mass")[1]
    )
    assert value.note == "Uses the selected run."


def test_stale_or_edited_loaded_merged_branch_cannot_gain_common_scope(series):
    series[4]["calibration"]["mass"]["embedding"] = "older-merged-value"
    value = scopes.row_scope(context(series), "calibration", "mass")
    assert value.scope == "run"
    assert (
        "stale merged" in scopes.common_calibration_origin(context(series), "mass")[1]
    )
    assert value.note == "Uses the selected run."


def test_unbound_external_config_cannot_claim_common_or_selected_run(series):
    current = replace(context(series), loaded_path=Path("C:/external/another-run.toml"))
    value = scopes.row_scope(current, "calibration", "mass")
    assert value.scope == "run"
    assert "run not confirmed" in value.line
    assert "not bound" in scopes.common_calibration_origin(current, "mass")[1]
    assert value.note == "Uses the selected run."


@pytest.mark.parametrize(
    "item_id", ["cropping", "mass", "segmentation", "fingers", "thresholding", "volume"]
)
def test_original_analysis_remains_one_run_even_with_common_calibration(
    series, item_id
):
    value = scopes.row_scope(context(series), "analysis", item_id)
    assert value.scope == "run"
    assert value.line == "Applies to: run AC28 only"
    assert value.run_selector_enabled
    assert value.note.startswith("Uses the selected run.")
    if item_id == "mass":
        assert "Calibration defaults come from common.toml" in value.note


@pytest.mark.parametrize("item_id", ["reset", "delete"])
def test_danger_rows_are_selected_config_invocations_not_invented_series_jobs(
    series, item_id
):
    value = scopes.row_scope(context(series), "calibration", item_id)
    assert value.scope == "run"
    assert value.line == "Applies to: run AC28 only"
    assert "selected run" in value.note


def test_no_calibration_section_defaults_to_actual_run_scope(series):
    _root, _manifest, common, _run, loaded = series
    common.write_bytes(b"[rig]\nheight = 0.5\n")
    loaded.pop("calibration")
    value = scopes.row_scope(context(series), "calibration", "mass")
    assert value.scope == "run"
    assert (
        "No [calibration.mass]"
        in scopes.common_calibration_origin(context(series), "mass")[1]
    )
    assert value.note == "Uses the selected run."


def test_unknown_row_has_no_fabricated_scope_and_missing_run_is_not_first_run(series):
    current = replace(context(series), selected_run="")
    assert scopes.row_scope(current, "analysis", "imaginary") is None
    assert scopes.row_scope(current, "setup", "rig") is None
    assert "choose a run" in scopes.row_scope(current, "calibration", "mass").line


def test_one_context_reads_only_three_bound_tomls_and_pure_row_labels_do_no_io(
    series, monkeypatch
):
    root, manifest, common, run, loaded = series
    original = Path.open
    paths = []

    def allowed(path, *args, **kwargs):
        paths.append(path)
        assert path in {manifest, common, run}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", allowed)
    before = copy.deepcopy(loaded)
    current = scopes.load_scope_context(
        root, "ffac", "ac28", root / "config/merged/ac28.toml", loaded
    )
    assert paths == [manifest, common, run]
    for item in ("color", "mass", "default_mass", "mass_optuna", "reset", "delete"):
        scopes.row_scope(current, "calibration", item)
    assert paths == [manifest, common, run]
    assert loaded == before


def test_missing_or_broken_metadata_never_claims_common_or_writes_files(tmp_path):
    before = set(tmp_path.iterdir())
    missing = scopes.load_scope_context(tmp_path / "missing", "demo", "run1")
    assert missing.notes
    assert scopes.row_scope(missing, "calibration", "mass").scope == "run"
    assert set(tmp_path.iterdir()) == before


def test_custom_manifest_paths_are_respected(series):
    root, manifest, common, run, loaded = series
    custom = root / "shared.toml"
    custom.write_bytes(common.read_bytes())
    manifest.write_bytes(
        manifest.read_bytes()
        + b'[paths]\ncommon_config = "shared.toml"\nmerged_configs = "controller-configs"\n'
    )
    current = scopes.load_scope_context(
        root, "ffac", "ac28", root / "controller-configs/ac28.toml", loaded
    )
    assert current.common_path == custom
    assert current.merged_path == root / "controller-configs/ac28.toml"
    assert scopes.row_scope(current, "calibration", "mass").line == COMMON_LINE


def test_unloaded_explicit_canonical_merged_file_is_metadata_only_fallback(
    series, monkeypatch
):
    root, manifest, common, run, _loaded = series
    merged = root / "config/merged/ac28.toml"
    merged.parent.mkdir(parents=True)
    merged.write_bytes(common.read_bytes() + b'\n[data]\nresults = "per-run-results"\n')
    original = Path.open
    reads = []

    def allowed(path, *args, **kwargs):
        reads.append(path)
        assert path in {manifest, common, run, merged}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", allowed)
    current = scopes.load_scope_context(root, "ffac", "ac28", merged)
    assert reads == [manifest, common, run, merged]
    assert scopes.row_scope(current, "calibration", "mass").line == COMMON_LINE


def test_unloaded_external_path_is_not_read_or_assumed_common(series, monkeypatch):
    root, manifest, common, run, _loaded = series
    original = Path.open

    def allowed(path, *args, **kwargs):
        assert path in {manifest, common, run}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", allowed)
    current = scopes.load_scope_context(root, "ffac", "ac28", root / "external.toml")
    value = scopes.row_scope(current, "calibration", "mass")
    assert value.scope == "run"
    assert "run not confirmed" in value.line


def test_valid_empty_run_list_reports_zero_not_unknown(tmp_path):
    root = tmp_path / "empty-series"
    root.mkdir()
    (root / "series.toml").write_bytes(
        b'schema_version = 1\nseries_id = "EMPTY"\n[runs]\nids = []\n'
    )
    current = scopes.load_scope_context(root, "empty")
    assert current.run_count_known
    assert (
        scopes.series_scope(current).line == "Applies to: whole series EMPTY (0 runs)"
    )


def test_missing_or_invalid_manifest_reports_unknown_count(tmp_path):
    missing = scopes.load_scope_context(tmp_path / "absent", "missing")
    assert not missing.run_count_known
    assert "run count unavailable" in scopes.series_scope(missing).line
    root = tmp_path / "invalid"
    root.mkdir()
    (root / "series.toml").write_bytes(
        b'schema_version = 1\nseries_id = "INVALID"\n[runs]\nids = "not a list"\n'
    )
    invalid = scopes.load_scope_context(root, "invalid")
    assert not invalid.run_count_known
    assert "run count unavailable" in scopes.series_scope(invalid).line


def test_product_notes_have_no_cli_or_internal_key_descriptions(series):
    current = context(series)
    selections = [
        ("calibration", key)
        for key in ("color", "mass", "default_mass", "mass_optuna", "reset", "delete")
    ]
    selections += [
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
    selections += [("phase_model", "training")]
    for action, item in selections:
        value = scopes.row_scope(current, action, item)
        assert "--config" not in value.note
        assert "[calibration." not in value.note
        assert "CLI" not in value.note
        assert "data.cache" not in value.note
