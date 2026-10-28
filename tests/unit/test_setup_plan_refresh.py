"""Direct saved-choice refresh, without mocked planners or GUI callbacks."""
import datetime as dt

import toml

from darsia.fluidflower.image_normalisation import MATCH_REFERENCE, save_choice
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia.fluidflower.setup_plan_refresh import inspect_refresh


def test_real_saved_normalization_branch_preserves_original_toml_types(tmp_path):
    series = tmp_path / "series"
    series.mkdir()
    path = series / "series.toml"
    original = toml.dumps(dict(schema_version=1,series_id="synthetic",
        recorded_at=dt.datetime(2025,1,2,3,4,tzinfo=dt.timezone.utc),
        runs=dict(ids=["run1","run2"]))).encode()
    path.write_bytes(original)
    old_pin = descriptor(path)
    updated = save_choice(series,"synthetic",["run1","run2"],MATCH_REFERENCE,"run2",expected_bytes=original)
    result = inspect_refresh(tmp_path / "unstarted-job",dict(series_manifest=old_pin))
    assert result["previous_bytes"] == original
    assert result["current_bytes"] == updated
    assert result["previous_document"]["recorded_at"] == result["current_document"]["recorded_at"]
    assert result["current_document"]["image_normalisation"]["reference_run"] == "run2"

def test_added_storage_aliases_with_a_new_choice_still_refresh_but_other_changes_do_not(tmp_path):
    import pytest
    from darsia.fluidflower.image_normalisation import EQUALISE_SAND
    from darsia.fluidflower.setup_plan_refresh import _normalisation_only
    series = tmp_path / "series"
    series.mkdir()
    path = series / "series.toml"
    base = dict(schema_version=1, series_id="synthetic", runs=dict(ids=["run1", "run2"]),
                storage=dict(path_aliases={"//nas/share/a.toml": "copies/a.toml"}))
    original = toml.dumps(base).encode()
    path.write_bytes(original)
    old_pin = descriptor(path)
    relocated = dict(base, storage=dict(path_aliases={"//nas/share/a.toml": "copies/a.toml",
                                                      "//nas/share/b.toml": "copies/b.toml"}))
    moved = toml.dumps(relocated).encode()
    path.write_bytes(moved)
    updated = save_choice(series, "synthetic", ["run1", "run2"], EQUALISE_SAND, "run2",
                          color_method="uniform24", exclusion_band_mm=40, expected_bytes=moved)
    job = tmp_path / "job"
    (job / "setup-controls").mkdir(parents=True)
    (job / "setup-controls" / "series-manifest.toml").write_bytes(original)  # the retained Setup snapshot
    result = inspect_refresh(job, dict(series_manifest=old_pin))
    assert result["current_bytes"] == updated
    assert result["current_document"]["image_normalisation"]["exclusion_band"]["width_mm"] == 40.0
    changed = toml.dumps(dict(relocated, storage=dict(path_aliases={"//nas/share/a.toml": "other/a.toml"}),
                              image_normalisation=result["current_document"]["image_normalisation"])).encode()
    with pytest.raises(ValueError, match="beyond the lighting choice"):
        _normalisation_only(original, changed, series)
    other = toml.dumps(dict(relocated, extra=1,
                            image_normalisation=result["current_document"]["image_normalisation"])).encode()
    with pytest.raises(ValueError, match="beyond the lighting choice"):
        _normalisation_only(original, other, series)
    with pytest.raises(ValueError, match="beyond the lighting choice"):
        _normalisation_only(original, moved, series)  # aliases alone are no lighting choice
