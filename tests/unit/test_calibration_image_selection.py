from copy import deepcopy
import tomllib

import pytest

from darsia.fluidflower.calibration_image_selection import (
    calibration_image_count, calibration_statistics_selection,
    save_calibration_image_selection,
)
from darsia.fluidflower.series_method import ManifestChangedError


def test_calibration_count_is_independent_of_setup_and_production():
    assert calibration_image_count({"image_selection": {
        "color_paths": {"max_images_per_run": 15}, "production": {"max_images_per_run": 80}}}) == 30
    for invalid in (True, 2, 0, 1001, 30.):
        with pytest.raises(ValueError):
            calibration_image_count({"calibration": {"mass": {"image_selection": {"max_images_per_run": invalid}}}})


def test_shared_targets_filter_objective_and_keep_original_anchor_and_history():
    times = [index / 2 for index in range(1, 81)]
    statistics = {run: [dict(frame_index=index, selected_target_h=time,
        elapsed_h=time + .001, case_id=f"{run}-{index}") for index, time in enumerate(times)]
        for run in ("first", "second")}
    original = deepcopy(statistics)
    rules = dict(stage1_window_h=[.5, 2.5], stage2_start_h=3., anchor_target_h=2.5)
    selected, chosen, record = calibration_statistics_selection(statistics, rules, 30)
    assert len(chosen["target_times_h"]) == 30
    assert record["max_images_per_run"] == 30
    assert chosen["target_times_h"][-1] == 40.
    assert [entry["selected_target_h"] for entry in selected["first"]] == [
        entry["selected_target_h"] for entry in selected["second"]]
    assert statistics == original and "target_times_h" not in rules
    # With only three observations the exact original anchor remains a support
    # row, but cannot enter either stage's numerical objective.
    selected, chosen, record = calibration_statistics_selection(statistics, rules, 3)
    assert chosen["target_times_h"] == [.5, 3., 40.]
    assert [entry["selected_target_h"] for entry in selected["first"]] == [.5, 2.5, 3., 40.]
    assert record["runs"]["first"] == dict(observations=3, anchor_support=1)
    assert sum(entry["selected_target_h"] in chosen["target_times_h"] for entry in selected["first"]) == 3
    assert statistics == original


def test_selection_respects_existing_rules_and_short_run_coverage():
    statistics = {"long": [dict(frame_index=i, selected_target_h=float(i)) for i in range(1, 41)],
                  "short": [dict(frame_index=i, selected_target_h=float(i)) for i in range(1, 11)]}
    rules = dict(stage1_window_h=[1, 2], stage2_start_h=3, anchor_target_h=2,
                 target_times_h=[1., 3., 5., 9., 20., 40.])
    selected, chosen, _ = calibration_statistics_selection(statistics, rules, 3)
    assert chosen["target_times_h"] == [1., 3., 40.]
    assert [entry["selected_target_h"] for entry in selected["short"]] == [1., 2., 3.]
    assert all(entry["frame_index"] <= 10 for entry in selected["short"])


def test_saved_count_uses_lossless_cas_and_preserves_source_pool(tmp_path):
    path = tmp_path / "series.toml"
    raw = b'\xef\xbb\xbfschema_version = 1\r\n# operator note\r\n[image_selection.color_paths]\r\nmax_images_per_run = 15\r\n[future]\r\nopaque = [1, 2]\r\n'
    path.write_bytes(raw)
    saved = save_calibration_image_selection(tmp_path, 30, expected_bytes=raw)
    document = tomllib.loads(saved.decode("utf-8-sig"))
    assert document["calibration"]["mass"]["image_selection"]["max_images_per_run"] == 30
    assert document["image_selection"]["color_paths"]["max_images_per_run"] == 15
    assert document["future"]["opaque"] == [1, 2]
    assert b"\r\n" in saved and saved.startswith(b'\xef\xbb\xbf')
    with pytest.raises(ManifestChangedError):
        save_calibration_image_selection(tmp_path, 20, expected_bytes=raw)
    assert path.read_bytes() == saved
