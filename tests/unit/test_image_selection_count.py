"""Exact-count selection uses local protocol and quality metadata, never RAW."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.image_selection_count import propose_image_count
from darsia.fluidflower.image_selection_proposal import QUALITY_RULE, load_saved_quality


def metadata(times, **kwargs):
    return SimpleNamespace(
        frames=[
            dict(source_path=f"not-read/image_{index}.raw", elapsed_h=time)
            for index, time in enumerate(times)
        ],
        run_id="ac01",
        series_root="not-read/series",
        injection_start_h=kwargs.pop("injection_start_h", 0),
        **kwargs,
    )


def paths(meta, indices):
    return [meta.frames[index]["source_path"] for index in indices]


def saved_quality(tmp_path, meta, values):
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(
            dict(
                format="fluidflower-quality-camera-grid-v1",
                class_measurement=QUALITY_RULE,
                selected=[
                    dict(
                        **meta.frames[index],
                        source_sha256=f"{index:064x}",
                        quality={"class": value},
                    )
                    for index, value in values.items()
                ],
            )
        ),
        encoding="utf8",
    )
    return load_saved_quality(path, meta)


def test_exact_80_preserves_baseline_all_saved_images_and_late_bound_without_raw(
    monkeypatch,
):
    meta = metadata(range(-5, 201))
    retained = paths(meta, [0, 5, 10, 11, 25, 75, 150, 205])
    before = deepcopy(meta.frames)

    def no_file_access(*args, **kwargs):
        raise AssertionError("Selection must not open or inspect image files")

    monkeypatch.setattr(Path, "open", no_file_access)
    monkeypatch.setattr(Path, "is_file", no_file_access)
    result = propose_image_count(meta, retained)
    assert result["count"] == len(result["selected_paths"]) == 80
    assert len(set(result["selected_paths"])) == 80
    assert result["retained_paths"] == retained
    assert set(retained) <= set(result["selected_paths"])
    assert result["selected"][0]["elapsed_h"] == -5
    assert result["selected"][-1]["elapsed_h"] == 200
    assert sum(row["retained"] for row in result["selected"]) == len(retained)
    assert result["unrecorded_quality"] == 80
    assert all(
        row["quality"] == {"class": "Not recorded", "recorded": False}
        for row in result["selected"]
    )
    assert meta.frames == before


def test_unequal_original_intervals_are_filled_before_subdividing():
    meta = metadata([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 26, 51, 76, 101])
    retained = paths(meta, [0, 4, 8, 12])
    first = propose_image_count(meta, retained, 5)
    assert [row["elapsed_h"] for row in first["selected"]] == [-1, -0.5, 0, 1, 101]
    filled = propose_image_count(meta, retained, 7)
    assert [row["elapsed_h"] for row in filled["selected"]] == [
        -1,
        -0.5,
        0,
        0.5,
        1,
        51,
        101,
    ]
    subdivided = propose_image_count(meta, retained, 8)
    assert -0.75 in [row["elapsed_h"] for row in subdivided["selected"]]


def test_ties_are_earlier_and_repeated_calls_identical():
    meta = metadata([-1, -0.75, -0.25, 0])
    result = propose_image_count(meta, paths(meta, [3, 0]), 3)
    assert result["selected_paths"] == paths(meta, [0, 1, 3])
    assert result == propose_image_count(meta, paths(meta, [0, 3]), 3)


def test_candidate_times_outside_retained_window_are_not_used():
    meta = metadata([-5, -1, 0, 1, 2, 3, 100])
    result = propose_image_count(meta, paths(meta, [1, 5]), 5)
    assert result["selected_paths"] == paths(meta, [1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Only 5 eligible distinct"):
        propose_image_count(meta, paths(meta, [1, 5]), 6)


@pytest.mark.parametrize(
    "flag", ["blacklisted", "is_blacklisted", "excluded", "missing"]
)
def test_blacklisted_or_missing_retained_image_fails_and_candidate_is_excluded(flag):
    meta = metadata([-1, 0, 1, 2, 3])
    meta.frames[2][flag] = True
    with pytest.raises(ValueError, match="missing, blacklisted or unavailable"):
        propose_image_count(meta, paths(meta, [0, 2, 4]), 4)
    result = propose_image_count(meta, paths(meta, [0, 4]), 4)
    assert result["selected_paths"] == paths(meta, [0, 1, 3, 4])


def test_quality_is_source_bound_reject_is_excluded_and_unknown_stays_unknown(tmp_path):
    meta = metadata([-1, 0, 1, 2, 3])
    quality = saved_quality(tmp_path, meta, {0: "ok", 1: "minor", 2: "reject"})
    result = propose_image_count(meta, paths(meta, [0, 4]), 4, quality=quality)
    assert result["selected_paths"] == paths(meta, [0, 1, 3, 4])
    assert [row["quality"]["class"] for row in result["selected"]] == [
        "ok",
        "minor",
        "Not recorded",
        "Not recorded",
    ]
    assert result["selected"][0]["quality"]["source_sha256"] == "0" * 64
    assert result["unrecorded_quality"] == 2
    with pytest.raises(ValueError, match="saved quality reject"):
        propose_image_count(meta, paths(meta, [0, 2, 4]), 4, quality=quality)
    altered = deepcopy(quality)
    altered["classes"][meta.frames[2]["source_path"]]["class"] = "ok"
    with pytest.raises(ValueError, match="exact source receipt"):
        propose_image_count(meta, paths(meta, [0, 4]), 4, quality=altered)


def test_distinct_source_identities_at_same_time_do_not_inflate_count():
    meta = metadata([-1, 0, 0, 1, 2])
    result = propose_image_count(meta, paths(meta, [0, 4]), 4)
    assert result["selected_paths"] == paths(meta, [0, 1, 3, 4])
    with pytest.raises(ValueError, match="Only 4 eligible distinct"):
        propose_image_count(meta, paths(meta, [0, 4]), 5)
    with pytest.raises(ValueError, match="distinct acquisition times"):
        propose_image_count(meta, paths(meta, [0, 1, 2, 4]), 4)


def test_duplicate_protocol_paths_use_original_rules():
    meta = metadata([-1, 0, 1])
    meta.frames.append(dict(meta.frames[1]))
    assert propose_image_count(meta, paths(meta, [0, 2]), 3)["count"] == 3
    meta.frames[-1]["elapsed_h"] = 0.5
    with pytest.raises(ValueError, match="conflicting protocol times"):
        propose_image_count(meta, paths(meta, [0, 2]), 3)


def test_saved_selection_must_include_baseline_or_actual_first_usable():
    meta = metadata([-1, 0, 1])
    with pytest.raises(ValueError, match="must precede injection"):
        propose_image_count(meta, paths(meta, [1, 2]), 2)
    meta.injection_start_h = -2
    with pytest.raises(ValueError, match="first usable image explicitly"):
        propose_image_count(meta, paths(meta, [1, 2]), 2)
    assert propose_image_count(meta, paths(meta, [0, 2]), 3)["count"] == 3


def test_retained_count_can_be_returned_without_new_images():
    meta = metadata([-1])
    result = propose_image_count(meta, paths(meta, [0]), 1)
    assert result["selected_paths"] == paths(meta, [0])
    with pytest.raises(ValueError, match="Only 1 eligible distinct"):
        propose_image_count(meta, paths(meta, [0]), 2)


@pytest.mark.parametrize("count", [0, -1, True, 3.0])
def test_invalid_count(count):
    meta = metadata([-1, 0, 1])
    with pytest.raises(ValueError, match="positive integer"):
        propose_image_count(meta, paths(meta, [0, 2]), count)


def test_missing_selection_reducing_count_duplicates_and_unknown_sources_fail():
    meta = metadata([-1, 0, 1])
    with pytest.raises(ValueError, match="saved image selection is required"):
        propose_image_count(meta, [], 3)
    with pytest.raises(ValueError, match="smaller than the saved selection"):
        propose_image_count(meta, paths(meta, [0, 1, 2]), 2)
    with pytest.raises(ValueError, match="duplicate source paths"):
        propose_image_count(meta, paths(meta, [0, 0, 2]), 3)
    with pytest.raises(ValueError, match="missing, blacklisted or unavailable"):
        propose_image_count(meta, [*paths(meta, [0, 2]), "absent.raw"], 3)
