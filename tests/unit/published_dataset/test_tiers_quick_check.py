"""Tier nesting and tier-against-job frames; the quick check's frame dependencies."""

import json

import pytest

from darsia.fluidflower.published_dataset import quick_check, tiers

JOB = {"r1": [{"raw": {"name": f"IMG_{i}.ARW", "sha256": f"{i:064x}"}} for i in range(3)],
       "r2": [{"raw": {"name": "IMG_9.ARW", "sha256": "9" * 64}}]}


def _write(tmp_path, document):
    path = tmp_path / "tiers.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_read_tiers_and_refuse_bad_files(tmp_path):
    good = {"format": tiers.TIERS_FORMAT, "tiers": {"S50": {"runs": {"r1": ["IMG_0.ARW"]}}}}
    assert tiers.read_tiers(_write(tmp_path, good)) == {"S50": {"r1": ["IMG_0.ARW"]}}
    for document in ({"format": "other"}, {"format": tiers.TIERS_FORMAT, "tiers": {}},
                     {"format": tiers.TIERS_FORMAT, "tiers": {"S70": {"runs": {"r1": ["a"]}}}},
                     {"format": tiers.TIERS_FORMAT, "tiers": {"S50": {"runs": {"r1": []}}}},
                     {"format": tiers.TIERS_FORMAT, "tiers": {"S50": {"runs": {"r1": ["a", "A"]}}}}):
        with pytest.raises(ValueError):
            tiers.read_tiers(_write(tmp_path, document))


def test_tiers_must_nest():
    nested = {"S50": {"r1": ["a"]}, "S100": {"r1": ["a", "b"]}, "S200": {"r1": ["A", "b", "c"]}}
    assert tiers.nesting_problems(nested) == []
    broken = {"S50": {"r1": ["a", "z"], "r2": ["x"]}, "S100": {"r1": ["a", "b"]}}
    problems = tiers.nesting_problems(broken)
    assert problems == ["Tier S50, run r1: 1 frame(s) not in S100, first z", "Tier S50 has run r2, which S100 lacks"]


def test_a_tier_must_hold_exactly_the_jobs_frames():
    exact = {"S50": {"r1": ["IMG_0.ARW", "IMG_1.ARW", "2".rjust(64, "0")], "r2": ["img_9.arw"]}}
    assert tiers.tier_job_problems("S50", exact, JOB) == []
    assert tiers.frame_key("IMG_1.ARW", JOB["r1"]) == 1 and tiers.frame_key("nope", JOB["r1"]) is None
    fewer = {"S50": {"r1": ["IMG_0.ARW", "IMG_1.ARW"], "r2": ["IMG_9.ARW"]}}
    assert "the job analysed 3" in tiers.tier_job_problems("S50", fewer, JOB)[0]
    other = {"S50": {"r1": ["IMG_0.ARW", "IMG_1.ARW", "IMG_7.ARW"], "r3": ["x"]}}
    problems = tiers.tier_job_problems("S50", other, JOB)
    assert any("lacks the job's runs: r2" in p for p in problems)
    assert any("runs the job did not run: r3" in p for p in problems)
    assert any("did not analyse, first IMG_7.ARW" in p for p in problems)
    assert tiers.tier_job_problems("S100", exact, JOB) == ["The tier file does not declare S100"]


def _config(*, later=False, broken=False):
    frames, previous = [], "b" * 64
    for index in range(3):
        sha = f"{index + 1:064x}"
        frames.append({"case_id": f"r_f{index}", "frame_index": index + 1, "elapsed_h": index + 0.5,
                       "source": {"source_sha256": sha, "previous_source_sha256": previous}})
        previous = sha
    if later:
        frames[0]["phase_roles"] = {"anchor": {"source_sha256": frames[2]["source"]["source_sha256"]}}
    if broken:
        frames[2]["source"]["previous_source_sha256"] = "e" * 64
    return {"run": "r", "history_policy": "camera chronology", "frames": frames,
            "phase": {"initial_previous": {"source": {"source_sha256": "b" * 64}}}}


def test_frame_dependencies_name_earlier_and_later_frames():
    rows = quick_check.frame_dependencies(_config(later=True))
    assert rows[0] == {"case_id": "r_f0", "earlier": [], "later": ["r_f2"]}
    assert rows[1]["earlier"] == ["r_f0"] and rows[2]["later"] == []
    assert quick_check.history_chain(_config()) == []
    assert quick_check.history_chain(_config(broken=True)) == ["r_f2: its predecessor is not the previous frame"]


def test_quick_check_is_valid_only_without_later_frames():
    valid = quick_check.quick_check_validity("native-phase-n1", [_config()])
    assert valid["valid"] is True and valid["runs"][0]["frames_naming_later_frames"] == []
    later = quick_check.quick_check_validity("native-phase-n1", [_config(later=True)])
    assert later["valid"] is False and "names later frame(s): r_f2" in later["reasons"][0]
    assert quick_check.quick_check_validity("unknown-pipeline", [_config()])["valid"] is False
    assert quick_check.quick_check_validity("n1", [_config(broken=True)])["valid"] is False
