"""Check-score reader: held-out checks 1-3 validated by structure, as plain rows.

Synthetic documents only; nothing is computed from the numbers and nothing is
written.
"""

import copy
import hashlib
import json

import pytest

from darsia.fluidflower import calibration_checks as checks
from darsia.fluidflower.calibration_checks import (
    CheckScoreError,
    normalise_check_scores,
)


def _run(delta, m_over_d, w_over_d, counted=True, run_set="heldout"):
    return {
        "delta": delta,
        "M_over_d": m_over_d,
        "W_over_d": w_over_d,
        "counted": counted,
        "set": run_set,
    }


def _time(per_run, median=None, passed=None):
    entry = {
        "runs_with_frame": len(per_run),
        "runs_counted": sum(run["counted"] for run in per_run.values()),
        "assessable": median is not None,
        "per_run": per_run,
    }
    if median is not None:
        entry.update(median_delta=median, time_pass=passed)
    return entry


def _rule(name, status, need, per_time):
    return {
        "rule": name,
        "status": status,
        "assessable_times": sum(entry["assessable"] for entry in per_time.values()),
        "need_assessable": need,
        "per_time": per_time,
    }


def score_document():
    """A small synthetic check-score document of the expected structure."""
    return {
        "format": "synthetic-format-name",
        "verdict": "PASS",
        "counts": {
            "heldout_rows": 12,
            "heldout_runs": ["r01", "r02", "r03"],
            "loro_runs": ["r04"],
        },
        "integrity_problems": [],
        "check1": {
            "status": "PASS",
            "per_time": {
                "1.0": {
                    "runs": 3,
                    "median_abs_error": 0.0412,
                    "group_median_ratio": {"group_a": 0.951, "group_b": 1.0234},
                    "per_run_ratio": {"r01": 0.951, "r02": 0.9123, "r03": 1.0234},
                    "time_pass": True,
                    "missing": [],
                },
                "2.5": {
                    "runs": 2,
                    "median_abs_error": 0.0301,
                    "group_median_ratio": {"group_a": 0.97, "group_b": 1.03},
                    "per_run_ratio": {"r01": 0.97, "r03": 1.03},
                    "time_pass": True,
                    "missing": ["r02"],
                },
            },
        },
        "check2": {
            "primary": _rule(
                "capacity_rule",
                "PASS",
                2,
                {
                    "4.0": _time(
                        {"r01": _run(0.0, 0.95, 0.01), "r02": _run(0.052, 0.9, 0.02)},
                        0.026,
                        True,
                    ),
                    "8.0": _time(
                        {"r01": _run(0.02, 0.93, 0.01), "r02": _run(0.04, 0.91, 0.03)},
                        0.03,
                        True,
                    ),
                },
            ),
            "fallback": _rule(
                "zero_rule",
                "PASS",
                2,
                {
                    "4.0": _time(
                        {"r01": _run(0.01, 0.95, 0.0), "r02": _run(0.06, 0.9, 0.0)},
                        0.035,
                        True,
                    ),
                    "8.0": _time(
                        {"r01": _run(0.03, 0.93, 0.0), "r02": _run(0.05, 0.91, 0.0)},
                        0.04,
                        True,
                    ),
                },
            ),
            "decision": {"deciding_rule": "capacity_rule", "status": "PASS"},
        },
        "check3": {
            "primary": _rule(
                "capacity_rule",
                "NOT_ASSESSABLE",
                2,
                {
                    "50.0": _time(
                        {
                            "r01": _run(0.0, 0.97, 0.02),
                            "r04": _run(
                                0.1, 0.8, 0.09, counted=False, run_set="fit_loro"
                            ),
                        },
                        0.0,
                        True,
                    ),
                    "60.0": _time(
                        {
                            "r04": _run(
                                None, 0.85, None, counted=False, run_set="fit_loro"
                            )
                        }
                    ),
                },
            ),
            "fallback": _rule(
                "zero_rule",
                "FAIL",
                2,
                {
                    "50.0": _time(
                        {
                            "r01": _run(0.01, 0.97, 0.0),
                            "r04": _run(0.2, 0.8, 0.0, run_set="fit_loro"),
                        },
                        0.105,
                        False,
                    ),
                    "60.0": _time(
                        {"r04": _run(0.12, 0.85, 0.0, run_set="fit_loro")}, 0.12, False
                    ),
                },
            ),
            "decision": {"deciding_rule": "zero_rule", "status": "FAIL"},
        },
    }


def write_scores(path, document=None):
    """Write a check-score document as JSON and return its bytes."""
    raw = json.dumps(
        score_document() if document is None else document, indent=1
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _tree_texts(rows):
    return [row.texts for row in rows]


def test_verdict_checks_rules_and_sets_are_read_by_structure():
    scores = normalise_check_scores(score_document(), name="score.json")
    assert scores.verdict == "PASS" and scores.name == "score.json"
    first, second = scores.check1.times
    assert (first.time, first.runs, first.median_abs_error) == ("1.0", 3, 0.0412)
    assert first.group_ratios == (("group_a", 0.951), ("group_b", 1.0234))
    assert dict(first.run_ratios) == {"r01": 0.951, "r02": 0.9123, "r03": 1.0234}
    assert second.missing == ("r02",) and second.time_pass is True
    check2, check3 = scores.rule_checks
    assert (check2.key, check2.status, check2.deciding_role) == (
        "check2",
        "PASS",
        "primary",
    )
    assert (check3.key, check3.status, check3.deciding_role) == (
        "check3",
        "FAIL",
        "fallback",
    )
    primary = check3.rules[0]
    assert (primary.role, primary.name, primary.status) == (
        "primary",
        "capacity_rule",
        "NOT_ASSESSABLE",
    )
    assert (primary.assessable_times, primary.need_assessable) == (1, 2)
    late = primary.times[1]
    # A time that is not assessable has no median and no pass mark of its own.
    assert (late.time, late.assessable, late.median_delta, late.time_pass) == (
        "60.0",
        False,
        None,
        None,
    )
    (loro,) = late.runs
    assert (loro.run, loro.delta, loro.w_over_d, loro.counted, loro.loro) == (
        "r04",
        None,
        None,
        False,
        True,
    )
    assert not primary.times[0].runs[0].loro
    assert scores.counts == (
        ("Held-out rows", "12"),
        ("Held-out runs", "3: r01, r02, r03"),
        ("LORO-predicted fit runs", "1: r04"),
    )
    assert scores.integrity_problems == () and scores.thresholds == ()
    assert scores.entries == (("format", "synthetic-format-name"),)


def test_any_format_name_or_none_reads_the_same_checks():
    plain = normalise_check_scores(score_document())
    document = score_document()
    del document["format"]
    assert normalise_check_scores(document).rule_checks == plain.rule_checks
    document["format"] = "another-name-v9"
    assert normalise_check_scores(document).check1 == plain.check1


def test_optional_sections_may_be_absent():
    document = score_document()
    for key in ("counts", "integrity_problems", "format"):
        del document[key]
    scores = normalise_check_scores(document)
    assert (
        scores.counts is None
        and scores.integrity_problems is None
        and scores.entries == ()
    )
    rows = dict(checks.summary_rows([scores]))
    assert rows["Integrity problems"] == ("Not in the file",)
    assert "Counts: Held-out runs" not in rows


def _without(path):
    def change(document):
        *parents, key = path
        target = document
        for parent in parents:
            target = target[parent]
        del target[key]

    return change


def _set(path, value):
    def change(document):
        *parents, key = path
        target = document
        for parent in parents:
            target = target[parent]
        target[key] = value

    return change


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda document: document.clear() or document.update(x=1),
            "The file has no 'verdict'.",
        ),
        (_without(("check3",)), "The file has no 'check3'."),
        (_set(("check1",), []), "check1 is not a JSON object."),
        (
            _without(("check1", "per_time", "1.0", "median_abs_error")),
            "check1 › per_time › 1.0 has no 'median_abs_error'.",
        ),
        (
            _set(("check1", "per_time", "1.0", "group_median_ratio", "group_a"), "0.9"),
            "check1 › per_time › 1.0 › group_median_ratio › group_a "
            "is not a number.",
        ),
        (
            _set(("check1", "per_time", "2.5", "missing"), "r02"),
            "check1 › per_time › 2.5 › missing is not a list of text.",
        ),
        (_without(("check2", "fallback")), "check2 has no 'fallback'."),
        (
            _without(("check2", "primary", "per_time", "4.0", "median_delta")),
            "check2 › primary › per_time › 4.0 has no 'median_delta'.",
        ),
        (
            _set(("check2", "primary", "per_time", "8.0", "runs_counted"), 2.5),
            "check2 › primary › per_time › 8.0 › runs_counted "
            "is not a whole number.",
        ),
        (
            _set(
                ("check3", "fallback", "per_time", "60.0", "per_run", "r04", "counted"),
                "yes",
            ),
            "check3 › fallback › per_time › 60.0 › per_run › r04 › counted "
            "is not true or false.",
        ),
        (
            _set(("check2", "decision", "deciding_rule"), "other_rule"),
            "check2 › decision › deciding_rule names 'other_rule', which is "
            "neither the primary nor the fallback rule.",
        ),
        (
            _set(("check2", "fallback", "rule"), "capacity_rule"),
            "check2: the primary and the fallback rule are both named "
            "'capacity_rule', so the deciding rule is ambiguous.",
        ),
        (_set(("counts",), [1, 2]), "counts is not a JSON object."),
    ],
)
def test_other_structures_are_refused_with_the_first_mismatch(change, message):
    document = score_document()
    change(document)
    with pytest.raises(CheckScoreError) as error:
        normalise_check_scores(document)
    assert str(error.value) == message


def test_a_document_that_is_not_an_object_is_refused():
    with pytest.raises(CheckScoreError, match="^The file is not a JSON object.$"):
        normalise_check_scores([score_document()])


def test_load_reads_one_file_and_writes_nothing(tmp_path):
    path = tmp_path / "scores" / "score.json"
    path.parent.mkdir()
    raw = write_scores(path)
    before = {
        item: (item.stat().st_mtime_ns, item.read_bytes())
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    scores = checks.load_check_scores(path)
    assert (scores.name, scores.path) == ("score.json", str(path))
    assert scores.sha256 == hashlib.sha256(raw).hexdigest()
    after = {
        item: (item.stat().st_mtime_ns, item.read_bytes())
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    assert after == before
    # A byte-order mark, as some editors write it, is accepted.
    path.write_bytes(b"\xef\xbb\xbf" + raw)
    assert checks.load_check_scores(path).check1 == scores.check1


def test_load_refuses_text_that_is_not_json_and_files_over_the_limit(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.json"
    path.write_text("verdict: pass", encoding="utf-8")
    with pytest.raises(CheckScoreError, match="^The file is not JSON"):
        checks.load_check_scores(path)
    path.write_bytes(b"[" * 200000 + b"]" * 200000)
    with pytest.raises(
        CheckScoreError, match="^The file is nested too deeply to read.$"
    ):
        checks.load_check_scores(path)
    write_scores(path)
    monkeypatch.setattr(checks, "MAX_SCORE_BYTES", 64)
    with pytest.raises(
        CheckScoreError, match="larger than the display limit of 64 bytes"
    ):
        checks.load_check_scores(path)
    with pytest.raises(OSError):
        checks.load_check_scores(tmp_path / "absent.json")


def test_check1_table_shows_the_file_numbers_per_time_group_and_run():
    check = normalise_check_scores(score_document()).check1
    headers, rows = checks.check1_table(check)
    assert headers == ("Item", "Time 1.0", "Time 2.5")
    assert rows == (
        ("Runs", "3", "2"),
        ("Median |M/d − 1|", "0.041", "0.030"),
        ("group_a median M/d", "0.951", "0.970"),
        ("group_b median M/d", "1.023", "1.030"),
        ("Time pass", "Yes", "Yes"),
        ("Missing runs", "None", "r02"),
        ("r01 M/d", "0.951", "0.970"),
        ("r02 M/d", "0.912", "—"),
        ("r03 M/d", "1.023", "1.030"),
    )


def test_rule_tree_marks_the_deciding_rule_and_the_loro_predicted_fit_runs():
    check3 = normalise_check_scores(score_document()).rule_checks[1]
    primary, fallback = checks.rule_tree(check3)
    assert primary.texts[:3] == (
        "Primary rule: capacity_rule",
        "Not assessable",
        "1 of 2 times (2 needed)",
    )
    assert primary.mark == ""
    assert fallback.texts[0] == "Fallback rule: zero_rule (decides this check)"
    assert fallback.mark == "deciding"
    assert _tree_texts(primary.children) == [
        ("Time 50.0", "Pass", "Yes", "1 of 2", "median 0.000", "", "", ""),
        ("Time 60.0", "—", "No", "0 of 1", "—", "", "", ""),
    ]
    held, loro = primary.children[0].children
    assert (
        held.texts == ("r01", "", "", "Counted", "0.000", "0.970", "0.020", "Held-out")
        and held.mark == ""
    )
    assert loro.texts == (
        "r04",
        "",
        "",
        "Not counted",
        "0.100",
        "0.800",
        "0.090",
        "Fit run, LORO-predicted",
    )
    assert loro.mark == "loro"
    assert primary.children[1].children[0].texts[4:7] == ("—", "0.850", "—")
    assert len(checks.RULE_TREE_HEADERS) == len(primary.texts)
    note = checks.rule_check_note(check3)
    assert note.startswith(
        "Decision in the file: Fail, decided by the fallback rule (zero_rule)."
    )
    assert "fit runs, each predicted by its leave-one-run-out (LORO) fold" in note
    assert "LORO" not in checks.rule_check_note(
        normalise_check_scores(score_document()).rule_checks[0]
    )


def test_an_unnamed_set_is_shown_as_written():
    document = score_document()
    document["check2"]["primary"]["per_time"]["4.0"]["per_run"]["r01"][
        "set"
    ] = "descriptive"
    tree = checks.rule_tree(normalise_check_scores(document).rule_checks[0])
    row = tree[0].children[0].children[0]
    assert row.texts[-1] == "descriptive" and row.mark == ""


def test_pass_marks_are_the_files_own_and_none_is_invented():
    document = score_document()
    # Far above any limit one might expect, yet the file says it passed.
    document["check1"]["per_time"]["1.0"]["median_abs_error"] = 0.5
    document["check2"]["primary"]["per_time"]["4.0"]["median_delta"] = 0.9
    scores = normalise_check_scores(document)
    rows = dict((row[0], row[1:]) for row in checks.check1_table(scores.check1)[1])
    assert rows["Median |M/d − 1|"] == ("0.500", "0.030")
    assert rows["Time pass"] == ("Yes", "Yes")
    assert checks.rule_tree(scores.rule_checks[0])[0].children[0].texts[1] == "Pass"
    assert checks.check1_note(scores.check1) == (
        "Status in the file: Pass. The file states no thresholds: "
        "Time pass is the file's "
        "own mark, and no other pass mark is added."
    )
    assert "states no thresholds" in checks.rule_check_note(scores.rule_checks[0])
    assert dict(checks.summary_rows([scores]))["Thresholds"] == (
        "None stated in the file",
    )


def test_thresholds_stated_in_the_file_are_shown_as_written():
    document = score_document()
    document["thresholds"] = {"rule_text": "fixed before any result"}
    document["check1"]["thresholds"] = {
        "median_abs_error_max": 0.1,
        "group_ratio_range": [0.9, 1.1],
    }
    document["check2"]["primary"]["thresholds"] = {
        "median_delta_max": 0.1,
        "counted_runs_min": 7,
    }
    scores = normalise_check_scores(document)
    assert checks.check1_note(scores.check1) == (
        "Status in the file: Pass. Thresholds stated in the file: "
        "median_abs_error_max = 0.1; group_ratio_range = [0.9, 1.1]."
    )
    assert (
        "Primary rule: median_delta_max = 0.1; counted_runs_min = 7."
        in checks.rule_check_note(scores.rule_checks[0])
    )
    assert "states no thresholds" in checks.rule_check_note(scores.rule_checks[1])
    (thresholds,) = dict(checks.summary_rows([scores]))["Thresholds"]
    assert thresholds == (
        "File: rule_text = fixed before any result · "
        "Check 1: median_abs_error_max = 0.1; group_ratio_range = [0.9, 1.1] · "
        "Check 2, primary rule: median_delta_max = 0.1; "
        "counted_runs_min = 7"
    )


def test_summary_rows_put_two_files_side_by_side():
    first = normalise_check_scores(
        score_document(), name="variant_on.json", sha256="a" * 64
    )
    document = score_document()
    document["verdict"] = "FAIL"
    document["check2"]["primary"]["per_time"]["4.0"]["median_delta"] = 0.04
    document["check2"]["primary"]["per_time"]["12.0"] = _time(
        {"r01": _run(0.0, 0.95, 0.01)}, 0.01, True
    )
    document["integrity_problems"] = ["row 3: duplicate"]
    del document["counts"]
    second = normalise_check_scores(document, name="variant_off.json")
    rows = checks.summary_rows([first, second])
    labels = [label for label, _values in rows]
    values = dict(rows)
    assert labels[:3] == ["Verdict", "Integrity problems", "Check 1"]
    assert values["Verdict"] == ("Pass", "Fail")
    assert values["Integrity problems"] == ("None reported", "1: row 3: duplicate")
    assert values["Check 1 at 1.0: median |M/d − 1|"] == ("0.041", "0.041")
    assert values["Check 1 at 2.5: group_b median M/d"] == ("1.030", "1.030")
    assert values["Check 2"] == (
        "Pass, decided by the primary rule",
        "Pass, decided by the primary rule",
    )
    assert values["Check 2, primary rule at 4.0: median δ"] == ("0.026", "0.040")
    # A time only the second file has follows its neighbour, not the end of the list.
    index = labels.index("Check 2, primary rule at 12.0: median δ")
    assert labels[index - 1] == "Check 2, primary rule at 8.0: median δ"
    assert values["Check 2, primary rule at 12.0: median δ"] == (
        "Not in the file",
        "0.010",
    )
    assert values["Check 3, primary rule at 60.0: median δ"] == (
        "Not assessable",
        "Not assessable",
    )
    assert values["Check 3"] == ("Fail, decided by the fallback rule",) * 2
    assert values["Counts: Held-out runs"] == ("3: r01, r02, r03", "Not in the file")
    assert values["File SHA-256"] == ("a" * 64, "—")
    assert values["File entry: format"] == ("synthetic-format-name",) * 2


def test_file_titles_stay_distinct():
    def named(name, path):
        return normalise_check_scores(score_document(), name=name, path=path)

    assert checks.file_titles(
        [named("on.json", "x/on.json"), named("off.json", "x/off.json")]
    ) == ("on.json", "off.json")
    assert checks.file_titles(
        [named("s.json", "a/s.json"), named("s.json", "b/s.json")]
    ) == ("a/s.json", "b/s.json")
    assert checks.file_titles(
        [named("s.json", "a/s.json"), named("s.json", "a/s.json")]
    ) == ("a/s.json (1)", "a/s.json (2)")


@pytest.mark.parametrize(
    "value, text",
    [
        ("PASS", "Pass"),
        ("FAIL", "Fail"),
        ("NOT_ASSESSABLE", "Not assessable"),
        ("", "—"),
    ],
)
def test_status_text_reads_the_files_status(value, text):
    assert checks.status_text(value) == text


def test_number_and_flag_texts():
    assert checks.number_text(None) == "—" and checks.number_text(1.23456) == "1.235"
    assert checks.flag_text(None) == "—" and checks.flag_text(True) == "Yes"
    assert checks.flag_text(False) == "No"
    assert (
        checks.set_text("heldout") == "Held-out" and checks.set_text("other") == "other"
    )


def test_documents_are_not_changed_by_reading():
    document = score_document()
    before = copy.deepcopy(document)
    normalise_check_scores(document)
    assert document == before
