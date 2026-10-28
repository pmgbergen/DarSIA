"""Held-out check results (checks 1-3) read from a check-score file.

A check-score file is a JSON object. It is read by its structure only, never
by a format name or version string:

- ``verdict``: the file's overall result;
- ``check1``: ``status`` and ``per_time``. Each time holds ``runs``,
  ``median_abs_error`` (the median |M/d - 1|), ``group_median_ratio`` (the
  median M/d of each group), ``per_run_ratio`` (M/d of each run),
  ``time_pass`` and ``missing`` (the runs without a value at that time);
- ``check2`` and ``check3``: a ``primary`` and a ``fallback`` rule and the
  ``decision`` with its ``deciding_rule`` and ``status``. Each rule holds
  ``rule``, ``status``, ``assessable_times``, ``need_assessable`` and
  ``per_time``. Each time holds ``runs_with_frame``, ``runs_counted``,
  ``assessable`` and ``per_run`` (``delta``, ``M_over_d``, ``W_over_d``,
  ``counted`` and ``set`` of each run), plus ``median_delta`` and
  ``time_pass`` at an assessable time;
- optionally ``counts`` (the rows and runs of each set),
  ``integrity_problems``, and ``thresholds`` at the top, in a check or in a
  rule.

Statuses and pass marks are the file's own. Nothing is computed from the
numbers, a threshold is shown only where the file states one, and a file of
another structure is refused with a message that names the first mismatch.
The functions here read one explicitly chosen file and never write.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_SCORE_BYTES = 16 * 1024 * 1024
RULE_CHECKS = ("check2", "check3")
RULE_ROLES = ("primary", "fallback")
CHECK_TITLES = {"check1": "Check 1", "check2": "Check 2", "check3": "Check 3"}
ROLE_TITLES = {"primary": "Primary rule", "fallback": "Fallback rule"}
# Display names of the per-run sets; a set not named here is shown as written.
SET_TITLES = {"heldout": "Held-out", "fit_loro": "Fit run, LORO-predicted"}
LORO_SETS = frozenset({"fit_loro"})
COUNT_TITLES = {
    "heldout_rows": "Held-out rows",
    "loro_rows": "LORO rows",
    "heldout_runs": "Held-out runs",
    "loro_runs": "LORO-predicted fit runs",
    "heldout_parameters": "Held-out parameters",
}
NOT_IN_FILE = "Not in the file"
NO_VALUE = "—"
RULE_TREE_HEADERS = (
    "Rule, time or run",
    "Status",
    "Assessable",
    "Runs counted",
    "δ",
    "M/d",
    "W/d",
    "Set",
)
_DOCUMENT_KEYS = frozenset(
    {"verdict", "check1", *RULE_CHECKS, "counts", "integrity_problems", "thresholds"}
)


class CheckScoreError(ValueError):
    """The file is not a check-score file of the expected structure."""


@dataclass(frozen=True)
class Check1Time:
    """Check 1 at one time: M/d of the runs and the median of each group."""

    time: str
    runs: int
    median_abs_error: float
    group_ratios: tuple[tuple[str, float], ...]
    run_ratios: tuple[tuple[str, float], ...]
    time_pass: bool
    missing: tuple[str, ...]


@dataclass(frozen=True)
class Check1:
    """Check 1 as the file states it."""

    status: str
    times: tuple[Check1Time, ...]
    thresholds: tuple[str, ...]


@dataclass(frozen=True)
class RunDelta:
    """One run of a rule at one time."""

    run: str
    delta: float | None
    m_over_d: float
    w_over_d: float | None
    counted: bool
    run_set: str

    @property
    def loro(self) -> bool:
        """True for a fit run predicted by its leave-one-run-out fold."""
        return self.run_set in LORO_SETS


@dataclass(frozen=True)
class RuleTime:
    """One rule at one time; the median and pass mark only where assessable."""

    time: str
    runs_with_frame: int
    runs_counted: int
    assessable: bool
    median_delta: float | None
    time_pass: bool | None
    runs: tuple[RunDelta, ...]


@dataclass(frozen=True)
class Rule:
    """The primary or the fallback rule of check 2 or 3."""

    role: str
    name: str
    status: str
    assessable_times: int
    need_assessable: int
    times: tuple[RuleTime, ...]
    thresholds: tuple[str, ...]


@dataclass(frozen=True)
class RuleCheck:
    """Check 2 or 3: both rules and the file's decision between them."""

    key: str
    status: str
    deciding_rule: str
    deciding_role: str
    rules: tuple[Rule, ...]
    thresholds: tuple[str, ...]


@dataclass(frozen=True)
class CheckScores:
    """One check-score file, validated and normalised for display."""

    name: str
    path: str
    sha256: str
    verdict: str
    check1: Check1
    rule_checks: tuple[RuleCheck, ...]
    counts: tuple[tuple[str, str], ...] | None
    integrity_problems: tuple[str, ...] | None
    thresholds: tuple[str, ...]
    entries: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TreeRow:
    """One row of the rule tree; mark is "deciding", "loro" or ""."""

    texts: tuple[str, ...]
    children: tuple[TreeRow, ...] = ()
    mark: str = ""


# ---------------------------------------------------------------- validation


def _at(where: str, key: str) -> str:
    """The location of a key inside a location, for messages."""
    return f"{where} › {key}"


def _field(mapping: Mapping, key: str, where: str):
    """The value of a required key; refused when the key is absent."""
    if key not in mapping:
        raise CheckScoreError(f"{where} has no '{key}'.")
    return mapping[key]


def _mapping(value, where: str) -> Mapping:
    """A JSON object, else refused."""
    if not isinstance(value, Mapping):
        raise CheckScoreError(f"{where} is not a JSON object.")
    return value


def _text(value, where: str) -> str:
    """A JSON string, else refused."""
    if not isinstance(value, str):
        raise CheckScoreError(f"{where} is not text.")
    return value


def _whole(value, where: str) -> int:
    """A JSON whole number (not true or false), else refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckScoreError(f"{where} is not a whole number.")
    return value


def _number(value, where: str, *, optional: bool = False) -> float | None:
    """A JSON number as float, or None where optional and null; else refused."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckScoreError(f"{where} is not a number.")
    return float(value)


def _flag(value, where: str) -> bool:
    """A JSON true or false, else refused."""
    if not isinstance(value, bool):
        raise CheckScoreError(f"{where} is not true or false.")
    return value


def _texts(value, where: str) -> tuple[str, ...]:
    """A JSON list of strings, else refused."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CheckScoreError(f"{where} is not a list of text.")
    return tuple(value)


def _plain(value) -> str:
    """A JSON value as written: text unchanged, anything else as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _thresholds(mapping: Mapping) -> tuple[str, ...]:
    """The thresholds a mapping states, as 'name = value' lines; () if none."""
    if "thresholds" not in mapping:
        return ()
    value = mapping["thresholds"]
    if isinstance(value, Mapping):
        return tuple(f"{key} = {_plain(item)}" for key, item in value.items())
    if isinstance(value, list):
        return tuple(_plain(item) for item in value)
    return (_plain(value),)


def _check1(document: Mapping) -> Check1:
    """Check 1 of a document, validated."""
    check = _mapping(_field(document, "check1", "The file"), "check1")
    status = _text(_field(check, "status", "check1"), "check1 › status")
    where = "check1 › per_time"
    per_time = _mapping(_field(check, "per_time", "check1"), where)
    times = []
    for time, value in per_time.items():
        at = _at(where, time)
        entry = _mapping(value, at)
        groups = _mapping(
            _field(entry, "group_median_ratio", at), _at(at, "group_median_ratio")
        )
        runs = _mapping(_field(entry, "per_run_ratio", at), _at(at, "per_run_ratio"))
        times.append(
            Check1Time(
                time=time,
                runs=_whole(_field(entry, "runs", at), _at(at, "runs")),
                median_abs_error=_number(
                    _field(entry, "median_abs_error", at), _at(at, "median_abs_error")
                ),
                group_ratios=tuple(
                    (name, _number(ratio, _at(_at(at, "group_median_ratio"), name)))
                    for name, ratio in groups.items()
                ),
                run_ratios=tuple(
                    (run, _number(ratio, _at(_at(at, "per_run_ratio"), run)))
                    for run, ratio in runs.items()
                ),
                time_pass=_flag(_field(entry, "time_pass", at), _at(at, "time_pass")),
                missing=_texts(_field(entry, "missing", at), _at(at, "missing")),
            )
        )
    return Check1(status, tuple(times), _thresholds(check))


def _run(run: str, value, where: str) -> RunDelta:
    """One per-run entry of a rule at one time, validated."""
    entry = _mapping(value, where)
    return RunDelta(
        run=run,
        delta=_number(
            _field(entry, "delta", where), _at(where, "delta"), optional=True
        ),
        m_over_d=_number(_field(entry, "M_over_d", where), _at(where, "M_over_d")),
        w_over_d=_number(
            _field(entry, "W_over_d", where), _at(where, "W_over_d"), optional=True
        ),
        counted=_flag(_field(entry, "counted", where), _at(where, "counted")),
        run_set=_text(_field(entry, "set", where), _at(where, "set")),
    )


def _rule_time(time: str, value, where: str) -> RuleTime:
    """One time of a rule, validated; median and pass mark required where assessable."""
    entry = _mapping(value, where)
    assessable = _flag(_field(entry, "assessable", where), _at(where, "assessable"))
    if assessable:
        median = _number(
            _field(entry, "median_delta", where), _at(where, "median_delta")
        )
        passed = _flag(_field(entry, "time_pass", where), _at(where, "time_pass"))
    else:
        # A time that is not assessable may leave out its median and pass mark.
        median = _number(
            entry.get("median_delta"), _at(where, "median_delta"), optional=True
        )
        passed = entry.get("time_pass")
        if passed is not None:
            passed = _flag(passed, _at(where, "time_pass"))
    per_run = _mapping(_field(entry, "per_run", where), _at(where, "per_run"))
    return RuleTime(
        time=time,
        runs_with_frame=_whole(
            _field(entry, "runs_with_frame", where), _at(where, "runs_with_frame")
        ),
        runs_counted=_whole(
            _field(entry, "runs_counted", where), _at(where, "runs_counted")
        ),
        assessable=assessable,
        median_delta=median,
        time_pass=passed,
        runs=tuple(
            _run(run, item, _at(_at(where, "per_run"), run))
            for run, item in per_run.items()
        ),
    )


def _rule(role: str, value, where: str) -> Rule:
    """The primary or the fallback rule of a check, validated."""
    rule = _mapping(value, where)
    per_time = _mapping(_field(rule, "per_time", where), _at(where, "per_time"))
    return Rule(
        role=role,
        name=_text(_field(rule, "rule", where), _at(where, "rule")),
        status=_text(_field(rule, "status", where), _at(where, "status")),
        assessable_times=_whole(
            _field(rule, "assessable_times", where), _at(where, "assessable_times")
        ),
        need_assessable=_whole(
            _field(rule, "need_assessable", where), _at(where, "need_assessable")
        ),
        times=tuple(
            _rule_time(time, item, _at(_at(where, "per_time"), time))
            for time, item in per_time.items()
        ),
        thresholds=_thresholds(rule),
    )


def _rule_check(document: Mapping, key: str) -> RuleCheck:
    """Check 2 or 3 of a document; its decision must name exactly one of its rules."""
    check = _mapping(_field(document, key, "The file"), key)
    rules = tuple(
        _rule(role, _field(check, role, key), _at(key, role)) for role in RULE_ROLES
    )
    where = _at(key, "decision")
    decision = _mapping(_field(check, "decision", key), where)
    deciding = _text(
        _field(decision, "deciding_rule", where), _at(where, "deciding_rule")
    )
    status = _text(_field(decision, "status", where), _at(where, "status"))
    roles = [rule.role for rule in rules if rule.name == deciding]
    if not roles:
        raise CheckScoreError(
            f"{_at(where, 'deciding_rule')} names '{deciding}', which is neither "
            "the primary nor the fallback rule."
        )
    if len(roles) > 1:
        raise CheckScoreError(
            f"{key}: the primary and the fallback rule are both named '{deciding}', "
            "so the deciding rule is ambiguous."
        )
    return RuleCheck(key, status, deciding, roles[0], rules, _thresholds(check))


def _key_title(key: str) -> str:
    """A key as a title: some_key -> Some key."""
    words = key.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _counts(document: Mapping) -> tuple[tuple[str, str], ...] | None:
    """The optional counts as (title, text) rows; None when the file has none."""
    if "counts" not in document:
        return None
    counts = _mapping(document["counts"], "counts")
    rows = []
    for key, value in counts.items():
        if isinstance(value, list):
            text = (
                f"{len(value)}: " + ", ".join(_plain(item) for item in value)
                if value
                else "0"
            )
        else:
            text = _plain(value)
        rows.append((COUNT_TITLES.get(key, _key_title(key)), text))
    return tuple(rows)


def _integrity(document: Mapping) -> tuple[str, ...] | None:
    """The optional integrity problems as texts; None when the file has none."""
    if "integrity_problems" not in document:
        return None
    value = document["integrity_problems"]
    return tuple(
        _plain(item) for item in (value if isinstance(value, list) else [value])
    )


def normalise_check_scores(
    document, *, name: str = "", path: str = "", sha256: str = ""
) -> CheckScores:
    """Validate a parsed check-score document and return it as plain values.

    ``name``, ``path`` and ``sha256`` identify the file for display only.
    Raises CheckScoreError, naming the first mismatch, when the structure is
    not the one described in this module. Other top-level entries with a
    plain value (text, number, true/false or null) are kept as written;
    nested ones are not shown.
    """
    document = _mapping(document, "The file")
    verdict = _text(_field(document, "verdict", "The file"), "verdict")
    check1 = _check1(document)
    rule_checks = tuple(_rule_check(document, key) for key in RULE_CHECKS)
    entries = tuple(
        (key, _plain(value))
        for key, value in document.items()
        if key not in _DOCUMENT_KEYS
        and (value is None or isinstance(value, (str, int, float, bool)))
    )
    return CheckScores(
        name=name,
        path=path,
        sha256=sha256,
        verdict=verdict,
        check1=check1,
        rule_checks=rule_checks,
        counts=_counts(document),
        integrity_problems=_integrity(document),
        thresholds=_thresholds(document),
        entries=entries,
    )


def load_check_scores(path) -> CheckScores:
    """Read one check-score file and return it validated and normalised.

    The file is read once, up to MAX_SCORE_BYTES, and never written. Raises
    OSError when it cannot be read and CheckScoreError when it is not JSON or
    not of the check-score structure.
    """
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_SCORE_BYTES + 1)
    if len(raw) > MAX_SCORE_BYTES:
        raise CheckScoreError(
            f"The file is larger than the display limit of {MAX_SCORE_BYTES} bytes."
        )
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckScoreError(f"The file is not JSON ({error}).") from error
    except RecursionError as error:
        raise CheckScoreError("The file is nested too deeply to read.") from error
    return normalise_check_scores(
        document, name=path.name, path=str(path), sha256=hashlib.sha256(raw).hexdigest()
    )


# ------------------------------------------------------------------- display


def status_text(value: str) -> str:
    """A file status made readable: NOT_ASSESSABLE reads Not assessable."""
    words = value.replace("_", " ").strip().lower()
    return words[:1].upper() + words[1:] if words else NO_VALUE


def number_text(value: float | None) -> str:
    """A ratio or δ with three decimals; a value the file leaves empty is '—'."""
    return NO_VALUE if value is None else f"{value:.3f}"


def flag_text(value: bool | None) -> str:
    """Yes, No, or '—' where the file states nothing."""
    return NO_VALUE if value is None else ("Yes" if value else "No")


def set_text(value: str) -> str:
    """The display name of a per-run set, or the set as written."""
    return SET_TITLES.get(value, value)


def _pass_text(value: bool | None) -> str:
    """The file's pass mark of a time: Pass, Fail, or '—' where it states none."""
    return NO_VALUE if value is None else ("Pass" if value else "Fail")


def _ordered(values: Iterable[str]) -> list[str]:
    """Distinct values in order of first appearance."""
    return list(dict.fromkeys(values))


def _merged(sequences: Iterable[Sequence[str]]) -> list[str]:
    """Labels of several lists in one order; an extra label follows its neighbour."""
    merged: list[str] = []
    for sequence in sequences:
        previous = None
        for label in sequence:
            if label not in merged:
                merged.insert(
                    merged.index(previous) + 1 if previous is not None else 0, label
                )
            previous = label
    return merged


def file_titles(files: Sequence[CheckScores]) -> tuple[str, ...]:
    """A distinct title per file: its name, with its folder where names repeat."""
    names = [scores.name or "Unnamed file" for scores in files]
    if len(set(names)) == len(names):
        return tuple(names)
    names = [
        f"{Path(scores.path).parent.name}/{name}" if scores.path else name
        for scores, name in zip(files, names)
    ]
    if len(set(names)) == len(names):
        return tuple(names)
    return tuple(f"{name} ({index})" for index, name in enumerate(names, 1))


def check1_table(check: Check1) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Headers and rows of check 1, one column per time.

    The rows are the runs, the median |M/d − 1|, each group's median M/d, the
    file's time pass, the missing runs and then M/d of each run; a value the
    file does not give at a time reads '—'.
    """
    groups = _ordered(
        name for time in check.times for name, _ratio in time.group_ratios
    )
    runs = _ordered(run for time in check.times for run, _ratio in time.run_ratios)
    group_ratios = [dict(time.group_ratios) for time in check.times]
    run_ratios = [dict(time.run_ratios) for time in check.times]
    rows = [
        ("Runs", *(str(time.runs) for time in check.times)),
        (
            "Median |M/d − 1|",
            *(number_text(time.median_abs_error) for time in check.times),
        ),
        *(
            (
                f"{name} median M/d",
                *(number_text(ratios.get(name)) for ratios in group_ratios),
            )
            for name in groups
        ),
        ("Time pass", *(flag_text(time.time_pass) for time in check.times)),
        ("Missing runs", *(", ".join(time.missing) or "None" for time in check.times)),
        *(
            (f"{run} M/d", *(number_text(ratios.get(run)) for ratios in run_ratios))
            for run in runs
        ),
    ]
    return ("Item", *(f"Time {time.time}" for time in check.times)), tuple(rows)


def rule_tree(check: RuleCheck) -> tuple[TreeRow, ...]:
    """Rows of both rules of check 2 or 3, each with its times and per-run rows."""
    rows = []
    for rule in check.rules:
        deciding = rule.role == check.deciding_role
        times = []
        for time in rule.times:
            runs = tuple(
                TreeRow(
                    (
                        run.run,
                        "",
                        "",
                        "Counted" if run.counted else "Not counted",
                        number_text(run.delta),
                        number_text(run.m_over_d),
                        number_text(run.w_over_d),
                        set_text(run.run_set),
                    ),
                    mark="loro" if run.loro else "",
                )
                for run in time.runs
            )
            median = (
                NO_VALUE
                if time.median_delta is None
                else "median " + number_text(time.median_delta)
            )
            times.append(
                TreeRow(
                    (
                        f"Time {time.time}",
                        _pass_text(time.time_pass),
                        flag_text(time.assessable),
                        f"{time.runs_counted} of {time.runs_with_frame}",
                        median,
                        "",
                        "",
                        "",
                    ),
                    runs,
                )
            )
        title = f"{ROLE_TITLES[rule.role]}: {rule.name}"
        rows.append(
            TreeRow(
                (
                    title + (" (decides this check)" if deciding else ""),
                    status_text(rule.status),
                    f"{rule.assessable_times} of {len(rule.times)} times "
                    f"({rule.need_assessable} needed)",
                    "",
                    "",
                    "",
                    "",
                    "",
                ),
                tuple(times),
                "deciding" if deciding else "",
            )
        )
    return tuple(rows)


def _threshold_sentence(
    levels: Sequence[tuple[str, tuple[str, ...]]], fallback: str
) -> str:
    """The thresholds stated at each (level, lines), or fallback if none is."""
    stated = [
        (f"{level}: " if level else "") + "; ".join(lines)
        for level, lines in levels
        if lines
    ]
    if stated:
        return "Thresholds stated in the file: " + " · ".join(stated) + "."
    return fallback


def check1_note(check: Check1) -> str:
    """The line above check 1: its status in the file and any stated thresholds."""
    return f"Status in the file: {status_text(check.status)}. " + _threshold_sentence(
        [("", check.thresholds)],
        "The file states no thresholds: Time pass is the file's own mark, and no "
        "other pass mark is added.",
    )


def rule_check_note(check: RuleCheck) -> str:
    """The line above check 2 or 3: decision, deciding rule and any thresholds."""
    role = ROLE_TITLES[check.deciding_role].lower()
    text = (
        f"Decision in the file: {status_text(check.status)}, decided by the {role} "
        f"({check.deciding_rule}). "
    )
    text += _threshold_sentence(
        [("", check.thresholds)]
        + [(ROLE_TITLES[rule.role], rule.thresholds) for rule in check.rules],
        "The file states no thresholds: the statuses and Pass or Fail per time are the "
        "file's own marks, and no other pass mark is added.",
    )
    if any(
        run.loro for rule in check.rules for time in rule.times for run in time.runs
    ):
        text += (
            " Rows marked 'Fit run, LORO-predicted' are fit runs, each predicted by "
            "its leave-one-run-out (LORO) fold."
        )
    return text


def _summary(scores: CheckScores) -> dict[str, str]:
    """The summary items of one file, label -> text, in display order."""
    values = {"Verdict": status_text(scores.verdict)}
    problems = scores.integrity_problems
    values["Integrity problems"] = (
        NOT_IN_FILE
        if problems is None
        else (
            f"{len(problems)}: " + "; ".join(problems) if problems else "None reported"
        )
    )
    check = scores.check1
    values["Check 1"] = status_text(check.status)
    for time in check.times:
        values[f"Check 1 at {time.time}: median |M/d − 1|"] = number_text(
            time.median_abs_error
        )
        for name, ratio in time.group_ratios:
            values[f"Check 1 at {time.time}: {name} median M/d"] = number_text(ratio)
    for rule_check in scores.rule_checks:
        title = CHECK_TITLES[rule_check.key]
        role = ROLE_TITLES[rule_check.deciding_role].lower()
        values[title] = f"{status_text(rule_check.status)}, decided by the {role}"
        for rule in rule_check.rules:
            label = f"{title}, {ROLE_TITLES[rule.role].lower()}"
            values[label] = (
                f"{status_text(rule.status)}, {rule.assessable_times} of "
                f"{len(rule.times)} times assessable ({rule.need_assessable} needed)"
            )
            for time in rule.times:
                values[f"{label} at {time.time}: median δ"] = (
                    number_text(time.median_delta)
                    if time.median_delta is not None
                    else ("Not assessable" if not time.assessable else NO_VALUE)
                )
    for title, text in scores.counts or ():
        values[f"Counts: {title}"] = text
    levels = [("File", scores.thresholds), ("Check 1", check.thresholds)]
    for rule_check in scores.rule_checks:
        title = CHECK_TITLES[rule_check.key]
        levels.append((title, rule_check.thresholds))
        levels.extend(
            (f"{title}, {ROLE_TITLES[rule.role].lower()}", rule.thresholds)
            for rule in rule_check.rules
        )
    stated = [f"{level}: " + "; ".join(lines) for level, lines in levels if lines]
    values["Thresholds"] = " · ".join(stated) if stated else "None stated in the file"
    values["File SHA-256"] = scores.sha256 or NO_VALUE
    for key, text in scores.entries:
        values[f"File entry: {key}"] = text
    return values


def summary_rows(
    files: Sequence[CheckScores],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Summary rows, one text per file: verdict, statuses, medians and counts.

    With two files the rows put the same item side by side; an item that one
    file lacks reads 'Not in the file'.
    """
    columns = [_summary(scores) for scores in files]
    return tuple(
        (label, tuple(column.get(label, NOT_IN_FILE) for column in columns))
        for label in _merged(list(column) for column in columns)
    )
