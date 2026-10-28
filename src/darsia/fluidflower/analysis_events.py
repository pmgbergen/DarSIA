"""Events: when something happens in an area of the rig, for every run.

A question (``EventQuestion``) is built from menus as one or more conditions:

- *what*: all CO2, dissolved CO2 or free gas;
- *event*: is first detected, reaches a threshold, falls below a threshold it
  had reached, reaches its maximum, or is no longer detected;
- *where*: an area with a saved mass series: the whole rig, the two boxes of
  the mass analysis, or a named box or sand layer from "Mass from saved maps";
- a threshold where the event needs one, in grams or in percent of the mass
  injected up to that frame.

Conditions combine with "all" (and) or "any" (or). The answer for a run is the
first frame where the question holds, with its time in hours since injection
start. An optional noise filter sets the smallest mass that counts as detected
and asks the answer to hold for a number of frames in a row. Two answers can
be compared per run: did the first event come before the second?

Answers come from the per-frame masses that a finished mass analysis already
recorded (``RunSeries``, read by ``read_job_series``); no map is read, so
answering is light. A question is saved by name in the series folder ([paths]
event_questions; by default config/events) and can be asked again when runs
are added.
"""
from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

FORMAT = "darsia-event-question-v1"
QUESTIONS_KEY = "event_questions"
BOX_ANALYSIS_FORMAT = "darsia-saved-box-mass-analysis-v1"

WHAT = {"co2": "All CO2", "dissolved": "Dissolved CO2", "gas": "Free gas"}
# The mass component each choice reads: total, aqueous or gas.
COMPONENT = {"co2": "total", "dissolved": "aqueous", "gas": "gas"}
EVENTS = {
    "detected": "is first detected",
    "reaches": "reaches",
    "falls_below": "falls below",
    "maximum": "reaches its maximum",
    "gone": "is no longer detected",
}
THRESHOLD_EVENTS = frozenset({"reaches", "falls_below"})
UNITS = {"percent_injected": "% of injected", "g": "g"}
COMBINE = {"all": "all conditions hold (and)", "any": "any condition holds (or)"}
RELATIONS = {"before": "before", "after": "after"}
WHOLE_RIG = "full_rig"
# The areas every mass analysis integrates per frame, in its result records.
RESULT_AREAS = {"full_rig": "Whole rig", "box1": "Box 1", "box2": "Box 2"}
BOX_PREFIX = "box:"
LAYER_PREFIX = "layer:"


def area_label(area: str) -> str:
    """A plain name for an area key: "Whole rig", "Box 1", a named box or a sand layer."""
    if area in RESULT_AREAS:
        return RESULT_AREAS[area]
    for prefix in (BOX_PREFIX, LAYER_PREFIX):
        if area.startswith(prefix):
            return area[len(prefix):]
    return area


def _number(value) -> float | None:
    """A finite float, or None for a missing, boolean or non-finite value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class Condition:
    """One condition: what happens to which phase, where, and its threshold."""

    what: str
    event: str
    where: str = WHOLE_RIG
    threshold: float | None = None
    unit: str = "percent_injected"

    def check(self) -> "Condition":
        """This condition, or a ValueError that says what is wrong with it."""
        if self.what not in WHAT:
            raise ValueError(f"Unknown choice of what to look at: {self.what!r}")
        if self.event not in EVENTS:
            raise ValueError(f"Unknown event: {self.event!r}")
        if not isinstance(self.where, str) or not self.where:
            raise ValueError("Choose where the event is looked for")
        if self.event in THRESHOLD_EVENTS:
            if _number(self.threshold) is None or self.threshold < 0:
                raise ValueError(f"{EVENTS[self.event].capitalize()} needs a threshold of 0 or more")
            if self.unit not in UNITS:
                raise ValueError(f"Unknown threshold unit: {self.unit!r}")
        return self

    def text(self) -> str:
        """The condition as a sentence, for example "Free gas reaches 5 % of injected in Box 1"."""
        words = [WHAT.get(self.what, self.what), EVENTS.get(self.event, self.event)]
        if self.event in THRESHOLD_EVENTS:
            words.append(f"{self.threshold:g} {UNITS.get(self.unit, self.unit)}")
        return " ".join(words) + " in " + area_label(self.where)


@dataclass(frozen=True)
class EventQuestion:
    """A named question: its conditions, how they combine and the noise filter.

    ``noise_min`` (in ``noise_unit``) is the smallest mass that counts as
    detected; 0 counts any mass. ``hold_frames`` is the number of frames in a
    row the answer must hold; 1 takes the first frame where it holds.
    """

    name: str
    conditions: tuple[Condition, ...]
    combine: str = "all"
    noise_min: float = 0.0
    noise_unit: str = "g"
    hold_frames: int = 1

    def check(self) -> "EventQuestion":
        """This question, or a ValueError that says what is wrong with it."""
        if not str(self.name).strip():
            raise ValueError("Give the question a name")
        if not self.conditions:
            raise ValueError("Add at least one condition")
        for condition in self.conditions:
            condition.check()
        if self.combine not in COMBINE:
            raise ValueError(f"Unknown way to combine conditions: {self.combine!r}")
        if _number(self.noise_min) is None or self.noise_min < 0:
            raise ValueError("The smallest detected mass must be 0 or more")
        if self.noise_unit not in UNITS:
            raise ValueError(f"Unknown unit of the smallest detected mass: {self.noise_unit!r}")
        if isinstance(self.hold_frames, bool) or not isinstance(self.hold_frames, int) or self.hold_frames < 1:
            raise ValueError("Frames in a row must be a whole number of 1 or more")
        return self

    def text(self) -> str:
        """The question as one line of text."""
        joiner = " and " if self.combine == "all" else " or "
        return joiner.join(condition.text() for condition in self.conditions)

    def to_dict(self) -> dict:
        """The question as the saved recipe (``FORMAT``)."""
        return dict(format=FORMAT, name=str(self.name).strip(), combine=self.combine,
                    noise_min=float(self.noise_min), noise_unit=self.noise_unit,
                    hold_frames=int(self.hold_frames),
                    conditions=[dict(what=c.what, event=c.event, where=c.where,
                                     threshold=None if c.threshold is None else float(c.threshold),
                                     unit=c.unit) for c in self.conditions])

    @classmethod
    def from_dict(cls, value: Mapping) -> "EventQuestion":
        """A checked question from a saved recipe; ValueError when it is not one."""
        if not isinstance(value, Mapping) or value.get("format") != FORMAT:
            raise ValueError(f"Not a saved event question ({FORMAT})")
        rows = value.get("conditions")
        if not isinstance(rows, list):
            raise ValueError("A saved event question needs its list of conditions")
        try:
            conditions = tuple(Condition(what=row["what"], event=row["event"], where=row["where"],
                                         threshold=row.get("threshold"), unit=row.get("unit", "percent_injected"))
                               for row in rows)
            question = cls(name=value["name"], conditions=conditions, combine=value.get("combine", "all"),
                           noise_min=value.get("noise_min", 0.0), noise_unit=value.get("noise_unit", "g"),
                           hold_frames=value.get("hold_frames", 1))
        except (KeyError, TypeError) as error:
            raise ValueError(f"A saved event question lacks {error}") from None
        return question.check()


@dataclass(frozen=True)
class RunSeries:
    """One run's per-frame masses in each area, in time order.

    ``masses[area][component]`` holds grams per frame (None where unknown) for
    the components "total", "aqueous" and "gas". ``injected_g`` is the mass
    injected up to each frame, ``elapsed_h`` its time in hours since injection
    start.
    """

    run: str
    frames: tuple[str, ...]
    elapsed_h: tuple[float | None, ...]
    injected_g: tuple[float | None, ...]
    masses: Mapping[str, Mapping[str, tuple[float | None, ...]]]


@dataclass(frozen=True)
class EventAnswer:
    """When a question first held in one run: its frame and time, or why there is none."""

    run: str
    frame: str | None = None
    elapsed_h: float | None = None
    note: str = ""


def _and(values) -> bool | None:
    values = list(values)
    if any(value is False for value in values):
        return False
    return None if any(value is None for value in values) else True


def _or(values) -> bool | None:
    values = list(values)
    if any(value is True for value in values):
        return True
    return None if any(value is None for value in values) else False


def _not(value) -> bool | None:
    return None if value is None else not value


def _threshold_g(value, unit, injected) -> float | None:
    """A threshold in grams at one frame.

    A percentage of the injected mass is None (unknown) where that mass is
    missing, and cannot be reached (infinite) before anything was injected.
    """
    if unit == "g":
        return float(value)
    injected = _number(injected)
    if injected is None:
        return None
    return float(value) / 100.0 * injected if injected > 0 else math.inf


def _earlier(states) -> list[bool | None]:
    """For each frame, whether a state held at any earlier frame (three-valued)."""
    result, seen = [], False
    for state in states:
        result.append(seen)
        if state is True:
            seen = True
        elif state is None and seen is False:
            seen = None
    return result


def condition_states(condition: Condition, series: RunSeries, *, noise_min: float = 0.0,
                     noise_unit: str = "g") -> list[bool | None]:
    """Whether ``condition`` holds at each frame of ``series``: True, False or None (unknown).

    A frame is unknown when its mass, or the injected mass a percentage needs,
    is missing. "is no longer detected" holds from the frame after which the
    phase is never detected again, once it had been detected.
    """
    count = len(series.frames)
    amounts = [_number(value) for value in
               (series.masses.get(condition.where) or {}).get(COMPONENT[condition.what], (None,) * count)]
    if len(amounts) != count:
        raise ValueError(f"Run {series.run}: the masses of {area_label(condition.where)} do not match its frames")
    noise = [_threshold_g(noise_min, noise_unit, injected) for injected in series.injected_g]
    detected = [None if amount is None or limit is None else amount > limit
                for amount, limit in zip(amounts, noise)]
    if condition.event == "detected":
        return detected
    if condition.event in THRESHOLD_EVENTS:
        limits = [_threshold_g(condition.threshold, condition.unit, injected) for injected in series.injected_g]
        reached = [None if amount is None or limit is None else amount >= limit
                   for amount, limit in zip(amounts, limits)]
        if condition.event == "reaches":
            return reached
        return [_and((before, _not(now))) for before, now in zip(_earlier(reached), reached)]
    if condition.event == "maximum":
        known = [(amount, index) for index, amount in enumerate(amounts) if amount is not None]
        peak = max(known, key=lambda row: (row[0], -row[1]))[1] if known else None
        return [None if amount is None else index == peak for index, amount in enumerate(amounts)]
    # "gone": detected before, and not detected at this frame or any later one.
    later = [_or(detected[index:]) for index in range(count)]
    return [_and((before, _not(after))) for before, after in zip(_earlier(detected), later)]


def question_states(question: EventQuestion, series: RunSeries) -> list[bool | None]:
    """Whether the whole question holds at each frame, with its noise filter applied."""
    rows = [condition_states(condition, series, noise_min=question.noise_min, noise_unit=question.noise_unit)
            for condition in question.conditions]
    combine = _and if question.combine == "all" else _or
    combined = [combine(states) for states in zip(*rows)] if rows else []
    hold = int(question.hold_frames)
    if hold <= 1:
        return combined
    held = []
    for index in range(len(combined)):
        window = combined[index:index + hold]
        if False in window:
            held.append(False)
        else:
            held.append(True if len(window) == hold and None not in window else None)
    return held


def answer(question: EventQuestion, series: RunSeries) -> EventAnswer:
    """The first frame of one run where ``question`` holds, with its time."""
    question.check()
    missing = sorted({area_label(c.where) for c in question.conditions if c.where not in series.masses})
    if missing:
        return EventAnswer(series.run, note="No masses for " + ", ".join(missing))
    states = question_states(question, series)
    for index, state in enumerate(states):
        if state is True:
            unknown = sum(earlier is None for earlier in states[:index])
            note = f"{unknown} earlier frame(s) unknown" if unknown else ""
            return EventAnswer(series.run, series.frames[index], _number(series.elapsed_h[index]), note)
    if any(state is None for state in states):
        return EventAnswer(series.run, note="Not found; some frames are unknown")
    return EventAnswer(series.run, note="Never")


def answer_all(question: EventQuestion, series: Sequence[RunSeries]) -> list[EventAnswer]:
    """``answer`` for every run, in the order given."""
    question.check()
    return [answer(question, row) for row in series]


def compare(first: Sequence[EventAnswer], second: Sequence[EventAnswer],
            relation: str = "before") -> dict[str, bool | None]:
    """Per run: did the first event happen ``relation`` ("before" or "after") the second?

    None when either event has no time in that run. Equal times are neither
    before nor after.
    """
    if relation not in RELATIONS:
        raise ValueError(f"Unknown relation: {relation!r}")
    times = {row.run: row.elapsed_h for row in second}
    result = {}
    for row in first:
        other = times.get(row.run)
        if row.elapsed_h is None or other is None:
            result[row.run] = None
        else:
            result[row.run] = row.elapsed_h < other if relation == "before" else row.elapsed_h > other
    return result


def answers_csv(question: EventQuestion, answers: Sequence[EventAnswer], *,
                comparison: tuple[str, Mapping[str, bool | None]] | None = None) -> str:
    """The answers as CSV text: one row per run, with the question in the header.

    ``comparison`` adds a column: its title and, per run, the result of
    ``compare`` (written as yes, no or empty).
    """
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["# " + question.name + ": " + question.text()])
    writer.writerow(["run", "frame", "elapsed_h", "note"] + ([comparison[0]] if comparison else []))
    for row in answers:
        values = [row.run, row.frame or "", "" if row.elapsed_h is None else repr(row.elapsed_h), row.note]
        if comparison:
            result = comparison[1].get(row.run)
            values.append("" if result is None else ("yes" if result else "no"))
        writer.writerow(values)
    return text.getvalue()


def areas(series: Sequence[RunSeries]) -> dict[str, str]:
    """The areas with masses in any run, as key -> label: the whole rig first, then sand layers and boxes."""
    keys = []
    for row in series:
        keys.extend(key for key in row.masses if key not in keys)
    order = list(RESULT_AREAS)

    def rank(key):
        if key in order:
            return order.index(key), ""
        return len(order) + (0 if key.startswith(LAYER_PREFIX) else 1), area_label(key).casefold()

    keys.sort(key=rank)
    return {key: area_label(key) for key in keys}


# ----------------------------------------------------------- saved questions


def questions_folder(series_root) -> Path:
    """Where the series keeps its event questions: [paths] event_questions, else config/events."""
    from .series_paths import series_folder_path
    try:
        return series_folder_path(series_root, QUESTIONS_KEY)
    except KeyError:
        return series_folder_path(series_root, "config") / "events"


def _file_name(name: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "_", str(name).strip()).strip(" ._")
    if not stem:
        raise ValueError("Give the question a name with letters or digits")
    return stem + ".json"


def question_path(series_root, name: str) -> Path:
    """The file a question of this name is saved in."""
    return questions_folder(series_root) / _file_name(name)


def save_question(series_root, question: EventQuestion, *, replace: bool = False) -> Path:
    """Save ``question`` as a recipe in the series folder and return its file.

    An existing question of the same name is replaced only with ``replace``.
    The file is written whole (temporary file, then renamed).
    """
    question.check()
    path = question_path(series_root, question.name)
    if path.exists() and not replace:
        raise FileExistsError(f"A question named {question.name!r} is already saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(question.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_questions(series_root) -> tuple[list[EventQuestion], list[str]]:
    """The series' saved questions sorted by name, and a line per file that could not be read."""
    folder = questions_folder(series_root)
    questions, problems = [], []
    try:
        paths = sorted(folder.glob("*.json"))
    except OSError as error:
        return [], [f"{folder}: {error}"]
    for path in paths:
        try:
            questions.append(EventQuestion.from_dict(json.loads(path.read_text(encoding="utf-8-sig"))))
        except (OSError, ValueError) as error:
            problems.append(f"{path.name}: {error}")
    questions.sort(key=lambda question: question.name.casefold())
    return questions, problems


# ------------------------------------------------- masses of a finished job


def _box_tables(root: Path) -> tuple[dict, dict, str]:
    """The newest complete "Mass from saved maps" result of a job, and its folder name.

    Returns its box tables and its sand-layer tables (when it integrated
    layers), each as run -> CSV rows. Each table is checked against the
    SHA-256 its manifest recorded.
    """
    newest, created = None, ""
    for manifest in sorted((root / "analysis" / "boxes").glob("*/manifest.json")):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if value.get("format") == BOX_ANALYSIS_FORMAT and value.get("complete") is True:
            if str(value.get("created_utc") or "") >= created:
                newest, created = (manifest, value), str(value.get("created_utc") or "")
    if newest is None:
        return {}, {}, ""
    manifest, value = newest
    tables, layers = {}, {}
    for row in value.get("runs") or []:
        for key, found in (("table", tables), ("layer_table", layers)):
            pin = row.get(key)
            if not pin:
                continue
            path = manifest.parent / Path(str(pin.get("path") or "")).name
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != pin.get("sha256"):
                raise ValueError(f"{path} changed after its box analysis was recorded")
            found[str(row.get("run"))] = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    return tables, layers, manifest.parent.name


def _csv_number(text) -> float | None:
    try:
        return _number(float(text)) if str(text).strip() else None
    except ValueError:
        return None


def read_job_series(job, *, runs: Sequence[str] | None = None,
                    check_cancelled: Callable[[], None] | None = None,
                    on_progress: Callable[[int, int], None] | None = None) -> tuple[list[RunSeries], str]:
    """Per-frame masses of a finished mass analysis, from small records only.

    ``job`` is a finished job reader (``ProductionJob``). For every finished
    run, each frame's result record gives its time, the mass injected so far
    and the masses in the whole rig and the analysis's two boxes. A record
    without its areas gives the whole rig from its own totals, and one
    without the injected mass takes it from the run's mass table. The newest
    complete "Mass from saved maps" result adds the named boxes and sand
    layers it integrated, where their mass is known. No map is read. Returns
    the series and a line saying where the masses came from.
    """
    from .production_results_view import COMPLETE

    check = check_cancelled or (lambda: None)
    progress = on_progress or (lambda done, total: None)
    with getattr(job, "lock", None) or contextlib.nullcontext():
        finished = [row.run for row in job.runs if row.state == COMPLETE]
        selected = finished if runs is None else [run for run in runs if run in finished]
        try:
            boxes, layers, box_source, box_problem = *_box_tables(Path(job.root)), ""
        except (OSError, ValueError) as error:
            boxes, layers, box_source, box_problem = {}, {}, "", str(error)
        series = _read_series(job, selected, {BOX_PREFIX: boxes, LAYER_PREFIX: layers}, check, progress)
    source = "Masses from the mass analysis's frame records"
    if box_source:
        source += f" and from Mass from saved maps ({box_source})"
    if box_problem:
        source += f". Mass from saved maps was left out: {box_problem}"
    return series, source


def _read_series(job, selected, derived, check, progress) -> list[RunSeries]:
    """The series of the ``selected`` finished runs (see ``read_job_series``).

    ``derived`` maps an area prefix to the derived tables of that kind (run -> rows).
    """
    total = sum(len(job.frames(run)) for run in selected)
    done, series = 0, []
    for run in selected:
        frames = tuple(job.frames(run))
        hours, injected = [], []
        mass_rows = getattr(job, "mass_rows", None)
        table = {str(row.get("case_id")): row for row in (mass_rows(run) if mass_rows else [])}
        masses = {area: {component: [] for component in COMPONENT.values()} for area in RESULT_AREAS}
        for case in frames:
            check()
            result = job.result(run, case)
            hours.append(job.elapsed_h(run, case))
            value = result.get("injected_mass_g")
            injected.append(_number(value if value is not None else (table.get(case) or {}).get("injected_mass_g")))
            domains = result.get("domains") or {}
            for area in RESULT_AREAS:
                # The frame's own totals are the whole rig's masses.
                values = domains.get(area) or (result if area == WHOLE_RIG else {})
                for component in COMPONENT.values():
                    masses[area][component].append(_number(values.get(component + "_mass_g")))
            done += 1
            progress(done, total)
        index = {case: position for position, case in enumerate(frames)}
        for prefix, tables in derived.items():
            for row in tables.get(run, ()):
                position = index.get(row.get("case_id"))
                if position is None:
                    continue
                area = masses.setdefault(prefix + str(row.get("name")),
                                         {component: [None] * len(frames) for component in COMPONENT.values()})
                for component in COMPONENT.values():
                    area[component][position] = _csv_number(row.get(component + "_mass_g"))
        # An area without any known mass (a box the analysis did not use) is left out.
        kept = {area: {component: tuple(values) for component, values in parts.items()}
                for area, parts in masses.items()
                if area == WHOLE_RIG or any(value is not None for values in parts.values() for value in values)}
        series.append(RunSeries(run, frames, tuple(hours), tuple(injected), kept))
    return series
