"""Events: first frames where a question holds, from per-frame masses (synthetic runs only)."""
import hashlib
import json
import threading
from types import SimpleNamespace as NS

import pytest

from darsia.fluidflower import analysis_events as events
from darsia.fluidflower.analysis_events import Condition, EventQuestion, RunSeries

FRAMES = ("f0", "f1", "f2", "f3", "f4", "f5")
HOURS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
INJECTED = (0.0, 10.0, 20.0, 30.0, 40.0, 40.0)


def series(run="r1", *, aqueous=(0, 0, .2, 1., 3., 1.5), gas=(0, .5, .3, 0, 0, 0), injected=INJECTED,
           frames=FRAMES, hours=HOURS, area="box1"):
    """One run with dissolved and free-gas masses in ``area`` and the whole rig."""
    parts = dict(aqueous=tuple(aqueous), gas=tuple(gas),
                 total=tuple(None if a is None or g is None else a + g for a, g in zip(aqueous, gas)))
    return RunSeries(run, tuple(frames), tuple(hours), tuple(injected), {"full_rig": parts, area: parts})


def ask(*conditions, **options):
    return EventQuestion("Question", tuple(conditions), **options)


def first(question, run_series):
    found = events.answer(question, run_series)
    return found.frame, found.elapsed_h, found.note


def test_each_event_gives_its_first_frame():
    run = series()
    assert first(ask(Condition("dissolved", "detected", "box1")), run) == ("f2", 2.0, "")
    # 5 % of the injected mass is 0.5, 1.0, 1.5, 2.0 and 2.0 g; nothing can be reached before injection.
    reaches = Condition("dissolved", "reaches", "box1", 5.0, "percent_injected")
    assert first(ask(reaches), run) == ("f4", 4.0, "")
    falls = Condition("dissolved", "falls_below", "box1", 5.0, "percent_injected")
    assert first(ask(falls), run) == ("f5", 5.0, "")
    assert first(ask(Condition("dissolved", "reaches", "box1", 1.0, "g")), run) == ("f3", 3.0, "")
    assert first(ask(Condition("dissolved", "maximum", "box1")), run) == ("f4", 4.0, "")
    assert first(ask(Condition("gas", "gone", "box1")), run) == ("f3", 3.0, "")
    assert first(ask(Condition("co2", "detected", "full_rig")), run) == ("f1", 1.0, "")
    # The noise filter's smallest detected mass delays detection.
    assert first(ask(Condition("dissolved", "detected", "box1"), noise_min=.5), run) == ("f3", 3.0, "")
    assert first(ask(Condition("gas", "reaches", "box1", 9.0, "g")), run) == (None, None, "Never")


def test_conditions_combine_with_and_or_and_must_hold_frames_in_a_row():
    run = series(aqueous=(0, .1, 0, .1, .2, .3), gas=(0, 0, 0, 0, .4, .4))
    dissolved, gas = Condition("dissolved", "detected", "box1"), Condition("gas", "detected", "box1")
    assert first(ask(dissolved, gas, combine="all"), run)[0] == "f4"
    assert first(ask(dissolved, gas, combine="any"), run)[0] == "f1"
    # The flicker at f1 does not hold for two frames in a row; f3 does.
    assert first(ask(dissolved, hold_frames=2), run)[0] == "f3"
    # Near the end a hold cannot be confirmed.
    late = series(aqueous=(0, 0, 0, 0, 0, 1.))
    assert first(ask(dissolved, hold_frames=2), late) == (None, None, "Not found; some frames are unknown")


def test_unknown_frames_and_missing_areas_are_named():
    run = series(aqueous=(None, 0, 1., 1., 1., 1.))
    assert first(ask(Condition("dissolved", "detected", "box1")), run) == ("f2", 2.0, "1 earlier frame(s) unknown")
    unknown = series(aqueous=(None,) * 6)
    assert first(ask(Condition("dissolved", "detected", "box1")), unknown)[2] == "Not found; some frames are unknown"
    question = ask(Condition("dissolved", "detected", "box:Upper box"))
    assert first(question, run) == (None, None, "No masses for Upper box")
    # A percentage needs the injected mass of each frame.
    blind = series(injected=(None,) * 6)
    assert first(ask(Condition("dissolved", "reaches", "box1", 5.0)), blind)[0] is None


def test_answers_for_all_runs_compare_and_export():
    early, late = series("r1"), series("r2", aqueous=(0, 0, 0, 0, .2, .3), gas=(0,) * 6)
    question = ask(Condition("dissolved", "detected", "box1"))
    answers = events.answer_all(question, [early, late])
    assert [(row.run, row.frame) for row in answers] == [("r1", "f2"), ("r2", "f4")]
    gas = events.answer_all(ask(Condition("gas", "detected", "box1")), [early, late])
    assert events.compare(answers, gas, "before") == {"r1": False, "r2": None}
    assert events.compare(gas, answers, "before") == {"r1": True, "r2": None}
    assert events.compare(answers, gas, "after") == {"r1": True, "r2": None}
    text = events.answers_csv(question, answers)
    assert text.splitlines() == ["# Question: Dissolved CO2 is first detected in Box 1",
                                 "run,frame,elapsed_h,note", "r1,f2,2.0,", "r2,f4,4.0,"]
    compared = events.answers_csv(question, answers, comparison=("before Gas", {"r1": False, "r2": None}))
    assert compared.splitlines()[1:] == ["run,frame,elapsed_h,note,before Gas", "r1,f2,2.0,,no", "r2,f4,4.0,,"]
    assert events.areas([early, series(area="box:Upper")]) == {
        "full_rig": "Whole rig", "box1": "Box 1", "box:Upper": "Upper"}


def test_a_question_is_checked_and_reads_as_a_sentence():
    question = ask(Condition("gas", "reaches", "box2", 5, "percent_injected"),
                   Condition("dissolved", "detected"), combine="any")
    assert question.text() == "Free gas reaches 5 % of injected in Box 2 or Dissolved CO2 is first detected in Whole rig"
    assert EventQuestion.from_dict(question.to_dict()) == question.check()
    bad = [
        ask(),
        ask(Condition("oil", "detected")),
        ask(Condition("gas", "melts")),
        ask(Condition("gas", "reaches")),
        ask(Condition("gas", "detected"), combine="xor"),
        ask(Condition("gas", "detected"), hold_frames=0),
        ask(Condition("gas", "detected"), noise_min=-1),
        EventQuestion(" ", (Condition("gas", "detected"),)),
    ]
    for question in bad:
        with pytest.raises(ValueError):
            question.check()
    with pytest.raises(ValueError):
        EventQuestion.from_dict({"format": "something else"})


def test_questions_are_saved_in_the_series_folder(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    question = ask(Condition("dissolved", "detected", "box1"), noise_min=.1, hold_frames=2)
    path = events.save_question(root, question)
    assert path == root / "config" / "events" / "Question.json"
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        events.save_question(root, question)
    changed = EventQuestion("Question", question.conditions, combine="any")
    events.save_question(root, changed, replace=True)
    (path.parent / "broken.json").write_text("{", encoding="utf-8")
    loaded, problems = events.load_questions(root)
    assert loaded == [changed] and len(problems) == 1 and problems[0].startswith("broken.json")
    # [paths] event_questions moves the folder; a name becomes a safe file name.
    (root / "series.toml").write_text('[paths]\nevent_questions = "analysis/questions"\n', encoding="utf-8")
    moved = events.save_question(root, EventQuestion("Gas / layer 4?", question.conditions))
    assert moved == root / "analysis" / "questions" / "Gas _ layer 4.json"
    assert [row.name for row in events.load_questions(root)[0]] == ["Gas / layer 4?"]
    with pytest.raises(ValueError):
        events.question_path(root, "///")


class _Job:
    """A finished job reader with two runs of three frames each (no map is read)."""

    def __init__(self, root):
        self.root, self.lock = root, threading.RLock()
        self.runs = [NS(run="r1", state="Complete"), NS(run="r2", state="Complete"), NS(run="r3", state="Failed")]
        self.locked = []

    def frames(self, run):
        return ["c0", "c1", "c2"] if run != "r3" else []

    def elapsed_h(self, run, case):
        return float(case[1])

    def result(self, run, case):
        self.locked.append(self.lock._is_owned())
        step = int(case[1])
        masses = lambda scale: dict(total_mass_g=3. * scale, aqueous_mass_g=2. * scale, gas_mass_g=1. * scale)
        return dict(injected_mass_g=10. * step,
                    domains=dict(full_rig=masses(step), box1=masses(step / 2), box2=dict()))


def _box_analysis(root, rows, *, change=False, layers=None):
    folder = root / "analysis" / "boxes" / "20261028T000000Z-abcd1234"
    folder.mkdir(parents=True)
    text = "run,case_id,name,total_mass_g,aqueous_mass_g,gas_mass_g\n" + "".join(
        f"r1,{case},Upper,{total},{aqueous},{gas}\n" for case, total, aqueous, gas in rows)
    table = folder / "run-0001.csv"
    table.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    if change:
        table.write_text(text + "r1,c9,Upper,1,1,1\n", encoding="utf-8")
    run = dict(run="r1", table=dict(path=str(table), sha256=digest))
    if layers is not None:
        # The sand-layer table Mass from saved maps writes next to its box table.
        layer_table = folder / "layers-run-0001.csv"
        layer_table.write_text("run,case_id,name,total_mass_g,aqueous_mass_g,gas_mass_g\n" + "".join(
            f"r1,{case},{name},{total},{aqueous},{gas}\n" for case, name, total, aqueous, gas in layers),
            encoding="utf-8")
        run["layer_table"] = dict(path=str(layer_table), sha256=hashlib.sha256(layer_table.read_bytes()).hexdigest())
    manifest = dict(format=events.BOX_ANALYSIS_FORMAT, complete=True, created_utc="2026-10-28T00:00:00+00:00",
                    runs=[run])
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_a_finished_job_gives_each_run_its_frame_masses_and_saved_boxes(tmp_path):
    job = _Job(tmp_path)
    _box_analysis(tmp_path, [("c0", 0, 0, 0), ("c1", 1.5, 1.0, .5), ("c2", "", "", "")])
    progress = []
    found, source = events.read_job_series(job, on_progress=lambda done, total: progress.append((done, total)))
    assert [row.run for row in found] == ["r1", "r2"] and all(job.locked)
    first_run = found[0]
    assert first_run.frames == ("c0", "c1", "c2") and first_run.elapsed_h == (0.0, 1.0, 2.0)
    assert first_run.injected_g == (0.0, 10.0, 20.0)
    assert first_run.masses["full_rig"]["aqueous"] == (0.0, 2.0, 4.0)
    assert first_run.masses["box1"]["gas"] == (0.0, .5, 1.0)
    assert "box2" not in first_run.masses  # No known mass: the analysis did not use it.
    assert first_run.masses["box:Upper"]["aqueous"] == (0.0, 1.0, None)
    assert "box:Upper" not in found[1].masses
    assert progress[-1] == (6, 6)
    assert "Mass from saved maps (20261028T000000Z-abcd1234)" in source
    answer = events.answer(ask(Condition("dissolved", "detected", "box:Upper")), first_run)
    assert (answer.frame, answer.elapsed_h) == ("c1", 1.0)


def test_a_changed_box_table_is_left_out_and_said(tmp_path):
    _box_analysis(tmp_path, [("c0", 0, 0, 0)], change=True)
    found, source = events.read_job_series(_Job(tmp_path))
    assert all("box:Upper" not in row.masses for row in found)
    assert "Mass from saved maps was left out" in source and "changed" in source


def test_sand_layers_from_mass_from_saved_maps_are_areas(tmp_path):
    _box_analysis(tmp_path, [("c0", 0, 0, 0)], layers=[
        ("c0", "Upper sand", 0, 0, 0), ("c1", "Upper sand", .5, .5, 0), ("c2", "Upper sand", "", "", ""),
        ("c0", "Lower sand", 0, 0, 0)])
    found, _source = events.read_job_series(_Job(tmp_path))
    first = found[0]
    assert first.masses["layer:Upper sand"]["aqueous"] == (0.0, .5, None)
    assert "layer:Upper sand" not in found[1].masses
    assert events.areas([first]) == {"full_rig": "Whole rig", "box1": "Box 1", "layer:Lower sand": "Lower sand",
                                     "layer:Upper sand": "Upper sand", "box:Upper": "Upper"}
    answer = events.answer(ask(Condition("dissolved", "detected", "layer:Upper sand")), first)
    assert (answer.frame, answer.elapsed_h) == ("c1", 1.0)
