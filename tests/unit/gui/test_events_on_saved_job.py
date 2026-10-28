"""Events on a finished job's own records: the real job reader, Mass from saved maps and the page.

The job is the synthetic finished production job of the Results page tests,
with saved facies; no real series, map or NAS input is read.
"""
import threading

from darsia.fluidflower import analysis_box_reanalysis as boxes
from darsia.fluidflower import analysis_events as events
from darsia.fluidflower.analysis_events import Condition, EventQuestion
from darsia.fluidflower.events_page import EventsPage
from darsia.fluidflower.production_results_view import ProductionJob
from tests.unit.test_analysis_box_reanalysis import FACIES, saved_job

NAMES = {1: "Upper sand", 2: "Lower sand"}


def _job_with_layers(tmp_path, monkeypatch):
    fixture, config = saved_job(tmp_path, monkeypatch, facies=FACIES)
    boxes.reanalyze_saved_boxes(fixture.job, config, fixture.job / "analysis" / "boxes" / "20261028T010000Z-00000000",
                                actor="Tester", reason="Events", layers=True, layer_names=NAMES)
    return fixture.job


def test_a_finished_job_gives_events_its_frames_boxes_and_layers(tmp_path, monkeypatch):
    root = _job_with_layers(tmp_path, monkeypatch)
    series, source = events.read_job_series(ProductionJob(root))
    assert [row.run for row in series] == ["r1", "r2"]
    first = series[0]
    assert first.frames == ("r1_f1", "r1_f2") and first.elapsed_h == (0.5, 1.0)
    assert first.injected_g == (0.012, 0.024)  # From the run's mass table.
    assert first.masses["full_rig"]["aqueous"] == (0.007, 0.014)  # The frame's own totals.
    assert set(events.areas(series)) == {"full_rig", "layer:Upper sand", "layer:Lower sand", "box:Lower", "box:All"}
    assert "Mass from saved maps (20261028T010000Z-00000000)" in source
    answer = events.answer(EventQuestion("Upper", (Condition("dissolved", "detected", "layer:Upper sand"),)), first)
    assert (answer.frame, answer.elapsed_h, answer.note) == ("r1_f1", 0.5, "")


def test_the_events_page_reads_a_finished_job_off_the_gui_thread(qtbot, tmp_path, monkeypatch):
    root = _job_with_layers(tmp_path, monkeypatch)
    page = EventsPage(jobs_reader=lambda series_root, series_id: [dict(name="Mass analysis", root=str(root))])
    qtbot.addWidget(page)
    threads = []
    original = events.read_job_series

    def recorded(*args, **kwargs):
        threads.append(threading.current_thread() is threading.main_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(events, "read_job_series", recorded)
    page.set_series(tmp_path, "S")
    qtbot.waitUntil(lambda: bool(page.series), timeout=20000)
    assert threads == [False]
    assert page.status.text().startswith("2 finished runs, 4 frames.")
    row = page.rows[0]
    row.where.setCurrentIndex(row.where.findData("layer:Lower sand"))
    page.answer_button.click()
    assert page.table.rowCount() == 2 and page.sentence.text().startswith("Dissolved CO2 is first detected in Lower sand")
