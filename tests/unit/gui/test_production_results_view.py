"""Analysis > Production results: a finished production job, read-only, and its films.

Offscreen Qt on a synthetic job in the job layer's formats, written below
tmp_path: per frame products.npz and result.json, a stored coarse layer and
each run's mass table. Machine names are fictional. Nothing is computed; the
only writes are films into a chosen folder below tmp_path.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QScrollArea

from darsia.fluidflower import production_results_view as view
from tests.unit.published_dataset.synthetic import (
    PRODUCER,
    TOKEN,
    as_json,
    layer_arrays,
    make_series,
    npz_bytes,
    sha,
    write,
)

from .test_native_series_method import _phase_choice, _roots, method_window  # noqa: F401

SHAPE = (6, 8)
ALL_ARRAYS = ("phase_labels", "concentration_aqueous", "saturation_free_gas",
              "mass_total", "mass_aqueous", "mass_free_gas")
UNITS = {"concentration_aqueous": "g/L", "saturation_free_gas": "1", "mass_total": "g/L",
         "mass_aqueous": "g/L", "mass_free_gas": "g/L", "phase_labels": "category"}


def frame_arrays(index, names=ALL_ARRAYS):
    """A frame's products: dissolved CO2 grows to the right, one gas pixel, an excluded bottom row."""
    labels = np.zeros(SHAPE, np.uint8)
    labels[:, : index + 2] = 1
    labels[0, 0] = 2
    labels[-1, :] = 255
    concentration = np.where(labels == 1, 1.2 + 0.1 * index, 0.0).astype(np.float32)
    saturation = np.where(labels == 2, 0.5, 0.0).astype(np.float32)
    aqueous = (concentration * 0.5).astype(np.float32)
    gas = (saturation * 1.8).astype(np.float32)
    arrays = dict(phase_labels=labels, concentration_aqueous=concentration, saturation_free_gas=saturation,
                  mass_total=np.add(gas, aqueous), mass_aqueous=aqueous, mass_free_gas=gas)
    return {key: value for key, value in arrays.items() if key in names} | {
        "main_mass_eligible_mask": labels != 255}


def finished_run(job, producer, run, cases, *, names, layers, table_time):
    """A finished run: completion, done record, products index, products and shared copies."""
    attempt = producer / job.name / run / ".attempts" / TOKEN
    lease = job / "leases" / f"{run}.{TOKEN}" / "artifacts"
    products = []

    def product(product_id, data, kind, local, *, shared):
        write(local, data)
        row = {"product_id": product_id, "machine_id": PRODUCER, "local_path": str(local), "sha256": sha(data),
               "bytes": len(data), "kind": kind}
        if shared:
            write(lease / "metadata" / product_id, data)
            row["metadata_copy"] = {"relative_path": f"metadata/{product_id}", "sha256": sha(data),
                                    "bytes": len(data)}
        products.append(row)

    columns = ["case_id", "frame_index"] + (["elapsed_h"] if table_time else []) + [
        "total_mass_g", "aqueous_mass_g", "gas_mass_g", "injected_mass_g"]
    table = ",".join(columns) + "\n"
    for index, case in enumerate(cases):
        arrays = frame_arrays(index, names)
        archive = npz_bytes(arrays)
        folder = attempt / "frames" / case
        masses = {"total_mass_g": 0.01 * (index + 1), "aqueous_mass_g": 0.007 * (index + 1),
                  "gas_mass_g": 0.003 * (index + 1)}
        result = {"complete": True, "run": run, "case_id": case, "frame_index": index + 1,
                  "elapsed_h": 0.5 * (index + 1), "precision": "float32", **masses,
                  "products": {"path": "products.npz", "sha256": sha(archive), "size_bytes": len(archive)},
                  "units": {key: UNITS[key] for key in arrays if key in UNITS}}
        product(f"frames/{case}/products.npz", archive, "map", folder / "products.npz", shared=False)
        product(f"frames/{case}/result.json", as_json(result), "frame_scalar", folder / "result.json",
                shared=True)
        if layers:
            stage = attempt / "preparation" / case
            receipt = {"format": "darsia-storage-frame-v1", "case_id": case,
                       "coarse": {"grid": {"format": "darsia-metric-storage-grid-v1", "shape": [4, 5],
                                           "extent_m": [0.0, 0.005, 0.0, 0.004], "resolution_mm": 1.0,
                                           "axes": "row=y increasing, column=x increasing"}}}
            product(f"attempts/{TOKEN}/preparation/{case}/storage_stored_frame.npz",
                    npz_bytes(layer_arrays(0, index)), "intermediate", stage / "storage_stored_frame.npz",
                    shared=False)
            product(f"attempts/{TOKEN}/preparation/{case}/storage_completed.json", as_json(receipt),
                    "operational_receipt", stage / "storage_completed.json", shared=True)
        values = [case, str(index + 1)] + ([repr(0.5 * (index + 1))] if table_time else []) + [
            repr(masses["total_mass_g"]), repr(masses["aqueous_mass_g"]), repr(masses["gas_mass_g"]),
            repr(0.012 * (index + 1))]
        table += ",".join(values) + "\n"
    product("mass_table.csv", table.encode("utf-8"), "scalar_table", attempt / "mass_table.csv", shared=True)
    index = as_json({"format": "darsia-products-index-v1", "run": run, "attempt_id": TOKEN,
                     "publishing_machine_id": PRODUCER, "products": products})
    write(lease / "products_index.json", index)
    completed = as_json({"complete": True, "run": run, "pipeline": "native-phase-n1", "precision": "float32",
                         "case_count": len(cases), "ordered_case_ids": cases})
    write(job / "results" / run / "completed.json", completed)
    write(job / "done" / f"{run}.json", as_json({
        "format": "darsia-alias-done-v1", "run": run, "attempt_id": TOKEN,
        "completed": {"relative_path": f"results/{run}/completed.json", "sha256": sha(completed),
                      "bytes": len(completed)},
        "products_index": {"relative_path": f"leases/{run}.{TOKEN}/artifacts/products_index.json",
                           "sha256": sha(index), "bytes": len(index)}}))
    return attempt


def production_job(root, *, states=(("r1", "complete"), ("r2", "failed"), ("r3", "queued")), frames=3,
                   names=ALL_ARRAYS, layers=True, table_time=True):
    """A production job below ``root`` whose runs are finished, failed, queued or running."""
    root = Path(root)
    job, producer = root / "jobs" / "job-1", root / "producer"
    write(job / "job.json", as_json({"format": "darsia-alias-job-v1", "commit": "c" * 40, "precision": "float32",
                                     "expected_runs": [run for run, _state in states], "series_id": "SYN",
                                     "storage_profile": "full"}))
    attempts = {}
    for run, state in states:
        cases = [f"{run}_f{index + 1}" for index in range(frames)]
        write(job / "results" / run / "task.json", as_json({
            "format": "darsia-alias-task-v1", "run": run, "pipeline": "native-phase-n1",
            "ordered_case_ids": cases}))
        if state == "complete":
            attempts[run] = finished_run(job, producer, run, cases, names=names, layers=layers,
                                         table_time=table_time)
        elif state == "failed":
            write(job / "failed" / f"{run}.json", as_json({"run": run, "complete": False,
                                                            "traceback": "ValueError: example failure\n"}))
        elif state == "queued":
            write(job / "queue" / f"{run}.json", as_json({"run": run}))
        elif state == "running":
            write(job / "claimed" / "example-worker" / f"{run}.json", as_json({"run": run}))
    return SimpleNamespace(root=root, job=job, producer=producer, attempts=attempts)


def snapshot(root):
    """Every file below ``root``: its size, modification time and SHA-256."""
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns,
                                          hashlib.sha256(path.read_bytes()).hexdigest())
            for path in Path(root).rglob("*") if path.is_file()}


@pytest.fixture
def page(qtbot):
    widget = view.ProductionResultsPage()
    qtbot.addWidget(widget)
    widget.resize(1200, 900)
    widget.show()
    return widget


def _settled(page, qtbot):
    """Wait until what the page reads in the background is shown."""
    qtbot.waitUntil(lambda: not page.loading, timeout=10000)


def _choose(page, qtbot, field, source=None):
    if source is not None:
        page.source_combo.setCurrentIndex(page.source_combo.findData(source))
        _settled(page, qtbot)
    page.field_combo.setCurrentIndex(page.field_combo.findData(field))
    _settled(page, qtbot)


# ------------------------------------------------------------------ helpers


def test_display_stride_keeps_the_longest_side_within_the_limit():
    assert view.display_stride((6, 8)) == 1
    assert view.display_stride((3572, 6304)) == 5 and 6304 / 5 <= view.MAX_DISPLAY_PIXELS
    assert view.display_stride((10, 10), limit=4) == 3


def test_nice_limit_and_fit_limit_round_up_to_two_digits():
    assert view.nice_limit(1.5422) == 1.6 and view.nice_limit(0.88) == 0.88
    assert view.nice_limit(6.61e-6) == 6.7e-6 and view.nice_limit(2.0) == 2.0
    assert view.nice_limit(0) == view.nice_limit(float("nan")) == view.nice_limit("x") == 1.0
    image = view.FieldImage(np.array([[0.2, 5.0], [np.nan, 1.31]]), np.array([[False, True], [False, False]]),
                            False, "g/L", "upper", None, 1, (2, 2))
    assert view.fit_limit(image) == 1.4
    image.mask[:] = True
    assert view.fit_limit(image) == 1.0


def test_folder_helpers_and_titles(tmp_path):
    (tmp_path / "job" / "inner").mkdir(parents=True)
    assert view.inside_folder(tmp_path / "job" / "inner", tmp_path / "job")
    assert view.inside_folder(tmp_path / "job", tmp_path / "job")
    assert not view.inside_folder(tmp_path, tmp_path / "job")
    assert view.fresh_folder(tmp_path, "job") == tmp_path / "job-2"
    assert view.fresh_folder(tmp_path, "film") == tmp_path / "film"
    assert view.format_hours(12.5) == "12.50 h" and view.format_hours(None) == "time not recorded"
    assert view.frame_title("ac52", "gas_mass", 0.125) == "ac52 · Free-gas mass · 0.12 h"
    assert view.frame_footer(1, 3, "r1_f1", None) == "Frame 1 of 3 (r1_f1)"
    notes = []
    for stride in (1, 2, 3, 5, 11, 21):
        image = view.FieldImage(np.zeros((1, 1)), None, False, "", "upper", None, stride, (1, 1))
        notes.append(view.frame_footer(2, 3, "c", image).removeprefix("Frame 2 of 3 (c)"))
    assert notes == ["", " · every 2nd cell shown", " · every 3rd cell shown", " · every 5th cell shown",
                     " · every 11th cell shown", " · every 21st cell shown"]


def test_label_colours_follow_the_phase_legend():
    rgb, legend = view.label_colours(np.array([[0, 1], [2, 7]], np.uint8))
    assert tuple(rgb[1, 0]) == (242, 76, 157) and tuple(rgb[0, 1]) == (49, 217, 160)
    assert [name for name, _colour in legend] == ["Water", "Dissolved CO2", "Free gas", "Other label"]


def test_render_map_and_curves_give_fixed_size_pixels():
    image = view.FieldImage(np.ones((3, 4), np.float32), None, False, "g/L", "upper", None, 1, (3, 4))
    pixels = view.render_map(image, title="r1 · Concentration · 0.50 h", scale_max=2.0, label="Concentration")
    assert pixels.shape == (720, 1280, 4) and pixels.dtype == np.uint8
    labels = view.FieldImage(np.zeros((3, 4), np.uint8), None, True, "", "lower", (0, 1, 0, 1), 1, (3, 4))
    assert view.render_map(labels, title="t").shape == (720, 1280, 4)
    assert view.render_map(None, title="t", message="Not readable here").shape == (720, 1280, 4)
    rows = [{"case_id": "a", "elapsed_h": 0.5, "total_mass_g": 1.0}, {"case_id": "b", "elapsed_h": 1.0,
                                                                       "total_mass_g": 2.0}]
    assert view.render_mass_curves(rows, run="r1", marker=1.0).shape == (400, 1280, 4)
    assert view.render_mass_curves([], run="r1").shape == (400, 1280, 4)


def test_curve_series_uses_frame_numbers_without_elapsed_time():
    rows = [{"case_id": "a", "total_mass_g": 1.0, "gas_mass_g": "n/a"}, {"case_id": "b", "total_mass_g": 2.0}]
    axis, xs, series = view.curve_series(rows)
    assert (axis, xs, list(series)) == ("Frame", [1, 2], ["total_mass_g"])


# ---------------------------------------------------------------- reading


def test_job_lists_every_run_and_reads_only_finished_runs(tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"), ("r2", "failed"), ("r3", "queued"),
                                            ("r4", "running"), ("r5", "waiting")))
    job = view.ProductionJob(data.job)
    assert [(state.run, state.state, state.frames) for state in job.runs] == [
        ("r1", "Complete", 3), ("r2", "Failed", 3), ("r3", "Queued", 3), ("r4", "Claimed", 3),
        ("r5", "Not started", 3)]
    assert list(job.readers) == ["r1"] and job.readers["r1"].job["expected_runs"] == ["r1"]
    assert job.frames("r2") == [] and job.frames("r1") == ["r1_f1", "r1_f2", "r1_f3"]
    assert job.sources("r1") == ["products", "layer"]
    assert job.fields("r1", "products") == list(view.FIELDS)
    assert job.fields("r1", "layer") == list(view.FIELDS)
    assert "series SYN" in job.facts() and "5 runs, 1 complete" in job.facts()
    assert job.elapsed_h("r1", "r1_f2") == 1.0
    assert [row["total_mass_g"] for row in job.mass_rows("r1")] == [0.01, 0.02, 0.03]


def test_an_unreadable_finished_run_does_not_hide_the_others(tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"), ("r2", "complete")))
    completed = data.job / "results" / "r1" / "completed.json"
    completed.write_bytes(completed.read_bytes() + b" ")
    job = view.ProductionJob(data.job)
    first, second = job.runs
    assert first.state == "Unreadable" and "differs from its record" in first.note and first.frames == 3
    assert second.state == "Complete" and list(job.readers) == ["r2"]


def test_products_fields_match_the_stored_arrays_and_mask_excluded_pixels(tmp_path):
    data = production_job(tmp_path)
    job = view.ProductionJob(data.job)
    image = job.field_image("r1", "r1_f2", "concentration", "products")
    expected = frame_arrays(1)
    assert not image.categorical and image.unit == "g/L" and image.stride == 1 and image.origin == "upper"
    np.testing.assert_array_equal(image.values, expected["concentration_aqueous"])
    np.testing.assert_array_equal(image.mask, expected["phase_labels"] == 255)
    labels = job.field_image("r1", "r1_f2", "phase_labels", "products")
    assert labels.categorical
    np.testing.assert_array_equal(labels.values, expected["phase_labels"])
    assert job.field_image("r1", "r1_f2", "saturation", "products").unit == "fraction"
    assert job.field_image("r1", "r1_f2", "concentration", "products") is image  # Kept in memory.


def test_only_the_stored_fields_are_offered(tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),), layers=False,
                          names=("phase_labels", "concentration_aqueous", "saturation_free_gas"))
    job = view.ProductionJob(data.job)
    assert job.sources("r1") == ["products"]
    assert job.fields("r1", "products") == ["phase_labels", "concentration", "saturation"]


def test_large_grids_show_every_nth_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(view, "MAX_DISPLAY_PIXELS", 4)
    data = production_job(tmp_path, states=(("r1", "complete"),))
    job = view.ProductionJob(data.job)
    image = job.field_image("r1", "r1_f1", "aqueous_mass", "products")
    assert image.stride == 2 and image.shape == SHAPE
    np.testing.assert_array_equal(image.values, frame_arrays(0)["mass_aqueous"][::2, ::2])


def test_the_published_dataset_job_fixture_opens_with_its_stored_layers(tmp_path):
    series = make_series(tmp_path)
    job = view.ProductionJob(series.job)
    assert [state.state for state in job.runs] == ["Complete", "Complete"]
    assert job.sources("r1") == ["layer"]
    assert job.fields("r1", "layer") == list(view.FIELDS)
    image = job.field_image("r1", "r1_f2", "gas_mass", "layer")
    np.testing.assert_array_equal(image.values, layer_arrays(0, 1)["cell_mass_gas_g"])
    assert image.unit == "g per cell" and image.origin == "lower" and image.extent == (0.0, 0.005, 0.0, 0.004)
    phases = job.field_image("r1", "r1_f2", "phase_labels", "layer")
    arrays = layer_arrays(0, 1)
    water = arrays["phase_water_area_fraction"] >= arrays["phase_dissolved_area_fraction"]
    np.testing.assert_array_equal(phases.values, np.where(water, 0, 1))
    # The mass table of this fixture has no elapsed time; the result records do.
    assert job.elapsed_h("r1", "r1_f3") == 1.5


def test_products_only_on_the_producing_machine_are_named_not_read(tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    (data.attempts["r1"] / "frames" / "r1_f1" / "products.npz").unlink()
    (data.attempts["r1"] / "preparation" / "r1_f1" / "storage_stored_frame.npz").unlink()
    job = view.ProductionJob(data.job)
    for source in ("products", "layer"):
        with pytest.raises(view.JobError, match="only on the machine that produced it; open the job there"):
            job.field_image("r1", "r1_f1", "concentration", source)


def test_a_folder_without_a_production_job_is_refused(tmp_path):
    with pytest.raises(view.JobError, match="No job.json"):
        view.ProductionJob(tmp_path)
    write(tmp_path / "job.json", as_json({"format": "another-job", "expected_runs": ["r1"]}))
    with pytest.raises(view.JobError, match="darsia-alias-job-v1"):
        view.ProductionJob(tmp_path)


# ------------------------------------------------------------------- page


def test_page_lists_all_runs_and_opens_the_first_finished_one(page, qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r2", "failed"), ("r1", "complete"), ("r3", "queued")))
    assert page.open_job(data.job)
    assert page.status_label.text() == "Reading the job's finished runs…"
    _settled(page, qtbot)
    table = page.runs_table
    assert [[table.item(row, column).text() for column in range(3)] for row in range(table.rowCount())] == [
        ["r2", "Failed", "3"], ["r1", "Complete", "3"], ["r3", "Queued", "3"]]
    assert page.run == "r1" and page.shown == ("r1", "r1_f1", "phase_labels", "products")
    assert [page.field_combo.itemText(index) for index in range(page.field_combo.count())] == [
        "Phase labels", "Concentration", "Gas saturation", "Aqueous mass", "Free-gas mass"]
    assert page.source_combo.isVisible() and page.source_combo.itemText(1) == "Stored coarse layer (1 mm)"
    assert not page.map_view.source_pixmap().isNull() and not page.curves_view.source_pixmap().isNull()
    assert page.frame_label.text() == "Frame 1 of 3 · r1_f1 · 0.50 h"
    assert page.fps_spin.value() == 3.0 and page.export_button.isEnabled()
    table.selectRow(0)
    assert page.run == "r2" and not page.export_button.isEnabled()
    assert page.map_view.text().startswith("Run r2 is failed; its products open when it is complete.")


def test_the_page_names_an_unreadable_run_and_opens_the_next(page, qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"), ("r2", "complete")))
    completed = data.job / "results" / "r1" / "completed.json"
    completed.write_bytes(completed.read_bytes() + b" ")
    assert page.open_job(data.job)
    _settled(page, qtbot)
    assert page.run == "r2" and page.runs_table.item(0, 1).text() == "Unreadable"
    assert "differs from its record" in page.runs_table.item(0, 1).toolTip()
    page.runs_table.selectRow(0)
    assert page.map_view.text().startswith("Run r1 has a done record, but it could not be read:")


def test_the_page_says_where_products_are_when_they_are_elsewhere(page, qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),), layers=False)
    (data.attempts["r1"] / "frames" / "r1_f1" / "products.npz").unlink()
    assert page.open_job(data.job)
    _settled(page, qtbot)
    assert "only on the machine that produced it" in page.map_view.text()
    assert page.field_combo.count() == 0 and not page.export_button.isEnabled()
    assert page.export_film(tmp_path / "films") is None and not (tmp_path / "films").exists()


def test_the_slider_moves_through_the_run_with_a_fixed_scale(page, qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    page.open_job(data.job)
    _settled(page, qtbot)
    _choose(page, qtbot, "concentration")
    assert page.scale_edit.isVisible() and page.scale_edit.text() == "1.2"
    page.frame_slider.setValue(2)
    assert page.frame_label.text() == "Frame 3 of 3 · r1_f3 · 1.50 h"
    qtbot.waitUntil(lambda: page.shown == ("r1", "r1_f3", "concentration", "products"), timeout=5000)
    np.testing.assert_array_equal(page.image.values, frame_arrays(2)["concentration_aqueous"])
    assert page.scale_edit.text() == "1.2"  # The run's scale stays fixed while the slider moves.
    page.fit_scale()
    assert page.scale_edit.text() == "1.4"
    page.scale_edit.setText("2.5")
    page.scale_edit.editingFinished.emit()
    assert page._scales[("r1", "products", "concentration")] == 2.5
    _choose(page, qtbot, "saturation")
    assert page.scale_edit.text() == "1"  # Saturation starts at its physical bound.
    _choose(page, qtbot, "phase_labels")
    assert not page.scale_edit.isVisible() and page.image.categorical
    _choose(page, qtbot, "gas_mass", source="layer")
    assert page.shown == ("r1", "r1_f3", "gas_mass", "layer") and page.image.unit == "g per cell"


def test_mass_curves_come_from_the_run_mass_table(page, qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    page.open_job(data.job)
    _settled(page, qtbot)
    axis, xs, series = view.curve_series(page.job.mass_rows("r1"))
    assert axis == "Elapsed time (h)" and xs == [0.5, 1.0, 1.5]
    for key, step in (("total_mass_g", 0.01), ("aqueous_mass_g", 0.007), ("gas_mass_g", 0.003),
                      ("injected_mass_g", 0.012)):
        assert series[key] == [step * frame for frame in (1, 2, 3)]  # As written, read back exactly.


def test_opening_and_browsing_write_nothing(page, qtbot, tmp_path):
    data = production_job(tmp_path)
    before = snapshot(tmp_path)
    page.open_job(data.job)
    _settled(page, qtbot)
    for field in view.FIELDS:
        for source in ("products", "layer"):
            _choose(page, qtbot, field, source)
            for index in range(3):
                page.show_frame(index)
                _settled(page, qtbot)
    page.runs_table.selectRow(1)
    _settled(page, qtbot)
    assert snapshot(tmp_path) == before


def test_the_page_is_one_scroll_area_at_most(page):
    assert page.findChildren(QScrollArea) == [page.scroll]
    assert page.title.text() == "Results"
    assert page.open_button.text() == "Open finished job…" and page.export_button.text() == "Export film…"


def test_open_finished_job_asks_for_the_folder(page, qtbot, tmp_path, monkeypatch):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    asked = []
    monkeypatch.setattr(view.QFileDialog, "getExistingDirectory",
                        lambda *args: asked.append(args[1]) or str(data.job))
    assert page.open_button.isEnabled()
    page.open_button.click()
    _settled(page, qtbot)
    assert asked == ["Open finished job"] and page.run == "r1"
    monkeypatch.setattr(view.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    assert not page.choose_job()
    assert page.status_label.text().startswith("Could not open the job: No job.json")
    assert page.run == "r1"  # The open job stays.


# ------------------------------------------------------------------- film


def _decoded(path):
    video = cv2.VideoCapture(str(path))
    try:
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT)), video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()


def test_export_film_writes_an_mp4_at_the_chosen_fps_outside_the_job(page, qtbot, tmp_path, monkeypatch):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    films = tmp_path / "films"
    films.mkdir()
    page.open_job(data.job)
    _settled(page, qtbot)
    _choose(page, qtbot, "saturation")
    job_before, producer_before = snapshot(data.job), snapshot(data.producer)
    monkeypatch.setattr(view.QFileDialog, "getExistingDirectory", lambda *args: str(films))
    page.export_button.click()
    result = page.last_film
    assert result is not None and result.folder == films / "r1_saturation_film"
    assert result.movie == result.folder / "r1_saturation.mp4" and result.frames_folder is None
    assert _decoded(result.movie) == (3, 3.0)
    assert page.status_label.text().startswith("Film written:")
    record = json.loads((result.folder / "film.json").read_text(encoding="utf-8"))
    assert record["format"] == view.FILM_FORMAT and record["fps"] == 3.0
    assert record["colour_scale"] == [0.0, 1.0] and record["field_label"] == "Gas saturation"
    assert [row["case_id"] for row in record["frames"]] == ["r1_f1", "r1_f2", "r1_f3"]
    assert sorted(path.name for path in result.folder.iterdir()) == ["film.json", "r1_saturation.mp4"]
    assert snapshot(data.job) == job_before and snapshot(data.producer) == producer_before
    page.fps_spin.setValue(1.5)
    second = page.export_film(films)
    assert second.folder == films / "r1_saturation_film-2" and _decoded(second.movie) == (3, 1.5)


def test_export_film_refuses_the_job_and_product_folders(page, qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    page.open_job(data.job)
    _settled(page, qtbot)
    before = snapshot(tmp_path)
    for folder in (data.job, data.job / "results", data.attempts["r1"], data.attempts["r1"] / "frames"):
        assert page.export_film(folder) is None
        assert page.status_label.text().startswith("Choose a folder outside the job folder")
    assert snapshot(tmp_path) == before
    job = view.ProductionJob(data.job)
    with pytest.raises(ValueError, match="outside the job folder"):
        view.write_film(job, "r1", "concentration", "products", data.job / "results", fps=3)


def test_without_an_mp4_encoder_the_png_frames_are_written(page, qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower import post_materialization

    def no_encoder(*_args, **_kwargs):
        raise RuntimeError("No compatible MP4 video encoder is available")

    monkeypatch.setattr(post_materialization, "_write_movie", no_encoder)
    data = production_job(tmp_path, states=(("r1", "complete"),))
    page.open_job(data.job)
    _settled(page, qtbot)
    _choose(page, qtbot, "phase_labels")
    result = page.export_film(tmp_path / "films")
    assert result.movie is None and result.encoder is None
    assert sorted(path.name for path in result.frames_folder.iterdir()) == [
        "frame_000001.png", "frame_000002.png", "frame_000003.png"]
    assert cv2.imread(str(result.frames_folder / "frame_000001.png")).shape == (720, 1280, 3)
    assert "No MP4 encoder is available here" in page.status_label.text()
    record = json.loads((result.folder / "film.json").read_text(encoding="utf-8"))
    assert record["movie"] is None and record["png_frames"] == "frames" and record["colour_scale"] is None


def test_a_cancelled_film_writes_nothing(tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    job = view.ProductionJob(data.job)
    films = tmp_path / "films"
    films.mkdir()
    seen = []
    assert view.write_film(job, "r1", "concentration", "products", films, fps=3, scale_max=1.5,
                           progress=lambda done, total: seen.append((done, total)) or done < 2) is None
    assert seen == [(1, 3), (2, 3)] and list(films.iterdir()) == []
    with pytest.raises(ValueError, match="positive number of frames"):
        view.write_film(job, "r1", "concentration", "products", films, fps=0)


# ---------------------------------------------------------------- sidebar


def test_the_analysis_row_opens_the_page_in_the_phase_map_group(method_window, qtbot):  # noqa: F811
    window = method_window
    qtbot.waitUntil(lambda: not window._series_loading and not window._series_loader.busy(), timeout=10000)
    assert not window._series_load_error
    _phase_choice(_roots(window)["series-a"])
    window._refresh_method_state()
    section = window.sidebar._sections["analysis"]
    row = section.get_row("jobs:production_results")
    assert row is not None and row.label == "Results" and not row.isHidden()
    assert row in window._method_group_widgets["analysis"]["phase_model"]
    assert ("jobs", "production_results") not in window.workflow_status.rows
    window._select_row(("jobs", "production_results"))
    page = window.production_results_page
    assert isinstance(page, view.ProductionResultsPage)
    assert window.content_stack.currentWidget() is page
    assert window.sidebar.get_selected() == ("jobs", "production_results")
    assert window.current_row_scope.scope == "series" and not window.run_combo.isVisible()
    assert window.status_header.line.text().endswith(" · Whole series (0 runs)")
    assert not window.toolbar_builder.play_action.isEnabled()
    window._select_row(("start", "series"))
    qtbot.waitUntil(lambda: not window._series_loading and not window._series_loader.busy(), timeout=10000)
    assert not window._series_load_error
    window._select_row(("jobs", "production_results"))
    assert window.content_stack.currentWidget() is page
    assert window.ensure_production_results() is page  # Created once.
