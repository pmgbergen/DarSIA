"""Figures and series films use the existing finished-job and rendering routes."""
import json
from pathlib import Path

import pytest

from darsia.fluidflower import production_results_view as view
from .test_production_results_view import _settled, as_json, finished_run, production_job, sha, write


def test_figures_export_actual_frames_and_saved_mass_curves(tmp_path):
    data = production_job(tmp_path / "source", states=(("r1", "complete"),))
    job = view.ProductionJob(data.job)
    output = view.write_figures(job, "r1", "concentration", "products", tmp_path / "figures")
    receipt = json.loads((output / "figures.json").read_text())
    assert receipt["complete"] and receipt["mass_curves"] == "mass_curves.png"
    assert [row["case_id"] for row in receipt["frames"]] == job.frames("r1")
    assert all((output / row["path"]).is_file() for row in receipt["frames"])
    assert (output / "mass_curves.png").is_file()
    assert receipt["source_products_index"] == job._reader("r1").done["r1"]["products_index"]
    assert receipt["source_completion"] == job._reader("r1").done["r1"]["completed"]


def test_same_mm_uses_exact_saved_metric_layer_and_finer_export_refuses(tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    job = view.ProductionJob(data.job)
    rows = list(view.export_field_images(job, "r1", "aqueous_mass", "products", 1))
    assert [case for case, _ in rows] == job.frames("r1")
    assert all(image.unit == "g per cell" for _, image in rows)
    output = view.write_figures(job, "r1", "aqueous_mass", "products", tmp_path / "figures", resolution_mm=1)
    receipt = json.loads((output / "figures.json").read_text())
    assert receipt["source"] == "layer" and receipt["view_source"] == "products"
    with pytest.raises(ValueError):
        list(view.export_field_images(job, "r1", "aqueous_mass", "layer", .5))
    with pytest.raises(ValueError, match="no prescribed reduction"):
        list(view.export_field_images(job, "r1", "phase_labels", "layer", 2))


def test_missing_frame_never_becomes_a_successful_placeholder_film(tmp_path, monkeypatch):
    data = production_job(tmp_path / "source", states=(("r1", "complete"),))
    job = view.ProductionJob(data.job)
    monkeypatch.setattr(job, "field_image", lambda *args, **kwargs: (_ for _ in ()).throw(
        view.JobError("saved product missing")))
    with pytest.raises(view.JobError, match="saved product missing"):
        view.write_film(job, "r1", "concentration", "products", tmp_path / "films", fps=3)
    assert not (tmp_path / "films").exists()


def test_series_figures_export_only_explicit_finished_scope(qtbot, tmp_path):
    data = production_job(tmp_path / "source", states=(("r1", "complete"), ("r2", "complete"), ("r3", "queued")))
    page = view.ProductionResultsPage()
    qtbot.addWidget(page)
    assert page.open_job(data.job)
    _settled(page, qtbot)
    page.export_scope_combo.setCurrentIndex(page.export_scope_combo.findData("series"))
    _settled(page, qtbot)
    outputs = page.export_media("figures", tmp_path / "figures")
    assert len(outputs) == 2
    assert {json.loads((path / "figures.json").read_text())["run"] for path in outputs} == {"r1", "r2"}
    assert "2 finished runs" in page.status_label.text()


def _saved_resolutions(data, run, resolutions):
    """Change synthetic grid receipts and rebind their actual published bytes."""
    done_path = data.job / "done" / (run + ".json")
    done = json.loads(done_path.read_bytes())
    index_path = data.job / done["products_index"]["relative_path"]
    index = json.loads(index_path.read_bytes())
    for number, resolution in enumerate(resolutions, 1):
        suffix = f"/preparation/{run}_f{number}/storage_completed.json"
        product = next(row for row in index["products"] if row["product_id"].endswith(suffix))
        receipt = json.loads(Path(product["local_path"]).read_bytes())
        grid = receipt["coarse"]["grid"]
        if resolution is None:
            receipt["coarse"]["grid"] = {}
        else:
            grid.update(resolution_mm=resolution, extent_m=[0., .005 * resolution, 0., .004 * resolution])
        raw = as_json(receipt)
        write(Path(product["local_path"]), raw)
        write(index_path.parent / product["metadata_copy"]["relative_path"], raw)
        product.update(sha256=sha(raw), bytes=len(raw))
        product["metadata_copy"].update(sha256=sha(raw), bytes=len(raw))
    raw = as_json(index)
    write(index_path, raw)
    done["products_index"].update(sha256=sha(raw), bytes=len(raw))
    write(done_path, as_json(done))


def _availability(page):
    combo = page.export_resolution_combo
    return {combo.itemData(index): combo.model().item(index).isEnabled() for index in range(combo.count())}


def _page(qtbot, data, field="concentration"):
    page = view.ProductionResultsPage()
    qtbot.addWidget(page)
    assert page.open_job(data.job)
    _settled(page, qtbot)
    page.field_combo.setCurrentIndex(page.field_combo.findData(field))
    _settled(page, qtbot)
    return page


def test_resolution_choices_check_every_frame_and_selected_field(qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"),))
    _saved_resolutions(data, "r1", [1, 5, 2])
    page = _page(qtbot, data)
    assert _availability(page) == {None: True, 1: False, 2: False, 5: True, 10: True}
    assert "5 mm layer" in page.export_resolution_note.text()
    page.field_combo.setCurrentIndex(page.field_combo.findData("phase_labels"))
    _settled(page, qtbot)
    assert _availability(page) == {None: True, 1: False, 2: False, 5: False, 10: False}
    assert "exact stored resolution" in page.export_resolution_note.text()


def test_scope_and_run_changes_update_choices_and_reset_an_unavailable_selection(qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"), ("r2", "complete"), ("r3", "queued")))
    _saved_resolutions(data, "r2", [5, 5, 5])
    page = _page(qtbot, data)
    assert all(_availability(page).values())
    page.export_resolution_combo.setCurrentIndex(page.export_resolution_combo.findData(1))
    page.export_scope_combo.setCurrentIndex(page.export_scope_combo.findData("series"))
    _settled(page, qtbot)
    assert page.export_resolution_combo.currentData() is None
    assert _availability(page) == {None: True, 1: False, 2: False, 5: True, 10: True}
    assert "r2" in page.export_resolution_note.text() and "r3" not in page.export_resolution_note.text()
    page.export_scope_combo.setCurrentIndex(page.export_scope_combo.findData("selected"))
    _settled(page, qtbot)
    assert all(_availability(page).values())
    page.select_run("r2")
    _settled(page, qtbot)
    assert _availability(page) == {None: True, 1: False, 2: False, 5: True, 10: True}


@pytest.mark.parametrize("layers", [True, False])
def test_missing_metric_metadata_keeps_only_the_current_display_available(qtbot, tmp_path, layers):
    data = production_job(tmp_path, states=(("r1", "complete"),), layers=layers)
    if layers:
        _saved_resolutions(data, "r1", [1, None, 1])
    page = _page(qtbot, data)
    assert _availability(page) == {None: True, 1: False, 2: False, 5: False, 10: False}
    assert "unavailable" in page.export_resolution_note.text()
    assert page.figures_button.isEnabled() and page.export_button.isEnabled()
    index = page.export_resolution_combo.findData(1)
    assert page.export_resolution_combo.model().item(index).toolTip()


def test_metric_choices_follow_layer_field_but_current_display_follows_selected_source(qtbot, tmp_path):
    data = production_job(tmp_path, states=(("r1", "complete"), ("r2", "complete")))
    finished_run(data.job, data.producer, "r2", [f"r2_f{i}" for i in range(1, 4)],
                 names=("phase_labels",), layers=True, table_time=True)
    page = _page(qtbot, data)
    page.export_scope_combo.setCurrentIndex(page.export_scope_combo.findData("series"))
    _settled(page, qtbot)
    assert not _availability(page)[None] and _availability(page)[1]
    assert not page.figures_button.isEnabled() and page.export_button.isEnabled()
    assert "Frame products" in page.export_resolution_note.text()
    page.source_combo.setCurrentIndex(page.source_combo.findData("layer"))
    _settled(page, qtbot)
    assert _availability(page)[None] and page.figures_button.isEnabled()
    assert page.field_combo.currentData() == "concentration"


def test_invalid_scope_resolution_is_rejected_before_any_media_is_written(qtbot, tmp_path, monkeypatch):
    data = production_job(tmp_path / "source", states=(("r1", "complete"), ("r2", "complete")))
    _saved_resolutions(data, "r2", [5, 5, 5])
    page = _page(qtbot, data)
    page.export_scope_combo.setCurrentIndex(page.export_scope_combo.findData("series"))
    _settled(page, qtbot)
    page.export_resolution_combo.blockSignals(True)
    page.export_resolution_combo.setCurrentIndex(page.export_resolution_combo.findData(1))
    page.export_resolution_combo.blockSignals(False)
    monkeypatch.setattr(view, "write_figures", lambda *a, **k: pytest.fail("Unavailable scope must be rejected first"))
    output = tmp_path / "unavailable"
    assert page.export_media("figures", output) is None
    assert not output.exists() and "r2" in page.status_label.text()
