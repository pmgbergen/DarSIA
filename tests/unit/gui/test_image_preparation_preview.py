"""Real preview widgets with finite source producers, no scientific writes."""
import math
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from darsia.fluidflower.image_normalisation_step import ImageNormalisationChoice
from darsia.fluidflower.image_preparation_preview import (ImagePreparationPreview, PreparationSource,
                                                           percent_correction)
from darsia.fluidflower import image_normalisation as method


def choice(qtbot, tmp_path):
    path = tmp_path / "series.toml"
    path.write_text('series_id="sample"\n[runs]\nids=["r1","r2"]\n', encoding="utf8")
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "sample")
    qtbot.waitUntil(panel.can_close)
    panel.method.setCurrentIndex(panel.method.findData(method.MATCH_REFERENCE))
    panel.reference.setCurrentIndex(panel.reference.findData("r1"))
    panel.stable_band.setValue(3)
    return panel, path


def pixels(run="r1"):
    mask = np.zeros((8, 12), bool)
    mask[:, :2] = True
    return dict(run=run, rgb=np.full((8, 12, 3), 50, np.uint8), stable_mask=mask,
                grid=dict(shape=[8, 12]), source_pins={"baseline": "test fixture"})


def test_preview_overlay_and_proposal_never_save_choice(qtbot, tmp_path):
    selected, path = choice(qtbot, tmp_path)
    original = path.read_bytes()
    calls = []
    def load(operation, root, series, **options):
        calls.append((operation, root, series, options))
        options["check_cancelled"]()
        options["on_progress"](dict(run="r1", position=0, total=2))
        if operation == "preview":
            return pixels(options["run"])
        return dict(reference_run="r2", ranking=[
            dict(run="r2", score=0.1, lighting_score=0.1, color_score=None),
            dict(run="r1", score=0.2, lighting_score=0.2, color_score=None)], source_pins={})
    preview = ImagePreparationPreview(selected, loader=load)
    qtbot.addWidget(preview)
    preview.show()
    preview.preview.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    assert preview._mask.sum() == 16
    assert preview.photo.pixmap_item.pixmap().size().width() == 12
    assert preview.mask_item.isVisible()
    preview.show_mask.setChecked(False)
    assert not preview.mask_item.isVisible()
    assert not preview.photo.pixmap_item.pixmap().isNull()
    preview.propose.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    assert preview.use_proposal.isEnabled()
    assert preview.ranking.rowCount() == 2
    assert selected.reference.currentData() == "r1"
    assert path.read_bytes() == original
    preview.use_proposal.click()
    assert selected.reference.currentData() == "r2"
    assert "until you save" in preview.status.text()
    assert path.read_bytes() == original
    assert calls[0][3]["run"] == "r1" and calls[0][3]["band_width_mm"] == 3
    assert calls[1][3]["use_lighting"] is True and calls[1][3]["use_color"] is False


def test_changing_band_discards_slow_preview_and_close_does_not_wait(qtbot, tmp_path):
    selected, _path = choice(qtbot, tmp_path)
    entered, release, retired = Event(), Event(), Event()
    def load(operation, root, series, **options):
        entered.set()
        release.wait(5)
        try:
            options["check_cancelled"]()
            return pixels()
        finally:
            retired.set()
    preview = ImagePreparationPreview(selected, loader=load)
    qtbot.addWidget(preview)
    preview.show()
    try:
        preview.preview.click()
        qtbot.waitUntil(entered.is_set)
        assert preview.reader.busy
        selected.stable_band.setValue(4)
        assert not preview.reader.busy and preview._mask is None
        assert preview.preview.isEnabled()
        preview.close()
        assert not preview.isVisible()
    finally:
        release.set()
        qtbot.waitUntil(retired.is_set)
    qtbot.wait(20)
    assert preview._mask is None


def test_reference_change_or_new_run_discards_old_proposal(qtbot, tmp_path):
    selected, _path = choice(qtbot, tmp_path)
    preview = ImagePreparationPreview(selected, loader=lambda *a, **k: dict(
        reference_run="r2", ranking=[dict(run="r2", score=0.1)]))
    qtbot.addWidget(preview)
    preview.show()
    preview.propose.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    assert preview.use_proposal.isEnabled()
    preview.run.setCurrentIndex(preview.run.findData("r2"))
    assert not preview.use_proposal.isEnabled()
    assert preview.ranking.isHidden()


def test_colour_fit_button_uses_saved_settings_and_shows_readonly_inline_failure(qtbot, tmp_path):
    selected, path = choice(qtbot, tmp_path)
    selected.color_method.setCurrentIndex(selected.color_method.findData(method.COLOR_UNIFORM24))
    calls = []
    def load(operation, root, series, **options):
        calls.append((operation, options))
        if operation == 'preview':
            return pixels(options['run'])
        assert operation == 'colour_fit'
        assert options['run'] == 'r2' and 'band_width_mm' not in options
        return dict(run='r2', reference_run='r1', failed_run='r2', accepted=False,
            complete=False, error='Neutral tone curve is outside physical bounds for channel 1',
            diagnostics=dict(channel=1, source_swatches_srgb=[[[.5,.99,.5]]]),
            note='Read-only rejected fit; no result was saved.')
    preview = ImagePreparationPreview(selected, loader=load)
    qtbot.addWidget(preview)
    preview.show()
    assert not preview.colour_fit.isEnabled()  # Unsaved choices cannot pretend to be the job's settings.
    selected.save_button.click()
    qtbot.waitUntil(selected.can_close)
    preview.run.setCurrentIndex(preview.run.findData('r2'))
    qtbot.waitUntil(lambda: not preview.reader.busy and preview.colour_fit.isEnabled())
    original = path.read_bytes()
    preview.colour_fit.click()
    qtbot.waitUntil(lambda: any(row[0] == 'colour_fit' for row in calls) and not preview.reader.busy)
    assert 'R2 → R1' in preview.colour_status.text()
    assert 'No correction was accepted' in preview.colour_status.text()
    assert preview.colour_details.isHidden() and preview.fit_details.isVisible()
    preview.fit_details.click()
    assert preview.colour_details.isVisible()
    assert 'source_swatches_srgb' in preview.colour_details.toPlainText()
    assert path.read_bytes() == original
    selected.stable_band.setValue(4)
    assert not preview.colour_fit.isEnabled() and preview.fit_details.isHidden()


def test_misaligned_mask_is_refused_without_displaying_it(qtbot, tmp_path):
    selected, _path = choice(qtbot, tmp_path)
    wrong = pixels()
    wrong["stable_mask"] = np.zeros((4, 6), bool)
    preview = ImagePreparationPreview(selected, loader=lambda *a, **k: wrong)
    qtbot.addWidget(preview)
    preview.show()
    preview.preview.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    assert "same declared image grid" in preview.status.text()
    assert preview._mask is None


def test_default_source_passes_only_explicit_jobs_to_coordinator(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import current_image_preparation as api
    source = PreparationSource(registry_root=tmp_path / "local-registry")
    (tmp_path / "series.toml").write_text('series_id="SAMPLE_X"\n[runs]\nids=["r1"]\n')
    jobs = [dict(root="explicit-job", job=dict(sha256="test"))]
    class Registry:
        def jobs(self, root, series):
            assert Path(root) == tmp_path and series == "SAMPLE_X"
            return jobs
    source.registry = Registry()
    calls = []
    monkeypatch.setattr(api, "get_preparation_preview", lambda *args, **kwargs: calls.append((args, kwargs)), raising=False)
    source("preview", tmp_path, "samplex", run="r1", band_width_mm=3)
    assert calls[0][0] == (tmp_path, "SAMPLE_X")
    assert calls[0][1] == dict(registrations=jobs, run="r1", band_width_mm=3,
                             context_cache=source._context_cache, operations_cache=source._operations_cache)
    monkeypatch.setattr(api, "get_reference_proposal", lambda *args, **kwargs: calls.append((args, kwargs)))
    source("proposal", tmp_path, "samplex", band_width_mm=3, use_lighting=True, use_color=False)
    assert calls[1][0] == (tmp_path, "SAMPLE_X")
    monkeypatch.setattr(api, "get_colour_fit_preview", lambda *args, **kwargs: calls.append((args, kwargs)))
    source("colour_fit", tmp_path, "samplex", run="r1")
    assert calls[2][0] == (tmp_path, "SAMPLE_X")
    assert calls[2][1] == dict(registrations=jobs, run="r1", operations_cache=source._operations_cache)
    with pytest.raises(ValueError, match="another series"):
        source("preview", tmp_path, "other", run="r1", band_width_mm=3)


def test_selected_baseline_loads_on_open_and_run_change_without_saving(qtbot, tmp_path):
    selected, path = choice(qtbot, tmp_path)
    original, calls, displayed = path.read_bytes(), [], []
    def load(operation, root, series, **options):
        assert operation == "preview"
        calls.append(options["run"])
        return pixels(options["run"])
    preview = ImagePreparationPreview(selected, loader=load)
    preview.preview_loaded.connect(displayed.append)
    qtbot.addWidget(preview)
    preview.show()
    qtbot.waitUntil(lambda: len(displayed) == 1)
    assert calls == ["r1"] and displayed[0]["run"] == "r1"
    assert preview.mask_item.isVisible()
    preview.run.setCurrentIndex(preview.run.findData("r2"))
    assert preview.photo.pixmap_item.pixmap().isNull()  # Never show R1 as R2.
    qtbot.waitUntil(lambda: len(displayed) == 2)
    assert calls == ["r1", "r2"] and displayed[1]["run"] == "r2"
    assert path.read_bytes() == original


def test_save_and_optical_choices_keep_loaded_baseline_and_zoom(qtbot, tmp_path):
    selected, path = choice(qtbot, tmp_path)
    calls = []
    def load(operation, root, series, **options):
        calls.append(options["run"])
        return pixels(options["run"])
    preview = ImagePreparationPreview(selected, loader=load)
    qtbot.addWidget(preview)
    preview.show()
    qtbot.waitUntil(lambda: preview._mask is not None)
    preview.photo.scale(2., 2.)
    zoom = preview.photo.transform().m11()
    selected.color_method.setCurrentIndex(selected.color_method.findData(method.COLOR_UNIFORM24))
    selected.reference.setCurrentIndex(selected.reference.findData("r2"))
    selected.save_button.click()
    qtbot.waitUntil(selected.can_close)
    qtbot.wait(150)
    # Saving refreshes source metadata without clearing/refitting the image;
    # the production loader's same-receipt pixel cache is covered separately.
    assert calls == ["r1", "r1"] and not preview.photo.pixmap_item.pixmap().isNull()
    assert preview.photo.transform().m11() == zoom
    selected.stable_band.setValue(4.)
    qtbot.waitUntil(lambda: len(calls) == 3 and not preview.reader.busy)
    assert preview.photo.transform().m11() == zoom
    assert method.saved_choice(__import__("tomllib").loads(path.read_text()),
        "sample", ("r1", "r2"))["reference_run"] == "r2"


def test_baseline_can_be_shown_without_stable_mask_and_wrong_run_is_refused(qtbot, tmp_path):
    selected, _ = choice(qtbot, tmp_path)
    selected.stable_band.setValue(0.)
    result = dict(pixels(), stable_mask=None, note="Choose a stable-band width.")
    preview = ImagePreparationPreview(selected, loader=lambda *a, **k: result)
    qtbot.addWidget(preview)
    preview.show()
    qtbot.waitUntil(lambda: not preview.photo.pixmap_item.pixmap().isNull())
    assert preview._mask is None and preview.fit.isEnabled()
    assert not preview.mask_item.isVisible()
    preview.run.setCurrentIndex(preview.run.findData("r2"))
    qtbot.waitUntil(lambda: "another run" in preview.status.text())
    assert preview.photo.pixmap_item.pixmap().isNull()


def test_original_optical_proposal_reaches_form_without_automatic_save(qtbot, tmp_path):
    from darsia.fluidflower.optical_reference import reference_samples, propose_reference
    selected, path = choice(qtbot, tmp_path)
    path.write_text('series_id="sample"\n[runs]\nids=["r1","r2","r3"]\n', encoding="utf8")
    selected.set_context(tmp_path, "sample")
    qtbot.waitUntil(selected.can_close)
    selected.method.setCurrentIndex(selected.method.findData(method.MATCH_REFERENCE))
    selected.reference.setCurrentIndex(selected.reference.findData("r1"))
    selected.stable_band.setValue(3)
    original = path.read_bytes()
    def load(operation, root, series, **options):
        mask = np.zeros((24, 24), bool)
        mask[:, :4] = True
        samples = [reference_samples(run, np.full((24, 24, 3), .3 * exposure), mask,
                                     grid_shape=(6, 6))
                   for run, exposure in zip(("r1", "r2", "r3"), (.6, 1., 1.6))]
        result = propose_reference(samples, use_lighting=options["use_lighting"],
                                   use_color=options["use_color"])
        for row in result["ranking"]:
            row["score"] = row["mean_log_correction"]
        return result
    preview = ImagePreparationPreview(selected, loader=load)
    qtbot.addWidget(preview)
    preview.show()
    preview.propose.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    assert preview._proposal["reference_run"] == "r2"
    assert preview._proposal["selected"] is False
    assert preview.ranking.item(0, 0).text() == "1"
    assert preview.ranking.item(0, 1).text() == "R2 (proposed)"
    assert selected.reference.currentData() == "r1"
    preview.use_proposal.click()
    assert selected.reference.currentData() == "r2"
    assert path.read_bytes() == original


def test_a_log_correction_reads_as_the_percent_change_it_makes():
    assert percent_correction(math.log(1.39)) == "39 %"
    assert percent_correction(math.log(1.042)) == "4.2 %"
    assert percent_correction(0.0) == "0.0 %"
    for missing in (None, math.nan, True, "0.1"):
        assert percent_correction(missing) == "—"


def test_the_proposal_shows_rank_percent_and_the_saved_and_proposed_runs(qtbot, tmp_path):
    selected, _path = choice(qtbot, tmp_path)
    selected.method.setCurrentIndex(selected.method.findData(method.EQUALISE_SAND))
    selected._saved_form = selected._form()  # The saved reference is r1.
    def load(operation, root, series, **options):
        if operation == "preview":
            return pixels(options["run"])
        return dict(reference_run="r2", note="Smallest mean optical correction across baseline images.", ranking=[
            dict(run="r2", score=math.log(1.39), maximum_log_correction=math.log(1.5),
                 mean_light_correction=math.log(1.2), mean_colour_correction=math.log(1.6)),
            dict(run="r1", score=math.log(1.042), maximum_log_correction=None,
                 mean_light_correction=math.log(1.042))])
    preview = ImagePreparationPreview(selected, loader=load)
    qtbot.addWidget(preview)
    preview.show()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    preview.propose.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    table = preview.ranking
    assert [table.horizontalHeaderItem(column).text() for column in range(6)] == [
        "Rank", "Run", "Mean distance (%)", "Largest distance (%)", "Sand distance (%)", "ColorChecker distance (%)"]
    assert all(table.horizontalHeaderItem(column).toolTip() for column in range(6))
    assert [[table.item(row, column).text() for column in range(6)] for row in range(2)] == [
        ["1", "R2 (proposed)", "39 %", "50 %", "20 %", "60 %"],
        ["2", "R1 (saved)", "4.2 %", "—", "4.2 %", "—"]]
    assert preview.status.text().endswith(
        "Runs are compared with their own light divided out, outside the band along the frame.")
    selected._saved_form = selected._form()[:2] + ("r2",) + selected._form()[3:]
    preview.propose.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    assert table.item(0, 1).text() == "R2 (proposed, saved)" and table.item(1, 1).text() == "R1"
    selected.method.setCurrentIndex(selected.method.findData(method.MATCH_REFERENCE))
    qtbot.waitUntil(lambda: not preview.reader.busy)
    preview.propose.click()
    qtbot.waitUntil(lambda: not preview.reader.busy)
    assert "own light divided out" not in preview.status.text()
