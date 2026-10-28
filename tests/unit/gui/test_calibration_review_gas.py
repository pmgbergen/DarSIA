"""Saved local calibration → real contribution reader → embedded Review panels."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass.gas_contribution_review import capture_gas_contributions, save_gas_contributions, contribution_scales
from darsia.fluidflower.mass.n1_pipeline import _write_json
from darsia.fluidflower.calibration_review_gas import local_reviews, gas_figure, gas_summary_text
from darsia.fluidflower.storage_saved_rendering import release_stored_figure
from ..test_calibration_review_native import native_attachment, _review, tiny_config, phase1_data  # noqa: F401


@pytest.fixture
def frozen(tmp_path):
    rows = []
    for run in ("ac28", "ac47"):
        for index in (0, 1):
            shape = (3, 5)
            source = SimpleNamespace(population=SimpleNamespace(eligible=np.ones(shape, bool),
                pore_volume_m3=np.full(shape, 1e-6)),
                prepared={"frame": SimpleNamespace(dimensions_m=(1., 2.), frame_index=index, elapsed_h=float(index))},
                statistics_descriptor={}, probability_roles={}, phase_source={"run": run})
            before = np.ones(shape, np.float32)
            after = np.ones(shape, np.float32)*(index+1)
            before[:, 3:] = 0.
            after[:, 3:] = 3.  # An added-only review cell must visibly retain its own hue.
            after[0, 0], after[0, 1] = 0., 3.
            folder = tmp_path / run / str(index)
            folder.mkdir(parents=True)
            result = dict(report=dict(baseline_gas_mass_g=float(before.sum())*.001,
                gas_mass_g=float(after.sum())*.001), statistics={},
                gas_contribution_maps=capture_gas_contributions(source, before, after, max_axis=2))
            target = dict(kind="own_measurement" if run == "ac28" else "pressure_matched_estimate",
                          estimated=run != "ac28")
            record = save_gas_contributions(folder, source, result, target)
            rows.append(dict(run=run, case_id=f"image-{index}", frame_index=index, elapsed_h=float(index),
                gas_contribution=record, phase_split_target=target))
    preparation = _write_json(tmp_path / "preparation.json", dict(frames=rows,
        gas_contribution_review=contribution_scales(rows)))
    return _write_json(tmp_path / "frozen.json", dict(frozen=True,
        format="darsia-local-overflow-frozen-calibration-v1", overflow_preparation=preparation,
        calibration_runs=["ac28", "ac47"], parameters=dict(gain=1., beta=0.)))


def test_fixed_scales_native_edges_and_all_views_read_actual_saved_contributions(frozen):
    reader = local_reviews(frozen, dataset="series-a")[0]
    first, second = reader.gas(0), reader.gas(1)
    assert first["display_scale_g_per_cell"] == second["display_scale_g_per_cell"]
    assert first["summary"]["model_gas_mass_g"] == pytest.approx(.009)
    assert first["summary"]["final_gas_mass_g"] == pytest.approx(.028)
    for data in (first, second):
        assert data["maps"]["cell_model_before_g"].sum() == pytest.approx(data["summary"]["model_gas_mass_g"])
        assert data["maps"]["cell_final_gas_g"].sum() == pytest.approx(data["summary"]["final_gas_mass_g"])
        assert data["maps"]["cell_overflow_change_g"].sum() == pytest.approx(data["summary"]["overflow_change_g"])
    originals = {key: value.tobytes() for key, value in first["maps"].items()}
    for view in ("total", "model", "overflow", "origin"):
        figure = gas_figure(first, view)
        try:
            figure.canvas.draw()
            axis = figure.axes[0]
            coordinates = axis.collections[0].get_coordinates()
            np.testing.assert_allclose(coordinates[0, :, 0], [0., 1.2, 2.])
            np.testing.assert_allclose(coordinates[:, 0, 1], [1., 1/3, 0.])
            assert "g" in gas_summary_text(first)
            if view != "origin":
                same = gas_figure(second, view)
                try:
                    assert axis.collections[0].get_clim() == same.axes[0].collections[0].get_clim()
                finally:
                    release_stored_figure(same)
            else:
                from matplotlib.colors import to_rgb
                np.testing.assert_allclose(axis.collections[0].get_facecolors()[1, :3], to_rgb("#f2993f"))
                assert len(axis.patches) == 1  # Removal hatching only where removal was recorded.
                assert "mutually exclusive" in figure.texts[0].get_text()
        finally:
            release_stored_figure(figure)
    assert originals == {key: value.tobytes() for key, value in first["maps"].items()}
    estimated = local_reviews(frozen, dataset="series-a")[1].gas(0)
    assert "ESTIMATED" in gas_summary_text(estimated)


def test_old_scalar_report_never_fabricates_a_spatial_decomposition(tmp_path):
    preparation = _write_json(tmp_path / "old.json", dict(frames=[dict(run="old", case_id="one",
        report=dict(baseline_gas_mass_g=2., gas_mass_g=1.))]))
    bound = _write_json(tmp_path / "frozen.json", dict(frozen=True, overflow_preparation=preparation))
    data = local_reviews(bound, dataset="series")[0].gas(0)
    assert data["maps"] is None and "-1" in gas_summary_text(data)
    with pytest.raises(ValueError, match="no spatial"):
        gas_figure(data, "origin")


def test_local_panels_route_to_real_reader_and_keep_synchronized_choices(qtbot, frozen):
    from darsia.fluidflower.calibration_review_panels import CalibrationReviewPanels, _PanelRead
    local = local_reviews(frozen, dataset="series-a")[0]
    widget = CalibrationReviewPanels()
    qtbot.addWidget(widget)
    widget.set_local_result(local, 0)
    widget.panels[0].content.setCurrentIndex(widget.panels[0].content.findData("mass_gas"))
    widget.panels[0].gas_view.setCurrentIndex(widget.panels[0].gas_view.findData("origin"))
    widget.panels[1].gas_view.setCurrentIndex(widget.panels[1].gas_view.findData("overflow"))
    requests = [dict(content="mass_gas", gas_view=panel.gas_view.currentData(),
        gas_palette=panel.gas_palette.currentData()) for panel in widget.panels[:2]]
    worker = _PanelRead(1, local, local.frozen["path"], 0, requests, widget)
    received = []
    worker.ready.connect(lambda *args: received.append(args))
    worker.run()
    assert received and all(not error and not image.isNull() for image, error in received[0][1])
    assert "AC28" in worker.gas_notice and "Measured" in worker.gas_notice
    widget.set_local_result(local, 1)
    assert widget.panels[0].gas_view.currentData() == "origin"
    assert widget.panels[1].gas_view.currentData() == "overflow"
    assert not widget.panels[1].mass_controls.isVisibleTo(widget)
    widget.stop()


def test_existing_review_workspace_attaches_exact_local_frozen_and_exposes_its_runs(qtbot, monkeypatch, tmp_path, frozen):
    from .test_calibration_review_approval import _page, _series
    page, _ = _page(qtbot, monkeypatch, _series(tmp_path))
    saved = page.saved_results
    try:
        page.show_view("review")
        saved.attach_local_calibration(frozen, name="Night calibration")
        qtbot.waitUntil(lambda: bool(saved._local_loaded), timeout=10000)
        qtbot.waitUntil(lambda: saved._local_review is not None, timeout=10000)
        assert saved.mass_calibration.currentText().startswith("Night calibration")
        assert saved.result_path == __import__('pathlib').Path(frozen["path"])
        assert saved.view_experiment.count() == 2
        saved.view_experiment.setCurrentIndex(saved.view_experiment.findData("ac47"))
        assert saved._local_review.run == "ac47"
        assert saved.slider.maximum() == 1
        assert saved.review_panels.result is saved._local_review
        assert saved.review_panels.parent() is saved
        page.show_view("results")
        assert saved._local_review is None
    finally:
        page._stop_lookups()


def test_opening_review_automatically_uses_completed_refit_snapshot(frozen):
    from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
    calls = []
    saved = SimpleNamespace(set_overflow_comparison_mode=lambda mode: None,
        attach_local_calibration=lambda bound, **kwargs: calls.append((bound, kwargs)))
    owner = SimpleNamespace(phase_model_workspace=SimpleNamespace(series_id="series"),
        ensure_mass_analysis=lambda: SimpleNamespace(show_view=lambda view: None, saved_results=saved),
        mass_calibration_display=SimpleNamespace(_refit_series="root", refit_panel=SimpleNamespace(
            snapshot=dict(frozen=({}, frozen)), name_edit=SimpleNamespace(text=lambda: "Night fit"))),
        _method_series_root=lambda: "root", overflow_page=SimpleNamespace(mode="compare"),
        content_stack=SimpleNamespace(setCurrentWidget=lambda page: None))
    NativeFluidFlowerMainWindow._show_mass_analysis_view(owner, "review")
    assert calls == [(frozen, dict(name="Night fit"))]


@pytest.mark.parametrize("destination", ["legacy", "native"])
def test_local_gas_choices_do_not_leak_into_older_result_displays(qtbot, monkeypatch, frozen,
        tmp_path, native_attachment, destination):
    from darsia.fluidflower import calibration_review_panels as module
    from .test_calibration_review_panels import _ManualRead, _result
    actual_read, requests = module._PanelRead, []
    def capture(*args):
        worker = _ManualRead(*args)
        requests.append(worker)
        return worker
    monkeypatch.setattr(module, "_PanelRead", capture)
    widget = module.CalibrationReviewPanels()
    qtbot.addWidget(widget)
    local = local_reviews(frozen, dataset="series-a")[0]
    widget.layout_choice.setCurrentIndex(widget.layout_choice.findData("one"))
    widget.set_local_result(local, 0)
    panel = widget.panels[0]
    assert panel.gas_view.currentData() == "origin" and not panel.gas_controls.isHidden()
    panel.mass_limit.setText("1.0")
    if destination == "legacy":
        result = _result()
        widget.set_result(result, tmp_path/"old.json", 0, [])
    else:
        _config, completion, index = native_attachment
        result = _review(completion, index)
        widget.set_native_result(result, 0)
    try:
        # A subsequent content change must not revive unsupported controls.
        panel.content.setCurrentIndex(panel.content.findData("photograph"))
        panel.content.setCurrentIndex(panel.content.findData("mass_gas"))
        assert panel.gas_controls.isHidden() and not panel.gas_view.isEnabled() and not panel.gas_palette.isEnabled()
        assert panel.gas_view.currentData() == "origin"  # Retained local preference, never applied to this result.
        widget.set_active(True)
        request = requests[-1].requests[0]
        assert request["gas_view"] == "total" and not request["error"]
        if destination == "native":
            worker = actual_read(1, result, widget.path, 0, [request], widget)
            loaded = []
            worker.ready.connect(lambda *args: loaded.append(args))
            worker.run()
            assert loaded and not loaded[0][1][0][0].isNull() and not loaded[0][1][0][1]
            # Explicit unsupported decomposition still fails in the reader.
            unknown = actual_read(2, result, widget.path, 0, [dict(request, gas_view="origin")], widget)
            refused = []
            unknown.ready.connect(lambda *args: refused.append(args))
            unknown.run()
            assert "no recorded model/overflow" in refused[0][1][0][1]
        widget.set_active(False)
        widget.set_local_result(local, 0)
        assert not panel.gas_controls.isHidden() and panel.gas_view.isEnabled() and panel.gas_palette.isEnabled()
        assert panel.gas_view.currentData() == "origin"
    finally:
        widget.stop()
