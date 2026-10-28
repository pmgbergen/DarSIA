"""Finite local inline review with the original parser, store and window."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace as NS

import pytest
from PySide6.QtCore import QProcess, QRectF, Qt
from PySide6.QtGui import QAction, QFont, QFontDatabase, QImage
from PySide6.QtWidgets import QApplication, QComboBox, QDockWidget, QLabel, QLineEdit, QMainWindow, QStackedWidget, QTextEdit, QWidget

from darsia.fluidflower import native_inline_phase_review as inline
from darsia.fluidflower.native_phase_review_owner import NativePhaseReviewOwner
from darsia.fluidflower import native_phase_review_session as lifecycle


@pytest.fixture
def review(qtbot, monkeypatch, tmp_path, request):
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "series.toml").write_text('schema_version=1\nseries_id="ffac"\n[paths]\n', encoding="utf-8")
    monkeypatch.setenv("DARSIA_FFAC_PROJECT_ROOT", str(configured))
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent.json"))
    monkeypatch.delenv("DARSIA_FLUIDFLOWER_REVIEW_SCRIPT", raising=False)
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "postflight_manifest.json").write_text("{}", encoding="utf-8")
    from darsia.fluidflower.tools import calibration_review_desktop as desktop
    from darsia.fluidflower import series_native_context
    from darsia.fluidflower.phase_workflow import build_phase_review_command

    context = ContextVar("inline-test-reader", default="outside")
    @contextmanager
    def bound_context():
        token = context.set("exact-reader")
        try:
            yield
        finally:
            context.reset(token)
    monkeypatch.setattr(series_native_context, "saved_reader_ui_context", bound_context)
    monkeypatch.setattr(desktop.review, "discover_review_inventory", lambda root, **kw:
        desktop.review.ReviewInventory(pairs=(), unpaired={}, manifest_path=root / "postflight_manifest.json"))
    host = QMainWindow()
    if getattr(request, "param", None) == "panels":
        from .test_native_phase_steps import FakeStudio
        from darsia.fluidflower.native_phase_steps import NativePhaseSteps
        original_studio = FakeStudio()
        qtbot.addWidget(original_studio)
        steps = NativePhaseSteps(original_studio)
        stack = steps.workspace_stack
        host.setCentralWidget(steps)
    else:
        stack = QStackedWidget()
        host.setCentralWidget(stack)
        steps = QWidget()
        stack.addWidget(steps)
    steps.series_id = "ffac"
    panel = inline.NativeInlinePhaseReview(host, stack, steps)
    stack.addWidget(panel)
    events = []
    outer_save = QAction("Save series", host)
    outer_save.setShortcut("Ctrl+S")
    outer_save.triggered.connect(lambda: events.append("series-save"))
    host.addAction(outer_save)
    model = dict(model_id="saved-model", root=str(tmp_path / "saved-model"))
    studio = NS(review_save_root=QLineEdit(str(tmp_path / "reviews"), steps),
        reviewer=QLineEdit("Reviewer", steps), review_status=QLabel(steps), bundles=QTextEdit(steps),
        training_process=None, _series_batch_active=False,
        review_opened=NS(emit=lambda binding: events.append("opened")),
        _start_review_process=lambda *a, **kw: pytest.fail("External launch"),
        _selected_validation_model=lambda: model,
        _refresh_review_sessions=lambda: events.append("refreshed"))
    def command(mode, *, binding_override, session_id):
        return [sys.executable, *build_phase_review_command(mode=mode, reviewer="Reviewer",
            save_root=tmp_path / "reviews", cache_root=tmp_path / "cache",
            campaign_root=tmp_path / "campaign", selected_candidates={"ac01": "candidate-a"},
            target_model_id=model["model_id"], review_session_id=session_id, use_gpu=False)[1:]]
    studio._review_command = command
    workflow = NS(studio=studio, steps=steps, refresh=lambda: events.append("refresh-page"))
    owner = NativePhaseReviewOwner(workflow)
    owner.process_factory = panel.process_factory
    host.resize(1500, 1000)
    host.show()
    host.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is host)
    monkeypatch.setattr(QProcess, "start", lambda *a, **kw: pytest.fail("Inline review must not launch a process"))
    def clean(_widget):
        process = owner.process
        if process is not None:
            if process.loader is not None:
                qtbot.waitUntil(lambda: process.loader is None, timeout=10000)
            process.request_close()
            qtbot.waitUntil(lambda: owner.process is None, timeout=10000)
    qtbot.addWidget(host, before_close_func=clean)
    yield NS(host=host, stack=stack, steps=steps, panel=panel, owner=owner,
             studio=studio, model=model, desktop=desktop, context=context,
             root=tmp_path, events=events, outer_save=outer_save)
    clean(host)
    assert context.get() == "outside"


def opened(review, qtbot):
    review.owner.start("open_correction", {"selected_candidates": {"ac01": "candidate-a"}})
    qtbot.waitUntil(lambda: review.owner.process is None or review.owner.process.review_window is not None, timeout=10000)
    assert review.owner.process is not None, review.panel.notice.text() + review.studio.review_status.text()
    qtbot.waitUntil(lambda: review.owner.process.state() == QProcess.ProcessState.Running, timeout=10000)
    return review.owner.process, review.owner.process.review_window


@pytest.mark.parametrize("size", [(1280, 720), (3840, 1600)])
def test_waiting_message_stays_beside_header_in_a_large_workspace(qtbot, size):
    stack = QStackedWidget()
    qtbot.addWidget(stack)
    panel = inline.NativeInlinePhaseReview(stack, stack, QWidget())
    stack.addWidget(panel)
    stack.resize(*size)
    stack.show()
    panel.show_requirement("Loading the selected model's saved images…")
    qtbot.waitExposed(stack)
    assert panel.heading.geometry().top() < 25
    assert panel.notice.geometry().top() < 70
    assert panel.notice.geometry().bottom() < 120
    assert not panel.review_scroll.isVisible()


@pytest.mark.parametrize("review", ["panels"], indirect=True)
def test_basis_to_new_labels_draws_saves_and_reopens_the_same_inline_session(review, qtbot, monkeypatch):
    from .test_native_phase_scratch_review import make_collection
    from darsia.fluidflower.native_phase_panel_navigation import NativePhasePanelNavigation
    from darsia.fluidflower.phase_workflow import build_phase_review_command

    r = review
    font_file = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_file.is_file():
        QFontDatabase.addApplicationFont(str(font_file))
    data = make_collection(r.root / "scratch", monkeypatch)
    r.studio.review_save_root.setText(str(data.save_root))
    r.model["validation_binding"] = data.binding
    r.steps.studio._selected_validation_model = lambda: r.model
    r.steps._ready = True
    r.steps._legacy_availability["open_model_correction"] = True
    r.steps._original_review = lambda mode: r.owner.start(mode, data.binding)
    r.steps.phase_workflow.review_owner = r.owner
    r.owner.workflow.refresh = r.steps.refresh

    def command(mode, *, binding_override, session_id):
        return [sys.executable, *build_phase_review_command(mode=mode, reviewer="Reviewer",
            save_root=data.save_root, cache_root=r.root / "cache", campaign_root=data.source,
            selected_candidates=binding_override["selected_candidates"],
            target_model_id="source-model-is-not-the-basis", review_session_id=session_id,
            use_gpu=False)[1:]]

    r.studio._review_command = command
    navigation = NativePhasePanelNavigation(r.steps, r.panel)
    r.steps.scratch_basis.click()
    navigation.sections[2].toggle.click()
    qtbot.waitUntil(lambda: r.owner.process is None or r.owner.process.review_window is not None,
                   timeout=10000)
    process = r.owner.process
    assert process is not None, r.panel.notice.text() + r.studio.review_status.text()
    window = process.review_window
    qtbot.waitUntil(lambda: not window.viewer.base_item.pixmap().isNull(), timeout=10000)
    assert not window.isWindow() and window.window() is r.host
    assert not window.annotations and not window.workspace.target_model_id
    assert not r.steps.existing_basis.isEnabled()
    window.box_tool.click()
    window.annotation_category.setCurrentIndex(window.annotation_category.findData("confirmed_only_free_gas"))
    window.viewer.rectangle_drawn.emit(QRectF(.2, .2, .2, .2))
    window._save_now()
    qtbot.waitUntil(lambda: not window.save_inflight and window.store.json_path.is_file(), timeout=10000)
    annotation = window.store.decisions()["ac01"]["annotations"][0]
    assert annotation["category"] == "confirmed_only_free_gas"
    assert annotation["photo_identity"]["source_sha256"] == data.frames[0]["acquisition"]["sha256"]
    session_id = window.workspace.review_session_id
    assert r.host.grab().save(str(r.root / "basis-direct-label.png"))
    navigation.sections[3].toggle.click()
    assert r.owner.process is process and window.workflow_role.currentData() == "blind"
    navigation.sections[4].toggle.click()
    assert r.stack.currentWidget() is r.steps.workspace_overview
    assert r.studio.training_process is None
    process.request_close()
    qtbot.waitUntil(lambda: r.owner.process is None, timeout=10000)
    assert r.steps.scratch_basis.isChecked() and r.steps.existing_basis.isEnabled()
    navigation.sections[2].toggle.click()
    qtbot.waitUntil(lambda: r.owner.process is None or r.owner.process.review_window is not None,
                   timeout=10000)
    assert r.owner.process is not None, r.panel.notice.text()
    reopened = r.owner.process.review_window
    assert reopened.workspace.review_session_id == session_id
    assert reopened.annotations[0]["id"] == annotation["id"]


def test_original_window_and_adapters_are_inline_and_navigation_keeps_ownership(review, qtbot):
    before = set(QApplication.topLevelWidgets())
    process, window = opened(review, qtbot)
    assert isinstance(window, review.desktop.CalibrationReviewWindow)
    assert not window.isWindow() and window.window() is review.host
    assert not any(w.isVisible() for w in set(QApplication.topLevelWidgets()) - before)
    assert all(not dock.features() & QDockWidget.DockWidgetFeature.DockWidgetFloatable
               for dock in window.findChildren(QDockWidget))
    assert review.stack.currentWidget() is review.panel
    assert [type(adapter).__name__ for adapter in window._workflow_adapters] == [
        "ReviewProposalGeneration", "CalibrationContradictionDock", "PhasePredictionDisclosure"]
    assert window._workflow_request_context["model_root"] == Path(review.model["root"])
    assert window.workspace.selected_candidates == {"ac01": "candidate-a"}
    assert window.store.output_dir == window.workspace.output_dir
    assert review.context.get() == "exact-reader"
    review.panel.show_steps()
    assert review.owner.process is process and review.owner.close_block_reason()
    assert process.state() == QProcess.ProcessState.Running
    review.owner.start("blind_region_labeling", {})
    assert review.owner.process is process and review.stack.currentWidget() is review.panel
    process.request_close()
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert review.context.get() == "outside"
    assert lifecycle.read_current_review_session(review.root / "reviews", series="ffac", reviewer="Reviewer")["status"] == "active"


def test_label_phase_map_uses_original_overlay_and_blind_role_clears_pending_images(review, qtbot, monkeypatch):
    _process, window = opened(review, qtbot)
    retained_store = window.store
    callbacks = []
    window._workflow_prediction_disclosure = NS(request=callbacks.append)
    photo = QImage(40, 30, QImage.Format.Format_RGB32)
    photo.fill(Qt.GlobalColor.darkBlue)
    overlay = QImage(40, 30, QImage.Format.Format_ARGB32)
    overlay.fill(Qt.GlobalColor.green)
    window.viewer.set_image(photo, fit=True)
    window.current_pair = {"candidate_a": {"frame_count": 1}}
    window.base_key = window.displayed_base_key = "photo"
    window.reference_mode = False
    monkeypatch.setattr(window, "_render_vectors", lambda: None)
    monkeypatch.setattr(window, "_overlay_asset", lambda side, index: review.root / "overlay.png")
    monkeypatch.setattr(window.image_cache, "get", lambda source: overlay)
    monkeypatch.setattr(window.image_cache, "prefetch", lambda sources, **kwargs: None)
    assert window.contours.isVisible() and window.contour_filter.isVisible()
    window.contours.click()
    assert callbacks and window.viewer.overlay_item.pixmap().isNull()
    callbacks.pop()()
    assert not window.viewer.overlay_item.pixmap().isNull()
    assert window.viewer.overlay_item.pixmap().toImage().pixelColor(0, 0).green() == 255
    overlay_key = window.overlay_key
    window.contours.click()
    assert window.viewer.overlay_item.pixmap().isNull()
    window.contours.click()
    assert not window.viewer.overlay_item.pixmap().isNull()
    window.workflow_role.setCurrentIndex(window.workflow_role.findData("blind"))
    assert window.contours.isHidden() and window.contour_filter.isHidden()
    assert window.viewer.overlay_item.pixmap().isNull()
    window._image_loaded(overlay_key, overlay, "")
    assert window.viewer.overlay_item.pixmap().isNull()
    window.workflow_role.setCurrentIndex(window.workflow_role.findData("label"))
    assert not window.viewer.overlay_item.pixmap().isNull()
    assert window.store is retained_store
    assert window.workspace.review_surface == "blind_regions"


@pytest.mark.parametrize("review", ["panels"], indirect=True)
def test_actual_review_keeps_phase_panels_available_on_the_left(review, qtbot):
    process, window = opened(review, qtbot)
    panels = review.steps.phase_tabs.sections
    assert all(panel.isVisible() for panel in panels)
    assert review.stack.currentWidget() is review.panel
    assert review.steps.phase_workspace.widget(1) is review.stack
    blind = next(panel for panel in panels if panel.content is review.steps.groups[3])
    blind.toggle.click()
    assert blind.is_expanded() and blind.content.isVisible()
    assert review.stack.currentWidget() is review.panel
    assert review.owner.process is process and window.isVisible()
    review.panel.show_steps()
    assert review.stack.currentWidget() is review.steps.workspace_overview
    assert all(panel.isVisible() for panel in panels)
    process.show_review()
    assert review.stack.currentWidget() is review.panel
    process.request_close()
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert review.stack.currentWidget() is review.steps.workspace_overview
    assert all(panel.isVisible() for panel in panels)


@pytest.mark.parametrize("review", ["panels"], indirect=True)
def test_full_screen_keeps_only_the_review_and_restores_every_part(review, qtbot):
    host = review.panel.window()
    dock = QDockWidget("Logging", host)
    host.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    host.statusBar().showMessage("Ready")
    panel = review.panel
    assert not panel.full_screen_button.isVisible()  # No images, nothing to enlarge.
    process, window = opened(review, qtbot)
    controls = review.steps.phase_workspace.controls_scroll
    assert panel.full_screen_button.isVisible() and controls.isVisible() and dock.isVisible()
    panel.full_screen_button.click()
    assert host.isFullScreen() and window.isVisible() and panel.full_screen_button.text() == "Exit full screen"
    assert not controls.isVisible() and not dock.isVisible() and not host.statusBar().isVisible()
    panel.full_screen_button.click()
    assert not host.isFullScreen() and panel.full_screen_button.text() == "Full screen"
    assert controls.isVisible() and dock.isVisible() and host.statusBar().isVisible()
    panel.full_screen_button.click()
    process.request_close()
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert not host.isFullScreen() and controls.isVisible() and not panel.full_screen_button.isVisible()


@pytest.mark.parametrize("mode", ["Light", "Dark"])
def test_embedded_review_uses_host_theme_font_and_live_theme_changes(review, qtbot, mode):
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QListWidgetItem
    from darsia.gui.ui.theme import apply_theme, get_theme, success_color

    old_mode, old_font = get_theme(), QApplication.font()
    try:
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        apply_theme(QApplication.instance(), mode)
        QApplication.setFont(QFont("Segoe UI", 12))
        _process, window = opened(review, qtbot)
        for widget in (window, window.quiz_progress, window.new_round, window.reviewer):
            assert widget.font().pointSizeF() == review.host.font().pointSizeF()
            assert widget.font().family() == review.host.font().family()
        assert window.quiz_progress.palette().color(QPalette.ColorRole.Window) == review.host.palette().color(QPalette.ColorRole.Window)
        assert window.new_round.palette().color(QPalette.ColorRole.Button) == review.host.palette().color(QPalette.ColorRole.Button)
        assert "font-size" not in window.styleSheet()
        window.grab().save(str(review.root / f"phase-review-{mode.lower()}.png"))

        window.quiz_entry_help.setStyleSheet("color:#9bd4a7;")
        assert success_color(QApplication.palette()).name() in window.quiz_entry_help.styleSheet()
        # A theme change must retain the meaning of the latest original style,
        # rather than remapping the colour produced for the previous theme.
        apply_theme(QApplication.instance(), "Dark" if mode == "Light" else "Light")
        assert success_color(QApplication.palette()).name() in window.quiz_entry_help.styleSheet()
        QApplication.setFont(QFont("Segoe UI", 14))
        qtbot.waitUntil(lambda: window.new_round.font().pointSizeF() == 14)

        row = QListWidgetItem("NEW · example")
        row.setData(Qt.ItemDataRole.ForegroundRole, QPalette().brush(QPalette.ColorRole.Link))
        window.new_region_list.addItem(row)
        window._native_review_theme.refresh_rows()
        assert row.data(Qt.ItemDataRole.ForegroundRole) is None
        window._refresh_quiz()
        assert window.new_round.font().pointSizeF() == 14
    finally:
        apply_theme(QApplication.instance(), old_mode)
        QApplication.setFont(old_font)


def test_native_and_review_save_shortcuts_are_never_simultaneously_live(review, qtbot):
    process, window = opened(review, qtbot)
    assert not review.outer_save.shortcuts()
    saves = [a for a in window.findChildren(QAction) if a.shortcut().toString() == "Ctrl+S"]
    assert len(saves) == 1
    saves[0].triggered.connect(lambda: review.events.append("review-save"))
    qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert "review-save" in review.events and "series-save" not in review.events
    review.panel.show_steps()
    assert not saves[0].shortcuts() and review.outer_save.shortcuts()
    qtbot.keyClick(review.host, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert review.events[-1] == "series-save"
    process.show_review()
    assert saves[0].shortcuts() and not review.outer_save.shortcuts()


def test_original_controls_scroll_without_expanding_the_native_window(review, qtbot):
    original_font = QApplication.font()
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    QApplication.setFont(QFont("Segoe UI", 10))
    try:
        _process, window = opened(review, qtbot)
        review.host.resize(1280, 900)
        qtbot.waitUntil(lambda: review.host.width() == 1280)
        scroll = review.panel.review_scroll
        qtbot.waitUntil(lambda: scroll.horizontalScrollBar().maximum() > 0)
        scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().maximum())
        assert window.x() < 0
        assert window.x() + window.width() == scroll.viewport().width()
    finally:
        QApplication.setFont(original_font)


def test_metadata_reader_is_owned_and_captures_exact_context(review, qtbot, monkeypatch):
    entered, release = Event(), Event()
    original = review.desktop.build_workspace_model
    seen = []
    def load(*args, **kwargs):
        seen.append(review.context.get())
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(review.desktop, "build_workspace_model", load)
    review.owner.start("open_correction", {})
    try:
        qtbot.waitUntil(entered.is_set)
        process = review.owner.process
        assert process.loader.isRunning() and not review.panel.close_button.isEnabled()
        review.panel.show_steps()
        assert review.owner.process is process and review.owner.close_block_reason()
        with pytest.raises(ValueError, match="Close review"):
            review.owner.run_training(lambda: pytest.fail("Must not train during review opening"))
    finally:
        release.set()
    qtbot.waitUntil(lambda: process.state() == QProcess.ProcessState.Running)
    assert seen == ["exact-reader"]
    assert review.stack.currentWidget() is review.steps
    assert review.outer_save.shortcuts()
    assert not any(action.shortcuts() for action in process.review_window.findChildren(QAction))
    process.show_review()
    assert not review.outer_save.shortcuts()
    assert any(action.shortcut().toString() == "Ctrl+S"
               for action in process.review_window.findChildren(QAction))


def test_external_review_script_is_refused_inline_before_any_store_write(review, qtbot):
    review.studio._review_command = lambda *a, **kw: [sys.executable, str(review.root / "custom.py")]
    review.owner.start("open_correction", {})
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert "external REVIEW_SCRIPT cannot be embedded" in review.panel.notice.text()
    assert review.stack.currentWidget() is review.panel
    assert not list((review.root / "reviews").rglob("review_session.json"))
    assert review.context.get() == "outside"


def test_explicit_missing_external_script_is_not_silently_replaced(review, qtbot, monkeypatch):
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_REVIEW_SCRIPT", str(review.root / "missing-custom.py"))
    review.owner.start("open_correction", {})
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert "external REVIEW_SCRIPT cannot be embedded" in review.panel.notice.text()
    assert not list((review.root / "reviews").rglob("review_session.json"))


def test_changed_source_binding_is_rejected_before_original_store_reopens(review, qtbot):
    process, window = opened(review, qtbot)
    binding_path = window.workspace.output_dir / "review_source_binding.json"
    retained = binding_path.read_bytes()
    process.request_close()
    qtbot.waitUntil(lambda: review.owner.process is None)
    review.model["root"] = str(review.root / "another-model-folder")
    review.owner.start("open_correction", {})
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert "another model folder" in review.panel.notice.text()
    assert binding_path.read_bytes() == retained
    assert review.context.get() == "outside"


def test_original_async_final_save_finishes_before_owner_retires(review, qtbot, monkeypatch):
    process, window = opened(review, qtbot)
    entered, release = Event(), Event()
    target = window.store.json_path
    calls = []
    def durable_save(request):
        entered.set()
        assert release.wait(5)
        calls.append(request)
        target.write_text(json.dumps(request), encoding="utf-8")
        return request
    monkeypatch.setattr(window.store, "save", durable_save)
    window.current_run = "ac01"
    window.annotations = [{"id": "retained-label", "label": "water"}]
    window.answers = [{"proposal_id": "retained-box", "human_choice": "water"}]
    window.notes.setPlainText("Saved inline")
    process.request_close()
    try:
        qtbot.waitUntil(entered.is_set)
        assert review.owner.process is process and review.owner.close_block_reason()
        assert window.save_inflight and window._workflow_closing
        assert review.context.get() == "exact-reader"
        review.panel.show_steps()
        assert review.owner.process is process
    finally:
        release.set()
    qtbot.waitUntil(lambda: review.owner.process is None)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert len(calls) == 1 and saved["notes"] == "Saved inline"
    assert saved["annotations"][0]["id"] == "retained-label"
    assert saved["phase_quiz_answers"][0]["proposal_id"] == "retained-box"
    assert review.context.get() == "outside"


def test_final_save_failure_keeps_original_review_open_for_retry(review, qtbot, monkeypatch):
    process, window = opened(review, qtbot)
    def failed_save(request):
        raise OSError("Finite fixture: disk unavailable")
    monkeypatch.setattr(window.store, "save", failed_save)
    window.current_run = "ac01"
    process.request_close()
    qtbot.waitUntil(lambda: window.sync_error_pending and not window._workflow_closing)
    assert review.owner.process is process and review.owner.close_block_reason()
    assert window.centralWidget().isEnabled() and review.context.get() == "exact-reader"
    monkeypatch.setattr(window.store, "save", lambda request: request)
    window._save_now()
    qtbot.waitUntil(lambda: not window.save_inflight and not window.sync_error_pending)
    process.request_close()
    qtbot.waitUntil(lambda: review.owner.process is None)


def test_later_adapter_failure_retains_original_close_and_workers(review, qtbot, monkeypatch):
    from darsia.fluidflower.ui import phase_contradiction_review
    def unavailable(*args):
        raise ValueError("Finite adapter fixture failure")
    monkeypatch.setattr(phase_contradiction_review, "install", unavailable)
    review.owner.start("open_correction", {})
    qtbot.waitUntil(lambda: "Finish its saved work" in review.panel.notice.text())
    process = review.owner.process
    assert process.review_window is not None and review.context.get() == "exact-reader"
    assert review.owner.close_block_reason()
    process.request_close()
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert review.context.get() == "outside"


def test_invalid_arguments_are_visible_in_page_instead_of_exiting_gui(review, qtbot):
    command = review.studio._review_command
    review.studio._review_command = lambda *a, **kw: [*command(*a, **kw), "--unrecognized-fixture-flag"]
    review.owner.start("open_correction", {})
    qtbot.waitUntil(lambda: review.owner.process is None)
    assert "invalid arguments" in review.panel.notice.text()
    assert review.context.get() == "outside"


def test_series_navigation_reverts_selection_while_review_owns_context(review, qtbot):
    from darsia.fluidflower.native_darsia_app import PhaseModelWorkspace
    from darsia.fluidflower.native_phase_subpage import NativePhaseModelStudioSubpage
    process, _window = opened(review, qtbot)
    page = NS(studio=review.studio, inline_review=review.panel,
        native_steps=NS(phase_workflow=NS(close_block_reason=review.owner.close_block_reason)),
        series_id="ffac", project_root=review.root)
    page.review_context_block_reason = lambda: NativePhaseModelStudioSubpage.review_context_block_reason(page)
    selector = QComboBox(review.host)
    selector.addItem("FF_AC", "ffac")
    selector.addItem("FF_UM", "ffum")
    selector.setCurrentIndex(1)
    workspace = NS(studio_page=page, series_selector=selector, _selected_series="ffac")
    PhaseModelWorkspace._series_changed(workspace, 1)
    assert selector.currentData() == "ffac" and review.owner.process is process
    assert "Close review" in review.panel.notice.text()
    with pytest.raises(ValueError, match="Close review"):
        NativePhaseModelStudioSubpage.bind_context(page, "ffum", review.root / "other")
    assert page.series_id == "ffac" and page.project_root == review.root


def test_windows_bootstrapped_runtime_is_the_same_loaded_gui_environment(monkeypatch, tmp_path):
    from darsia.fluidflower import phase_workflow
    monkeypatch.delenv("DARSIA_FLUIDFLOWER_REVIEW_SCRIPT", raising=False)
    runtime = tmp_path / "venv"
    program = runtime / "Scripts" / "pythonw.exe"
    monkeypatch.setattr(phase_workflow, "phase_python_executable", lambda **kw: str(program))
    monkeypatch.setattr(inline.sys, "executable", str(tmp_path / "base" / "python.exe"))
    monkeypatch.setattr(inline.PySide6, "__file__", str(runtime / "Lib" / "site-packages" / "PySide6" / "__init__.py"))
    args = NS(workspace="training", phase_workflow_series="ffac", phase_workflow_role="label",
              evidence_audit=None, review_current_model_root=tmp_path / "model", online=True)
    desktop = NS(build_parser=lambda: NS(parse_args=lambda values: args))
    assert inline._arguments(program, ["-m", inline._MODULE], desktop) is args
    monkeypatch.setattr(inline.PySide6, "__file__", str(tmp_path / "another-env" / "PySide6" / "__init__.py"))
    with pytest.raises(ValueError, match="this GUI's Python runtime"):
        inline._arguments(program, ["-m", inline._MODULE], desktop)
