"""Native presentation of the original model-bound phase correction window."""

from __future__ import annotations

from contextlib import ExitStack
from contextvars import copy_context
from copy import deepcopy
import os
from pathlib import Path
import sys

import PySide6
from PySide6.QtCore import QByteArray, QEvent, QProcess, QSignalBlocker, QThread, QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDockWidget, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea, QToolBar,
    QVBoxLayout, QWidget,
)
from shiboken6 import isValid


_MODULE = "darsia.fluidflower.tools.calibration_review_desktop"


def _desktop_module():
    from .tools import calibration_review_desktop
    return calibration_review_desktop


def _arguments(program, arguments, desktop, *, review_basis="model"):
    script = os.environ.get("DARSIA_FLUIDFLOWER_REVIEW_SCRIPT", "").strip()
    if script and Path(script).expanduser().resolve() != Path(desktop.__file__).resolve():
        raise ValueError("The configured external REVIEW_SCRIPT cannot be embedded. Use the original standalone review application, or select the packaged review tool before opening inline review.")
    executable = Path(program).resolve()
    current = Path(sys.executable).resolve()
    # The Windows launcher can bootstrap the venv's packages into base Python.
    # Accept that same loaded environment, never an unrelated interpreter.
    from .phase_workflow import phase_python_executable
    configured = Path(phase_python_executable(gui=True)).resolve()
    loaded_environment = (executable == configured and
        Path(PySide6.__file__).resolve().is_relative_to(configured.parent.parent))
    if (not loaded_environment and executable.parent != current.parent
            or executable.stem.casefold() not in {"python", "pythonw", current.stem.casefold()}):
        raise ValueError("Inline phase review requires this GUI's Python runtime. The configured external interpreter remains available through the standalone review application.")
    if list(arguments[:2]) != ["-m", _MODULE]:
        raise ValueError("The configured external REVIEW_SCRIPT cannot be embedded. Use the original standalone review application, or select the packaged review tool before opening inline review.")
    try:
        args = desktop.build_parser().parse_args(arguments[2:])
    except SystemExit as error:
        raise ValueError("The configured phase review command has invalid arguments.") from error
    if (args.workspace != "training" or not args.phase_workflow_series
            or not args.phase_workflow_role or args.evidence_audit
            or review_basis not in {"model", "scratch"}
            or review_basis == "model" and not args.review_current_model_root
            or review_basis == "scratch" and (args.review_current_model_root or args.target_model_id)):
        raise ValueError("Inline review requires the original reserved, model-bound phase correction command.")
    args.native_review_basis = review_basis
    if args.online:
        args.offline_bundle = None
    return args


def _prepare(desktop, args, reviewer, save_root):
    """Original preflight and store builder; no widgets are created by this read."""
    if getattr(args, "native_review_basis", "model") == "scratch":
        from .native_phase_scratch_review import prepare
        return prepare(desktop, args, reviewer, save_root)
    from .native_phase_review_session import read_current_review_session
    from .ui.phase_model_correction import validate_workspace

    workspace = desktop.local_reviewer_workspace(
        desktop.review_workspaces(args)["training"], save_root, reviewer)
    reservation = read_current_review_session(
        save_root, series=args.phase_workflow_series, reviewer=reviewer)
    if reservation is None:
        raise ValueError("The native review session has not been reserved")
    binding = validate_workspace(workspace, reviewer=reviewer, reservation=reservation,
                                 model_root=args.review_current_model_root)
    desktop.ensure_writable_directory(workspace.output_dir)
    binding_path = Path(workspace.output_dir) / "review_source_binding.json"
    if not binding_path.exists():
        desktop.atomic_json(binding_path, binding)
    model, store = desktop.build_workspace_model(args, workspace, reviewer_name=reviewer)
    return workspace, binding, model, store


class _Preparation(QThread):
    def __init__(self, callback, parent):
        super().__init__(parent)
        self.callback, self.context = callback, copy_context()
        self.result, self.error = None, ""

    def run(self):
        try:
            self.result = self.context.run(self.callback)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"


class _DesktopPresentation:
    def __init__(self, desktop, process):
        self.desktop = desktop

        class EmbeddedWindow(desktop.CalibrationReviewWindow):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                process.review_window = self
                self.setParent(process.panel, Qt.WindowType.Widget)
                self.setMinimumSize(0, 0)
                from .native_phase_review_theme import NativePhaseReviewTheme
                self._native_review_theme = NativePhaseReviewTheme(self)

            def _refresh_quiz(self):
                super()._refresh_quiz()
                theme = getattr(self, "_native_review_theme", None)
                if theme is not None:
                    theme.refresh_rows()

            def event(self, event):
                result = super().event(event)
                if event.type() == QEvent.Type.Close and event.isAccepted():
                    # The original override has finished its save/reader checks.
                    QTimer.singleShot(0, process._closed)
                return result

        self.CalibrationReviewWindow = EmbeddedWindow

    def __getattr__(self, name):
        return getattr(self.desktop, name)


class InlineReviewProcess(QProcess):
    """Retain the existing owner protocol without starting an OS process."""

    already_open_message = (
        "Phase correction is already open on this page. "
        "Switch between labels and blind validation here."
    )

    def __init__(self, panel, parent):
        super().__init__(parent)
        self.panel = panel
        self.review_window = None
        self.loader = None
        self.lifetime = ExitStack()
        self._state = QProcess.ProcessState.NotRunning
        self._error = ""
        self.review_basis = "model"

    def state(self):
        return self._state

    def close_block_reason(self):
        return "Use Close review on the phase model page and let its saved work finish before continuing."

    def errorString(self):
        return self._error

    def readAllStandardError(self):
        return QByteArray(self._error.encode("utf-8"))

    def readAllStandardOutput(self):
        return QByteArray()

    def start(self, program, arguments):
        self.program, self.arguments = program, list(arguments)
        self._state = QProcess.ProcessState.Starting
        self.panel.process = self
        if "--phase-workflow-role" in self.arguments:
            self.panel.requested_role = self.arguments[self.arguments.index("--phase-workflow-role") + 1]
        self.panel.notice.setText("Opening the exact saved phase-review session…")
        self.panel.close_button.setEnabled(False)
        self.show_review()
        QTimer.singleShot(0, self._begin)

    def _begin(self):
        try:
            from .native_phase_review_session import review_editing_lease
            from .series_native_context import saved_reader_ui_context

            self.desktop = _desktop_module()
            self.args = _arguments(self.program, self.arguments, self.desktop,
                                   review_basis=self.review_basis)
            self.reviewer, self.save_root = self.desktop.effective_reviewer_settings(self.args)
            self.lifetime.enter_context(saved_reader_ui_context())
            self.lifetime.enter_context(review_editing_lease(self.save_root,
                series=self.args.phase_workflow_series, reviewer=self.reviewer))
            self.loader = _Preparation(lambda: _prepare(
                self.desktop, self.args, self.reviewer, self.save_root), self)
            self.loader.finished.connect(self._prepared)
            self.loader.start()
        except Exception as error:
            self._failed(f"{type(error).__name__}: {error}")

    def _prepared(self):
        loader = self.loader
        if loader.isRunning():
            QTimer.singleShot(0, self._prepared)
            return
        self.loader = None
        loader.deleteLater()
        if loader.error:
            self._failed(loader.error)
            return
        try:
            from .native_phase_review_session import attach_review_manifest
            from .ui.phase_model_correction import create_window
            from .ui.phase_review_proposal_generation import install as proposals
            from .ui.phase_contradiction_review import install as contradictions
            from .ui.phase_prediction_disclosure import install as prediction_disclosure

            workspace, binding, model, store = loader.result
            attach_review_manifest(self.save_root, series=self.args.phase_workflow_series,
                reviewer=self.reviewer, review_session_id=workspace.review_session_id,
                session_path=workspace.output_dir / "review_session.json")
            desktop = _DesktopPresentation(self.desktop, self)
            scratch = self.review_basis == "scratch"
            if scratch:
                from .native_phase_scratch_review import create_window
            window = create_window(desktop, model, store, self.args.cache_root, workspace,
                reviewer_name=self.reviewer, role=self.panel.requested_role or self.args.phase_workflow_role,
                use_opengl=not self.args.no_gpu and
                    self.desktop.os.environ.get("QT_QPA_PLATFORM", "").casefold() != "offscreen",
                on_back=self.panel.show_steps)
            self.review_window = window
            window._workflow_model_root = self.args.review_current_model_root
            window._workflow_request_context = dict(save_root=Path(self.save_root),
                series=self.args.phase_workflow_series, reviewer=self.reviewer,
                model_root=self.args.review_current_model_root, binding=deepcopy(binding),
                workspace={key: deepcopy(getattr(workspace, key)) for key in (
                    "key", "single_model", "review_surface", "review_session_id", "target_model_id", "output_dir",
                    "combined_root", "ffac_root", "ffum_root", "selected_candidates", "ffac_runs", "ffum_runs")})
            if not scratch:
                window._workflow_adapters.append(proposals(window, desktop))
                window._workflow_adapters.append(contradictions(window, desktop))
                window._workflow_prediction_disclosure = prediction_disclosure(window, desktop)
                window._workflow_adapters.append(window._workflow_prediction_disclosure)
            self._embed_docks()
            self.panel.review_scroll.setWidget(window)
            window.show()
            self._state = QProcess.ProcessState.Running
            self.panel.notice.setText("New phase labels · Autosaved in this session."
                if scratch else "Phase model correction · Every edit uses the original saved review session.")
            self.panel.close_button.setEnabled(True)
            self.panel.refresh_shortcuts()
            self.started.emit()
        except Exception as error:
            self._failed(f"{type(error).__name__}: {error}")

    def show_review(self):
        self.panel.heading.setText("Blindly validate phase model"
            if self.panel.requested_role == "blind" else "Label phases")
        self.panel.close_button.show()
        self.panel.full_screen_button.show()
        self.panel.review_scroll.show()
        self.panel.empty_space.hide()
        self.panel.stack.setCurrentWidget(self.panel)
        if self.review_window is not None:
            self.review_window.show()

    def request_close(self):
        if self.review_window is not None:
            self.review_window.close()

    def _failed(self, message):
        self._error = message
        self.panel.notice.setText("Could not open inline phase review: " + message)
        if self.review_window is not None:
            # A later adapter error must not release a writer's lease or destroy
            # an already constructed window with active image/save workers.
            self._state = QProcess.ProcessState.Running
            self.panel.review_scroll.setWidget(self.review_window)
            self._embed_docks()
            self.review_window.show()
            self.panel.close_button.setEnabled(True)
            self.panel.notice.setText(
                "Phase review could not finish opening: " + message +
                " Finish its saved work with Close review before retrying."
            )
            self.panel.refresh_shortcuts()
            return
        self.panel.close_button.setEnabled(False)
        self.panel.process = None
        self.panel.close_button.hide()
        self.panel.set_full_screen(False)
        self.panel.full_screen_button.hide()
        self.panel.review_scroll.hide()
        self.panel.empty_space.show()
        self.lifetime.close()
        self._state = QProcess.ProcessState.NotRunning
        self.errorOccurred.emit(QProcess.ProcessError.FailedToStart)

    def _embed_docks(self):
        for dock in self.review_window.findChildren(QDockWidget):
            dock.setFeatures(dock.features() & ~QDockWidget.DockWidgetFeature.DockWidgetFloatable)

    def _closed(self):
        if self._state == QProcess.ProcessState.NotRunning:
            return
        window, self.review_window = self.review_window, None
        self.panel.set_full_screen(False)
        self.panel.full_screen_button.hide()
        self.panel.review_scroll.takeWidget()
        window.deleteLater()
        self.panel.process = None
        self.panel.close_button.setEnabled(False)
        self.panel.notice.setText("Phase correction closed. Saved labels and validation boxes remain in the active session.")
        self.lifetime.close()
        self._state = QProcess.ProcessState.NotRunning
        self.panel.show_steps()
        self.finished.emit(0, QProcess.ExitStatus.NormalExit)


class NativeInlinePhaseReview(QWidget):
    def __init__(self, subpage, stack, steps):
        super().__init__(subpage)
        self.stack, self.steps = stack, steps
        self.process = None
        self.requested_role = None
        self._outer_shortcuts, self._inner_shortcuts, self._chrome = [], [], []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.heading = QLabel("Phase model")
        heading_font = self.heading.font()
        heading_font.setBold(True)
        self.heading.setFont(heading_font)
        controls.addWidget(self.heading)
        self.close_button = QPushButton("Close review")
        self.close_button.setToolTip("Finish the original autosave and readers, then close this review.")
        self.close_button.setEnabled(False)
        self.close_button.hide()
        self.close_button.clicked.connect(lambda: self.process.request_close() if self.process else None)
        controls.addWidget(self.close_button)
        # Full screen keeps only the images and their own tools (labels, blind
        # validation and the model's proposals); F11 toggles it as well.
        self._full_screen_hidden, self._full_screen_state = [], None
        self.full_screen_button = QPushButton("Full screen")
        self.full_screen_button.setCheckable(True)
        self.full_screen_button.setToolTip(
            "Show only the images and their tools over the whole screen. F11 or Exit full screen returns.")
        self.full_screen_button.toggled.connect(self.set_full_screen)
        self.full_screen_button.hide()
        controls.addWidget(self.full_screen_button)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(
            lambda: self.full_screen_button.toggle() if self.full_screen_button.isVisible() else None)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.notice)
        self.review_scroll = QScrollArea()
        self.review_scroll.setWidgetResizable(True)
        self.review_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.review_scroll.setMinimumSize(0, 0)
        layout.addWidget(self.review_scroll, 1)
        # Keep a waiting/error message beside the panel header when the
        # editor is hidden, rather than distributing it across a tall window.
        self.empty_space = QWidget()
        layout.addWidget(self.empty_space, 1)
        self.empty_space.hide()

    def process_factory(self, parent):
        return InlineReviewProcess(self, parent)

    def open_role(self, mode):
        """Focus the requested tools in the same original saved review."""
        self.requested_role = "blind" if mode in {"blind", "blind_region_labeling"} else "label"
        self.stack.setCurrentWidget(self)
        process = self.process
        if process is not None:
            window = process.review_window
            if window is not None:
                index = window.workflow_role.findData(self.requested_role)
                window.workflow_role.setCurrentIndex(index)
                # Clicking the active role must still bring its drawing tools
                # back into view after the user has scrolled elsewhere.
                window._focus_workflow_role()
            process.show_review()

    def show_requirement(self, text):
        self.stack.setCurrentWidget(self)
        self.heading.setText("Blindly validate phase model"
            if self.requested_role == "blind" else "Label phases")
        self.close_button.setVisible(self.process is not None)
        self.full_screen_button.setVisible(self.process is not None)
        self.review_scroll.setVisible(self.process is not None)
        self.empty_space.setVisible(self.process is None)
        self.notice.setText(text)

    def set_full_screen(self, on):
        """Show only the review images and their tools over the whole screen, or return.

        The main window's sidebar, docks, status lines and the phase panels are
        hidden and the window goes full screen; leaving restores each part as
        it was. The page itself, its heading and the review stay unchanged.
        """
        root = self.window()
        on = bool(on) and isinstance(root, QMainWindow) and root is not self
        if on and not self._full_screen_hidden:
            splitter = getattr(root, "root_splitter", None)
            workspace = getattr(self.steps, "phase_workspace", None)
            parts = [splitter.widget(0) if splitter is not None else None, root.statusBar(),
                     getattr(root, "series_header", None), getattr(root, "status_header", None),
                     getattr(getattr(root, "workflow_status", None), "panel", None),
                     getattr(workspace, "controls_scroll", None),
                     *root.findChildren(QDockWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)]
            self._full_screen_hidden = [(part, not part.isHidden()) for part in parts
                                        if isinstance(part, QWidget) and not part.isAncestorOf(self)]
            for part, _visible in self._full_screen_hidden:
                part.hide()
            self._full_screen_state = root.windowState()
            root.showFullScreen()
        elif not on and self._full_screen_hidden:
            for part, visible in self._full_screen_hidden:
                if isValid(part):
                    part.setVisible(visible)
            self._full_screen_hidden = []
            if self._full_screen_state is not None and isValid(root):
                root.setWindowState(self._full_screen_state)
            self._full_screen_state = None
        with QSignalBlocker(self.full_screen_button):
            self.full_screen_button.setChecked(bool(self._full_screen_hidden))
        self.full_screen_button.setText("Exit full screen" if self._full_screen_hidden else "Full screen")

    def show_steps(self):
        self.stack.setCurrentWidget(getattr(self.steps, "workspace_overview", self.steps))

    def refresh_shortcuts(self):
        if self.isVisible():
            self._restore(self._inner_shortcuts)
            self._inner_shortcuts = []
        else:
            self._suspend_inner_shortcuts()

    def _suspend_inner_shortcuts(self):
        existing = {action for action, _ in self._inner_shortcuts}
        self._inner_shortcuts.extend(
            (action, action.shortcuts()) for action in self.findChildren(QAction)
            if action not in existing and action.shortcuts())
        for action, _ in self._inner_shortcuts:
            if isValid(action):
                action.setShortcuts([])

    @staticmethod
    def _restore(saved):
        for action, shortcuts in saved:
            if isValid(action):
                action.setShortcuts(shortcuts)

    def showEvent(self, event):
        super().showEvent(event)
        root = self.window()
        if root is self or not hasattr(root, "menuBar"):
            return
        own = set(self.findChildren(QAction))
        self._outer_shortcuts = [(action, action.shortcuts()) for action in root.findChildren(QAction)
                                 if action not in own and action.shortcuts()]
        for action, _ in self._outer_shortcuts:
            action.setShortcuts([])
        self.refresh_shortcuts()
        chrome = [root.menuBar(), *root.findChildren(
            QToolBar, options=Qt.FindChildOption.FindDirectChildrenOnly)]
        self._chrome = [(widget, not widget.isHidden()) for widget in chrome]
        for widget, _ in self._chrome:
            widget.hide()

    def hideEvent(self, event):
        super().hideEvent(event)
        if not self.window().isMinimized():
            self.set_full_screen(False)  # Leaving the review leaves full screen too.
        self._suspend_inner_shortcuts()
        self._restore(self._outer_shortcuts)
        for widget, visible in self._chrome:
            if isValid(widget):
                widget.setVisible(visible)
        self._outer_shortcuts, self._chrome = [], []
