"""Native Depth UI plus sequencing of the two unchanged DarSIA commands.

The pane only emits an explicit user's requested depth settings. The adapter
must persist those selected fields using its existing configuration save before
starting the sequence. No physical defaults, interpolation or SCI code lives
here. Original ProcessRunner owns output capture, Done/Error and process abort.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DepthSetupPanel(QWidget):
    compute_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        layout = QVBoxLayout(self)
        heading = QLabel("Depth")
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Constant depth", "constant")
        self.mode.addItem("Measured points (CSV)", "Load from CSV")
        form.addRow("Depth source", self.mode)
        self.constant_label = QLabel("Constant depth [m]")
        self.constant_depth = QLineEdit()
        self.constant_depth.setPlaceholderText("Enter the measured constant depth")
        form.addRow(self.constant_label, self.constant_depth)
        self.csv_label = QLabel("Measurements CSV")
        path_row = QWidget()
        paths = QHBoxLayout(path_row)
        paths.setContentsMargins(0, 0, 0, 0)
        self.measurements = QLineEdit()
        self.measurements.setPlaceholderText("Choose the measurements file")
        self.browse = QPushButton("Browse…")
        self.browse.clicked.connect(self._browse)
        paths.addWidget(self.measurements, 1)
        paths.addWidget(self.browse)
        form.addRow(self.csv_label, path_row)
        layout.addLayout(form)
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        layout.addWidget(self.validation)
        self.compute_button = QPushButton("Compute depth map")
        self.compute_button.clicked.connect(self._request_compute)
        layout.addWidget(self.compute_button)
        self.status = QLabel("No depth map loaded.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._map_layout = QVBoxLayout()
        layout.addLayout(self._map_layout, 1)
        self.map_canvas = None
        self.map_figure = None
        self.map_artist = None
        self.saved_map = None
        self.mode.currentIndexChanged.connect(self._update_controls)
        self.constant_depth.textChanged.connect(self._update_controls)
        self.measurements.textChanged.connect(self._update_controls)
        self.set_configuration({})

    def set_configuration(
        self, depth: Mapping, *, measured_csv: str | Path | None = None
    ) -> None:
        """Display provided existing config; this never reads or writes a file.

        measured_csv is an optional explicit existing-series reference supplied
        by the adapter, not a guessed catalogue constant or physical default.
        """
        mode = str(depth.get("measurements_mode", "Load from CSV"))
        self.mode.setCurrentIndex(self.mode.findData(mode))
        self.constant_depth.setText(
            str(depth["constant_depth"]) if "constant_depth" in depth else ""
        )
        self.measurements.setText(str(depth.get("measurements") or measured_csv or ""))
        self._update_controls()

    def requested_settings(self) -> dict:
        mode = self.mode.currentData()
        if mode not in {"constant", "Load from CSV"}:
            raise ValueError("Choose constant depth or measured points.")
        target = self.measurements.text().strip()
        if not target:
            raise ValueError("Choose the measurements CSV.")
        requested = {"measurements_mode": mode, "measurements": target}
        if mode == "constant":
            try:
                depth = float(self.constant_depth.text())
            except ValueError as error:
                raise ValueError(
                    "Enter a positive constant depth in metres."
                ) from error
            if not math.isfinite(depth) or depth <= 0:
                raise ValueError("Enter a positive constant depth in metres.")
            requested["constant_depth"] = depth
        return requested

    def _update_controls(self, *_args) -> None:
        constant = self.mode.currentData() == "constant"
        self.constant_label.setVisible(constant)
        self.constant_depth.setVisible(constant)
        self.csv_label.setText(
            "Generated measurements CSV" if constant else "Measurements CSV"
        )
        for widget in (self.mode, self.constant_depth, self.measurements, self.browse):
            widget.setEnabled(not self._busy)
        try:
            self.requested_settings()
        except ValueError as error:
            self.validation.setText(str(error))
            valid = False
        else:
            self.validation.clear()
            valid = True
        self.compute_button.setEnabled(valid and not self._busy)

    def _request_compute(self) -> None:
        if self.compute_button.isEnabled():
            self.compute_requested.emit(self.requested_settings())

    def _browse(self) -> None:
        method = (
            QFileDialog.getSaveFileName
            if self.mode.currentData() == "constant"
            else QFileDialog.getOpenFileName
        )
        path, _selected = method(
            self, "Depth measurements CSV", self.measurements.text(), "CSV (*.csv)"
        )
        if path:
            self.measurements.setText(path)

    def set_running(self, running: bool) -> None:
        self._busy = bool(running)
        self._update_controls()

    def set_status(self, text: str) -> None:
        """The adapter supplies saved-map range/status; no map is calculated here."""
        self.status.setText(text)

    def set_saved_depth_map(self, saved: SavedDepthMap) -> None:
        """Show the already read map inline, without recomputing or saving it."""
        self.saved_map = saved
        self.map_artist = None
        self.set_status(saved.status)
        if self.map_figure is not None:
            self.map_figure.clear()
            self.map_canvas.hide()
        if saved.image is None:
            return
        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        if self.map_canvas is None:
            self.map_figure = Figure(layout="constrained")
            self.map_canvas = FigureCanvasQTAgg(self.map_figure)
            self.map_canvas.setMinimumSize(280, 160)
            self._map_layout.addWidget(self.map_canvas)
        try:
            axis = self.map_figure.add_subplot(111)
            self.map_artist = axis.imshow(
                np.ma.masked_invalid(np.asarray(saved.image.img) * 1000.0),
                extent=saved.image.domain,
                origin="upper",
                interpolation="nearest",
            )
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            self.map_figure.colorbar(self.map_artist, ax=axis, label="Depth [mm]")
            self.map_canvas.show()
            self.map_canvas.draw_idle()
        except Exception as error:
            self.map_figure.clear()
            self.map_artist = None
            self.set_status(f"{saved.status}\nDepth map display unavailable: {error}")


def depth_workflow_commands(
    python_executable: str | Path,
    config_path: str | Path,
    *,
    force: bool = False,
    show: bool = False,
) -> tuple[list[str], list[str]]:
    """The two original CLI entrypoints, never --all or a generated wrapper."""
    prefix = [str(python_executable), "-m"]
    first = prefix + [
        "darsia.presets.workflows.user_interface_preprocessing",
        "--config",
        str(config_path),
        "--depth-measurements",
    ]
    second = prefix + [
        "darsia.presets.workflows.user_interface_setup",
        "--config",
        str(config_path),
        "--depth",
    ]
    if force:
        first.append("--force")
    if show:
        first.append("--show")
        second.append("--show")
    return first, second


class DepthWorkflowSequence(QObject):
    """A UI sequence; ProcessRunner and the two old commands do the work.

    Caller saves the pane's settings first, then starts this sequence. Disable
    series/run/depth selection while busy. Original runner handles each command's
    output and Done/Error dialogs unchanged. A successful first command advances
    after its original completion handling. Failure/abort never starts stage 2.
    """

    stage_changed = Signal(str)
    busy_changed = Signal(bool)
    completed = Signal(bool, str)

    def __init__(
        self,
        main_window,
        python_executable,
        *,
        resolve_conflict=None,
        config_is_current=None,
    ):
        super().__init__(main_window if isinstance(main_window, QObject) else None)
        self.window = main_window
        self.python_executable = python_executable
        self.resolve_conflict = resolve_conflict
        self.config_is_current = config_is_current
        self.current_process = None
        self._running = False
        self._cancelled = False
        self._stage = -1
        self._commands = ()
        self._config = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self, config_path: str | Path, *, show: bool = False) -> bool:
        if self._running or self.window.process_runner.active:
            self.completed.emit(False, "Another calculation is already running.")
            return False
        path = Path(config_path)
        resolver = self.resolve_conflict
        if resolver is None:
            from darsia.gui.ui.preprocessing import resolve_depth_measurements_conflict

            resolver = lambda chosen: resolve_depth_measurements_conflict(
                self.window, chosen
            )
        try:
            decision = resolver(path)
        except Exception as error:
            self.window.print_log(str(error))
            self.completed.emit(
                False, f"Depth measurements could not be checked: {error}"
            )
            return False
        if decision is None:
            self.completed.emit(False, "Depth map calculation cancelled.")
            return False
        self._config = path
        self._commands = depth_workflow_commands(
            self.python_executable, path, force=bool(decision), show=show
        )
        self._cancelled = False
        self._running = True
        self.busy_changed.emit(True)
        self._launch(0)
        return self._running

    def _is_current(self) -> bool:
        if self.config_is_current is not None:
            return bool(self.config_is_current(self._config))
        current = getattr(self.window, "config_file", None)
        return not current or os.path.normcase(
            os.path.abspath(current)
        ) == os.path.normcase(os.path.abspath(self._config))

    def _launch(self, stage: int) -> None:
        if not self._running:
            return
        if self._cancelled:
            self._finish(False, "Depth map calculation stopped.")
            return
        if not self._is_current():
            self._finish(
                False, "The selected run changed. The next depth step was not started."
            )
            return
        self._stage = stage
        workflow, action = (
            ("preprocessing", "depth_measurements")
            if stage == 0
            else ("setup", "depth")
        )
        self.stage_changed.emit(
            "Preparing depth measurements…" if stage == 0 else "Computing depth map…"
        )
        toolbar = self.window.toolbar_builder
        try:
            process = self.window.process_runner.start_workflow_process(
                self._commands[stage],
                toolbar.play_action,
                toolbar.stop_action,
                cwd=Path.cwd(),
                workflow=workflow,
                actions=[action],
                config_path=self._config,
            )
        except Exception as error:
            self._finish(False, f"Depth step could not start: {error}")
            return
        self.current_process = process
        if process is None:
            self._finish(False, "Depth map calculation cancelled before starting.")
            return
        # The native adapter can also route Stop directly to abort(). Keeping
        # both original tab handles current preserves the existing toolbar path.
        self.window.preprocessing_tab.process = process
        self.window.setup_tab.process = process
        process.finished.connect(
            lambda code, status: self._stage_finished(process, stage, code, status)
        )
        process.errorOccurred.connect(
            lambda error: self._process_error(process, workflow, error)
        )

    def _process_error(self, process, workflow: str, error) -> None:
        if (
            process is not self.current_process
            or error != QProcess.ProcessError.FailedToStart
        ):
            return
        text = process.errorString()
        runner = self.window.process_runner
        if runner.active.get(workflow, (None,))[0] is process:
            runner.active.pop(workflow, None)
        self.window.print_log(text)
        self.window.toolbar_builder.play_action.setEnabled(True)
        self.window.toolbar_builder.stop_action.setEnabled(False)
        self.window.toolbar_builder.stop_action.setVisible(False)
        # Reuse the original runner's error presentation where available.
        show_error = getattr(runner, "_show_error_dialog", None)
        if show_error is not None:
            show_error(workflow, -1, [text])
        self._finish(False, f"Depth step could not start: {text}")

    def _stage_finished(self, process, stage: int, code: int, status) -> None:
        if process is not self.current_process or not self._running:
            return
        self.current_process = None
        if self._cancelled or process.property("darsia_aborted"):
            self._finish(False, "Depth map calculation stopped.")
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            self._finish(False, f"Depth step failed (exit code {code}).")
        elif stage == 0:
            QTimer.singleShot(0, lambda: self._launch(1))
        else:
            self._finish(True, "Depth map saved.")

    def abort(self) -> None:
        if not self._running:
            return
        self._cancelled = True
        if self.current_process is not None:
            self.window.process_runner.abort_workflow_process(self.current_process)
        else:
            self._finish(False, "Depth map calculation stopped.")

    def _finish(self, success: bool, message: str) -> None:
        if not self._running:
            return
        self._running = False
        self.current_process = None
        self.busy_changed.emit(False)
        self.completed.emit(success, message)


@dataclass(frozen=True)
class SavedDepthMap:
    path: Path | None
    image: object | None
    status: str


def effective_depth_settings(config: Mapping, *, config_path=None) -> dict:
    """Use the numerical DepthConfig's storage aliases and relative-path rules."""
    from darsia.presets.workflows.config.utils import (
        _config_path,
        _with_storage_aliases,
    )

    depth = dict(_with_storage_aliases(dict(config.get("depth", {}))))
    data = _with_storage_aliases(dict(config.get("data", {})))
    for key in ("measurements", "depth_map"):
        if depth.get(key):
            depth[key] = _config_path(depth[key], config_path)
    if not depth.get("depth_map") and data.get("results"):
        depth["depth_map"] = (
            _config_path(data["results"], config_path)
            / "setup"
            / "depth"
            / "depth_map.npz"
        )
    return depth


def load_saved_depth_map(
    config: Mapping, *, config_path=None, reader=None
) -> SavedDepthMap:
    """Describe one already saved map using DarSIA's original image reader.

    This is an on-demand display range, not a new depth calculation. No path is
    guessed beyond DepthConfig's existing results/setup/depth default.
    """
    chosen = effective_depth_settings(config, config_path=config_path).get("depth_map")
    if not chosen:
        return SavedDepthMap(None, None, "No depth map configured.")
    path = Path(chosen)
    if not path.is_file():
        return SavedDepthMap(path, None, f"Depth map not saved yet: {path}")
    try:
        if reader is None:
            from darsia import imread

            reader = imread
        import numpy as np

        image = reader(path)
        array = np.asarray(image.img)
        finite = array[np.isfinite(array)]
        if not finite.size:
            return SavedDepthMap(
                path, None, f"Saved depth map has no finite values: {path}"
            )
        minimum, maximum = float(finite.min()), float(finite.max())
    except Exception as error:
        return SavedDepthMap(path, None, f"Saved depth map range unavailable: {error}")
    return SavedDepthMap(
        path, image, f"Saved depth map: {minimum:.6g}–{maximum:.6g} m · {path}"
    )


def saved_depth_map_status(config: Mapping, *, config_path=None, reader=None) -> str:
    return load_saved_depth_map(config, config_path=config_path, reader=reader).status


class NativeDepthWorkflow(QObject):
    """Bind the pane to the native window, keeping original save/run semantics.

    Only this native instance's save sync and intermediate Done presentation are
    adapted. The two numerical commands and original Error/Abort handling stay
    unchanged. Navigation alone never saves a config or starts a process.
    """

    def __init__(self, window):
        super().__init__(window)
        import sys

        self.window = window
        self.panel = DepthSetupPanel()
        self._play_help = window.toolbar_builder.play_action.toolTip()
        self._opening = False
        self._saving = False
        self._embedded_active = False
        self._displayed_context = None
        self._authored_measurements = None
        self._displayed_measurements = ""
        self.sequence = DepthWorkflowSequence(
            window, sys.executable, config_is_current=self._config_is_current
        )
        self.panel.compute_requested.connect(self.compute)
        self.sequence.busy_changed.connect(self._busy_changed)
        self.sequence.stage_changed.connect(self.panel.set_status)
        self.sequence.completed.connect(self._completed)
        factory = window.settings_factory
        original_sync = factory._sync_settings_inputs_to_config_dict

        def sync(*args, **kwargs):
            # The original hidden form must not overwrite the explicit Depth
            # values, including between its two commands when the runner saves.
            if self._saving or self.sequence.running:
                return None
            return original_sync(*args, **kwargs)

        sync.__wrapped__ = original_sync
        factory._sync_settings_inputs_to_config_dict = sync
        original_refresh = factory.refresh_current_view

        def refresh(*args, **kwargs):
            # Save refreshes the last generic form, even while it is hidden.
            # Its callbacks must not materialize unrelated defaults between
            # the two depth commands. The depth pane refreshes on completion.
            if self._saving or self.sequence.running:
                return None
            return original_refresh(*args, **kwargs)

        refresh.__wrapped__ = original_refresh
        factory.refresh_current_view = refresh
        runner = window.process_runner
        original_done = runner._show_done_dialog

        def done(workflow, actions, config_path):
            if (
                self.sequence.running
                and self.sequence._stage == 0
                and workflow == "preprocessing"
                and actions == ["depth_measurements"]
            ):
                return None
            return original_done(workflow, actions, config_path)

        done.__wrapped__ = original_done
        runner._show_done_dialog = done

    def _config_is_current(self, path) -> bool:
        from darsia.fluidflower.native_darsia_app import _same_file

        window = self.window
        return (
            window._config_series == window.phase_model_workspace.series_id
            and _same_file(window.config_file, path)
            and _same_file(window.run_combo.currentData(), path)
        )

    def show(self) -> None:
        """Retain the standalone adapter for callers outside the native route."""
        self._embedded_active = False
        if self.window.content_stack.indexOf(self.panel) < 0:
            self.window.content_stack.addWidget(self.panel)
        self._open(embedded=False)

    def set_embedded_active(self, active: bool) -> None:
        """Reuse the same pane within Rig Geometry; opening never computes."""
        self._embedded_active = bool(active)
        if self._embedded_active:
            self._open(embedded=True)

    def _context_key(self):
        window = self.window
        return (window.phase_model_workspace.series_id, window.config_file,
                id(window.config_dict), getattr(window, "_config_stamp", None))

    def _visible(self) -> bool:
        return self._embedded_active or self.window.content_stack.currentWidget() is self.panel

    def _open(self, *, embedded: bool) -> None:
        if self._opening:
            return
        window = self.window
        if self.sequence.running:
            if not embedded:
                window.content_stack.setCurrentWidget(self.panel)
            return
        self._opening = True
        try:
            # Native Rig Geometry has already settled its previous editor.
            # Its embedded pane must not resync an unrelated hidden form.
            if not embedded:
                window.settings_factory._sync_settings_inputs_to_config_dict()
            chosen = window.run_combo.currentData()
            if chosen and not self._config_is_current(chosen):
                if not window._load_run_config(str(chosen)):
                    chosen = None
            if not embedded:
                window.content_stack.setCurrentWidget(self.panel)
                window.selected_action, window.selected_checkbox_id = "setup", "depth"
            if chosen and self._config_is_current(chosen):
                if self._displayed_context != self._context_key():
                    self._display_configuration()
            else:
                self._displayed_context = None
                self.panel.set_configuration({})
                self.panel.set_saved_depth_map(
                    SavedDepthMap(None, None, "Choose a run with a configuration first.")
                )
            window.toolbar_builder.play_action.setToolTip(
                "Use Compute depth map in the Depth pane."
            )
        finally:
            self._opening = False

    def refresh_if_visible(self, *, reload_selected=False) -> None:
        window = self.window
        if (
            self._opening
            or self.sequence.running
            or not self._visible()
        ):
            return
        if reload_selected:
            self._open(embedded=self._embedded_active)
            return
        chosen = window.run_combo.currentData()
        if chosen and self._config_is_current(chosen):
            self._display_configuration()
        else:
            self._displayed_context = None
            self.panel.set_configuration({})
            self.panel.set_saved_depth_map(
                SavedDepthMap(None, None, "Choose a run with a configuration first.")
            )

    def _display_configuration(self) -> None:
        window = self.window
        self._authored_measurements = window.config_dict.get("depth", {}).get(
            "measurements"
        )
        self.panel.set_configuration(
            effective_depth_settings(window.config_dict, config_path=window.config_file)
        )
        self._displayed_measurements = self.panel.measurements.text()
        self.panel.set_saved_depth_map(
            load_saved_depth_map(window.config_dict, config_path=window.config_file)
        )
        self._displayed_context = self._context_key()

    def compute(self, settings: Mapping) -> None:
        window = self.window
        path = window.run_combo.currentData()
        if self.sequence.running or window.process_runner.active:
            self.panel.set_status("Another calculation is already running.")
            return
        if not path or not self._config_is_current(path):
            self.panel.set_status("Choose a run with a configuration first.")
            return
        from darsia.fluidflower.native_darsia_app import (
            _ask_config_reload,
            _file_stamp,
        )

        recorded = getattr(window, "_config_stamp", None)
        if (
            recorded is not None
            and recorded[0] == str(path)
            and _file_stamp(path) != recorded[1]
        ):
            decision = _ask_config_reload(window, str(path))
            if decision == "cancel":
                self.panel.set_status("Depth map calculation cancelled.")
                return
            if decision == "reload":
                window._load_run_config(str(path))
                self.refresh_if_visible()
                self.panel.set_status(
                    "Configuration reloaded. Review its depth settings, then Compute depth map."
                )
                return
        allowed = {"measurements_mode", "measurements", "constant_depth"}
        if set(settings) - allowed:
            raise ValueError("Unexpected depth settings")
        settings = dict(settings)
        if (
            self._authored_measurements
            and settings.get("measurements") == self._displayed_measurements
        ):
            # Showing the effective location is not a request to rewrite its
            # authored relative path or storage alias when computing a map.
            settings["measurements"] = self._authored_measurements
        self._saving = True
        try:
            window.config_dict.setdefault("depth", {}).update(dict(settings))
            window.settings_factory.save_settings()
        except Exception as error:
            window.print_log(f"Depth settings could not be saved: {error}")
            self.panel.set_status(f"Depth settings could not be saved: {error}")
            return
        finally:
            self._saving = False
        self.sequence.start(path)

    def keep_selection_disabled_if_running(self) -> None:
        if not self.sequence.running:
            return
        self.window.run_combo.setEnabled(False)
        self.window.phase_model_workspace.series_selector.setEnabled(False)
        self.window.home_page.table.setEnabled(False)

    def sync_play_help(self) -> None:
        self.window.toolbar_builder.play_action.setToolTip(
            "Use Compute depth map in the Depth pane."
            if self._visible()
            else self._play_help
        )

    def _busy_changed(self, busy: bool) -> None:
        window = self.window
        self.panel.set_running(busy)
        window.phase_model_workspace.series_selector.setEnabled(not busy)
        window.home_page.table.setEnabled(not busy)
        window._refresh_scope_display()
        self.keep_selection_disabled_if_running()
        window._update_run_action()

    def _completed(self, success: bool, message: str) -> None:
        if success and self._config_is_current(self.sequence._config):
            self._display_configuration()
        else:
            self.panel.set_status(message)
        window = self.window
        if window.content_stack.currentWidget() is self.panel:
            window.selected_action, window.selected_checkbox_id = "setup", "depth"
            window._select_row(("setup", "depth"))
            window.toolbar_builder.wizard_action.setEnabled(False)
            if not success:
                self.panel.set_status(message)
        window._update_run_action()
