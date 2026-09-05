import os
from pathlib import Path

import psutil
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from darsia.presets.workflows.analysis.progress import try_decode_progress_line
from darsia.presets.workflows.results_folder import has_workflow_output

from .about_dialog import AboutDialog
from .analysis import AnalysisTab
from .calibration import CalibrationTab
from .comparison import ComparisonTab
from .config_controller import ConfigController
from .helper import HelperTab
from .menu import MenuBuilder
from .process_runner import ProcessRunner
from .settings import SettingsFactory
from .setup import SetupTab
from .sidebar import Sidebar
from .streaming import StreamingPanel
from .theme import apply_theme
from .theme import set_theme as save_theme
from .theme import theme_signal
from .toolbar import ToolbarBuilder
from .utils_tab import UtilsTab

_BATCH_DURATION_WINDOW = 5


def _format_duration(seconds) -> str:
    """Format seconds as H:MM:SS / M:SS, or 'n/a' for missing/invalid input."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 0:
        return "n/a"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class MainWindow(QMainWindow):
    """The main class containing the window and the relevant methods for the visualization."""

    log_message = Signal(str)

    _SIDEBAR_WIDTH_SETTINGS_KEY = "ui/sidebar_width"
    _DEFAULT_SIDEBAR_WIDTH = 120
    _LOG_VISIBLE_SETTINGS_KEY = "ui/log_visible"
    _VIEW_VISIBLE_SETTINGS_KEY = "ui/view_visible"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DarSIA")

        # Connect log_message signal to append_log slot for thread-safe logging
        self.log_message.connect(self._append_log)

        # Set window icon
        logo_path = (
            Path(__file__).parent
            / "../../presets/workflows/interface/DarSIA_Horisontal_Positiv_part.png"
        )
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # Initialize core state needed by menu/toolbar builders
        self.config_file = ""
        self.config_dict = {}
        self.settings_inputs = {}
        self._last_settings_view = None

        # Initialize settings container and layout before settings factory and menu builder
        self.settings_container = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)

        # Initialize settings factory before menu builder (which uses it)
        self.settings_factory = SettingsFactory(self)

        # Initialize config controller before menu builder (which uses it)
        self.config_controller = ConfigController(self)

        # Initialize process runner
        self.process_runner = ProcessRunner(self)

        # View dock (right of the tabs): streaming preview + on-disk results
        # browsing (StreamingPanel's two modes). Created before the menu
        # builder since it wires a View-menu toggle action for this dock.
        self.streaming_panel = StreamingPanel(self)
        self.streaming_dock = QDockWidget("View", self)
        self.streaming_dock.setWidget(self.streaming_panel)
        # Closable only: no float/drag, so the fixed edge-toggle button (see
        # below) always points at where the panel actually is.
        self.streaming_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.streaming_dock)
        view_visible = QSettings().value(
            self._VIEW_VISIBLE_SETTINGS_KEY, True, type=bool
        )
        self.streaming_dock.setVisible(view_visible)

        # Logging dock (bottom), created before the menu builder for the same
        # reason as the View dock: it wires a View-menu toggle action.
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.addWidget(QLabel("Logging:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.log_dock = QDockWidget("Logging", self)
        self.log_dock.setWidget(log_container)
        # Closable only: no float/drag, so the fixed edge-toggle button (see
        # below) always points at where the panel actually is.
        self.log_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        log_visible = QSettings().value(self._LOG_VISIBLE_SETTINGS_KEY, True, type=bool)
        self.log_dock.setVisible(log_visible)

        # Set up the menu bar
        self.menu_builder = MenuBuilder(self)
        self.menu_builder.build()

        # Set up the toolbar
        self.toolbar_builder = ToolbarBuilder(self, self.menu_builder)
        self.toolbar_builder.build()

        # Hidden label to track current config path
        self.config_path_label = QLabel("No config loaded.")
        self.config_path_label.setWordWrap(True)

        # Storage for file/folder chooser widgets used by settings.py's
        # FileDialogHelper (e.g. depth.measurements, facies.props).
        self.chosen_files = {}

        # Setting up the layout containers
        upper_mid_container = QWidget()
        upper_mid_layout = QVBoxLayout(upper_mid_container)

        upper_right_container = QWidget()
        upper_right_layout = QVBoxLayout(upper_right_container)

        # Setting up the middle upper layout with sidebar
        # Initialize tab managers
        self.setup_tab = SetupTab(self)
        self.calibration_tab = CalibrationTab(self)
        self.analysis_tab = AnalysisTab(self)
        self.helper_tab = HelperTab(self)
        self.comparison_tab = ComparisonTab(self)
        self.utils_tab = UtilsTab(self)

        # Build action dispatch dict (needed by toolbar Play/Stop)
        self.action_dispatch = {
            "setup": self.setup_tab,
            "calibration": self.calibration_tab,
            "analysis": self.analysis_tab,
            "helper": self.helper_tab,
            "comparison": self.comparison_tab,
            "utils": self.utils_tab,
        }

        # Build sidebar from tab-manager declarative data
        sidebar_data = {
            "setup": ("Setup", "fa5s.cogs", self.setup_tab.sidebar_items()),
            "calibration": (
                "Calibration",
                "fa5s.balance-scale",
                self.calibration_tab.sidebar_items(),
            ),
            "analysis": (
                "Analysis",
                "fa5s.chart-line",
                self.analysis_tab.sidebar_items(),
            ),
            "helper": ("Helper", "fa5s.life-ring", self.helper_tab.sidebar_items()),
            "comparison": (
                "Comparison",
                "fa5s.exchange-alt",
                self.comparison_tab.sidebar_items(),
            ),
            "utils": ("Utils", "fa5s.toolbox", self.utils_tab.sidebar_items()),
        }
        self.sidebar = Sidebar(sidebar_data)
        self.sidebar.selection_changed.connect(self._on_sidebar_selection)
        upper_mid_layout.addWidget(self.sidebar)

        # Initialize selection state (will be set when sidebar row is clicked)
        self.selected_action = None
        self.selected_checkbox_id = None

        # Setting up the right upper layout
        # Add scroll area for settings
        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setWidget(self.settings_container)
        self.settings_scroll_area.setWidgetResizable(True)
        upper_right_layout.addWidget(self.settings_scroll_area)

        # Horizontal splitter: sidebar (left) and settings column (right).
        # Logging lives in self.log_dock (bottom dock, built earlier) instead
        # of a fixed-proportion splitter pane, so it no longer competes with
        # the settings form for space by default.
        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.addWidget(upper_mid_container)
        root_splitter.addWidget(upper_right_container)
        root_splitter.setStretchFactor(0, 1)  # sidebar: 1/7 of space
        root_splitter.setStretchFactor(1, 7)  # settings: 6/7 of space

        # Load persisted sidebar width or use default
        sidebar_width = QSettings().value(
            self._SIDEBAR_WIDTH_SETTINGS_KEY, self._DEFAULT_SIDEBAR_WIDTH
        )
        self.sidebar.setMinimumWidth(100)
        self.sidebar.setMaximumWidth(600)

        # Set initial splitter sizes deterministically
        root_splitter.setSizes([sidebar_width, 1000])

        # Persist sidebar width on user resize
        root_splitter.splitterMoved.connect(self._on_splitter_moved)
        self.root_splitter = root_splitter

        # Create central widget with all components. Outer layout is
        # horizontal so the streaming edge-handle can span the *entire*
        # window height on the right, edge to edge (matching the log
        # edge-handle spanning the entire window width along the bottom);
        # everything else (content + log bar) lives in a left column next to it.
        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(main_container)

        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(0)
        left_column_layout.addWidget(root_splitter, 1)

        # Thin edge handle to reveal/hide the Logging dock, spanning the
        # full width of the left column along the bottom.
        self.log_toggle_button = QPushButton()
        self.log_toggle_button.setFixedHeight(14)
        self.log_toggle_button.setFlat(True)
        self.log_toggle_button.setCursor(Qt.PointingHandCursor)
        self.log_toggle_button.clicked.connect(
            self.menu_builder.show_logging_action.trigger
        )
        self._refresh_log_toggle_style()
        theme_signal.theme_changed.connect(self._refresh_log_toggle_style)
        self.log_dock.visibilityChanged.connect(self._sync_log_toggle_button)
        self._sync_log_toggle_button(self.log_dock.isVisible())
        left_column_layout.addWidget(self.log_toggle_button)

        main_layout.addWidget(left_column, 1)

        # Thin edge handle to reveal/hide the View dock, sitting
        # at the boundary between the main content and the (right-docked,
        # now closable-only) dock, spanning the full window height. Drives
        # the same toggleViewAction as Ctrl+P/View menu/toolbar, so all four
        # controls stay in sync for free.
        self.streaming_toggle_button = QPushButton("<")
        self.streaming_toggle_button.setFixedWidth(14)
        self.streaming_toggle_button.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding
        )
        self.streaming_toggle_button.setFlat(True)
        self.streaming_toggle_button.setCursor(Qt.PointingHandCursor)
        self.streaming_toggle_button.clicked.connect(
            self.menu_builder.streaming_toggle_action.trigger
        )
        self._refresh_streaming_toggle_style()
        theme_signal.theme_changed.connect(self._refresh_streaming_toggle_style)
        self.streaming_dock.visibilityChanged.connect(
            self._sync_streaming_toggle_button
        )
        self._sync_streaming_toggle_button(self.streaming_dock.isVisible())
        main_layout.addWidget(self.streaming_toggle_button)

        self._init_dashboard()

        self.showMaximized()

        # Display welcome message
        self.welcome_message()

    def _sync_streaming_toggle_button(self, visible):
        """Keep the edge-toggle glyph/tooltip in sync with the dock's actual
        visibility, whatever triggered the change (Ctrl+P, View menu,
        toolbar, this button, or the dock's own close button)."""
        self.streaming_toggle_button.setText(">" if visible else "<")
        self.streaming_toggle_button.setToolTip(
            "Hide View (Ctrl+P)" if visible else "Show View (Ctrl+P)"
        )
        QSettings().setValue(self._VIEW_VISIBLE_SETTINGS_KEY, visible)

    def _refresh_streaming_toggle_style(self):
        """Rebuild the edge-toggle button's palette-aware styling (theme-safe)."""
        pal = QApplication.instance().palette()
        bg_base = pal.color(QPalette.Button)
        is_dark_mode = pal.color(QPalette.Window).lightnessF() < 0.5
        bg_color = bg_base.lighter(130) if is_dark_mode else bg_base.darker(120)
        bg = bg_color.name()
        hover = bg_color.lighter(115).name()
        pressed = bg_color.darker(110).name()
        border = pal.color(QPalette.Mid).name()
        self.streaming_toggle_button.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; border: none; "
            f"border-left: 1px solid {border}; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:pressed {{ background-color: {pressed}; }}"
        )

    def _sync_log_toggle_button(self, visible):
        """Keep the log edge-toggle glyph/tooltip in sync with the dock's
        actual visibility, and persist it as the default for next launch."""
        self.log_toggle_button.setText("⌄" if visible else "⌃")
        self.log_toggle_button.setToolTip(
            "Hide Logging (Ctrl+L)" if visible else "Show Logging (Ctrl+L)"
        )
        QSettings().setValue(self._LOG_VISIBLE_SETTINGS_KEY, visible)

    def _refresh_log_toggle_style(self):
        """Rebuild the log edge-toggle button's palette-aware styling (theme-safe)."""
        pal = QApplication.instance().palette()
        bg_base = pal.color(QPalette.Button)
        is_dark_mode = pal.color(QPalette.Window).lightnessF() < 0.5
        bg_color = bg_base.lighter(130) if is_dark_mode else bg_base.darker(120)
        bg = bg_color.name()
        hover = bg_color.lighter(115).name()
        pressed = bg_color.darker(110).name()
        border = pal.color(QPalette.Mid).name()
        self.log_toggle_button.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; border: none; "
            f"border-top: 1px solid {border}; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:pressed {{ background-color: {pressed}; }}"
        )

    def _init_dashboard(self):
        """Set up the status-bar dashboard (CPU / memory / process status)."""
        self.dashboard_cpu_label = QLabel()
        self.dashboard_memory_label = QLabel()
        self.batch_progress_label = QLabel()
        self.dashboard_process_label = QLabel()
        self._batch_image_durations = []
        status_bar = self.statusBar()
        status_bar.addPermanentWidget(self.dashboard_cpu_label)
        status_bar.addPermanentWidget(self.dashboard_memory_label)
        status_bar.addPermanentWidget(self.batch_progress_label)
        status_bar.addPermanentWidget(self.dashboard_process_label)

        self._dashboard_timer = QTimer(self)
        self._dashboard_timer.timeout.connect(self._update_dashboard)
        self._dashboard_timer.start(1000)
        self._update_dashboard()

    def _update_dashboard(self):
        """Refresh the status-bar CPU/memory/process-status labels (polled 1/s)."""
        try:
            cpu_text = f"CPU: {psutil.cpu_percent(interval=None):.1f}%"
            memory_text = (
                f"Memory: {psutil.virtual_memory().percent:.1f}% system, "
                f"{psutil.Process(os.getpid()).memory_info().rss / (1024**2):.1f} MB GUI"
            )
        except Exception:
            cpu_text = "CPU: n/a"
            memory_text = "Memory: n/a"

        active = list(self.process_runner.active.values())
        if active:
            process, label = active[0]
            process_text = f"Workflow: running (pid={process.processId()}, {label})"
        else:
            process_text = "Workflow: idle"

        self.dashboard_cpu_label.setText(cpu_text)
        self.dashboard_memory_label.setText(memory_text)
        self.dashboard_process_label.setText(process_text)

    def reset_batch_progress(self):
        """Clear the status-bar batch-progress label; called at the start of
        every Analysis run."""
        self._batch_image_durations = []
        self.batch_progress_label.setText("")

    def update_batch_progress(self, line):
        """Update the status-bar batch-progress label from one progress-notify
        stdout line (minimal batch monitor: image count, elapsed time, and a
        rolling-average ETA — see darsia.presets.workflows.analysis.progress).
        """
        try:
            is_progress_line, event = try_decode_progress_line(line)
        except Exception:
            return
        if not is_progress_line or event is None:
            return

        event_type = event.get("event")
        step = event.get("step", "")
        image_total = event.get("image_total")

        if event_type == "step_start":
            self._batch_image_durations = []
            self.batch_progress_label.setText(f"Images: 0/{image_total} ({step})")
            return

        if event_type == "image_progress":
            image_index = event.get("image_index")
            duration = event.get("image_duration_s")
            if isinstance(duration, (int, float)):
                self._batch_image_durations.append(duration)
                self._batch_image_durations = self._batch_image_durations[
                    -_BATCH_DURATION_WINDOW:
                ]
            elapsed_text = _format_duration(event.get("step_elapsed_s"))
            text = (
                f"Images: {image_index}/{image_total} ({step}) — elapsed {elapsed_text}"
            )
            if (
                len(self._batch_image_durations) >= 2
                and isinstance(image_index, int)
                and isinstance(image_total, int)
            ):
                remaining = image_total - image_index
                if remaining > 0:
                    average = sum(self._batch_image_durations) / len(
                        self._batch_image_durations
                    )
                    text += f", ETA {_format_duration(average * remaining)}"
            self.batch_progress_label.setText(text)
            return

        if event_type == "step_complete":
            elapsed_text = _format_duration(event.get("step_elapsed_s"))
            self.batch_progress_label.setText(
                f"Images: {image_total}/{image_total} ({step}) — elapsed {elapsed_text}"
            )

    def _on_splitter_moved(self):
        """Save the sidebar width when splitter is moved."""
        sizes = self.root_splitter.sizes()
        if sizes:
            QSettings().setValue(self._SIDEBAR_WIDTH_SETTINGS_KEY, sizes[0])

    def welcome_message(self):
        """Display a welcome message in the log window."""
        self.print_log("Welcome to DarSIA!")
        self.print_log("Load a config file to get started, or create a new one.")
        self.print_log(
            "Use the sidebar on the left to navigate through the application."
        )
        self.print_log(
            "For help, visit the <a href='https://docs.darsia.xyz'>DarSIA documentation</a>."
        )

    def _on_sidebar_selection(self, action: str, checkbox_id: str):
        """Handle sidebar row selection: update state and auto-open settings."""
        self.selected_action = action
        self.selected_checkbox_id = checkbox_id
        self.settings_factory.display_settings(action, [checkbox_id])
        self.streaming_panel.refresh_results()

    def _on_open_full_config(self):
        """Handle opening full config: deselect sidebar and show all settings."""
        self.sidebar.deselect_all()
        self.settings_factory.display_full_settings()

    # (action, checkbox_id) -> the exact action-label string
    # results_folder.has_workflow_output expects, for the few cases that
    # aren't a mechanical underscore-to-space rename of the checkbox_id
    # (e.g. Calibration's "color" step is labeled "color embedding" there).
    _PROGRESS_ACTION_LABEL_OVERRIDES = {("calibration", "color"): "color embedding"}

    def action_label_for(self, category: str, checkbox_id: str) -> str:
        """Map a (category, checkbox_id) pair to the action-label string
        results_folder.py's resolvers expect (see _PROGRESS_ACTION_LABEL_OVERRIDES
        docstring above for why most are a mechanical rename)."""
        return self._PROGRESS_ACTION_LABEL_OVERRIDES.get(
            (category, checkbox_id), checkbox_id.replace("_", " ")
        )

    def refresh_sidebar_progress(self):
        """Update sidebar completion dots from on-disk workflow output.

        Best-effort and skipped without a loaded config. Coverage matches
        what results_folder.suggested_workflow_results_folder understands
        per category/step; a step outside that coverage (e.g. Setup's Crop
        correction, which has no known output folder at all) simply keeps
        its default, uninformative dot rather than showing a wrong one.
        Helper's items are inspection tools, not one-time pipeline steps, so
        they're intentionally left out of scope.
        """
        if not self.config_file or not Path(self.config_file).exists():
            return
        config_path = Path(self.config_file)
        tab_managers = {
            "setup": self.setup_tab,
            "calibration": self.calibration_tab,
            "analysis": self.analysis_tab,
            "comparison": self.comparison_tab,
            "utils": self.utils_tab,
        }
        for category, tab_manager in tab_managers.items():
            for _group_label, items in tab_manager.sidebar_items():
                for _label, checkbox_id, _icon, _help in items:
                    action_label = self.action_label_for(category, checkbox_id)
                    done = has_workflow_output(category, config_path, [action_label])
                    self.sidebar.set_item_state(
                        category, checkbox_id, "done" if done else "none"
                    )
        self.streaming_panel.refresh_results()

    def run_selected_workflow(self):
        """Run the currently selected sidebar workflow (Play button / Ctrl+Return)."""
        if self.selected_action is None:
            self.print_log("Select an item in the sidebar first.")
            return

        tab_manager = self.action_dispatch.get(self.selected_action)
        if tab_manager:
            tab_manager.on_run_clicked()

    def abort_selected_workflow(self):
        """Abort the currently running workflow (Stop button / Ctrl+Escape)."""
        if self.selected_action is None:
            self.print_log("No workflow running.")
            return

        tab_manager = self.action_dispatch.get(self.selected_action)
        if tab_manager:
            tab_manager.on_abort_clicked()

    def set_theme(self, mode: str):
        """Set the application theme (System/Light/Dark).

        Parameters
        ----------
        mode : str
            One of "System", "Light", or "Dark".
        """
        from PySide6.QtWidgets import QApplication

        apply_theme(QApplication.instance(), mode)
        save_theme(mode)
        self.print_log(f"Theme set to {mode}")

    def show_about_dialog(self):
        """Show the About dialog."""
        AboutDialog(self).exec()

    def print_log(self, text):
        """Emit log_message signal to append text to log window (thread-safe via Qt signal)."""
        self.log_message.emit(text)

    def _append_log(self, text):
        """Slot that appends text to log window and prints to console.

        Reveals the log dock on an error line if it's currently collapsed, so
        a failure is never silently missed behind a closed panel. A plain
        substring check rather than a structured severity level, since no
        call site currently tags log lines with one.
        """
        self.log_text.append(text)
        print(text)
        if "error" in text.lower() and not self.log_dock.isVisible():
            self.log_dock.show()

    def closeEvent(self, event):
        """Clean up the streaming preview cache directory on window close."""
        self.streaming_panel.cleanup()
        super().closeEvent(event)
