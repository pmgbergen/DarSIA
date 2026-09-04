import os
from pathlib import Path

import psutil
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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
from .theme import apply_theme
from .theme import set_theme as save_theme
from .toolbar import ToolbarBuilder
from .utils_tab import UtilsTab


class MainWindow(QMainWindow):
    """The main class containing the window and the relevant methods for the visualization."""

    log_message = Signal(str)

    _SIDEBAR_WIDTH_SETTINGS_KEY = "ui/sidebar_width"
    _DEFAULT_SIDEBAR_WIDTH = 120

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

        # Create logging container with its own scroll area
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_label = QLabel("Logging:")
        log_layout.addWidget(log_label)

        # Add a text edit for logging output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.log_scroll_area = QScrollArea()
        self.log_scroll_area.setWidget(log_container)
        self.log_scroll_area.setWidgetResizable(True)

        # Vertical splitter between the log-window and settings/tabs
        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(upper_right_container)
        content_splitter.addWidget(self.log_scroll_area)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)

        # Horizontal splitter: sidebar (left) and settings+log column (right)
        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.addWidget(upper_mid_container)
        root_splitter.addWidget(content_splitter)
        root_splitter.setStretchFactor(0, 1)  # sidebar: 1/7 of space
        root_splitter.setStretchFactor(1, 7)  # settings+log: 6/7 of space

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

        # Create central widget with all components
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        self.setCentralWidget(main_container)
        main_layout.addWidget(root_splitter)

        self._init_dashboard()

        self.showMaximized()

        # Display welcome message
        self.welcome_message()

    def _init_dashboard(self):
        """Set up the status-bar dashboard (CPU / memory / process status)."""
        self.dashboard_cpu_label = QLabel()
        self.dashboard_memory_label = QLabel()
        self.dashboard_process_label = QLabel()
        status_bar = self.statusBar()
        status_bar.addPermanentWidget(self.dashboard_cpu_label)
        status_bar.addPermanentWidget(self.dashboard_memory_label)
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

    def _on_open_full_config(self):
        """Handle opening full config: deselect sidebar and show all settings."""
        self.sidebar.deselect_all()
        self.settings_factory.display_full_settings()

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

    def toggle_logging(self, visible: bool):
        """Show or hide the logging panel (View > Show Logging / Ctrl+L)."""
        self.log_scroll_area.setVisible(visible)

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
        """Slot that appends text to log window and prints to console."""
        self.log_text.append(text)
        print(text)
