from pathlib import Path

import psutil
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QTabWidget,
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
from .settings import SettingsFactory, unwrap_composite_widget
from .setup import SetupTab
from .theme import apply_theme
from .theme import set_theme as save_theme
from .toolbar import ToolbarBuilder
from .utils_tab import UtilsTab


class MainWindow(QMainWindow):
    """The main class containing the window and the relevant methods for the visualization."""

    log_message = Signal(str)

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

        # Set up the menu bar
        self.menu_builder = MenuBuilder(self)
        self.menu_builder.build()

        # Set up the toolbar
        self.toolbar_builder = ToolbarBuilder(self, self.menu_builder)
        self.toolbar_builder.build()

        # Setting up the three upper layouts
        upper_container = QWidget()
        upper_layout = QVBoxLayout(upper_container)
        upper_container.setFixedHeight(100)

        upper_mid_container = QWidget()
        upper_mid_layout = QVBoxLayout(upper_mid_container)

        upper_right_container = QWidget()
        upper_right_layout = QVBoxLayout(upper_right_container)

        # Setting up the left upper layout
        upper_layout.addWidget(QLabel("Loaded config:"))
        self.config_path_label = QLabel("No config loaded.")
        self.config_path_label.setWordWrap(True)
        upper_layout.addWidget(self.config_path_label)

        # Storage for file/folder chooser widgets used by settings.py's
        # FileDialogHelper (e.g. depth.measurements, facies.props).
        self.chosen_files = {}

        # Add stretch to push the config label to the top
        upper_layout.addStretch()

        # Setting up the middle upper layout with tabs
        tabs = QTabWidget()

        # Initialize tab managers
        self.setup_tab = SetupTab(self)
        self.calibration_tab = CalibrationTab(self)
        self.analysis_tab = AnalysisTab(self)
        self.helper_tab = HelperTab(self)
        self.comparison_tab = ComparisonTab(self)
        self.utils_tab = UtilsTab(self)

        # Add tabs
        tabs.addTab(self.setup_tab.create_tab(), "Setup")
        tabs.addTab(self.calibration_tab.create_tab(), "Calibration")
        tabs.addTab(self.analysis_tab.create_tab(), "Analysis")
        tabs.addTab(self.helper_tab.create_tab(), "Helper")
        tabs.addTab(self.comparison_tab.create_tab(), "Comparison")
        tabs.addTab(self.utils_tab.create_tab(), "Utils")

        upper_mid_layout.addWidget(tabs)

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

        log_scroll_area = QScrollArea()
        log_scroll_area.setWidget(log_container)
        log_scroll_area.setWidgetResizable(True)

        # Splitter for the upper half of the GUI
        upper_splitter = QSplitter(Qt.Horizontal)
        upper_splitter.addWidget(upper_mid_container)
        upper_splitter.addWidget(upper_right_container)
        upper_splitter.setStretchFactor(0, 1)  # left panel: 1/7 of space
        upper_splitter.setStretchFactor(1, 7)  # right panel: 6/7 of space

        # Vertical splitter between the log-window and the rest of the GUI
        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(upper_splitter)
        content_splitter.addWidget(log_scroll_area)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)

        # Create central widget with all components
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        self.setCentralWidget(main_container)
        main_layout.addWidget(upper_container)
        main_layout.addWidget(content_splitter)

        self.showMaximized()

        # Display welcome message
        self.print_log("No config loaded. Use:")
        self.print_log("- <b><i>File > New</i></b> to start a new config, ")
        self.print_log(
            "- <b><i>File > Open Config...</i></b> to open an existing one, or "
        )
        self.print_log(
            "- <b><i>File > Open Recent</i></b> to open a recently-used config."
        )

    def get_checked_checkbox_ids(self, checkboxes):
        """
        Function to get the ids of checked checkboxes from a list of (id, checkbox) tuples.
        """
        checked_ids = []
        for checkbox_id, checkbox in checkboxes:
            if checkbox.isChecked():
                checked_ids.append(checkbox_id)
        return checked_ids

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

    def start_workflow_process(self, argv, run_button, abort_button, cwd=None):
        """Launch argv as a QProcess, streaming merged stdout/stderr to the log.
        Disables run_button and shows/enables abort_button while running; restores
        button state and logs completion/abort/error when the process finishes.
        Returns the QProcess (caller must keep a reference alive, e.g. on the tab,
        and call abort_workflow_process(process) to abort it).
        """
        process = QProcess(self)
        process.setProgram(argv[0])
        process.setArguments(argv[1:])
        if cwd:
            process.setWorkingDirectory(str(cwd))
        process.setProcessChannelMode(QProcess.MergedChannels)

        def handle_output():
            data = bytes(process.readAllStandardOutput()).decode(errors="replace")
            for line in data.splitlines():
                if line:
                    self.print_log(line)

        def handle_finished(exit_code, exit_status):
            run_button.setEnabled(True)
            abort_button.setVisible(False)
            abort_button.setEnabled(False)
            if exit_status == QProcess.CrashExit:
                self.print_log("Process aborted.")
            elif exit_code != 0:
                self.print_log(f"Process exited with code {exit_code}.")
            else:
                self.print_log("Completed successfully!")

        process.readyReadStandardOutput.connect(handle_output)
        process.finished.connect(handle_finished)
        run_button.setEnabled(False)
        abort_button.setVisible(True)
        abort_button.setEnabled(True)
        process.start()
        return process

    def abort_workflow_process(self, process):
        """Abort a process started via start_workflow_process, killing its whole tree."""
        if process is None or process.state() == QProcess.NotRunning:
            return
        self._kill_process_tree(process.processId())

    @staticmethod
    def _kill_process_tree(pid):
        """Best-effort kill of a process and all its descendants (children, grandchildren)."""
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
