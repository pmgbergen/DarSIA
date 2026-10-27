"""Analysis workflow tab for DarSIA GUI."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

STEPS_USING_MASS_CALIBRATION = (
    "mass",
    "volume",
    "segmentation",
    "fingers",
    "thresholding",
)


def _confirm_outdated_mass_calibration(main_window, config_file: str, selected_id: str):
    """Ask the user whether to continue if the mass calibration is outdated.

    The mass calibration is outdated if the color paths were recalibrated after it
    (e.g. a different number of key colors for some label).

    Returns
    -------
        True if the analysis may start, False if the user cancelled.

    """
    if selected_id not in STEPS_USING_MASS_CALIBRATION:
        return True
    try:
        from darsia.presets.workflows.analysis.analysis_context import (
            infer_require_color_to_mass_from_config,
        )
        from darsia.presets.workflows.calibration.consistency import (
            outdated_mass_calibration,
        )

        path = Path(config_file).resolve()
        if not infer_require_color_to_mass_from_config(
            path,
            include_segmentation=selected_id == "segmentation",
            include_fingers=selected_id == "fingers",
            include_thresholding=selected_id == "thresholding",
            include_mass=selected_id == "mass",
            include_volume=selected_id == "volume",
        ):
            return True
        inconsistencies = outdated_mass_calibration(path)
    except Exception as error:  # a check must never block the analysis
        main_window.print_log(f"Skipped the mass calibration check: {error}")
        return True
    if len(inconsistencies) == 0:
        return True

    details = "\n".join(
        f"  label {label}: {message}"
        for label, message in sorted(inconsistencies.items())
    )
    box = QMessageBox(main_window)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Mass calibration outdated")
    box.setText(
        "The color paths have been recalibrated after the last mass calibration."
    )
    box.setInformativeText(
        "The analysis would use the color paths stored with the mass calibration, "
        "not the current ones. It is recommended to recalibrate the mass "
        "calibration first.\n\nAffected labels:\n" + details + "\n\nContinue anyway?"
    )
    continue_button = box.addButton("Continue anyway", QMessageBox.AcceptRole)
    cancel_button = box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(cancel_button)
    box.exec()
    if box.clickedButton() is not continue_button:
        main_window.print_log(
            "Analysis cancelled: recalibrate the mass calibration for the current "
            "color paths."
        )
        return False
    return True


class AnalysisTab:
    """Manages the analysis tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_analysis()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_analysis(self):
        """Run analysis workflow based on selected sidebar item."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        selected_id = self.main_window.selected_checkbox_id
        if not selected_id:
            self.main_window.print_log("Please select an option in the sidebar.")
            return

        if not _confirm_outdated_mass_calibration(
            self.main_window, config_file, selected_id
        ):
            return

        # Sync GUI widgets to config_dict before reading live values
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()
        show_plots = bool(
            self.main_window.settings_factory.get_value(
                self.main_window.config_dict, "options.analysis.show_plots"
            )
        )
        stream_preview = bool(
            self.main_window.settings_factory.get_value(
                self.main_window.config_dict, "options.analysis.stream_preview"
            )
        )

        # Build options dictionary matching the CLI interface
        options = {
            "cropping": selected_id == "cropping",
            "segmentation": selected_id == "segmentation",
            "fingers": selected_id == "fingers",
            "mass": selected_id == "mass",
            "volume": selected_id == "volume",
            "thresholding": selected_id == "thresholding",
            "show": show_plots,
        }

        self.main_window.print_log(
            f"Starting analysis with options: {[k for k, v in options.items() if v]}"
        )

        # Build command-line arguments for subprocess
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_analysis",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if options["cropping"]:
            argv.append("--cropping")
        if options["segmentation"]:
            argv.append("--segmentation")
        if options["fingers"]:
            argv.append("--fingers")
        if options["mass"]:
            argv.append("--mass")
        if options["volume"]:
            argv.append("--volume")
        if options["thresholding"]:
            argv.append("--thresholding")
        if options["show"]:
            argv.append("--show")
        argv.append("--progress")

        cache_dir = None
        if stream_preview:
            cache_dir = self.main_window.streaming_panel.prepare_cache_dir()
            argv += ["--stream", "--stream-cache-dir", str(cache_dir)]
        self.main_window.streaming_panel.reset_for_run(stream_preview, cache_dir)
        self.main_window.reset_batch_progress()

        def _on_progress_line(line):
            self.main_window.update_batch_progress(line)
            if stream_preview:
                self.main_window.streaming_panel.handle_progress_line(line)

        # Launch workflow in a separate process
        play_action = self.main_window.toolbar_builder.play_action
        stop_action = self.main_window.toolbar_builder.stop_action
        self.process = self.main_window.process_runner.start_workflow_process(
            argv,
            play_action,
            stop_action,
            cwd=Path.cwd(),
            workflow="analysis",
            actions=[selected_id],
            config_path=Path(config_file),
            on_stream_line=(
                self.main_window.streaming_panel.handle_stream_line
                if stream_preview
                else None
            ),
            on_progress_line=_on_progress_line,
        )

    def sidebar_items(self):
        """Return sidebar data structure for Analysis category."""
        from .help_text import get_help_text

        return [
            (
                "Analysis steps",
                [
                    (
                        "Cropping",
                        "cropping",
                        "fa5s.circle",
                        get_help_text("analysis", "cropping", "Cropping"),
                    ),
                    (
                        "Mass",
                        "mass",
                        "fa5s.circle",
                        get_help_text("analysis", "mass", "Mass"),
                    ),
                    (
                        "Segmentation",
                        "segmentation",
                        "fa5s.circle",
                        get_help_text("analysis", "segmentation", "Segmentation"),
                    ),
                    (
                        "Fingers",
                        "fingers",
                        "fa5s.circle",
                        get_help_text("analysis", "fingers", "Fingers"),
                    ),
                    (
                        "Thresholding",
                        "thresholding",
                        "fa5s.circle",
                        get_help_text("analysis", "thresholding", "Thresholding"),
                    ),
                    (
                        "Volume",
                        "volume",
                        "fa5s.circle",
                        get_help_text("analysis", "volume", "Volume"),
                    ),
                ],
            ),
        ]
