"""Utils workflow tab for DarSIA GUI."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from darsia.presets.workflows.utils.calibration_bundle import (
    preview_calibration_bundle_import_conflicts,
)
from darsia.presets.workflows.utils.utils_download import prepare_download_data

_PREVIEW_LIMIT = 8


class UtilsTab:
    """Manages the utils tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_utils()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_utils(self):
        """Run utils workflow based on selected sidebar item."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        selected_id = self.main_window.selected_checkbox_id
        if not selected_id:
            self.main_window.print_log("Please select an option in the sidebar.")
            return

        config_paths = [Path(config_file)]

        # Sync GUI widgets to config_dict before reading bundle paths, so
        # values edited via "Open Full Config" are picked up.
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()
        export_bundle = self.main_window.settings_factory.get_value(
            self.main_window.config_dict, "workflow_utils.export_calibration_bundle"
        )
        import_bundle = self.main_window.settings_factory.get_value(
            self.main_window.config_dict, "workflow_utils.import_calibration_bundle"
        )
        export_bundle = (export_bundle or "").strip() or None
        import_bundle = (import_bundle or "").strip() or None

        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_utils",
            "--config",
            str(config_paths[0].resolve()),
        ]

        if selected_id == "build_media":
            argv.append("--build-media")

        elif selected_id == "download_data":
            try:
                plan = prepare_download_data(config_paths)
            except Exception as e:
                self.main_window.print_log(f"Could not prepare download: {e}")
                return
            if not plan.image_paths:
                QMessageBox.information(
                    self.main_window,
                    "Download data",
                    "No files selected for download.",
                )
                return
            confirmed = QMessageBox.question(
                self.main_window,
                "Confirm data download",
                f"About to download {len(plan.image_paths)} files\n"
                f"Total size: {plan.total_size_string}\n"
                f"Destination: {plan.destination_dir}\n\nProceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirmed != QMessageBox.Yes:
                self.main_window.print_log("Download cancelled.")
                return
            argv.append("--download-data")

        elif selected_id == "export_calibration":
            argv.append("--export-calibration")
            if export_bundle:
                argv += ["--calibration-bundle", export_bundle]

        elif selected_id == "import_calibration":
            if not import_bundle:
                QMessageBox.critical(
                    self.main_window,
                    "Import calibration",
                    "Import calibration requires an input bundle zip path. Set "
                    "workflow_utils.import_calibration_bundle via Open Full Config.",
                )
                return
            try:
                conflicts = preview_calibration_bundle_import_conflicts(
                    config_paths, Path(import_bundle)
                )
            except Exception as e:
                self.main_window.print_log(f"Could not check import conflicts: {e}")
                return
            conflict_action = "error"
            if conflicts:
                preview_paths = conflicts[:_PREVIEW_LIMIT]
                preview_text = "\n".join(str(p) for p in preview_paths)
                remaining = len(conflicts) - _PREVIEW_LIMIT
                if remaining > 0:
                    preview_text += f"\n... and {remaining} more."
                box = QMessageBox(self.main_window)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle("Calibration import conflicts")
                box.setText(
                    "Some calibration files already exist and would be "
                    f"overwritten.\n\n{preview_text}"
                )
                yes_button = box.addButton("Overwrite all", QMessageBox.YesRole)
                no_button = box.addButton("Skip existing", QMessageBox.NoRole)
                box.addButton("Cancel", QMessageBox.RejectRole)
                box.exec()
                clicked = box.clickedButton()
                if clicked is yes_button:
                    conflict_action = "overwrite_all"
                elif clicked is no_button:
                    conflict_action = "skip_all"
                else:
                    self.main_window.print_log("Calibration import cancelled by user.")
                    return
            argv += [
                "--import-calibration",
                "--calibration-bundle",
                import_bundle,
                "--conflict-action",
                conflict_action,
            ]

        self.main_window.print_log(f"Starting utils: {selected_id}")

        play_action = self.main_window.toolbar_builder.play_action
        stop_action = self.main_window.toolbar_builder.stop_action
        self.process = self.main_window.process_runner.start_workflow_process(
            argv,
            play_action,
            stop_action,
            cwd=Path.cwd(),
            workflow="utils",
            actions=[selected_id],
            config_paths=config_paths,
        )

    def sidebar_items(self):
        """Return sidebar data structure for Utils category."""
        from .help_text import get_help_text

        return [
            (
                "Actions",
                [
                    (
                        "Build protocol-time media (MP4/GIF)",
                        "build_media",
                        "fa5s.circle",
                        get_help_text(
                            "utils",
                            "build_media",
                            "Build protocol-time media (MP4/GIF)",
                        ),
                    ),
                    (
                        "Download/cache data",
                        "download_data",
                        "fa5s.circle",
                        get_help_text("utils", "download_data", "Download/cache data"),
                    ),
                    (
                        "Export calibration",
                        "export_calibration",
                        "fa5s.circle",
                        get_help_text(
                            "utils", "export_calibration", "Export calibration"
                        ),
                    ),
                    (
                        "Import calibration",
                        "import_calibration",
                        "fa5s.circle",
                        get_help_text(
                            "utils", "import_calibration", "Import calibration"
                        ),
                    ),
                ],
            ),
        ]
