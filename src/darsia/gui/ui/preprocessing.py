"""Preprocessing workflow tab for DarSIA GUI: protocols, depth measurements,
crop correction.

These are preparation steps that run ahead of, and independently from, the
main Setup tab's routines (depth map, segmentation, facies, rig).
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

CONFLICT_PREVIEW_LIMIT = 8


def resolve_overwrite_conflicts(
    main_window, conflicts: list[Path], noun: str
) -> bool | None:
    """Ask the user what to do about output files that already exist.

    Shared by every preprocessing routine that can overwrite an existing file
    (protocols, depth measurements, ...), so they all ask the same way.

    Parameters
    ----------
    conflicts : list[Path]
        Files that would be overwritten. An empty list means nothing to ask.
    noun : str
        What to call the files in the dialog (e.g. "Protocol files").

    Returns
    -------
        False if nothing would be overwritten, True if the user approved
        overwriting (pass ``--force``), or None if they cancelled — in which
        case the caller must not run preprocessing.
    """
    if not conflicts:
        return False

    preview_text = "\n".join(str(p) for p in conflicts[:CONFLICT_PREVIEW_LIMIT])
    if len(conflicts) > CONFLICT_PREVIEW_LIMIT:
        preview_text += f"\n... and {len(conflicts) - CONFLICT_PREVIEW_LIMIT} more."

    result = QMessageBox.question(
        main_window,
        f"{noun} exist",
        f"{noun} already exist:\n\n{preview_text}\n\nOverwrite?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if result != QMessageBox.Yes:
        main_window.print_log(
            "Preprocessing cancelled: user chose not to overwrite existing files."
        )
        return None
    return True


def resolve_protocol_conflicts(main_window, config_path: Path) -> bool | None:
    """Ask the user what to do about protocol files that already exist.

    Shared by the Preprocessing tab and the preprocessing wizard, so both
    answer the question the same way. See :func:`resolve_overwrite_conflicts`
    for the return value.
    """
    try:
        from darsia.presets.workflows.setup.setup_protocols import (
            preview_protocol_setup_conflicts,
        )

        conflicts = preview_protocol_setup_conflicts([config_path])
    except Exception as e:
        main_window.print_log(f"Error checking protocol conflicts: {str(e)}")
        return None
    return resolve_overwrite_conflicts(main_window, conflicts, "Protocol files")


def resolve_depth_measurements_conflict(main_window, config_path: Path) -> bool | None:
    """Ask the user what to do if the depth-measurements target already exists.

    No-op (returns False) unless [depth].measurements_mode is 'constant' — see
    :func:`preview_depth_measurements_conflict`.
    """
    try:
        from darsia.presets.workflows.setup.setup_depth import (
            preview_depth_measurements_conflict,
        )

        conflicts = preview_depth_measurements_conflict([config_path])
    except Exception as e:
        main_window.print_log(f"Error checking depth-measurements conflicts: {str(e)}")
        return None
    return resolve_overwrite_conflicts(
        main_window, conflicts, "Depth measurements file"
    )


class PreprocessingTab:
    """Manages the preprocessing tab UI and workflow execution."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.process = None

    def on_run_clicked(self):
        """Handle run button click."""
        self.run_preprocessing()

    def on_abort_clicked(self):
        """Handle abort button click."""
        if self.process is not None:
            self.main_window.process_runner.abort_workflow_process(self.process)

    def run_preprocessing(self):
        """Run preprocessing workflow based on selected sidebar item."""
        config_file = self.main_window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            self.main_window.print_log("Please select a config file first.")
            return

        selected_id = self.main_window.selected_checkbox_id
        if not selected_id:
            self.main_window.print_log("Please select an option in the sidebar.")
            return

        # Sync GUI widgets to config_dict to read current show_plots setting
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()
        show_plots = bool(
            self.main_window.settings_factory.get_value(
                self.main_window.config_dict, "options.preprocessing.show_plots"
            )
        )

        # Build options dictionary matching the CLI interface
        options = {
            "all": selected_id == "all",
            "protocols": selected_id == "protocols",
            "depth_measurements": selected_id == "depth_measurements",
            "crop": selected_id == "crop",
            "show": show_plots,
            "force": False,
        }

        self.main_window.print_log(
            """Starting preprocessing with options: """
            f"""{[k for k, v in options.items() if v and k != "force"]}"""
        )

        # Check for output-file conflicts and ask user if overwrite is needed
        config_paths = [Path(config_file)]
        if options["all"] or options["protocols"]:
            decision = resolve_protocol_conflicts(self.main_window, config_paths[0])
            if decision is None:
                return
            options["force"] = decision
        if options["all"] or options["depth_measurements"]:
            decision = resolve_depth_measurements_conflict(
                self.main_window, config_paths[0]
            )
            if decision is None:
                return
            options["force"] = options["force"] or decision

        # Build command-line arguments for subprocess
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_preprocessing",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if options["all"]:
            argv.append("--all")
        if options["protocols"]:
            argv.append("--protocol")
        if options["depth_measurements"]:
            argv.append("--depth-measurements")
        if options["crop"]:
            argv.append("--crop")
        if options["force"]:
            argv.append("--force")
        if options["show"]:
            argv.append("--show")

        # Launch workflow in a separate process
        play_action = self.main_window.toolbar_builder.play_action
        stop_action = self.main_window.toolbar_builder.stop_action

        self.process = self.main_window.process_runner.start_workflow_process(
            argv,
            play_action,
            stop_action,
            cwd=Path.cwd(),
            workflow="preprocessing",
            actions=[selected_id],
            config_path=config_paths[0],
        )

    def sidebar_items(self):
        """Return sidebar data structure for Preprocessing category."""
        from .help_text import get_help_text

        return [
            (
                "Preparation",
                [
                    (
                        "Protocols",
                        "protocols",
                        "fa5s.circle",
                        get_help_text("preprocessing", "protocols", "Protocols"),
                    ),
                    (
                        "Depth measurements",
                        "depth_measurements",
                        "fa5s.circle",
                        get_help_text(
                            "preprocessing",
                            "depth_measurements",
                            "Depth measurements",
                        ),
                    ),
                    (
                        "Crop correction",
                        "crop",
                        "fa5s.circle",
                        get_help_text("preprocessing", "crop", "Crop correction"),
                    ),
                ],
            ),
        ]
