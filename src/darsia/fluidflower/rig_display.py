"""Read-only physical-rig header for the native Setup page.

The Start wizard's explicit profile name is identity metadata; dimensions come
from an existing run/common configuration or from that existing catalogue entry.
No dimensions, profiles or corrections are created or inferred from numbers.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .series_paths import series_key


@dataclass(frozen=True)
class RigDisplay:
    line: str
    note: str = ""
    profile: str = ""
    physical_rig: str = ""
    rig_class: str = ""
    width: float | None = None
    height: float | None = None


def _read_toml(path: Path, *, limit=1024 * 1024) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Metadata file is too large: {path.name}")
    return tomllib.loads(raw.decode("utf-8"))


def _dimensions(values) -> tuple[float | None, float | None]:
    def positive(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None

    if isinstance(values, Mapping):
        return positive(values.get("width")), positive(values.get("height"))
    return positive(getattr(values, "width", None)), positive(
        getattr(values, "height", None)
    )


def read_rig_display(
    series_root: str | Path | None,
    series_id: str = "",
    *,
    loaded_config: Mapping | None = None,
    catalogue_loader=None,
) -> RigDisplay:
    """Read one explicit series manifest and, at most, its small common config.

    The caller must supply only a configuration already known to belong to this
    series when loaded_config is supplied. Missing physical identity is reported
    honestly. FF_AC -> Albus is the project owner's explicit historical identity, not a guess
    based on dimensions, and is never written back by this display helper.
    """
    if series_root is None:
        return RigDisplay("Rig: no series selected")
    root = Path(series_root)
    try:
        manifest = _read_toml(root / "series.toml", limit=64 * 1024)
    except (OSError, ValueError, UnicodeError) as error:
        return RigDisplay("Rig: identity not recorded", str(error))
    actual_id = str(manifest.get("series_id") or "")
    if series_id and actual_id:
        try:
            matches = series_key(actual_id) == series_key(series_id)
        except ValueError:
            matches = False
        if not matches:
            return RigDisplay("Rig: series identity does not match")
    rig_class = str(manifest.get("rig_class") or "")
    profile_value = manifest.get("rig_profile")
    profile = profile_value.strip() if isinstance(profile_value, str) else ""
    chosen_at_start = bool(profile)
    known_albus = not profile and actual_id == "FF_AC"
    if known_albus:
        profile = "FF AC"
    # These are existing Start choices, not a new physical-rig catalogue.
    from darsia.fluidflower.series_wizard import PHYSICAL_RIGS

    physical = {profile: name for name, profile in PHYSICAL_RIGS}.get(profile, "")
    notes = []
    if chosen_at_start:
        notes.append("Physical rig profile recorded by the Start choice.")
    elif known_albus:
        notes.append("FF_AC is Albus, as explicitly identified by the project owner.")
    else:
        notes.append("The physical rig choice is not recorded for this series.")
    config = loaded_config
    if config is None:
        paths = manifest.get("paths")
        common = paths.get("common_config") if isinstance(paths, Mapping) else None
        name = Path(str(common or "config/common.toml"))
        common_path = name if name.is_absolute() else root / name
        try:
            config = _read_toml(common_path)
        except (OSError, ValueError, UnicodeError):
            config = {}
        else:
            notes.append("Dimensions from the existing series common configuration.")
    else:
        notes.append("Dimensions from the selected run's existing configuration.")
    rig = config.get("rig", {}) if isinstance(config, Mapping) else {}
    width, height = _dimensions(rig)
    has_dimensions = isinstance(rig, Mapping) and ("width" in rig or "height" in rig)
    if profile and not has_dimensions:
        if catalogue_loader is None:
            from darsia.presets.workflows.config.catalogue import load_catalogue

            catalogue_loader = load_catalogue
        try:
            width, height = _dimensions(catalogue_loader("rig").get(profile))
        except (KeyError, OSError, ValueError):
            pass
        else:
            notes.append("Dimensions from the existing named rig profile.")
    details = [value for value in (profile, rig_class if not profile else "") if value]
    if width is not None and height is not None:
        details.append(f"{width:.2f} × {height:.2f} m")
    else:
        details.append("dimensions not recorded")
    label = physical or (profile if profile else "physical rig not recorded")
    line = f"Rig: {label} ({', '.join(details)})"
    if chosen_at_start:
        line += " — chosen at Start"
    return RigDisplay(
        line,
        " ".join(notes),
        profile,
        physical,
        rig_class,
        width,
        height,
    )


class NativeRigContext:
    """Native-instance-only Setup presentation of the Start rig choice."""

    def __init__(self, window):
        self.window = window
        self.display = RigDisplay("Rig: no series selected")
        factory = window.settings_factory
        render = factory._render_settings_tabs

        def render_settings(settings_by_section):
            view = window._last_settings_view
            if view and view[0] == "action" and view[1] in {"setup", "preprocessing"}:
                settings_by_section = {
                    key: value
                    for key, value in settings_by_section.items()
                    if key != "rig"
                }
            return render(settings_by_section)

        render_settings.__wrapped__ = render
        factory._render_settings_tabs = render_settings
        workspace = window.geometry_workspace
        for name in (
            "_load_rig_class",
            "_show_page",
            "_install_rig_frame_panel",
            "_create_advanced_page",
        ):
            original = getattr(workspace, name)

            def update(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                self.hide_geometry_questions()
                return result

            update.__wrapped__ = original
            setattr(workspace, name, update)
        self.hide_geometry_questions()

    def refresh(self, *, display=None) -> None:
        from darsia.fluidflower.native_darsia_app import _same_file

        provider = getattr(self, 'prepared_display_provider', None)
        if display is None and provider is not None:
            display = provider()
            if display is None and getattr(self.window, '_series_loading', False):
                return

        window = self.window
        workspace = window.phase_model_workspace
        series = workspace.series_id
        config = None
        if window._config_series == series and _same_file(
            window.config_file, window.run_combo.currentData()
        ):
            config = window.config_dict
        self.display = display if display is not None else read_rig_display(
            workspace.series_roots.get(series), series, loaded_config=config
        )
        window.rig_label.setText(self.display.line)
        window.rig_label.setToolTip(self.display.note)
        window.rig_label.setVisible(
            window._current_category in {"setup", "preprocessing", "geometry_facies", "image_selection"}
        )
        self.hide_geometry_questions()

    def hide_geometry_questions(self) -> None:
        from PySide6.QtWidgets import QLabel

        workspace = self.window.geometry_workspace
        workspace.rig_class.setEnabled(False)
        workspace.rig_class.hide()
        for label in workspace.findChildren(QLabel):
            if label.text() == "Rig class":
                label.hide()
        panels = (
            getattr(workspace.simple_view, "rig_panel", None),
            workspace.rig_panel,
        )
        for panel in panels:
            if panel is not None:
                self._prepare_panel(panel)
        page = getattr(workspace, "page", None)
        if page is not None:
            for name in ("width_m", "height_m"):
                field = getattr(page, name, None)
                if field is not None:
                    field.setEnabled(False)
                    field.setToolTip(
                        "Rig dimensions come from the existing series configuration."
                    )

    def _prepare_panel(self, panel) -> None:
        if not getattr(panel, "_native_rig_readonly", False):
            panel._native_rig_readonly = True
            original_hint = panel._update_class_hint
            original_frame = panel._show_frame

            def update_hint(*args, **kwargs):
                result = original_hint(*args, **kwargs)
                self._lock_panel(panel)
                return result

            def show_frame(*args, **kwargs):
                result = original_frame(*args, **kwargs)
                self._lock_panel(panel)
                return result

            update_hint.__wrapped__ = original_hint
            show_frame.__wrapped__ = original_frame
            panel._update_class_hint = update_hint
            panel._show_frame = show_frame
        self._lock_panel(panel)

    def _lock_panel(self, panel) -> None:
        from PySide6.QtCore import QSignalBlocker
        from PySide6.QtWidgets import QLabel

        reference = panel.reference()
        frame = panel.frame
        # Before the first crop exists the original panel has no frame and
        # shows zero sizes. Bind only missing UI values to the SAME reference's
        # existing preset/config dimensions. This never calls a save or a model.
        previous_loading = panel._loading
        panel._loading = True
        try:
            for field, attribute in (
                (panel.width_m, "width_m"),
                (panel.height_m, "height_m"),
            ):
                recorded = getattr(frame, attribute, None)
                value = getattr(reference, attribute, None)
                if recorded is None and value is not None:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if math.isfinite(float(value)) and float(value) > 0:
                            with QSignalBlocker(field):
                                field.setValue(float(value))
                field.setEnabled(False)
        finally:
            panel._loading = previous_loading
        panel.class_row.hide()
        panel.use_class_size.setEnabled(False)

        def hide_row(layout, target):
            if layout is None:
                return False
            if any(layout.itemAt(i).widget() is target for i in range(layout.count())):
                for i in range(layout.count()):
                    widget = layout.itemAt(i).widget()
                    if widget is not None:
                        widget.hide()
                return True
            return any(
                hide_row(layout.itemAt(i).layout(), target)
                for i in range(layout.count())
                if layout.itemAt(i).layout() is not None
            )

        hide_row(panel.width_m.parentWidget().layout(), panel.width_m)
        for label in panel.findChildren(QLabel):
            if label.text().startswith(
                "Mark the four inner corners of the rig and enter"
            ):
                label.setText(
                    label.text().replace(
                        " and enter its inner size.",
                        ". The existing series rig configuration supplies its dimensions.",
                    )
                )
        status = panel.frame_status.text()
        status = status.replace(
            " in a photograph and enter the inner width and height.",
            " in a photograph.",
        )
        status = status.replace(" Enter the rig's inner width and height.", "")
        if not panel.width_m.value() or not panel.height_m.value():
            status += " Rig dimensions are missing from the series configuration."
        panel.frame_status.setText(status)

    def prepare_wizard(self, dialog, *, extra_policy=None) -> None:
        """Hide the two repeated rig questions on this native dialog only.

        Its config was already filled by the explicit Start preset or by loading
        a registered run. Nothing is applied, saved or launched by this adapter.
        The remaining original pages, Finish and commands stay unchanged.
        """
        if getattr(dialog, "_native_rig_readonly", False):
            if extra_policy is not None:
                extra_policy(dialog)
            return
        if not hasattr(dialog, "_fill_rig_page"):
            return
        from PySide6.QtWidgets import QLabel
        from darsia.gui.ui.preprocessing_wizard import STEPS

        dialog._native_rig_readonly = True
        original_header = dialog._page_header
        original_goto = dialog._goto

        def readonly_rig(layout):
            label = QLabel(self.display.line)
            label.setWordWrap(True)
            label.setToolTip(self.display.note)
            layout.addWidget(label)
            note = QLabel("Uses the existing rig configuration chosen for this series.")
            note.setWordWrap(True)
            layout.addWidget(note)

        def header(layout, index):
            if STEPS[index][0] not in {"series", "rig"}:
                return original_header(layout, index)

        def goto(index):
            result = original_goto(index)
            if STEPS[dialog._index][0] in {"series", "rig"}:
                dialog._title_label.setText("Series rig")
                dialog._subtitle_label.setText("The existing rig choice is read-only.")
            return result

        # Old preset checkbox widgets may be deleted when the summary is built.
        # Clear their registry and disable live copying before that occurs.
        dialog._piece_checkboxes = {}
        dialog._selected_preset = None
        dialog._fill_series_page = readonly_rig
        dialog._fill_rig_page = readonly_rig
        dialog._page_header = header
        dialog._goto = goto
        for index, step in enumerate(STEPS):
            if step[0] in {"series", "rig"}:
                dialog._built_pages.discard(index)
        if extra_policy is not None:
            extra_policy(dialog)
