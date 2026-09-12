"""Guided setup wizard: a linear, explanatory view over the same config TOML.

The wizard holds no state of its own. Every field it shows is one of the standard
settings widgets bound to ``main_window.config_dict``, so setting something here is
identical to setting it in the expert-mode Settings tabs, or by editing the TOML by
hand — the three are just different speeds of the same editor. Only the
series-catalogue step is wizard-specific, and it writes plain config sections too.
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QMovie, QPalette, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from darsia.presets.workflows.config.catalogue import (
    PIECE_LABELS,
    SeriesCatalogue,
    load_catalogue,
)

from .schema.dataclass_introspection import get_section_fields
from .settings import unwrap_composite_widget
from .theme import muted_text_color, success_color, theme_signal

ASSETS_DIR = Path(__file__).parent / "assets" / "setup_wizard"
NO_SERIES = "None — start blank"

PROTOCOL_MODE_KEYS = (
    "protocols.imaging_mode",
    "protocols.injection_mode",
    "protocols.pressure_temperature_mode",
)

STEPS = [
    (
        "data",
        "Data & results",
        "Point DarSIA at your images, and say where results should go.",
        (
            "Add one folder per imaging phase — a run split into an injection "
            "folder and a rest folder is two entries here. The baseline image is "
            "the reference frame everything else is compared against; the results "
            "folder collects everything DarSIA produces for this run. Every later "
            "step builds on what you set here."
        ),
    ),
    (
        "series",
        "Rig & corrections",
        "Reuse a known rig setup instead of re-entering its geometry.",
        (
            "Experiments run on the same physical rig share their geometry, image "
            "corrections and depth measurements. Pick the matching series and those "
            "settings are copied into your config; uncheck anything you would rather "
            "set yourself. Illumination and colour corrections are not part of these "
            "presets yet — set those in the Corrections tab."
        ),
    ),
    (
        "protocols",
        "Protocols",
        "Tell DarSIA when each image was taken, and what happened when.",
        (
            "The protocol CSVs are the reference DarSIA reads: which image belongs to "
            "which point in time, when injection ran, and the pressure/temperature "
            "conditions. If you already wrote them, say so and they are left alone. "
            "Otherwise pick how they should be generated: read each image's own "
            "timestamp (EXIF or file time), or compute timestamps from a cadence you "
            "declare — useful when the camera fired every N seconds."
        ),
    ),
    (
        "review",
        "Review & finish",
        "Check what will be written, then save it to your config file.",
        (
            "Finishing writes everything above into the TOML config file. Nothing on "
            "disk changes until then — and you can still adjust any of it afterwards "
            "in the Settings tabs, or in the file itself."
        ),
    ),
]


def _build_illustration(step_id: str) -> QLabel | None:
    """Return a label showing this step's bundled image/GIF, or None if there is none.

    Drop ``<step_id>.gif`` / ``.png`` / ``.jpg`` into ``assets/setup_wizard/`` and it
    is picked up here automatically; no other wiring is needed.
    """
    for suffix in (".gif", ".png", ".jpg"):
        path = ASSETS_DIR / f"{step_id}{suffix}"
        if not path.exists():
            continue
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        if suffix == ".gif":
            movie = QMovie(str(path))
            label.setMovie(movie)
            movie.start()
            # Keep the movie alive for as long as the label it drives.
            label._darsia_movie = movie
            return label
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            continue
        if pixmap.width() > 560:
            pixmap = pixmap.scaledToWidth(560, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        return label
    return None


class _StepRail(QWidget):
    """Vertical, colour-coded step indicator: done / current / still to come."""

    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self._badges = []
        self._labels = []
        self._current = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(16)

        for index, step in enumerate(steps):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)

            badge = QLabel(str(index + 1))
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(26, 26)
            label = QLabel(step[1])
            label.setWordWrap(True)

            row_layout.addWidget(badge, alignment=Qt.AlignTop)
            row_layout.addWidget(label, stretch=1)
            layout.addWidget(row)

            self._badges.append(badge)
            self._labels.append(label)

        layout.addStretch(1)
        theme_signal.theme_changed.connect(lambda _mode: self._restyle())
        self._restyle()

    def set_current(self, index: int) -> None:
        self._current = index
        self._restyle()

    def _restyle(self) -> None:
        palette = self.palette()
        done = success_color(palette)
        muted = muted_text_color(palette)
        accent = palette.color(QPalette.Highlight)
        on_accent = palette.color(QPalette.HighlightedText)
        text = palette.color(QPalette.WindowText)
        window = palette.color(QPalette.Window)

        self.setStyleSheet(
            f"_StepRail {{ background-color: {palette.color(QPalette.Base).name()}; }}"
        )

        for index, (badge, label) in enumerate(zip(self._badges, self._labels)):
            if index < self._current:
                badge.setText("✓")
                fill, border, glyph = done, done, window
                label.setStyleSheet(f"color: {text.name()};")
            elif index == self._current:
                badge.setText(str(index + 1))
                fill, border, glyph = accent, accent, on_accent
                label.setStyleSheet(f"color: {text.name()}; font-weight: 600;")
            else:
                badge.setText(str(index + 1))
                fill, border, glyph = window, muted, muted
                label.setStyleSheet(f"color: {muted.name()};")

            badge.setStyleSheet(
                f"background-color: {fill.name()};"
                f"color: {glyph.name()};"
                f"border: 1px solid {border.name()};"
                "border-radius: 13px;"
                "font-weight: 600;"
            )


class SetupWizardDialog(QDialog):
    """Guided, step-by-step editor for the setup part of a run config."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Setup Wizard")
        self.setModal(True)
        self.resize(1060, 740)

        self._index = 0
        self._selected_preset = None
        self._piece_checkboxes: dict[str, QCheckBox] = {}
        self._saved_modes: dict[str, str] = {}
        self._built_pages: set[int] = set()
        self._run_depth_checkbox: QCheckBox | None = None
        self._catalogue = self._load_catalogue()

        # The wizard takes over the shared widget registry while it is open: flush
        # whatever the Settings tabs currently hold, then start from a clean slate so
        # our widgets are the ones a save reads back.
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()
        self.main_window.settings_inputs.clear()

        self._build_ui()
        self._goto(0)

    # ---------------------------------------------------------------- building

    def _load_catalogue(self) -> SeriesCatalogue:
        try:
            return load_catalogue("series")
        except Exception as exc:  # pragma: no cover - defensive, surfaced in the log
            self.main_window.print_log(f"Could not load series catalogue: {exc}")
            return SeriesCatalogue()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._rail = _StepRail(STEPS)
        self._rail.setFixedWidth(250)
        self._rail.setAutoFillBackground(True)
        outer.addWidget(self._rail)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        outer.addWidget(divider)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(28, 24, 28, 20)
        right_layout.setSpacing(6)

        self._title_label = QLabel()
        title_font = self._title_label.font()
        title_font.setPointSize(title_font.pointSize() + 5)
        title_font.setBold(True)
        self._title_label.setFont(title_font)

        self._subtitle_label = QLabel()
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setStyleSheet(
            f"color: {muted_text_color(self.palette()).name()};"
        )

        header_line = QFrame()
        header_line.setFrameShape(QFrame.HLine)
        header_line.setFrameShadow(QFrame.Sunken)

        right_layout.addWidget(self._title_label)
        right_layout.addWidget(self._subtitle_label)
        right_layout.addSpacing(6)
        right_layout.addWidget(header_line)
        right_layout.addSpacing(10)

        self._page_containers = []
        self._page_area = QVBoxLayout()
        right_layout.addLayout(self._page_area, stretch=1)

        for _step in STEPS:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(12)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            scroll.hide()

            self._page_area.addWidget(scroll)
            self._page_containers.append((scroll, page_layout))

        footer = QHBoxLayout()
        self._config_label = QLabel()
        self._config_label.setStyleSheet(
            f"color: {muted_text_color(self.palette()).name()};"
        )
        config_file = self.main_window.config_file
        if config_file:
            self._config_label.setText(f"Saves to {Path(config_file).name}")
            self._config_label.setToolTip(config_file)
        else:
            self._config_label.setText("No config file")

        self._back_button = QPushButton("‹  Back")
        self._back_button.clicked.connect(lambda: self._goto(self._index - 1))
        self._next_button = QPushButton("Next  ›")
        self._next_button.setDefault(True)
        self._next_button.clicked.connect(self._on_next)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        footer.addWidget(self._config_label, stretch=1)
        footer.addWidget(self._back_button)
        footer.addWidget(self._next_button)
        footer.addWidget(cancel_button)

        right_layout.addSpacing(10)
        right_layout.addLayout(footer)
        outer.addWidget(right, stretch=1)

    def _page_header(self, layout, step_index: int) -> None:
        """Add this step's explanatory paragraph and optional illustration."""
        explanation = QLabel(STEPS[step_index][3])
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        illustration = _build_illustration(STEPS[step_index][0])
        if illustration is not None:
            layout.addWidget(illustration)

        layout.addSpacing(4)

    def _build_section_form(self, layout, section: str) -> None:
        """Render a config section with the same widgets the Settings tabs use."""
        fields = get_section_fields(section)
        if not fields:
            layout.addWidget(QLabel(f"No schema found for [{section}]."))
            return
        holder = QWidget()
        form = QFormLayout(holder)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addWidget(holder)
        self.main_window.settings_factory.build_tab_form(form, fields)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _drop_settings_keys(self, prefix: str) -> None:
        """Forget widgets we are about to destroy, so a later save never reads them."""
        stale = [
            key
            for key in self.main_window.settings_inputs
            if key == prefix.rstrip(".") or key.startswith(prefix)
        ]
        for key in stale:
            del self.main_window.settings_inputs[key]

    # ------------------------------------------------------------------ pages

    def _ensure_page(self, index: int) -> None:
        _scroll, layout = self._page_containers[index]
        step_id = STEPS[index][0]

        # Data and series are built once; protocols and review are rebuilt on every
        # visit because they read state the earlier steps may just have changed.
        if step_id in ("data", "series") and index in self._built_pages:
            return

        if step_id == "protocols":
            self._drop_settings_keys("protocols.")
        self._clear_layout(layout)

        self._page_header(layout, index)
        if step_id == "data":
            self._build_section_form(layout, "data")
        elif step_id == "series":
            self._fill_series_page(layout)
        elif step_id == "protocols":
            self._fill_protocols_page(layout)
        else:
            self._fill_review_page(layout)
        layout.addStretch(1)

        self._built_pages.add(index)

    def _fill_series_page(self, layout) -> None:
        combo = QComboBox()
        combo.addItem(NO_SERIES)
        for name in self._catalogue.names():
            combo.addItem(name)

        row = QHBoxLayout()
        row.addWidget(QLabel("Experiment series:"))
        row.addWidget(combo, stretch=1)
        layout.addLayout(row)

        self._series_description = QLabel()
        self._series_description.setWordWrap(True)
        self._series_description.setStyleSheet(
            f"color: {muted_text_color(self.palette()).name()};"
        )
        layout.addWidget(self._series_description)

        self._pieces_box = QGroupBox("Copy these into my config")
        self._pieces_layout = QVBoxLayout(self._pieces_box)
        layout.addWidget(self._pieces_box)

        self._series_combo = combo
        combo.currentTextChanged.connect(self._on_series_changed)
        self._on_series_changed(combo.currentText())

    def _on_series_changed(self, name: str) -> None:
        self._clear_layout(self._pieces_layout)
        self._piece_checkboxes = {}

        if name == NO_SERIES or name not in self._catalogue.presets:
            self._selected_preset = None
            self._series_description.setText(
                "Nothing will be copied — fill in rig, corrections and depth yourself "
                "in the Settings tabs."
            )
            self._pieces_box.setEnabled(False)
            return

        preset = self._catalogue.get(name)
        self._selected_preset = preset
        self._series_description.setText(preset.description)
        self._pieces_box.setEnabled(True)

        available = preset.pieces()
        for piece, label in PIECE_LABELS.items():
            if piece not in available:
                continue
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            self._pieces_layout.addWidget(checkbox)
            self._piece_checkboxes[piece] = checkbox

    def _fill_protocols_page(self, layout) -> None:
        have_csv = QCheckBox(
            "I already have all my protocol CSV files — don't generate anything"
        )
        have_csv.setToolTip(
            "Sets every protocol to 'detailed', so setup leaves your files untouched."
        )
        layout.addWidget(have_csv)
        self._have_csv_checkbox = have_csv

        self._build_section_form(layout, "protocols")

        protocols = self.main_window.config_dict.get("protocols", {})
        already_detailed = all(
            protocols.get(key.split(".", 1)[1], "") == "detailed"
            for key in PROTOCOL_MODE_KEYS
        )
        have_csv.blockSignals(True)
        have_csv.setChecked(already_detailed)
        have_csv.blockSignals(False)
        have_csv.toggled.connect(self._on_have_csv_toggled)

    def _mode_combo(self, key: str) -> QComboBox | None:
        widget = unwrap_composite_widget(self.main_window.settings_inputs.get(key))
        return widget if isinstance(widget, QComboBox) else None

    def _on_have_csv_toggled(self, checked: bool) -> None:
        if checked:
            self._saved_modes = {}
            for key in PROTOCOL_MODE_KEYS:
                combo = self._mode_combo(key)
                if combo is None:
                    continue
                self._saved_modes[key] = combo.currentText()
                combo.setCurrentText("detailed")
            return

        for key, previous in self._saved_modes.items():
            combo = self._mode_combo(key)
            if combo is not None and previous:
                combo.setCurrentText(previous)
        self._saved_modes = {}

    def _fill_review_page(self, layout) -> None:
        config = self.main_window.config_dict
        data = config.get("data", {})
        protocols = config.get("protocols", {})

        folders = data.get("folders") or []
        if not folders and data.get("folder"):
            folders = [data["folder"]]

        summary = QFormLayout()
        summary.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        def add(label: str, value: str) -> None:
            field = QLabel(value or "—")
            field.setWordWrap(True)
            summary.addRow(f"{label}:", field)

        add("Config file", self.main_window.config_file)
        add("Data folders", "\n".join(str(folder) for folder in folders))
        add("Baseline image", str(data.get("baseline", "")))
        add("Results folder", str(data.get("results", "")))

        if self._selected_preset is None:
            add("Experiment series", "None — nothing copied")
        else:
            chosen = [
                PIECE_LABELS[piece]
                for piece, checkbox in self._piece_checkboxes.items()
                if checkbox.isChecked()
            ]
            add("Experiment series", self._series_combo.currentText())
            add("Copied pieces", "\n".join(chosen) if chosen else "None selected")

        add("Imaging protocol", str(protocols.get("imaging_mode", "exif")))
        add("Injection protocol", str(protocols.get("injection_mode", "constant")))
        add(
            "Pressure/temperature",
            str(protocols.get("pressure_temperature_mode", "constant")),
        )

        holder = QWidget()
        holder.setLayout(summary)
        layout.addWidget(holder)

        depth_configured = bool(
            config.get("depth", {}).get("measurements")
        ) or "depth" in {
            piece
            for piece, checkbox in self._piece_checkboxes.items()
            if checkbox.isChecked()
        }
        self._run_depth_checkbox = QCheckBox(
            "Compute the depth map now after finishing"
        )
        self._run_depth_checkbox.setEnabled(depth_configured)
        if not depth_configured:
            self._run_depth_checkbox.setToolTip(
                "Needs depth measurements — set them in the Depth tab first."
            )
        layout.addWidget(self._run_depth_checkbox)

        note = QLabel(
            "Rig setup itself is not run here: it is slow and partly interactive "
            "(crop correction). Run it from the Setup sidebar when you are ready."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {muted_text_color(self.palette()).name()};")
        layout.addWidget(note)

    # ------------------------------------------------------------- navigation

    def _goto(self, index: int) -> None:
        index = max(0, min(index, len(STEPS) - 1))

        # Flush the page we are leaving, so the next page sees current values
        # (per-folder protocol rows are keyed off [data].folders, for instance).
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()

        self._ensure_page(index)

        for position, (scroll, _layout) in enumerate(self._page_containers):
            scroll.setVisible(position == index)

        self._index = index
        self._rail.set_current(index)
        self._title_label.setText(STEPS[index][1])
        self._subtitle_label.setText(STEPS[index][2])
        self._back_button.setEnabled(index > 0)
        self._next_button.setText("Finish" if index == len(STEPS) - 1 else "Next  ›")

    def _on_next(self) -> None:
        if self._index < len(STEPS) - 1:
            self._goto(self._index + 1)
            return
        self._finish()

    # ----------------------------------------------------------------- finish

    def _finish(self) -> None:
        factory = self.main_window.settings_factory
        factory._sync_settings_inputs_to_config_dict()

        if self._selected_preset is not None and self._piece_checkboxes:
            enabled = {
                piece: checkbox.isChecked()
                for piece, checkbox in self._piece_checkboxes.items()
            }
            applied = self.main_window.config_controller.apply_series_preset(
                self._selected_preset, enabled
            )
            if applied:
                self.main_window.print_log(
                    f"Wizard applied series preset "
                    f"'{self._series_combo.currentText()}': {', '.join(applied)}."
                )

        # Writes config_dict to the TOML the main window has open.
        factory.save_settings()

        run_depth = (
            self._run_depth_checkbox is not None
            and self._run_depth_checkbox.isChecked()
            and self._run_depth_checkbox.isEnabled()
        )
        self.accept()
        if run_depth:
            self._start_depth_setup()

    def _start_depth_setup(self) -> None:
        config_file = self.main_window.config_file
        if not config_file or not Path(config_file).exists():
            self.main_window.print_log(
                "Skipping depth map: no config file on disk to run against."
            )
            return
        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_setup",
            "--config",
            str(Path(config_file).resolve()),
            "--depth",
        ]
        self.main_window.process_runner.start_workflow_process(
            argv,
            self.main_window.toolbar_builder.play_action,
            self.main_window.toolbar_builder.stop_action,
            cwd=Path.cwd(),
            workflow="setup",
            actions=["depth"],
            config_path=Path(config_file),
        )

    def done(self, result: int) -> None:
        # Hand the widget registry back to the Settings tabs: ours are about to be
        # destroyed, so rebuild theirs from config_dict (which holds every edit).
        self.main_window.settings_inputs.clear()
        self.main_window.settings_factory.refresh_current_view()
        super().done(result)
