"""Guided setup wizard: a linear, explanatory view over the same config TOML.

The wizard holds no state of its own. Every field it shows is one of the standard
settings widgets bound to ``main_window.config_dict``, so setting something here is
identical to setting it in the expert-mode Settings tabs, or by editing the TOML by
hand — the three are just different speeds of the same editor. Only the
series-catalogue step is wizard-specific, and it writes plain config sections too.
"""

import sys
from dataclasses import fields as dataclass_fields
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
from darsia.presets.workflows.config.protocols import (
    ImagingProtocolConfig,
    InjectionProtocolConfig,
    PressureTemperatureProtocolConfig,
)

from .schema.dataclass_introspection import get_section_fields
from .setup import resolve_overwrite_conflicts
from .theme import muted_text_color, success_color, theme_signal

ASSETS_DIR = Path(__file__).parent / "assets" / "setup_wizard"
NO_SERIES = "None — start blank"

PROTOCOL_STEP_CONFIGS = {
    "imaging": ImagingProtocolConfig,
    "injection": InjectionProtocolConfig,
    "pressure_temperature": PressureTemperatureProtocolConfig,
}
"""Wizard step id -> the protocol sub-config whose fields that step edits."""


def _protocol_fields(config_class) -> list[dict]:
    """Schema for only the [protocols] fields declared by one sub-config.

    The three protocol configs are mixed into a single flat ProtocolsConfig (so the
    TOML stays flat), which means the generic schema is flat too. Filtering by the
    declaring class is what gives each wizard step its own fields, and keeps the
    split honest: add a field to a sub-config and it appears on that step.

    Group boxes are dropped, since the step's own page title already names the
    protocol; expert mode keeps its grouping untouched.
    """
    own_names = {f.name for f in dataclass_fields(config_class)}
    selected = []
    for setting in get_section_fields("protocols") or []:
        if setting["key"].rsplit(".", 1)[-1] not in own_names:
            continue
        setting = dict(setting)
        setting.pop("group_name", None)
        selected.append(setting)
    return selected


def _corrections_fields(*keys: str) -> list[dict]:
    """Schema for just the given [corrections] sub-sections (e.g. "curvature").

    Each of type/resize/curvature is its own nested group in the underlying
    schema; this picks out only the ones named, so each wizard step can render
    exactly its own piece without pulling in drift/color/illumination/etc.
    (those stay expert-mode only).
    """
    wanted = set(keys)
    return [
        setting
        for setting in get_section_fields("corrections") or []
        if setting["key"].rsplit(".", 1)[-1] in wanted
    ]


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
        "Experiment series",
        "Reuse a known rig setup, or start from scratch.",
        (
            "Experiments run on the same physical rig share their geometry, "
            "depth measurements and curvature correction. Pick a matching series "
            "to copy those in, or leave it as 'None' to set everything up "
            "yourself on the next few steps. Uncheck a piece here to keep setting "
            "just that one up manually, even if you picked a series for the rest "
            "— the next few steps update to match. Illumination and colour "
            "corrections are not part of these presets yet — set those in the "
            "Corrections tab."
        ),
    ),
    (
        "rig",
        "Rig geometry",
        "The physical size of the FluidFlower cell.",
        (
            "Width, height and dimensionality of the rig, in meters — used "
            "everywhere downstream to relate pixels to physical coordinates. "
            "Copied automatically if the series you picked provides it; "
            "otherwise type it in here."
        ),
    ),
    (
        "depth",
        "Depth measurements",
        "Where the sand is deep, and where it isn't — part of the rig's geometry.",
        (
            "'constant' generates a uniform-depth CSV automatically from the value "
            "below, in meters — right when the sand layer is (approximately) flat. "
            "Choose 'Load from CSV' if you already have, or will hand-author, a real "
            "scattered-point measurements CSV; it is then left untouched. Copied "
            "automatically if the series you picked provides it."
        ),
    ),
    (
        "type_resize",
        "Type & resize",
        "Two corrections that trade a little accuracy for speed.",
        (
            "These are performance options, not experiment ones, so they are "
            "always set here directly — never copied from a series preset. "
            "Converting images to float32 instead of float64 roughly halves "
            "memory use and speeds up every later processing step, usually with "
            "no visible cost. Resizing (downsampling) trades resolution for a "
            "faster turnaround — handy for a quick first pass before committing "
            "to a full-resolution run."
        ),
    ),
    (
        "curvature",
        "Curvature correction",
        "Straightening the image and cropping to the rig's four corners.",
        (
            "Corner points come from the actual baseline image, so the button "
            "above is usually easier than typing pixel coordinates by hand. "
            "Copied automatically if the series you picked provides it; "
            "otherwise launch the interactive picker or type the corners in "
            "below."
        ),
    ),
    (
        "imaging",
        "Imaging protocol",
        "Which image was taken when.",
        (
            "This is the timeline everything else hangs off. Pick how it should be "
            "built: read each image's own timestamp ('exif' from the camera, 'ctime' "
            "from the file), or compute timestamps from a cadence you declare "
            "('interval') — useful when the camera fired every N seconds and the "
            "metadata is unreliable. Choose 'Load from CSV' if you already wrote the CSV "
            "yourself, and it will be left untouched. The start reference is what "
            "counts as time zero, and anchors the other two protocols as well."
        ),
    ),
    (
        "injection",
        "Injection protocol",
        "When injection ran, where, and at what rate.",
        (
            "'constant' writes a single row spanning the whole run at the rate and "
            "coordinates you give — right when the experiment injected steadily from "
            "one port. Pick the rate's unit alongside it; if it's mL/s, mL/min or "
            "mL/hr, also give the fluid density so it can be converted to kg/s. "
            "Coordinates (x, y) are in meters, in the same physical frame as the "
            "rig's width/height. For anything time-varying, choose 'Load from CSV': "
            "an existing file is left untouched, and if none exists yet you get an "
            "empty template with just the column headers to fill in."
        ),
    ),
    (
        "pressure_temperature",
        "Pressure & temperature",
        "The conditions the experiment ran under.",
        (
            "'constant' writes a single row from the values below — fine when the rig "
            "held steady conditions throughout. Choose 'Load from CSV' to keep a full "
            "time-resolved CSV you maintain yourself; the file is then left untouched."
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
        self._series_name: str = NO_SERIES
        self._piece_checkboxes: dict[str, QCheckBox] = {}
        self._built_pages: set[int] = set()
        self._run_protocols_checkbox: QCheckBox | None = None
        self._run_depth_measurements_checkbox: QCheckBox | None = None
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

    def _forget_fields(self, settings_list: list[dict]) -> None:
        """Forget widgets we are about to destroy, so a later save never reads them.

        Keyed on the exact fields being rebuilt, so sibling protocol steps that are
        still alive keep their own registered widgets.
        """
        for setting in settings_list:
            self.main_window.settings_inputs.pop(setting["key"], None)

    # ------------------------------------------------------------------ pages

    # Steps whose fields must be forgotten + rebuilt on every visit, because
    # they either read state an earlier step may have changed (imaging's
    # per-folder rows), summarise everything (review), or their lock state
    # depends on the series step's checkboxes (rig/depth/type_resize/curvature).
    _LOCKABLE_STEP_FIELDS = {
        "rig": lambda: get_section_fields("rig") or [],
        "depth": lambda: get_section_fields("depth") or [],
        "type_resize": lambda: _corrections_fields("type", "resize"),
        "curvature": lambda: _corrections_fields("curvature"),
    }

    def _ensure_page(self, index: int) -> None:
        _scroll, layout = self._page_containers[index]
        step_id = STEPS[index][0]

        rebuild_always = step_id in ("imaging", "review", *self._LOCKABLE_STEP_FIELDS)
        if not rebuild_always and index in self._built_pages:
            return

        if step_id == "imaging":
            self._forget_fields(_protocol_fields(ImagingProtocolConfig))
        elif step_id in self._LOCKABLE_STEP_FIELDS:
            self._forget_fields(self._LOCKABLE_STEP_FIELDS[step_id]())
        self._clear_layout(layout)

        self._page_header(layout, index)
        if step_id == "data":
            self._build_section_form(layout, "data")
        elif step_id == "series":
            self._fill_series_page(layout)
        elif step_id == "rig":
            self._fill_rig_page(layout)
        elif step_id == "depth":
            self._fill_depth_page(layout)
        elif step_id == "type_resize":
            self._fill_type_resize_page(layout)
        elif step_id == "curvature":
            self._fill_curvature_page(layout)
        elif step_id in PROTOCOL_STEP_CONFIGS:
            self._fill_protocol_page(layout, step_id)
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
        self._series_name = name

        if name == NO_SERIES or name not in self._catalogue.presets:
            self._selected_preset = None
            self._series_description.setText(
                "Nothing will be copied — set rig, depth and curvature correction "
                "yourself on the next few steps."
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
            checkbox.toggled.connect(self._on_piece_toggled)
            self._pieces_layout.addWidget(checkbox)
            self._piece_checkboxes[piece] = checkbox

        self._apply_live()

    def _on_piece_toggled(self, _checked: bool) -> None:
        self._apply_live()

    def _apply_live(self) -> None:
        """Write the currently-checked preset pieces into config_dict right away.

        Runs the moment the preset or a piece checkbox changes, so the Rig/
        Depth/Curvature steps — built fresh on every visit, see
        _LOCKABLE_STEP_FIELDS — always show what is actually about to be saved,
        locked or not, rather than a promise kept until Finish.
        """
        if self._selected_preset is None or not self._piece_checkboxes:
            return
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

    def _is_piece_locked(self, piece: str) -> bool:
        """Whether `piece` is currently supplied by a checked series-preset box.

        A locked piece's step shows its values read-only, with a note pointing
        back to the Experiment series step — unchecking that box there is what
        unlocks manual editing again (see the confirmed per-piece design).
        """
        checkbox = self._piece_checkboxes.get(piece)
        return checkbox is not None and checkbox.isChecked()

    def _build_piece_form(self, layout, fields: list[dict], piece: str | None) -> None:
        """Render `fields` as a form, disabled with an explanatory note if the
        given series-preset `piece` is currently locked (piece=None means this
        content is never preset-provided, so it is always editable)."""
        locked = piece is not None and self._is_piece_locked(piece)
        if locked:
            label = PIECE_LABELS.get(piece, piece)
            note = QLabel(
                f"{label} is copied from the '{self._series_name}' series. "
                "Uncheck it on the Experiment series step to edit it here instead."
            )
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {muted_text_color(self.palette()).name()};")
            layout.addWidget(note)

        holder = QWidget()
        form = QFormLayout(holder)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addWidget(holder)
        if fields:
            self.main_window.settings_factory.build_tab_form(form, fields)
        holder.setEnabled(not locked)

    def _fill_rig_page(self, layout) -> None:
        self._build_piece_form(layout, get_section_fields("rig") or [], "rig")

    def _fill_depth_page(self, layout) -> None:
        self._build_piece_form(layout, get_section_fields("depth") or [], "depth")

    def _fill_type_resize_page(self, layout) -> None:
        layout.addWidget(QLabel("Type correction"))
        self._build_piece_form(layout, _corrections_fields("type"), None)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Resize correction"))
        self._build_piece_form(layout, _corrections_fields("resize"), None)

    def _fill_curvature_page(self, layout) -> None:
        locked = self._is_piece_locked("curvature")

        crop_button = QPushButton("Launch interactive crop correction…")
        if locked:
            crop_button.setEnabled(False)
            crop_button.setToolTip(
                f"Curvature correction is copied from the '{self._series_name}' "
                "series. Uncheck it on the Experiment series step to set it up "
                "interactively instead."
            )
        else:
            crop_button.setToolTip(
                "Opens a separate window to click the four corners of the rig on "
                "your baseline image. Saves your config first — fill in the Data "
                "step before using this."
            )
            crop_button.clicked.connect(self._launch_crop_correction)
        layout.addWidget(crop_button)
        layout.addSpacing(8)

        self._build_piece_form(layout, _corrections_fields("curvature"), "curvature")

    def _launch_crop_correction(self) -> None:
        """Save the config, then spawn the existing interactive crop assistant.

        Crop correction reads [data]/[corrections]/[rig] from the TOML file on
        disk (not from config_dict) and writes its result straight back to that
        file, so it needs a real, current config file to run against.
        """
        self.main_window.settings_factory._sync_settings_inputs_to_config_dict()
        config_file = self.main_window.config_file
        if not config_file:
            self.main_window.print_log(
                "Crop correction: create or open a config file first."
            )
            return
        self.main_window.settings_factory.save_settings()

        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_setup",
            "--config",
            str(Path(config_file).resolve()),
            "--crop",
        ]
        self.main_window.print_log("Wizard launching interactive crop correction…")
        self._run_subprocess(argv, ["crop"], Path(config_file))

    def _fill_protocol_page(self, layout, step_id: str) -> None:
        """Render just one protocol's fields, straight from its sub-config."""
        settings_list = _protocol_fields(PROTOCOL_STEP_CONFIGS[step_id])
        if not settings_list:
            layout.addWidget(QLabel("No settings found for this protocol."))
            return
        holder = QWidget()
        form = QFormLayout(holder)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addWidget(holder)
        self.main_window.settings_factory.build_tab_form(form, settings_list)

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
        add(
            "Depth measurements",
            str(config.get("depth", {}).get("measurements_mode", "Load from CSV")),
        )

        holder = QWidget()
        holder.setLayout(summary)
        layout.addWidget(holder)

        run_box = QGroupBox("Run after finishing")
        run_layout = QVBoxLayout(run_box)

        self._run_protocols_checkbox = QCheckBox("Generate the protocol CSV files")
        self._run_protocols_checkbox.setToolTip(
            "Runs protocol setup with the modes chosen above. Protocols set to "
            "'Load from CSV' are left untouched; you are asked before anything is "
            "overwritten."
        )
        self._run_protocols_checkbox.setChecked(True)
        run_layout.addWidget(self._run_protocols_checkbox)

        self._run_depth_measurements_checkbox = QCheckBox("Generate depth measurements")
        self._run_depth_measurements_checkbox.setToolTip(
            "Runs depth-measurements setup with the mode chosen on the Depth "
            "measurements step. No-op if that mode is 'Load from CSV'; you are asked "
            "before an existing 'constant'-mode file is overwritten."
        )
        self._run_depth_measurements_checkbox.setChecked(True)
        run_layout.addWidget(self._run_depth_measurements_checkbox)

        depth_configured = bool(
            config.get("depth", {}).get("measurements")
        ) or "depth" in {
            piece
            for piece, checkbox in self._piece_checkboxes.items()
            if checkbox.isChecked()
        }
        self._run_depth_checkbox = QCheckBox("Compute the depth map")
        self._run_depth_checkbox.setEnabled(depth_configured)
        if not depth_configured:
            self._run_depth_checkbox.setToolTip(
                "Needs depth measurements — set them in the Depth tab first."
            )
        run_layout.addWidget(self._run_depth_checkbox)
        layout.addWidget(run_box)

        note = QLabel(
            "Rig setup itself (baseline correction, depth map, image porosity) is "
            "not run here — it's slow and best run once everything above is "
            "correct. Run it from the Setup sidebar when you are ready."
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
        # The series preset (if any) was already applied live as it was picked/
        # toggled (see _apply_live) — nothing deferred to do with it here.
        factory = self.main_window.settings_factory
        factory._sync_settings_inputs_to_config_dict()

        # Writes config_dict to the TOML the main window has open. Setup runs as a
        # subprocess reading that file, so it has to land on disk first.
        factory.save_settings()

        actions = []
        if self._checked(self._run_protocols_checkbox):
            actions.append("protocol")
        if self._checked(self._run_depth_measurements_checkbox):
            actions.append("depth_measurements")
        if self._checked(self._run_depth_checkbox):
            actions.append("depth")

        config_file = self.main_window.config_file
        force = False
        if config_file and ("protocol" in actions or "depth_measurements" in actions):
            conflicts = self._collect_conflicts(actions, Path(config_file))
            if conflicts is None:
                # A conflict check itself failed; already logged, stay open.
                return
            decision = resolve_overwrite_conflicts(
                self.main_window, conflicts, "Setup output files"
            )
            if decision is None:
                # Cancelled at the overwrite prompt: stay open so the choice can be
                # revised rather than silently finishing without generating.
                return
            force = decision

        self.accept()
        if actions:
            self._start_setup(actions, force)

    def _collect_conflicts(
        self, actions: list[str], config_path: Path
    ) -> list[Path] | None:
        """Gather overwrite conflicts across whichever checked actions can cause
        one, so Finish asks about all of them in a single combined prompt (they
        share one `--force` flag on the subprocess either way).

        Returns None if a check itself failed (already logged to the user).
        """
        conflicts: list[Path] = []
        if "protocol" in actions:
            try:
                from darsia.presets.workflows.setup.setup_protocols import (
                    preview_protocol_setup_conflicts,
                )

                conflicts += preview_protocol_setup_conflicts([config_path])
            except Exception as exc:
                self.main_window.print_log(f"Error checking protocol conflicts: {exc}")
                return None
        if "depth_measurements" in actions:
            try:
                from darsia.presets.workflows.setup.setup_depth import (
                    preview_depth_measurements_conflict,
                )

                conflicts += preview_depth_measurements_conflict([config_path])
            except Exception as exc:
                self.main_window.print_log(
                    f"Error checking depth-measurements conflicts: {exc}"
                )
                return None
        return conflicts

    @staticmethod
    def _checked(checkbox: QCheckBox | None) -> bool:
        return checkbox is not None and checkbox.isEnabled() and checkbox.isChecked()

    def _run_subprocess(
        self, argv: list[str], actions: list[str], config_path: Path
    ) -> None:
        """Launch a setup subprocess the same way the flat Setup sidebar does."""
        process = self.main_window.process_runner.start_workflow_process(
            argv,
            self.main_window.toolbar_builder.play_action,
            self.main_window.toolbar_builder.stop_action,
            cwd=Path.cwd(),
            workflow="setup",
            actions=actions,
            config_path=config_path,
        )
        if process is not None:
            process.finished.connect(self._refresh_after_subprocess)

    def _refresh_after_subprocess(self, *_args) -> None:
        """Re-render the page we're on, now that the subprocess has finished.

        Interactive crop correction (launched from the Curvature step while the
        wizard stays open) writes straight to the config file on disk. On
        success, process_runner's own finished-handler — connected before this
        one, so it runs first — already reloads main_window.config_dict from
        that file. Lockable steps already rebuild unconditionally in
        _ensure_page, so simply re-running it here is enough to pick up
        whatever changed; a no-op if we've since navigated elsewhere or the
        wizard has already closed (e.g. after the Review step's Finish).
        """
        if self.isVisible():
            self._ensure_page(self._index)

    def _start_setup(self, actions: list[str], force: bool) -> None:
        """Run the requested setup steps in one subprocess, as the Setup tab does.

        Out of process on purpose: a protocol run scans every image's metadata and
        would otherwise freeze the GUI, and this way it is abortable and streams
        into the log dock.
        """
        config_file = self.main_window.config_file
        if not config_file or not Path(config_file).exists():
            self.main_window.print_log(
                "Skipping setup: no config file on disk to run against."
            )
            return

        argv = [
            sys.executable,
            "-m",
            "darsia.presets.workflows.user_interface_setup",
            "--config",
            str(Path(config_file).resolve()),
        ]
        if "protocol" in actions:
            argv.append("--protocol")
        if "depth_measurements" in actions:
            argv.append("--depth-measurements")
        if "depth" in actions:
            argv.append("--depth")
        if force:
            argv.append("--force")

        self.main_window.print_log(
            f"Wizard starting setup: {', '.join(actions)}"
            f"{' (overwriting existing files)' if force else ''}."
        )
        self._run_subprocess(argv, actions, Path(config_file))

    def done(self, result: int) -> None:
        # Hand the widget registry back to the Settings tabs: ours are about to be
        # destroyed, so rebuild theirs from config_dict (which holds every edit).
        self.main_window.settings_inputs.clear()
        self.main_window.settings_factory.refresh_current_view()
        super().done(result)
