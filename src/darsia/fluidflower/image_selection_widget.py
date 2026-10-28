"""Setup's explicit image-choice editor; no processing or automatic writes."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import tomllib

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.series_image_selection import (
    _eligible_frames,
    preview_targets,
    read_series_image_selection,
    save_run_image_selection,
    selected_frame_rows,
    load_run_protocol_metadata,
    suggest_target_hours,
)
from darsia.fluidflower.series_method import ManifestChangedError


class _ManualImageSelectionDialog(QDialog):
    """One run at a time; all choices are saved explicitly in series.toml.

    ``rule_interpretation`` has no default numerical interpretation. The owner
    must resolve the 8+12/pre-injection count question before Suggest is enabled.
    Existing saved choices/manual editing work independently of that decision.
    """

    saved = Signal(str)

    def __init__(
        self,
        series_root,
        parent=None,
        *,
        metadata_reader=load_run_protocol_metadata,
        rule_interpretation="",
        image_reader=QImageReader,
    ):
        super().__init__(parent)
        self.root = Path(series_root).resolve()
        self.metadata_reader, self.image_reader = metadata_reader, image_reader
        self.rule_interpretation = (
            ""  # Q6: do not guess a grid that contradicts Studio20
        )
        self.saved_table, self.expected_bytes = read_series_image_selection(self.root)
        self.document = tomllib.loads(self.expected_bytes.decode("utf-8-sig"))
        self.metadata = None
        self.drafts = {}
        self._loaded_run = ""
        self._rows = []
        self._eligible_count = 0
        self._changing = False
        self._dirty_runs = set()
        self.setWindowTitle("Image selection")
        self.resize(1220, 900)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Select the series images used for review and analysis. The rule is 20 images per run: "
            "1 before injection + 8 evenly over injection + 11 log-spaced after. "
            "Inspect suggestions, edit target times, or tick/untick actual images. Nothing is calculated here."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.run_combo = QComboBox()
        for run in self.document.get("runs", {}).get("ids", []):
            self.run_combo.addItem(str(run).upper(), run)
        line = QWidget()
        horizontal = QHBoxLayout(line)
        horizontal.setContentsMargins(0, 0, 0, 0)
        horizontal.addWidget(self.run_combo)
        self.load_button = QPushButton("Load this run's saved protocols")
        horizontal.addWidget(self.load_button)
        form.addRow("Run", line)
        counts = QWidget()
        horizontal = QHBoxLayout(counts)
        horizontal.setContentsMargins(0, 0, 0, 0)
        self.early_count = QSpinBox()
        self.early_count.setRange(2, 1000)
        self.early_count.setValue(8)
        self.early_count.setEnabled(False)
        self.rest_count = QSpinBox()
        self.rest_count.setRange(2, 1000)
        self.rest_count.setValue(11)
        self.rest_count.setEnabled(False)
        horizontal.addWidget(QLabel("During injection"))
        horizontal.addWidget(self.early_count)
        horizontal.addWidget(QLabel("After injection"))
        horizontal.addWidget(self.rest_count)
        self.suggest_button = QPushButton("Suggest images from protocol times")
        self.suggest_button.setEnabled(False)
        if not rule_interpretation:
            self.suggest_button.setToolTip(
                "Automatic grid awaits Q6: the actual stored Studio20 frames do not match the stated rule. Manual editing remains available."
            )
        horizontal.addWidget(self.suggest_button)
        form.addRow("Suggested counts (editable)", counts)
        self.targets = QLineEdit()
        self.targets.setPlaceholderText(
            "Comma-separated hours from first injection start"
        )
        form.addRow("Target hours (editable)", self.targets)
        self.tolerance = QLineEdit("15")
        form.addRow("Maximum target difference (seconds)", self.tolerance)
        self.match_button = QPushButton("Match edited target times")
        form.addRow(self.match_button)
        layout.addLayout(form)
        self.status = QLabel(
            "Load a run's saved protocol metadata to see its images; no raw folder is scanned."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            [
                "Use",
                "Source image",
                "Time from injection (h)",
                "Original index",
                "Source path",
            ]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(2, 190)
        self.table.setColumnWidth(3, 100)
        self.table.setColumnWidth(4, 550)
        layout.addWidget(self.table, 3)
        preview_line = QHBoxLayout()
        self.preview_button = QPushButton("Preview highlighted original image")
        preview_line.addWidget(self.preview_button)
        self.preview_note = QLabel(
            "Image pixels are read only when Preview is pressed."
        )
        self.preview_note.setWordWrap(True)
        preview_line.addWidget(self.preview_note, 1)
        layout.addLayout(preview_line)
        self.preview_label = QLabel("No image preview requested")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(160)
        self.preview_label.setMaximumHeight(280)
        layout.addWidget(self.preview_label, 1)
        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save this run's image choices to series.toml")
        self.close_button = QPushButton("Close")
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self.load_button.clicked.connect(self._load_clicked)
        self.suggest_button.clicked.connect(self._suggest_clicked)
        self.match_button.clicked.connect(self._match_clicked)
        self.preview_button.clicked.connect(self._preview_clicked)
        self.save_button.clicked.connect(self._save_clicked)
        self.close_button.clicked.connect(self.close)
        self.table.itemChanged.connect(self._choice_changed)
        self.targets.textEdited.connect(self._draft_changed)
        self.tolerance.textEdited.connect(self._draft_changed)
        self.run_combo.currentIndexChanged.connect(self._run_changed)
        self._update_controls()

    def _update_controls(self):
        ready = (
            self.metadata is not None
            and self.run_combo.currentData() == self._loaded_run
        )
        self.table.setEnabled(ready)
        for control in (self.match_button, self.preview_button, self.save_button):
            control.setEnabled(ready)
        self.suggest_button.setEnabled(ready and bool(self.rule_interpretation))

    def _remember(self):
        if self.metadata is not None:
            self.drafts[self._loaded_run] = {
                "metadata": self.metadata,
                "paths": self.selected_paths(),
                "targets": self.targets.text(),
                "tolerance": self.tolerance.text(),
            }

    def _run_changed(self):
        self._remember()
        # Changing the combo never reads a second run or drops a previous draft.
        self._update_controls()
        self.status.setText(
            "Press Load to read this run's saved metadata. Other run drafts are retained."
        )

    def selected_paths(self):
        return [
            row["source_path"]
            for i, row in enumerate(self._rows)
            if self.table.item(i, 0).checkState() == Qt.CheckState.Checked
        ]

    def load_run(self):
        self._remember()
        run = self.run_combo.currentData()
        if not run:
            raise ValueError("Choose a registered run.")
        draft = self.drafts.get(run)
        # An explicit Load rereads this run, including a changed saved protocol.
        # Manual choices remain a draft rather than being reset by that read.
        metadata = self.metadata_reader(self.root, run)
        eligible, _omitted = _eligible_frames(
            list(metadata.frames), check_files=False, source_root=None
        )
        saved = self.saved_table["runs"].get(run, {})
        paths = (
            draft["paths"]
            if draft
            else [r["source_path"] for r in saved.get("selected", [])]
        )
        # A stale/manual reference remains explicitly checked and removable; it
        # is never silently dropped or accepted by the save helper.
        available = {
            str(r["source_path"]).replace("\\", "/").casefold() for r in eligible
        }
        saved_rows = {r["source_path"]: r for r in saved.get("selected", [])}
        stale = [
            {
                "source_path": p,
                "image_id": "Unavailable / excluded: " + Path(p).name,
                "elapsed_h": saved_rows.get(p, {}).get("elapsed_h"),
                "frame_index": saved_rows.get(p, {}).get("frame_index", "—"),
                "ineligible": True,
            }
            for p in paths
            if str(p).replace("\\", "/").casefold() not in available
        ]
        selected = {str(p).replace("\\", "/").casefold() for p in paths}
        self.metadata, self._loaded_run, self._rows = metadata, run, eligible + stale
        self._eligible_count = len(eligible)
        self._changing = True
        try:
            self.table.setRowCount(len(self._rows))
            for i, row in enumerate(self._rows):
                use = QTableWidgetItem()
                use.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                )
                use.setCheckState(
                    Qt.CheckState.Checked
                    if row["source_path"].replace("\\", "/").casefold() in selected
                    else Qt.CheckState.Unchecked
                )
                self.table.setItem(i, 0, use)
                shown_time = (
                    "Unknown" if row["elapsed_h"] is None else f'{row["elapsed_h"]:.9g}'
                )
                for col, value in enumerate(
                    (
                        row["image_id"],
                        shown_time,
                        row["frame_index"],
                        row["source_path"],
                    ),
                    1,
                ):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(str(row["elapsed_h"]) if col == 2 else str(value))
                    self.table.setItem(i, col, item)
            self.targets.setText(
                draft["targets"]
                if draft
                else ", ".join(str(t) for t in saved.get("target_hours", []))
            )
            self.tolerance.setText(
                draft["tolerance"]
                if draft
                else str(saved.get("tolerance_seconds", 300))
            )
        finally:
            self._changing = False
        self.preview_label.clear()
        self.preview_label.setText("No image preview requested")
        self.preview_note.setText("Image pixels are read only when Preview is pressed.")
        self._update_controls()
        self._selection_status(
            "Unavailable/excluded saved images remain checked. Untick them explicitly before saving."
            if stale
            else ""
        )
        return metadata

    def _selection_status(self, extra=""):
        if self.metadata is None:
            return
        selected = self.selected_paths()
        self.status.setText(
            f"{self._loaded_run.upper()}: {len(selected)} chosen; {self._eligible_count} eligible protocol images. "
            + " ".join(self.metadata.notes)
            + (" " + extra if extra else "")
        )

    def _choice_changed(self, *_args):
        if not self._changing:
            self._draft_changed()
            self._selection_status("Unsaved manual image choices.")

    def _draft_changed(self, *_args):
        if self._loaded_run:
            self._dirty_runs.add(self._loaded_run)

    def suggest(self):
        if self.metadata is None:
            raise ValueError("Load this run's metadata first.")
        targets = suggest_target_hours(
            self.metadata,
            early_count=self.early_count.value(),
            rest_count=self.rest_count.value(),
            interpretation=self.rule_interpretation,
        )
        self.targets.setText(", ".join(repr(t) for t in targets))
        return self.match_targets()

    def match_targets(self):
        if self.metadata is None:
            raise ValueError("Load this run's metadata first.")
        targets = [
            float(t.strip()) for t in self.targets.text().split(",") if t.strip()
        ]
        receipt = preview_targets(
            self.metadata, targets, tolerance_seconds=float(self.tolerance.text())
        )
        selected = {r["source_path"] for r in receipt["runs"][0]["selected"]}
        self._changing = True
        try:
            for i, row in enumerate(self._rows):
                self.table.item(i, 0).setCheckState(
                    Qt.CheckState.Checked
                    if row["source_path"] in selected
                    else Qt.CheckState.Unchecked
                )
        finally:
            self._changing = False
        self._draft_changed()
        issues = [str(i["message"]) for i in receipt["runs"][0]["issues"]]
        self._selection_status(" ".join(issues + receipt["selection_notes"]))
        return receipt

    def preview_image(self):
        index = self.table.currentRow()
        if index < 0 or index >= len(self._rows):
            raise ValueError("Highlight an image row first.")
        path = self._rows[index]["source_path"]
        reader = self.image_reader(path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(
                size.scaled(QSize(1000, 260), Qt.AspectRatioMode.KeepAspectRatio)
            )
        image = reader.read()
        if image.isNull():
            raise ValueError(f"Could not preview {path}: {reader.errorString()}")
        self.preview_label.setPixmap(QPixmap.fromImage(image))
        self.preview_note.setText("Original source image: " + path)
        return path

    def save_run(self):
        if self.metadata is None or self.run_combo.currentData() != self._loaded_run:
            raise ValueError("Load this run's metadata first.")
        targets = [
            float(t.strip()) for t in self.targets.text().split(",") if t.strip()
        ]
        table, raw = save_run_image_selection(
            self.root,
            self.metadata,
            self.selected_paths(),
            expected_bytes=self.expected_bytes,
            target_hours=targets,
            tolerance_seconds=float(self.tolerance.text()),
            rule="manual_or_edited_targets",
        )
        self.saved_table, self.expected_bytes = table, raw
        self.targets.setText(
            ", ".join(str(t) for t in table["runs"][self._loaded_run]["target_hours"])
        )
        self._dirty_runs.discard(self._loaded_run)
        self._remember()
        self._selection_status("Saved in series.toml; no processing started.")
        self.saved.emit(str(self.root / "series.toml"))
        return self.root / "series.toml"

    def _clicked(self, title, operation):
        try:
            return operation()
        except (OSError, ValueError, RuntimeError, AssertionError, KeyError) as error:
            QMessageBox.warning(self, title, f"{type(error).__name__}: {error}")

    def _load_clicked(self):
        self._clicked("Image metadata was not loaded", self.load_run)

    def _suggest_clicked(self):
        self._clicked("Image suggestion was not made", self.suggest)

    def _match_clicked(self):
        self._clicked("Target times were not matched", self.match_targets)

    def _preview_clicked(self):
        self._clicked("Image preview was not opened", self.preview_image)

    def _save_clicked(self):
        self._clicked("Image choices were not saved", self.save_run)

    def closeEvent(self, event):
        production = getattr(self, "production_controls", None)
        if production is not None and production.busy:
            production.status.setText("The production metadata read is finishing; keep this editor open briefly.")
            event.ignore()
            return
        if (
            self._dirty_runs
            and QMessageBox.question(
                self,
                "Unsaved image choices",
                "Discard unsaved image choices for "
                + ", ".join(sorted(self._dirty_runs))
                + "?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            != QMessageBox.StandardButton.Discard
        ):
            event.ignore()
            return
        event.accept()


class SeriesImageSelectionDialog(_ManualImageSelectionDialog):
    """Q6: historical stored choice and a separate, explicitly applied proposal."""

    def reject(self):
        production = getattr(self, "production_controls", None)
        if production is not None and production.busy:
            production.status.setText("The production metadata read is finishing; keep this editor open briefly.")
            return
        super().reject()

    def __init__(self, *args, stored_plan_provider=None, **kwargs):
        from darsia.fluidflower.image_selection_proposal import RULE

        self.stored_plan_provider = stored_plan_provider
        self.proposal = self.applied_proposal = self.stored_plan = self.quality = None
        self.manual_edits_after_apply = False
        super().__init__(*args, **kwargs)
        self.rule_interpretation = RULE
        self.resize(1580, 1040)
        self.layout().itemAt(0).widget().setText(
            "Stored plan (Studio 20) is a historical manual choice and is never replaced automatically. "
            "Default 20 (1 + 8 + 11) is a separate camera-time proposal. Inspect, Apply and Save are explicit; "
            "no image calculation or new classification runs here."
        )
        self.suggest_button.setText("Propose Default 20 (1 + 8 + 11)")
        self.suggest_button.setToolTip(
            "Q6 camera-time proposal; never replaces the stored choice automatically."
        )
        form = self.layout().itemAt(1).layout()
        label = form.labelForField(self.early_count.parentWidget())
        if label is not None:
            label.setText("Fixed proposal counts")
        self.d1_edit = QLineEdit("0.25")
        form.addRow("First post-injection delay d1 (hours)", self.d1_edit)
        self.d1_edit.textEdited.connect(self._proposal_parameter_changed)
        self._build_time_grid_row(form)
        from darsia.fluidflower.image_selection_count_ui import ImageCountControls
        self.count_controls = ImageCountControls(self, form)
        from darsia.fluidflower.production_image_grid_ui import ProductionGridControls
        self.production_controls = ProductionGridControls(self, form)
        self.stored_status = QLabel(
            "Stored plan: Load the run to inspect its explicit source binding."
        )
        self.stored_status.setTextFormat(Qt.TextFormat.PlainText)
        self.stored_status.setWordWrap(True)
        self.proposal_status = QLabel("No Default20 proposal requested.")
        self.proposal_status.setTextFormat(Qt.TextFormat.PlainText)
        self.proposal_status.setWordWrap(True)
        self.import_stored_button = QPushButton("Import stored native manifest…")
        self.import_quality_button = QPushButton(
            "Load existing source-bound quality receipt…"
        )
        self.apply_stored_button = QPushButton(
            "Use stored selection in the manual draft"
        )
        self.apply_button = QPushButton("Apply this proposal to the manual draft")
        self.stored_table = QTableWidget(0, 4)
        self.stored_table.setHorizontalHeaderLabels(
            [
                "Stored image",
                "Stored Studio h",
                "Protocol camera h",
                "Original protocol index",
            ]
        )
        self.stored_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stored_table.setMaximumHeight(180)
        self.proposal_table = QTableWidget(0, 7)
        self.proposal_table.setHorizontalHeaderLabels(
            [
                "Target",
                "Target h",
                "Chosen image",
                "Camera h",
                "Offset s",
                "Saved class",
                "Flags",
            ]
        )
        self.proposal_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.proposal_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            [
                "Use",
                "Source image",
                "Time from injection (h)",
                "Original index",
                "Source path",
                "Saved 28 Sep class",
            ]
        )
        position = self.layout().indexOf(self.table)
        self.layout().takeAt(position)
        tables = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.addWidget(self.stored_status)
        left.addWidget(self.import_stored_button)
        left.addWidget(self.stored_table)
        left.addWidget(self.apply_stored_button)
        left.addWidget(QLabel("Current explicit choice / manual draft"))
        left.addWidget(self.table, 1)
        right.addWidget(self.proposal_status)
        right.addWidget(self.import_quality_button)
        right.addWidget(self.proposal_table, 1)
        right.addWidget(self.apply_button)
        tables.addLayout(left, 1)
        tables.addLayout(right, 1)
        self.layout().insertLayout(position, tables, 3)
        for table, stretch_column in (
            (self.stored_table, None),
            (self.table, 4),
            (self.proposal_table, 6),
        ):
            header = table.horizontalHeader()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            if stretch_column is not None:
                header.setSectionResizeMode(
                    stretch_column, QHeaderView.ResizeMode.Stretch
                )
            for column in range(table.columnCount()):
                item = table.horizontalHeaderItem(column)
                item.setToolTip(item.text())
        self.import_stored_button.clicked.connect(self._import_stored_clicked)
        self.import_quality_button.clicked.connect(self._import_quality_clicked)
        self.apply_button.clicked.connect(
            lambda: self._clicked("Proposal was not applied", self.apply_proposal)
        )
        self.apply_stored_button.clicked.connect(
            lambda: self._clicked(
                "Stored selection was not applied", self.apply_stored_plan
            )
        )
        self._update_controls()

    def _build_time_grid_row(self, form):
        """The time grid choice and its two proposal buttons."""
        from darsia.fluidflower.image_selection_grid import (
            grid_families,
            read_time_grids,
        )

        self.grid_proposal = self.applied_grid = None
        self.grid_edited = False
        try:
            self.grids, problem = read_time_grids(self.root), ""
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            self.grids, problem = {}, f"{type(error).__name__}: {error}"
        line = QWidget()
        horizontal = QHBoxLayout(line)
        horizontal.setContentsMargins(0, 0, 0, 0)
        self.grid_combo = QComboBox()
        for family, names in grid_families(self.grids).items():
            times = {
                round(value, 9) for name in names for value in self.grids[name]["times_h"]
            }
            count = len(times)
            self.grid_combo.addItem(f"{family} ({count} times)", names)
            self.grid_combo.setItemData(
                self.grid_combo.count() - 1, ", ".join(names), Qt.ItemDataRole.ToolTipRole
            )
        if not self.grids:
            self.grid_combo.addItem(problem or "No time grids in config/common.toml", None)
        self.grid_combo.setToolTip(
            "Times after injection start from the [[data_time]] tables in config/common.toml. "
            "Each time gets the nearest camera image within 15 s up to 3 h and 150 s after; "
            "a rejected image is replaced by an ok one within 60 s or 600 s."
        )
        self.grid_button = QPushButton("Propose from time grid")
        self.grid_button.setToolTip(
            "Match this run's camera images to the chosen grid; inspect, then Apply and Save."
        )
        self.grid_all_button = QPushButton("Propose and save for all runs…")
        self.grid_all_button.setToolTip(
            "For every run: propose from the chosen grid, apply it and save it to series.toml, "
            "one run after another, after one question."
        )
        horizontal.addWidget(self.grid_combo, 1)
        horizontal.addWidget(self.grid_button)
        horizontal.addWidget(self.grid_all_button)
        form.addRow("Time grid", line)
        self.grid_button.clicked.connect(
            lambda: self._clicked("Time grid not proposed", self.propose_from_time_grid)
        )
        self.grid_all_button.clicked.connect(self._grid_all_clicked)

    def _chosen_grids(self):
        names = self.grid_combo.currentData() if hasattr(self, "grid_combo") else None
        if not names:
            raise ValueError("Choose a time grid from config/common.toml.")
        return list(names)

    def _update_controls(self):
        super()._update_controls()
        if not hasattr(self, "apply_button"):
            return
        ready = (
            self.metadata is not None
            and self.run_combo.currentData() == self._loaded_run
        )
        self.suggest_button.setEnabled(ready)
        self.apply_button.setEnabled(
            ready and (self.proposal is not None or self.grid_proposal is not None
                       or self.count_controls.proposed is not None)
        )
        self.count_controls.update(ready)
        if hasattr(self, "grid_button"):
            has_grid = bool(self.grids)
            self.grid_button.setEnabled(ready and has_grid)
            self.grid_all_button.setEnabled(has_grid and self.run_combo.count() > 0)
        self.apply_stored_button.setEnabled(ready and self.stored_plan is not None)
        self.import_stored_button.setEnabled(ready)
        self.import_quality_button.setEnabled(ready)
        if getattr(self, "_source_errors", {}):
            self.save_button.setEnabled(False)

    def _remember(self):
        previous = (
            deepcopy(self.drafts.get(self._loaded_run, {}))
            if self.metadata is not None
            else {}
        )
        super()._remember()
        if self.metadata is not None and hasattr(self, "d1_edit"):
            errors = getattr(self, "_source_errors", {})
            self.drafts[self._loaded_run].update(
                d1=self.d1_edit.text(),
                stored_plan=deepcopy(self.stored_plan),
                quality=deepcopy(self.quality),
                proposal=deepcopy(self.proposal),
                applied_proposal=deepcopy(self.applied_proposal),
                manual_edits_after_apply=self.manual_edits_after_apply,
            )
            for field in errors:
                if field in previous:
                    self.drafts[self._loaded_run][field] = deepcopy(previous[field])

    def load_run(self):
        from darsia.fluidflower.image_selection_proposal import (
            legacy_stored_plan_status,
            load_stored_native_plan,
            load_saved_quality,
        )

        metadata = super().load_run()
        draft = self.drafts.get(self._loaded_run, {})
        saved = self.saved_table["runs"].get(self._loaded_run, {})
        self.stored_plan = self.quality = self.proposal = self.applied_proposal = None
        self.grid_proposal = self.applied_grid = None
        self.grid_edited = False
        self.count_controls.reset()
        self.manual_edits_after_apply = False
        self.d1_edit.setText(draft.get("d1", str(saved.get("default20_d1_h", 0.25))))
        self.proposal_table.setRowCount(0)
        self.proposal_status.setText(
            "No proposal requested; stored/manual images are unchanged."
        )
        notes = []
        self._source_errors = {}
        try:
            source = (draft.get("stored_plan") or {}).get("source") or saved.get(
                "stored_plan_source"
            )
            if source is None and self.stored_plan_provider is not None:
                source = self.stored_plan_provider(metadata)
            if source:
                self.stored_plan = load_stored_native_plan(
                    source["path"],
                    metadata,
                    series_id=self.document["series_id"],
                    expected=source,
                )
            else:
                notes.append(legacy_stored_plan_status(self.root)["note"])
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            ManifestChangedError,
        ) as error:
            self._source_errors["stored_plan"] = str(error)
            notes.append(type(error).__name__ + ": " + str(error))
        try:
            source = (draft.get("quality") or {}).get("source") or saved.get(
                "quality_source"
            )
            if source:
                self.quality = load_saved_quality(
                    source["path"], metadata, expected=source
                )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            ManifestChangedError,
        ) as error:
            self._source_errors["quality"] = str(error)
            notes.append(type(error).__name__ + ": " + str(error))
        self._show_stored(" ".join(notes))
        self._show_quality_classes()
        if draft.get("applied_proposal") and not self._source_errors:
            from darsia.fluidflower.image_selection_proposal import default20_proposal

            fresh = default20_proposal(
                metadata, d1_h=float(self.d1_edit.text()), quality=self.quality
            )
            if all(
                fresh[key] == draft["applied_proposal"][key]
                for key in ("parameters", "target_matches")
            ):
                self.applied_proposal = draft["applied_proposal"]
                self.manual_edits_after_apply = draft.get(
                    "manual_edits_after_apply", False
                )
        self._update_controls()
        return metadata

    def _show_stored(self, note=""):
        rows = (self.stored_plan or {}).get("selected", [])
        self.stored_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(
                (
                    row["image_id"],
                    row["historical_elapsed_h"],
                    row["protocol_elapsed_h"],
                    row["frame_index"],
                )
            ):
                item = QTableWidgetItem(str(value))
                item.setToolTip(
                    f"{self.stored_table.horizontalHeaderItem(j).text()}: {value}\n"
                    + row["source_path"]
                    + (
                        "\nStored and protocol clock values differ; both are retained."
                        if row["clock_difference"]
                        else ""
                    )
                )
                self.stored_table.setItem(i, j, item)
        self.stored_status.setText(
            (
                self.stored_plan["label"]
                + f": {len(rows)} images. "
                + self.stored_plan["note"]
            )
            if rows
            else "Stored plan (Studio 20): " + note
        )

    def _show_quality_classes(self):
        from darsia.fluidflower.image_selection_proposal import _key

        classes = (self.quality or {}).get("classes", {})
        previous = self._changing
        self._changing = True
        try:
            for i, row in enumerate(self._rows):
                self.table.setItem(
                    i,
                    5,
                    QTableWidgetItem(
                        classes.get(_key(row["source_path"]), {}).get(
                            "class", "Not recorded"
                        )
                    ),
                )
        finally:
            self._changing = previous

    def _draft_changed(self, *args):
        super()._draft_changed(*args)
        if not self._changing and hasattr(self, "count_controls"):
            self.count_controls.proposed = None
            self._update_controls()
        if self.applied_proposal is not None:
            self.manual_edits_after_apply = True
        if getattr(self, "applied_grid", None) is not None:
            self.grid_edited = True

    def _proposal_parameter_changed(self, *_args):
        self.count_controls.reset()
        self.proposal = None
        self.grid_proposal = None
        self.proposal_table.setRowCount(0)
        self.proposal_status.setText(
            "Parameters changed; Propose again before Apply. The current draft is unchanged."
        )
        if self._loaded_run:
            self._dirty_runs.add(self._loaded_run)
        if self.applied_proposal is not None:
            self.manual_edits_after_apply = True
        self._update_controls()

    def suggest(self):
        from darsia.fluidflower.image_selection_proposal import default20_proposal, RULE

        if self.metadata is None:
            raise ValueError("Load this run's metadata first.")
        self.count_controls.reset()
        self.proposal = default20_proposal(
            self.metadata, d1_h=float(self.d1_edit.text()), quality=self.quality
        )
        self.grid_proposal = None
        rows = self.proposal["target_matches"]
        self.proposal_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            image = row.get("chosen_image", {})
            values = (
                i,
                row["target_time_h"],
                image.get("image_id", "Not selected"),
                row.get("camera_elapsed_h", "—"),
                row.get("offset_seconds", "—"),
                row.get("frame_list_class", "Not recorded"),
                "; ".join(row["flags"]),
            )
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(
                    f"{self.proposal_table.horizontalHeaderItem(j).text()}: {value}\n"
                    + str(image.get("source_path") or "")
                )
                self.proposal_table.setItem(i, j, item)
        params = "; ".join(
            f"{key}={value}" for key, value in self.proposal["parameters"].items()
        )
        self.proposal_status.setText(
            RULE
            + f': {self.proposal["selected_frame_count"]} distinct images. '
            + params
            + ". "
            + " ".join(self.proposal["issues"])
            + " Explicit Apply required."
        )
        self._update_controls()
        return self.proposal

    def _set_draft_paths(self, paths):
        selected = set(paths)
        self._changing = True
        try:
            for i, row in enumerate(self._rows):
                self.table.item(i, 0).setCheckState(
                    Qt.CheckState.Checked
                    if row["source_path"] in selected
                    else Qt.CheckState.Unchecked
                )
        finally:
            self._changing = False

    def propose_from_time_grid(self):
        """Propose this run's images for the chosen time grid (shown, not applied)."""
        from darsia.fluidflower.image_selection_grid import (
            proposal_summary,
            propose_from_grid,
        )

        if self.metadata is None or self.run_combo.currentData() != self._loaded_run:
            raise ValueError("Load this run's metadata first.")
        self.count_controls.reset()
        result = propose_from_grid(
            self.metadata, self.grids, self._chosen_grids(), quality=self.quality
        )
        self.grid_proposal, self.proposal = result, None
        rows = result["target_matches"]
        self.proposal_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            if row["selected"]:
                flags = []
                if row.get("replacement"):
                    replaced = Path(str(row["replacement"].get("source_path", ""))).name
                    flags.append(f"replaces rejected {replaced}")
                if not (row.get("quality") or {}).get("recorded", True):
                    flags.append("quality not recorded")
                values = (
                    i,
                    row["target_hour"],
                    Path(str(row["source_path"])).name,
                    row["elapsed_h"],
                    round(row["offset_seconds"], 1),
                    row.get("frame_quality_class", "Not recorded"),
                    "; ".join(flags),
                )
            else:
                reason = row.get("reason", "")
                values = (i, row["target_hour"], "Not selected", "—", "—", "—", reason)
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(
                    f"{self.proposal_table.horizontalHeaderItem(j).text()}: {value}\n"
                    + str(row.get("source_path") or "")
                )
                self.proposal_table.setItem(i, j, item)
        self.proposal_status.setText(
            "Time grid proposal: " + proposal_summary(result) + " Explicit Apply required."
        )
        self._update_controls()
        return result

    def _apply_time_grid(self):
        """Put the inspected time grid proposal in the manual draft."""
        from darsia.fluidflower.image_selection_grid import propose_from_grid

        fresh = propose_from_grid(
            self.metadata, self.grids, self.grid_proposal["grids"], quality=self.quality
        )
        if fresh["selected_paths"] != self.grid_proposal["selected_paths"]:
            raise ValueError("The time grid proposal changed; inspect it again before Apply.")
        self._set_draft_paths(fresh["selected_paths"])
        matched = [
            row["target_hour"] for row in fresh["target_matches"] if row["selected"]
        ]
        self.targets.setText(", ".join(repr(value) for value in matched))
        self.tolerance.setText("150")
        self.applied_grid, self.applied_proposal = fresh, None
        self.grid_edited = False
        self._dirty_runs.add(self._loaded_run)
        self._selection_status(
            "Time grid proposal applied to the draft only. Save remains explicit."
        )
        return self.selected_paths()

    def _saved_rule(self):
        from darsia.fluidflower.image_selection_proposal import RULE

        if rule := self.count_controls.rule():
            return rule

        if self.applied_proposal is not None:
            return RULE
        if self.applied_grid is not None and not self.grid_edited:
            return "time grid: " + ", ".join(self.applied_grid["grids"])
        return "manual_or_edited_targets"

    def propose_and_save_all(self):
        """For every run in turn: load it, propose from the chosen grid, apply
        and save. Stops at the first run that cannot be saved; returns one
        result line per run."""
        from darsia.fluidflower.image_selection_grid import proposal_summary

        names = self._chosen_grids()
        unsaved = sorted(run.upper() for run in self._dirty_runs)
        if unsaved:
            raise ValueError(
                "Save or reload the runs with unsaved image choices first: "
                + ", ".join(unsaved)
            )
        start = self.run_combo.currentIndex()
        lines = []
        try:
            for index in range(self.run_combo.count()):
                run = self.run_combo.itemData(index)
                self.run_combo.setCurrentIndex(index)
                try:
                    self.load_run()
                    result = self.propose_from_time_grid()
                    self.apply_proposal()
                    self.save_run()
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                    lines.append(f"{str(run).upper()}: not saved: {failure}")
                    lines.append("The remaining runs were not changed.")
                    raise RuntimeError("\n".join(lines)) from error
                lines.append(f"{str(run).upper()}: saved; " + proposal_summary(result))
        finally:
            self.run_combo.setCurrentIndex(start)
        self.status.setText(
            f"Saved the {', '.join(names)} choices of {len(lines)} runs in series.toml."
        )
        return lines

    def _grid_all_clicked(self):
        try:
            names = self._chosen_grids()
        except ValueError as error:
            QMessageBox.warning(self, "Time grid not proposed", str(error))
            return
        answer = QMessageBox.question(
            self,
            "Propose and save for all runs",
            f"For each of the {self.run_combo.count()} runs, choose the camera images for "
            f"{', '.join(names)} and save them to series.toml, one run after another? "
            "Each run's earlier choice is replaced; the first run that cannot be saved stops "
            "the rest. No image is decoded and no calculation starts.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            lines = self.propose_and_save_all()
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Not all runs were saved", str(error))
            return
        QMessageBox.information(self, "Saved for all runs", "\n".join(lines))

    def apply_proposal(self):
        from darsia.fluidflower.image_selection_proposal import default20_proposal

        if self.count_controls.proposed is not None:
            return self.count_controls.apply()

        if self.proposal is None and self.grid_proposal is not None:
            return self._apply_time_grid()
        if self.proposal is None:
            raise ValueError("Request and inspect a proposal first.")
        fresh = default20_proposal(
            self.metadata, d1_h=float(self.d1_edit.text()), quality=self.quality
        )
        if fresh != self.proposal:
            raise ValueError("The proposal changed; inspect it again before Apply.")
        self._set_draft_paths(row["source_path"] for row in fresh["selected"])
        self.targets.setText(
            ", ".join(repr(row["target_time_h"]) for row in fresh["target_matches"])
        )
        self.applied_proposal = deepcopy(fresh)
        self.manual_edits_after_apply = False
        self._dirty_runs.add(self._loaded_run)
        self._selection_status(
            "Default20 applied to the draft only. Save remains explicit; historical Studio files are unchanged."
        )
        return self.selected_paths()

    def import_stored_plan(self, path):
        from darsia.fluidflower.image_selection_proposal import load_stored_native_plan

        if self.metadata is None:
            raise ValueError("Load this run's saved protocol metadata first.")
        value = load_stored_native_plan(
            path, self.metadata, series_id=self.document["series_id"]
        )
        self.stored_plan = value
        getattr(self, "_source_errors", {}).pop("stored_plan", None)
        self._show_stored()
        self._update_controls()
        return value

    def apply_stored_plan(self):
        if self.stored_plan is None:
            raise ValueError("Read an explicitly bound stored selection first.")
        self._set_draft_paths(
            row["source_path"] for row in self.stored_plan["selected"]
        )
        self.applied_proposal = None
        self._dirty_runs.add(self._loaded_run)
        self._selection_status(
            "Stored plan copied to the manual draft; no Studio file or series manifest was written."
        )
        return self.selected_paths()

    def import_quality(self, path):
        from darsia.fluidflower.image_selection_proposal import load_saved_quality

        if self.metadata is None:
            raise ValueError("Load this run's saved protocol metadata first.")
        value = load_saved_quality(path, self.metadata)
        self.quality = value
        getattr(self, "_source_errors", {}).pop("quality", None)
        self._show_quality_classes()
        self._proposal_parameter_changed()
        self._draft_changed()
        return value

    def _import_stored_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Read an existing source-bound native manifest", "", "JSON (*.json)"
        )
        if path:
            self._clicked(
                "Stored plan was not imported", lambda: self.import_stored_plan(path)
            )

    def _import_quality_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Read an existing source-bound 28 September quality receipt",
            "",
            "JSON (*.json)",
        )
        if path:
            self._clicked(
                "Saved quality was not imported", lambda: self.import_quality(path)
            )

    def save_run(self):
        if self.metadata is None or self.run_combo.currentData() != self._loaded_run:
            raise ValueError("Load this run first.")
        if getattr(self, "_source_errors", {}):
            raise ValueError(
                "Imported selection metadata changed; reimport the intended source before saving."
            )
        targets = [
            float(value.strip())
            for value in self.targets.text().split(",")
            if value.strip()
        ]
        table, raw = save_run_image_selection(
            self.root,
            self.metadata,
            self.selected_paths(),
            expected_bytes=self.expected_bytes,
            target_hours=targets,
            tolerance_seconds=float(self.tolerance.text()),
            rule=self._saved_rule(),
            proposal=self.applied_proposal,
            stored_plan=self.stored_plan,
            quality=self.quality,
            manual_edits_after_apply=self.manual_edits_after_apply,
            default20_d1_h=float(self.d1_edit.text()),
        )
        self.saved_table, self.expected_bytes = table, raw
        self._dirty_runs.discard(self._loaded_run)
        self._remember()
        self._selection_status(
            "Saved explicit selection, rule and parameters in series.toml; no processing started."
        )
        self.saved.emit(str(self.root / "series.toml"))
        return self.root / "series.toml"
