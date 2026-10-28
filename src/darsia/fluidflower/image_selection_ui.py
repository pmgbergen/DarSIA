"""Native-only Image selection row; no protocol or image read on navigation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.series_image_selection import read_series_image_selection
from darsia.fluidflower.image_selection_status import (
    production_selection_note,
    setup_selection_summary,
)
from darsia.fluidflower.stored_image_selection import loaded_studio_stored_plan
from darsia.fluidflower.image_selection_widget import SeriesImageSelectionDialog


class NativeImageSelectionWorkflow:
    def __init__(
        self,
        window,
        *,
        dialog_factory=SeriesImageSelectionDialog,
        rule_interpretation="",
        stored_plan_provider=None,
    ):
        self.window, self.dialog_factory = window, dialog_factory
        self.stored_plan_provider = (
            stored_plan_provider
            if stored_plan_provider is not None
            else lambda metadata: loaded_studio_stored_plan(self.window, metadata)
        )
        # Kept for existing callers; the dialog names Q6 Default20 explicitly.
        self.rule_interpretation = rule_interpretation
        self.panel = QWidget()
        outer = QVBoxLayout(self.panel)
        title = QLabel("Image selection")
        outer.addWidget(title)
        self.status = QLabel("Choose a registered series on Start.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.note = QLabel(
            "Setup reference photographs are chosen under Rig Geometry. "
            "The stored choices below are review/calibration inputs, including any selected baselines; "
            "Calibration determines which runs and images enter its fit. Production uses its separate grid. "
            "Load, Preview and Save are explicit actions; opening this page starts no calculation."
        )
        self.note.setWordWrap(True)
        outer.addWidget(self.note)
        self.edit_button = QPushButton("Edit image selection…")
        self.edit_button.clicked.connect(self.open)
        outer.addWidget(self.edit_button)
        self.saved_choices = QTableWidget(0, 4)
        self.saved_choices.setHorizontalHeaderLabels(
            ["Run", "Stored image choices", "Time span [h]", "Source binding recorded"]
        )
        self.saved_choices.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.saved_choices.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.saved_choices, 1)
        self.production_status = QLabel()
        self.production_status.setWordWrap(True)
        outer.addWidget(self.production_status)
        self.dialog = None

    def refresh(self):
        root = self.window._method_series_root()
        self.edit_button.setEnabled(root is not None)
        self.saved_choices.setRowCount(0)
        self.production_status.clear()
        if root is None:
            self.status.setText("Choose a registered series on Start.")
            return
        try:
            _table, raw = read_series_image_selection(root)
            import tomllib

            document = tomllib.loads(raw.decode("utf-8-sig"))
            summary = setup_selection_summary(document)
            self.status.setText(summary["summary"] + "\n" + summary["note"])
            self.production_status.setText(production_selection_note(document))
            ids = document["runs"]["ids"]
            self.saved_choices.setRowCount(len(ids))
            for index, run in enumerate(ids):
                row = document.get("image_selection", {}).get("runs", {}).get(run, {})
                selected = row.get("selected", [])
                times = [float(value["elapsed_h"]) for value in selected]
                span = f"{min(times):g} to {max(times):g}" if times else "Not selected"
                excluded = summary["excluded_runs"].get(run, "")
                name = str(run).upper() + (" (excluded)" if excluded else "")
                values = [
                    name,
                    str(len(selected)),
                    span,
                    "Yes" if selected and row.get("source_metadata") else "Not recorded",
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if excluded:
                        item.setToolTip(excluded)
                    self.saved_choices.setItem(index, column, item)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.status.setText("Not done — " + str(error))

    def show(self):
        self.refresh()
        self.window.content_stack.setCurrentWidget(self.panel)

    def open(self, *_args, production=False):
        """The image-selection editor; ``production`` opens it at the production grid."""
        root = self.window._method_series_root()
        if root is None:
            self.window.print_log(
                "Choose a registered series on Start before editing image selection."
            )
            return
        try:
            dialog = self.dialog_factory(
                root,
                self.window,
                rule_interpretation=self.rule_interpretation,
                **(
                    {"stored_plan_provider": self.stored_plan_provider}
                    if self.stored_plan_provider is not None
                    else {}
                ),
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.warning(
                self.window, "Image selection is unavailable", str(error)
            )
            return
        self.dialog = dialog
        current_run = self.window.run_combo.currentText()
        index = dialog.run_combo.findData(current_run)
        if index >= 0:
            dialog.run_combo.setCurrentIndex(index)
        dialog.saved.connect(self._saved)
        controls = getattr(dialog, "production_controls", None)
        if production and controls is not None:
            controls.group.setChecked(True)
        dialog.exec()

    def _saved(self, _run):
        self.refresh()
        # Rebuild one bounded cached context using the actual saved manifest.
        # Numerical actions and other calibration/method widgets are untouched.
        self.window._refresh_scope_display()
