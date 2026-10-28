"""Explicit image-count proposals in the existing image selection editor."""

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout, QMessageBox, QPushButton, QSpinBox, QTableWidgetItem, QWidget,
)

from .image_selection_count import propose_image_count


class ImageCountControls:
    def __init__(self, dialog, form):
        self.dialog = dialog
        self.proposed = self.applied = None
        line = QWidget()
        layout = QHBoxLayout(line)
        layout.setContentsMargins(0, 0, 0, 0)
        self.count = QSpinBox()
        self.count.setRange(2, 10000)
        self.count.setValue(80)
        self.count.setToolTip(
            "Total images, including the baseline and late images. Keep the current "
            "choice and add images between it, retaining its sampling density."
        )
        self.button = QPushButton("Propose image count")
        self.button.setToolTip(
            "Keep the current images and add images between them. The complete camera "
            "timeline is required; missing quality measurements remain explicit."
        )
        self.all_button = QPushButton("Propose and save image count…")
        self.all_button.setToolTip(
            "Apply the total to runs that already have saved image choices. "
            "Runs without a saved choice are not included."
        )
        layout.addWidget(self.count)
        layout.addWidget(self.button)
        layout.addWidget(self.all_button)
        form.addRow("Total images per run", line)
        self.count.valueChanged.connect(self.invalidate)
        self.button.clicked.connect(
            lambda: dialog._clicked("Image count not proposed", self.propose)
        )
        self.all_button.clicked.connect(self._save_all_clicked)

    def reset(self):
        self.proposed = self.applied = None

    def invalidate(self, *_args):
        if self.proposed is not None:
            self.proposed = None
            self.dialog.proposal_table.setRowCount(0)
            self.dialog.proposal_status.setText(
                "Image count changed; propose again before Apply. The draft is unchanged."
            )
        self.dialog._update_controls()

    def update(self, ready):
        healthy = not getattr(self.dialog, "_source_errors", {})
        self.button.setEnabled(ready and healthy)
        self.all_button.setEnabled(bool(self._runs()) and healthy)

    def _runs(self):
        return {
            run for run, choice in self.dialog.saved_table.get("runs", {}).items()
            if choice.get("selected")
        }

    def propose(self):
        dialog = self.dialog
        if dialog.metadata is None or dialog.run_combo.currentData() != dialog._loaded_run:
            raise ValueError("Load this run's saved protocols first.")
        if getattr(dialog, "_source_errors", {}):
            raise ValueError("Reload the changed selection metadata before proposing images.")
        result = propose_image_count(
            dialog.metadata, dialog.selected_paths(), self.count.value(),
            quality=dialog.quality,
        )
        self.proposed = result
        dialog.proposal = dialog.grid_proposal = None
        dialog.proposal_table.setRowCount(len(result["selected"]))
        for index, row in enumerate(result["selected"]):
            quality = row["quality"]
            values = (
                index, row["elapsed_h"], Path(row["source_path"]).name,
                row["elapsed_h"], 0, quality["class"],
                "Kept from current choice" if row["retained"] else "Added between current images",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(row["source_path"])
                dialog.proposal_table.setItem(index, column, item)
        dialog.proposal_status.setText(
            f"{len(result['selected_paths'])} images total, including baseline and late images. "
            f"{result['unrecorded_quality']} without a saved quality class. "
            "Inspect the proposal, then Apply and Save."
        )
        dialog._update_controls()
        return result

    def apply(self):
        dialog = self.dialog
        result = self.proposed
        fresh = propose_image_count(
            dialog.metadata, result["retained_paths"], result["count"],
            quality=dialog.quality,
        )
        if fresh != result:
            raise ValueError("The image-count proposal changed; inspect it again before Apply.")
        dialog._set_draft_paths(fresh["selected_paths"])
        dialog.targets.setText(", ".join(repr(row["elapsed_h"]) for row in fresh["selected"]))
        dialog.tolerance.setText("0")
        dialog.applied_grid = dialog.applied_proposal = None
        self.applied = fresh
        self.proposed = None
        dialog._dirty_runs.add(dialog._loaded_run)
        dialog._selection_status("Image-count proposal applied to the draft only. Save remains explicit.")
        dialog._update_controls()
        return fresh["selected_paths"]

    def rule(self):
        if self.applied is not None and self.dialog.selected_paths() == self.applied["selected_paths"]:
            return f"{self.applied['count']} images including baseline; retain current images and fill intervals"
        return None

    def save_all(self):
        dialog = self.dialog
        if dialog._dirty_runs:
            raise ValueError("Save or reload runs with unsaved image choices first.")
        runs = self._runs()
        if not runs:
            raise ValueError("Save an initial image choice before requesting a count.")
        start = dialog.run_combo.currentIndex()
        lines = []
        try:
            for index in range(dialog.run_combo.count()):
                run = dialog.run_combo.itemData(index)
                if run not in runs:
                    continue
                dialog.run_combo.setCurrentIndex(index)
                try:
                    dialog.load_run()
                    result = self.propose()
                    self.apply()
                    dialog.save_run()
                except Exception as error:
                    lines.append(f"{str(run).upper()}: not saved: {error}")
                    raise RuntimeError("\n".join(lines) + "\nRemaining runs were not changed.") from error
                lines.append(f"{str(run).upper()}: saved {result['count']} images")
        finally:
            dialog.run_combo.setCurrentIndex(start)
        return lines

    def _save_all_clicked(self):
        total = self.count.value()
        answer = QMessageBox.question(
            self.dialog, "Propose and save image count",
            f"For each of the {len(self._runs())} runs with a saved choice, retain its images "
            f"and add images to reach exactly {total} total, including baseline and late images? "
            "This saves each run in turn and stops at the first failure. "
            "Images without saved quality measurements remain unclassified. No calculation starts.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.dialog._clicked("Image counts not saved", self.save_all)
