"""Mass analysis > Image selection: the images per run that mass analysis computes.

Mass analysis computes one shared production time grid in every run, saved
apart from the calibration and review choices (production_image_grid). This
page shows the saved grid per run and opens the existing image-selection
editor at its production section. Opening the page reads series.toml and the
saved grid only; nothing is computed or written.
"""
from __future__ import annotations

import json
from pathlib import Path
import tomllib

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

COLUMNS = ("Run", "Baseline", "Main images", "Late images", "Total", "Missing targets")
COUNTS = ("baselines", "main", "late", "selected", "missing")


def saved_production_grid(root):
    """The production grid that series.toml in ``root`` binds, or None when none is saved.

    The grid must still have its recorded SHA-256 and size. The result holds
    its parameters, who saved it and when, and the image counts per run in the
    registered run order.
    """
    from .production_image_grid import FORMAT, KEY
    from .series_image_selection import _descriptor, _located
    from .storage_paths import storage_path_remapper
    root = Path(root)
    document = tomllib.loads((root / "series.toml").read_bytes().decode("utf-8-sig"))
    choice = document.get(KEY)
    if not choice:
        return None
    pin = choice.get("selection_manifest") if isinstance(choice, dict) else None
    if not isinstance(pin, dict) or not pin.get("path"):
        raise ValueError("The recorded production grid reference is incomplete")
    path = Path(_located(pin["path"], root, storage_path_remapper()))
    actual = _descriptor(path)
    if any(actual[key] != pin.get(key) for key in ("sha256", "size_bytes")):
        raise ValueError("The saved production grid changed after it was recorded: " + str(path))
    saved = json.loads(path.read_bytes().decode("utf-8-sig"))
    if not isinstance(saved, dict) or saved.get("format") != FORMAT:
        raise ValueError("The recorded file is not a production grid: " + str(path))
    counts = {run: dict(row.get("counts") or {}) for run, row in (saved.get("runs") or {}).items()}
    return dict(path=str(path), parameters=dict(saved.get("parameters") or {}),
                created_utc=str(saved.get("created_utc") or ""), created_by=str(saved.get("created_by") or ""),
                runs=[(run, counts[run]) for run in document["runs"]["ids"] if run in counts])


def grid_summary(grid) -> str:
    """One paragraph on the saved grid, or on its absence."""
    if grid is None:
        return "No production image selection is saved yet. Mass analysis needs one; use Choose production images."
    values = grid["parameters"]
    numbers = [values.get(key) for key in ("main_count", "first_h", "through_h", "late_count")]
    text = "Saved production grid"
    if all(isinstance(value, (int, float)) for value in numbers):
        main, first, through, late = numbers
        text += (f": {main} images from {first:g} h through {through:g} h after injection, "
                 f"up to {late} later images, plus each run's baseline")
    saved = " ".join(part for part in (grid["created_utc"][:16].replace("T", " "),
                                        ("by " + grid["created_by"]) if grid["created_by"] else "") if part)
    return text + (f" (saved {saved} UTC)." if saved else ".") + "\n" + grid["path"]


class ProductionImageSelectionPage:
    """The Mass analysis Image selection row; opening it starts no calculation."""

    def __init__(self, window):
        self.window = window
        self.panel = QWidget()
        outer = QVBoxLayout(self.panel)
        outer.addWidget(QLabel("Image selection"))
        note = QLabel(
            "Mass analysis computes these images in every run: one shared time grid after injection, "
            "plus each run's baseline. They are separate from the calibration images and from the "
            "review choices under Setup > Image selection. Opening this page starts no calculation.")
        note.setWordWrap(True)
        outer.addWidget(note)
        self.status = QLabel("Choose a registered series on Start.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.status)
        self.edit_button = QPushButton("Choose production images…")
        self.edit_button.setToolTip("Opens the image-selection editor at its production grid; "
                                    "proposing and saving are separate explicit steps.")
        self.edit_button.clicked.connect(self.open)
        outer.addWidget(self.edit_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.table, 1)

    def refresh(self):
        root = self.window._method_series_root()
        self.edit_button.setEnabled(root is not None)
        self.table.setRowCount(0)
        if root is None:
            self.status.setText("Choose a registered series on Start.")
            return
        try:
            grid = saved_production_grid(root)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.status.setText("The saved production grid could not be read: " + str(error))
            return
        self.status.setText(grid_summary(grid))
        rows = grid["runs"] if grid is not None else []
        self.table.setRowCount(len(rows))
        for index, (run, counts) in enumerate(rows):
            for column, value in enumerate((run.upper(), *(counts.get(key, "") for key in COUNTS))):
                self.table.setItem(index, column, QTableWidgetItem(str(value)))

    def show(self):
        self.refresh()
        self.window.content_stack.setCurrentWidget(self.panel)

    def open(self, *_args):
        """The existing image-selection editor, opened at its production grid."""
        self.window.image_selection_workflow.open(production=True)
        self.refresh()
