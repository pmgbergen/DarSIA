"""The review window of a map placed in all runs: one overlay per run.

The list shows the runs in review order (the reference run first) with their
shift from the reference run's grid, their largest layer offset and the
checks; below it, the selected run's offset per pair of layers. The viewer
shows the selected run's overlay at full resolution, zoomed with the mouse
wheel and moved by dragging, with the legend of its two line types. Display
only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.map_placement import OVERLAY_LEGEND

# A layer pair counts as changed in the table only with this many measured
# samples (shorter boundaries give unsteady medians; the details list all)
# and a change of more than LARGE_CHANGE_MM.
MIN_SAMPLES = 100
LARGE_CHANGE_MM = 5.0

_COLUMNS = (
    "Run",
    "Status",
    "Shift from reference (px)",
    "Typical offset change (mm)",
    f"Pairs changed > {LARGE_CHANGE_MM:g} mm",
    "Gaps to fill (px)",
    "Water pockets (px)",
    "Unknown layers",
)


def offset_summary(row: Mapping[str, Any]) -> tuple[float | None, int]:
    """A run's typical change of the layer-pair offsets from the reference
    run and the number of pairs that changed by more than LARGE_CHANGE_MM.

    The typical change is the median of the pairs' changes weighted by their
    measured samples, so short boundaries with unsteady medians weigh little;
    None when no pair has a change. A pair counts as changed only with at
    least MIN_SAMPLES measured samples.
    """

    rows = [
        (float(boundary["change_from_reference_mm"]), int(boundary.get("measured_sample_count") or 0))
        for boundary in (dict(row.get("layer_offsets") or {})).get("boundaries") or []
        if boundary.get("change_from_reference_mm") is not None
    ]
    large = sum(1 for change, count in rows if count >= MIN_SAMPLES and abs(change) > LARGE_CHANGE_MM)
    rows = sorted((change, count) for change, count in rows if count > 0)
    total = sum(count for _change, count in rows)
    if not total:
        return None, large
    running = 0
    for change, count in rows:
        running += count
        if 2 * running >= total:
            return change, large
    return rows[-1][0], large


def _cells(row: Mapping[str, Any]) -> tuple[str, ...]:
    """The table cells of one run of a placement index."""

    checks = dict(row.get("checks") or {})
    shift = dict(row.get("offset_px") or {})
    typical, large = offset_summary(row)
    unknown = checks.get("unknown_layer_ids") or []
    return (
        str(row.get("run_id") or "").upper(),
        str(row.get("status") or ""),
        f"{shift['median_px']:.1f} / {shift['max_px']:.1f}" if shift else "",
        f"{typical:+.1f}" if typical is not None else ("none measured" if checks else ""),
        str(large) if checks else "",
        str(checks.get("inside_gap_pixels", checks.get("gap_pixels", ""))),
        str(checks.get("water_pocket_pixels", "")),
        ", ".join(str(value) for value in unknown) or ("none" if checks else ""),
    )


def offset_lines(row: Mapping[str, Any], layers: Mapping[str, Any]) -> list[str]:
    """One line per measured pair of layers of a run: median, P10 and P90."""

    def name(layer: int) -> str:
        entry = dict(layers.get(str(layer)) or {})
        return str(entry.get("name") or f"layer {layer}")

    def mm(value: Any) -> str:
        return "–" if value is None else f"{float(value):+.1f}"

    lines = []
    for boundary in (dict(row.get("layer_offsets") or {})).get("boundaries") or []:
        first, second = boundary["labels"]
        lines.append(
            f"{name(first)} | {name(second)}: change {mm(boundary.get('change_from_reference_mm'))} mm; "
            f"median {mm(boundary['median_mm'])} (P10 {mm(boundary['p10_mm'])}, "
            f"P90 {mm(boundary['p90_mm'])}); "
            f"{boundary['measured_sample_count']} of {boundary['sample_count']} samples"
        )
    return lines


class OverlayView(QGraphicsView):
    """A photograph at full resolution: wheel to zoom, drag to move."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the OverlayView display and controls."""
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setMinimumSize(320, 240)

    def show_image(self, path: Path | str | None) -> bool:
        """Show an image file, fitted to the view; False when it cannot be read."""

        pixmap = QPixmap(str(path)) if path else QPixmap()
        self.item.setPixmap(pixmap)
        if pixmap.isNull():
            return False
        self.scene().setSceneRect(self.item.boundingRect())
        self.fit()
        return True

    def show_rgb(self, image: Any) -> bool:
        """Show a uint8 RGB array, fitted to the view; False when it is empty."""

        import numpy as np

        array = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        if array.size == 0:
            self.item.setPixmap(QPixmap())
            return False
        height, width = array.shape[:2]
        # The QImage only borrows the buffer: copy it while the array lives.
        picture = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.item.setPixmap(QPixmap.fromImage(picture))
        self.scene().setSceneRect(self.item.boundingRect())
        self.fit()
        return True

    def fit(self) -> None:
        """Show the whole image."""

        self.resetTransform()
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: Any) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)


class MapPlacementDialog(QDialog):
    """Review a map placed in all runs: the runs, their checks, their overlays."""

    def __init__(self, placement: Mapping[str, Any], parent: QWidget | None = None) -> None:
        """Build a read-only placement summary for the saved map version."""
        super().__init__(parent)
        self.placement = dict(placement)
        self.folder = Path(str(self.placement["folder"]))
        self.layers = dict(self.placement.get("layers") or {})
        self.resize(1500, 900)
        layout = QVBoxLayout(self)
        version = dict(self.placement.get("map_version") or {})
        name = str(version.get("name") or "saved map")
        reference = str(self.placement.get("reference_run") or "reference run").upper()
        self.setWindowTitle(f"Sand map placement summary – {name} in all runs")
        self.lead = QLabel(f"Summary of how the approved sand map ({name} from {reference}) fits each run's "
                           "photograph. Read only: nothing is adopted or changed.")
        self._approval_lead = self.lead.text()
        self.refresh_approval_state(getattr(parent, "map_approvals", []))
        self.lead.setWordWrap(True)
        layout.addWidget(self.lead)
        note = QLabel(
            f"Map {version.get('name') or ''} (SHA-256 {str(version.get('sha256') or '')[:12]}…) "
            f"from {str(self.placement.get('reference_run') or '').upper()}, placed in each run "
            "through its measured geometry. Review only: compare the boundaries with each "
            "run's own photograph. Nothing is adopted and no run configuration changed. "
            "Layer offsets: the strongest colour edge within ±25 mm of each boundary, positive "
            "from the lower towards the higher layer id; the change is the difference from the "
            "reference run's own offset for the same pair. Gaps: pixels inside the map that it "
            "does not cover. Water pockets: free water apart from the main free water. "
        )
        positive_continuation = self.placement.get("format") == "darsia-fluidflower-map-placement-v2"
        if positive_continuation:
            note.setText(note.text() + " Known photo-edge layers continue only inside each target rig frame. "
                         "Cyan marks continuation beyond the reference photo; filled/support flags remain separate. "
                         "Zero or unknown edge donors remain true gaps.")
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        column_notes = {
            1: "No geometry means this run is not among the analysed runs.",
            2: "Shift is camera displacement, compensated by the measured frame model.",
            3: f"Typical offset change is the boundary fit compared with {reference}; near 0 is good.",
            4: f"Pairs changed counts boundaries that moved more than {LARGE_CHANGE_MM:g} mm, "
               f"with at least {MIN_SAMPLES} measured samples.",
        }
        self.column_legend = QLabel(" ".join(column_notes.values()))
        self.column_legend.setWordWrap(True)
        layout.addWidget(self.column_legend)
        project = getattr(parent, "project", None)
        if project is not None:
            try:
                location = self.folder.relative_to(Path(project.root))
            except ValueError:
                location = None
            if location is not None:
                label = QLabel(f"Saved placement: {location.as_posix()}")
                label.setWordWrap(True)
                layout.addWidget(label)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        side = QSplitter(Qt.Orientation.Vertical)
        columns = _COLUMNS + (("Continued beyond photo (px)",) if positive_continuation else ())
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(list(columns))
        for column, tooltip in column_notes.items():
            self.table.horizontalHeaderItem(column).setToolTip(tooltip)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.rows = list(self.placement.get("runs") or [])
        self.table.setRowCount(len(self.rows))
        for number, row in enumerate(self.rows):
            cells = _cells(row) + ((str((row.get("continuation") or {}).get("continued_pixels", "")),) if positive_continuation else ())
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if column in column_notes:
                    item.setToolTip(column_notes[column])
                self.table.setItem(number, column, item)
        self.table.itemSelectionChanged.connect(self._selected)
        side.addWidget(self.table)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        side.addWidget(self.details)
        side.setSizes([500, 300])
        splitter.addWidget(side)
        self.view = OverlayView()
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 980])
        layout.addWidget(splitter, 1)
        legend = [
            (str(row.get("label") or ""), tuple(row.get("rgb") or (255, 255, 255)))
            for row in self.placement.get("overlay_legend") or []
        ] or list(OVERLAY_LEGEND)
        self.legend = QLabel(
            "&nbsp;&nbsp;&nbsp;".join(
                f"<span style='color: rgb{tuple(int(value) for value in color)}'>■</span> {label}"
                for label, color in legend
            )
        )
        layout.addWidget(self.legend)
        self.status = QLabel("Choose a run to see its overlay. Wheel: zoom · drag: move.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Close summary")
        self.fit_button = buttons.addButton("Show whole photo", QDialogButtonBox.ButtonRole.ActionRole)
        self.fit_button.clicked.connect(self.view.fit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if self.rows:
            self.table.selectRow(0)

    def refresh_approval_state(self, approvals):
        """Mark withdrawn approval from cached log DATA; change no placement bytes."""
        from .layer_painting import approval_withdrawal
        version = self.placement.get("map_version") or {}
        withdrawn = approval_withdrawal(approvals, version.get("approval"))
        self.lead.setText(self._approval_lead + (
            " Approval withdrawn " + withdrawn["recorded_utc"] + "; retained placement evidence only."
            if withdrawn else ""))

    def show_run(self, number: int) -> bool:
        """Show the overlay and layer offsets of the run in row ``number``;
        False when it has no overlay."""

        row = self.rows[number]
        overlay = row.get("overlay")
        shown = self.view.show_image(self.folder / overlay if overlay else None)
        run = str(row.get("run_id") or "").upper()
        lines = offset_lines(row, self.layers)
        if row.get("layer_offsets") is None:
            self.details.setPlainText(f"{run}: layer offsets not measured.")
        else:
            self.details.setPlainText(
                f"{run}, layer offsets (change from the reference run first):\n"
                + ("\n".join(lines) or "no measurable boundary")
            )
        self.status.setText(
            f"{run}: the placed map on its reference photograph. Wheel: zoom · drag: move."
            if shown
            else f"{run}: {row.get('status') or 'no overlay'}."
        )
        return shown

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.show_run(rows[0].row())


__all__ = ["MapPlacementDialog", "OverlayView", "offset_lines", "offset_summary"]
