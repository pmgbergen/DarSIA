"""A read-only view of each run's measured geometry.

A geometry index (format ``darsia-measured-run-geometry-v1``) holds, per
run, a frame model: the homography between the run's camera grid and the
metric rig frame (millimetres, origin at the inner left wall and the bottom,
x to the right, y up), the measured width, the optical offset to the sand
and the area factor; and, when they are built, per-frame models
(``frames/*.json``) and pixel-area maps (``areas/*.npy``) beside it. This
window lists them per run and draws the frame model on the run's grid, on
its reference photograph or on one of its area maps. Display only: it
writes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.map_placement import RunGeometry, grid_spacing_mm, read_run_geometry
from darsia.fluidflower.map_placement_view import OverlayView

FRAME_COLOR = (255, 255, 0)
GRID_COLOR = (0, 255, 255)
GRID_STEP_MM = 100.0
# The longest side of the drawn grid, in pixels.
DISPLAY_LIMIT_PX = 2000

_COLUMNS = (
    "Run",
    "Reference photograph",
    "Grid (px)",
    "Width (mm)",
    "Optical offset (mm)",
    "Area factor",
    "mm per px (down / along)",
    "Frame models",
    "Area maps",
)


def model_files(geometry: RunGeometry) -> dict[str, list[Path]]:
    """A run's per-frame models and pixel-area maps: in the folder of the
    frame data its index row binds (``frame_data``, checked by size and
    SHA-256), or else beside its reference model."""

    from darsia.fluidflower.project import sha256_file

    folder = Path(geometry.source).parent
    bound = geometry.entry.get("frame_data")
    if isinstance(bound, Mapping) and bound.get("path"):
        index = Path(str(bound["path"]))
        if not index.is_file():
            raise FileNotFoundError(f"{geometry.run_id}: the frame data index is missing: {index}")
        if "size_bytes" in bound and index.stat().st_size != int(bound["size_bytes"]):
            raise ValueError(f"{geometry.run_id}: the frame data index changed (size)")
        if bound.get("sha256") and sha256_file(index) != bound["sha256"]:
            raise ValueError(f"{geometry.run_id}: the frame data index changed (SHA-256)")
        folder = index.parent / geometry.run_id
    return {
        "frames": sorted((folder / "frames").glob("*.json")),
        "areas": sorted((folder / "areas").glob("*.npy")),
    }


def measured_width_mm(geometry: RunGeometry) -> float | None:
    """The measured rig width of a run's frame model, in millimetres, or None."""

    frame = dict(geometry.model.get("metric_frame") or {})
    value = frame.get("measured_width_mm")
    return float(value) if value is not None else None


def frame_lines(geometry: RunGeometry, step_mm: float = GRID_STEP_MM) -> dict[str, list]:
    """The frame model's lines in the run's grid pixels (x, y): ``frame`` is
    the bottom and the two walls of the measured width, ``grid`` the lines
    every ``step_mm`` inside it. The walls and the grid run upwards as far
    as the width; the drawing clips them at the grid's edge."""

    width = measured_width_mm(geometry)
    if width is None:
        return {"frame": [], "grid": []}

    def to_grid(points_mm: list[tuple[float, float]]) -> np.ndarray:
        points = np.array([[x, y, 1.0] for x, y in points_mm]).T
        mapped = geometry.metric_to_grid @ points
        return (mapped[:2] / mapped[2]).T

    top = width
    frame = [
        to_grid([(0.0, 0.0), (width, 0.0)]),
        to_grid([(0.0, 0.0), (0.0, top)]),
        to_grid([(width, 0.0), (width, top)]),
    ]
    steps = np.arange(step_mm, width, step_mm)
    grid = [to_grid([(x, 0.0), (x, top)]) for x in steps]
    grid += [to_grid([(0.0, y), (width, y)]) for y in steps]
    return {"frame": frame, "grid": grid}


def draw_frame_model(
    geometry: RunGeometry,
    shape_hw: tuple[int, int],
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """The run's frame model drawn on its grid reduced to at most
    DISPLAY_LIMIT_PX, over ``background`` (uint8 RGB on the full grid) or a
    dark canvas. Returns the drawing and the reduction factor."""

    import cv2

    factor = max(1, int(np.ceil(max(shape_hw) / DISPLAY_LIMIT_PX)))
    height, width = int(shape_hw[0]) // factor, int(shape_hw[1]) // factor
    if background is None:
        canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    else:
        canvas = np.ascontiguousarray(background[: height * factor, : width * factor, :3])
        if factor > 1:
            canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        canvas = canvas.copy()
    lines = frame_lines(geometry)
    for kind, color, thickness in (("grid", GRID_COLOR, 2), ("frame", FRAME_COLOR, 3)):
        for line in lines[kind]:
            points = np.round(np.asarray(line) / factor).astype(np.int32)
            cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, color, thickness)
    return canvas, factor


def area_map_image(path: Path | str) -> tuple[np.ndarray, dict[str, float]]:
    """A pixel-area map (2-D .npy) as a colour image and its statistics.

    Colours run from the 1st to the 99th percentile of the finite values.
    """

    import cv2

    values = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{path} is not a two-dimensional map")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"{path} has no finite value")
    low, high = np.percentile(values[finite], [1, 99])
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    scaled[~finite] = 0.0
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    stats = {
        "min": float(values[finite].min()),
        "max": float(values[finite].max()),
        "mean": float(values[finite].mean()),
    }
    return np.ascontiguousarray(colored[..., ::-1]), stats


def _cells(geometry: RunGeometry) -> tuple[str, ...]:
    """The table cells of one run's frame model."""

    model = geometry.model
    shape = geometry.shape_hw
    try:
        files = model_files(geometry)
        frames, areas = str(len(files["frames"])), str(len(files["areas"]))
    except (OSError, ValueError):
        frames = areas = "unreadable"
    width = measured_width_mm(geometry)
    offset = model.get("optical_offset_mm")
    factor = model.get("area_factor")
    spacing = grid_spacing_mm(geometry, shape) if shape else None
    return (
        geometry.run_id.upper(),
        Path(str(geometry.reference_image.get("path") or "")).name,
        f"{shape[1]} x {shape[0]}" if shape else "",
        f"{width:g}" if width is not None else "",
        f"{float(offset):g}" if offset is not None else "",
        f"{float(factor):.4f}" if factor is not None else "",
        f"{spacing[0]:.4f} / {spacing[1]:.4f}" if spacing else "",
        frames,
        areas,
    )


class MeasuredGeometryDialog(QDialog):
    """Each run's measured geometry from a geometry index, read only."""

    def __init__(
        self,
        index_path: Path | str,
        *,
        photo_loader: Callable[[str, RunGeometry], np.ndarray] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        import json

        self.index_path = Path(index_path)
        self.photo_loader = photo_loader
        self.geometries = read_run_geometry(self.index_path)
        index = json.loads(self.index_path.read_text(encoding="utf-8-sig"))
        order = [str(run).casefold() for run in index.get("run_order") or []]
        self.runs = [run for run in order if run in self.geometries] + sorted(
            run for run in self.geometries if run not in order
        )
        missing = [run for run in order if run not in self.geometries]
        self.setWindowTitle("Measured geometry of the runs")
        self.resize(1500, 900)
        layout = QVBoxLayout(self)
        state = str(index.get("state") or "")
        note = QLabel(
            "Read only: each run's frame model from the geometry index (the homography "
            "between its camera grid and the metric rig frame, the measured width, the "
            "optical offset and the area factor), with its per-frame models and pixel-area "
            "maps when they are built. Nothing is adopted. "
            + (f"Index state: {state}. " if state else "")
            + (f"Not yet in the index: {', '.join(run.upper() for run in missing)}. " if missing else "")
            + f"Index: {self.index_path}"
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(len(self.runs), len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        for number, run in enumerate(self.runs):
            for column, value in enumerate(_cells(self.geometries[run])):
                self.table.setItem(number, column, QTableWidgetItem(value))
        self.table.itemSelectionChanged.connect(self._selected)
        splitter.addWidget(self.table)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.view = OverlayView()
        right_layout.addWidget(self.view, 1)
        controls = QHBoxLayout()
        self.photo_button = QPushButton("Show reference photograph")
        self.photo_button.setEnabled(photo_loader is not None)
        self.photo_button.clicked.connect(self.show_photograph)
        controls.addWidget(self.photo_button)
        self.area_choice = QComboBox()
        self.area_choice.setMinimumContentsLength(18)
        controls.addWidget(self.area_choice)
        self.area_button = QPushButton("Show area map")
        self.area_button.clicked.connect(self.show_area_map)
        controls.addWidget(self.area_button)
        fit = QPushButton("Fit")
        fit.clicked.connect(self.view.fit)
        controls.addWidget(fit)
        controls.addStretch(1)
        right_layout.addLayout(controls)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 800])
        layout.addWidget(splitter, 1)
        self.legend = QLabel(
            f"<span style='color: rgb{FRAME_COLOR}'>■</span> Rig frame: the bottom and the "
            "walls at 0 mm and the measured width&nbsp;&nbsp;&nbsp;"
            f"<span style='color: rgb{GRID_COLOR}'>■</span> Every {GRID_STEP_MM:g} mm"
        )
        layout.addWidget(self.legend)
        self.status = QLabel("Choose a run. Wheel: zoom · drag: move.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.current: str | None = None
        if self.runs:
            self.table.selectRow(0)

    def show_run(self, number: int, background: np.ndarray | None = None) -> bool:
        """Draw the frame model of the run in row ``number`` on its grid, or
        on ``background`` (uint8 RGB on that grid)."""

        run = self.runs[number]
        geometry = self.geometries[run]
        self.current = run
        self.area_choice.clear()
        try:
            areas = model_files(geometry)["areas"]
        except (OSError, ValueError) as error:
            areas = []
            self.status.setText(f"{run.upper()}: the frame data cannot be read: {error}")
        for path in areas:
            self.area_choice.addItem(path.stem, str(path))
        self.area_button.setEnabled(self.area_choice.count() > 0)
        shape = geometry.shape_hw or (background.shape[:2] if background is not None else None)
        if shape is None:
            self.status.setText(f"{run.upper()}: the model gives no grid size.")
            return self.view.show_rgb(np.zeros((0, 0, 3), dtype=np.uint8))
        drawing, factor = draw_frame_model(geometry, shape, background)
        self.status.setText(
            f"{run.upper()}: frame model on the {'reference photograph' if background is not None else 'camera grid'}"
            f" ({shape[1]} x {shape[0]} px, shown 1:{factor}). Wheel: zoom · drag: move."
        )
        return self.view.show_rgb(drawing)

    def show_photograph(self) -> bool:
        """Draw the frame model of the selected run on its reference photograph."""

        if self.current is None or self.photo_loader is None:
            return False
        geometry = self.geometries[self.current]
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            photo = np.asarray(self.photo_loader(self.current, geometry))
        except Exception as error:  # noqa: BLE001 - shown to the user
            self.status.setText(f"{self.current.upper()}: the photograph could not be read: {error}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        if geometry.shape_hw is not None and tuple(photo.shape[:2]) != tuple(geometry.shape_hw):
            self.status.setText(f"{self.current.upper()}: the photograph is not on the model's grid.")
            return False
        return self.show_run(self.runs.index(self.current), photo)

    def show_area_map(self) -> bool:
        """Show the chosen pixel-area map of the selected run."""

        path = self.area_choice.currentData()
        if not path:
            return False
        try:
            image, stats = area_map_image(path)
        except (OSError, ValueError) as error:
            self.status.setText(f"The area map could not be read: {error}")
            return False
        self.status.setText(
            f"{(self.current or '').upper()} area map {Path(path).stem}: "
            f"min {stats['min']:.6g}, mean {stats['mean']:.6g}, max {stats['max']:.6g} "
            "(colours from the 1st to the 99th percentile)."
        )
        return self.view.show_rgb(image)

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.show_run(rows[0].row())


__all__ = [
    "MeasuredGeometryDialog",
    "area_map_image",
    "draw_frame_model",
    "frame_lines",
    "measured_width_mm",
    "model_files",
]
