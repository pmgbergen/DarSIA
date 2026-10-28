"""Simple view of the Geometry and facies tab for registered series.

Four steps sit beside the raw baseline photograph of the reference run:

1. Rig frame, injection points and analysis boxes (the rig frame panel).
2. Mark layers: rough polygons (and brush strokes) per layer on the photograph.
3. Propose boundaries: a pixel classifier trained on the marks labels the rest
   of the photograph inside the rig frame and places the boundaries near the
   polygon edges; adjust polygons or paint corrections and propose again.
4. Facies and export: tick the layers of each facies and give the facies a
   role, porosity and permeability, then save a geometry revision and export it
   for native Setup.

The full geometry-and-facies page remains available as the advanced view.
"""

from __future__ import annotations

import hashlib
from html import escape
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from . import geometry_workspace_frame as workspace_frame
from .rig_point_review import load_review as load_point_review
from typing import Any

import numpy as np
from shiboken6 import isValid
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
    QTransform,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as lpm
from darsia.fluidflower import painting_layer_priority as layer_priority
from darsia.fluidflower import layer_painting_polygons_only as lpp
from darsia.fluidflower import layer_painting_version_archive as lpv
from darsia.fluidflower import painting_facies_approval as facies_approval
from darsia.fluidflower.painting_native_cache import ASYNC_PIXELS, NativeProposalCache
from darsia.fluidflower.layer_painting_join import (
    JoinEdgesError, JoinPreview, join_to_neighbour,
)
from darsia.fluidflower.image_formats import image_family, photo_dialog_filter
from darsia.fluidflower.project import utc_now
from darsia.fluidflower.series_geometry import (
    RigFrame,
    SeriesRunInputs,
    default_actor,
    export_native_setup,
    photo_size,
    rig_frame_from_inputs,
    saved_rig_frames,
)

STROKE_OPACITY = 0.75
# A polygon point is hit within this many screen pixels.
_VERTEX_HIT_PX = 8.0
# A right press that moves less than this (screen pixels) is a click.
_CLICK_SLOP_PX = 4.0
# Point handles of the Edit tool (screen pixels); selected points are larger.
_HANDLE_PX = 7.0
_SELECTED_HANDLE_PX = 11.0
# Round handles: a point that several polygons share (Edit), and the points of
# a new polygon that snapped onto a shown polygon or follow its outline (Draw).
_SHARED_HANDLE_PX = 9.0
_FOLLOWED_HANDLE_PX = 5.0
# Held, this key makes a Draw click place a free point, and an Edit press on
# a shared point take only the point of the polygon pressed into.
_ALONE = Qt.KeyboardModifier.AltModifier
# Screen pixels that an arrow key moves the selected points (with Shift).
_NUDGE_PX = 1.0
_NUDGE_SHIFT_PX = 10.0
# A reference photograph at least this large (a camera RAW or a full-size
# JPEG) is decoded in the background; a smaller one, such as a fixture, at once.
_BACKGROUND_DECODE_BYTES = 8 * 1024 * 1024
# Arrow keys of the Edit tool: (rows, columns) per step.
_NUDGE_KEYS = {
    Qt.Key.Key_Left: (0, -1),
    Qt.Key.Key_Right: (0, 1),
    Qt.Key.Key_Up: (-1, 0),
    Qt.Key.Key_Down: (1, 0),
}
_TOO_FEW_POINTS = (
    "A polygon needs at least three points; use Delete polygon in its "
    "right-click menu."
)
_BRUSH_STEPS = 100
_BRUSH_MIN_FRACTION = 0.0005
_BRUSH_MAX_FRACTION = 0.05
_BRUSH_DEFAULT_FRACTION = 0.006
_UNDO_LIMIT = 200
# Tools that draw a path while the button is held, and those of them with a brush.
_STROKE_TOOLS = frozenset({"paint", "erase", "correct", "restore", "lasso"})
_BRUSH_TOOLS = frozenset({"paint", "erase", "correct", "restore"})
# Tools that correct the proposal by hand; they never change the marks.
_CORRECTION_TOOLS = frozenset({"correct", "restore", "lasso", "fill"})
# Tools that draw on the reference photo.
_DRAWING_TOOLS = frozenset({"polygon", "edit", "paint", "erase"}) | _CORRECTION_TOOLS
# Outline of the corrected areas on the proposal (RGBA).
_CORRECTED_EDGE = (255, 196, 0, 255)
# Polygons, strokes, corrections and priority, as kept for undo and redo.
# An optional fifth field holds the facies property rows and permeability unit.
_Snapshot = tuple
# Column of the layer table whose box shows or hides a layer (display only).
_SHOW_COLUMN = 4
_TOP_COLUMN = 5
_TOP_TIP = "Always on top in the drawing and manual map. Keeps the polygons underneath unchanged; uses Manual map while enabled."
_SHOW_TIP = "Show or hide this layer using its own colour (By layer, display only)"
# Another photograph of the reference run shown under the marks (display only).
_BACKGROUND_TIP = (
    "Photograph shown under the marks. Choose another photo of the same run and "
    "camera position, for example one taken before the indicator was added. "
    "Display only: the proposal always uses the reference photo. T switches "
    "between the reference photo and the chosen photo."
)
_BACKGROUND_SETTINGS = ("DarSIA", "FluidFlower layer painting")
# Shown while the version library or the reference point review is read in
# the background, which takes seconds on a series share.
_VERSIONS_READING = "Reading the saved versions…"
_POINT_REVIEW_CHECKING = "Checking the reference point review…"
# How long a slow reference point check's answer is shown without checking again.
_SLOW_CHECK_FRESH_S = 10.0
# The measured geometry index last used to place a map in all runs.
_GEOMETRY_INDEX_KEY = "placement/geometry_index"
# The runs last chosen to review first after the reference run.
_FIRST_RUNS_KEY = "placement/first_runs"
_BACKGROUND_CHOOSE = "choose"
# Whether Draw follows the edges of shown polygons: the user's choice, kept in
# the same settings and not in the painting session.
_FOLLOW_EDGES_KEY = "draw/follow_edges"
_FOLLOW_EDGES_TIP = (
    "Draw: a click near a point or edge of a shown polygon snaps onto it, and "
    "between two such clicks on one polygon the new polygon follows its "
    "outline, the shorter way round (Tab: the other way). Neighbouring layers "
    "then share their boundary exactly. Alt+click places a free point."
)
# A session painted on the raw reference photo, in a project whose geometry
# is in the saved corrected rig frame, stays on the raw photo.
_RAW_SESSION_NOTE = (
    "Painting on the raw reference photo, where these marks were drawn. The "
    "project's saved corrected rig maps are not shown here."
)
_RAW_SESSION_EXPORT_NOTE = (
    "This painting is on the raw reference photo, while the project's geometry "
    "is in the saved corrected rig frame. Exporting it into the project is a "
    "separate step."
)
# Short role names for the facies table.
_ROLE_SHORT = {"porous_medium": "Porous", "free_water": "Free water"}
_CANVAS_HINT = (
    "Wheel: zoom · right or middle drag: pan · Ctrl+Z / Ctrl+Y: undo / redo. "
    "On the photo: L draw · V edit · B paint · E erase · H pan · [ ] brush "
    "size · 1-9 layer · P propose · F fit · on the proposal: C correct · "
    "R restore · A lasso · G fill"
)
_DRAFT_HINT = (
    "Polygon with {count} point{plural}: left click adds a point · right click, "
    "Enter or a double click closes · Backspace removes the last point · Esc "
    "cancels"
)
_FOLLOW_HINT = " · Tab: follow the other way round · Alt+click: a free point"
_MOVE_SHARED_TIP = (
    "Edges that polygons of different layers share, for example drawn with "
    "Follow edges. Yes: the model places them within the snap distance, like "
    "other edges. No: the boundary lies exactly on the shared edge, also in "
    "the exported map; other edges keep their snap distance."
)
# How saturated the photograph is shown, in percent: the user's choice, kept
# with the background choice and not in the painting session.
_SATURATION_KEY = "display/saturation"
_COLOR_MODE_KEY = "display/color_mode"
_SATURATION_TIP = (
    "Colour saturation of the photograph shown, from 0 % (grey) to 100 % (its "
    "own colours). Grey can make layer boundaries easier to trace. Display "
    "only: the proposal, the session and the export always use the photograph "
    "itself."
)


def _facies_color(name: str) -> QColor:
    """A stable display colour, independent of layer order and saved colours."""
    digest = hashlib.sha256(name.encode("utf8")).digest()
    return QColor.fromHsv(int.from_bytes(digest[:2], "big") % 360,
                         180 + digest[2] % 45, 205 + digest[3] % 40)


# The manual map: the polygons exactly as drawn, without a classifier.
_MANUAL = lpm.MANUAL_MAP_KIND
_MAP_SHOWN_TIP = (
    "The map drawn over the photograph and saved by 'Save version…': the proposal "
    "with its corrections, or the manual map, which is the polygons exactly as "
    "drawn at full resolution, without a classifier"
)
# Manual-map pixels inside the rig frame that no polygon covers, pixels the
# frame-edge fill gave a layer (hatched), and where the manual map and the
# proposal disagree (RGBA).
_UNCOVERED_COLOR = (255, 0, 255, 230)
_EDGE_FILL_COLOR = (255, 255, 255, 150)
_DIFFERENCE_COLOR = (230, 25, 25, 215)


def _any_pixel(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A mask on a coarser grid of ``size`` (width, height), true in each cell
    that holds a true pixel, so that thin areas stay visible."""

    import cv2

    width, height = int(size[0]), int(size[1])
    if mask.shape == (height, width):
        return mask.copy()
    area = cv2.resize(
        mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
    )
    return area > 0.0


def _muted(widget: QWidget) -> QColor:
    palette = widget.palette()
    foreground = palette.color(QPalette.ColorRole.WindowText)
    background = palette.color(QPalette.ColorRole.Window)
    return QColor(
        int(0.55 * foreground.red() + 0.45 * background.red()),
        int(0.55 * foreground.green() + 0.45 * background.green()),
        int(0.55 * foreground.blue() + 0.45 * background.blue()),
    )


def _state_color(widget: QWidget, state: str) -> QColor:
    """Status colours that stay legible in light and dark themes."""

    base = {
        "done": QColor(46, 139, 87),
        "warn": QColor(201, 138, 20),
        "error": QColor(196, 60, 50),
    }.get(state)
    if base is None:
        return _muted(widget)
    foreground = widget.palette().color(QPalette.ColorRole.WindowText)
    return QColor(
        int(0.7 * base.red() + 0.3 * foreground.red()),
        int(0.7 * base.green() + 0.3 * foreground.green()),
        int(0.7 * base.blue() + 0.3 * foreground.blue()),
    )


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("secondaryText")
    return label


def _swatch(color: str, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


def photo_with_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    """An RGB uint8 photograph shown with less colour saturation.

    ``saturation`` 1 (or more) returns ``rgb`` itself, unchanged; 0 (or less)
    gives its grey, the luma with OpenCV's RGB-to-grey weights in all three
    channels; a value between blends the two. The result is uint8 RGB of the
    same shape. For display only.
    """

    import cv2

    value = min(max(float(saturation), 0.0), 1.0)
    if value >= 1.0:
        return rgb
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if value <= 0.0:
        return grey
    return cv2.addWeighted(image, value, grey, 1.0 - value, 0.0)


def _parse_number(text: str) -> float | None:
    cleaned = str(text).replace(",", ".").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _fit_table_height(table: QTableWidget, *, max_rows: int = 8) -> None:
    """Make a table as tall as its rows (up to ``max_rows``)."""

    rows = min(table.rowCount(), max_rows)
    height = table.horizontalHeader().height() + 2 * table.frameWidth() + 2
    for row in range(rows):
        height += table.rowHeight(row)
    if table.rowCount() == 0:
        height += 24
    table.setFixedHeight(height)


def insert_point_on_nearest_edge(
    points: Sequence[tuple[float, float]], row: float, col: float
) -> tuple[list[tuple[float, float]], int]:
    """The points of a closed polygon with one point added on its nearest edge.

    Every edge counts, also the closing edge from the last point back to the
    first. The new point is where (row, col) projects at right angles onto the
    edge nearest to it, kept between the edge's two end points; it is inserted
    between those two points. Returns the new list of points and the index of
    the new point. With fewer than two points, or a position that is not a
    finite number, the points come back unchanged with index -1. ``points``
    itself is not changed.
    """

    vertices = [(float(point[0]), float(point[1])) for point in points]
    row, col = float(row), float(col)
    count = len(vertices)
    if count < 2 or not (math.isfinite(row) and math.isfinite(col)):
        return vertices, -1
    best: tuple[float, int, tuple[float, float]] | None = None
    for index in range(count):
        start = vertices[index]
        end = vertices[(index + 1) % count]
        delta_row, delta_col = end[0] - start[0], end[1] - start[1]
        length2 = delta_row * delta_row + delta_col * delta_col
        fraction = 0.0
        if length2 > 0.0:
            fraction = (
                (row - start[0]) * delta_row + (col - start[1]) * delta_col
            ) / length2
        if fraction <= 0.0:
            point = start
        elif fraction >= 1.0:
            point = end
        else:
            point = (start[0] + fraction * delta_row, start[1] + fraction * delta_col)
        distance = math.hypot(row - point[0], col - point[1])
        if best is None or distance < best[0]:
            best = (distance, index, point)
    assert best is not None
    _distance, edge, point = best
    vertices.insert(edge + 1, point)
    return vertices, edge + 1


def _segment_distance(point: QPointF, start: QPointF, end: QPointF) -> float:
    """Distance from a point to the segment between two points."""

    delta_x, delta_y = end.x() - start.x(), end.y() - start.y()
    length2 = delta_x * delta_x + delta_y * delta_y
    fraction = 0.0
    if length2 > 0.0:
        fraction = (
            (point.x() - start.x()) * delta_x + (point.y() - start.y()) * delta_y
        ) / length2
        fraction = min(max(fraction, 0.0), 1.0)
    return math.hypot(
        point.x() - (start.x() + fraction * delta_x),
        point.y() - (start.y() + fraction * delta_y),
    )


def _ways_round(
    points: Sequence[tuple[float, float]],
    start: tuple[int, float],
    end: tuple[int, float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The points of a closed outline passed on the way from one place on it
    to another, going forwards and going backwards.

    A place is (edge, fraction): on the edge from point ``edge`` to the next
    point, ``fraction`` of the way along (0: on point ``edge`` itself).
    """

    count = len(points)
    first = start[0] + start[1]
    last = end[0] + end[1]
    ways = []
    for step in (1, -1):
        span = ((last - first) * step) % count
        if step > 0:
            vertex = start[0] + 1
        else:
            vertex = start[0] if start[1] > 0.0 else start[0] - 1
        passed = []
        for _number in range(count):
            if ((vertex - first) * step) % count >= span - 1.0e-9:
                break
            row, col = points[vertex % count]
            passed.append((float(row), float(col)))
            vertex += step
        ways.append(passed)
    return ways[0], ways[1]


def insert_point_on_shared_edge(
    polygons: Mapping[int, Sequence[tuple[float, float]]],
    index: int,
    row: float,
    col: float,
    *,
    tolerance: float = lp.SHARED_POINT_PX,
) -> dict[int, tuple[list[tuple[float, float]], int]]:
    """One point added on the nearest edge of polygon ``index`` and on that
    edge in every other polygon that shares it.

    ``polygons`` maps polygon indices to their points. The point goes where
    :func:`insert_point_on_nearest_edge` puts it in polygon ``index``. Another
    polygon gets the same point when one of its edges joins two points within
    ``tolerance`` raw pixels of that edge's end points. Returns {polygon:
    (new points, index of the new point)}, empty when no point can be added.
    ``polygons`` itself is not changed.
    """

    points, inserted = insert_point_on_nearest_edge(polygons[index], row, col)
    if inserted < 0:
        return {}
    changes = {int(index): (points, inserted)}
    point = points[inserted]
    ends = (points[inserted - 1], points[(inserted + 1) % len(points)])

    def near(first: Sequence[float], second: Sequence[float]) -> bool:
        return math.hypot(first[0] - second[0], first[1] - second[1]) <= tolerance

    for other, other_points in polygons.items():
        count = len(other_points)
        if other == index or count < 2:
            continue
        for vertex in range(count):
            start, end = other_points[vertex], other_points[(vertex + 1) % count]
            if (near(start, ends[0]) and near(end, ends[1])) or (
                near(start, ends[1]) and near(end, ends[0])
            ):
                new_points = [
                    (float(value[0]), float(value[1])) for value in other_points
                ]
                new_points.insert(vertex + 1, point)
                changes[int(other)] = (new_points, vertex + 1)
                break
    return changes


def _detail_photo(path, camera_u, display_shape):
    """The photograph at up to ``DETAIL_LIMIT_PX`` as uint8 RGB, for zooming in.

    None when it would be no larger than the display copy of ``display_shape``,
    or when it cannot be read. Display only: the marks, the proposal and saving
    never use it.
    """
    from darsia.fluidflower.photograph_io import load_reduced_rgb8

    try:
        width, height = photo_size(Path(path), camera_u=camera_u)
        factor = max(1, int(math.ceil(max(width, height) / lp.DETAIL_LIMIT_PX)))
        detail = load_reduced_rgb8(Path(path), factor, camera_u=camera_u)
    except (OSError, ValueError):
        return None
    rows, cols = display_shape[:2]
    if detail.shape[0] <= rows or detail.shape[1] <= cols:
        return None
    return detail


def _decode_reference_photo(path, camera_u):
    """Decode a reference photograph for painting and its zoom detail; no Qt is touched."""
    image = lp.load_painting_image(Path(path), camera_u=camera_u)
    return image, _detail_photo(image.path, image.camera_u, image.display.shape)


def _decode_background_photo(path, camera_u, size, display_shape):
    """A background photograph at the display copy's size and its zoom detail; no Qt is touched.

    ``size`` is the photograph's (width, height). The display copy is float32
    RGB in [0, 1] of the reference's ``display_shape``.
    """
    from darsia.fluidflower.photograph_io import load_reduced

    shape = display_shape[:2]
    factor = max(1, int(math.ceil(max(size) / lp.DISPLAY_LIMIT_PX)))
    rgb = np.asarray(load_reduced(Path(path), factor, camera_u=camera_u), dtype=np.float32)
    if rgb.shape[:2] != shape:
        import cv2

        rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.clip(rgb, 0.0, 1.0), _detail_photo(path, camera_u, display_shape)


def _decode_frame_photo(path, camera_u):
    """Another photograph reduced for display, and its (width, height); no Qt is touched.

    The display copy is float RGB in [0, 1], reduced to at most
    ``DISPLAY_LIMIT_PX`` on its longer side.
    """
    from darsia.fluidflower.photograph_io import load_reduced

    width, height = photo_size(Path(path), camera_u=camera_u)
    factor = max(1, int(math.ceil(max(width, height) / lp.DISPLAY_LIMIT_PX)))
    return load_reduced(Path(path), factor, camera_u=camera_u), (int(width), int(height))


@dataclass
class _DraftPoint:
    """A point clicked while drawing a polygon."""

    # Raw (row, col): where it was clicked, or the point or edge position of a
    # shown polygon it snapped onto.
    point: tuple[float, float]
    snapped: bool = False
    # Ways round a shown polygon's outline from the previous point, as the
    # outline points passed (raw), shortest first; and the way taken.
    ways: list[list[tuple[float, float]]] = field(default_factory=list)
    way: int = 0

    def followed(self) -> list[tuple[float, float]]:
        """The outline points followed on the way to this point."""

        return list(self.ways[self.way]) if self.ways else []


@dataclass
class _PointDrag:
    """Selected points being dragged with the Edit tool."""

    # Viewport position and raw (row, col) of the press.
    press: QPointF
    start: tuple[float, float]
    # Points of every polygon with selected points, before and while dragging.
    before: dict[int, list[tuple[float, float]]]
    after: dict[int, list[tuple[float, float]]]
    # The mouse has moved further than a click.
    moving: bool = False


# ---------------------------------------------------------------------------
# Canvas


class LayerCanvas(QGraphicsView):
    """Photograph with polygons, painted strokes and the proposed layers on top.

    The tools are ``pan``, ``polygon`` (the Draw button), ``edit``, ``paint``
    and ``erase``, and the correction tools ``correct``, ``restore``, ``lasso``
    and ``fill``. The right or middle button drags the view in every tool.

    Draw (``polygon``) only draws new polygons: a left click adds a point, a
    right click (without dragging), Enter or a double click closes the
    polygon, Backspace removes the last point and Esc cancels it. Existing
    polygons show no point handles here, and a click on one of their points
    adds a point to the new polygon; it never moves the existing point.

    With ``follow_edges`` (the default) a Draw click within the point-hit
    distance of a point of a shown polygon snaps onto that point, else onto
    the nearest position on an edge within that distance; Alt places a free
    point. When two consecutive clicked points lie on one shown polygon's
    outline, the new polygon follows the outline between them and takes its
    points, the shorter way round; Tab takes the other way (Shift+Tab back),
    and the preview shows the way the next click would follow. Backspace
    removes the last clicked point with the outline followed to it. When the
    polygon is closed, its points that snapped onto an edge are added to the
    polygons whose outline they lie on: just before ``polygon_finished``,
    ``finished_outline_points`` holds {index: new points} of those polygons,
    so that the new polygon and their new points are one change.

    Edit (``edit``) changes existing polygons and never starts a polygon, a
    stroke or a draft. Every shown polygon has point handles, and selected
    points (``selected_points``) are drawn larger, white with a dark outline.
    A left press on a point selects only that point, or keeps the whole
    selection when the point is already selected; Ctrl+click adds the point
    to the selection or takes it out. Dragging moves all selected points by
    the same raw-pixel offset; they stop together at the edge of the
    photograph. A drag on empty space selects the points inside its rectangle
    (Ctrl adds them to the selection); a click there clears the selection.
    Delete or Backspace removes the selected points, but a polygon that would
    keep fewer than three points is left unchanged and ``notice`` says so.
    The arrow keys move the selected points by one screen pixel (Shift: ten)
    and Esc clears the selection. A double click on an edge adds a point there
    and selects it. Each of these changes is reported once, so that it is
    undone in one step: ``polygon_edited`` (index and new points) when one
    polygon changed, ``polygons_edited`` ({index: new points}) when several
    did.

    Points of shown polygons at one position (within ``lp.SHARED_POINT_PX``)
    are one shared point, with a round handle. A click selects all of them,
    so that dragging, nudging and deleting act on all of them; an Alt+press
    selects and drags only the point of the polygon pressed into, which is
    then no longer shared. A double click on an edge that several polygons
    share adds the point to all of them.

    A right click on a polygon asks for its menu (``polygon_context``: the
    polygon, the point under the click or -1, and the screen position); with
    the Edit tool, a click near an edge counts too, also just outside the
    polygon. Just before that, ``last_context_raw`` is set to the raw
    (row, col) of the click, so that the menu can add a point to the polygon
    on its nearest edge.

    ``set_hidden_layers`` hides the polygons and strokes of some layers; this
    changes only what is shown. Polygons of hidden layers are not drawn, get
    no point handles and cannot be selected, dragged or right-clicked; hiding
    a layer drops its points from the selection. Strokes of hidden layers are
    left out of the stroke image; eraser strokes still apply to the strokes
    that are shown. ``set_overlays_visible`` (all marks, the proposal) works
    on top of this.

    The correction tools draw on top of the proposal and never change the
    marks: ``correct`` and ``restore`` are brushes, ``lasso`` draws a closed
    outline while the button is held, and ``fill`` reports a click.
    ``correction_drawn`` gives the finished ``lp.Correction`` (without the
    proposal it was drawn on and its time) and ``fill_requested`` the raw
    (row, col) of a Fill click.

    The canvas also serves the rig frame panel: ``image_point_selected`` gives
    clicks in place mode as fractions of the photograph, ``preview`` identifies
    the photograph shown and ``set_model`` shows another one.

    ``set_photo`` may also get the photograph at a higher resolution
    (``detail``, normally its raw pixels). It is shown over the display copy
    whenever one of its pixels covers at least half a screen pixel, so that
    zooming in shows the photograph's own pixels; the display copy, which sets
    the scene grid for the marks, strokes and the proposal, is shown when
    zoomed out. Display only.
    """

    # The detail is shown from this many screen pixels per detail pixel.
    DETAIL_FROM = 0.5

    image_point_selected = Signal(float, float)
    # Original camera-U pixels, including the full sensor margin outside U>=0.
    raw_point_selected = Signal(float, float)
    stroke_finished = Signal(object)
    polygon_finished = Signal(object)
    # Index of an edited polygon and its new points (raw pixels).
    polygon_edited = Signal(int, object)
    # Several polygons changed in one step: {index: new points (raw pixels)}.
    polygons_edited = Signal(object)
    # Polygon index, point index (-1: not on a point) and global position.
    polygon_context = Signal(int, int, QPoint)
    # Points of the polygon being drawn (0: none).
    draft_changed = Signal(int)
    notice = Signal(str)
    # A hand correction drawn with Correct, Restore or Lasso (an lp.Correction).
    correction_drawn = Signal(object)
    # The raw (row, col) clicked with Fill.
    fill_requested = Signal(float, float)

    TOOLS = (
        "pan",
        "polygon",
        "edit",
        "paint",
        "erase",
        "correct",
        "restore",
        "lasso",
        "fill",
    )

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        # The photograph at a higher resolution, over the display copy.
        self.detail_item = QGraphicsPixmapItem()
        self.detail_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self.detail_item.setZValue(1.0)
        self.detail_item.hide()
        self.proposal_item = QGraphicsPixmapItem()
        self.proposal_item.setZValue(5.0)
        self.stroke_item = QGraphicsPixmapItem()
        self.stroke_item.setZValue(10.0)
        self.stroke_item.setOpacity(STROKE_OPACITY)
        self.active_item = QGraphicsPathItem()
        self.active_item.setZValue(12.0)
        self.active_item.setOpacity(STROKE_OPACITY)
        self.draft_item = QGraphicsPathItem()
        self.draft_item.setZValue(14.0)
        self.rubber_item = QGraphicsPathItem()
        self.rubber_item.setZValue(14.0)
        # The rectangle dragged on empty space to select points (Edit tool).
        self.band_item = QGraphicsRectItem()
        self.band_item.setZValue(25.0)
        band_pen = QPen(QColor(255, 255, 255, 230), 1.0, Qt.PenStyle.DashLine)
        band_pen.setCosmetic(True)
        self.band_item.setPen(band_pen)
        self.band_item.setBrush(QColor(255, 255, 255, 40))
        self.band_item.hide()
        self.cursor_item = QGraphicsEllipseItem()
        self.cursor_item.setZValue(30.0)
        cursor_pen = QPen(QColor(255, 255, 255, 230), 1.5)
        cursor_pen.setCosmetic(True)
        self.cursor_item.setPen(cursor_pen)
        self.cursor_item.hide()
        # Where a Draw click would snap to (follow edges).
        self.snap_item = QGraphicsEllipseItem(-6.0, -6.0, 12.0, 12.0)
        self.snap_item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
        )
        snap_pen = QPen(QColor(255, 255, 255, 240), 2.0)
        snap_pen.setCosmetic(True)
        self.snap_item.setPen(snap_pen)
        self.snap_item.setZValue(29.0)
        self.snap_item.hide()
        for item in (
            self.pixmap_item,
            self.detail_item,
            self.proposal_item,
            self.stroke_item,
            self.active_item,
            self.draft_item,
            self.rubber_item,
            self.band_item,
            self.cursor_item,
            self.snap_item,
        ):
            self.scene().addItem(item)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setBackgroundBrush(QColor("#090b0d"))
        self.setMinimumSize(480, 360)

        self.preview: dict[str, Any] | None = None
        self.place_mode = False
        self.tool = "pan"
        self.paint_enabled = False
        self.layer_id = 0
        self.layer_color = QColor("#ffffff")
        self.radius_px = 20.0
        self._raw_size: tuple[int, int] | None = None
        self.photo_display_mapping = None
        self._photo_display_size = None
        self._photo_valid_support = None
        self._display_shape: tuple[int, int] | None = None
        self._stroke_image: QImage | None = None
        self._points: list[tuple[float, float]] = []
        self._path: QPainterPath | None = None
        self._pan_origin: QPointF | None = None
        self._right_press: QPointF | None = None
        self._right_dragged = False
        self._user_zoomed = False
        self._photo_scrollbar_policies = None
        self._marks_visible = True
        self._polygon_fill = True
        self._polygons: list[lp.Polygon] = []
        self._polygon_colors: dict[int, QColor] = {}
        self._polygon_items: list[QGraphicsPathItem] = []
        # Point handles of each polygon (Edit tool only).
        self._handle_items: list[list[QGraphicsItem]] = []
        self._draft: list[tuple[float, float]] = []
        self._draft_handles: list[QGraphicsItem] = []
        # The points clicked for the polygon being drawn; _draft holds them
        # with the outline points followed between them.
        self._clicks: list[_DraftPoint] = []
        # Draw snaps onto and follows the edges of shown polygons.
        self.follow_edges = True
        # Set only while polygon_finished is emitted (see the class notes).
        self.finished_outline_points: dict[int, list[tuple[float, float]]] = {}
        # The last viewport position over the canvas while drawing, and whether
        # Alt was held, for the preview.
        self._draw_hover: tuple[QPointF, bool] | None = None
        # Edit tool: the selected points as (polygon index, point index), a
        # drag of them in progress and where a rubber band started (viewport).
        self._selection: set[tuple[int, int]] = set()
        # Shared points of the shown polygons: the (polygon, point) indices of
        # each point's shared point.
        self._shared: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        self._drag: _PointDrag | None = None
        self._band_origin: QPointF | None = None
        # Layers whose polygons and strokes are not shown (display only).
        self._hidden_layers: frozenset[int] = frozenset()
        # The strokes last rendered, to draw them again when layers are hidden.
        self._strokes: list[lp.Stroke] = []
        self._stroke_colors: dict[int, QColor] = {}
        # Raw (row, col) of the last right click that opened a polygon menu.
        self.last_context_raw: tuple[float, float] | None = None

    # ----------------------------------------------------------- geometry

    def _scale(self) -> tuple[float, float]:
        """Display pixels per raw pixel along rows and columns."""

        if self._raw_size is None or self._display_shape is None:
            return 1.0, 1.0
        width, height = self._photo_display_size or self._raw_size
        return self._display_shape[0] / height, self._display_shape[1] / width

    def raw_to_scene(self, row: float, col: float) -> QPointF:
        sy, sx = self._scale()
        if self.photo_display_mapping is not None:
            col, row = self.photo_display_mapping.camera_u_to_display_xy([[col, row]])[0]
        return QPointF((col + 0.5) * sx - 0.5, (row + 0.5) * sy - 0.5)

    def scene_to_raw(self, point: QPointF) -> tuple[float, float]:
        sy, sx = self._scale()
        row, col = (point.y() + 0.5) / sy - 0.5, (point.x() + 0.5) / sx - 0.5
        if self.photo_display_mapping is not None:
            col, row = self.photo_display_mapping.display_to_camera_u_xy([[col, row]])[0]
        return float(row), float(col)

    def display_radius(self, radius_px: float | None = None) -> float:
        sy, sx = self._scale()
        return (
            float(radius_px if radius_px is not None else self.radius_px)
            * 0.5
            * (sx + sy)
        )

    @property
    def raw_size(self) -> tuple[int, int] | None:
        return self._raw_size

    def _clamped_scene(self, point: QPointF) -> QPointF:
        bounds = self.pixmap_item.boundingRect()
        return QPointF(
            min(max(point.x(), 0.0), max(0.0, bounds.width() - 1.0)),
            min(max(point.y(), 0.0), max(0.0, bounds.height() - 1.0)),
        )

    # ------------------------------------------------------------- photo

    def set_photo(
        self,
        display: np.ndarray,
        raw_size: tuple[int, int],
        *,
        token: dict[str, Any],
        reset_view: bool,
        detail: np.ndarray | None = None,
        display_mapping=None,
        display_size=None,
        valid_support=None,
    ) -> None:
        """Show an RGB uint8 image that represents a raw photograph.

        ``detail`` is the same photograph at a higher resolution, shown when
        zoomed in (see the class notes); ``None`` shows the display copy only.
        """

        image = np.ascontiguousarray(display, dtype=np.uint8)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        resized = self._display_shape != (height, width)
        self._display_shape = (height, width)
        self._raw_size = (int(raw_size[0]), int(raw_size[1]))
        self.photo_display_mapping = display_mapping
        self._photo_display_size = display_size
        self._photo_valid_support = valid_support
        self.preview = token
        self._set_detail(detail)
        if resized or self._stroke_image is None:
            self._stroke_image = QImage(
                width, height, QImage.Format.Format_ARGB32_Premultiplied
            )
            self._stroke_image.fill(Qt.GlobalColor.transparent)
            self.stroke_item.setPixmap(QPixmap())
            self.proposal_item.setPixmap(QPixmap())
        self._position_proposal()
        if self._photo_scrollbar_policies is None:
            self._photo_scrollbar_policies = (
                self.horizontalScrollBarPolicy(), self.verticalScrollBarPolicy()
            )
            # A photo plus half a viewport of margin always needs both bars.
            # Keeping their geometry stable also prevents Fit/resize feedback.
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        if reset_view:
            self.fit_image()
        else:
            self._update_pan_margins()
        self.refresh_detail()

    def _set_detail(self, detail: np.ndarray | None) -> None:
        """Place ``detail`` on the scene grid of the display copy."""

        if detail is None or self._display_shape is None:
            self.detail_item.setPixmap(QPixmap())
            self.detail_item.hide()
            return
        image = np.ascontiguousarray(detail, dtype=np.uint8)
        rows, cols = image.shape[:2]
        qimage = QImage(
            image.data, cols, rows, int(image.strides[0]), QImage.Format.Format_RGB888
        ).copy()
        self.detail_item.setPixmap(QPixmap.fromImage(qimage))
        # Each detail pixel centre lands where raw_to_scene puts that point of
        # the photograph, so the marks sit on the pixels they were drawn on.
        height, width = self._display_shape
        self.detail_item.setTransform(QTransform.fromScale(width / cols, height / rows))
        self.detail_item.setPos(-0.5, -0.5)

    @property
    def detail_shown(self) -> bool:
        """Whether the higher-resolution photograph is shown now."""

        return self.detail_item.isVisible()

    def refresh_detail(self) -> None:
        """Show the detail when zoomed in far enough, else the display copy."""

        pixmap = self.detail_item.pixmap()
        if pixmap.isNull() or self._display_shape is None:
            self.detail_item.hide()
            return
        scene_per_detail = self._display_shape[1] / pixmap.width()
        screen_per_scene = abs(self.transform().m11())
        self.detail_item.setVisible(
            screen_per_scene * scene_per_detail >= self.DETAIL_FROM
        )

    def set_model(self, preview: Mapping[str, Any], _config: Mapping[str, Any]) -> None:
        """Show ``preview["baseline"]`` (float RGB in [0, 1] or uint8)."""

        rgb = np.asarray(preview["baseline"])
        if rgb.dtype != np.uint8:
            rgb = np.rint(255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        size = preview.get("raw_size") or (rgb.shape[1], rgb.shape[0])
        self.set_photo(rgb, size, token=dict(preview), reset_view=True)

    def clear_photo(self) -> None:
        """Show nothing (a run without a readable baseline photograph)."""

        for item in (self.pixmap_item, self.detail_item, self.proposal_item, self.stroke_item):
            item.setPixmap(QPixmap())
        self.detail_item.hide()
        self._cancel_stroke()
        self.cancel_polygon()
        self._end_drag()
        self._cancel_band()
        self._selection = set()
        self.render_polygons([], {})
        self._strokes = []
        self._stroke_colors = {}
        self._display_shape = None
        self._raw_size = None
        self.photo_display_mapping = None
        self._photo_display_size = None
        self._photo_valid_support = None
        self._stroke_image = None
        self._position_proposal()
        self.preview = None
        self.paint_enabled = False
        self.cursor_item.hide()
        self.scene().setSceneRect(QRectF())
        if self._photo_scrollbar_policies is not None:
            horizontal, vertical = self._photo_scrollbar_policies
            self._photo_scrollbar_policies = None
            self.setHorizontalScrollBarPolicy(horizontal)
            self.setVerticalScrollBarPolicy(vertical)

    def _update_pan_margins(self) -> None:
        """Scene-only room to centre any photograph edge at the current zoom.

        The photograph and every mark retain their existing scene coordinates.
        The extra two screen pixels allow for integer scrollbar rounding.
        """
        if self._display_shape is None or self.pixmap_item.pixmap().isNull():
            return
        viewport = self.viewport().rect()
        visible = self.mapToScene(viewport).boundingRect()
        if visible.width() <= 0.0 or visible.height() <= 0.0:
            return
        centre = self.mapToScene(viewport.center())
        origin = self.mapToScene(QPoint(0, 0))
        rounding = self.mapToScene(QPoint(2, 2)) - origin
        margin_x = visible.width() * 0.5 + abs(rounding.x())
        margin_y = visible.height() * 0.5 + abs(rounding.y())
        self.scene().setSceneRect(self.pixmap_item.boundingRect().adjusted(
            -margin_x, -margin_y, margin_x, margin_y
        ))
        self.centerOn(centre)

    def fit_image(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._user_zoomed = False
        self._update_pan_margins()
        self.refresh_detail()
        self._update_cursor()

    def set_overlays_visible(self, marks: bool, proposal: bool) -> None:
        """Show or hide the polygons and strokes, and the proposal."""

        self._marks_visible = bool(marks)
        if not marks:
            self.cancel_polygon()
            self._end_drag()
            self._cancel_band()
            self.snap_item.hide()
        self.stroke_item.setVisible(marks)
        self.active_item.setVisible(marks)
        for item, polygon in zip(self._polygon_items, self._polygons):
            item.setVisible(marks and polygon.layer_id not in self._hidden_layers)
        self._update_handles()
        self.proposal_item.setVisible(proposal)

    @property
    def hidden_layers(self) -> frozenset[int]:
        return self._hidden_layers

    def set_hidden_layers(self, layer_ids: Iterable[int]) -> None:
        """Hide the polygons and strokes of these layers; show the others.

        Display only: the marks themselves are not changed.
        """

        hidden = frozenset(int(value) for value in layer_ids)
        if hidden == self._hidden_layers:
            return
        self._hidden_layers = hidden
        if self._drag is not None and any(
            index < len(self._polygons) and self._polygons[index].layer_id in hidden
            for index in self._drag.before
        ):
            self._end_drag()
        # Points of hidden layers cannot stay selected.
        self._selection = {
            point for point in self._selection if self._selectable(*point)
        }
        for item, polygon in zip(self._polygon_items, self._polygons):
            item.setVisible(self._marks_visible and polygon.layer_id not in hidden)
        self._refresh_shared()
        self._update_handles()
        self._draw_strokes()

    def set_proposal(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self.proposal_item.setPixmap(QPixmap())
            self._position_proposal()
            return
        image = np.ascontiguousarray(rgba, dtype=np.uint8)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        self.proposal_item.setPixmap(QPixmap.fromImage(qimage))
        self._position_proposal()

    def _position_proposal(self) -> None:
        """Place the unchanged camera-U raster over an expanded photograph.

        The proposal may have its own reduced resolution. Map its pixel
        centres through camera-U, rather than scaling it to the full RAW
        rectangle, so the saved mask stays aligned with polygons and strokes.
        """
        self.proposal_item.setTransform(QTransform())
        self.proposal_item.setPos(0.0, 0.0)
        pixmap = self.proposal_item.pixmap()
        if (pixmap.isNull() or self.photo_display_mapping is None
                or self._raw_size is None or self._display_shape is None):
            return
        raw_width, raw_height = self._raw_size
        origin = self.raw_to_scene(-0.5, -0.5)
        along_x = self.raw_to_scene(-0.5, raw_width / pixmap.width() - 0.5) - origin
        along_y = self.raw_to_scene(raw_height / pixmap.height() - 0.5, -0.5) - origin
        self.proposal_item.setTransform(QTransform(
            along_x.x(), along_x.y(), along_y.x(), along_y.y(),
            origin.x(), origin.y(),
        ))

    # ----------------------------------------------------------- strokes

    def _stroke_pen(self, stroke: lp.Stroke, color: QColor) -> QPen:
        width = max(1.0, 2.0 * self.display_radius(stroke.radius_px))
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    def _paint(
        self, painter: QPainter, stroke: lp.Stroke, colors: Mapping[int, QColor]
    ) -> None:
        if stroke.layer_id == lp.ERASER:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            color = QColor(0, 0, 0, 255)
        else:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            color = QColor(colors.get(stroke.layer_id, QColor("#ffffff")))
            color.setAlpha(255)
        points = [self.raw_to_scene(row, col) for row, col in stroke.points]
        if not points:
            return
        if len(points) == 1:
            radius = max(0.5, self.display_radius(stroke.radius_px))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(points[0], radius, radius)
            return
        painter.setPen(self._stroke_pen(stroke, color))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.drawPath(path)

    def _stroke_shown(self, stroke: lp.Stroke) -> bool:
        return (
            stroke.layer_id == lp.ERASER or stroke.layer_id not in self._hidden_layers
        )

    def _draw_strokes(self) -> None:
        """Draw the stored strokes, leaving out those of hidden layers."""

        if self._stroke_image is None:
            return
        self._stroke_image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self._stroke_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for stroke in self._strokes:
            if self._stroke_shown(stroke):
                self._paint(painter, stroke, self._stroke_colors)
        painter.end()
        self.stroke_item.setPixmap(QPixmap.fromImage(self._stroke_image))

    def render_strokes(
        self, strokes: Sequence[lp.Stroke], colors: Mapping[int, QColor]
    ) -> None:
        self._strokes = list(strokes)
        self._stroke_colors = dict(colors)
        self._draw_strokes()

    def add_stroke_to_image(
        self, stroke: lp.Stroke, colors: Mapping[int, QColor]
    ) -> None:
        self._strokes.append(stroke)
        self._stroke_colors = dict(colors)
        if self._stroke_image is None or not self._stroke_shown(stroke):
            return
        painter = QPainter(self._stroke_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint(painter, stroke, colors)
        painter.end()
        self.stroke_item.setPixmap(QPixmap.fromImage(self._stroke_image))

    # ---------------------------------------------------------- polygons

    def _polygon_path(self, points: Sequence[tuple[float, float]]) -> QPainterPath:
        path = QPainterPath()
        if not points:
            return path
        path.moveTo(self.raw_to_scene(*points[0]))
        for row, col in points[1:]:
            path.lineTo(self.raw_to_scene(row, col))
        path.closeSubpath()
        return path

    def _style_polygon(self, item: QGraphicsPathItem, color: QColor) -> None:
        pen = QPen(QColor(color), 2.0)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        item.setPen(pen)
        if self._polygon_fill:
            fill = QColor(color)
            fill.setAlpha(55)
            item.setBrush(fill)
        else:
            item.setBrush(Qt.BrushStyle.NoBrush)

    def set_polygon_fill(self, filled: bool) -> None:
        """Fill the polygons lightly, or draw outlines only (over a proposal)."""

        if bool(filled) == self._polygon_fill:
            return
        self._polygon_fill = bool(filled)
        for item, polygon in zip(self._polygon_items, self._polygons):
            self._style_polygon(
                item, QColor(self._polygon_colors.get(polygon.layer_id, "#ffffff"))
            )

    def render_polygons(
        self, polygons: Sequence[lp.Polygon], colors: Mapping[int, QColor],
        *, always_on_top: Iterable[int] = (),
    ) -> None:
        """Draw the polygons; explicit priority lies above the usual area order."""

        for item in self._polygon_items:
            self.scene().removeItem(item)
        self._polygon_items = []
        self._polygons = list(polygons)
        self._always_on_top = frozenset(always_on_top)
        self._polygon_colors = dict(colors)
        drag = self._drag
        if drag is not None and any(
            index >= len(self._polygons) or list(self._polygons[index].points) != points
            for index, points in drag.before.items()
        ):
            # The polygons changed under the drag: it no longer applies.
            self._drag = drag = None
        # Selected points must still exist and be shown.
        self._selection = {
            point for point in self._selection if self._selectable(*point)
        }
        if self._display_shape is not None:
            areas = [polygon.area() for polygon in self._polygons]
            for index, polygon in enumerate(self._polygons):
                points = (
                    drag.after[index]
                    if drag is not None and index in drag.after
                    else polygon.points
                )
                item = QGraphicsPathItem(self._polygon_path(points))
                self._style_polygon(
                    item, QColor(colors.get(polygon.layer_id, QColor("#ffffff")))
                )
                # Smaller polygons win where layers overlap, so they lie on top.
                rank = sum(1 for other in areas if other > areas[index])
                base = 11.0 if polygon.layer_id in self._always_on_top else 8.0
                item.setZValue(base + min(rank, 999) * 1.0e-3)
                item.setVisible(
                    self._marks_visible and polygon.layer_id not in self._hidden_layers
                )
                self.scene().addItem(item)
                self._polygon_items.append(item)
        else:
            self._polygon_items = []
        self._refresh_shared()
        self._update_handles()

    def _handle(
        self, point: QPointF, color: QColor, *, size: float = 7.0, rounded: bool = False
    ) -> QGraphicsRectItem | QGraphicsEllipseItem:
        shape = QGraphicsEllipseItem if rounded else QGraphicsRectItem
        item = shape(-size / 2.0, -size / 2.0, size, size)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        item.setPos(point)
        pen = QPen(QColor(20, 20, 20, 230), 1.0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QColor(color))
        item.setZValue(20.0)
        self.scene().addItem(item)
        return item

    def _style_handle(
        self,
        item: QGraphicsRectItem | QGraphicsEllipseItem,
        color: QColor,
        selected: bool,
        shared: bool = False,
    ) -> None:
        """A point handle in its layer's colour; larger and white when selected.

        A shared point's handle is round, a little larger, with a white rim.
        """

        size = _SELECTED_HANDLE_PX if selected else _HANDLE_PX
        if shared and not selected:
            size = _SHARED_HANDLE_PX
        item.setRect(-size / 2.0, -size / 2.0, size, size)
        if shared and not selected:
            pen = QPen(QColor(255, 255, 255, 240), 1.5)
        else:
            pen = QPen(
                QColor(20, 20, 20, 255 if selected else 230), 2.0 if selected else 1.0
            )
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QColor(255, 255, 255) if selected else QColor(color))
        item.setZValue(21.0 if selected else 20.0)

    def _update_handles(self) -> None:
        """Point handles for every shown polygon, with the Edit tool only."""

        for handles in self._handle_items:
            for item in handles:
                self.scene().removeItem(item)
        self._handle_items = []
        if (
            self.tool != "edit"
            or not self._marks_visible
            or not self.paint_enabled
            or self._display_shape is None
        ):
            return
        drag = self._drag
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers:
                # No handles, but keep one list per polygon (indexed by polygon).
                self._handle_items.append([])
                continue
            points = (
                drag.after[index]
                if drag is not None and index in drag.after
                else polygon.points
            )
            color = QColor(self._polygon_colors.get(polygon.layer_id, "#ffffff"))
            handles = []
            for vertex, (row, col) in enumerate(points):
                shared = (index, vertex) in self._shared
                item = self._handle(self.raw_to_scene(row, col), color, rounded=shared)
                self._style_handle(
                    item, color, (index, vertex) in self._selection, shared
                )
                handles.append(item)
            self._handle_items.append(handles)

    def _restyle_handles(self, points: Iterable[tuple[int, int]]) -> None:
        """Show whether these points are selected, without making the handles
        again."""

        for index, vertex in points:
            if not (
                0 <= index < min(len(self._handle_items), len(self._polygons))
                and 0 <= vertex < len(self._handle_items[index])
            ):
                continue
            color = QColor(
                self._polygon_colors.get(self._polygons[index].layer_id, "#ffffff")
            )
            self._style_handle(
                self._handle_items[index][vertex],
                color,
                (index, vertex) in self._selection,
                (index, vertex) in self._shared,
            )

    def _hit_vertex(self, position: QPointF) -> tuple[int, int] | None:
        """Polygon and point index of the point under a viewport position."""

        best: tuple[float, int, int] | None = None
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers:
                continue
            for vertex, (row, col) in enumerate(polygon.points):
                spot = self.mapFromScene(self.raw_to_scene(row, col))
                distance = math.hypot(spot.x() - position.x(), spot.y() - position.y())
                if distance <= _VERTEX_HIT_PX and (best is None or distance < best[0]):
                    best = (distance, index, vertex)
        return None if best is None else (best[1], best[2])

    def _hit_polygon(self, position: QPointF) -> int | None:
        """The smallest shown polygon under a viewport position."""

        scene_point = self.mapToScene(position.toPoint())
        hits = [
            index
            for index, item in enumerate(self._polygon_items)
            if self._polygons[index].layer_id not in self._hidden_layers
            and item.path().contains(scene_point)
        ]
        if not hits:
            return None
        return min(hits, key=lambda index: (
            self._polygons[index].layer_id not in self._always_on_top,
            self._polygons[index].area()))

    def _hit_edge(self, position: QPointF) -> int | None:
        """The shown polygon with an edge within the point-hit distance of a
        viewport position: the nearest edge, and the smaller polygon where
        two edges are as near (a shared boundary)."""

        transform = self.viewportTransform()
        best: tuple[float, float, int] | None = None
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers or len(polygon.points) < 2:
                continue
            spots = [
                transform.map(self.raw_to_scene(row, col))
                for row, col in polygon.points
            ]
            distance = min(
                _segment_distance(position, start, spots[(number + 1) % len(spots)])
                for number, start in enumerate(spots)
            )
            if distance <= _VERTEX_HIT_PX:
                key = (distance, polygon.area(), index)
                if best is None or key < best:
                    best = key
        return None if best is None else best[2]

    def polygon_at(self, position: QPointF) -> tuple[int, int] | None:
        """Polygon (and point, or -1) under a viewport position.

        With the Edit tool, a position near an edge counts too, also just
        outside the polygon.
        """

        vertex = self._hit_vertex(position)
        if vertex is not None:
            return vertex
        index = self._hit_polygon(position)
        if index is None and self.tool == "edit":
            index = self._hit_edge(position)
        return None if index is None else (index, -1)

    # ------------------------------------------------------ shared points

    def _refresh_shared(self) -> None:
        """Find the shared points of the shown polygons."""

        shown = {
            index: polygon.points
            for index, polygon in enumerate(self._polygons)
            if polygon.layer_id not in self._hidden_layers
        }
        self._shared = {}
        for group in lp.shared_points(shown):
            members = tuple(group)
            for member in members:
                self._shared[member] = members

    def shared_with(self, index: int, vertex: int) -> list[tuple[int, int]]:
        """The (polygon, point) indices of a shown polygon's point and of the
        points it is shared with (only itself when it is not shared)."""

        point = (int(index), int(vertex))
        return list(self._shared.get(point, (point,)))

    def _with_shared(self, points: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
        chosen: set[tuple[int, int]] = set()
        for index, vertex in points:
            chosen.update(self.shared_with(index, vertex))
        return chosen

    def _pressed_point(
        self, hit: tuple[int, int], position: QPointF
    ) -> tuple[int, int]:
        """Of a shared point, the point of the polygon pressed into: the
        smallest shown polygon that contains the viewport position, else the
        polygon of the selected layer, else the point hit."""

        members = self.shared_with(*hit)
        if len(members) < 2:
            return hit
        scene_point = self.mapToScene(position.toPoint())
        inside = [
            member
            for member in members
            if member[0] < len(self._polygon_items)
            and self._polygon_items[member[0]].path().contains(scene_point)
        ]
        if inside:
            return min(inside, key=lambda member: self._polygons[member[0]].area())
        for member in members:
            if self._polygons[member[0]].layer_id == self.layer_id:
                return member
        return hit

    # ---------------------------------------------------- selecting points

    @property
    def selected_points(self) -> list[tuple[int, int]]:
        """Points selected with the Edit tool: sorted (polygon, point) indices."""

        return sorted(self._selection)

    def _selectable(self, index: int, vertex: int) -> bool:
        return (
            0 <= index < len(self._polygons)
            and 0 <= vertex < len(self._polygons[index].points)
            and self._polygons[index].layer_id not in self._hidden_layers
        )

    def set_selection(self, points: Iterable[tuple[int, int]]) -> None:
        """Select these (polygon, point) indices instead of the current ones.

        Only the Edit tool has a selection; points of hidden layers and points
        that do not exist are left out.
        """

        chosen = {(int(index), int(vertex)) for index, vertex in points}
        if self.tool != "edit":
            chosen = set()
        before = self._selection
        self._selection = {point for point in chosen if self._selectable(*point)}
        self._restyle_handles(before ^ self._selection)

    def clear_selection(self) -> None:
        """Select no point; a drag or a rubber band in progress is cancelled."""

        self._cancel_drag()
        self._cancel_band()
        if self._selection:
            before, self._selection = self._selection, set()
            self._restyle_handles(before)

    def selection_without(
        self, index: int, vertices: Iterable[int]
    ) -> list[tuple[int, int]]:
        """The selection as it will be once these points of polygon ``index``
        are removed: they leave it, and later points of that polygon move up."""

        return self.selection_without_points((index, vertex) for vertex in vertices)

    def selection_without_points(
        self, points: Iterable[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """The selection as it will be once these (polygon, point) points are
        removed: they leave it, and later points of their polygons move up."""

        removed: dict[int, set[int]] = {}
        for index, vertex in points:
            removed.setdefault(int(index), set()).add(int(vertex))
        kept = []
        for polygon, vertex in self._selection:
            gone = removed.get(polygon, set())
            if vertex not in gone:
                earlier = sum(1 for number in gone if number < vertex)
                kept.append((polygon, vertex - earlier))
        return sorted(kept)

    def _selected_by_polygon(self) -> dict[int, set[int]]:
        grouped: dict[int, set[int]] = {}
        for index, vertex in self._selection:
            if self._selectable(index, vertex):
                grouped.setdefault(index, set()).add(vertex)
        return grouped

    def _emit_edits(self, changes: Mapping[int, list[tuple[float, float]]]) -> None:
        """Report changed polygons as one change: ``polygon_edited`` for one
        polygon, ``polygons_edited`` for several."""

        if len(changes) == 1:
            ((index, points),) = changes.items()
            self.polygon_edited.emit(int(index), list(points))
        elif changes:
            self.polygons_edited.emit(
                {int(index): list(points) for index, points in changes.items()}
            )

    def _raw_limits(self) -> tuple[float, float, float, float]:
        """First and last raw row and column that points are kept within (as
        for the points of a polygon being drawn)."""

        bounds = self.pixmap_item.boundingRect()
        top, left = self.scene_to_raw(QPointF(0.0, 0.0))
        bottom, right = self.scene_to_raw(
            QPointF(max(0.0, bounds.width() - 1.0), max(0.0, bounds.height() - 1.0))
        )
        return top, bottom, left, right

    def _clamp_offset(
        self, points: Sequence[tuple[float, float]], rows: float, cols: float
    ) -> tuple[float, float]:
        """Limit an offset so that it moves none of these points out of the
        photograph; points already outside it are not pulled in."""

        if not points:
            return 0.0, 0.0
        top, bottom, left, right = self._raw_limits()
        row_values = [point[0] for point in points]
        col_values = [point[1] for point in points]
        rows = min(
            max(rows, min(0.0, top - min(row_values))),
            max(0.0, bottom - max(row_values)),
        )
        cols = min(
            max(cols, min(0.0, left - min(col_values))),
            max(0.0, right - max(col_values)),
        )
        return rows, cols

    def _moved_points(
        self,
        polygons: Mapping[int, Sequence[tuple[float, float]]],
        rows: float,
        cols: float,
    ) -> dict[int, list[tuple[float, float]]]:
        """The points of these polygons with the selected ones moved by an
        offset (raw pixels), limited to the photograph."""

        selected = self._selected_by_polygon()
        moving = [
            polygons[index][vertex]
            for index, vertices in selected.items()
            if index in polygons
            for vertex in vertices
            if vertex < len(polygons[index])
        ]
        rows, cols = self._clamp_offset(moving, rows, cols)
        moved = {}
        for index, points in polygons.items():
            new_points = list(points)
            for vertex in selected.get(index, ()):
                if vertex < len(new_points):
                    row, col = new_points[vertex]
                    new_points[vertex] = (row + rows, col + cols)
            moved[index] = new_points
        return moved

    def nudge_selected_points(self, screen_rows: float, screen_cols: float) -> bool:
        """Move the selected points by screen pixels as one change (limited to
        the photograph); False when nothing moved."""

        selected = self._selected_by_polygon()
        if not selected:
            return False
        transform = self.transform()
        scale_rows, scale_cols = self._scale()
        zoom_rows = abs(transform.m22()) * scale_rows
        zoom_cols = abs(transform.m11()) * scale_cols
        if zoom_rows <= 0.0 or zoom_cols <= 0.0:
            return False
        before = {index: list(self._polygons[index].points) for index in selected}
        after = self._moved_points(
            before, screen_rows / zoom_rows, screen_cols / zoom_cols
        )
        changes = {
            index: points for index, points in after.items() if points != before[index]
        }
        self._emit_edits(changes)
        return bool(changes)

    def delete_selected_points(self) -> int:
        """Remove the selected points as one change; returns how many went.

        A polygon that would keep fewer than three points is left unchanged,
        its points stay selected and ``notice`` says so; other polygons still
        lose their selected points.
        """

        selection = set(self._selection)
        changes: dict[int, list[tuple[float, float]]] = {}
        refused = False
        for index, vertices in sorted(self._selected_by_polygon().items()):
            kept = [
                point
                for vertex, point in enumerate(self._polygons[index].points)
                if vertex not in vertices
            ]
            if len(kept) < 3:
                refused = True
            else:
                changes[index] = kept
        removed = 0
        if changes:
            self._emit_edits(changes)
            applied = {
                index
                for index, points in changes.items()
                if index < len(self._polygons)
                and list(self._polygons[index].points) == points
            }
            removed = sum(1 for index, _vertex in selection if index in applied)
            self.set_selection(point for point in selection if point[0] not in applied)
        if refused:
            self.notice.emit(_TOO_FEW_POINTS)
        return removed

    def _insert_point_at(self, position: QPointF) -> bool:
        """Add a point on the polygon edge under a viewport position (not on a
        point), also in the shown polygons that share that edge, and select
        it."""

        if self._hit_vertex(position) is not None:
            return False
        index = self._hit_edge(position)
        if index is None:
            return False
        row, col = self.scene_to_raw(self.mapToScene(position.toPoint()))
        shown = {
            number: polygon.points
            for number, polygon in enumerate(self._polygons)
            if polygon.layer_id not in self._hidden_layers
        }
        added = insert_point_on_shared_edge(shown, index, row, col)
        if not added:
            return False
        self._emit_edits(
            {number: points for number, (points, _vertex) in added.items()}
        )
        applied = [
            (number, vertex)
            for number, (points, vertex) in added.items()
            if number < len(self._polygons)
            and list(self._polygons[number].points) == points
        ]
        if applied:
            self.set_selection(applied)
        return bool(applied)

    def _edit_press(self, event: Any) -> None:
        """A left press with the Edit tool: select points, then drag them or
        a rubber band."""

        # A drag or band whose release never came (a dialog took the mouse).
        self._cancel_drag()
        self._cancel_band()
        position = event.position()
        extend = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        hit = self._hit_vertex(position)
        if hit is None:
            if not extend:
                self.set_selection(())
            self._band_origin = QPointF(position)
            return
        selection = set(self._selection)
        if event.modifiers() & _ALONE:
            # Only the pressed polygon's point, to move it off a shared point.
            hit = self._pressed_point(hit, position)
            selection = {hit}
        else:
            # A shared point is selected, and moves, in all its polygons.
            members = set(self.shared_with(*hit))
            if extend:
                selection = (
                    selection - members if members <= selection else selection | members
                )
            elif not members <= selection:
                selection = members
        self.set_selection(selection)
        if hit in self._selection:
            self._start_drag(position)

    def _band_rect(self, position: QPointF) -> QRectF:
        """Scene rectangle from where the rubber band started to a position."""

        assert self._band_origin is not None
        return QRectF(
            QPointF(self.mapToScene(self._band_origin.toPoint())),
            QPointF(self.mapToScene(position.toPoint())),
        ).normalized()

    def _move_band(self, position: QPointF) -> None:
        if self._band_origin is None:
            return
        if (
            not self.band_item.isVisible()
            and (position - self._band_origin).manhattanLength() <= _CLICK_SLOP_PX
        ):
            return
        self.band_item.setRect(self._band_rect(position))
        self.band_item.show()

    def _finish_band(self, position: QPointF) -> None:
        """Add the shown points inside the rubber band to the selection."""

        if self._band_origin is None:
            return
        dragged = (
            self.band_item.isVisible()
            or (position - self._band_origin).manhattanLength() > _CLICK_SLOP_PX
        )
        rect = self._band_rect(position)
        self._cancel_band()
        if not dragged:
            return
        chosen: set[tuple[int, int]] = set()
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers:
                continue
            for vertex, (row, col) in enumerate(polygon.points):
                if rect.contains(self.raw_to_scene(row, col)):
                    chosen.add((index, vertex))
        self.set_selection(set(self._selection) | self._with_shared(chosen))

    def _cancel_band(self) -> None:
        self._band_origin = None
        self.band_item.hide()
        self.band_item.setRect(QRectF())

    def _edit_key(self, event: Any) -> bool:
        """Keys of the Edit tool; False when the key is not one of them."""

        key = event.key()
        busy = self._drag is not None or self._band_origin is not None
        if key == Qt.Key.Key_Escape:
            if not (self._selection or busy):
                return False
            self.clear_selection()
            return True
        if not self._selection or busy:
            return False
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_points()
            return True
        step = _NUDGE_KEYS.get(key)
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if step is None or modifiers & ~Qt.KeyboardModifier.ShiftModifier:
            return False
        pixels = (
            _NUDGE_SHIFT_PX
            if modifiers & Qt.KeyboardModifier.ShiftModifier
            else _NUDGE_PX
        )
        self.nudge_selected_points(step[0] * pixels, step[1] * pixels)
        return True

    # ------------------------------------------------------ drawing polygons

    def drafting(self) -> bool:
        return bool(self._draft)

    @property
    def draft_points(self) -> list[tuple[float, float]]:
        return list(self._draft)

    def _drawing(self) -> bool:
        return (
            self.paint_enabled
            and not self.place_mode
            and self.tool == "polygon"
            and self._raw_size is not None
            and self._marks_visible
        )

    def _draft_pen(self, *, dashed: bool = False) -> QPen:
        pen = QPen(QColor(self.layer_color), 2.0)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _render_draft(
        self,
        cursor: QPointF | None = None,
        way: Sequence[tuple[float, float]] = (),
    ) -> None:
        """Draw the polygon being drawn, and a dashed line from its last point
        to the cursor, along ``way`` (raw points) when the next click would
        follow an outline, and back to its first point."""

        for item in self._draft_handles:
            self.scene().removeItem(item)
        self._draft_handles = []
        if not self._draft:
            self.draft_item.setPath(QPainterPath())
            self.rubber_item.setPath(QPainterPath())
            return
        scene_points = [self.raw_to_scene(row, col) for row, col in self._draft]
        path = QPainterPath(scene_points[0])
        for point in scene_points[1:]:
            path.lineTo(point)
        self.draft_item.setPen(self._draft_pen())
        self.draft_item.setPath(path)
        rubber = QPainterPath()
        if cursor is not None:
            rubber.moveTo(scene_points[-1])
            for row, col in way:
                rubber.lineTo(self.raw_to_scene(row, col))
            rubber.lineTo(cursor)
            if len(scene_points) >= 2:
                rubber.lineTo(scene_points[0])
        self.rubber_item.setPen(self._draft_pen(dashed=True))
        self.rubber_item.setPath(rubber)
        for number, click in enumerate(self._clicks):
            for row, col in click.followed():
                self._draft_handles.append(
                    self._handle(
                        self.raw_to_scene(row, col),
                        self.layer_color,
                        size=_FOLLOWED_HANDLE_PX,
                        rounded=True,
                    )
                )
            # The first point is larger: clicking it closes the polygon. A
            # point that snapped onto a polygon is round.
            size = 10.0 if number == 0 else 7.0
            self._draft_handles.append(
                self._handle(
                    self.raw_to_scene(*click.point),
                    self.layer_color,
                    size=size + 2.0 if click.snapped else size,
                    rounded=click.snapped,
                )
            )

    def _update_draft(self) -> None:
        """The polygon's points: the clicked points with the outline points
        followed between them."""

        self._draft = [
            point
            for click in self._clicks
            for point in (*click.followed(), click.point)
        ]

    def set_follow_edges(self, follow: bool) -> None:
        """Whether Draw snaps onto, and follows, the edges of shown polygons."""

        self.follow_edges = bool(follow)
        self._refresh_preview()

    def _shown_polygons(self) -> list[tuple[int, lp.Polygon]]:
        return [
            (index, polygon)
            for index, polygon in enumerate(self._polygons)
            if polygon.layer_id not in self._hidden_layers
        ]

    def _viewport_points(self, points: Sequence[tuple[float, float]]) -> np.ndarray:
        """Viewport (x, y) of raw (row, col) points, one row per point."""

        raw = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        scale_rows, scale_cols = self._scale()
        x = (raw[:, 1] + 0.5) * scale_cols - 0.5
        y = (raw[:, 0] + 0.5) * scale_rows - 0.5
        transform = self.viewportTransform()
        return np.column_stack(
            (
                transform.m11() * x + transform.m21() * y + transform.dx(),
                transform.m12() * x + transform.m22() * y + transform.dy(),
            )
        )

    def snap_point(self, position: QPointF) -> tuple[float, float] | None:
        """Where a Draw click at a viewport position snaps to (raw row, col).

        The nearest point of a shown polygon within the point-hit distance,
        else the nearest position on an edge of a shown polygon within that
        distance; None when there is neither.
        """

        target = np.asarray([position.x(), position.y()], dtype=np.float64)
        shown = []
        for _index, polygon in self._shown_polygons():
            points = np.asarray(polygon.points, dtype=np.float64).reshape(-1, 2)
            if len(points):
                shown.append((points, self._viewport_points(points)))
        best: tuple[float, tuple[float, float]] | None = None
        for points, spots in shown:
            distance = np.hypot(spots[:, 0] - target[0], spots[:, 1] - target[1])
            nearest = int(np.argmin(distance))
            if distance[nearest] <= _VERTEX_HIT_PX and (
                best is None or distance[nearest] < best[0]
            ):
                best = (float(distance[nearest]), tuple(map(float, points[nearest])))
        if best is not None:
            return best[1]
        for points, spots in shown:
            if len(points) < 2:
                continue
            delta = np.roll(spots, -1, axis=0) - spots
            length2 = np.einsum("ij,ij->i", delta, delta)
            along = np.einsum("ij,ij->i", target - spots, delta)
            fraction = np.clip(along / np.maximum(length2, 1e-12), 0.0, 1.0)
            near = spots + fraction[:, None] * delta
            distance = np.hypot(near[:, 0] - target[0], near[:, 1] - target[1])
            edge = int(np.argmin(distance))
            if distance[edge] <= _VERTEX_HIT_PX and (
                best is None or distance[edge] < best[0]
            ):
                start, end = points[edge], points[(edge + 1) % len(points)]
                raw = start + fraction[edge] * (end - start)
                best = (float(distance[edge]), (float(raw[0]), float(raw[1])))
        return None if best is None else best[1]

    def _outline_places(
        self, point: tuple[float, float]
    ) -> list[tuple[int, int, float]]:
        """Shown polygons whose outline passes through a raw point (within
        ``lp.SHARED_POINT_PX``): (polygon, edge, fraction) per polygon, with
        fraction 0 on the edge's first point."""

        row, col = point
        places = []
        for index, polygon in self._shown_polygons():
            points = np.asarray(polygon.points, dtype=np.float64)
            if len(points) < 2:
                continue
            distance = np.hypot(points[:, 0] - row, points[:, 1] - col)
            vertex = int(np.argmin(distance))
            if distance[vertex] <= lp.SHARED_POINT_PX:
                places.append((index, vertex, 0.0))
                continue
            delta = np.roll(points, -1, axis=0) - points
            length2 = np.einsum("ij,ij->i", delta, delta)
            along = (row - points[:, 0]) * delta[:, 0]
            along += (col - points[:, 1]) * delta[:, 1]
            fraction = np.clip(along / np.maximum(length2, 1e-12), 0.0, 1.0)
            near = points + fraction[:, None] * delta
            distance = np.hypot(near[:, 0] - row, near[:, 1] - col)
            edge = int(np.argmin(distance))
            if distance[edge] <= lp.SHARED_POINT_PX and 0.0 < fraction[edge] < 1.0:
                places.append((index, edge, float(fraction[edge])))
        return places

    def _ways_between(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[list[tuple[float, float]]]:
        """Ways round the outline of a shown polygon through both raw points,
        as the outline points passed, shortest first; empty when no shown
        polygon's outline passes through both."""

        ends = {
            index: (edge, fraction)
            for index, edge, fraction in self._outline_places(end)
        }
        found: dict[tuple[tuple[float, float], ...], list[tuple[float, float]]] = {}
        for index, edge, fraction in self._outline_places(start):
            if index not in ends:
                continue
            outline = self._polygons[index].points
            for way in _ways_round(outline, (edge, fraction), ends[index]):
                found.setdefault(tuple(way), way)

        def length(way: Sequence[tuple[float, float]]) -> float:
            points = [start, *way, end]
            return sum(
                math.hypot(second[0] - first[0], second[1] - first[1])
                for first, second in zip(points, points[1:])
            )

        return sorted(found.values(), key=length)

    def _add_draft_point(self, scene_point: QPointF, *, free: bool = False) -> None:
        """Add a clicked point to the polygon being drawn.

        With ``follow_edges`` and without ``free`` it snaps onto a shown
        polygon, and after a point that snapped onto the same outline the
        polygon follows the outline to it.
        """

        raw = self.scene_to_raw(self._clamped_scene(scene_point))
        snapped = None
        if self.follow_edges and not free:
            snapped = self.snap_point(self.viewportTransform().map(scene_point))
        if snapped is not None:
            raw = snapped
            scene_point = self.raw_to_scene(*raw)
        if self._draft:
            last = self.mapFromScene(self.raw_to_scene(*self._draft[-1]))
            spot = self.mapFromScene(scene_point)
            if math.hypot(last.x() - spot.x(), last.y() - spot.y()) < 2.0:
                return
        click = _DraftPoint(raw, snapped is not None)
        if click.snapped and self._clicks and self._clicks[-1].snapped:
            click.ways = self._ways_between(self._clicks[-1].point, raw)
        self._clicks.append(click)
        self._update_draft()
        self._render_draft(scene_point)
        self.draft_changed.emit(len(self._draft))

    def switch_way(self, step: int = 1) -> bool:
        """Follow the outline the other way round to the last clicked point
        (Tab; Shift+Tab goes back). False when there is no other way."""

        click = self._clicks[-1] if self._clicks else None
        if click is None or len(click.ways) < 2:
            return False
        click.way = (click.way + int(step)) % len(click.ways)
        self._update_draft()
        self._refresh_preview()
        self.draft_changed.emit(len(self._draft))
        return True

    def _hover_draw(self, position: QPointF | None, *, free: bool = False) -> None:
        """Show where a Draw click at a viewport position would go: a ring
        where it would snap, and the dashed line from the last point, along
        the outline the click would follow."""

        self._draw_hover = None if position is None else (QPointF(position), bool(free))
        snapped = None
        if position is not None and self.follow_edges and not free:
            snapped = self.snap_point(position)
        if snapped is None:
            self.snap_item.hide()
        else:
            self.snap_item.setPos(self.raw_to_scene(*snapped))
            self.snap_item.show()
        if not self._draft:
            return
        if position is None:
            self._render_draft()
        elif snapped is None:
            self._render_draft(self._clamped_scene(self.mapToScene(position.toPoint())))
        else:
            way: list[tuple[float, float]] = []
            if self._clicks and self._clicks[-1].snapped:
                ways = self._ways_between(self._clicks[-1].point, snapped)
                way = ways[0] if ways else []
            self._render_draft(self.raw_to_scene(*snapped), way)

    def _refresh_preview(self) -> None:
        if self._draw_hover is not None and self._drawing():
            position, free = self._draw_hover
            self._hover_draw(position, free=free)
        else:
            self.snap_item.hide()
            self._render_draft()

    def remove_last_point(self) -> None:
        """Remove the last clicked point, with the outline followed to it."""

        if not self._clicks:
            return
        self._clicks.pop()
        self._update_draft()
        self._render_draft()
        self.draft_changed.emit(len(self._draft))

    def cancel_polygon(self) -> None:
        had = bool(self._clicks)
        self._clicks = []
        self._update_draft()
        self._render_draft()
        if had:
            self.draft_changed.emit(0)

    def _outline_points(self) -> dict[int, list[tuple[float, float]]]:
        """The shown polygons that get the new polygon's points that snapped
        onto their edges: {index: new points}."""

        added: dict[int, list[tuple[int, float, tuple[float, float]]]] = {}
        for click in self._clicks:
            if not click.snapped:
                continue
            for index, edge, fraction in self._outline_places(click.point):
                if fraction > 0.0:
                    added.setdefault(index, []).append((edge, fraction, click.point))
        changes = {}
        for index, rows in added.items():
            points = [
                (float(row), float(col)) for row, col in self._polygons[index].points
            ]
            done: set[tuple[int, float]] = set()
            # From the last edge position back, so that earlier edges keep
            # their indices.
            for edge, fraction, point in sorted(
                rows, key=lambda row: row[:2], reverse=True
            ):
                if (edge, round(fraction, 9)) in done:
                    continue
                done.add((edge, round(fraction, 9)))
                points.insert(edge + 1, (float(point[0]), float(point[1])))
            changes[index] = points
        return changes

    def close_polygon(self) -> bool:
        """Finish the polygon being drawn; False when it has too few points."""

        if not self._draft:
            return False
        if len(self._draft) < 3:
            self.notice.emit(
                "A polygon needs at least three points; keep clicking, or press "
                "Esc to cancel."
            )
            return False
        points = list(self._draft)
        # The implicit last-to-first edge follows the same shown outline as
        # two explicit snapped clicks. Free endpoints and Follow off stay free.
        if (self.follow_edges and self._clicks
                and self._clicks[0].snapped and self._clicks[-1].snapped):
            ways = self._ways_between(self._clicks[-1].point, self._clicks[0].point)
            if ways:
                points.extend(ways[0])
        polygon = lp.Polygon(int(self.layer_id), points)
        if polygon.area() < 1.0:
            self.notice.emit("The polygon encloses no area.")
            return False
        outline = self._outline_points()
        self._clicks = []
        self._update_draft()
        self._render_draft()
        self.draft_changed.emit(0)
        self.finished_outline_points = outline
        try:
            self.polygon_finished.emit(polygon)
        finally:
            self.finished_outline_points = {}
        return True

    def _near_first_point(self, position: QPointF) -> bool:
        if len(self._draft) < 3:
            return False
        first = self.mapFromScene(self.raw_to_scene(*self._draft[0]))
        return math.hypot(first.x() - position.x(), first.y() - position.y()) <= (
            _VERTEX_HIT_PX
        )

    def _editing(self) -> bool:
        return (
            self.paint_enabled
            and not self.place_mode
            and self.tool == "edit"
            and self._raw_size is not None
            and self._marks_visible
        )

    def _start_drag(self, position: QPointF) -> None:
        """Start dragging the selected points from a viewport position."""

        before = {
            index: list(self._polygons[index].points)
            for index in sorted(self._selected_by_polygon())
        }
        if not before:
            return
        self._drag = _PointDrag(
            QPointF(position),
            self.scene_to_raw(self.mapToScene(position.toPoint())),
            before,
            {index: list(points) for index, points in before.items()},
        )

    def _move_drag(self, position: QPointF) -> None:
        """Move the dragged points with the mouse, all by the same offset."""

        drag = self._drag
        if drag is None:
            return
        if not drag.moving:
            if (position - drag.press).manhattanLength() <= _CLICK_SLOP_PX:
                return
            drag.moving = True
        row, col = self.scene_to_raw(self.mapToScene(position.toPoint()))
        drag.after = self._moved_points(
            drag.before, row - drag.start[0], col - drag.start[1]
        )
        for index, points in drag.after.items():
            if index < len(self._polygon_items):
                self._polygon_items[index].setPath(self._polygon_path(points))
            handles = (
                self._handle_items[index] if index < len(self._handle_items) else []
            )
            for vertex, item in enumerate(handles):
                if vertex < len(points):
                    item.setPos(self.raw_to_scene(*points[vertex]))

    def _end_drag(self) -> None:
        """Finish a drag: one change for every polygon whose points moved."""

        drag, self._drag = self._drag, None
        if drag is None:
            return
        changes = {
            index: points
            for index, points in drag.after.items()
            if points != drag.before[index]
        }
        # Without a change the outlines and handles are where they were.
        self._emit_edits(changes)

    def _cancel_drag(self) -> None:
        """Stop a drag and put the points back where they were."""

        drag, self._drag = self._drag, None
        if drag is None:
            return
        for index, points in drag.before.items():
            if index < len(self._polygon_items):
                self._polygon_items[index].setPath(self._polygon_path(points))
        self._update_handles()

    # ------------------------------------------------------------ tools

    def set_tool(self, tool: str) -> None:
        if tool not in self.TOOLS:
            raise ValueError(f"Unknown tool {tool!r}")
        if tool != self.tool:
            self.cancel_polygon()
            # Only the Edit tool has selected points.
            self.clear_selection()
            self.snap_item.hide()
            self._draw_hover = None
        self.tool = tool
        self._cancel_stroke()
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            {
                "pan": Qt.CursorShape.OpenHandCursor,
                "edit": Qt.CursorShape.ArrowCursor,
            }.get(tool, Qt.CursorShape.CrossCursor)
        )
        self._update_handles()
        self._update_cursor()

    def set_layer(self, layer_id: int, color: QColor) -> None:
        self.layer_id = int(layer_id)
        self.layer_color = QColor(color)
        if self._draft:
            self._render_draft()

    def _painting(self) -> bool:
        return (
            self.paint_enabled
            and not self.place_mode
            and self.tool in _STROKE_TOOLS
            and self._raw_size is not None
        )

    def _update_cursor(self, scene_point: QPointF | None = None) -> None:
        if not self._painting() or self.tool == "lasso":
            self.cursor_item.hide()
            return
        if scene_point is None:
            position = self.viewport().mapFromGlobal(self.cursor().pos())
            if not self.viewport().rect().contains(position):
                self.cursor_item.hide()
                return
            scene_point = self.mapToScene(position)
        radius = max(0.5, self.display_radius())
        self.cursor_item.setRect(
            QRectF(
                scene_point.x() - radius,
                scene_point.y() - radius,
                2 * radius,
                2 * radius,
            )
        )
        self.cursor_item.show()

    def _cancel_stroke(self) -> None:
        self._points = []
        self._path = None
        self.active_item.setPath(QPainterPath())

    def _extend(self, scene_point: QPointF) -> None:
        raw = self.scene_to_raw(scene_point)
        if self._points:
            previous = self.raw_to_scene(*self._points[-1])
            if (scene_point - previous).manhattanLength() < 1.0:
                return
        self._points.append(raw)
        if self._path is None:
            self._path = QPainterPath(scene_point)
        else:
            self._path.lineTo(scene_point)
        path = QPainterPath(self._path)
        if len(self._points) == 1 and self.tool != "lasso":
            radius = max(0.5, self.display_radius())
            path.addEllipse(scene_point, radius, radius)
        self.active_item.setPath(path)

    def _start_stroke(self, scene_point: QPointF) -> None:
        self._cancel_stroke()
        clearing = self.tool in {"erase", "restore"}
        color = QColor(255, 255, 255) if clearing else QColor(self.layer_color)
        if self.tool == "lasso":
            outline = QPen(color, 2.0)
            outline.setCosmetic(True)
            self.active_item.setPen(outline)
            self.active_item.setBrush(Qt.BrushStyle.NoBrush)
            self._extend(scene_point)
            return
        pen = QPen(color, max(1.0, 2.0 * self.display_radius()))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if clearing:
            pen.setStyle(Qt.PenStyle.SolidLine)
            color.setAlpha(120)
            pen.setColor(color)
        self.active_item.setPen(pen)
        self.active_item.setBrush(color)
        self._extend(scene_point)

    def _finish_stroke(self) -> None:
        if not self._points:
            self._cancel_stroke()
            return
        points, tool = list(self._points), self.tool
        self._cancel_stroke()
        if tool in _CORRECTION_TOOLS:
            self._finish_correction(tool, points)
            return
        stroke = lp.Stroke(
            lp.ERASER if tool == "erase" else int(self.layer_id),
            float(self.radius_px),
            points,
        )
        self.stroke_finished.emit(stroke)

    def _finish_correction(self, tool: str, points: list[tuple[float, float]]) -> None:
        """Report a finished Correct, Restore or Lasso path as a correction."""

        kind = {"correct": "brush", "restore": "restore", "lasso": "lasso"}[tool]
        try:
            correction = lp.Correction(
                kind,
                int(self.layer_id),
                points,
                radius_px=0.0 if kind == "lasso" else float(self.radius_px),
            )
        except ValueError as error:
            self.notice.emit(
                "Hold the button and go round the area to draw a lasso."
                if kind == "lasso"
                else str(error)
            )
            return
        self.correction_drawn.emit(correction)

    # ------------------------------------------------------------ events

    def wheelEvent(self, event: Any) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.scale(factor, factor)
        self._user_zoomed = True
        self._update_pan_margins()
        self.refresh_detail()
        self._update_cursor(self.mapToScene(event.position().toPoint()))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit_image()
        else:
            self._update_pan_margins()

    def mousePressEvent(self, event: Any) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        button = event.button()
        if button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_origin = event.position()
            if button == Qt.MouseButton.RightButton:
                self._right_press = event.position()
                self._right_dragged = False
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self.place_mode:
            point = self.mapToScene(event.position().toPoint())
            bounds = self.pixmap_item.boundingRect()
            if bounds.contains(point):
                if self.photo_display_mapping is not None:
                    row, col = self.scene_to_raw(point)
                    x, y = self.photo_display_mapping.camera_u_to_display_xy([[col, row]])[0]
                    support = self._photo_valid_support
                    ix, iy = int(round(x)), int(round(y))
                    if (support is None or not (0 <= iy < support.shape[0]
                            and 0 <= ix < support.shape[1] and support[iy, ix])):
                        event.accept()
                        return
                    width, height = self._raw_size
                    self.raw_point_selected.emit(row, col)
                    self.image_point_selected.emit(col / max(1, width-1), row / max(1, height-1))
                else:
                    row, col = self.scene_to_raw(point)
                    self.raw_point_selected.emit(row, col)
                    self.image_point_selected.emit(
                        float(point.x() / max(1.0, bounds.width() - 1.0)),
                        float(point.y() / max(1.0, bounds.height() - 1.0)),
                    )
                event.accept()
                return
        if button == Qt.MouseButton.LeftButton and self._editing():
            self._edit_press(event)
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._drawing():
            # Draw only adds points: a click on a point of an existing polygon
            # adds a point to the new polygon there.
            point = self.mapToScene(event.position().toPoint())
            if self._near_first_point(event.position()):
                self.close_polygon()
                event.accept()
                return
            if self.pixmap_item.boundingRect().contains(point) or self._draft:
                self._add_draft_point(point, free=bool(event.modifiers() & _ALONE))
                event.accept()
                return
        if (
            button == Qt.MouseButton.LeftButton
            and self.tool == "fill"
            and self.paint_enabled
            and not self.place_mode
            and self._raw_size is not None
        ):
            point = self.mapToScene(event.position().toPoint())
            if self.pixmap_item.boundingRect().contains(point):
                row, col = self.scene_to_raw(point)
                self.fill_requested.emit(float(row), float(col))
                event.accept()
                return
        if button == Qt.MouseButton.LeftButton and self._painting():
            point = self.mapToScene(event.position().toPoint())
            if self.pixmap_item.boundingRect().contains(point):
                self._start_stroke(point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drawing():
            if self._draft:
                # The first click of the double click added the last point.
                self.close_polygon()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._editing():
            self._cancel_band()
            self._insert_point_at(event.position())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._pan_origin is not None:
            if self._right_press is not None and not self._right_dragged:
                moved = event.position() - self._right_press
                self._right_dragged = moved.manhattanLength() > _CLICK_SLOP_PX
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(round(delta.x()))
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(round(delta.y()))
            )
            event.accept()
            return
        point = self.mapToScene(event.position().toPoint())
        if self._drag is not None:
            self._move_drag(event.position())
            event.accept()
            return
        if self._band_origin is not None:
            self._move_band(event.position())
            event.accept()
            return
        if self._drawing():
            self._hover_draw(event.position(), free=bool(event.modifiers() & _ALONE))
        else:
            self.snap_item.hide()
        if self._points and self._painting():
            self._extend(self._clamped_scene(point))
            self._update_cursor(point)
            event.accept()
            return
        self._update_cursor(point)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        button = event.button()
        if self._pan_origin is not None and button in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._pan_origin = None
            clicked = (
                button == Qt.MouseButton.RightButton
                and self._right_press is not None
                and not self._right_dragged
            )
            self._right_press = None
            self.set_tool(self.tool)
            if clicked:
                self._right_click(event)
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._drag is not None:
            self._end_drag()
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._band_origin is not None:
            self._finish_band(event.position())
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._points:
            self._finish_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _right_click(self, event: Any) -> None:
        """A right click without dragging: close the polygon or open a menu."""

        if self.place_mode:
            return
        if self._draft and self._drawing():
            self.close_polygon()
            return
        if not self._marks_visible or not self.paint_enabled:
            return
        hit = self.polygon_at(event.position())
        if hit is not None:
            self.last_context_raw = self.scene_to_raw(
                self.mapToScene(event.position().toPoint())
            )
            self.polygon_context.emit(
                hit[0], hit[1], self.viewport().mapToGlobal(event.position().toPoint())
            )

    def event(self, event: QEvent) -> bool:
        # While a polygon is drawn with Follow edges, Tab (Shift+Tab back)
        # takes the other way round the outline instead of moving the
        # keyboard focus.
        if (
            event.type() == QEvent.Type.KeyPress
            and self._clicks
            and self.follow_edges
            and event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab)
        ):
            self.switch_way(-1 if event.key() == Qt.Key.Key_Backtab else 1)
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event: Any) -> None:
        if self._draft:
            key = event.key()
            if key == Qt.Key.Key_Escape:
                self.cancel_polygon()
                event.accept()
                return
            if key == Qt.Key.Key_Backspace:
                self.remove_last_point()
                event.accept()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.close_polygon()
                event.accept()
                return
        if self._editing() and self._edit_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self.cursor_item.hide()
        self.snap_item.hide()
        self._draw_hover = None
        if self._draft:
            self._render_draft()
        super().leaveEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        event.accept()


# ---------------------------------------------------------------------------
# Step sections


class _JoinEdgesDialog(QDialog):
    """Preview the old and shared stretches on the photograph before applying.

    Every joinable stretch is listed with its length and a check box, all
    checked at first; only the checked ones are joined, and the preview
    follows each change at once. A refused join is explained in plain words
    and its point is ringed in red on the canvas (raw coordinates only in
    the tooltips). A distance at or below the chosen one that is checked to
    work is then offered as one button; that search runs in a worker thread,
    which is freed here in the GUI thread after wait(), never by deleteLater
    in its own thread.
    """

    # The offered distance is a whole multiple of this (mm).
    TOLERANCE_MM = 0.05
    # Radius of the red ring at a refused point, in screen pixels.
    RING_PX = 12.0
    # The checked stretch nearest to the ring is marked in the list, and any
    # other at most this many raw pixels farther from it.
    AT_RING_PX = 0.5
    _INPUTS = ("Selected polygon", "Neighbour polygon")

    class _DistanceSearch(QObject):
        """Find a distance that is checked to work, for one request, in its own thread."""

        finished = Signal(object)

        def __init__(self, request, pixels_per_mm, decimals):
            """One request: (generation, points, neighbour, distance px, tolerance px)."""
            super().__init__()
            self.request = request
            self.pixels_per_mm = pixels_per_mm
            self.decimals = decimals
            self.stop = False

        @Slot()
        def run(self):
            """Emit (generation, mm or None); the mm is checked as the dialog will use it."""
            from darsia.fluidflower.layer_painting_join import verified_working_distance

            generation, points, other, distance_px, tolerance_px = self.request
            offer = None
            try:
                found = verified_working_distance(
                    points, other, distance_px, tolerance_px, cancelled=lambda: self.stop
                )
                if found is not None and not self.stop:
                    # The spin box keeps its decimals and the dialog joins at
                    # value * pixels per mm: check exactly that distance too.
                    mm = float(f"{found / self.pixels_per_mm:.{self.decimals}f}")
                    preview = join_to_neighbour(points, other, mm * self.pixels_per_mm)
                    offer = mm if preview.stretches else None
            except JoinEdgesError:
                offer = None
            finally:
                self.finished.emit((generation, offer))

    def __init__(self, parent, canvas, points, neighbours, pixels_per_mm):
        super().__init__(parent)
        from PySide6.QtWidgets import QListWidget

        self.setWindowTitle("Join to neighbour's edge")
        self.canvas = canvas
        self.points = points
        self.neighbours = neighbours
        self.pixels_per_mm = pixels_per_mm
        self.preview: JoinPreview | None = None
        self.error: JoinEdgesError | None = None
        # The joinable stretches at the chosen neighbour and distance.
        self.stretches: tuple = ()
        self.offered_mm: float | None = None
        self._no_smaller = False
        self._generation = 0
        self._thread = self._worker = self._pending = None
        layout = QVBoxLayout(self)
        intro = QLabel(
            "The selected polygon follows the neighbour's outline within the distance; "
            "the neighbour keeps its outline. Orange: old stretches; cyan: shared edge. "
            "Only the checked stretches are joined."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Neighbour"))
        self.neighbour = QComboBox()
        for index, label, _points in neighbours:
            self.neighbour.addItem(label, index)
        row.addWidget(self.neighbour)
        row.addWidget(QLabel("Distance"))
        self.distance = QDoubleSpinBox()
        self.distance.setRange(0.0, lp.MAX_SNAP_MM)
        self.distance.setDecimals(2)
        self.distance.setValue(5.0)
        self.distance.setSuffix(" mm")
        row.addWidget(self.distance)
        layout.addLayout(row)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        offer = QHBoxLayout()
        self.use_button = QPushButton()
        self.use_button.setAutoDefault(False)
        self.use_button.setToolTip(
            "A distance at or below the chosen one that was checked to give a valid "
            "join. It is not necessarily the largest distance that works."
        )
        self.use_button.clicked.connect(self._use_offered)
        self.use_button.hide()
        offer.addWidget(self.use_button)
        offer.addStretch(1)
        layout.addLayout(offer)
        layout.addWidget(QLabel("Stretches to join"))
        self.stretch_list = QListWidget()
        self.stretch_list.setToolTip(
            "Uncheck a stretch to keep its old outline. "
            "Select one to show it on the photograph."
        )
        self.stretch_list.setMaximumHeight(9 * self.fontMetrics().height())
        self.stretch_list.itemChanged.connect(self._stretch_checked)
        self.stretch_list.currentRowChanged.connect(self._show_current)
        layout.addWidget(self.stretch_list)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.old_item = QGraphicsPathItem()
        self.new_item = QGraphicsPathItem()
        self.current_item = QGraphicsPathItem()
        self.error_item = QGraphicsPathItem()
        for item, color, style, width, z in (
            (self.current_item, QColor(255, 255, 255, 170), Qt.PenStyle.SolidLine, 9.0, 39.0),
            (self.old_item, QColor("#ff9900"), Qt.PenStyle.DashLine, 3.0, 40.0),
            (self.new_item, QColor("#00ffff"), Qt.PenStyle.SolidLine, 3.0, 40.0),
            (self.error_item, QColor("#ff3333"), Qt.PenStyle.SolidLine, 3.0, 41.0),
        ):
            pen = QPen(color, width, style)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(z)
            canvas.scene().addItem(item)
        self.neighbour.currentIndexChanged.connect(self.refresh_preview)
        self.distance.valueChanged.connect(self.refresh_preview)
        self.finished.connect(self.clear_preview)
        self.refresh_preview()

    def _path(self, stretches):
        path = QPainterPath()
        for points in stretches:
            if points:
                path.moveTo(self.canvas.raw_to_scene(*points[0]))
                for point in points[1:]:
                    path.lineTo(self.canvas.raw_to_scene(*point))
        return path

    def _other(self):
        """The outline of the chosen neighbour."""
        return self.neighbours[self.neighbour.currentIndex()][2]

    def _distance_px(self):
        """The chosen distance in raw pixels."""
        return self.distance.value() * self.pixels_per_mm

    def checked_stretches(self) -> list[int]:
        """The numbers (0, 1, ...) of the checked stretches, in list order."""
        return [
            row for row in range(self.stretch_list.count())
            if self.stretch_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def refresh_preview(self, *_args):
        """List the stretches at this neighbour and distance, all checked, and preview."""
        from darsia.fluidflower.layer_painting_join import join_stretches

        self._stop_search()
        other, distance_px = self._other(), self._distance_px()
        try:
            self.preview = join_to_neighbour(self.points, other, distance_px)
            self.error = None
            self.stretches = self.preview.stretches
        except JoinEdgesError as error:
            self.preview, self.error = None, error
            try:
                # A refused join still lists its stretches, so some can be unchecked.
                self.stretches = join_stretches(self.points, other, distance_px)
            except JoinEdgesError:
                self.stretches = ()
        self._fill_list()
        self._show()
        # A smaller distance cannot mend a refused input polygon.
        if self.error is not None and self.error.outline not in self._INPUTS:
            self._start_search()

    def _fill_list(self):
        """One checked row per joinable stretch, with its length in mm."""
        from PySide6.QtWidgets import QListWidgetItem

        self.stretch_list.blockSignals(True)
        self.stretch_list.clear()
        for stretch in self.stretches:
            item = QListWidgetItem()
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(Qt.CheckState.Checked)
            self.stretch_list.addItem(item)
        self.stretch_list.blockSignals(False)
        self.current_item.setPath(QPainterPath())

    def _stretch_checked(self, _item):
        """A check box changed: preview the checked stretches at once."""
        self._update_preview()

    def _update_preview(self):
        """Preview the checked stretches; all checked is the default join."""
        checked = self.checked_stretches()
        include = None if len(checked) == len(self.stretches) else checked
        try:
            self.preview = join_to_neighbour(
                self.points, self._other(), self._distance_px(), include=include
            )
            self.error = None
        except JoinEdgesError as error:
            self.preview, self.error = None, error
        self._show()

    def _show(self):
        """Draw the stretches and any ring, say what Apply does and enable it."""
        checked = self.checked_stretches()
        shown = (
            self.preview.stretches if self.preview is not None
            else [self.stretches[number] for number in checked]
        )
        self.old_item.setPath(self._path([row.before for row in shown]))
        self.new_item.setPath(self._path([row.after for row in shown]))
        self.error_item.setPath(QPainterPath())
        self.error_item.setToolTip("")
        self.summary.setToolTip("")
        at_ring = set()
        if self.error is not None:
            self.summary.setText(
                self._plain_refusal(self.error)
                + (" No smaller distance was found that works." if self._no_smaller else "")
            )
            self.summary.setToolTip(str(self.error))
            if self.error.location is not None:
                at_ring = self._ring(self.error.location, checked)
        elif self.preview.stretches:
            count = len(self.preview.stretches)
            length = self.preview.length_px / self.pixels_per_mm
            self.summary.setText(
                f"{count} stretch{'es' if count != 1 else ''}, {length:.2f} mm of old outline. "
                "Apply changes both polygons in one undo step."
            )
        elif self.stretches and not checked:
            self.summary.setText("No stretch is checked. Check a stretch to join it.")
        else:
            self.summary.setText("No stretch can be joined at this distance.")
        self.stretch_list.blockSignals(True)
        for number, stretch in enumerate(self.stretches):
            self.stretch_list.item(number).setText(
                f"Stretch {number + 1}: {stretch.length_px / self.pixels_per_mm:.2f} mm"
                + (" (at the red ring)" if number in at_ring else "")
            )
        self.stretch_list.blockSignals(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).setEnabled(
            self.preview is not None and bool(self.preview.stretches)
        )

    def _ring(self, location, checked):
        """Ring a refused raw point on the canvas; return the checked stretches nearest to it.

        Only stretches within the join distance of the point count, so a
        refusal elsewhere marks none.
        """
        from shapely.geometry import LineString, Point

        point = self.canvas.raw_to_scene(*location)
        radius = self.RING_PX / max(abs(self.canvas.transform().m11()), 1.0e-6)
        path = QPainterPath()
        path.addEllipse(point, radius, radius)
        self.error_item.setPath(path)
        self.error_item.setToolTip(f"Raw row {location[0]:.2f}, column {location[1]:.2f}")
        self.canvas.ensureVisible(
            QRectF(point.x() - radius, point.y() - radius, 2.0 * radius, 2.0 * radius)
        )
        where = Point(location)
        reach = {
            number: min(
                LineString(points).distance(where)
                for points in (self.stretches[number].before, self.stretches[number].after)
                if len(points) > 1
            )
            for number in checked
            if len(self.stretches[number].before) > 1 or len(self.stretches[number].after) > 1
        }
        if not reach or min(reach.values()) > self._distance_px() + self.AT_RING_PX:
            return set()
        nearest = min(reach.values())
        return {
            number for number, value in reach.items()
            if value <= nearest + self.AT_RING_PX
        }

    @staticmethod
    def _plain_refusal(error):
        """The refusal in plain words; its raw coordinates stay in the tooltips."""
        crosses = "Self-intersection" in (error.reason or "")
        marked = " at the marked point" if error.location is not None else ""
        if error.outline == "Joined polygon":
            problem = "would cross itself" if crosses else "would not be a valid outline"
            return (
                f"The joined outline {problem}{marked}. Use a smaller distance, uncheck "
                "the stretch there, or join from the other polygon."
            )
        if error.outline == "Neighbour polygon after endpoint insertion":
            return (
                f"The neighbour's outline would not stay valid{marked}. Use a smaller "
                "distance, or join from the other polygon."
            )
        if error.outline in _JoinEdgesDialog._INPUTS:
            problem = "crosses itself" if crosses else "is not a valid outline"
            if error.outline == "Selected polygon":
                return (
                    f"The selected polygon {problem}{marked}. "
                    "Correct its outline before joining."
                )
            return (
                f"This neighbour {problem}{marked}. "
                "Correct its outline, or choose another neighbour."
            )
        return str(error)

    def _show_current(self, row):
        """Highlight the old outline of the selected stretch on the canvas."""
        if 0 <= row < len(self.stretches):
            self.current_item.setPath(self._path([self.stretches[row].before]))
        else:
            self.current_item.setPath(QPainterPath())

    def _stop_search(self):
        """Drop the offer and stop a running search; its thread ends after one more join."""
        self._generation += 1
        self._pending = None
        if self._worker is not None:
            self._worker.stop = True
        self.offered_mm = None
        self._no_smaller = False
        self.use_button.hide()

    def _start_search(self):
        """Search a working distance below the chosen one, off the GUI thread."""
        self._pending = (
            self._generation,
            list(self.points),
            list(self._other()),
            self._distance_px(),
            self.TOLERANCE_MM * self.pixels_per_mm,
        )
        self.use_button.setText("Finding a distance that works…")
        self.use_button.setEnabled(False)
        self.use_button.show()
        if self._thread is None:
            self._run_pending()

    def _run_pending(self):
        """Start the latest search request in its own thread (0008 rule)."""
        request, self._pending = self._pending, None
        if request is None:
            return
        thread = QThread(self)
        worker = self._DistanceSearch(request, self.pixels_per_mm, self.distance.decimals())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(self._search_finished)
        thread.finished.connect(self._thread_finished)
        self._thread, self._worker = thread, worker
        thread.start()

    def _search_finished(self, result):
        """Offer the distance found for the current request; older results are dropped."""
        generation, offer = result
        if generation != self._generation:
            return
        if offer is None:
            self.use_button.hide()
            self._no_smaller = True
            self._show()
            return
        self.offered_mm = offer
        self.use_button.setText(f"Use {offer:.2f} mm")
        self.use_button.setEnabled(True)
        self.use_button.show()

    def _thread_finished(self):
        """Free the finished search here, then start a request that waited for it."""
        thread = self._thread
        if thread is not None:
            # After wait() the worker is freed here, in the GUI thread.
            thread.wait()
        self._thread = self._worker = None
        if thread is not None:
            thread.deleteLater()
        self._run_pending()

    def _use_offered(self):
        """Take the offered distance; the preview follows."""
        if self.offered_mm is not None:
            self.distance.setValue(self.offered_mm)

    def clear_preview(self, *_args):
        """Remove the preview and the ring from the canvas, and end a running search."""
        self._stop_search()
        thread = self._thread
        if thread is not None:
            # A stopped search returns after at most one more join.
            thread.wait()
            # After wait() the worker is freed here, in the GUI thread.
            self._thread = self._worker = None
            thread.deleteLater()
        for item in (self.old_item, self.new_item, self.current_item, self.error_item):
            if item.scene() is not None:
                item.scene().removeItem(item)


class StepSection(QFrame):
    """A workflow section with a header that folds its content."""

    def __init__(
        self, number: int | None, title: str, content: QWidget, *, expanded: bool = True
    ) -> None:
        """Build one foldable step and its status header."""
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setText(title)
        font = self.toggle.font()
        font.setBold(True)
        self.toggle.setFont(font)
        self.toggle.toggled.connect(self.set_expanded)
        header.addWidget(self.toggle)
        header.addStretch(1)
        self.status = QLabel()
        header.addWidget(self.status)
        layout.addLayout(header)
        self.content = content
        layout.addWidget(content)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.blockSignals(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.blockSignals(False)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(bool(expanded))

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()

    def set_status(self, text: str, state: str = "") -> None:
        self.status.setText(text)
        color = _state_color(self, state).name()
        self.status.setStyleSheet(f"color: {color};")


# ---------------------------------------------------------------------------
# Workers


class _CorrectedGrid:
    """The corrected proposal on one grid, extended as corrections are added.

    ``get`` applies only the corrections added since the last call when the
    proposal, grid and layers are the same and the earlier corrections are
    unchanged; otherwise it starts again from the proposal.
    """

    def __init__(self) -> None:
        self.key: Any = None
        self.done: tuple[lp.Correction, ...] = ()
        self.state: tuple[np.ndarray, np.ndarray] | None = None

    def get(
        self,
        key: Any,
        base: np.ndarray,
        corrections: Sequence[lp.Correction],
        scale: tuple[float, float],
        layer_ids: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """The corrected classes on ``base``'s grid and the corrected pixels."""

        if (
            self.state is None
            or self.key != key
            or len(self.done) > len(corrections)
            or any(old is not new for old, new in zip(self.done, corrections))
        ):
            self.key, self.done = key, ()
            self.state = (base, np.zeros(base.shape, dtype=bool))
        rest = list(corrections[len(self.done):])
        if rest:
            result, decided, _skipped = lp.apply_corrections(
                base,
                rest,
                scale,
                inside=base != lp.OUTSIDE,
                layer_ids=set(layer_ids),
                start=self.state,
            )
            self.state = (result, decided)
            self.done = tuple(corrections)
        return self.state


class _ProposalWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image: lp.PaintingImage,
        session: lp.PaintingSession,
        frame: RigFrame | None,
        destination: Path,
        *,
        method: str = lp.PROPOSAL_METHOD,
    ) -> None:
        super().__init__()
        self.image = image
        self.session = session
        self.frame = frame
        self.destination = destination
        self.method = method

    @Slot()
    def run(self) -> None:
        try:
            propose = lpp.make_proposal if self.method == lpp.METHOD else lp.propose_layers
            if self.method not in {lp.PROPOSAL_METHOD, lpp.METHOD}:
                raise ValueError("Choose a supported proposal method")
            proposal = propose(
                self.image, self.session, frame=self.frame, progress=self.progress.emit
            )
        except ValueError as error:
            self.failed.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        try:
            proposal.save(self.destination)
        except OSError as error:
            self.progress.emit(f"The proposal could not be saved: {error}")
        self.finished.emit(proposal)


class _ManualMapWorker(QObject):
    """Draw the manual map at full resolution and, when asked, the proposal's
    map to compare it with (for display; nothing is saved)."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, session, image_size, frame, request) -> None:
        super().__init__()
        self.session = session
        self.image_size = tuple(image_size)
        self.frame = frame
        self.request = dict(request)

    @Slot()
    def run(self) -> None:
        result = {
            key: self.request[key] for key in ("run_id", "manual_key", "compare_key")
        }
        try:
            if result["manual_key"]:
                result["manual"] = lpm.manual_map(
                    self.session,
                    self.image_size,
                    frame=self.frame,
                    progress=self.progress.emit,
                )
            if result["compare_key"]:
                self.progress.emit("Drawing the proposal at full resolution to compare…")
                painted = (lpp.painted_layers if lpp.is_polygons_only(self.request["proposal"])
                           else lpm.painted_layers)
                result["layers"] = painted(
                    self.session,
                    self.request["proposal"],
                    frame=self.frame,
                    corrections=self.request["corrections"],
                    **({"require_complete": False} if lpp.is_polygons_only(self.request["proposal"]) else {}),
                )
        except Exception as error:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(result)


def _reviewed_geometry_options(frame, options):
    """Freeze approved manual point transforms before the existing transfer starts."""
    document = options.get('reference_point_review')
    if document is None:
        return
    from .rig_point_alignment import capture_reviewed_selection
    from .painting_frame_models import capture_frame_model_selection
    selection = options.get('geometry_selection') or capture_frame_model_selection(
        options['project'],override_path=options['geometry_index'])
    selected = capture_reviewed_selection(options['project'],selection,document,frame,options['runs'])
    options.update(geometry_index=selected.index_path,geometry_selection=selected)
    reuse = options.get('local_reuse')
    if reuse is not None:
        bound = reuse['review']['geometry_index']
        if any(bound.get(key)!=value for key,value in dict(path=selected.index_path,
                sha256=selected.sha256,size_bytes=selected.size_bytes).items()):
            raise ValueError('The reference points changed. Transfer the map again before adopting it.')


def _new_placement_contracts(record, frame, options, session, layers, *, grid, run_progress=None):
    """GUI metadata-worker capture; old programmatic callers keep V1."""
    if not options.get("continue_photo_edges"):
        return None, None
    from .painting_frame_models import verify_frame_model_selection
    from .photo_edge_continuation import build_contracts
    from .painted_adoption_table import capture_version
    verify_frame_model_selection(options["geometry_selection"])
    project = options["continuation_project"]
    approved = capture_version(record, options.get("approval"), frame, layers, project=project)
    from .painting_run_preview import read_preview_geometry
    models, _binding = read_preview_geometry(options["geometry_index"])
    requested = options.get("continuation_runs", options["runs"])
    contracts = build_contracts(project, options["geometry_index"], session.run_id,
        [str(run).casefold() for run in requested if str(run).casefold() in models], [layer.layer_id for layer in session.layers], target_grid=grid, source_frame=frame,
        **({'run_progress': run_progress} if run_progress is not None else {}))
    verify_frame_model_selection(options["geometry_selection"])
    return contracts, approved


class _AdoptionWorker(QObject):
    """Prepare the adoption of an approved map version on every run's label
    grid (a preparation folder only; nothing is adopted)."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    gap_review_ready = Signal(object)

    def __init__(self, record, frame, options) -> None:
        super().__init__()
        self.record = dict(record)
        self.frame = frame
        self.options = dict(options)
        if self.options.get("display_context") is not None:
            self.options["display_context"] = deepcopy(self.options["display_context"])

    @Slot()
    def run(self) -> None:
        """Check the selected source closure before preparing the saved map."""
        from .raw_shape_basis_reader import project_retained_source_context
        try:
            with project_retained_source_context(self.options.get("project")):
                result = self._prepare()
        except Exception as error:  # noqa: BLE001 - reported to the user
            from .painted_analysis_exclusions import GapReviewRequired
            if isinstance(error, GapReviewRequired):
                self.gap_review_ready.emit(error.review)
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(result)

    def _prepare(self):
        _reviewed_geometry_options(self.frame,self.options)
        if self.options.get("local_reuse") is not None:
            from .local_map_adoption import prepare_local_adoption
            from uuid import uuid4
            request = self.options["local_reuse"]
            review = request["review"]
            result = prepare_local_adoption(review["review_binding"],
                source_context=review["preparation_context"],
                destination=Path(self.options["destination"]) / ("chart-" + uuid4().hex),
                analysis_exclusions=self.options.get("analysis_exclusions"),
                chart_transfers=request["approvals"],
                detection_manifest=request["detection_manifest"])
            result["gui_preparation_actions"] = dict(
                analysis_exclusions=deepcopy(self.options.get("analysis_exclusions")),
                target_corrections=[])
            return result
        from darsia.fluidflower import map_adoption as ma
        from .raw_shape_basis_reader import require_basis_index
        from .painting_frame_models import read_adoption_cohort, verify_adoption_cohort

        cohort = read_adoption_cohort(self.options["geometry_index"], self.options["runs"], self.record["run_id"])
        self.progress.emit(cohort['summary'])
        require_basis_index(self.options["geometry_index"],
                            [self.record["run_id"], *cohort["runs"]])
        self.progress.emit("Drawing the approved map at full resolution…")
        version_layers = lpp.version_layers if self.record.get("proposal_method") == lpp.METHOD else lpm.version_layers
        session, layers = version_layers(self.record, frame=self.frame)
        continuation, approved = _new_placement_contracts(self.record, self.frame,
            dict(self.options, continuation_runs=cohort["runs"]), session, layers, grid="native")
        fixture_options = {}
        if any(session.facies_of(row.layer_id).role == "colorchecker" for row in session.layers):
            from .layer_painting_fixtures import version_fixture_underlay
            underlay = version_fixture_underlay(self.record, frame=self.frame)
            if underlay is not None:
                fixture_options = dict(fixture_underlay=underlay, fixture_frame=self.frame)
        verify_adoption_cohort(cohort)
        result = ma.prepare_adoption(
            self.options["destination"],
            layers,
            reference_run=session.run_id,
            geometry_index=self.options["geometry_index"],
            runs=cohort["runs"],
            roles={
                row.layer_id: session.facies_of(row.layer_id).role for row in session.layers
            },
            names={row.layer_id: row.name for row in session.layers},
            facies={row.layer_id: row.facies for row in session.layers},
            version={
                key: self.record.get(key)
                for key in ("name", "kind", "sha256", "created_utc", "folder")
            }
            | ({"proposal_method": lpp.METHOD} if self.record.get("proposal_method") == lpp.METHOD else {})
            | {"approval": self.options.get("approval")},
            current_labels=self.options.get("current_labels"),
            progress=self.progress.emit,
            **fixture_options,
            painting_version=self.record,
            painting_frame=self.frame,
            painting_project=self.options.get("project"),
            continuation=continuation, approved_source=approved,
            review_findings=True,
            **({"analysis_exclusions": self.options["analysis_exclusions"]}
               if self.options.get("analysis_exclusions") is not None else {}),
            **({"target_corrections": self.options["target_corrections"]}
               if self.options.get("target_corrections") is not None else {}),
            **({"display_context": deepcopy(self.options["display_context"])}
               if continuation is not None and self.options.get("display_context") is not None else {}),
        )
        verify_adoption_cohort(cohort)
        if any(result['geometry_index'].get(key) != val for key, val in cohort['geometry_index'].items()):
            raise ValueError('The preparation used another adoption cohort index')
        result = dict(result, gui_adoption_cohort=cohort)
        if self.options.get('analysis_exclusions') is not None or self.options.get('target_corrections') is not None:
            from .painted_adoption_table import _checked
            completed_approval = (None if result.get('analysis_exclusions') is None else
                _checked(result['analysis_exclusions']['approval'], metadata=True))
            completed_targets = deepcopy((result.get('target_corrections') or {}).get('bindings') or [])
            result['gui_preparation_actions'] = dict(analysis_exclusions=completed_approval, target_corrections=completed_targets)
        return result


class _PlacementWorker(QObject):
    """Place an approved map version in all runs (a review folder only)."""

    progress = Signal(str)
    run_progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    gap_review_ready = Signal(object)

    def __init__(self, record, frame, options) -> None:
        super().__init__()
        self.record = dict(record)
        self.frame = frame
        self.options = dict(options)
        if self.options.get("display_context") is not None:
            self.options["display_context"] = deepcopy(self.options["display_context"])

    @Slot()
    def run(self) -> None:
        from .raw_shape_basis_reader import project_retained_source_context
        try:
            with project_retained_source_context(self.options.get("project")):
                result = self._place()
        except Exception as error:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(result)

    def _place(self):
        from darsia.fluidflower import map_placement as mp
        from .painting_frame_models import verify_frame_model_selection

        _reviewed_geometry_options(self.frame,self.options)
        verify_frame_model_selection(self.options.get("geometry_selection"))
        self.progress.emit("Drawing the approved map at full resolution…")
        version_layers = lpp.version_layers if self.record.get("proposal_method") == lpp.METHOD else lpm.version_layers
        session, layers = version_layers(self.record, frame=self.frame)
        continuation, approved = _new_placement_contracts(self.record, self.frame, self.options,
            session, layers, grid="camera_u", run_progress=self.run_progress.emit)
        roles = {
            row.layer_id: session.facies_of(row.layer_id).role for row in session.layers
        }
        version = {
            key: self.record.get(key)
            for key in ("name", "kind", "sha256", "created_utc", "folder")
        } | ({"proposal_method": lpp.METHOD} if self.record.get("proposal_method") == lpp.METHOD else {}) | {"approval": self.options.get("approval")}
        native_review_builder = None
        if continuation is not None:
            from .painted_analysis_exclusions import build_transfer_gap_review, _same
            native_contracts, native_approved = _new_placement_contracts(self.record, self.frame,
                self.options, session, layers, grid="native", run_progress=self.run_progress.emit)
            if not _same(approved, native_approved):
                raise ValueError("The native findings and camera-U figures must bind the same saved painting")
            underlay = None
            if any(session.facies_of(row.layer_id).role == "colorchecker" for row in session.layers):
                from .layer_painting_fixtures import version_fixture_underlay
                underlay = version_fixture_underlay(self.record, frame=self.frame)
            def native_review_builder(folder):
                return build_transfer_gap_review(folder, layers, painting_binding=native_approved,
                    geometry_index=self.options["geometry_index"], map_version=version,
                    contracts=native_contracts, fixture_underlay=underlay, progress=self.progress.emit,
                    run_progress=self.run_progress.emit,
                    **({"display_context": deepcopy(self.options["display_context"])}
                       if native_contracts is not None and self.options.get("display_context") is not None else {}))
        result = mp.place_map_in_runs(
            self.options["destination"],
            layers,
            reference_run=session.run_id,
            geometry_index=self.options["geometry_index"],
            runs=self.options["runs"],
            photo_loader=self.options["photo_loader"],
            roles=roles,
            names={row.layer_id: row.name for row in session.layers},
            facies={row.layer_id: row.facies for row in session.layers},
            version=version,
            first_runs=self.options.get("first_runs") or (),
            continuation=continuation, approved_source=approved,
            **({"native_review_builder": native_review_builder} if native_review_builder is not None else {}),
            **({"display_context": deepcopy(self.options["display_context"])}
               if continuation is not None and self.options.get("display_context") is not None else {}),
            progress=self.progress.emit,
            run_progress=self.run_progress.emit,
        )
        verify_frame_model_selection(self.options.get("geometry_selection"))
        if result.get("gap_review") is not None:
            from .painted_analysis_exclusions import read_gap_review
            self.progress.emit('Opening the completed transfer review…')
            self.gap_review_ready.emit(read_gap_review(result["gap_review"]["path"]))
        return result


class _SaveWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        project: Any,
        session: lp.PaintingSession,
        proposal: lp.LayerProposal,
        reference: SeriesRunInputs,
        frame: RigFrame | None,
        actor: str,
        *,
        corrections: Sequence[lp.Correction] = (),
    ) -> None:
        super().__init__()
        self.project = project
        self.session = session
        self.proposal = proposal
        self.reference = reference
        self.frame = frame
        self.actor = actor
        self.corrections = tuple(corrections)

    @Slot()
    def run(self) -> None:
        try:
            save_revision = lpp.save_painted_revision if lpp.is_polygons_only(self.proposal) else lp.save_painted_revision
            save_revision(
                self.project,
                self.session,
                self.proposal,
                self.reference,
                frame=self.frame,
                actor=self.actor,
                progress=self.progress.emit,
                corrections=self.corrections,
            )
            self.progress.emit("Exporting for native Setup…")
            export = export_native_setup(
                self.project, reference=self.reference, actor=self.actor, replace=True
            )
        except ValueError as error:
            self.failed.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(export)


# ---------------------------------------------------------------------------
# View


class LayerPainterView(QWidget):
    """Paint layers, propose boundaries, group facies and export."""

    exported = Signal(object)
    changed = Signal()
    display_styles_changed = Signal()
    # True while a proposal or a save runs; False when it has finished.
    busy_changed = Signal(bool)
    workflow_scope_requested = Signal(str)

    def __init__(
        self,
        project: Any,
        reference: Callable[[], SeriesRunInputs | None],
        parent: QWidget | None = None,
        *,
        run_inputs: Callable[[], Sequence[SeriesRunInputs]] | None = None,
        set_reference: Callable[[str], Any] | None = None,
    ) -> None:
        """Build the painting editor and its explicit saved-version workflow."""
        super().__init__(parent)
        self.setObjectName("seriesLayerView")
        self.project = project
        self._reference = reference
        self._static_reference_runs = run_inputs
        self._workspace_frame_config: dict[str, Any] = {}
        # True while a session painted on the raw reference photo is shown in
        # a project whose geometry is in the saved corrected rig frame.
        self._raw_session_frame = False
        # The reference photograph on the fine proposal grid: (source, fine image).
        self._fine_image: tuple[lp.PaintingImage, lp.PaintingImage] | None = None
        self._saved_corrected_boundaries: np.ndarray | None = None
        self.session: lp.PaintingSession | None = None
        self.image: lp.PaintingImage | None = None
        self.proposal: lp.LayerProposal | None = None
        self.last_export: Any = None
        # Snapshots of (polygons, strokes, corrections) for undo and redo.
        self._undo_stack: list[_Snapshot] = []
        self._redo_stack: list[_Snapshot] = []
        # Hand corrections drawn on top of the proposal, in drawing order.
        # They have their own file; drawing them never writes the session.
        self.corrections: list[lp.Correction] = []
        self._corrections_dirty = False
        # Why the run's correction file could not be read; no correction is
        # drawn or saved then, so that the file is never overwritten.
        self._corrections_error = ""
        # The corrected proposal on the display grid, extended as corrections
        # are added.
        self._display_corrections = _CorrectedGrid()
        self._native_corrections_key = None
        self._native_corrections_state = None
        self._native_corrections_gaps = 0
        self._native_corrections_gap_mask = None
        self._native_corrections_cache = {}
        self._native_display_key = self._native_display_state = None
        self._new_export_check_key = self._new_export_check_problems = None
        self._new_map_digest_key = self._new_map_digest = None
        self._native_async = NativeProposalCache(self)
        self._native_async.changed.connect(self._native_cache_changed)
        # Saved versions of the run's map, oldest first (records with a folder).
        self.versions: list[dict[str, Any]] = []
        self._version_preview_identity = None
        self._version_snapshot_session = None
        # Approvals of the run's saved versions, oldest first, and why they
        # could not be read ("" when they could).
        self.map_approvals: list[dict[str, Any]] = []
        self._map_approvals_error = ""
        # Layers hidden in the photograph: a display choice for the session
        # shown, never saved and never part of the proposal.
        self._hidden_layers: set[int] = set()
        self._hidden_facies: set[str] = set()
        self._hidden_session: lp.PaintingSession | None = None
        self._color_mode = self._stored_color_mode()
        self._highlight_facies: str | None = None
        self._layer_table_menu_widget: QMenu | None = None
        self._facies_table_menu_widget: QMenu | None = None
        self._dirty = False
        self._export_stale = True
        self._export_key = ""
        self._loading = False
        self.analysis_boxes = None
        self._analysis_boxes_load_key = None
        self._analysis_boxes_reload_pending = False
        self._analysis_boxes_syncing = False
        self._analysis_boxes_closed = False
        self._analysis_boxes_window = None
        self._analysis_boxes_preview_items = []
        self._analysis_boxes_preview_key = None
        self.measured_edges = None
        self.other_frames = None
        self._measured_edges_reload_pending = False
        self._measured_edges_syncing = False
        self._waiting_for_workers = False
        self._facies_menu_open = False
        self._facies_refresh_pending = False
        self._facies_buttons: dict[str, QToolButton] = {}
        self._polygon_menu_widget: QMenu | None = None
        self._other_photo: Path | None = None
        self._workflow_scope = 'all'
        self._rig_previous_preview_run = None
        from .rig_photo_loader import RigPhotoLoader
        self._rig_photo_loader = RigPhotoLoader()
        self._rig_photo_loader.ready.connect(self._rig_photo_ready)
        self.destroyed.connect(self._rig_photo_loader.dispose)
        self._rig_photo_epoch = 0
        self._rig_photo_key = None
        self._rig_photo_packet = None
        self._rig_photo_path = None
        self._rig_photo_display_reference = None
        self._rig_photo_reset = True
        self._reference_token: dict[str, Any] | None = None
        self._plain: np.ndarray | None = None
        self._enhanced: np.ndarray | None = None
        # Another photograph of the reference run, shown under the marks
        # instead of the reference photo when chosen (display only).
        self._background_path: Path | None = None
        self._background_rgb: np.ndarray | None = None
        self._background_plain: np.ndarray | None = None
        self._background_enhanced: np.ndarray | None = None
        self._background_token: dict[str, Any] | None = None
        self._background_shown = False
        # The reference run's configured camera-U profile (N106), cached per run.
        self._camera_u_key: tuple[str, str] | None = None
        self._camera_u: Any = None
        # The reference and background photos at full resolution (uint8),
        # plain and with enhanced contrast, shown when zoomed in.
        self._detail_plain: np.ndarray | None = None
        self._detail_enhanced: np.ndarray | None = None
        self._background_detail_plain: np.ndarray | None = None
        self._background_detail_enhanced: np.ndarray | None = None
        self._proposal_thread: QThread | None = None
        self._proposal_worker: _ProposalWorker | None = None
        self._save_thread: QThread | None = None
        self._save_worker: _SaveWorker | None = None
        # Placing an approved map version in all runs (a review folder only).
        self._placement_thread: QThread | None = None
        self._placement_worker: _PlacementWorker | None = None
        self.last_placement: dict[str, Any] | None = None
        self._placement_dialog: QDialog | None = None
        self._geometry_dialog: QDialog | None = None
        # Preparing an approved version's adoption (a preparation folder only).
        self._adoption_thread: QThread | None = None
        self._adoption_worker: _AdoptionWorker | None = None
        self.last_adoption: dict[str, Any] | None = None
        self._adoption_message: QMessageBox | None = None
        self._adoption_dialog: QDialog | None = None
        self._saved_transfers_dialog: QDialog | None = None
        # Installed only by the separately reviewed automatic RAW adoption UI.
        self._series_adopt_action = None
        # The manual map (the polygons as drawn, at full resolution) and the
        # proposal's full-resolution map it is compared with, each with the
        # key of what it was drawn from; display data, saved only as a version.
        self._manual: lpm.ManualMap | None = None
        self._manual_key = ""
        self._manual_run = ""
        self._compare_layers: np.ndarray | None = None
        self._compare_key = ""
        self._compare_run = ""
        self._manual_display: dict[str, Any] | None = None
        # The worker's progress, or why the manual map could not be drawn.
        self._manual_message = ""
        self._manual_thread: QThread | None = None
        self._manual_worker: _ManualMapWorker | None = None
        self._manual_pending = False
        # The manual map is drawn again once the marks rest for a moment.
        self._manual_timer = QTimer(self)
        self._manual_timer.setSingleShot(True)
        self._manual_timer.setInterval(500)
        self._manual_timer.timeout.connect(self._update_manual_map)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self.save_session)
        # The saturation is applied once the slider rests for a moment.
        self._saturation_timer = QTimer(self)
        self._saturation_timer.setSingleShot(True)
        self._saturation_timer.setInterval(150)
        self._saturation_timer.timeout.connect(self.apply_saturation)

        self.canvas = LayerCanvas()
        self.canvas.stroke_finished.connect(self._stroke_finished)
        self.canvas.polygon_finished.connect(self._polygon_finished)
        self.canvas.polygon_edited.connect(self._polygon_edited)
        self.canvas.polygons_edited.connect(self._polygons_edited)
        self.canvas.polygon_context.connect(self._polygon_menu)
        self.canvas.draft_changed.connect(self._draft_changed)
        self.canvas.notice.connect(self._canvas_notice)
        self.canvas.correction_drawn.connect(self._correction_drawn)
        self.canvas.fill_requested.connect(self._fill_requested)

        from darsia.fluidflower.series_rig_frame_panel import RigFramePanel

        self.rig_panel = RigFramePanel(self, project, reference, scrollable=False)
        self.rig_panel.changed.connect(self._frame_changed)
        static = getattr(self.rig_panel, 'static_references', None)
        if static is not None:
            static.busy_changed.connect(self.busy_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_steps())
        from darsia.fluidflower.painting_run_preview_ui import PaintingRunPreview
        self.run_preview = PaintingRunPreview(self, self._build_canvas_column(), LayerCanvas,
            run_inputs=run_inputs, set_reference=set_reference)
        self.display_styles_changed.connect(self.run_preview.redraw_display)
        from .painting_analysis_boxes_ui import PaintingAnalysisBoxesController
        self.analysis_boxes = PaintingAnalysisBoxesController(self)
        self.step_frame.content.layout().addWidget(self.analysis_boxes.widget)
        self.analysis_boxes.changed.connect(self._analysis_boxes_changed)
        self.analysis_boxes.saved.connect(self._analysis_boxes_changed)
        self.analysis_boxes.busy_changed.connect(self._analysis_boxes_busy_changed)
        from .painting_measured_edges_ui import PaintingMeasuredEdgesController
        self.measured_edges = PaintingMeasuredEdgesController(self)
        self.step_frame.content.layout().addWidget(self.measured_edges.widget)
        self.measured_edges.busy_changed.connect(self._measured_edges_busy_changed)
        self.busy_changed.connect(self._refresh_measured_edges)
        self.run_preview.geometry_button.clicked.connect(self._measured_edges_context_changed)
        self.rig_panel.mark_corners.toggled.connect(self._refresh_measured_edges)
        self.rig_panel.place_reference.toggled.connect(self._refresh_measured_edges)
        self.rig_panel.place_point.toggled.connect(self._refresh_measured_edges)
        self.rig_panel.mark_crop.toggled.connect(self._refresh_measured_edges)
        from .painting_other_frames_ui import PaintingOtherFramesController
        self.other_frames = PaintingOtherFramesController(self)
        self.step_frame.content.layout().addWidget(self.other_frames.widget)
        self.other_frames.busy_changed.connect(self._other_frames_busy_changed)
        from .painting_fixture_preview import PaintingFixturePreview
        self.fixture_preview = PaintingFixturePreview(self)
        self.run_preview.widget.layout().addWidget(self.fixture_preview.widget)
        self.fixture_preview.busy_changed.connect(self._fixture_preview_busy_changed)
        for button in (self.analysis_boxes.draw_button, self.analysis_boxes.edit_button):
            button.toggled.connect(self._analysis_boxes_tool_changed)
        self.run_preview.display_changed.connect(self._sync_analysis_boxes)
        self.run_preview.widget.destroyed.connect(self._invalidate_analysis_boxes)
        self.run_preview.canvas.destroyed.connect(self._invalidate_analysis_boxes)
        self.run_preview.geometry_button.clicked.connect(lambda: self._sync_analysis_boxes(force=True))
        self.destroyed.connect(self._invalidate_analysis_boxes)
        self.run_preview.runs.currentIndexChanged.connect(self._fill_region_context_changed)
        self.busy_changed.connect(self._refresh_fill_region)
        self.busy_changed.connect(self._sync_analysis_boxes)
        self.rig_panel.mark_corners.toggled.connect(self._refresh_fill_region)
        self.rig_panel.place_reference.toggled.connect(self._refresh_fill_region)
        self.rig_panel.place_point.toggled.connect(self._refresh_fill_region)
        self.rig_panel.mark_crop.toggled.connect(self._refresh_fill_region)
        self.rig_panel.mark_corners.toggled.connect(self._sync_analysis_boxes)
        self.rig_panel.place_reference.toggled.connect(self._sync_analysis_boxes)
        self.rig_panel.place_point.toggled.connect(self._sync_analysis_boxes)
        self.rig_panel.mark_crop.toggled.connect(self._sync_analysis_boxes)
        splitter.addWidget(self.run_preview.widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([580, 1020])
        self.splitter = splitter
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self._install_shortcuts()
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.wait_for_workers)
            application.focusChanged.connect(self._refresh_analysis_box_undo)
        from .painting_series_adoption_ui import SeriesAdoptionController
        from .painting_guided_review_bindings import guided_gap_dialog, RegionPreviewSession, dispatch_findings
        preview_session = RegionPreviewSession()
        self.adoption_controller = SeriesAdoptionController(self, gap_dialog_factory=
            lambda source, **options: guided_gap_dialog(source, preview_session=preview_session,
                                                        series_root=self._series_root(), **options),
            catalogue_refresher=lambda request: dispatch_findings(self.adoption_controller, request))
        self.adoption_controller.findings_changed.connect(self._refresh_use_map)
        self.run_preview.display_changed.connect(self._refresh_transfer_runs)
        self.run_preview.runs.currentIndexChanged.connect(self._sync_transfer_preparation)
        self._transfer_review_active = False
        self.step_paint.toggle.clicked.connect(self.show_mark_layers)
        self.run_preview.stack.currentChanged.connect(self._keep_transfer_review_visible)
        self._series_adopt_action = (self.adoption_controller.start
            if self.adoption_controller.available else None)
        self._configure_rig_sections()
        self.reload()

    # ------------------------------------------------------------ layout

    def set_workflow_scope(self, scope: str) -> None:
        """Show a sidebar route on the same drawing and transfer-review session."""
        if scope not in {'all', 'layers', 'apply', 'rig_frame', 'analysis_boxes'}:
            raise ValueError('Unknown geometry workflow scope')
        previous = self._workflow_scope
        self._workflow_scope = scope
        if previous == 'analysis_boxes' and previous != scope:
            self.analysis_boxes.set_mode(None)
            for control, enabled in getattr(self, '_box_scope_controls', ()):
                control.setEnabled(enabled)
            for shortcut, enabled in getattr(self, '_box_scope_shortcuts', ()):
                shortcut.setEnabled(enabled)
            self._box_scope_controls, self._box_scope_shortcuts = [], []
            self._set_rig_photo_tools(False)
        if scope != 'rig_frame':
            self._point_review_active = False
            if previous == 'rig_frame':
                self.show_mark_layers()
                self._set_rig_photo_tools(False)
        if previous != scope and not ({previous, scope} <= {'layers', 'rig_frame'}):
            self._invalidate_rig_photo()
        editing = scope in {'all', 'layers'}
        self.import_map_start.setVisible(editing)
        self.workflow_intro.setVisible(editing)
        self.step_frame.setVisible(editing or scope == 'rig_frame')
        self._configure_rig_sections()
        for section in (self.step_paint, self.step_facies, self.step_propose):
            section.setVisible(editing)
        self.step_use_map.setVisible(scope in {'all', 'apply'})
        self._set_transfer_review_space(scope == 'apply')
        if scope == 'layers' and previous != scope:
            # The rig route expands this same section into several editors.
            # Keep that temporary expansion out of the normal painting route.
            self.step_frame.set_expanded(False)
            self.step_paint.set_expanded(True)
        if previous == 'rig_frame' and scope != previous:
            self.rig_panel._set_mode('')
            for control, enabled in self._rig_control_states:
                control.setEnabled(enabled)
            for shortcut, enabled in self._rig_nav_states:
                shortcut.setEnabled(enabled)
            index = self.run_preview.runs.findData(self._rig_previous_preview_run)
            if index >= 0:
                blocked = self.run_preview.runs.blockSignals(True)
                self.run_preview.runs.setCurrentIndex(index)
                self.run_preview.runs.blockSignals(blocked)
            self._show_reference(reset_view=True)
            self.run_preview._selected()
            self._version_preview_identity = None
            self._version_selected()
        if scope == 'apply':
            self.step_use_map.set_expanded(True)
            self._guard_reference_display_controls()
            self.show_transfer_review_prompt()
            self.adoption_controller.open_chart_review(automatic=True)
        else:
            self.show_mark_layers()
            if scope in {'layers', 'analysis_boxes'} and previous != 'rig_frame':
                self._show_reference(reset_view=previous != scope)
            if scope == 'analysis_boxes':
                if previous != scope:
                    self._box_scope_controls = [(control, control.isEnabled()) for control in self._rig_preview_controls()]
                    self._box_scope_shortcuts = [(shortcut, shortcut.isEnabled())
                        for shortcut in [*self.canvas.findChildren(QShortcut), *self.run_preview.navigation_keys]
                        if shortcut.key().toString() not in {'F', 'H'}]
                self.step_analysis_boxes.set_expanded(True)
                self._show_analysis_boxes_editor()
            if scope == 'rig_frame':
                self.step_frame.set_expanded(True)
                if previous != scope:
                    self._rig_control_states = [(control, control.isEnabled())
                        for control in self._rig_preview_controls()]
                    self._rig_nav_states = [(shortcut, shortcut.isEnabled())
                        for shortcut in self.run_preview.navigation_keys]
                    # A saved-map reader already in flight may finish, but it
                    # no longer owns the photo while the rig editor is open.
                    self.run_preview._version_requested = None
                    self.run_preview._reference_version_key = None
                    self._rig_previous_preview_run = self.run_preview.runs.currentData()
                    reference = self.reference()
                    index = self.run_preview.runs.findData(reference.run_id) if reference is not None else -1
                    if index >= 0:
                        blocked = self.run_preview.runs.blockSignals(True)
                        self.run_preview.runs.setCurrentIndex(index)
                        self.run_preview.runs.blockSignals(blocked)
                self._show_rig_editor()
                self._show_reference(reset_view=True)
            if scope == 'layers' and previous != scope:
                # The transfer/box editor may own the stack while an immutable
                # version remains selected. Restore that selection's viewer.
                self.run_preview._selected()

        self.analysis_boxes.refresh()
        self._refresh_analysis_box_preview()

    def _configure_rig_sections(self) -> None:
        """Split only the rig route, retaining the existing painting step layout."""
        rig_scope = self._workflow_scope == 'rig_frame'
        box_scope = self._workflow_scope == 'analysis_boxes'
        self.rig_curvature_host.setVisible(rig_scope and self.rig_curvature_host.layout().count() > 0)
        self.step_depth.setVisible(rig_scope)
        self.step_frame.toggle.setText('Reference quad' if rig_scope
            else 'Reference points, injection points and analysis boxes' if self._workflow_scope == 'all'
            else 'Rig Geometry')
        self.step_analysis_boxes.toggle.setText('Analysis boxes')
        self.rig_panel.move_injection_controls(self.injection_host if rig_scope else None)
        boxes = getattr(self, 'analysis_boxes', None)
        if boxes is not None:
            destination = self.step_frame.content if self._workflow_scope == 'all' else self.analysis_boxes_host
            measured = getattr(self, 'measured_edges', None)
            before = destination.layout().indexOf(measured.widget) if not rig_scope and measured is not None else -1
            if before >= 0:
                destination.layout().insertWidget(before, boxes.widget)
            else:
                destination.layout().addWidget(boxes.widget)
            boxes.widget.setVisible(box_scope or self._workflow_scope == 'all')
        for section in (self.step_injection, self.step_point_review):
            section.setVisible(rig_scope)
        self.step_analysis_boxes.setVisible(box_scope)
        if not rig_scope:
            self._set_point_review_space(False)
        self.measured_geometry_button.setVisible(not rig_scope and not box_scope)
        self.rig_panel.review_points.hide()
        for controller in (getattr(self, 'measured_edges', None), getattr(self, 'other_frames', None)):
            if controller is not None:
                controller.widget.setVisible(not rig_scope and not box_scope)
        if rig_scope and not hasattr(self, '_rig_aux_visibility'):
            self._rig_aux_visibility = []
            for controller, name in ((getattr(self, 'measured_edges', None), 'show_edges'),
                                     (getattr(self, 'other_frames', None), 'show_frames')):
                control = getattr(controller, name, None)
                if control is not None:
                    self._rig_aux_visibility.append((control, control.isChecked()))
                    blocked = control.blockSignals(True)
                    control.setChecked(False)
                    control.blockSignals(blocked)
                    controller.refresh()
        elif not rig_scope and hasattr(self, '_rig_aux_visibility'):
            for control, checked in self._rig_aux_visibility:
                blocked = control.blockSignals(True)
                control.setChecked(checked)
                control.blockSignals(blocked)
            del self._rig_aux_visibility
            self._refresh_measured_edges()

    def _rig_edit_section_clicked(self, expanded: bool, selected=None) -> None:
        if expanded and self._workflow_scope == 'rig_frame' and getattr(self, '_point_review_active', False):
            self.close_reference_point_review()
            if selected is not None:
                for section in (self.step_frame, self.step_injection, self.step_analysis_boxes):
                    section.set_expanded(section is selected)

    def _set_point_review_space(self, active: bool) -> None:
        """Give the active run list the remaining height of the left pane."""
        layout = self._steps_layout
        layout.setStretch(layout.indexOf(self.step_point_review), 1 if active else 0)
        layout.setStretch(self._steps_bottom_spacer, 0 if active else 1)

    def _set_transfer_review_space(self, active: bool) -> None:
        """Give transfer approval the same full-height run list as point review."""
        layout = self._steps_layout
        layout.setStretch(layout.indexOf(self.step_use_map), 1 if active else 0)
        self.step_use_map.layout().setStretch(1, 1 if active else 0)
        self.use_map_runs.setMaximumHeight(16777215 if active else 320)
        point_review = self._workflow_scope == 'rig_frame' and getattr(self, '_point_review_active', False)
        layout.setStretch(self._steps_bottom_spacer, 0 if active or point_review else 1)

    def _point_review_section_clicked(self, expanded: bool) -> None:
        if self._workflow_scope != 'rig_frame':
            return
        if expanded:
            self.open_reference_point_review()
        elif getattr(self, '_point_review_active', False):
            self.close_reference_point_review()

    def _rig_preview_controls(self):
        preview = self.run_preview
        return (preview.runs, preview.previous, preview.next, preview.reference_button,
            preview.sources, preview.refresh_button, preview.geometry_button)

    def _rig_overlay_visibility_changed(self, *_args) -> None:
        if self._workflow_scope == 'rig_frame':
            self.rig_panel.set_overlay_visibility(points=self.rig_show_points.isChecked(),
                lines=self.rig_show_lines.isChecked())

    def _set_rig_photo_tools(self, active: bool) -> None:
        """Temporarily replace painting controls with the compact rig toolbar."""
        if active and not hasattr(self, '_rig_photo_visibility'):
            controls = [*self._painting_photo_groups, self.photo_return_button]
            def collect(layout):
                for index in range(layout.count()):
                    item = layout.itemAt(index)
                    if item.widget() is not None and item.widget() is not self.run_preview.stack:
                        controls.append(item.widget())
                    elif item.layout() is not None:
                        collect(item.layout())
            collect(self.run_preview.widget.layout())
            self._rig_photo_visibility = [(control, control.isHidden()) for control in controls]
            for control, _hidden in self._rig_photo_visibility:
                control.hide()
        elif not active and hasattr(self, '_rig_photo_visibility'):
            for control, hidden in self._rig_photo_visibility:
                control.setVisible(not hidden)
            del self._rig_photo_visibility
        self.rig_photo_toolbar.setVisible(active and not getattr(self, '_point_review_active', False))
        self.rig_panel.set_overlay_visibility(points=self.rig_show_points.isChecked() if active else True,
            lines=self.rig_show_lines.isChecked() if active else True)

    def static_reference_runs(self):
        """Capture registered inputs on the GUI thread before a finite read."""
        if self._static_reference_runs is not None:
            return list(self._static_reference_runs())
        reference = self.reference()
        return [] if reference is None else [reference]

    def static_reference_context(self):
        """Only the loaded reference's full field may define static areas."""
        reference = self.static_reference_current_run()
        run = reference.run_id if reference is not None else ''
        token = self.canvas.preview or {}
        ready = (self._workflow_scope == 'rig_frame' and reference is not None
            and reference.baseline_image is not None and not self._rig_photo_loader.busy()
            and self._rig_photo_packet is not None and token.get('run_id') == run
            and token.get('image_rgb') is not None
            and Path(token.get('path') or '') == Path(reference.baseline_image))
        return run, bool(ready)

    def static_reference_current_run(self):
        return self._rig_photo_display_reference or self.reference()

    def show_static_reference_run(self, run_id):
        """Browse static-area observations without changing the map reference."""
        if self._workflow_scope != 'rig_frame':
            return False
        reference = next((row for row in self.static_reference_runs() if row.run_id == run_id), None)
        if reference is None or reference.baseline_image is None:
            return False
        return self._request_rig_photo(Path(reference.baseline_image), reset_view=False,
            display_reference=reference)

    def use_static_reference_frame(self, run_id, corners_rc, width_m, height_m):
        """Save the explicitly selected run's frame, never its neighbour's."""
        from .series_geometry import save_rig_frame
        reference = self.static_reference_current_run()
        if (reference is None or reference.run_id != run_id
                or self.static_reference_context() != (run_id, True)):
            return False
        points = np.asarray(corners_rc, dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError('The measured rig needs four finite camera-U corners')
        previous = saved_rig_frames(self.project).get(run_id)
        frame = RigFrame(dict(zip(('top_left', 'top_right', 'bottom_right', 'bottom_left'),
            [tuple(point) for point in points])), width_m=float(width_m), height_m=float(height_m),
            run_id=run_id, source='computed from static reference areas', image=str(reference.baseline_image),
            image_size=tuple(self.canvas.raw_size),
            disabled_corners=previous.disabled_corners if previous is not None else ())
        from darsia.fluidflower.series_decisions import project_series_root, series_actor
        save_rig_frame(self.project, frame, actor=series_actor(project_series_root(self.project)))
        if self.reference() is not None and self.reference().run_id == run_id:
            self.rig_panel.frame = frame
            self.rig_panel._loading = True
            try:
                self.rig_panel._show_frame()
            finally:
                self.rig_panel._loading = False
            self.rig_panel._draw_overlay()
            self.rig_panel.changed.emit()
        return True

    def read_static_reference_packet(self, reference):
        """Read a run's own complete photograph without using Qt state."""
        from .rig_photo_loader import read_static_reference_packet
        profile = lp.reference_camera_u(self.project, reference)
        return read_static_reference_packet(reference, profile)

    def _static_reference_context_changed(self):
        panel = getattr(getattr(self, 'rig_panel', None), 'static_references', None)
        if panel is not None:
            panel.context_changed()

    def _show_rig_editor(self) -> None:
        """Temporarily use the working canvas without changing the saved map choice."""
        if getattr(self,'_point_review_active',False):
            self._set_rig_photo_tools(True)
            self.rig_panel.setEnabled(False)
            self.run_preview.stack.setCurrentWidget(self._point_review_panel)
            return
        self.set_reference_display_mode(True)
        self._set_rig_photo_tools(True)
        self.run_preview.stack.setCurrentIndex(0)
        self.rig_panel.setEnabled(workspace_frame.frame_kind(self._workspace_frame_config) == 'raw'
            and not (self._rig_photo_key is not None and self._rig_photo_packet is None))
        for control in self._rig_preview_controls():
            control.setEnabled(False)
        for shortcut in self.run_preview.navigation_keys:
            shortcut.setEnabled(False)
        reference = self.reference()
        self.run_preview.note.setText('Rig editor · reference run ' + (reference.run_id if reference else ''))

    def _show_analysis_boxes_editor(self) -> None:
        """Use the original measured-box editor without enabling Setup edits."""
        self.set_reference_display_mode(True)
        self._set_rig_photo_tools(True)
        self.rig_photo_toolbar.hide()
        self.run_preview.stack.setCurrentIndex(0)
        for control, _enabled in getattr(self, '_box_scope_controls', ()):
            control.setEnabled(False)
        for shortcut, _enabled in getattr(self, '_box_scope_shortcuts', ()):
            shortcut.setEnabled(False)
        self.rig_panel._set_mode('')
        self.rig_panel.setEnabled(False)
        self.rig_panel.set_overlay_visibility(points=False, lines=False)
        self.canvas.paint_enabled = False
        self.canvas.set_tool('pan')
        self.canvas.set_overlays_visible(False, False)
        self.analysis_boxes.widget.setEnabled(True)
        self.analysis_boxes.refresh()

    def open_reference_point_review(self):
        """Review each run's same physical points in the existing full-photo area."""
        frame = self.current_frame()
        if frame is None or not frame.complete() or not frame.has_size():
            self.step_point_review.set_expanded(False)
            self.step_frame.set_expanded(True)
            self._set_point_review_space(False)
            self.rig_panel.reference_error.setText('Define four visible reference points and their physical dimensions first.')
            return
        self.set_workflow_scope('rig_frame')
        for section in (self.step_frame, self.step_injection, self.step_analysis_boxes):
            section.set_expanded(False)
        self.step_point_review.set_expanded(True)
        self._set_point_review_space(True)
        panel = getattr(self,'_point_review_panel',None)
        if panel is None:
            from .rig_point_review_ui import RigPointReviewPanel
            panel = RigPointReviewPanel(self)
            self._point_review_panel = panel
            panel.changed.connect(self._refresh_use_map)
            panel.changed.connect(self.adoption_controller._queue_automatic_chart_review)
            panel.return_requested.connect(self.close_reference_point_review)
        self._point_review_active = True
        self.rig_panel.setEnabled(False)
        self.show_transfer_review(panel)
        panel.prepare()

    def close_reference_point_review(self):
        self._point_review_active = False
        self.step_point_review.set_expanded(False)
        self.step_frame.set_expanded(True)
        self._set_point_review_space(False)
        self.show_mark_layers()
        self._show_rig_editor()
        self._show_reference(reset_view=False)

    def reviewed_point_selection(self,base=None):
        cached=getattr(self,'_reviewed_point_selection',None)
        if base is None:
            base=getattr(getattr(self,'run_preview',None),'_geometry_selection',None)
        if (cached and (base is None or cached['base']==base)
                and cached['document']==load_point_review(self.project)
                and self.current_frame() is not None and cached['frame']==self.current_frame().to_dict()):
            return cached['selection']
        return None

    def _restore_reviewed_point_selection(self):
        document=load_point_review(self.project)
        if document is None or self.reviewed_point_selection() is not None:
            return
        required=document.get('required_runs') or list(document.get('runs') or {})
        if not required or any(document['runs'][run]['status']!='approved' for run in required):
            return
        panel=getattr(self,'_point_review_panel',None)
        if panel is None:
            from .rig_point_review_ui import RigPointReviewPanel
            panel=RigPointReviewPanel(self)
            self._point_review_panel=panel
            panel.hide()
            panel.changed.connect(self._refresh_use_map)
            panel.changed.connect(self.adoption_controller._queue_automatic_chart_review)
            panel.return_requested.connect(self.close_reference_point_review)
        panel.document=document
        panel._runs={row.run_id.casefold():row for row in self.static_reference_runs()}
        panel.prepare_alignment()

    def _reference_point_review_blocker(self):
        """Readiness of the reference point review; the worker rechecks every photo before transfer.

        The check resolves every run's photograph path, which takes seconds
        on a series share, and several controls ask on each refresh. It runs
        in the background (_point_review_problem).
        """
        from .series_geometry import native_series_link
        if not native_series_link(self.project).get('series_root'):
            return None
        from .rig_point_review import load_review
        document = load_review(self.project)
        frame = self.current_frame()
        if document is None or frame is None:
            return 'Review and approve the reference points in all transfer runs first.'
        required = document.get('required_runs') or list(document['runs'])
        required = list(dict.fromkeys([frame.run_id.casefold(),*required]))
        problem = self._point_review_problem(document, frame, required)
        return problem or (None if self.reviewed_point_selection() is not None
            else 'Preparing approved point geometry. Open reference point review if it needs attention.')

    def _point_review_problem(self, document, frame, required):
        """review_blocker's answer for this review, frame and runs, read off the GUI thread.

        A quick check answers at once, as before. A slow one answers later
        (_point_review_checked); until then the review counts as being
        checked. A slow check's answer is shown again for a few seconds, and
        then while the next check runs.
        """
        key = json.dumps([document, frame.to_dict(), required], sort_keys=True, default=str)
        last = self.__dict__.get('_point_review_last')
        known = last is not None and last[0] == key
        if known and last[3] and time.monotonic() - last[1] < _SLOW_CHECK_FRESH_S:
            return last[2]
        reader = self.__dict__.get('_point_review_reader')
        if reader is None:
            from .background_read import BackgroundRead
            reader = self._point_review_reader = BackgroundRead(self)
            reader.answered.connect(self._point_review_checked)
        if reader.busy() and reader.key == key:
            return last[2] if known else _POINT_REVIEW_CHECKING
        from .rig_point_review import review_blocker
        document, frame = deepcopy(document), deepcopy(frame)
        bindings = {run: row['photo'] for run, row in document['runs'].items()}
        done, problem, error = reader.read(
            key, lambda: review_blocker(document, frame, required, bindings),
            call=("darsia.fluidflower.rig_point_review:review_blocker", (document, frame, required, bindings)),
            wait_s=0.0 if known and last[3] else None)
        if done:
            return self._keep_point_review_problem(key, problem, error, slow=False)
        return last[2] if known else _POINT_REVIEW_CHECKING

    def _keep_point_review_problem(self, key, problem, error, *, slow):
        if error:
            problem = f'The reference point review could not be checked: {error}'
        self._point_review_last = (key, time.monotonic(), problem, slow)
        return problem

    def _point_review_checked(self, key, problem, error) -> None:
        """Show a slow reference point check's answer in the controls that ask for it."""
        last = self.__dict__.get('_point_review_last')
        problem = self._keep_point_review_problem(key, problem, error, slow=True)
        if last is None or last[0] != key or last[2] != problem:
            self._refresh_versions()
        self._background_reading_answered()

    def _background_reading_answered(self) -> None:
        """Let an automatic review that waited for a background reading of this view try again."""
        controller = getattr(self, 'adoption_controller', None)
        if controller is not None:
            controller.background_reading_answered()

    def show_transfer_review_prompt(self, reason='') -> None:
        """Keep the photograph and its approval controls in the same workspace."""
        if self._transfer_review_active:
            self.show_mark_layers()
        panel = getattr(self, '_transfer_review_prompt', None)
        if panel is None:
            panel = QFrame(self.run_preview.widget)
            panel.setFrameShape(QFrame.Shape.StyledPanel)
            layout = QVBoxLayout(panel)
            title = self._transfer_review_title = QLabel('Run approval')
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            layout.addWidget(title)
            self._transfer_review_message = QLabel()
            self._transfer_review_message.setWordWrap(True)
            layout.addWidget(self._transfer_review_message)
            for text in ('The sand-layer map outlines are valid in this photograph',
                         'The ColorChecker outline is correct in this photograph'):
                check = QCheckBox(text)
                check.setEnabled(False)
                layout.addWidget(check)
            self._transfer_review_prepare = QPushButton('Prepare transfer review')
            self._transfer_review_prepare.clicked.connect(self._place_version_clicked)
            layout.addWidget(self._transfer_review_prepare)
            self._transfer_review_prompt = panel
            self.run_preview.widget.layout().addWidget(panel)
        self._transfer_review_message.setText(reason or
            'Loading the saved transfer review. If it has not been prepared for this map, '
            'use Prepare transfer review; the approval controls will appear here below the photograph.')
        panel.show()
        self._sync_transfer_preparation()

    def _sync_transfer_preparation(self, *_args):
        panel = getattr(self, '_transfer_review_prompt', None)
        if panel is None or panel.isHidden():
            return
        run = str(self.run_preview.runs.currentData() or '').upper()
        self._transfer_review_title.setText(f'{run} · Run approval' if run else 'Run approval')
        self._transfer_review_prepare.setEnabled(self.place_version_button.isEnabled())

    def show_transfer_review(self, panel) -> None:
        """Use the existing main-photo area for the selected saved transfer."""
        if hasattr(panel, 'set_sidebar_navigation'):
            panel.set_sidebar_navigation(True)
        stack = self.run_preview.stack
        if not self._transfer_review_active:
            self._transfer_previous_page = stack.currentWidget()
            # Hidden drawing pages retain their large canvas minimum. They
            # must not force the review controls below a smaller window.
            self._transfer_size_policies = (stack.sizePolicy(), self.run_preview.widget.sizePolicy())
            stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            self.run_preview.widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
            controls = []
            def collect(layout):
                for index in range(layout.count()):
                    item = layout.itemAt(index)
                    if item.widget() is not None and item.widget() is not stack:
                        control = item.widget()
                        controls.append((control, control.isHidden()))
                        control.hide()
                    elif item.layout() is not None:
                        collect(item.layout())
            collect(self.run_preview.widget.layout())
            self._transfer_hidden_controls = controls
        self._transfer_review_active = True
        self._transfer_review_panel = panel
        if stack.indexOf(panel) < 0:
            stack.addWidget(panel)
        stack.setCurrentWidget(panel)
        panel.show()

    def _keep_transfer_review_visible(self, _index) -> None:
        """A retiring ordinary preview must not replace the active transfer page."""
        if getattr(self, '_restoring_photo_page', False):
            return
        if self._workflow_scope == 'rig_frame':
            self._show_rig_editor()
            return
        panel = getattr(self, '_transfer_review_panel', None)
        if (getattr(self, '_transfer_review_active', False) and panel is not None
                and self.run_preview.stack.currentWidget() is not panel):
            self.run_preview.stack.setCurrentWidget(panel)

    def show_adoption_panel(self, panel) -> None:
        """Keep adoption and its confirmation in the existing photo workspace."""
        if not getattr(panel, '_workspace_return_connected', False):
            was_review = self._transfer_review_active
            previous = self.run_preview.stack.currentWidget()
            panel.set_embedded(self.run_preview.stack)
            def restore(_result):
                if self._transfer_review_panel is not panel:
                    return
                if was_review and self.run_preview.stack.indexOf(previous) >= 0:
                    self._transfer_review_panel = previous
                    self.run_preview.stack.setCurrentWidget(previous)
                    previous.show()
                else:
                    self.show_mark_layers()
            panel.finished.connect(restore)
            panel._workspace_return_connected = True
        self.show_transfer_review(panel)

    def show_mark_layers(self, *_args) -> None:
        """Return to the normal photo tools without clearing per-run reviews."""
        if getattr(self, '_transfer_review_active', False):
            self._transfer_review_active = False
            stack_policy, preview_policy = self._transfer_size_policies
            self.run_preview.stack.setSizePolicy(stack_policy)
            self.run_preview.widget.setSizePolicy(preview_policy)
            del self._transfer_size_policies
            self._restoring_photo_page = True
            try:
                for control, hidden in self._transfer_hidden_controls:
                    control.setVisible(not hidden)
                page = self._transfer_previous_page
                if self.run_preview.stack.indexOf(page) >= 0:
                    self.run_preview.stack.setCurrentWidget(page)
            finally:
                self._restoring_photo_page = False
                del self._transfer_hidden_controls
        if (prompt := getattr(self, '_transfer_review_prompt', None)) is not None:
            prompt.hide()
        # Transfer review can have hidden controls which the rig route had
        # already hidden. Unwind that inner snapshot before the rig snapshot.
        if self._workflow_scope != 'rig_frame':
            self._set_rig_photo_tools(False)


    def _build_steps(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(500)
        content = QWidget()
        scroll.setWidget(content)
        self.steps_scroll = scroll
        layout = QVBoxLayout(content)
        self._steps_layout = layout
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        self.import_map_start = QWidget()
        import_layout = QVBoxLayout(self.import_map_start)
        import_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.import_map_start)
        self.workflow_intro = _note(
                "Outline the layers on the raw baseline photograph of the reference "
                "run, propose the boundaries, adjust and propose again until they "
                "fit; then group the layers into facies and export them."
        )
        layout.addWidget(self.workflow_intro)
        # Native Rig Geometry may attach its original curvature editor here.
        # The standalone painting workspace leaves the host empty.
        self.rig_curvature_host = QWidget()
        QVBoxLayout(self.rig_curvature_host).setContentsMargins(0, 0, 0, 0)
        self.rig_curvature_host.hide()
        layout.addWidget(self.rig_curvature_host)
        frame_step = QWidget()
        frame_layout = QVBoxLayout(frame_step)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.rig_panel)
        measured = QHBoxLayout()
        # Outside the rig panel: the view stays available where the panel is
        # disabled (a corrected workspace).
        self.measured_geometry_button = QPushButton("Measured geometry of all runs…")
        self.measured_geometry_button.setToolTip(
            "Show each run's measured frame model, area factor and area maps from a "
            "geometry index; read only"
        )
        self.measured_geometry_button.clicked.connect(self._measured_geometry_clicked)
        measured.addWidget(self.measured_geometry_button)
        measured.addStretch(1)
        frame_layout.addLayout(measured)
        self.step_frame = StepSection(
            1, "Reference points, injection points and analysis boxes", frame_step, expanded=False
        )
        self.injection_host = QWidget()
        QVBoxLayout(self.injection_host).setContentsMargins(0, 0, 0, 0)
        self.step_injection = StepSection(2, "Injection points", self.injection_host, expanded=False)
        self.analysis_boxes_host = QWidget()
        QVBoxLayout(self.analysis_boxes_host).setContentsMargins(0, 0, 0, 0)
        self.step_analysis_boxes = StepSection(3, "Analysis boxes", self.analysis_boxes_host, expanded=False)
        self.point_review_host = QWidget()
        self.point_review_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        QVBoxLayout(self.point_review_host).setContentsMargins(0, 0, 0, 0)
        self.step_point_review = StepSection(4, "Review reference points on all runs", self.point_review_host, expanded=False)
        self.step_point_review.layout().setStretch(1, 1)
        self.step_reference_review = self.step_point_review
        self.rig_depth_host = QWidget()
        QVBoxLayout(self.rig_depth_host).setContentsMargins(0, 0, 0, 0)
        self.step_depth = StepSection(None, "Depth", self.rig_depth_host, expanded=False)
        for section in (self.step_frame, self.step_injection, self.step_analysis_boxes):
            section.toggle.clicked.connect(lambda checked, selected=section: self._rig_edit_section_clicked(checked, selected))
        self.step_point_review.toggle.clicked.connect(self._point_review_section_clicked)
        self.step_paint = StepSection(2, "Mark layers", self._build_paint_step())
        self.step_facies = StepSection(3, "Facies", self._build_export_step())
        self.step_propose = StepSection(4, "Versions", self._build_propose_step())
        self.step_use_map = StepSection(None, "Use the map in the series", self._build_use_map_step(), expanded=False)
        for section in (
            self.step_frame,
            self.step_injection,
            self.step_analysis_boxes,
            self.step_point_review,
            self.step_depth,
            self.step_paint,
            self.step_facies,
            self.step_propose,
            self.step_use_map,
        ):
            layout.addWidget(section)
        self._steps_bottom_spacer = layout.count()
        layout.addStretch(1)
        return scroll

    def _build_paint_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            _note(
                "Choose a layer and draw it with Draw (L): left click adds a "
                "point, right click or Enter closes the polygon. Rough outlines "
                "are enough; the boundaries are placed within the snap distance. "
                "Draw only adds new polygons and never moves a point "
                "that is already there. With Follow edges, a click near a point "
                "or edge of a shown polygon snaps onto it, and between two such "
                "clicks the new polygon follows that polygon's outline (Tab: the "
                "other way round; Alt+click: a free point), so that neighbouring "
                "layers share their boundary. To change existing polygons, use "
                "Edit (V): click a point to select it and drag to move it, "
                "Ctrl+click to select more points, or drag on empty space to "
                "select the points in a rectangle. A point that several polygons "
                "share has a round handle and moves in all of them; Alt+drag "
                "moves it in the polygon pressed into only. Delete removes the "
                "selected points, the arrow keys move them a little (Shift: "
                "more) and Esc clears the selection. Double-click an edge to add "
                "a point there. Right-"
                "click a polygon to add or remove a point, delete the polygon or "
                "give it to another layer. The brush (B) paints strokes, for "
                "example as corrections. Mark the ColorChecker, tape and fittings "
                "as Excluded. Untick Show to hide a layer in the photograph; this "
                "changes only what is shown, and its points cannot be edited "
                "while it is hidden. Right-click the list to show only one layer, "
                "or all layers again. Background (T) shows another photo of the "
                "same run under the marks, for example one taken before the "
                "indicator was added; the proposal still uses the reference photo."
            )
        )
        self.layer_table = QTableWidget(0, 6)
        self.layer_table.setHorizontalHeaderLabels(
            ["", "Layer", "Polygons", "Strokes", "Show", "Top"]
        )
        show_header = self.layer_table.horizontalHeaderItem(_SHOW_COLUMN)
        if show_header is not None:
            show_header.setToolTip(_SHOW_TIP)
        self.layer_table.horizontalHeaderItem(_TOP_COLUMN).setToolTip(_TOP_TIP)
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.layer_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.layer_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = self.layer_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 28)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            _SHOW_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(_TOP_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        self.layer_table.itemSelectionChanged.connect(self._layer_selected)
        self.layer_table.itemChanged.connect(self._layer_item_changed)
        self.layer_table.cellDoubleClicked.connect(self._layer_double_clicked)
        self.layer_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_table.customContextMenuRequested.connect(self._layer_table_menu)
        self.working_layers_note = _note("Saved version: layer visibility is display only. Choose Current map to draw or edit.")
        self.working_layers_note.hide()
        layout.addWidget(self.working_layers_note)
        layout.addWidget(self.layer_table)
        layout.addLayout(self._visibility_buttons("layer"))
        buttons = QHBoxLayout()
        self.add_layer_button = QPushButton("Add layer")
        self.add_layer_button.clicked.connect(self.add_layer)
        self.remove_layer_button = QPushButton("Remove layer")
        self.remove_layer_button.clicked.connect(lambda: self.remove_layer())
        self.clear_strokes_button = QPushButton("Clear its marks")
        self.clear_strokes_button.setToolTip(
            "Remove the polygons and strokes of the selected layer"
        )
        self.clear_strokes_button.clicked.connect(lambda: self.clear_layer_marks())
        for button in (
            self.add_layer_button,
            self.remove_layer_button,
            self.clear_strokes_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return widget

    def _build_propose_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            _note(
                "A model learns from the marked pixels and labels the rest of the "
                "photograph inside the rig frame. Inside a polygon its layer is "
                "kept; near polygon edges the model places the boundary within the "
                "snap distance, choosing only between layers marked nearby. Where "
                "it is wrong, move or add a polygon point with Edit (V) or paint "
                "a short stroke of the right layer, and propose again. Or correct "
                "the proposal by hand with Correct, Restore, Lasso and Fill: the "
                "corrections are kept on top of it and saved with the map."
            )
        )
        row = QHBoxLayout()
        self.propose_button = QPushButton("Propose boundaries")
        self.propose_button.setToolTip("Shortcut: P")
        self.propose_button.clicked.connect(self.propose)
        row.addWidget(self.propose_button)
        self.propose_busy = QProgressBar()
        self.propose_busy.setRange(0, 0)
        self.propose_busy.setMaximumWidth(110)
        self.propose_busy.setTextVisible(False)
        self.propose_busy.hide()
        row.addWidget(self.propose_busy)
        self.propose_status = QLabel()
        self.propose_status.setWordWrap(True)
        row.addWidget(self.propose_status, 1)
        layout.addLayout(row)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Smoothing"))
        self.smoothing = QComboBox()
        for key, label in (("low", "Low"), ("medium", "Medium"), ("high", "High")):
            self.smoothing.addItem(label, key)
        self.smoothing.setToolTip("How strongly the proposed boundaries are smoothed")
        self.smoothing.currentIndexChanged.connect(self._settings_changed)
        settings.addWidget(self.smoothing)
        settings.addSpacing(10)
        settings.addWidget(QLabel("Merge regions under"))
        self.min_region = QDoubleSpinBox()
        self.min_region.setRange(0.0, 500.0)
        self.min_region.setDecimals(1)
        self.min_region.setSingleStep(0.5)
        self.min_region.setSuffix(" cm²")
        self.min_region.setToolTip(
            "Islands smaller than this join the surrounding layer, unless they "
            "contain painted pixels. Without the rig size, the area is taken "
            "relative to a 1 m² window."
        )
        self.min_region.editingFinished.connect(self._settings_changed)
        settings.addWidget(self.min_region)
        settings.addStretch(1)
        layout.addLayout(settings)
        snap_row = QHBoxLayout()
        snap_row.addWidget(QLabel("Snap distance"))
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0.0, lp.MAX_SNAP_MM)
        self.snap.setDecimals(0)
        self.snap.setSingleStep(1.0)
        self.snap.setSuffix(" mm")
        self.snap.setToolTip(
            "How far polygon edges may lie from the true boundary. Within this "
            "distance the model places the boundary; 0 uses the polygons exactly. "
            "Without the rig size, the photograph is taken as 1 m wide."
        )
        self.snap.editingFinished.connect(self._settings_changed)
        snap_row.addWidget(self.snap)
        snap_row.addSpacing(10)
        snap_row.addWidget(QLabel("Detail"))
        self.detail = QComboBox()
        for key, label in (("normal", "Normal"), ("fine", "Fine")):
            self.detail.addItem(label, key)
        self.detail.setToolTip(
            "The grid the model labels. Fine uses the photograph as shown on screen, "
            "with about twice the pixels per mm of Normal, so thin layers keep their "
            "width; a proposal then takes about twice as long. A layer can also get "
            "its own snap distance: right-click it in the layer list."
        )
        self.detail.currentIndexChanged.connect(self._settings_changed)
        snap_row.addWidget(self.detail)
        snap_row.addStretch(1)
        layout.addLayout(snap_row)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.proposal_method = QComboBox()
        self.proposal_method.addItem("Classifier", lp.PROPOSAL_METHOD)
        self.proposal_method.addItem("Polygons only", lpp.METHOD)
        self.proposal_method.setToolTip(
            "Polygons only draws the exact polygons at photograph resolution. "
            "Strokes, classifier settings and automatic edge fill are ignored. "
            "Unassigned pixels stay magenta until corrected; approval and export require no gaps."
        )
        self.proposal_method.currentIndexChanged.connect(self._proposal_method_changed)
        method_row.addWidget(self.proposal_method)
        method_row.addStretch(1)
        layout.addLayout(method_row)
        shared_row = QHBoxLayout()
        shared_label = QLabel("Classifier may move shared edges")
        shared_label.setToolTip(_MOVE_SHARED_TIP)
        shared_row.addWidget(shared_label)
        self.move_shared_edges = QComboBox()
        self.move_shared_edges.addItem("Yes", True)
        self.move_shared_edges.addItem("No", False)
        self.move_shared_edges.setToolTip(_MOVE_SHARED_TIP)
        self.move_shared_edges.currentIndexChanged.connect(self._settings_changed)
        shared_row.addWidget(self.move_shared_edges)
        shared_row.addStretch(1)
        layout.addLayout(shared_row)
        self.proposal_summary = QLabel()
        self.proposal_summary.setWordWrap(True)
        layout.addWidget(self.proposal_summary)
        layout.addWidget(self._build_versions())
        return widget

    def _build_versions(self) -> QWidget:
        """Saved versions of the map: save one, see them, restore one."""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 6, 0, 0)
        title = QLabel("<b>Versions</b>")
        layout.addWidget(title)
        layout.addWidget(
            _note(
                "Save the map shown under the photograph as a version at any "
                "time: the proposal with its corrections, or the manual map, "
                "which is the polygons exactly as drawn. A saved version is never "
                "changed. Restoring one first saves the current map as a version, "
                "so that it can be restored again. Approve the version that is "
                "the run's map; the approval records its SHA-256, the reviewer "
                "and the time, and changes nothing else."
            )
        )
        buttons = QHBoxLayout()
        self.save_version_button = QPushButton("Save version…")
        self.save_version_button.setToolTip(
            "Save the map shown (the proposal and its corrections, or the manual "
            "map) as a new version"
        )
        self.save_version_button.clicked.connect(self._save_version_clicked)
        buttons.addWidget(self.save_version_button)
        self.restore_version_button = QPushButton("Restore version…")
        self.restore_version_button.setToolTip(
            "Make the selected version the current map (double-click works too)"
        )
        self.restore_version_button.clicked.connect(self._restore_version_clicked)
        buttons.addWidget(self.restore_version_button)
        self.approve_version_button = QPushButton("Approve version…")
        self.approve_version_button.setToolTip(
            "Approve the selected version as this run's map: records its SHA-256, "
            "the reviewer and the time; changes no map, proposal or geometry"
        )
        self.approve_version_button.clicked.connect(self._approve_version_clicked)
        buttons.addWidget(self.approve_version_button)
        buttons.addStretch(1)
        self.versions_table = QTableWidget(0, 6)
        self.versions_table.setHorizontalHeaderLabels(
            ["Name", "Kind", "Settings", "Saved UTC", "SHA-256", "State"]
        )
        self.versions_table.verticalHeader().setVisible(False)
        self.versions_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.versions_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.versions_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        palette = self.versions_table.palette()
        for role in (QPalette.ColorRole.Highlight, QPalette.ColorRole.HighlightedText):
            palette.setColor(QPalette.ColorGroup.Inactive, role,
                             palette.color(QPalette.ColorGroup.Active, role))
        self.versions_table.setPalette(palette)
        header = self.versions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.versions_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.versions_table.itemSelectionChanged.connect(self._version_selected)
        self.versions_table.cellClicked.connect(self._version_clicked)
        self.versions_table.itemDoubleClicked.connect(self._version_double_clicked)
        layout.addWidget(self.versions_table)
        layout.addLayout(buttons)
        self.versions_note = QLabel()
        self.versions_note.setWordWrap(True)
        self.versions_note.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.versions_note)
        self.version_snapshot_note = QLabel("Select a saved version to preview its exact saved map. The working map stays unchanged.")
        self.version_snapshot_note.setWordWrap(True)
        layout.addWidget(self.version_snapshot_note)
        self.facies_version_note = QLabel()
        self.facies_version_note.setWordWrap(True)
        layout.addWidget(self.facies_version_note)
        from .painting_versions_ui import PaintingVersionsActions
        self.version_actions = PaintingVersionsActions(self, layout)
        return widget

    def _build_use_map_step(self) -> QWidget:
        """One compact saved-map selection and the actual per-run review state."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.use_map_version = QComboBox()
        self.use_map_version.setPlaceholderText("Choose a saved version")
        self.use_map_version.setMinimumContentsLength(14)
        self.use_map_version.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.use_map_version.setToolTip("The same saved version selected in Versions; selecting it never restores or approves it.")
        self.use_map_version.currentIndexChanged.connect(self._use_map_version_selected)
        layout.addWidget(self.use_map_version)
        self.map_status = QLabel("Choose a saved version above.")
        self.map_status.setWordWrap(True)
        self.map_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.map_status)
        from .painting_series_use_ui import SeriesUseStatus
        self.series_use_status = SeriesUseStatus(self)
        self.series_use_status.changed.connect(self._refresh_use_map)
        self.series_use_status.label.setParent(widget)
        self.series_use_status.label.hide()
        self.use_map_reasons = {}
        choices = (
            ("place_version_button", "Transfer map to all runs…", self._place_version_clicked,
             "Transfer the approved saved map through each run's measured frame model."),
            ("chart_review_button", "Review the transfer…", lambda: self.adoption_controller.open_chart_review(),
             "Review each run on its own saved photograph and approve its outlines."),
            ("gap_review_button", "Review findings…", self._open_gap_review_clicked,
             "Open the selected transfer's saved gaps and findings before Check and adoption."),
            ("check_gaps_button", "Check again", self._recheck_gap_review_clicked,
             "Run the existing fresh check after resolving the review findings."),
            ("adopt_button", "Adopt as the series' sand map…", self._adopt_clicked,
             "Check and confirm the selected saved map before recording a series revision."),
        )
        for name, text, callback, explanation in choices:
            button = QPushButton(text)
            button.clicked.connect(callback)
            button.setToolTip(explanation)
            setattr(self, name, button)
            reason = QLabel(widget)
            reason.hide()
            self.use_map_reasons[button] = reason
        layout.addWidget(self.place_version_button)
        self.use_map_progress = QLabel()
        self.use_map_progress.setWordWrap(True)
        self.use_map_progress.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.use_map_progress)
        self.use_map_progress_bar = QProgressBar()
        self.use_map_progress_bar.hide()
        layout.addWidget(self.use_map_progress_bar)
        self.use_map_elapsed = QLabel()
        self.use_map_elapsed.hide()
        layout.addWidget(self.use_map_elapsed)
        self._placement_started = None
        self._placement_progress_text = ''
        self._placement_stage = None
        self._placement_timer = QTimer(self)
        self._placement_timer.setInterval(1000)
        self._placement_timer.timeout.connect(self._update_placement_progress)
        self.use_map_runs = QTableWidget(0, 2)
        self.use_map_runs.setHorizontalHeaderLabels(["Run", "Status"])
        self.use_map_runs.verticalHeader().hide()
        self.use_map_runs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.use_map_runs.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.use_map_runs.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.use_map_runs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.use_map_runs.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.use_map_runs.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.use_map_runs.setMinimumHeight(180)
        self.use_map_runs.setMaximumHeight(320)
        self.use_map_runs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        palette = self.use_map_runs.palette()
        for role in (QPalette.ColorRole.Highlight, QPalette.ColorRole.HighlightedText):
            palette.setColor(QPalette.ColorGroup.Inactive, role,
                             palette.color(QPalette.ColorGroup.Active, role))
        self.use_map_runs.setPalette(palette)
        self.use_map_runs.itemSelectionChanged.connect(self._use_map_run_selected)
        layout.addWidget(self.use_map_runs, 1)
        self.use_map_run_note = _note("Run statuses load from the selected map's saved review.")
        layout.addWidget(self.use_map_run_note)
        review_row = QHBoxLayout()
        review_row.addWidget(self.chart_review_button)
        review_row.addWidget(self.gap_review_button)
        review_row.addWidget(self.check_gaps_button)
        layout.addLayout(review_row)
        layout.addWidget(self.adopt_button)
        self.adoption_status = QLabel("Not yet saved as the active series' sand map.")
        self.adoption_status.setAccessibleName("Series sand map adoption status")
        self.adoption_status.setWordWrap(True)
        self.adoption_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.adoption_status)
        self.use_map_compact_reason = _note("")
        layout.addWidget(self.use_map_compact_reason)
        self.recorded_review_button = QPushButton("View recorded decisions…", widget)
        self.recorded_review_button.clicked.connect(self._open_recorded_review_clicked)
        secondary = QHBoxLayout()
        secondary.addWidget(self.recorded_review_button)
        secondary.addStretch(1)
        layout.insertLayout(layout.indexOf(self.adopt_button), secondary)
        self.prepare_adoption_button = self.adopt_button
        self.apply_build_job_label = QLabel("Apply and Build job folder")
        self.apply_build_job_edit = QLineEdit()
        self.apply_build_job_edit.setAccessibleName("Apply and Build job folder")
        self.apply_build_job_edit.setPlaceholderText(r"\\server\share\series\jobs\new-setup-job")
        self.apply_build_job_edit.setToolTip(
            "Fresh shared folder for this Apply and Build job. You can change the suggested location.")
        layout.addWidget(self.apply_build_job_label)
        layout.addWidget(self.apply_build_job_edit)
        self.apply_build_job_label.hide()
        self.apply_build_job_edit.hide()
        self.apply_build_button = QPushButton("Apply to all runs and Build run models")
        self.apply_build_button.setEnabled(False)
        self.apply_build_button.setToolTip("Unavailable until the existing RAW route-2 Apply and Build job is ready.")
        layout.addWidget(self.apply_build_button)
        self.apply_build_reason = _note("Apply and Build are unavailable: the RAW route-2 backend is not ready. Nothing has been applied or built by this tab.")
        layout.addWidget(self.apply_build_reason)
        self.step_export = StepSection(None, "Advanced: older native Setup export", self._build_legacy_export_step(), expanded=False)
        self.step_export.setParent(self)
        self.step_export.hide()
        return widget

    def _build_export_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            _note(
                "A facies groups layers that share role, porosity and permeability. "
                "Tick its layers in the Layers list; each layer belongs to one "
                "facies. Free water and excluded regions get no properties; "
                "everything outside the rig frame is added as a facies of its own."
            )
        )
        self.facies_table = QTableWidget(0, 6)
        self.facies_table.verticalHeader().setVisible(False)
        self.facies_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.facies_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.facies_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        header = self.facies_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.facies_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.facies_table.itemChanged.connect(self._facies_item_changed)
        self.facies_table.itemSelectionChanged.connect(self._facies_selection_changed)
        self.facies_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.facies_table.customContextMenuRequested.connect(self._facies_table_menu)
        self.working_facies_note = _note("Saved version: facies visibility is display only. Choose Current map to edit the groups.")
        self.working_facies_note.hide()
        layout.addWidget(self.working_facies_note)
        layout.addWidget(self.facies_table)
        layout.addLayout(self._visibility_buttons("facies"))
        facies_buttons = QHBoxLayout()
        self.add_facies_button = QPushButton("Add facies")
        self.add_facies_button.setToolTip(
            "Add an empty facies, then tick its layers in the Layers list"
        )
        self.add_facies_button.clicked.connect(lambda: self.add_facies())
        facies_buttons.addWidget(self.add_facies_button)
        self.remove_facies_button = QPushButton("Remove facies")
        self.remove_facies_button.setToolTip(
            "Remove the selected facies; each of its layers gets a facies of its own"
        )
        self.remove_facies_button.clicked.connect(lambda: self.remove_facies())
        facies_buttons.addWidget(self.remove_facies_button)
        facies_buttons.addStretch(1)
        layout.addLayout(facies_buttons)
        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Permeability unit"))
        self.unit = QComboBox()
        for value in lp.PERMEABILITY_UNITS:
            self.unit.addItem(value, value)
        self.unit.currentIndexChanged.connect(self._unit_changed)
        unit_row.addWidget(self.unit)
        unit_row.addStretch(1)
        layout.addLayout(unit_row)
        self.undo_facies_button = QPushButton("Undo last facies change")
        self.undo_facies_button.setToolTip("Restore one role, porosity, permeability or unit edit; the saved approval stays unchanged")
        self.undo_facies_button.clicked.connect(self._undo_facies_change)
        layout.addWidget(self.undo_facies_button)
        self.facies_approval_note = QLabel()
        self.facies_approval_note.setWordWrap(True)
        layout.addWidget(self.facies_approval_note)
        return widget

    def _build_legacy_export_step(self) -> QWidget:
        """Build the optional older Setup export controls below the series route."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_note("Older native Setup export route. The numbered series-map route above uses the selected saved version."))
        self.export_problems = QLabel()
        self.export_problems.setWordWrap(True)
        self.export_problems.linkActivated.connect(self._open_round5_step)
        layout.addWidget(self.export_problems)
        row = QHBoxLayout()
        self.export_button = QPushButton("Save and export to native Setup…")
        self.export_button.setToolTip(
            "Save the proposed layers as a geometry revision on the raw photograph's "
            "grid and write the colored label image, facies table and [labeling] "
            "and [facies] sections"
        )
        self.export_button.clicked.connect(self.save_and_export)
        row.addWidget(self.export_button)
        self.export_busy = QProgressBar()
        self.export_busy.setRange(0, 0)
        self.export_busy.setMaximumWidth(110)
        self.export_busy.setTextVisible(False)
        self.export_busy.hide()
        row.addWidget(self.export_busy)
        row.addStretch(1)
        layout.addLayout(row)
        self.export_status = QLabel()
        self.export_status.setWordWrap(True)
        self.export_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.export_status)
        self.open_export_button = QPushButton("Open older native Setup export")
        self.open_export_button.clicked.connect(self._open_export_folder)
        open_row = QHBoxLayout()
        open_row.addWidget(self.open_export_button)
        open_row.addStretch(1)
        layout.addLayout(open_row)
        return widget

    def _build_canvas_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        self.rig_photo_toolbar = QWidget()
        rig_tools = QHBoxLayout(self.rig_photo_toolbar)
        rig_tools.setContentsMargins(0, 0, 0, 0)
        self.rig_show_points = QCheckBox('Points')
        self.rig_show_lines = QCheckBox('Lines')
        for box in (self.rig_show_points, self.rig_show_lines):
            box.setChecked(True)
            box.toggled.connect(self._rig_overlay_visibility_changed)
            rig_tools.addWidget(box)
        self.rig_show_points.setToolTip('Show reference points, their numbers and injection markers. Display only.')
        self.rig_show_lines.setToolTip('Show the reference quadrilateral lines. Display only.')
        self.rig_enhance = QCheckBox('Enhance contrast')
        rig_tools.addWidget(self.rig_enhance)
        self.rig_fit = QToolButton()
        self.rig_fit.setText('Fit')
        self.rig_fit.setToolTip('Show the whole photograph (F)')
        self.rig_fit.clicked.connect(self.canvas.fit_image)
        rig_tools.addWidget(self.rig_fit)
        guidance = QLabel('Wheel: zoom · Right or middle drag: pan')
        guidance.setWordWrap(True)
        guidance.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        rig_tools.addWidget(guidance, 1)
        layout.addWidget(self.rig_photo_toolbar)
        self.rig_photo_toolbar.hide()
        def group(row):
            widget = QWidget()
            row.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(row)
            return widget
        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_buttons: dict[str, QToolButton] = {}
        for key, label, tip in (
            ("pan", "Pan", "Drag to move the photograph (H)"),
            (
                "polygon",
                "Draw",
                (
                    "Draw a new polygon for the selected layer (L): left click adds "
                    "a point, right click or Enter closes it"
                ),
            ),
            (
                "edit",
                "Edit",
                (
                    "Edit existing polygons (V): click a point to select it and "
                    "drag to move; Ctrl+click selects more points; drag on empty "
                    "space selects points in a rectangle; Delete removes the "
                    "selected points; double-click or right-click an edge to add a "
                    "point"
                ),
            ),
            ("paint", "Paint", "Paint strokes of the selected layer (B)"),
            ("erase", "Erase", "Erase strokes (E); polygons are kept"),
        ):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, name=key: self.set_tool(name))
            self.tool_group.addButton(button)
            self.tool_buttons[key] = button
            tools.addWidget(button)
        self.tool_group.setExclusive(True)
        tools.addSpacing(8)
        self.follow_edges = QCheckBox("Follow edges")
        self.follow_edges.setToolTip(_FOLLOW_EDGES_TIP)
        self.follow_edges.setChecked(self._stored_follow_edges())
        self.canvas.set_follow_edges(self.follow_edges.isChecked())
        self.follow_edges.toggled.connect(self._follow_edges_toggled)
        tools.addWidget(self.follow_edges)
        self.join_edge_button = QToolButton()
        self.join_edge_button.setText("Join edge…")
        self.join_edge_button.setToolTip(
            "Join the selected polygon to a shown neighbour's edge; preview before applying"
        )
        self.join_edge_button.clicked.connect(lambda: self._join_selected_polygon())
        tools.addWidget(self.join_edge_button)
        tools.addSpacing(8)
        tools.addWidget(QLabel("Brush"))
        self.brush_size = QSlider(Qt.Orientation.Horizontal)
        self.brush_size.setRange(0, _BRUSH_STEPS)
        self.brush_size.setFixedWidth(110)
        self.brush_size.setToolTip("Brush size ([ and ])")
        self.brush_size.valueChanged.connect(self._brush_changed)
        tools.addWidget(self.brush_size)
        self.brush_label = QLabel()
        self.brush_label.setMinimumWidth(70)
        tools.addWidget(self.brush_label)
        self.undo_button = QToolButton()
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip(
            "Undo the last change to the polygons or strokes (Ctrl+Z)"
        )
        self.undo_button.clicked.connect(self.undo)
        tools.addWidget(self.undo_button)
        self.redo_button = QToolButton()
        self.redo_button.setText("Redo")
        self.redo_button.setToolTip("Redo (Ctrl+Y)")
        self.redo_button.clicked.connect(self.redo)
        tools.addWidget(self.redo_button)
        tools.addStretch(1)
        fit = self.map_fit = QToolButton()
        fit.setText("Fit")
        fit.setToolTip("Show the whole photograph (F)")
        fit.clicked.connect(self._fit_map_photo)
        tools.addWidget(fit)
        self.paint_tools_widget = group(tools)
        layout.addWidget(self.paint_tools_widget)
        self.correction_tools_widget = group(self._build_correction_tools())
        layout.addWidget(self.correction_tools_widget)
        from .painting_fill_region_tool import FillRegionTool
        self.fill_region_tool = FillRegionTool(self)
        self.fill_region_tool.button.toggled.connect(self._fill_region_tool_toggled)
        self.fill_region_tool.cut_button.toggled.connect(self._fill_region_tool_toggled)
        region_tools = QHBoxLayout()
        for button in (self.fill_region_tool.button, self.fill_region_tool.cut_button,
                       self.fill_region_tool.clear_cut_button):
            region_tools.addWidget(button)
        self.fill_region_tool.note.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        region_tools.addWidget(self.fill_region_tool.note, 1)
        self.fill_tools_widget = group(region_tools)
        layout.addWidget(self.fill_tools_widget)

        display = QHBoxLayout()
        display.addWidget(QLabel("Show"))
        self.show_strokes = QCheckBox("Marks")
        self.show_strokes.setToolTip("Polygons and strokes")
        self.show_strokes.setChecked(True)
        self.show_proposal = QCheckBox("Proposal")
        self.show_proposal.setChecked(True)
        self.show_uncertain = QCheckBox("Uncertain areas")
        self.show_uncertain.setToolTip(
            "Hatch pixels where the model is unsure; paint strokes there first"
        )
        for box in (self.show_strokes, self.show_proposal, self.show_uncertain):
            box.toggled.connect(self._display_changed)
            display.addWidget(box)
        display.addSpacing(8)
        display.addWidget(QLabel("Opacity"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(45)
        self.opacity.setFixedWidth(90)
        self.opacity.setToolTip("Opacity of the proposal")
        self.opacity.valueChanged.connect(self._display_changed)
        display.addWidget(self.opacity)
        display.addSpacing(8)
        self.enhance = QCheckBox("Enhance contrast")
        self.enhance.setChecked(True)
        self.enhance.setToolTip(
            "Stretch the colors of the rig for display; the model always uses "
            "the original photograph"
        )
        self.enhance.toggled.connect(self._enhance_toggled)
        display.addWidget(self.enhance)
        display.addSpacing(8)
        background_label = QLabel("Background")
        background_label.setToolTip(_BACKGROUND_TIP)
        display.addWidget(background_label)
        self.background = QComboBox()
        self.background.setToolTip(_BACKGROUND_TIP)
        self.background.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.background.activated.connect(self._background_activated)
        display.addWidget(self.background)
        self._refresh_background_choice()
        display.addStretch(1)
        # The saturation of the photograph, in a row of its own so that the
        # display row keeps its width.
        photo = QHBoxLayout()
        saturation_label = QLabel("Saturation")
        saturation_label.setToolTip(_SATURATION_TIP)
        photo.addWidget(saturation_label)
        self.saturation = QSlider(Qt.Orientation.Horizontal)
        self.saturation.setRange(0, 100)
        self.saturation.setFixedWidth(120)
        self.saturation.setToolTip(_SATURATION_TIP)
        self.saturation.setValue(self._stored_saturation())
        self.saturation.valueChanged.connect(self._saturation_changed)
        photo.addWidget(self.saturation)
        self.saturation_value = QLabel(f"{self.saturation.value()} %")
        self.saturation_value.setToolTip(_SATURATION_TIP)
        photo.addWidget(self.saturation_value)
        photo.addStretch(1)

        self.photo_note_row = QWidget()
        note_layout = QHBoxLayout(self.photo_note_row)
        note_layout.setContentsMargins(0, 0, 0, 0)
        self.photo_note = QLabel()
        self.photo_note.setWordWrap(True)
        note_layout.addWidget(self.photo_note, 1)
        back = QPushButton("Back to the reference photo")
        self.photo_return_button = back
        back.clicked.connect(self.show_reference_photo)
        note_layout.addWidget(back)
        self.photo_note_row.hide()
        layout.addWidget(self.photo_note_row)
        layout.addWidget(self.canvas, 1)
        self.paint_display_controls = QWidget()
        display_layout = QVBoxLayout(self.paint_display_controls)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.addLayout(display)
        display_layout.addLayout(photo)
        display_layout.addLayout(self._build_map_choice())
        self.facies_legend = QLabel()
        self.facies_legend.setTextFormat(Qt.TextFormat.RichText)
        self.facies_legend.setWordWrap(True)
        self.facies_legend.setToolTip("Facies display colours; the saved layer colours and map stay unchanged")
        self.facies_legend.hide()
        display_layout.addWidget(self.facies_legend)
        self.manual_status = QLabel()
        self.manual_status.setWordWrap(True)
        self.manual_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.manual_status.hide()
        display_layout.addWidget(self.manual_status)
        self.canvas_hint = _note(_CANVAS_HINT)
        display_layout.addWidget(self.canvas_hint)
        layout.addWidget(self.paint_display_controls)
        self._painting_photo_groups = (self.paint_tools_widget, self.correction_tools_widget,
            self.fill_tools_widget, self.paint_display_controls)
        self.rig_enhance.setChecked(self.enhance.isChecked())
        self.rig_enhance.toggled.connect(self.enhance.setChecked)
        self.enhance.toggled.connect(self.rig_enhance.setChecked)
        return column

    def _build_map_choice(self) -> QHBoxLayout:
        """Which map is shown and saved as a version, and the layer of their
        differences."""

        row = QHBoxLayout()
        label = QLabel("Map")
        label.setToolTip(_MAP_SHOWN_TIP)
        row.addWidget(label)
        self.map_shown = QComboBox()
        self.map_shown.addItem("Proposal", "proposal")
        self.map_shown.addItem("Manual map", _MANUAL)
        self.map_shown.setToolTip(_MAP_SHOWN_TIP)
        self.map_shown.currentIndexChanged.connect(self._map_shown_changed)
        row.addWidget(self.map_shown)
        self.show_differences = QCheckBox("Differences")
        self.show_differences.setToolTip(
            "Mark in red where the manual map and the proposal with its corrections "
            "give a pixel different layers, and give their area in cm²"
        )
        self.show_differences.toggled.connect(self._differences_toggled)
        row.addWidget(self.show_differences)
        self.show_proposal_boundaries = QCheckBox("Proposal boundaries")
        self.show_proposal_boundaries.setChecked(True)
        self.show_proposal_boundaries.setToolTip(
            "White edges between regions of the proposal. Display only; "
            "manual maps keep their drawn polygon outlines."
        )
        self.show_proposal_boundaries.toggled.connect(self._display_changed)
        row.addWidget(self.show_proposal_boundaries)
        row.addSpacing(12)
        row.addWidget(QLabel("Colours"))
        self.color_mode = QComboBox()
        self.color_mode.addItem("By layer", "layer")
        self.color_mode.addItem("By facies", "facies")
        self.color_mode.setCurrentIndex(self.color_mode.findData(self._color_mode))
        self.color_mode.setToolTip("By layer: each layer's own colour. By facies: one colour per group. Display only.")
        self.color_mode.currentIndexChanged.connect(self._color_mode_changed)
        self.color_mode.activated.connect(self._color_mode_activated)
        row.addWidget(self.color_mode)
        self.clear_highlight_button = QToolButton()
        self.clear_highlight_button.setText("Clear facies highlight")
        self.clear_highlight_button.clicked.connect(lambda: self.highlight_facies(None))
        self.clear_highlight_button.hide()
        row.addWidget(self.clear_highlight_button)
        row.addStretch(1)
        return row

    def _build_correction_tools(self) -> QHBoxLayout:
        """The tools that correct the proposal by hand, and their state."""

        row = QHBoxLayout()
        title = QLabel("Correct the proposal")
        title.setToolTip(
            "Corrections are drawn on top of the proposal and kept in their own "
            "file; the polygons, strokes and the proposal stay as they are"
        )
        row.addWidget(title)
        for key, label, tip in (
            ("correct", "Correct", "Paint the selected layer onto the proposal (C)"),
            (
                "restore",
                "Restore",
                "Brush the proposal back where corrections were drawn (R)",
            ),
            (
                "lasso",
                "Lasso",
                "Hold the button and go round an area: it takes the selected "
                "layer (A)",
            ),
            (
                "fill",
                "Fill",
                "Click a region of the proposal: it takes the selected layer (G)",
            ),
        ):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, name=key: self.set_tool(name))
            self.tool_group.addButton(button)
            self.tool_buttons[key] = button
            row.addWidget(button)
        self.show_corrections = QCheckBox("Show")
        self.show_corrections.setChecked(True)
        self.show_corrections.setToolTip(
            "Show the corrections on the proposal; they are always saved with the map"
        )
        self.show_corrections.toggled.connect(self._corrections_shown)
        row.addWidget(self.show_corrections)
        self.clear_corrections_button = QToolButton()
        self.clear_corrections_button.setText("Clear")
        self.clear_corrections_button.setToolTip(
            "Remove all corrections of this run (Ctrl+Z brings them back)"
        )
        self.clear_corrections_button.clicked.connect(self._clear_corrections_clicked)
        row.addWidget(self.clear_corrections_button)
        self.corrections_label = QLabel()
        self.corrections_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        row.addWidget(self.corrections_label, 1)
        return row

    def _install_shortcuts(self) -> None:
        """Single keys act on the photograph only, so that typing in a table
        or field is never taken as a shortcut; undo and redo work in the view."""

        def add(
            sequence: str | QKeySequence,
            slot: Callable[[], Any],
            parent: QWidget,
            context: Qt.ShortcutContext,
        ) -> None:
            shortcut = QShortcut(QKeySequence(sequence), parent)
            shortcut.setContext(context)
            shortcut.activated.connect(slot)

        canvas_only = Qt.ShortcutContext.WidgetShortcut
        for key, slot in (
            ("L", lambda: self.set_tool("polygon")),
            ("V", lambda: self.set_tool("edit")),
            ("B", lambda: self.set_tool("paint")),
            ("E", lambda: self.set_tool("erase")),
            ("C", lambda: self.set_tool("correct")),
            ("R", lambda: self.set_tool("restore")),
            ("A", lambda: self.set_tool("lasso")),
            ("G", lambda: self.set_tool("fill")),
            ("H", lambda: self.set_tool("pan")),
            ("F", self._fit_map_photo),
            ("P", self.propose),
            ("T", self.toggle_background),
            ("[", lambda: self._step_brush(-6)),
            ("]", lambda: self._step_brush(6)),
        ):
            add(key, slot, self.canvas, canvas_only)
        for number in range(1, 10):
            add(
                str(number),
                lambda index=number - 1: self.select_layer_row(index),
                self.canvas,
                canvas_only,
            )
        # One shortcut per action with all platform keys, so that no key is
        # registered twice (an ambiguous shortcut never fires).
        for standard, slot, extra in (
            (QKeySequence.StandardKey.Undo, self.undo, ()),
            (QKeySequence.StandardKey.Redo, self.redo, ("Ctrl+Y",)),
        ):
            keys = list(QKeySequence.keyBindings(standard))
            for text in extra:
                if QKeySequence(text) not in keys:
                    keys.append(QKeySequence(text))
            shortcut = QShortcut(self)
            shortcut.setKeys(keys)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # ------------------------------------------------------------- state

    def _session_painted_on_raw_photo(self) -> bool:
        """Whether the run's saved session has marks on its raw reference photo.

        The marks fit the raw photo when the session's recorded image size is
        the raw photo's size. Nothing is written.
        """

        try:
            reference = self._reference()
            if reference is None or reference.baseline_image is None:
                return False
            if not lp.session_path(self.project, reference.run_id).is_file():
                return False
            saved = lp.load_painting_session(self.project, reference)
            raw_size = photo_size(Path(reference.baseline_image),
                                  camera_u=lp.reference_camera_u(self.project, reference))
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return (
            bool(saved.polygons or saved.strokes)
            and saved.image_size is not None
            and tuple(saved.image_size) == tuple(raw_size)
        )

    def reference(self) -> SeriesRunInputs | None:
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            return workspace_frame.workspace_reference(self._workspace_frame_config, self._reference(), check_files=False)
        return self._reference()

    def current_frame(self) -> RigFrame | None:
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            reference = self.reference()
            return workspace_frame.corrected_rig_frame(self._workspace_frame_config, reference.run_id) if reference else None
        frame = self.rig_panel.frame
        if frame is None:
            reference = self.reference()
            if reference is not None:
                frame = saved_rig_frames(self.project).get(
                    reference.run_id
                ) or rig_frame_from_inputs(reference)
        return frame if frame is not None and frame.complete() else None

    def busy(self) -> bool:
        static = getattr(getattr(self, 'rig_panel', None), 'static_references', None)
        if static is not None and static.busy():
            return True
        chooser = getattr(self, "_saved_transfers_dialog", None)
        if chooser is not None and chooser.busy():
            return True
        target = getattr(self, "_target_correction_dialog", None)
        if target is not None and target.busy():
            return True
        recorded = getattr(self, "_recorded_review_dialog", None)
        if recorded is not None and recorded.busy():
            return True
        gaps = getattr(self, "_gap_review_dialog", None)
        if gaps is not None and gaps.busy():
            return True
        fixture = getattr(self, "fixture_preview", None)
        if fixture is not None and fixture.busy():
            return True
        other = getattr(self, "other_frames", None)
        if other is not None and other.busy():
            return True
        measured = getattr(self, "measured_edges", None)
        if measured is not None and measured.busy():
            return True
        boxes = getattr(self, "analysis_boxes", None)
        if boxes is not None and boxes.busy():
            return True
        if self._native_async.busy():
            return True
        for thread in (
            self._proposal_thread,
            self._save_thread,
            self._placement_thread,
            self._adoption_thread,
        ):
            try:
                if thread is not None and thread.isRunning():
                    return True
            except RuntimeError:
                continue
        dialog = self._adoption_dialog
        return bool(dialog is not None and getattr(dialog, "busy", lambda: False)())

    def wait_for_workers(self) -> None:
        """Let a running proposal or save finish (before closing)."""

        previous = self._waiting_for_workers
        self._waiting_for_workers = True
        try:
            point_review = getattr(self,'_point_review_panel',None)
            if point_review is not None:
                point_review.wait_for_workers()
            static = getattr(getattr(self, 'rig_panel', None), 'static_references', None)
            if static is not None:
                static.wait_for_workers()
            self._invalidate_rig_photo()
            self._rig_photo_loader.wait()
            adoption = getattr(self, "adoption_controller", None)
            if adoption is not None:
                adoption.wait_for_workers()
            for controller in (self.analysis_boxes, self.measured_edges, self.other_frames,
                               getattr(self, "fixture_preview", None)):
                if controller is not None:
                    while controller.busy():
                        controller.wait_for_workers()
            chooser = getattr(self, "_saved_transfers_dialog", None)
            if chooser is not None:
                chooser.wait_for_workers()
            status = getattr(self, "series_use_status", None)
            if status is not None:
                status.wait_for_workers()
            self.run_preview.wait()
            self._native_async.wait()
            dialog_thread = getattr(self._adoption_dialog, "_thread", None)
            for thread in (
                self._proposal_thread,
                self._save_thread,
                self._placement_thread,
                self._adoption_thread,
                dialog_thread,
                self._manual_thread,
            ):
                try:
                    if thread is not None and thread.isRunning():
                        thread.wait()
                except RuntimeError:  # already deleted
                    continue
            try:
                self.save_session()
            except RuntimeError:  # the view is being deleted
                pass
        finally:
            self._waiting_for_workers = previous

    def reload(self) -> None:
        """Load the session, photograph and proposal of the reference run."""

        self._invalidate_rig_photo()
        self._reset_fill_region()
        self._analysis_boxes_load_key = None
        self.save_session()
        # Which layers are hidden is a display choice for the session shown.
        self._clear_background()
        self._hidden_layers = set()
        self._hidden_facies = set()
        self._hidden_session = None
        self.canvas.set_hidden_layers(())
        self._raw_session_frame = False
        try:
            self._workspace_frame_config = workspace_frame.current_configuration(self.project)
        except (OSError, ValueError) as error:
            self.session = None
            self.image = None
            self.canvas.clear_photo()
            self.propose_status.setText(f"Could not read the workspace coordinate frame: {error}")
            self.propose_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self._refresh_fill_region()
            return
        if (
            workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
            and self._session_painted_on_raw_photo()
        ):
            # Marks painted on the raw reference photo only fit that photo:
            # this tab then works in the raw frame. The saved corrected maps
            # of the project are left as they are.
            self._raw_session_frame = True
            self._workspace_frame_config = {}
        self.rig_panel.setEnabled(workspace_frame.frame_kind(self._workspace_frame_config) == "raw")
        self._saved_corrected_boundaries = None
        # Manual maps are drawn on the raw reference photo only.
        corrected = workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
        if corrected and self._map_kind() == _MANUAL:
            self.map_shown.blockSignals(True)
            self.map_shown.setCurrentIndex(0)
            self.map_shown.blockSignals(False)
            self._refresh_manual_status()
        self.map_shown.setEnabled(not corrected)
        self.show_proposal.setText(self._overlay_text())
        self._manual_display = None
        self._manual_message = ""
        self._loading = True
        try:
            self.session = None
            self.image = None
            self.proposal = None
            self._undo_stack = []
            self._redo_stack = []
            self.corrections = []
            self._corrections_dirty = False
            self._corrections_error = ""
            self.versions = []
            self.map_approvals, self._map_approvals_error = [], ""
            self.canvas.cancel_polygon()
            self.canvas.clear_selection()
            self._other_photo = None
            self._reference_token = None
            self._detail_plain = None
            self._detail_enhanced = None
            try:
                reference = self.reference()
            except (OSError, ValueError) as error:
                self.canvas.clear_photo()
                self.propose_status.setText(f"Could not read the selected run's corrected maps: {error}")
                self.propose_button.setEnabled(False)
                self.export_button.setEnabled(False)
                return
            self.rig_panel.load()
            if reference is None or reference.baseline_image is None:
                self.canvas.clear_photo()
                self.propose_status.setText(
                    "The reference run has no baseline photograph."
                )
            else:
                try:
                    self.session = lp.load_painting_session(self.project, reference)
                except (OSError, ValueError, KeyError) as error:
                    QMessageBox.warning(
                        self,
                        "Could not read the painted layers",
                        f"{error}\n\nStarting with the default layers.",
                    )
                    self.session = lp.PaintingSession.default(
                        reference.run_id, reference.baseline_image
                    )
                self._load_reference_photo(reference)
                self._restore_background()
                proposal = lp.load_proposal(self.project, reference.run_id)
                if proposal is None and lp.proposal_path(self.project, reference.run_id).is_file():
                    try:
                        proposal = lpp.load_proposal(lp.proposal_path(self.project, reference.run_id))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        self.propose_status.setText(f"The saved proposal could not be read: {error}")
                if (
                    proposal is not None
                    and self.image is not None
                    and tuple(proposal.image_size) == tuple(self.image.image_size)
                ):
                    self.proposal = proposal
                    if lpp.is_polygons_only(proposal):
                        try:
                            self.session = lpp.checked_current_session(
                                proposal, self.session, self.current_frame())
                        except ValueError:
                            # Keep the independently saved draft. Showing a stale
                            # proposal does not adopt its vertices or input image.
                            self.propose_status.setText("The current draft changed; propose the polygons again.")
                self.corrections = self._read_corrections(reference.run_id)
                self.versions = self._read_versions(reference.run_id)
                self._read_approvals(reference.run_id)
            self._sync_settings()
            if self.session is not None and self.session.photo_changed:
                self.propose_status.setText(
                    "The baseline photograph of this run has changed since the "
                    "strokes were painted; check that they still fit."
                )
            elif self._raw_session_frame and self.image is not None:
                self.propose_status.setText(_RAW_SESSION_NOTE)
            self.last_export = None
            self._refresh_layers()
            self._refresh_facies_table()
            self._render_marks()
            self._render_proposal()
            self._refresh_export_info()
            self._refresh_status()
            frame_done = (
                self.current_frame() is not None
                and self.current_frame().has_size()  # type: ignore[union-attr]
            )
            self.step_frame.set_expanded(not frame_done)
            self.set_tool("polygon" if self.session is not None else "pan")
            self._update_manual_map()
            self._refresh_manual_status()
        finally:
            self._loading = False
            self.run_preview.refresh()
            self._version_selected()
            if self._workflow_scope == 'rig_frame':
                self._show_rig_editor()
            self._refresh_fill_region()
            self._sync_analysis_boxes()
            self._measured_edges_context_changed()
            QTimer.singleShot(0,self._restore_reviewed_point_selection)

    def _load_reference_photo(self, reference: SeriesRunInputs) -> None:
        assert reference.baseline_image is not None
        self._photo_wanted = None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
                image = workspace_frame.load_painting_image(self._workspace_frame_config, reference)
                if (self.session is not None and self.session.image_size not in (None, image.image_size)
                        and (self.session.strokes or self.session.polygons)):
                    raise ValueError("Saved marks belong to another coordinate frame; the draft was preserved")
                self._saved_corrected_boundaries = workspace_frame.saved_boundaries(
                    self._workspace_frame_config, reference.run_id, image.display.shape[:2], project=self.project)
            else:
                decoded = self._decoded_photo(Path(reference.baseline_image),
                                              lp.reference_camera_u(self.project, reference))
                if decoded is None:
                    # Decoded in the background; reload() shows it when it is ready.
                    self.image = None
                    self.canvas.clear_photo()
                    self.propose_status.setText("Loading the reference photograph…")
                    return
                image, detail = decoded
        except (OSError, ValueError) as error:
            self.image = None
            self.canvas.clear_photo()
            self.propose_status.setText(
                f"Could not read the baseline photograph: {error}"
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        session = self.session
        assert session is not None
        if session.image_size is None:
            session.image_size = image.image_size
        if not session.image_path:
            session.image_path = str(reference.baseline_image)
        self.image = image
        self._plain = np.rint(255.0 * np.clip(image.display, 0.0, 1.0)).astype(np.uint8)
        self._reference_token = {
            "baseline": self._plain,
            "path": str(image.path),
            "raw_size": image.image_size,
            "reference": True,
        }
        # The full-resolution photo for zooming in; the saved corrected maps
        # are shown at their display size only.
        self._detail_plain = (
            None
            if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
            else detail
        )
        self._update_enhanced()
        self._show_reference(reset_view=True)

    def _display_mask(self) -> np.ndarray | None:
        """Pixels used to stretch the display contrast (inside the rig frame)."""

        if self.image is None:
            return None
        frame = self.current_frame()
        shape = self.image.display.shape[:2]
        scale = (
            shape[0] / self.image.image_size[1],
            shape[1] / self.image.image_size[0],
        )
        if frame is not None:
            return lp.frame_mask(frame, shape, scale)
        # Without a frame, leave out the border where the rig frame usually is.
        mask = np.zeros(shape, dtype=bool)
        rows, cols = int(0.12 * shape[0]), int(0.08 * shape[1])
        mask[rows : shape[0] - rows, cols : shape[1] - cols] = True
        return mask

    def _full_resolution(self, path: Path | str, *, camera_u: Any = None) -> np.ndarray | None:
        """The photograph at up to ``DETAIL_LIMIT_PX`` as uint8 RGB.

        It is shown when zoomed in. None when it would be no larger than the
        display copy, or when it cannot be read. Display only: the marks, the
        proposal and saving never use it.
        """

        if self.image is None:
            return None
        return _detail_photo(path, camera_u, self.image.display.shape)

    def _decoded_photo(self, path: Path, camera_u: Any):
        """The decoded reference photograph and its zoom detail, or None while it decodes.

        A camera RAW takes 5-25 s to decode. A large photograph is therefore
        decoded in the background, and reload() runs again when it is ready.
        The last two decoded photographs are kept, so opening the page again
        or restoring a version shows them at once. A small one decodes at once.
        """
        try:
            info = path.stat()
            key = (str(path), repr(camera_u), info.st_size, info.st_mtime_ns)
        except OSError:
            info = key = None
        self._photo_wanted = key
        kept = self.__dict__.setdefault("_decoded_photos", {})
        if key in kept:
            return kept[key]
        if info is not None and info.st_size >= _BACKGROUND_DECODE_BYTES:
            self._start_photo_decode(key, path, camera_u)
            return None
        decoded = _decode_reference_photo(path, camera_u)
        self._keep_decoded("_decoded_photos", key, decoded)
        return decoded

    def _keep_decoded(self, store: str, key, decoded) -> None:
        """Keep a decoded photograph under ``key`` in ``store``, the last two only."""
        if key is None:
            return
        kept = self.__dict__.setdefault(store, {})
        kept[key] = decoded
        while len(kept) > 2:
            kept.pop(next(iter(kept)))

    def _start_photo_decode(self, key, path: Path, camera_u: Any) -> None:
        from darsia.fluidflower.ui_metadata_preview import MetadataPreview

        reader = self.__dict__.get("_photo_decoder")
        if reader is None:
            reader = self._photo_decoder = MetadataPreview(self)
            reader.completed.connect(self._photo_decoded)
        elif reader.busy and self.__dict__.get("_photo_decoding") == key:
            return
        self._photo_decoding = key
        reader.start(lambda _cancelled: (key, _decode_reference_photo(path, camera_u)))

    def _photo_decoded(self, value, error) -> None:
        """Keep the decoded photograph and show it, or say why it could not be read.

        A photograph no longer wanted, because another run or photograph was
        opened meanwhile, is only kept.
        """
        decoding = self.__dict__.pop("_photo_decoding", None)
        key = decoding if error or value is None else value[0]
        if not error and value is not None:
            self._keep_decoded("_decoded_photos", key, value[1])
        if key is None or key != self.__dict__.get("_photo_wanted"):
            return
        if error or value is None:
            self.image = None
            self.canvas.clear_photo()
            self.propose_status.setText(
                f"Could not read the baseline photograph: {error or 'no answer was returned'}")
            return
        self.reload()

    def _start_background_decode(self, key, path: Path, camera_u: Any, size) -> None:
        from darsia.fluidflower.ui_metadata_preview import MetadataPreview

        reader = self.__dict__.get("_background_decoder")
        if reader is None:
            reader = self._background_decoder = MetadataPreview(self)
            reader.completed.connect(self._background_photo_decoded)
        elif reader.busy and self.__dict__.get("_background_decoding") == key:
            return
        self._background_decoding = key
        shape = self.image.display.shape
        reader.start(lambda _cancelled: (key, _decode_background_photo(path, camera_u, size, shape)))

    def _background_photo_decoded(self, value, error) -> None:
        """Keep a decoded background photograph and install the choice waiting for it.

        A choice made meanwhile, or another run opened meanwhile, is left as
        it is; the photograph is then only kept.
        """
        decoding = self.__dict__.pop("_background_decoding", None)
        key = decoding if error or value is None else value[0]
        if not error and value is not None:
            self._keep_decoded("_decoded_backgrounds", key, value[1])
        pending = self.__dict__.get("_background_pending")
        if pending is None or key is None or pending[0] != key:
            return
        self._background_pending = None
        _key, path, show, quiet, run_id = pending
        reference = self.reference()
        if self.image is None or reference is None or reference.run_id != run_id:
            return
        if self._other_photo is None:
            self.photo_note_row.hide()
        if error or value is None:
            if not quiet:
                QMessageBox.warning(self, "Could not read the photograph", error or "No answer was returned.")
            self._refresh_background_choice()
            return
        self.set_background_photo(path, show=show, quiet=quiet)

    def _start_frame_decode(self, key, path: Path, camera_u: Any) -> None:
        from darsia.fluidflower.ui_metadata_preview import MetadataPreview

        reader = self.__dict__.get("_frame_decoder")
        if reader is None:
            reader = self._frame_decoder = MetadataPreview(self)
            reader.completed.connect(self._frame_photo_decoded)
        elif reader.busy and self.__dict__.get("_frame_decoding") == key:
            return
        self._frame_decoding = key
        reader.start(lambda _cancelled: (key, _decode_frame_photo(path, camera_u)))

    def _frame_photo_decoded(self, value, error) -> None:
        """Keep a decoded photograph and show it if it is still the one asked for.

        The page may have gone back to the reference photo or on to another
        photograph meanwhile; the photograph is then only kept.
        """
        decoding = self.__dict__.pop("_frame_decoding", None)
        key = decoding if error or value is None else value[0]
        if not error and value is not None:
            self._keep_decoded("_decoded_frames", key, value[1])
        if key is None or key != self.__dict__.get("_frame_wanted"):
            return
        path = Path(key[0])
        if self._other_photo is None or Path(self._other_photo) != path:
            return
        if error or value is None:
            self.photo_note.setText(f"Could not show {path.name}: {error or 'no answer was returned'}")
            self.photo_note_row.show()
            return
        self._install_frame_photo(path, value[1])
        # The rig frame panel draws its frame on the photograph it asked for.
        rig_panel = getattr(self, "rig_panel", None)
        if rig_panel is not None and rig_panel._photo_path == path:
            rig_panel._shown_preview = self.canvas.preview
            rig_panel._draw_overlay()

    def _stretched_detail(
        self, detail: np.ndarray | None, display: np.ndarray
    ) -> np.ndarray | None:
        """``detail`` with the same contrast stretch as its display copy."""

        if detail is None:
            return None
        lower, upper = lp.contrast_limits(display, self._display_mask())
        return lp.stretch_contrast_rgb8(detail, lower, upper)

    def _update_enhanced(self) -> None:
        if self.image is None:
            self._enhanced = None
            self._detail_enhanced = None
            return
        self._enhanced = lp.enhance_contrast(self.image.display, self._display_mask())
        self._detail_enhanced = self._stretched_detail(self._detail_plain, self.image.display)
        self._update_background_enhanced()

    def _show_reference(self, *, reset_view: bool) -> None:
        if self.image is None or self._reference_token is None:
            return
        photo_path = self._background_path if self._background_shown and self._background_path is not None else self.image.path
        if self._request_rig_photo(Path(photo_path), reset_view=reset_view):
            return
        if self._other_photo is not None:
            self._reset_fill_region()
        enhanced = self.enhance.isChecked()
        token = self._reference_token
        display = self._enhanced if enhanced else self._plain
        detail = self._detail_enhanced if enhanced else self._detail_plain
        if self._background_shown and self._background_token is not None:
            token = self._background_token
            stretched = enhanced and self._background_enhanced is not None
            display = self._background_enhanced if stretched else self._background_plain
            detail = (
                self._background_detail_enhanced
                if stretched
                else self._background_detail_plain
            )
        if self.canvas.preview is not token:
            self._reset_fill_region()
        assert display is not None
        saturation = self.saturation.value() / 100.0
        display = photo_with_saturation(display, saturation)
        if detail is not None:
            detail = photo_with_saturation(detail, saturation)
        self.canvas.set_photo(
            display,
            self.image.image_size,
            token=token,
            reset_view=reset_view,
            detail=detail,
        )
        if (workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
                or self._workflow_scope in {'rig_frame', 'analysis_boxes'}
                or getattr(self, '_display_only_preview', False)):
            self.canvas.paint_enabled = False
        else:
            self.canvas.paint_enabled = True
        self._other_photo = None
        self.rig_panel.use_host_photo(self.image.image_size, self.canvas.preview)
        self.photo_note_row.hide()
        self._render_marks()
        self._render_proposal()
        self._display_changed()
        self._refresh_fill_region()

    def show_reference_photo(self) -> None:
        self._show_reference(reset_view=False)

    def show_frame_photo(self, path: Path) -> bool:
        """Show a raw photograph for the rig frame panel.

        A large photograph not decoded yet is decoded off the GUI thread: the
        page says it is loading and shows it when it is ready. The last two
        decoded photographs are kept, so the saturation and contrast controls
        show it again at once, at the same zoom.
        """

        if self._request_rig_photo(Path(path), reset_view=True):
            return True
        if self.image is not None and Path(path) == Path(self.image.path):
            if self.canvas.preview is not self._reference_token:
                self._show_reference(reset_view=False)
            return True
        path = Path(path)
        # A RAW of a run with a camera profile is shown on U, the grid of the marks.
        camera_u = self._camera_u_profile() if image_family(path) == "raw" else None
        try:
            info = path.stat()
            key = (str(path), repr(camera_u), info.st_size, info.st_mtime_ns)
        except OSError:
            info = key = None
        self._frame_wanted = key
        kept = self.__dict__.setdefault("_decoded_frames", {})
        if key in kept:
            return self._install_frame_photo(path, kept[key])
        if info is not None and info.st_size >= _BACKGROUND_DECODE_BYTES:
            self._start_frame_decode(key, path, camera_u)
            self._reset_fill_region()
            self.canvas.clear_photo()
            self.canvas.paint_enabled = False
            self._other_photo = path
            self.photo_note.setText(f"Loading {path.name}…")
            self.photo_note_row.show()
            return True
        try:
            decoded = _decode_frame_photo(path, camera_u)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not show the photograph", str(error))
            return False
        self._keep_decoded("_decoded_frames", key, decoded)
        return self._install_frame_photo(path, decoded)

    def _install_frame_photo(self, path: Path, decoded) -> bool:
        """Show a decoded photograph (``_decode_frame_photo``) instead of the reference.

        The view is kept when the same photograph is shown again, for example
        at another saturation.
        """

        rgb, (width, height) = decoded
        shown = self.canvas.preview
        same = isinstance(shown, dict) and shown.get("path") == str(path)
        plain = np.rint(255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        display = lp.enhance_contrast(rgb) if self.enhance.isChecked() else plain
        display = photo_with_saturation(display, self.saturation.value() / 100.0)
        token = {"baseline": plain, "path": str(path), "raw_size": (width, height)}
        self._reset_fill_region()
        self.canvas.set_photo(display, (width, height), token=token, reset_view=not same)
        self.canvas.paint_enabled = False
        self.canvas.set_overlays_visible(False, False)
        self._other_photo = Path(path)
        self.photo_note.setText(
            f"Showing {Path(path).name}. Painting works on the reference photo."
        )
        self.photo_note_row.show()
        self._refresh_fill_region()
        self._sync_analysis_boxes()
        self._refresh_measured_edges()
        return True

    def _invalidate_rig_photo(self) -> None:
        self._rig_photo_epoch += 1
        self._rig_photo_loader.invalidate()
        self._rig_photo_key = self._rig_photo_packet = self._rig_photo_path = None
        self._rig_photo_display_reference = None
        rig_panel = getattr(self, 'rig_panel', None)
        if rig_panel is not None:
            rig_panel.corners.parentWidget().setEnabled(True)
            rig_panel.points.parentWidget().setEnabled(not getattr(self, '_point_review_active', False))
        self._static_reference_context_changed()

    def _request_rig_photo(self, path: Path, *, reset_view: bool, display_reference=None) -> bool:
        if self._workflow_scope not in {'rig_frame', 'layers', 'analysis_boxes'} or image_family(path) != 'raw':
            return False
        selected = self._rig_photo_display_reference
        reference = display_reference or (selected if selected is not None
            and selected.baseline_image is not None and Path(selected.baseline_image) == path
            else self.reference())
        profile = (lp.reference_camera_u(self.project, reference) if display_reference is not None
            or (selected is not None and reference is selected) else self._camera_u_profile())
        if profile is None:
            return False
        self._rig_photo_display_reference = reference
        key = (self._rig_photo_epoch, str(self.project.root), reference.run_id,
            str(path), str(profile.path), profile.sha256, profile.profile_id)
        if key == self._rig_photo_key and self._rig_photo_packet is not None:
            self._apply_rig_photo(self._rig_photo_packet, path, reset_view=reset_view)
            return True
        if key != self._rig_photo_key:
            self._rig_photo_packet = None
        self._rig_photo_key, self._rig_photo_path = key, path
        self._rig_photo_reset = reset_view
        self.rig_panel._set_mode('')
        self.canvas.clear_photo()
        self.rig_panel._draw_overlay()
        self.rig_panel.setEnabled(False)
        self._static_reference_context_changed()
        self.photo_note.setText('Loading the complete lens-corrected RAW photograph…')
        self.photo_note_row.show()
        if self._workflow_scope == 'layers':
            self.run_preview.refresh_full_reference_photo()
        self._rig_photo_loader.request(key, path, profile)
        return True

    @Slot(object, object, str)
    def _rig_photo_ready(self, key, packet, error) -> None:
        reference = self._rig_photo_display_reference
        if (self._workflow_scope not in {'rig_frame', 'layers', 'analysis_boxes'} or key != self._rig_photo_key
                or reference is None or key[1:3] != (str(self.project.root), reference.run_id)):
            return
        if error:
            self.photo_note.setText('Could not load the complete RAW photograph: ' + error)
            self.photo_note_row.show()
            if self._workflow_scope == 'layers' and self.run_preview._reference_version() is not None:
                self.run_preview.canvas.clear_photo()
                self.run_preview.note.setText(self.photo_note.text())
            return
        self._rig_photo_packet = packet
        self._apply_rig_photo(packet, self._rig_photo_path, reset_view=self._rig_photo_reset)

    def _apply_rig_photo(self, packet, path, *, reset_view: bool) -> None:
        rig_editor = self._workflow_scope == 'rig_frame'
        if rig_editor:
            self._show_rig_editor()
        plain = packet['display_plain']
        display = packet['display_enhanced'] if self.enhance.isChecked() else plain
        display = photo_with_saturation(display, self.saturation.value() / 100.)
        full = packet['image_rgb']
        size = tuple(packet['camera_u_wh'])
        reference = self._rig_photo_display_reference or self.reference()
        token = dict(packet, path=str(path), baseline=plain, raw_size=size,
            run_id=reference.run_id if reference is not None else '', full_raw_sensor_field=True)
        self.canvas.set_photo(display, size, token=token, reset_view=reset_view,
            detail=full, display_mapping=packet['mapping'],
            display_size=(full.shape[1], full.shape[0]), valid_support=packet['valid_support'])
        self.canvas.set_overlays_visible(False, False)
        from .painting_run_preview import _same_path
        known_reference = self.image is not None and _same_path(self.image.path, path)
        known_background = (self._background_shown and self._background_path is not None
                            and _same_path(self._background_path, path))
        known_photo = known_reference or known_background
        # The full-field renderer owns the currently displayed reference token.
        # Editing guards must retain that identity after replacing its pixmap.
        if known_reference:
            self._reference_token = token
        elif known_background:
            self._background_token = token
        self._other_photo = None if known_photo else path
        self.canvas.paint_enabled = (not rig_editor and known_photo
            and not getattr(self, '_display_only_preview', False)
            and workspace_frame.frame_kind(self._workspace_frame_config) == 'raw')
        self.rig_panel.setEnabled(workspace_frame.frame_kind(self._workspace_frame_config) == 'raw'
            and not getattr(self,'_point_review_active',False))
        self.rig_panel.use_host_photo(size, token)
        if self._workflow_scope == 'analysis_boxes':
            self._show_analysis_boxes_editor()
            self.photo_note.setText('Complete RAW photograph, lens corrected. Edit measured analysis boxes on this bound photograph.')
            self.photo_note_row.show()
            self._sync_analysis_boxes()
            return
        if not rig_editor:
            self.photo_note.setText('Complete RAW photograph, lens corrected.')
            self.photo_note_row.show()
            self._render_marks()
            self._render_proposal()
            self._display_changed()
            self._refresh_fill_region()
            self.run_preview.refresh_full_reference_photo()
            return
        map_reference = self.reference()
        foreign_run = (map_reference is not None and reference is not None
            and map_reference.run_id != reference.run_id)
        self.rig_panel.corners.parentWidget().setEnabled(not foreign_run)
        self.rig_panel.points.parentWidget().setEnabled(not foreign_run and not getattr(self, '_point_review_active', False))
        if foreign_run:
            self.run_preview.note.setText('Static references · ' + reference.run_id.upper()
                + ' · map reference remains ' + map_reference.run_id.upper())
        if self._other_photo is not None:
            self.rig_panel._photo_path = path
        self.photo_note.setText('Complete RAW photograph, lens corrected. ' + (
            'Inspect and correct static reference areas for ' + reference.run_id.upper() + '.'
            if foreign_run else 'Edit the rig corners and injection points on this photograph.'))
        self.photo_note_row.show()
        self._sync_analysis_boxes()
        self._refresh_measured_edges()
        self._static_reference_context_changed()

    def uses_full_reference_photo(self) -> bool:
        return (self._workflow_scope == 'layers' and self.image is not None
            and image_family(Path(self.image.path)) == 'raw' and self._camera_u_profile() is not None)

    def show_full_reference_preview(self, canvas, run_id, *, reset_view=True) -> bool:
        """Place a saved map on the same full photograph without changing its grid."""
        packet = self._rig_photo_packet
        reference = self.reference()
        if (self._workflow_scope != 'layers' or packet is None or reference is None
                or reference.run_id.casefold() != str(run_id).casefold()
                or self.image is None or tuple(packet['camera_u_wh']) != tuple(self.image.image_size)):
            return False
        full = packet['image_rgb']
        display = packet['display_enhanced'] if self.enhance.isChecked() else packet['display_plain']
        display = photo_with_saturation(display, self.saturation.value() / 100.)
        canvas.set_photo(display, tuple(packet['camera_u_wh']),
            token=dict(packet, run_id=run_id, path=str(self._rig_photo_path)), reset_view=reset_view,
            detail=full, display_mapping=packet['mapping'], display_size=(full.shape[1], full.shape[0]),
            valid_support=packet['valid_support'])
        return True

    # ----------------------------------------------------- background photo

    def background_available(self) -> bool:
        """A background photo needs the raw reference photograph of the run."""

        if self.image is None:
            return False
        try:
            return workspace_frame.frame_kind(self._workspace_frame_config) == "raw"
        except (AttributeError, TypeError, ValueError):
            return False

    @property
    def background_path(self) -> Path | None:
        """The chosen background photo of this run, if any."""

        return self._background_path

    @property
    def background_shown(self) -> bool:
        return bool(self._background_shown and self._background_token is not None)

    def set_background_photo(
        self, path: Path | str | None, *, show: bool = True, quiet: bool = False,
    ) -> bool:
        """Show another photograph of the reference run under the marks.

        The photograph must have the raw size of the reference photo, for
        example one taken earlier in the same run before the indicator was
        added. Only the display changes: marks, strokes and the proposal stay
        on top, painting works as on the reference photo, and the proposal is
        always computed from the reference photo. ``None`` forgets the choice.
        The choice is remembered per user, series folder and run.
        A large photograph not decoded yet is decoded off the GUI thread and
        False is returned; the choice is installed (and shown with ``show``)
        when it is ready, unless another choice or run was opened meanwhile.
        The last two decoded photographs are kept.
        """

        self._background_pending = None
        if path is None:
            self._clear_background()
            self._remember_background(None)
            if self.image is not None and self._other_photo is None:
                self._show_reference(reset_view=False)
            return True
        if not self.background_available():
            if not quiet:
                QMessageBox.information(
                    self,
                    "Background photo",
                    "A background photo can be used with the raw reference "
                    "photograph only.",
                )
            return False
        assert self.image is not None
        path = Path(path)
        expected = (int(self.image.image_size[0]), int(self.image.image_size[1]))
        # A camera RAW of a run with a configured camera profile is shown on the
        # camera grid U (N106). U is also the grid of the camera's JPEG and of a
        # RAW reference shown on U; otherwise both must be the same kind.
        camera_u = self._camera_u_profile() if image_family(path) == "raw" else None
        reference_on_u = getattr(self.image, "camera_u", None) is not None
        if image_family(path) != image_family(self.image.path):
            if camera_u is None and not reference_on_u:
                if not quiet:
                    QMessageBox.warning(
                        self,
                        "Background photo",
                        f"{path.name} and the reference photo {Path(self.image.path).name} "
                        "are not the same kind of file (camera RAW or JPEG/TIFF/PNG). "
                        "A camera's JPEG is cropped and lens-corrected, so the two do "
                        "not line up without a camera profile for this run. Choose a "
                        "photograph of the same kind.",
                    )
                return False
        try:
            width, height = photo_size(path, camera_u=camera_u)
        except (OSError, ValueError) as error:
            if not quiet:
                QMessageBox.warning(self, "Could not read the photograph", str(error))
            return False
        if (int(width), int(height)) != expected:
            if not quiet:
                QMessageBox.warning(
                    self,
                    "Background photo",
                    f"{path.name} is {width} x {height} pixels, the reference "
                    f"photo {expected[0]} x {expected[1]}. Choose a photograph "
                    "of the same run and camera position.",
                )
            return False
        size = (int(width), int(height))
        try:
            info = path.stat()
            key = (str(path), repr(camera_u), self.image.display.shape[:2], info.st_size, info.st_mtime_ns)
        except OSError:
            info = key = None
        kept = self.__dict__.setdefault("_decoded_backgrounds", {})
        if key in kept:
            rgb, detail = kept[key]
        elif info is not None and info.st_size >= _BACKGROUND_DECODE_BYTES:
            # Installed when it is decoded (seconds for a camera RAW).
            reference = self.reference()
            self._background_pending = (key, path, show, quiet,
                                        reference.run_id if reference is not None else None)
            self._start_background_decode(key, path, camera_u, size)
            if show and not quiet and self._other_photo is None:
                self.photo_note.setText(f"Loading the background photo {path.name}…")
                self.photo_note_row.show()
            return False
        else:
            try:
                rgb, detail = _decode_background_photo(path, camera_u, size, self.image.display.shape)
            except (OSError, ValueError) as error:
                if not quiet:
                    QMessageBox.warning(self, "Could not read the photograph", str(error))
                return False
            self._keep_decoded("_decoded_backgrounds", key, (rgb, detail))
        return self._install_background(path, rgb, detail, expected, show)

    def _camera_u_profile(self):
        """The reference run's configured camera-U profile, or None (cached per run)."""

        reference = self.reference()
        root = getattr(self.project, "root", None)
        if reference is None or root is None:
            return None
        key = (str(root), str(reference.run_id))
        if self._camera_u_key != key:
            try:
                self._camera_u = lp.run_camera_u(self.project, reference.run_id)
            except (OSError, ValueError, KeyError, TypeError):
                self._camera_u = None
            self._camera_u_key = key
        return self._camera_u

    def _install_background(self, path, rgb, detail, expected, show) -> bool:
        """Install a loaded background photo (display copy and detail) and show it."""

        self._background_path = path
        self._background_rgb = rgb
        self._background_plain = np.rint(255.0 * rgb).astype(np.uint8)
        self._background_detail_plain = detail
        self._update_background_enhanced()
        self._background_token = {
            "baseline": self._background_plain,
            "path": str(path),
            "raw_size": expected,
            "reference": False,
            "background": True,
        }
        self._background_shown = bool(show)
        self._remember_background(path)
        self._refresh_background_choice()
        if self._other_photo is None:
            self._show_reference(reset_view=False)
        return True

    def show_background(self, shown: bool) -> None:
        """Show the chosen background photo (True) or the reference photo."""

        self._background_shown = bool(shown) and self._background_token is not None
        self._refresh_background_choice()
        if self.image is not None:
            self._show_reference(reset_view=False)

    def toggle_background(self) -> None:
        """Switch between the reference photo and the chosen background (T)."""

        if self._background_token is None:
            self.choose_background_photo()
            return
        self.show_background(not self._background_shown)

    def choose_background_photo(self) -> bool:
        if not self.background_available():
            self._refresh_background_choice()
            return False
        assert self.image is not None
        start = Path(self.image.path).parent
        start = start.parent if start.parent != start else start
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Background photo of the same run",
            str(start),
            photo_dialog_filter(self.image.path),
        )
        if not path:
            self._refresh_background_choice()
            return False
        return self.set_background_photo(Path(path))

    def _update_background_enhanced(self) -> None:
        if self._background_rgb is None or self.image is None:
            self._background_enhanced = None
            self._background_detail_enhanced = None
            return
        self._background_enhanced = lp.enhance_contrast(
            self._background_rgb, self._display_mask()
        )
        self._background_detail_enhanced = self._stretched_detail(
            self._background_detail_plain, self._background_rgb
        )

    def _clear_background(self) -> None:
        self._background_path = None
        self._background_rgb = None
        self._background_plain = None
        self._background_enhanced = None
        self._background_detail_plain = None
        self._background_detail_enhanced = None
        self._background_token = None
        self._background_shown = False
        self._refresh_background_choice()

    def _refresh_background_choice(self) -> None:
        box = getattr(self, "background", None)
        if box is None:
            return
        box.blockSignals(True)
        try:
            box.clear()
            box.addItem("Reference photo", None)
            if self._background_path is not None:
                box.addItem(self._background_path.name, str(self._background_path))
            box.addItem("Choose photo...", _BACKGROUND_CHOOSE)
            box.setCurrentIndex(
                1 if self._background_shown and self._background_path is not None else 0
            )
            box.setEnabled(self.background_available())
        finally:
            box.blockSignals(False)

    def _background_activated(self, index: int) -> None:
        data = self.background.itemData(index)
        if data == _BACKGROUND_CHOOSE:
            self.choose_background_photo()
        else:
            self.show_background(data is not None)

    def _background_settings_key(self) -> str | None:
        reference = self.reference()
        if reference is None:
            return None
        root = str(getattr(self.project, "root", "") or "")
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
        return f"background/{digest}/{reference.run_id}"

    def _remember_background(self, path: Path | None) -> None:
        key = self._background_settings_key()
        if key is None:
            return
        settings = QSettings(*_BACKGROUND_SETTINGS)
        if path is None:
            settings.remove(key)
        else:
            settings.setValue(key, str(path))
        settings.sync()

    def _stored_follow_edges(self) -> bool:
        """The user's last Follow edges choice; on when none is stored."""

        value = QSettings(*_BACKGROUND_SETTINGS).value(_FOLLOW_EDGES_KEY, True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)

    def _stored_saturation(self) -> int:
        """The user's last saturation of the photograph in percent; 100 when
        none is stored."""

        value = QSettings(*_BACKGROUND_SETTINGS).value(_SATURATION_KEY, 100)
        try:
            return min(max(int(float(value)), 0), 100)
        except (TypeError, ValueError):
            return 100

    def _saturation_changed(self, value: int) -> None:
        self.saturation_value.setText(f"{int(value)} %")
        self._saturation_timer.start()

    def apply_saturation(self) -> None:
        """Show the photograph at the slider's saturation, and remember it."""

        self._saturation_timer.stop()
        settings = QSettings(*_BACKGROUND_SETTINGS)
        settings.setValue(_SATURATION_KEY, int(self.saturation.value()))
        settings.sync()
        if self._other_photo is not None:
            self.show_frame_photo(self._other_photo)
        elif self.image is not None:
            self._show_reference(reset_view=False)

    def _follow_edges_toggled(self, checked: bool) -> None:
        self.canvas.set_follow_edges(checked)
        settings = QSettings(*_BACKGROUND_SETTINGS)
        settings.setValue(_FOLLOW_EDGES_KEY, bool(checked))
        settings.sync()
        if self.canvas.drafting():
            self._draft_changed(len(self.canvas.draft_points))

    def _restore_background(self) -> None:
        """Load the background photo last chosen for this run (not shown)."""

        key = self._background_settings_key()
        if key is None or not self.background_available():
            self._refresh_background_choice()
            return
        value = QSettings(*_BACKGROUND_SETTINGS).value(key, "")
        if value and Path(str(value)).is_file():
            self.set_background_photo(Path(str(value)), show=False, quiet=True)
        else:
            self._refresh_background_choice()

    def _sync_settings(self) -> None:
        session = self.session
        widgets = (
            self.smoothing,
            self.min_region,
            self.snap,
            self.detail,
            self.proposal_method,
            self.move_shared_edges,
            self.unit,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if session is not None:
                self.smoothing.setCurrentIndex(
                    max(0, self.smoothing.findData(session.smoothing))
                )
                self.min_region.setValue(float(session.min_region_cm2))
                self.snap.setValue(float(session.snap_mm))
                self.detail.setCurrentIndex(max(0, self.detail.findData(session.detail)))
                self.proposal_method.setCurrentIndex(
                    1 if lpp.is_polygons_only(self.proposal) else 0)
                self.move_shared_edges.setCurrentIndex(
                    0 if session.move_shared_edges else 1
                )
                self.unit.setCurrentIndex(
                    max(0, self.unit.findData(session.permeability_unit))
                )
            self._brush_from_session()
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        for widget in (
            self.propose_button,
            self.add_layer_button,
            self.remove_layer_button,
            self.clear_strokes_button,
            self.smoothing,
            self.min_region,
            self.snap,
            self.detail,
            self.proposal_method,
            self.move_shared_edges,
            self.unit,
            self.add_facies_button,
            self.remove_facies_button,
        ):
            widget.setEnabled(session is not None)
        self._sync_method_controls()

    # ----------------------------------------------------------- saving

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._export_stale = True
        if not self._loading:
            self._save_timer.start()
        self._schedule_manual_map()
        self.changed.emit()

    def save_session(self) -> None:
        self._save_timer.stop()
        self._save_corrections()
        if not self._dirty or self.session is None:
            return
        try:
            save = (lpp.save_current_session if lpp.is_polygons_only(self.proposal)
                    or self.proposal_method.currentData() == lpp.METHOD else lp.save_painting_session)
            save(self.project, self.session, actor=default_actor())
        except OSError as error:
            self.propose_status.setText(f"The strokes could not be saved: {error}")
            return
        self._dirty = False

    def hideEvent(self, event: Any) -> None:
        self._reset_fill_region()
        boxes = getattr(self, "analysis_boxes", None)
        if boxes is not None:
            boxes.set_mode(None)
        self.save_session()
        super().hideEvent(event)

    def closeEvent(self, event: Any) -> None:
        if not self.can_close():
            event.ignore()
            self._canvas_notice("Wait for the current painting or metadata operation before closing.")
            return
        self._reset_fill_region()
        super().closeEvent(event)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        window = self.window()
        if window is not self and window is not self._analysis_boxes_window:
            if self._analysis_boxes_window is not None and isValid(self._analysis_boxes_window):
                self._analysis_boxes_window.removeEventFilter(self)
            self._analysis_boxes_window = window
            window.installEventFilter(self)
        self._sync_analysis_boxes()

    def can_close(self) -> bool:
        boxes = getattr(self, "analysis_boxes", None)
        measured = getattr(self, "measured_edges", None)
        other = getattr(self, "other_frames", None)
        status = getattr(self, "series_use_status", None)
        point_review = getattr(self,'_point_review_panel',None)
        return (not self.busy() and not self.run_preview.busy() and not self._rig_photo_loader.busy()
                and (point_review is None or not point_review.busy())
                and (status is None or not status.busy())
                and (boxes is None or boxes.can_close())
                and (measured is None or measured.can_close())
                and (other is None or other.can_close()))

    def eventFilter(self, watched, event):
        if self._analysis_boxes_closed or not isValid(self):
            return False
        if watched is self._analysis_boxes_window and event.type() == QEvent.Type.Close:
            point_review=getattr(self,'_point_review_panel',None)
            if point_review is not None and point_review.busy():
                event.ignore()
                self._canvas_notice('Wait for the reference-point photograph or geometry read before closing.')
                return True
            static = getattr(self.rig_panel, 'static_references', None)
            if static is not None and not static.can_close():
                event.ignore()
                self._canvas_notice('Wait for the current static-area search before closing.')
                return True
            status = getattr(self, "series_use_status", None)
            if status is not None and status.busy():
                event.ignore()
                status.label.setText("Wait for the current finite status read before closing.")
                return True
            for controller in (self.analysis_boxes, self.measured_edges, self.other_frames,
                               getattr(self, "fixture_preview", None)):
                if controller is not None and not controller.can_close():
                    event.ignore()
                    controller.note.setText("Wait for the current metadata operation before closing.")
                    return True
        return super().eventFilter(watched, event)

    def _invalidate_analysis_boxes(self, *_args):
        # Destruction invalidation performs no Qt access.
        self._analysis_boxes_closed = True
        self._analysis_boxes_preview_items = []

    def _sync_analysis_boxes(self, *_args, force=False):
        boxes = getattr(self, "analysis_boxes", None)
        if (self._waiting_for_workers or self._analysis_boxes_closed or self._analysis_boxes_syncing or boxes is None
                or not boxes._alive()):
            return
        self._analysis_boxes_syncing = True
        try:
            boxes.refresh()
            self._refresh_analysis_box_preview()
            if self._loading or self.session is None or self.image is None:
                return
            reference = self.reference()  # GUI getter: capture before starting the worker.
            if reference is None or reference.run_id.casefold() != self.session.run_id.casefold():
                return
            from .series_geometry import native_series_link
            linked = bool(native_series_link(self.project).get("series_root"))
            if not linked and not force:
                return
            frame = self.current_frame()
            frame_key = None if frame is None else (
                tuple(sorted(frame.corners.items())), frame.width_m, frame.height_m)
            packet_binding = getattr(self.run_preview.last_result, "geometry_binding", None)
            packet_models = packet_binding.get("models") if isinstance(packet_binding, dict) else None
            packet_source = None if not isinstance(packet_binding, dict) else (
                packet_binding.get("geometry_index"),
                packet_models.get(self.session.run_id.casefold()) if isinstance(packet_models, dict) else None)
            key = (boxes._key(), str(reference.config_path), frame_key,
                   json.dumps(self._workspace_frame_config, sort_keys=True, default=str),
                   json.dumps(packet_source, sort_keys=True, default=str))
            if not force and key == self._analysis_boxes_load_key:
                return
            if boxes.busy() or self.busy() or self.run_preview.busy():
                self._analysis_boxes_reload_pending = True
                return
            expected_run, config_path = reference.run_id.casefold(), Path(reference.config_path)
            if linked:
                from .painting_analysis_box_sources import load_legacy_box_source
                def original_source(project, run):
                    if str(run).casefold() != expected_run:
                        raise ValueError("The captured original-box reference run changed")
                    return load_legacy_box_source(project, run, reference_config_path=config_path)
                boxes.legacy_source_provider = original_source
            else:
                boxes.legacy_source_provider = None
            self._analysis_boxes_load_key = key
            self._analysis_boxes_reload_pending = False
            boxes.reload(preserve_draft=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            boxes._operation_error = str(error)
            boxes.refresh()
        finally:
            self._analysis_boxes_syncing = False
            self._refresh_analysis_box_undo()

    def _analysis_boxes_busy_changed(self, *_args):
        if self._analysis_boxes_closed or not isValid(self):
            return
        self._refresh_status()
        self._sync_analysis_boxes(force=self._analysis_boxes_reload_pending)
        self.busy_changed.emit(self.busy())
        if (not self.busy() and not self.run_preview.busy() and not self.run_preview.is_reference()
                and self.run_preview.last_result is None):
            self.run_preview.request()

    def _analysis_boxes_changed(self, *_args):
        if not self._analysis_boxes_closed and isValid(self):
            self._refresh_status()
            self._refresh_analysis_box_preview()

    def _measured_edges_context_changed(self, *_args):
        self._measured_edges_reload_pending = True
        other = getattr(self, "other_frames", None)
        if other is not None:
            other.context_changed()
        self._refresh_measured_edges()

    def _refresh_measured_edges(self, *_args):
        other = getattr(self, "other_frames", None)
        if other is not None:
            other.refresh()
        measured = getattr(self, "measured_edges", None)
        if (measured is None or self._measured_edges_syncing or not measured._alive()):
            return
        self._measured_edges_syncing = True
        try:
            measured.refresh()
            if (self._measured_edges_reload_pending and not self._loading
                    and not self._waiting_for_workers and not self.busy()
                    and not self.run_preview.busy()):
                self._measured_edges_reload_pending = False
                measured.context_changed()
        finally:
            self._measured_edges_syncing = False

    def _measured_edges_busy_changed(self, *_args):
        measured = getattr(self, "measured_edges", None)
        if measured is not None and measured._alive():
            self._refresh_status()
            self.busy_changed.emit(self.busy())

    def _fixture_preview_busy_changed(self, *_args):
        """Publish fixture-preview activity through the host lifecycle guard."""
        fixture = getattr(self, "fixture_preview", None)
        if fixture is not None and fixture._alive():
            self._refresh_status()
            self.busy_changed.emit(self.busy())

    def _other_frames_busy_changed(self, *_args):
        other = getattr(self, "other_frames", None)
        if other is not None and other._alive():
            self._refresh_status()
            self.busy_changed.emit(self.busy())

    def _analysis_boxes_tool_changed(self, *_args):
        if self._analysis_boxes_closed or not isValid(self):
            return
        active = self.analysis_boxes.mode is not None
        self.follow_edges.setEnabled(not active)
        self.join_edge_button.setEnabled(not active)
        if active:
            self.brush_size.setEnabled(False)
            self.brush_label.setEnabled(False)
        self._refresh_analysis_box_undo()

    def _analysis_boxes_owns_undo(self):
        boxes = getattr(self, "analysis_boxes", None)
        if boxes is None or not boxes._alive():
            return False
        focus = QApplication.focusWidget()
        return self._workflow_scope == 'analysis_boxes' or boxes.mode is not None or (focus is not None and (focus is boxes.widget or boxes.widget.isAncestorOf(focus)))

    def _refresh_analysis_box_undo(self, *_args):
        boxes = getattr(self, "analysis_boxes", None)
        if (self._analysis_boxes_closed or not isValid(self) or boxes is None
                or not boxes._alive() or not hasattr(self, "undo_button")):
            return
        owned = self._analysis_boxes_owns_undo()
        self.undo_button.setEnabled(not boxes.busy() and (boxes.undo_button.isEnabled() if owned else bool(self._undo_stack)))
        self.redo_button.setEnabled(not boxes.busy() and (boxes.redo_button.isEnabled() if owned else bool(self._redo_stack)))
        self.undo_button.setToolTip("Undo the last analysis-box edit (Ctrl+Z)" if owned
                                   else "Undo the last change to the polygons or strokes (Ctrl+Z)")
        self.redo_button.setToolTip("Redo the last analysis-box edit (Ctrl+Y)" if owned else "Redo (Ctrl+Y)")

    def _refresh_analysis_box_preview(self):
        boxes = getattr(self, "analysis_boxes", None)
        if self._analysis_boxes_closed or boxes is None or not boxes._alive():
            return
        preview = self.run_preview
        if not isValid(preview.canvas):
            self._analysis_boxes_preview_items = []
            return
        result = preview.last_result
        valid = (boxes.in_analysis_context() and not preview.is_reference()
                 and result is not None and boxes.draft is not None
                 and not boxes._binding_error and self.session is not None
                 and result.run_id == str(preview.runs.currentData() or "").casefold()
                 and result.source_run == self.session.run_id.casefold()
                 and tuple(preview.canvas.raw_size or ()) == (result.photo.shape[1], result.photo.shape[0])
                 and str((preview.canvas.preview or {}).get("path", "")) == result.photo_path)
        model = boxes.models.get(result.run_id) if valid else None
        if model is not None:
            problem = boxes.preview_geometry_problem(getattr(result, "geometry_binding", None),
                source_run=result.source_run, target_run=result.run_id)
            if problem:
                boxes._operation_error = problem
                boxes.note.setText(problem)
                model = None
        transform = None
        if model is not None:
            native_shape = getattr(result, "native_shape_hw", None)
            display_shape = getattr(result, "display_shape_hw", None)
            try:
                transform = np.asarray(getattr(result, "H_native_to_display", None), np.float64)
            except (TypeError, ValueError):
                transform = np.asarray(())
            if (native_shape is None or tuple(native_shape) != tuple(model.shape_hw or ())
                    or display_shape is None or tuple(display_shape) != tuple(result.photo.shape[:2])
                    or transform.shape != (3, 3) or not np.isfinite(transform).all()):
                model = None
            else:
                sy, sx = display_shape[0]/native_shape[0], display_shape[1]/native_shape[1]
                expected = np.asarray(((sx, 0., (sx-1.)/2.), (0., sy, (sy-1.)/2.), (0., 0., 1.)))
                if not np.array_equal(transform, expected):
                    model = None
        configuration = boxes.draft.configuration() if valid and model is not None else None
        key = None if configuration is None else (id(result), id(model), json.dumps(configuration, sort_keys=True))
        if key == self._analysis_boxes_preview_key:
            return
        self._analysis_boxes_preview_key = key
        for item in self._analysis_boxes_preview_items:
            if isValid(item) and item.scene() is not None:
                item.scene().removeItem(item)
        self._analysis_boxes_preview_items = []
        if configuration is None:
            return
        try:
            outlines = boxes.passive_outlines(model, photo_path=result.photo_path,
                shape=result.native_shape_hw, workspace_config=self._workspace_frame_config)
        except (OSError, ValueError, KeyError):
            return  # No projection is inferred from matching dimensions alone.
        for outline in outlines:
            path = QPainterPath()
            for number, point in enumerate(outline["points_rc"]):
                mapped = transform @ np.asarray((point[1], point[0], 1.))
                value = preview.canvas.raw_to_scene(float(mapped[1]), float(mapped[0]))
                path.moveTo(value) if number == 0 else path.lineTo(value)
            path.closeSubpath()
            item = QGraphicsPathItem(path)
            pen = QPen(QColor("#ffd200" if outline["role"] == "box1" else "#00ffff"), 2.)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setZValue(30.)
            item.setToolTip(outline["name"] + " · analysis-box preview, read only")
            preview.canvas.scene().addItem(item)
            self._analysis_boxes_preview_items.append(item)

    # ------------------------------------------------------------- tools

    def _reset_fill_region(self) -> None:
        tool = getattr(self, "fill_region_tool", None)
        if tool is not None and (tool.active or tool.cut_line or tool._draft):
            tool.deactivate()

    def _fill_region_context_changed(self, *_args: Any) -> None:
        self._reset_fill_region()
        self._refresh_fill_region()

    def _fill_region_tool_toggled(self, checked: bool) -> None:
        if checked:
            self.brush_size.setEnabled(False)
            self.brush_label.setEnabled(False)

    def _refresh_fill_region(self, *_args: Any) -> None:
        tool = getattr(self, "fill_region_tool", None)
        if tool is None or not tool._alive():
            return
        problem = tool._problem()
        previous = getattr(self, "_fill_region_problem", "")
        tool.refresh()
        if problem:
            tool.note.setText(problem)
        elif previous or not tool.note.text():
            tool.note.setText("Fill an uncovered region as a polygon; a cut can divide the region first.")
        self._fill_region_problem = problem

    def _active_map_canvas(self):
        preview = getattr(self, "run_preview", None)
        return preview.active_canvas() if preview is not None else self.canvas

    def _fit_map_photo(self) -> None:
        self._active_map_canvas().fit_image()

    def set_tool(self, tool: str) -> None:
        if self._workflow_scope == 'analysis_boxes':
            if tool == 'pan':
                self.canvas.set_tool('pan')
            return
        if getattr(self, "_display_only_preview", False):
            if tool == "pan":
                self._active_map_canvas().set_tool("pan")
                self.tool_buttons["pan"].setChecked(True)
            return
        boxes = getattr(self, "analysis_boxes", None)
        if boxes is not None and boxes.mode is not None:
            boxes.set_mode(None, restore_tool=False)
        if tool in _DRAWING_TOOLS and self._other_photo is not None:
            self.show_reference_photo()
        if tool in _CORRECTION_TOOLS and not self.corrections_available():
            self._canvas_notice(self._corrections_unavailable())
            tool = self.canvas.tool
        self.canvas.set_tool(tool)
        button = self.tool_buttons.get(tool)
        if button is not None:
            button.setChecked(True)
        painting = tool in _BRUSH_TOOLS
        self.brush_size.setEnabled(painting)
        self.brush_label.setEnabled(painting)

    def _brush_limits(self) -> tuple[float, float]:
        width = float(self.canvas.raw_size[0]) if self.canvas.raw_size else 2000.0
        return (
            max(1.0, _BRUSH_MIN_FRACTION * width),
            max(4.0, _BRUSH_MAX_FRACTION * width),
        )

    def _radius_from_slider(self, value: int) -> float:
        low, high = self._brush_limits()
        return low * (high / low) ** (value / _BRUSH_STEPS)

    def _slider_from_radius(self, radius: float) -> int:
        low, high = self._brush_limits()
        radius = min(max(radius, low), high)
        return int(round(_BRUSH_STEPS * math.log(radius / low) / math.log(high / low)))

    def _brush_from_session(self) -> None:
        if self.image is None:
            return
        radius = _BRUSH_DEFAULT_FRACTION * self.image.image_size[0]
        if self.session is not None and self.session.strokes:
            radius = self.session.strokes[-1].radius_px
        self.brush_size.blockSignals(True)
        self.brush_size.setValue(self._slider_from_radius(radius))
        self.brush_size.blockSignals(False)
        self._brush_changed(self.brush_size.value())

    def _brush_changed(self, value: int) -> None:
        radius = self._radius_from_slider(int(value))
        self.canvas.radius_px = radius
        per_metre = lp.frame_pixels_per_metre(self.current_frame())
        if per_metre:
            self.brush_label.setText(f"Ø {2000.0 * radius / per_metre:.0f} mm")
        else:
            self.brush_label.setText(f"Ø {2.0 * radius:.0f} px")
        self.canvas._update_cursor()

    def _step_brush(self, steps: int) -> None:
        self.brush_size.setValue(self.brush_size.value() + steps)

    # ------------------------------------------------------------ layers

    def _layer_colors(self) -> dict[int, QColor]:
        return self.display_layer_styles()[0]

    def _display_session(self):
        """Use the saved map's labels for its display-only controls."""
        preview = getattr(self, "run_preview", None)
        if preview is not None and preview._reference_version() is not None:
            return self._version_snapshot_session
        return self.session

    def display_layer_styles(self, session=None) -> tuple[dict[int, QColor], frozenset[int]]:
        """Colours and effective hidden IDs for a read-only renderer of this map.

        A saved-version renderer can supply its own session. Visibility and
        colour mode stay display choices and never enter that session's data.
        """
        session = self.session if session is None else session
        if session is None:
            return {}, frozenset()
        known = {row.layer_id for row in session.layers}
        hidden = frozenset(self._hidden_layers & known if self._color_mode == "layer" else {
            row.layer_id for row in session.layers if row.facies in self._hidden_facies})
        if self._color_mode == "layer" and self._highlight_facies is None:
            return {row.layer_id: QColor(row.color) for row in session.layers}, hidden
        return {row.layer_id: (
            QColor("#999999") if self._highlight_facies is not None and row.facies != self._highlight_facies
            else _facies_color(row.facies)) for row in session.layers}, hidden

    def _stored_color_mode(self) -> str:
        mode = QSettings(*_BACKGROUND_SETTINGS).value(_COLOR_MODE_KEY, "layer")
        return "facies" if mode == "facies" else "layer"

    def set_color_mode(self, mode: str) -> None:
        """Choose display colours without changing any saved map data."""
        if mode not in {"layer", "facies"}:
            raise ValueError("Choose layer or facies display colours")
        index = self.color_mode.findData(mode)
        if self.color_mode.currentIndex() == index:
            self._color_mode_activated(index)
        else:
            self.color_mode.setCurrentIndex(index)

    def _color_mode_activated(self, index: int) -> None:
        if self._highlight_facies is not None:
            self._color_mode_changed(index)

    def _color_mode_changed(self, _index: int) -> None:
        self._color_mode = str(self.color_mode.currentData())
        self._highlight_facies = None
        QSettings(*_BACKGROUND_SETTINGS).setValue(_COLOR_MODE_KEY, self._color_mode)
        self._refresh_display_colors()

    def _refresh_display_colors(self) -> None:
        self._refresh_layers()
        self._refresh_facies_table()
        self._render_marks()
        self._render_proposal()
        self.display_styles_changed.emit()

    def _refresh_facies_legend(self) -> None:
        if not hasattr(self, "facies_legend"):
            return
        self._sync_facies_visibility()
        session = self._display_session()
        visible = session is not None and (self._color_mode == "facies" or self._highlight_facies is not None)
        self.facies_legend.setVisible(visible)
        self.clear_highlight_button.setVisible(self._highlight_facies is not None)
        if not visible:
            self.facies_legend.clear()
            return
        colors, hidden_ids = self.display_layer_styles(session)
        entries = []
        for name in session.facies_order():
            members = session.layers_in_facies(name)
            if members:
                color = colors[members[0].layer_id].name()
                hidden = " (hidden)" if all(row.layer_id in hidden_ids for row in members) else ""
                entries.append(f'<span style="color:{color}">■</span> {escape(name)}{hidden}')
        prefix = f"Highlight: {escape(self._highlight_facies)} · " if self._highlight_facies is not None else "Facies: "
        self.facies_legend.setText(prefix + " &nbsp; ".join(entries))

    def _visibility_buttons(self, owner: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Layer visibility" if owner == "layer" else "Facies visibility"))
        for text, visible, suffix in (("All on", True, "on"), ("All off", False, "off")):
            button = QToolButton()
            button.setText(text)
            button.setCheckable(True)
            button.setStyleSheet(
                "QToolButton:checked:enabled { background-color: palette(highlight); "
                "color: palette(highlighted-text); border: 1px solid palette(highlight); "
                "border-radius: 3px; font-weight: 600; }"
            )
            target = "layer" if owner == "layer" else "facies"
            colours = "each layer's own colour" if target == "layer" else "one colour per facies group"
            button.setToolTip(
                f"Show all with {colours} (By {target}, display only)"
                if visible else f"Hide all in By {target} (display only)"
            )
            button.clicked.connect(lambda _checked=False, value=visible, kind=target:
                self.set_all_display_visible(kind, value))
            setattr(self, f"{owner}_all_{suffix}_button", button)
            row.addWidget(button)
        row.addStretch(1)
        return row

    def _facies_visibility_state(self, name: str) -> Qt.CheckState:
        session = self._display_session()
        members = session.layers_in_facies(name) if session is not None else []
        return Qt.CheckState.Checked if members and name not in self._hidden_facies else Qt.CheckState.Unchecked

    def _sync_facies_visibility(self) -> None:
        if not hasattr(self, "facies_table"):
            return
        table = self.facies_table
        blocked = table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item is not None:
                    item.setCheckState(self._facies_visibility_state(str(item.data(Qt.ItemDataRole.UserRole) or "")))
        finally:
            table.blockSignals(blocked)
        self._sync_visibility_buttons()
        self._guard_reference_display_controls()

    def _sync_visibility_buttons(self) -> None:
        session = self._display_session()
        known = {row.layer_id for row in session.layers} if session is not None else set()
        hidden = self.display_layer_styles(session)[1] & known
        for owner in ("layer", "facies"):
            for suffix, active in (("on", bool(known) and not hidden),
                                   ("off", bool(known) and hidden == known)):
                button = getattr(self, f"{owner}_all_{suffix}_button", None)
                if button is not None:
                    blocked = button.blockSignals(True)
                    try:
                        button.setEnabled(bool(known))
                        button.setChecked(owner == self._color_mode and active)
                    finally:
                        button.blockSignals(blocked)

    def set_reference_display_mode(self, reference: bool) -> None:
        """Keep the existing display filters usable on a read-only run photo."""
        if self._workflow_scope in {'rig_frame', 'analysis_boxes'}:
            reference = True
        self._sync_visibility_buttons()
        read_only = not reference
        if read_only == getattr(self, "_display_only_preview", False):
            return
        self._display_only_preview = read_only
        if read_only:
            self.canvas.paint_enabled = False
            self._preview_disabled_controls = {}
            self._preview_table_modes = {
                table: (table.editTriggers(), table.contextMenuPolicy())
                for table in (self.layer_table, self.facies_table)
            }
            self._guard_reference_display_controls()
        else:
            for widget, enabled in self._preview_disabled_controls.items():
                if isValid(widget):
                    widget.setEnabled(enabled)
            for table, (triggers, policy) in self._preview_table_modes.items():
                table.setEditTriggers(triggers)
                table.setContextMenuPolicy(policy)
            self._preview_disabled_controls = {}
            self._preview_table_modes = {}
            self.show_proposal.setText(self._overlay_text())
            self.canvas_hint.show()
            self._refresh_manual_status()
            self._refresh_layers()
            self._refresh_facies_table()
            self._refresh_status()

    def _guard_reference_display_controls(self) -> None:
        if not getattr(self, "_display_only_preview", False):
            return
        # Shared photo tools remain visible outside the canvas stack. Only
        # navigation and display changes apply to an immutable saved map.
        editing = [button for name, button in self.tool_buttons.items() if name != "pan"]
        editing += [self.follow_edges, self.join_edge_button, self.brush_size,
            self.brush_label, self.undo_button, self.redo_button,
            self.correction_tools_widget, self.fill_tools_widget,
            self.show_strokes, self.show_uncertain, self.background,
            self.map_shown, self.show_differences, self.show_proposal_boundaries]
        for widget in editing:
            self._preview_disabled_controls.setdefault(widget, widget.isEnabled())
            widget.setEnabled(False)
        self.tool_buttons["pan"].setChecked(True)
        self.show_proposal.setText("Saved map" if self.run_preview._reference_version() is not None else "Map")
        self.manual_status.hide()
        self.canvas_hint.hide()
        tables = (self.layer_table, self.facies_table)
        safe = (*tables, self.layer_all_on_button, self.layer_all_off_button,
                self.facies_all_on_button, self.facies_all_off_button,
                self.step_paint.toggle, self.step_facies.toggle)
        if self._workflow_scope == 'apply':
            # A read-only photograph does not own the separate transfer route.
            # In particular, another run's preview must not disable its start.
            self._preview_disabled_controls.pop(self.step_use_map, None)
            self.step_use_map.setEnabled(True)
            safe += (self.step_use_map, *self.step_use_map.findChildren(QWidget))
        chooser = getattr(self, "run_preview", None)
        if chooser is not None and chooser._reference_version() is not None:
            tables += (self.versions_table,)
            safe += (self.versions_table,
                     self.restore_version_button, self.approve_version_button, self.step_propose.toggle,
                     self.step_use_map, *self.step_use_map.findChildren(QWidget))
        ancestors = set()
        root = self.steps_scroll.widget()
        for widget in safe:
            while widget is not None:
                ancestors.add(widget)
                if widget is root:
                    break
                widget = widget.parentWidget()
        for widget in root.findChildren(QWidget):
            if widget in ancestors:
                if widget in self._preview_disabled_controls:
                    widget.setEnabled(self._preview_disabled_controls[widget])
                continue
            if any(table.isAncestorOf(widget) for table in tables):
                continue
            if widget.parentWidget() in ancestors:
                self._preview_disabled_controls.setdefault(widget, widget.isEnabled())
                widget.setEnabled(False)
        for table, show_column in ((self.layer_table, _SHOW_COLUMN), (self.facies_table, 0)):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            blocked = table.blockSignals(True)
            try:
                for row in range(table.rowCount()):
                    for column in range(table.columnCount()):
                        item = table.item(row, column)
                        if item is not None:
                            flags = item.flags() & ~Qt.ItemFlag.ItemIsEditable
                            if column != show_column:
                                flags &= ~Qt.ItemFlag.ItemIsUserCheckable
                            item.setFlags(flags)
                        widget = table.cellWidget(row, column)
                        if widget is not None:
                            widget.setEnabled(False)
            finally:
                table.blockSignals(blocked)

    def show_only_facies(self, facies: str) -> None:
        session = self._display_session()
        if session is None or not session.layers_in_facies(facies):
            return
        self._hidden_facies = set(session.facies_order()) - {facies}
        self.set_color_mode("facies")
        self._set_hidden_layers(self._hidden_layers)

    def highlight_facies(self, facies: str | None) -> None:
        """Highlight a facies while retaining every existing hidden-layer choice."""
        session = self._display_session()
        if facies is not None and (session is None or not session.layers_in_facies(facies)):
            return
        if facies != self._highlight_facies:
            self._highlight_facies = facies
            self._refresh_display_colors()

    def _refresh_layers(self) -> None:
        self._sync_layer_priority()
        table = self.layer_table
        self._prune_hidden_layers()
        table.blockSignals(True)
        try:
            current = self.canvas.layer_id
            session = self._display_session()
            layers = session.layers if session is not None else []
            strokes = session.stroke_counts() if session is not None else {}
            polygons = session.polygon_counts() if session is not None else {}
            table.setRowCount(len(layers))
            selected_row = -1
            for row, layer in enumerate(layers):
                swatch = QTableWidgetItem()
                swatch.setBackground(QColor(layer.color))
                # Not selectable, so that the selection does not hide the color.
                swatch.setFlags(Qt.ItemFlag.ItemIsEnabled)
                swatch.setToolTip("This layer's colour. Double-click to change the color")
                swatch.setData(Qt.ItemDataRole.UserRole, layer.layer_id)
                table.setItem(row, 0, swatch)
                name = QTableWidgetItem(layer.name)
                name.setToolTip(
                    (f"Key {row + 1}. " if row < 9 else "")
                    + f"Facies: {layer.facies}. Double-click to rename."
                )
                table.setItem(row, 1, name)
                for column, counts in ((2, polygons), (3, strokes)):
                    text = str(counts.get(layer.layer_id, 0))
                    if column == 2 and layer.snap_mm is not None:
                        # The layer's own snap distance, shown with its polygons.
                        text += f" · {layer.snap_mm:g} mm"
                    count = QTableWidgetItem(text)
                    count.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if column == 2 and layer.snap_mm is not None:
                        count.setToolTip(
                            f"This layer's own snap distance: {layer.snap_mm:g} mm. "
                            "Right-click to change it."
                        )
                    table.setItem(row, column, count)
                shown = QTableWidgetItem()
                shown.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                shown.setCheckState(
                    Qt.CheckState.Unchecked
                    if layer.layer_id in self._hidden_layers
                    else Qt.CheckState.Checked
                )
                shown.setToolTip(_SHOW_TIP)
                table.setItem(row, _SHOW_COLUMN, shown)
                top = QTableWidgetItem()
                top.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                             | Qt.ItemFlag.ItemIsUserCheckable)
                top.setCheckState(Qt.CheckState.Checked if layer.always_on_top else Qt.CheckState.Unchecked)
                reason = self._manual_unavailable()
                if reason:
                    top.setFlags(Qt.ItemFlag.ItemIsSelectable)
                top.setToolTip(reason or _TOP_TIP)
                table.setItem(row, _TOP_COLUMN, top)
                if layer.layer_id == current:
                    selected_row = row
            if selected_row < 0 and layers:
                selected_row = 0
            if selected_row >= 0:
                table.selectRow(selected_row)
                if not getattr(self, "_display_only_preview", False):
                    self._use_layer(layers[selected_row].layer_id)
        finally:
            table.blockSignals(False)
        _fit_table_height(table)
        self._refresh_facies_legend()

    def _use_layer(self, layer_id: int) -> None:
        layer = self.session.layer(layer_id) if self.session is not None else None
        if layer is None:
            return
        self.canvas.set_layer(layer.layer_id, self._layer_colors()[layer.layer_id])

    def selected_layer_id(self) -> int | None:
        rows = {index.row() for index in self.layer_table.selectedIndexes()}
        session = self._display_session()
        if not rows or session is None:
            return None
        row = min(rows)
        if row >= len(session.layers):
            return None
        return session.layers[row].layer_id

    def select_layer_row(self, row: int) -> None:
        if self.session is None or not 0 <= row < len(self.session.layers):
            return
        self.layer_table.selectRow(row)

    def _layer_selected(self) -> None:
        if getattr(self, "_display_only_preview", False):
            return
        layer_id = self.selected_layer_id()
        if layer_id is None:
            return
        self._use_layer(layer_id)
        if self.canvas.tool == "pan" and not self._loading:
            self.set_tool("polygon")

    def _layer_item_changed(self, item: QTableWidgetItem) -> None:
        session = self._display_session()
        if session is None:
            return
        if getattr(self, "_display_only_preview", False) and item.column() != _SHOW_COLUMN:
            return
        if item.column() == _TOP_COLUMN:
            if item.row() < len(self.session.layers):
                self.set_layer_always_on_top(self.session.layers[item.row()].layer_id,
                    item.checkState() == Qt.CheckState.Checked)
            return
        if item.column() == _SHOW_COLUMN:
            if item.row() < len(session.layers):
                self.set_layer_visible(
                    session.layers[item.row()].layer_id,
                    item.checkState() == Qt.CheckState.Checked,
                )
            return
        if item.column() != 1:
            return
        layer = (
            self.session.layers[item.row()]
            if item.row() < len(self.session.layers)
            else None
        )
        if layer is None:
            return
        try:
            self.session.rename_layer(layer.layer_id, item.text())
        except ValueError as error:
            QMessageBox.information(self, "Layer name in use", str(error))
        self._refresh_layers()
        self._refresh_facies_table()
        self._render_proposal()
        self._mark_dirty()
        self._refresh_status()

    def _layer_double_clicked(self, row: int, column: int) -> None:
        if getattr(self, "_display_only_preview", False):
            return
        if column != 0 or self.session is None or row >= len(self.session.layers):
            return
        layer = self.session.layers[row]
        color = QColorDialog.getColor(
            QColor(layer.color), self, f"Color of {layer.name}"
        )
        if color.isValid():
            self.set_layer_color(layer.layer_id, color.name())

    def set_layer_color(self, layer_id: int, color: str) -> None:
        layer = self.session.layer(layer_id) if self.session is not None else None
        if layer is None:
            return
        layer.color = str(color)
        self._refresh_layers()
        self._refresh_facies_table()
        self._render_marks()
        self._render_proposal()
        self._mark_dirty()

    def add_layer(self) -> None:
        if self.session is None:
            return
        try:
            layer = self.session.add_layer()
        except ValueError as error:
            QMessageBox.information(self, "Cannot add a layer", str(error))
            return
        self.canvas.set_layer(layer.layer_id, self._layer_colors()[layer.layer_id])
        self._refresh_layers()
        self._refresh_facies_table()
        self._mark_dirty()
        self._refresh_status()
        row = len(self.session.layers) - 1
        self.layer_table.editItem(self.layer_table.item(row, 1))

    def _mark_text(self, layer_id: int) -> str:
        assert self.session is not None
        polygons = self.session.polygon_counts().get(layer_id, 0)
        strokes = self.session.stroke_counts().get(layer_id, 0)
        parts = []
        if polygons:
            parts.append(f"{polygons} polygon{'s' if polygons != 1 else ''}")
        if strokes:
            parts.append(f"{strokes} stroke{'s' if strokes != 1 else ''}")
        return " and ".join(parts)

    def remove_layer(self, *, confirm: bool = True) -> None:
        layer_id = self.selected_layer_id()
        if self.session is None or layer_id is None:
            return
        layer = self.session.layer(layer_id)
        assert layer is not None
        marks = self._mark_text(layer_id)
        if confirm:
            question = (
                f"Remove {layer.name} and its {marks}?"
                if marks
                else f"Remove {layer.name}?"
            )
            answer = QMessageBox.question(
                self, "Remove layer", question + " This cannot be undone."
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.canvas.cancel_polygon()
        self.canvas.clear_selection()
        self.session.remove_layer(layer_id)
        if layer_id in self._hidden_layers:
            # A later layer may get the same ID; it starts shown.
            self._hidden_layers.discard(layer_id)
            self.canvas.set_hidden_layers(self.hidden_layers)
        # Earlier snapshots may refer to the removed layer.
        self._undo_stack = []
        self._redo_stack = []
        if self.session.layers:
            first = self.session.layers[0]
            self.canvas.set_layer(first.layer_id, QColor(first.color))
        self._after_mark_change()
        self._refresh_facies_table()

    def clear_layer_marks(self, *, confirm: bool = True) -> None:
        layer_id = self.selected_layer_id()
        if self.session is None or layer_id is None:
            return
        layer = self.session.layer(layer_id)
        marks = self._mark_text(layer_id)
        if not marks or layer is None:
            return
        if confirm:
            answer = QMessageBox.question(
                self, "Clear marks", f"Remove the {marks} of {layer.name}?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        # Removing polygons renumbers the others.
        self.canvas.clear_selection()
        self._push_undo()
        self.session.clear_layer_marks(layer_id)
        self._after_mark_change()

    def clear_layer_strokes(self, *, confirm: bool = True) -> None:
        """Former name of :meth:`clear_layer_marks`."""

        self.clear_layer_marks(confirm=confirm)

    # --------------------------------------------------- layer visibility

    @property
    def hidden_layers(self) -> frozenset[int]:
        """Layers hidden in the photograph (display only, never saved)."""

        return self.display_layer_styles()[1]

    def layer_visible(self, layer_id: int) -> bool:
        return int(layer_id) not in self.hidden_layers

    def _sync_layer_priority(self) -> None:
        marked = layer_priority.layer_ids(self.session) if self.session is not None else ()
        box = getattr(self, "map_shown", None)
        if box is None:
            return
        if marked and workspace_frame.frame_kind(self._workspace_frame_config) != "rig-corrected":
            box.blockSignals(True)
            box.setCurrentIndex(box.findData(_MANUAL))
            box.blockSignals(False)
            self.show_proposal.setText(self._overlay_text())
        # Proposal arrays retain their own scientific identity. Priority uses
        # the explicitly selected manual map rather than decorating a proposal.
        proposal_item = box.model().item(box.findData("proposal"))
        if proposal_item is not None:
            proposal_item.setEnabled(not marked)
        box.setToolTip(_TOP_TIP if marked else _MAP_SHOWN_TIP)

    def set_layer_always_on_top(self, layer_id: int, enabled: bool) -> None:
        """Set persistent manual-map priority in one undo step; keep all points."""
        if self.session is None or not self.run_preview.is_reference():
            return
        layer = self.session.layer(layer_id)
        if layer is None or layer.always_on_top == bool(enabled):
            return
        if enabled and self._manual_unavailable():
            self.canvas_hint.setText(self._manual_unavailable())
            self._refresh_layers()
            return
        boxes = getattr(self, "analysis_boxes", None)
        if boxes is not None:
            if boxes.busy():
                self.canvas_hint.setText("Wait for the analysis-box operation before changing layer priority.")
                self._refresh_layers()
                return
            boxes.set_mode(None)
        self.layer_table.setFocus()
        self._push_undo()
        layer.always_on_top = bool(enabled)
        self._manual_display = None
        self._after_mark_change()
        self._render_proposal()
        self._schedule_manual_map()
        self.run_preview.refresh()

    def set_layer_visible(self, layer_id: int, visible: bool) -> None:
        """Show or hide a layer in the photograph.

        Only the display changes: its polygons, strokes and proposed pixels.
        The session, the proposal and what is saved or exported stay the same.
        """

        hidden = set(self._hidden_layers)
        if visible:
            hidden.discard(int(layer_id))
        else:
            hidden.add(int(layer_id))
        self.set_color_mode("layer")
        self._set_hidden_layers(hidden)

    def show_only_layer(self, layer_id: int) -> None:
        session = self._display_session()
        if session is None or session.layer(layer_id) is None:
            return
        self.set_color_mode("layer")
        self._set_hidden_layers(
            row.layer_id for row in session.layers if row.layer_id != layer_id
        )

    def show_all_layers(self) -> None:
        self.set_color_mode("layer")
        self._set_hidden_layers(())

    def hide_all_layers(self) -> None:
        session = self._display_session()
        if session is None:
            self.show_all_layers()
            return
        self.set_color_mode("layer")
        self._set_hidden_layers(row.layer_id for row in session.layers)

    def set_all_display_visible(self, kind: str, visible: bool) -> None:
        """Choose one colour mode and change only that mode's Show choices."""
        session = self._display_session()
        if session is None:
            return
        if kind not in {"layer", "facies"}:
            raise ValueError("Choose layer or facies visibility")
        if kind == "facies":
            self._hidden_facies = set() if visible else set(session.facies_order())
            self.set_color_mode("facies")
            self._set_hidden_layers(self._hidden_layers)
        elif visible:
            self.show_all_layers()
        else:
            self.hide_all_layers()

    def set_facies_visible(self, facies: str, visible: bool) -> None:
        """Use grouped facies colours and preserve the layer mode's Show choices."""
        session = self._display_session()
        if session is None:
            return
        if visible:
            self._hidden_facies.discard(facies)
        elif session.layers_in_facies(facies):
            self._hidden_facies.add(facies)
        self.set_color_mode("facies")
        self._set_hidden_layers(self._hidden_layers)

    def _set_hidden_layers(self, layer_ids: Iterable[int]) -> None:
        session = self._display_session()
        known = (
            {row.layer_id for row in session.layers}
            if session is not None
            else set()
        )
        hidden = {int(value) for value in layer_ids} & known
        if hidden != self._hidden_layers:
            self._hidden_layers = hidden
        effective = self.display_layer_styles(session)[1]
        if session is self.session and effective != self.canvas.hidden_layers:
            self.canvas.set_hidden_layers(effective)
            self._render_proposal()
        self._sync_show_column()
        self._refresh_facies_legend()
        self.display_styles_changed.emit()

    def _prune_hidden_layers(self) -> None:
        """Hidden layers belong to the session shown: another session (another
        run or an imported map) shows every layer; removed layers are dropped.
        Another session also starts with no point selected."""

        session = self._display_session()
        if session is not self._hidden_session:
            self._hidden_session = session
            self._highlight_facies = None
            self._hidden_facies = set()
            hidden: set[int] = set()
            if session is self.session:
                self.canvas.clear_selection()
        elif session is None:
            hidden = set()
            self._hidden_facies = set()
        else:
            hidden = self._hidden_layers & {row.layer_id for row in session.layers}
            self._hidden_facies.intersection_update(session.facies_order())
        if (self._highlight_facies is not None
                and (session is None or not session.layers_in_facies(self._highlight_facies))):
            self._highlight_facies = None
        if hidden != self._hidden_layers:
            self._hidden_layers = hidden
        if session is self.session:
            self.canvas.set_hidden_layers(self.hidden_layers)

    def _sync_show_column(self) -> None:
        """Tick the Show box of every shown layer."""

        session = self._display_session()
        if session is None:
            return
        table = self.layer_table
        table.blockSignals(True)
        try:
            for row, layer in enumerate(session.layers):
                item = table.item(row, _SHOW_COLUMN)
                state = (
                    Qt.CheckState.Unchecked
                    if layer.layer_id in self._hidden_layers
                    else Qt.CheckState.Checked
                )
                if item is not None and item.checkState() != state:
                    item.setCheckState(state)
                if item is not None:
                    item.setToolTip(_SHOW_TIP)
        finally:
            table.blockSignals(False)

    def _reveal_layer(self, layer_id: int) -> None:
        """A layer that gets a new mark is shown, so that the mark can be seen."""

        if int(layer_id) not in self.hidden_layers or self.session is None:
            return
        layer = self.session.layer(layer_id)
        if layer is not None and self._color_mode == "facies":
            self.set_facies_visible(layer.facies, True)
        else:
            self.set_layer_visible(layer_id, True)
        if layer is not None:
            self._canvas_notice(f"{layer.name} is shown again for its new mark.")

    def _layer_table_menu(self, position: QPoint) -> None:
        """Right-click menu of the layer list: which layers are shown."""

        if self.session is None:
            return
        row = self.layer_table.indexAt(position).row()
        layer = (
            self.session.layers[row] if 0 <= row < len(self.session.layers) else None
        )
        menu = QMenu(self)
        only = menu.addAction("Show only this layer")
        only.setEnabled(layer is not None)
        if layer is not None:
            only.triggered.connect(
                lambda _checked=False, layer_id=layer.layer_id: self.show_only_layer(
                    layer_id
                )
            )
        self._add_facies_display_actions(menu, layer.facies if layer is not None else None)
        every = menu.addAction("Show all layers")
        every.setEnabled(bool(self._hidden_layers) or self._color_mode != "layer")
        every.triggered.connect(lambda _checked=False: self.show_all_layers())
        menu.addSeparator()
        own = menu.addAction("Snap distance of this layer…")
        own.setEnabled(layer is not None)
        common = menu.addAction("Use the common snap distance")
        common.setEnabled(layer is not None and layer.snap_mm is not None)
        if layer is not None:
            own.triggered.connect(
                lambda _checked=False, layer_id=layer.layer_id: self._ask_layer_snap(layer_id)
            )
            common.triggered.connect(
                lambda _checked=False, layer_id=layer.layer_id: self.set_layer_snap(
                    layer_id, None
                )
            )
        self._layer_table_menu_widget = menu
        menu.popup(self.layer_table.viewport().mapToGlobal(position))

    def _add_facies_display_actions(self, menu: QMenu, facies: str | None) -> None:
        members = self.session.layers_in_facies(facies) if self.session is not None and facies is not None else []
        only = menu.addAction("Show only this facies")
        only.setEnabled(bool(members))
        focus = menu.addAction("Highlight this facies")
        focus.setEnabled(bool(members))
        if members:
            only.triggered.connect(lambda _checked=False, name=facies: self.show_only_facies(name))
            focus.triggered.connect(lambda _checked=False, name=facies: self.highlight_facies(name))
        clear = menu.addAction("Clear facies highlight")
        clear.setEnabled(self._highlight_facies is not None)
        clear.triggered.connect(lambda _checked=False: self.highlight_facies(None))

    def _facies_table_menu(self, position: QPoint) -> None:
        if self.session is None:
            return
        row = self.facies_table.indexAt(position).row()
        item = self.facies_table.item(row, 0) if row >= 0 else None
        facies = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None
        menu = QMenu(self)
        self._add_facies_display_actions(menu, facies)
        every = menu.addAction("Show all layers")
        every.setEnabled(bool(self._hidden_facies) or self._color_mode != "facies")
        every.triggered.connect(lambda _checked=False: self.set_all_display_visible("facies", True))
        self._facies_table_menu_widget = menu
        menu.popup(self.facies_table.viewport().mapToGlobal(position))

    def _ask_layer_snap(self, layer_id: int) -> None:
        """Ask for one layer's own snap distance and set it."""

        if self.session is None:
            return
        layer = self.session.layer(layer_id)
        if layer is None:
            return
        value, accepted = QInputDialog.getDouble(
            self,
            "Snap distance of this layer",
            f"Snap distance for {layer.name} (mm). 0 keeps its polygons exactly, "
            "for thin layers; the common distance is "
            f"{self.session.snap_mm:g} mm.",
            self.session.layer_snap_mm(layer_id),
            0.0,
            lp.MAX_SNAP_MM,
            1,
        )
        if accepted:
            self.set_layer_snap(layer_id, float(value))

    def set_layer_snap(self, layer_id: int, snap_mm: float | None) -> None:
        """Give one layer its own snap distance, or the common one with None."""

        if self.session is None:
            return
        layer = self.session.layer(layer_id)
        if layer is None:
            return
        value = None if snap_mm is None else float(min(max(snap_mm, 0.0), lp.MAX_SNAP_MM))
        if layer.snap_mm == value:
            return
        layer.snap_mm = value
        self._mark_dirty()
        self._refresh_layers()
        self._refresh_status()

    # ------------------------------------------------------------- marks

    def _snapshot(self) -> _Snapshot:
        assert self.session is not None
        return (
            [
                lp.Polygon(row.layer_id, list(row.points))
                for row in self.session.polygons
            ],
            [
                lp.Stroke(row.layer_id, row.radius_px, list(row.points))
                for row in self.session.strokes
            ],
            # Corrections are never changed once drawn: the list is enough.
            list(self.corrections),
            layer_priority.layer_ids(self.session),
        )

    def _restore(self, snapshot: _Snapshot) -> None:
        assert self.session is not None
        polygons, strokes, corrections = snapshot[:3]
        marked = frozenset(snapshot[3]) if len(snapshot) > 3 else frozenset()
        for layer in self.session.layers:
            layer.always_on_top = layer.layer_id in marked
        self.corrections = list(corrections)
        self.session.polygons = [
            lp.Polygon(row.layer_id, list(row.points)) for row in polygons
        ]
        self.session.strokes = [
            lp.Stroke(row.layer_id, row.radius_px, list(row.points)) for row in strokes
        ]
        if len(snapshot) > 4:
            rows, unit = snapshot[4]
            for name, role, porosity, permeability in rows:
                if name in self.session.facies:
                    self.session.facies[name] = lp.FaciesEntry(role, porosity, permeability)
            self.session.permeability_unit = unit

    def _push_undo(self) -> None:
        if self.session is None:
            return
        self._undo_stack.append(self._snapshot())
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []

    def _stroke_finished(self, stroke: lp.Stroke) -> None:
        if self.session is None:
            return
        snapshot = self._snapshot()
        try:
            self.session.add_stroke(stroke)
        except ValueError:
            self._render_marks()
            return
        self._undo_stack.append(snapshot)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []
        self._reveal_layer(stroke.layer_id)
        self.canvas.add_stroke_to_image(stroke, self._layer_colors())
        self._after_mark_change(render=False)

    def add_stroke(self, stroke: lp.Stroke) -> None:
        """Add a stroke as if it had been painted (used by scripts and tests)."""

        self._stroke_finished(stroke)

    def _polygon_finished(self, polygon: lp.Polygon) -> None:
        if self.session is None:
            return
        snapshot = self._snapshot()
        # Points drawn onto edges of other polygons become their points too,
        # in the same change (Follow edges).
        outline = dict(self.canvas.finished_outline_points)
        try:
            for index, points in sorted(outline.items()):
                if 0 <= index < len(self.session.polygons):
                    layer_id = self.session.polygons[index].layer_id
                    self.session.replace_polygon(
                        index, lp.Polygon(layer_id, list(points))
                    )
            self.session.add_polygon(polygon)
        except ValueError as error:
            self._restore(snapshot)
            self._canvas_notice(str(error))
            return
        self._undo_stack.append(snapshot)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []
        self._reveal_layer(polygon.layer_id)
        self._after_mark_change()

    def add_polygon(self, polygon: lp.Polygon) -> None:
        """Add a polygon as if it had been drawn (used by scripts and tests)."""

        self._polygon_finished(polygon)

    def _polygon_edited(
        self, index: int, points: Sequence[tuple[float, float]]
    ) -> None:
        self._polygons_edited({index: points})

    def _polygons_edited(
        self, changes: Mapping[int, Sequence[tuple[float, float]]]
    ) -> list[int]:
        """Validate and apply the whole shared edit as one undoable change.

        A refused neighbour keeps every member at its original geometry, so
        moving a shared point cannot split an edge through a partial edit.
        """
        if self.session is None:
            return []
        prepared: dict[int, lp.Polygon] = {}
        try:
            for index, points in sorted(dict(changes).items()):
                index = int(index)
                if not 0 <= index < len(self.session.polygons):
                    raise ValueError("The selected polygon changed; select its points again.")
                polygon = lp.Polygon(self.session.polygons[index].layer_id, list(points))
                self.session._check_polygon(polygon)
                prepared[index] = polygon
        except ValueError as error:
            self._canvas_notice(str(error))
            self._render_marks()
            return []
        if not prepared:
            return []
        snapshot = self._snapshot()
        for index, polygon in prepared.items():
            self.session.replace_polygon(index, polygon)
        self._undo_stack.append(snapshot)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []
        self._after_mark_change()
        return list(prepared)

    def delete_polygon(self, index: int) -> None:
        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        # Deleting a polygon renumbers the polygons after it.
        self.canvas.clear_selection()
        self._push_undo()
        self.session.remove_polygon(index)
        self._after_mark_change()

    def move_polygon(self, index: int, layer_id: int) -> None:
        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        if self.session.polygons[index].layer_id == layer_id:
            return
        self.canvas.clear_selection()
        self._push_undo()
        self.session.move_polygon(index, layer_id)
        self._reveal_layer(layer_id)
        self._after_mark_change()

    def remove_polygon_point(self, index: int, vertex: int) -> None:
        """Remove a polygon point, from every shown polygon that shares it, as
        one change; a polygon that would keep fewer than three points keeps
        it."""

        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        polygon = self.session.polygons[index]
        if len(polygon.points) <= 3 or not 0 <= vertex < len(polygon.points):
            return
        removed: dict[int, set[int]] = {}
        for number, shared in self.canvas.shared_with(index, vertex):
            if 0 <= number < len(self.session.polygons):
                removed.setdefault(number, set()).add(shared)
        changes = {
            number: [
                point
                for position, point in enumerate(self.session.polygons[number].points)
                if position not in gone
            ]
            for number, gone in removed.items()
            if len(self.session.polygons[number].points) - len(gone) >= 3
        }
        # The other selected points keep their selection.
        selection = self.canvas.selection_without_points(
            (number, shared) for number in changes for shared in removed[number]
        )
        if index in self._polygons_edited(changes):
            self.canvas.set_selection(selection)
        if len(changes) < len(removed):
            self._canvas_notice(_TOO_FEW_POINTS)

    def remove_selected_points(self) -> int:
        """Remove the points selected with the Edit tool, as one change to undo.

        A polygon that would keep fewer than three points is left unchanged
        and the hint line says so. Returns how many points were removed.
        """

        return self.canvas.delete_selected_points()

    def add_polygon_point(self, index: int, row: float, col: float) -> int:
        """Add a point to a polygon on its edge nearest to (row, col).

        (row, col) is in raw pixels, for example where the polygon was
        right-clicked; the point goes where it projects onto that edge. Like a
        moved point, the change can be undone. With the Edit tool the new
        point is selected. Returns the index of the new point, or -1.
        """

        if self.session is None or not 0 <= index < len(self.session.polygons):
            return -1
        # Shown polygons that share the edge get the point too.
        shown = {
            number: polygon.points
            for number, polygon in enumerate(self.session.polygons)
            if number == index or polygon.layer_id not in self.canvas.hidden_layers
        }
        added = insert_point_on_shared_edge(shown, index, row, col)
        if not added:
            return -1
        applied = self._polygons_edited(
            {number: points for number, (points, _vertex) in added.items()}
        )
        if index not in applied:
            return -1  # refused; the notice says why
        self.canvas.set_selection([(number, added[number][1]) for number in applied])
        return added[index][1]

    def _polygon_menu(self, index: int, vertex: int, position: QPoint) -> None:
        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        polygon = self.session.polygons[index]
        layer = self.session.layer(polygon.layer_id)
        menu = QMenu(self)
        title = menu.addAction(f"Polygon of {layer.name if layer else 'a layer'}")
        title.setEnabled(False)
        if vertex >= 0:
            remove_point = menu.addAction("Remove this point")
            remove_point.setEnabled(len(polygon.points) > 3)
            remove_point.triggered.connect(
                lambda _checked=False: self.remove_polygon_point(index, vertex)
            )
            selected = self.canvas.selected_points
            if len(selected) > 1 and (index, vertex) in selected:
                remove_selected = menu.addAction(
                    f"Remove {len(selected)} selected points"
                )
                remove_selected.triggered.connect(
                    lambda _checked=False: self.remove_selected_points()
                )
        elif self.canvas.last_context_raw is not None:
            # Where the polygon was right-clicked, in raw pixels.
            clicked = self.canvas.last_context_raw
            add_point = menu.addAction("Add point here")
            add_point.triggered.connect(
                lambda _checked=False, at=clicked: self.add_polygon_point(
                    index, at[0], at[1]
                )
            )
        join = menu.addAction("Join to neighbour's edge…")
        join.triggered.connect(lambda _checked=False: self._join_polygon_dialog(index))
        move = menu.addMenu("Move to layer")
        for row in self.session.layers:
            action = move.addAction(_swatch(row.color), row.name)
            action.setEnabled(row.layer_id != polygon.layer_id)
            action.triggered.connect(
                lambda _checked=False, target=row.layer_id: self.move_polygon(
                    index, target
                )
            )
        menu.addSeparator()
        delete = menu.addAction("Delete polygon")
        delete.triggered.connect(lambda _checked=False: self.delete_polygon(index))
        self._polygon_menu_widget = menu
        menu.popup(position)

    def _join_selected_polygon(self) -> None:
        indices = {index for index, _vertex in self.canvas.selected_points}
        if len(indices) != 1:
            self._canvas_notice("Select points of one polygon with Edit, or right-click its outline.")
            return
        self._join_polygon_dialog(next(iter(indices)))

    def _join_polygon_dialog(self, index: int) -> None:
        if self.session is None or self.image is None or self.busy():
            return
        if not 0 <= index < len(self.session.polygons):
            return
        target = self.session.polygons[index]
        if target.layer_id in self.canvas.hidden_layers:
            return
        neighbours = []
        from shapely.geometry import LineString

        outline = LineString([*target.points, target.points[0]])
        for number, polygon in enumerate(self.session.polygons):
            if number == index or polygon.layer_id in self.canvas.hidden_layers:
                continue
            layer = self.session.layer(polygon.layer_id)
            label = f"{layer.name if layer else 'Layer'} · polygon {number + 1}"
            points = list(polygon.points)
            distance = outline.distance(LineString([*points, points[0]]))
            neighbours.append((distance, number, label, points))
        if not neighbours:
            self._canvas_notice("Show another polygon to join to its edge.")
            return
        neighbours.sort(key=lambda row: (row[0], row[1]))
        per_mm = lp.snap_pixels(1.0, self.current_frame(), (1.0, 1.0), self.image.image_size)
        dialog = _JoinEdgesDialog(self, self.canvas, list(target.points),
                                  [(n, label, pts) for _d, n, label, pts in neighbours], per_mm)
        # The modal dialog prevents edits between its preview and Apply.
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.preview is not None:
                self._apply_join_preview(index, int(dialog.neighbour.currentData()), dialog.preview)
        finally:
            dialog.clear_preview()
            dialog.deleteLater()

    def _apply_join_preview(self, index: int, neighbour: int, preview: JoinPreview) -> bool:
        if self.session is None or not preview.stretches or self.busy():
            return False
        if index == neighbour or any(
            not 0 <= number < len(self.session.polygons)
            or self.session.polygons[number].layer_id in self.canvas.hidden_layers
            for number in (index, neighbour)
        ):
            return False
        changes = {
            index: lp.Polygon(self.session.polygons[index].layer_id, list(preview.target)),
            neighbour: lp.Polygon(self.session.polygons[neighbour].layer_id, list(preview.neighbour)),
        }
        try:
            # Validate both before either changes, so this is an atomic edit.
            for polygon in changes.values():
                self.session._check_polygon(polygon)
        except ValueError as error:
            self._canvas_notice(str(error))
            return False
        self._push_undo()
        for number, polygon in changes.items():
            self.session.replace_polygon(number, polygon)
        self.canvas.clear_selection()
        self._after_mark_change()
        self._canvas_notice("Joined to the neighbour's edge. Ctrl+Z restores both polygons.")
        return True

    def _draft_changed(self, count: int) -> None:
        if count:
            hint = _DRAFT_HINT.format(count=count, plural="s" if count != 1 else "")
            if self.canvas.follow_edges:
                hint += _FOLLOW_HINT
            self.canvas_hint.setText(hint)
        else:
            self.canvas_hint.setText(_CANVAS_HINT)

    def _canvas_notice(self, message: str) -> None:
        self.canvas_hint.setText(str(message))

    def show_frame_hint(self, text: str) -> None:
        """What to click while the rig frame panel marks corners or points."""

        if getattr(self, "canvas_hint", None) is None:
            return  # the panel is created before the canvas column
        if text:
            self.canvas_hint.setText(text)
        elif self.canvas.drafting():
            self._draft_changed(len(self.canvas.draft_points))
        else:
            self.canvas_hint.setText(_CANVAS_HINT)

    def undo(self) -> None:
        boxes = getattr(self, "analysis_boxes", None)
        if boxes is not None and boxes.busy():
            return
        if self._analysis_boxes_owns_undo():
            boxes.undo()
            return
        if self.session is None or not self.run_preview.is_reference():
            return
        if self.canvas.drafting():
            self.canvas.remove_last_point()
            return
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        current = self._facies_snapshot() if len(snapshot) > 4 else self._snapshot()
        if snapshot[0] != current[0]:
            self.canvas.clear_selection()  # the polygons change
        self._redo_stack.append(current)
        self._restore(snapshot)
        self._after_restore(current, snapshot)

    def redo(self) -> None:
        boxes = getattr(self, "analysis_boxes", None)
        if boxes is not None and boxes.busy():
            return
        if self._analysis_boxes_owns_undo():
            boxes.redo()
            return
        if self.session is None or not self._redo_stack or not self.run_preview.is_reference():
            return
        snapshot = self._redo_stack.pop()
        current = self._facies_snapshot() if len(snapshot) > 4 else self._snapshot()
        if snapshot[0] != current[0]:
            self.canvas.clear_selection()  # the polygons change
        self._undo_stack.append(current)
        self._restore(snapshot)
        self._after_restore(current, snapshot)

    def _after_restore(self, before: _Snapshot, after: _Snapshot) -> None:
        """Refresh what an undo or redo changed: the marks, the corrections or both.

        A change of the corrections alone does not write the session.
        """

        priority_changed = before[3:4] != after[3:4]
        if priority_changed:
            self._manual_display = None
        if before[:2] != after[:2] or priority_changed:
            self._after_mark_change()
        if priority_changed:
            self._render_proposal()
            self._schedule_manual_map()
        if before[2] != after[2]:
            self._after_correction_change()
        if before[4:] != after[4:]:
            self._sync_settings()
            self._facies_changed()

    # ------------------------------------------------------- corrections

    def _layer_ids(self) -> set[int]:
        if self.session is None:
            return set()
        return {layer.layer_id for layer in self.session.layers}

    def _read_corrections(self, run_id: str) -> list[lp.Correction]:
        """The run's corrections; none, and correcting off, when unreadable."""

        try:
            return lp.load_corrections(self.project, run_id)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._corrections_error = (
                f"The corrections of {run_id} could not be read ({error}); "
                "correcting is off until the file is fixed."
            )
            return []

    def _save_corrections(self) -> None:
        if not self._corrections_dirty or self.session is None:
            return
        try:
            lp.save_corrections(self.project, self.session.run_id, self.corrections)
        except OSError as error:
            self.propose_status.setText(f"The corrections could not be saved: {error}")
            return
        self._corrections_dirty = False

    def _corrections_unavailable(self) -> str:
        """Why no correction can be drawn now ("" when one can)."""

        if self.session is None or self.image is None:
            return "Choose a reference run with a baseline photograph first."
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            return "Saved corrected rig maps are shown; corrections need a proposal."
        if self._map_kind() == _MANUAL:
            return (
                "Corrections are drawn on the proposal; show the proposal to draw "
                "them. A manual map changes with its polygons."
            )
        if self._corrections_error:
            return self._corrections_error
        if self.proposal is None or tuple(self.proposal.image_size) != tuple(
            self.image.image_size
        ):
            return (
                "Propose the boundaries first; corrections are drawn on "
                "the proposal."
            )
        if lpp.is_polygons_only(self.proposal) and not self.proposal_current():
            return "The exact polygons or frame changed; propose again before correcting this map."
        return ""

    def corrections_available(self) -> bool:
        """True when hand corrections can be drawn on the proposal shown."""

        return not self._corrections_unavailable()

    @Slot(object)
    def _correction_drawn(self, correction: lp.Correction) -> None:
        self.add_correction(correction)

    def add_correction(self, correction: lp.Correction) -> bool:
        """Add a hand correction on top of the proposal shown.

        Used by the correction tools, scripts and tests. The correction is
        bound to the proposal it is drawn on and is one step to undo. Returns
        False, and says why, when no correction can be drawn now.
        """

        problem = self._corrections_unavailable()
        if problem:
            self._canvas_notice(problem)
            return False
        assert self.proposal is not None
        correction.proposal_digest = self.proposal.digest
        correction.created_utc = utc_now()
        self._push_undo()
        self.corrections.append(correction)
        if correction.kind != "restore":
            self._reveal_layer(correction.layer_id)
        self._after_correction_change()
        return True

    @Slot(float, float)
    def _fill_requested(self, row: float, col: float) -> None:
        self.fill_region_at(row, col)

    def fill_region_at(self, row: float, col: float) -> bool:
        """Give the region of the corrected proposal under a raw point the selected
        layer, as one correction."""

        problem = self._corrections_unavailable()
        if problem:
            self._canvas_notice(problem)
            return False
        assert self.proposal is not None
        if lpp.is_polygons_only(self.proposal):
            state = self._corrected_native_state()
            if state is None:
                self._canvas_notice("Computing the exact native corrections; wait before filling a region.")
                return False
            classes = state[0]
        else:
            classes = lp.corrected_classes(self.proposal, self.corrections,
                                           layer_ids=self._layer_ids())[0]
        scale = lp.proposal_grid_scale(self.proposal)
        current = lp.layer_at(classes, (row, col), scale)
        layer = int(self.canvas.layer_id)
        inside_gap = False
        if lpp.is_polygons_only(self.proposal):
            seed = lp._grid_seed((row, col), classes.shape, scale)
            inside_gap = seed is not None and bool(self.proposal.inside[seed])
        if current is None or (current == lp.OUTSIDE and not inside_gap):
            self._canvas_notice("Fill works on the regions inside the rig frame.")
            return False
        if current == layer:
            self._canvas_notice("That region already has the selected layer.")
            return False
        return self.add_correction(
            lp.Correction("fill", layer, [(row, col)], replaced_layer=current)
        )

    def _corrected_native_state(self, *, shown: bool = False):
        """Cache this method's exact native corrections before display reduction.

        Editing and export always use all corrections. The visibility switch
        affects only the separately keyed display state.
        """
        proposal = self.proposal
        assert lpp.is_polygons_only(proposal)
        if self._native_async_required():
            result = self._large_native_result()
            if result is None or result.error:
                return None
            visible = not shown or self.show_corrections.isChecked()
            state = (result.classes, result.decided, list(result.skipped)) if visible else (
                proposal.classes, result.base_decided, [])
            self._native_corrections_state = state
            self._native_corrections_gap_mask = result.gap_mask
            self._native_corrections_gaps = result.gap_count
            self._native_corrections_key = result.key
            return state
        visible = not shown or self.show_corrections.isChecked()
        corrections = tuple(self.corrections) if visible else ()
        layer_ids = frozenset(self._layer_ids())
        key = (id(proposal), proposal.digest, id(proposal.classes), id(proposal.inside),
               lpp._json_bytes(proposal.source), lp.corrections_digest(corrections),
               layer_ids, visible)
        cached = self._native_corrections_cache.get(key)
        if cached is None:
            # A stale draft may have added/removed a layer. Rendering remains
            # passive; currentness/export gates handle that input change.
            known = layer_ids & set(proposal.source["layer_ids"])
            state = lpp.corrected_classes(proposal, corrections, layer_ids=known)
            gaps = proposal.inside & (state[0] == lp.OUTSIDE)
            cached = state, gaps, int(np.count_nonzero(gaps))
            if len(self._native_corrections_cache) >= 2:
                self._native_corrections_cache.clear()
            self._native_corrections_cache[key] = cached
        self._native_corrections_key = key
        self._native_corrections_state, self._native_corrections_gap_mask, self._native_corrections_gaps = cached
        return self._native_corrections_state

    def _native_async_required(self):
        return lpp.is_polygons_only(self.proposal) and self.proposal.classes.size > ASYNC_PIXELS

    def _large_native_result(self):
        """Request the exact current snapshot; only its worker may fill the cache."""
        assert self._native_async_required() and self.session is not None and self.image is not None
        proposal, frame, reference = self.proposal, self.current_frame(), self.reference()
        payload = lpv._drawing_payload(self.session)
        payload.pop("updated_utc", None)
        payload.pop("updated_by", None)
        reference_key = (str(reference.run_id), str(reference.baseline_image),
                         reference.width_m, reference.height_m) if reference is not None else None
        layer_ids = frozenset(self._layer_ids())
        shape = tuple(self.image.display.shape[:2])
        key = (id(proposal), proposal.digest, id(proposal.classes), id(proposal.confidence),
               id(proposal.inside), lpp._json_bytes(proposal.source),
               lpp._json_bytes([payload, frame.to_dict() if frame else None, reference_key]),
               lp.corrections_digest(self.corrections), layer_ids, shape)
        was_busy = self._native_async.busy()
        result = self._native_async.request(key, proposal, self.session, self.corrections,
            frame, reference, layer_ids=layer_ids, display_shape=shape)
        if not was_busy and self._native_async.busy():
            self.busy_changed.emit(True)
        return result

    @Slot()
    def _native_cache_changed(self):
        # Every callback executes in the view's thread. A result for a prior
        # draft can be reused by Undo, but is never shown as the current map.
        if self._native_async_required() and self.session is not None and self.image is not None:
            self._render_proposal()
            self._refresh_status()
            self._schedule_manual_map()
            self._refresh_manual_status()
        self.busy_changed.emit(self.busy())

    def clear_corrections(self, *, confirm: bool = True) -> bool:
        """Remove all corrections of the run shown, as one step to undo."""

        if not self.corrections:
            return False
        if confirm:
            answer = QMessageBox.question(
                self,
                "Clear corrections",
                f"Remove all {len(self.corrections)} corrections of this run? "
                "Undo (Ctrl+Z) brings them back.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self._push_undo()
        self.corrections = []
        self._after_correction_change()
        return True

    def _clear_corrections_clicked(self) -> None:
        self.clear_corrections(confirm=True)

    def _corrections_shown(self, _checked: bool) -> None:
        self._render_proposal()

    def _mark_corrections_dirty(self) -> None:
        self._corrections_dirty = True
        self._export_stale = True
        if not self._loading:
            self._save_timer.start()
        self._schedule_manual_map()
        self.changed.emit()

    def _after_correction_change(self) -> None:
        self._mark_corrections_dirty()
        self._render_proposal()
        self._refresh_status()

    def _corrections_text(self) -> str:
        if self._corrections_error:
            return self._corrections_error
        if lpp.is_polygons_only(self.proposal):
            if self._native_async_required():
                result = self._large_native_result()
                if result is None:
                    return f"Computing native corrections… {len(self.corrections)} correction(s) queued."
                if result.error:
                    return "Native corrections unavailable: " + result.error
                text = f"{len(self.corrections)} correction(s)" if self.corrections else "No corrections"
                text += f" · {result.gap_count:,} unassigned pixels (magenta)"
                if result.skipped:
                    text += f" · {len(result.skipped)} no longer apply"
                return text
            _classes, _decided, skipped = self._corrected_native_state()
            text = f"{len(self.corrections)} correction(s)" if self.corrections else "No corrections"
            text += f" · {self._native_corrections_gaps:,} unassigned pixels (magenta)"
            if skipped:
                text += f" · {len(skipped)} no longer apply"
            return text
        if not self.corrections:
            return "No corrections." if self.proposal is not None else ""
        parts = [f"{len(self.corrections)} correction(s)"]
        if self.proposal is not None:
            earlier = sum(
                row.proposal_digest != self.proposal.digest for row in self.corrections
            )
            if earlier:
                parts.append(f"{earlier} drawn on an earlier proposal")
            missed = lp.corrections_not_applied(
                self.proposal, self.corrections, layer_ids=self._layer_ids()
            )
            if missed:
                parts.append(f"{len(missed)} no longer apply")
        return " · ".join(parts)

    def _refresh_corrections(self) -> None:
        """The corrections line and which correction tools can be used."""

        if getattr(self, "corrections_label", None) is None:
            return  # Called before the canvas column is built.
        self.corrections_label.setText(self._corrections_text())
        self.clear_corrections_button.setEnabled(
            bool(self.corrections) and not self._corrections_error
        )
        available = self.corrections_available()
        for key in _CORRECTION_TOOLS:
            self.tool_buttons[key].setEnabled(available)
        if not available and self.canvas.tool in _CORRECTION_TOOLS:
            self.set_tool("pan")

    # ---------------------------------------------------------- versions

    def _read_versions(self, run_id: str) -> list[dict[str, Any]]:
        """The run's saved versions, read off the GUI thread.

        A quick reading answers at once. The library is checked file by file,
        which takes seconds on a series share: the rows already shown for the
        same run then stay (none for another run) until _versions_read shows
        the answer.
        """
        reader = self.__dict__.get("_versions_reader")
        if reader is None:
            from .background_read import BackgroundRead
            reader = self._versions_reader = BackgroundRead(self)
            reader.answered.connect(self._versions_read)
        project = self.project
        key = (id(project), run_id)
        done, rows, error = reader.read(
            key, lambda: lpv.list_version_metadata(project, run_id),
            call=("darsia.fluidflower.layer_painting_version_archive:list_version_metadata", (project, run_id)))
        if done:
            return self._versions_answer(key, rows, error)
        if hasattr(self, "versions_note"):
            self.versions_note.setText(_VERSIONS_READING)
        return list(self.versions) if self.__dict__.get("_versions_shown") == key else []

    def versions_reading(self) -> bool:
        """True while a slow reading of the version library runs (see _read_versions)."""
        reader = self.__dict__.get("_versions_reader")
        return reader is not None and reader.busy()

    def _versions_answer(self, key, rows, error) -> list[dict[str, Any]]:
        self._versions_shown = key
        if error:
            if hasattr(self, "versions_note"):
                self.versions_note.setText(f"The version library could not be read: {error}")
            return []
        return rows

    def _versions_read(self, key, rows, error) -> None:
        """Show a slow reading's versions, unless another project is open meanwhile."""
        if self.project is None or key[0] != id(self.project):
            return
        note = self.versions_note.text() if hasattr(self, "versions_note") else ""
        self.versions = self._versions_answer(key, rows, error)
        self._versions_arriving = True
        try:
            self._refresh_versions()
        finally:
            self._versions_arriving = False
        if not error and note and note != _VERSIONS_READING:
            # A message written after the reading started, such as a saved version's, stays.
            self.versions_note.setText(note)
        self._background_reading_answered()

    def _version_selection_key(self):
        if self.session is None:
            return None
        root = str(Path(self.project.root).resolve()).casefold()
        identity = hashlib.sha256((root + "\n" + self.session.run_id.casefold()).encode()).hexdigest()
        return "versions/selected/" + identity

    def _remembered_version(self):
        key = self._version_selection_key()
        if key is None:
            return None
        settings = QSettings(*_BACKGROUND_SETTINGS)
        settings.sync()
        try:
            saved = json.loads(str(settings.value(key, "null")))
        except (ValueError, TypeError):
            saved = None
        if not isinstance(saved, dict):
            return None
        record = next((row for row in self.versions
            if all(row.get(field) == saved.get(field) for field in ("folder", "sha256", "kind"))
            and (Path(row["folder"]) / "version.json").is_file()), None)
        if record is None and self.versions_reading():
            return None  # Known once the versions are read; the remembered choice stays.
        if record is None:
            settings.remove(key)
            self.version_snapshot_note.setText("The remembered version is unavailable. Choose another saved version; the working map is unchanged.")
        return record

    def show_version_metadata(self, session, record):
        self._version_snapshot_session = session
        name = record.get("display_name") or record.get("name")
        self.version_snapshot_note.setText(f"Viewing {name} ({record.get('kind')}, SHA-256 {str(record.get('sha256'))[:12]}). "
            "Saved map is read only. Restore version to edit it; the working map remains unchanged.")
        self._refresh_layers()
        self._refresh_facies_table()
        self._guard_reference_display_controls()
        self.display_styles_changed.emit()

    def _read_approvals(self, run_id: str) -> None:
        """Read the run's map approvals; an unreadable file is said, never
        taken as "not approved"."""

        try:
            self.map_approvals = lp.read_map_approvals(self.project, run_id)
            self._map_approvals_error = ""
        except (OSError, ValueError) as error:
            self.map_approvals = []
            self._map_approvals_error = str(error)

    def _approval_text(self, approval: Mapping[str, Any]) -> str:
        when = str(approval.get("approved_utc") or "")[:16].replace("T", " ")
        reviewer = approval.get("reviewer")
        return f"approved {when} UTC" + (f" by {reviewer}" if reviewer else "")

    def _versions_unavailable(self) -> str:
        """Why no version can be saved now ("" when one can)."""

        if self._native_async_required():
            result = self._large_native_result()
            if result is None or self._native_async.busy():
                return "Computing the exact native map; wait before saving a version."
            if result.error:
                return "The exact native map is unavailable: " + result.error
        if self.busy():
            return "Wait until the proposal or export has finished."
        if self._map_kind() == _MANUAL:
            problem = self._manual_unavailable()
            if not problem and not self._manual_current():
                problem = "The manual map is being drawn; save it once it is shown."
            return problem
        problem = self._corrections_unavailable()
        if problem:
            return problem
        return ""

    def _refresh_versions(self) -> None:
        """The versions table, newest first, and its buttons."""

        if getattr(self, "versions_table", None) is None:
            return  # Called before the steps are built.
        native_ready = True
        if self._native_async_required():
            native = self._large_native_result()
            native_ready = native is not None and not native.error
        selected = self._selected_version() or self._remembered_version()
        keep = selected.get("folder") if selected else None
        table = self.versions_table
        table.blockSignals(True)
        table.clearSelection()
        table.setRowCount(len(self.versions))
        for row, record in enumerate(reversed(self.versions)):
            digest = str(record.get("sha256") or "")
            count = int(record.get("correction_count") or 0)
            kind = f"edited ({count})" if count else "proposal"
            if record.get("kind") == lpv.DRAWING_KIND:
                kind = "drawing"
            if lpm.is_manual_version(record):
                kind = "manual map (gaps)" if record.get("uncovered_pixels") else "manual map"
            saved = str(record.get("created_utc") or "")[:16].replace("T", " ")
            approval = self._approval_for_version(record)
            approved = (
                str(approval.get("approved_utc") or "")[:16].replace("T", " ")
                if approval
                else ""
            )
            from .painting_versions_ui import settings_text
            state = f"Approved {approved} UTC" if approval else "Not approved"
            if approval is None and record.get("approval_withdrawn_utc"):
                state = "approval withdrawn " + str(record["approval_withdrawn_utc"])
            if record.get("placement_withdrawn"):
                state += " · Placement approval withdrawn"
            if record.get("kind") == lpv.DRAWING_KIND:
                state = "Drawing only"
            elif record.get("uncovered_pixels"):
                state = f"{int(record['uncovered_pixels']):,} unassigned pixels"
            history_error = str(record.get("history_error") or self._map_approvals_error or "")
            if history_error:
                state = "History UNKNOWN"
            for flag, label in (("placed", "Placed"), ("adopted", "Adopted")):
                if record.get(flag):
                    state += " · " + label
            for column, value in enumerate((str(record.get("display_name") or record.get("name") or ""),
                    kind, settings_text(record), saved, digest[:12], state)):
                item = QTableWidgetItem(value)
                tooltip = (f"SHA-256 {digest}\n" + str(record.get("note") or "")
                    + "\n" + json.dumps(record.get("settings") or {}, ensure_ascii=False, sort_keys=True))
                if approval is not None and not history_error:
                    tooltip += "\nApproval: " + json.dumps(approval, ensure_ascii=False, sort_keys=True)
                if history_error:
                    tooltip += "\nHistory UNKNOWN: " + history_error
                item.setToolTip(tooltip)
                item.setData(Qt.ItemDataRole.UserRole, record.get("folder"))
                table.setItem(row, column, item)
            if keep is not None and record.get("folder") == keep:
                table.selectRow(row)
        table.blockSignals(False)
        available = not self._versions_unavailable()
        self.save_version_button.setEnabled(available)
        self.restore_version_button.setEnabled(
            self._selected_version() is not None and not self.busy()
        )
        self.approve_version_button.setEnabled(
            self._selected_version() is not None
            and self._selected_version().get("kind") != lpv.DRAWING_KIND
            and not self.busy()
            and not self._map_approvals_error
            and native_ready
        )
        self.place_version_button.setEnabled(self._placeable(self._selected_version()))
        self.prepare_adoption_button.setEnabled(self._placeable(self._selected_version()))
        self.adopt_button.setEnabled(not self.busy() and not self._map_approvals_error and native_ready)
        self._version_selected()
        self._refresh_use_map()
        if hasattr(self, "version_actions"):
            self.version_actions.refresh()
        if hasattr(self, "run_preview"):
            self.run_preview._refresh_sources()

    def _refresh_map_status(self) -> None:
        """Name the selected saved identity used by the series-map route."""

        label = getattr(self, "map_status", None)
        if label is None:
            return  # Called before the steps are built.
        if self._map_approvals_error:
            label.setText(
                f"Map approvals could not be read: {self._map_approvals_error}"
            )
            return
        manual = self._selected_manual_version()
        if manual is not None:
            version, approval = manual
            name = version.get("display_name") or version.get("name")
            if approval is not None:
                label.setText(f"Using {name} ({self._approval_text(approval)}) "
                              f"(SHA-256 {str(version['sha256'])[:12]}…). "
                              "This selected approval counts for Use the map in the series.")
            else:
                label.setText(f"Map selected: {name}, not approved. Approve this saved manual-map "
                              "version under Versions before transferring it or checking the series revision.")
            return
        if self.session is None or self.proposal is None:
            label.setText("")
            return
        if self._native_async_required():
            native = self._large_native_result()
            if native is None:
                label.setText("Computing the exact native map and its SHA-256…")
                return
            if native.error:
                label.setText("The exact native map is unavailable: " + native.error)
                return
        try:
            digest = self._current_map_digest()
        except (OSError, ValueError, TypeError) as error:
            label.setText(f"The map's SHA-256 could not be computed: {error}")
            return
        version = next(
            (row for row in reversed(self.versions) if row.get("sha256") == digest),
            None,
        )
        approval = lp.map_approval(self.map_approvals, digest)
        if approval is not None:
            label.setText(
                f"Map to save: the approved version {approval.get('name')} "
                f"(SHA-256 {digest[:12]}…, {self._approval_text(approval)})."
            )
        elif version is not None:
            label.setText(
                f"Map to save: the saved version {version.get('name')} (SHA-256 "
                f"{digest[:12]}…), not approved. Approve it under Versions."
            )
        else:
            label.setText(
                "Map to save: not saved as a version. Save it as a version in "
                "Versions and approve it there before using it in the series."
            )

    def _approval_for_version(self, record):
        """Return only the literal saved version's readable existing approval."""
        if record is None or self._map_approvals_error or record.get("history_error"):
            return None
        folder = Path(str(record.get("folder") or ""))
        files = record.get("files")
        if not record.get("folder") or not isinstance(files, Mapping):
            return None
        row = lp.map_approval(self.map_approvals, record.get("sha256"), version=folder.name)
        return row if (row is not None and str(row.get("kind") or "") == str(record.get("kind") or "")
            and row.get("run_id") == record.get("run_id") and row.get("files") == files) else None

    def _selected_manual_version(self):
        """The explicit saved identity used by round 5, separate from drafts.

        Reading an existing approval here never hashes a proposal or redraws a
        manual map. The actual transfer/adoption route keeps its own checks.
        """
        version = self._selected_version()
        if version is None or not lpm.is_manual_version(version):
            return None
        return version, self._approval_for_version(version)

    def _refresh_manual_export_route(self) -> bool:
        """Replace the older export warning only for the selected manual map."""
        if not hasattr(self, "export_button"):
            return False
        manual = self._selected_manual_version()
        self.export_button.setVisible(manual is None)
        if manual is None:
            return False
        _version, approval = manual
        if self._map_approvals_error or _version.get("history_error"):
            self.step_export.set_status("Approval history cannot be read", "warn")
            detail = "The approval history cannot be read; refresh it before continuing."
        elif approval is None:
            self.step_export.set_status("Approve the selected manual map under Versions", "warn")
            detail = "Approve the selected saved manual-map version under Versions."
        else:
            self.step_export.set_status("Approved manual map: use it in the series", "done")
            detail = "The selected approved manual map uses Use the map in the series."
        self.export_problems.setText(detail + " Transfer map to all runs → Review placement → "
            "Check the series revision → Adopt as the series' sand map. <a href='round5'>Open Use the map in the series</a>. "
            "The older native Setup export is hidden for this route.")
        self.export_problems.setStyleSheet("")
        self.export_button.setEnabled(False)
        return True

    def _open_round5_step(self, _link=None) -> None:
        """Open the series-map actions for the selected saved manual map."""
        self.workflow_scope_requested.emit('apply')
        self.set_workflow_scope('apply')
        self.step_use_map.set_expanded(True)
        self.place_version_button.setFocus()
        parent = self.place_version_button.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(self.place_version_button)
                break
            parent = parent.parentWidget()

    def _selected_version(self) -> dict[str, Any] | None:
        table = getattr(self, "versions_table", None)
        if table is None:
            return None
        rows = table.selectionModel().selectedRows()
        item = table.item(rows[0].row(), 0) if rows else None
        folder = None if item is None else item.data(Qt.ItemDataRole.UserRole)
        return next((row for row in self.versions if row.get("folder") == folder), None)

    def _version_clicked(self, _row, _column):
        self._version_preview_identity = None
        self._version_selected()

    def _version_selected(self) -> None:
        record = self._selected_version()
        self._sync_use_map_version()
        self._refresh_transfer_runs()
        key = self._version_selection_key()
        # No row is selected while the versions are still read; the remembered choice stays.
        if key is not None and (record is not None or not self.versions_reading()):
            settings = QSettings(*_BACKGROUND_SETTINGS)
            if record is None:
                settings.remove(key)
            else:
                settings.setValue(key, json.dumps({field: record.get(field) for field in ("folder", "sha256", "kind")}))
            settings.sync()
        if record is not None:
            text = f"{record.get('name')}: SHA-256 {record.get('sha256')}"
            approval = self._approval_for_version(record)
            if lpm.is_manual_version(record):
                approval = self._selected_manual_version()[1]
            if approval is not None:
                text += f" · {self._approval_text(approval)}"
                if lpm.is_manual_version(record):
                    text += " · this selected approval counts for round 5"
            gaps = self._version_gaps(record)
            if gaps:
                text += f" · {gaps}"
            placement = self.last_placement or {}
            version = placement.get("map_version") if isinstance(placement, Mapping) else None
            rows = placement.get("runs") if isinstance(placement, Mapping) else None
            if (not self.busy() and approval is not None and isinstance(version, Mapping)
                    and version.get("folder") == record.get("folder")
                    and all(version.get(key) == record.get(key) for key in ("name", "kind", "sha256"))
                    and version.get("approval") == approval and isinstance(rows, list)
                    and all(isinstance(row, Mapping) for row in rows)
                    and isinstance(placement.get("folder"), str)):
                placed = sum(row.get("status") == "placed" for row in rows)
                text += f" · Placed in {placed} of {len(rows)} runs: {placement['folder']}"
            self.versions_note.setText("Selected version · " + text)
            identity = (str(self.project.root), self.session.run_id, record.get("folder"), record.get("sha256"))
            if (self._workflow_scope not in {'rig_frame', 'analysis_boxes'} and not self._loading and hasattr(self, "run_preview")
                    and identity != self._version_preview_identity):
                self._version_preview_identity = identity
                # Versions that arrive from a slow reading keep the transfer review
                # of this route open, as when they were read before it opened.
                if not (self._workflow_scope == 'apply' and self.__dict__.get('_versions_arriving')):
                    self.show_mark_layers()
                self.run_preview.show_saved_version(record)
        elif self._version_preview_identity is not None:
            self._version_preview_identity = None
            self._version_snapshot_session = None
            self.run_preview.show_working_map()
        self.restore_version_button.setEnabled(record is not None and not self.busy())
        native_ready = True
        if self._native_async_required():
            native = self._large_native_result()
            native_ready = native is not None and not native.error
        self.approve_version_button.setEnabled(
            record is not None and not self.busy() and not self._map_approvals_error
            and record.get("kind") != lpv.DRAWING_KIND
            and native_ready
        )
        self.place_version_button.setEnabled(self._placeable(record))
        self.prepare_adoption_button.setEnabled(self._placeable(record))
        self._refresh_map_status()
        self._refresh_use_map()
        if not self._refresh_manual_export_route() and self.session is not None:
            self._refresh_legacy_export_status()
        if hasattr(self, "version_actions"):
            self.version_actions.refresh()
        self._refresh_facies_approval()

    def _sync_use_map_version(self):
        if not hasattr(self, 'use_map_version'):
            return
        selected = self._selected_version()
        combo = self.use_map_version
        blocked = combo.blockSignals(True)
        combo.clear()
        for record in reversed(self.versions):
            name = record.get('display_name') or record.get('name')
            combo.addItem(f"{name} · {record.get('kind')} · {str(record.get('sha256'))[:12]}", record['folder'])
        combo.setCurrentIndex(combo.findData(selected['folder']) if selected is not None else -1)
        combo.blockSignals(blocked)

    def _use_map_version_selected(self, index):
        folder = self.use_map_version.itemData(index)
        for row in range(self.versions_table.rowCount()):
            if self.versions_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == folder:
                self.versions_table.selectRow(row)
                if getattr(self, '_workflow_scope', None) == 'apply':
                    QTimer.singleShot(0, lambda: self.adoption_controller.open_chart_review(automatic=True)
                        if getattr(self, '_workflow_scope', None) == 'apply' else None)
                break

    def _refresh_transfer_runs(self, *_args):
        if not hasattr(self, 'use_map_runs'):
            return
        panel = getattr(getattr(self, 'adoption_controller', None), 'chart_dialog', None)
        context = getattr(getattr(self, 'adoption_controller', None), '_chart_context', None) or {}
        selected, version = self._selected_version(), context.get('version') or {}
        session = getattr(panel, 'session', None)
        current = bool(selected and session is not None and version.get('folder') == selected.get('folder')
                       and version.get('sha256') == selected.get('sha256'))
        chooser = getattr(self, 'run_preview', None)
        registered = [str(chooser.runs.itemData(index)).casefold()
                      for index in range(chooser.runs.count())] if chooser is not None else []
        active_runs = set(registered) if self._series_root() is not None else None
        table = self.use_map_runs
        blocked = table.blockSignals(True)
        table.setRowCount(0)
        if not current:
            reference = str((selected or {}).get('run_id') or '').strip().casefold()
            rows = [(reference, 'Reference · not transferred', '')] if reference else []
            active = reference or None
            self.use_map_run_note.setText('Transfer the selected map to show its runs, or open its saved transfer review.'
                if selected else 'Select a saved map version.')
        else:
            approvals = panel.approvals
            active = getattr(panel, '_excluded_run', None) or panel.runs.currentData()
            rows = [(row['run_id'], 'Approved' if row['run_id'] in approvals else 'Waiting for approval', '')
                    for row in session.review['runs']]
            rows.extend((row['run_id'], 'Excluded', str(row.get('reason') or ''))
                        for row in panel.excluded_runs)
            if active_runs is not None:
                rows = [row for row in rows if row[0].casefold() in active_runs]
            reviewed = [run for run, state, _reason in rows if state != 'Excluded']
            approved = sum(run in approvals for run in reviewed)
            self.use_map_run_note.setText(f'{approved} of {len(reviewed)} approved. '
                'Select a run to review its photograph.')
        table.setRowCount(len(rows))
        for index, (run, state, reason) in enumerate(rows):
            for column, text in enumerate((run.upper(), state)):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, run)
                item.setToolTip(reason or state)
                if state == 'Approved':
                    item.setForeground(QColor('#299666'))
                table.setItem(index, column, item)
            if run == active:
                table.selectRow(index)
        table.blockSignals(blocked)

    def _use_map_run_selected(self):
        panel = getattr(self.adoption_controller, 'chart_dialog', None)
        if panel is not None and getattr(panel, 'bulk_approval_active', False):
            self._refresh_transfer_runs()
            return
        rows = self.use_map_runs.selectionModel().selectedRows()
        if not rows:
            return
        run = self.use_map_runs.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        context = getattr(self.adoption_controller, '_chart_context', None) or {}
        selected, version = self._selected_version(), context.get('version') or {}
        if (panel is None or panel.session is None or not selected
                or any(selected.get(key) != version.get(key) for key in ('folder', 'sha256'))):
            index = self.run_preview.runs.findData(run)
            if index >= 0:
                self.run_preview.runs.setCurrentIndex(index)
            return
        self.show_transfer_review(panel)
        index = panel.runs.findData(run)
        if index >= 0:
            panel.runs.setCurrentIndex(index)
        else:
            excluded = next((row for row in panel.excluded_runs if row['run_id'] == run), None)
            if excluded is not None:
                panel.show_excluded_run(run, str(excluded.get('reason') or ''))

    def _refresh_use_map(self) -> None:
        """Refresh action readiness from the literal selected version approval."""
        if not hasattr(self, "use_map_reasons"):
            return
        record = self._selected_version()
        approval = self._approval_for_version(record)
        if self._map_approvals_error or (record is not None and record.get("history_error")):
            problem = "The approval history could not be read; refresh it before continuing."
        elif record is None:
            problem = "Choose a saved map version under Versions."
        elif record.get("kind") == lpv.DRAWING_KIND:
            problem = "This is a drawing. Save a manual map or a proposal version under Versions."
        elif self._version_gaps(record):
            problem = self._version_gaps(record)
        elif approval is None:
            problem = "Approve this saved version under Versions."
        elif self.busy():
            problem = "Wait for the current operation to finish."
        elif self._reference_point_review_blocker():
            problem = self._reference_point_review_blocker()
        elif not self._placeable(record):
            problem = "Wait for the checked native map before continuing."
        else:
            problem = ""
        for button in (self.place_version_button, self.adopt_button):
            button.setEnabled(not problem)
            self.use_map_reasons[button].setText(problem)
        route_problem = "The rigs need their RAW basis and the automatic adoption backend before the map can be adopted."
        if self._series_adopt_action is None:
            self.adopt_button.setEnabled(False)
            self.use_map_reasons[self.adopt_button].setText(problem or route_problem)
        review_ready = approval is not None or (record is None and any(
            self._approval_for_version(version) for version in self.versions))
        self.chart_review_button.setEnabled(not self.busy() and review_ready)
        current_findings = self.adoption_controller.current_gap_review()
        self.gap_review_button.setEnabled(review_ready and (not self.busy() or current_findings is not None))
        findings = getattr(self, '_gap_review_dialog', None)
        self.check_gaps_button.setEnabled(not self.busy() and findings is not None and findings.recheck.isEnabled())
        self.use_map_reasons[self.check_gaps_button].setText('Resolve the findings in the review window first.'
            if not self.check_gaps_button.isEnabled() else 'Run the original fresh preparation and Check.')
        if record is not None and approval is not None:
            name = record.get("display_name") or record.get("name")
            self.map_status.setText(f"Using {name} ({self._approval_text(approval)}), "
                                    f"SHA-256 {record['sha256'][:12]}…. This selected version counts.")
        elif not lpm.is_manual_version(record or {}):
            self.map_status.setText(problem)
        def matches(value):
            version = (value or {}).get("map_version") or {}
            return bool(record and version.get("sha256") == record.get("sha256")
                        and (version.get("folder") == record.get("folder") or
                             (version.get("approval") or {}).get("version") == Path(record["folder"]).name))
        transferred = matches(self.last_placement)
        reviewed = transferred and getattr(self, "_reviewed_transfer", None) == self.last_placement.get("folder")
        prepared = matches(self.last_adoption)
        next_action = (self.place_version_button if not transferred else self.gap_review_button if not reviewed
                       else self.adopt_button)
        for button in self.use_map_reasons:
            primary = button is next_action and button.isEnabled() and not problem
            button.setDefault(primary)
            button.setProperty("seriesMapNext", primary)
            button.setStyleSheet("font-weight: bold; border: 2px solid palette(highlight);" if primary else "")
        self.use_map_compact_reason.setText(problem or (
            self.use_map_reasons[self.adopt_button].text() if not self.adopt_button.isEnabled() else ''))
        if self.project is not None:
            self.series_use_status.refresh(lp.painting_folder(self.project).parent,
                Path(self.project.root) / "overrides/geometry_facies/current.json", record, approval)
        else:
            self.series_use_status.refresh(None, None, record, approval)
        completed = None if problem else self.series_use_status.completed_adoption(record, approval)
        adopting = (self.adoption_controller.context is not None
                    or getattr(self, "_adoption_thread", None) is not None)
        if completed and not adopting:
            adoption_text = "✓ Saved as the active series' sand map."
            self.adoption_status.setStyleSheet("color: #299666; font-weight: bold;")
            self.adoption_status.setToolTip(f"Geometry revision {completed['revision_id']}")
        else:
            adoption_text = ("Adoption in progress…" if adopting else
                "Reading saved adoption status…" if self.series_use_status.busy() else
                "Not yet saved as the active series' sand map.")
            self.adoption_status.setStyleSheet("")
            self.adoption_status.setToolTip("")
        self.adoption_status.setText(adoption_text)
        if hasattr(self, "step_use_map"):
            self.step_use_map.set_status(adoption_text if completed and not adopting else problem or (
                "Ready to use the selected version" if not prepared else "Revision check prepared; review before adoption"),
                "done" if completed and not adopting else "warn" if problem else "")
        self._sync_transfer_preparation()

    @staticmethod
    def _version_gaps(record: Mapping[str, Any]) -> str:
        """Why a saved manual map cannot be approved ("" when it can, or when
        the version is not a manual map)."""

        uncovered = int(record.get("uncovered_pixels") or 0)
        if not (lpm.is_manual_version(record) or record.get("proposal_method") == lpp.METHOD) or not uncovered:
            return ""
        return (
            f"{uncovered:,} px inside the rig frame are covered by no polygon; cover "
            "them, save a new version and approve that"
        )

    def _placeable(self, record: Mapping[str, Any] | None) -> bool:
        """Whether a version may be placed in all runs: it is approved."""

        if self._native_async_required():
            native = self._large_native_result()
            if native is None or native.error:
                return False
        return (
            record is not None
            and record.get("kind") != lpv.DRAWING_KIND
            and not record.get("uncovered_pixels")
            and not self.busy()
            and not self._map_approvals_error
            and not record.get("history_error")
            and self._approval_for_version(record) is not None
            and not self._reference_point_review_blocker()
        )

    def _series_root(self) -> Path | None:
        """The native series folder this project belongs to, or None."""

        from darsia.fluidflower.series_geometry import native_series_link

        root = native_series_link(self.project).get("series_root") if self.project else None
        return Path(str(root)) if root else None

    def _series_runs(self) -> list[str]:
        """The registered runs of the series the geometry belongs to."""

        from darsia.fluidflower.series_geometry import read_series_manifest

        root = self._series_root()
        if root is None:
            return []
        return [str(run) for run in (read_series_manifest(root).get("runs") or {}).get("ids") or []]

    def _capture_review_display_context(self):
        from .painted_review_comparison import capture_display_context
        if not isinstance(getattr(self.project, "manifest", None), dict):
            return None
        from .series_geometry import native_series_link
        if not native_series_link(self.project).get("series_root"):
            return None
        prior = getattr(getattr(self, "adoption_controller", None), "display_context", None) or {}
        return capture_display_context(self.project, declared_order=self._series_runs(),
            image_roles=deepcopy(prior.get("image_roles") or {}),
            limitations=deepcopy(prior.get("limitations") or {}))

    def _reference_photo_loader(self):
        """Load a run's reference photograph, as its geometry names it, on its
        camera grid (uint8 RGB); the file's size and SHA-256 are checked."""

        from darsia.fluidflower.photograph_io import load_reduced_rgb8
        from darsia.fluidflower.project import sha256_file
        from darsia.fluidflower.raw_shape_basis_reader import reviewed_photo_path
        from darsia.fluidflower.rig_point_alignment import model_photo_path

        project = self.project

        def load(run: str, geometry) -> np.ndarray:
            image = dict(geometry.reference_image)
            path = model_photo_path(geometry)
            if path == Path(str(image.get("path") or "")):
                path = reviewed_photo_path(project, path)
            if not path.is_file():
                raise FileNotFoundError(f"{run}: reference photograph not found: {path}")
            if "size_bytes" in image and path.stat().st_size != int(image["size_bytes"]):
                raise ValueError(f"{run}: the reference photograph changed size: {path}")
            if image.get("sha256") and sha256_file(path) != image["sha256"]:
                raise ValueError(f"{run}: the reference photograph changed: {path}")
            camera_u = lp.photo_camera_u(project, run, path)
            return load_reduced_rgb8(path, 1, camera_u=camera_u)

        return load

    def _measured_geometry_clicked(self) -> None:
        """Choose the runs' geometry index and show each run's measured geometry."""

        settings = QSettings(*_BACKGROUND_SETTINGS)
        start = str(settings.value(_GEOMETRY_INDEX_KEY, "") or "")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Measured geometry of the runs (index.json)",
            start,
            "Geometry index (index.json);;JSON (*.json)",
        )
        if not path:
            return
        settings.setValue(_GEOMETRY_INDEX_KEY, path)
        settings.sync()
        self.show_measured_geometry(path)

    def show_measured_geometry(
        self,
        index_path: Path | str,
        *,
        photo_loader: Callable[..., np.ndarray] | None = None,
    ) -> QDialog | None:
        """Open the read-only view of each run's measured geometry; None when
        the index cannot be read (a message says why)."""

        from darsia.fluidflower.measured_geometry_view import MeasuredGeometryDialog

        try:
            dialog = MeasuredGeometryDialog(
                index_path,
                photo_loader=photo_loader or self._reference_photo_loader(),
                parent=self,
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not read the geometry index", str(error))
            return None
        self._geometry_dialog = dialog
        dialog.show()
        return dialog

    def _place_version_clicked(self) -> None:
        """Start the selected map's transfer directly from the visible run list."""

        record = self._selected_version()
        if not self._placeable(record):
            self.versions_note.setText(self._reference_point_review_blocker() or "Approve the version before placing it in all runs.")
            return
        try:
            selection = self.run_preview.geometry_selection()
        except ValueError as error:
            self.versions_note.setText(str(error))
            self.use_map_progress.setText(str(error))
            return
        path = selection.index_path
        try:
            runs = self._series_runs()
        except (OSError, ValueError) as error:
            self.use_map_progress.setText(str(error))
            return
        if not runs:
            self.use_map_progress.setText('No active runs are available for transfer.')
            return
        self.place_version(
            record,
            path,
            runs=runs,
            first_runs=[],
            continue_photo_edges=True,
        )

    def place_version(
        self,
        record: Mapping[str, Any],
        geometry_index: Path | str,
        *,
        runs: Sequence[str] | None = None,
        first_runs: Sequence[str] = (),
        photo_loader: Callable[..., np.ndarray] | None = None,
        continue_photo_edges: bool = False,
    ) -> bool:
        """Start placing an approved version in all runs; False when it cannot
        start (the note says why). ``first_runs`` are reviewed right after
        this run. The review window opens when it is done."""

        self._read_approvals(str(record.get("run_id") or (self.session.run_id if self.session else "")))
        if not self._placeable(record) or self.session is None:
            self.versions_note.setText(self._reference_point_review_blocker() or "Approve the version before placing it in all runs.")
            return False
        from copy import deepcopy

        from .painting_frame_models import capture_frame_model_selection

        # The normal button uses the same immutable choice as preview and boxes.
        # Programmatic callers may still supply an explicit override.
        try:
            selected = self.run_preview.geometry_selection()
        except ValueError:
            selected = None
        if selected is None or Path(selected.index_path) != Path(geometry_index):
            try:
                selected = capture_frame_model_selection(self.project, override_path=str(geometry_index))
            except ValueError as error:
                self.versions_note.setText(str(error))
                self.use_map_progress.setText(str(error))
                return False
        approval = self._approval_for_version(record)
        try:
            display_context = self._capture_review_display_context() if continue_photo_edges else None
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.versions_note.setText(str(error))
            return False
        thread = QThread(self)
        worker = _PlacementWorker(
            record,
            # A copy: the frame stays editable while the map is placed.
            deepcopy(self.current_frame()),
            {
                "destination": lp.painting_folder(self.project).parent / "map_placement",
                "geometry_index": str(geometry_index),
                "geometry_selection": selected,
                "runs": list(runs if runs is not None else self._series_runs()),
                "photo_loader": photo_loader or self._reference_photo_loader(),
                "approval": approval,
                "project": self.project,
                "reference_point_review": load_point_review(self.project),
                "first_runs": list(first_runs),
                **({"display_context": deepcopy(display_context)} if display_context is not None else {}),
                **({"continue_photo_edges": True, "continuation_project": deepcopy(self.project)}
                   if continue_photo_edges else {}),
            },
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._placement_progress)
        worker.run_progress.connect(self._placement_run_progress)
        worker.finished.connect(self._placement_finished)
        worker.failed.connect(self._placement_failed)
        controller = getattr(self, "adoption_controller", None)
        if hasattr(worker, "gap_review_ready") and controller is not None:
            worker.gap_review_ready.connect(controller.transfer_review_ready,
                                           Qt.ConnectionType.QueuedConnection)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._placement_thread_finished)
        self._placement_thread, self._placement_worker = thread, worker
        self._start_placement_progress(len(worker.options['runs']))
        thread.start()
        self._refresh_versions()
        self.busy_changed.emit(True)
        return True

    def _placement_finished(self, result: object) -> None:
        """Keep the current transfer and its approval in the same photo area."""

        self._placement_timer.stop()
        self._placement_started = None
        self.use_map_progress_bar.setRange(0, 1)
        self.use_map_progress_bar.setValue(1)
        self.use_map_progress_bar.setFormat('Processing finished')
        self.last_placement = dict(result)
        if self.last_placement.get('gap_review') is not None:
            controller = getattr(self, 'adoption_controller', None)
            if controller is not None:
                controller.transfer_review_ready(self.last_placement['gap_review'])
        placed = sum(1 for row in self.last_placement["runs"] if row.get("status") == "placed")
        self.use_map_progress.setText(
            f"Placed in {placed} of {len(self.last_placement['runs'])} runs: "
            f"{self.last_placement['folder']}"
        )
        if self.last_placement.get('gap_review') is None:
            from darsia.fluidflower.map_placement_view import MapPlacementDialog

            dialog = MapPlacementDialog(self.last_placement, self)
            self._placement_dialog = dialog
            dialog.show()
        self._reviewed_transfer = self.last_placement["folder"]
        self.series_use_status.invalidate()
        self._refresh_use_map()

    def _open_placement_clicked(self) -> None:
        """List the project's saved transfers; no path picker is involved."""
        existing = self._saved_transfers_dialog
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        if self.busy() or self.project is None:
            self.versions_note.setText("Wait for the current operation before opening the transfer list.")
            return
        from .painting_saved_transfers_ui import SavedTransferChooser
        preferred = self._selected_version()
        if self._approval_for_version(preferred) is None:
            preferred = next((version for row in reversed(self.map_approvals) for version in self.versions
                if self._approval_for_version(version) == row), None)
        project_root = Path(self.project.root)
        root = lp.painting_folder(self.project).parent / "map_placement"
        dialog = SavedTransferChooser(root, preferred, self)
        def chosen(folder):
            if self.project is None or Path(self.project.root) != project_root:
                self.versions_note.setText("The project changed; open its current transfer list again.")
                return
            self.open_placement(folder)
        dialog.transfer_selected.connect(chosen)
        def loading_changed(_loading):
            self._refresh_versions()
            self.busy_changed.emit(self.busy())
        dialog.loading_changed.connect(loading_changed)
        self._saved_transfers_dialog = dialog
        dialog.show()
        self._refresh_versions()
        self.busy_changed.emit(self.busy())

    def open_placement(self, folder: Path | str) -> QDialog | None:
        """Open the review window of a placement folder; None when it cannot be
        read (a message says why)."""

        from darsia.fluidflower import map_placement as mp
        from darsia.fluidflower.map_placement_view import MapPlacementDialog

        try:
            placement = mp.read_placement(folder)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not open the placement review", str(error))
            return None
        self.last_placement = dict(placement)
        dialog = MapPlacementDialog(placement, self)
        self._placement_dialog = dialog
        dialog.show()
        self._reviewed_transfer = placement["folder"]
        self._refresh_use_map()
        return dialog

    def _current_label_fields(self):
        """The label grid the adopted maps take, for every run: the imported
        parent's labels of the current geometry revision (their DarSIA
        metadata, shape and file binding), as the store's registered runs
        keep it; None when the revision has no imported parent. The files
        are read on first use, in the worker that prepares the adoption."""

        from darsia.fluidflower.geometry_facies import (
            GeometryRevisionStore,
            _darsia_metadata,
            _load_array,
        )
        from darsia.fluidflower.geometry_raster_sources import descriptor

        project = self.project
        cache: dict[str, Any] = {}

        def lookup(run: str) -> dict[str, Any] | None:
            if "grid" not in cache:
                try:
                    config = GeometryRevisionStore(project).current().get("configuration") or {}
                    parent = (config.get("canonical_import_parent") or {}).get(
                        "imported_configuration"
                    ) or {}
                    text = str(parent.get("labels_path") or "")
                    cache["grid"] = (
                        {
                            "metadata": _darsia_metadata(Path(text)),
                            "shape_hw": list(_load_array(Path(text)).shape),
                            "source": descriptor(text),
                        }
                        if text and Path(text).is_file()
                        else None
                    )
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    cache["grid"] = None
            return cache["grid"]

        return lookup

    def _adopt_clicked(self) -> None:
        """Use one automatic check/adopt route and its latest literal preparation.

        The controller uses the existing preparation, checks and final human
        question. The manual facies-ID dialog is not a fallback for this button.
        """
        record = self._selected_version()
        if self._series_adopt_action is None:
            self.use_map_progress.setText("The rigs need their RAW basis and the automatic adoption backend before the map can be adopted.")
            return
        if not self._placeable(record):
            self.use_map_progress.setText("Choose and approve this saved version under Versions before adoption.")
            return
        try:
            selection = self.run_preview.geometry_selection()
        except ValueError as error:
            self.use_map_progress.setText(str(error))
            return
        from .painting_series_use_ui import matches_version
        prepared = self.series_use_status.result.get("preparation")
        if not matches_version(prepared, record):
            prepared = None
        # The actual controller retains the existing dry-run checks and final
        # human confirmation; neither metadata display nor this seam adopts.
        self.use_map_progress.setText("Preparing the selected map for adoption…")
        self._series_adopt_action(record, selection, prepared)
        self._refresh_use_map()

    def _open_gap_review_clicked(self) -> None:
        """Resolve this selected version's known review without a file picker."""
        self.adoption_controller.open_latest_gap_review()

    def _open_recorded_review_clicked(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self, "View recorded decisions", "", "Recorded review (*.json)")
        if source:
            self.adoption_controller.open_recorded_review(source)

    def _open_other_gap_review_clicked(self) -> None:
        """Choose an external review explicitly; original admission still applies."""
        source, _ = QFileDialog.getOpenFileName(self, "Open other bound gap review", "", "Gap review (*.json)")
        if source:
            self.open_gap_review(source)

    def _recheck_gap_review_clicked(self) -> None:
        """Delegate a human click to the existing review's guarded Check action."""
        dialog = getattr(self, '_gap_review_dialog', None)
        if dialog is not None and dialog.recheck.isEnabled():
            dialog.recheck.click()
        else:
            self.use_map_progress.setText('Open the findings and resolve the required choices before checking again.')

    def register_gap_review(self, binding) -> None:
        """Register only a transient technical descriptor for later human review."""
        self.adoption_controller.register_gap_review(binding)

    def open_gap_review(self, source) -> QDialog | None:
        """Review a typed failure or Root-prepared retained manifest read only."""
        return self.adoption_controller.open_gap_review(source)

    def open_adoption(self, folder: Path | str) -> QDialog | None:
        """Open the adoption window of a preparation folder; None when it
        cannot be read (a message says why)."""

        if self.session is not None:
            self._read_approvals(self.session.run_id)
        if self._native_async_required():
            native = self._large_native_result()
            if native is None or native.error or self._native_async.busy():
                self.versions_note.setText("Wait for the exact native map before opening adoption.")
                return None
        if self._map_approvals_error:
            self.versions_note.setText("Version history is unreadable; adoption remains blocked.")
            return None
        from darsia.fluidflower.map_adoption_view import AdoptionDialog
        from darsia.fluidflower.series_geometry import default_actor

        try:
            dialog = AdoptionDialog(
                self.project, folder, actor=self._reviewer() or default_actor(), parent=self
            )
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Could not open the adoption", str(error))
            return None
        self._adoption_dialog = dialog
        self.show_adoption_panel(dialog)
        return dialog

    def _prepare_adoption_clicked(self) -> None:
        """Retain old programmatic entry points as the same single GUI action."""
        self._adopt_clicked()

    def prepare_adoption_of(
        self,
        record: Mapping[str, Any],
        geometry_index: Path | str,
        *,
        runs: Sequence[str] | None = None,
        current_labels: Callable[[str], Mapping[str, Any] | None] | None = None,
        continue_photo_edges: bool = False,
        analysis_exclusions: Mapping[str, Any] | None = None,
        target_corrections: Sequence[Mapping[str, Any]] | None = None,
        local_reuse: Mapping[str, Any] | None = None,
    ) -> bool:
        """Start preparing an approved version's adoption on every run's label
        grid; False when it cannot start (the note says why)."""

        self._read_approvals(str(record.get("run_id") or (self.session.run_id if self.session else "")))
        if not self._placeable(record) or self.session is None:
            self.versions_note.setText(self._reference_point_review_blocker() or "Approve the version before preparing its adoption.")
            return False
        from copy import deepcopy

        continuation_options = {}
        if local_reuse is not None:
            continuation_options["local_reuse"] = deepcopy(local_reuse)
        if continue_photo_edges:
            from .painting_frame_models import capture_frame_model_selection
            try:
                selection = capture_frame_model_selection(self.project, override_path=str(geometry_index))
            except ValueError as error:
                self.versions_note.setText(str(error))
                return False
            continuation_options.update(continue_photo_edges=True,
                continuation_project=deepcopy(self.project), geometry_selection=selection)

        if analysis_exclusions is not None:
            continuation_options["analysis_exclusions"] = deepcopy(analysis_exclusions)
        if target_corrections is not None:
            continuation_options["target_corrections"] = deepcopy(target_corrections)

        try:
            display_context = self._capture_review_display_context() if continue_photo_edges else None
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.versions_note.setText(str(error))
            return False
        if display_context is not None:
            continuation_options["display_context"] = deepcopy(display_context)

        thread = QThread(self)
        worker = _AdoptionWorker(
            record,
            # A copy: the frame stays editable while the adoption is prepared.
            deepcopy(self.current_frame()),
            {
                "destination": lp.painting_folder(self.project).parent / "map_adoption",
                "geometry_index": str(geometry_index),
                "runs": list(runs if runs is not None else self._series_runs()),
                "approval": self._approval_for_version(record),
                "current_labels": current_labels or self._current_label_fields(),
                "project": self.project,
                "reference_point_review": load_point_review(self.project),
                **continuation_options,
            },
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.use_map_progress.setText)
        worker.finished.connect(self._adoption_finished, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._adoption_failed, Qt.ConnectionType.QueuedConnection)
        controller = getattr(self, "adoption_controller", None)
        if hasattr(worker, "gap_review_ready") and controller is not None:
            worker.gap_review_ready.connect(controller.gap_review_ready,
                                           Qt.ConnectionType.QueuedConnection)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._adoption_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._adoption_thread, self._adoption_worker = thread, worker
        self.use_map_progress.setText("Checking the selected map for adoption…")
        thread.start()
        self._refresh_versions()
        self.busy_changed.emit(True)
        return True

    @Slot(object)
    def _adoption_finished(self, result: object) -> None:
        """Say what the preparation holds and what an adoption still lacks."""

        self.last_adoption = dict(result)
        rows = self.last_adoption["runs"]
        prepared = sum(1 for row in rows if row.get("status") == "prepared")
        text = (
            f"Prepared {prepared} of {len(rows)} runs on their rig-corrected label grid. "
            "Nothing is adopted."
        )
        blockers = list(self.last_adoption.get("blockers") or [])
        detail = text + (
            "\n\nStill missing:\n" + "\n".join(f"- {line}" for line in blockers)
            if blockers
            else ""
        ) + f"\n\nFolder: {self.last_adoption['folder']}"
        self.use_map_progress.setText(f"{text} {self.last_adoption['folder']}")
        self.series_use_status.invalidate()
        controller = getattr(self, "adoption_controller", None)
        if controller is not None and controller.prepared(self.last_adoption):
            return
        message = QMessageBox(self)
        message.setWindowTitle("Adoption prepared")
        message.setText(detail)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._adoption_message = message
        message.show()

    @Slot(str)
    def _adoption_failed(self, message: str) -> None:
        """Say why the adoption was not prepared."""

        self.use_map_progress.setText(f"The adoption was not prepared: {message}")
        controller = getattr(self, "adoption_controller", None)
        if controller is not None:
            controller.failed(message)

    @Slot()
    def _adoption_thread_finished(self) -> None:
        """Drop the finished worker and its thread here, in the GUI thread."""

        thread, worker = self._adoption_thread, self._adoption_worker
        if thread is not None:
            thread.wait()
        self._adoption_thread = None
        self._adoption_worker = None
        if thread is not None:
            thread.deleteLater()
        self._refresh_versions()
        self.busy_changed.emit(self.busy())
        controller = getattr(self, "adoption_controller", None)
        if controller is not None:
            controller.preparation_retired()

    def _placement_failed(self, message: str) -> None:
        """Say why the map was not placed."""

        self._placement_timer.stop()
        self._placement_started = None
        self.use_map_progress_bar.hide()
        self.use_map_progress.setText(f"The map was not transferred to all runs: {message}")

    def _start_placement_progress(self, total):
        self._placement_started = time.monotonic()
        self._placement_progress_text = f'Preparing transfer of {total} runs…'
        self._placement_stage = None
        self.use_map_progress_bar.setRange(0, 0)
        self.use_map_progress_bar.show()
        self.use_map_elapsed.show()
        self._placement_timer.start()
        self._update_placement_progress()

    @Slot(str)
    def _placement_progress(self, message):
        self._placement_progress_text = message
        self._update_placement_progress()

    @Slot(object)
    def _placement_run_progress(self, progress):
        self._placement_stage = dict(progress)
        total, completed = progress['total'], progress['completed']
        self.use_map_progress_bar.setRange(0, total)
        self.use_map_progress_bar.setValue(completed)
        self.use_map_progress_bar.setFormat(f'{completed} of {total} in this stage')
        labels = {'raw_basis': 'Checking source data',
                  'frame_models_camera': 'Preparing map placement',
                  'frame_models_native': 'Preparing review geometry',
                  'transfer': 'Transferring maps', 'native_review': 'Preparing run review'}
        phase = labels.get(progress['phase'], progress['phase'])
        self._placement_progress_text = f"{phase}: {completed} of {total} complete · {progress['run_id'].upper()}"
        self._update_placement_progress()

    def _update_placement_progress(self):
        self.use_map_progress.setText(self._placement_progress_text)
        if self._placement_started is not None:
            elapsed = max(0, int(time.monotonic() - self._placement_started))
            self.use_map_elapsed.setText(f'Elapsed: {elapsed // 60}:{elapsed % 60:02d}')

    def _placement_thread_finished(self) -> None:
        """Drop the finished worker and its thread here, in the GUI thread."""

        self._placement_timer.stop()
        self._placement_started = None
        thread, worker = self._placement_thread, self._placement_worker
        if thread is not None:
            thread.wait()
        self._placement_thread = None
        self._placement_worker = None
        if thread is not None:
            thread.deleteLater()
        self._refresh_versions()
        self.busy_changed.emit(self.busy())
        controller = getattr(self, "adoption_controller", None)
        if controller is not None:
            controller.placement_retired()

    def _version_double_clicked(self, _item: QTableWidgetItem) -> None:
        self._restore_version_clicked()

    def _save_version_clicked(self) -> None:
        problem = self._versions_unavailable()
        if problem:
            self.versions_note.setText(problem)
            return
        name, accepted = QInputDialog.getText(
            self,
            "Save version",
            "Name of this version (left empty: the kind of map and the time):",
            text=self.default_version_name(),
        )
        if accepted:
            self.save_version(name)

    def default_version_name(self) -> str:
        """The name a version gets when none is given: "Proposal", "Edited map"
        or "Manual map" with the local date and time."""

        kind = "Edited map" if self.corrections else "Proposal"
        if self._map_kind() == _MANUAL:
            kind = "Manual map"
        return f"{kind} {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    def save_version(self, name: str = "") -> dict[str, Any] | None:
        """Save the map shown as a version: the proposal with its corrections,
        or the manual map with the session.

        An empty name gives the default name. Returns the version's record, or
        None (the note says why).
        """

        problem = self._versions_unavailable()
        if problem:
            self.versions_note.setText(problem)
            return None
        assert self.session is not None
        name = str(name).strip() or self.default_version_name()
        self.save_session()
        try:
            if self._map_kind() == _MANUAL:
                assert self._manual is not None
                record = lpm.save_manual_version(
                    self.project, self.session, self._manual, name=name
                )
            else:
                assert self.proposal is not None
                method = lpp.METHOD if lpp.is_polygons_only(self.proposal) else lp.PROPOSAL_METHOD
                record = lpv.save_map_version(
                    self.project, self.session, self.proposal, self.corrections, name=name,
                    method=method, settings=(dict(method=lpp.METHOD, grid="native photograph",
                        automatic_edge_fill=False, strokes_used=False)
                        if method == lpp.METHOD else self.session.to_dict()["proposal_settings"]),
                    proposal_writer=lpp.save_proposal if method == lpp.METHOD else None,
                )
        except (OSError, ValueError) as error:
            self.versions_note.setText(f"The version could not be saved: {error}")
            return None
        self.versions = self._read_versions(self.session.run_id)
        self._refresh_versions()
        text = f"Saved {record['name']} ({record['kind']}): SHA-256 {record['sha256']}"
        gaps = self._version_gaps(record)
        self.versions_note.setText(text + (f" · {gaps}" if gaps else ""))
        return record

    def _reviewer(self) -> str:
        """The series' reviewer setting ("" when none is set or it cannot be read)."""

        from darsia.fluidflower.reviewer_preferences import load_reviewer_name

        try:
            return load_reviewer_name(getattr(self.project, "root", ""))
        except (OSError, ValueError):
            return ""

    def _approve_version_clicked(self) -> None:
        record = self._selected_version()
        if record is None:
            return
        gaps = self._version_gaps(record)
        if gaps:
            self.versions_note.setText(f"{record.get('name')} cannot be approved: {gaps}.")
            return
        reviewer = self._reviewer()
        who = (
            f"The reviewer recorded is {reviewer} (the series' reviewer setting)."
            if reviewer
            else "No reviewer is set for this series, so the record names nobody."
        )
        run = record.get("run_id") or "this run"
        answer = QMessageBox.question(
            self,
            "Approve version",
            f"Approve {record.get('name')} ({record.get('kind')}) as the map of "
            f"{run}? The approval records its SHA-256 {record.get('sha256')} and "
            f"the time. {who} It changes no map, proposal or geometry revision.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.approve_version(record)

    def approve_version(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Approve a saved version as the run's map, with the series' reviewer
        setting. Returns the approval record, or None (the note says why)."""

        self._read_approvals(str(record.get("run_id") or (self.session.run_id if self.session else "")))
        if self._native_async_required():
            native = self._large_native_result()
            if native is None or native.error or self._native_async.busy():
                self.versions_note.setText("Computing the exact native map; approval remains blocked.")
                return None
        if self.busy() or self._map_approvals_error or record.get("history_error"):
            self.versions_note.setText(
                self._map_approvals_error or str(record.get("history_error") or "")
                or "Wait until the proposal or export has finished."
            )
            return None
        try:
            reviewer = self._reviewer() or None
            if record.get("kind") == lpv.DRAWING_KIND:
                raise ValueError("A drawing has no map to approve; propose or save a manual map first")
            if record.get("proposal_method") == lpp.METHOD:
                approval = lpp.approve_version(record, reviewer=reviewer)
            elif lpm.is_manual_version(record):
                approval = lpm.approve_manual_version(record, reviewer=reviewer)
            else:
                approval = lp.approve_version(record, reviewer=reviewer)
        except (OSError, ValueError, KeyError) as error:
            self.versions_note.setText(f"The version could not be approved: {error}")
            return None
        self._read_approvals(str(record.get("run_id") or ""))
        self._refresh_versions()
        for row in range(self.versions_table.rowCount()):
            if self.versions_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == record.get("folder"):
                self.versions_table.selectRow(row)
                break
        self.step_propose.set_expanded(False)
        self.step_use_map.set_expanded(True)
        self._refresh_use_map()
        self.versions_note.setText(
            f"{approval['name']}: SHA-256 {approval['sha256']} · "
            f"{self._approval_text(approval)}"
        )
        return approval

    def _restore_version_clicked(self) -> None:
        record = self._selected_version()
        if record is None:
            return
        if record.get("kind") == lpv.DRAWING_KIND:
            self.restore_version(record)
            return
        saved = str(record.get("created_utc") or "")[:16].replace("T", " ")
        keep = (
            "The current map is saved as a version first, so that it can be "
            "restored again."
            if self.proposal is not None or self._map_kind() == _MANUAL
            else "The current drawing is saved as a version first."
        )
        if lpm.is_manual_version(record):
            keep += (
                " A manual map brings back its polygons, layers and facies; the "
                "proposal and its corrections stay as they are."
            )
        answer = QMessageBox.question(
            self,
            "Restore version",
            f"Restore {record.get('name')} ({record.get('kind')}, saved {saved} UTC)? "
            + keep,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.restore_version(record)

    def restore_version(self, record: Mapping[str, Any]) -> bool:
        """Make a saved version the current map.

        The current map is saved as a version first unless an identical one
        is already saved. The version itself is never changed.
        """

        if self.busy() or self.session is None:
            return False
        if self.versions_reading():
            # Whether the current map is already saved is known once the versions are read.
            self.versions_note.setText(_VERSIONS_READING)
            return False
        if record.get("kind") == lpv.DRAWING_KIND:
            restored = self.version_actions.use_drawing(record)
            if restored:
                self.run_preview.show_working_map()
            return restored
        restored_new = None
        if record.get("proposal_method") == lpp.METHOD:
            try:
                from .painting_run_preview import _same_path
                restored_new = lpp.load_version(record)
                loaded, painted, _corrections = restored_new
                if (loaded.run_id != self.session.run_id
                        or tuple(loaded.image_size or ()) != tuple(self.session.image_size or ())
                        or not _same_path(loaded.image_path, self.session.image_path)
                        or loaded.image_stamp != self.session.image_stamp
                        or not lpp.proposal_current(painted, loaded, self.current_frame())):
                    raise ValueError("This version belongs to another photograph or full rig frame; no draft was changed")
            except (OSError, ValueError, KeyError) as error:
                self.versions_note.setText(f"The version could not be restored: {error}")
                return False
        if self._map_kind() == _MANUAL:
            # The manual map shown is kept as a version first.
            problem = self._versions_unavailable()
            if problem:
                self.versions_note.setText(problem)
                return False
            assert self._manual is not None
            current = lpm.manual_map_digest(self.session, self._manual)
            if not any(row.get("sha256") == current for row in self.versions):
                if self.save_version(f"Before restoring {record.get('name')}") is None:
                    return False
        elif self.proposal is not None and not self._corrections_error:
            if lpp.is_polygons_only(self.proposal) and not self.proposal_current():
                try:
                    lpv.save_drawing_version(self.project, self.session,
                        name=f"Drawing before restoring {record.get('name')}")
                except (OSError, ValueError) as error:
                    self.versions_note.setText(f"The exact current drawing could not be kept: {error}")
                    return False
            else:
                current = self._current_map_digest()
                if not any(row.get("sha256") == current for row in self.versions):
                    if self.save_version(f"Before restoring {record.get('name')}") is None:
                        return False
        else:
            try:
                lpv.save_drawing_version(self.project, self.session,
                    name=f"Drawing before restoring {record.get('name')}")
            except (OSError, ValueError) as error:
                self.versions_note.setText(f"The current drawing could not be kept: {error}")
                return False
        self.save_session()
        try:
            if restored_new is not None:
                loaded, painted, corrections = restored_new
                lpp.save_current_session(self.project, loaded, actor=default_actor())
                painted.save(lp.proposal_path(self.project, loaded.run_id))
                if corrections or lp.corrections_path(self.project, loaded.run_id).exists():
                    lp.save_corrections(self.project, loaded.run_id, corrections)
                self.session = loaded
                self._dirty = self._corrections_dirty = False
            elif lpm.is_manual_version(record):
                lpm.restore_manual_version(self.project, record, actor=default_actor())
            else:
                lp.restore_version(self.project, record, actor=default_actor())
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Could not restore the version", str(error))
            return False
        self.reload()
        self.run_preview.show_working_map()
        self.versions_note.setText(
            f"Restored {record.get('name')}: SHA-256 {record.get('sha256')}"
        )
        return True

    def _after_mark_change(self, *, render: bool = True) -> None:
        if render:
            self._render_marks()
        self._refresh_layers()
        self._mark_dirty()
        self._refresh_status()

    def _render_polygons(self) -> None:
        if self.session is None:
            self.canvas.render_polygons([], {})
            return
        self.canvas.render_polygons(self.session.polygons, self._layer_colors(),
                                    always_on_top=layer_priority.layer_ids(self.session))

    def _render_marks(self) -> None:
        if self.session is None:
            self.canvas.render_strokes([], {})
            self.canvas.render_polygons([], {})
            return
        colors = self._layer_colors()
        self.canvas.render_strokes(self.session.strokes, colors)
        self.canvas.render_polygons(self.session.polygons, colors,
                                    always_on_top=layer_priority.layer_ids(self.session))

    def _render_strokes(self) -> None:
        """Former name of :meth:`_render_marks`."""

        self._render_marks()

    # ---------------------------------------------------------- proposal

    def _sync_method_controls(self) -> None:
        classifier = self.proposal_method.currentData() != lpp.METHOD
        available = self.session is not None
        for widget in (self.smoothing, self.min_region, self.snap, self.detail,
                       self.move_shared_edges):
            widget.setEnabled(available and classifier)

    def _proposal_method_changed(self, *_args: Any) -> None:
        self._sync_method_controls()
        if not self._loading:
            self._refresh_status()

    def _settings_changed(self, *_args: Any) -> None:
        if self.session is None or self._loading:
            return
        self.session.smoothing = str(self.smoothing.currentData() or "medium")
        self.session.min_region_cm2 = float(self.min_region.value())
        self.session.snap_mm = float(self.snap.value())
        self.session.detail = str(self.detail.currentData() or "normal")
        self.session.move_shared_edges = (
            self.move_shared_edges.currentData() is not False
        )
        self._mark_dirty()
        self._refresh_status()

    def proposal_current(self) -> bool:
        return (
            self.proposal is not None
            and self.session is not None
            and lpp.proposal_current(self.proposal, self.session, self.current_frame())
        )

    def propose(self) -> None:
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            self.propose_status.setText("Saved corrected rig maps are shown. The reference labels remain editable in Advanced; production registration is a separate approved step.")
            return
        if self.session is None or self.image is None or self.busy():
            return
        reference = self.reference()
        if reference is None:
            return
        if self._other_photo is not None:
            self.show_reference_photo()
        self.save_session()
        method = str(self.proposal_method.currentData() or lp.PROPOSAL_METHOD)
        if method == lpp.METHOD:
            try:
                lpp.save_current_session(self.project, self.session, actor=default_actor())
            except (OSError, ValueError) as error:
                self.propose_status.setText(f"The exact drawing could not be saved: {error}")
                return
        snapshot = (deepcopy(self.session) if method == lpp.METHOD
                    else lp.PaintingSession.from_dict(self.session.to_dict()))
        thread = QThread(self)
        worker = _ProposalWorker(
            self._proposal_image(),
            snapshot,
            deepcopy(self.current_frame()) if method == lpp.METHOD else self.current_frame(),
            lp.proposal_path(self.project, reference.run_id),
            method=method,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.propose_status.setText)
        worker.finished.connect(self._proposal_worker_finished)
        worker.failed.connect(self._proposal_worker_failed)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._proposal_thread_finished)
        self._proposal_thread = thread
        self._proposal_worker = worker
        self.propose_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.propose_busy.show()
        self.propose_status.setText("Starting…")
        self.step_propose.set_status("Working…", "warn")
        thread.start()
        self.busy_changed.emit(True)

    def _proposal_image(self) -> lp.PaintingImage:
        """The reference photograph on the grid the session's detail asks for.

        Fine uses the displayed photograph as the analysis grid. That image is
        kept while the photograph stays, so its features are computed once.
        """

        assert self.image is not None and self.session is not None
        if self.session.detail != "fine":
            return self.image
        if self._fine_image is None or self._fine_image[0] is not self.image:
            fine = lp.with_analysis_width(self.image, int(self.image.display.shape[1]))
            self._fine_image = (self.image, fine)
        return self._fine_image[1]

    @Slot(object)
    def _proposal_worker_finished(self, proposal: lp.LayerProposal) -> None:
        """Hand a finished proposal to _proposal_ready in this view's thread.

        The worker's signal reaches a slot of the view, so Qt queues it into
        the GUI thread. _proposal_ready may be replaced by an extension's
        plain function, which Qt would call in the worker's thread, so the
        worker never connects to it directly.
        """

        self._proposal_ready(proposal)

    @Slot(str)
    def _proposal_worker_failed(self, message: str) -> None:
        """Hand a failed proposal's message to _proposal_failed in this view's thread."""

        self._proposal_failed(message)

    @Slot(object)
    def _proposal_ready(self, proposal: lp.LayerProposal) -> None:
        worker = self._proposal_worker
        if (
            worker is not None
            and self.session is not None
            and worker.session.run_id != self.session.run_id
        ):
            return  # the reference run changed while the model worked
        self.proposal = proposal
        self.propose_status.setText(f"Proposed in {proposal.elapsed_s:.1f} s.")
        if not self.show_proposal.isChecked():
            self.show_proposal.setChecked(True)
        self._render_proposal()
        self._refresh_facies_table()
        self.step_propose.set_expanded(True)
        self._schedule_manual_map()

    @Slot(str)
    def _proposal_failed(self, message: str) -> None:
        worker = self._proposal_worker
        if (
            worker is not None
            and self.session is not None
            and worker.session.run_id != self.session.run_id
        ):
            return
        self.propose_status.setText(message)
        QMessageBox.information(self, "No proposal", message)

    def _proposal_thread_finished(self) -> None:
        thread = self._proposal_thread
        if thread is not None:
            thread.wait()
        self._proposal_thread = None
        self._proposal_worker = None
        if thread is not None:
            thread.deleteLater()
        self.propose_busy.hide()
        self.propose_button.setEnabled(self.session is not None)
        self._refresh_status()
        self.busy_changed.emit(self.busy())

    def _layer_overlay(
        self, classes: np.ndarray, *, proposal_boundaries: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """The colours of a layer map on the display grid, and its pixels of
        shown layers.

        Each layer in its colour at the chosen opacity, the area outside the
        rig frame dark and hidden layers transparent. The proposal renderer
        can add white boundaries where a shown layer meets another; the manual
        map uses the drawn polygon outlines instead (display only).
        """

        alpha = int(round(2.55 * self.opacity.value()))
        table = np.zeros((256, 4), dtype=np.uint8)
        table[lp.OUTSIDE] = (0, 0, 0, 110)
        assert self.session is not None
        colors = self._layer_colors()
        for layer in self.session.layers:
            color = colors[layer.layer_id]
            table[layer.layer_id] = (color.red(), color.green(), color.blue(), alpha)
        # Hidden layers are fully transparent (display only).
        hidden = np.zeros(256, dtype=bool)
        for layer_id in self.hidden_layers:
            if 0 < layer_id < 256:
                hidden[layer_id] = True
        table[hidden] = 0
        rgba = table[classes]
        shown = ~hidden[classes]
        if not proposal_boundaries:
            return rgba, shown
        boundary_classes = classes
        if self._color_mode == "facies" or self._highlight_facies is not None:
            groups = np.arange(256, dtype=np.int16) + 256
            names = {name: number for number, name in enumerate(self.session.facies_order())}
            for layer in self.session.layers:
                groups[layer.layer_id] = names[layer.facies]
            boundary_classes = groups[classes]
        # A boundary pixel is drawn where it touches a shown layer (or the area
        # outside the rig frame); between hidden layers there is none.
        boundary = np.zeros(classes.shape, dtype=bool)
        boundary[:, 1:] |= ((boundary_classes[:, 1:] != boundary_classes[:, :-1])
                            | (shown[:, 1:] != shown[:, :-1])) & (
            shown[:, 1:] | shown[:, :-1]
        )
        boundary[1:, :] |= ((boundary_classes[1:, :] != boundary_classes[:-1, :])
                            | (shown[1:, :] != shown[:-1, :])) & (
            shown[1:, :] | shown[:-1, :]
        )
        rgba[boundary] = (255, 255, 255, 235)
        return rgba, shown

    def _render_proposal(self) -> None:
        manual_shown = self._map_kind() == _MANUAL
        self.show_proposal_boundaries.setEnabled(
            not manual_shown and self.proposal is not None and self._other_photo is None
        )
        if manual_shown:
            self._render_manual_map()
            return
        if (workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
                and self._other_photo is None and self.proposal is None):
            self.canvas.set_proposal(self._saved_corrected_boundaries)
            return
        if (
            self.proposal is None
            or self.image is None
            or self.session is None
            or self._other_photo is not None
            or tuple(self.proposal.image_size) != tuple(self.image.image_size)
        ):
            self.canvas.set_proposal(None)
            return
        import cv2

        height, width = self.image.display.shape[:2]
        if lpp.is_polygons_only(self.proposal):
            if self._native_async_required():
                native = self._large_native_result()
                if native is None or native.error:
                    self.canvas.set_proposal(None)
                    self._update_polygon_fill()
                    return
                visible = self.show_corrections.isChecked()
                classes = native.display_classes if visible else native.base_display_classes
                gaps = native.display_gaps if visible else native.base_display_gaps
                corrected = native.display_decided if visible else None
            else:
                native, decided, _skipped = self._corrected_native_state(shown=True)
                display_key = self._native_corrections_key, height, width
                if self._native_display_key != display_key:
                    classes = cv2.resize(native, (width, height), interpolation=cv2.INTER_NEAREST)
                    gaps = cv2.resize(self._native_corrections_gap_mask.astype(np.uint8),
                                      (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    corrected = cv2.resize(decided.astype(np.uint8), (width, height),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    self._native_display_key = display_key
                    self._native_display_state = classes, gaps, corrected
                classes, gaps, corrected = self._native_display_state
            rgba, _shown = self._layer_overlay(
                classes, proposal_boundaries=self.show_proposal_boundaries.isChecked()
            )
            rgba[gaps] = (255, 0, 255, 150)
            if corrected is not None and self.corrections and self.show_corrections.isChecked():
                inner = cv2.erode(corrected.astype(np.uint8), np.ones((3, 3), np.uint8))
                rgba[corrected & ~inner.astype(bool)] = _CORRECTED_EDGE
            self._mark_differences(rgba)
            self.canvas.set_proposal(rgba)
            self._update_polygon_fill()
            return
        classes = cv2.resize(
            self.proposal.classes, (width, height), interpolation=cv2.INTER_NEAREST
        )
        corrected = None
        if self.corrections and self.show_corrections.isChecked():
            layer_ids = frozenset(self._layer_ids())
            classes, corrected = self._display_corrections.get(
                (id(self.proposal), self.proposal.digest, classes.shape, layer_ids),
                classes,
                self.corrections,
                (height / self.image.image_size[1], width / self.image.image_size[0]),
                layer_ids,
            )
        rgba, shown = self._layer_overlay(
            classes, proposal_boundaries=self.show_proposal_boundaries.isChecked()
        )
        if self.show_uncertain.isChecked():
            uncertain = cv2.resize(
                self.proposal.uncertain().astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            rows, cols = np.indices(classes.shape)
            hatch = ((rows + cols) // 3) % 2 == 0
            if corrected is not None:
                uncertain &= ~corrected  # A correction decided these pixels.
            rgba[uncertain & hatch & shown] = (255, 0, 255, 200)
        if corrected is not None and np.any(corrected):
            inner = cv2.erode(corrected.astype(np.uint8), np.ones((3, 3), np.uint8))
            rgba[corrected & ~inner.astype(bool)] = _CORRECTED_EDGE
        self._mark_differences(rgba)
        self.canvas.set_proposal(rgba)
        self._update_polygon_fill()

    def _display_changed(self, *_args: Any) -> None:
        reference_shown = self._other_photo is None and self._workflow_scope != 'rig_frame'
        self.canvas.set_overlays_visible(
            self.show_strokes.isChecked() and reference_shown,
            self.show_proposal.isChecked() and reference_shown,
        )
        self._update_polygon_fill()
        sender = self.sender()
        if sender in (self.opacity, self.show_uncertain, self.show_proposal_boundaries):
            self._render_proposal()
        self._sync_analysis_boxes()
        self._refresh_measured_edges()

    def _update_polygon_fill(self) -> None:
        """Outlines only where the proposal or the manual map already colours
        the layers."""

        drawn = (
            self._manual_display_layers() is not None
            if self._map_kind() == _MANUAL
            else self.proposal is not None
        )
        self.canvas.set_polygon_fill(
            not (drawn and self.show_proposal.isChecked() and self._other_photo is None)
        )

    def _enhance_toggled(self, _checked: bool) -> None:
        if self._other_photo is not None:
            self.show_frame_photo(self._other_photo)
        else:
            self._show_reference(reset_view=False)

    def _frame_changed(self) -> None:
        self._reset_fill_region()
        self._update_enhanced()
        if self._other_photo is None and self.enhance.isChecked():
            self._show_reference(reset_view=False)
        self._brush_changed(self.brush_size.value())
        self._refresh_status()
        self._schedule_manual_map()

    # ---------------------------------------------------------- manual map

    def _map_kind(self) -> str:
        """The map shown and saved as a version: the proposal or the manual map."""

        if (self.session is not None and layer_priority.layer_ids(self.session)
                and workspace_frame.frame_kind(self._workspace_frame_config) != "rig-corrected"):
            return _MANUAL
        box = getattr(self, "map_shown", None)
        return str(box.currentData() or "proposal") if box is not None else "proposal"

    def _overlay_text(self) -> str:
        """The name of the map drawn over the photograph."""

        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            return "Saved layer boundaries"
        return "Manual map" if self._map_kind() == _MANUAL else "Proposal"

    def _manual_unavailable(self) -> str:
        """Why no manual map can be drawn now ("" when one can)."""

        if self.session is None or self.image is None:
            return "Choose a reference run with a baseline photograph first."
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            return "A manual map is drawn on the raw reference photograph."
        return ""

    def _manual_wanted(self) -> bool:
        """Whether the manual map is shown or compared with the proposal."""

        box = getattr(self, "show_differences", None)
        compared = box is not None and box.isChecked()
        return (self._map_kind() == _MANUAL or compared) and not self._manual_unavailable()

    def _manual_inputs_key(self) -> str:
        """SHA-256 of what the manual map is drawn from: the polygons, the
        edge fill, the photograph's size and the rig frame ("" without them)."""

        if self._manual_unavailable():
            return ""
        assert self.session is not None and self.image is not None
        frame = self.current_frame()
        payload = [
            self.session.to_dict().get("polygons"),
            self.session.edge_fill_mm,
            list(self.image.image_size),
            frame.to_dict() if frame is not None else None,
        ]
        marked = layer_priority.layer_ids(self.session)
        if marked:
            payload.append({"always_on_top": marked})
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _compare_inputs_key(self) -> str:
        """SHA-256 of what the proposal's map to compare with is drawn from
        ("" when the differences are not shown or there is no proposal)."""

        if (
            not self.show_differences.isChecked()
            or self._manual_unavailable()
            or self.proposal is None
            or self._corrections_error
        ):
            return ""
        assert self.session is not None and self.image is not None
        if tuple(self.proposal.image_size) != tuple(self.image.image_size):
            return ""
        if self._native_async_required():
            native = self._large_native_result()
            if native is None or native.error:
                return ""
        frame = self.current_frame()
        payload = [
            self._current_map_digest(),
            list(self.image.image_size),
            frame.to_dict() if frame is not None else None,
        ]
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _manual_current(self) -> bool:
        """Whether the manual map drawn is that of the polygons shown now."""

        return (
            self._manual is not None
            and self.session is not None
            and self._manual_run == self.session.run_id
            and bool(self._manual_key)
            and self._manual_key == self._manual_inputs_key()
        )

    def _schedule_manual_map(self) -> None:
        """Draw the manual map again once the marks, the proposal or the frame
        rest for a moment, when it is shown or compared."""

        timer = getattr(self, "_manual_timer", None)
        if timer is not None and not self._loading and self._manual_wanted():
            timer.start()

    def _update_manual_map(self) -> bool:
        """Start drawing the manual map, and the proposal's map it is compared
        with, in a worker when either is out of date. True when one started."""

        self._manual_timer.stop()
        if self._manual_thread is not None:
            self._manual_pending = True  # Drawn again when this one is done.
            return False
        if not self._manual_wanted():
            self._refresh_manual_status()
            return False
        assert self.session is not None and self.image is not None
        manual_key = self._manual_inputs_key()
        compare_key = self._compare_inputs_key()
        compared = bool(compare_key) and not (
            compare_key == self._compare_key and self._compare_run == self.session.run_id
        )
        request = {
            "run_id": self.session.run_id,
            "manual_key": "" if self._manual_current() else manual_key,
            "compare_key": compare_key if compared else "",
            "proposal": self.proposal if compared else None,
            "corrections": deepcopy(self.corrections) if compared else [],
        }
        if not request["manual_key"] and not request["compare_key"]:
            self._refresh_manual_status()
            return False
        thread = QThread(self)
        worker = _ManualMapWorker(
            # Copies: the marks and the frame stay editable while it draws.
            deepcopy(self.session),
            self.image.image_size,
            deepcopy(self.current_frame()),
            request,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._manual_progress)
        worker.finished.connect(self._manual_map_ready)
        worker.failed.connect(self._manual_map_failed)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._manual_thread_finished)
        self._manual_thread, self._manual_worker = thread, worker
        self._manual_message = (
            "Drawing the manual map…" if request["manual_key"] else "Comparing with the proposal…"
        )
        thread.start()
        self._refresh_manual_status()
        return True

    def _manual_progress(self, message: str) -> None:
        """Show the worker's progress under the photograph."""

        self._manual_message = message
        self._refresh_manual_status()

    def _manual_map_ready(self, result: object) -> None:
        """Keep what the worker drew, in the GUI thread, and show it."""

        drawn = dict(result)  # type: ignore[call-overload]
        if "manual" in drawn:
            self._manual = drawn["manual"]
            self._manual_key = drawn["manual_key"]
            self._manual_run = drawn["run_id"]
        if "layers" in drawn:
            self._compare_layers = drawn["layers"]
            self._compare_key = drawn["compare_key"]
            self._compare_run = drawn["run_id"]
        self._manual_display = None
        self._manual_message = ""
        self._render_proposal()
        self._refresh_versions()
        self._refresh_manual_status()

    def _manual_map_failed(self, message: str) -> None:
        """Say why the manual map could not be drawn."""

        self._manual_message = f"The manual map could not be drawn: {message}"
        self._refresh_manual_status()

    def _manual_thread_finished(self) -> None:
        """Drop the finished worker and its thread here, in the GUI thread, and
        draw again if the marks changed meanwhile."""

        thread = self._manual_thread
        self._manual_thread = None
        self._manual_worker = None
        if thread is not None:
            thread.deleteLater()
        if self._manual_pending:
            self._manual_pending = False
            self._update_manual_map()
        self._refresh_versions()
        self._refresh_manual_status()

    def _manual_display_layers(self) -> dict[str, Any] | None:
        """The manual map of the run shown, and its differences from the
        proposal, on the display grid (None when there is none to show).

        Kept until either map changes; thin areas stay visible.
        """

        manual = self._manual
        if (
            manual is None
            or self.image is None
            or self.session is None
            or self._manual_run != self.session.run_id
            or tuple(manual.image_size) != tuple(self.image.image_size)
        ):
            return None
        import cv2

        height, width = self.image.display.shape[:2]
        layers = self._compare_layers if self._compare_run == self._manual_run else None
        if self._native_async_required():
            current = self._compare_inputs_key()
            if not current or self._compare_key != current:
                layers = None
        live_key = self._manual_inputs_key()
        live = bool(live_key and self._manual_key == live_key)
        key = (id(manual), id(layers), height, width, live_key, live)
        if self._manual_display is not None and self._manual_display["key"] == key:
            return self._manual_display
        classes = cv2.resize(manual.classes, (width, height), interpolation=cv2.INTER_NEAREST)
        extra_pixels, coverage_reason = 0, ""
        if live:
            from .painting_drawing_coverage import add_explicit_outside
            try:
                classes, extra_pixels = add_explicit_outside(
                    self.session, self.current_frame(), classes)
            except ValueError as error:
                coverage_reason = str(error)
        display: dict[str, Any] = {
            "key": key,
            "classes": classes,
            "extra_drawing_pixels": extra_pixels,
            "coverage_reason": coverage_reason,
            "uncovered": _any_pixel(manual.uncovered(), (width, height)),
            "edge": _any_pixel(manual.edge_filled, (width, height)),
            "uncovered_pixels": manual.uncovered_pixels(),
            "edge_filled_pixels": manual.edge_filled_pixels(),
            "differences": None,
            "difference_pixels": None,
        }
        if layers is not None and layers.shape == manual.classes.shape:
            differ = lpm.manual_map_differences(manual, layers)
            display["differences"] = _any_pixel(differ, (width, height))
            display["difference_pixels"] = int(np.count_nonzero(differ))
        self._manual_display = display
        return display

    def _render_manual_map(self) -> None:
        """Show the manual map: its layers, the pixels no polygon covers in
        magenta and the frame-edge fill hatched (display only)."""

        display = self._manual_display_layers()
        if display is None or self._other_photo is not None:
            self.canvas.set_proposal(None)
            self._update_polygon_fill()
            return
        rgba, shown = self._layer_overlay(display["classes"])
        rows, cols = np.indices(display["classes"].shape)
        hatch = ((rows + cols) // 3) % 2 == 0
        rgba[display["edge"] & hatch & shown] = _EDGE_FILL_COLOR
        rgba[display["uncovered"]] = _UNCOVERED_COLOR
        self._mark_differences(rgba)
        self.canvas.set_proposal(rgba)
        self._update_polygon_fill()

    def _mark_differences(self, rgba: np.ndarray) -> None:
        """Mark in red where the manual map and the proposal disagree, when
        the differences are shown and both maps are drawn."""

        if not self.show_differences.isChecked() or self.proposal is None:
            return
        display = self._manual_display_layers()
        if display is None or display["differences"] is None:
            return
        if display["differences"].shape == rgba.shape[:2]:
            rgba[display["differences"]] = _DIFFERENCE_COLOR

    def _map_shown_changed(self, _index: int) -> None:
        """Show the other map; the manual map is drawn when it is out of date."""

        self._sync_layer_priority()
        self.show_proposal.setText(self._overlay_text())
        if not self.show_proposal.isChecked():
            self.show_proposal.setChecked(True)
        self._update_manual_map()
        self._render_proposal()
        self._refresh_corrections()
        self._refresh_versions()
        self._refresh_manual_status()

    def _differences_toggled(self, _checked: bool) -> None:
        """Show or hide where the manual map and the proposal disagree."""

        self._update_manual_map()
        self._render_proposal()
        self._refresh_manual_status()

    def _refresh_manual_status(self) -> None:
        """Say under the photograph what the manual map covers and how much it
        differs from the proposal."""

        label = getattr(self, "manual_status", None)
        if label is None:
            return
        if not (self._map_kind() == _MANUAL or self.show_differences.isChecked()):
            label.hide()
            label.setText("")
            return
        label.show()
        problem = self._manual_unavailable()
        if problem:
            label.setText(problem)
            return
        assert self.session is not None and self.image is not None
        parts = [self._manual_message] if self._manual_message else []
        display = self._manual_display_layers()
        frame = self.current_frame()
        area = lpm.pixel_area_cm2(frame, self.image.image_size)
        scale = "" if lp.frame_pixels_per_metre(frame) else " at a nominal scale"
        if display is not None:
            uncovered = display["uncovered_pixels"]
            if uncovered:
                parts.append(
                    f"Manual map: {uncovered:,} px inside the rig frame "
                    f"({uncovered * area:.1f} cm²{scale}) are covered by no polygon "
                    "(magenta). A version of it can be saved, but approved only "
                    "when polygons cover them."
                )
            else:
                parts.append("Manual map: every pixel inside the rig frame has a layer.")
            if display.get("extra_drawing_pixels"):
                parts.append("Explicit polygons outside the rig frame are shown as drawing context and kept in the live placement preview. "
                             "Saved manual versions keep their original frame-limited map. Drawing cannot cover areas outside this photo.")
            if display.get("coverage_reason"):
                parts.append("Extra drawing context unavailable: " + display["coverage_reason"])
            filled = display["edge_filled_pixels"]
            if filled:
                parts.append(
                    f"The {self._manual.edge_fill_mm:g} mm edge fill gives {filled:,} px "
                    "along the frame edge the nearest polygon's layer (hatched)."
                )
            if self.session.strokes:
                parts.append(
                    f"The {len(self.session.strokes)} stroke(s) are not part of it; "
                    "strokes guide the proposal only."
                )
        if self.show_differences.isChecked():
            native = self._large_native_result() if self._native_async_required() else None
            if self._native_async_required() and native is None:
                parts.append("Computing exact native corrections before comparing the maps.")
            elif self._native_async_required() and native.error:
                parts.append("Differences unavailable: " + native.error)
            elif self.proposal is None:
                parts.append("There is no proposal to compare it with.")
            elif display is not None and display["difference_pixels"] is not None:
                pixels = display["difference_pixels"]
                parts.append(
                    f"It and the proposal disagree on {pixels * area:.1f} cm²{scale} "
                    f"({pixels:,} px, red)."
                )
        label.setText(" ".join(parts))

    # ------------------------------------------------------------ facies

    @staticmethod
    def _clear_cell_widgets(table: QTableWidget) -> None:
        """Remove cell widgets now; Qt would only delete them later."""

        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                widget = table.cellWidget(row, column)
                if widget is not None:
                    widget.hide()
                    table.removeCellWidget(row, column)

    def _refresh_facies_table(self) -> None:
        """Rebuild the facies table: name, layers, role and properties."""

        if self._facies_menu_open:
            # A layer list is open; rebuild once it has closed.
            self._facies_refresh_pending = True
            return
        session = self._display_session()
        names = session.facies_order() if session is not None else []
        colors = self.display_layer_styles(session)[0]
        selected = self.selected_facies()
        table = self.facies_table
        table.blockSignals(True)
        try:
            unit = (session.permeability_unit if session is not None
                and getattr(self, "_display_only_preview", False)
                else str(self.unit.currentData() or "darcy"))
            table.setHorizontalHeaderLabels(
                ["Facies", "Layers", "Role", "Porosity", f"Perm. [{unit}]", "Approval comparison"]
            )
            for column, tip in (
                (3, "Porosity [-], between 0 and 1"),
                (4, f"Permeability [{unit}]"),
            ):
                header_item = table.horizontalHeaderItem(column)
                if header_item is not None:
                    header_item.setToolTip(tip)
            self._clear_cell_widgets(table)
            self._facies_buttons = {}
            table.setRowCount(len(names))
            for row, name in enumerate(names):
                assert session is not None
                entry = session.facies.get(name, lp.FaciesEntry())
                members = session.layers_in_facies(name)
                item = QTableWidgetItem(
                    _swatch(colors[members[0].layer_id].name()) if members else QIcon(), name
                )
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEditable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(self._facies_visibility_state(name))
                if members:
                    item.setToolTip(
                        "Double-click to rename; the name of another facies merges "
                        "the two. Tick to show or hide the group with one colour "
                        "(By facies, display only)."
                    )
                else:
                    item.setForeground(_muted(self))
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                    item.setToolTip(
                        "No layers: tick layers in the Layers list, or remove the "
                        "facies. Facies without layers are not exported."
                    )
                table.setItem(row, 0, item)
                # The cell behind the layer list keeps the facies for selection.
                holder = QTableWidgetItem()
                holder.setData(Qt.ItemDataRole.UserRole, name)
                holder.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                table.setItem(row, 1, holder)
                table.setCellWidget(row, 1, self._layers_button(name))
                role = QComboBox()
                for key, text in lp.LAYER_ROLES.items():
                    role.addItem(_ROLE_SHORT.get(key, text), key)
                    role.setItemData(
                        role.count() - 1, text, Qt.ItemDataRole.ToolTipRole
                    )
                role.setCurrentIndex(max(0, role.findData(entry.role)))
                role.setToolTip(
                    "Porous medium needs porosity and permeability. In mass "
                    "integrals free water has porosity 1 and excluded regions "
                    "(ColorChecker, tape, fittings) 0."
                )
                role.currentIndexChanged.connect(
                    lambda _index, combo=role, facies=name: self._role_edited(
                        combo, facies
                    )
                )
                table.setCellWidget(row, 2, role)
                porous = entry.role == "porous_medium"
                fixed = entry.effective_porosity()
                for column, value in ((3, entry.porosity), (4, entry.permeability)):
                    if porous:
                        text = _format_number(value)
                    elif column == 3 and fixed is not None:
                        text = _format_number(fixed)
                    else:
                        text = "—"
                    cell = QTableWidgetItem(text)
                    cell.setData(Qt.ItemDataRole.UserRole, name)
                    if not porous and column == 3 and fixed is not None:
                        cell.setToolTip(
                            "Free water fills the whole volume: porosity 1 in mass "
                            "integrals."
                            if fixed > 0.0
                            else "Left out of mass integrals: porosity 0."
                        )
                    if porous:
                        cell.setFlags(
                            Qt.ItemFlag.ItemIsEnabled
                            | Qt.ItemFlag.ItemIsSelectable
                            | Qt.ItemFlag.ItemIsEditable
                        )
                        invalid = (
                            value is None
                            or (column == 3 and not 0.0 < value <= 1.0)
                            or (column == 4 and value <= 0.0)
                        )
                        if invalid and members:
                            cell.setBackground(QColor(201, 138, 20, 70))
                            cell.setToolTip(
                                "Porosity between 0 and 1"
                                if column == 3
                                else "A positive permeability"
                            )
                    else:
                        cell.setFlags(
                            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                        )
                        cell.setForeground(_muted(self))
                    table.setItem(row, column, cell)
                if name == selected:
                    table.selectRow(row)
        finally:
            table.blockSignals(False)
        _fit_table_height(table, max_rows=10)
        self._sync_facies_visibility()
        self._facies_selection_changed()
        self._refresh_facies_approval()
        self._refresh_status()

    def _layers_text(self, facies: str) -> str:
        session = self._display_session()
        if session is None:
            return ""
        members = session.layers_in_facies(facies)
        if not members:
            return "No layers: tick…"
        return ", ".join(layer.name for layer in members)

    def _layers_button(self, facies: str) -> QToolButton:
        button = QToolButton()
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        button.setText(self._layers_text(facies))
        button.setToolTip(
            "Tick the layers of this facies. A layer belongs to one facies: ticking "
            "moves it here, unticking gives it a facies of its own."
        )
        menu = QMenu(button)
        menu.aboutToShow.connect(
            lambda target=menu, name=facies: self._fill_layer_menu(target, name)
        )
        menu.aboutToHide.connect(self._layer_menu_closed)
        button.setMenu(menu)
        self._facies_buttons[facies] = button
        return button

    def _fill_layer_menu(self, menu: QMenu, facies: str) -> None:
        self._facies_menu_open = True
        menu.clear()
        if self.session is None:
            return
        for layer in self.session.layers:
            box = QCheckBox(layer.name)
            box.setIcon(_swatch(layer.color))
            box.setChecked(layer.facies == facies)
            if layer.facies != facies and layer.facies != layer.name:
                box.setText(f"{layer.name}   (now in {layer.facies})")
            box.toggled.connect(
                lambda checked, layer_id=layer.layer_id, name=facies: (
                    self.set_layer_in_facies(layer_id, name, checked)
                )
            )
            holder = QWidget()
            row = QHBoxLayout(holder)
            row.setContentsMargins(10, 3, 12, 3)
            row.addWidget(box)
            action = QWidgetAction(menu)
            action.setDefaultWidget(holder)
            menu.addAction(action)

    def _layer_menu_closed(self) -> None:
        self._facies_menu_open = False
        # The menu belongs to a button in the table: rebuild after it has closed.
        if self._facies_refresh_pending:
            self._facies_refresh_pending = False
            QTimer.singleShot(0, self._refresh_facies_table)

    def selected_facies(self) -> str:
        rows = {index.row() for index in self.facies_table.selectedIndexes()}
        if not rows:
            return ""
        item = self.facies_table.item(min(rows), 0)
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    def _facies_selection_changed(self) -> None:
        self.remove_facies_button.setEnabled(
            not getattr(self, "_display_only_preview", False)
            and self.session is not None and bool(self.selected_facies())
        )

    def _role_edited(self, box: QComboBox, facies: str) -> None:
        try:
            role = str(box.currentData())
        except RuntimeError:
            return
        self.set_facies_role(facies, role)

    def _facies_state(self) -> tuple:
        """Capture the current facies properties and unit for one undo step."""
        if self.session is None:
            return (), ""
        return (tuple((name, value.role, value.porosity, value.permeability)
                      for name, value in self.session.facies.items()), self.session.permeability_unit)

    def _facies_snapshot(self) -> _Snapshot:
        """Append facies state only to commands that actually edit properties."""
        return self._snapshot() + (self._facies_state(),)

    def _remember_facies_change(self, before: _Snapshot) -> bool:
        """Remember one changed property command without creating no-op entries."""
        if before[4] == self._facies_state():
            return False
        self._undo_stack.append(before)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []
        return True

    def _undo_facies_change(self) -> None:
        """Undo the latest facies command while preserving the saved approval."""
        if (self.session is None or not self.run_preview.is_reference() or self.busy()
                or not self._undo_stack or len(self._undo_stack[-1]) < 5
                or self._undo_stack[-1][4] == self._facies_state()):
            return
        current, previous = self._facies_snapshot(), self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore(previous)
        self._after_restore(current, previous)

    def _refresh_facies_approval(self) -> None:
        """Show current draft differences from the literal selected approval."""
        label = getattr(self, "facies_approval_note", None)
        if label is None:
            return
        reference, changed, error = None, {}, ""
        record = self._selected_version()
        approval = self._approval_for_version(record)
        if self.session is not None and record and lpm.is_manual_version(record) and approval and not self._map_approvals_error and not record.get("history_error"):
            try:
                key = facies_approval.metadata_key(record, approval)
                cache = getattr(self, "_facies_approval_cache", None)
                if cache is None or cache[0] != key:
                    cache = key, facies_approval.approved_reference(record, approval)
                    self._facies_approval_cache = cache
                reference = cache[1]
                if reference["run_id"] != self.session.run_id:
                    raise ValueError("The selected approved session belongs to another run")
                changed = facies_approval.differences(self.session, reference)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        if error:
            text = "Facies comparison with the selected approved version is unavailable: " + error
        elif reference is None:
            text = "Select an approved manual-map version to compare the current facies."
        elif changed:
            details = "; ".join(f"{name} ({', '.join(fields)})" for name, fields in changed.items())
            text = (f"Current session differs from {reference['name']} (approved): {details}. "
                    "Undo the change, or save and approve a new version if it is intended. The saved approval is unchanged.")
        else:
            text = f"Current facies match {reference['name']} (approved)."
        label.setText(text)
        label.setStyleSheet(f"color: {_state_color(self, 'warn').name()};" if changed or error else "")
        self.facies_version_note.setText(text if changed or error else "")
        self.facies_version_note.setVisible(bool(changed or error))
        self.facies_version_note.setStyleSheet(label.styleSheet())
        table = self.facies_table
        blocked = table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                name = "" if item is None else str(item.data(Qt.ItemDataRole.UserRole) or "")
                fields = changed.get(name, [])
                note = (f"changed since {reference['name']} (approved)" if fields else
                        "Comparison unavailable" if error else "")
                state = QTableWidgetItem(note)
                state.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                state.setToolTip(text if fields or error else "Role, porosity, permeability and unit match the selected approved session" if reference else text)
                if fields or error:
                    state.setForeground(_state_color(self, "warn"))
                table.setItem(row, 5, state)
        finally:
            table.blockSignals(blocked)
        self.undo_facies_button.setEnabled(bool(self.session and self.run_preview.is_reference()
            and not self.busy() and self._undo_stack and len(self._undo_stack[-1]) > 4
            and self._undo_stack[-1][4] != self._facies_state()))

    def _facies_changed(self) -> None:
        self._mark_dirty()
        if self._facies_menu_open:
            self._facies_refresh_pending = True
        else:
            QTimer.singleShot(0, self._refresh_facies_table)
        self._refresh_layers()
        self._render_marks()
        self._render_proposal()

    def set_layer_in_facies(self, layer_id: int, facies: str, member: bool) -> None:
        """Tick or untick a layer in a facies' layer list."""

        if self.session is None:
            return
        self.session.set_layer_in_facies(layer_id, facies, bool(member))
        button = self._facies_buttons.get(facies)
        if button is not None:
            try:
                button.setText(self._layers_text(facies))
            except RuntimeError:  # the table was rebuilt
                pass
        self._facies_changed()

    def set_layer_facies(self, layer_id: int, facies: str) -> None:
        """Move a layer into a facies (created when new)."""

        if self.session is None:
            return
        layer = self.session.layer(layer_id)
        if (
            layer is None
            or not str(facies).strip()
            or str(facies).strip() == layer.facies
        ):
            return
        self.session.set_layer_facies(layer_id, facies)
        self._facies_changed()

    def add_facies(self, name: str | None = None) -> str:
        if self.session is None:
            return ""
        try:
            created = self.session.add_facies(name)
        except ValueError as error:
            QMessageBox.information(self, "Facies name in use", str(error))
            return ""
        self._mark_dirty()
        self._refresh_facies_table()
        row = self.session.facies_order().index(created)
        self.facies_table.selectRow(row)
        if name is None:
            self.facies_table.editItem(self.facies_table.item(row, 0))
        return created

    def remove_facies(self, name: str | None = None, *, confirm: bool = True) -> None:
        if self.session is None:
            return
        facies = name or self.selected_facies()
        if facies not in self.session.facies:
            return
        members = self.session.layers_in_facies(facies)
        if confirm and members:
            answer = QMessageBox.question(
                self,
                "Remove facies",
                f"Remove the facies {facies}? Its layers ("
                + ", ".join(layer.name for layer in members)
                + ") get a facies of their own.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.session.remove_facies(facies)
        self._facies_changed()

    def rename_facies(self, old: str, new: str) -> None:
        if self.session is None or not str(new).strip() or old == str(new).strip():
            return
        self.session.rename_facies(old, new)
        if old in self._hidden_facies:
            self._hidden_facies.discard(old)
            self._hidden_facies.add(str(new).strip())
        self._facies_changed()

    def set_facies_role(self, facies: str, role: str) -> None:
        if self.session is None or facies not in self.session.facies:
            return
        if self.session.facies[facies].role == role:
            return
        before = self._facies_snapshot()
        self.session.set_facies(facies, role=role)
        self._remember_facies_change(before)
        self._facies_changed()

    def set_facies_properties(
        self,
        facies: str,
        *,
        porosity: float | None = None,
        permeability: float | None = None,
    ) -> None:
        if self.session is None:
            return
        before = self._facies_snapshot()
        self.session.set_facies(facies, porosity=porosity, permeability=permeability)
        self._remember_facies_change(before)
        self._mark_dirty()
        self._refresh_facies_table()

    def _facies_item_changed(self, item: QTableWidgetItem) -> None:
        session = self._display_session()
        if session is None:
            return
        facies = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if facies not in session.facies:
            return
        if item.column() == 0:
            if item.text() == facies:
                self.set_facies_visible(facies, item.checkState() != Qt.CheckState.Unchecked)
                return
            if getattr(self, "_display_only_preview", False):
                return
            self.rename_facies(facies, item.text())
            return
        if getattr(self, "_display_only_preview", False):
            return
        if item.column() not in (3, 4):
            return
        value = _parse_number(item.text())
        before = self._facies_snapshot()
        if item.column() == 3:
            self.session.set_facies(facies, porosity=value)
        else:
            self.session.set_facies(facies, permeability=value)
        self._remember_facies_change(before)
        self._mark_dirty()
        QTimer.singleShot(0, self._refresh_facies_table)

    def _present_layers(self) -> set[int] | None:
        if self.proposal is None:
            return None
        return {
            int(row["layer_id"]) for row in self.proposal.layers if row.get("pixels")
        }

    def _unit_changed(self, _index: int) -> None:
        if self.session is None or self._loading:
            return
        unit = str(self.unit.currentData() or "darcy")
        if self.session.permeability_unit == unit:
            return
        before = self._facies_snapshot()
        self.session.permeability_unit = unit
        self._remember_facies_change(before)
        self._mark_dirty()
        self._refresh_facies_table()

    # ------------------------------------------------------------ export

    def _current_map_digest(self) -> str:
        if not lpp.is_polygons_only(self.proposal):
            return lp.map_digest(self.session, self.proposal, self.corrections)
        if self._native_async_required():
            native = self._large_native_result()
            if native is None:
                raise ValueError("Computing the exact native map and its SHA-256")
            if native.error:
                raise ValueError(native.error)
            return native.map_digest
        payload = self.session.to_dict()
        payload.pop("updated_utc", None)
        payload.pop("updated_by", None)
        key = (id(self.proposal), self.proposal.digest, id(self.proposal.classes),
               id(self.proposal.confidence), lpp._json_bytes(payload),
               lp.corrections_digest(self.corrections))
        if self._new_map_digest_key != key:
            self._new_map_digest = lp.map_digest(self.session, self.proposal, self.corrections)
            self._new_map_digest_key = key
        return self._new_map_digest

    def export_problems_now(self) -> list[str]:
        if self.session is not None and layer_priority.layer_ids(self.session):
            return ["Always on top uses the manual map. Save and approve a manual-map version; "
                    "proposal export does not include this priority."]
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            return ["Corrected-frame production export and Build run model are not qualified yet."]
        if self._raw_session_frame:
            return [_RAW_SESSION_EXPORT_NOTE]
        if self.session is None:
            return ["Choose a reference run with a baseline photograph."]
        if self._native_async_required():
            native = self._large_native_result()
            if native is None:
                return ["Computing the exact native map; approval and export wait for the checked result."]
            return list(native.problems)
        if lpp.is_polygons_only(self.proposal):
            reference, frame = self.reference(), self.current_frame()
            try:
                source = lpp._source(self.session, frame, tuple(self.proposal.image_size))
                roles = [(row.layer_id, row.name, row.facies) for row in self.session.layers]
                properties = [(name, value.role, value.porosity, value.permeability)
                              for name, value in self.session.facies.items()]
                key = (id(self.proposal), self.proposal.digest, id(self.proposal.classes),
                       id(self.proposal.inside), lpp._json_bytes(source),
                       lpp._json_bytes([roles, properties, self.session.permeability_unit]),
                       lp.corrections_digest(self.corrections),
                       str(reference.baseline_image) if reference else None,
                       reference.width_m if reference else None,
                       reference.height_m if reference else None)
                if self._new_export_check_key != key:
                    self._new_export_check_problems = lpp.check_ready_for_revision(
                        self.session, self.proposal, frame, reference, self.corrections)
                    self._new_export_check_key = key
                return list(self._new_export_check_problems)
            except (ValueError, TypeError, KeyError) as error:
                return [str(error)]
        check = lpp.check_ready_for_revision if lpp.is_polygons_only(self.proposal) else lp.check_ready_for_revision
        return check(
            self.session,
            self.proposal,
            self.current_frame(),
            self.reference(),
            self.corrections,
        )

    def save_and_export(self) -> None:
        if self.session is not None and layer_priority.layer_ids(self.session):
            self.export_status.setText(self.export_problems_now()[0])
            return
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            self.export_status.setText("Corrected-frame production export and Build run model are not qualified yet.")
            return
        if self._raw_session_frame:
            self.export_status.setText(_RAW_SESSION_EXPORT_NOTE)
            return
        if self.busy() or self.session is None or self.proposal is None:
            if self.proposal is None:
                QMessageBox.information(
                    self, "Not ready to export", "Propose the boundaries first."
                )
            return
        problems = self.export_problems_now()
        if problems:
            QMessageBox.information(self, "Not ready to export", "\n".join(problems))
            return
        reference = self.reference()
        assert reference is not None
        self.save_session()
        thread = QThread(self)
        worker = _SaveWorker(
            self.project,
            deepcopy(self.session) if lpp.is_polygons_only(self.proposal)
            else lp.PaintingSession.from_dict(self.session.to_dict()),
            self.proposal,
            reference,
            deepcopy(self.current_frame()) if lpp.is_polygons_only(self.proposal) else self.current_frame(),
            default_actor(),
            corrections=tuple(self.corrections),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.export_status.setText)
        worker.finished.connect(self._export_ready)
        worker.failed.connect(self._export_failed)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._save_thread_finished)
        self._save_thread = thread
        self._save_worker = worker
        self.export_button.setEnabled(False)
        self.propose_button.setEnabled(False)
        # The revision is written from a snapshot; the frame and points are
        # also written by the save, so they wait until it has finished.
        self.rig_panel.setEnabled(False)
        self._export_key = self._export_state_key()
        self.export_busy.show()
        self.export_status.setText("Starting…")
        self.step_export.set_status("Working…", "warn")
        thread.start()
        self.busy_changed.emit(True)

    @Slot(object)
    def _export_ready(self, export: Any) -> None:
        worker = self._save_worker
        if (
            worker is None
            or self.session is None
            or worker.session.run_id == self.session.run_id
        ):
            self.last_export = export
            # Edits made while the revision was written are not in it.
            self._export_stale = self._export_state_key() != self._export_key
            self._refresh_export_info()
        self.exported.emit(export)

    @Slot(str)
    def _export_failed(self, message: str) -> None:
        self.export_status.setText(message)
        QMessageBox.warning(self, "Could not save and export", message)

    def _save_thread_finished(self) -> None:
        thread = self._save_thread
        if thread is not None:
            thread.wait()
        self._save_thread = None
        self._save_worker = None
        if thread is not None:
            thread.deleteLater()
        self.export_busy.hide()
        self.propose_button.setEnabled(self.session is not None)
        self.rig_panel.setEnabled(True)
        self._refresh_status()
        self.busy_changed.emit(self.busy())

    def _export_state_key(self) -> str:
        """Everything a saved revision depends on."""

        import hashlib
        import json

        if self.session is None:
            return ""
        payload = self.session.to_dict()
        payload.pop("updated_utc", None)
        payload.pop("updated_by", None)
        payload["proposal"] = self.proposal.digest if self.proposal else ""
        payload["points"] = self.rig_panel.current_points()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _refresh_export_info(self) -> None:
        export = self.last_export
        if export is not None and not self._current_revision_is(export.revision_id):
            # A later revision was saved (for example in the advanced view).
            self.last_export = export = None
            self._export_stale = True
        if export is not None:
            self.export_status.setText(
                f"Saved revision {str(export.revision_id)[:8]} and exported "
                f"{export.region_count} regions in {export.facies_count} facies to "
                f"{export.destination}."
            )
            self.open_export_button.setEnabled(True)
            return
        info = self._latest_painted_revision()
        if info:
            self.export_status.setText(info)
        else:
            self.export_status.setText("")
        self.open_export_button.setEnabled(self._export_folder() is not None)

    def _current_revision_is(self, revision_id: str) -> bool:
        from darsia.fluidflower.geometry_facies import GeometryRevisionStore

        try:
            current = GeometryRevisionStore(self.project).current()
        except (OSError, ValueError):
            return True
        return str(current.get("revision_id") or "") == str(revision_id)

    def _latest_painted_revision(self) -> str:
        from darsia.fluidflower.geometry_facies import (
            GEOMETRY_REVISION_FORMAT,
            GeometryRevisionStore,
        )

        reference = self.reference()
        try:
            current = GeometryRevisionStore(self.project).current()
        except (OSError, ValueError):
            return ""
        if current.get("format") != GEOMETRY_REVISION_FORMAT:
            return ""
        generation = dict(
            (current.get("configuration") or {}).get("facies_generation") or {}
        )
        if generation.get("method") != lp.PROPOSAL_METHOD:
            return "The current revision was saved in the advanced view."
        run = str(generation.get("reference_run") or "")
        when = str(current.get("recorded_utc") or "")[:16].replace("T", " ")
        text = f"Current revision {str(current.get('revision_id'))[:8]} ({when} UTC)"
        if reference is not None and run and run != reference.run_id:
            text += f" was painted on run {run}."
        else:
            text += " was saved from these layers."
        return text

    def _export_folder(self) -> Path | None:
        if self.last_export is not None:
            return Path(self.last_export.destination)
        root = Path(self.project.root) / "exports" / "native_setup"
        try:
            folders = [
                path
                for path in root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
        except OSError:
            return None
        if not folders:
            return None
        return max(folders, key=lambda path: path.stat().st_mtime)

    def _open_export_folder(self) -> None:
        folder = self._export_folder()
        if folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------ status

    def _refresh_legacy_export_status(self) -> None:
        """The existing native Setup status; round 5 does not use this gate."""
        if not hasattr(self, "export_problems"):
            return
        problems = self.export_problems_now()
        if self._save_thread is not None:
            self.step_export.set_status("Working…", "warn")
        elif problems:
            self.step_export.set_status("Missing input", "warn")
        elif self.last_export is not None and not self._export_stale:
            self.step_export.set_status("Exported", "done")
        else:
            self.step_export.set_status("Ready to export", "done")
        self.export_problems.setText(" ".join(problems))
        self.export_problems.setStyleSheet(
            f"color: {_state_color(self, 'warn').name()};" if problems else ""
        )
        self.export_button.setEnabled(not self.busy() and not problems)

    def _refresh_status(self) -> None:
        fixture = getattr(self, "fixture_preview", None)
        if fixture is not None:
            fixture.refresh()
        self._refresh_fill_region()
        self._sync_analysis_boxes()
        self._refresh_measured_edges()
        session = self.session
        frame = self.current_frame()
        points = self.rig_panel.current_points()
        located = sum(
            1
            for point in points
            if point["x_m"] is not None and point["y_m"] is not None
        )
        self.step_injection.set_status(f"{located} of {len(points)} located" if points else "No injection points",
            "done" if points and located == len(points) else "warn")
        point_text = f" · {located} injection point(s)" if located and self._workflow_scope != 'rig_frame' else ""
        partial_frame = self.rig_panel.frame
        if frame is None and partial_frame is not None and partial_frame.disabled_corners:
            visible = sum(name not in partial_frame.disabled_corners for name in partial_frame.corners)
            self.step_frame.set_status(f"{visible}/4 visible corners · references needed" + point_text, "warn")
        elif frame is None:
            self.step_frame.set_status("No rig frame" + point_text, "warn")
        elif not frame.has_size():
            self.step_frame.set_status("Enter the inner size" + point_text, "warn")
        else:
            self.step_frame.set_status("Frame set" + point_text, "done")

        if session is None:
            self.step_paint.set_status("No photograph", "warn")
            self.step_propose.set_status("", "")
            self.step_export.set_status("", "")
            self.export_problems.setText("")
            self.export_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)
            self._refresh_corrections()
            self._refresh_versions()
            self._guard_reference_display_controls()
            return
        marked = session.marked_layers()
        parts = [f"{len(marked)} of {len(session.layers)} layers marked"]
        if session.polygons:
            parts.append(f"{len(session.polygons)} polygon(s)")
        if session.strokes:
            parts.append(f"{len(session.strokes)} stroke(s)")
        self.step_paint.set_status(
            " · ".join(parts), "done" if len(marked) >= 2 else "warn"
        )
        self.undo_button.setEnabled(bool(self._undo_stack))
        self.redo_button.setEnabled(bool(self._redo_stack))
        self._refresh_analysis_box_undo()

        busy = self.busy()
        native = self._large_native_result() if self._native_async_required() else None
        if self._proposal_thread is not None:
            self.step_propose.set_status("Working…", "warn")
        elif self._native_async_required() and native is None:
            self.step_propose.set_status("Computing native corrections…", "warn")
        elif self._native_async_required() and native.error:
            self.step_propose.set_status("Native corrections unavailable", "warn")
        elif self.proposal is None:
            self.step_propose.set_status("Not proposed yet", "warn")
            self.proposal_summary.setText("")
        elif self.proposal_current():
            self.step_propose.set_status("Up to date", "done")
        else:
            self.step_propose.set_status("Out of date: propose again", "warn")
        if self.proposal is not None:
            self.proposal_summary.setText(self._proposal_text())
        self._refresh_corrections()
        self._refresh_versions()

        if not self._refresh_manual_export_route():
            self._refresh_legacy_export_status()
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            self.propose_button.setEnabled(False)
            self.rig_panel.setEnabled(False)
            if self.image is not None and self._saved_corrected_boundaries is not None:
                self.step_propose.set_status("Saved rig boundaries", "done")
                self.proposal_summary.setText("Read-only old rig maps; no classifier or registration was run.")
            else:
                self.step_propose.set_status("Saved rig maps unavailable", "warn")
                self.proposal_summary.setText("No raw-photo or common-map fallback was used.")
        self._guard_reference_display_controls()

    def _proposal_text(self) -> str:
        proposal = self.proposal
        session = self.session
        if proposal is None or session is None:
            return ""
        native = self._large_native_result() if self._native_async_required() else None
        if self._native_async_required() and native is None:
            return "Polygons only · computing the exact native map and gap count…"
        if self._native_async_required() and native.error:
            return "Polygons only · native map unavailable: " + native.error
        parts = []
        for row in proposal.layers:
            if not row.get("pixels"):
                continue
            layer = session.layer(int(row["layer_id"]))
            name = layer.name if layer is not None else str(row.get("name") or "")
            regions = int(row.get("regions") or 0)
            parts.append(f"{name}: {regions} region{'s' if regions != 1 else ''}")
        when = proposal.created_utc[11:16] if proposal.created_utc else ""
        sources = []
        if proposal.polygon_count:
            sources.append(f"{proposal.polygon_count} polygon(s)")
        if proposal.stroke_count or not sources:
            sources.append(f"{proposal.stroke_count} stroke(s)")
        text = (
            "From "
            + " and ".join(sources)
            + (f" at {when} UTC" if when else "")
            + ". "
            + " · ".join(parts)
            + "."
        )
        if proposal.merged_regions:
            text += f" {proposal.merged_regions} small island(s) merged."
        if lpp.is_polygons_only(proposal):
            if self._native_async_required():
                gap_count = native.gap_count
            else:
                self._corrected_native_state()
                gap_count = self._native_corrections_gaps
            return "Polygons only · exact photograph grid. " + text + (
                f" {gap_count:,} unassigned pixels (magenta); "
                "approval and export require no gaps.")
        uncertain = int(np.count_nonzero(proposal.uncertain()))
        inside = int(np.count_nonzero(proposal.classes != lp.OUTSIDE))
        if inside:
            text += f" Uncertain: {100.0 * uncertain / inside:.1f} % of the rig."
        return text


__all__ = [
    "LayerCanvas",
    "LayerPainterView",
    "StepSection",
    "insert_point_on_nearest_edge",
]
