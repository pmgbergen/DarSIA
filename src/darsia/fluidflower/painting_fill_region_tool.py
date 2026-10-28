"""A drawing-only gap-fill tool on the existing raw-coordinate canvas."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem, QLabel, QToolButton
from shiboken6 import isValid

from . import layer_painting as lp
from . import layer_painting_fill_region as geometry
from .series_geometry import CROP_CORNERS


class FillRegionTool(QObject):
    """Temporary cut preview plus an atomic, single-undo polygon edit.

    The host places the four exposed widgets and calls ``refresh`` alongside
    its normal status refresh. Only the editable canvas is watched. Geometry
    is always planned from all marks, including marks hidden from display.
    """

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.canvas = view.canvas
        self._viewport = self.canvas.viewport()
        self._closed = False
        self.button = QToolButton(view)
        self.button.setText("Fill region")
        self.button.setToolTip("Make a polygon in an uncovered region, bounded by existing polygons and the rig frame")
        self.cut_button = QToolButton(view)
        self.cut_button.setText("Cut line")
        self.cut_button.setToolTip("Drag a temporary line between boundaries, then click Fill region")
        self.clear_cut_button = QToolButton(view)
        self.clear_cut_button.setText("Clear cut")
        self.clear_cut_button.setToolTip("Clear the temporary cut without changing marks (Escape)")
        self.note = QLabel(view)
        self.note.setWordWrap(True)
        self._mode = None
        self._cut = ()
        self._cut_context = None
        self._draft = []
        self._cut_item = None
        self._changing = False
        for button, mode in ((self.button, "fill"), (self.cut_button, "cut")):
            button.setCheckable(True)
            view.tool_group.addButton(button)
            button.toggled.connect(lambda checked, value=mode: self._toggled(value, checked))
        for button in view.tool_buttons.values():
            button.toggled.connect(self._old_tool_selected)
        self.clear_cut_button.clicked.connect(self.clear_cut)
        for watched in (view, self.canvas, self._viewport):
            watched.destroyed.connect(self._invalidate)
        self._viewport.installEventFilter(self)
        self.canvas.installEventFilter(self)
        self.refresh()

    def _alive(self):
        return (not self._closed and isValid(self.view) and isValid(self.canvas)
                and isValid(self._viewport))

    def _invalidate(self, *_args):
        # Child destruction can deliver another filtered event after the
        # canvas C++ object has gone. Never call back into Qt from this slot.
        self._closed = True
        self._mode = None
        self._cut, self._draft, self._cut_context = (), [], None
        self._cut_item = None

    @property
    def active(self):
        return self._mode is not None

    @property
    def cut_line(self):
        return tuple(self._cut)

    def _notice(self, message):
        if not self._alive():
            return
        self.note.setText(str(message))
        self.view._canvas_notice(str(message))

    def _problem(self):
        if not self._alive():
            return "The drawing canvas is closed."
        view = self.view
        if view.session is None or view.image is None or not self.canvas.raw_size:
            return "Choose a reference photograph before filling a region."
        if view.busy():
            return "Wait for the current painting operation to finish."
        controller = getattr(view, "run_preview", None)
        if controller is not None and not controller.is_reference():
            return "The placed map is read only. Return to the reference run to draw."
        if view._other_photo is not None:
            return "Return to the reference photograph before filling a region."
        if not self.canvas.paint_enabled:
            return "This photograph is read only. Return to an editable reference photograph to draw."
        if self.canvas.place_mode:
            return "Finish placing the frame points before filling a region."
        frame = view.current_frame()
        if frame is None or not frame.complete():
            return "Mark a complete rig frame before filling a region."
        if frame.run_id and frame.run_id.casefold() != view.session.run_id.casefold():
            return "The rig frame belongs to another reference run."
        from .storage_paths import same_storage_path
        if frame.image and not same_storage_path(frame.image, view.session.image_path):
            return "The rig frame belongs to another reference photograph."
        if (frame.image_size is not None and tuple(frame.image_size) != tuple(view.session.image_size or ())):
            return "The rig frame and drawing have different photograph grids."
        if tuple(view.session.image_size or ()) != tuple(self.canvas.raw_size):
            return "The canvas and drawing have different photograph grids."
        if view.session.layer(int(self.canvas.layer_id)) is None:
            return "Choose a layer before filling a region."
        return ""

    def _context(self):
        session = self.view.session
        frame = self.view.current_frame()
        return (session.run_id, session.image_path, tuple(session.image_size or ()),
                tuple(tuple(frame.corners[name]) for name in CROP_CORNERS))

    def refresh(self):
        if not self._alive():
            return
        problem = self._problem()
        self.button.setEnabled(not problem)
        self.cut_button.setEnabled(not problem)
        self.clear_cut_button.setEnabled(bool(self._cut or self._draft))
        if problem:
            if self.active or self._cut or self._draft:
                self.deactivate()
                self._notice(problem)
            return
        if self._cut and self._cut_context != self._context():
            self.clear_cut()
        elif self._cut or self._draft:
            self._draw_cut()

    def _toggled(self, mode, checked):
        if not self._alive():
            return
        if self._changing:
            return
        if not checked:
            if self._mode == mode:
                self._mode = None
            return
        problem = self._problem()
        if problem:
            self.deactivate()
            self._notice(problem)
            return
        self._mode = mode
        # Internal Pan is deliberate: custom left events are consumed below,
        # while its existing right/middle panning and wheel logic stay intact.
        self.canvas.set_tool("pan")
        self.canvas.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self._notice("Click an uncovered region to make a polygon." if mode == "fill"
                     else "Drag a cut between boundaries. It changes no marks until a fill succeeds.")

    def _old_tool_selected(self, checked):
        if checked and not self._changing:
            self.deactivate(select_pan=False)

    def deactivate(self, *, select_pan=True):
        if not self._alive():
            self._invalidate()
            return
        self._mode = None
        self._changing = True
        try:
            self.button.setChecked(False)
            self.cut_button.setChecked(False)
            self.clear_cut()
            if select_pan:
                self.view.set_tool("pan")
        finally:
            self._changing = False

    def clear_cut(self):
        self._cut, self._draft, self._cut_context = (), [], None
        if self._cut_item is not None and isValid(self._cut_item):
            scene = self._cut_item.scene()
            if scene is not None:
                scene.removeItem(self._cut_item)
            self._cut_item = None
        if self._alive():
            self.clear_cut_button.setEnabled(False)

    def _draw_cut(self):
        points = self._draft or self._cut
        if not points:
            return
        path = QPainterPath(self.canvas.raw_to_scene(*points[0]))
        for point in points[1:]:
            path.lineTo(self.canvas.raw_to_scene(*point))
        if self._cut_item is None:
            self._cut_item = QGraphicsPathItem()
            pen = QPen(QColor("#ffdd44"), 2., Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._cut_item.setPen(pen)
            self._cut_item.setZValue(10000)
            self.canvas.scene().addItem(self._cut_item)
        self._cut_item.setPath(path)
        self.clear_cut_button.setEnabled(True)

    def _raw_point(self, event):
        return tuple(map(float, self.canvas.scene_to_raw(
            self.canvas.mapToScene(event.position().toPoint()))))

    def eventFilter(self, watched, event):
        if not self._alive() or not self.active:
            return False
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.deactivate()
            self._notice("The temporary cut was cleared. No marks were changed.")
            return True
        if watched is not self._viewport:
            return False
        kind = event.type()
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            problem = self._problem()
            if problem:
                self.deactivate()
                self._notice(problem)
                return True
            self.canvas.setFocus(Qt.FocusReason.MouseFocusReason)
            if kind == QEvent.Type.MouseButtonDblClick:
                return True
            point = self.canvas.mapToScene(event.position().toPoint())
            if not self.canvas.pixmap_item.boundingRect().contains(point):
                self._notice("Click inside the photograph.")
                return True
            if self._mode == "fill":
                self.fill_at(*self._raw_point(event))
            else:
                self.clear_cut()
                self._draft = [self._raw_point(event)]
                self._cut_context = self._context()
                self._draw_cut()
            return True
        if kind == QEvent.Type.MouseMove and self._draft:
            if event.buttons() & Qt.MouseButton.LeftButton:
                point = self._raw_point(event)
                if point != self._draft[-1]:
                    self._draft.append(point)
                    self._draw_cut()
                return True
        if kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self._draft:
                point = self._raw_point(event)
                if point != self._draft[-1]:
                    self._draft.append(point)
                if len(self._draft) < 2:
                    self.clear_cut()
                    self._notice("Drag a cut with two distinct endpoints.")
                else:
                    self._cut, self._draft = tuple(self._draft), []
                    self._draw_cut()
                    self.button.setChecked(True)
                    self._notice("Cut ready. Click the uncovered part to fill; Escape clears the cut.")
            return True
        return False

    def fill_at(self, row, col, cut_line=None):
        """Apply one checked vector plan, or leave the entire drawing alone."""
        problem = self._problem()
        if problem:
            self._notice(problem)
            return False
        session = self.view.session
        context = self._context()
        if cut_line is None and self._cut and self._cut_context != context:
            self.clear_cut()
            self._notice("The photograph or frame changed. Draw the cut again.")
            return False
        points = {index: tuple(polygon.points) for index, polygon in enumerate(session.polygons)}
        identities = [(polygon.layer_id, tuple(polygon.points)) for polygon in session.polygons]
        try:
            frame = self.view.current_frame()
            plan = geometry.fill_region(points, [frame.corners[name] for name in CROP_CORNERS],
                (row, col), layer_id=self.canvas.layer_id,
                cut_line=(self._cut or None) if cut_line is None else cut_line)
            replacements = {}
            for update in plan.updates:
                if update.index not in points or points[update.index] != update.before:
                    raise ValueError("The shared outline changed. Try filling again.")
                replacements[update.index] = lp.Polygon(
                    session.polygons[update.index].layer_id, list(update.after))
            added = lp.Polygon(plan.layer_id, list(plan.polygon_points))
            for polygon in [*replacements.values(), added]:
                session._check_polygon(polygon)
            if (self.view.session is not session or self._context() != context
                    or [(p.layer_id, tuple(p.points)) for p in session.polygons] != identities):
                raise ValueError("The drawing changed. Try filling again.")
        except (ValueError, IndexError, TypeError) as error:
            self._notice(error)
            return False
        snapshot = self.view._snapshot()
        undo, redo = list(self.view._undo_stack), list(self.view._redo_stack)
        flags = self.view._dirty, self.view._export_stale
        save_pending = self.view._save_timer.isActive()
        manual_pending = self.view._manual_timer.isActive()
        hidden = tuple(self.canvas.hidden_layers)
        try:
            self.view._push_undo()
            for index, polygon in replacements.items():
                session.replace_polygon(index, polygon)
            session.add_polygon(added)
            self.canvas.clear_selection()
            self.view._reveal_layer(added.layer_id)
            self.view._after_mark_change()
        except Exception as error:
            self.view._restore(snapshot)
            self.view._undo_stack, self.view._redo_stack = undo, redo
            self.view._dirty, self.view._export_stale = flags
            if not save_pending:
                self.view._save_timer.stop()
            if not manual_pending:
                self.view._manual_timer.stop()
            if tuple(self.canvas.hidden_layers) != hidden:
                self.view._set_hidden_layers(hidden)
            self.view._render_marks()
            self._notice(f"The fill could not be applied: {error}. Nothing was changed.")
            return False
        self.clear_cut()
        self._notice("Filled the region as a polygon. Ctrl+Z restores all shared outlines.")
        return True
