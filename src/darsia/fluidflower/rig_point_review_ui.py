"""Per-run reference-point review inside the existing photograph area."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPolygonItem,
    QGraphicsSimpleTextItem, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView, QHeaderView, QApplication)

from . import rig_point_review as review
from . import layer_painting as painting
from .painting_frame_models import read_adoption_cohort, verify_frame_model_selection
from .painting_run_preview import read_preview_geometry
from .rig_photo_loader import RigPhotoLoader, read_rig_photo
from .series_geometry import save_rig_frame


class _Prepare(QObject):
    finished = Signal()

    def __init__(self, frame, runs, selection, existing=None):
        super().__init__()
        self.frame, self.runs, self.selection = frame, runs, selection
        self.existing = existing
        self.document, self.error = None, ''

    @Slot()
    def run(self):
        try:
            models = None
            required = [row.run_id.casefold() for row in self.runs]
            if self.selection is not None:
                verify_frame_model_selection(self.selection)
                models, _ = read_preview_geometry(self.selection.index_path)
                required = read_adoption_cohort(self.selection.index_path, required, self.frame.run_id)['runs']
            inputs = []
            for row in self.runs:
                key = row.run_id.casefold()
                model = (models or {}).get(key)
                photo = str(row.baseline_image or '')
                relocation = None
                from .storage_paths import same_storage_path
                if model is not None and not same_storage_path(model.reference_image.get('path', ''), photo):
                    try:
                        relocation = review.verify_photo_relocation(model.reference_image, photo)
                    except (OSError, ValueError):
                        models.pop(key)  # A different exposure cannot inherit this model's proposal.
                        model = None
                size = tuple(model.shape_hw[::-1]) if model is not None and model.shape_hw else None
                if key == self.frame.run_id.casefold() and size is None:
                    size = self.frame.image_size
                inputs.append(dict(run_id=key, path=photo, image_size=size, photograph_relocation=relocation))
            frame = review.RigFrame.from_dict(self.existing['reference']['frame']) if self.existing else self.frame
            prepared = review.prepare_review(frame, inputs, models)
            if self.existing:
                self.document = review.recover_missing_proposals(self.existing, prepared)
            else:
                self.document = prepared
                self.document['required_runs'] = required
            if self.selection is not None:
                verify_frame_model_selection(self.selection)
        except Exception as error:
            self.error = str(error)
        finally:
            self.finished.emit()


class _Align(QObject):
    finished = Signal()

    def __init__(self,project,selection,document,frame,runs):
        super().__init__()
        self.project,self.selection,self.document,self.frame,self.runs=project,selection,document,frame,runs
        self.result,self.error=None,''

    @Slot()
    def run(self):
        try:
            from .rig_point_alignment import capture_reviewed_selection
            self.result=capture_reviewed_selection(self.project,self.selection,self.document,self.frame,self.runs)
        except Exception as error:
            self.error=str(error)
        finally:
            self.finished.emit()


def _read_photo(path, profile):
    from .image_formats import image_family
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path)
    if image_family(Path(path)) == 'raw':
        return read_rig_photo(path, profile)
    from .photograph_io import load_reduced_rgb8
    rgb = load_reduced_rgb8(Path(path), 1)
    image = Image.fromarray(rgb)
    image.thumbnail((painting.DISPLAY_LIMIT_PX, painting.DISPLAY_LIMIT_PX), Image.Resampling.BOX)
    display = np.asarray(image).copy()
    return dict(image_rgb=rgb, display_plain=display,
        display_enhanced=painting.enhance_contrast(display.astype(np.float32)/255.),
        camera_u_wh=[rgb.shape[1], rgb.shape[0]], mapping=None,
        valid_support=np.ones(rgb.shape[:2], bool))


class _Point(QGraphicsEllipseItem):
    def __init__(self, point, name, moved):
        super().__init__(-6, -6, 12, 12)
        self.name, self.moved = name, moved
        self.setPos(point)
        self.setPen(QPen(QColor('#00e5ff'), 2))
        self.setBrush(QColor('#003c46'))
        self.setFlags(self.GraphicsItemFlag.ItemIsMovable | self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(30)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event):
        self._start = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.pos() != self._start:
            point = self.pos()
            QTimer.singleShot(0, lambda: self.moved(self.name, point))


class RigPointReviewPanel(QWidget):
    changed = Signal()
    return_requested = Signal()

    def __init__(self, view):
        super().__init__(view)
        from .series_layer_view import LayerCanvas
        self.view = view
        self.document = None
        self._runs = {}
        self._thread = self._worker = None
        self._alignment_thread = self._alignment_worker = None
        self._loading = False
        self._photo = None
        self._photo_key = None
        self._viewed_identity = None
        self._items = []
        self._generation = 0
        self._joining = False
        self.loader = RigPhotoLoader(reader=_read_photo)
        self.loader.ready.connect(self._loaded)
        self.destroyed.connect(self.loader.dispose)
        if QApplication.instance() is not None:
            QApplication.instance().aboutToQuit.connect(self.wait_for_workers)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.back = QPushButton('Previous')
        self.next = QPushButton('Next')
        self.return_button = QPushButton('Return to reference points')
        self.refresh = QPushButton('Prepare point proposals')
        self.use_points = QPushButton('Use approved points')
        self.use_points.setEnabled(False)
        self.summary = QLabel()
        self.run_title = QLabel()
        for item in (self.back, self.next, self.run_title, self.return_button):
            row.addWidget(item)
        layout.addLayout(row)
        self.sidebar = QWidget(view)
        sidebar = QVBoxLayout(self.sidebar)
        sidebar.setContentsMargins(0, 0, 0, 0)
        self.scope_note = QLabel()
        self.scope_note.setWordWrap(True)
        sidebar.addWidget(self.summary)
        sidebar.addWidget(self.scope_note)
        sidebar.addWidget(self.refresh)
        self.runs = QTableWidget(0, 2)
        self.runs.setHorizontalHeaderLabels(['Run', 'Status'])
        self.runs.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.runs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs.verticalHeader().hide()
        self.runs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.runs.horizontalHeader().setStretchLastSection(True)
        self.runs.setMinimumHeight(230)
        sidebar.addWidget(self.runs, 1)
        sidebar.addWidget(self.use_points)
        host = getattr(view, 'point_review_host', None)
        if host is not None:
            host.layout().addWidget(self.sidebar)
        photograph = QVBoxLayout()
        self.legend = QLabel('Yellow: original proposal · Cyan: current points. Drag cyan points, or select a row and click Place selected point.')
        self.legend.setWordWrap(True)
        visibility = QHBoxLayout()
        self.show_points = QCheckBox('Points')
        self.show_lines = QCheckBox('Lines')
        for control in (self.show_points, self.show_lines):
            control.setChecked(True)
            control.toggled.connect(self._draw)
            visibility.addWidget(control)
        visibility.addWidget(self.legend, 1)
        photograph.addLayout(visibility)
        self.canvas = LayerCanvas()
        self.canvas.set_tool('pan')
        photograph.addWidget(self.canvas, 1)
        self.points = QTableWidget(4, 4)
        self.points.setHorizontalHeaderLabels(['Reference point', 'Before [px] (row; col)', 'After [px] (row; col)', 'Moved [px]'])
        self.points.verticalHeader().hide()
        self.points.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.points.setMaximumHeight(165)
        photograph.addWidget(self.points)
        controls = QHBoxLayout()
        self.place = QPushButton('Place selected point')
        self.place.setCheckable(True)
        self.checked = QCheckBox('I checked that these are the same physical reference points.')
        self.approve = QPushButton('Approve this run')
        self.auto_next = QCheckBox('Next after approval')
        self.auto_next.setChecked(True)
        photograph.addWidget(self.checked)
        for item in (self.place, self.approve, self.auto_next):
            controls.addWidget(item)
        photograph.addLayout(controls)
        self.note = QLabel()
        self.note.setWordWrap(True)
        photograph.addWidget(self.note)
        layout.addLayout(photograph, 1)
        self.back.clicked.connect(lambda: self.runs.selectRow(self.runs.currentRow()-1))
        self.next.clicked.connect(lambda: self.runs.selectRow(self.runs.currentRow()+1))
        self.return_button.clicked.connect(self.return_requested)
        self.refresh.clicked.connect(lambda: self.prepare(restart=True))
        self.use_points.clicked.connect(self.prepare_alignment)
        self.runs.itemSelectionChanged.connect(self._selected)
        self.points.itemChanged.connect(self._edited)
        self.points.itemSelectionChanged.connect(self._update_controls)
        self.place.toggled.connect(self._place_toggled)
        self.canvas.raw_point_selected.connect(self._clicked)
        self.checked.toggled.connect(self._update_controls)
        self.approve.clicked.connect(self._approve)
        self.points.selectRow(0)
        self._update_controls()

    def busy(self):
        return self._thread is not None or self._alignment_thread is not None or self.loader.busy()

    def wait_for_workers(self):
        self._joining = True
        self._generation += 1
        self._photo_key = None
        self.loader.wait()
        if self._thread is not None:
            self._thread.wait()
            self._prepared()
        if self._alignment_thread is not None:
            self._alignment_thread.wait()
            self._aligned()
        self._joining = False

    def prepare(self, *, restart=False):
        if self.busy():
            return
        if restart:
            self.view._reviewed_point_selection=None
            self.changed.emit()
        frame = self.view.current_frame()
        if frame is None or not frame.complete() or not frame.has_size():
            self.note.setText('Define four reference points and their physical dimensions first.')
            return
        inputs = self.view.static_reference_runs()
        self._runs = {row.run_id.casefold(): row for row in inputs}
        existing = review.load_review(self.view.project)
        resume = bool(existing and review.reference_matches(existing, frame)
                and set(self._runs) <= set(existing['runs']))
        if resume:
            self.document = existing
            self._populate()
            missing = any(existing['runs'][run]['current_rc'] is None
                          and existing['runs'][run]['status'] == 'pending' for run in self._runs)
            if not missing:
                self.prepare_alignment()
                return
        try:
            preview = getattr(self.view, 'run_preview', None)
            selection = preview.geometry_selection() if preview is not None else None
        except ValueError:
            selection = None
        self._generation += 1
        # With no models, an existing manual-only draft cannot gain a new proposal.
        if resume and selection is None:
            return
        worker = _Prepare(deepcopy(frame), inputs, deepcopy(selection), deepcopy(existing) if resume else None)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._prepared)
        self._thread, self._worker = thread, worker
        self.note.setText('Preparing unapproved point proposals for the series…')
        self.refresh.setEnabled(False)
        thread.start()

    @Slot()
    def _prepared(self):
        thread, worker = self._thread, self._worker
        if thread is None:
            return
        thread.wait()
        self._thread = self._worker = None
        thread.deleteLater()
        self.refresh.setEnabled(True)
        if worker.error:
            self.note.setText(worker.error)
            return
        if not review.reference_matches(worker.document, self.view.current_frame()):
            self.note.setText('The reference changed during preparation. Prepare point proposals again.')
            return
        if worker.existing is not None and review.load_review(self.view.project) != worker.existing:
            self.note.setText('The review changed during preparation. Open the point review again.')
            return
        self.document = worker.document
        unchanged = worker.existing is not None and worker.document == worker.existing
        if (unchanged or self._save()) and not self._joining:
            self._populate()

    def _save(self):
        try:
            review.save_review(self.view.project, self.document)
        except (OSError, ValueError) as error:
            self.note.setText('Could not save point review: '+str(error))
            return False
        self.changed.emit()
        return True

    def _populate(self):
        self._loading = True
        self.runs.setRowCount(0)
        source = self.document['reference']['frame']['run_id'].casefold()
        active = [run for run in self._runs if run in self.document['runs']]
        order = ([source] if source in active else []) + [run for run in active if run != source]
        for run in order:
            index = self.runs.rowCount()
            self.runs.insertRow(index)
            item = QTableWidgetItem(run.upper())
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.runs.setItem(index, 0, item)
            self.runs.setItem(index, 1, QTableWidgetItem())
        self._loading = False
        self._refresh_states()
        if order:
            self.runs.selectRow(0)
        else:
            self._generation += 1
            self._photo_key = self._photo = self._viewed_identity = None
            self.run_title.clear()
            self.canvas.clear_photo()
            self._draw()
            self._update_controls()

    def _required_active_runs(self):
        """The active view never rewrites the frozen review's historical cohort."""
        if not self.document:
            return []
        required = self.document.get('required_runs') or list(self.document['runs'])
        return [run for run in required if run in self._runs and run in self.document['runs']]

    def _refresh_states(self):
        required = self._required_active_runs()
        count = 0
        outside = []
        for index in range(self.runs.rowCount()):
            run = self.runs.item(index, 0).data(Qt.ItemDataRole.UserRole)
            row = self.document['runs'][run]
            if row['status'] == 'approved':
                state = ('Approved directly' if row['current_rc'] == row['proposed_rc']
                         else 'Approved with changes')
            elif row['current_rc'] is None:
                state = 'No point proposal'
            else:
                state = {'pending':'Not reviewed', 'corrected':'Changed · review'}[row['status']]
            if run not in required:
                outside.append(run.upper())
                state = 'Outside transfer · ' + state
            self.runs.item(index, 1).setText(state)
            self.runs.item(index, 1).setToolTip(row['proposal_binding']['reason'] + (
                '. This run is outside the selected geometry and transfer cohort.' if run not in required else ''))
            self.runs.item(index, 1).setForeground(QColor('#45b76d' if row['status']=='approved' else '#e9b541'))
            count += run in required and row['status']=='approved'
        self.summary.setText(f'{count} of {len(required)} approved')
        self.scope_note.setText('Outside the selected transfer: '+', '.join(outside) if outside else '')
        self.scope_note.setVisible(bool(outside))
        self.use_points.setEnabled(count==len(required) and bool(required) and self._alignment_thread is None)

    def prepare_alignment(self):
        if not self.document or self._alignment_thread is not None:
            return
        required=self._required_active_runs()
        if not required or any(self.document['runs'][run]['status']!='approved' for run in required):
            return
        self.view.run_preview.ensure_saved_geometry_index()
        selection=self.view.run_preview._geometry_selection
        if selection is None:
            self.note.setText('Select the measured frame models before using the approved reference points.')
            return
        current=self.view.reviewed_point_selection(selection)
        if current is not None:
            return
        worker=_Align(deepcopy(self.view.project),deepcopy(selection),deepcopy(self.document),
            deepcopy(self.view.current_frame()),list(self._runs))
        thread=QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit,Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._aligned)
        self._alignment_thread,self._alignment_worker=thread,worker
        self.use_points.setEnabled(False)
        self.note.setText('Preparing the approved point geometry for transfer…')
        thread.start()

    @Slot()
    def _aligned(self):
        thread,worker=self._alignment_thread,self._alignment_worker
        if thread is None:
            return
        thread.wait()
        self._alignment_thread=self._alignment_worker=None
        thread.deleteLater()
        self._refresh_states()
        if worker.error:
            self.note.setText(worker.error)
        elif (worker.document==review.load_review(self.view.project)
                and worker.frame.to_dict()==self.view.current_frame().to_dict()):
            self.view._reviewed_point_selection=dict(base=worker.selection,selection=worker.result,
                document=worker.document,frame=worker.frame.to_dict())
            self.note.setText('Approved reference points are ready for transfer.')
            self.changed.emit()
        else:
            self.note.setText('The reference points changed during preparation. Use the approved points again.')
        if not self._joining and self.runs.rowCount()==0 and getattr(self.view,'_point_review_active',False):
            self._populate()

    def current_run(self):
        item = self.runs.item(self.runs.currentRow(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _selected(self):
        if self._loading or not self.document or self.current_run() is None:
            return
        run = self.current_run()
        self.run_title.setText(run.upper())
        self._generation += 1
        self._photo_key = (self._generation, run)
        self._photo = self._viewed_identity = None
        self.checked.setChecked(False)
        self.place.setChecked(False)
        self.canvas.clear_photo()
        self._draw()
        row = self.document['runs'][run]
        self.note.setText('Loading '+run.upper()+' · '+row['proposal_binding']['reason'])
        try:
            reference = self._runs[run]
            profile = painting.reference_camera_u(self.view.project, reference)
            self.loader.request(self._photo_key, Path(row['photo']['path']), profile)
        except (OSError, ValueError, KeyError) as error:
            self.note.setText(str(error))
        self._update_controls()

    @Slot(object, object, str)
    def _loaded(self, key, packet, error):
        if key != self._photo_key or key[1] != self.current_run():
            return
        if error:
            self.note.setText('Could not load the complete photograph: '+error)
            self._update_controls()
            return
        row = self.document['runs'][self.current_run()]
        actual = review.photo_binding(row['photo']['path'], row['photo'].get('image_size'))
        if actual != row['photo']:
            self.note.setText('The photograph changed. Prepare point proposals again before reviewing.')
            return
        if row['photo'].get('image_size') and tuple(row['photo']['image_size']) != tuple(packet['camera_u_wh']):
            self.note.setText('The loaded photograph has another camera grid. Prepare point proposals again.')
            return
        self._photo = packet
        self._viewed_identity = actual['identity']
        full = packet['image_rgb']
        self.canvas.set_photo(packet['display_plain'], tuple(packet['camera_u_wh']),
            token=dict(packet, run_id=self.current_run(), path=row['photo']['path']), reset_view=True,
            detail=full, display_mapping=packet.get('mapping'), display_size=(full.shape[1],full.shape[0]),
            valid_support=packet['valid_support'])
        self.canvas.paint_enabled = False
        if row['current_rc'] is None:
            self.note.setText(self._missing_proposal_message(row))
        else:
            self.note.setText('Review the same physical points. Drag cyan markers or edit After. Yellow points retain the original proposal.')
        self._draw()
        self._update_controls()

    def _missing_proposal_message(self, row):
        required = self._required_active_runs()
        if self.current_run() not in required:
            return ('This run is outside the selected geometry and transfer cohort, so it has no measured point proposal. '
                    'It is not required for this transfer. To review it separately, select each reference point and click Place selected point.')
        return (row['proposal_binding']['reason'] + '. Select each reference point and click Place selected point, '
                'or prepare proposals with geometry models that include this run.')

    def _draw(self):
        from .series_rig_frame_panel import CORNER_LABELS, _pixel_item
        from .rig_reference_points import point_description
        from .series_geometry import RigFrame
        for item in self._items:
            if item.scene() is not None:
                self.canvas.scene().removeItem(item)
        self._items = []
        if not self.document or self.current_run() is None:
            return
        row = self.document['runs'][self.current_run()]
        old, current = row['before_rc'] or {}, row['current_rc'] or {}
        if self._photo is not None and self.show_lines.isChecked():
            for points,color,style in ((old,'#ffd740',Qt.PenStyle.DashLine),(current,'#00e5ff',Qt.PenStyle.SolidLine)):
                if set(points)==set(review.POINT_NAMES):
                    outline = QGraphicsPolygonItem(QPolygonF([self.canvas.raw_to_scene(*points[name]) for name in review.POINT_NAMES]))
                    pen = QPen(QColor(color),1)
                    pen.setCosmetic(True)
                    pen.setStyle(style)
                    outline.setPen(pen)
                    outline.setZValue(27)
                    self._add_item(outline)
        frame = RigFrame.from_dict(self.document['reference']['frame'])
        self._loading = True
        for index,name in enumerate(review.POINT_NAMES):
            description = point_description(frame,name).replace('_',' ').capitalize()
            values = [QTableWidgetItem(f'{index+1} {CORNER_LABELS[name]} · {description}'),
                _pixel_item(*old[name]) if name in old else QTableWidgetItem('—'),
                _pixel_item(*current[name]) if name in current else QTableWidgetItem(),
                QTableWidgetItem(f'{np.linalg.norm(np.asarray(current[name])-old[name]):.2f}'
                    if name in old and name in current else '—')]
            for column,item in enumerate(values):
                if column != 2:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.points.setItem(index,column,item)
            if self._photo is None:
                continue
            if name in old and self.show_points.isChecked():
                point = self.canvas.raw_to_scene(*old[name])
                circle = QGraphicsEllipseItem(-8,-8,16,16)
                circle.setPos(point)
                circle.setFlag(circle.GraphicsItemFlag.ItemIgnoresTransformations)
                circle.setPen(QPen(QColor('#ffd740'),2))
                circle.setZValue(29)
                self._add_item(circle)
            if name in current and self.show_points.isChecked():
                point = self.canvas.raw_to_scene(*current[name])
                marker = _Point(point, name, self._moved)
                self._add_item(marker)
                label = QGraphicsSimpleTextItem(str(index+1))
                label.setBrush(QColor('#00e5ff'))
                label.setPos(point.x()+8,point.y()+5)
                label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations)
                label.setZValue(31)
                self._add_item(label)
                if self.show_lines.isChecked() and name in old and current[name] != old[name]:
                    before = self.canvas.raw_to_scene(*old[name])
                    line = QGraphicsLineItem(before.x(),before.y(),point.x(),point.y())
                    pen = QPen(QColor('#ffd740'),1)
                    pen.setCosmetic(True)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    line.setPen(pen)
                    line.setZValue(28)
                    self._add_item(line)
        self._loading = False

    def _add_item(self,item):
        self.canvas.scene().addItem(item)
        self._items.append(item)

    def _set_point(self,name,row,col):
        try:
            if not review.reference_matches(self.document,self.view.current_frame()):
                raise ValueError('The reference changed. Prepare point proposals again.')
            self.document = review.set_point(self.document,self.current_run(),name,row,col)
            self.checked.setChecked(False)
            self._save()
            self._refresh_states()
        except (ValueError,OSError) as error:
            self.note.setText(str(error))
        self._draw()
        self._update_controls()

    def _edited(self,item):
        if self._loading or item.column()!=2:
            return
        from .series_rig_frame_panel import _pixel_value
        try:
            point = _pixel_value(item)
            if point is not None:
                self._set_point(review.POINT_NAMES[item.row()],*point)
        except ValueError as error:
            self.note.setText(str(error))

    def _moved(self,name,point):
        if self._photo is not None:
            self._set_point(name,*self.canvas.scene_to_raw(point))

    def _place_toggled(self,checked):
        if checked:
            self.show_points.setChecked(True)
        self.canvas.place_mode = checked

    def _clicked(self,row,col):
        if self.place.isChecked() and self.points.currentRow()>=0 and self._viewed_identity is not None:
            self._set_point(review.POINT_NAMES[self.points.currentRow()],row,col)
            self.place.setChecked(False)

    def _points_visible(self):
        if self._photo is None or self.current_run() is None:
            return False
        points = self.document['runs'][self.current_run()]['current_rc'] or {}
        if set(points)!=set(review.POINT_NAMES):
            return False
        support = self._photo['valid_support']
        for row,col in points.values():
            mapping = self._photo.get('mapping')
            x,y = mapping.camera_u_to_display_xy([[col,row]])[0] if mapping is not None else (col,row)
            x,y = int(round(x)),int(round(y))
            if not (0<=y<support.shape[0] and 0<=x<support.shape[1] and support[y,x]):
                return False
        return True

    def _update_controls(self):
        loaded = self._viewed_identity is not None and self._photo is not None
        self.place.setEnabled(loaded and self.points.currentRow()>=0)
        self.checked.setEnabled(loaded)
        ready = loaded and self.checked.isChecked() and self._points_visible()
        self.approve.setEnabled(bool(ready))
        self.approve.setToolTip('All four points must be visible in this photograph before approval.')
        self.back.setEnabled(self.runs.currentRow()>0)
        self.next.setEnabled(0<=self.runs.currentRow()<self.runs.rowCount()-1)

    def _approve(self):
        if not self.approve.isEnabled():
            return
        try:
            frame = self.view.current_frame()
            if not review.reference_matches(self.document,frame):
                raise ValueError('The reference changed. Prepare point proposals again.')
            from .series_decisions import project_series_root, series_actor
            actor = series_actor(project_series_root(self.view.project))
            document = review.approve_run(self.document,self.current_run(),viewed_identity=self._viewed_identity,actor=actor)
            approved = review.reviewed_frame(document,self.current_run())
            save_rig_frame(self.view.project,approved,actor=actor)
            self.document = document
            if not self._save():
                return
            if self.current_run()==frame.run_id.casefold():
                self.view.rig_panel.load()
            self._refresh_states()
            self.note.setText('Approved and saved. Original proposed points and manual corrections are retained.')
            self.prepare_alignment()
            if self.auto_next.isChecked() and self.runs.currentRow()+1<self.runs.rowCount():
                self.runs.selectRow(self.runs.currentRow()+1)
        except (ValueError,OSError) as error:
            self.note.setText(str(error))
