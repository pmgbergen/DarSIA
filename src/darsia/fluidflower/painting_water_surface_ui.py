"""Explicit human confirmation of a source-bound, display-only water line."""
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


class WaterSurfaceControls(QWidget):
    """Consume the pure producer's packet; viewing never confirms a line."""
    line_changed = Signal()
    pick_requested = Signal(bool)
    confirmed = Signal(object)
    basis_ready = Signal(object)
    tolerances_changed = Signal(object)
    line_edited = Signal(str)

    def __init__(self, *, actor, parent=None, api=None):
        super().__init__(parent)
        self.actor, self._api = str(actor or '').strip(), api
        self.packet = self.action = self.measurements = self.line_photo_xy = None
        self.photo_to_display_h = None
        self._active = False
        self._manual_points=[]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.outline = QCheckBox('Show water line and bolt-height ruler')
        self.outline.setChecked(True)
        self.outline.toggled.connect(lambda *_: self.line_changed.emit())
        row.addWidget(self.outline)
        self.move = QPushButton('Move the water line with one click')
        self.move.setCheckable(True)
        self.move.setAutoDefault(False)
        self.move.toggled.connect(self._move_toggled)
        row.addWidget(self.move)
        self.manual = QPushButton('Set the water line with two points')
        self.manual.setCheckable(True)
        self.manual.setAutoDefault(False)
        self.manual.toggled.connect(self._manual_toggled)
        row.addWidget(self.manual)
        self.confirm = QPushButton('Confirm water line')
        self.confirm.setAutoDefault(False)
        self.confirm.clicked.connect(self._confirm)
        row.addWidget(self.confirm)
        layout.addLayout(row)
        self.status = QLabel('Water line: a verified clear-water photograph and measurement basis are missing.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.readout = QLabel('No confirmed water line. The physical frame and scale stay unchanged.')
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        self.origins=QLabel('Own geometry starts at the inner left-wall and bottom intersection. Bolt height starts at the observed bottom-left bolt centre.')
        self.origins.setWordWrap(True)
        layout.addWidget(self.origins)
        row = QHBoxLayout()
        self.tolerance_label=QLabel('Display warning tolerances from the confirmed series:')
        row.addWidget(self.tolerance_label)
        self.height_tolerance, self.angle_tolerance = QDoubleSpinBox(), QDoubleSpinBox()
        for control, value, suffix in ((self.height_tolerance, 3., ' mm'), (self.angle_tolerance, .5, ' degrees')):
            control.setDecimals(3)
            control.setRange(0., 1000000.)
            control.setSuffix(suffix)
            control.setValue(value)
            row.addWidget(control)
        self.apply = QPushButton('Apply water-line tolerances')
        self.apply.setAutoDefault(False)
        self.apply.clicked.connect(self._tolerances)
        row.addWidget(self.apply)
        layout.addLayout(row)
        self._buttons()

    def _producer(self):
        if self._api is None:
            from . import painted_water_surface
            self._api = painted_water_surface
        return self._api

    @staticmethod
    def _displayable(points, matrix):
        points=np.asarray(points,np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('Invalid photograph points in the water-line display')
        mapped=np.column_stack((points,np.ones(len(points)))) @ matrix.T
        if not np.isfinite(mapped).all() or np.any(mapped[:,2]==0):
            raise ValueError('The water line cannot be shown in the exact photograph plane')
        if not np.isfinite(mapped[:,:2]/mapped[:,2:3]).all():
            raise ValueError('The water line lacks finite display coordinates')

    def set_packet(self, packet, *, display_grid, active):
        """The explicit map joins basis-photo pixels to the visible photo plane."""
        self.packet = self.action = self.measurements = self.line_photo_xy = None
        self.photo_to_display_h = None
        self._active = False
        self.move.setChecked(False)
        self.manual.setChecked(False)
        self.status.setText('Water line: a verified clear-water photograph and measurement basis are missing.')
        self.readout.setText('No confirmed water line. The physical frame and scale stay unchanged.')
        try:
            if packet is not None:
                owned = deepcopy(packet)
                matrix = np.asarray(owned['photo_to_display_h'], np.float64)
                if (display_grid is None or owned['display_grid'] != display_grid or matrix.shape != (3, 3)
                        or not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0):
                    raise ValueError('An explicit water-line photograph to display-plane binding is missing')
                # Admission and measurements belong to the original pure module.
                basis = owned['basis']
                line = owned['proposal']['line_photo_xy']
                if line is not None:
                    coordinates=np.asarray(line,np.float64)
                    if coordinates.shape != (2,2) or not np.isfinite(coordinates).all() or np.array_equal(coordinates[0],coordinates[1]):
                        raise ValueError('The proposal lacks two finite, distinct photograph endpoints')
                    line=coordinates.tolist()
                    self._displayable(line,matrix)
                if basis.get('left_bolts_photo_xy') is not None:
                    self._displayable(basis['left_bolts_photo_xy'],matrix)
                self._producer()  # Fail visibly if this optional capability is absent.
                self.packet, self.photo_to_display_h = owned, matrix.copy()
                self.line_photo_xy = deepcopy(line)
                stale=False
                if owned.get('action') is not None:
                    try:
                        self.measurements = self._producer().water_surface_measurements(
                            owned['action'], expected_basis=basis)
                        self.action = deepcopy(owned['action'])
                        self._displayable(self.action['line_photo_xy'],matrix)
                        self.line_photo_xy = deepcopy(self.action['line_photo_xy'])
                    except (KeyError,TypeError,ValueError):
                        stale=True
                        self.action=self.measurements=None
                self._active = bool(active)
                self.basis_ready.emit(deepcopy(basis))
                self._readout()
                if stale:
                    self.status.setText('The previous confirmation belongs to another photograph or measurement basis. Confirm the new proposal yourself.')
                elif not self._active:
                    self.status.setText(self.status.text()+' Select the bound clear-water photograph to confirm or move the line.')
        except (ImportError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
            self.packet = self.action = self.measurements = self.line_photo_xy = None
            self.photo_to_display_h = None
            self.status.setText('The water line is unavailable: ' + str(error))
        self._buttons()
        self.line_changed.emit()

    def set_active(self, active):
        self._active = bool(active)
        if not self._active:
            self.move.setChecked(False)
            self.manual.setChecked(False)
        self._buttons()

    def _buttons(self):
        available=self.packet is not None
        for control in (self.outline,self.move,self.manual,self.confirm,self.readout,self.origins,
                        self.tolerance_label,self.height_tolerance,self.angle_tolerance,self.apply):
            control.setVisible(available)
        ready = self._active and self.packet is not None and self.line_photo_xy is not None
        self.move.setEnabled(ready and self.packet['basis'].get('photo_to_metric_h') is not None)
        self.manual.setEnabled(self._active and self.packet is not None)
        self.confirm.setEnabled(ready and bool(self.actor) and not self.manual.isChecked() and not self.move.isChecked())
        self.apply.setEnabled(self.packet is not None)
        if not self.move.isEnabled():
            self.move.setChecked(False)

    def _move_toggled(self, armed):
        if armed:
            self.manual.setChecked(False)
            self.status.setText('Click one point on the visible water line. Its angle in the existing metric plane stays unchanged.')
        if armed or not self.manual.isChecked():
            self.pick_requested.emit(bool(armed))
        self._buttons()

    def _manual_toggled(self, armed):
        self._manual_points=[]
        if armed:
            self.move.setChecked(False)
            self.status.setText('Click the first and then the second endpoint of the water line. This approves nothing.')
        if armed or not self.move.isChecked():
            self.pick_requested.emit(bool(armed))
        self._buttons()

    def select_photo_point(self, point_photo_xy):
        if not self.manual.isChecked():
            self.move_to_photo_point(point_photo_xy)
            return
        try:
            point=np.asarray(point_photo_xy,np.float64)
            height,width=self.packet['basis']['photo_shape']
            if point.shape != (2,) or not np.isfinite(point).all() or not (0 <= point[0] <= width-1 and 0 <= point[1] <= height-1):
                raise ValueError('Choose an actual point inside the bound clear-water photograph; coordinates are not clipped')
            if self._manual_points and np.array_equal(point,self._manual_points[0]):
                raise ValueError('The two endpoints must be distinct')
            if self._manual_points:
                self._displayable([self._manual_points[0],point.tolist()],self.photo_to_display_h)
            self._manual_points.append(point.tolist())
            if len(self._manual_points)==1:
                self.status.setText('First endpoint selected. Click the second, then confirm the water line yourself.')
                self.pick_requested.emit(True)
            else:
                line=deepcopy(self._manual_points)
                self.manual.setChecked(False)
                self.line_photo_xy=line
                self.action=self.measurements=None
                self._readout()
                self.line_changed.emit()
                self.line_edited.emit(self.packet['basis']['run_id'])
        except (KeyError,TypeError,ValueError) as error:
            self.status.setText('The endpoint could not be used: '+str(error))
            self.pick_requested.emit(True)
        self._buttons()

    def move_to_photo_point(self, point_photo_xy):
        if not self.move.isEnabled() or not self.move.isChecked():
            return
        self.move.setChecked(False)
        try:
            point=np.asarray(point_photo_xy,np.float64)
            height,width=self.packet['basis']['photo_shape']
            if point.shape != (2,) or not np.isfinite(point).all() or not (0 <= point[0] <= width-1 and 0 <= point[1] <= height-1):
                raise ValueError('Choose a point inside the bound clear-water photograph; coordinates are not clipped')
            line = self._producer().move_water_surface_to_photo_point(
                deepcopy(self.line_photo_xy), list(point_photo_xy), basis=deepcopy(self.packet['basis']))
            self._displayable(line,self.photo_to_display_h)
            self.line_photo_xy=deepcopy(line)
            self.action = self.measurements = None
            self._readout()
            self.line_changed.emit()
            self.line_edited.emit(self.packet['basis']['run_id'])
        except (KeyError, TypeError, ValueError) as error:
            self.status.setText('The water line could not be moved: ' + str(error))
        self._buttons()

    def _confirm(self):
        if not self.confirm.isEnabled():
            return
        try:
            basis = deepcopy(self.packet['basis'])
            action = self._producer().make_water_surface_action(basis, deepcopy(self.line_photo_xy),
                actor=self.actor, confirmed_utc=datetime.now(timezone.utc).isoformat())
            measured = self._producer().water_surface_measurements(action, expected_basis=basis)
            self.action, self.measurements = deepcopy(action), deepcopy(measured)
            self._readout()
            self.confirmed.emit(dict(basis=basis, action=deepcopy(action), measurements=deepcopy(measured)))
        except (KeyError, TypeError, ValueError) as error:
            self.status.setText('The water line could not be confirmed: ' + str(error))
        self._buttons()

    def _readout(self):
        if self.action is None:
            self.status.setText('Proposed water line – unconfirmed. Moving it changes only the proposal.'
                if self.line_photo_xy is not None else 'No water line was proposed in this photograph.')
            self.readout.setText('No confirmed water line. The physical frame and scale stay unchanged.')
            return
        result = self.measurements
        def value(key, unit):
            return 'unavailable' if result.get(key) is None else f'{result[key]:.3f} {unit}'
        location = {'left-bolt-column':'left bolt column', 'bound-port-column':'the bound port column'}.get(result.get('bolt_height_location'),'unknown measurement column')
        lines = [f"Calculated height from own geometry (inner left-wall and bottom intersection): {value('height_mm', 'mm')}",
                 f"Bolt height from the observed bottom-left bolt centre (0/195/390 mm), at {location}: {value('bolt_height_mm', 'mm')}",
                 f"Angle relative to the bottom: {value('angle_degrees', 'degrees')}",
                 f"Own calculated top minus water line: {value('own_top_minus_water_mm', 'mm')}",
                 f"Transferred AC17 top minus water line: {value('transported_top_minus_water_mm', 'mm')}"]
        if result.get('ruler_extrapolated'):
            lines.append('The bolt height lies beyond the observed 0/195/390 mm bolts (extrapolated).')
        if result.get('unavailable'):
            lines.append('Unavailable measurement basis: ' + ', '.join(result['unavailable']))
        self.readout.setText('\n'.join(lines) + '\nDisplay and review only; physical geometry stays unchanged.')
        self.status.setText('You have confirmed this water line on the exact bound photograph.')

    def _tolerances(self):
        if self.apply.isEnabled():
            self.tolerances_changed.emit(dict(height_tolerance_mm=self.height_tolerance.value(),
                angle_tolerance_degrees=self.angle_tolerance.value()))
