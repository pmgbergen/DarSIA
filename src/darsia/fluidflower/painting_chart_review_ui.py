"""Per-run chart proposals on the original saved photograph, with explicit approval."""
from copy import deepcopy
from threading import Event
import time
import numpy as np
from PySide6.QtCore import QEvent, QRectF, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPen, QBrush, QPainterPath
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QCheckBox, QGraphicsPathItem, QGraphicsView, QMessageBox, QProgressBar,
    QScrollArea, QSizePolicy, QFrame)

from .painting_chart_transfer import MOVED_RUNS, confirm_chart_transfer, _validate_approval
from .painting_gap_review_ui import _Read, _Photo, _rgb, _gap_contours, _path


def read_chart_preview(review, run):
    """Use the full same-exposure camera display and saved continuation mask."""
    from .painted_camera_jpg_display import read_transfer_preview
    from .painted_analysis_exclusions import _arrays
    packet = read_transfer_preview(review, run)
    row = next(row for row in review['runs'] if row['run_id'] == run)
    packet['continued_mask'] = _arrays(row['product'])['continued']
    return packet


class ChartReviewDialog(QWidget):
    """One finite reader at a time; viewing and navigation never approve a run."""
    approvals_changed = Signal(object)
    reuse_requested = Signal(object)
    proposal_ready = Signal(object)
    review_loaded = Signal()
    run_changed = Signal(str)
    bulk_progress = Signal(int, int)

    def __init__(self, source, *, actor, parent=None, session_factory=None,
                 preview_reader=None, review_validator=None, excluded_runs=None,
                 registered_runs=None):
        super().__init__(parent)
        from .painting_chart_transfer_reader import ChartTransferSession
        self.actor = actor
        self._factory = session_factory or ChartTransferSession
        self._preview_reader = preview_reader or read_chart_preview
        self._validator = review_validator
        self.session = self.proposal = self.packet = None
        self.approvals, self.adjustments, self.confirmations = {}, {}, {}
        self._approval_store = None
        self._approval_payloads = {}
        self._save_error = ''
        self.display_options = {}
        self.expected_runs = set()
        self._thread = self._worker = None
        self._closing = False
        self._overlays = []
        self._photo_run = None
        self._photo_grid = None
        self._generation = 0
        self._pending_request = self._active_request = None
        self._sidebar_navigation = False
        self._bulk_approval_active = False
        self._bulk_context = None
        self._bulk_cancel = None
        self.bulk_progress.connect(self._show_bulk_progress)
        self._adjust_timer = QTimer(self)
        self._adjust_timer.setSingleShot(True)
        self._adjust_timer.setInterval(200)
        self._adjust_timer.timeout.connect(self._dispatch_pending)
        self.setObjectName('transferReviewPanel')
        self.setWindowTitle('Review the transfer — ColorChecker masks')
        layout = QVBoxLayout(self)
        introduction = QLabel('Review every adoption run on its own saved photograph. '
            'Changed proposals follow the detected shift with a fixed 5 mm margin. '
            'Other masks stay unchanged unless you adjust them. Approve every run before saving.')
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        navigation = QHBoxLayout()
        self.back = QPushButton('Previous')
        self.runs = QComboBox()
        self.run_title = QLabel()
        self.run_title.setStyleSheet('font-weight: 600;')
        self.run_title.hide()
        self.next = QPushButton('Next')
        for widget in (self.back, self.runs, self.run_title, self.next):
            navigation.addWidget(widget)
        navigation.setStretch(1, 1)
        navigation.setStretch(2, 1)
        self.photo_view = QComboBox()
        self.photo_view.addItem('Full photograph', 'full')
        self.photo_view.addItem('Rig area', 'rig')
        self.photo_view.model().item(1).setEnabled(False)
        self.photo_view.setToolTip('Display only. Rig area uses the same fixed area for every run.')
        self.photo_view.currentIndexChanged.connect(self._photo_view_changed)
        navigation.addWidget(self.photo_view)
        layout.addLayout(navigation)
        self.counter = QLabel('0 approved')
        layout.addWidget(self.counter)
        excluded = excluded_runs or []
        self.excluded_runs = deepcopy(excluded)
        self._excluded_run = None
        self.excluded = QLabel('Outside adoption: ' + '; '.join(
            f"{row['run_id'].upper()}: {row['reason']}" for row in excluded) if excluded else '')
        self.excluded.setWordWrap(True)
        layout.addWidget(self.excluded)
        self.photo = _Photo()
        self.photo.setMinimumSize(0, 80)
        self.photo.auto_fit_on_resize = True
        self.photo.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        layout.addWidget(self.photo, 1)
        self._loading_photo_run = None
        self._photo_load_started = None
        self.photo_loading = QWidget(self.photo.viewport())
        self.photo_loading.setStyleSheet('background: #252525; color: white;')
        loading_layout = QVBoxLayout(self.photo_loading)
        loading_layout.addStretch(1)
        self.photo_loading_text = QLabel()
        self.photo_loading_text.setWordWrap(True)
        self.photo_loading_text.setTextFormat(Qt.TextFormat.PlainText)
        self.photo_loading_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.photo_loading_text)
        self.photo_loading_bar = QProgressBar()
        self.photo_loading_bar.setRange(0, 0)
        loading_layout.addWidget(self.photo_loading_bar)
        loading_layout.addStretch(1)
        self.photo_loading.hide()
        self.photo.viewport().installEventFilter(self)
        self._photo_load_timer = QTimer(self)
        self._photo_load_timer.setInterval(1000)
        self._photo_load_timer.timeout.connect(self._update_photo_loading)
        self.excluded_notice = QLabel()
        self.excluded_notice.setWordWrap(True)
        self.excluded_notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.excluded_notice.hide()
        layout.addWidget(self.excluded_notice, 1)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_scroll.setMinimumSize(0, 90)
        self.controls_scroll.setMaximumHeight(260)
        self.controls_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.controls_panel = QWidget()
        control_layout = QVBoxLayout(self.controls_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_scroll.setWidget(self.controls_panel)
        layout.addWidget(self.controls_scroll)
        self.motion = QLabel()
        self.motion.setWordWrap(True)
        control_layout.addWidget(self.motion)
        overlays = QGridLayout()
        self.overlays = {}
        for key, title, default in (('sand', 'Sand layers', True), ('chart', 'ColorChecker', True),
                ('continued', 'Continued area', False), ('rig_frame', 'Rig frame', True),
                ('silicone', 'Silicone', True),
                ('gaps', 'Gaps', True)):
            control = QCheckBox(title)
            control.setChecked(default)
            self.overlays[key] = control
            control.toggled.connect(self._overlay_changed)
            index = len(self.overlays) - 1
            overlays.addWidget(control, index // 3, index % 3)
        self.overlays['rig_frame'].setToolTip('Saved rig boundary used for coverage and unmapped-gap checks. '
            'It does not trace the sand layers. This display control does not change the analysis.')
        self.overlays['silicone'].setToolTip('Saved silicone/excluded-material outlines. '
            'This display control does not change their analysis masks.')
        control_layout.addLayout(overlays)
        controls = QGridLayout()
        self.right, self.up = QDoubleSpinBox(), QDoubleSpinBox()
        for row, (label, control) in enumerate((('ColorChecker right (mm)', self.right), ('ColorChecker up (mm)', self.up))):
            controls.addWidget(QLabel(label), row, 0)
            control.setRange(-500., 500.)
            control.setDecimals(2)
            control.setSingleStep(.5)
            control.setMinimumWidth(80)
            controls.addWidget(control, row, 1)
            control.valueChanged.connect(self._edited)
        self.preview = QPushButton('Update proposal')
        self.preview.clicked.connect(self._load_run)
        controls.addWidget(self.preview, 0, 2)
        self.hatching = QCheckBox('Hatch excluded ColorChecker')
        self.hatching.setChecked(False)
        self.hatching.toggled.connect(self._overlay_changed)
        controls.addWidget(self.hatching, 1, 2)
        control_layout.addLayout(controls)
        self.sand_valid = QCheckBox('The sand-layer map outlines are valid in this photograph')
        self.chart_valid = QCheckBox('The ColorChecker outline is correct in this photograph')
        for control in (self.sand_valid, self.chart_valid):
            control.toggled.connect(self._checks_changed)
            control_layout.addWidget(control)
        actions = QGridLayout()
        self.approve = QPushButton('Approve this run')
        self.approve.clicked.connect(self._approve)
        self.approve_all = QPushButton('Approve all runs…')
        self.approve_all.setToolTip('Confirm the sand-layer and ColorChecker outlines for every run, '
            'using its current adjustment. Existing valid approvals are kept.')
        self.approve_all.clicked.connect(self._approve_all)
        self.reuse = QPushButton('Use approved masks with the saved maps…')
        self.reuse.clicked.connect(self._reuse)
        actions.addWidget(self.approve, 0, 0)
        actions.addWidget(self.approve_all, 0, 1)
        actions.addWidget(self.reuse, 1, 0, 1, 2)
        control_layout.addLayout(actions)
        self.status = QLabel('Loading the saved map and ColorChecker detections…')
        self.status.setWordWrap(True)
        control_layout.addWidget(self.status)
        self.back.clicked.connect(lambda: self.runs.setCurrentIndex(max(0, self.runs.currentIndex()-1)))
        self.next.clicked.connect(lambda: self.runs.setCurrentIndex(min(self.runs.count()-1, self.runs.currentIndex()+1)))
        self.runs.currentIndexChanged.connect(self._selected)
        def load_session():
            session = self._factory(source)
            if registered_runs is not None:
                from .painting_frame_models import read_adoption_cohort
                from .storage_paths import same_storage_descriptor
                review = session.review
                cohort = read_adoption_cohort(review['geometry_index']['path'],
                    registered_runs, review['source']['reference_run'])
                if (not same_storage_descriptor(cohort['geometry_index'], review['geometry_index'])
                        or cohort['runs'] != [row['run_id'] for row in review['runs']]):
                    raise ValueError('The saved review belongs to another selected adoption cohort.')
                session.excluded_runs = cohort['excluded_runs']
            from .painting_chart_approval_store import ChartApprovalStore
            session._approval_store = ChartApprovalStore.for_session(session)
            session._approval_state = (session._approval_store.load() if session._approval_store is not None
                else dict(approvals={}, payloads={}, adjustments={}, confirmations={}))
            return session
        self._start('session', load_session)

    def busy(self):
        return self._thread is not None or self._pending_request is not None or self._adjust_timer.isActive()

    @property
    def bulk_approval_active(self):
        return self._bulk_approval_active

    def set_sidebar_navigation(self, enabled):
        """Use the host's run/status list instead of a second run selector."""
        self._sidebar_navigation = bool(enabled)
        self.runs.setVisible(not enabled)
        self.run_title.setVisible(enabled)
        self.counter.setVisible(not enabled)
        self.excluded.setVisible(not enabled)
        self._update_run_title()

    def _update_run_title(self):
        run = self._excluded_run or self.runs.currentData()
        state = ('Excluded' if self._excluded_run else
            'Loading photograph…' if self._loading_photo_run == run and self._photo_load_timer.isActive() else
            'Photograph unavailable' if self._loading_photo_run == run else
            'Approved' if run in self.approvals else 'Waiting for approval')
        self.run_title.setText(f'{str(run).upper()} · {state}' if run else 'Choose a run in the list')

    def eventFilter(self, watched, event):
        if watched is self.photo.viewport() and event.type() == QEvent.Type.Resize:
            self.photo_loading.setGeometry(self.photo.viewport().rect())
        return super().eventFilter(watched, event)

    def _show_photo_loading(self, run):
        if self._loading_photo_run != run:
            self._photo_load_started = time.monotonic()
        self._loading_photo_run = run
        self.photo_loading.setGeometry(self.photo.viewport().rect())
        self.photo_loading_bar.show()
        self.photo_loading.show()
        self.photo_loading.raise_()
        self._photo_load_timer.start()
        self._update_photo_loading()

    def _update_photo_loading(self):
        if self._loading_photo_run is None:
            return
        elapsed = max(0, int(time.monotonic() - self._photo_load_started))
        self.photo_loading_text.setText(f'Loading {self._loading_photo_run.upper()} photograph…\n'
            f'Elapsed: {elapsed // 60}:{elapsed % 60:02d}')

    def _finish_photo_loading(self, error=None):
        self._photo_load_timer.stop()
        if error is not None and self._loading_photo_run is not None:
            self.photo_loading_bar.hide()
            self.photo_loading_text.setText(f'Could not load {self._loading_photo_run.upper()} photograph.\n{error}')
            return
        self.photo_loading.hide()
        self._loading_photo_run = self._photo_load_started = None

    def _buttons(self):
        ready = not self._closing and self.session is not None
        idle = ready and not self.busy()
        for control in (self.runs, self.right, self.up, self.preview):
            control.setEnabled(ready and not self.bulk_approval_active)
        if self._excluded_run is not None:
            for control in (self.right, self.up, self.preview):
                control.setEnabled(False)
        self.back.setEnabled(ready and not self.bulk_approval_active and self.runs.currentIndex() > 0)
        self.next.setEnabled(ready and not self.bulk_approval_active and self.runs.currentIndex() < self.runs.count()-1)
        for control in (self.sand_valid, self.chart_valid):
            control.setEnabled(idle and self.proposal is not None)
        self.approve.setEnabled(idle and self.proposal is not None
            and self.sand_valid.isChecked() and self.chart_valid.isChecked())
        self.approve_all.setEnabled(idle and bool(self.expected_runs)
            and set(self.approvals) != self.expected_runs)
        self.reuse.setEnabled(idle and not self._save_error and bool(self.expected_runs) and set(self.approvals) == self.expected_runs)
        self.counter.setText(f'{len(self.approvals)} of {len(self.expected_runs)} approved')
        if self._save_error:
            self.status.setText('Chart decisions could not be saved: ' + self._save_error)
        self._update_run_title()

    def _start(self, operation, function):
        if self._thread is not None or self._closing:
            return
        self._operation = operation
        self._thread = QThread(self)
        self._worker = _Read(function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._retired)
        self._buttons()
        self._thread.start()

    @Slot()
    def _retired(self):
        if self._thread is None:
            return
        thread, worker, operation = self._thread, self._worker, self._operation
        thread.wait()
        self._thread = self._worker = None
        thread.deleteLater()
        worker.deleteLater()
        if operation == 'approve_all':
            self._finish_bulk_approval(worker)
            return
        if self._closing:
            return
        if operation == 'proposal' and self._active_request[0] != self._generation:
            self._buttons()
            if not self._adjust_timer.isActive():
                self._dispatch_pending()
            return
        try:
            if worker.error:
                raise ValueError(worker.error)
            if operation == 'session':
                if self._validator is not None:
                    self._validator(worker.result.review)
                self.session = worker.result
                self._approval_store = self.session._approval_store
                saved = self.session._approval_state
                self.approvals = deepcopy(saved['approvals'])
                self._approval_payloads = deepcopy(saved['payloads'])
                self.adjustments = deepcopy(saved['adjustments'])
                self.confirmations = deepcopy(saved['confirmations'])
                if registered_runs := getattr(self.session, 'excluded_runs', None):
                    self.excluded_runs = deepcopy(registered_runs)
                    self.excluded.setText('Outside adoption: ' + '; '.join(
                        f"{row['run_id'].upper()}: {row['reason']}" for row in registered_runs))
                self.expected_runs = {row['run_id'] for row in self.session.review['runs']}
                self.runs.blockSignals(True)
                for row in self.session.review['runs']:
                    run = row['run_id']
                    self.runs.addItem(run.upper() + (' — changed proposal' if run in MOVED_RUNS else ''), run)
                self.runs.blockSignals(False)
                self.review_loaded.emit()
                self.approvals_changed.emit(deepcopy(self.approvals))
                self._selected()
                return
            self.proposal, self.packet = worker.result
            metadata = self.proposal.validate()
            if metadata['run_id'] != self.runs.currentData():
                raise ValueError('The ColorChecker preview belongs to another selected run.')
            if self.packet.get('run_id') != metadata['run_id']:
                raise ValueError('The photograph belongs to another selected run.')
            self._draw()
            self.proposal_ready.emit(metadata)
            shift = metadata['motion']
            self.motion.setText(f'Detected: {shift["dx_mm"]:+.2f} mm right, '
                f'{shift["dy_up_mm"]:+.2f} mm up. '
                f'ColorChecker adjustment: {metadata["correction_mm"][0]:+.2f} mm right, '
                f'{metadata["correction_mm"][1]:+.2f} mm up. '
                f'Applied rotation: {metadata.get("applied_motion", {}).get("rotation_degrees", 0.):+.2f}°. '
                + ('Changed proposal. Margin: 5 mm. ' if metadata['moved'] else 'Original mask unchanged. Adjustment adds a 5 mm margin. ')
                +
                'Yellow: previous ColorChecker. Cyan: proposed ColorChecker. '
                + metadata['source_binding'].get('detection', {}).get('basis', ''))
            self.motion.setStyleSheet('')
            self.status.setText(f'{len(self.approvals)} of {len(self.expected_runs)} approved. '
                + ('This run is approved.' if metadata['run_id'] in self.approvals
                   else 'Check both outlines, then approve this run.'))
        except Exception as error:
            self.proposal = None
            self._finish_photo_loading(error)
            self.status.setText(str(error))
        self._buttons()
        if self._pending_request is not None and not self._adjust_timer.isActive():
            self._dispatch_pending()

    def _selected(self):
        self._excluded_run = None
        self.excluded_notice.hide()
        self.photo.show()
        self.motion.show()
        defaults = dict(sand=True, chart=True, continued=False, rig_frame=True,
            silicone=True, gaps=True, hatching=False)
        values = self.display_options.get(self.runs.currentData(), defaults)
        for key, control in {**self.overlays, 'hatching': self.hatching}.items():
            control.blockSignals(True)
            control.setChecked(values[key])
            control.blockSignals(False)
        values = self.adjustments.get(self.runs.currentData(), [0., 0.])
        for control, value in zip((self.right, self.up), values):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self._set_checks(self.confirmations.get(self.runs.currentData(), [False, False]))
        self._load_run()
        self.run_changed.emit(str(self.runs.currentData()))

    def show_excluded_run(self, run, reason):
        self._cancel_bulk_approval()
        self._excluded_run = run
        self._generation += 1
        self._pending_request = None
        self._adjust_timer.stop()
        self.proposal = self.packet = None
        self._finish_photo_loading()
        self.runs.blockSignals(True)
        self.runs.setPlaceholderText(run.upper() + ' — excluded')
        self.runs.setCurrentIndex(-1)
        self.runs.blockSignals(False)
        self.photo.hide()
        self.motion.hide()
        self.excluded_notice.setText(run.upper() + ' — excluded from this review.\n' + reason
                                     + '\nNo photograph is available in the selected review.')
        self.excluded_notice.show()
        self.status.setText('This run is outside the selected adoption cohort.')
        self._set_checks([False, False])
        self._buttons()
        self.run_changed.emit(run)

    def _set_checks(self, values):
        for control, value in zip((self.sand_valid, self.chart_valid), values):
            control.blockSignals(True)
            control.setChecked(value)
            control.blockSignals(False)

    def _checks_changed(self):
        self._cancel_bulk_approval()
        run = self.runs.currentData()
        if run not in self.expected_runs:
            return
        values = [self.sand_valid.isChecked(), self.chart_valid.isChecked()]
        self.confirmations[run] = values
        if not all(values) and run in self.approvals:
            self.approvals.pop(run)
            self._approval_payloads.pop(run, None)
            self.approvals_changed.emit(deepcopy(self.approvals))
        self._persist_changes()
        self._buttons()

    def _overlay_changed(self):
        self.display_options[self.runs.currentData()] = {
            key: control.isChecked() for key, control in {**self.overlays, 'hatching': self.hatching}.items()}
        self._draw()

    def _edited(self):
        run = self.runs.currentData()
        self.adjustments[run] = [self.right.value(), self.up.value()]
        self.confirmations[run] = [False, False]
        self._set_checks([False, False])
        self.approvals.pop(run, None)
        self._approval_payloads.pop(run, None)
        self.proposal = None
        self.overlays['chart'].setChecked(True)
        self.approvals_changed.emit(deepcopy(self.approvals))
        self._persist_changes()
        self._queue_proposal(immediate=False)

    def _load_run(self):
        self._queue_proposal(immediate=True)

    def _queue_proposal(self, *, immediate):
        if self.session is None or self._closing:
            return
        self._cancel_bulk_approval()
        self.proposal = None
        run, correction = self.runs.currentData(), [self.right.value(), self.up.value()]
        self._generation += 1
        self._pending_request = (self._generation, run, correction)
        if self._photo_run != run:
            self.packet = None
            # Preserve the readout's height so loading cannot shift the viewport.
            self.motion.setStyleSheet('color: transparent;')
            self._show_photo_loading(run)
            self.status.setText(f'Loading {run.upper()} photograph and transferred outlines…')
        else:
            self.status.setText(f'{run.upper()}: updating the ColorChecker at '
                f'{correction[0]:+.2f} mm right, {correction[1]:+.2f} mm up…')
        self._adjust_timer.stop()
        if immediate:
            self._dispatch_pending()
        else:
            self._adjust_timer.start()
        self._buttons()

    def _dispatch_pending(self):
        if self._thread is not None or self._closing or self._pending_request is None:
            return
        self._active_request, self._pending_request = self._pending_request, None
        _generation, run, correction = self._active_request
        packet = self.packet if self.packet is not None and self.packet.get('run_id') == run else None
        session = self.session
        approval = deepcopy(self.approvals.get(run))
        def read():
            if approval is not None and list(approval.get('correction_mm', [])) == list(correction):
                from .painting_chart_transfer_reader import proposal_for_approval
                proposal = proposal_for_approval(session, run, approval)
                _validate_approval(proposal.validate(), proposal.payload, approval)
            else:
                proposal = session.proposal(run, correction_mm=correction)
            current = packet if packet is not None else self._preview_reader(session.review, run)
            from .painting_chart_transfer import DETECTED_FORMAT
            if ((session.review.get('transfer_policy') is not None
                    or getattr(session, 'decision_proposal_format', None) == DETECTED_FORMAT)
                    and (current.get('full_camera_photograph') is not True
                         or not hasattr(session, 'restored'))):
                raise ValueError('New transfers require the complete photograph and original source map; '
                                 'the legacy cropped preview is not supported.')
            if (current.get('full_camera_photograph') is True
                    and 'material_display' not in current and hasattr(session, 'restored')):
                from .painting_transfer_overlays import full_material_display
                current = dict(current)
                current['material_display'] = full_material_display(
                    session.restored['material_labels'], proposal.metadata()['source_to_native_h'],
                    current['native_to_photo_h'], current['own_frame_native_xy'],
                    current['image_rgb'].shape[:2])
            return proposal, current
        self._start('proposal', read)

    def _photo_crop_rect(self):
        crop = self.packet.get('review_crop_xywh') if self.packet is not None else None
        if (isinstance(crop, (list, tuple)) and len(crop) == 4
                and np.isfinite(crop).all() and crop[2] > 0 and crop[3] > 0):
            return QRectF(*crop)
        return None

    @Slot()
    def _photo_view_changed(self):
        rectangle = self._photo_crop_rect() if self.photo_view.currentData() == 'rig' else None
        self.photo.set_view_rect(rectangle)

    def _draw(self):
        if self.proposal is None or self.packet is None:
            return
        for item in self._overlays:
            self.photo.scene().removeItem(item)
        self._overlays = []
        if self._photo_run != self.packet['run_id']:
            crop = self._photo_crop_rect()
            available = crop is not None
            self.photo_view.model().item(1).setEnabled(available)
            if not available and self.photo_view.currentData() == 'rig':
                blocked = self.photo_view.blockSignals(True)
                self.photo_view.setCurrentIndex(0)
                self.photo_view.blockSignals(blocked)
            rectangle = crop if available and self.photo_view.currentData() == 'rig' else None
            view_changed = self.photo.set_view_rect(rectangle, refit=False)
            grid = self.packet.get('display_grid')
            plane = ('camera_u_to_photo_h' if isinstance(grid, dict)
                     and 'camera_u_to_photo_h' in grid else 'native_to_photo_h')
            fields = ('shape_hw', plane, 'runs')
            common_grid = ({key: deepcopy(grid[key]) for key in fields}
                if self.packet.get('full_camera_photograph') is True
                and isinstance(grid, dict) and all(key in grid for key in fields) else None)
            same_grid = common_grid is not None and common_grid == self._photo_grid
            self.photo.show_rgb(_rgb(self.packet), refit=not same_grid or view_changed)
            self._photo_run = self.packet['run_id']
            self._photo_grid = common_grid
        from .painting_transfer_overlays import sand_boundary_path
        material = self.proposal.result['material_labels']
        displayed_material = self.packet.get('material_display', material)
        material_transform = (np.eye(3) if 'material_display' in self.packet
                              else self.packet['native_to_photo_h'])
        roles = self.session.roles
        porous = [key for key, role in roles.items() if role == 'porous_medium']
        silicone = [key for key, role in roles.items() if role in {'excluded', 'frame', 'seal'}]
        lines = []
        if self.overlays['sand'].isChecked():
            lines.append((sand_boundary_path(displayed_material, porous, material_transform), 'white', False))
        if self.overlays['chart'].isChecked():
            for key, color in (('original_chart_mask', 'yellow'), ('chart_mask', 'cyan')):
                outlines = self.proposal.display_outlines
                points = outlines[key] if outlines is not None else _gap_contours(self.proposal.result[key])
                lines.append((points, color, key == 'chart_mask'))
        if self.overlays['continued'].isChecked():
            lines.append((_gap_contours(self.packet['continued_mask']), 'orange', False))
        if self.overlays['rig_frame'].isChecked():
            lines.append(([self.packet['own_frame_native_xy']], 'lime', False))
        if self.overlays['silicone'].isChecked():
            lines.append((_path(_gap_contours(np.isin(displayed_material, silicone)),
                                material_transform), 'violet', False))
        if self.overlays['gaps'].isChecked():
            lines.append((_gap_contours(self.proposal.result['unassigned_mask']), 'magenta', False))
        for points, color, chart_mask in lines:
            item = QGraphicsPathItem(points if isinstance(points, QPainterPath)
                                     else _path(points, self.packet['native_to_photo_h']))
            pen = QPen(QColor(color), 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            if chart_mask and self.hatching.isChecked():
                item.setBrush(QBrush(QColor(0, 255, 255, 100), Qt.BrushStyle.BDiagPattern))
            item.setZValue(2)
            self.photo.scene().addItem(item)
            self._overlays.append(item)
        self._finish_photo_loading()

    def _approve(self):
        if (self.proposal is None or self.busy() or not self.sand_valid.isChecked()
                or not self.chart_valid.isChecked()):
            return
        try:
            if self._validator is not None:
                self._validator(self.session.review)
            reason = self.sand_valid.text() + '. ' + self.chart_valid.text() + '.'
            record = confirm_chart_transfer(self.proposal, actor=self.actor, reason=reason)
            approved = dict(self.approvals, **{record['run_id']: record})
            payloads = dict(self._approval_payloads, **{record['run_id']: self.proposal.payload.decode('utf8')})
            self._save_decisions(approved, payloads, self.adjustments, self.confirmations)
            self.approvals, self._approval_payloads = approved, payloads
            self.approvals_changed.emit(deepcopy(self.approvals))
            self.status.setText(f'{record["run_id"].upper()} approved. {len(self.approvals)} of {len(self.expected_runs)} approved.')
        except (OSError, ValueError) as error:
            self.status.setText(str(error))
        self._buttons()

    def _save_decisions(self, approvals, payloads, adjustments, confirmations):
        if self._approval_store is not None:
            self._approval_store.save(approvals, payloads, adjustments, confirmations)
        self._save_error = ''

    def _persist_changes(self):
        try:
            self._save_decisions(self.approvals, self._approval_payloads, self.adjustments, self.confirmations)
            return True
        except (OSError, ValueError, KeyError) as error:
            self._save_error = str(error)
            self.status.setText('Chart decisions could not be saved: ' + str(error))
            return False

    def _cancel_bulk_approval(self):
        if self._bulk_cancel is not None:
            self._bulk_cancel.set()

    def _bulk_context_matches(self, context):
        return (not self._closing and self.session is context['session']
            and self._generation == context['generation']
            and self.session.review == context['review']
            and self.expected_runs == set(context['runs'])
            and self.runs.currentData() == context['selected_run']
            and self.adjustments == context['adjustments']
            and self.approvals == context['approvals']
            and self.confirmations == context['confirmations'])

    def _approve_all(self):
        if self.busy() or self._closing or self.session is None or not self.expected_runs:
            return
        context = dict(session=self.session, review=deepcopy(self.session.review),
            generation=self._generation, runs=tuple(row['run_id'] for row in self.session.review['runs']),
            selected_run=self.runs.currentData(), adjustments=deepcopy(self.adjustments),
            approvals=deepcopy(self.approvals), confirmations=deepcopy(self.confirmations))
        answer = QMessageBox.question(self, 'Approve all runs',
            f'Approve the sand-layer map outlines and ColorChecker outline for all {len(context["runs"])} runs?\n\n'
            'Each run keeps its current adjustment. Existing valid approvals are kept. '
            'This is a bulk confirmation; photographs are not opened individually.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self._bulk_context_matches(context):
            self.status.setText('The review changed. No additional runs were approved; try again.')
            return
        try:
            if self._validator is not None:
                self._validator(self.session.review)
            if not self._bulk_context_matches(context):
                raise ValueError('The review changed before bulk approval.')
        except Exception as error:
            self.status.setText(str(error))
            return
        self._bulk_context = context
        self._bulk_cancel = Event()
        self._bulk_approval_active = True
        current = self.proposal
        cancelled = self._bulk_cancel
        actor = self.actor

        def approve_runs():
            approved, payloads = {}, {}
            for index, run in enumerate(context['runs'], 1):
                if cancelled.is_set():
                    raise ValueError('The review changed during bulk approval.')
                correction = context['adjustments'].get(run, [0., 0.])
                record = context['approvals'].get(run)
                from .painting_chart_transfer_reader import proposal_for_approval
                if current is not None and run == context['selected_run']:
                    proposal = current
                elif record is not None and list(record.get('correction_mm', [])) == list(correction):
                    proposal = proposal_for_approval(context['session'], run, record)
                else:
                    proposal = context['session'].proposal(run, correction_mm=correction)
                metadata = proposal.validate()
                if metadata['run_id'] != run or list(metadata['correction_mm']) != list(correction):
                    raise ValueError('A proposal does not match its run and current adjustment.')
                try:
                    _validate_approval(metadata, proposal.payload, record)
                except (ValueError, TypeError, KeyError):
                    record = confirm_chart_transfer(proposal, actor=actor,
                        reason='Bulk approval of the sand-layer map outlines and ColorChecker outline '
                        'for all selected runs, using each current adjustment.')
                approved[run] = record
                payloads[run] = proposal.payload.decode('utf8')
                del proposal
                self.bulk_progress.emit(index, len(context['runs']))
            return dict(approvals=approved, payloads=payloads)

        self.status.setText(f'Approving all runs: 0 of {len(context["runs"])} prepared…')
        self._start('approve_all', approve_runs)

    @Slot(int, int)
    def _show_bulk_progress(self, done, total):
        if self.bulk_approval_active and not self._closing:
            self.status.setText(f'Approving all runs: {done} of {total} prepared…')

    def _finish_bulk_approval(self, worker):
        context = self._bulk_context
        cancelled = self._bulk_cancel
        self._bulk_approval_active = False
        self._bulk_context = self._bulk_cancel = None
        if self._closing:
            return
        try:
            if cancelled.is_set() or not self._bulk_context_matches(context):
                raise ValueError('The review changed during bulk approval.')
            if worker.error:
                raise ValueError(worker.error)
            if self._validator is not None:
                self._validator(self.session.review)
            if not self._bulk_context_matches(context):
                raise ValueError('The review changed during bulk approval.')
            if set(worker.result['approvals']) != self.expected_runs:
                raise ValueError('The bulk approval did not include every selected run.')
            confirmations = dict(self.confirmations, **{run: [True, True] for run in context['runs']})
            self._save_decisions(worker.result['approvals'], worker.result['payloads'], self.adjustments, confirmations)
            self.approvals = deepcopy(worker.result['approvals'])
            self._approval_payloads = deepcopy(worker.result['payloads'])
            self.confirmations = confirmations
            self._set_checks(self.confirmations.get(self.runs.currentData(), [False, False]))
            self.approvals_changed.emit(deepcopy(self.approvals))
            self.status.setText(f'All {len(self.expected_runs)} runs are approved. '
                'Use approved masks with the saved maps to continue.')
        except Exception as error:
            self.status.setText(f'No additional runs were approved. {error}')
        self._buttons()
        if self._pending_request is not None and not self._adjust_timer.isActive():
            self._dispatch_pending()

    def _reuse(self):
        if self.busy() or not self.expected_runs or set(self.approvals) != self.expected_runs:
            return
        if not self._persist_changes():
            self._buttons()
            return
        self.reuse_requested.emit(dict(review=deepcopy(self.session.review),
            approvals=deepcopy(self.approvals), detection_manifest=deepcopy(self.session.manifest_binding)))

    def wait_for_workers(self):
        self._closing = True
        self._cancel_bulk_approval()
        self._adjust_timer.stop()
        self._photo_load_timer.stop()
        self._pending_request = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._retired()

    def closeEvent(self, event):
        self.wait_for_workers()
        super().closeEvent(event)
