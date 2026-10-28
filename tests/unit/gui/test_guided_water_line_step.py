"""Rowless water measurements through the original finite GUI reader lifetime.

Review metadata and decoded photographs are finite UI fixtures, not campaign
source or calibration qualification. The actual photo canvas, water action API,
owned _Read/QThread retirement and unchanged Check/approval predicates run.
"""
from copy import deepcopy
import threading

import numpy as np
import pytest
from PySide6.QtCore import QTimer

from .test_painting_gap_review_ui import _packet
from .test_painting_guided_review_ui import (
    GuidedBackend, _bound_review, _dialog, _settle, _stage,
)
from .test_painting_water_surface_ui import _api, _packet as _water_packet


def _review_for(runs):
    review = _bound_review(gaps=False)
    template = deepcopy(review['runs'][-1])
    review['runs'] = [dict(deepcopy(template), run_id=run) for run in runs]
    review['issues'] = []
    review['display_context'] = dict(format='finite-GUI-reader-context', declared_run_order=list(runs))
    return review


class WaterBackend(GuidedBackend):
    def __init__(self, runs):
        super().__init__()
        self.water_reads = []
        self.packets = {run: _packet(run) for run in runs}
        for packet in self.packets.values():
            packet['gap_mask'] = np.zeros_like(packet['gap_mask'])
            packet.update(image_role='current', display_grid=dict(
                format='finite-bound-native-display', run_id=packet['run_id']),
                image_choices=[dict(image_id='clearwater', available=True)])

    def water(self, review, run, component, role):
        assert run in [row['run_id'] for row in review['runs']]
        assert component is None and role == 'clearwater'
        self.water_reads.append((run, component, role, threading.current_thread() is threading.main_thread()))
        base = self.packets[run]
        water = _water_packet(bolts=False)
        water['basis']['run_id'] = run
        water['basis']['photo']['path'] = f'C:/synthetic/{run}/clearwater.ARW'
        water['display_grid'] = deepcopy(base['display_grid'])
        water['photo_to_display_h'] = np.asarray(base['native_to_photo_h']).tolist()
        return dict(base, component_id=None, image_role=role,
            image_rgb=np.full((64, 64, 3), 80, np.uint8), valid_support=np.ones((64, 64), bool),
            water_surface=water)


def _water_dialog(qtbot, *, runs=('ac17', 'ac18'), backend=None, reader=None):
    backend = backend or WaterBackend(runs)
    api, calls = _api()
    widget, _ = _dialog(qtbot, backend=backend, review=_review_for(runs), editor=False,
        water_surface_comparison_reader=reader or backend.water, water_surface_api=api)
    return widget, backend, calls


def test_measurement_step_lists_exact_39_runs_without_inventing_findings_or_reading_the_cohort(qtbot):
    runs = tuple(f'ac{number}' for number in range(17, 56))  # Explicit finite cohort, not the campaign cohort.
    widget, backend, calls = _water_dialog(qtbot, runs=runs)
    assert [stage for stage, _ in widget._questions] == ['source_water', 'water_line', 'check', 'adopt']
    before = deepcopy(widget._rows), deepcopy(widget._report), deepcopy(widget.review)
    requests = []
    for signal in (widget.approval_changed, widget.state_requested, widget.recheck_requested,
                   widget.adopt_requested, widget.water_surface_confirmed):
        signal.connect(lambda *args: requests.append(args))
    assert backend.water_reads == []
    _stage(qtbot, widget, 'water_line')
    assert [widget.water_run_choice.itemData(index) for index in range(widget.water_run_choice.count())] == list(runs)
    assert widget._current_row() is None and 'AC17' in widget.question_title.text()
    assert widget._preview_key == ('ac17', None) and widget.images.currentData() == 'clearwater'
    assert widget.water_surface.packet['basis']['run_id'] == 'ac17'
    assert widget.water_surface.isVisible() and not widget.reason_panel.isVisible()
    assert backend.water_reads == [('ac17', None, 'clearwater', False)]
    widget.water_run_choice.setCurrentIndex(widget.water_run_choice.findData(runs[-1]))
    _settle(qtbot, widget)
    assert widget._preview_key == (runs[-1], None) and widget.water_surface.packet['basis']['run_id'] == runs[-1]
    assert [value[0] for value in backend.water_reads] == ['ac17', runs[-1]]
    assert {run for run, _ in backend.reads if run != 'review'} == {'ac17', runs[-1]}
    assert backend.comparisons == []  # Ordinary N467 comparison reader was never used for measurements.
    widget.next_button.click()
    _settle(qtbot, widget)
    assert widget.guided_stage == 'check'
    assert (widget._rows, widget._report, widget.review) == before
    assert requests == calls == [] and widget.approval is widget._candidate is None
    assert widget._acquisition_actions == [] and widget._warning_actions == {} and not widget.can_recheck()
    # Explicit capability is required; legacy callers keep their original question list.
    legacy, _ = _dialog(qtbot, backend=WaterBackend(runs), review=_review_for(runs), editor=False)
    assert 'water_line' not in [stage for stage, _ in legacy._questions]
    # The rowless empty-mask route must not relax the original selected-gap refusal.
    gap_backend = GuidedBackend()
    gap_backend.packets['ac18']['gap_mask'][:] = False
    gap, _ = _dialog(qtbot, backend=gap_backend, editor=False)
    assert gap._current_row()['kind'] == 'gap' and gap.packet is None
    assert 'selected component mask is empty' in gap.status.text()
    assert gap.approval is gap._candidate is None


def test_measurement_uses_original_water_actions_and_preserves_photo_view_without_implicit_answers(qtbot):
    widget, backend, calls = _water_dialog(qtbot)
    _stage(qtbot, widget, 'water_line')
    qtbot.waitUntil(lambda: widget._pending_photo_fit is None, timeout=3000)
    widget.photo.scale(30., 30.)
    widget.photo.centerOn(25., 20.)
    horizontal, vertical = widget.photo.horizontalScrollBar(), widget.photo.verticalScrollBar()
    qtbot.waitUntil(lambda: horizontal.isVisible() and vertical.isVisible()
        and widget.photo.viewport().width() + vertical.width() == widget.photo.maximumViewportSize().width()
        and widget.photo.viewport().height() + horizontal.height() == widget.photo.maximumViewportSize().height(), timeout=2000)
    transform = widget.photo.transform()
    center = widget.photo.mapToScene(widget.photo.viewport().rect().center())
    rows = deepcopy(widget._rows)
    widget.image_shortcut.activated.emit()
    assert widget.images.currentData() == 'current' and calls == []
    assert widget.photo.transform() == transform
    assert widget.photo.mapToScene(widget.photo.viewport().rect().center()) == center
    widget.image_shortcut.activated.emit()
    _settle(qtbot, widget)
    assert widget.images.currentData() == 'clearwater' and widget.photo.transform() == transform
    assert widget.photo.mapToScene(widget.photo.viewport().rect().center()) == center
    assert len(backend.water_reads) == 2 and backend.comparisons == [] and calls == []
    for _ in range(5):
        widget.image_shortcut.activated.emit()
        assert widget.images.currentData() == 'current'
        assert widget.photo.transform() == transform
        assert widget.photo.mapToScene(widget.photo.viewport().rect().center()) == center
        widget.image_shortcut.activated.emit()
        _settle(qtbot, widget)
        assert widget.images.currentData() == 'clearwater'
        assert widget.photo.transform() == transform
        assert widget.photo.mapToScene(widget.photo.viewport().rect().center()) == center
        assert calls == [] and widget.water_surface_confirmations == {}
    assert widget.water_surface.action is None and widget.water_surface_confirmations == {}
    basis = deepcopy(widget.water_surface.packet['basis'])
    widget.water_surface.confirm.click()
    record = widget.water_surface_confirmations['ac17']
    assert [name for name, *_ in calls] == ['confirm']
    assert record['action']['run_id'] == 'ac17' and record['action']['actor'] == 'Synthetic reviewer'
    assert record['measurements']['height_basis'] == 'existing-own-geometry'
    assert record['measurements']['height_mm'] == 16. and record['measurements']['bolt_height_mm'] is None
    assert widget.water_surface.packet['basis'] == basis and widget._rows == rows
    assert widget.approval is widget._candidate is None and widget._acquisition_actions == []
    assert widget._display_preferences_dirty and not widget.recheck.isEnabled()
    widget.water_surface.move.click()
    widget.photo.point_clicked.emit((8., 16.))  # Original explicit display inverse: source (5,11).
    assert widget._unconfirmed_water_edits == {'ac17'} and widget.water_surface_confirmations == {}
    starts = []
    widget.recheck_requested.connect(starts.append)
    widget._recheck()
    assert starts == [] and [name for name, *_ in calls] == ['confirm', 'move']
    widget.clear_water_confirmations.click()
    assert widget._unconfirmed_water_edits == set() and widget.water_surface_confirmations == {}
    assert widget._display_preferences_dirty and widget.approval is None


@pytest.mark.parametrize('damage', ['role', 'run', 'grid'])
def test_measurement_refuses_a_wrong_photo_role_run_or_native_plane_without_recording_it(qtbot, damage):
    backend = WaterBackend(('ac17', 'ac18'))

    def damaged(*args):
        packet = backend.water(*args)
        if damage == 'role':
            packet['image_role'] = 'previous-run'
        elif damage == 'run':
            packet['water_surface']['basis']['run_id'] = 'ac18'
        else:
            packet['display_grid'] = dict(packet['display_grid'], run_id='ac18')
        return packet

    widget, _, calls = _water_dialog(qtbot, backend=backend, reader=damaged)
    default_reader = widget._comparison_reader
    _stage(qtbot, widget, 'water_line')
    assert widget.images.currentData() == 'current' and widget.water_surface.packet is None
    assert 'could not be displayed' in widget.status.text()
    assert not widget.water_surface.confirm.isEnabled() and widget.water_surface_confirmations == {}
    assert widget.approval is widget._candidate is None and calls == []
    assert widget._comparison_reader is default_reader


def test_close_during_rowless_water_read_uses_original_natural_retirement(qtbot):
    backend = WaterBackend(('ac17', 'ac18'))
    entered, release = threading.Event(), threading.Event()

    def delayed(*args):
        if args[1] == 'ac18':
            entered.set()
            assert release.wait(5), 'The finite original reader must always be released'
        return backend.water(*args)

    widget, _, calls = _water_dialog(qtbot, backend=backend, reader=delayed)
    _stage(qtbot, widget, 'water_line')
    try:
        widget.water_run_choice.setCurrentIndex(widget.water_run_choice.findData('ac18'))
        qtbot.waitUntil(entered.is_set, timeout=5000)
        owned = widget.thread
        assert owned is not None and widget.worker.key[0] == 'comparison'
        assert not widget.water_run_choice.isEnabled() and not widget.next_button.isEnabled()
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks), timeout=1000)
        widget.close()
        assert widget.busy() and widget._closing and widget._pending is None and widget.isVisible()
        assert widget.thread is owned and not release.is_set()
        release.set()
        qtbot.waitUntil(lambda: not widget.busy() and not widget.isVisible(), timeout=5000)
        assert widget.thread is widget.worker is None and widget.water_surface_confirmations == {}
        assert calls == [] and widget.approval is widget._candidate is None
    finally:
        release.set()
        widget.wait_for_workers()
