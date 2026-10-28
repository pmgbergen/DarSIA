"""Adoption stays inside the main photo workspace through Check and Save."""
from copy import deepcopy
from threading import Event

import pytest
from PySide6.QtWidgets import QLabel, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import series_layer_view as editor

from ..test_chart_transfer_findings import _current_chart_case
from ..test_painted_adoption_table import (  # noqa: F401
    imported, registered_setup, synthetic_payload_admission,
)
from .test_painting_series_adoption_controller import _Host


class _EmbeddedHost(_Host):
    show_transfer_review = editor.LayerPainterView.show_transfer_review
    show_adoption_panel = editor.LayerPainterView.show_adoption_panel
    show_mark_layers = editor.LayerPainterView.show_mark_layers
    _keep_transfer_review_visible = editor.LayerPainterView._keep_transfer_review_visible

    def __init__(self, *args):
        super().__init__(*args)
        self._workflow_scope = 'apply'
        self._transfer_review_active = False
        self.run_preview.widget = QWidget(self)
        self.run_preview.stack = QStackedWidget()
        preview = QVBoxLayout(self.run_preview.widget)
        preview.addWidget(self.run_preview.stack)
        self.photo = QLabel('Original photograph')
        self.run_preview.stack.addWidget(self.photo)
        self.run_preview.stack.currentChanged.connect(self._keep_transfer_review_visible)
        QVBoxLayout(self).addWidget(self.run_preview.widget)
        self.resize(900, 700)

    def _set_rig_photo_tools(self, _active):
        pass


def _host(qtbot, data, tmp_path, monkeypatch):
    _, _, prepared, _ = _current_chart_case(data, tmp_path, monkeypatch)
    session, _, frame = table.verify_version(prepared['painting_facies'])
    record = dict(prepared['painting_facies']['record'], folder=prepared['map_version']['folder'])
    host = _EmbeddedHost(data, prepared, record, session, frame)
    host.series_use_status.result = dict(preparation=deepcopy(prepared))
    qtbot.addWidget(host, before_close_func=lambda owned: owned.wait_for_workers())
    host.show()
    monkeypatch.setattr(QMessageBox, 'question',
                        lambda *a, **k: pytest.fail('Embedded adoption must not open a modal question'))
    return host, prepared


@pytest.mark.parametrize('prior_review', [False, True])
def test_inline_save_once_and_back_restores_prior_surface(
        qtbot, registered_setup, tmp_path, monkeypatch, prior_review):
    host, prepared = _host(qtbot, registered_setup, tmp_path, monkeypatch)
    previous = host.photo
    if prior_review:
        previous = QLabel('Existing transfer review')
        host.show_transfer_review(previous)
    calls = []
    original_save = geometry.GeometryRevisionStore.save_revision

    def save(*args, **kwargs):
        calls.append(True)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(geometry.GeometryRevisionStore, 'save_revision', save)
    controller = host.adoption_controller
    assert controller.start(host.record, host.selection, prepared)
    panel = controller.dialog
    qtbot.waitUntil(lambda: panel.confirmation.isVisible(), timeout=15000)
    assert not panel.isWindow()
    assert panel.window() is host
    assert host.run_preview.stack.currentWidget() is panel
    assert calls == []
    panel.confirm_save.click()
    panel.confirm_save.click()  # A duplicate click cannot append twice.
    qtbot.waitUntil(lambda: controller.context is None and not panel.busy(), timeout=15000)
    assert len(calls) == 1
    assert panel.revision is not None, panel.status.text()
    panel.close_button.click()
    assert host.run_preview.stack.currentWidget() is previous
    assert not panel.isVisible()

    # A cold owner reads the saved revision instead of deriving another table
    # against that revision as if it were the preparation's original parent.
    revision = geometry.GeometryRevisionStore(registered_setup.project).current()
    session, _, frame = table.verify_version(prepared['painting_facies'])
    reopened = _EmbeddedHost(registered_setup, prepared, host.record, session, frame)
    reopened.series_use_status.result = dict(preparation=deepcopy(prepared))
    qtbot.addWidget(reopened, before_close_func=lambda owned: owned.wait_for_workers())
    reopened.show()
    with monkeypatch.context() as guard:
        guard.setattr(adoption, 'adoption_candidate',
                      lambda *a, **k: pytest.fail('Saved adoption must not run Check again'))
        assert reopened.adoption_controller.start(reopened.record, reopened.selection, prepared)
    restored = reopened.adoption_controller.dialog
    assert reopened.adoption_controller.context is None
    assert restored.revision == revision
    assert not restored.check_button.isEnabled() and not restored.save_button.isEnabled()
    assert not restored.confirmation.isVisible() and not restored.busy()
    assert 'Already adopted' in reopened.use_map_progress.text()
    assert len(calls) == 1
    assert geometry.GeometryRevisionStore(registered_setup.project).current() == revision
    unrelated = deepcopy(revision)
    unrelated['configuration'][table.KEY]['preparation']['sha256'] = '0' * 64
    assert table.saved_adoption_revision(unrelated, prepared['folder']) is None

    csv_path = revision['artifacts']['facies_properties_path']
    from pathlib import Path
    original_csv = Path(csv_path).read_bytes()
    Path(csv_path).write_bytes(original_csv + b'\n')
    with pytest.raises(ValueError):
        table.saved_adoption_revision(revision, prepared['folder'])


def test_busy_embedded_close_preserves_worker_then_inline_cancel_returns(
        qtbot, registered_setup, tmp_path, monkeypatch):
    host, prepared = _host(qtbot, registered_setup, tmp_path, monkeypatch)
    entered, release = Event(), Event()
    original = adoption.adoption_candidate

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(adoption, 'adoption_candidate', slow)
    controller = host.adoption_controller
    assert controller.start(host.record, host.selection, prepared)
    panel = controller.dialog
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        thread = panel._thread
        assert panel.busy() and not panel.close_button.isEnabled()
        assert not panel.close()
        panel.reject()
        assert panel.isVisible() and panel._thread is thread
    finally:
        release.set()
    qtbot.waitUntil(lambda: panel.confirmation.isVisible(), timeout=15000)
    panel.confirm_cancel.click()
    assert controller.context is None
    assert panel.revision is None and not panel.busy()
    assert host.run_preview.stack.currentWidget() is host.photo
