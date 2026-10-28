"""Explicit recorded-policy GUI action; original publisher and guards run.

Pressure and authority metadata are exact original source copies; the native
and geometry specimens are finite test metadata. Only file choices, strict
same-byte source locators and scheduling are finite UI seams; no source,
policy, adoption or plan validator is replaced.
"""
from pathlib import Path
import threading

import pytest
from PySide6.QtWidgets import QFileDialog

from darsia.fluidflower import accepted_pressure_policy as policy
from darsia.fluidflower.analysis_preparation_ui import AnalysisPreparationDialog
from darsia.fluidflower.jobs_panel import JobsPanel
from ..test_accepted_pressure_policy import recorded_pressure, saved_products, phase1_data  # noqa: F401


def _dialog(qtbot, case):
    panel = JobsPanel()
    panel.series_root = case.series_root
    qtbot.addWidget(panel)
    dialog = AnalysisPreparationDialog(panel, panel)
    qtbot.addWidget(dialog)
    dialog.pressure.setText(case.original_index['path'])
    dialog.geometry.setText(case.geometry['path'])
    dialog.inputs.setText(str(case.series_root / 'unchanged-source-inputs.json'))
    dialog.show()
    return panel, dialog


@pytest.mark.parametrize('stale', [False, True])
def test_recorded_decision_uses_original_owned_worker_and_only_prefills_current_choices(
        qtbot, recorded_pressure, tmp_path, monkeypatch, stale):
    case = recorded_pressure
    panel, dialog = _dialog(qtbot, case)
    target = case.series_root / 'derived-pressure-index.json'
    before = {Path(bound['path']): Path(bound['path']).read_bytes()
              for bound in (case.original_index, case.original_atmosphere, case.authority)}
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (case.authority['path'], ''))
    def choose_output(*args, **kwargs):
        assert Path(args[2]) == case.series_root / 'accepted-pressure-index.json'
        assert 'registered input root' in args[1]
        return str(target), ''
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', choose_output)
    original = policy.prepare_accepted_pressure_index
    entered, release = threading.Event(), threading.Event()
    calls, adopted = [], []
    owner_thread = threading.get_ident()
    def publish(*args, **kwargs):
        calls.append((threading.get_ident(), args, kwargs))
        entered.set()
        if not release.wait(10):
            raise RuntimeError('The finite recorded-policy call was not released')
        return original(*args, **kwargs, source_aliases=case.source_aliases)
    monkeypatch.setattr(policy, 'prepare_accepted_pressure_index', publish)
    dialog.adopted.connect(adopted.append)
    try:
        dialog.pressure_decision_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert len(calls) == 1 and calls[0][0] != owner_thread
        assert calls[0][1][0] == case.original_index
        assert calls[0][2]['authority'] == case.authority
        assert not dialog.pressure_decision_button.isEnabled()
        assert not dialog.prepare_button.isEnabled() and not dialog.collect_button.isEnabled()
        assert not dialog.adopt_button.isEnabled()
        dialog.reject()
        assert dialog.isVisible() and not dialog.can_close()
        if stale:
            dialog.pressure.setText(str(tmp_path / 'different-current-pressure.json'))
        selected = dialog.pressure.text()
        release.set()
        qtbot.waitUntil(lambda: dialog._thread is None, timeout=10000)
        assert target.is_file() and dialog.can_close()
        assert dialog.pressure_decision_button.isEnabled()
        if stale:
            assert dialog.pressure.text() == selected and 'earlier choices' in dialog.status.text()
        else:
            assert dialog.pressure.text() == str(target.resolve())
            assert 'existing recorded pressure decision' in dialog.status.text()
        assert all(path.read_bytes() == data for path, data in before.items())
        assert dialog.prepared_result is None and not dialog.adopt_button.isEnabled()
        assert adopted == [] and panel.series_root == case.series_root
        assert not list(tmp_path.rglob('execution.toml'))
        assert not list(tmp_path.rglob('job.json'))
    finally:
        release.set()
        qtbot.waitUntil(lambda: dialog._thread is None, timeout=10000)
        dialog.close()


def test_recorded_decision_refuses_modal_drift_other_stage_and_plan_folder(
        qtbot, recorded_pressure, tmp_path, monkeypatch):
    case = recorded_pressure
    panel, dialog = _dialog(qtbot, case)
    target = tmp_path / 'never-written.json'
    before = dialog.pressure.text()
    def choose(*a, **k):
        dialog.probability.setChecked(not dialog.probability.isChecked())
        return case.authority['path'], ''
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', choose)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(target), ''))
    dialog.pressure_decision_button.click()
    assert dialog._thread is None and 'changed while choosing files' in dialog.status.text()
    assert dialog.pressure.text() == before and not target.exists()
    dialog.stage.setCurrentIndex(1)
    assert not dialog.pressure_decision_button.isEnabled()
    dialog.use_recorded_pressure_decision()
    assert dialog._thread is None and not target.exists()
    dialog.stage.setCurrentIndex(0)
    panel.series_root = tmp_path / 'different-series'
    dialog.use_recorded_pressure_decision()
    assert 'selected series changed' in dialog.status.text()
    assert dialog._thread is None and dialog.pressure.text() == before and not target.exists()
    panel.series_root = case.series_root
    dialog.folder.setText(str(tmp_path / 'fresh-plan'))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (case.authority['path'], ''))
    protected = tmp_path / 'fresh-plan' / 'pressure.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(protected), ''))
    dialog.pressure_decision_button.click()
    assert dialog._thread is None and 'outside the fresh plan folder' in dialog.status.text()
    assert not protected.exists() and dialog.pressure.text() == before
    # The immutable proof copies also reserve a fresh sibling namespace.
    # An outside JSON must not occupy the selected future plan directory.
    dialog.folder.setText(str(tmp_path / 'derived-pressure-index_policy_sources'))
    sibling = tmp_path / 'derived-pressure-index.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(sibling), ''))
    dialog.pressure_decision_button.click()
    assert dialog._thread is None and 'outside the fresh plan folder' in dialog.status.text()
    assert not sibling.exists() and not Path(dialog.folder.text()).exists()
    assert dialog.pressure.text() == before
