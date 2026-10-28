"""Ordinary Qt Check carries an explicit source statement without changing gates."""
from copy import deepcopy
import threading

import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import map_adoption_view as view

from ..test_geometry_registered_export_apply import registered_setup  # noqa: F401
from ..test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401
from ..test_painted_source_acquisition import _statement


@pytest.mark.parametrize("state", ["clearwater", "indicator"])
def test_dialog_snapshots_and_forwards_the_exact_source_statement_with_original_warning_flow(
        qtbot, registered_setup, tmp_path, monkeypatch, state):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    declaration = _statement(prepared, state)
    expected = deepcopy(declaration)
    store = geometry.GeometryRevisionStore(data.project)
    before = store.current_path.read_bytes()
    calls = []
    original = adoption.adoption_candidate

    def candidate(*args, **kwargs):
        calls.append((deepcopy(kwargs), threading.get_ident()))
        return original(*args, **kwargs)

    monkeypatch.setattr(adoption, "adoption_candidate", candidate)
    dialog = view.AdoptionDialog(data.project, prepared["folder"], actor="synthetic source owner",
        facies_source_acquisition=declaration)
    qtbot.addWidget(dialog, before_close_func=lambda owned: owned.wait_for_workers())
    declaration["acquisition_state"] = "clearwater" if state == "indicator" else "indicator"
    declaration["photograph"]["sha256"] = "0" * 64
    declaration["decision"]["statement"] = "Caller changed its own metadata."
    assert dialog.check_button.isEnabled() and not dialog.save_button.isEnabled()
    assert dialog.check()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert dialog.candidate is not None, dialog.status.text()
    assert len(calls) == 1 and calls[0][0] == dict(facies_source_acquisition=expected)
    assert calls[0][1] != threading.get_ident()
    assert all(row["acquisition_state"] == state for row in dialog.candidate["run_geometry"]["runs"])
    assert dialog.candidate["provenance"]["facies_source_acquisition"] == expected
    assert dialog.save_button.isEnabled()
    shown = dialog.check_messages.toPlainText()
    assert ("indicator_facies_carryover_risk" in shown) == (state == "indicator")
    questions = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args:
        questions.append(args[2]) or QMessageBox.StandardButton.No)
    dialog._save_clicked()
    assert len(questions) == 1
    assert ("indicator_facies_carryover_risk" in questions[0]) == (state == "indicator")
    assert store.current_path.read_bytes() == before and dialog.revision is None


def test_worker_snapshots_statement_before_a_caller_can_mutate_it(monkeypatch, tmp_path):
    declaration = dict(acquisition_state="indicator", decision=dict(statement="Synthetic first statement"))
    expected = deepcopy(declaration)
    calls, results = [], []

    def candidate(current, folder, ids, *, facies_source_acquisition):
        calls.append(deepcopy(facies_source_acquisition))
        return {}, {}

    monkeypatch.setattr(adoption, "adoption_candidate", candidate)
    monkeypatch.setattr(geometry, "validate_geometry_config", lambda _config: {"issues": []})
    worker = view._CheckWorker({}, tmp_path, None, facies_source_acquisition=declaration)
    worker.finished.connect(results.append)
    declaration["decision"]["statement"] = "Caller changed metadata before run."
    worker.run()
    assert calls == [expected] and len(results) == 1


def test_no_statement_preserves_injected_original_worker_and_candidate_signatures(
        qtbot, registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    original_worker, original_candidate = view._CheckWorker, adoption.adoption_candidate
    calls = []

    class OriginalSignatureWorker(original_worker):
        def __init__(self, current, folder, ids):
            calls.append("worker")
            super().__init__(current, folder, ids)

    def candidate(current, folder, ids):
        calls.append("candidate")
        return original_candidate(current, folder, ids)

    monkeypatch.setattr(view, "_CheckWorker", OriginalSignatureWorker)
    monkeypatch.setattr(adoption, "adoption_candidate", candidate)
    dialog = view.AdoptionDialog(data.project, prepared["folder"], actor="synthetic source owner")
    qtbot.addWidget(dialog, before_close_func=lambda owned: owned.wait_for_workers())
    assert dialog.check()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=30000)
    assert dialog.candidate is not None, dialog.status.text()
    assert calls == ["worker", "candidate"]
    assert "facies_source_acquisition" not in dialog.candidate["provenance"]
    assert all(row["acquisition_state"] == "unknown" for row in dialog.candidate["run_geometry"]["runs"])
