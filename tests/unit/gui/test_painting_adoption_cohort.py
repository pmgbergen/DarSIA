"""GUI cohort arguments and final human scope text; no scientific qualification.

The metadata is private synthetic DATA. Heavy native drawing/preparation are
replaced only to inspect GUI argument plumbing, never claimed as source proof.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower import painting_frame_models as scope
from darsia.fluidflower import series_layer_view as editor
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import raw_shape_basis_reader as reader
from darsia.fluidflower.map_adoption_view import AdoptionDialog
from darsia.fluidflower.map_placement import GEOMETRY_FORMAT


@pytest.fixture
def cohort(tmp_path):
    """Explicit 39-model cohort out of 43 registered synthetic runs."""
    registered=[f"run{i:02d}" for i in range(43)]
    selected=registered[:39]
    value=dict(format=GEOMETRY_FORMAT,run_order=selected,run_count=39,
        runs={run:{"reference_model":{"path":"not-read-by-GUI"}} for run in selected},
        reference_models_complete=True,pending_runs={},complete=False)
    path=tmp_path/"index.json"
    path.write_text(json.dumps(value),encoding="utf8")
    return path,registered,selected,value


def test_explicit_39_of_43_scope_preserves_order_and_shows_four_remaining(cohort):
    path,registered,selected,_=cohort
    result=scope.read_adoption_cohort(path,registered,selected[0])
    assert result["runs"]==selected and result["registered"]==registered
    assert result["remaining"]==registered[39:]
    assert "39 of 43" in result["summary"] and "4 remain outside" in result["summary"]
    for run in registered[39:]:assert run in result["summary"]
    scope.verify_adoption_cohort(result)


@pytest.mark.parametrize("damage",["duplicate","foreign","count","partial","pending","missing_model"])
def test_inconsistent_foreign_or_duplicate_scope_refuses(cohort,damage):
    path,registered,selected,value=cohort
    if damage=="duplicate":value["run_order"][1]=value["run_order"][0]
    elif damage=="foreign":
        old=value["run_order"][-1];value["run_order"][-1]="foreign"
        value["runs"]["foreign"]=value["runs"].pop(old)
    elif damage=="count":value["run_count"]+=1
    elif damage=="partial":value.pop("run_order")
    elif damage=="pending":value["reference_models_complete"]=False
    else:value["runs"].pop(selected[-1])
    path.write_text(json.dumps(value),encoding="utf8")
    with pytest.raises(ValueError):scope.read_adoption_cohort(path,registered,registered[0])


def test_legacy_headerless_index_keeps_all_registered_default(cohort):
    path,registered,_,value=cohort
    path.write_text(json.dumps({"format":GEOMETRY_FORMAT,"runs":value["runs"]}),encoding="utf8")
    result=scope.read_adoption_cohort(path,registered,registered[0])
    assert result["runs"]==registered and result["remaining"]==[] and not result["explicit"]


def test_real_gui_preparation_worker_passes_only_declared_cohort(qtbot,cohort,monkeypatch):
    path,registered,selected,_=cohort
    checked=[];prepared=[];finished=[];failed=[]
    monkeypatch.setattr(reader,"require_basis_index",lambda index,runs:checked.append(list(runs)))
    session=SimpleNamespace(run_id=selected[0],layers=[])
    monkeypatch.setattr(editor.lpm,"version_layers",lambda *a,**k:(session,np.zeros((1,1),np.uint8)))
    def prepare(*args,**kwargs):
        prepared.append(kwargs)
        bound=scope.read_adoption_cohort(path,registered,selected[0])["geometry_index"]
        return {"geometry_index":bound,"runs":[]}
    monkeypatch.setattr(adoption,"prepare_adoption",prepare)
    worker=editor._AdoptionWorker({"run_id":selected[0]},None,
        dict(geometry_index=str(path),runs=registered,destination=path.parent/"unused"))
    worker.finished.connect(finished.append);worker.failed.connect(failed.append)
    worker.run()
    assert not failed and len(finished)==1
    assert checked==[[selected[0],*selected]]
    assert prepared[0]["runs"]==selected
    assert finished[0]["gui_adoption_cohort"]["remaining"]==registered[39:]


def test_final_existing_human_confirmation_includes_cohort_before_decision(qtbot,cohort,monkeypatch):
    path,registered,selected,_=cohort
    summary=scope.read_adoption_cohort(path,registered,selected[0])["summary"]
    questions=[];decisions=[];saves=[]
    def question(owner,title,text):
        questions.append(text)
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox,"question",question)
    dialog=SimpleNamespace(candidate={"actual_gui_argument_test":True},automatic=False,
        findings_managed=False,review_resolutions=None,
        adoption_scope_note=summary,confirmation_decided=SimpleNamespace(emit=decisions.append),
        save=lambda:saves.append(True))
    AdoptionDialog._save_clicked(dialog)
    assert len(questions)==1 and questions[0].startswith(summary)
    assert "39 of 43" in questions[0] and "4 remain outside" in questions[0]
    assert decisions==[False] and saves==[]


def test_changed_index_after_scope_capture_refuses(cohort):
    path,registered,selected,_=cohort
    captured=scope.read_adoption_cohort(path,registered,selected[0])
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError,match="changed"):scope.verify_adoption_cohort(captured)


@pytest.mark.parametrize('as_list', [False, True])
def test_excluded_scope_preserves_only_saved_reasons(cohort, as_list):
    path, registered, selected, value = cohort
    reasons = {registered[39]: 'Saved exclusion reason', registered[40]: ''}
    value['excluded_runs'] = ([dict(run_id=run, reason=reason) for run, reason in reasons.items()]
                              if as_list else reasons)
    path.write_text(json.dumps(value), encoding='utf8')
    result = scope.read_adoption_cohort(path, registered, selected[0])
    assert result['excluded_runs'] == [dict(run_id=run, reason=(
        'Saved exclusion reason' if run == registered[39] else 'Outside the selected adoption cohort'))
        for run in registered[39:]]
