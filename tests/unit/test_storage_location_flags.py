"""Actual admitted output-location metadata only; no producer or network I/O."""
from types import SimpleNamespace
import pytest
from darsia.fluidflower.mass import n1_pipeline as n1
from darsia.fluidflower.pipeline import run_io
from darsia.fluidflower import job_filesystem as fs


def binding(monkeypatch,tmp_path,proof):
    seen=[];root=tmp_path/"results"
    value=SimpleNamespace(local_output=SimpleNamespace(admission=proof,results_root=root),
        require_output=lambda path:seen.append(path))
    monkeypatch.setattr(run_io,"current_binding",lambda:value)
    return root,seen


def test_legacy_direct_path_flags_are_unchanged(monkeypatch,tmp_path):
    monkeypatch.setattr(run_io,"current_binding",lambda:None)
    assert n1._storage_facts(tmp_path)==dict(local_retained=False,direct_NAS_write=True)


def test_original_positive_local_flags_remain_exact(monkeypatch,tmp_path):
    proof=fs.LocalFileAdmission(True,"fixture-fixed","synthetic OS","fixture",str(tmp_path))
    root,seen=binding(monkeypatch,tmp_path,proof)
    assert n1._storage_facts(root)==dict(local_retained=True,direct_NAS_write=False,direct_local_write=True)
    assert seen==[root]


@pytest.mark.parametrize("protocol",["smb","nfs"])
def test_actual_remote_proof_is_never_reported_as_local(monkeypatch,tmp_path,protocol):
    source=fs.SharedFolderAdmission(True,protocol,"synthetic OS","fixture")
    proof=fs.ResultsRootAdmission(True,protocol,False,"synthetic OS","fixture",str(tmp_path),str(tmp_path/"results"),source)
    root,seen=binding(monkeypatch,tmp_path,proof)
    assert n1._storage_facts(root)==dict(local_retained=False,direct_NAS_write=True,direct_local_write=False)
    assert seen==[root]


def test_new_positive_local_proof_preserves_old_three_flags(monkeypatch,tmp_path):
    source=fs.LocalFileAdmission(True,"fixture-fixed","synthetic OS","fixture",str(tmp_path))
    proof=fs.ResultsRootAdmission(True,"fixture-fixed",True,"synthetic OS","fixture",str(tmp_path),str(tmp_path/"results"),source)
    root,seen=binding(monkeypatch,tmp_path,proof)
    assert n1._storage_facts(root)==dict(local_retained=True,direct_NAS_write=False,direct_local_write=True)


@pytest.mark.parametrize("allowed,protocol",[(False,"smb"),(True,"unknown")])
def test_refused_transport_never_becomes_a_storage_claim(monkeypatch,tmp_path,allowed,protocol):
    source=fs.SharedFolderAdmission(allowed,protocol,"synthetic refused OS","fixture")
    proof=fs.ResultsRootAdmission(True,protocol,False,"fixture","fixture",str(tmp_path),str(tmp_path/"results"),source)
    root,seen=binding(monkeypatch,tmp_path,proof)
    with pytest.raises(ValueError,match="positive actual"):n1._storage_facts(root)
