"""Synthetic OS evidence and output-only bindings; no real network access."""
from dataclasses import replace
from pathlib import Path
import pytest
from darsia.fluidflower import job_filesystem as fs


def inspect(tmp_path, protocol, **facts):
    return fs.inspect_results_root(tmp_path / "new-results",
        locality_provider=lambda _: fs.LocalFilesystemFacts("remote",False,"synthetic OS evidence"),
        filesystem_provider=lambda _: fs.FilesystemFacts(protocol,"synthetic OS evidence",**facts),sync_roots=[])


@pytest.mark.parametrize("protocol",["smb","nfs"])
def test_positive_os_remote_output_evidence(tmp_path,protocol):
    value=inspect(tmp_path,protocol)
    assert value.allowed and value.local is False and value.protocol==protocol
    assert fs.valid_results_admission(value,tmp_path/"new-results")
    assert not (tmp_path/"new-results").exists()
    assert value.source_admission.server_mtime_verified is False
    assert value.source_admission.atomic_rename_tested is False


@pytest.mark.parametrize("protocol,flags",[("unknown",{}),("local",{}),
    ("smb",{"offline_cache":True}),("smb",{"synchronized":True})])
def test_spelling_or_bad_os_evidence_is_not_remote_admission(tmp_path,protocol,flags):
    value=inspect(tmp_path,protocol,**flags)
    assert not value.allowed and not fs.valid_results_admission(value,tmp_path/"new-results")
    assert not (tmp_path/"new-results").exists()


def test_existing_positive_local_output_proof_remains_valid(tmp_path):
    root=tmp_path/"results"
    old=fs.LocalFileAdmission(True,"fixture-fixed","synthetic local proof","fixture",str(tmp_path))
    assert fs.valid_results_admission(old,root)
    value=fs.inspect_results_root(root,
        locality_provider=lambda _: fs.LocalFilesystemFacts("fixture-fixed",True,"fixture"),sync_roots=[])
    assert value.local is True and fs.valid_results_admission(value,root)
    assert not root.exists()


@pytest.mark.parametrize("filesystem",["unknown","synchronized","offline","SMB","nfs",None,""])
def test_legacy_local_proof_cannot_admit_remote_unknown_or_offline(tmp_path,filesystem):
    old=fs.LocalFileAdmission(True,filesystem,"fixture","fixture",str(tmp_path))
    assert not fs.valid_results_admission(old,tmp_path/"results")


def test_legacy_local_proof_cannot_admit_unc_without_probing(tmp_path,monkeypatch):
    old=fs.LocalFileAdmission(True,"ntfs","fixture","fixture",str(tmp_path))
    monkeypatch.setattr(Path,"resolve",lambda *a,**k:pytest.fail("UNC spelling must fail before any path probe"))
    assert not fs.valid_results_admission(old,r"\\not-a-real-host\share\out")


@pytest.mark.parametrize("field,value",[("local",1),("protocol","unknown"),("allowed",1),
    ("source_admission",None),("results_root","/unrelated"),("checked_container","/missing-container")])
def test_remote_proof_types_and_location_bindings_are_exact(tmp_path,field,value):
    proof=inspect(tmp_path,"smb")
    assert not fs.valid_results_admission(replace(proof,**{field:value}),tmp_path/"new-results")


def test_remote_wrapper_cannot_discard_refused_original_proof(tmp_path):
    proof=inspect(tmp_path,"smb",offline_cache=True)
    assert not fs.valid_results_admission(replace(proof,allowed=True),tmp_path/"new-results")


def test_remote_output_permission_never_changes_local_settings_admission(tmp_path):
    value=fs.inspect_local_file(tmp_path/"stores.toml",
        locality_provider=lambda _:fs.LocalFilesystemFacts("smb",False,"synthetic OS"),sync_roots=[])
    assert value.allowed is False
