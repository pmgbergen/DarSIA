"""Operational startup with injected transport facts only; no Git/job/claim/run."""
import pytest
from darsia.fluidflower.pipeline import worker
from darsia.fluidflower import job_filesystem as filesystem


@pytest.mark.parametrize("facts", [
    filesystem.FilesystemFacts("unknown", "unknown fixture"),
    filesystem.FilesystemFacts("smb", "offline fixture", offline_cache=True),
    filesystem.FilesystemFacts("smb", "synchronized fixture", synchronized=True),
])
def test_startup_refuses_transport_before_reading_job_or_calling_git(monkeypatch,tmp_path,facts):
    seen=[]
    monkeypatch.setattr(worker,"require_unc",lambda path:seen.append("original UNC spelling check"))
    def transport(path):
        seen.append("actual transport admission")
        return filesystem.classify_shared_folder(path,facts)
    monkeypatch.setattr(filesystem,"inspect_shared_folder",transport)
    monkeypatch.setattr(worker,"read_json",lambda *_:pytest.fail("Job must not be read on transport refusal"))
    monkeypatch.setattr(worker.subprocess,"check_output",lambda *_a,**_k:pytest.fail("Git must not be queried on transport refusal"))
    with pytest.raises(ValueError,match="Shared queue transport refused"):
        worker.verify_startup(tmp_path)
    assert seen==["actual transport admission"]


def test_positive_explicit_transport_continues_original_startup_without_claim_or_new_api(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(worker,"require_unc",lambda path:seen.append("unc"))
    def transport(path):
        seen.append("actual transport admission")
        return filesystem.classify_shared_folder(path,filesystem.FilesystemFacts("nfs","positive injected OS facts"))
    monkeypatch.setattr(filesystem,"inspect_shared_folder",transport)
    def original_job_reader(path):
        seen.append("original job read")
        raise RuntimeError("original reader sentinel; no actual job")
    monkeypatch.setattr(worker,"read_json",original_job_reader)
    with pytest.raises(RuntimeError,match="original reader sentinel"):
        worker.verify_startup(tmp_path)
    assert seen==["actual transport admission","original job read"]



def test_legacy_non_unc_is_still_refused_before_git_or_claim(monkeypatch,tmp_path):
    seen=[]
    def transport(path):
        seen.append("positive shared transport")
        return filesystem.classify_shared_folder(path,filesystem.FilesystemFacts("smb","explicit fixture facts"))
    def original_job(path):
        seen.append("legacy job metadata only")
        return {"format":"darsia-whole-run-job-v1"}
    monkeypatch.setattr(filesystem,"inspect_shared_folder",transport)
    monkeypatch.setattr(worker,"read_json",original_job)
    monkeypatch.setattr(worker.subprocess,"check_output",lambda *_a,**_k:pytest.fail("Legacy local spelling must fail before Git"))
    monkeypatch.setattr(worker,"JobQueue",lambda *_a,**_k:pytest.fail("Startup must never claim"))
    # Real require_unc remains active: positive transport is insufficient to
    # admit local-spelled legacy jobs, including on Windows.
    with pytest.raises(ValueError,match="UNC path required"):
        worker.verify_startup(tmp_path)
    assert seen==["positive shared transport","legacy job metadata only"]


def test_positive_nfs_alias_still_requires_explicit_runtime_schema_before_git(monkeypatch,tmp_path):
    reads=[]
    monkeypatch.setattr(filesystem,"inspect_shared_folder",lambda path:filesystem.classify_shared_folder(path,filesystem.FilesystemFacts("nfs","positive explicit OS facts")))
    monkeypatch.setattr(worker,"require_unc",lambda *_:pytest.fail("Positive alias NFS does not use legacy UNC spelling"))
    def pending_job(path):
        reads.append(path)
        return {"format":"darsia-alias-job-v1","format_version":1,"legacy_paths":False,"execution_ready":False}
    monkeypatch.setattr(worker,"read_json",pending_job)
    monkeypatch.setattr(worker.subprocess,"check_output",lambda *_a,**_k:pytest.fail("Pending alias must fail before Git"))
    monkeypatch.setattr(worker,"JobQueue",lambda *_a,**_k:pytest.fail("Startup must never claim"))
    with pytest.raises(RuntimeError,match="not explicitly ready"):
        worker.verify_startup(tmp_path)
    assert reads==[tmp_path/"job.json"]
