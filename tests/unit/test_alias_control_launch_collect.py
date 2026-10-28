"""Portable UI metadata and explicit collect commands with synthetic files only."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_control,job_launch_state,portable_job_launch,products
from darsia.fluidflower.pipeline import alias_job,job_queue as jobs,products_publication as publication
from .test_products_publication import fixture as product_fixture,publish as publish_fixture


def alias_root(tmp_path, *, master="Master"):
    root=tmp_path/"job"
    jobs.write_immutable(root/"job.json",dict(format="darsia-alias-job-v1",format_version=1))
    jobs.write_immutable(root/"ui/master.json",dict(format="darsia-job-ui-master-v2",
        machine=master,job=alias_job.job_metadata_reference(root,root/"job.json")))
    return root


def test_alias_control_open_is_read_only_and_commands_are_job_relative(tmp_path):
    root=alias_root(tmp_path)
    before={str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert job_control.read_control(root)==dict(action="run",sequence=0)
    after={str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert before==after and not (root/"ui/commands").exists()
    command=job_control.publish_control(root,"pause",machine="Master",expected_previous=None)
    assert set(command)=={"relative_path","sha256","bytes"}
    state=job_control.read_control(root)
    assert state["action"]=="pause" and state["sequence"]==1 and state["command"]==command
    assert state["format"]=="darsia-job-ui-control-v2"
    assert set(state["job"])==set(command) and set(state["master"])==set(command)


def test_alias_control_stale_displayed_previous_never_rebases_or_overwrites(tmp_path):
    root=alias_root(tmp_path)
    first=job_control.publish_control(root,"pause",machine="Master",expected_previous=None)
    path=root/first["relative_path"];raw=path.read_bytes()
    with pytest.raises(ValueError,match="changed; refresh"):
        job_control.publish_control(root,"stop",machine="Master",expected_previous=None)
    assert path.read_bytes()==raw
    assert len(list((root/"ui/commands").glob("*.json")))==1
    second=job_control.publish_control(root,"run",machine="Master",expected_previous=first)
    assert job_control.read_control(root)["previous"]==first and second!=first


def test_alias_control_nonmaster_cannot_create_command_namespace(tmp_path):
    root=alias_root(tmp_path)
    with pytest.raises(ValueError,match="recorded master machine"):
        job_control.publish_control(root,"stop",machine="Other",expected_previous=None)
    assert not (root/"ui/commands").exists()


def test_alias_control_chain_remains_bound_after_shared_root_moves(tmp_path):
    root=alias_root(tmp_path)
    first=job_control.publish_control(root,"pause",machine="Master",expected_previous=None)
    moved=tmp_path/"another-mount";root.rename(moved)
    assert job_control.read_control(moved)["command"]==first
    job_control.publish_control(moved,"run",machine="Master",expected_previous=first)
    assert job_control.read_control(moved)["sequence"]==2


def test_portable_launch_error_retains_unknown_reservation_and_literal_console_path(tmp_path):
    root=alias_root(tmp_path)
    request=job_launch_state.begin_launch_request(root,machine="Main",backend="cpu",workers=2,kind="ssh",
        now=datetime(2026,1,1,tzinfo=timezone.utc))
    source=Path(request["source"]["path"]);original=source.read_bytes()
    saved=jobs.read_json(source)
    assert saved["format"]=="darsia-job-ui-launch-request-v2"
    assert saved["launch_directory"].startswith("ui/launches/")
    assert saved["console"].endswith("/console.log")
    assert saved["service_log"]=="logs/main-cpu.log"
    assert "job_root" not in saved and "path" not in saved["job"]
    job_launch_state.record_launch_error(request,"RuntimeError: original SSH transport failure")
    [row]=job_launch_state.read_launch_requests(root)
    assert row["released"] is False and row["state"]==job_launch_state.UNKNOWN
    assert row["error_traceback"]=="RuntimeError: original SSH transport failure"
    assert row["console"]==request["console"] and source.read_bytes()==original
    assert row["worker_liveness_verified"] is False and row["numerical_progress_claimed"] is False


def test_operator_declaration_is_explicit_and_never_scientific_or_retirement_evidence(tmp_path):
    root=alias_root(tmp_path)
    request=portable_job_launch.begin(root,machine="Main",backend="cpu",workers=1,kind="local")
    [row]=portable_job_launch.read(root)
    assert row["released"] is False
    [asset]=portable_job_launch.declare(root,[row])
    declaration=jobs.read_json(Path(asset["path"]))
    assert declaration["scientific_evidence"] is False
    assert declaration["worker_liveness_verified"] is False
    assert declaration["numerical_progress_claimed"] is False
    [current]=portable_job_launch.read(root)
    assert current["released"] is True
    assert current["state"]=="Operator declared stopped; not independently verified"
    assert "terminal_report" not in current
    assert current["source"]==request["source"]


def test_only_new_matching_original_service_terminal_releases_launch(tmp_path):
    root=alias_root(tmp_path);log=root/"logs/main-cpu.log";log.parent.mkdir()
    time=datetime(2026,1,1,tzinfo=timezone.utc)
    old=(time-timedelta(seconds=1)).isoformat()+" "+json.dumps({"kind":"stopped"})+"\n"
    log.write_text(old,encoding="utf-8")
    portable_job_launch.begin(root,machine="Main",backend="cpu",workers=1,kind="ssh",now=time)
    [row]=portable_job_launch.read(root)
    assert row["released"] is False
    with log.open("a",encoding="utf-8") as stream:
        stream.write((time+timedelta(seconds=1)).isoformat()+" "+json.dumps({"kind":"startup_failure","traceback":"literal failure"})+"\n")
    [row]=portable_job_launch.read(root)
    assert row["released"] is True and row["terminal_report"]["event"]["traceback"]=="literal failure"


def test_collect_cli_uses_explicit_index_sha_and_shared_payload_without_original_files(tmp_path,monkeypatch):
    data=product_fixture(tmp_path)
    index=publish_fixture(tmp_path,data)
    payload=publication.publish_indexed_products(index,tmp_path/"published",
        current_machine_id="Main",file_writers_closed=True)
    Path(data["film"]["path"]).unlink()
    monkeypatch.setattr(products,"read_store_map",lambda *a,**k:SimpleNamespace(machine_id="Different-machine"))
    target=tmp_path/"collected"
    result=products.main(["collect","--index",index["path"],"--index-sha256",index["sha256"],
        "--index-bytes",str(index.get("bytes",index.get("size_bytes"))),"--product-id","film.mp4",
        "--target",str(target),"--published-bundle",payload["path"]])
    assert Path(result["path"]).name=="collection.json"
    assert (target/"products/film.mp4").read_bytes()==b"synthetic film bytes\x00\xff"


def test_collect_cli_wrong_explicit_index_sha_refuses_before_new_target(tmp_path,monkeypatch):
    data=product_fixture(tmp_path);index=publish_fixture(tmp_path,data)
    monkeypatch.setattr(products,"read_store_map",lambda *a,**k:SimpleNamespace(machine_id="Main"))
    target=tmp_path/"collected"
    with pytest.raises(ValueError,match="Metadata bytes differ"):
        products.main(["collect","--index",index["path"],"--index-sha256","0"*64,
            "--index-bytes",str(index.get("bytes",index.get("size_bytes"))),"--product-id","film.mp4","--target",str(target)])
    assert not target.exists()


def test_collect_cli_duplicate_explicit_source_is_not_silently_overridden(tmp_path,monkeypatch):
    data=product_fixture(tmp_path);index=publish_fixture(tmp_path,data)
    monkeypatch.setattr(products,"read_store_map",lambda *a,**k:SimpleNamespace(machine_id="Main"))
    target=tmp_path/"collected"
    with pytest.raises(ValueError,match="distinct PRODUCT_ID"):
        products.main(["collect","--index",index["path"],"--index-sha256",index["sha256"],
            "--index-bytes",str(index.get("bytes",index.get("size_bytes"))),"--product-id","film.mp4","--target",str(target),
            "--source","film.mp4="+data["film"]["path"],"--source","film.mp4="+data["export"]["path"]])
    assert not target.exists()
