"""Portable UI launch reservations, never worker/process/scientific evidence."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import uuid

from .pipeline.job_queue import descriptor,name,read_json,write_immutable,_own_asset
from .pipeline.alias_job import job_metadata_reference

FORMAT="darsia-job-ui-launch-request-v2"
def _ref(root,path):
    return job_metadata_reference(root,path)
def _relative_path(root,value):
    path=Path(value)
    relative=path.relative_to(root).as_posix()
    if not relative or any(x in {"",".",".."} for x in relative.split("/")):
        raise ValueError("Launch metadata path escaped its own job")
    return relative
def _runtime(root,value,bound):
    identifier=Path(bound["path"]).name.removesuffix(".request.json")
    folder=root/"ui/launches"/identifier
    if (value.get("format")!=FORMAT or value.get("job")!=_ref(root,root/"job.json")
            or value.get("launch_directory")!=folder.relative_to(root).as_posix()
            or value.get("console")!=(folder/"console.log").relative_to(root).as_posix()
            or value.get("service_log")!="logs/"+name(value["machine"]+"-"+value["backend"])+".log"):
        raise ValueError("Alias launch request differs from its immutable job/namespace")
    from .job_launch_state import launch_key,_epoch,UNKNOWN
    launch_key(root,value["machine"],value["backend"])
    if (_epoch(value.get("requested_utc")) is None
            or type(value.get("workers")) is not int or value["workers"]<1
            or value.get("kind") not in {"local","ssh"}
            or type(value.get("service_log_offset")) is not int or value["service_log_offset"]<0):
        raise ValueError("Alias launch request lacks its actual count/time/log position")
    # Absolute paths below are this reader's in-memory locator only.
    return dict(value,source=bound,job_root=str(root),launch_directory=str(folder),
                console=str(folder/"console.log"),service_log=str(root/value["service_log"]),
                state=UNKNOWN,released=False,error_traceback="")

def begin(root,*,machine,backend,workers,kind,now=None):
    from .job_launch_state import launch_key,_epoch
    root=Path(root);launch_key(root,machine,backend)
    if kind not in {"local","ssh"} or type(workers) is not int or workers<1:
        raise ValueError("Explicit launch kind/count required")
    if read_json(root / "job.json").get("pipeline") == "preset-wasserstein" and (backend != "cpu" or workers != 1):
        raise ValueError("Original Comparison starts require CPU and exactly one worker per service")
    identifier=uuid.uuid4().hex;folder=root/"ui/launches"/identifier
    log=root/"logs"/(name(machine+"-"+backend)+".log")
    requested=(now or datetime.now(timezone.utc)).isoformat()
    if _epoch(requested) is None:raise ValueError("Aware launch request time required")
    value=dict(format=FORMAT,job=_ref(root,root/"job.json"),machine=name(machine),backend=backend,
               workers=workers,kind=kind,requested_utc=requested,
               launch_directory=folder.relative_to(root).as_posix(),
               console=(folder/"console.log").relative_to(root).as_posix(),
               service_log=log.relative_to(root).as_posix(),service_log_offset=log.stat().st_size if log.exists() else 0,
               numerical_progress_claimed=False,worker_liveness_verified=False)
    path=root/"ui/launches"/(identifier+".request.json")
    write_immutable(path,value)
    return _runtime(root,value,descriptor(path))

def error(request,text):
    root=Path(request["job_root"]);source=Path(request["source"]["path"])
    if descriptor(source)!=request["source"]:
        raise ValueError("Launch request changed before recording its original failure")
    original=read_json(source);_runtime(root,original,request["source"])
    path=source.with_name(source.name.replace(".request.json",".error.json"))
    write_immutable(path,dict(format="darsia-job-ui-launch-error-v2",
        request=_ref(root,source),traceback=text,console=original["console"],
        worker_liveness_verified=False,numerical_progress_claimed=False))

def read(root,*,reader=None):
    root=Path(root)
    from .job_launch_state import _read_bound,_terminal_report,_epoch
    result=[]
    for path in sorted((root/"ui/launches").glob("*.request.json")):
        raw,bound=_read_bound(path,reader=reader);row=_runtime(root,raw,bound)
        request_ref=_ref(root,path);identifier=path.name.removesuffix(".request.json")
        ep=path.with_name(identifier+".error.json")
        if ep.exists():
            value,_=_read_bound(ep,reader=reader)
            if (value.get("format")!="darsia-job-ui-launch-error-v2" or value.get("request")!=request_ref
                    or not isinstance(value.get("traceback"),str)):
                raise ValueError("Alias launch error differs from original request")
            row["error_traceback"]=value["traceback"]
        lp=Path(row["launch_directory"])/"launch.json"
        if lp.exists():
            value,asset=_read_bound(lp,reader=reader)
            if (value.get("format") not in {"darsia-job-local-launch-v2","darsia-job-ssh-launch-v2"}
                    or value.get("job")!=_ref(root,root/"job.json")):
                raise ValueError("Alias launcher receipt differs from its immutable job")
            row["launcher_receipt"]=dict(value,source=asset)
            if (value.get("format") == "darsia-job-local-launch-v2" and value.get("started") is False
                    and value.get("launcher_pid") is None):
                row.update(released=True, state="Launcher did not create a process")
        terminal=_terminal_report(row)
        if terminal:row.update(released=True,state="Original service reported "+terminal["event"]["kind"],terminal_report=terminal)
        preparation = Path(row["launch_directory"]) / "preparation-failure.json"
        if preparation.is_file():
            failure, asset = _read_bound(preparation, reader=reader)
            if (failure.get("format") != "darsia-job-startup-preparation-failure-v1"
                    or failure.get("job") != raw["job"] or failure.get("request") != request_ref
                    or failure.get("stage") != "prepare" or failure.get("worker_started") is not False
                    or failure.get("cache_scope_allocated") is not False
                    or _epoch(failure.get("utc")) is None or not isinstance(failure.get("traceback"), str)):
                raise ValueError("Startup preparation failure is not bound to its original request")
            row.update(released=True, state="Startup preparation failed; worker was not started",
                       error_traceback=failure["traceback"], preparation_failure=asset)
        for dp in sorted(path.parent.glob(identifier+".declaration.*.json")):
            value,asset=_read_bound(dp,reader=reader)
            if (value.get("format")!="darsia-job-ui-operator-declaration-v2"
                    or value.get("request")!=request_ref or value.get("job")!=raw["job"]
                    or value.get("operator_declares_previous_service_stopped") is not True
                    or value.get("scientific_evidence") is not False
                    or value.get("worker_liveness_verified") is not False
                    or value.get("numerical_progress_claimed") is not False
                    or _epoch(value.get("declared_utc")) is None):
                raise ValueError("Operator declaration has no original request or falsely claims evidence")
            row.update(released=True,state="Operator declared stopped; not independently verified",operator_declaration=asset)
        result.append(row)
    from .job_local_stop import apply_confirmations
    return apply_confirmations(root, result)

def declare(root,requests):
    root=Path(root);current={r["source"]["path"]:r for r in read(root)};written=[]
    for chosen in requests:
        row=current.get(chosen["source"]["path"])
        if row is None or row["source"]!=chosen["source"]:
            raise ValueError("Launch request changed before operator declaration")
        if row["released"]:continue
        path=Path(row["source"]["path"]);target=path.with_name(path.name.replace(".request.json",".declaration."+uuid.uuid4().hex+".json"))
        write_immutable(target,dict(format="darsia-job-ui-operator-declaration-v2",
            request=_ref(root,path),job=_ref(root,root/"job.json"),declared_utc=datetime.now(timezone.utc).isoformat(),
            operator_declares_previous_service_stopped=True,scientific_evidence=False,
            worker_liveness_verified=False,numerical_progress_claimed=False))
        written.append(descriptor(target))
    return written
