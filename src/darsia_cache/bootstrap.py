"""Explicit stdlib-only admission before DarSIA/Numba/Qt imports.

Prepared only. No default quota, path, profile, cleanup or launcher. This module
can reserve/attach an import-scope lease; it never calls an exit 'retirement'.
Its own imports precede admission. Updating the current bytecode prefix below
routes later imports only; it does not retroactively cover startup/bootstrap
bytecode or an independent child's initially inherited GUI prefix.
"""
from __future__ import annotations
import atexit
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import sys
import tomllib
from types import SimpleNamespace
import uuid

from .policy import CachePolicyError, cache_settings, nonnegative, _descriptor
from .windows import WindowsNamespace, _parts
from .control import CacheBudgetLedger, _encoded
from .capacity import _capacity_host_context, read_bound, volume_info
from .environment import (derive_cache_environment, WRITER_SUBDIRECTORIES, WRITER_ENVIRONMENT_KEYS,
    FIRST_WRITER_SUBDIRECTORIES, FIRST_WRITER_ENVIRONMENT_KEYS, first_qualification_environment)

_LEASE = None
POLICY_ENV = "DARSIA_CACHE_POLICY"
RECEIPT_ENV = "DARSIA_CACHE_BOOTSTRAP_RECEIPT"
EXPECTED_STORES_ENV = "DARSIA_CACHE_BOOTSTRAP_EXPECTED_STORES"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _outside(path, package_file):
    path = Path(path).expanduser()
    if not path.is_absolute() or str(path).startswith("\\\\"):
        raise CachePolicyError("Explicit outside-repository local cache metadata required")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or getattr(before,"st_reparse_tag",0) or getattr(before,"st_file_attributes",0)&0x400:
        raise CachePolicyError("Cache metadata reparse/nonregular file refused")
    resolved=path.resolve(strict=True)
    if str(resolved).startswith("\\\\") or os.path.normcase(str(resolved))!=os.path.normcase(os.path.abspath(path)):
        raise CachePolicyError("Cache metadata aliases/reparse parents are not admitted implicitly")
    path=resolved
    roots = [Path(package_file).resolve().parent.parent]
    roots += [p for p in Path(package_file).resolve().parents if (p/".git").exists()]
    if any(path.is_relative_to(root) for root in roots):
        raise CachePolicyError("Cache policy/receipt must be outside repository/package")
    return path


def _explicit_document(path, package_file):
    path = _outside(path, package_file)
    before = path.stat()
    if not 0 < before.st_size <= 1024**2:
        raise CachePolicyError("Explicit stores/policy metadata exceeds one MiB")
    raw = path.read_bytes(); after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if version(before) != version(after) or len(raw) != before.st_size:
        raise CachePolicyError("Explicit stores/cache policy changed while reading")
    descriptor = dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    return tomllib.loads(raw.decode("utf-8-sig")), descriptor


def _exact(binding, package_file):
    _outside(binding["path"],package_file)
    return read_bound(binding)[0]


def _exact_authority_list(binding, package_file):
    """The N43 answers role alone is a pinned local list of record objects."""
    pin = _descriptor(binding)
    path = _outside(pin["path"], package_file)
    before = path.stat()
    if before.st_size != pin["size_bytes"]:
        raise CachePolicyError("Authority metadata SHA/size/version changed before read")
    with path.open("rb") as stream:
        raw = stream.read(pin["size_bytes"] + 1)
    after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if (version(before) != version(after) or len(raw) != pin["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != pin["sha256"]):
        raise CachePolicyError("Authority metadata SHA/size/version changed")
    return _authority_records(raw)


def _authority_records(raw):
    """Decode one captured raw authority list with the original strict parser."""
    from .capacity import _json_bytes
    # Reuse the original duplicate-key/nonfinite parser without changing the
    # original artifact bytes or the ordinary object-only metadata reader.
    wrapped = _json_bytes(('{"authority_records":' + raw.decode("utf-8-sig")
        + '}').encode("utf-8"))
    if set(wrapped) != {"authority_records"}:
        raise CachePolicyError("Authority metadata must be one list document")
    document = wrapped["authority_records"]
    if not isinstance(document, list) or any(not isinstance(row, dict) for row in document):
        raise CachePolicyError("Authority metadata must be a list of record objects")
    return document


class _HistoricalAuthorityReader:
    """Root-bound Q134 history for named preserved requests, never a fallback.

    New authority reads use the full immutable current snapshot. Only an exact
    preserved request may resolve the one declared old answers descriptor.
    Original requests, descriptors, and live authority files are never changed.
    """
    def __init__(self, binding, package_file):
        self.package_file = package_file
        self.binding = _descriptor(binding)
        value = _exact(self.binding, package_file)
        if (set(value) != {"format", "authority_ids", "current_authorities",
                "preserved_requests", "historical_authorities"}
                or value["format"] != "darsia-cache-historical-authority-resolution-v1"
                or value["authority_ids"] != ["Q134"]):
            raise CachePolicyError("Exact Root-bound Q134 historical authority map required")
        current = value["current_authorities"]
        if not isinstance(current, dict) or set(current) != {"coordinator_answers", "coordinator_status"}:
            raise CachePolicyError("Both immutable current authority snapshots are required")
        self.current = {}
        for role, row in current.items():
            if not isinstance(row, dict) or set(row) != {"source", "snapshot"}:
                raise CachePolicyError("Exact current authority source/snapshot pair required")
            source, snapshot = _descriptor(row["source"]), _descriptor(row["snapshot"])
            if (PureWindowsPath(source["path"]).name.casefold() != role + ".json"
                    or self._path(source) == self._path(snapshot)
                    or (source["sha256"], source["size_bytes"]) != (snapshot["sha256"], snapshot["size_bytes"])):
                raise CachePolicyError("Immutable snapshot must preserve exact current authority bytes outside its live path")
            _outside(source["path"], package_file)
            document = (_exact_authority_list(snapshot, package_file) if role == "coordinator_answers"
                else _exact(snapshot, package_file))
            self.current[role] = (source, snapshot, document)
        answers = self.current["coordinator_answers"][2]
        if not any(row.get("id") == "Q134-answer-option1-historical-authority-resolver-and-immutable-authority-snapshots"
                and row.get("decided_by") == "coordinator" for row in answers):
            raise CachePolicyError("Actual current Q134 authority required")
        requests = value["preserved_requests"]
        if not isinstance(requests, list) or not requests:
            raise CachePolicyError("Explicit preserved original request pins required")
        self.requests = [_descriptor(row) for row in requests]
        if len({self._path(row) for row in self.requests}) != len(self.requests):
            raise CachePolicyError("Preserved original request paths must be unique")
        history = value["historical_authorities"]
        if not isinstance(history, list) or len(history) != 1:
            raise CachePolicyError("Exactly one declared historical answers version required")
        row = history[0]
        if not isinstance(row, dict) or set(row) != {"original", "snapshot", "entry_count"}:
            raise CachePolicyError("Exact historical original/snapshot/count required")
        original, snapshot = _descriptor(row["original"]), _descriptor(row["snapshot"])
        historical = _exact_authority_list(snapshot, package_file)
        if (self._path(original) != self._path(self.current["coordinator_answers"][0])
                or self._path(original) == self._path(snapshot)
                or (original["sha256"], original["size_bytes"]) != (snapshot["sha256"], snapshot["size_bytes"])
                or type(row["entry_count"]) is not int or not 0 < row["entry_count"] < len(answers)
                or len(historical) != row["entry_count"] or answers[:len(historical)] != historical):
            raise CachePolicyError("Historical raw pin/path/entry-prefix provenance differs")
        self.original, self.snapshot = original, snapshot
        self._verify_actual_prefix(answers)

    @staticmethod
    def _path(pin):
        return str(PureWindowsPath(pin["path"])).casefold()

    def _verify_actual_prefix(self, snapshot):
        # This live read proves prefix provenance only; appended records never
        # replace the immutable request-time authority document used by the job.
        source = self.current["coordinator_answers"][0]
        path = _outside(source["path"], self.package_file)
        before = path.stat()
        if not 0 < before.st_size <= 16 * 1024**2:
            raise CachePolicyError("Bounded current authority prefix source required")
        with path.open("rb") as stream:
            raw = stream.read(before.st_size + 1)
        after = path.stat()
        version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if version(before) != version(after) or len(raw) != before.st_size:
            raise CachePolicyError("Current authority prefix source changed while reading")
        actual = _authority_records(raw)
        if actual[:len(snapshot)] != snapshot:
            raise CachePolicyError("Current authority entry-prefix provenance differs")

    def current_authorities(self, binding):
        pin = _descriptor(binding)
        if pin != self.current["coordinator_answers"][1]:
            raise CachePolicyError("Every new request must bind its immutable full-current answers snapshot")
        return _exact_authority_list(pin, self.package_file)

    def for_original(self, request):
        from copy import deepcopy
        if _descriptor(request) not in self.requests:
            raise CachePolicyError("Historical authority map is only for its preserved original requests")
        def read(binding):
            pin = _descriptor(binding)
            if self._path(pin) == self._path(self.original):
                if pin != self.original:
                    raise CachePolicyError("Historical original authority descriptor differs")
                # Recheck immutable bytes on every fresh reconstruction.
                return _exact_authority_list(self.snapshot, self.package_file)
            if pin == self.current["coordinator_answers"][1]:
                return self.current_authorities(pin)
            # Other already immutable historical snapshots keep their original
            # exact reader. No failed ordinary read is ever redirected.
            return deepcopy(_exact_authority_list(pin, self.package_file))
        return read


def _python_module_invocation():
    # Parse only interpreter options before the entry. A later script/module
    # argument spelled '-m pytest' must never reclassify the actual entry.
    argv=list(getattr(sys,"orig_argv",sys.argv));i=1
    flags={"-b","-bb","-B","-d","-E","-h","-i","-I","-O","-OO","-P",
           "-q","-R","-s","-S","-u","-v","-vv","-V","-x"}
    while i<len(argv):
        value=argv[i]
        if value=="-m":
            return (argv[i+1],argv[i+2:]) if i+1<len(argv) else None
        if value in {"-c","--","-"} or not value.startswith("-"):
            return None
        if value in {"-X","-W","--check-hash-based-pycs"}:
            if i+1>=len(argv):return None
            i+=2
        elif value in flags or value.startswith(("-X","-W")):
            i+=1
        else:
            return None  # Unsupported interpreter form is not silently tests.
    return None


def _worker_arguments():
    entry=_python_module_invocation()
    return entry[1] if entry and entry[0]=="darsia.fluidflower.worker" else None


def _stores_argument():
    tail=_worker_arguments()
    if tail is None:
        return None
    values=[]
    for i,value in enumerate(tail):
        if value=="--":break
        if value=="--stores":
            if i+1==len(tail) or tail[i+1].startswith("--"):
                raise CachePolicyError("Worker --stores requires its original explicit path")
            values.append(tail[i+1])
        elif value.startswith("--stores="):
            values.append(value.split("=",1)[1])
    if len(values)>1 or (values and not values[0]):
        raise CachePolicyError("Direct worker accepts one nonempty --stores cache policy before imports")
    return values[0] if values else None


def _default_store_map_path():
    # Same locator as fluidflower.store_map.default_store_map_path, without
    # importing darsia or probing/creating any cache root.
    if os.name=="nt":
        base=os.environ.get("LOCALAPPDATA")
        return Path(base)/"DarSIA"/"stores.toml" if base else None
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home()/".config"))/"darsia"/"stores.toml"



def _legacy_default_without_cache(path):
    """Classify only an ordinary default; this never admits a managed alias."""
    path=Path(path).expanduser()
    if not path.is_absolute() or str(path).startswith(("\\\\","//")):
        raise CachePolicyError("Default stores classification requires a local absolute path")
    resolved=path.resolve(strict=True)
    if str(resolved).startswith(("\\\\","//")):
        raise CachePolicyError("Default stores classification cannot read a network alias")
    try:
        before=resolved.stat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024**2:
            raise CachePolicyError("Default stores classification requires bounded regular metadata")
        with resolved.open("rb") as stream:
            raw=stream.read(1024**2+1)
        after=resolved.stat()
        if path.resolve(strict=True)!=resolved:
            raise CachePolicyError("Default stores alias retargeted during unmanaged classification")
    except FileNotFoundError as error:
        raise CachePolicyError("Default stores disappeared during unmanaged classification") from error
    # Reads may update access time; compare identity/content-version fields only.
    stable_fields=("st_dev","st_ino","st_mode","st_nlink","st_size",
                   "st_mtime_ns","st_ctime_ns","st_birthtime_ns",
                   "st_file_attributes","st_reparse_tag")
    if (any(getattr(before,name,None)!=getattr(after,name,None) for name in stable_fields)
            or len(raw)!=before.st_size or len(raw)>1024**2):
        raise CachePolicyError("Default stores changed during unmanaged classification")
    document=tomllib.loads(raw.decode("utf-8-sig"))
    return "cache" not in document and "cache_bootstrap" not in document


def _entry_kind():
    if _worker_arguments() is not None:
        return "worker"
    argv=list(getattr(sys,"orig_argv",sys.argv))
    entry=_python_module_invocation()
    if entry and entry[0]=="pytest":
        return "tests"
    if any(Path(x).name.casefold() in {"pytest","pytest.exe","pytest-script.py","py.test","py.test.exe"}
           for x in argv[:2]):
        raise CachePolicyError("Managed pytest console entry is unqualified; exact -m pytest is required")
    return "runtime"


def select_entry_role(document, *, entry_kind, role_identifier=None):
    policy=cache_settings(document,required=True)  # Pure validation before allocation.
    boot=document.get("cache_bootstrap")
    keys={"entry_roles","roles","namespace_admission","accounting_entry_limit"}
    if not isinstance(boot,dict) or set(boot)!=keys:
        raise CachePolicyError("Explicit entry roles/profiles/namespace/accounting bound required")
    entries,roles=boot["entry_roles"],boot["roles"]
    if (not isinstance(entries,dict) or set(entries)!={"runtime","worker","tests"}
            or not isinstance(roles,dict) or not roles):
        raise CachePolicyError("Configure runtime, worker and tests roles explicitly")
    for key,value in roles.items():
        if (not isinstance(key,str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}",key) is None
                or not isinstance(value,dict) or set(value)!={"purpose","profile"}
                or value["purpose"] not in {"jobs","gates","tests","display","logs"}):
            raise CachePolicyError("Invalid explicit cache role")
        _descriptor(value["profile"])
        if value["purpose"] not in policy.profiles:
            raise CachePolicyError("Every cache role purpose needs its explicit cache.profiles binding")
    if any(not isinstance(value,str) or value not in roles for value in entries.values()):
        raise CachePolicyError("Every entry must name an explicitly configured role")
    if entry_kind not in entries:
        raise CachePolicyError("Unknown cache entry kind")
    selected=entries[entry_kind]
    if role_identifier is not None and role_identifier!=selected:
        raise CachePolicyError("Inherited role differs from original entry mapping")
    role=roles[selected]
    if ((entry_kind=="worker" and role["purpose"]!="jobs")
            or (entry_kind=="tests" and role["purpose"]!="tests")):
        raise CachePolicyError("Worker/tests role purpose differs from its explicit entry")
    return dict(role_identifier=selected,entry_kind=entry_kind,
                purpose=role["purpose"],profile=_descriptor(role["profile"]))


def _owner():
    if os.name!="nt":
        raise CachePolicyError("Prepared cache bootstrap currently supports Windows local NTFS only")
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    kernel.GetProcessTimes.restype=wintypes.BOOL
    times=[wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(),*[ctypes.byref(t) for t in times]):
        raise ctypes.WinError(ctypes.get_last_error())
    creation=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
    context=_capacity_host_context()
    return dict(machine=context["host"],pid=os.getpid(),parent_pid=os.getppid(),
        process_create_time=str((creation-116444736000000000)/10000000),
        creation_filetime_100ns=str(creation),argv=list(sys.argv)),context


class ImportLease:
    """Active compatible pins, persistent reservation and actual owner records.

    No automatic adoption of external files and no cleanup on atexit. A later
    observer must prove all actual registered/declared writers retired.
    """
    def __init__(self,document,source,package_file,*,entry_kind,role_identifier=None):
        self.package_file=package_file;self.document=document;self.source=source
        self.policy=cache_settings(document,required=True)
        if self.policy.volume_cap is None and "DARSIA_CACHE_FIRST_QUALIFICATION" not in os.environ:
            raise CachePolicyError("Managed startup requires an accepted actual fixed-volume write-cap binding")
        role=select_entry_role(document,entry_kind=entry_kind,role_identifier=role_identifier)
        boot=document["cache_bootstrap"]
        self.role_identifier,self.entry_kind=role["role_identifier"],role["entry_kind"]
        self.purpose,self.profile_binding=role["purpose"],role["profile"]
        self.entry_limit=nonnegative(boot["accounting_entry_limit"],"Accounting entry limit")
        if not self.entry_limit:raise CachePolicyError("Positive explicit accounting limit required")
        from .first_qualification import REQUEST_ENV, FirstQualification, resolve_profile
        self.first_qualification=None
        if REQUEST_ENV in os.environ:
            if entry_kind!="worker" or self.purpose!="jobs":
                raise CachePolicyError("Finite first qualification requires the original worker/jobs entry")
            first_binding = json.loads(os.environ[REQUEST_ENV])
            first_type = FirstQualification
            first_format = _exact(first_binding, package_file).get("format")
            if first_format == "darsia-common-phase-qualification-request-v1":
                from .common_qualification import CommonQualification
                first_type = CommonQualification
            elif first_format == "darsia-setup-service-request-v1":
                from .setup_qualification import PositiveSetupServiceQualification
                first_type = PositiveSetupServiceQualification
            elif first_format == "darsia-setup-job-service-request-v1":
                from .setup_session import SetupJobService
                first_type = SetupJobService
            elif first_format == "darsia-preset-wasserstein-first-request-v1":
                from .preset_qualification import PresetQualification
                first_type = PresetQualification
            self.first_qualification=first_type(first_binding,
                provider_binding=self.profile_binding,stores_source=self.source,
                namespace=boot["namespace_admission"],read=lambda pin:_exact(pin,package_file),
                read_authorities=lambda pin:_exact_authority_list(pin,package_file))
            first=self.first_qualification.provider
            # Operational predeclared closure, explicitly NOT a measured profile.
            self.profile=dict(format="darsia-cache-first-entry-contract-v1",
                python_executable=first["python_executable"],module_closure=first["module_closure"],
                native_supervision=dict(startup=first["startup"]))
            self.maximum=0  # Not an entitlement: derived under the host lock in start().
        else:
            self.profile=resolve_profile(self.profile_binding,"import",lambda pin:_exact(pin,package_file),
                stores_source=self.source,
                read_authorities=lambda pin:_exact_authority_list(pin,package_file))
        if self.policy.volume_cap is None and not (self.first_qualification is not None
                and self.first_qualification.reference_build
                and self.first_qualification.reference_administration["host_policy"]=="second-n43-one-run-uncapped-d"):
            raise CachePolicyError("Managed startup requires an accepted actual fixed-volume write-cap binding")
        from .first_qualification import BOUND_IMPORT_PROFILE, checked_bound_import_context
        bounded = self.first_qualification is None and self.profile.get("format") == BOUND_IMPORT_PROFILE
        common = (self.first_qualification is None
                  and self.profile.get("format") == "darsia-common-phase-import-profile-v1")
        setup = (self.first_qualification is None
                 and self.profile.get("format") == "darsia-setup-service-import-profile-v1")
        preset = (self.first_qualification is None
                  and self.profile.get("format") == "darsia-preset-wasserstein-bounded-profile-v1")
        if preset:
            from .preset_qualification import checked_import_context
            self.maximum = checked_import_context(self.profile, purpose=self.purpose,
                role_identifier=self.role_identifier, entry_kind=self.entry_kind,
                policy=self.policy, namespace=boot["namespace_admission"],
                machine_id=self.document.get("machine", {}).get("id"))
        if setup:
            from .setup_qualification import checked_import_context
            self.maximum = checked_import_context(self.profile, purpose=self.purpose,
                role_identifier=self.role_identifier, entry_kind=self.entry_kind,
                policy=self.policy, namespace=boot["namespace_admission"])
        if common:
            from .common_qualification import checked_import_context
            self.maximum = checked_import_context(self.profile, purpose=self.purpose,
                role_identifier=self.role_identifier, entry_kind=self.entry_kind,
                policy=self.policy, namespace=boot["namespace_admission"])
        if bounded:
            self.maximum = checked_bound_import_context(self.profile, purpose=self.purpose,
                role_identifier=self.role_identifier, entry_kind=self.entry_kind,
                policy=self.policy, namespace=boot["namespace_admission"])
        if self.first_qualification is None and not bounded and not common and not setup and not preset and (self.profile.get("format")!="darsia-cache-import-profile-v1" or self.profile.get("purpose")!=self.purpose
                or self.profile.get("role_identifier")!=self.role_identifier
                or self.profile.get("entry_kind")!=self.entry_kind
                or self.profile.get("writer_contract")!="cooperative-start-admission-v1"):
            raise CachePolicyError("An applicable explicit import/service cache profile is required")
        if self.first_qualification is None:
            self.maximum=nonnegative(self.profile["maximum_cache_bytes"],"Import maximum cache bytes")
            complete=_exact(self.profile["completion"],package_file)
            if not self.maximum or complete.get("complete") is not True or complete.get("passed") is not True:
                raise CachePolicyError("Accepted completed source-bound import cache profile required")
        # No optional numerical/runtime identity or guessed extension of a profile.
        executable=Path(sys.executable).resolve()
        bound=self.profile["python_executable"]
        if executable!=Path(bound["path"]).resolve() or hashlib.sha256(executable.read_bytes()).hexdigest()!=bound["sha256"] or executable.stat().st_size!=bound["size_bytes"]:
            raise CachePolicyError("Import profile interpreter identity differs")
        for bound in self.profile["module_closure"]:
            path=Path(bound["path"]).resolve();raw=path.read_bytes()
            if len(raw)!=bound["size_bytes"] or hashlib.sha256(raw).hexdigest()!=bound["sha256"]:
                raise CachePolicyError("Import profile operational/source closure differs")
        self.admission=_exact(boot["namespace_admission"],package_file)
        self.owner,self.context=_owner();self.pins=[];self.receipt=None
        self.ledger=CacheBudgetLedger(self.policy,SimpleNamespace(**self.context),
            measurement_provider=self.measure,namespace_admission=self.admission,
            verify_host_admission=self._host_binding)

    def _host_binding(self,admitted,actual):
        return (admitted.get("host")==actual["host"]
            and actual["service_account_sid"] in admitted.get("authorized_sids",[])
            and Path(admitted["host_ledger"]["path"])==Path(actual["common_root"]))

    def measure(self,root, *, include_file_stats=False, hash_files=False):
        """Original live accounting; explicit quiet snapshots may also hash retained files."""
        if type(include_file_stats) is not bool or type(hash_files) is not bool or hash_files and not include_file_stats:
            raise CachePolicyError("Explicit retained-file inventory options required")
        if str(root)!=self.policy.root:raise CachePolicyError("Different configured cache root")
        rootpath=Path(root);tokens={};sizes={};control=self.ledger.control
        if include_file_stats:
            import time
            inventory_started=time.perf_counter()
            inventory_started_utc=_utc()
        for name in control.names():
            if re.fullmatch(r"cache-r-[0-9a-f]{32}\.json",name):sizes[name[8:-5]]=0
            if re.fullmatch(r"cache-s-[0-9a-f]{32}\.json",name):
                raw,_=control.read_once(name);row=json.loads(raw)
                relative=Path(row["root"]).relative_to(rootpath)
                if relative in tokens:raise CachePolicyError("Overlapping bound cache scopes")
                tokens[relative]=row["token"]
        entries=0;total=0;files=[]
        with_namespace=WindowsNamespace(rootpath,self.admission["cache_identity"])
        def walk(handle,relative):
            nonlocal entries,total
            directory_before=(rootpath/relative).lstat() if include_file_stats else None
            for name in with_namespace.api.names(handle):
                entries+=1
                if entries>self.entry_limit:raise CachePolicyError("Explicit accounting limit exceeded; no admission")
                rel=relative/name;st=(rootpath/rel).lstat()
                if getattr(st,"st_reparse_tag",0) or getattr(st,"st_file_attributes",0)&0x400:
                    raise CachePolicyError("Reparse cache entry preserved; no accounting traversal")
                directory=stat.S_ISDIR(st.st_mode)
                if directory:
                    child,identity=with_namespace.api.child(handle,name,directory=True)
                    try:walk(child,rel)
                    finally:with_namespace.api.close(child)
                else:
                    # Accounting does not open active payload files with the
                    # exclusive immutable-reader API. It takes lstat lengths;
                    # original remaining reservations still cover active scopes.
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:
                        raise CachePolicyError("Unqualified cache file type/link accounting")
                    total+=st.st_size
                    if include_file_stats:
                        row=dict(relative_path=str(rel),dev=st.st_dev,ino=st.st_ino,
                            size_bytes=st.st_size,mtime_ns=st.st_mtime_ns)
                        if hash_files:
                            # Only explicit before-launch/retired common snapshots
                            # request payload hashes; ordinary live accounting does not.
                            version=lambda info:(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
                            digest=hashlib.sha256();count=0
                            with (rootpath/rel).open("rb") as stream:
                                if version(os.fstat(stream.fileno()))!=version(st):
                                    raise CachePolicyError("Retained cache file changed before hash snapshot")
                                while block:=stream.read(1024*1024):
                                    digest.update(block);count+=len(block)
                                after=os.fstat(stream.fileno())
                            if (count!=st.st_size or version(after)!=version(st)
                                    or version((rootpath/rel).lstat())!=version(st)):
                                raise CachePolicyError("Retained cache file changed during hash snapshot")
                            row["sha256"]=digest.hexdigest()
                        files.append(row)
                    matches=[token for scope,token in tokens.items() if rel.is_relative_to(scope)]
                    if len(matches)>1:raise CachePolicyError("Overlapping bound scope accounting")
                    if matches:sizes[matches[0]]+=st.st_size
            if include_file_stats:
                directory_after=(rootpath/relative).lstat()
                version=lambda info:(info.st_dev,info.st_ino,info.st_mtime_ns)
                if version(directory_before)!=version(directory_after):
                    raise CachePolicyError("Retained cache directory changed during finite inventory")
        try:walk(with_namespace.handle,Path())
        finally:with_namespace.close()
        result=dict(root=self.policy.root,fresh=True,root_identity=self.admission["cache_identity"],
            no_reparse_traversal=True,active_scope_bytes=sizes,logical_bytes=total,
            free_bytes=volume_info(root)["free_bytes"],measured_utc=_utc(),eligible_reclaimable_bytes=0)
        if include_file_stats:
            result.update(files=files,payload_hashes_included=hash_files,
                observed_scan_anomalies=[], inventory_started_utc=inventory_started_utc,
                inventory_elapsed_seconds=time.perf_counter()-inventory_started,
                inventory_scope="entire original owned cache root, including redirected driver JIT files; outside-root OS writes unobserved")
        return result

    def _record(self,name,document):
        control=WindowsNamespace(self.admission["receipts"]["path"],self.admission["receipts"]["identity"])
        raw=_encoded(document)
        try:control.write_once(name,raw)
        finally:control.close()
        return dict(path=str(Path(self.admission["receipts"]["path"])/name),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))

    def start(self):
        token=uuid.uuid4().hex
        container=self.policy.purpose(self.purpose)/("current" if self.purpose=="tests" else "")
        bound=self.admission["purposes"][self.purpose]
        if Path(bound["path"])!=container:raise CachePolicyError("Admitted purpose differs from explicit policy")
        parent=tree=None
        with self.ledger.locked():
            if getattr(self,"first_qualification",None) is not None:
                self.maximum=self.first_qualification.service_entitlement(self)
            self.reservation=self.ledger.reserve(token,self.owner,self.maximum)
            self._record(token+".import-intent.json",dict(owner=self.owner,reservation=self.reservation,
                policy=self.source,profile=self.profile_binding,maximum_bytes=self.maximum))
            try:
                parent=WindowsNamespace(bound["path"],bound["identity"])
                tree=parent.scope(token,create=True)
                # FILE_CREATE plus this still-exclusive handle establishes that
                # no inherited cache contents precede this service's writers.
                initial_names=tree.api.names(tree.handles[""])
                if initial_names:
                    raise CachePolicyError("New exclusive UUID scope was not observed empty")
                first_writers = (getattr(self, "first_qualification", None) is not None
                    or self.profile.get("format") in {
                        "darsia-cache-volume-bound-import-profile-v1", "darsia-common-phase-import-profile-v1",
                        "darsia-setup-service-import-profile-v1", "darsia-preset-wasserstein-bounded-profile-v1"})
                writer_directories = FIRST_WRITER_SUBDIRECTORIES if first_writers else WRITER_SUBDIRECTORIES
                writer_keys = FIRST_WRITER_ENVIRONMENT_KEYS if first_writers else WRITER_ENVIRONMENT_KEYS
                fresh=dict(format="darsia-fresh-exclusive-empty-service-scope-v1",
                    token=token, owner=dict(self.owner), purpose=self.purpose,
                    root_identity=dict(tree.directories[""]),
                    creation_disposition="NtCreateFile.FILE_CREATE", exclusive_root_handle=True,
                    initial_names=[], initial_empty_observed_utc=_utc(),
                    observation_precedes_writer_subdirectories_and_imports=True,
                    writer_subdirectories=list(writer_directories),
                    writer_environment_keys=list(writer_keys))
                for name in writer_directories:
                    tree._parent(tuple(_parts(name)))  # Validated directory components; no phantom file.
                self.directories=dict(tree.directories);self.root=container/token
                manifest=dict(format="darsia-cache-external-import-scope-v1",token=token,owner=self.owner,
                    purpose=self.purpose,role_identifier=self.role_identifier,entry_kind=self.entry_kind,
                    root=str(self.root),root_identity=self.directories[""],
                    reservation=self.reservation,maximum_bytes=self.maximum,
                    fresh_scope_provenance=fresh,external_writer_file_ownership_not_yet_frozen=True)
                if getattr(self, "native_launch_context", None) is not None:
                    manifest["native_launch_context"] = dict(self.native_launch_context)
                self.ledger.require_unbound_scope(token)
                self.ledger.control.write_once("cache-s-"+token+".json",_encoded(manifest))
            finally:
                if tree is not None:tree.close()
                if parent is not None:parent.close()
        self.manifest=manifest
        # Reopen original IDs BEFORE enabling imports or any external writer.
        self._pin()
        receipt=dict(format="darsia-cache-import-service-receipt-v1",utc=_utc(),policy=self.source,
            role_identifier=self.role_identifier,entry_kind=self.entry_kind,
            namespace=self.document["cache_bootstrap"]["namespace_admission"],
            profile=self.profile_binding,manifest=manifest,directories=self.directories)
        if getattr(self,"first_qualification",None) is not None:
            receipt["first_qualification"]=dict(request=self.first_qualification.binding,
                provider=self.first_qualification.provider_binding,
                administrative_entitlement=self.first_qualification.administrative)
        self.receipt=self._record(token+".import-service.json",receipt)
        self.attach_owner()

    def attach(self,receipt):
        if receipt.get("format")!="darsia-cache-import-service-receipt-v1" or receipt["policy"]!=self.source:
            raise CachePolicyError("Inherited import receipt policy differs")
        if (receipt["namespace"]!=self.document["cache_bootstrap"]["namespace_admission"]
                or receipt["profile"]!=self.profile_binding
                or receipt.get("role_identifier")!=self.role_identifier
                or receipt.get("entry_kind")!=self.entry_kind):
            raise CachePolicyError("Inherited namespace/profile/role differs")
        manifest=receipt["manifest"];self.root=Path(manifest["root"]);self.reservation=manifest["reservation"]
        if getattr(self,"first_qualification",None) is not None:
            first=receipt.get("first_qualification",{})
            admin=first.get("administrative_entitlement",{})
            if (first.get("request")!=self.first_qualification.binding
                    or first.get("provider")!=self.first_qualification.provider_binding
                    or admin.get("measured_profile") is not False
                    or admin.get("maximum_bytes")!=manifest["maximum_bytes"]):
                raise CachePolicyError("Inherited first-mode administrative binding differs")
            self.maximum=nonnegative(manifest["maximum_bytes"],"Original first service entitlement")
            if not self.maximum:raise CachePolicyError("No first service entitlement")
        self.manifest=manifest
        self.directories=receipt["directories"]
        expected=self.policy.purpose(self.purpose)/("current" if self.purpose=="tests" else "")/self.reservation["token"]
        if (self.root!=expected or manifest["maximum_bytes"]!=self.maximum
                or manifest.get("role_identifier")!=self.role_identifier
                or manifest.get("entry_kind")!=self.entry_kind
                or manifest.get("purpose")!=self.purpose):
            raise CachePolicyError("Inherited scope/role differs from original reservation")
        with self.ledger.locked():
            self.ledger.require_reservation(self.reservation,manifest["owner"],self.maximum)
            raw,_=self.ledger.control.read_once("cache-s-"+self.reservation["token"]+".json")
            if json.loads(raw)!=manifest:raise CachePolicyError("Inherited scope manifest differs")
        self._pin();self.attach_owner()

    def _pin(self):
        try:
            for relative,identity in self.directories.items():
                if relative:_parts(relative)
                self.pins.append(WindowsNamespace(self.root/relative,identity))
        except BaseException:
            self.close_pins();raise

    def attach_owner(self):
        from .service_lifecycle import ServiceLifecycle
        ServiceLifecycle(self).register_owner()  # Atomic with the original-token seal.
        token=self.reservation["token"]
        self._record(token+".attach-"+str(self.owner["pid"])+"-"+self.owner["creation_filetime_100ns"]+".json",
            dict(utc=_utc(),owner=self.owner,reservation=self.reservation,policy=self.source,
                 registration_is_not_retirement=True,non_DarSIA_or_remote_children_not_implicitly_covered=True))

    def close_pins(self):
        errors=[]
        while self.pins:
            try:self.pins.pop().close()
            except BaseException as error:errors.append(str(error))
        if errors:raise CachePolicyError("Active pin close incomplete: "+repr(errors))

    def exit_intent(self):
        try:
            document=dict(utc=_utc(),owner=self.owner,reservation=self.reservation,
                actual_OS_retirement_not_proven=True)
            if getattr(self,"first_qualification",None) is not None:
                document["first_qualification_modules"]=self.first_qualification.loaded_modules()
            self._record(self.reservation["token"]+".exit-intent-"+str(self.owner["pid"])+"-"+self.owner["creation_filetime_100ns"]+".json",
                document)
        finally:self.close_pins()
        # No reservation close, adoption of arbitrary generated files or cleanup.


def before_darsia_import(*,package_file):
    if "DARSIA_SETUP_STARTUP_CAPTURE" in os.environ:
        from .setup_service import capture_startup
        capture_startup()  # Always exits before application imports or allocation.
    global _LEASE
    if _LEASE is not None:return _LEASE
    worker=_worker_arguments() is not None
    direct=_stores_argument();inherited=os.environ.get(RECEIPT_ENV)
    if POLICY_ENV in os.environ and not os.environ[POLICY_ENV]:
        raise CachePolicyError("Explicit cache policy environment path must be nonempty")
    from .folders import FOLDER_ENV, folder_roots, activate_folder, worker_selection
    folder_binding = json.loads(os.environ[FOLDER_ENV]) if FOLDER_ENV in os.environ else None
    if worker or folder_binding is not None:
        folder_path = direct or (folder_binding["source"]["path"] if folder_binding else None) or _default_store_map_path()
        if folder_path is None:
            raise CachePolicyError("Choose a Cache folder in Start > Series folders before starting a worker")
        folder_document, folder_source = _explicit_document(folder_path, package_file)
        has_folder = any(isinstance(roots, dict) and "cache" in roots
                         for roots in folder_document.get("stores", {}).values())
        series_id, job_root = ((worker_selection(_worker_arguments()) if worker else
                               (folder_binding["series_id"], folder_binding["job_root"]))
                              if has_folder or folder_binding is not None else (None, None))
        if series_id is not None and folder_roots(folder_document, series_id) is not None:
            if folder_binding is not None and folder_binding["source"] != folder_source:
                raise CachePolicyError("Selected stores changed before folder worker bootstrap")
            if EXPECTED_STORES_ENV in os.environ and json.loads(os.environ[EXPECTED_STORES_ENV]) != folder_source:
                raise CachePolicyError("Selected stores changed before folder worker bootstrap")
            # FolderContext is deliberately separate from _LEASE. No managed
            # namespace, entitlement or qualification is manufactured here.
            return activate_folder(folder_document, folder_source, series_id, job_root=job_root)
        if folder_binding is not None:
            raise CachePolicyError("Selected Cache folder was removed; reopen Start > Series folders")
        if "cache" not in folder_document and "cache_bootstrap" not in folder_document:
            raise CachePolicyError("Choose a Cache folder in Start > Series folders before starting a worker")
    binding=json.loads(inherited) if inherited is not None else None
    receipt=_exact(binding,package_file) if inherited is not None else None
    policy_path=direct or (receipt["policy"]["path"] if receipt else os.environ.get(POLICY_ENV))
    if policy_path:
        # Explicit worker/policy/receipt requests never fall back to unmanaged.
        document,source=_explicit_document(policy_path,package_file)
    else:
        policy_path=_default_store_map_path()
        if policy_path is None:
            if worker:raise CachePolicyError("Worker requires one configured external stores cache policy before imports")
            return None
        try:
            # A default legacy store map is not a managed-cache admission. Only
            # ordinary unbound entries may classify its bounded TOML via an OS
            # virtualized alias. Managed defaults still use the ORIGINAL path.
            if (not worker and EXPECTED_STORES_ENV not in os.environ
                    and _legacy_default_without_cache(policy_path)):
                return None
            document,source=_explicit_document(policy_path,package_file)
        except FileNotFoundError:
            if worker:raise CachePolicyError("Worker requires one configured external stores cache policy before imports")
            return None
        if "cache" not in document and "cache_bootstrap" not in document:
            if worker:raise CachePolicyError("Worker stores needs its complete managed cache policy before imports")
            return None
    if receipt and source!=receipt["policy"]:raise CachePolicyError("Explicit stores differs from inherited cache policy")
    if not inherited and any(name in sys.modules for name in ("numba","matplotlib")):
        raise CachePolicyError("Cache producer already imported without a pre-reserved receipt; no retroactive bootstrap claim")
    if EXPECTED_STORES_ENV in os.environ:
        expected=json.loads(os.environ[EXPECTED_STORES_ENV])
        if not isinstance(expected,dict) or expected!=source:
            raise CachePolicyError("Selected/default stores identity changed before child bootstrap")
    entry_kind=receipt["entry_kind"] if receipt else _entry_kind()
    role_identifier=receipt["role_identifier"] if receipt else None
    lease=ImportLease(document,source,package_file,entry_kind=entry_kind,role_identifier=role_identifier)
    # Only a separately qualified explicit profile enables native supervision.
    # Earlier cooperative profiles remain visible/unqualified for auto cleanup.
    native = "native_supervision" in getattr(lease, "profile", {})
    if native:
        from .owned_process import prepare_native_entry
        lease.native_launch_context = prepare_native_entry(lease,
            inherited_receipt=receipt is not None)
    elif "DARSIA_CACHE_NATIVE_LAUNCH" in os.environ:
        raise CachePolicyError("Native launch binding has no qualified original profile")
    try:
        if receipt:
            lease.receipt=binding;lease.attach(receipt)
        else:lease.start()
        q123 = getattr(getattr(lease, "first_qualification", None), "q123_cold_scope", None)
        if q123 is not None:
            q123.require_service(lease.first_qualification, lease)
        environment=derive_cache_environment(document,purpose=lease.purpose,token=lease.reservation["token"],
            scope_root=lease.root,parent_environment=os.environ)
        first_writers = (getattr(lease, "first_qualification", None) is not None
            or getattr(lease, "profile", {}).get("format") in {
                "darsia-cache-volume-bound-import-profile-v1", "darsia-common-phase-import-profile-v1",
                "darsia-setup-service-import-profile-v1", "darsia-preset-wasserstein-bounded-profile-v1"})
        if first_writers:
            environment = first_qualification_environment(environment, scope_root=lease.root)
            if q123 is not None:
                environment["MPLCONFIGDIR"] = q123.matplotlib_root(lease.first_qualification, lease)
        environment[RECEIPT_ENV]=json.dumps(lease.receipt,sort_keys=True,separators=(",",":"))
        writer_keys = FIRST_WRITER_ENVIRONMENT_KEYS if first_writers else WRITER_ENVIRONMENT_KEYS
        for key in (*writer_keys, RECEIPT_ENV):
            os.environ[key]=environment[key]
        # CPython uses this live attribute, not a later environment assignment.
        # Startup and the imports needed to reach this line are not covered.
        sys.pycache_prefix=environment["PYTHONPYCACHEPREFIX"]
        # This also covers already-imported stdlib tempfile in a legacy parent.
        import tempfile
        tempfile.tempdir=None
        if native:
            from .owned_process import report_native_service
            report_native_service(lease)
        _LEASE=lease;atexit.register(lease.exit_intent)
        return lease
    except BaseException:
        lease.close_pins()  # Persist reservation and every partial; no blind rollback.
        raise


def current_import_lease():
    if _LEASE is None:raise CachePolicyError("No pre-import reservation attached")
    return _LEASE
