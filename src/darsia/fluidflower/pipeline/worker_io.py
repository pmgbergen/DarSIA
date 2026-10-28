"""Worker-side operational I/O wiring; no scientific executor is replaced."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from .job_queue import LocalOutputBinding, name, read_json
from .store_runtime import (resolve_catalog,verify_inputs,local_attempt_output,
                            RunUnavailable,InputHashMismatch)
from .run_io import RunIOBinding,alias_resume_output


class _PublicationRouter:
    def __init__(self, runtime):
        self.runtime=runtime
    def __call__(self,claim,candidate_descriptor,success):
        return self.runtime.factory(claim)(claim,candidate_descriptor,success)
    def publish_progress(self,claim,event):
        factory=self.runtime.factory(claim)
        callback=getattr(factory,"publish_progress",None)
        if not callable(callback):
            raise RuntimeError("Alias progress requires its actual shared-metadata publisher")
        return callback(claim,event)


class WorkerRunIO:
    def __init__(self,job_root,job,stores,service_id,backend,commit):
        if stores is None:
            raise RunUnavailable("Alias worker has no startup-admitted store map")
        if name(stores.machine_id+"-"+backend)!=service_id:
            raise RunUnavailable("Store-map machine differs from the actual worker service")
        self.root,self.job,self.stores=Path(job_root),deepcopy(job),stores
        self.service_id,self.commit=service_id,commit
        self.backend=backend
        self.repository=Path(__file__).resolve().parents[4]
        self.bindings,self.factories={},{}
        self.capacities={}
        self.capacity_refusals={}
        self.setup_admissions,self.setup_verified_inputs,self.setup_output_scopes={},{},{}
        from .capacity import remote_results_owner
        self.remote_results_scope=remote_results_owner(stores,job["series_id"],stores.roots(job["series_id"])["results"])
        self.publications=_PublicationRouter(self)
        from darsia_cache.folders import folder_mode, require_durable_path
        self.folder_mode = folder_mode(stores, job["series_id"])
        if self.folder_mode:
            from darsia_cache.adapters import require_same_startup_stores
            require_same_startup_stores(stores)
            require_durable_path(stores, job["series_id"], self.root)
            self.cache_adapter = None
        else:
            from darsia_cache.worker_adapter import WorkerCacheAdapter
            self.cache_adapter=WorkerCacheAdapter(self)

    def capacity_claim(self,owner,on_refusal=None):
        from .capacity import ClaimCapacity
        return ClaimCapacity(self,owner,on_refusal)

    def _setup_task(self,task):
        """Recognize only a typed Setup task joined to its original job."""
        if task.get("pipeline") != "series-setup":
            return False
        from .series_setup_protocol import validate_setup_claim
        validate_setup_claim(task,self.job)
        return True

    def _setup_helper(self,claim):
        """Require the actual retained whole-operation admission for this claim."""
        helper=self.setup_admissions.get(claim.token)
        if helper is None or helper.admission.task != claim.task:
            raise ValueError("Setup IO differs from its exact original claim admission")
        return helper

    @contextmanager
    def output_scope(self,claim):
        """Keep the actual Setup writer active through input replay and publication."""
        if not self._setup_task(claim.task):
            yield None
            return
        helper=self._setup_helper(claim)
        if not helper.bound or helper.sink is None:
            raise ValueError("Setup output requires all actual volume reservations")
        scopes=self.setup_output_scopes
        if claim.token in scopes:
            raise ValueError("Setup output scope cannot be reopened")
        state=dict(active=True,IO_closed=False,capacity_closed=False)
        scopes[claim.token]=state
        try:
            with helper.sink.activate():
                yield helper.sink
        finally:
            state["active"]=False
            # A leaked stream refuses closure and leaves every cap retained.
            # It is never forcibly closed or treated as executor retirement.
            state["evidence"]=helper.sink.close()
            state["IO_closed"]=True

    def capacity_progress(self,claim,owner,event):
        if self._setup_task(claim.task):
            if owner != self._setup_helper(claim).owner:
                raise ValueError("Capacity progress owner differs from actual Setup claim")
            return  # An operation event never releases a frame-sized prefix.
        ledger,reservation,original_owner=self.capacities[claim.token]
        if owner!=original_owner:
            raise ValueError("Capacity progress owner differs from actual claim")
        ledger.progress(reservation,owner,event)

    def output_factory(self,task,token):
        path,evidence=local_attempt_output(self.stores,self.job["series_id"],
                                          task["job_id"],task["run"],token)
        return LocalOutputBinding(self.stores.machine_id,
            Path(self.stores.roots(self.job["series_id"])["results"]),path,evidence)

    def availability(self,task):
        setup=self._setup_task(task)
        if setup and name(task["target_machine"]) != name(self.stores.machine_id):
            return None
        if setup and task.get("operation") == "image-preparation":
            from .setup_image_preparation import build_dependencies_ready
            if not build_dependencies_ready(self.root, self.job):
                return None
        if (task.get("execution_ready") is not True
                or task["input_catalog"].get("runtime_support")!="darsia-role-io-runtime-v1"):
            raise RunUnavailable("Prepared J2 metadata is not runtime-ready")
        checkpoint = getattr(self, "checkpoint", None)
        rows=(resolve_catalog(task,self.stores,self.job["series_id"])
            if checkpoint is None else resolve_catalog(task,self.stores,self.job["series_id"], checkpoint=checkpoint))
        output=self.output_factory(task,task["attempt_id"])
        if setup or task.get("pipeline") == "preset-wasserstein":
            if not setup:
                from ..preset_wasserstein_protocol import validate_preset_wasserstein_claim
                validate_preset_wasserstein_claim(task, self.job)
            return rows
        # Reuse the exact original resume selector before claim. It reads only
        # named own-job lineage/retirement metadata; full prefix SHA checks still
        # run through original verify_frame_prefix after actual claim.
        from .checkpoint import resume_source_for_claim
        if task.get("pipeline") == "native-phase-statistics":
            from .phase1_protocol import phase1_resume_source
            phase1_resume_source(SimpleNamespace(root=self.root),
                SimpleNamespace(task=task,token=task["attempt_id"],output=output.path,
                                local_output=output))
        else:
            resume_source_for_claim(SimpleNamespace(root=self.root),
                SimpleNamespace(task=task,token=task["attempt_id"],output=output.path,
                                local_output=output))
        return rows

    def prepare(self,claim):
        try:
            setup=self._setup_task(claim.task)
            if setup:
                helper=self._setup_helper(claim)
                if not self.setup_output_scopes.get(claim.token,{}).get("active"):
                    raise ValueError("Setup input replay requires its actual active output scope")
            if setup and getattr(self, "folder_mode", False):
                verified=verify_inputs(claim.task,claim.input_rows,
                    previous=self.setup_verified_inputs.get(claim.token),
                    **({"checkpoint": self.checkpoint} if hasattr(self, "checkpoint") else {}))
            else:
                verified=verify_inputs(claim.task,claim.input_rows,
                    **({"checkpoint": self.checkpoint} if hasattr(self, "checkpoint") else {}))
            if setup:
                self.setup_verified_inputs[claim.token] = verified
            binding=RunIOBinding.replay(claim.task,verified,claim.local_output,
                repository=Path(__file__).resolve().parents[4],
                admitted_commit=self.commit)
            lease = getattr(self.cache_adapter, "lease", None)
            first = getattr(lease, "first_qualification", None)
            if setup:
                if helper.admission.configuration != binding.configuration:
                    raise ValueError("Setup replay differs from its admitted operation controls")
            elif claim.task.get("pipeline") == "preset-wasserstein":
                from ..preset_wasserstein_protocol import validate_preset_wasserstein_claim
                validate_preset_wasserstein_claim(claim.task, self.job, config=binding.configuration)
                from darsia_cache.preset_qualification import PresetQualification
                if isinstance(first, PresetQualification):
                    first.require_task(self.cache_adapter.lease, claim.task)
                    first.require_output(claim.output)
                    if binding.configuration != first.configuration:
                        raise ValueError("Preset first replay differs from its exact released operation")
                elif not getattr(self, "folder_mode", False) and (first is not None or self.cache_adapter.lease.profile.get("format") not in {
                        "darsia-preset-wasserstein-bounded-profile-v1"}):
                    from ..preset_wasserstein_capacity import MEASURED_UNAVAILABLE
                    raise ValueError(MEASURED_UNAVAILABLE)
            elif first is not None and getattr(first, "common_mode", False):
                from darsia_cache.common_qualification import _compute_control, require_phase2_predecessor
                if first.stage != "calibration" and _compute_control(binding.configuration, first.stage) is None:
                    raise ValueError("The released common proof needs its declared CPU or auto control")
                first.require_task(self.cache_adapter.lease, claim.task)
                require_phase2_predecessor(first, binding.configuration)
                binding.opencl_proof_released = True
                binding.common_closure_released = True
                binding.common_job_binding = deepcopy(first.request["job"])
                binding.opencl_compare_cpu = True
            elif getattr(lease, "profile", {}).get("format") == "darsia-common-phase-import-profile-v1":
                from darsia_cache.common_qualification import runtime_compute_policy
                binding.opencl_policy = runtime_compute_policy(
                    self.cache_adapter.lease.profile, claim.task, binding.configuration)
        except (InputHashMismatch,RunUnavailable):
            raise
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise InputHashMismatch("Input admission failed before science: "+str(error)) from error
        self.bindings[claim.token]=binding
        return binding

    def finalize_common(self, claim):
        """Close science inside its original admitted slot, before budget/exit closure."""
        if self._setup_task(claim.task) or claim.task.get("pipeline") == "preset-wasserstein":
            return None
        lease = getattr(self.cache_adapter, "lease", None)
        first = getattr(lease, "first_qualification", None)
        if first is None or not getattr(first, "common_mode", False):
            return None
        binding = self.bindings[claim.token]
        from .common_worker_closure import finalize_worker_stage
        with binding.activate():
            return finalize_worker_stage(first, lease, claim=claim, binding=binding)

    def factory(self,claim):
        if claim.token not in self.factories:
            if claim.token not in self.bindings:
                raise RuntimeError("No admitted per-run binding for publication")
            from .publication_runtime import make_publication_factory
            self.factories[claim.token]=make_publication_factory(
                self.root,self.job,self.stores.machine_id,
                producer_origin_for_attempt=self.bindings[claim.token].producer_origin_for_attempt)
        return self.factories[claim.token]

    def capacity_close(self,claim,*,retain_capacity=False):
        if self._setup_task(claim.task):
            state=self.setup_output_scopes.get(claim.token,{})
            if state.get("capacity_closed"):
                return
            helper=self._setup_helper(claim)
            if state.get("active"):
                raise ValueError("Setup IO scope must close before capacity closure")
            if helper.sink is not None:
                helper.sink.close()
            else:
                helper.preclaim_sink.close()
            if retain_capacity:
                return
            if not state.get("IO_closed") or not helper.bound:
                raise ValueError("Setup capacity requires its actual closed output scope")
            self.factories.pop(claim.token,None)
            self.bindings.pop(claim.token,None)
            self.setup_verified_inputs.pop(claim.token,None)
            helper.admission.release(helper.owner)
            state["capacity_closed"]=True
            self.setup_admissions.pop(claim.token,None)
            return
        if self.cache_adapter is not None:
            self.cache_adapter.close_claim(claim)
        capacity=self.capacities.get(claim.token)
        if capacity is not None:
            ledger,reservation,owner=capacity
            ledger.release(reservation,owner)
            self.capacities.pop(claim.token)

    def release(self,claim,*,retain_capacity=False):
        if self._setup_task(claim.task):
            try:
                self.capacity_close(claim,retain_capacity=retain_capacity)
            finally:
                self.factories.pop(claim.token,None)
                self.bindings.pop(claim.token,None)
                self.setup_verified_inputs.pop(claim.token,None)
            return
        self.capacity_close(claim)
        self.factories.pop(claim.token,None)
        self.bindings.pop(claim.token,None)


def owner_for_attempt(root,task,service_id):
    """Read an exact live or archived owner; never scan another job/attempt."""
    run,token=name(task["run"]),task["attempt_id"]
    root=Path(root)
    sources=[root/"leases"/(run+"."+token)/"owner.json",
             root/"reclaimed"/run/token/"lease"/"owner.json",
             root/"publishing"/run/"owner.json"]
    matches=[]
    for path in sources:
        # publishing/<run> is a marker file in the in-place layout (POSIX:
        # NotADirectoryError, Windows: FileNotFoundError); its lease is above.
        try: owner=read_json(path)
        except (FileNotFoundError,NotADirectoryError): continue
        if owner.get("token")==token:
            if owner.get("machine")!=service_id or owner.get("task")!=task:
                raise ValueError("Actual owner and original task/service differ")
            matches.append(owner)
    if not matches or any(value!=matches[0] for value in matches):
        raise ValueError("No unique actual owner for this original attempt")
    return matches[0]


def retirement_output(root,task,service_id,stores):
    if stores is None:
        raise RunUnavailable("Alias parent has no startup store binding")
    owner=owner_for_attempt(root,task,service_id)
    path,evidence=local_attempt_output(stores,read_json(Path(root)/"job.json")["series_id"],
                                      task["job_id"],task["run"],task["attempt_id"])
    binding=LocalOutputBinding(stores.machine_id,
        Path(stores.roots(read_json(Path(root)/"job.json")["series_id"])["results"]),path,evidence)
    actual=alias_resume_output(owner,binding,task,task["attempt_id"])
    if actual.resolve()!=path.resolve():
        raise RunUnavailable("Actual owned retirement output differs")
    return actual,owner


def claimed_task_after_archive(root,service_id,event):
    """Bind a fast admission reject/publication after its atomic claim moved."""
    root=Path(root);run=name(event["run"]);token=event["attempt_id"]
    tasks=[]
    paths=[(root/"claimed"/service_id/(run+".json"),False),
           (root/"reclaimed"/run/token/"task.json",False),
           (root/"reclaimed"/run/token/"lease"/"owner.json",True),
           (root/"leases"/(run+"."+token)/"owner.json",True),
           (root/"publishing"/run/"owner.json",True)]
    for path,owner in paths:
        try: value=read_json(path)
        except (FileNotFoundError,NotADirectoryError): continue
        task=value.get("task") if owner else value
        if isinstance(task,dict) and task.get("attempt_id")==token:
            if owner and value.get("machine")!=service_id:
                raise ValueError("Claim archive belongs to another actual worker service")
            tasks.append(task)
    if not tasks or any(task!=tasks[0] for task in tasks):
        raise ValueError("No exact immutable current/archive task for claimed generation")
    return tasks[0]
