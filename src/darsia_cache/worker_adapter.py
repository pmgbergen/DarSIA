"""Operational claim admission against the already admitted service cache.

No scratch allocator, producer invocation or cleanup lives here. Only an exact
observed service-only writer profile is supported; other profiles refuse before
claim. A zero claim entitlement is not a zero shared-service cache assertion.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path, PureWindowsPath

from .adapters import require_same_startup_stores, worker_identity_epoch
from .bootstrap import _exact, _owner
from .policy import CachePolicyError, _descriptor, _module_source_descriptor


def _key(path):
    return str(PureWindowsPath(path)).casefold()


def _coverage_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _source_pins(rows):
    if not isinstance(rows, list) or not rows:
        raise CachePolicyError("Observed exact producer/runtime source pins required")
    paths = set()
    for value in rows:
        pin = _module_source_descriptor(value)
        if _key(pin["path"]) in paths:
            raise CachePolicyError("Duplicate source identity in writer qualification")
        paths.add(_key(pin["path"]))
        # Descriptor checks only. Original ImportLease.__init__ already read
        # and verified this exact source/runtime closure at service startup.
    return paths


class WorkerCacheAdapter:
    """One service lease, explicit qualified claims, no invented claim files."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.lease = require_same_startup_stores(runtime.stores)
        self.namespace_admission = self.lease.admission
        self.profiles, self.claims = {}, {}
        first=getattr(self.lease,"first_qualification",None)
        setup=getattr(runtime,"job",{}).get("pipeline")=="series-setup"
        if setup:
            from darsia.fluidflower.pipeline.series_setup_protocol import validate_setup_job
            validate_setup_job(runtime.job)
        if first is not None and first.reference_build and not setup:
            from .first_qualification import ReferencePublicationFence
            original_factory=runtime.factory
            def reference_factory(claim):
                first.require_task(self.lease,claim.task)
                return ReferencePublicationFence(original_factory(claim),first,self.lease)
            runtime.factory=reference_factory

    def verify_preimport(self, stores, policy, *, create_binding=True):
        lease = require_same_startup_stores(stores)
        if lease is not self.lease or policy != lease.policy:
            raise CachePolicyError("Claim cache differs from the actual early service lease")
        receipt = _exact(lease.receipt, lease.package_file)
        if (receipt["policy"] != {k: stores.source[k] for k in ("path", "sha256", "size_bytes")}
                or receipt["profile"] != lease.profile_binding
                or receipt["manifest"]["reservation"] != lease.reservation
                or receipt["manifest"]["root"] != str(lease.root)):
            raise CachePolicyError("Actual pre-import service receipt differs")
        actual, _ = _owner()
        if any(actual[k] != lease.owner[k] for k in ("pid", "process_create_time")):
            raise CachePolicyError("Current worker did not attach its real pre-import owner")
        transaction = lease.ledger.locked() if create_binding else lease.ledger.locked(create_binding=False)
        with transaction:
            lease.ledger.require_reservation(lease.reservation,
                receipt["manifest"]["owner"], lease.maximum)
        return True

    def verify_host_admission(self, admitted, actual):
        return (admitted == self.namespace_admission
                and self.lease._host_binding(admitted, actual) is True)

    def budget_ledger(self, result_capacity):
        """Reuse the service ledger; its accounting callback needs its lock."""
        context = self.lease.context
        if (result_capacity.host != context["host"]
                or result_capacity.account != context["service_account_sid"]
                or _key(result_capacity.common_root) != _key(context["common_root"])):
            raise CachePolicyError("Result/cache physical host/account authority differs")
        return self.lease.ledger

    def project_expected_space(self, stores, series_id, tasks, estimates, result_ledger, *, job_root):
        """Real read projection for this existing admitted sequential worker slot.

        It does not bootstrap a service, initialize a binding, reserve, claim,
        register a writer or release/clean anything. Other service reservations
        remain accounted by the same host ledger. Unstarted services need their
        own admission; this method does not invent a concurrency measurement.
        """
        from .control import applicable_job_cache_profile
        from .capacity import volume_info
        if (series_id != self.runtime.job["series_id"]
                or job_root is None or _key(job_root) != _key(self.runtime.root)
                or stores.source != self.runtime.stores.source
                or len(tasks) != len(estimates) or not tasks):
            raise CachePolicyError("Projection requires the actual worker job/stores and exact estimates")
        if self.verify_preimport(stores, self.lease.policy, create_binding=False) is not True:
            raise CachePolicyError("Projection has no actual admitted pre-import service")
        ledger = self.budget_ledger(result_ledger)
        original_receipt = _exact(self.lease.receipt, self.lease.package_file)
        reservation_owner = original_receipt["manifest"]["owner"]
        profiles = []
        for task, estimate in zip(tasks, estimates):
            maximum, binding, profile, measured, completion = applicable_job_cache_profile(
                self.lease.policy, self, task, estimate, projection=True)
            from .first_qualification import BOUND_JOB_PROFILE
            if profile.get("format") == "darsia-preset-wasserstein-bounded-profile-v1":
                profiles.append(dict(run=task["run"], profile=binding, maximum_new_claim_bytes=maximum,
                    measurement=profile["measurement"], completion=profile["completion"],
                    physical_bound=profile["capacity_bound"],
                    operational_service_entitlement_bytes=profile["maximum_cache_bytes"],
                    sampled_cache_highwater_bytes=profile["sampled_cache_highwater_bytes"],
                    not_a_measured_profile=True, sampled_highwater_is_lifetime_peak=False, actual_writer_IO_proven=False))
            elif profile.get("format") in {BOUND_JOB_PROFILE, "darsia-common-phase-job-profile-v1"}:
                profiles.append(dict(run=task["run"], profile=binding, maximum_new_claim_bytes=maximum,
                    measurement=profile["measurement"], completion=profile["completion"],
                    physical_bound=profile["capacity_bound"],
                    operational_service_entitlement_bytes=profile["operational_service_entitlement_bytes"],
                    sampled_cache_highwater_bytes=profile["sampled_cache_highwater_bytes"],
                    sampled_highwater_is_lifetime_peak=False, actual_writer_IO_proven=False))
            else:
                profiles.append(dict(run=task["run"], profile=binding, maximum_new_claim_bytes=maximum,
                    measurement=profile["measurement"], completion=profile["completion"],
                    observed_service_peak_bytes=measured["worker_cache_service_coverage"]["maximum_service_cache_bytes"]))
        bounded = ["physical_bound" in row for row in profiles]
        if any(bounded) and not all(bounded):
            raise CachePolicyError("Mixed measured-lifetime and volume-bound projections require their own applicability")
        # The current qualified service-only adapter supports only zero new
        # claim-local bytes. Sum is conservative even if later adapters retain
        # per-run allocations; no service budget is multiplied by frame count.
        additional = sum(row["maximum_new_claim_bytes"] for row in profiles)
        products = sum(row.products_estimated_bytes for row in estimates)
        transient = sum(row.transient_estimated_bytes for row in estimates)
        with result_ledger.locked(create=False):
            other = result_ledger.remaining(exclude_job=job_root)
            result_volume = result_ledger.fresh_volume()
            with ledger.locked(create_binding=False):
                ledger.require_reservation(self.lease.reservation, reservation_owner, self.lease.maximum)
                cache_volume = volume_info(self.lease.policy.root)
                same = (result_ledger.remote_scope is None
                        and cache_volume["volume_id"] == result_volume["volume_id"])
                noncache = products + transient + other if same else 0
                budget = ledger.project(additional, noncache_bytes=noncache)
        result = dict(complete=True, state="QUALIFIED_EXISTING_SERVICE_READ_PROJECTION",
            sufficient=budget["sufficient"], required_cache_bytes=self.lease.maximum + additional,
            additional_cache_reservation_bytes=additional,
            already_reserved_service_maximum_bytes=self.lease.maximum,
            existing_service_reservation=self.lease.reservation,
            service_reservation_original_owner=reservation_owner,
            attached_service_actor=self.lease.owner,
            existing_service_profile=self.lease.profile_binding, profiles=profiles,
            eligible_reclaimable_logical_bytes=budget["eligible_reclaimable_logical_bytes"],
            quota_headroom_bytes=budget["quota_headroom_bytes"],
            post_admission_quota_headroom_bytes=budget["post_admission_quota_headroom_bytes"],
            fresh_cache_budget=budget, result_other_remaining_bytes=other,
            result_volume=result_volume, cache_volume=cache_volume,
            same_cache_and_results_volume=same,
            service_budget_counted_once_in_existing_used_and_unwritten=True,
            required_cache_semantics="one existing service maximum plus additional requested claim maxima; service is already reserved, not reserved again",
            scope="all supplied runs on this actual sequential worker slot; already active other services counted by shared ledger; no new service/concurrency promise",
            projection_only=True, reservation_created=False, claim_created=False,
            cleanup_performed=False, physical_reclaim_not_inferred=True)
        if all(bounded):
            result.update(qualification_kind="physical-volume-bound-sampled-operational-reserve",
                sampled_cache_highwater_bytes=max(row["sampled_cache_highwater_bytes"] for row in profiles),
                sampled_highwater_is_lifetime_peak=False, actual_writer_IO_proven=False)
        else:
            result["observed_service_peak_bytes"] = max(row["observed_service_peak_bytes"] for row in profiles)
        return result

    def measure(self, root):
        return self.lease.measure(root)

    def prepare_first_claim(self, task, estimate):
        """Explicit finite administration, never a qualified profile projection."""
        first=self.lease.first_qualification
        if self.runtime.backend!="cpu" or self.runtime.commit!=first.provider["scope"]["commit"]:
            raise CachePolicyError("First worker runtime/source differs from released scope")
        first.require_task(self.lease,task)
        if task.get("pipeline") == "preset-wasserstein":
            from .preset_qualification import PresetQualification
            from darsia.fluidflower.preset_wasserstein_capacity import (
                PresetWassersteinEntitlement, require_estimate, require_service_sources,
            )
            if not isinstance(first, PresetQualification) or not isinstance(estimate, PresetWassersteinEntitlement):
                raise CachePolicyError("Only the explicit preset first-service administration is applicable")
            require_estimate(task, estimate, allow_administration=True)
            if estimate.request != first.binding or estimate.administration != first.request["output_entitlement_plan"]:
                raise CachePolicyError("Preset cache/result administration differs from its original release")
            require_service_sources(self.runtime.repository,
                _source_pins(first.provider["producer_source_closure"]),
                _source_pins(first.provider["runtime_source_closure"]))
        if first.reference_build:
            from .first_qualification import reference_writer_coverage
            reference_writer_coverage(first, self.lease, writer_environment=True)
        self.profiles[task["attempt_id"]]=dict(task=deepcopy(task),binding=first.binding,
            first_administration=True,coverage_sha256=None)
        return first.binding

    def verify_profile(self, task, estimate, binding, profile, measured, complete):
        return self._verify_profile(task, estimate, binding, profile, measured, complete, remember=True)

    def verify_profile_projection(self, task, estimate, binding, profile, measured, complete):
        return self._verify_profile(task, estimate, binding, profile, measured, complete, remember=False)

    def _verify_profile(self, task, estimate, binding, profile, measured, complete, *, remember):
        """Require actual observed coverage; a source audit alone is insufficient.

        Existing measured-profile-v1 remains exact-count/commit. Its measurement
        must additionally bind this adapter's service-only writer qualification;
        no profile is synthesized from39 products or absence of a search hit.
        """
        from .first_qualification import BOUND_JOB_PROFILE
        if profile.get("format") == "darsia-preset-wasserstein-bounded-profile-v1":
            return self._verify_preset_bounded(task, estimate, binding, profile, measured, complete, remember=remember)
        if profile.get("format") == "darsia-common-phase-job-profile-v1":
            return self._verify_common_profile(task, estimate, binding, profile, measured, complete,
                                               remember=remember)
        if profile.get("format") == BOUND_JOB_PROFILE:
            return self._verify_bound_profile(task, estimate, binding, profile, measured, complete, remember=remember)
        preset = task.get("pipeline") == "preset-wasserstein"
        if preset:
            from darsia.fluidflower.preset_wasserstein_capacity import PROFILE_FORMAT, require_estimate
            require_estimate(task, estimate)
            if (profile.get("format") != PROFILE_FORMAT or self.runtime.backend != "cpu"
                    or task.get("precision") != "original" or task.get("backend") != "cpu"
                    or self.runtime.commit != task.get("commit")
                    or profile.get("scope") != estimate.scope or measured.get("scope") != estimate.scope
                    or complete.get("scope") != estimate.scope
                    or self.lease.profile.get("format") != "darsia-cache-import-profile-v1"
                    or getattr(self.lease, "first_qualification", None) is not None):
                raise CachePolicyError("Exact observed original preset service/task scope required")
        elif (self.runtime.backend != "cpu" or task.get("pipeline") not in {"native-phase-n1", "n1"}
                or task.get("precision") != "float32"):
            raise CachePolicyError("This adapter requires a separately observed CPU float32 native-phase-n1 or N1 writer route")
        coverage = measured.get("worker_cache_service_coverage")
        if not isinstance(coverage, dict):
            raise CachePolicyError("No observed worker/service writer qualification")
        fields = {"format", "complete", "passed", "commit", "pipeline", "backend",
                  "precision", "case_count", "allocation_model", "service_profile",
                  "namespace_admission", "stores_source", "producer_source_closure",
                  "runtime_source_closure", "shared_service_writers",
                  "claim_local_disposable_writers", "unclassified_writers",
                  "actual_writer_IO_observed", "maximum_service_cache_bytes"}
        if (set(coverage) != fields
                or coverage["format"] != "darsia-worker-cache-service-only-qualification-v1"
                or coverage["complete"] is not True or coverage["passed"] is not True
                or coverage["actual_writer_IO_observed"] is not True
                or coverage["allocation_model"] != "shared-service-only-no-claim-files-v1"
                or coverage["commit"] != self.runtime.commit):
            raise CachePolicyError("Observed worker cache qualification is not applicable")
        if (coverage["commit"] != task["commit"]
                or coverage["pipeline"] != task.get("pipeline")
                or coverage["backend"] != self.runtime.backend
                or coverage["precision"] != task.get("precision")
                or type(coverage["case_count"]) is not int
                or coverage["case_count"] != len(task["ordered_case_ids"])
                or coverage["service_profile"] != self.lease.profile_binding
                or coverage["namespace_admission"] != self.lease.document["cache_bootstrap"]["namespace_admission"]
                or coverage["stores_source"] != self.lease.source
                or coverage["claim_local_disposable_writers"] != []
                or coverage["unclassified_writers"] != []
                or coverage["shared_service_writers"] != ["bytecode", "matplotlib", "numba", "tempfile"]
                or complete.get("worker_cache_service_coverage_sha256") != _coverage_hash(coverage)):
            raise CachePolicyError("Source/runtime/role/writer coverage is unqualified")
        if profile["maximum_cache_bytes"] != 0 or measured["maximum_observed_cache_bytes"] != 0:
            raise CachePolicyError("No actual positive claim-local writer adapter; no claim")
        service_peak = coverage["maximum_service_cache_bytes"]
        if type(service_peak) is not int or not 0 <= service_peak <= self.lease.maximum:
            raise CachePolicyError("Observed full service-cache requirement exceeds its existing reservation")
        producers = _source_pins(coverage["producer_source_closure"])
        runtime_sources = _source_pins(coverage["runtime_source_closure"])
        root = self.runtime.repository / "src" / "darsia" / "fluidflower" / "pipeline"
        required = [Path(__file__), root / "worker_io.py", root / "capacity.py",
                    root / "runner.py", root / "series.py", root.parent / "native" / "sidecar.py",
                    root.parent / "mass" / "n1_pipeline.py"]
        if preset:
            from darsia.fluidflower.preset_wasserstein_capacity import require_service_sources
            require_service_sources(self.runtime.repository, producers, runtime_sources)
            required = []  # The separate preset boundary set was checked above.
        elif task["pipeline"] == "n1":
            required += [root / "staged_data.py", root / "job_inputs.py",
                         root.parent / "mass" / "pixel_pipeline.py", root.parent / "mass" / "pixels.py",
                         root.parent / "geometry" / "pixels.py", root.parent / "storage_extensive.py",
                         root.parent / "storage_extensive_io.py", root.parent / "storage_n1_outputs.py",
                         root.parent / "storage_n1_saved_products.py", root.parent / "storage_outputs.py",
                         root.parent / "storage_resolution.py"]
        if not {_key(p) for p in required}.issubset(producers):
            raise CachePolicyError("Observed producer closure omits an actual worker/output boundary")
        combined = coverage["producer_source_closure"] + coverage["runtime_source_closure"]
        startup_closure = self.lease.profile["module_closure"]
        if (len({_key(pin["path"]) for pin in combined}) != len(combined)
                or any(pin not in combined for pin in startup_closure)
                or any(pin not in startup_closure for pin in combined)):
            raise CachePolicyError("Claim proof differs from the exact startup-verified source/runtime closure")
        for suffix in ("numba/core/caching.py", "matplotlib/font_manager.py",
                       "tempfile.py", "importlib/_bootstrap_external.py"):
            if not any(p.replace("\\", "/").endswith(suffix) for p in runtime_sources):
                raise CachePolicyError("Observed runtime writer is not source-bound: " + suffix)
        if remember:
            self.profiles[task["attempt_id"]] = dict(task=deepcopy(task), binding=dict(binding),
                coverage_sha256=_coverage_hash(coverage))
        return True

    def _verify_preset_bounded(self, task, estimate, binding, profile, measured, complete, *, remember):
        from .preset_qualification import checked_import_context
        from darsia.fluidflower.preset_wasserstein_capacity import require_bounded_estimate, require_service_sources
        require_bounded_estimate(task, estimate, profile, binding)
        lease = self.lease
        checked_import_context(profile, purpose=lease.purpose, role_identifier=lease.role_identifier,
            entry_kind=lease.entry_kind, policy=lease.policy,
            namespace=lease.document["cache_bootstrap"]["namespace_admission"],
            machine_id=lease.document.get("machine", {}).get("id"))
        if (getattr(lease, "first_qualification", None) is not None or profile != lease.profile
                or binding != lease.profile_binding or profile["stores_source"] != lease.source
                or self.runtime.backend != "cpu" or self.runtime.commit != task["commit"]
                or profile["scope"]["physical_host"].casefold() != lease.context["host"].casefold()
                or profile["scope"]["target_machine"] != lease.document.get("machine", {}).get("id")
                or profile["scope"]["target_machine"] != self.runtime.stores.machine_id
                or profile["maximum_cache_bytes"] != lease.maximum
                or complete.get("format") != "darsia-preset-wasserstein-bounded-completion-v1"
                or complete.get("complete") is not True or complete.get("passed") is not True
                or complete.get("not_a_measured_profile") is not True
                or measured.get("complete") is not True or measured.get("first_stop") is not None
                or measured.get("output_hard_quota") is not False
                or measured.get("sampled_highwater_is_lifetime_peak") is not False):
            raise CachePolicyError("Preset bounded claim differs from its actual admitted physical service and closed first execution")
        require_service_sources(self.runtime.repository, _source_pins(profile["producer_source_closure"]),
                                _source_pins(profile["runtime_source_closure"]))
        combined = profile["producer_source_closure"] + profile["runtime_source_closure"]
        if (len({_key(pin["path"]) for pin in combined}) != len(combined)
                or {_coverage_hash(pin) for pin in combined} != {_coverage_hash(pin) for pin in lease.profile["module_closure"]}):
            raise CachePolicyError("Preset bounded proof changed the exact startup-verified source/runtime closure")
        if remember:
            self.profiles[task["attempt_id"]] = dict(task=deepcopy(task), binding=dict(binding),
                coverage_sha256=_coverage_hash(profile), qualification_kind="explicit-preset-physical-bound-administration")
        return True

    def _verify_common_profile(self, task, estimate, binding, profile, measured, complete, *, remember):
        """An actually resolved common DAG, never a full-profile compatibility cast."""
        from .common_qualification import require_applicable, IMPORT_PROFILE
        stage = require_applicable(profile, task)
        facts = getattr(estimate, "estimate", None)
        scope = getattr(estimate, "scope", None)
        control = (facts.get("compute_control") if isinstance(facts, dict)
                   else scope.get("compute_control") if isinstance(scope, dict) else None)
        if control != profile.get("compute_profiles", {}).get(stage, {}).get("control"):
            raise CachePolicyError("Common result projection changes its qualified optional-compute control")
        imported = self.lease.profile
        if (imported.get("format") != IMPORT_PROFILE or self.runtime.backend != "cpu"
                or self.runtime.commit != task["commit"] or task["backend"] != self.runtime.backend
                or binding != self.lease.profile_binding or profile.get("provider") != binding
                or profile.get("stores_source") != self.lease.source
                or profile.get("namespace_admission") != self.lease.document["cache_bootstrap"]["namespace_admission"]
                or any(profile.get(key) != imported.get(key) for key in (
                    "scope", "stages", "stage_routes", "device_identity", "python_executable", "module_closure",
                    "producer_source_closure", "runtime_source_closure", "capacity_bound",
                    "compute_profiles", "result_spaces"))
                or profile.get("operational_service_entitlement_bytes") != self.lease.maximum
                or profile.get("additional_claim_cache_bytes") != 0
                or profile.get("allocation_model") != "shared-service-only-no-claim-files-v1"
                or profile.get("actual_writer_IO_proven") is not False
                or profile.get("sampled_highwater_is_lifetime_peak") is not False
                or measured.get("complete") is not True or measured.get("first_stop") is not None
                or measured.get("cache_operational_reserve_crossed") is not False
                or complete.get("complete") is not True or complete.get("passed") is not True):
            raise CachePolicyError("Common profile differs from the actual early service/runtime/source closure")
        cap = self.lease.policy.volume_cap
        observation = profile["capacity_bound"]["volume_cap_observation"]
        if (cap is None or cap.maximum_size_bytes != self.lease.policy.quota_bytes
                or observation.get("binding") != cap.document()
                or observation.get("virtual_size_bytes") != cap.maximum_size_bytes):
            raise CachePolicyError("Common profile has another actual physical-volume bound")
        combined = profile["producer_source_closure"] + profile["runtime_source_closure"]
        if (len({_key(pin["path"]) for pin in combined}) != len(combined)
                or {_coverage_hash(pin) for pin in combined} != {
                    _coverage_hash(pin) for pin in imported["module_closure"]}):
            raise CachePolicyError("Common profile omits an actual startup producer/runtime boundary")
        if task.get("pipeline") == "n1-calibration-refit":
            from darsia.fluidflower.pipeline.refit_capacity import require_estimate, require_service_sources
            require_estimate(task, estimate)
            require_service_sources(self.runtime.repository, _source_pins(profile["producer_source_closure"]),
                                    _source_pins(profile["runtime_source_closure"]))
        if remember:
            self.profiles[task["attempt_id"]] = dict(task=deepcopy(task), binding=dict(binding),
                coverage_sha256=_coverage_hash(profile["stages"]), qualification_kind="common-phase-physical-bound")
        return True

    def _verify_bound_profile(self, task, estimate, binding, profile, measured, complete, *, remember):
        """A typed volume-bound profile, for every task on its qualified head and route.

        The task may be another run than the qualified one (N188 D2; another
        run must be a full task, see applicable_job_cache_profile): its claim
        adds no cache bytes, and the shared service keeps the qualified
        entitlement. Commit, backend, pipeline and precision stay exact. A
        native-phase-n1 profile also covers a phase-one statistics task and a
        phase-two n1 task on its qualified head, which run the same stages;
        each route's own boundary modules must be in the qualified closure.
        """
        from .first_qualification import BOUND_WRITER_CONTRACT
        qualified = profile.get("scope") if isinstance(profile.get("scope"), dict) else {}
        # A phase-one statistics task and a phase-two n1 task run the qualified
        # native-phase-n1 stages.
        staged = (task.get("pipeline") in {"native-phase-statistics", "n1", "n1-calibration-refit"}
                  and qualified.get("pipeline") == "native-phase-n1")
        phase1 = staged and task.get("pipeline") == "native-phase-statistics"
        refit = staged and task.get("pipeline") == "n1-calibration-refit"
        if refit:
            from darsia.fluidflower.pipeline.refit_capacity import require_estimate
            require_estimate(task, estimate)
        keys = ("commit", "backend", "precision") if staged else ("commit", "pipeline", "backend", "precision")
        if (not qualified
                or any(task.get(key) != qualified.get(key) for key in keys)
                or task.get("commit") != self.runtime.commit
                or task.get("backend") != self.runtime.backend or self.runtime.backend != "cpu"
                or task.get("pipeline") != (task.get("pipeline") if staged else "native-phase-n1")
                or task.get("precision") != "float32"
                or profile.get("provider") != binding or binding != self.lease.profile_binding
                or profile.get("namespace_admission") != self.lease.document["cache_bootstrap"]["namespace_admission"]
                or profile.get("stores_source") != self.lease.source
                or profile.get("writer_contract") != BOUND_WRITER_CONTRACT
                or profile.get("additional_claim_cache_bytes") != 0
                or profile.get("allocation_model") != "shared-service-only-no-claim-files-v1"
                or profile.get("actual_writer_IO_proven") is not False
                or profile.get("sampled_highwater_is_lifetime_peak") is not False
                or profile.get("operational_service_entitlement_bytes") != self.lease.maximum
                or measured.get("complete") is not True
                or measured.get("cache_operational_reserve_crossed") is not False
                or measured.get("first_stop") is not None
                or complete.get("complete") is not True or complete.get("passed") is not True
                or complete.get("scope") != qualified):
            raise CachePolicyError("Typed physical-bound qualification is not applicable to this actual service/task")
        cap = self.lease.policy.volume_cap
        if (cap is None or cap.maximum_size_bytes != 500*1024**3
                or profile["capacity_bound"]["volume_cap_observation"].get("binding") != cap.document()):
            raise CachePolicyError("This ordinary service has another physical cache bound")
        producers = _source_pins(profile["producer_source_closure"])
        runtime_sources = _source_pins(profile["runtime_source_closure"])
        root = self.runtime.repository / "src" / "darsia" / "fluidflower" / "pipeline"
        required = [Path(__file__), root / "worker_io.py", root / "capacity.py", root / "runner.py",
            root / "series.py", root.parent / "native" / "sidecar.py", root.parent / "mass" / "n1_pipeline.py"]
        if phase1:
            # The phase-one route's own boundary: its runner, DATA, protocol and writers.
            required += [root / "phase_statistics.py", root / "staged_data.py", root / "phase1_protocol.py",
                         root / "run_io.py", root / "phase1_capacity.py", root.parent / "storage_outputs.py",
                         root.parent / "storage_original_photo.py",
                         root.parent / "phase" / "probability_storage.py"]
        elif staged and not refit:
            # The n1 route's own boundary, as the measured n1 route requires.
            required += [root / "staged_data.py", root / "job_inputs.py", root / "n1_capacity.py",
                         root.parent / "mass" / "pixel_pipeline.py", root.parent / "mass" / "pixels.py",
                         root.parent / "geometry" / "pixels.py", root.parent / "storage_extensive.py",
                         root.parent / "storage_extensive_io.py", root.parent / "storage_n1_outputs.py",
                         root.parent / "storage_n1_saved_products.py", root.parent / "storage_outputs.py",
                         root.parent / "storage_resolution.py"]
        combined = profile["producer_source_closure"] + profile["runtime_source_closure"]
        if (not {_key(path) for path in required} <= producers
                or len({_key(pin["path"]) for pin in combined}) != len(combined)
                or {_coverage_hash(pin) for pin in combined} != {
                    _coverage_hash(pin) for pin in self.lease.profile["module_closure"]}):
            raise CachePolicyError("Typed qualification differs from the actual startup writer/source/runtime closure")
        for suffix in ("numba/core/caching.py", "matplotlib/font_manager.py", "tempfile.py", "importlib/_bootstrap_external.py"):
            if not any(path.replace("\\", "/").endswith(suffix) for path in runtime_sources):
                raise CachePolicyError("Actual bounded runtime writer is missing: " + suffix)
        if refit:
            # Service-only cache applicability requires this original fitter's
            # complete source closure; native scientific evidence is not reused.
            from darsia.fluidflower.pipeline.refit_capacity import require_service_sources
            require_service_sources(self.runtime.repository, producers, runtime_sources)
        if remember:
            self.profiles[task["attempt_id"]] = dict(task=deepcopy(task), binding=dict(binding),
                coverage_sha256=_coverage_hash(profile["qualification"]), qualification_kind="physical-volume-bound")
        return True

    def _is_current_claim_owner(self, owner):
        """True only if a claim owner is this very process, compared exactly.

        A claim owner is the queue's worker identity, whose epoch is the
        worker_identity_epoch form. The current process must still be the lease
        owner (pid, exact epoch and FILETIME count), and the claim owner's pid
        and epoch must equal this process in the worker identity form.
        """
        actual, _ = _owner()
        return (all(actual.get(k) == self.lease.owner.get(k)
                    for k in ("pid", "process_create_time", "creation_filetime_100ns"))
                and owner.get("pid") == actual["pid"]
                and owner.get("process_create_time") == worker_identity_epoch(actual["pid"]))

    def register_claim_allocation(self, claim, ledger, reservation, owner, maximum, profile):
        """Bind a qualified zero entitlement; never allocate artificial scratch."""
        proof = self.profiles[claim.token]
        first=getattr(self.lease,"first_qualification",None)
        if first is not None:
            first.require_task(self.lease,claim.task)
            if proof.get("first_administration") is not True or profile!=first.binding:
                raise CachePolicyError("First claim is not the exact released administrative binding")
        if (ledger is not self.lease.ledger or maximum != 0
                or claim.token != claim.task["attempt_id"]
                or proof["task"] != claim.task or proof["binding"] != profile
                or claim.token in self.claims
                or not self._is_current_claim_owner(owner)):
            raise CachePolicyError("Actual claim/cache/service/owner binding differs")
        ledger.require_reservation(reservation, owner, 0)
        ledger.require_unbound_scope(claim.token)
        receipt = self.lease._record(self.lease.reservation["token"] + ".claim-" + claim.token + ".json",
            dict(format=("darsia-worker-cache-first-claim-service-binding-v1" if first is not None
                         else "darsia-worker-cache-claim-service-binding-v1"), claim_token=claim.token,
                 task=claim.task, owner=owner, profile=profile,
                 writer_coverage_sha256=proof["coverage_sha256"], claim_reservation=reservation,
                 claim_disposable_entitlement_bytes=0, claim_scope_created=False,
                 service_reservation=self.lease.reservation, service_reservation_closed=False,
                 process_retirement_claimed=False, IO_closed_claimed=False))
        self.claims[claim.token] = dict(reservation=dict(reservation), owner=dict(owner),
            task=deepcopy(claim.task), proof=proof, receipt=receipt, closed=None, budget_closed=False)

    def close_claim(self, claim):
        """Close only a zero claim entitlement; shared writers remain active."""
        row = self.claims.get(claim.token)
        if row is None:
            raise CachePolicyError("Unbound claim cache lifecycle; preserve admission evidence")
        if row["task"] != claim.task:
            raise CachePolicyError("Claim changed before its budget close")
        if row["closed"] is not None:
            return row["closed"]
        if row["budget_closed"]:
            raise CachePolicyError("Zero entitlement closed but receipt incomplete; preserve evidence, no retry")
        if not self._is_current_claim_owner(row["owner"]):
            raise CachePolicyError("Only the actual live claim owner may close its zero entitlement")
        ledger = self.lease.ledger
        with ledger.locked():
            ledger.require_unbound_scope(claim.token)
            def no_claim_entitlement(value):
                return (value["token"] == claim.token and value["maximum_bytes"] == 0
                        and value["owner"] == row["owner"] and row["proof"] is self.profiles[claim.token])
            ledger.close(row["reservation"], owner=row["owner"],
                         verify_no_further_writes=no_claim_entitlement)
            row["budget_closed"] = True
            row["closed"] = self.lease._record(self.lease.reservation["token"] + ".claim-" + claim.token + ".budget-closed.json",
                dict(claim_binding=row["receipt"], claim_reservation=row["reservation"],
                     zero_claim_entitlement_closed=True, no_claim_scope_was_allocated=True,
                     service_reservation=self.lease.reservation, service_reservation_closed=False,
                     service_writers_may_still_be_active=True,
                     process_retirement_claimed=False, IO_closed_claimed=False))
        return row["closed"]
