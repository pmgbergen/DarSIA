"""Q123 Main bound qualification; retained files are used bytes, never zero.

This reuses the original append-only close verifier and FILE_CREATE service
provenance. It cannot rearm a consumed attempt or choose another cache root.
"""
from copy import deepcopy
import hashlib
from pathlib import Path, PureWindowsPath
import re

from .policy import CachePolicyError, _descriptor


def require_prior_reservations_closed(names):
    """Refuse unless every cache-r reservation in these ledger names has its cache-c.

    A fully written live reservation has zero remaining bytes. It still forbids
    an exclusive start, so each one must be genuinely closed.
    """
    for name in names:
        if name.startswith("cache-r-"):
            if re.fullmatch(r"cache-r-[0-9a-f]{32}\.json", name) is None:
                raise CachePolicyError("Unknown prior reservation prevents exclusivity")
            token = name[8:-5]
            if "cache-c-" + token + ".json" not in names:
                raise CachePolicyError("Every prior reservation must be genuinely closed")


class RetainedColdScope:
    def __init__(self, first, binding, *, read, read_authorities, historical_authorities=None):
        from .first_qualification import _digest, _exact_keys
        from .service_context import RetainedServiceCloseContext
        self.binding = _descriptor(binding)
        self.value = deepcopy(read(self.binding))
        _exact_keys(self.value, {"format", "authority_ids", "answers_source",
            "original_close_result", "retained_cache_inventory", "matplotlib_namespace"},
            "Exact Q123 bound cold-scope admission required")
        if (not first.bound_mode or first.reference_build
                or self.value["format"] != "darsia-q123-bound-cold-scope-v1"
                or self.value["authority_ids"] != ["Q123"] or not callable(read_authorities)):
            raise CachePolicyError("Q123 retained admission belongs only to Main BOUND_REQUEST")
        for key in ("answers_source", "original_close_result", "retained_cache_inventory"):
            _descriptor(self.value[key])
        answers = read_authorities(self.value["answers_source"])
        if not any(row.get("id") == "Q123-answer-option1-append-only-closeout-and-new-empty-exclusive-scope-retained-bytes-counted"
                and row.get("decided_by") == "coordinator" for row in answers):
            raise CachePolicyError("Actual Q123 authority required")
        self.read = read
        result = read(self.value["original_close_result"])
        _exact_keys(result, {"format", "request", "reservation", "close_record",
            "original_service_receipt", "original_lifecycle_eligibility", "original_native_retirement",
            "retained_cache_inventory", "all_files_retained", "retained_bytes_are_still_used",
            "cleanup_authorized", "rearm_authorized"}, "Actual original retained close result required")
        self.closed = RetainedServiceCloseContext(result["request"], package_file=__file__,
            retained_close=_descriptor(result["close_record"]),
            historical_authorities=historical_authorities)
        original = self.closed.original
        if (result["format"] != "darsia-cache-retained-service-close-result-v1"
                or result["reservation"] != original.reservation
                or any(result[key] != self.closed.request[key] for key in (
                    "original_service_receipt", "original_lifecycle_eligibility", "original_native_retirement"))
                or result["retained_cache_inventory"] != self.value["retained_cache_inventory"]
                or result["retained_cache_inventory"] != self.closed.request["retained_cache_inventory"]
                or result["all_files_retained"] is not True or result["retained_bytes_are_still_used"] is not True
                or result["cleanup_authorized"] is not False or result["rearm_authorized"] is not False):
            raise CachePolicyError("Original close chain/retained accounting differs")
        self.close_record = _descriptor(result["close_record"])
        expected = Path(original.ledger.bound["path"]) / ("cache-c-" + original.reservation["token"] + ".json")
        if PureWindowsPath(self.close_record["path"]) != PureWindowsPath(str(expected)):
            raise CachePolicyError("Close must remain in the original canonical host ledger")
        self.matplotlib_namespace = _exact_keys(self.value["matplotlib_namespace"], {"path", "identity"},
            "Exact job-owned external Matplotlib namespace required")
        self._seal = _digest(self.value)

    def admit(self, first, lease, report):
        """Called under the existing host lock, before any new reservation."""
        from .first_qualification import _digest
        from .service_context import verify_retained_cache_inventory
        from .control import _json
        if _digest(self.value) != self._seal:
            raise CachePolicyError("Q123 admission changed in memory")
        old = self.closed.original
        cap = getattr(lease.policy, "volume_cap", None)
        oldcap = getattr(old.policy, "volume_cap", None)
        if (lease.context["host"] != "reference-host" or cap is None or oldcap is None
                or cap.document() != oldcap.document() or cap.maximum_size_bytes != 500 * 1024**3
                or PureWindowsPath(lease.policy.root) != PureWindowsPath(old.policy.root)
                or lease.admission["cache_identity"] != old.ledger.admission["cache_identity"]
                or lease.admission["host_ledger"] != old.ledger.admission["host_ledger"]):
            raise CachePolicyError("Q123 retains the same Main canonical root, identity and Q87 cap")
        require_prior_reservations_closed(lease.ledger.control.names())
        # Original project/remaining has already validated each close's owner,
        # token and raw reservation SHA, including fully allocated reservations.
        raw, _ = lease.ledger.control.read_once(Path(self.close_record["path"]).name)
        if len(raw) != self.close_record["size_bytes"] or hashlib.sha256(raw).hexdigest() != self.close_record["sha256"]:
            raise CachePolicyError("Original retained close bytes changed")
        closed, _ = _json(lease.ledger.control, Path(self.close_record["path"]).name)
        if (closed.get("owner") != old.manifest["owner"] or closed.get("token") != old.reservation["token"]
                or closed.get("reservation_sha256") != old.reservation["sha256"]
                or closed.get("no_further_writes") is not True):
            raise CachePolicyError("Actual close does not bind the original owner/token/reservation SHA")
        # Recheck original kernel/registered owner retirement, with exact epochs.
        self.closed.retired_after_close_locked(lease.ledger, self.close_record)
        inventory = verify_retained_cache_inventory(self.read(self.value["retained_cache_inventory"]), lease.policy.root)
        if (report["current_cache_logical_bytes"] != inventory["logical_bytes"]
                or report["active_cache_remaining_bytes"] != 0
                or report.get("hard_filesystem_quota_verified") is not True):
            raise CachePolicyError("Whole retained cache bytes stay used; active reservations are forbidden")
        self.matplotlib_root(first, lease, require_empty=True)
        return inventory

    def matplotlib_root(self, first, lease=None, *, require_empty=False):
        from .windows import WindowsNamespace
        value = self.matplotlib_namespace
        path = PureWindowsPath(value["path"])
        output = PureWindowsPath(first.output_entitlement["watched_output_scope"]["path"])
        cache = PureWindowsPath(lease.policy.root if lease is not None else self.closed.original.policy.root)
        if (not path.is_absolute() or ".." in path.parts or path == output or output not in path.parents
                or path == cache or cache in path.parents or path in cache.parents):
            raise CachePolicyError("External Matplotlib must be a counted diagnostic child of this job's watched output")
        namespace = WindowsNamespace(value["path"], value["identity"])
        try:
            if require_empty and namespace.names():
                raise CachePolicyError("New external Matplotlib namespace must be empty at admission")
        finally:
            namespace.close()
        return value["path"]

    def require_service(self, first, lease):
        """Existing FILE_CREATE proof is available only after ImportLease.start."""
        from .environment import FIRST_WRITER_SUBDIRECTORIES, FIRST_WRITER_ENVIRONMENT_KEYS
        from .control import _time
        manifest = lease.manifest
        fresh = manifest.get("fresh_scope_provenance", {})
        token = lease.reservation["token"]
        if (token == self.closed.original.reservation["token"]
                or manifest.get("reservation") != lease.reservation or not isinstance(manifest.get("owner"), dict)
                or manifest.get("token") != token or manifest.get("purpose") != "jobs"
                or PureWindowsPath(manifest.get("root", "")) != PureWindowsPath(str(lease.policy.purpose("jobs") / token))
                or manifest.get("root_identity") != lease.directories[""]
                or manifest.get("root_identity") == self.closed.original.manifest["root_identity"]
                or fresh.get("format") != "darsia-fresh-exclusive-empty-service-scope-v1"
                or fresh.get("token") != token or fresh.get("owner") != manifest["owner"]
                or fresh.get("purpose") != "jobs" or fresh.get("root_identity") != manifest["root_identity"]
                or fresh.get("creation_disposition") != "NtCreateFile.FILE_CREATE"
                or fresh.get("exclusive_root_handle") is not True or fresh.get("initial_names") != []
                or fresh.get("observation_precedes_writer_subdirectories_and_imports") is not True
                or fresh.get("writer_subdirectories") != list(FIRST_WRITER_SUBDIRECTORIES)
                or fresh.get("writer_environment_keys") != list(FIRST_WRITER_ENVIRONMENT_KEYS)):
            raise CachePolicyError("Q123 requires the original genuinely new empty exclusive service proof")
        _time(fresh["initial_empty_observed_utc"])
        self.matplotlib_root(first, lease)
        return True
