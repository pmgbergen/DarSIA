"""Explicit whole-operation Setup ceilings on the actual affected volumes.

The existing capacity ledgers own host, account, locking, reservations and
retirement. This positive adapter does not qualify a measured cache profile or
start a writer. RAW is still written only by its original OwnedRawOutput. The
separate metadata/scratch allowance must be enforced by the caller's bounded
sink, which receives this typed admission after its real claim is bound.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path, PureWindowsPath
import re

from .series_setup_capacity import CappedSetupOutput, capped_setup_output


RESERVATION_FORMAT = "darsia-series-setup-volume-reservation-v1"
RESERVATION_MODEL = "explicit-setup-whole-volume-output-caps"


def _reservation_version(info):
    """Reuse an admitted digest only while its original file version is unchanged."""
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _reservation_pin(path, *, with_version=False):
    """Capture one bounded, version-stable original reservation record."""
    from .job_inputs import _json_bytes
    before = path.stat()
    if before.st_size > 1024**2:
        raise ValueError("Setup reservation exceeds its metadata bound")
    with path.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    after = path.stat()
    if _reservation_version(before) != _reservation_version(after) or len(raw) != before.st_size:
        raise ValueError("Setup reservation changed during its bounded read")
    _json_bytes(raw)
    pin = len(raw), hashlib.sha256(raw).hexdigest()
    return (pin, _reservation_version(after)) if with_version else pin


def _integer(value, label):
    """Keep byte ceilings and observations literal, without coercion."""
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a literal nonnegative integer")
    return value


@dataclass(frozen=True)
class SetupVolumeReservation:
    """One whole cap; committed-frame progress must never reduce it.

    A metadata allowance covering several volumes is reserved in full on each
    distinct volume. This conservative reservation does not promise that the
    allowance may be spent once per volume: the one bounded sink owns its
    cumulative global ceiling.
    """
    run: str
    operation: str
    source_commit: str
    maximum_output_bytes: int
    raw_maximum_output_bytes: int
    metadata_maximum_output_bytes: int
    volume_id: str
    capacity_key: str
    roots: tuple[dict, ...]
    setup_plan: dict
    raw_build_request: dict | None
    case_count: int = 1
    transient_estimated_bytes: int = 0
    frame_progress_release_allowed: bool = False
    format: str = RESERVATION_FORMAT
    model: str = RESERVATION_MODEL

    @property
    def products_per_frame_estimated_bytes(self):
        """Compatibility slot; this is a declared whole cap, not an estimate."""
        return self.maximum_output_bytes

    @property
    def products_estimated_bytes(self):
        """Retain the complete cap until the existing release boundary."""
        return self.maximum_output_bytes

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=None):
        """Include every other still-open reservation on this actual volume."""
        if reserve_bytes is None:
            from .capacity import DEFAULT_RESERVE_BYTES
            reserve_bytes = DEFAULT_RESERVE_BYTES
        return (self.maximum_output_bytes
                + _integer(other_remaining_bytes, "Other open reservations")
                + _integer(reserve_bytes, "The existing free-space reserve"))


def _descriptor(value, label):
    """Validate a retained literal descriptor without reopening its source."""
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value["path"], str) or not value["path"]
            or not (Path(value["path"]).is_absolute() or PureWindowsPath(value["path"]).is_absolute())
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError(label + " differs from an exact retained path/SHA/size descriptor")
    _integer(value["size_bytes"], label + " byte count")


def validate_retained_reservation(row, *, capacity_key, volume_id):
    """Check the frozen whole-cap record without requiring live old inputs.

    Current request/source revalidation belongs to new admission. Historical
    remaining-space accounting must retain a genuine old maximum even if its
    original request, plan, or project is no longer available.
    """
    value = row.get("estimate")
    if (row.get("format") != "darsia-local-capacity-reservation-v1"
            or row.get("reservation_model") != "setup-explicit-whole-operation"
            or row.get("not_a_measured_profile") is not True
            or not isinstance(value, dict)
            or set(value) != {field.name for field in fields(SetupVolumeReservation)}):
        raise ValueError("The retained Setup whole-operation record differs")
    maximum = _integer(value["maximum_output_bytes"], "The retained whole output cap")
    raw = _integer(value["raw_maximum_output_bytes"], "The retained RAW cap")
    metadata = _integer(value["metadata_maximum_output_bytes"], "The retained metadata cap")
    row_maximum = _integer(row.get("maximum_output_bytes"), "The retained row output cap")
    if maximum <= 0 or maximum != raw + metadata or row_maximum != maximum:
        raise ValueError("The retained Setup output ceilings disagree")
    if (_integer(row.get("products_per_frame_estimated_bytes"), "The retained compatibility cap") != maximum
            or _integer(row.get("transient_estimated_bytes"), "The retained compatibility transient") != 0):
        raise ValueError("The Setup whole cap differs from its original ledger compatibility fields")
    operation = value["operation"]
    if (not isinstance(value["run"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value["run"]) is None
            or not isinstance(operation, str) or operation not in {"apply", "build", "image-preparation"}
            or row.get("run") != value["run"]
            or type(value["case_count"]) is not int or value["case_count"] != 1
            or type(row.get("case_count")) is not int or row.get("case_count") != 1
            or row.get("ordered_case_ids") != [operation]
            or value["transient_estimated_bytes"] != 0
            or type(value["transient_estimated_bytes"]) is not int
            or value["frame_progress_release_allowed"] is not False
            or value["format"] != RESERVATION_FORMAT or value["model"] != RESERVATION_MODEL):
        raise ValueError("The retained Setup operation is not one whole unchanged cap")
    if (not isinstance(value["source_commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]) is None
            or value["capacity_key"] != capacity_key or value["volume_id"] != volume_id
            or row.get("volume_id") != volume_id):
        raise ValueError("The retained Setup source or actual capacity scope differs")
    _descriptor(value["setup_plan"], "The retained Setup plan")
    if operation == "build":
        _descriptor(value["raw_build_request"], "The retained full RAW request")
    elif value["raw_build_request"] is not None or raw:
        raise ValueError("Apply cannot retain a future RAW writer allowance")
    roots = value["roots"]
    if not isinstance(roots, (list, tuple)) or not roots:
        raise ValueError("The retained Setup output domains are missing")
    identities, roles = set(), set()
    for root in roots:
        if (not isinstance(root, dict) or set(root) != {"role", "path"}
                or not isinstance(root["role"], str) or root["role"] not in {"raw", "metadata", "scratch"}
                or not isinstance(root["path"], str) or not root["path"]
                or not (Path(root["path"]).is_absolute() or PureWindowsPath(root["path"]).is_absolute())):
            raise ValueError("The retained Setup output domain differs")
        identity = (root["role"], root["path"])
        if identity in identities:
            raise ValueError("A retained Setup output domain is duplicated")
        identities.add(identity)
        roles.add(root["role"])
    if (("raw" in roles) != bool(raw)
            or bool(roles & {"metadata", "scratch"}) != bool(metadata)):
        raise ValueError("The retained Setup roots and their output ceilings disagree")
    return value


@dataclass(frozen=True)
class _Domain:
    role: str
    path: Path
    ledger: object
    volume_id: str
    identity: tuple[int, int]


class SetupBindError(RuntimeError):
    """Retain exact successful prefixes of a failed multi-volume bind.

    No release is inferred. The original owner or proven supervisor retirement
    must close every retained reservation through the existing ledger API.
    """
    def __init__(self, cause, bound):
        self.cause = cause
        self.bound_reservations = tuple(bound)
        super().__init__("Setup capacity binding failed; retained reservations: "
                         + repr(self.bound_reservations) + "; " + str(cause))


class SetupAdmission:
    """A fresh typed plan, locked admission and original claim reservations.

    Only :func:`prepare_setup_admission` constructs this object. Leaving
    ``locked`` releases locks, never capacity reservations. Call ``release``
    only after the original executor's IO scope is closed, or with the original
    supervisor's actual retirement proof.
    """
    def __init__(self, task, config, estimate, domains, scopes, inspectors):
        self._task, self._configuration = deepcopy(task), deepcopy(config)
        self._estimate = estimate
        self._domains, self._scopes = tuple(domains), tuple(scopes)
        self._inspect_local, self._inspect_results, self._valid_results = inspectors
        self._locks = False
        self._checked = False
        self._bound = []
        self._bound_pins = []
        self._bound_versions = []
        self._owner = None
        self._closed = False

    @property
    def task(self):
        """The exact original task captured for this operation's reservation."""
        return deepcopy(self._task)

    @property
    def metadata_limits(self):
        """The one global metadata/scratch sink's exact whole/per-file caps."""
        return deepcopy(self._estimate.metadata_output_authority)

    @property
    def raw_authority(self):
        """The unchanged full request's original RAW output authority."""
        return deepcopy(self._estimate.raw_output_authority)

    @property
    def configuration(self):
        """The immutable operation document which supplied this admission."""
        return deepcopy(self._configuration)

    @property
    def volume_reservations(self):
        """Read-only declared per-volume ceilings, without publication claims."""
        return deepcopy(tuple(estimate for _ledger, estimate in self._scopes))

    @property
    def bound_reservations(self):
        """Actual retained ledger paths; a partial prefix is never hidden."""
        return tuple(dict(capacity_key=ledger.key, path=str(path))
                     for ledger, path in self._bound)

    def _revalidate_domains(self, *, selected=None):
        """Check all admission roots, or a write plus its reserved volumes.

        A write cannot affect unrelated output directories. Their original
        identities remain captured and are checked when a write targets them;
        prospective check/bind still revalidates the complete operation scope.
        Every distinct reserved volume and local/shared admission class is
        checked even when this particular write targets another volume.
        """
        from ..rig_raw_write import _no_links
        reuse_locked_admission = self._locks and self._checked and selected is not None
        domains = self._domains
        if selected is not None:
            domains = tuple(domain for domain in domains
                if domain.role in {"metadata", "scratch"} and selected.is_relative_to(domain.path))
            if not domains:
                raise ValueError("The metadata/scratch path is outside the explicit Setup output roots")
            def scope(domain):
                return (domain.ledger.key, domain.volume_id,
                    domain.role == "scratch" and not getattr(domain.ledger, "folder_mode", False))
            covered = {scope(domain) for domain in domains}
            representatives = []
            for domain in self._domains:
                if scope(domain) not in covered:
                    covered.add(scope(domain))
                    representatives.append(domain)
            domains += tuple(representatives)
        for domain in domains:
            _no_links(domain.path)
            if domain.path.resolve() != domain.path:
                raise ValueError("A Setup output root changed its resolved location")
            info = domain.path.stat()
            if not domain.path.is_dir() or (info.st_dev, info.st_ino) != domain.identity:
                raise ValueError("A Setup output root changed its actual directory identity")
            if reuse_locked_admission:
                # The same lock still protects this checked claim transaction.
                # Keep path/link/inode checks for every write; refresh transport,
                # free space and reservations again at the final bind below.
                continue
            if domain.role == "scratch" and not getattr(domain.ledger, "folder_mode", False):
                proof = self._inspect_local(domain.path / ".darsia-location-probe")
                if proof.allowed is not True:
                    raise ValueError("Setup scratch roots must remain positively local")
            else:
                proof = self._inspect_results(domain.path)
                if not self._valid_results(proof, domain.path):
                    raise ValueError("A Setup output root lost positive filesystem admission")
            # A ledger may represent several same-volume domains. Query the
            # actual domain, not merely the first ledger's results_root.
            observed = domain.ledger.fresh_volume()
            if observed["volume_id"] != domain.volume_id:
                raise ValueError("A Setup output root changed its actual volume")

    @contextmanager
    def locked(self):
        """Use the original ledgers' fixed stable lock order, without a writer."""
        if self._locks:
            raise RuntimeError("Setup admission locks are already held")
        with ExitStack() as stack:
            for ledger, _estimate in self._scopes:
                stack.enter_context(ledger.locked())
            self._locks = True
            try:
                yield self
            finally:
                self._locks = False
                self._checked = False

    def check(self):
        """Fresh free space includes ALL retained open claims before bind."""
        if not self._locks or self._bound or self._closed:
            raise RuntimeError("Fresh Setup admission needs locked, unbound ledgers")
        self._checked = False
        fresh = capped_setup_output(self._task, self._configuration)
        if fresh != self._estimate:
            raise ValueError("The Setup request or output ceilings changed after planning")
        if self.raw_authority is not None:
            from ..rig_raw_write import _canonical
            if _canonical(self.raw_authority["destination"]).exists():
                raise ValueError("The original RAW output destination already exists; retained work cannot be relaunched")
        self._revalidate_domains()
        reports = []
        for ledger, estimate in self._scopes:
            volume = ledger.fresh_volume()
            if volume["volume_id"] != estimate.volume_id:
                raise ValueError("The Setup capacity scope changed its actual volume")
            other = _integer(ledger.remaining(), "Other open reservations")
            needed = estimate.needed(other_remaining_bytes=other, reserve_bytes=ledger.reserve)
            free = _integer(volume["free_bytes"], "Observed free bytes")
            reports.append(dict(capacity_key=ledger.key, volume_id=estimate.volume_id,
                                maximum_output_bytes=estimate.maximum_output_bytes,
                                other_open_bytes=other, reserve_bytes=ledger.reserve,
                                needed_bytes=needed, free_bytes=free,
                                sufficient=free >= needed, not_a_measured_profile=True))
        if not all(row["sufficient"] for row in reports):
            raise ValueError("Setup output ceilings plus existing open claims and reserve exceed free space: "
                             + repr(reports))
        self._checked = True
        return tuple(reports)

    def bind(self, job_root, claim, owner):
        """Reserve only the real unchanged claim; retain any failed prefix."""
        if not self._locks or not self._checked or self._bound or self._closed:
            raise RuntimeError("Setup bind requires its fresh locked admission")
        if (claim.task != self._task or claim.token != self._task.get("attempt_id")
                or self._task.get("ordered_case_ids") != [self._configuration["operation"]]):
            raise ValueError("Setup reservation differs from the original whole-operation claim")
        # Close the gap between the prospective check and the actual claim.
        self.check()
        self._owner = deepcopy(owner)
        try:
            for ledger, estimate in self._scopes:
                path = ledger.bind(job_root, claim, owner, estimate)
                self._bound.append((ledger, Path(path)))
                pin, version = _reservation_pin(Path(path) / "reserved.json", with_version=True)
                self._bound_pins.append(pin)
                self._bound_versions.append(version)
        except Exception as error:
            self._closed = True
            raise SetupBindError(error, self.bound_reservations) from error
        return self.bound_reservations

    def require_write(self, path):
        """Admit one metadata/scratch path only while ALL reservations are live.

        Call this before directory creation and before each stream write. The
        sink also owns cumulative byte accounting and post-write path checks.
        RAW-only roots cannot be used to bypass the original RAW writer.
        """
        if self._closed or len(self._bound) != len(self._scopes) or self._owner is None:
            raise RuntimeError("A complete active Setup reservation is required before writing")
        from ..rig_raw_write import _no_links
        _no_links(Path(path).expanduser())
        selected = Path(path).expanduser().resolve()
        candidates = [domain for domain in self._domains
                      if domain.role in {"metadata", "scratch"}
                      and selected.is_relative_to(domain.path)]
        if not candidates:
            raise ValueError("The metadata/scratch path is outside the explicit Setup output roots")
        self._revalidate_domains(selected=selected)
        # Existing release records are immutable. Do not continue writing after
        # an external original-owner/supervisor release of any affected scope.
        for (_ledger, reservation), version in zip(self._bound, self._bound_versions):
            if (reservation / "released.json").exists():
                raise RuntimeError("An affected Setup capacity reservation has already been released")
            if _reservation_version((reservation / "reserved.json").stat()) != version:
                raise ValueError("An active Setup capacity reservation changed before read")
        return selected

    def release(self, actor, *, retirement=None):
        """Delegate actual closure, preserving partial release evidence."""
        if not self._bound:
            raise RuntimeError("No actual Setup reservation exists to release")
        self._closed = True
        with self.locked():
            for ledger, path in self._bound:
                ledger.release(path, actor, retirement=retirement)


def metadata_roots_for_targets(config, targets):
    """Select only declared roots for the operation's actual metadata outputs."""
    if (config["operation"] not in {"build", "image-preparation"}
            or not isinstance(targets, tuple) or not targets):
        raise ValueError("Only Build/image preparation may restrict metadata admission to actual outputs")
    declared, selected = tuple(map(Path, config["output_scope"]["metadata_roots"])), []
    for target in targets:
        target = Path(target)
        matches = [root for root in declared if target == root or target.is_relative_to(root)]
        if not matches:
            raise ValueError("A Setup metadata target is outside its declared output roots")
        root = max(matches, key=lambda path: len(path.parts))
        if root not in selected:
            selected.append(root)
    return tuple(selected)


def prepare_setup_admission(stores, series_id, task, config, *, metadata_roots,
                            scratch_roots=(), ledger_factory=None,
                            inspect_local=None, inspect_results=None, valid_results=None,
                            metadata_targets=None):
    """Capture explicit roots and original ceilings without any claim or write.

    The injected providers support private deterministic tests of accounting;
    production defaults are the original OS evidence and CapacityLedger. Root
    spellings never imply locality or a shared physical volume.
    """
    from ..job_filesystem import inspect_local_file, inspect_results_root, valid_results_admission
    from .capacity import CapacityLedger
    from .series_setup_protocol import validate_setup_operation
    from ..rig_raw_write import _canonical, _no_links
    validate_setup_operation(config, task=task)
    estimate = capped_setup_output(task, config)
    if not isinstance(estimate, CappedSetupOutput):
        raise TypeError("The original explicit Setup output ceilings are required")
    if not isinstance(metadata_roots, (list, tuple)) or not metadata_roots:
        raise ValueError("Setup metadata needs explicit affected output roots")
    if not isinstance(scratch_roots, (list, tuple)):
        raise ValueError("Setup scratch roots must be explicit")
    scope = config["output_scope"]
    if (list(map(str, metadata_roots)) != scope["metadata_roots"]
            or list(map(str, scratch_roots)) != scope["scratch_roots"]):
        raise ValueError("Setup output roots differ from the exact operation resource scope")
    if metadata_targets is not None:
        # Build/image preparation write their own attempt plus job/lease publications.
        # Old immutable plans may list every other run's Apply/config roots;
        # those give this Build no useful writer authority or reservation.
        # Narrow to existing DECLARED roots; never invent/broaden a root, and
        # retain all usual checks on every root that can actually be written.
        metadata_roots = metadata_roots_for_targets(config, metadata_targets)
    factory = CapacityLedger if ledger_factory is None else ledger_factory
    local = inspect_local_file if inspect_local is None else inspect_local
    results = inspect_results_root if inspect_results is None else inspect_results
    valid = valid_results_admission if valid_results is None else valid_results
    authored = [("metadata", root) for root in metadata_roots]
    authored.extend(("scratch", root) for root in scratch_roots)
    if estimate.raw_output_authority is not None:
        raw_authority = estimate.raw_output_authority
        root = _canonical(raw_authority["intended_root"])
        target = _canonical(raw_authority["destination"])
        protected = raw_authority["forbidden_raw_roots"]
        if not isinstance(protected, list) or not protected:
            raise ValueError("The original RAW authority needs its explicit protected acquisition roots")
        protected = tuple(_canonical(path) for path in protected)
        if (target == root or not target.is_relative_to(root)
                or any(target == source or target.is_relative_to(source) or source.is_relative_to(target)
                       for source in protected)):
            raise ValueError("The original RAW output is not a fresh child outside its protected acquisition roots")
        if target.exists():
            raise ValueError("The original RAW output destination already exists; retained work cannot be relaunched")
        for _role, candidate in authored:
            selected = _canonical(candidate)
            if selected == target or selected.is_relative_to(target) or target.is_relative_to(selected):
                raise ValueError("Setup metadata/scratch roots overlap the original RAW output tree")
        authored.append(("raw", estimate.raw_output_authority["intended_root"]))
    domains, pools, physical = [], {}, {}
    for role, root in authored:
        path = Path(root).expanduser()
        _no_links(path)
        if not path.is_absolute() or path.resolve() != path or not path.is_dir():
            raise ValueError("Each Setup output root needs its exact absolute physical location")
        from darsia_cache.folders import folder_mode, require_durable_path
        ordinary = folder_mode(stores, series_id)
        if ordinary:
            require_durable_path(stores, series_id, path)
        if role == "scratch" and not ordinary:
            proof = local(path / ".darsia-location-probe")
            if proof.allowed is not True:
                raise ValueError("Setup scratch roots must be positively local")
        else:
            proof = results(path)
            if not valid(proof, path):
                raise ValueError("Setup output roots require positive output filesystem admission")
        ledger = factory(stores, series_id, path)
        if Path(ledger.results_root).resolve() != path:
            raise ValueError("The actual capacity ledger belongs to another output root")
        volume_id = ledger.volume["volume_id"]
        if not isinstance(volume_id, str) or not volume_id:
            raise ValueError("The actual Setup output volume needs a stable identity")
        if volume_id in physical and physical[volume_id] != ledger.key:
            raise ValueError("One physical volume has conflicting original capacity scopes")
        physical[volume_id] = ledger.key
        pool = pools.setdefault(ledger.key, dict(ledger=ledger, volume_id=volume_id,
                                                roles=set(), roots=[]))
        if pool["volume_id"] != volume_id:
            raise ValueError("One capacity scope claims several actual volumes")
        pool["roles"].add(role)
        root_row = dict(role=role, path=str(path))
        if root_row not in pool["roots"]:
            pool["roots"].append(root_row)
            info = path.stat()
            domains.append(_Domain(role, path, ledger, volume_id, (info.st_dev, info.st_ino)))
    scopes = []
    for key in sorted(pools):
        pool = pools[key]
        raw = estimate.raw_maximum_output_bytes if "raw" in pool["roles"] else 0
        metadata = (estimate.metadata_maximum_output_bytes
                    if pool["roles"] & {"metadata", "scratch"} else 0)
        reservation = SetupVolumeReservation(
            run=estimate.run, operation=estimate.operation, source_commit=task["commit"],
            maximum_output_bytes=raw + metadata,
            raw_maximum_output_bytes=raw, metadata_maximum_output_bytes=metadata,
            volume_id=pool["volume_id"], capacity_key=key,
            roots=tuple(sorted(pool["roots"], key=lambda row: (row["role"], row["path"]))),
            setup_plan=deepcopy(config["setup_plan"]),
            raw_build_request=deepcopy(estimate.raw_build_request))
        scopes.append((pool["ledger"], reservation))
    return SetupAdmission(task, config, estimate, domains, scopes, (local, results, valid))
