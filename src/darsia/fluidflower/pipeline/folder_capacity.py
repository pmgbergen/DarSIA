"""Fresh output-space checks with durable, shared, per-job reservations.

Folder caches hold no coordination state. A byte-range file lock in the job
serializes admission on SMB as well as on a local filesystem. The OS releases
the lock when an owner exits; retained reservations require explicit original
owner or supervisor retirement, just like the existing queue.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .job_queue import read_json, write_immutable
from .store_runtime import RunUnavailable


class CapacityLockBusy(RunUnavailable):
    """A live kernel lock is held; retain reservations and retry later."""


def _lock_identity(stream, path):
    """An opened inode must still be the named lock, never its replacement."""
    opened, named = os.fstat(stream.fileno()), path.stat()
    identity = opened.st_dev, opened.st_ino
    if path.is_symlink() or identity != (named.st_dev, named.st_ino):
        raise ValueError("Shared job capacity lock file was replaced")
    return identity


@dataclass(frozen=True)
class FolderSpaceEstimate:
    run: str
    case_count: int
    products_per_frame_estimated_bytes: int
    transient_estimated_bytes: int
    model: str = "Estimate from declared input bytes and existing output-size model; no measured cache qualification"

    @property
    def products_estimated_bytes(self):
        return self.products_per_frame_estimated_bytes * self.case_count

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=0):
        return self.products_estimated_bytes + self.transient_estimated_bytes + other_remaining_bytes + reserve_bytes


def folder_estimate(task, config):
    """Use existing size DATA, with an explicit metadata-only estimate otherwise.

    This is ordinary best-effort disk headroom, never a producer qualification
    or a hard output quota. Setup retains its actual enforced writer ceilings.
    """
    count = len(task["ordered_case_ids"])
    if count < 1:
        raise ValueError("Output estimate needs the actual nonempty task")
    inputs = sum(row["reference"]["bytes"] for row in task["input_catalog"]["inputs"])
    per_frame = max(16 * 1024**2, (2 * inputs + count - 1) // count)
    transient = max(16 * 1024**2, inputs)
    if task.get("pipeline") in {"n1", "native-phase-n1", "native-phase-statistics"}:
        from ..storage_profile import estimate_job_bytes, declared_storage_profile, declared_canvas_pixels, AC52_CANVAS_HW
        pipeline = "n1" if task["pipeline"] == "n1" else "native-phase-n1"
        base = estimate_job_bytes(declared_storage_profile(task), count, pipeline=pipeline)
        pixels = declared_canvas_pixels(config, pipeline=pipeline)
        scale = max(1.0, (pixels or 0) / (AC52_CANVAS_HW[0] * AC52_CANVAS_HW[1]))
        per_frame = max(per_frame, int(base["per_frame_bytes"] * scale))
        transient = max(transient, int((base["inflight_bytes"] + base["per_run_bytes"]) * scale))
    return FolderSpaceEstimate(task["run"], count, per_frame, transient)


class FolderCapacityLedger:
    def __init__(self, stores, series_id, results_root, *, job_root=None, volume_provider=None, **_kwargs):
        from .capacity import DEFAULT_RESERVE_BYTES, volume_info
        from darsia_cache.folders import require_durable_path
        self.stores, self.series_id = stores, series_id
        self.machine_id, self.results_root = stores.machine_id, Path(results_root).resolve()
        require_durable_path(stores, series_id, self.results_root)
        self.job_root = Path(job_root).resolve() if job_root is not None else None
        if self.job_root is not None:
            require_durable_path(stores, series_id, self.job_root)
        self.volume_provider = volume_info if volume_provider is None else volume_provider
        self.volume = self.volume_provider(self.results_root)
        self.reserve = stores.document.get("capacity", {}).get("reserve_bytes", DEFAULT_RESERVE_BYTES)
        # No host-local binding or local-only volume requirement. Reservation
        # files are in this shared job and every worker can read them.
        self.remote_scope = None
        self.folder_mode = True
        self.key = hashlib.sha256(str(self.volume["volume_id"]).encode()).hexdigest()
        self.root = (self.job_root / ".capacity" / self.key) if self.job_root is not None else None

    @contextmanager
    def locked(self, *, create=True, wait_seconds=30.0):
        if self.root is None:
            raise RunUnavailable("Choose the shared Jobs/<job-id> folder before output-space admission")
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise RunUnavailable("This job has no retained capacity state yet")
        path = self.root / "admission.lock"
        try:
            stream = path.open("x+b" if create else "r+b")
        except FileExistsError:
            stream = path.open("r+b")
        with stream:
            if path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            deadline = time.monotonic() + wait_seconds
            while True:
                checkpoint = getattr(self, "checkpoint", None)
                if checkpoint is not None:
                    checkpoint()
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise CapacityLockBusy("Shared job output-space admission is busy; no claim made")
                    time.sleep(0.05)
            try:
                identity = _lock_identity(stream, path)
                import psutil
                owner = dict(format="darsia-folder-capacity-lock-v1", nonce=uuid.uuid4().hex,
                    machine=self.machine_id, pid=os.getpid(),
                    process_create_time=str(psutil.Process().create_time()))
                owner_bytes = json.dumps(owner, sort_keys=True).encode("utf-8")
                stream.seek(1)
                stream.write(owner_bytes)
                stream.truncate()
                stream.flush()
                yield self
                if _lock_identity(stream, path) != identity:
                    raise ValueError("Shared job capacity lock identity changed")
                stream.seek(1)
                if stream.read() != owner_bytes:
                    raise ValueError("Shared job capacity lock owner or nonce changed")
            finally:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def fresh_volume(self):
        current = self.volume_provider(self.results_root)
        if current["volume_id"] != self.volume["volume_id"]:
            raise RunUnavailable("Output volume changed during admission")
        return current

    def remaining(self, *, exclude_job=None):
        if self.root is None or not self.root.exists():
            return 0
        if exclude_job is not None and Path(exclude_job).resolve() == self.job_root:
            return 0
        total = 0
        for path in self.root.glob("r-*/reserved.json"):
            row = read_json(path)
            self._require_row(path.parent, row)
            release = path.parent / "released.json"
            if release.exists():
                ended = read_json(release)
                if (ended.get("reservation_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                        or ended.get("owner") != row["owner"] or ended.get("attempt_id") != row["attempt_id"]):
                    raise ValueError("Shared job capacity release differs from its reservation")
                continue
            total += row["maximum_output_bytes"] + row["transient_estimated_bytes"]
        return total

    def _require_row(self, path, row):
        from .capacity import _owner, _reservation_id, _integer
        if (self.root is None or Path(path).parent != self.root
                or Path(row["job_root"]).resolve() != self.job_root
                or row["volume_id"] != self.volume["volume_id"]
                or Path(path).name != "r-" + _reservation_id(self.job_root, row)):
            raise ValueError("Shared job capacity reservation belongs to another scope")
        _owner(row["owner"])
        _integer(row["maximum_output_bytes"], "Reserved output bytes")
        _integer(row["transient_estimated_bytes"], "Reserved transient bytes")
        if row["reservation_model"] == "setup-explicit-whole-operation":
            from .series_setup_admission import validate_retained_reservation
            validate_retained_reservation(row, capacity_key=self.key, volume_id=self.volume["volume_id"])
        elif row["reservation_model"] != "folder-whole-run":
            raise ValueError("Unknown shared job capacity reservation")

    def bind(self, job_root, claim, owner, estimate):
        from .capacity import _owner, _reservation_id
        from .series_setup_admission import SetupVolumeReservation
        _owner(owner)
        if (self.job_root != Path(job_root).resolve() or claim.token != claim.task["attempt_id"]
                or estimate.case_count != len(claim.task["ordered_case_ids"])):
            raise ValueError("Output reservation differs from its actual shared claim")
        setup = isinstance(estimate, SetupVolumeReservation)
        if setup and (estimate.capacity_key != self.key or estimate.volume_id != self.volume["volume_id"]):
            raise ValueError("Setup reservation differs from its output volume")
        maximum = estimate.maximum_output_bytes if setup else estimate.products_estimated_bytes
        row = dict(format="darsia-local-capacity-reservation-v1" if setup else "darsia-folder-capacity-reservation-v1",
            reservation_model="setup-explicit-whole-operation" if setup else "folder-whole-run",
            not_a_measured_profile=True, owner=dict(owner), job_root=str(self.job_root),
            machine_id=self.machine_id, volume_id=self.volume["volume_id"],
            run=claim.task["run"], attempt_id=claim.token, case_count=estimate.case_count,
            ordered_case_ids=list(claim.task["ordered_case_ids"]), maximum_output_bytes=maximum,
            products_per_frame_estimated_bytes=maximum if setup else estimate.products_per_frame_estimated_bytes,
            transient_estimated_bytes=estimate.transient_estimated_bytes, estimate=asdict(estimate))
        destination = self.root / ("r-" + _reservation_id(job_root, claim.task))
        destination.mkdir()
        self._require_row(destination, row)
        write_immutable(destination / "reserved.json", row)
        return destination

    def progress(self, reservation, owner, event):
        row = read_json(Path(reservation) / "reserved.json")
        self._require_row(reservation, row)
        if owner != row["owner"]:
            raise ValueError("Output capacity progress owner differs from actual claim")
        # Retain the whole estimate until actual IO closure. Frame checkpoints
        # and progress already live in the queue/attempt; never duplicate them.

    def release(self, reservation, actor, *, retirement=None):
        from .capacity import _owner
        _owner(actor)
        path = Path(reservation) / "reserved.json"
        row = read_json(path)
        self._require_row(reservation, row)
        if retirement is None:
            if actor != row["owner"]:
                raise ValueError("Only the actual original owner may close its reservation")
        elif (retirement.get("complete") is not True or retirement.get("all_absent") is not True
                or retirement.get("worker_os_references_closed") is not True
                or retirement.get("supervisor") != actor or retirement.get("run") != row["run"]
                or retirement.get("attempt_id") != row["attempt_id"]
                or not any(i.get("pid") == row["owner"]["pid"]
                    and i.get("process_create_time") == row["owner"]["process_create_time"]
                    for i in retirement.get("identities", []))):
            raise ValueError("Supervisor retirement does not bind this output reservation owner")
        value = dict(format="darsia-folder-capacity-release-v1", owner=row["owner"], actor=actor,
            attempt_id=row["attempt_id"], reservation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            retirement=retirement)
        destination = Path(reservation) / "released.json"
        if not destination.exists():
            write_immutable(destination, value)
        elif read_json(destination).get("reservation_sha256") != value["reservation_sha256"]:
            raise ValueError("Output capacity release changed reservation identity")

    def release_local_stop(self, reservation, actor, proof):
        """Separate operator-verified absence from worker retirement records."""
        from .capacity import _owner
        _owner(actor)
        path = Path(reservation) / "reserved.json"
        row = read_json(path)
        self._require_row(reservation, row)
        if (proof.get("format") != "darsia-job-local-process-stop-claim-v1"
                or proof.get("all_absent") is not True
                or proof.get("run") != row["run"] or proof.get("attempt_id") != row["attempt_id"]
                or proof.get("actual_worker") != row["owner"]
                or not proof.get("confirmation") or not proof.get("claim") or not proof.get("owner")
                or not any(item.get("pid") == row["owner"]["pid"]
                    and item.get("process_create_time") == row["owner"]["process_create_time"]
                    for item in proof.get("identities", []))):
            raise ValueError("Local process-stop evidence differs from this reservation owner")
        value = dict(format="darsia-folder-capacity-release-v1", owner=row["owner"], actor=actor,
            attempt_id=row["attempt_id"], reservation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            local_stop=proof, retirement=None)
        destination = Path(reservation) / "released.json"
        if not destination.exists():
            write_immutable(destination, value)
        elif read_json(destination).get("reservation_sha256") != value["reservation_sha256"]:
            raise ValueError("Output capacity release changed reservation identity")
