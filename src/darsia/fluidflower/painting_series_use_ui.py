"""Read existing series-map progress records for display, without admission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QLabel

from .painting_saved_transfers_ui import _safe

# Complete multi-run indexes embed their retained source/geometry bindings.
# Keep both reads finite while admitting those metadata-only records.
_LIMIT = 32 * 1024 * 1024
_TOTAL = 128 * 1024 * 1024
_PLACEMENT = {"darsia-fluidflower-map-placement-v1", "darsia-fluidflower-map-placement-v2"}
_PREPARATION = "darsia-fluidflower-painted-map-adoption-preparation-v1"
_PREPARATIONS = {_PREPARATION, "darsia-fluidflower-painted-map-adoption-preparation-v2"}


def matches_version(value, record):
    """Match the saved folder through explicit aliases and its original SHA."""
    version = value.get("map_version") if isinstance(value, dict) else None
    from .storage_paths import resolve_storage_path
    return bool(record and isinstance(version, dict) and record.get("folder") and version.get("folder")
        and resolve_storage_path(version["folder"]) == resolve_storage_path(record["folder"])
        and version.get("sha256") == record.get("sha256"))


def _read(path, budget):
    """Read bounded metadata consistently; no NPZ, photo or source model is read."""
    path = _safe(path)
    before = path.stat()
    with path.open("rb") as stream:
        raw = stream.read(_LIMIT + 1)
    budget[0] += len(raw)
    if budget[0] > _TOTAL:
        raise ValueError("Saved series-map records exceed the total bounded read")
    after = path.stat()
    if len(raw) > _LIMIT or any(getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("The saved progress record changed or exceeds its bounded read")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Repeated progress metadata key")
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError("Nonfinite progress metadata")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("The saved progress record is not an object")
    return value, len(raw), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def existing_progress(root, current_path, record):
    """List only this selected version's retained transfer/check/adoption records."""
    root, current_path = _safe(root), _safe(current_path)
    result = dict(transfer=None, preparation=None, adopted=None, gap_reviews=[], errors=[])
    budget = [0]
    for name, formats, key in (("map_placement", _PLACEMENT, "transfer"),
                              ("map_adoption", _PREPARATIONS, "preparation")):
        folder = _safe(root / name)
        if not folder.exists():
            continue
        children = list(folder.iterdir())
        if len(children) > 4096:
            raise ValueError("Too many saved series-map records to list")
        candidates = []
        for child in children:
            if QThread.currentThread().isInterruptionRequested():
                return result
            try:
                path = _safe(child / "index.json")
                if not child.is_dir() or not path.is_file():
                    continue
                value, size, bound = _read(path, budget)
                if value.get("format") not in formats or not matches_version(value, record):
                    continue
                if not isinstance(value.get("runs"), list):
                    raise ValueError("The saved series-map record lacks its run list")
                if any(not isinstance(row, dict) for row in value["runs"]):
                    raise ValueError("The saved series-map run entries must be objects")
                if (not isinstance(value.get("created_utc"), str)
                        or len(value["created_utc"]) > 64):
                    raise ValueError("The saved series-map record lacks its bounded timestamp")
                candidates.append(dict(value, folder=str(child), index_binding=bound))
                if isinstance(value.get('gap_review'), dict):
                    result['gap_reviews'].append(deepcopy(value['gap_review']))
            except (OSError, ValueError, TypeError) as error:
                result["errors"].append(f"{child.name}: {error}")
                if budget[0] > _TOTAL:
                    break
        if candidates:
            result[key] = max(candidates, key=lambda row: (str(row.get("created_utc") or ""), row["folder"]))
        if budget[0] > _TOTAL:
            break
    if budget[0] <= _TOTAL and current_path.is_file():
        try:
            current, _, _ = _read(current_path, budget)
            config = current.get("configuration") or {}
            provenance = config.get("provenance") or {}
            preparation = provenance.get("preparation")
            table_preparation = (config.get("painted_adoption_table") or {}).get("preparation")
            from .storage_paths import same_storage_descriptor
            if (preparation is not None and table_preparation is not None
                    and not same_storage_descriptor(preparation, table_preparation)):
                raise ValueError("The saved adoption has inconsistent preparation identities")
            if preparation is None:
                preparation = table_preparation
            if isinstance(preparation, dict):
                path = _safe(Path(str(preparation.get("path") or "")))
                if path.resolve().is_relative_to((root / "map_adoption").resolve()):
                    value, _, bound = _read(path, budget)
                    expected = dict(preparation, path=str(path))
                    if same_storage_descriptor(bound, expected) and matches_version(value, record):
                        result["adopted"] = dict(revision_id=current.get("revision_id"),
                            created_utc=current.get("created_utc"), preparation=bound)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            result["errors"].append(f"Current revision: {error}")
    return result


def progress_text(record, approval, result):
    """Describe recorded evidence without inferring completion from one SHA."""
    if record is None:
        return "Select a saved map version in Versions."
    lines = [f"Selected: {record.get('display_name') or record.get('name') or 'Saved map'}",
             "Approved: " + (str(approval.get("approved_utc") or approval.get("utc") or "recorded")
                             if approval else "not approved")]
    transfer = result.get("transfer")
    if transfer:
        rows = transfer["runs"]
        count = sum(row.get("status") == "placed" for row in rows)
        lines.append(f"Transferred: {count} of {len(rows)} runs, {transfer.get('created_utc') or 'time not recorded'}; "
                     "each run's camera-photo grid (this record does not establish a RAW-basis transfer).")
        if record.get("placement_withdrawn"):
            lines.append("Transfer approval withdrawn; the saved placement remains available for review.")
    else:
        lines.append("Transferred: no record for this saved version.")
    preparation = result.get("preparation")
    if preparation:
        rows = preparation["runs"]
        count = sum(row.get("status") == "prepared" for row in rows)
        blockers = len(preparation.get("blockers") or [])
        lines.append(f"Checked: preparation {count} of {len(rows)} runs, {preparation.get('created_utc') or 'time not recorded'}; "
                     f"{blockers} recorded blocker(s). The final Store check remains separate.")
    else:
        lines.append("Checked: no saved preparation for this version.")
    adopted = result.get("adopted")
    lines.append("Adopted: " + (f"current revision {adopted.get('revision_id') or 'recorded'}"
                                if adopted else "no exact current-revision record."))
    lines.append("Applied/built: not established by these saved records.")
    if result.get("errors"):
        lines.append("Some status records are unavailable: " + "; ".join(result["errors"][:2]))
    return "\n".join(lines)


class _Read(QObject):
    """One finite, read-only GUI metadata operation."""
    finished = Signal()

    def __init__(self, key, root, current, record):
        """Freeze plain metadata and paths; retain no live project or canvas."""
        super().__init__()
        self.key, self.root, self.current, self.record = key, root, current, deepcopy(record)
        self.result = None

    @Slot()
    def run(self):
        """Read once, report errors explicitly and return ownership to the GUI."""
        try:
            self.result = existing_progress(self.root, self.current, self.record)
        except Exception as error:
            self.result = dict(transfer=None, preparation=None, adopted=None,
                               errors=[f"{type(error).__name__}: {error}"])
        finally:
            self.moveToThread(QCoreApplication.instance().thread())
            self.finished.emit()


class SeriesUseStatus(QObject):
    """Retain one finite metadata reader and ignore obsolete selection results."""
    changed = Signal()

    def __init__(self, parent):
        """Build a passive status label; start no read before a map is selected."""
        super().__init__(parent)
        self.label = QLabel("Select a saved map version in Versions.")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.thread = self.worker = None
        self.pending = None
        self.desired = self.cached = None
        self.record = self.approval = None
        self.result = {}
        self._waiting = False

    def refresh(self, root, current, record, approval):
        """Load only a changed literal selection; never poll unchanged records."""
        self.record, self.approval = deepcopy(record), deepcopy(approval)
        if record is None or root is None:
            self.desired, self.pending = None, None
            self.result = {}
            self.label.setText(progress_text(record, approval, {}))
            return
        key = json.dumps([str(root), str(current), record.get("folder"), record.get("sha256"), approval], sort_keys=True)
        self.desired = key
        if key == self.cached:
            self.pending = None
            self.label.setText(progress_text(record, approval, self.result))
            return
        if self.worker is not None and self.worker.key == key:
            return
        self.pending = key, root, current, deepcopy(record)
        self.label.setText("Reading this saved version's existing series records…")
        if self.thread is None:
            self._start()

    def invalidate(self):
        """Allow one fresh metadata snapshot after a completed user operation."""
        self.cached = None

    def completed_adoption(self, record, approval):
        """Display completion only from this selection's exact saved snapshot.

        Approval or preparation alone is not adoption. A newer preparation must
        itself be saved as the current revision before it receives a checkmark.
        This uses retained metadata only and grants no execution admission.
        """
        if (not approval or self.desired is None or self.cached != self.desired
                or self.approval != approval
                or not matches_version(dict(map_version=self.record), record)
                or self.result.get("errors")):
            return None
        adopted, prepared = self.result.get("adopted"), self.result.get("preparation")
        if not isinstance(adopted, dict) or not isinstance(prepared, dict):
            return None
        from .storage_paths import same_storage_descriptor
        if (not matches_version(prepared, record)
                or not same_storage_descriptor(adopted.get("preparation"), prepared.get("index_binding"))):
            return None
        return adopted

    def _start(self):
        """Launch only the latest pending finite metadata snapshot."""
        if self.pending is None:
            return
        args, self.pending = self.pending, None
        self.thread, self.worker = QThread(self), _Read(*args)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        thread = self.thread
        self.thread.finished.connect(lambda owned=thread: self._retired(owned))
        self.thread.start()

    def _retired(self, expected=None):
        """Wait before dropping Qt ownership, then display only the desired key."""
        if self.thread is None or expected is not None and self.thread is not expected:
            return
        thread, worker = self.thread, self.worker
        thread.wait()
        self.thread = self.worker = None
        if worker.key == self.desired:
            self.cached, self.result = worker.key, worker.result or {}
            self.label.setText(progress_text(self.record, self.approval, self.result))
        del worker
        thread.deleteLater()
        if not self._waiting:
            self._start()
            self.changed.emit()

    def busy(self):
        """Report the owned finite reader for safe view shutdown."""
        return self.thread is not None

    def wait_for_workers(self):
        """Retire the current read without starting obsolete queued selections."""
        self.pending = None
        self._waiting = True
        try:
            if self.thread is not None:
                self.thread.wait()
                self._retired()
        finally:
            self._waiting = False
