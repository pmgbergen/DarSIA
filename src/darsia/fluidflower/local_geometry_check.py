"""One actual local geometry Check, reused only while its inputs stay unchanged."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path

_KEY = object()
_STAMP_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
_RESULT_FIELDS = frozenset(("depth_path", "labels_path", "facies_path",
    "facies_source_path", "facies_properties_source_path", "protocol_source_path"))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stamp(path):
    path = Path(path)
    try:
        value = path.stat()
    except FileNotFoundError:
        return None
    return tuple(getattr(value, name) for name in _STAMP_FIELDS)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _direct_inputs(candidate):
    """Watch only the ordinary local top-level and per-run reader locations."""
    result = {}
    rows = [candidate, *((candidate.get("run_geometry") or {}).get("runs") or [])]
    for row in rows:
        for name, value in row.items():
            if not name.endswith("_path") or not isinstance(value, str) or not value.strip():
                continue
            path = Path(value).expanduser().resolve()
            # RGB/RAW display photographs are not numeric inputs to adoption.
            # Their file versions stay watched; do not rehash 39 baseline image
            # archives merely to reuse an already reviewed categorical map.
            result[str(path)] = dict(stamp=_stamp(path),
                sha256=_digest(path) if name in _RESULT_FIELDS and path.is_file() else None)
            if path.is_dir() and name == "protocol_source_path":
                # This is exactly the original injector reader's table set.
                for item in sorted(path.iterdir()):
                    if item.suffix.casefold() in {".csv", ".xlsx", ".xls"}:
                        result[str(item)] = dict(stamp=_stamp(item),
                            sha256=_digest(item) if item.is_file() else None)
    return result


@dataclass(frozen=True, init=False)
class CheckedGeometryCandidate:
    """In-process result from the real validator, never a JSON success flag."""
    _candidate: str
    _report: str
    _direct: str
    _bound: str

    def __init__(self, key, candidate, report, direct, bound):
        if key is not _KEY:
            raise TypeError("Use the actual local geometry Check")
        for name, value in (("_candidate", candidate), ("_report", report),
                            ("_direct", direct), ("_bound", bound)):
            object.__setattr__(self, name, _json(value))

    def _unchanged(self):
        from .painted_adoption_table import _FILES
        direct = json.loads(self._direct)
        for name, expected in direct.items():
            actual = _stamp(name)
            if (list(actual) if actual is not None else None) != expected["stamp"]:
                raise ValueError("Geometry inputs changed after Check: " + name)
        bound = json.loads(self._bound)
        for row in bound["files"].values():
            row["stamp"] = tuple(row["stamp"])
        if not _FILES.unchanged(bound):
            raise ValueError("A consumed geometry source changed after Check")

    def report_for(self, candidate):
        """Return the actual Check only for the exact unchanged candidate."""
        if _json(candidate) != self._candidate:
            raise ValueError("The geometry candidate changed after Check")
        self._unchanged()
        return json.loads(self._report)

    def report_for_save(self, candidate):
        """Recheck each result-bearing input's bytes once before Store writes.

        This small bound read set replaces complete validation/RAW replays.
        It also catches same-size files whose timestamps were restored.
        """
        report = self.report_for(candidate)
        expected = {name: row["sha256"] for name, row in json.loads(self._direct).items()
                    if row["sha256"] is not None}
        for row in json.loads(self._bound)["files"].values():
            if row["path"] in expected and expected[row["path"]] != row["sha256"]:
                raise ValueError("The Check consumed conflicting source bytes")
            expected[row["path"]] = row["sha256"]
        for name, digest in expected.items():
            if _digest(name) != digest:
                raise ValueError("Geometry source bytes changed after Check: " + name)
        self._unchanged()
        return report

    def materialized_report(self, candidate, input_depth, source_depth):
        """Reuse arithmetic only after the Store's sole depth relocation is exact."""
        from .geometry_facies import geometry_config_digest, _file_reference
        self._unchanged()
        original = json.loads(self._candidate)
        original_path = Path(original["depth_path"]).expanduser().resolve()
        if Path(input_depth["path"]).resolve() != original_path:
            raise ValueError("The Store copied another depth input")
        copied = Path(candidate["depth_path"]).resolve()
        if copied == original_path:
            raise ValueError("Materialized depth must be the actual new Store copy")
        expected = deepcopy(original)
        expected.update(depth_path=str(copied), depth_artifact_path=str(copied),
                        depth_source_reference=deepcopy(source_depth))
        if _json(expected) != _json(candidate):
            raise ValueError("The Store changed more than its exact depth copy")
        actual = _file_reference(copied, include_full_sha256=True)
        if (actual.get("sha256") != input_depth.get("sha256")
                or actual.get("size_bytes") != input_depth.get("size_bytes")):
            raise ValueError("The Store depth copy differs from the checked input")
        self._unchanged()
        report = json.loads(self._report)
        report["config_sha256"] = geometry_config_digest(candidate)
        return report, actual


def check_geometry_candidate(candidate, *, display_attachments=None):
    """Run the ordinary validator once and retain its actual file-version closure."""
    from .geometry_facies import validate_geometry_config
    from .painted_adoption_table import _FILES
    candidate = deepcopy(dict(candidate))
    direct = _direct_inputs(candidate)
    with _FILES.capture() as bound:
        report = (validate_geometry_config(candidate) if display_attachments is None
            else validate_geometry_config(candidate, display_attachments=display_attachments))
    checked = CheckedGeometryCandidate(_KEY, candidate, report, direct, bound)
    checked.report_for(candidate)
    return checked
