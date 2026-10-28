"""Read a finished production job (read-only) for a published dataset.

The job layer records what a job read and produced:

* ``job.json``: code commit, precision, calibration, environment lock, series;
* ``results/<run>/task.json``: the run's input catalogue, each input as a root
  alias, a path relative to that root, its SHA-256, size, role and uses;
* ``done/<run>.json`` and the products index it names: every final product
  with the producer's local path, SHA-256 and size, and a shared copy of the
  small metadata files (``result.json``, the mass table);
* ``results/<run>/completed.json``: the run's completion record.

Nothing here writes, and every byte read is checked against its recorded
SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from ..image_formats import image_family
from ..pipeline.job_queue import name
from .scan import ScanName, receipt_names

ALIAS_JOB_FORMAT = "darsia-alias-job-v1"
DONE_FORMAT = "darsia-alias-done-v1"
PRODUCTS_FORMAT = "darsia-products-index-v1"
LAYER_FILE = "storage_stored_frame.npz"
LAYER_RECEIPT = "storage_completed.json"
_SERVICE_SUFFIX = re.compile(r"-(?:cpu|cuda|any)$", re.IGNORECASE)


@dataclass(frozen=True)
class Product:
    """One final product of a run: where its producer wrote it, and a shared copy."""

    product_id: str
    machine_id: str
    local_path: str
    sha256: str
    size_bytes: int
    kind: str
    shared_copy: Path | None


@dataclass(frozen=True)
class Input:
    """One input the job read: its root alias, relative path, SHA-256, size, role and uses."""

    root_alias: str
    relative_path: str
    sha256: str
    size_bytes: int
    role: str
    uses: tuple[str, ...]

    @property
    def camera_file(self) -> bool:
        """True for a photograph read from outside the series folder (raw or JPEG)."""
        return self.root_alias != "series" and image_family(self.relative_path) != "other"


class JobError(ValueError):
    """The folder is not a finished production job this reader can use."""


def _strict_json(raw: bytes):
    """Decode JSON and refuse duplicate keys."""
    def pairs(rows):
        """One JSON object; a repeated key is an error."""
        value = {}
        for key, item in rows:
            if key in value:
                raise JobError("Duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs)


def _verified(path: Path, sha256: str, size: int) -> bytes:
    """A file's bytes, checked against its recorded SHA-256 and size."""
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
        raise JobError(f"File differs from its record: {path}")
    return raw


class FinishedJob:
    """A finished alias job: its runs, inputs, frames, products and receipts."""

    def __init__(self, root: str | Path):
        """Read and check the job's records; raise JobError for an unfinished or other job."""
        self.root = Path(root)
        path = self.root / "job.json"
        if not path.is_file():
            raise JobError(f"No job.json in {self.root}")
        self.job = _strict_json(path.read_bytes())
        if self.job.get("format") != ALIAS_JOB_FORMAT:
            raise JobError("A finished alias job (darsia-alias-job-v1) is required")
        self.runs = [str(run) for run in self.job.get("expected_runs") or ()]
        if not self.runs:
            raise JobError("The job names no runs")
        self.tasks, self.done, self.completed, self.products = {}, {}, {}, {}
        self._index_roots = {}
        for run in self.runs:
            key = name(run)
            self.tasks[run] = _strict_json((self.root / "results" / key / "task.json").read_bytes())
            done_path = self.root / "done" / f"{key}.json"
            if not done_path.is_file():
                raise JobError(f"Run {run} is not finished: no done record")
            done = _strict_json(done_path.read_bytes())
            if done.get("format") != DONE_FORMAT or done.get("run") != run:
                raise JobError(f"Unexpected done record for run {run}")
            self.done[run] = done
            completed = done["completed"]
            self.completed[run] = _strict_json(_verified(
                self._job_path(completed["relative_path"]), completed["sha256"], completed["bytes"]))
            if self.completed[run].get("complete") is not True:
                raise JobError(f"Run {run} has no complete result")
            index_ref = done["products_index"]
            index_path = self._job_path(index_ref["relative_path"])
            index = _strict_json(_verified(index_path, index_ref["sha256"], index_ref["bytes"]))
            if index.get("format") != PRODUCTS_FORMAT or index.get("run") != run:
                raise JobError(f"Unexpected products index for run {run}")
            self._index_roots[run] = index_path.parent
            self.products[run] = {row["product_id"]: self._product(run, row) for row in index["products"]}

    @property
    def series_id(self) -> str:
        """The series identity recorded in the job."""
        return str(self.job.get("series_id") or "")

    def _job_path(self, relative: str) -> Path:
        """A path inside the job folder from a recorded relative path."""
        parts = PurePosixPath(relative).parts
        if not parts or any(part in {"", ".", ".."} for part in parts) or "\\" in relative:
            raise JobError(f"Invalid job-relative path: {relative}")
        return self.root.joinpath(*parts)

    def _product(self, run, row) -> Product:
        """One products-index row, with its shared copy's location when there is one."""
        copy = row.get("metadata_copy")
        shared = None
        if isinstance(copy, dict):
            parts = PurePosixPath(copy["relative_path"]).parts
            if parts and not any(part in {"", ".", ".."} for part in parts):
                shared = self._index_roots[run].joinpath(*parts)
        return Product(row["product_id"], str(row.get("machine_id") or ""), str(row["local_path"]),
                       row["sha256"], int(row["bytes"]), str(row.get("kind") or ""), shared)

    def product(self, run: str, product_id: str) -> Product:
        """One final product of a run, by its product id."""
        try:
            return self.products[run][product_id]
        except KeyError:
            raise JobError(f"Run {run} has no product {product_id}") from None

    def product_source(self, product: Product) -> Path:
        """A readable file with the product's bytes: the shared copy or the producer's file.

        The producer's own file is only readable on the producing machine (or
        where its folder is reachable); the caller verifies size and SHA-256.
        """
        if product.shared_copy is not None and product.shared_copy.is_file():
            return product.shared_copy
        local = Path(product.local_path)
        if local.is_file():
            return local
        raise JobError(f"Product {product.product_id} is only on its producing machine; "
                       "run the build there")

    def product_bytes(self, product: Product) -> bytes:
        """The verified bytes of a (small) product."""
        return _verified(self.product_source(product), product.sha256, product.size_bytes)

    def frames(self, run: str) -> list[str]:
        """The run's case ids in time order."""
        return [str(case) for case in self.tasks[run]["ordered_case_ids"]]

    def frame_result(self, run: str, case: str) -> dict:
        """The frame's ``result.json`` (masses, time, source binding)."""
        return _strict_json(self.product_bytes(self.product(run, f"frames/{case}/result.json")))

    def layer(self, run: str, case: str, leaf: str = LAYER_FILE) -> Product | None:
        """The frame's stored coarse layer (or its receipt), when the job wrote one."""
        for product_id, product in self.products[run].items():
            parts = PurePosixPath(product_id).parts
            if parts[-1:] == (leaf,) and case in parts[-2:-1]:
                return product
        return None

    def mass_table(self, run: str) -> Product:
        """The run's scalar mass table product."""
        return self.product(run, "mass_table.csv")

    def inputs(self, run: str) -> list[Input]:
        """Every input the run read, as recorded in its input catalogue."""
        rows = []
        for entry in self.tasks[run]["input_catalog"]["inputs"]:
            reference = entry["reference"]
            rows.append(Input(reference["root_alias"], reference["relative_path"], reference["sha256"],
                              int(reference["bytes"]), reference["role"], tuple(entry["uses"])))
        return rows

    def plan_reference(self) -> dict:
        """The job's execution plan as an input reference (root alias, path, SHA-256)."""
        return dict(self.job["series"])

    def series_manifest_reference(self) -> dict:
        """The series.toml the job read, as an input reference."""
        return dict(self.job["series_manifest"])

    def receipt_names(self) -> tuple[ScanName, ...]:
        """Machine and user names the job's receipts record, for the name scan."""
        names = list(receipt_names(self.job, "production job"))
        for run in self.runs:
            names.extend(receipt_names(self.completed[run], "production receipt"))
            for product in self.products[run].values():
                if product.machine_id:
                    names.append(ScanName(product.machine_id, "production receipt"))
        for folder in ("claimed", "logs"):
            base = self.root / folder
            if base.is_dir():
                for child in base.iterdir():
                    stem = child.stem if child.is_file() else child.name
                    if folder == "logs" and not child.suffix == ".log":
                        continue
                    names.append(ScanName(stem, "production receipt"))
        for path in (self.root / "ui" / "master.json", *sorted((self.root / "ui" / "machines").glob("*.json"))):
            if path.is_file():
                names.extend(receipt_names(_strict_json(path.read_bytes()), "production receipt"))
        for lease in sorted((self.root / "leases").glob("*/owner.json")):
            names.extend(receipt_names(_strict_json(lease.read_bytes()), "production receipt"))
        expanded = []
        for item in names:
            expanded.append(item)
            stripped = _SERVICE_SUFFIX.sub("", item.text)
            if stripped != item.text:
                expanded.append(ScanName(stripped, item.source))
        return tuple(expanded)

    def measured(self, run: str) -> dict:
        """What the run's receipts measured: backend, precision, threads and times."""
        completed = self.completed[run]
        task = self.tasks[run]
        frames = completed.get("case_count") or len(self.frames(run))
        wall = completed.get("wall_seconds")
        computed = completed.get("newly_computed_frames")
        per_frame = None
        if isinstance(wall, (int, float)) and frames:
            per_frame = float(wall) / float(computed if isinstance(computed, int) and computed > 0 else frames)
        return {
            "backend": completed.get("backend") or task.get("backend"),
            "precision": completed.get("precision") or task.get("precision"),
            "threads": task.get("threads"),
            "frames": frames,
            "wall_seconds": wall,
            "seconds_per_frame": per_frame,
            "pipeline": completed.get("pipeline") or task.get("pipeline"),
        }


def producer_is_this_machine(job: FinishedJob, host_names) -> bool:
    """True when a product's producing machine id names this machine."""
    wanted = {_SERVICE_SUFFIX.sub("", str(value)).casefold() for value in host_names if value}
    for run in job.runs:
        for product in job.products[run].values():
            if product.machine_id and product.machine_id.casefold() in wanted:
                return True
    return False
