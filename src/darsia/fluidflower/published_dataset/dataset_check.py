"""The launcher's first steps: check a dataset folder and the results folder.

* The manifest is read and checked; the record files' sizes are checked at
  once, and the full SHA-256 check is offered (record files, unpacked series
  inputs and expected results).
* The installed DarSIA commit and environment are compared with the manifest:
  another commit is refused unless the user overrides it (the report then says
  "not a reproduction"); other package builds or another platform are allowed,
  listed in the report and judged by the cross-machine limits.
* The time and disk estimates use the producer's measurements, labelled as such.
* The results folder must be new or empty, outside the dataset, with enough space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import shutil
import subprocess

from .archives import CHUNK_BYTES
from .checksums import verify_checksums
from .layout import inside
from .manifest import load_manifest


@dataclass
class DatasetCheck:
    """A dataset folder as the launcher found it."""

    root: Path
    manifest: dict
    problems: list = field(default_factory=list)
    code: dict = field(default_factory=dict)
    environment: list = field(default_factory=list)

    @property
    def runs(self) -> list[str]:
        """The dataset's runs, in manifest order."""
        return [row["run"] for row in self.manifest["runs"]]

    def frames(self, run: str) -> list[dict]:
        """One run's frames in time order."""
        return next(row["frames"] for row in self.manifest["runs"] if row["run"] == run)


def installed_commit() -> dict:
    """The DarSIA commit this installation runs, and whether its checkout is clean."""
    from ..series_paths import darsia_source_checkout

    checkout = darsia_source_checkout()
    if checkout is None:
        return {"commit": None, "clean": False, "source": "no Git checkout (an installed copy)"}
    try:
        commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain",
                                         "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "clean": False, "source": "Git is unavailable"}
    return {"commit": commit, "clean": not dirty.strip(), "source": "Git checkout"}


def compare_code(manifest: dict, installed: dict) -> dict:
    """Whether the installed commit is the published one (a reproduction) or another."""
    published = manifest["code"]["published_commit"]
    same = installed.get("commit") == published and installed.get("clean") is True
    return {"published": published, "installed": installed.get("commit"), "clean": installed.get("clean"),
            "source": installed.get("source"), "same": same,
            "note": "" if same else ("The installed DarSIA is not the published commit (or has local changes); "
                                     "running it is not a reproduction.")}


def compare_environment(manifest: dict, *, fingerprint=None) -> list[str]:
    """Installed environment and platform differences from the producer's (listed, allowed)."""
    from .environment import compare_fingerprints, distribution_fingerprint, platform_facts

    lines = []
    published = (manifest.get("environment") or {}).get("fingerprint")
    if published:
        actual = fingerprint if fingerprint is not None else distribution_fingerprint(
            hash_files=published.get("file_digests") == "content")
        lines += compare_fingerprints(published, actual)
    else:
        lines.append("The dataset records no installed-environment fingerprint")
    producer = (manifest.get("machine") or {}).get("platform")
    here = platform_facts()
    if isinstance(producer, dict):
        for key in ("system", "machine", "python"):
            if producer.get(key) != here.get(key):
                lines.append(f"Platform {key}: {producer.get(key)} published, {here.get(key)} here")
    return lines


def record_problems(root: Path, manifest: dict) -> list[str]:
    """Missing or wrongly sized record files, and unpacked parts that are missing (a fast check)."""
    problems = []
    sizes = {row["name"]: row["size_bytes"] for row in manifest.get("files") or ()}
    for name, size in sizes.items():
        path = Path(root) / name
        if not path.is_file():
            problems.append(f"Missing record file: {name}")
        elif path.stat().st_size != size:
            problems.append(f"Size differs from the manifest: {name}")
    if manifest["identity"]["record"] == "R":
        tasks = [task["path"] for task in (manifest["plan"].get("tasks") or {}).values()]
        series_manifest = (manifest["staged_reproduction"]["roots"]["series"] + "/series.toml"
                           if manifest.get("staged_reproduction") else "series.toml")
        for name in (series_manifest, manifest["plan"]["path"], *tasks):
            if not inside(root, name).is_file():
                problems.append(f"Not unpacked: {name} (unpack series.zip into the dataset folder)")
        for run in manifest["expected"]["runs"].values():
            for frame in run["frames"].values():
                if not inside(root, frame["layer"]["path"]).is_file():
                    problems.append("Not unpacked: the expected results (unpack expected.zip into the dataset folder)")
                    return problems
    return problems


def _sha256(path: Path) -> str:
    """The SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_check(root: Path, manifest: dict, *, progress=None) -> list[str]:
    """The full SHA-256 check: record files, unpacked series inputs, prepared tasks and expected results."""
    root = Path(root)
    problems = verify_checksums(root)
    if manifest.get("staged_reproduction"):
        from .staged_package import readback_staged
        try:
            readback_staged(root, manifest)
        except (ValueError, OSError, KeyError, TypeError) as error:
            problems.append("Staged reproduction: " + str(error))
    items = [(row["path"], row["sha256"]) for row in (manifest.get("inputs") or {}).get("series", ())]
    if manifest["identity"]["record"] == "R":
        items += [(task["path"], task["sha256"]) for task in (manifest["plan"].get("tasks") or {}).values()]
        for run in manifest["expected"]["runs"].values():
            items.append((run["mass_table"]["path"], run["mass_table"]["sha256"]))
            items += [(frame["layer"]["path"], frame["layer"]["sha256"]) for frame in run["frames"].values()]
        maps = manifest.get("export_mass_maps") or {}
        if maps and maps.get("source") != "original stored layer":
            for run in maps["runs"].values():
                items += [(frame[kind]["path"], frame[kind]["sha256"])
                          for frame in run.values() for kind in ("arrays", "metadata")]
        software = manifest.get("software")
        if software is not None:
            assets = [software["environment_lock"], software["inventory"]]
            if software["mode"] == "include":
                assets += [software["code"], *software["libraries"]]
            items += [(asset["path"], asset["sha256"]) for asset in assets]
    for index, (relative, digest) in enumerate(items):
        path = inside(root, relative)
        if not path.is_file():
            problems.append(f"Missing: {relative}")
        elif _sha256(path) != digest:
            problems.append(f"Changed: {relative}")
        if progress is not None:
            progress(index + 1, len(items))
    return problems


def open_dataset(root: str | Path, *, fingerprint=None, installed=None) -> DatasetCheck:
    """Read a dataset folder: manifest, fast size check, code and environment comparison."""
    root = Path(root)
    manifest = load_manifest(root)
    check = DatasetCheck(root, manifest)
    check.problems = record_problems(root, manifest)
    check.code = compare_code(manifest, installed_commit() if installed is None else installed)
    check.environment = compare_environment(manifest, fingerprint=fingerprint)
    return check


def estimates(manifest: dict, runs: list[str], *, frames: dict | None = None, workers: int = 1) -> dict:
    """Time and disk estimates for runs (or the first ``frames[run]`` frames of each).

    Time: the producer's measured seconds per frame, labelled as measured on the
    reference machine; runs share the workers. Disk: the producer's measured
    output bytes per frame, else the job layer's storage estimate; plus room to
    unpack the largest raw archive in scope.
    """
    from ..storage_profile import ESTIMATE_LABEL, estimate_job_bytes

    rows = {row["run"]: row for row in manifest["runs"]}
    counts = {run: (frames or {}).get(run) or len(rows[run]["frames"]) for run in runs}
    total = sum(counts.values())
    staged = manifest.get("staged_reproduction")
    if staged:
        output = sum(staged["stages"][stage_id]["output_bytes"]
                     for run in runs for stage_id in staged["runs"][run].values())
        scratch = sum(row["size_bytes"] for row in staged["input_assets"])
        return dict(frames=total, seconds=None,
            seconds_basis="complete original P1 plus P2 runtime has not been measured",
            output_bytes=output, output_basis="original P1 and P2 published product byte counts",
            scratch_bytes=scratch, needed_bytes=output + scratch)
    facts = manifest.get("machine") or {}
    seconds = facts.get("seconds_per_frame")
    per_frame = facts.get("output_bytes_per_frame")
    if per_frame:
        output, basis = per_frame * total, "measured on the reference machine (output bytes per frame)"
    else:
        estimate = estimate_job_bytes(manifest["plan"].get("storage_profile", "full"), max(1, total),
                                      runs=max(1, len(runs)), pipeline=manifest["plan"].get("pipeline"))
        output, basis = estimate["needed_bytes"], ESTIMATE_LABEL
    scratch = max(((rows[run].get("archive") or {}).get("size_bytes") or 0) for run in runs) if runs else 0
    longest = max(counts.values()) if counts else 0
    wall = None
    if seconds:
        # Whole runs per worker: the longest run bounds the time from below.
        wall = max(seconds * total / max(1, workers), seconds * longest)
    return {"frames": total, "seconds": wall, "seconds_basis": "measured on the reference machine",
            "output_bytes": output, "output_basis": basis, "scratch_bytes": scratch,
            "needed_bytes": output + scratch}


def results_folder_problems(results: str | Path, dataset_root: str | Path, needed_bytes: int | None) -> list[str]:
    """Why a results folder cannot be used: not new or empty, inside the dataset, or too little space."""
    results, dataset = Path(results), Path(dataset_root)
    problems = []
    if results.exists() and (not results.is_dir() or any(results.iterdir())):
        problems.append(f"The results folder must be new or empty: {results}")
    resolved = results.resolve()
    if resolved.is_relative_to(dataset.resolve()) or dataset.resolve().is_relative_to(resolved):
        problems.append("The results folder must lie outside the dataset folder")
    existing = resolved
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if needed_bytes is not None and free < needed_bytes:
        problems.append(f"Too little free space: {free / 1e9:.1f} GB free, about {needed_bytes / 1e9:.1f} GB needed")
    return problems
