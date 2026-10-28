"""The reproduction launcher's steps, shared by the window and the command line.

1. Dataset folder: the manifest, the fast size check (the full SHA-256 check on
   request), the installed commit and environment against the manifest.
2. Results folder: new or empty, outside the dataset, with free space for the
   estimate.
3. Scope: the quick check (the first frames of one run, only when the manifest
   says it is valid), one run, or all runs.
4. Machine settings: workers from the published measurement, threads from the
   manifest, CPU unless CUDA is chosen; the precision is the manifest's.
5. Run: the dataset's prepared tasks, replayed on this machine with the
   production code by a runner (``replay.ReplayRunner``).
6. Compare and report: three verdicts per frame, report.json, report.html and
   English figures in the results folder.

Everything the launcher writes goes into the results folder; the dataset folder
is only read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .comparison import compare_frame, load_layer, summarize
from .dataset_check import (DatasetCheck, estimates, full_check, open_dataset, results_folder_problems)
from .layout import inside
from .quick_check import DEFAULT_QUICK_FRAMES
from .report import mass_rows, report_document, write_report
from .resources import default_workers

SCOPES = ("quick", "run", "all")
REPORT_FOLDER = "report"


class ReproductionRefused(ValueError):
    """The launcher refuses to start; the message says why."""


@dataclass
class Scope:
    """What to reproduce: the kind, the runs and, for the quick check, frames per run."""

    kind: str
    runs: list
    frames: dict = field(default_factory=dict)


@dataclass
class Settings:
    """The machine settings of one reproduction."""

    workers: int
    threads: int
    backend: str = "cpu"
    override_commit: bool = False
    remove_scratch: bool = True
    stage_runtimes: str | None = None


def choose_scope(manifest: dict, kind: str, *, run: str | None = None, quick_frames: int | None = None,
                 runs: list | None = None) -> Scope:
    """The scope of a reproduction; the quick check only when the manifest says it is valid.

    ``kind`` is "quick" (the first frames of ``run``), "run" (``run``, or the
    listed ``runs``) or "all".
    """
    known = [row["run"] for row in manifest["runs"]]
    if kind not in SCOPES:
        raise ReproductionRefused(f"Scope must be one of {', '.join(SCOPES)}")
    if kind == "all":
        return Scope("all", known)
    chosen = list(runs) if runs else [run or known[0]]
    unknown = [item for item in chosen if item not in known]
    if unknown:
        raise ReproductionRefused(f"The dataset has no run {', '.join(unknown)}")
    if kind == "run":
        return Scope("run", [item for item in known if item in chosen])
    selected = chosen[0]
    if not manifest.get("quick_check_valid"):
        raise ReproductionRefused("The quick check is not valid for this dataset: a step uses a run's later frames")
    available = len(next(row["frames"] for row in manifest["runs"] if row["run"] == selected))
    count = quick_frames or (manifest.get("quick_check") or {}).get("default_frames") or DEFAULT_QUICK_FRAMES
    return Scope("quick", [selected], {selected: max(1, min(int(count), available))})


def scope_frames(manifest: dict, scope: Scope) -> list[tuple[str, dict]]:
    """The (run, frame) pairs in scope, in time order within each run."""
    pairs = []
    for row in manifest["runs"]:
        if row["run"] in scope.runs:
            frames = row["frames"]
            pairs += [(row["run"], frame) for frame in frames[:scope.frames.get(row["run"], len(frames))]]
    return pairs


def plan_differences(manifest: dict, scope: Scope, settings: Settings, code: dict) -> list[str]:
    """The settings of a new analysis that differ from the dataset's published plan, in words."""
    machine = manifest.get("machine") or {}
    runs = [row["run"] for row in manifest["runs"]]
    lines = []
    if scope.kind == "quick":
        run = scope.runs[0]
        lines.append(f"Scope: the quick check, the first {scope.frames[run]} frame(s) of run {run} "
                     f"(the published plan has {len(runs)} run(s))")
    elif scope.runs != runs:
        lines.append(f"Scope: run(s) {', '.join(scope.runs)} of the published {len(runs)}")
    if machine.get("threads") is not None and settings.threads != machine["threads"]:
        lines.append(f"Threads per worker: {settings.threads} (published: {machine['threads']})")
    if machine.get("backend") is not None and settings.backend != machine["backend"]:
        lines.append(f"Backend: {settings.backend} (published: {machine['backend']})")
    if not code.get("same"):
        lines.append(f"DarSIA commit: {code.get('installed')} (published: {code.get('published')}); "
                     "not a reproduction")
    return lines


def _staged_bit_verdict(comparison, expected, reproduced):
    """Require exact final scalar encodings in addition to stored array bits.

    The established cross-machine comparator intentionally uses scalar value
    equality. Staged RAW replay has the stronger original-byte requirement,
    including signed zero and JSON number type, without changing that legacy
    comparator or accepting missing/nonfinite masses as matching output.
    """
    import math
    import struct
    from .comparison import Check, IDENTICAL, MASSES, OUTSIDE

    failures = []
    for name in MASSES:
        left, right = expected.get(name), reproduced.get(name)
        identical = type(left) is type(right) and type(left) in (int, float)
        if identical and type(left) is float:
            identical = math.isfinite(left) and math.isfinite(right) and struct.pack("!d", left) == struct.pack("!d", right)
        elif identical:
            identical = left == right
        detail = ("exact scalar type and bits" if identical else
                  f"{left!r} ({type(left).__name__}) vs {right!r} ({type(right).__name__}); exact scalar type and bits required")
        check = Check(name + " mass bits", "mass", 0.0 if identical else math.inf,
                      0.0, identical, detail, 0.0 if identical else math.inf)
        comparison.checks.append(check)
        if not identical:
            failures.append(dict(quantity=check.quantity, kind=check.kind, detail=check.detail))
    if failures or comparison.verdict != IDENTICAL:
        comparison.verdict = OUTSIDE
        if failures:
            comparison.first_failure = failures[0]
        elif comparison.first_failure is None:
            detail = "Exact stored layer dtype, shape and element bytes are required"
            comparison.checks.append(Check("stored layer bits", "map", math.inf, 0.0, False, detail, math.inf))
            comparison.first_failure = dict(quantity="stored layer bits", kind="map", detail=detail)
    return comparison


class Reproduction:
    """One reproduction of a dataset into a results folder, step by step."""

    def __init__(self, dataset_root, results_root, *, runner=None):
        """Bind the dataset and results folders and the runner (default: the production job layer)."""
        self.dataset_root, self.results_root = Path(dataset_root), Path(results_root)
        self.runner = runner
        self.check: DatasetCheck | None = None
        self.comparisons = []
        self.reproduced_masses = {}
        self.summary = None

    def open(self, *, installed=None, fingerprint=None) -> DatasetCheck:
        """Step 1: read and check the dataset folder (fast checks only)."""
        self.check = open_dataset(self.dataset_root, installed=installed, fingerprint=fingerprint)
        return self.check

    def verify(self, *, progress=None) -> list[str]:
        """The full SHA-256 check of the dataset folder."""
        return full_check(self.dataset_root, self.check.manifest, progress=progress)

    def advice(self, scope: Scope, *, threads=None, backend="cpu", inventory=None) -> dict:
        """Step 4: the worker default and its reason for this scope."""
        return default_workers(self.check.manifest, runs_in_scope=len(scope.runs), threads=threads,
                               backend=backend, inventory=inventory)

    def estimate(self, scope: Scope, workers: int) -> dict:
        """Time and disk estimates for the scope."""
        return estimates(self.check.manifest, scope.runs, frames=scope.frames, workers=workers)

    def preflight(self, scope: Scope, settings: Settings) -> list[str]:
        """Everything that refuses a start: dataset problems, commit, results folder, workers."""
        problems = list(self.check.problems)
        if not self.check.manifest.get("staged_reproduction") and not self.check.code["same"] and not settings.override_commit:
            problems.append(self.check.code["note"] + " Choose to override to run it anyway.")
        if type(settings.workers) is not int or settings.workers < 1:
            problems.append("Choose at least one worker")
        if settings.backend not in {"cpu", "cuda"} or type(settings.threads) is not int or settings.threads < 1:
            problems.append("Choose the CPU or CUDA backend and a positive thread count")
        problems += results_folder_problems(self.results_root, self.dataset_root,
                                            self.estimate(scope, max(1, settings.workers or 1))["needed_bytes"])
        runner_preflight = getattr(self.runner, "preflight", None)
        if callable(runner_preflight):
            problems += runner_preflight(self.results_root, check=self.check, scope=scope, settings=settings)
        return problems

    def run(self, scope: Scope, settings: Settings, *, progress=None) -> None:
        """Step 5: run the scope with the runner; refuse when the preflight finds anything."""
        problems = self.preflight(scope, settings)
        if problems:
            raise ReproductionRefused("\n".join(problems))
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.runner.run(self.check, self.results_root, scope, settings, progress=progress)

    def compare(self, scope: Scope) -> dict:
        """Step 6a: compare every reproduced frame in scope with the expected results."""
        manifest = self.check.manifest
        rule = manifest["reproducibility"]["other_machines"]
        expected = manifest["expected"]["runs"]
        self.comparisons, self.reproduced_masses = [], {}
        for run, frame in scope_frames(manifest, scope):
            case = frame["case_id"]
            entry = expected[run]["frames"][case]
            masses, layer = self.runner.reproduced(run, case)
            self.reproduced_masses[(run, case)] = masses
            compared = compare_frame(run, case, entry["masses_g"], masses,
                                     load_layer(inside(self.dataset_root, entry["layer"]["path"])), layer, rule)
            if manifest.get("staged_reproduction"):
                compared = _staged_bit_verdict(compared, entry["masses_g"], masses)
            self.comparisons.append(compared)
        self.summary = summarize(self.comparisons)
        if manifest.get("staged_reproduction"):
            from .comparison import IDENTICAL, OUTSIDE
            self.summary["bit_identical_required"] = True
            self.summary["reproduced"] = all(item.verdict == IDENTICAL for item in self.comparisons)
            if not self.summary["reproduced"]:
                self.summary["dataset"] = OUTSIDE
                self.summary["requirement_failure"] = "Staged RAW reproduction requires bit-identical final results"
        return self.summary

    def write_report(self, scope: Scope, settings: Settings, *, machine=None, timing=None) -> dict:
        """Step 6b: write report.json, report.html and the figures into the results folder."""
        from .environment import machine_facts

        manifest = self.check.manifest
        if manifest.get("staged_reproduction"):
            settings = getattr(getattr(self.runner, "staged_runner", None), "settings", None) or settings
        frames = scope_frames(manifest, scope)
        route = getattr(self.runner, "route", None)
        report = report_document(
            manifest=manifest, check={"code": self.check.code, "environment": self.check.environment},
            settings={"scope": scope.kind, "runs": ", ".join(scope.runs), "frames": len(frames),
                      **{key: value for key, value in asdict(settings).items()},
                      "precision": manifest["plan"].get("precision"), **({"route": route} if route else {})},
            summary=self.summary, frames=[comparison.as_dict() for comparison in self.comparisons],
            machine=machine if machine is not None else machine_facts(), timing=timing,
            differences=plan_differences(manifest, scope, settings, self.check.code))
        if manifest.get("staged_reproduction"):
            evidence = getattr(self.runner, "staged_evidence", None)
            if not evidence or not evidence.get("complete"):
                raise ReproductionRefused("The original staged replay has no complete runtime/coupling evidence")
            report["staged_reproduction"] = evidence
            report["reproduction"] = True
            report["note"] = ("Original per-stage code and locked environments verified; see staged evidence. "
                              "Staged RAW reproduction requires bit-identical final results.")
            report["settings"].pop("stage_runtimes", None)
            report["settings"]["machine_settings_scope"] = (
                "Final launcher settings; successful settings for each run are recorded in staged_reproduction.runs")
            report["differences_from_published_plan"] = [line for line in report["differences_from_published_plan"]
                                                       if not line.startswith("DarSIA commit:")]
        verdicts = {(item.run, item.case_id): item.verdict for item in self.comparisons}
        rows = mass_rows(manifest, verdicts, self.reproduced_masses)
        return write_report(self.results_root / REPORT_FOLDER, report, rows)


def execute(reproduction: Reproduction, scope: Scope, settings: Settings, *, progress=None,
            machine=None) -> dict:
    """Steps 5 and 6: run, compare and write the report; the window and the command line share it.

    Returns the dataset verdict summary and the report's files.
    """
    import time

    started = time.monotonic()
    reproduction.run(scope, settings, progress=progress)
    seconds = time.monotonic() - started
    summary = reproduction.compare(scope)
    frames = len(scope_frames(reproduction.check.manifest, scope))
    files = reproduction.write_report(scope, settings, machine=machine,
                                      timing={"seconds": seconds, "frames": frames,
                                              "seconds_per_frame_measured_here": seconds / max(1, frames)})
    return {"summary": summary, "files": files}
