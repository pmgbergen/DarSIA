"""``darsia-reproduce``: reproduce a published dataset's results on this machine.

Without arguments it opens the launcher window. With ``--dataset`` and
``--results`` it runs the same steps on the command line: check the dataset,
choose the scope and the machine settings, run, compare and write the report.
Exit status: 0 when the results are reproduced (every frame IDENTICAL or
WITHIN EXPECTED VARIATION), 1 when a frame is OUTSIDE or the launcher refuses,
2 for a usage error. The runs are replayed with the production code of a clean
Git clone of DarSIA checked out at the dataset's commit (not a source archive).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parser() -> argparse.ArgumentParser:
    """The ``darsia-reproduce`` argument parser."""
    result = argparse.ArgumentParser(prog="darsia-reproduce", description=__doc__.splitlines()[0])
    result.add_argument("--dataset", type=Path, help="The unpacked dataset folder")
    result.add_argument("--results", type=Path, help="A new or empty results folder")
    result.add_argument("--runs", nargs="+", help="Reproduce these runs (default: all)")
    result.add_argument("--quick", action="store_true",
                        help="The quick check: the first frames of one run (only where the dataset allows it)")
    result.add_argument("--quick-frames", type=int, help="Frames of the quick check (default: the dataset's)")
    result.add_argument("--workers", type=int, help="Workers (default: from the measured memory per worker)")
    result.add_argument("--threads", type=int, help="Threads per worker (default: the dataset's)")
    result.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--full-check", action="store_true", help="Check every SHA-256 before running")
    result.add_argument("--override-commit", action="store_true",
                        help="Run another DarSIA commit anyway; the report says it is not a reproduction")
    result.add_argument("--keep-scratch", action="store_true", help="Keep the unpacked raw images")
    result.add_argument("--stage-runtimes", type=Path,
                        help="Explicit Python and clean original checkout for each staged producer")
    result.add_argument("--gui", action="store_true", help="Open the launcher window")
    return result


def _print(text: str) -> None:
    """Print a line to the console when there is one."""
    if sys.stdout is not None:
        print(text, flush=True)


def run(arguments, *, runner=None, machine=None, installed=None) -> int:
    """Run the command-line steps; return the exit status."""
    from .reproduce import Reproduction, ReproductionRefused, Settings, choose_scope, execute
    from .readme import export_resolution

    if runner is None:
        from .replay import ReplayRunner

        runner = ReplayRunner()
    reproduction = Reproduction(arguments.dataset, arguments.results, runner=runner)
    check = reproduction.open(installed=installed)
    manifest = check.manifest
    _print(f"Dataset: {manifest['identity']['title']} ({manifest['identity']['series_id']}, record "
           f"{manifest['identity']['record']}, mass maps {export_resolution(manifest)}): {len(check.runs)} runs.")
    staged = bool(manifest.get("staged_reproduction"))
    if staged:
        _print("DarSIA: each original stage's code and environment are verified before scientific execution.")
    else:
        _print(f"DarSIA: published {check.code['published']}, installed {check.code['installed']}"
               + ("" if check.code["same"] else f". {check.code['note']}"))
    for line in check.environment:
        _print(("Launcher environment: " if staged else "Environment: ") + line)
    if arguments.full_check:
        problems = reproduction.verify()
        if problems:
            for line in problems:
                _print(f"Check: {line}")
            return 1
        _print("Every SHA-256 matches the dataset.")
    try:
        if arguments.quick:
            scope = choose_scope(manifest, "quick", runs=arguments.runs, quick_frames=arguments.quick_frames)
        elif arguments.runs:
            scope = choose_scope(manifest, "run", runs=arguments.runs)
        else:
            scope = choose_scope(manifest, "all")
        advice = reproduction.advice(scope, threads=arguments.threads, backend=arguments.backend)
        _print(f"Workers: {advice['reason']}")
        if advice.get("message"):
            _print(advice["message"])
        workers = arguments.workers if arguments.workers is not None else advice.get("workers")
        if not workers:
            raise ReproductionRefused("No worker count: choose one with --workers")
        settings = Settings(workers=workers, threads=advice["threads"], backend=arguments.backend,
                            override_commit=arguments.override_commit, remove_scratch=not arguments.keep_scratch,
                            stage_runtimes=str(arguments.stage_runtimes) if getattr(arguments, "stage_runtimes", None) else None)
        estimate = reproduction.estimate(scope, workers)
        if estimate["seconds"]:
            _print(f"About {estimate['seconds'] / 3600:.1f} h ({estimate['seconds_basis']}); "
                   f"about {estimate['needed_bytes'] / 1e9:.1f} GB of disk.")
        result = execute(reproduction, scope, settings, machine=machine,
                         progress=lambda states: _print("Runs: " + ", ".join(f"{run} {state}"
                                                                              for run, state in states.items())))
    except ReproductionRefused as error:
        _print(f"Refused: {error}")
        return 1
    summary = result["summary"]
    _print(f"Verdict: {summary['dataset']}. Report: {result['files']['html']}")
    if summary.get("first_divergence"):
        first = summary["first_divergence"]
        _print(f"First divergence: run {first['run']}, {first['case_id']}, {first['quantity']}")
    return 0 if summary["reproduced"] else 1


def main(argv=None) -> int:
    """Open the window without arguments; otherwise run the command line."""
    arguments = parser().parse_args(argv)
    if arguments.gui or (arguments.dataset is None and arguments.results is None):
        from .reproduce_window import main as window_main

        return window_main(arguments)
    if arguments.dataset is None or arguments.results is None:
        parser().error("--dataset and --results are both required on the command line")
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
