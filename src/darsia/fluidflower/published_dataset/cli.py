"""Command line of published datasets: ``darsia dataset build``.

The same builder as the GUI action Build published dataset (Analysis >
Results and export). ``--check`` checks without writing anything.
Exit status: 0 when built (or, with ``--check``, when nothing refuses),
1 when the builder refuses, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .layout import RECORD_TYPES, TIER_ORDER


def _pair(text: str) -> tuple[str, str]:
    """Parse one ``NAME=VALUE`` argument."""
    key, separator, value = text.partition("=")
    if not separator or not key or not value:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got {text!r}")
    return key, value


def parser() -> argparse.ArgumentParser:
    """The ``darsia dataset`` argument parser (subcommand ``build``)."""
    main_parser = argparse.ArgumentParser(prog="darsia dataset", description=__doc__.splitlines()[0])
    commands = main_parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Check and build a published dataset record")
    build.add_argument("--series", type=Path, required=True, help="The series folder (holds series.toml)")
    build.add_argument("--job", type=Path, required=True,
                       help="The finished production job that made the expected results")
    build.add_argument("--record", choices=RECORD_TYPES, required=True,
                       help="R: reproduction (raw images), S: simulator products")
    build.add_argument("--tier", choices=TIER_ORDER,
                       help="Optional legacy raw-image policy; otherwise keep the exact finished-job selection")
    build.add_argument("--output", type=Path, required=True, help="A new or empty folder for the record")
    build.add_argument("--title", default="", help="Record title (default: the series name)")
    build.add_argument("--tiers", type=Path, help="Tier file (default: image_selection/published_tiers.json)")
    build.add_argument("--root", type=_pair, action="append", default=[], metavar="ALIAS=FOLDER",
                       help="Where an input root of the job lies on this machine")
    build.add_argument("--published-plan", type=Path, help="A clean revision of the job's execution plan")
    build.add_argument("--published-commit", help="The DarSIA commit the record publishes (default: the job's)")
    build.add_argument("--proof", type=Path, help="Proof record for a published revision")
    build.add_argument("--proof-job", type=Path,
                       help="The finished job of the run that proved a published revision (its tasks are published)")
    build.add_argument("--known-variation", type=Path, help="JSON list of known variation between machines")
    build.add_argument("--peak-memory-bytes", type=int, help="Measured peak memory per worker, in bytes")
    build.add_argument("--peak-memory-basis", default="", help="How the peak memory was measured")
    build.add_argument("--expected-resolution-mm", type=float,
                       help="Optional check on the original stored layer resolution; otherwise read it from the job")
    build.add_argument("--simulator-resolution-mm", type=float,
                       help="Required for an S record; there is no default")
    build.add_argument("--simulator-geometry", type=Path,
                       help="An S record's simulator geometry request (the completed porous domain, N18)")
    build.add_argument("--mass-map-resolution-mm", type=int, choices=(1, 2, 5, 10),
                       help="Resolution of attached export maps; original expected results stay unchanged")
    build.add_argument("--software-mode", choices=("references", "include"),
                       help="Exact software references or included source and every locked library")
    build.add_argument("--software-inventory", type=Path,
                       help="Explicit inventory bound to the producing job's code and environment")
    build.add_argument("--software-stage-inventory", type=Path,
                       help="Exact software inventories for every original P1 and P2 stage")
    build.add_argument("--retain-producer-metadata", action="store_true",
                       help="Retain immutable staged producer receipts, including local paths and producer identifiers")
    build.add_argument("--software-folder", type=Path,
                       help="Folder with the job's unchanged lock; inclusion also needs source.bundle and wheels/")
    build.add_argument("--software-code-url", default="", help="HTTPS code reference when using --software-folder")
    build.add_argument("--software-python", default="", help="Exact producing Python version when using --software-folder")
    build.add_argument("--link", type=_pair, action="append", default=[], metavar="KIND=DOI",
                       help="other_record, code or article")
    build.add_argument("--no-fingerprint", action="store_true",
                       help="Do not record the installed-environment fingerprint")
    build.add_argument("--check", action="store_true", help="Check only; write no export files")
    build.add_argument("--report", type=Path, help="Also write the build report as JSON here")
    return main_parser


def request_from(arguments) -> "BuildRequest":
    """The build request of parsed ``build`` arguments."""
    from .builder import BuildRequest

    return BuildRequest(
        series_root=arguments.series, job_root=arguments.job, record=arguments.record, tier=arguments.tier,
        output=arguments.output, title=arguments.title, tiers_file=arguments.tiers,
        roots=dict(arguments.root) or None, published_plan=arguments.published_plan,
        published_commit=arguments.published_commit, proof=arguments.proof, proof_job=arguments.proof_job,
        known_variation=arguments.known_variation, peak_memory_bytes=arguments.peak_memory_bytes,
        peak_memory_basis=arguments.peak_memory_basis,
        expected_resolution_mm=arguments.expected_resolution_mm,
        simulator_resolution_mm=arguments.simulator_resolution_mm, simulator_geometry=arguments.simulator_geometry,
        require_simulator_geometry=arguments.record == "S" and arguments.tier is None,
        mass_map_resolution_mm=arguments.mass_map_resolution_mm,
        software_mode=arguments.software_mode or ("references" if arguments.record == "R" and arguments.tier is None else None),
        software_inventory=arguments.software_inventory,
        software_stage_inventory=arguments.software_stage_inventory,
        retain_producer_metadata=arguments.retain_producer_metadata,
        software_folder=arguments.software_folder, software_code_url=arguments.software_code_url,
        software_python=arguments.software_python,
        links=dict(arguments.link),
        fingerprint=not arguments.no_fingerprint)


def _print(text: str) -> None:
    """Print to the console when there is one.

    A GUI entry point on Windows has no console; ``--report`` still records the result.
    """
    if sys.stdout is not None:
        print(text, end="", flush=True)


def main(argv=None) -> int:
    """Run ``darsia dataset build``; return the exit status."""
    from .builder import build_record, check_build

    arguments = parser().parse_args(argv)
    request = request_from(arguments)
    if arguments.check:
        _plan, report = check_build(request)
    else:
        report = build_record(request)
    _print(report.text())
    if arguments.report is not None:
        arguments.report.write_text(json.dumps(report.as_dict(), indent=1) + "\n", encoding="utf-8")
    return 1 if report.refusals else 0
