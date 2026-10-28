"""Build a published dataset record from a series folder and its finished production job.

The builder first checks everything without writing export files: the job's inputs and
products, the tier, the expected results, the revision proof, the name scan
and the size budget. Only when no check refuses does it write the flat record
into a new or empty folder:

* R (reproduction): ``series.zip``, ``expected.zip`` and one ``raw_<run>.zip``
  per run, all stored without recompression;
* S (simulator): one ``simulator_<run>.zip`` per run with the run's stored
  coarse layer, mass table and protocol time series;

plus ``dataset_manifest.json``, ``README.md`` and ``CHECKSUMS.sha256``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tomllib

from .archives import Member, stored_size, write_stored_archive
from .checksums import write_checksums
from .environment import lock_summary
from .job_reader import (LAYER_FILE, LAYER_RECEIPT, FinishedJob, Input, JobError,
                         producer_is_this_machine)
from .layout import (ALIASES, EXPECTED_ARCHIVE, FILE_LIMIT, FORMAT, MANIFEST, PROVENANCE_FOLDER,
                     README, REPLAY_PIPELINES, SERIES_ARCHIVE, TIER_BUDGETS, raw_archive_name,
                     require_record_type, require_tier, task_path)
from .manifest import DEFAULT_RULE, LABEL_LEGEND, manifest_text, validate_manifest, validate_rule
from .proof import inputs_digest, load_proof, proof_problems
from .quick_check import quick_check_validity
from .readme import render_readme
from .scan import NameScanner, machine_names, refused
from .tiers import DEFAULT_TIERS_FILE, nesting_problems, read_tiers, tier_job_problems

PLAN_COPY = "plan/adopted_plan.toml"
PROVENANCE_FORMAT = "darsia-published-provenance-v1"
ALIASES_FORMAT = "darsia-published-dataset-aliases-v1"
MASS_FIELDS = (("total", "total_mass_g"), ("aqueous", "aqueous_mass_g"), ("gas", "gas_mass_g"))
EXPECTED_FOLDER = {"R": "expected", "S": "products"}


def simulator_archive_name(run: str) -> str:
    """The record file name of one run's simulator archive."""
    return "simulator_" + raw_archive_name(run)[len("raw_"):]


def run_archive_name(record: str, run: str) -> str:
    """The per-run archive of a record: raw images (R) or simulator products (S)."""
    return raw_archive_name(run) if record == "R" else simulator_archive_name(run)


@dataclass(frozen=True)
class BuildRequest:
    """What to build: the series, the job that made its results, the record and the tier.

    ``roots`` maps the job's input root aliases to folders on this machine
    (default: the series folder for ``series`` and the series' declared roots).
    ``published_plan`` is a clean revision of the job's execution plan; it and
    a ``published_commit`` other than the job's need a ``proof``, and an R
    record then needs ``proof_job``, the finished job of the run that proved the
    revision (its prepared tasks are published).
    ``tier=None`` preserves the complete finished job selection with the
    existing maximum record budget. Explicit legacy tiers keep their original
    subset-file checks and smaller budgets.
    ``simulator_resolution_mm`` is required for an S record; it has no default.
    ``simulator_geometry`` is an optional simulator geometry request (N18) for
    an S record; without it the record is built exactly as before.
    """

    series_root: Path
    job_root: Path
    record: str
    tier: str | None
    output: Path
    title: str = ""
    tiers_file: Path | None = None
    roots: dict | None = None
    published_plan: Path | None = None
    published_commit: str | None = None
    proof: Path | None = None
    proof_job: Path | None = None
    known_variation: Path | None = None
    peak_memory_bytes: int | None = None
    peak_memory_basis: str = ""
    expected_resolution_mm: float | None = 1.0
    simulator_resolution_mm: float | None = None
    simulator_geometry: Path | None = None
    mass_map_resolution_mm: float | None = None
    require_simulator_geometry: bool = False
    software_mode: str | None = None
    software_inventory: Path | None = None
    software_stage_inventory: Path | None = None
    retain_producer_metadata: bool = False
    software_folder: Path | None = None
    software_code_url: str = ""
    software_python: str = ""
    links: dict = field(default_factory=dict)
    rule: dict | None = None
    deny: tuple | None = None
    fingerprint: bool = True


@dataclass
class BuildReport:
    """What the builder found, refused and wrote."""

    refusals: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    hits: list = field(default_factory=list)
    exif: list = field(default_factory=list)
    files: list = field(default_factory=list)
    manifest: dict | None = None
    built: bool = False

    def refuse(self, message: str) -> None:
        """Record one reason the record cannot be built."""
        self.refusals.append(message)

    def text(self) -> str:
        """A plain-text build report: refusals, warnings, name findings and files."""
        if self.built:
            lines = ["Record built."]
        else:
            lines = ["Refused." if self.refusals else "Checked: nothing refuses the record."]
        if self.refusals:
            lines += ["", "Refusals:"] + [f"- {item}" for item in self.refusals]
        found = refused(self.hits)
        if found:
            lines += ["", f"Name scan, {len(found)} finding(s):"]
            lines += [f"- {hit.describe()}" for hit in found[:200]]
            if len(found) > 200:
                lines.append(f"- and {len(found) - 200} more")
        allowed = [hit for hit in self.hits if hit.allowed]
        if allowed:
            lines += ["", "Allowed names in validation records:"]
            lines += [f"- {hit.describe()}" for hit in allowed]
        if self.exif:
            lines += ["", "EXIF identity fields in raw images (reported, never edited):"]
            lines += [f"- {row['file']}: {row['field']} = {row['value']!r}" for row in self.exif[:50]]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"- {item}" for item in self.warnings]
        if self.files:
            lines += ["", "Files:"]
            lines += [f"- {row['name']}  {row['size_bytes']} B  {row['sha256']}" for row in self.files]
        return "\n".join(lines) + "\n"

    def as_dict(self) -> dict:
        """The report as a JSON-ready dictionary."""
        return {"built": self.built, "refusals": list(self.refusals), "warnings": list(self.warnings),
                "name_findings": [hit.as_dict() for hit in refused(self.hits)],
                "allowed_names": [hit.as_dict() for hit in self.hits if hit.allowed],
                "exif": list(self.exif), "files": list(self.files)}


@dataclass
class BuildPlan:
    """Everything a checked build writes: the manifest skeleton and every archive's members."""

    manifest: dict
    archives: dict  # record file name -> members, in writing order


def _sha(data: bytes) -> str:
    """The SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


def _series_document(root: Path) -> dict:
    """The series' ``series.toml``."""
    with (Path(root) / "series.toml").open("rb") as stream:
        return tomllib.load(stream)


def _check_output(request: BuildRequest, report: BuildReport) -> None:
    """Refuse an output folder that is not new or empty, or lies inside the series or job."""
    output = Path(request.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        report.refuse(f"The output folder must be new or empty: {output}")
    resolved = output.resolve()
    for other in (request.series_root, request.job_root):
        if resolved.is_relative_to(Path(other).resolve()):
            report.refuse("The output folder must lie outside the series folder and the job folder")
            break


def input_roots(request: BuildRequest, series_document: dict) -> dict[str, Path]:
    """Folders on this machine for the job's input root aliases.

    ``series`` is the series folder being published; the others are the
    series' declared ``[roots]``, overridden by the request's ``roots``.
    """
    from ..store_map import StoreMapError, manifest_roots

    roots = {}
    try:
        declared = manifest_roots(Path(request.series_root), series_document)
        roots.update({alias: Path(str(path)) for alias, path in declared.items()})
    except (StoreMapError, ValueError):
        pass
    roots.update({alias: Path(path) for alias, path in (request.roots or {}).items()})
    roots["series"] = Path(request.series_root)
    return roots


def job_frames(job: FinishedJob, inputs_by_run: dict, report: BuildReport, *,
               exact_acquisition: bool = False) -> tuple[dict, dict, dict]:
    """Per run: frames in time order with their raw image, other camera images, and all camera inputs.

    A frame's raw image is the camera input whose SHA-256 is the frame's
    acquisition identity. The run's other camera inputs (baseline and
    reference images) are its references. Staged RAW replay uses each frame's
    named P1 acquisition consumer, preserving distinct paths with equal bytes.
    """
    frames, references, cameras = {}, {}, {}
    for run in job.runs:
        by_sha = {}
        for item in inputs_by_run.get(run, ()):
            if item.camera_file:
                by_sha.setdefault(item.sha256, item)
        rows, used = [], set()
        for case in job.frames(run):
            try:
                result = job.frame_result(run, case)
            except (JobError, OSError, ValueError, KeyError) as error:
                report.refuse(f"Run {run}, {case}: no readable result ({error})")
                continue
            sha = ((result.get("bindings") or {}).get("source_sha256")
                   or (result.get("source_binding") or {}).get("source_sha256"))
            if exact_acquisition:
                from ..image_formats import image_family
                use = "frame." + case + ".native.acquisition"
                matches = [item for item in inputs_by_run.get(run, ()) if use in item.uses]
                if (len(matches) != 1 or not matches[0].camera_file or matches[0].role != "image"
                        or image_family(matches[0].relative_path) != "raw" or matches[0].sha256 != sha):
                    report.refuse(f"Run {run}, {case}: no unique original P1 RAW acquisition consumer")
                    continue
                raw = matches[0]
            else:
                raw = by_sha.get(sha)
            if raw is None:
                report.refuse(f"Run {run}, {case}: its raw image is not among the job's recorded inputs")
                continue
            used.add(raw.sha256)
            rows.append({
                "case_id": case, "frame_index": result.get("frame_index"),
                "elapsed_h": result.get("elapsed_h"),
                "raw": {"name": Path(raw.relative_path).name,
                        "path": f"{raw.root_alias}/{raw.relative_path}",
                        "sha256": raw.sha256, "size_bytes": raw.size_bytes},
                "masses_g": {key: result.get(source) for key, source in MASS_FIELDS},
            })
        frames[run] = rows
        references[run] = [
            {"name": Path(item.relative_path).name, "path": f"{item.root_alias}/{item.relative_path}",
             "sha256": item.sha256, "size_bytes": item.size_bytes, "uses": sorted(item.uses)[:8]}
            for item in sorted(by_sha.values(), key=lambda value: value.relative_path)
            if item.sha256 not in used]
        cameras[run] = sorted(by_sha.values(), key=lambda value: (value.root_alias, value.relative_path))
    return frames, references, cameras


def saved_result_frames(job: FinishedJob, report: BuildReport) -> dict:
    """Numerical exports use the exact saved results, without requiring RAW bytes.

    The acquisition SHA is taken from each original, content-verified result.
    This does not assert that the source photograph is present or reproducible.
    """
    frames = {}
    for run in job.runs:
        rows, previous = [], None
        cases = job.frames(run)
        if not cases or len(cases) != len(set(cases)):
            report.refuse(f"Run {run}: its saved selection is empty or repeats a case")
        for case in cases:
            try:
                result = job.frame_result(run, case)
                binding = result.get("bindings") or result.get("source_binding") or {}
                source_sha = binding.get("source_sha256")
                index, elapsed = result.get("frame_index"), result.get("elapsed_h")
                if re.fullmatch("[0-9a-f]{64}", str(source_sha)) is None:
                    raise ValueError("the saved result has no acquisition SHA-256")
                if (type(index) is not int or index < 0 or type(elapsed) not in (int, float)
                        or not math.isfinite(elapsed) or (previous is not None
                        and (index <= previous[0] or elapsed <= previous[1]))):
                    raise ValueError("the saved frame index/time order is invalid")
                rows.append(dict(case_id=case, frame_index=index, elapsed_h=elapsed,
                    source=dict(sha256=source_sha),
                    masses_g={key: result.get(source) for key, source in MASS_FIELDS}))
                previous = index, elapsed
            except (JobError, OSError, ValueError, KeyError, TypeError) as error:
                report.refuse(f"Run {run}, {case}: no usable saved result ({error})")
        frames[run] = rows
    return frames


def saved_selection(job: FinishedJob, frames: dict, report: BuildReport) -> dict:
    """Bind every selected frame to the original finished job's immutable records."""
    runs = {}
    for run in job.runs:
        cases = job.frames(run)
        if not cases or len(cases) != len(set(cases)) or job.completed[run].get("ordered_case_ids") != cases:
            report.refuse(f"Run {run}: prepared frame order differs from the original completion")
        if [row["case_id"] for row in frames.get(run, ())] != cases:
            report.refuse(f"Run {run}: export frames differ from the complete saved job selection")
        runs[run] = dict(ordered_case_ids=cases,
            configuration_sha256=job.tasks[run]["configuration"]["sha256"],
            completion_sha256=job.done[run]["completed"]["sha256"],
            products_index_sha256=job.done[run]["products_index"]["sha256"])
    return dict(kind="finished-job", runs=runs)


def record_budget(tier):
    """Keep the legacy policy when chosen; the saved-job route keeps its maximum."""
    return TIER_BUDGETS[tier] if tier is not None else max(TIER_BUDGETS.values())


def series_inputs(inputs_by_run: dict, report: BuildReport, *, what: str = "the job") -> dict:
    """The inputs under the series root, by relative path; refuse inputs outside the dataset.

    Camera images under another root go to the raw archives; any other input
    outside the series folder cannot be published.
    """
    files = {}
    for run, inputs in inputs_by_run.items():
        for item in inputs:
            if item.root_alias == "series":
                previous = files.get(item.relative_path)
                if previous is not None and previous.sha256 != item.sha256:
                    report.refuse(f"Two runs of {what} read different versions of {item.relative_path}")
                files[item.relative_path] = item
            elif not item.camera_file:
                report.refuse(f"A needed file of {what} lies outside the dataset folder: root "
                              f"{item.root_alias!r}, {item.relative_path} (publish it in the series "
                              "folder as a new revision)")
    return files


def check_tiers(request: BuildRequest, frames: dict, report: BuildReport) -> None:
    """Refuse a tier whose frames differ from the job's, and tiers that are not nested."""
    path = Path(request.tiers_file) if request.tiers_file else Path(request.series_root) / DEFAULT_TIERS_FILE
    if not path.is_file():
        report.refuse(f"No tier file: {path}")
        return
    try:
        tiers = read_tiers(path)
    except (ValueError, OSError) as error:
        report.refuse(f"Tier file unreadable: {error}")
        return
    for problem in nesting_problems(tiers):
        report.refuse("The tiers are not nested: " + problem)
    for problem in tier_job_problems(request.tier, tiers, frames):
        report.refuse(problem)


def expected_results(request: BuildRequest, job: FinishedJob, frames: dict,
                     report: BuildReport) -> tuple[dict, dict, dict]:
    """The mass tables and stored coarse layer of every frame of the job.

    Returns the manifest section per run, the archive members per run and the
    layer's grid. A missing layer, a layer at another resolution, or a missing
    mass refuses the build.
    """
    folder = EXPECTED_FOLDER[request.record]
    runs, members, grid = {}, {}, {}
    for run in job.runs:
        members[run] = []
        try:
            table = job.mass_table(run)
            source = job.product_source(table)
        except JobError as error:
            report.refuse(f"Run {run}: an expected result is missing: {error}")
            continue
        name = f"{folder}/{run}/mass_table.csv"
        members[run].append(Member(name, source, table.sha256, table.size_bytes))
        entry = {"mass_table": {"path": name, "sha256": table.sha256, "size_bytes": table.size_bytes},
                 "frames": {}}
        for frame in frames.get(run, ()):
            case = frame["case_id"]
            layer, receipt = job.layer(run, case), job.layer(run, case, LAYER_RECEIPT)
            if layer is None or receipt is None:
                report.refuse(f"Run {run}, {case}: the job has no stored coarse layer for this frame")
                continue
            try:
                source = job.product_source(layer)
                document = json.loads(job.product_bytes(receipt))
            except (JobError, OSError, ValueError) as error:
                report.refuse(f"Run {run}, {case}: an expected result is missing: {error}")
                continue
            coarse = document.get("coarse") or {}
            frame_grid = coarse.get("grid") or {}
            if request.expected_resolution_mm is not None and frame_grid.get("resolution_mm") != request.expected_resolution_mm:
                report.refuse(f"Run {run}, {case}: the job has no {request.expected_resolution_mm:g} mm "
                              f"layer (stored at {frame_grid.get('resolution_mm', 'native')})")
                continue
            if grid and grid.get("resolution_mm") != frame_grid.get("resolution_mm"):
                report.refuse(f"Run {run}, {case}: stored mass-map resolutions differ within the job")
                continue
            if any(value is None for value in frame["masses_g"].values()):
                report.refuse(f"Run {run}, {case}: an expected mass is missing")
            grid = grid or {key: frame_grid.get(key) for key in
                            ("format", "shape", "extent_m", "resolution_mm", "axes", "storage_precision")}
            name = f"{folder}/{run}/{case}/{LAYER_FILE}"
            members[run].append(Member(name, source, layer.sha256, layer.size_bytes))
            entry["frames"][case] = {"masses_g": frame["masses_g"],
                                     "layer": {"path": name, "sha256": layer.sha256,
                                               "size_bytes": layer.size_bytes,
                                               "arrays": coarse.get("arrays") or {}}}
        runs[run] = entry
    return runs, members, grid


def collect_plan_inputs(series_root: Path, plan_path: Path) -> tuple[dict, str, list]:
    """The inputs a (clean revision of an) execution plan reads per run, its SHA-256 and run configurations.

    The job layer's own input catalogue reads only metadata; no numerical
    call starts and nothing is written.
    """
    from ..pipeline.alias_job import read_alias_series
    from ..pipeline.job_inputs import InputCatalog, validate_catalog
    from ..pipeline.job_queue import descriptor
    from ..store_map import manifest_roots

    manifest = descriptor(Path(series_root) / "series.toml")
    plan = descriptor(Path(plan_path))
    value, configs, original = read_alias_series(Path(plan["path"]), expected_sha256=plan["sha256"],
                                                 series_manifest=manifest)
    roots = manifest_roots(Path(series_root), _series_document(Path(series_root)))
    result = {}
    for row, config in zip(value["runs"], configs):
        catalog = InputCatalog(roots, [*value.get("data_aliases", ()), *config.get("data_aliases", ())])
        catalog.add(original, "configuration", "run.execution_series")
        catalog.add(manifest, "configuration", "run.series_manifest")
        inputs = validate_catalog(catalog.collect(row["configuration"], config))
        result[row["run"]] = [
            Input(entry["reference"]["root_alias"], entry["reference"]["relative_path"],
                  entry["reference"]["sha256"], int(entry["reference"]["bytes"]),
                  entry["reference"]["role"], tuple(entry["uses"]))
            for entry in inputs["inputs"]]
    return result, plan["sha256"], configs


@dataclass
class Published:
    """What a record publishes: inputs per run, plan, commit, run configurations and the proof."""

    inputs_by_run: dict
    plan_sha256: str
    plan_path: Path | None
    commit: str
    configs: list
    proof: dict | None


def published_revision(request: BuildRequest, job: FinishedJob, roots: dict,
                       report: BuildReport) -> Published:
    """The published inputs, plan, commit, run configurations and proof (None when no revision).

    Without a published plan, the job's own inputs, plan and configurations are
    published. A published plan or commit other than the job's is a revision
    and needs a proof that binds both sides.
    """
    reference = job.plan_reference()
    commit = request.published_commit or job.job["commit"]
    plan_root = roots.get(reference["root_alias"])
    if plan_root is None:
        report.refuse("The original execution plan's input root has no bound folder")
    result = Published({run: job.inputs(run) for run in job.runs}, reference["sha256"],
                       None if plan_root is None else plan_root / reference["relative_path"], commit,
                       run_configurations(job, roots, report), None)
    if request.published_plan is not None:
        try:
            by_run, plan_sha, configs = collect_plan_inputs(Path(request.series_root),
                                                            Path(request.published_plan))
        except (OSError, ValueError, KeyError) as error:
            report.refuse(f"The published plan is not admissible: {error}")
            return result
        if set(by_run) != set(job.runs):
            report.refuse("The published plan runs other runs than the job")
        result = Published(by_run, plan_sha, Path(request.published_plan), commit, configs, None)
    if result.plan_sha256 == reference["sha256"] and commit == job.job["commit"]:
        return result
    if request.proof is None:
        report.refuse("A published revision has no proof against the revision that made the expected results")
        return result
    result.proof = load_proof(request.proof)
    files = series_inputs(result.inputs_by_run, BuildReport())
    digest = inputs_digest((item.root_alias, item.relative_path, item.sha256) for item in files.values())
    for problem in proof_problems(result.proof, expected_commit=job.job["commit"],
                                  expected_plan_sha256=reference["sha256"], published_commit=commit,
                                  published_plan_sha256=result.plan_sha256, published_inputs_sha256=digest):
        report.refuse("Proof: " + problem)
    return result


def published_tasks(request: BuildRequest, job: FinishedJob, published: Published,
                    report: BuildReport) -> dict[str, bytes]:
    """Each run's prepared task, exact bytes, by its path in the dataset (``plan/tasks/<run>.json``).

    The launcher replays these tasks with the production code. Without a
    revision they are the production job's own; a revision needs the finished
    job of the run that proved it (``proof_job``), whose commit, plan and inputs
    must be the published ones. A task must be a replayable alias task whose
    inputs are exactly the inputs the record publishes for its run.
    """
    from ..pipeline.job_inputs import validate_catalog
    from ..pipeline.job_queue import name

    source = job
    if published.plan_sha256 != job.plan_reference()["sha256"] or published.commit != job.job["commit"]:
        if request.proof_job is None:
            report.refuse("A published revision needs the finished job of the run that proved it "
                          "(its prepared tasks are published)")
            return {}
        try:
            source = FinishedJob(request.proof_job)
        except (JobError, OSError, KeyError, ValueError) as error:
            report.refuse(f"The proof job is not a finished job: {error}")
            return {}
        if source.job.get("commit") != published.commit or source.plan_reference()["sha256"] != published.plan_sha256:
            report.refuse("The proof job ran another commit or plan than the published revision")
            return {}
        if source.runs != job.runs:
            report.refuse("The proof job ran other runs than the production job")
            return {}
    files = {}
    for run in job.runs:
        path = Path(source.root) / "results" / name(run) / "task.json"
        try:
            raw = path.read_bytes()
            task = json.loads(raw)
            validate_catalog(task["input_catalog"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.refuse(f"Run {run}: its prepared task is unreadable ({error})")
            continue
        if task != source.tasks[run] or task.get("format") != "darsia-alias-task-v1" or task.get("run") != run:
            report.refuse(f"Run {run}: its prepared task is not the job's alias task")
            continue
        if task.get("pipeline") not in REPLAY_PIPELINES:
            report.refuse(f"Run {run}: the launcher cannot replay the pipeline {task.get('pipeline')!r}")
            continue
        inputs = {(entry["reference"]["root_alias"], entry["reference"]["relative_path"], entry["reference"]["sha256"])
                  for entry in task["input_catalog"]["inputs"]}
        if inputs != {(item.root_alias, item.relative_path, item.sha256)
                      for item in published.inputs_by_run.get(run, ())}:
            report.refuse(f"Run {run}: its prepared task reads other inputs than the record publishes")
            continue
        files[task_path(run)] = raw
    return files


def simulator_geometry_members(request: BuildRequest, job: FinishedJob, published: Published, roots: dict,
                               report: BuildReport) -> tuple[dict, dict | None]:
    """Each run's completed simulator geometry (N18) as archive members, and the manifest section.

    Only an S record with a simulator geometry request has them. The request
    names the series' geometry revision, the inner frame and sand surface, the
    fixture roles and polygons, and the depth points (default: each run
    configuration's own). A missing or unusable input refuses the record; it
    is never inferred.
    """
    if request.simulator_geometry is None:
        return {}, None
    from . import simulator_geometry as geometry

    if request.record != "S":
        report.refuse("A simulator geometry belongs to an S record")
        return {}, None
    try:
        raw = Path(request.simulator_geometry).read_bytes()
        wanted = geometry.load_request(request.simulator_geometry)
        revision = json.loads(geometry.series_file(roots["series"], wanted["geometry"]).read_text(encoding="utf-8-sig"))
        configuration = revision.get("configuration") or revision
        width, height = float(configuration["width_m"]), float(configuration["height_m"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.refuse(f"The simulator geometry request is not usable: {error}")
        return {}, None
    roles = tuple(wanted.get("fixture_roles") or geometry.DEFAULT_FIXTURE_ROLES)
    section = {"format": geometry.GEOMETRY_FORMAT, "request_sha256": _sha(raw),
               "resolution_mm": request.simulator_resolution_mm, "fixture_roles": list(roles), "runs": {}}
    members = {}
    for run, config in zip(job.runs, published.configs):
        own = ((wanted.get("runs") or {}).get(run) or {}).get("fixtures_m")
        fixtures = own if own is not None else wanted.get("fixtures_m")
        try:
            facies, by_facies, table, source = geometry.run_geometry(revision, run)
            points = geometry.depth_points_for(wanted, roots["series"], config)
            npz, text, counts, warnings = geometry.export_geometry(
                facies, by_facies, table, width_m=width, height_m=height, depth_points=points,
                resolution_mm=request.simulator_resolution_mm, inner_frame=wanted["inner_frame"],
                sand_surface=wanted["sand_surface"], fixture_polygons_m=fixtures, fixture_roles=roles,
                permeability_unit=wanted.get("permeability_unit"),
                depth_source="request" if wanted.get("depth_points") is not None else "run configuration",
                source=source)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report.refuse(f"Run {run}: the simulator geometry could not be made ({error})")
            continue
        report.warnings.extend(f"Run {run}: {line}" for line in warnings)
        base = f"{EXPECTED_FOLDER['S']}/{run}/geometry/{geometry.FILE_STEM}"
        members[run] = [Member(base + ".npz", data=npz), Member(base + ".json", data=text)]
        section["runs"][run] = {"npz": {"path": base + ".npz", "sha256": _sha(npz), "size_bytes": len(npz)},
                                "json": {"path": base + ".json", "sha256": _sha(text), "size_bytes": len(text)},
                                "counts": counts}
    return members, section


def run_configurations(job: FinishedJob, roots: dict, report: BuildReport) -> list[dict]:
    """The job's run configurations, read from the series folder, in run order."""
    configs = []
    for run in job.runs:
        reference = job.tasks[run]["configuration"]
        path = roots.get(reference["root_alias"])
        try:
            raw = (Path(path) / reference["relative_path"]).read_bytes()
            if _sha(raw) != reference["sha256"]:
                raise ValueError("changed since the job read it")
            configs.append(json.loads(raw))
        except (OSError, ValueError, TypeError) as error:
            report.refuse(f"Run {run}: its run configuration is unreadable ({error})")
    return configs


def reference_machine(request: BuildRequest, job: FinishedJob, report: BuildReport) -> tuple[dict, dict | None]:
    """The reference machine's measured facts, and its environment when it is this machine."""
    measured = [job.measured(run) for run in job.runs]

    def one(key):
        """The runs' common value of ``key``; a warning when the runs differ."""
        values = sorted({row[key] for row in measured if row.get(key) is not None}, key=str)
        if len(values) > 1:
            report.warnings.append(f"Runs differ in {key}: {', '.join(map(str, values))}")
        return values[0] if values else None

    seconds = sorted(row["seconds_per_frame"] for row in measured if row.get("seconds_per_frame"))
    output = sorted(sum(product.size_bytes for product in job.products[run].values())
                    // max(1, len(job.frames(run))) for run in job.runs)
    machine = {"label": "reference machine", "backend": one("backend"), "precision": one("precision"),
               "threads": one("threads"), "seconds_per_frame": seconds[len(seconds) // 2] if seconds else None,
               "seconds_per_frame_basis": "median over the runs of wall time per computed frame",
               "output_bytes_per_frame": output[len(output) // 2] if output else None,
               "peak_memory_bytes": request.peak_memory_bytes,
               "peak_memory_basis": request.peak_memory_basis or (
                   "not recorded" if request.peak_memory_bytes is None else "measured on the reference machine")}
    if request.peak_memory_bytes is None:
        report.warnings.append("No measured peak memory per worker: the launcher will ask for a worker count")
    hosts = [item.text for item in machine_names() if item.source == "build machine host name"]
    if not producer_is_this_machine(job, hosts):
        report.warnings.append("The build does not run on the producing machine: its operating system, "
                               "CPU and installed environment are not recorded")
        return machine, None
    from .environment import cpu_model, distribution_fingerprint, instruction_sets, platform_facts

    machine["platform"] = platform_facts()
    machine["cpu"] = {"model": cpu_model(), "instruction_sets": instruction_sets()["summary"]}
    return machine, (distribution_fingerprint() if request.fingerprint else None)


def lock_facts(job: FinishedJob) -> dict:
    """The job's environment lock: path and SHA-256, with its pins when the checkout has it."""
    lock = dict(job.job.get("environment_lock") or {})
    from ..series_paths import darsia_source_checkout

    repository = darsia_source_checkout()
    if repository is None or not lock.get("path"):
        return lock
    try:
        raw = subprocess.check_output(["git", "-C", str(repository), "show",
                                       f"{job.job['commit']}:{lock['path']}"], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return lock
    if _sha(raw) != lock.get("sha256"):
        return lock
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "lock.txt"
        path.write_bytes(raw)
        summary = lock_summary(path)
    return {**lock, "pins": summary["pins"], "python": summary["python"], "hashes": summary["hashes"]}


def provenance_records(job: FinishedJob, machine: dict, environment: dict | None) -> dict[str, bytes]:
    """One neutral provenance record per run: code, environment, machine facts, no names."""
    files = {}
    for run in job.runs:
        measured = job.measured(run)
        document = {
            "format": PROVENANCE_FORMAT, "run": run, "machine": "reference machine",
            "code": {"commit": job.job["commit"]},
            "environment": {"fingerprint_sha256": (environment or {}).get("sha256"),
                            "lock_sha256": (job.job.get("environment_lock") or {}).get("sha256")},
            "platform": machine.get("platform", "not recorded"), "cpu": machine.get("cpu", "not recorded"),
            "backend": measured["backend"], "precision": measured["precision"],
            "threads": measured["threads"], "frames": measured["frames"],
            "seconds_per_frame": measured["seconds_per_frame"],
            "peak_memory_bytes": machine["peak_memory_bytes"], "pipeline": measured["pipeline"],
        }
        files[f"{PROVENANCE_FOLDER}/{run}.json"] = (json.dumps(document, indent=1) + "\n").encode("utf-8")
    return files


def alias_bindings(camera_roots) -> bytes:
    """``dataset_aliases.toml``: each input root of the dataset, relative to the dataset folder."""
    lines = ["# Where each input root of this dataset lies, relative to the dataset folder.",
             "# The raw archives unpack below the folder named after their root.",
             "schema_version = 1", f'format = "{ALIASES_FORMAT}"', "", "[roots]", 'series = "."']
    lines += [f'{alias} = "{alias}"' for alias in sorted(set(camera_roots)) if alias != "series"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def name_scan(request: BuildRequest, scanner: NameScanner, archives: dict, roots: dict,
              report: BuildReport) -> None:
    """Scan every member and record file name; read series files and check they are unchanged."""
    for archive, members in archives.items():
        report.hits.extend(scanner.path(archive, file=archive))
        for member in members:
            report.hits.extend(scanner.path(member.name, file=archive, where="member name"))
            label = f"{archive}:{member.name}"
            if member.data is not None or member.generate is not None:
                data = member.data if member.data is not None else b"".join(member.chunks())
                if member.sha256 is not None and (_sha(data) != member.sha256 or len(data) != member.size_bytes):
                    report.refuse("Generated export changed since preflight: " + member.name)
                report.hits.extend(scanner.content(data, file=label, suffix=Path(member.name).suffix))
                continue
            if archive.startswith("raw_"):
                _hits, fields = scanner.raw_image(member.source, file=label)
                report.exif.extend(fields)
                continue
            try:
                data = Path(member.source).read_bytes()
            except OSError as error:
                report.refuse(f"Unreadable: {member.name} ({error})")
                continue
            if member.sha256 is not None and (_sha(data) != member.sha256 or len(data) != member.size_bytes):
                report.refuse(f"Changed since the job read or wrote it: {member.name} "
                              "(publish a clean revision with a proof)")
            report.hits.extend(scanner.content(data, file=label, suffix=Path(member.name).suffix))


def check_build(request: BuildRequest) -> tuple[BuildPlan | None, BuildReport]:
    """Check without writing export files; return a plan when nothing refuses.

    Included Git objects are checked offline in a disposable local repository.
    """
    report = BuildReport()
    try:
        require_record_type(request.record)
        if request.tier is not None:
            require_tier(request.tier)
    except ValueError as error:
        report.refuse(str(error))
        return None, report
    if request.record == "S" and request.simulator_resolution_mm is None:
        report.refuse("A simulator record needs an explicit simulator resolution; there is no default")
    if (request.record == "S" and (request.tier is None or request.require_simulator_geometry
                                  or request.mass_map_resolution_mm is not None)
            and request.simulator_geometry is None):
        report.refuse("Choose a simulator geometry request binding the inner frame, sand, fixtures and measured depth")
    _check_output(request, report)
    try:
        job = FinishedJob(request.job_root)
        series_document = _series_document(Path(request.series_root))
    except (JobError, OSError, KeyError, ValueError) as error:
        report.refuse(f"Not a finished production job of this series: {error}")
        return None, report
    key = lambda value: str(value).casefold().replace("_", "").replace("-", "")
    if key(series_document.get("series_id", "")) != key(job.series_id):
        report.refuse("The job belongs to another series")
    try:
        from .settings import load_deny_list

        deny = request.deny if request.deny is not None else load_deny_list(
            forbidden_roots=(Path(request.series_root),))
    except (ValueError, OSError) as error:
        report.refuse(f"Publication settings: {error}")
        deny = ()
    roots = input_roots(request, series_document)
    published = published_revision(request, job, roots, report)
    staged = None
    if request.record == "R" and any(task.get("pipeline") == "n1" for task in job.tasks.values()):
        try:
            from .staged_package import prepare_staged
            staged = prepare_staged(request, job, published, roots, report)
        except (ValueError, OSError, KeyError, TypeError) as error:
            report.refuse("Full staged RAW reproduction: " + str(error))
            return None, report
    numerical = request.record == "S" and request.tier is None
    if numerical:
        frames, references, cameras = saved_result_frames(job, report), {}, {}
    else:
        frames, references, cameras = job_frames(job, staged.inputs_by_run if staged else published.inputs_by_run,
                                                report, exact_acquisition=staged is not None)
    if staged:
        for run in job.runs:
            cameras[run] = list({(item.root_alias, item.relative_path): item
                for item in staged.inputs_by_run[run] if item.camera_file}.values())
            selected_paths = {row["raw"]["path"] for row in frames[run]}
            references[run] = [dict(name=Path(item.relative_path).name,
                path=f"{item.root_alias}/{item.relative_path}", sha256=item.sha256,
                size_bytes=item.size_bytes, uses=list(item.uses))
                for item in cameras[run] if f"{item.root_alias}/{item.relative_path}" not in selected_paths]
    selection = saved_selection(job, frames, report) if request.tier is None or staged else None
    for alias in sorted({item.root_alias for items in cameras.values() for item in items}):
        if alias not in roots:
            report.refuse(f"The input root {alias!r} has no folder on this machine")
    series_files = {} if numerical or staged else series_inputs(published.inputs_by_run, report)
    tasks = staged.tasks if staged else (published_tasks(request, job, published, report) if request.record == "R" else {})
    geometry_members, geometry_section = simulator_geometry_members(request, job, published, roots, report)
    if request.tier is not None:
        check_tiers(request, frames, report)
    commit, proof = published.commit, published.proof
    pipeline = job.tasks[job.runs[0]].get("pipeline") or "native-phase-n1"
    if numerical:
        quick = {"valid": False, "reasons": ["Numerical-model products do not replay the analysis"]}
    elif staged:
        quick = {"valid": False, "reasons": ["Coupled RAW reproduction requires the complete sealed selection"]}
    elif pipeline not in REPLAY_PIPELINES:
        quick = {"valid": False, "reasons": ["The complete producing pipeline has no published replay route"]}
        if request.record == "R":
            report.refuse("Full reproduction needs the original phase-one job and RAW-input catalogue, "
                          "plus the bound staged replay route; the phase-two N1 catalogue alone is incomplete")
    else:
        quick = (quick_check_validity(pipeline, published.configs) if len(published.configs) == len(job.runs)
                 else {"valid": False, "reasons": ["A run configuration is unreadable"]})
    expected, expected_members, grid = expected_results(request, job, frames, report)
    if (request.record == "S" and request.tier is not None and request.mass_map_resolution_mm is None
            and grid and request.simulator_resolution_mm is not None
            and request.simulator_resolution_mm != grid.get("resolution_mm")):
        report.refuse(f"The simulator resolution {request.simulator_resolution_mm:g} mm is not the stored "
                      f"layer's {grid.get('resolution_mm'):g} mm; a re-export of stored native data at "
                      "another resolution is not available yet")
    map_members, map_section, software, software_section = {}, None, [], None
    if request.mass_map_resolution_mm is not None:
        try:
            from ..storage_coarsening import _target
            target = _target(request.mass_map_resolution_mm)
            source_resolution = grid.get("resolution_mm")
            if source_resolution is None or target < source_resolution:
                raise ValueError("Export maps must use the stored resolution or a coarser one")
            if target == source_resolution:
                map_section = dict(resolution_mm=target, source="original stored layer", runs=expected)
            else:
                from .export_mass_maps import mass_map_members
                map_members, map_section = mass_map_members(job, target, EXPECTED_FOLDER[request.record])
        except (JobError, ValueError, OSError, KeyError, TypeError) as error:
            report.refuse("Export mass maps: " + str(error))
    try:
        if staged:
            from .staged_software import staged_software_members
            from .staged_package import attach_staged_software
            software, stage_software = staged_software_members(request, staged.stage_jobs)
            attach_staged_software(staged, stage_software, software)
        else:
            from .export_software import software_members
            software, software_section = software_members(request, job, published.commit)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.refuse("Reproduction software: " + str(error))
    machine, environment = reference_machine(request, job, report)
    plan_reference = job.plan_reference()
    plan_sha = published.plan_sha256
    try:
        if published.plan_path is None:
            raise ValueError("no bound folder for its original input root")
        plan_bytes = published.plan_path.read_bytes()
    except (OSError, ValueError) as error:
        report.refuse(f"The execution plan is unreadable: {error}")
        plan_bytes = b""
    try:
        rule = validate_rule(request.rule or DEFAULT_RULE)
    except ValueError as error:
        report.refuse(str(error))
        rule = DEFAULT_RULE
    variation = []
    if request.known_variation is not None:
        variation = json.loads(Path(request.known_variation).read_text(encoding="utf-8-sig"))
        if not isinstance(variation, list) or any(not isinstance(item, dict) for item in variation):
            report.refuse("Known variation must be a JSON list of entries")
            variation = []
    camera_roots = {item.root_alias for items in cameras.values() for item in items}
    manifest = {
        "format": FORMAT,
        # Keep the legacy budget token readable by the exact published code.
        # The new selection record, not this token, binds the saved frames.
        "identity": {"series_id": job.series_id, "record": request.record,
                     "tier": request.tier or max(TIER_BUDGETS, key=TIER_BUDGETS.__getitem__),
                     "title": request.title or str(series_document.get("name") or job.series_id),
                     "links": {key: (request.links or {}).get(key) for key in ("other_record", "code", "article")}},
        "code": {"expected_results_commit": job.job["commit"], "published_commit": commit,
                 "doi": (request.links or {}).get("code"),
                 "proof": None if proof is None else {
                     "sha256": _sha(json.dumps(proof, sort_keys=True).encode("utf-8")),
                     "frames": len(proof.get("frames") or ()), "verdict": proof.get("verdict"),
                     "machine": proof.get("machine")}},
        "environment": {"python": (environment or {}).get("python"), "lock": lock_facts(job),
                        "fingerprint": environment},
        "plan": {"path": PLAN_COPY, "sha256": plan_sha, "pipeline": pipeline,
                 "precision": job.job.get("precision"),
                 "storage_profile": job.job.get("storage_profile", "full"),
                 "job_reference": {key: plan_reference[key] for key in ("root_alias", "relative_path")},
                 "series_manifest": job.series_manifest_reference()["relative_path"],
                 "tasks": {run: {"path": task_path(run), "sha256": _sha(tasks[task_path(run)]),
                                 "size_bytes": len(tasks[task_path(run)])}
                           for run in job.runs if task_path(run) in tasks}},
        "roots": {"series": ".", **{alias: alias for alias in sorted(camera_roots) if alias != "series"}},
        "inputs": {"series": [{"path": relative, "sha256": item.sha256, "size_bytes": item.size_bytes,
                               "role": item.role} for relative, item in sorted(series_files.items())]},
        "runs": [{"run": run, "frames": [{key: frame[key] for key in ("case_id", "frame_index", "elapsed_h",
                                         "source" if numerical else "raw")}
                                         for frame in frames.get(run, ())],
                  "references": references.get(run, [])} for run in job.runs],
        "machine": machine,
        "quick_check_valid": bool(quick.get("valid")),
        "quick_check": quick,
        "reproducibility": {"same_machine": "bit-identical", "other_machines": rule},
        "known_variation": variation,
        "units": {"masses": "g", "areas": "cm2", "time": "h since the injection start",
                  "label_legend": dict(LABEL_LEGEND)},
    }
    if selection is not None:
        manifest["selection"] = selection
    if staged:
        manifest["staged_reproduction"] = staged.section
    layer = {**grid, "file": LAYER_FILE}
    if map_section is not None:
        manifest["export_mass_maps"] = map_section
    if software_section is not None:
        manifest["software"] = software_section
    if request.record == "R":
        manifest["expected"] = {"layer": layer, "runs": expected}
    else:
        manifest["simulator"] = {"layer": layer, "resolution_mm": request.simulator_resolution_mm,
                                 "gate": "the products are the in-run stored layer itself, not a re-export",
                                 "runs": expected}
        if geometry_section is not None:
            manifest["simulator"]["geometry"] = geometry_section
    archives = {}
    if request.record == "R":
        generated = {PLAN_COPY: plan_bytes, ALIASES: alias_bindings(camera_roots),
                     **provenance_records(job, machine, environment), **tasks}
        archives[SERIES_ARCHIVE] = (
            [Member(relative, roots["series"] / relative, item.sha256, item.size_bytes)
             for relative, item in sorted(series_files.items())]
            + [Member(name, data=data) for name, data in sorted(generated.items())] + software)
        if staged:
            archives[SERIES_ARCHIVE].extend(staged.members)
        archives[EXPECTED_ARCHIVE] = [member for run in job.runs
                                     for member in [*expected_members.get(run, ()), *map_members.get(run, ())]]
        for run in job.runs:
            archives[raw_archive_name(run)] = [
                Member(f"{item.root_alias}/{item.relative_path}", roots[item.root_alias] / item.relative_path,
                       item.sha256, item.size_bytes)
                for item in cameras.get(run, ()) if item.root_alias in roots]
    else:
        folder = EXPECTED_FOLDER["S"]
        for run in job.runs:
            protocols = {(f"{item.root_alias}/{item.relative_path}" if numerical else item.relative_path): item
                         for item in published.inputs_by_run.get(run, ())
                         if item.role == "protocol" and (numerical or item.root_alias == "series")}
            for item in protocols.values():
                if item.root_alias not in roots:
                    report.refuse(f"Run {run}: the protocol root {item.root_alias!r} has no bound folder")
            archives[simulator_archive_name(run)] = list(expected_members.get(run, ())) + [
                Member(f"{folder}/{run}/timeseries/{relative}", roots[item.root_alias] / item.relative_path,
                       item.sha256, item.size_bytes) for relative, item in sorted(protocols.items())
                if item.root_alias in roots] + geometry_members.get(run, []) + map_members.get(run, [])
    scanner = NameScanner((*machine_names(), *job.receipt_names()), deny=deny)
    name_scan(request, scanner, archives, roots, report)
    report.hits.extend(scanner.content(manifest_text(manifest).encode("utf-8"), file=MANIFEST, suffix=".json"))
    if staged:
        from .staged_package import provenance_findings
        provenance_findings(request, staged, report)
    check_size(request, archives, manifest, report)
    found = refused(report.hits)
    names = sum(hit.kind == "name" for hit in found)
    if names:
        report.refuse(f"The name scan found {names} name(s) of people, machines or users")
    if len(found) - names:
        report.refuse(f"The name scan found {len(found) - names} absolute path(s) without an alias; "
                      "published paths must be relative to the dataset")
    if report.refusals:
        return None, report
    return BuildPlan(manifest, archives), report


def check_size(request: BuildRequest, archives: dict, manifest: dict, report: BuildReport) -> None:
    """Refuse a record above its tier's budget or with 100 files or more."""
    files = 3 + len(archives)
    total = sum(stored_size(members) for members in archives.values())
    total += 4 * len(manifest_text(manifest).encode("utf-8")) + 64 * 1024
    budget = record_budget(request.tier)
    if files >= FILE_LIMIT:
        report.refuse(f"The record would have {files} files; it must have fewer than {FILE_LIMIT}")
    if total > budget:
        policy = request.tier or "saved-job export"
        report.refuse(f"The record would be {total / 1e9:.2f} GB, above the {policy} budget of "
                      f"{budget / 1e9:.0f} GB")


def build_record(request: BuildRequest) -> BuildReport:
    """Check, then write the record. Nothing is written when a check refuses.

    A failure while writing removes the files this build wrote, never others.
    """
    plan, report = check_build(request)
    if plan is None:
        return report
    request = replace(request, tier=plan.manifest["identity"]["tier"])
    output = Path(request.output)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        rows = []
        for name, members in plan.archives.items():
            rows.append(write_stored_archive(output / name, members))
            written.append(output / name)
        manifest = plan.manifest
        manifest["files"] = [{key: row[key] for key in ("name", "sha256", "size_bytes")} for row in rows]
        for row in manifest["runs"]:
            archive = run_archive_name(request.record, row["run"])
            row["archive"] = next((dict(item) for item in manifest["files"] if item["name"] == archive), None)
        archives_bytes = sum(row["size_bytes"] for row in rows)
        manifest["size"] = {"files": len(rows) + 3, "bytes": None,
                            "limit_bytes": record_budget(request.tier), "limit_files": FILE_LIMIT}
        # The size counts the manifest, the README and the checksum list themselves.
        listed = [row["name"] for row in rows] + [MANIFEST, README]
        checksums = sum(64 + 2 + len(name.encode("utf-8")) + 1 for name in listed)
        for _ in range(5):
            total = (archives_bytes + len(render_readme(manifest).encode("utf-8"))
                     + len(manifest_text(manifest).encode("utf-8")) + checksums)
            if manifest["size"]["bytes"] == total:
                break
            manifest["size"]["bytes"] = total
        validate_manifest(manifest)
        for name, text in ((MANIFEST, manifest_text(manifest)), (README, render_readme(manifest))):
            with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
                written.append(output / name)
                stream.write(text)
        digests = {row["name"]: row["sha256"] for row in rows}
        digests[MANIFEST] = _sha((output / MANIFEST).read_bytes())
        digests[README] = _sha((output / README).read_bytes())
        written.append(write_checksums(output, digests))
        report.files = [{"name": name, "sha256": digest, "size_bytes": (output / name).stat().st_size}
                        for name, digest in sorted(digests.items())]
        report.manifest = manifest
        report.built = True
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return report
