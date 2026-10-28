"""Create RAW review assets after an existing production job has finished.

The scientific worker and the RAW review writer are unchanged. The Jobs page
observes newly completed jobs while it is open; a saved request can also be run
explicitly from the command line when the GUI was closed. A failed or partial
output is preserved and never retried automatically.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import tomllib

REQUEST_FORMAT = "darsia-raw-review-postrun-request-v1"
COMPLETION_FORMAT = "darsia-raw-review-postrun-completed-v1"


class ReviewInputsMissing(ValueError):
    """The job has no explicitly attached original RAW preparation inputs."""


def _key(path):
    """Normalize a recorded path for case-insensitive comparisons."""
    return ntpath.normcase(ntpath.normpath(str(path)))


def _same(left, right):
    """Compare saved values with the existing JSON equality rules."""
    from .pipeline.staged_data import _same_json
    return _same_json(left, right)


def _pin(path):
    """The exact current file; hashing never changes it."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    after = path.stat()
    stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if stamp(before) != stamp(after):
        raise ValueError("A review input changed while being read: " + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def _bound(value):
    """Require a complete saved path, SHA-256 digest and byte count."""
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("A full saved path, SHA-256 and byte count are required")
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def _read(value):
    """Read size-bounded JSON only after verifying its pinned bytes."""
    from .pipeline.job_inputs import _json_bytes, MAX_METADATA_BYTES
    value = _bound(value)
    if _pin(value["path"]) != value:
        raise ValueError("A saved review source differs from its recorded bytes: " + value["path"])
    if value["size_bytes"] > MAX_METADATA_BYTES:
        raise ValueError("Review metadata exceeds its explicit size limit")
    raw = Path(value["path"]).read_bytes()
    if len(raw) != value["size_bytes"] or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("Review metadata changed during its read")
    return _json_bytes(raw)


def _write(path, value):
    """Create a new durable JSON file and return its byte descriptor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return _pin(path)


def request_path(job_root):
    """The explicitly attached request; this is not a producer-path search."""
    return Path(job_root) / "ui" / "raw_review_request.json"


def declared_preparations(series_root, *, expected_series, runs):
    """Read explicitly saved acquisition descriptors, never search for inputs.

    ``series.toml`` declares ``[raw_review.preparations.<run>]`` with path,
    sha256 and size_bytes. Relative paths resolve against that series folder.
    Payload hashes are verified again by ``build_request`` before any write.
    """
    from .pipeline.job_inputs import MAX_METADATA_BYTES
    from .series_paths import series_key
    if series_root is None:
        raise ReviewInputsMissing("Choose this job's registered series first")
    root = Path(series_root)
    path = root / "series.toml"
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Series metadata exceeds its size limit")
    value = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    selected = (value.get("series") or {}).get("id") or value.get("dataset") or value.get("series_id")
    if not selected or series_key(selected) != series_key(expected_series):
        raise ValueError("The saved acquisition preparations belong to another series")
    declarations = (value.get("raw_review") or {}).get("preparations") or {}
    if not isinstance(declarations, Mapping) or not runs:
        raise ReviewInputsMissing("Save this series' original RAW acquisition preparations in raw_review.preparations")
    missing = [run for run in runs if run not in declarations]
    if missing:
        raise ReviewInputsMissing("Original RAW acquisition preparations are not declared for: " + ", ".join(missing))
    result = {}
    for run in runs:
        descriptor = _bound(declarations[run])
        declared_path = Path(descriptor["path"])
        if not declared_path.is_absolute():
            declared_path = root / declared_path
        result[run] = dict(descriptor, path=str(declared_path))
    return result


def _completion_pins(job):
    """Pin the original closed task and final-publication records, without rewriting them."""
    from .pipeline.job_queue import name
    return {run: {"task": _pin(Path(job.root) / "results" / name(run) / "task.json"),
                  "done": _pin(Path(job.root) / "done" / (name(run) + ".json"))}
            for run in job.runs}


def _finished(job_root):
    """Require verified final publications and closed completion for every run."""
    from .jobs_metadata import JobSnapshotReader
    from .published_dataset.job_reader import FinishedJob
    root = Path(job_root)
    snapshot = JobSnapshotReader()(dict(root=str(root)))
    if (snapshot.get("complete") is not True or snapshot.get("errors")
            or not snapshot.get("runs")
            or any(row.get("state") != "Complete" for row in snapshot["runs"])):
        raise ValueError("RAW review needs a genuinely completed job with verified final publications")
    job = FinishedJob(root)
    if any(result.get("whole_run") is not True or result.get("worker_references_closed") is not True
           for result in job.completed.values()):
        raise ValueError("Every selected run must have its original whole-run closed completion")
    return job


class _Sources:
    """Read only named catalogue inputs and exact indexed producer products."""

    def __init__(self, job, run, stores):
        """Validate the input catalogue and initialize explicit source bindings."""
        from .pipeline.job_inputs import validate_catalog
        self.job, self.run, self.stores = job, run, stores
        self.task = job.tasks[run]
        self.catalog = validate_catalog(self.task["input_catalog"])
        self.locations, self.verified = {}, {}
        self.bindings, self.literal_sources, self.physical_bindings = {}, {}, {}
        self.protected_folders = set()
        self._protected_source_ids = set()

    def register(self, original, physical):
        """Bind a saved descriptor to one physical path without identity conflicts."""
        original = dict(original)
        if re.fullmatch(r"[0-9a-f]{64}", original.get("sha256", "")) is None:
            raise ValueError("A producer asset needs its original SHA-256")
        identity = (_key(original["path"]), original["sha256"])
        physical_key = _key(physical)
        physical_identity = self.physical_bindings.get(physical_key)
        if physical_identity is not None and (
                physical_identity["sha256"] != original["sha256"]
                or "size_bytes" in original and "size_bytes" in physical_identity
                and original["size_bytes"] != physical_identity["size_bytes"]):
            raise ValueError("One physical source has conflicting saved byte identities")
        self.physical_bindings[physical_key] = dict(physical_identity or {}, **original)
        previous = self.locations.get(identity)
        if previous is not None and _key(previous) != _key(physical):
            raise ValueError("A saved asset has conflicting explicit locations")
        saved = self.bindings.get(identity, {})
        if ("size_bytes" in saved and "size_bytes" in original
                and saved["size_bytes"] != original["size_bytes"]):
            raise ValueError("A saved asset has conflicting explicit byte counts")
        self.bindings[identity] = dict(saved, **original)
        self.locations[identity] = Path(physical)
        return dict(original, path=str(physical))

    def input(self, original, use, role):
        """Resolve one exact catalogue input with the required use and role."""
        from .pipeline.store_runtime import _available_input
        rows = [row for row in self.catalog["inputs"] if use in row["uses"]]
        if len(rows) != 1:
            raise ValueError("No unique original input for " + use)
        ref = rows[0]["reference"]
        if (ref["role"] != role or ref["sha256"] != original.get("sha256")
                or "size_bytes" in original and ref["bytes"] != original["size_bytes"]):
            raise ValueError("The saved input descriptor differs for " + use)
        path = _available_input(ref, self.stores, self.job.series_id, self.stores.roots(self.job.series_id))
        return self.register(dict(original, size_bytes=ref["bytes"]), path)

    def configuration(self):
        """Resolve and read the exact configuration recorded in the task catalogue."""
        rows = [row for row in self.catalog["inputs"] if "run.configuration" in row["uses"]]
        if len(rows) != 1 or rows[0]["reference"] != self.task["configuration"]:
            raise ValueError("The run configuration differs from its original catalogue")
        from .pipeline.store_runtime import _available_input
        ref = rows[0]["reference"]
        path = _available_input(ref, self.stores, self.job.series_id, self.stores.roots(self.job.series_id))
        bound = dict(path=str(path), sha256=ref["sha256"], size_bytes=ref["bytes"])
        self.register(bound, path)
        return bound, self.json(bound)

    def product(self, original):
        """Resolve one indexed published product matching the saved descriptor."""
        rows = [row for row in self.job.products[self.run].values()
                if _key(row.local_path) == _key(original["path"])
                and row.sha256 == original.get("sha256")
                and ("size_bytes" not in original or row.size_bytes == original["size_bytes"])]
        if len(rows) != 1:
            raise ValueError("No unique published product for " + original["path"])
        product = rows[0]
        if product.shared_copy is None and product.machine_id.casefold() != self.stores.machine_id.casefold():
            raise ValueError("RAW review needs this product on its original producing machine: "
                             + product.machine_id + " / " + product.product_id)
        path = self.job.product_source(product)
        return self.register(dict(original, size_bytes=product.size_bytes), path)

    def original(self, original, *, aliases=()):
        """Admit only a literal source or an alias explicitly in its saved configuration."""
        candidates = [row["physical"] for row in aliases
                      if _key(row["logical"]["path"]) == _key(original["path"])]
        if len(candidates) > 1:
            raise ValueError("Ambiguous saved source alias")
        physical = candidates[0] if candidates else original
        if (physical.get("sha256") != original.get("sha256")
                or "size_bytes" in original and physical.get("size_bytes") != original["size_bytes"]):
            raise ValueError("A saved source alias changes its byte identity")
        known = self.locations.get((_key(original["path"]), original["sha256"]))
        if known is not None:
            # A named catalogue binding has already resolved these exact bytes.
            return dict(original, path=str(known))
        return self.register(original, Path(physical["path"]))

    def path(self, value):
        """Return an admitted source path after checking its recorded byte identity."""
        identity = (_key(value["path"]), value["sha256"])
        path = self.locations.get(identity)
        if path is None:
            raise ValueError("Asset is outside the explicit post-run source bindings: " + value["path"])
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = self.verified.get(str(path))
        actual = cached[1] if cached is not None and cached[0] == stamp else _pin(path)
        if (actual["sha256"] != value["sha256"]
                or "size_bytes" in value and actual["size_bytes"] != value["size_bytes"]):
            raise ValueError("A post-run source asset changed: " + str(path))
        self.verified[str(path)] = (stamp, actual)
        if identity in self._protected_source_ids:
            self._protect_opened(path)
        return path

    def json(self, value):
        """Read bounded JSON from a verified source binding."""
        from .pipeline.job_inputs import _json_bytes, MAX_METADATA_BYTES
        path = self.path(value)
        if path.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("Review metadata exceeds its explicit size limit")
        return _json_bytes(path.read_bytes())

    def array(self, value):
        """Open a verified array read-only without loading pickled objects."""
        import numpy as np
        return np.load(self.path(value), allow_pickle=False, mmap_mode="r")

    def protect(self, bound):
        """Original producer/RAW data folders are never destinations for this output."""
        identity = (_key(bound["path"]), bound["sha256"])
        self._protected_source_ids.add(identity)
        physical = self.locations.get(identity)
        for path in (bound["path"], physical):
            if path is not None:
                self.protected_folders.add(_key(ntpath.dirname(ntpath.abspath(str(path)))))
        # Logical descriptors stay passive. Resolve a current-host locator only
        # after this reader has actually opened and SHA-verified its bytes.
        verified = self.verified.get(str(physical)) if physical is not None else None
        if verified is not None and verified[1]["sha256"] == bound["sha256"]:
            self._protect_opened(physical)

    def _protect_opened(self, path):
        """Called only after the exact physical/literal file was opened and hashed."""
        self.protected_folders.add(_key(Path(path).resolve(strict=True).parent))

    @staticmethod
    def _check_bytes(path, bound):
        """Rehash a source and require its original digest and byte count."""
        actual = _pin(path)
        if (actual["sha256"] != bound["sha256"]
                or "size_bytes" in bound and actual["size_bytes"] != bound["size_bytes"]):
            raise ValueError("An original consumed RAW review source changed: " + str(path))
        return actual

    def revalidate(self):
        """Rehash every consumed binding, including literal compositor paths; decode nothing."""
        checked = {}
        values = [(physical, self.bindings[identity]) for identity, physical in self.locations.items()]
        values += [(Path(bound["path"]), bound) for bound in self.literal_sources.values()]
        for path, bound in values:
            identity = (_key(path), bound["sha256"])
            actual = checked.get(identity)
            if actual is None:
                actual = checked[identity] = self._check_bytes(path, bound)
            elif "size_bytes" in bound and actual["size_bytes"] != bound["size_bytes"]:
                raise ValueError("An original consumed RAW review source changed its byte count")

    def full(self, full, *, input_prefix=None):
        """Admit and protect the exact centre and ring assets consumed by display."""
        centre = full["native_centre"]
        items = [(centre["log_linear_rgb"], ".centre.log"),
                 (centre["valid_mask"], ".centre.valid")]
        items += [(window["log_linear_rgb"], ".ring." + str(index))
                  for index, window in enumerate(full["ring_windows"])]
        for bound, use in items:
            if input_prefix is None:
                self.product(bound)
            else:
                self.input(bound, input_prefix + use, "image" if use != ".centre.valid" else "rig_model")
            self.protect(bound)
            # The unchanged RAW-detail composer reads these recorded literals.
            # Do not rewrite a scientific record to make a relocated asset fit.
            self.path(bound)  # The resolved named input/indexed producer product.
            literal = Path(bound["path"])
            self._check_bytes(literal, bound)  # What the unchanged compositor actually opens.
            self._protect_opened(literal)
            self.literal_sources[(_key(literal), bound["sha256"])] = dict(bound)


class _PhotoEntry(dict):
    """Keep per-frame support transient instead of holding every frame's raster."""
    def __init__(self, value, photo):
        """Retain frame metadata and a lazy photographic support loader."""
        super().__init__(value)
        self.photo = photo

    def __getitem__(self, key):
        """Materialize photographic support only when that entry is requested."""
        return self.photo() if key == "photo_support" else super().__getitem__(key)


def _photo_factory(sources, roles, native, shape, aliases=()):
    """Bind a lazy support projection to its original photo proof and transform."""
    from .native import support
    import numpy as np
    photo = roles.get("photo_sand_reference")
    proof = roles.get("photo_proof") or {}
    if (not isinstance(photo, Mapping) or proof.get("sha256") != photo.get("sha256")
            or proof.get("phase_labels_used") is not False):
        raise ValueError("Original source-bound photographic support is missing")
    sources.original(photo, aliases=aliases)
    h = np.asarray(roles.get("reference_to_current_h"), np.float64)
    parameters = native["support"]
    support._normal_h(h, parameters=parameters)
    def load():
        sand = support.source_photo_grid(sources.path(photo), photo["sha256"], parameters=parameters)
        return support.project_source_photo_safe_core(sand, h, shape_hw=tuple(shape),
            guard_px=parameters["guard_px"], enclosed_hole_policy=parameters["fill_policy"],
            parameters=parameters)
    return load


@dataclass
class PreparedRun:
    """Hold verified source bindings and inputs needed to render one review run."""
    run: str
    configuration: dict
    native_configuration: dict
    native_parameters: dict
    raw_binding: dict
    reference_manifest: dict
    reference_index: int
    reference_record: dict
    reference_photo: object
    frames: list
    sources: _Sources
    input_pins: list
    phase_configuration: dict | None = None


def _reference(job, run, config, sources, native, preparation, raw_binding, phase):
    """Use the preparation's own photo mask and source H, never current_h as a substitute."""
    import numpy as np
    from .native.raw_rig_source import validate_binding, validate_frame
    manifest = _read(preparation)
    if (manifest.get("format") != "darsia-raw-acquisition-preparation-v1"
            or manifest.get("run") != run):
        raise ValueError("Choose the original RAW acquisition preparation manifest for " + run)
    record = config["phase"]["reference"]
    matches = [(index, value) for index, value in enumerate(manifest.get("records", ()))
               if _same(value, record)]
    if len(matches) != 1:
        raise ValueError("The original preparation has no exact, unique saved baseline record")
    preparation_config = _read(manifest["configuration"])
    sources.original(manifest["configuration"])
    if (not _same(preparation_config.get("native"), config["native"])
            or not _same(preparation_config.get("phase"), {"configuration": config["phase"]["configuration"]})):
        raise ValueError("Baseline preparation uses another native or phase configuration")
    matches_specs = [item for item in preparation_config.get("acquisitions", ())
                     if _same(item.get("source"), record.get("source"))
                     and _same(item.get("phase_roles", {}).get("current_h"), record.get("current_h"))]
    if len(matches_specs) != 1:
        raise ValueError("The original baseline has no exact source/geometry preparation")
    spec = matches_specs[0]
    if (spec.get("native_task", {}).get("kind") != "raw_rig"
            or not _same(spec["native_task"]["processor"]["raw_binding"], raw_binding)):
        raise ValueError("Baseline RAW source binding differs from the completed run")
    aliases = preparation_config.get("data_aliases", ())
    geometry_bound = sources.original(spec["native_task"]["geometry"], aliases=aliases)
    geometry = sources.json(spec["native_task"]["geometry"])
    binding = sources.json(raw_binding)
    validate_binding(binding, native)
    validate_frame(geometry, binding, native)
    h = spec["phase_roles"].get("reference_to_current_h")
    if not np.array_equal(np.asarray(h, np.float64), np.asarray(geometry["reference_to_native_h"], np.float64)):
        raise ValueError("Baseline photographic H differs from its original RAW geometry")
    sources.input(record["native_row"], "phase.reference.native_row", "configuration")
    sources.protect(record["native_row"])
    row = sources.json(record["native_row"])
    sources.input(row["raw_producer_audit"], "phase.reference.raw_audit", "configuration")
    audit = sources.json(row["raw_producer_audit"])
    from .pipeline.phase_inputs import verify_raw_acquisition
    verify_raw_acquisition(record, row, audit, parameters=phase["parameters"]["inputs"])
    h_key = native["source"]["symbols"]["reference_to_frame_h_key"]
    if not np.array_equal(np.asarray(h, np.float64), np.asarray(audit[h_key], np.float64)):
        raise ValueError("Baseline photographic H differs from its original producer audit")
    sources.input(row["raw_acquisition"], "phase.reference.raw_acquisition", "image")
    sources.protect(row["raw_acquisition"])
    sources.register(audit["sources"]["camera_arw"], sources.path(row["raw_acquisition"]))
    sources.protect(audit["sources"]["camera_arw"])
    sources.full(record["full"], input_prefix="phase.reference.full")
    sources.original(preparation)
    shape = record["full"]["native_centre"]["native_shape_hw"]
    photo = _photo_factory(sources, spec["phase_roles"], native, shape, aliases=aliases)
    return matches[0][0], record, photo, [preparation, manifest["configuration"], geometry_bound]


def prepare_run(job, run, stores, preparation):
    """Bind one complete RAW run to its exact configuration and published native products."""
    import numpy as np
    sources = _Sources(job, run, stores)
    configuration, config = sources.configuration()
    if (config.get("run") != run or config.get("format") != "darsia-native-phase-n1-run-v1"
            or [item["case_id"] for item in config.get("frames", ())] != job.frames(run)
            or any(item.get("native_task", {}).get("kind") != "raw_rig" for item in config.get("frames", ()))):
        raise ValueError("The completed job is not the original ordered RAW native/phase run")
    native_bound = sources.input(config["native"]["configuration"], "native.configuration", "configuration")
    native = sources.json(config["native"]["configuration"])
    sources.input(config["phase"]["configuration"], "phase.configuration", "configuration")
    phase = sources.json(config["phase"]["configuration"])
    bindings = [item["native_task"]["processor"]["raw_binding"] for item in config["frames"]]
    if not bindings or any(not _same(item, bindings[0]) for item in bindings):
        raise ValueError("One run must keep its single original RAW binding")
    raw_binding = bindings[0]
    for spec in config["frames"]:
        sources.input(raw_binding, "frame." + spec["case_id"] + ".native.raw_binding", "configuration")
    reference_index, reference, reference_photo, originals = _reference(
        job, run, config, sources, native, preparation, raw_binding, phase)
    if _has_baseline_geometry(reference):
        geometry = reference["full"]["binding"]["geometry"]
        for bound in (geometry["frame_photo_top_keep"], phase["chain_assets"]["floor"]):
            sources.original(bound, aliases=config.get("data_aliases", ()))
            sources.protect(bound)
            sources.path(bound)
    frames = []
    for index, spec in enumerate(config["frames"]):
        case = spec["case_id"]
        result = job.frame_result(run, case)
        stage = (result.get("prediction") or {}).get("pipeline") or {}
        if (stage.get("complete") is not True or stage.get("format") != "darsia-series-frame-v1"
                or stage.get("case_id") != case or not _same(stage.get("source"), spec["source"])
                or stage.get("configuration_sha256") != configuration["sha256"]):
            raise ValueError("Published native stage differs from its source/configuration: " + case)
        native_record = stage.get("files", {}).get("native.json")
        if native_record is None:
            raise ValueError("The completed frame retains no native record: " + case)
        sources.product(native_record)
        sources.protect(native_record)
        document = sources.json(native_record)
        if not _same(document, stage.get("native")):
            raise ValueError("Published native file differs from its committed record: " + case)
        sources.product(document["audit"])
        audit = sources.json(document["audit"])
        sources.full(document["full"])
        use = "frame." + case
        sources.input(spec["native_task"]["geometry"], use + ".native_geometry", "rig_model")
        geometry = sources.json(spec["native_task"]["geometry"])
        sources.input(spec["geometry"], use + ".rig_geometry", "rig_model")
        sources.input(geometry["acquisition"], use + ".native.acquisition", "image")
        sources.protect(geometry["acquisition"])
        photo = spec["phase_roles"]["photo_sand_reference"]
        sources.input(photo, use + ".phase.photo_sand_reference", "rig_model")
        h_key = native["source"]["symbols"]["reference_to_frame_h_key"]
        if not np.array_equal(np.asarray(spec["phase_roles"]["reference_to_current_h"], np.float64),
                              np.asarray(audit[h_key], np.float64)):
            raise ValueError("Frame photographic H differs from its bound native audit: " + case)
        if (audit.get("sources", {}).get("camera_arw", {}).get("sha256") != geometry["acquisition"]["sha256"]
                or document.get("source_sha256") != spec["source"]["source_sha256"]):
            raise ValueError("Frame RAW source differs from its completed native record: " + case)
        camera = audit["sources"]["camera_arw"]
        sources.register(camera, sources.path(geometry["acquisition"]))
        sources.protect(camera)
        sources.product(stage["phase"]["classes"])
        values = dict(case_id=case, frame_index=index, elapsed_h=spec["elapsed_h"],
            native_record=native_record, native_geometry=spec["native_task"]["geometry"],
            rig_geometry=spec["geometry"], rig_classes=stage["phase"]["classes"])
        frames.append(_PhotoEntry(values, _photo_factory(sources, spec["phase_roles"], native,
            document["full"]["native_centre"]["native_shape_hw"])))
    # render_original passes this descriptor's path to the unchanged RAW renderer.
    sources.register(native_bound, native_bound["path"])
    return PreparedRun(run, configuration, native_bound, native, raw_binding,
        preparation, reference_index, reference, reference_photo, frames, sources,
        [configuration, native_bound, *originals], phase_configuration=config["phase"]["configuration"])


def _stores(bound, series_root):
    """Read the explicitly selected store map after verifying its saved bytes."""
    from .store_map import read_store_map
    _read_store_bytes = _bound(bound)
    if _pin(_read_store_bytes["path"]) != _read_store_bytes:
        raise ValueError("The explicitly selected local store map changed")
    return read_store_map(bound["path"], forbidden_roots=(Path(series_root), Path(__file__).resolve().parents[3]))


def _separate_output(output, prepared_runs):
    """Compare explicit paths only; never search or probe another machine's source folders."""
    candidate = _key(ntpath.abspath(str(output)))
    for prepared in prepared_runs:
        for original in prepared.sources.protected_folders:
            try:
                inside = ntpath.commonpath((candidate, original)) == original
            except ValueError:  # Different recorded drives cannot contain each other.
                inside = False
            if inside:
                raise ValueError("RAW review output must be separate from original RAW producer/data folders")


def build_request(job_root, stores_path, preparation_paths, output_root, *, series_root):
    """Prepare exact post-run DATA from explicitly selected original preparations."""
    job = _finished(job_root)
    preparations = {}
    for path in preparation_paths:
        if isinstance(path, Mapping):
            bound = _bound(path)
            if _pin(bound["path"]) != bound:
                raise ValueError("A declared RAW acquisition preparation changed")
        else:
            bound = _pin(path)
        value = _read(bound)
        run = value.get("run")
        if (value.get("format") != "darsia-raw-acquisition-preparation-v1"
                or run not in job.runs or run in preparations):
            raise ValueError("Choose exactly one original RAW acquisition preparation for each job run")
        preparations[run] = bound
    if set(preparations) != set(job.runs):
        raise ValueError("Original RAW acquisition preparations are missing for: "
                         + ", ".join(sorted(set(job.runs) - set(preparations))))
    store_bound = _pin(stores_path)
    stores = _stores(store_bound, series_root)
    configurations, prepared_runs = {}, []
    for run in job.runs:
        prepared = prepare_run(job, run, stores, preparations[run])
        configurations[run] = prepared.configuration
        prepared_runs.append(prepared)
    output = Path(output_root).resolve()
    if output.is_relative_to(Path(job_root).resolve()):
        raise ValueError("RAW review output must be separate from the original production job")
    _separate_output(output, prepared_runs)
    return dict(format=REQUEST_FORMAT, job_root=str(Path(job_root).resolve()),
        job=_pin(Path(job_root) / "job.json"), series_id=job.series_id,
        completion=_completion_pins(job),
        series_root=str(Path(series_root).resolve()), stores=store_bound,
        preparations=preparations, configurations=configurations,
        output_root=str(output), old_results_unchanged=True, adopted=False)


def attach_request(request):
    """Create one immutable UI request; never replace a request or producer receipt."""
    path = request_path(request["job_root"])
    if path.exists():
        existing = _read(_pin(path))
        if not _same(existing, request):
            raise ValueError("Different RAW review inputs are already attached; the saved request is preserved")
        return _pin(path)
    return _write(path, request)


def _request(value):
    """Verify request structure and unchanged original job completion records."""
    request = _read(value)
    required = {"format", "job_root", "job", "series_id", "completion", "series_root", "stores",
                "preparations", "configurations", "output_root", "old_results_unchanged", "adopted"}
    if (set(request) != required or request.get("format") != REQUEST_FORMAT
            or request.get("old_results_unchanged") is not True or request.get("adopted") is not False):
        raise ValueError("Explicit RAW post-run review request DATA is required")
    if _pin(Path(request["job_root"]) / "job.json") != request["job"]:
        raise ValueError("The completed production job changed after review preparation")
    for records in request["completion"].values():
        for recorded in records.values():
            if _pin(recorded["path"]) != recorded:
                raise ValueError("An original task or final publication changed after review preparation")
    return request


def _has_baseline_geometry(reference):
    """Report whether the saved reference retains its own geometry binding."""
    return "geometry" in (reference.get("full", {}).get("binding") or {})


def _baseline_paths(review_root, run):
    """Return the class archive and provenance output paths for one run."""
    folder = Path(review_root) / "baseline_classes"
    return folder / (run.casefold() + ".npz"), folder / (run.casefold() + ".json")


def _baseline_provenance(prepared, classes):
    """Describe display-only classes and their exact baseline source bindings."""
    from .raw_review_baseline_classes import CONTRACT
    return dict(format=CONTRACT, run=prepared.run, display_only=True,
        acquisition_manifest=prepared.reference_manifest, record_index=prepared.reference_index,
        native_row=prepared.reference_record["native_row"],
        phase_configuration=prepared.phase_configuration, classes=classes)


def _write_baseline_classes(prepared, review_root):
    """Write baseline classes and provenance when original geometry is bound."""
    from .raw_review_baseline_classes import write_baseline_rig_classes
    if not _has_baseline_geometry(prepared.reference_record):
        return None
    classes_path, provenance_path = _baseline_paths(review_root, prepared.run)
    phase = prepared.sources.json(prepared.phase_configuration)
    classes = write_baseline_rig_classes(classes_path, prepared.reference_record, phase,
        reader=prepared.sources)
    _write(provenance_path, _baseline_provenance(prepared, classes))
    prepared.sources.register(classes, classes_path)
    return classes


def _verify_baseline_classes(prepared, review_root, provenance):
    """Read exact saved output and compare its masks with the original geometry; write nothing."""
    import numpy as np
    from .raw_review_baseline_classes import baseline_rig_classes
    classes_path, provenance_path = _baseline_paths(review_root, prepared.run)
    if _key(_bound(provenance)["path"]) != _key(provenance_path):
        raise ValueError("Existing baseline class provenance is outside its expected output path")
    value = _read(provenance)
    classes = _bound(value.get("classes"))
    if _key(classes["path"]) != _key(classes_path) or _pin(classes_path) != classes:
        raise ValueError("Existing baseline classes differ from their pinned output path or bytes")
    if not _same(value, _baseline_provenance(prepared, classes)):
        raise ValueError("Existing baseline class provenance differs from its original sources")
    expected = baseline_rig_classes(prepared.reference_record,
        prepared.sources.json(prepared.phase_configuration), reader=prepared.sources)
    with np.load(classes_path, allow_pickle=False) as arrays:
        if (set(arrays.files) != set(expected)
                or any(arrays[key].dtype != expected[key].dtype
                       or not np.array_equal(arrays[key], expected[key]) for key in expected)):
            raise ValueError("Existing baseline class arrays differ from their original geometry")
    prepared.sources.register(classes, classes_path)
    return classes


def _change_status(prepared):
    """Explain whether the saved geometry classes support Change display."""
    if not _has_baseline_geometry(prepared.reference_record):
        return "Unavailable: the saved baseline has no own rig geometry; no later frame is substituted"
    if not all(frame.get("rig_classes") for frame in prepared.frames):
        return "Unavailable: current-frame rig classes are missing"
    return "Available from the baseline's own source geometry and saved current-frame classes"


def _verify_reuse(completed, request_bound, request, prepared_runs):
    """Verify existing review outputs against the original pinned request."""
    from .raw_photographic_review import (validate_raw_review_manifest, raw_asset_descriptor,
        raw_sources, _descriptor, ORIGINAL_MODE, RAW_DETAIL_MODE, CHANGE_MODE)
    from .phase_saved_assets import VerifiedAssetReader
    if (completed.get("format") != COMPLETION_FORMAT or completed.get("complete") is not True
            or not _same(completed.get("request"), request_bound)
            or not _same(completed.get("original_job"), request["job"])
            or completed.get("review_root") != request["output_root"]
            or completed.get("original_results_unchanged") is not True
            or completed.get("scientific_worker_changed") is not False
            or completed.get("adopted") is not False):
        raise ValueError("Existing RAW review completion belongs to another request")
    manifests = completed.get("manifests")
    runs = [prepared.run for prepared in prepared_runs]
    if (not isinstance(manifests, list) or not manifests or len(manifests) != len(runs)
            or [row.get("run") for row in manifests if isinstance(row, Mapping)] != runs
            or len({run.casefold() for run in runs}) != len(runs)):
        raise ValueError("Existing RAW review completion must name every original run exactly once in order")

    def same_fields(actual, expected, label):
        if (not isinstance(actual, Mapping)
                or any(not _same(actual.get(key), value) for key, value in expected.items())):
            raise ValueError("Existing RAW review differs from the original " + label)

    def source_fields(sources):
        return {key: sources[key] for key in ("acquisition", "audit", "profile_id")}

    reader = VerifiedAssetReader()
    baseline_outputs = completed.get("baseline_classes", {})
    expected_baselines = {prepared.run for prepared in prepared_runs
                          if _has_baseline_geometry(prepared.reference_record)}
    if not isinstance(baseline_outputs, Mapping) or set(baseline_outputs) != expected_baselines:
        raise ValueError("Existing RAW review must bind exactly its original baseline class outputs")
    for bound, prepared in zip(manifests, prepared_runs):
        expected_path = Path(request["output_root"]) / "native_frames" / prepared.run.casefold() / "manifest.json"
        if _key(_bound(bound)["path"]) != _key(expected_path):
            raise ValueError("Existing RAW review manifest is outside its expected output run folder")
        manifest = validate_raw_review_manifest(_read(bound), prepared.run)
        classes = (_verify_baseline_classes(prepared, request["output_root"], baseline_outputs[prepared.run])
                   if prepared.run in expected_baselines else None)
        modes = [ORIGINAL_MODE, *prepared.native_parameters["display_modes"]["modes"], RAW_DETAIL_MODE]
        if classes and all(frame.get("rig_classes") for frame in prepared.frames):
            modes.append(CHANGE_MODE)
        same_fields(manifest, dict(
            native_configuration=_descriptor(prepared.native_configuration, "Native configuration"),
            raw_binding=_descriptor(prepared.raw_binding, "RAW source binding"),
            modes=modes, frame_count=len(prepared.frames)), "configuration, binding or photographic modes")
        if "training_context" in manifest:
            raise ValueError("Existing RAW review has training context absent from this post-run request")
        row = prepared.sources.json(prepared.reference_record["native_row"])
        reference = raw_sources(prepared.reference_record["full"], row["raw_producer_audit"],
                                reader=prepared.sources)
        same_fields(manifest["reference"], dict(
            acquisition_manifest=_descriptor(prepared.reference_manifest, "Reference acquisition manifest"),
            record_index=prepared.reference_index, **source_fields(reference)), "baseline acquisition")
        geometry = {key: list(reference[key]) for key in
                    ("phase_shape_hw", "native_shape_hw", "native_origin_xy", "content_box_xywh")}
        geometry["canvas_wh"] = list(reversed(reference["phase_shape_hw"]))
        same_fields(manifest, geometry, "baseline canvas")
        if len(manifest["frames"]) != len(prepared.frames):
            raise ValueError("Existing RAW review has a different original frame count")
        for entry, frame in zip(manifest["frames"], prepared.frames):
            native = prepared.sources.json(frame["native_record"])
            sources = raw_sources(native["full"], native["audit"], reader=prepared.sources)
            expected = dict(case_id=str(frame["case_id"]), frame_index=int(frame["frame_index"]),
                elapsed_h=float(frame["elapsed_h"]),
                **{key: _descriptor(frame[key], key) for key in
                   ("native_record", "native_geometry", "rig_geometry")}, **source_fields(sources))
            if frame.get("rig_classes"):
                expected["rig_classes"] = _descriptor(frame["rig_classes"], "Rig classes")
            elif entry.get("rig_classes"):
                raise ValueError("Existing RAW review has unbound frame rig classes")
            same_fields(entry, expected, "ordered frame source/geometry")
            same_fields(manifest, {key: list(sources[key]) for key in
                ("phase_shape_hw", "native_shape_hw", "native_origin_xy", "content_box_xywh")}, "frame canvas")
        if classes:
            same_fields(manifest["reference"], {"rig_classes": _descriptor(classes, "Rig classes")},
                "baseline rig classes")
        elif manifest["reference"].get("rig_classes"):
            raise ValueError("Existing RAW review has baseline rig classes absent from this post-run request")
        folder = expected_path.parent
        for entry in (manifest["reference"], *manifest["frames"]):
            if not isinstance(entry.get("assets"), Mapping) or set(entry["assets"]) != set(modes):
                raise ValueError("Existing RAW review is missing a declared photographic mode")
            if not isinstance(entry.get("photo_support"), Mapping):
                raise ValueError("Existing RAW review is missing source-bound photo support")
            for asset in [*entry["assets"].values(), entry["photo_support"]]:
                reader.path(raw_asset_descriptor(folder, asset))
    return completed


def render_run(prepared, review_root):
    """Call the existing writer with its original records and explicit native configuration."""
    from .native.config import configure_parameters, reset_parameters
    from .raw_photographic_review import write_raw_review_root
    reference = _PhotoEntry(dict(acquisition_manifest=prepared.reference_manifest,
        record_index=prepared.reference_index), prepared.reference_photo)
    classes = _write_baseline_classes(prepared, review_root)
    if classes:
        reference["rig_classes"] = classes
    token = configure_parameters(prepared.native_parameters)
    try:
        return write_raw_review_root(review_root, run=prepared.run,
            native_configuration=prepared.native_configuration, raw_binding=prepared.raw_binding,
            reference=reference, frames=prepared.frames, reader=prepared.sources)
    finally:
        reset_parameters(token)


def run_request(bound):
    """Render once after genuine completion; keep every failure and partial file."""
    request = _request(bound)
    job = _finished(request["job_root"])
    if job.series_id != request["series_id"] or set(job.runs) != set(request["preparations"]):
        raise ValueError("The saved post-run request names different original runs")
    stores = _stores(request["stores"], request["series_root"])
    root = Path(request["output_root"])
    if root.resolve().is_relative_to(Path(request["job_root"]).resolve()):
        raise ValueError("RAW review output must be separate from its original production job")
    completed_path = root / "postrun_completed.json"
    reusable = None
    if root.exists():
        if completed_path.is_file():
            reusable = _read(_pin(completed_path))
        else:
            raise ValueError("A RAW review attempt already exists; partial files are preserved, with no automatic retry")
    # All runs are preflighted before the exclusive once-only output is created.
    # Masks and photographs remain transient and are not materialised here.
    prepared_runs = []
    for run in job.runs:
        prepared = prepare_run(job, run, stores, request["preparations"][run])
        if not _same(prepared.configuration, request["configurations"][run]):
            raise ValueError("The exact closed run configuration changed")
        prepared_runs.append(prepared)
    _separate_output(root, prepared_runs)
    if reusable is not None:
        for prepared in prepared_runs:
            prepared.sources.revalidate()
        return _verify_reuse(reusable, bound, request, prepared_runs)
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "postrun_started.json", dict(format="darsia-raw-review-postrun-started-v1",
        utc=datetime.now(timezone.utc).isoformat(), request=bound, pid=os.getpid(),
        no_original_result_writes=True, automatic_retry=False))
    manifests = []
    try:
        for prepared in prepared_runs:
            run = prepared.run
            path = render_run(prepared, root)
            manifests.append(dict(_pin(path), run=run))
        # Recheck the immutable request, original terminal publications and inputs.
        if not _same(_request(bound), request):
            raise ValueError("The review request changed during rendering")
        closed = _finished(request["job_root"])
        if _pin(request["stores"]["path"]) != request["stores"]:
            raise ValueError("The explicit local store map changed during rendering")
        for prepared in prepared_runs:
            prepared.sources.revalidate()
        for run in job.runs:
            _read(request["preparations"][run])
            _read(request["configurations"][run])
            for case in job.frames(run):
                closed.frame_result(run, case)
        if not _same(_request(bound), request):
            raise ValueError("The original final publications changed during source verification")
        result = dict(format=COMPLETION_FORMAT, complete=True,
            utc=datetime.now(timezone.utc).isoformat(), request=bound, manifests=manifests,
            review_root=str(root),
            original_job=request["job"], original_results_unchanged=True,
            scientific_worker_changed=False, adopted=False,
            baseline_classes={prepared.run: _pin(_baseline_paths(root, prepared.run)[1])
                for prepared in prepared_runs if _has_baseline_geometry(prepared.reference_record)},
            change_display="; ".join(prepared.run + ": " + _change_status(prepared)
                for prepared in prepared_runs),
            gui_closed="Run this same saved request explicitly from the command line when the GUI was closed")
        _write(completed_path, result)
        return result
    except Exception as error:
        _write(root / "postrun_error.json", dict(complete=False,
            utc=datetime.now(timezone.utc).isoformat(), request=bound,
            reason=str(error), completed_manifests=manifests,
            partials_preserved=True, automatic_retry=False))
        raise


def prepare_and_run(job_root, stores_path, preparation_paths, *, series_root, output_root):
    """The Jobs action: attach checked inputs, then execute their one post-run step."""
    bound = attach_request(build_request(job_root, stores_path, preparation_paths,
        output_root, series_root=series_root))
    return run_request(bound)


def default_output(job_root, series_root):
    """A separate, series-owned output for this exact original job."""
    from .series_paths import series_folder_path
    return series_folder_path(series_root, "phase_annotations") / "raw_reviews" / _pin(Path(job_root) / "job.json")["sha256"]


def run_attached(job_root):
    """Automatic terminal observer route; missing inputs require an explicit GUI choice."""
    path = request_path(job_root)
    if not path.is_file():
        raise ReviewInputsMissing("Choose the original RAW acquisition preparations with Prepare RAW review")
    return run_request(_pin(path))


def main(argv=None):
    """Explicitly prepare or run post-run review; no worker or scheduler is installed."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--job-root", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--stores", type=Path)
    parser.add_argument("--series-root", type=Path)
    parser.add_argument("--preparation", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.request is not None:
        if (args.request_sha256 is None or args.stores is not None or args.series_root is not None
                or args.preparation or args.output_root is not None):
            parser.error("A saved request needs only --request and --request-sha256")
        bound = _pin(args.request)
        if bound["sha256"] != args.request_sha256:
            raise ValueError("The selected RAW review request SHA differs")
        result = run_request(bound)
    else:
        if (args.stores is None or args.series_root is None
                or args.request_sha256 is not None):
            parser.error("A finished job needs --stores and --series-root")
        preparations = args.preparation
        if not preparations:
            job = _read(_pin(args.job_root / "job.json"))
            preparations = list(declared_preparations(args.series_root,
                expected_series=job["series_id"], runs=job["expected_runs"]).values())
        result = prepare_and_run(args.job_root, args.stores, preparations,
            series_root=args.series_root,
            output_root=args.output_root or default_output(args.job_root, args.series_root))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
