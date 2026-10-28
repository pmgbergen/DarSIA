"""Frozen I/O for the original preset comparison, without changing its science.

Preparation reads configuration/protocol metadata and declares the original Rig
archives. It never constructs a Rig, decodes its rasters, or runs a comparison.
The opt-in reader scope supplies authenticated bytes to the original parsers;
ordinary local readers retain their existing behavior outside that scope.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import io
import json
import ntpath
import os
from pathlib import Path
import stat as stat_module
import sys

FORMAT = "darsia-preset-wasserstein-inputs-v1"
_ACTIVE = ContextVar("preset-wasserstein-inputs", default=None)
_STAMP = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def current_scope():
    """Return only this explicitly entered preset reader scope."""
    return _ACTIVE.get()


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_location(path):
    """Resolve declared storage moves at IO, keeping caller contracts literal."""
    from .storage_paths import resolve_storage_path
    return resolve_storage_path(path)


def read_bytes(path):
    scope = current_scope()
    return _read_location(path).read_bytes() if scope is None else scope.read(path)


def read_text(path, encoding=None, errors=None):
    if current_scope() is None:
        return _read_location(path).read_text(encoding=encoding, errors=errors)
    with open_read(path, "r", encoding=encoding, errors=errors) as stream:
        return stream.read()


def open_read(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
    """A read-only stream; consumers parse the bytes that were authenticated."""
    if mode not in {"r", "rt", "rb"}:
        raise ValueError("The preset input seam only opens read-only streams")
    scope = current_scope()
    if scope is None:
        return open(_read_location(path), mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)
    buffer = io.BytesIO(scope.read(path))
    if "b" in mode:
        return buffer
    return io.TextIOWrapper(buffer, encoding=encoding,
                            errors=errors, newline=newline)


def source(path):
    """Keep parser options unchanged and substitute only a verified byte stream."""
    if hasattr(path, "read"):
        return path
    if current_scope() is None:
        located = _read_location(path)
        return located if isinstance(path, Path) else str(located)
    return io.BytesIO(current_scope().read(path))


def table_source(path, sheet=None):
    scope = current_scope()
    if scope is None:
        located = _read_location(path)
        return located if isinstance(path, Path) else str(located)
    with scope.role("protocol"):
        raw = scope.read(path, kind="table", sheet=sheet)
    return io.BytesIO(raw)


def exists(path, kind="exists"):
    scope = current_scope()
    return getattr(_read_location(path), kind)() if scope is None else scope.exists(path, kind)


def glob_paths(path, pattern):
    scope = current_scope()
    return _read_location(path).glob(pattern) if scope is None else scope.inventory(path, pattern)


def list_names(path):
    scope = current_scope()
    return os.listdir(_read_location(path)) if scope is None else [p.name for p in scope.inventory(path, None)]


def make_directory(path, *args, **kwargs):
    """Config construction is inert in this scope; writers use owned output seams."""
    if current_scope() is None:
        return Path(path).mkdir(*args, **kwargs)


def file_stat(path):
    scope = current_scope()
    return _read_location(path).stat() if scope is None else scope.stat(path)


def _guard(method):
    """An optional original config section cannot swallow a failed bound read."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            self.failed = str(error)
            raise
    return guarded


class _Scope:
    def __init__(self, *, roots=None, frozen=None, binding=None, catalog=None, path_aliases=()):
        self.roots = dict(roots or {})
        frozen = deepcopy(frozen)
        self.frozen, self.binding, self.catalog = frozen, binding, catalog
        self.working_directory = (str(Path.cwd().resolve()) if frozen is None
                                  else frozen["working_directory"])
        self.path_aliases = deepcopy(list(path_aliases) if frozen is None else frozen["path_aliases"])
        self.files, self.inventories, self.existence, self.config_roots = {}, {}, {}, {}
        self.consumed, self.failed, self._role = {}, None, "rig_model"
        if frozen is not None:
            self.expected = {_key(r["descriptor"]["path"]): r for r in frozen["files"]}
            self.expected_inventories = {(_key(r["path"]), r["pattern"]): r for r in frozen["inventories"]}
            self.expected_existence = {(_key(r["path"]), r["kind"]): r for r in frozen["existence"]}
            self.expected_roots = {tuple(map(_key, r["paths"])): r for r in frozen["config_roots"]}

    @contextmanager
    def enter(self):
        if current_scope() is not None:
            raise ValueError("Preset input scopes cannot be nested")
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    @contextmanager
    def role(self, role):
        previous, self._role = self._role, role
        try:
            yield
        finally:
            self._role = previous

    def fail(self, message):
        self.failed = message
        raise ValueError(message)

    def absolute(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = Path(self.working_directory) / path
        # Lexical normalization is independent of the worker's mounted roots.
        return Path(os.path.normpath(str(path)))

    def admit(self, path, *, directory=False):
        from .store_map import input_reference
        path = self.absolute(path)
        if directory and any(_key(path) == _key(root) for root in self.roots.values()):
            return path
        input_reference(str(path), self.roots, sha256="0" * 64, size_bytes=0, role=self._role)
        return path

    def _location(self, path):
        path = self.absolute(path)
        key = _key(path)
        if key not in self.expected:
            self.fail("Original preset reader requested an unbound input: " + str(path))
        record = self.expected[key]
        if self.binding is not None:
            row = self.binding.locations.get(key)
            if row is None or row.reference["sha256"] != record["descriptor"]["sha256"]:
                self.fail("Preset input lacks its exact admitted worker location: " + str(path))
            self.binding._input_digest(row)
            physical = row.path
        else:
            physical = self.catalog.physical(record["descriptor"])["path"]
        return path, Path(physical), record

    @_guard
    def descriptor(self, path, expected=None):
        path = self.absolute(path)
        key = _key(path)
        if self.frozen is not None:
            _, physical, record = self._location(path)
            actual = deepcopy(record["descriptor"])
            if self.binding is None:
                # Initial catalogue already checked root membership and size.
                self.catalog.add(actual, record["role"], record["uses"][0])
        else:
            physical = self.admit(path)
            before = physical.stat()
            if not stat_module.S_ISREG(before.st_mode):
                self.fail("Preset input is not a regular file: " + str(path))
            digest = hashlib.sha256()
            with physical.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = physical.stat()
            if any(getattr(before, k) != getattr(after, k) for k in _STAMP):
                self.fail("Preset source changed while being bound: " + str(path))
            actual = dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)
        if expected is not None and any(actual[k] != expected[k] for k in ("sha256", "size_bytes") if k in expected):
            self.fail("Original preset source descriptor changed: " + str(path))
        old = self.files.get(key)
        if old is not None and old["descriptor"] != actual:
            self.fail("Preset source changed between consumers: " + str(path))
        if old is None:
            self.files[key] = dict(descriptor=actual, role=self._role,
                                   uses=["preset-input:" + str(len(self.files))])
        self.consumed.setdefault(key, {"descriptor": actual, "reads": []})
        return deepcopy(actual)

    @_guard
    def read(self, path, *, kind="bytes", sheet=None):
        actual = self.descriptor(path)
        if self.frozen is None:
            physical = self.admit(path)
        else:
            _, physical, _ = self._location(path)
        before = physical.stat()
        with physical.open("rb") as stream:
            raw = stream.read(actual["size_bytes"] + 1)
        after = physical.stat()
        if (len(raw) != actual["size_bytes"] or hashlib.sha256(raw).hexdigest() != actual["sha256"]
                or any(getattr(before, k) != getattr(after, k) for k in _STAMP)):
            self.fail("Preset input changed at its actual parser read: " + str(path))
        if self.binding is not None:
            self._location(path)
        use = dict(kind=kind, sheet=sheet)
        reads = self.consumed[_key(self.absolute(path))]["reads"]
        if use not in reads:
            reads.append(use)
        return raw

    @_guard
    def stat(self, path):
        self.descriptor(path)
        physical = self.admit(path) if self.frozen is None else self._location(path)[1]
        return physical.stat()

    @_guard
    def exists(self, path, kind):
        path, key = self.absolute(path), (_key(self.absolute(path)), kind)
        if kind not in {"exists", "is_file", "is_dir"}:
            self.fail("Unsupported preset existence query")
        if self.frozen is None:
            try:
                self.admit(path, directory=True)
            except ValueError:
                self.fail("Preset existence query leaves registered roots: " + str(path))
            value = getattr(path, kind)()
            row = dict(path=str(path), kind=kind, exists=value)
        elif key in self.expected_existence:
            row = self.expected_existence[key]
        elif _key(path) in self.expected:
            row = dict(path=str(path), kind=kind, exists=kind != "is_dir")
        else:
            self.fail("Original preset reader requested an unfrozen existence query: " + str(path))
        self.existence[key] = deepcopy(row)
        return row["exists"]

    @_guard
    def inventory(self, path, pattern):
        path = self.absolute(path)
        key = (_key(path), pattern)
        if self.frozen is None:
            self.admit(path, directory=True)
            selected = list(path.iterdir()) if pattern is None else list(path.glob(pattern))
            # Preserve the reader's directory result before its own sorting.
            names = [str(p.relative_to(path)) for p in selected]
            row = dict(path=str(path), pattern=pattern, names=names)
        else:
            if key not in self.expected_inventories:
                self.fail("Original preset reader requested an unfrozen inventory: " + str(path))
            row = self.expected_inventories[key]
        self.inventories[key] = deepcopy(row)
        return [path / name for name in row["names"]]

    def config_path(self, value, configs):
        from darsia.presets.workflows.config.utils import _is_relative_path
        path = Path(value)
        if configs is None or not _is_relative_path(value):
            return self.absolute(path)
        paths = [configs] if isinstance(configs, (str, os.PathLike)) else list(configs)
        paths = [str(self.absolute(p)) for p in paths]
        key = tuple(map(_key, paths))
        if key in self.config_roots:
            row = self.config_roots[key]
        else:
            roots = set()
            for config in paths:
                for parent in (Path(config).parent, *Path(config).parent.parents):
                    marker = parent / "series.toml"
                    # This is the original ancestor-discovery stat, not content
                    # admission. Even outside a registered root, record absence
                    # rather than changing the original relative-path basis.
                    # A found manifest still needs ordinary root admission before
                    # any bytes are opened or added to the input catalogue.
                    if self.frozen is not None:
                        observation = self.expected_existence.get((_key(marker), "is_file"))
                        if observation is None:
                            self.fail("Preset ancestor-discovery observation was not frozen")
                        found = observation["exists"]
                    else:
                        try:
                            found = marker.is_file()
                        except OSError:
                            found = False  # Preserve original _series_root behavior.
                    self.existence[(_key(marker), "is_file")] = dict(
                        path=str(marker), kind="is_file", exists=found)
                    if found:
                        with self.role("configuration"):
                            self.descriptor(marker)
                        roots.add(str(parent))
                        break
            row = dict(paths=paths, root=roots.pop() if len(roots) == 1 else None)
            if self.frozen is not None and row != self.expected_roots.get(key):
                self.fail("Original preset relative-path basis changed")
        self.config_roots[key] = deepcopy(row)
        if row["root"] is not None:
            with self.role("configuration"):
                self.descriptor(Path(row["root"]) / "series.toml")
        return self.absolute(path) if row["root"] is None else Path(row["root"]) / path

    def check(self):
        if self.failed is not None:
            raise ValueError(self.failed)

    @_guard
    def verify_final(self):
        """Rehash consumed sources without retaining payloads at publication."""
        for record in self.consumed.values():
            bound = record["descriptor"]
            _, physical, _ = self._location(bound["path"])
            before = physical.stat()
            digest, size = hashlib.sha256(), 0
            with physical.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
            after = physical.stat()
            if (size != bound["size_bytes"] or digest.hexdigest() != bound["sha256"]
                    or any(getattr(before, k) != getattr(after, k) for k in _STAMP)):
                self.fail("Original preset source changed before publication: " + bound["path"])
            self._location(bound["path"])


def _rig_inputs(scope, folder):
    """Follow the actual Rig.load archive layout and native proof metadata."""
    folder = scope.absolute(folder)
    metadata = json.loads(scope.read(folder / "meta_data.json"))
    for name in ("baseline", "depth", "labels", "facies", "porosity", "permeability", "image_porosity"):
        scope.descriptor(folder / (name + ".npz"))
    optional = folder / "shape_corrected_baseline.npz"
    if scope.exists(optional, "exists"):
        scope.descriptor(optional)
    shape = scope.inventory(folder, "shape_correction_*.npz")
    color = scope.inventory(folder, "color_correction_*.npz")
    corrections = shape + color if shape or color else scope.inventory(folder, "correction_*.npz")
    for path in sorted(corrections):
        scope.descriptor(path)
    if "analysis_domain" in metadata:
        from .painted_native_product import native_analysis_exclusion_dependencies
        row, proof = metadata["analysis_domain"], metadata["analysis_domain"]["proof"]
        scope.descriptor(row["masks"]["path"], row["masks"])
        for bound in proof["dependencies"]:
            scope.descriptor(bound["path"], bound)
        contract = proof["native_product"] if proof["native_product"] is not None else proof["contract"]
        class Reader:
            def path(self, bound):
                scope.descriptor(bound["path"], bound)
        native_analysis_exclusion_dependencies(contract, reader=Reader())
        preparation = json.loads(scope.read(contract["preparation"]["path"]))
        painting = preparation["painting_facies"]
        # The original approval verifier also reads this implicit history.
        scope.descriptor(Path(painting["version"]["path"]).parent.parent / "approvals.jsonl")
        # Native gap/target replay invokes the original full review reader,
        # including its declared display metadata (but no photograph decode).
        from .painted_analysis_exclusions import read_gap_review
        from .painted_review_comparison import validate_display_context
        gap = preparation.get("analysis_exclusions")
        if gap is not None and proof["camera"]["run_id"] in gap["runs"]:
            approval = json.loads(scope.read(gap["approval"]["path"]))
            read_gap_review(approval["review"]["path"])
            scope.descriptor(Path(contract["preparation"]["path"]).parent / "source_map.npz")
            if preparation.get("display_context") is not None:
                validate_display_context(preparation["display_context"])
        target = ((preparation.get("target_corrections") or {}).get("runs") or {}).get(proof["camera"]["run_id"])
        for target_binding in (target or {}).get("bindings", []):
            version = json.loads(scope.read(target_binding["version"]["path"]))
            read_gap_review(version["basis"]["review"]["path"])
        certificate = folder / "baseline_native_renderer.json"
        if scope.exists(certificate, "exists"):
            scope.read(certificate)
    return str(folder)


def _controls(scope, config_path):
    from darsia.presets.workflows.config.multi_fluidflower_config import MultiFluidFlowerConfig
    from darsia.presets.workflows.utils.mass import select_data_path
    with scope.role("configuration"):
        config = MultiFluidFlowerConfig(scope.absolute(config_path), require_data=False, require_results=True)
    scope.check()
    w = config.wasserstein
    if w is None or config.runs is None or len(w.runs) < 2 or not w.roi or not w.times:
        raise ValueError("Preset Wasserstein needs its original runs, ROIs and requested times")
    times = [dict(order=i, time_hours=t, tolerance_hours=u) for i, (t, u) in enumerate(w.times)]
    selected = []
    for row in times:
        for run in w.runs:
            with scope.role("protocol"):
                path = select_data_path(config.runs.config[run], "mass", row["time_hours"],
                                        row["tolerance_hours"])
            with scope.role("image"):
                bound = None if path is None else scope.descriptor(path)
            selected.append(dict(time_order=row["order"], run=run, descriptor=bound))
    with scope.role("rig_model"):
        rig_folder = _rig_inputs(scope, config.runs.config[w.runs[0]].rig.path)
    return config, dict(runs=deepcopy(w.runs), rois=[roi.name for roi in w.roi.values()], requested_times=times,
                        reference_run=w.runs[0], rig_folder=rig_folder, selected_mass=selected)


def compile_preset_wasserstein_inputs(config_path, *, roots, series_manifest=None, path_aliases=None):
    """Prepare a source snapshot; no raster decode, scientific run, or output write."""
    if path_aliases is None:
        from .storage_paths import _alias_rows
        path_aliases = [dict(source=source, destination=destination) for source, destination in _alias_rows()]
    scope = _Scope(roots=roots, path_aliases=path_aliases)
    with scope.enter():
        with scope.role("configuration"):
            source_configuration = scope.descriptor(config_path)
            manifest = None if series_manifest is None else scope.descriptor(
                series_manifest["path"] if isinstance(series_manifest, dict) else series_manifest,
                series_manifest if isinstance(series_manifest, dict) else None)
        _, controls = _controls(scope, source_configuration["path"])
        scope.check()
    result = dict(format=FORMAT, source_configuration=source_configuration,
        working_directory=scope.working_directory, series_manifest=manifest, path_aliases=scope.path_aliases,
        config_paths=[deepcopy(r["descriptor"]) for r in scope.files.values()
                      if r["role"] == "configuration"],
        files=list(scope.files.values()), inventories=list(scope.inventories.values()),
        existence=list(scope.existence.values()), config_roots=list(scope.config_roots.values()), **controls)
    _json(result)
    from .preset_wasserstein_protocol import validate_inputs_contract
    validate_inputs_contract(result)
    return result


def catalogue_preset_wasserstein_inputs(catalog, operation):
    """Admit first, then replay original controls against only frozen input bytes."""
    from .preset_wasserstein_protocol import validate_inputs_contract
    frozen = validate_inputs_contract(operation["preset_inputs"])
    # The protocol validator may validate in-place and return None.
    frozen = operation["preset_inputs"] if frozen is None else frozen
    for row in frozen["files"]:
        for use in row["uses"]:
            catalog.add(row["descriptor"], row["role"], use)
    scope = _Scope(frozen=frozen, binding=getattr(catalog, "binding", None), catalog=catalog)
    with scope.enter():
        with scope.role("configuration"):
            scope.descriptor(frozen["source_configuration"]["path"])
            if frozen["series_manifest"] is not None:
                scope.descriptor(frozen["series_manifest"]["path"])
        _, controls = _controls(scope, frozen["source_configuration"]["path"])
        scope.check()
    if any(_json(controls[k]) != _json(frozen[k]) for k in controls):
        raise ValueError("Frozen preset controls differ from the original readers")
    if scope.files != scope.expected:
        raise ValueError("Preset declared files differ from the original metadata consumer closure")
    return frozen


class _Runtime:
    def __init__(self, task, binding, frozen):
        self.task, self.binding, self.frozen = deepcopy(task), binding, deepcopy(frozen)
        self.scope = _Scope(frozen=self.frozen, binding=binding)
        self.loaded = set()

    def __enter__(self):
        from darsia.presets.workflows.rig import Rig
        from darsia.experiment.experiment import ProtocolledExperiment
        from darsia.presets.workflows.config.multi_fluidflower_config import MultiFluidFlowerConfig
        self.context = self.scope.enter()
        self.context.__enter__()
        try:
            with self.scope.role("configuration"):
                self.config = MultiFluidFlowerConfig(Path(self.frozen["source_configuration"]["path"]),
                    require_data=False, require_results=True)
            self.scope.check()
            w = self.config.wasserstein
            expected = self.task["comparison_time"]
            if (w is None or list(w.runs) != self.frozen["runs"]
                    or [roi.name for roi in w.roi.values()] != self.frozen["rois"]
                    or expected not in self.frozen["requested_times"]
                    or list(w.times[expected["order"]]) != [expected["time_hours"], expected["tolerance_hours"]]):
                raise ValueError("Worker comparison control differs from original preset")
            w.times = [(expected["time_hours"], expected["tolerance_hours"])]
            first = self.config.runs.config[w.runs[0]]
            with self.scope.role("rig_model"):
                self.reference_rig = Rig.load(first.rig.path)
            with self.scope.role("protocol"):
                self.reference_experiment = ProtocolledExperiment.init_from_config(first)
            self.scope.check()
            return self
        except BaseException:
            self.context.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *args):
        return self.context.__exit__(*args)

    def load_mass(self, run_config, data, time, tol):
        from darsia.presets.workflows.utils.mass import load_data, select_data_path
        run = next((key for key, value in self.config.runs.config.items() if value is run_config), None)
        target = self.task["comparison_time"]
        if (run not in self.frozen["runs"] or data != "mass"
                or time != target["time_hours"] or tol != target["tolerance_hours"]):
            raise ValueError("Comparison requested mass outside its frozen task")
        expected = next(row["descriptor"] for row in self.frozen["selected_mass"]
                        if row["run"] == run and row["time_order"] == target["order"])
        with self.scope.role("protocol"):
            selected = select_data_path(run_config, data, time, tol)
        if expected is None:
            if selected is not None:
                raise ValueError("Original frozen no-match selection changed")
            self.loaded.add(run)
            self.scope.check()
            return None
        if selected is None or _key(self.scope.absolute(selected)) != _key(expected["path"]):
            raise ValueError("Original nearest-match selection changed")
        with self.scope.role("image"):
            image = load_data(run_config, data, time, tol)
        if image is None:
            raise ValueError("Original mass reader did not return its selected image")
        self.loaded.add(run)
        self.scope.check()
        return image

    def verify_consumed(self):
        self.scope.check()
        if self.loaded != set(self.frozen["runs"]):
            raise ValueError("Not every requested original run resolved its frozen mass selection")
        self.scope.verify_final()
        return dict(format="darsia-preset-wasserstein-consumption-v1",
            comparison_time=deepcopy(self.task["comparison_time"]),
            reference_run=self.frozen["reference_run"],
            files=[dict(**deepcopy(row), uses=self.scope.expected[key]["uses"])
                   for key, row in self.scope.consumed.items()],
            frozen_inventories=list(self.scope.inventories.values()),
            config_roots=list(self.scope.config_roots.values()))


def load_preset_wasserstein_runtime(task):
    """Enter only the exact whole-run task's authenticated original-reader scope."""
    from .pipeline.run_io import for_task
    binding = for_task(task)
    if binding is None or task.get("pipeline") != "preset-wasserstein":
        raise ValueError("Preset comparison requires its exact active run I/O binding")
    frozen = binding.configuration["preset_inputs"]
    if hashlib.sha256(_json(frozen).encode()).hexdigest() != task["preset_inputs_sha256"]:
        raise ValueError("Preset input snapshot differs from its admitted task")
    return _Runtime(task, binding, frozen)
