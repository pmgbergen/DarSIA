"""Named control inputs for the explicit series Setup operation.

This is a control catalogue, not a complete scientific input closure. The
original corrected-geometry proof readers run after capacity/startup/claim
admission and bind the sources they actually validate into the operation's
receipt. No map, image or RAW acquisition is decoded here. Control locators
stay physical; consumed native files may use declared aliases or an exact
file binding, without granting access to a containing directory.
"""
from __future__ import annotations

from copy import deepcopy
import json
import ntpath
from pathlib import Path
import re
import hashlib

from .series_setup_protocol import (
    APPLY_COMPLETE_FORMAT, validate_setup_operation, validate_setup_plan,
)
from ..preset_wasserstein_inputs import _Scope
from .job_inputs import MAX_METADATA_BYTES


SCOPE = "series-setup-controls-only"
_IDENTITY = ("path", "sha256", "size_bytes")
_REQUEST_FIELDS = {"format", "run_id", "configuration_files", "series_context",
                   "baseline", "profile", "operation_overrides", "protocol_inputs",
                   "output", "shape_basis_certificate"}


def _key(path):
    """Compare current-host spellings without relocating recorded history."""
    return ntpath.normcase(ntpath.normpath(str(path)))


def _same(left, right):
    """Keep bool/int and int/float differences in authored JSON metadata."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


class _Controls:
    """Use only the original catalogue's admitted named consumers."""

    def __init__(self, catalog):
        self.catalog = catalog

    def source(self, bound, role, use, *, metadata=False):
        """Reject relocation before stat/open; metadata uses the strict reader."""
        actual = self.catalog.physical(bound)
        if _key(actual["path"]) != _key(bound["path"]):
            raise ValueError("Setup controls require their original physical locators")
        operation = self.catalog.doc if metadata else self.catalog.add
        return operation(bound, role, "setup." + use)

    def named(self, path, role, use, *, metadata=False):
        """Bind one finite canonical control path, never a recursive discovery.

        Preparation checks containment before hashing. Runtime replay uses its
        already verified named reference and must not recapture a changed SHA.
        The temporary zero digest is only a containment check, never evidence.
        """
        name = "setup." + use
        binding = getattr(self.catalog, "binding", None)
        if binding is not None:
            row = binding.verified.for_use(name)
            if _key(row.path) != _key(path):
                raise ValueError("Setup control replay changed its original physical locator")
            bound = dict(path=str(path), sha256=row.reference["sha256"],
                         size_bytes=row.reference["bytes"])
        else:
            from ..store_map import input_reference
            from ..corrected_setup import _descriptor
            input_reference(str(path), self.catalog.roots, sha256="0" * 64,
                            size_bytes=0, role=role)
            bound = _descriptor(path)
        return bound, self.source(bound, role, use, metadata=metadata)


class _NativeInputs(_Scope):
    """Catalogue selected scientific inputs; preserve every logical descriptor.

    Only configured storage aliases may change a scientific source locator.
    The original request parser runs under this metadata-only scope: binary
    payload hashes belong to the worker's actual verified-input admission.
    """
    def __init__(self, controls):
        from ..storage_paths import _alias_rows
        self.controls, self.catalog = controls, controls.catalog
        self.path_aliases = [dict(source=source, destination=destination) for source, destination in _alias_rows()]
        self.declared, self.native, self.dependencies = {}, {}, {}
        self.raw_metadata, self.metadata_bytes = {}, 0

    def source(self, bound, role, use, *, metadata=False, strict=False):
        from ..storage_paths import _resolve_with
        if (not isinstance(bound, dict) or not set(_IDENTITY) <= set(bound)
                or type(bound["size_bytes"]) is not int or bound["size_bytes"] < 0):
            raise ValueError("Native Setup input requires its original path/SHA/size")
        original = {key: bound[key] for key in _IDENTITY}
        physical = dict(original)
        if not strict:
            physical["path"] = str(_resolve_with(original["path"],
                [(row["source"], row["destination"]) for row in self.path_aliases]))
            old = self.catalog.aliases.get(_key(original["path"]))
            if old is not None and not _same(old, physical):
                raise ValueError("Native Setup source conflicts with its declared storage alias")
            if _key(physical["path"]) != _key(original["path"]):
                self.catalog.aliases[_key(original["path"])] = physical
            operation = (self.catalog.doc if metadata else self.catalog.add_native
                         if use.startswith("raw.native.") else self.catalog.add)
            result = operation(original, role, "setup." + use)
        else:
            result = self.controls.source(original, role, use, metadata=metadata)
        binding = getattr(self.catalog, "binding", None)
        if binding is not None:
            row = binding.verified.for_use("setup." + use)
            physical = dict(original, path=str(row.path))
        record = dict(logical=original, physical=physical)
        for path in (original["path"], physical["path"]):
            key = _key(path)
            if key in self.declared and not _same(self.declared[key], record):
                previous = self.declared[key]
                if any(previous["logical"][name] != original[name] for name in ("sha256", "size_bytes")):
                    raise ValueError("A native Setup source changed between named consumers")
            self.declared[key] = record
        if not strict:
            self.native[_key(original["path"])] = record
        return result

    def path(self, bound):
        key = (_key(bound["path"]), bound["sha256"], bound["size_bytes"])
        if key not in self.dependencies:
            use = "raw.native." + str(len(self.dependencies))
            self.source(bound, "rig_model", use)
            self.dependencies[key] = use
        return Path(bound["path"])

    def descriptor(self, path, expected=None):
        record = self.declared.get(_key(path))
        if record is None:
            raise ValueError("Native Setup parser requested an undeclared input: " + str(path))
        value = dict(record["logical"], path=str(path))
        if expected is not None and any(value[name] != expected[name] for name in ("sha256", "size_bytes") if name in expected):
            raise ValueError("Native Setup parser source identity changed")
        if self.catalog.file_stat(record["physical"]["path"]).st_size != value["size_bytes"]:
            raise ValueError("Native Setup source size changed")
        return value

    def read(self, path, *, kind="bytes", sheet=None):
        if kind not in {"bytes", "raw-build-metadata"} or sheet is not None or Path(path).suffix.casefold() != ".json":
            raise ValueError("Native Setup catalogue may read only its declared JSON metadata")
        bound = self.descriptor(path)
        if bound["size_bytes"] > MAX_METADATA_BYTES:
            raise ValueError("Native Setup metadata exceeds its original read limit")
        physical = self.declared[_key(path)]["physical"]["path"]
        key = (_key(physical), bound["sha256"])
        raw = self.raw_metadata.get(key)
        if raw is None:
            raw = self.catalog.metadata_bytes(physical)
            if (not isinstance(raw, bytes) or len(raw) != bound["size_bytes"]
                    or hashlib.sha256(raw).hexdigest() != bound["sha256"]):
                raise ValueError("Native Setup metadata SHA/size changed")
            if self.metadata_bytes + len(raw) <= 16 * 1024**2:
                self.raw_metadata[key] = raw
                self.metadata_bytes += len(raw)
        return raw

    def stat(self, path):
        self.descriptor(path)
        return self.catalog.file_stat(self.declared[_key(path)]["physical"]["path"])

    def exists(self, *_a, **_kw):
        raise ValueError("Native Setup catalogue cannot probe undeclared filesystem state")

    def inventory(self, *_a, **_kw):
        raise ValueError("Native Setup catalogue cannot enumerate scientific inputs")


def _plan(controls, config):
    """Join the immutable plan to the literal selected project and revision."""
    plan = validate_setup_plan(controls.source(
        config["setup_plan"], "configuration", "plan", metadata=True))
    fields = ("project_root", "project_manifest", "adopted_revision")
    if config["operation"] != "image-preparation":
        fields += ("series_manifest",)
    if any(not _same(plan[key], config[key]) for key in fields):
        raise ValueError("The Setup plan belongs to another project or adopted revision")
    runs, plans = plan["runs"], plan["build_plans"]
    if config["operation"] == "image-preparation":
        if config["runs"] != runs or config["native_parameters"] != plan["native_parameters"]:
            raise ValueError("Image preparation must retain the original geometric run/native authority")
        selected = plan
    elif config["operation"] == "apply":
        if (not _same(config["runs"], runs) or not _same(config["build_plans"], plans)
                or any(not _same(config[key], plan[key]) for key in
                       ("native_parameters", "target_machine", "metadata_output", "output_scope"))):
            raise ValueError("Apply differs from its exact immutable run templates")
        selected = plan
    else:
        matches = [row for row in plans if row["run_id"] == config["run"]]
        if len(matches) != 1:
            raise ValueError("Build names a run outside its immutable Setup plan")
        selected = matches[0]
        for key in ("native_parameters", "target_machine", "metadata_output", "output_scope"):
            if not _same(config[key], selected.get(key)):
                raise ValueError("Build changed its original plan field: " + key)
    if not _same(config["native_parameters"], selected.get("native_parameters")):
        raise ValueError("Setup native parameters differ from the selected original run")
    return plan, selected


def _completed_controls(controls, config, plan, revision):
    """Catalogue completed Apply metadata without replaying scientific proofs."""
    completed = controls.source(config["coordinator_completion"], "configuration",
                                "coordinator_completion", metadata=True)
    expected = {"format", "complete", "setup_plan", "adopted_revision", "project_manifest",
                "series_manifest", "revision_id", "current_sha256", "runs", "Build_run",
                "optical_calibration_run"}
    if (set(completed) != expected or completed["format"] != APPLY_COMPLETE_FORMAT
            or completed["complete"] is not True or completed["Build_run"] is not False
            or completed["optical_calibration_run"] is not False
            or any(not _same(completed[key], config[key]) for key in
                   ("setup_plan", "adopted_revision", "project_manifest", "series_manifest"))
            or completed["revision_id"] != revision.get("revision_id")
            or completed["current_sha256"] != config["adopted_revision"]["sha256"]
            or _key(config["coordinator_completion"]["path"])
                != _key(Path(plan["control_root"]) / "apply-complete.json")):
        raise ValueError("Build lacks its exact complete coordinator Apply controls")
    rows = completed["runs"]
    if (not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) !=
            {"run_id", "export", "apply_completion", "configuration_files"} for row in rows)
            or [row["run_id"] for row in rows] != plan["runs"]):
        raise ValueError("Coordinator Apply rows changed the original ordered run scope")
    selected = next(row for row in rows if row["run_id"] == config["run"])
    if any(not _same(selected[key], config[key]) for key in
           ("apply_completion", "configuration_files")):
        raise ValueError("Build is not bound to its actual selected Apply postimages")
    exported = controls.source(selected["export"], "configuration", "export", metadata=True)
    applied = controls.source(config["apply_completion"], "configuration", "apply_completion", metadata=True)
    template = next(row["request_template"] for row in plan["build_plans"] if row["run_id"] == config["run"])
    for field in ("analysis_exclusions", "native_product"):
        if not _same((exported.get("binding") or {}).get(field), template.get(field)):
            raise ValueError("The actual Apply export differs from its planned " + field)
    if (exported.get("run_id") != config["run"]
            or exported.get("revision_id") != revision.get("revision_id")
            or exported.get("current_sha256") != completed["current_sha256"]
            or applied.get("format") != "darsia-corrected-setup-apply-complete-v1"
            or applied.get("complete") is not True or applied.get("run_id") != config["run"]
            or applied.get("revision_id") != revision.get("revision_id")
            or not _same(applied.get("export"), selected["export"])):
        raise ValueError("Build's actual export and per-run Apply controls disagree")
    destination = Path(selected["export"]["path"]).parent
    if _key(config["apply_completion"]["path"]) != _key(destination / "apply-complete.json"):
        raise ValueError("The selected Apply completion belongs to another export")
    controls.named(destination / "apply-claim.json", "configuration", "apply_claim", metadata=True)
    controls.source(applied["attempt_receipt"], "configuration", "apply_attempt", metadata=True)
    for index, row in enumerate(applied["configurations"]):
        for side in ("before", "after"):
            controls.source(row[side], "configuration", "applied." + str(index) + "." + side)
    if exported.get("root_export") is not None:
        controls.source(exported["root_export"], "configuration", "reference_export", metadata=True)
    return selected


def _raw_request(controls, config, selected):
    """Admit every original request pin before its unchanged verifying reader."""
    from ..rig_raw_build import REQUEST_FORMAT, _native_request_dependencies, read_build_request
    request = controls.source(config["raw_build_request"], "configuration", "raw_request", metadata=True)
    if (not _REQUEST_FIELDS <= set(request) <= _REQUEST_FIELDS | {"analysis_exclusions", "native_product"}
            or request["format"] != REQUEST_FORMAT
            or request["run_id"] != config["run"]
            or not _same(request["configuration_files"], config["configuration_files"])
            or not _same({key: value for key, value in request.items() if key != "configuration_files"},
                         selected["request_template"])):
        raise ValueError("The full RAW request differs from its actual run, Apply postimages or original template")
    scope = _NativeInputs(controls)
    scope.source(config["raw_build_request"], "configuration", "raw_request", strict=True)
    for index, row in enumerate(config["configuration_files"]):
        scope.source(row, "configuration", "raw.configuration." + str(index), strict=True)
    scope.source(request["baseline"], "image", "raw.baseline")
    scope.source(request["profile"], "rig_model", "raw.profile", metadata=True)
    if not _same(request["series_context"]["manifest"], config["series_manifest"]):
        raise ValueError("The full RAW request belongs to another original series manifest")
    scope.source(request["series_context"]["manifest"], "configuration", "raw.series_manifest", strict=True)
    from ..protocol_content import current_source
    for index, row in enumerate(request["protocol_inputs"]):
        # A corrected injection position keeps the Build (protocol_content).
        scope.source(current_source(row), "protocol", "raw.protocol." + str(index) + "." + row["role"])
    scope.source(request["shape_basis_certificate"], "rig_model", "raw.shape_basis", metadata=True)
    with scope.enter():
        _native_request_dependencies(request, reader=scope)
        verified, _pins = read_build_request(config["raw_build_request"], artifact_kind="raw_rig_baseline")
    if not _same(verified, request):
        raise ValueError("The original RAW request changed after its named controls were read")
    return verified, list(scope.native.values())


def catalogue_setup_inputs(catalog, config):
    """Bind finite Setup controls; leave actual scientific proof replay explicit.

    The enclosing catalogue retains its existing schema. Callers must record
    the returned ``scope`` and must not describe this as all scientific input
    admission. Apply target TOMLs and future Build template paths are not read
    or frozen here: original Apply owns their CAS and retained-before proof.
    """
    validate_setup_operation(config)
    controls = _Controls(catalog)
    controls.source(config["project_manifest"], "configuration", "project_manifest", metadata=True)
    revision = controls.source(config["adopted_revision"], "rig_model", "adopted_revision", metadata=True)
    controls.source(config["series_manifest"], "configuration", "series_manifest")
    controls.source(config["native_parameters"], "configuration", "native_parameters", metadata=True)
    if config.get("reuse_image_basis") is not None:
        controls.source(config["reuse_image_basis"], "configuration", "reused_image_basis", metadata=True)
    plan, selected = _plan(controls, config)
    if (revision.get("format") != "darsia-fluidflower-geometry-facies-v1"
            or revision.get("schema") != 1
            or re.fullmatch(r"[0-9a-f]{32}", str(revision.get("revision_id"))) is None):
        raise ValueError("Setup requires its actual saved adopted revision controls")
    request, native_inputs = None, []
    if config["operation"] == "build":
        _completed_controls(controls, config, plan, revision)
        request, native_inputs = _raw_request(controls, config, selected)
    return dict(scope=SCOPE, plan=deepcopy(plan), revision=deepcopy(revision),
                raw_request=deepcopy(request), native_inputs=deepcopy(native_inputs))
