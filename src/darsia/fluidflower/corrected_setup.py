"""Explicit per-run export/apply of already corrected workspace rasters.

No segmentation, registration, interpolation or rig calculation is implemented
here. Original raster archives and the selected run's original property table
are copied verbatim. Only four configuration leaves are applied explicitly.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tomllib
import uuid
import numpy as np

from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT, GeometryRevisionStore, _load_array
from darsia.fluidflower.geometry_workspace_frame import adopted_content_digest, frame_kind, selected_fields
from darsia.fluidflower.project import utc_now
from darsia.fluidflower.series_geometry import (
    NATIVE_SETUP_EXPORT_FORMAT, NATIVE_LABEL_COLORS, geometry_package_root,
    native_color_index, native_series_link, _storage_path,
)
from darsia.fluidflower.setup_geometry_apply import ReadBinding, _exclusive as _original_exclusive
from darsia.fluidflower.setup_rig import _require_exact_grid

FORMAT = NATIVE_SETUP_EXPORT_FORMAT
LIMIT = 1024 * 1024
NAMES = {"labels": "native_labels.npz", "facies": "native_facies.npz"}
EXPORT_FILES = {*NAMES.values(), "facies_properties.csv", "native_setup.toml", "colored_labels.png"}


def _exclusive(path, data):
    """Keep the original writer unless an admitted Setup operation is active."""
    from .pipeline.series_setup_sink import current_sink
    sink = current_sink()
    if sink is None:
        return _original_exclusive(path, data)
    return sink.write_bytes(path, data)


def _copy_output(source, destination):
    """Bound an explicit Setup copy while retaining the default byte copier."""
    from .pipeline.series_setup_sink import current_sink
    sink = current_sink()
    if sink is None:
        return shutil.copyfile(source, destination)
    return sink.copy_file(source, destination)


def _mkdir_output(path, *, parents=False, exist_ok=False):
    """Check a positive Setup output domain before creating a directory."""
    from .pipeline.series_setup_sink import current_sink
    sink = current_sink()
    if sink is None:
        return path.mkdir(parents=parents, exist_ok=exist_ok)
    return sink.mkdir(path, parents=parents, exist_ok=exist_ok)


def _move_output(source, destination, *, replace=False):
    """Preserve the original rename/CAS boundary and the whole byte ceiling."""
    from .pipeline.series_setup_sink import current_sink
    sink = current_sink()
    if sink is None:
        return (os.replace if replace else os.rename)(source, destination)
    return sink.rename(source, destination, replace=replace)


def _save_output_png(image, path):
    """Use the original PNG serializer through a capped stream when active."""
    from .pipeline.series_setup_sink import current_sink
    sink = current_sink()
    if sink is None:
        return image.save(path)
    with sink.open_exclusive(path) as stream:
        return image.save(stream, format="PNG")


def _small(path):
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError(f"Corrected setup metadata exceeds 1 MiB: {path}")
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError(f"Corrected setup metadata exceeds 1 MiB: {path}")
    return data


def _descriptor(path):
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path)
    from .setup_source_reads import source_descriptor
    return source_descriptor(path, reader=_read_descriptor)


def _read_descriptor(path):
    """Hash the original bytes without retaining the source contents."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during corrected setup verification: {path}")
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def _verified(row):
    actual = _descriptor(row["path"])
    if actual["sha256"] != row.get("sha256") or actual["size_bytes"] != row.get("size_bytes"):
        raise ValueError(f"Corrected setup source identity mismatch: {actual['path']}")
    return actual


def _same(first, second):
    from .storage_paths import resolve_storage_path
    first, second = resolve_storage_path(first), resolve_storage_path(second)
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _identity(row):
    return {key: row[key] for key in ("path", "sha256", "size_bytes")}


def _same_source(first, second):
    """Keep the exact source descriptor, allowing only declared storage aliases."""
    from .storage_paths import same_storage_descriptor
    return same_storage_descriptor(first, second)


def _revision_properties(config, saved, registered):
    """The full table authority for this explicit revision, without a fallback.

    The registered resolver has already validated a painted table against its
    approved session, parent CSV, preparation and exact materialized bytes.
    Every other revision keeps the selected run's original table authority.
    """
    if "painted_adoption_table" in config:
        if registered is None:
            raise ValueError("A painted adoption needs its verified registered full CSV")
        return _identity(registered["files"]["facies_properties"])
    return _identity(saved["facies_properties"])


def _replacements(destination):
    return {
        "labeling": {"labels": str(destination / NAMES["labels"]), "frame": "rig-corrected"},
        "facies": {"path": str(destination / NAMES["facies"]), "props": str(destination / "facies_properties.csv")},
    }


def _run_name(run):
    if not isinstance(run, str) or not run or run in (".", "..") or Path(run).name != run or any(c in run for c in "\\/:\x00"):
        raise ValueError("Corrected setup requires one explicit safe run ID.")
    return run


def _require_unedited_adoption(current, config):
    provenance = config.get("provenance") or {}
    if provenance.get("adopted_content_sha256") != adopted_content_digest(config):
        raise ValueError("This corrected revision was edited or lacks its adoption identity; export of edited/registered geometry is not qualified.")
    display = config["frame_binding"]["display_reference_run"]
    fields = selected_fields(config, display)
    for key, field in (("baseline_path", "shape_corrected_baseline"), ("labels_path", "labels"), ("facies_source_path", "facies"), ("depth_path", "depth")):
        actual = _descriptor(config[key])
        if actual["sha256"] != fields[field]["sha256"] or actual["size_bytes"] != fields[field]["size_bytes"]:
            raise ValueError("Saved corrected revision fields differ from the original adopted maps.")
    derived = (current.get("artifacts") or {}).get("facies_path")
    if not derived:
        raise ValueError("The genuine Store-derived facies artifact is missing.")
    if not np.array_equal(_load_array(Path(derived)), _load_array(Path(fields["facies"]["path"]))):
        raise ValueError("Store-derived facies differ from the original adopted facies; no corrected export was written.")


@dataclass(frozen=True)
class CorrectedSetupExport:
    revision_id: str
    run_id: str
    destination: Path
    toml_text: str
    current_json: Path
    current_bytes: bytes
    receipt: dict
    reused: bool = False


class CorrectedApplyError(RuntimeError):
    def __init__(self, message, receipt):
        self.receipt = Path(receipt)
        super().__init__(f"{message}. Application receipt and preserved originals: {receipt}")


def _active_setup_export_scope():
    from .pipeline.run_io import current_binding
    reader = current_binding()
    provider = getattr(reader, "corrected_setup_export_scope", None)
    return provider() if provider is not None else None


def _setup_scope_root(scope, revision, *, plan=None):
    """Resolve a job export from its original bound plan, never a loose path."""
    if (not isinstance(scope, dict) or set(scope) != {"format", "job_id", "project_root", "setup_plan"}
            or scope.get("format") != "darsia-job-corrected-export-scope-v1"):
        raise ValueError("The job-owned corrected export scope is malformed")
    from .pipeline.series_setup_protocol import validate_setup_plan
    if plan is None:
        from .pipeline.capacity import read_bound
        plan = read_bound(scope["setup_plan"])[0]
    validate_setup_plan(plan)
    controls = Path(plan["control_root"])
    if (controls.name != "setup-controls" or controls.parent.name != scope["job_id"]
            or not _same(plan["project_root"], scope["project_root"])
            or not _same(scope["setup_plan"]["path"], controls / "plan.json")):
        raise ValueError("The corrected export scope differs from its original job plan")
    return controls.parent / "exports" / "native_setup" / revision


def _canonical_location(package_root, config, revision, run, *, setup_scope=None):
    display = _run_name(config["frame_binding"]["display_reference_run"])
    runs = config["frame_binding"]["runs"]
    if run not in runs:
        raise ValueError("No corrected export binding exists for this selected run.")
    planned = {_run_name(key): f"runs/{key}/export.json" for key in sorted(runs) if key != display}
    active = _active_setup_export_scope()
    if setup_scope is None and active is not None:
        setup_scope = active[0]
    if setup_scope is not None:
        if not _same(package_root, setup_scope["project_root"]):
            raise ValueError("The corrected export scope belongs to another source project")
        plan = active[1] if active is not None and active[0] == setup_scope else None
        root = _setup_scope_root(setup_scope, revision, plan=plan)
    else:
        root = package_root / "exports" / "native_setup" / revision
    return root, root if run == display else root / "runs" / run, display, planned


CONFIGURATION_BINDING_FORMAT = "darsia-corrected-setup-configuration-dual-binding-v1"
CONFIGURATION_BINDING_V2_FORMAT = "darsia-corrected-setup-configuration-dual-binding-v2"
PROTOCOL_LEAVES = ("protocols.imaging", "protocols.injection", "protocols.imaging_mode")


def _typed_equal(first, second):
    """TOML/JSON equality without bool/int, int/float or container coercion."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_typed_equal(first[key], second[key]) for key in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(_typed_equal(a, b) for a, b in zip(first, second))
    return first == second


def _protocol_changes(original, current):
    """Exactly three explicit clock-pointer/mode changes, no other difference."""
    restored = deepcopy(current)
    changes = {}
    old_protocols, new_protocols = original.get("protocols"), restored.get("protocols")
    if not isinstance(old_protocols, dict) or not isinstance(new_protocols, dict):
        raise ValueError("Dual configuration binding requires both explicit protocol tables.")
    for leaf in PROTOCOL_LEAVES:
        key = leaf.split(".")[1]
        if key not in old_protocols or key not in new_protocols or _typed_equal(old_protocols[key], new_protocols[key]):
            raise ValueError("Dual configuration binding requires exactly the three protocol-leaf differences.")
        changes[leaf] = dict(before=deepcopy(old_protocols[key]), after=deepcopy(new_protocols[key]))
        new_protocols[key] = deepcopy(old_protocols[key])
    if not _typed_equal(original, restored):
        raise ValueError("Dual configuration binding changes a non-protocol value or TOML type.")
    return changes


def _configuration_identity(row):
    if (not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}
            or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
            or not isinstance(row["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            or type(row["size_bytes"]) is not int or row["size_bytes"] < 0):
        raise ValueError("Dual configuration identities require exact absolute path/SHA/byte descriptors.")
    return row


def _dual_configuration_binding(value, run, configuration, selected, *, destination, receipt):
    keys = {"format", "run_id", "source_configuration", "current_configuration",
            "apply_configurations", "protocol_changes", "comparison_sha256"}
    if (not isinstance(value, dict) or set(value) != keys
            or value["format"] != CONFIGURATION_BINDING_FORMAT or value["run_id"] != run
            or not isinstance(value["comparison_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["comparison_sha256"]) is None):
        raise ValueError("The selected-run dual configuration metadata is not explicit and complete.")
    original = _configuration_identity(value["source_configuration"])
    current = _configuration_identity(value["current_configuration"])
    targets = value["apply_configurations"]
    if (not _typed_equal(original, configuration) or not isinstance(targets, list)
            or len(selected) != 2 or len(targets) != 2):
        raise ValueError("The dual configuration binding belongs to another source or selected bundle.")
    for target, declared in zip(targets, selected):
        _configuration_identity(target)
        if not _same(target["path"], declared) or _same(target["path"], original["path"]):
            raise ValueError("Dual Apply targets must be the exact current declared bundle, never the historical source.")
    if not _typed_equal(current, targets[-1]):
        raise ValueError("The dual current configuration must be the declared merged target.")
    _verified(original)  # This physical source is immutable, including after Apply.
    raw_targets = [_original_configuration_bytes(target, destination, receipt) for target in targets]
    changes = _protocol_changes(tomllib.loads(_small(original["path"]).decode("utf-8-sig")),
                                tomllib.loads(raw_targets[-1].decode("utf-8-sig")))
    if not _typed_equal(value["protocol_changes"], changes):
        raise ValueError("The dual configuration metadata does not record the exact protocol-leaf changes.")
    return deepcopy(value)


def _configuration_bytes_v2(row):
    """Bind the exact bytes consumed by the new v2 TOML parser."""
    raw = _small(row["path"])
    if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("V2 consumed configuration bytes differ from the bound SHA/size identity.")
    return raw


def _dual_configuration_binding_v2(value, run, configuration, selected, *, series,
                                   project_root, revision, destination, receipt):
    """Bind one separately migrated output pointer, never a geometry override."""
    keys = {"format", "run_id", "source_configuration", "current_configuration",
            "apply_configurations", "pre_pointer_configurations", "protocol_change_mode", "protocol_changes",
            "output_pointer_change", "comparison_sha256"}
    if (not isinstance(value, dict) or set(value) != keys
            or value["format"] != CONFIGURATION_BINDING_V2_FORMAT or value["run_id"] != run
            or not isinstance(value["comparison_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["comparison_sha256"]) is None):
        raise ValueError("The v2 selected-run configuration metadata is not explicit and complete.")
    if (project_root is None or not Path(project_root).is_absolute()
            or not _same(project_root, geometry_package_root(Path(series)))
            or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{32}", revision) is None):
        raise ValueError("The v2 output pointer requires the actual series project and revision.")
    original = _configuration_identity(value["source_configuration"])
    current = _configuration_identity(value["current_configuration"])
    targets = value["apply_configurations"]
    if (not _typed_equal(original, configuration) or not isinstance(targets, list)
            or len(selected) != 2 or len(targets) != 2):
        raise ValueError("The v2 binding belongs to another source or selected bundle.")
    for target, declared in zip(targets, selected):
        _configuration_identity(target)
        if not _same(target["path"], declared) or _same(target["path"], original["path"]):
            raise ValueError("V2 Apply targets must be the exact current declared bundle, never the historical source.")
    if not _typed_equal(current, targets[-1]):
        raise ValueError("The v2 current configuration must be the declared merged target.")
    previous = value["pre_pointer_configurations"]
    if not isinstance(previous, list) or len(previous) != 2:
        raise ValueError("V2 requires exact preserved pre-pointer partial and merged identities.")
    for before, role in zip(previous, ("runs", "merged")):
        _configuration_identity(before)
        path = Path(before["path"])
        if (not path.resolve().is_relative_to(Path(series).resolve())
                or path.parent.name != role or path.name != run + ".toml"
                or any(_same(path, target["path"]) for target in targets)):
            raise ValueError("V2 pre-pointer identities must be preserved inside this series in exact run partial/merged order.")
    pointer = value["output_pointer_change"]
    expected = Path(project_root) / "results" / "rig_models" / revision / _run_name(run)
    if (not isinstance(pointer, dict) or set(pointer) != {"field", "before", "after"}
            or pointer["field"] != "rig.path"
            or not isinstance(pointer["before"], str) or not pointer["before"]
            or not Path(pointer["before"]).is_absolute()
            or not isinstance(pointer["after"], str) or pointer["after"] != str(expected)):
        raise ValueError("V2 permits only the explicit rig.path change to this project's fixed revision/run destination.")
    old_folder, new_folder = Path(pointer["before"]).resolve(), expected.resolve()
    if old_folder.is_relative_to(new_folder) or new_folder.is_relative_to(old_folder):
        raise ValueError("The v2 rig output must not overlap the preserved original rig folder.")
    source = tomllib.loads(_configuration_bytes_v2(original).decode("utf-8-sig"))
    if (source.get("rig") or {}).get("path") != pointer["before"]:
        raise ValueError("The v2 output pointer does not bind the original rig.path.")
    raw_targets = [_original_configuration_bytes(target, destination, receipt) for target in targets]
    configs = [tomllib.loads(raw.decode("utf-8-sig")) for raw in raw_targets]
    pre_configs = [tomllib.loads(_configuration_bytes_v2(before).decode("utf-8-sig")) for before in previous]
    if any((config.get("rig") or {}).get("path") != pointer["after"] for config in configs):
        raise ValueError("Both v2 declared Apply targets must carry the exact new rig.path.")
    for before, after in zip(pre_configs, configs):
        if (before.get("rig") or {}).get("path") != pointer["before"]:
            raise ValueError("Both preserved v2 pre-pointer configurations must carry the exact old rig.path.")
        restored = deepcopy(after)
        restored["rig"]["path"] = pointer["before"]
        if not _typed_equal(before, restored):
            raise ValueError("V2 target changes a value or TOML type beyond its own preserved rig.path migration.")
    mode = value["protocol_change_mode"]
    if mode == "clock-migrated-three-leaves":
        changes = _protocol_changes(source, pre_configs[-1])
    elif mode == "unchanged-zero-leaves":
        if not _typed_equal(source, pre_configs[-1]):
            raise ValueError("The zero-protocol v2 binding changes another value or TOML type.")
        changes = {}
    else:
        raise ValueError("V2 requires an explicit zero- or three-protocol change mode.")
    if not _typed_equal(value["protocol_changes"], changes):
        raise ValueError("V2 protocol metadata does not record its exact declared changes.")
    return deepcopy(value)


PRIVATE_MANUAL_CONFIGURATION_FORMAT = "darsia-private-Q121-corrected-setup-configuration-binding-v1"


def _private_manual_output_bytes(before, changes):
    """Only two physical output-pointer lines; every other byte is retained."""
    if set(changes)!={"rig.path","data.results"}:
        raise ValueError("Private Q121 requires exactly rig.path and data.results.")
    expected=tomllib.loads(before.decode("utf-8-sig"))
    bom=b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b""
    lines=before[len(bom):].decode("utf-8").splitlines(keepends=True)
    for leaf in ("rig.path","data.results"):
        section,key=leaf.split(".");change=changes[leaf]
        if (not isinstance(change,dict) or set(change)!={"field","before","after"}
                or change["field"]!=leaf or expected[section][key]!=change["before"]):
            raise ValueError("Private Q121 output source pointer differs.")
        headers=[(i,re.match(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$",line.rstrip("\r\n"))) for i,line in enumerate(lines)]
        starts=[i for i,m in headers if m and m.group(1).strip()==section]
        if len(starts)!=1:raise ValueError("Private Q121 needs one original output table.")
        start=starts[0];end=next((i for i in range(start+1,len(lines)) if lines[i].lstrip().startswith("[")),len(lines))
        found=[i for i in range(start+1,end) if re.match(rf"^\s*{re.escape(key)}\s*=",lines[i])]
        if len(found)!=1:raise ValueError("Private Q121 needs one original output pointer line.")
        index=found[0];line=lines[index]
        prefix=re.match(rf"^(\s*{re.escape(key)}\s*=\s*)",line).group(1)
        newline="\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        lines[index]=prefix+json.dumps(change["after"],ensure_ascii=False)+newline
        expected[section][key]=change["after"]
    after=bom+"".join(lines).encode("utf-8")
    if not _typed_equal(tomllib.loads(after.decode("utf-8-sig")),expected):
        raise ValueError("Private Q121 pointer patch changed another TOML value/type.")
    return after


def _private_manual_configuration_binding(value,run,configuration,selected,*,series,
        project_root,revision,destination,receipt):
    """Explicit private-only Q121 I/O, never an extension of ordinary dual V2."""
    keys={"format","run_id","source_configuration","current_configuration","apply_configurations",
        "pre_pointer_configurations","output_pointer_changes","manual_revision","comparison_sha256"}
    if (not isinstance(value,dict) or set(value)!=keys or value["format"]!=PRIVATE_MANUAL_CONFIGURATION_FORMAT
            or value["run_id"]!=run or re.fullmatch(r"[0-9a-f]{64}",str(value["comparison_sha256"])) is None
            or project_root is None or str(project_root).startswith(("\\\\","//"))
            or not _same(project_root,geometry_package_root(Path(series)))
            or re.fullmatch(r"[0-9a-f]{32}",str(revision)) is None):
        raise ValueError("Private Q121 output binding needs its exact local project/run/revision.")
    original=_configuration_identity(value["source_configuration"])
    targets=value["apply_configurations"];previous=value["pre_pointer_configurations"]
    if (not _typed_equal(original,configuration) or not isinstance(targets,list) or len(targets)!=2
            or len(selected)!=2 or not isinstance(previous,list) or len(previous)!=2):
        raise ValueError("Private Q121 requires original source and two pre-pointer/current targets.")
    root=Path(project_root).resolve();series=Path(series).resolve()
    for target,path in zip(targets,selected):
        _configuration_identity(target)
        if (not _same(target["path"],path) or _same(target["path"],original["path"])
                or not Path(target["path"]).resolve().is_relative_to(series)):
            raise ValueError("Private Q121 Apply target is foreign or is the preserved source.")
    if not _typed_equal(_configuration_identity(value["current_configuration"]),targets[-1]):
        raise ValueError("Private Q121 current must be its actual merged target.")
    source=_configuration_bytes_v2(original)
    for old,role in zip(previous,("runs","merged")):
        _configuration_identity(old);path=Path(old["path"])
        if (not path.resolve().is_relative_to(series) or path.parent.name!=role or path.name!=run+".toml"
                or any(_same(path,target["path"]) for target in targets)
                or _configuration_bytes_v2(old)!=source):
            raise ValueError("Private Q121 pre-pointer copies must retain ALL original bytes.")
    if _same(previous[0]["path"],previous[1]["path"]):
        raise ValueError("Private Q121 requires two distinct original pre-pointer copies.")
    expected={"rig.path":root/"results/rig_models"/revision/run,
              "data.results":root/"results/native_data"/revision/run}
    changes=value["output_pointer_changes"]
    if not isinstance(changes,dict) or set(changes)!=set(expected):
        raise ValueError("Private Q121 permits only its two explicit output pointer leaves.")
    for leaf,target in expected.items():
        change=changes[leaf]
        if (not isinstance(change,dict) or set(change)!={"field","before","after"}
                or change["field"]!=leaf or change["after"]!=str(target)
                or not isinstance(change["before"],str) or not Path(change["before"]).is_absolute()
                or target.resolve()!=target or Path(change["before"]).resolve().is_relative_to(root)):
            raise ValueError("Private Q121 output pointer is not its fixed local UUID/run path.")
    expected_raw=_private_manual_output_bytes(source,changes)
    for target in targets:
        if _original_configuration_bytes(target,destination,receipt)!=expected_raw:
            raise ValueError("Private Q121 changed bytes beyond the two output pointer lines.")
    record=_configuration_identity(value["manual_revision"])
    if not _same(record["path"],root/"results/geometry"/revision/"revision.json"):
        raise ValueError("Private Q121 manual provenance is not this new Store UUID.")
    current=json.loads(_configuration_bytes_v2(record))
    from .geometry_raster_sources import private_manual_label_edit_sources,MANUAL_LABEL_EDIT
    manual=private_manual_label_edit_sources(root,current,run)
    if manual.get("format")!=MANUAL_LABEL_EDIT or manual.get("revision_id")!=revision or manual.get("run_id")!=run:
        raise ValueError("Private Q121 output binding lacks actual manual parent/session/Store provenance.")
    return deepcopy(value)


def _require_fresh_private_manual_build(paths,dual):
    # Same effective-config/fresh-rig/cache checks. This is a field projection
    # of an already checked private record, not relabelling it as dual V2.
    return _require_fresh_v2_build(paths,dict(current_configuration=dual["current_configuration"],
        output_pointer_change=dual["output_pointer_changes"]["rig.path"]))


def _require_fresh_v2_build(paths, dual):
    """Check original dispatch's output/cache boundary without creating anything."""
    from darsia.presets.workflows.config.utils import _deep_merge
    effective = {}
    for path in [paths] if isinstance(paths, Path) else paths:
        _deep_merge(effective, tomllib.loads(_small(path).decode("utf-8-sig")))
    declared = tomllib.loads(_small(dual["current_configuration"]["path"]).decode("utf-8-sig"))
    if not _typed_equal(effective, declared):
        raise ValueError("V2 Build's entire effective configuration differs from its bound current merged configuration.")
    rig = effective["rig"]
    target = dual["output_pointer_change"]["after"]
    if rig.get("path") != target:
        raise ValueError("V2 Build's effective rig.path is not its bound fresh output.")
    folder = Path(target)
    if not _same(folder.resolve(), folder):
        raise ValueError("The fresh v2 rig output must not resolve through another location.")
    if folder.exists() and (not folder.is_dir() or next(folder.iterdir(), None) is not None):
        raise ValueError("The v2 rig output is not empty; partial or completed artifacts were preserved.")
    data = effective["data"]
    results = data.get("results")
    if (not isinstance(results, str) or not results or not Path(results).is_absolute()
            or not Path(results).is_dir()):
        raise ValueError("V2 Build requires the original data.results directory to exist before dispatch.")
    use_cache = data.get("use_cache", False)
    if type(use_cache) is not bool:
        raise ValueError("V2 Build requires an explicit boolean data.use_cache when present.")
    if use_cache:
        cache, raw_cache = Path(results) / "cache", Path(results) / "raw_cache"
        if not cache.is_dir() or not raw_cache.is_dir():
            raise ValueError("V2 Build requires the original cache and raw_cache directories before dispatch.")
        single = data.get("folder")
        multiple = data.get("folders")
        if single in ("", "No file chosen"):
            single = None
        if multiple == "":
            multiple = None
        if ((single is not None and (not isinstance(single, str) or not single))
                or (multiple is not None and (not isinstance(multiple, list) or not multiple
                    or any(not isinstance(item, str) or not item for item in multiple)))):
            raise ValueError("V2 Build requires the original explicit data folder(s).")
        folders = [Path(single)] if single is not None else []
        for item in multiple or []:
            if Path(item) not in folders:
                folders.append(Path(item))
        baseline = data.get("baseline")
        if not folders or not isinstance(baseline, str) or not baseline:
            raise ValueError("V2 Build requires the original baseline and data folder(s).")
        baseline = Path(baseline)
        if not baseline.is_absolute():
            candidates = [folder / baseline for folder in folders]
            baseline = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        relative = None
        for folder in folders:
            try:
                relative = baseline.resolve().relative_to(folder.resolve())
                break
            except ValueError:
                continue
        if relative is None:
            relative = Path(baseline.name)
        if not (raw_cache / relative.with_suffix(".npz")).is_file():
            raise ValueError("V2 Build requires the original baseline raw-cache file; no raw image was read or cache created.")


RAW_CONFIGURATION_FORMAT = "darsia-registered-raw-setup-configuration-v1"


def _raw_configuration_bundle(series, run, configuration, selected, *, project_root,
                              revision, destination=None, receipt=None):
    """Keep a registered map's historical source separate from current Apply targets.

    Current TOMLs are destination/configuration inputs, not replacements for
    the saved map's original evidence. The registered geometry reader and full
    RAW Build still verify their original sources and basis consistency.
    """
    if project_root is None or revision is None:
        return None
    if (len(selected) != 2 or len({str(Path(path).resolve()) for path in selected}) != 2
            or any(not Path(path).resolve().is_relative_to(Path(series).resolve()) for path in selected)):
        raise ValueError("Current RAW Apply needs two distinct declared targets inside this series")
    try:
        current = json.loads(_small(Path(project_root) / "results/geometry" / revision / "revision.json"))
    except FileNotFoundError:
        # Older canonical imports need no saved painting revision. They keep
        # the existing dual-binding validator, never this additive RAW path.
        return None
    config = current.get("configuration") or {}
    if "painted_adoption_table" not in config or current.get("revision_id") != revision:
        return None
    source = ((config.get("frame_binding") or {}).get("runs") or {}).get(run, {}).get("configuration")
    if not isinstance(source, dict) or not _typed_equal(_identity(source), _identity(configuration)):
        raise ValueError("Current RAW Setup must retain this saved run's historical configuration")
    if any(_same(configuration["path"], path) for path in selected):
        return None
    rows = [row for row in (config.get("run_geometry") or {}).get("runs", ())
            if row.get("run_id") == run and row.get("enabled", True)]
    if len(rows) != 1:
        return None
    model_binding = (rows[0].get("registration") or {}).get("target_model")
    if not isinstance(model_binding, dict):
        return None
    model = json.loads(_small(_verified(model_binding)["path"]))
    basis_binding = (model.get("baseline_native_grid") or {}).get("renderer")
    if not isinstance(basis_binding, dict):
        return None
    basis = json.loads(_small(_verified(basis_binding)["path"]))
    if (model.get("run_id", model.get("run")) != run or basis.get("run_id") != run
            or basis.get("format") != "darsia-raw-rig-baseline-build-v1"
            or basis.get("artifact_kind") != "raw_shape_basis"):
        raise ValueError("Current Apply targets need this registered run's exact RAW shape basis")
    previous = ((receipt or {}).get("binding") or {}).get("dual_configuration_binding") or {}
    if previous.get("format") == RAW_CONFIGURATION_FORMAT:
        targets = previous["apply_configurations"]
        if len(targets) != len(selected) or any(not _same(row["path"], path)
                                               for row, path in zip(targets, selected)):
            raise ValueError("The current RAW Apply target bundle changed")
    else:
        targets = [_descriptor(path) for path in selected]
    for target in targets:
        _original_configuration_bytes(target, destination, receipt)
    return dict(format=RAW_CONFIGURATION_FORMAT, run_id=run,
                source_configuration=_identity(configuration),
                apply_configurations=targets, current_configuration=targets[-1],
                target_model=model_binding, shape_basis_certificate=basis_binding)


def _configuration_bundle(series, run, configuration, *, destination=None, receipt=None,
                          project_root=None, revision=None):
    """Bind exact declared config paths, never a basename or path substring."""
    manifest_path = series / "series.toml"
    manifest_bytes = _small(manifest_path)
    manifest = tomllib.loads(manifest_bytes.decode("utf-8-sig"))
    table, paths = manifest.get("runs") or {}, manifest.get("paths") or {}
    ids = table.get("ids") or []
    if run not in ids or len(ids) != len(set(ids)):
        raise ValueError("The corrected export run is not uniquely registered in this series.")
    bundles = {}
    for key in ids:
        _run_name(key)
        values = (
            (table.get("config_files") or {}).get(key) or f"{paths.get('run_configs', 'config/runs')}/{key}.toml",
            (table.get("merged_files") or {}).get(key) or f"{paths.get('merged_configs', 'config/merged')}/{key}.toml",
        )
        own = []
        for value in values:
            item = _storage_path(value, series)
            if item is None or not item.resolve().is_relative_to(series.resolve()):
                raise ValueError("A corrected configuration bundle escapes the selected series.")
            if not any(_same(item, other) for other in own):
                own.append(str(item))
        bundles[key] = own
    selected = bundles[run]
    dual_bindings = table.get("geometry_configuration_bindings", {})
    if not isinstance(dual_bindings, dict):
        raise ValueError("Selected-run geometry configuration bindings must be an explicit table.")
    dual = dual_bindings.get(run)
    raw = _raw_configuration_bundle(series, run, configuration, selected,
        project_root=project_root, revision=revision, destination=destination, receipt=receipt)
    if raw is None and dual is None and not any(_same(configuration["path"], item) for item in selected):
        raise ValueError("The bound original configuration is not this run's declared configuration.")
    if any(_same(item, other) for key, rows in bundles.items() if key != run for item in selected for other in rows):
        raise ValueError("The selected corrected configuration bundle is ambiguous between runs.")
    common = _storage_path(paths.get("common_config", "config/common.toml"), series)
    if common is not None and any(_same(common, item) for rows in bundles.values() for item in rows):
        raise ValueError("The common configuration is ambiguous with a selected-run bundle.")
    common_binding = _descriptor(common) if common is not None and common.is_file() else None
    if (receipt is not None and common_binding is not None
            and common_binding != (receipt.get("binding") or {}).get("common_configuration")):
        from .painted_setup_apply import original_common_binding
        common_binding = original_common_binding(common, receipt, destination)
    result = dict(series_root=str(series), series_manifest=_descriptor(manifest_path),
                  configuration_bundles=bundles, common_configuration=common_binding)
    if raw is not None:
        result["dual_configuration_binding"] = raw
    elif dual is not None:
        if isinstance(dual,dict) and dual.get("format")==PRIVATE_MANUAL_CONFIGURATION_FORMAT:
            result["dual_configuration_binding"]=_private_manual_configuration_binding(
                dual,run,configuration,selected,series=series,project_root=project_root,
                revision=revision,destination=destination,receipt=receipt)
        elif isinstance(dual, dict) and dual.get("format") == CONFIGURATION_BINDING_V2_FORMAT:
            result["dual_configuration_binding"] = _dual_configuration_binding_v2(
                dual, run, configuration, selected, series=series, project_root=project_root,
                revision=revision, destination=destination, receipt=receipt)
        else:
            result["dual_configuration_binding"] = _dual_configuration_binding(
                dual, run, configuration, selected, destination=destination, receipt=receipt)
    return result


def _verify_export_files(receipt, destination):
    if set(receipt.get("files") or {}) != EXPORT_FILES:
        raise ValueError("The corrected export must contain exactly its five canonical named files.")
    for name, row in receipt["files"].items():
        if not _same(row["path"], destination / name):
            raise ValueError("A corrected export file belongs to another destination.")
        _verified(row)
    binding = receipt["binding"]
    for name, source in ((NAMES["labels"], binding["sources"]["labels"]),
                         (NAMES["facies"], binding["sources"]["facies"]),
                         ("facies_properties.csv", binding["properties"])):
        row = receipt["files"][name]
        if row["sha256"] != source["sha256"] or row["size_bytes"] != source["size_bytes"]:
            raise ValueError("The corrected export byte copies do not match their revision sources.")


def _apply_completion(destination, receipt):
    """Read one immutable successful application, refusing any partial claim."""
    complete = destination / "apply-complete.json"
    claim = destination / "apply-claim.json"
    if not complete.exists():
        if claim.exists():
            raise ValueError("An earlier corrected Apply is incomplete; all partial artifacts were preserved.")
        return None
    value = json.loads(_small(complete))
    if (value.get("format") != "darsia-corrected-setup-apply-complete-v1"
            or value.get("complete") is not True
            or value.get("run_id") != receipt["run_id"]
            or value.get("revision_id") != receipt["revision_id"]
            or value.get("export") != _descriptor(destination / "export.json")
            or value.get("input_binding") != receipt["binding"]):
        raise ValueError("An existing corrected Apply completion has a different identity; it was preserved.")
    claim_value = json.loads(_small(claim))
    if (claim_value.get("format") != "darsia-corrected-setup-apply-claim-v1"
            or claim_value.get("run_id") != receipt["run_id"]
            or claim_value.get("revision_id") != receipt["revision_id"]
            or claim_value.get("export") != value["export"]
            or claim_value.get("input_binding") != value["input_binding"]
            or not _same(Path(value["attempt_receipt"]["path"]).parent, claim_value.get("attempt_directory", ""))
            or Path(value["attempt_receipt"]["path"]).name != "receipt.json"
            or not Path(value["attempt_receipt"]["path"]).resolve().is_relative_to(destination.resolve())):
        raise ValueError("The corrected Apply completion does not match its preserved exclusive claim.")
    attempt = _verified(value["attempt_receipt"])
    record = json.loads(_small(attempt["path"]))
    if (record.get("state") != "complete" or record.get("run_id") != receipt["run_id"]
            or record.get("revision_id") != receipt["revision_id"]):
        raise ValueError("The preserved corrected Apply record is not complete for this run.")
    from .painted_setup_apply import completion_context, is_common_target, verify_applied_postimage
    facies_context = completion_context(value, claim_value, record, receipt=receipt)
    files = value.get("configurations")
    if not isinstance(files, list) or not files or len(files) != len(record.get("files") or []):
        raise ValueError("The corrected Apply completion has no exact configuration bundle.")
    for index, row in enumerate(files):
        old = record["files"][index]
        if (old.get("state") != "applied" or not _same(old["path"], row["after"]["path"])
                or old.get("before_sha256") != row["before"]["sha256"]
                or old.get("after_sha256") != row["after"]["sha256"]
                or not _same(row["before"]["path"], Path(attempt["path"]).parent / f"{index:02d}-before.toml")
                or not (any(_same(row["after"]["path"], p) for p in receipt["binding"]["configuration_bundles"][receipt["run_id"]])
                        or facies_context is not None and is_common_target(row["after"]["path"], receipt))):
            raise ValueError("The corrected Apply completion does not bind its actual before/after files.")
        _verified(row["before"])
        _verified(row["after"])
        if facies_context is not None:
            verify_applied_postimage(_small(row["before"]["path"]), _small(row["after"]["path"]),
                facies_context, receipt=receipt, target=row["after"]["path"])
        elif "native_product" in receipt["binding"] or "analysis_exclusions" in receipt["binding"]:
            from .painted_setup_apply import setup_native_rig
            expected = corrected_config_bytes(_small(row["before"]["path"]), receipt["binding"]["replacements"],
                native_rig=setup_native_rig(receipt["binding"]))
            if _small(row["after"]["path"]) != expected:
                raise ValueError("The applied native contract differs from its exact export-bound transformation")
    return value


def _original_configuration_bytes(configuration, destination, receipt=None):
    """Keep the source SHA; only a proved prior Apply may supply its backup."""
    before = _small(configuration["path"])
    if hashlib.sha256(before).hexdigest() == configuration["sha256"] and len(before) == configuration["size_bytes"]:
        return before
    if receipt is not None:
        completed = _apply_completion(destination, receipt)
        for row in (completed or {}).get("configurations", []):
            if (_same(row["after"]["path"], configuration["path"])
                    and row["before"]["sha256"] == configuration["sha256"]
                    and row["before"]["size_bytes"] == configuration["size_bytes"]):
                return _small(row["before"]["path"])
    raise ValueError(f"Corrected setup source identity mismatch: {configuration['path']}")


def _export_configuration_bytes(configuration, destination, receipt=None, *, bundle=None):
    """Read current RAW settings while retaining the map's historical source identity.

    The registered RAW bundle already binds the exact selected destinations and
    its model certificate. Historical geometry descriptors remain unchanged;
    the ordinary source reader still refuses every unproved live source edit.
    """
    binding = bundle if bundle is not None else ((receipt or {}).get("binding") or {})
    dual = binding.get("dual_configuration_binding") or {}
    selected = configuration
    if dual.get("format") == RAW_CONFIGURATION_FORMAT:
        targets = dual.get("apply_configurations")
        if (not _typed_equal(dual.get("source_configuration"), configuration)
                or not isinstance(targets, list) or len(targets) != 2
                or not _typed_equal(dual.get("current_configuration"), targets[-1])):
            raise ValueError("The registered RAW source does not match its selected configuration bundle")
        selected = targets[-1]
    return _original_configuration_bytes(selected, destination, receipt)


def _verify_immutable_export_inputs(receipt, *, common_after=None):
    """Source data that Apply never modifies; configuration targets are separate."""
    data = receipt["binding"]
    for item in (*data["sources"].values(), data["properties"], data["configured_depth_source"], data["series_manifest"]):
        _verified(item)
    if data.get("common_configuration") is not None:
        if common_after is not None:
            if _small(data["common_configuration"]["path"]) != common_after:
                raise RuntimeError("The common configuration differs from this transaction's expected-after bytes")
        else:
            try:
                _verified(data["common_configuration"])
            except ValueError:
                from .painted_setup_apply import original_common_binding
                destination = Path(receipt["files"][NAMES["labels"]]["path"]).parent
                original_common_binding(data["common_configuration"]["path"], receipt, destination)
    if data.get("dual_configuration_binding") is not None:
        dual = data["dual_configuration_binding"]
        # RAW configurations are mutable transaction targets. Apply checks their
        # exact postimages before completion; reuse checks its completed receipt.
        # Looking up that completion here would reject our own in-flight claim.
        if dual.get("format") != RAW_CONFIGURATION_FORMAT:
            _verified(data["configuration"])
        if dual["format"] in (CONFIGURATION_BINDING_V2_FORMAT,PRIVATE_MANUAL_CONFIGURATION_FORMAT):
            for before in dual["pre_pointer_configurations"]:
                _verified(before)
    if data.get("registered_geometry") is not None:
        from .geometry_raster_sources import verify_registered_sources
        verify_registered_sources(data["registered_geometry"])


def _verify_reusable_export(receipt, destination, *, revision, run, display, planned, current_sha, binding=None):
    active = _active_setup_export_scope()
    scope = receipt.get("setup_operation")
    if active is not None and scope != active[0]:
        raise ValueError("The corrected export belongs to another Setup job")
    if scope is not None:
        plan = active[1] if active is not None else None
        scoped_root = _setup_scope_root(scope, revision, plan=plan)
        expected = scoped_root if run == display else scoped_root / "runs" / run
        if not _same(expected, destination):
            raise ValueError("The corrected export lies outside its job-owned output")
    if (receipt.get("schema") != 1 or receipt.get("format") != FORMAT
            or receipt.get("revision_id") != revision or receipt.get("run_id") != run
            or receipt.get("frame") != "rig-corrected" or receipt.get("reference_run") != display
            or receipt.get("current_sha256") != current_sha
            or (binding is not None and receipt.get("binding") != binding)
            or (run == display and receipt.get("runs") != planned)):
        raise ValueError("An existing corrected export has a different identity; it was preserved.")
    _verify_export_files(receipt, destination)
    _verify_immutable_export_inputs(receipt)
    data = receipt["binding"]
    _export_configuration_bytes(data["configuration"], destination, receipt)


def _verify_reference_export(root, *, revision, display, planned, current_sha, child=None):
    """One shared root receipt/files/linkage check for child export/Apply/Build."""
    path = root / "export.json"
    if not path.is_file():
        raise ValueError("Export the display-reference run before a canonical selected-run child.")
    root_receipt = json.loads(_small(path))
    _verify_reusable_export(root_receipt, root, revision=revision, run=display,
                            display=display, planned=planned, current_sha=current_sha)
    root_binding = _descriptor(path)
    if child is not None and child.get("root_export") != root_binding:
        raise ValueError("The selected child is bound to another reference export; it was preserved.")
    return root_binding


def _require_painted_facies_documents(config, paths):
    """Check only explicit new-table run/common inputs before Apply or Build."""
    if "painted_adoption_table" not in config:
        return
    from .painted_facies_inputs import require_current_facies_inputs
    documents = {}
    for path in paths:
        path = Path(path)
        # Some series declare an optional merged file before its first build.
        # Every actual required path is still checked by the original guards.
        if path.is_file() and str(path) not in documents:
            documents[str(path)] = tomllib.loads(_small(path).decode("utf-8-sig"))
    require_current_facies_inputs(config, documents)


def require_corrected_export_for_build(paths):
    """Admit only this exact declared selected-run bundle and canonical export."""
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    labeling = _get_section_from_toml(paths, "labeling")
    facies = _get_section_from_toml(paths, "facies")
    labels_path = Path(labeling["labels"])
    destination = labels_path.parent
    receipt = json.loads(_small(destination / "export.json"))
    run, revision, display = _run_name(receipt.get("run_id")), receipt.get("revision_id"), _run_name(receipt.get("reference_run"))
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{32}", revision) is None:
        raise ValueError("Corrected Build requires the genuine revision UUID.")
    root = destination if run == display else destination.parent.parent
    expected = root if run == display else root / "runs" / run
    if (not _same(destination, expected) or root.name != revision
            or root.parent.name != "native_setup" or root.parent.parent.name != "exports"):
        raise ValueError("Corrected Build requires this run's canonical export location.")
    for name, value in ((NAMES["labels"], labeling["labels"]), (NAMES["facies"], facies["path"]), ("facies_properties.csv", facies["props"])):
        if not _same(value, destination / name):
            raise ValueError("Corrected Build inputs belong to different run exports.")
    source_project = root.parent.parent.parent
    scope = receipt.get("setup_operation")
    if scope is not None:
        active = _active_setup_export_scope()
        scoped_root = _setup_scope_root(scope, revision,
            plan=active[1] if active is not None and active[0] == scope else None)
        if not _same(root, scoped_root):
            raise ValueError("Corrected Build export lies outside its original Setup job")
        source_project = Path(scope["project_root"])
    current_path = source_project / "overrides/geometry_facies/current.json"
    current_bytes = _small(current_path)
    current = json.loads(current_bytes)
    config = current["configuration"]
    if "painted_adoption_table" in config:
        checked_paths = [paths] if isinstance(paths, Path) else list(paths)
        bound_inputs = receipt.get("binding") or {}
        checked_paths.extend((bound_inputs.get("configuration_bundles") or {}).get(run) or [])
        common_input = bound_inputs.get("common_configuration")
        if common_input is not None:
            checked_paths.append(common_input["path"])
        _require_painted_facies_documents(config, checked_paths)
    actual_root, selected, recorded_display, planned = _canonical_location(source_project, config, revision, run, setup_scope=scope)
    if current.get("revision_id") != revision or not _same(selected, destination) or recorded_display != display:
        raise ValueError("Corrected Build export belongs to another current revision or selected run.")
    registered = _registered_binding(source_project, current, config, run)
    if registered is None:
        _require_unedited_adoption(current, config)
    _verify_reusable_export(receipt, destination, revision=revision, run=run, display=display,
                            planned=planned, current_sha=hashlib.sha256(current_bytes).hexdigest())
    binding = receipt["binding"]
    saved = config["frame_binding"]["runs"][run]
    if ("analysis_exclusions" in config or "target_corrections" in config
            or "local_map_reuse" in (config.get("provenance") or {})):
        from .painted_analysis_exclusions import adopted_run_native_product
        expected_native = adopted_run_native_product(current, run,
            project_root=source_project, registered_sources=registered)
        if binding.get("native_product") != expected_native:
            raise ValueError("Corrected Build lost its adopted native product contract")
    elif "native_product" in binding:
        raise ValueError("Corrected Build cannot invent a native product policy")
    if "analysis_exclusions" in config:
        from .painted_analysis_exclusions import adopted_run_analysis_exclusions
        excluded = adopted_run_analysis_exclusions(current, run, project_root=source_project, registered_sources=registered)
        if binding.get("analysis_exclusions") != excluded:
            raise ValueError("Corrected Build lost its exact adopted analysis exclusions")
    elif "analysis_exclusions" in binding:
        raise ValueError("Corrected Build cannot invent analysis exclusions")
    expected_sources = {name: _identity(saved["fields"][name])
                        for name in ("shape_corrected_baseline", "labels", "facies", "depth")}
    if registered is not None:
        expected_sources.update({name: registered["files"][name] for name in ("labels", "facies")})
        if binding.get("registered_geometry") != registered:
            raise ValueError("Corrected Build lacks this genuine registered Store binding.")
    elif "registered_geometry" in binding:
        raise ValueError("An unedited adoption cannot claim a registered Store binding.")
    if (not _same_source(binding["configuration"], _identity(saved["configuration"]))
            or not _same_source(binding["properties"], _revision_properties(config, saved, registered))
            or not _same_source(binding["configured_depth_source"], _identity(saved["configured_depth_map"]))
            or binding["replacements"] != _replacements(destination)
            or any(not _same_source(binding["sources"][name], expected_sources[name])
                   for name in ("shape_corrected_baseline", "labels", "facies", "depth"))):
        raise ValueError("Corrected Build export inputs differ from this run's current revision binding.")
    if run != display:
        _verify_reference_export(actual_root, revision=revision, display=display,
                                 planned=planned, current_sha=hashlib.sha256(current_bytes).hexdigest(), child=receipt)
    series = Path(binding["series_root"])
    declared = _configuration_bundle(series, run, binding["configuration"], destination=destination, receipt=receipt,
                                     project_root=source_project, revision=revision)
    if any(binding.get(key) != value for key, value in declared.items()):
        raise ValueError("Corrected Build configuration bundle changed since export.")
    actual_paths = [paths] if isinstance(paths, Path) else paths
    selected_paths = binding["configuration_bundles"][run]
    common = binding.get("common_configuration")
    allowed = selected_paths + ([common["path"]] if common is not None else [])
    if (not actual_paths or not any(_same(p, q) for p in actual_paths for q in selected_paths)
            or any(not any(_same(p, q) for q in allowed) for p in actual_paths)):
        raise ValueError("Corrected export belongs to a different or ambiguous selected-run configuration bundle.")
    # Actual selected config bytes must either be their original bound source,
    # or the exact output of the immutable successful Apply record.
    completion = _apply_completion(destination, receipt)
    if completion is None:
        raise ValueError("Apply this run's bound corrected export before Build run model.")
    for path in actual_paths:
        if common is not None and _same(path, common["path"]):
            if completion.get("facies_rebuild") is None:
                _verified(common)
            else:
                from .painted_setup_apply import original_common_binding
                original_common_binding(path, receipt, destination)
        elif not any(_same(path, row["after"]["path"]) for row in completion["configurations"]):
            raise ValueError("The selected run configuration is not an actual completed Apply target.")
    if registered is not None:
        from .geometry_raster_sources import verify_registered_sources
        verify_registered_sources(registered)
    dual = binding.get("dual_configuration_binding")
    if dual is not None and dual["format"] == CONFIGURATION_BINDING_V2_FORMAT:
        _require_fresh_v2_build(paths, dual)
    elif dual is not None and dual["format"]==PRIVATE_MANUAL_CONFIGURATION_FORMAT:
        _require_fresh_private_manual_build(paths,dual)
    return receipt



def _registered_binding(project_root, current, config, run):
    from .setup_source_reads import reuse_registered_binding, setup_source_read_group
    with setup_source_read_group():
        return reuse_registered_binding(project_root, current, config, run,
                                        reader=_registered_binding_read)


def _registered_binding_read(project_root, current, config, run):
    if "manual_label_edit" in config:
        from .geometry_raster_sources import private_manual_label_edit_sources
        return private_manual_label_edit_sources(project_root, current, run)
    if config.get("source_mode") != "canonical-from-import":
        return None
    from .geometry_raster_sources import registered_run_sources
    return registered_run_sources(project_root, current, run)

def export_corrected_setup(project, *, reference, actor=""):
    """Explicit export only: full identity checks and byte copies, no Build."""
    if reference is None:
        raise ValueError("Choose the run whose corrected maps are to be exported.")
    run = _run_name(reference.run_id)
    current_path = GeometryRevisionStore(project).current_path
    before = _small(current_path)
    current = json.loads(before)
    if current.get("format") != GEOMETRY_REVISION_FORMAT or current.get("schema") != 1:
        raise ValueError("A saved original-schema workspace revision is required.")
    config = current["configuration"]
    if frame_kind(config) != "rig-corrected":
        raise ValueError("This export is only for an explicit corrected-frame revision.")
    revision = current.get("revision_id")
    if not isinstance(revision, str) or len(revision) != 32 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Corrected setup requires the original Store UUID.")
    fields = selected_fields(config, run)
    export_root, destination, display, planned = _canonical_location(project.root, config, revision, run)
    current_sha = hashlib.sha256(before).hexdigest()
    existing = json.loads(_small(destination / "export.json")) if destination.exists() else None
    root_binding = None
    if run != display:
        # A child never manufactures or mutates the ordinary reference export.
        root_binding = _verify_reference_export(export_root, revision=revision, display=display,
                                                planned=planned, current_sha=current_sha, child=existing)
    row = config["frame_binding"]["runs"][run]
    configuration = _identity(row["configuration"])
    series = Path(native_series_link(project)["series_root"])
    bundle = _configuration_bundle(series, run, configuration, destination=destination, receipt=existing,
                                   project_root=project.root, revision=revision)
    original = tomllib.loads(_export_configuration_bytes(
        configuration, destination, existing, bundle=bundle).decode("utf-8-sig"))
    depth_pointer = (original.get("depth") or {}).get("depth_map")
    configured_depth_path = _storage_path(depth_pointer, series)
    if configured_depth_path is None or not _same(
        configured_depth_path.resolve(), Path(row["configured_depth_map"]["path"]).resolve()
    ):
        raise ValueError("The original configured depth pointer changed.")
    # Preserve the selected run's full original table authority for ordinary
    # revisions. The explicit painted contract below selects its verified full
    # derived table, including all preserved source columns.
    properties = None
    if "painted_adoption_table" not in config:
        props_path = _storage_path((original.get("facies") or {}).get("props"), series)
        if props_path is None:
            raise ValueError("The selected run has no original facies property table.")
        _small(props_path)
        properties_binding = row.get("facies_properties")
        if not isinstance(properties_binding, dict) or not _same(properties_binding.get("path", ""), props_path):
            raise ValueError("The saved revision must bind this run's original facies property table and SHA.")
        properties = _verified(properties_binding)
    # Resolve cheap configuration mismatches before replaying any raster proof.
    # An adopted painting owns its derived table; the previous configuration's
    # properties pointer is a destination input that Apply replaces explicitly.
    registered = _registered_binding(project.root, current, config, run)
    if registered is None:
        _require_unedited_adoption(current, config)
    if "painted_adoption_table" in config:
        properties = _verified(_revision_properties(config, row, registered))
    configured_depth_source = _verified(row["configured_depth_map"])
    sources = {name: _verified(fields[name]) for name in ("shape_corrected_baseline", "labels", "facies", "depth")}
    if registered is not None:
        sources.update({name: registered["files"][name] for name in ("labels", "facies")})
    native_requested = ("analysis_exclusions" in config or "target_corrections" in config
                        or "local_map_reuse" in (config.get("provenance") or {}))
    native_product = None
    if native_requested:
        from .painted_analysis_exclusions import adopted_run_native_product
        native_product = adopted_run_native_product(current, run,
            project_root=project.root, registered_sources=registered)
    from .archive_grid import require_archive_grids
    if not require_archive_grids(sources, native_grid=(native_product or {}).get("native_grid")):
        import darsia
        baseline = darsia.imread(sources["shape_corrected_baseline"]["path"])
        for name in ("labels", "facies", "depth"):
            _require_exact_grid(Path(sources[name]["path"]), baseline, name)
    replacements = _replacements(destination)
    text = "\n".join(
        f"[{section}]\n" + "\n".join(f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in values.items())
        for section, values in replacements.items()
    ) + "\n"
    binding = dict(configuration=configuration, sources=sources, properties=properties,
                   configured_depth_source=configured_depth_source,
                   configured_depth_pointer=depth_pointer, replacements=replacements,
                   **bundle)
    if registered is not None:
        binding["registered_geometry"] = registered
    if native_requested:
        binding["native_product"] = native_product
    if "analysis_exclusions" in config:
        from .painted_analysis_exclusions import adopted_run_analysis_exclusions
        excluded = adopted_run_analysis_exclusions(current, run, project_root=project.root, registered_sources=registered)
        if excluded is not None:
            binding["analysis_exclusions"] = excluded
    from .painted_setup_apply import setup_native_rig, native_rig_toml
    native_rig = setup_native_rig(binding)
    if native_rig is not None:
        text += "[rig]\n" + "\n".join(f"{key} = {json.dumps(value, ensure_ascii=False)}"
            for key, value in native_rig_toml(native_rig).items()) + "\n"
    receipt_path = destination / "export.json"
    if existing is not None:
        receipt = existing
        _verify_reusable_export(receipt, destination, revision=revision, run=run,
                                display=display, planned=planned, current_sha=current_sha, binding=binding)
        if receipt.get("root_export") != root_binding:
            raise ValueError("The selected child is bound to another reference export; it was preserved.")
        if _small(current_path) != before:
            raise RuntimeError("The workspace revision changed during corrected export.")
        return CorrectedSetupExport(revision, run, destination, text, current_path, before, receipt, True)
    _mkdir_output(destination.parent, parents=True, exist_ok=True)
    staging = destination.with_name(f".{run}.{uuid.uuid4().hex}.tmp")
    _mkdir_output(staging)
    # Failed staging is intentionally preserved. No existing export is removed.
    for name, output_name in NAMES.items():
        _copy_output(sources[name]["path"], staging / output_name)
        copied = _descriptor(staging / output_name)
        if copied["sha256"] != sources[name]["sha256"]:
            raise RuntimeError("Corrected raster copy changed bytes.")
    _copy_output(properties["path"], staging / "facies_properties.csv")
    if _descriptor(staging / "facies_properties.csv")["sha256"] != properties["sha256"]:
        raise RuntimeError("Original facies property table copy changed bytes.")
    _exclusive(staging / "native_setup.toml", text.encode("utf-8"))
    # Exactly the existing native exporter colour renderer; presentation only.
    from PIL import Image
    labels = _load_array(Path(sources["labels"]["path"]))
    color_index = native_color_index(labels)
    palette = np.asarray([(255, 255, 255)] + [rgb for _name, rgb in NATIVE_LABEL_COLORS], dtype=np.uint8)
    _save_output_png(Image.fromarray(palette[color_index]), staging / "colored_labels.png")
    files = {}
    for name in sorted(EXPORT_FILES):
        files[name] = {**_descriptor(staging / name), "path": str(destination / name)}
    receipt = dict(schema=1, format=FORMAT, revision_id=revision, run_id=run, frame="rig-corrected",
                   reference_run=display, source_reference_run=config["frame_binding"]["source_reference_run"],
                   root_export=root_binding,
                   created_utc=utc_now(), created_by=str(actor), binding=binding, files=files,
                   current_sha256=hashlib.sha256(before).hexdigest(), lossless_rebuilt_gate="NOT RUN",
                   fields_are_original_byte_copies=True, actual_Build_run=False)
    scope = _active_setup_export_scope()
    if scope is not None:
        receipt["setup_operation"] = deepcopy(scope[0])
    if run == display:
        receipt.update(runs=planned, runs_are_planned_paths_only=True, child_completion_not_claimed=True)
    _exclusive(staging / "export.json", (json.dumps(receipt, ensure_ascii=False, indent=2)+"\n").encode())
    if _small(current_path) != before:
        raise RuntimeError("The workspace revision changed during corrected export.")
    selected_fields(config, run)
    _export_configuration_bytes(configuration, destination, existing, bundle=bundle)
    if registered is not None:
        from .geometry_raster_sources import verify_registered_sources
        verify_registered_sources(registered)
    if binding.get("dual_configuration_binding") is not None:
        final_bundle = _configuration_bundle(series, run, configuration, destination=destination,
                                            project_root=project.root, revision=revision)
        if any(binding.get(key) != value for key, value in final_bundle.items()):
            raise RuntimeError("The current dual configuration bundle changed during corrected export.")
    _move_output(staging, destination)
    return CorrectedSetupExport(revision, run, destination, text, current_path, before, receipt)


def corrected_config_bytes(before: bytes, replacements: dict, *, native_rig=None) -> bytes:
    """Keep the original four-line path; add only explicit bound native strings."""
    original = tomllib.loads(before.decode("utf-8-sig"))
    if set(replacements) != {"labeling", "facies"} or set(replacements["labeling"]) != {"labels", "frame"} or set(replacements["facies"]) != {"path", "props"} or replacements["labeling"]["frame"] != "rig-corrected":
        raise ValueError("Only the four explicit corrected setup leaves may be applied.")
    clear_exclusion = False
    if native_rig is not None:
        from .painted_setup_apply import native_rig_toml, clears_analysis_exclusions
        clear_exclusion = clears_analysis_exclusions(native_rig)
        replacements = dict(replacements, rig=native_rig_toml(native_rig))
    expected = deepcopy(original)
    if clear_exclusion:
        expected.setdefault("rig", {}).pop("analysis_exclusions", None)
    bom = b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b""
    text = before[len(bom):].decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    if clear_exclusion:
        # This retires one owned metadata value, never any approval/mask file.
        # Historical native_rig=None never enters this text transformation.
        retained, section, omitted = [], None, False
        for line in lines:
            header = re.match(r"^\s*\[\[?([^\[\]]+)\]\]?\s*(?:#.*)?$", line.rstrip("\r\n"))
            if header:
                section = header.group(1).strip()
                omitted = section == "rig.analysis_exclusions" or section.startswith("rig.analysis_exclusions.")
            if omitted or (section == "rig" and re.match(r"^\s*analysis_exclusions\s*=", line)):
                continue
            retained.append(line)
        lines = retained
    for section, updates in replacements.items():
        expected.setdefault(section, {}).update(updates)
        headers = [(i, re.match(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$", line.rstrip("\r\n"))) for i, line in enumerate(lines)]
        starts = [i for i, m in headers if m and m.group(1).strip() == section]
        if len(starts) > 1:
            raise ValueError("Duplicate setup table.")
        if not starts:
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += newline
            lines += [newline, f"[{section}]{newline}"]
            start = len(lines)-1
        else:
            start = starts[0]
        end = next((i for i in range(start+1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        for key, value in updates.items():
            found = [i for i in range(start+1, end) if re.match(rf"^\s*{re.escape(key)}\s*=", lines[i])]
            replacement = f"{key} = {json.dumps(value, ensure_ascii=False)}{newline}"
            if found:
                if len(found) != 1:
                    raise ValueError("Duplicate setup key.")
                lines[found[0]] = replacement
            else:
                if end and not lines[end-1].endswith(("\n", "\r")):
                    lines[end-1] += newline
                lines.insert(end, replacement)
                end += 1
    after = bom + "".join(lines).encode("utf-8")
    if tomllib.loads(after.decode("utf-8-sig")) != expected:
        raise ValueError("Corrected setup text patch changed unrelated configuration values.")
    return after


def apply_corrected_selected_run(project, export: CorrectedSetupExport, *, rebuild_facies_by_name=False,
                                 raw_setup_settings=None):
    """Selected-run CAS, with an explicit adopted-name rebuild before Build.

    The default changes only the original four pointers. The positive rebuild
    also updates the bound common inputs on the display-reference transaction.
    Explicit RAW settings share that transaction for selected-run targets only.
    Neither path runs registration, Build, calibration or adoption.
    """
    if raw_setup_settings is not None and not rebuild_facies_by_name:
        raise ValueError("RAW Setup settings require the explicit by-name configuration rebuild")
    root = Path(native_series_link(project)["series_root"])
    manifest_path = root / "series.toml"
    manifest_bytes = _small(manifest_path)
    manifest = tomllib.loads(manifest_bytes.decode("utf-8-sig"))
    if export.run_id not in (manifest.get("runs") or {}).get("ids", []):
        raise ValueError("The corrected export run is not in the selected series.")
    if _small(export.current_json) != export.current_bytes:
        raise RuntimeError("The current workspace revision changed; no setup was applied.")
    receipt_path = export.destination / "export.json"
    actual_receipt = json.loads(_small(receipt_path))
    if actual_receipt != export.receipt:
        raise RuntimeError("The corrected export receipt changed; no setup was applied.")
    current = json.loads(export.current_bytes)
    config = current["configuration"]
    facies_context = None
    if rebuild_facies_by_name:
        from .painted_setup_apply import context_for_current, raw_settings_for_run, require_raw_output
        if raw_setup_settings is not None:
            raw_setup_settings = raw_settings_for_run(raw_setup_settings, run_id=export.run_id)
            for key in ("native_product", "analysis_exclusions"):
                if key in raw_setup_settings and not _typed_equal(raw_setup_settings[key], actual_receipt["binding"].get(key)):
                    raise ValueError("RAW Setup native settings differ from their exact export binding")
            require_raw_output(raw_setup_settings, actual_receipt)
        facies_context = context_for_current(current, raw_setup_settings=raw_setup_settings)
    if "painted_adoption_table" in config:
        bound_inputs = actual_receipt.get("binding") or {}
        checked_paths = list((bound_inputs.get("configuration_bundles") or {}).get(export.run_id) or [])
        common_input = bound_inputs.get("common_configuration")
        if common_input is not None:
            checked_paths.append(common_input["path"])
        if facies_context is None:
            _require_painted_facies_documents(config, checked_paths)
        else:
            from .painted_setup_apply import require_rebuilt_inputs
            require_rebuilt_inputs(config, checked_paths, facies_context, revision_id=export.revision_id,
                                   receipt=actual_receipt)
    _export_root, expected_destination, display, planned = _canonical_location(
        project.root, config, export.revision_id, export.run_id
    )
    if not _same(export.destination, expected_destination):
        raise ValueError("The corrected export belongs to another selected-run destination.")
    _verify_reusable_export(actual_receipt, export.destination, revision=export.revision_id,
                            run=export.run_id, display=display, planned=planned,
                            current_sha=hashlib.sha256(export.current_bytes).hexdigest())
    if export.run_id != display:
        _verify_reference_export(_export_root, revision=export.revision_id, display=display,
                                 planned=planned, current_sha=hashlib.sha256(export.current_bytes).hexdigest(), child=actual_receipt)
    saved = json.loads(export.current_bytes)["configuration"]["frame_binding"]["runs"][export.run_id]
    registered = _registered_binding(project.root, current, config, export.run_id)
    expected_sources = {name: _identity(saved["fields"][name])
                        for name in ("shape_corrected_baseline", "labels", "facies", "depth")}
    binding = export.receipt["binding"]
    if ("analysis_exclusions" in config or "target_corrections" in config
            or "local_map_reuse" in (config.get("provenance") or {})):
        from .painted_analysis_exclusions import adopted_run_native_product
        native_product = adopted_run_native_product(current, export.run_id,
            project_root=project.root, registered_sources=registered)
        if binding.get("native_product") != native_product:
            raise ValueError("Corrected Apply lost its adopted native product contract")
    elif "native_product" in binding:
        raise ValueError("Corrected Apply cannot invent a native product policy")
    if "analysis_exclusions" in config:
        from .painted_analysis_exclusions import adopted_run_analysis_exclusions
        expected_exclusion = adopted_run_analysis_exclusions(current, export.run_id, project_root=project.root, registered_sources=registered)
        if binding.get("analysis_exclusions") != expected_exclusion:
            raise ValueError("Corrected Apply lost its exact unquantified native analysis mask")
    elif "analysis_exclusions" in binding:
        raise ValueError("A corrected export cannot invent analysis exclusions")
    if registered is not None:
        expected_sources.update({name: registered["files"][name] for name in ("labels", "facies")})
        if binding.get("registered_geometry") != registered:
            raise ValueError("Corrected Apply lacks this genuine registered Store binding.")
    elif "registered_geometry" in binding:
        raise ValueError("An unedited adoption cannot claim a registered Store binding.")
    if (not _same_source(binding["configuration"], _identity(saved["configuration"]))
            or not _same_source(binding["properties"], _revision_properties(config, saved, registered))
            or not _same_source(binding["configured_depth_source"], _identity(saved["configured_depth_map"]))
            or binding["replacements"] != _replacements(export.destination)):
        raise ValueError("The corrected export input identities or target paths are not bound to this revision.")
    if any(not _same_source(binding["sources"][name], expected_sources[name]) for name in ("shape_corrected_baseline", "labels", "facies", "depth")):
        raise ValueError("The corrected export field identities differ from this revision.")
    if set(export.receipt["files"]) != EXPORT_FILES:
        raise ValueError("The corrected export must contain exactly its five canonical named files.")
    for name, source in ((NAMES["labels"], expected_sources["labels"]), (NAMES["facies"], expected_sources["facies"]), ("facies_properties.csv", _revision_properties(config, saved, registered))):
        row = export.receipt["files"][name]
        if not _same(row["path"], export.destination / name) or row["sha256"] != source["sha256"] or row["size_bytes"] != source["size_bytes"]:
            raise ValueError("The corrected export byte copies do not match their revision sources.")
    for row in export.receipt["files"].values():
        _verified(row)
    _verified(export.receipt["binding"]["configured_depth_source"])
    configuration = export.receipt["binding"]["configuration"]
    _export_configuration_bytes(configuration, export.destination, export.receipt)
    for row in (*export.receipt["binding"]["sources"].values(), binding["properties"], binding["series_manifest"]):
        _verified(row)
    if binding.get("common_configuration") is not None:
        from .painted_setup_apply import original_common_binding
        original_common_binding(binding["common_configuration"]["path"], export.receipt, export.destination)
    expected_bundle = _configuration_bundle(root, export.run_id, configuration, destination=export.destination, receipt=export.receipt,
                                           project_root=project.root, revision=export.revision_id)
    if any(binding.get(key) != value for key, value in expected_bundle.items()):
        raise ValueError("The selected run's declared configuration bundle changed.")
    completed = _apply_completion(export.destination, export.receipt)
    if completed is not None:
        completed_raw = (completed.get("facies_rebuild") or {}).get("raw_setup_settings")
        if not _typed_equal(completed_raw, raw_setup_settings):
            raise ValueError("The completed application differs from the requested RAW Setup settings")
        # Pure revalidation/reuse: no new UUID, timestamp, write or replacement.
        if registered is not None:
            from .geometry_raster_sources import verify_registered_sources
            verify_registered_sources(registered)
        return Path(completed["attempt_receipt"]["path"])
    camera_before = None
    if raw_setup_settings is not None:
        from .rig_raw_input import profile_parameters
        from .pipeline.capacity import read_bound
        require_raw_output(raw_setup_settings, actual_receipt, fresh=True)
        camera = raw_setup_settings["raw_camera"]
        from .storage_paths import resolve_storage_path
        camera_source = resolve_storage_path(camera["configuration"])
        _parameters, camera_before = read_bound(dict(path=camera["configuration"],
            sha256=camera["sha256"], size_bytes=camera_source.stat().st_size))
        profile_parameters(camera)
    dual = binding.get("dual_configuration_binding")
    if dual is not None:
        targets = [Path(row["path"]) for row in dual["apply_configurations"]]
    else:
        targets = [Path(configuration["path"])]
        paths = manifest.get("paths") or {}
        table = manifest.get("runs") or {}
        partial = _storage_path((table.get("config_files") or {}).get(export.run_id) or f"{paths.get('run_configs','config/runs')}/{export.run_id}.toml", root)
        merged = _storage_path((table.get("merged_files") or {}).get(export.run_id) or f"{paths.get('merged_configs','config/merged')}/{export.run_id}.toml", root)
        declared = [p for p in (partial, merged) if p is not None]
        if not any(_same(configuration["path"], p) for p in declared):
            raise ValueError("The bound original configuration is not this run's declared configuration.")
        if partial is not None and not any(_same(partial, p) for p in targets):
            targets.append(partial)
        if merged is not None and merged.is_file() and not any(_same(merged, p) for p in targets):
            targets.append(merged)
    if facies_context is not None and binding.get("common_configuration") is not None:
        from .painted_setup_apply import rebuild_bytes
        common_path = Path(binding["common_configuration"]["path"])
        common_before = _small(common_path)
        common_after, _ = rebuild_bytes(common_before, facies_context, revision_id=export.revision_id,
                                        is_common=True)
        if common_after != common_before:
            if export.run_id != display:
                raise ValueError("Apply the display-reference run's by-name configurations first")
            if not any(_same(common_path, path) for path in targets):
                targets.insert(0, common_path)
    for path in targets:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Corrected setup target is outside the selected series.")
    bindings = [ReadBinding(manifest_path, manifest_bytes), ReadBinding(export.current_json, export.current_bytes), ReadBinding(receipt_path, _small(receipt_path))]
    if facies_context is not None:
        for bound in (facies_context["revision"], facies_context["parent_csv"]):
            path = Path(bound["path"])
            _verified(bound)
            bindings.append(ReadBinding(path, _small(path)))
    if camera_before is not None:
        bindings.append(ReadBinding(Path(raw_setup_settings["raw_camera"]["configuration"]), camera_before))
    changes = []
    for path in targets:
        before = _small(path)
        bindings.append(ReadBinding(path, before))
        if facies_context is None:
            from .painted_setup_apply import setup_native_rig
            after = corrected_config_bytes(before, export.receipt["binding"]["replacements"],
                native_rig=setup_native_rig(binding))
        else:
            from .painted_setup_apply import rebuild_bytes, is_common_target, setup_native_rig
            common = is_common_target(path, export.receipt)
            replacements = None if common else binding["replacements"]
            after, _ = rebuild_bytes(before, facies_context, revision_id=export.revision_id,
                                      replacements=replacements, is_common=common,
                                      native_rig=None if common else setup_native_rig(binding))
        changes.append((path, before, after))
    for item in bindings:
        item.require_unchanged()
    attempt = export.destination / ("apply-" + uuid.uuid4().hex)
    _mkdir_output(attempt)
    _exclusive(export.destination / "apply-claim.json", (json.dumps(dict(
        format="darsia-corrected-setup-apply-claim-v1", run_id=export.run_id,
        revision_id=export.revision_id, attempt_directory=str(attempt),
        export=_descriptor(receipt_path), input_binding=binding,
        **({"facies_rebuild": facies_context} if facies_context is not None else {}),
    ), ensure_ascii=False, indent=2)+"\n").encode())
    files, staged = [], []
    receipt = dict(format="darsia-corrected-setup-apply-v1", run_id=export.run_id,
                   revision_id=export.revision_id, created_utc=utc_now(), state="prepared", files=files,
                   crop_changed=False, depth_pointer_changed=False, registration_run=False, Build_run=False)
    if facies_context is not None:
        receipt["facies_rebuild"] = facies_context
    try:
        for i, (path, before, after) in enumerate(changes):
            _exclusive(attempt / f"{i:02d}-before.toml", before)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            _exclusive(temporary, after)
            staged.append((path, temporary))
            files.append(dict(path=str(path), before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(), state="staged"))
        applied = set()
        for i, (path, temporary) in enumerate(staged):
            for item in bindings:
                if str(item.path) not in applied:
                    item.require_unchanged()
            _move_output(temporary, path, replace=True)
            applied.add(str(path))
            files[i]["state"] = "applied"
        _verify_export_files(export.receipt, export.destination)
        if facies_context is None:
            _verify_immutable_export_inputs(export.receipt)
        else:
            from .painted_setup_apply import is_common_target
            common_after = next((after for path, _before, after in changes
                                 if is_common_target(path, export.receipt)), None)
            _verify_immutable_export_inputs(export.receipt, common_after=common_after)
        if registered is not None:
            from .geometry_raster_sources import verify_registered_sources
            verify_registered_sources(registered)
        if export.run_id != display:
            _verify_reference_export(_export_root, revision=export.revision_id, display=display,
                                     planned=planned, current_sha=hashlib.sha256(export.current_bytes).hexdigest(), child=actual_receipt)
        for item in bindings:
            if not any(_same(item.path, path) for path, _before, _after in changes):
                item.require_unchanged()
        # Recheck the entire planned result, including targets already replaced.
        # An observed external edit is a failure, never the new expected output.
        for index, (path, before, after) in enumerate(changes):
            if _small(path) != after:
                raise RuntimeError("Corrected Apply target differs from its expected-after bytes: " + str(path))
            if _small(attempt / f"{index:02d}-before.toml") != before:
                raise RuntimeError("A preserved corrected Apply before-copy changed.")
        receipt["state"] = "complete"
    except Exception as error:
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        _exclusive(attempt / "receipt.json", (json.dumps(receipt, indent=2)+"\n").encode())
        raise CorrectedApplyError(str(error), attempt / "receipt.json") from error
    _exclusive(attempt / "receipt.json", (json.dumps(receipt, indent=2)+"\n").encode())
    complete = dict(format="darsia-corrected-setup-apply-complete-v1", complete=True,
                    run_id=export.run_id, revision_id=export.revision_id,
                    export=_descriptor(receipt_path), input_binding=binding,
                    attempt_receipt=_descriptor(attempt / "receipt.json"),
                    configurations=[dict(before=dict(path=str(attempt / f"{index:02d}-before.toml"),
                                                     sha256=hashlib.sha256(before).hexdigest(), size_bytes=len(before)),
                                         after=dict(path=str(path), sha256=hashlib.sha256(after).hexdigest(), size_bytes=len(after)))
                                    for index, (path, before, after) in enumerate(changes)])
    if facies_context is not None:
        complete["facies_rebuild"] = facies_context
    _exclusive(export.destination / "apply-complete.json", (json.dumps(complete, ensure_ascii=False, indent=2)+"\n").encode())
    return attempt / "receipt.json"
