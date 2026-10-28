"""Discover completed, immutable mass results without repeating preparation.

This is a delivery index, not a second numerical cache or calculation method.
Published input/output checksums are verified once; unchanged filesystem stamps
allow the viewer to reopen a snapshot without rereading gigabytes from the NAS.
Changed files are always rehashed. Lookup belongs off the Qt GUI thread.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tomllib
import uuid
from darsia.fluidflower.phase_saved_runtime_views import selection, stamp
from darsia.fluidflower.phase_saved_review_metadata import configured_parameters


def _mass_policy():
    return dict(configured_parameters("mass_review_cache")["mass_policy"])


def __getattr__(name):
    if name != "DEFAULT_MASS_POLICY":
        raise AttributeError(name)
    return _mass_policy()




def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()






def index_path(request, method):
    root = request["candidate"].get("phase_map_output_dir")
    if not root:
        return None
    key = hashlib.sha256(json.dumps(selection(request, method), sort_keys=True).encode()).hexdigest()
    return Path(root)/"mass_results"/(key+".json")


def _selected_index(request, result):
    if request.get("reference_mode"):
        reference = request.get("reference")
        if reference is None:
            return next((i for i, row in enumerate(result["frames"]) if row["elapsed_h"] <= 0), None)
        return next((i for i, row in enumerate(result["frames"])
                     if row.get("source_name") == reference.get("source_name")
                     and abs(row["elapsed_h"]-reference["elapsed_h"]) < 1e-7), None)
    return next((i for i, row in enumerate(result["frames"])
                 if row.get("frame_index") == int(request.get("frame_index", 0))), None)


def register_result(path, request):
    path = Path(path).resolve()
    result = read(path)
    if result.get("complete") is not True:
        raise ValueError("Cannot publish an incomplete mass result")
    method = request["method"]
    expected = selection(request, method)
    actual = selection({"candidate": result["selected_inputs"]["candidate"],
                        "local_baseline_search": result["selected_inputs"].get("local_baseline_search"),
                        "mass_policy": result.get("mass_policy", _mass_policy()),
                        "analysis_width": request.get("analysis_width", 1024)}, method)
    if expected != actual or any(result["method"]["settings"].get(k) != v for k, v in method.items()):
        raise ValueError("Result differs from the requested model or concentration method")
    target = index_path(request, method)
    if target is None:
        raise ValueError("A registered phase-map output directory is required")
    bindings = {}
    def bind(file, expected_hash=None):
        file = Path(file).resolve()
        before = stamp(file)
        value = digest(file)
        if before != stamp(file) or (expected_hash and value != expected_hash):
            raise ValueError(f"Result input changed before publication: {file}")
        bindings[str(file)] = dict(sha256=value, stamp=before)
    inputs = result["selected_inputs"]
    for key in ("baseline", "native_manifest"):
        bind(inputs[key], inputs[key+"_sha256"])
    for frame in inputs["frames"]:
        for key in ("photo", "phase"):
            bind(frame[key], frame[key+"_sha256"])
    for source, source_hash in read(result["result_source_manifest"]).items():
        bind(source, source_hash)
    prepared = read(path.with_name("inputs.json"))["prepared"]
    for item in prepared.get("reference_inputs", []):
        bind(item["path"], item["sha256"])
    for source, source_hash in prepared.get("source_files", {}).items():
        bind(source, source_hash)
    # A delivered snapshot must not silently become the result of a changed
    # physical setup. Large serialized rigs use conservative stamp checks:
    # any file replacement or directory membership change invalidates lookup.
    directories = {}
    rig = prepared.get("correction_contract", {}).get("legacy_rig")
    if rig:
        directory = Path(rig).resolve()
        directories[str(directory)] = {str(p.relative_to(directory)): stamp(p)
                                      for p in directory.rglob('*') if p.is_file()}
    mirror = path.parents[2]/"preparation"/"resolved_um_config"
    provenance = mirror/(request['candidate']['run']+"_provenance.json")
    if provenance.is_file():
        config_receipt = read(provenance)
        config = Path(config_receipt['source'])
        bind(config, config_receipt['source_sha256'])
        values = tomllib.loads(config.read_text(encoding='utf-8-sig'))
        for item in values.get('protocols', {}).values():
            reference = item[0] if isinstance(item, list) else item
            if reference: bind(config.parent.parent/reference)
    for frame in result["frames"]:
        bind(frame["preview"])
        # Playback only reads PNGs. The numerical fields remain immutable
        # archived outputs, and are verified by the batch delivery audit.
        if not Path(frame["fields"]).is_file():
            raise ValueError("Numerical field archive missing")
    bind(result["mass_curve"])
    bind(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix("."+uuid.uuid4().hex+".tmp")
    temporary.write_text(json.dumps(dict(format="studio-mass-delivery-v1", selection=expected,
        result=str(path), created_utc=result["created_utc"], bindings=bindings,
        physical_directories=directories), indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def find_result(request, method):
    target = index_path(request, method)
    if target is None or not target.is_file():
        return None
    receipt = read(target)
    if receipt.get("format") != "studio-mass-delivery-v1" or receipt["selection"] != selection(request, method):
        return None
    for name, previous in receipt.get('physical_directories', {}).items():
        directory = Path(name)
        current = {str(p.relative_to(directory)): stamp(p) for p in directory.rglob('*') if p.is_file()}
        if current != previous:
            raise ValueError('Precomputed physical rig has changed; recalculate this run')
    for name, binding in receipt["bindings"].items():
        before = stamp(name)
        if before != binding["stamp"]:
            if digest(name) != binding["sha256"] or stamp(name) != before:
                raise ValueError(f"Precomputed result is stale: {Path(name).name}")
    result = read(receipt["result"])
    if result.get("complete") is not True or _selected_index(request, result) is None:
        return None
    return Path(receipt["result"])
