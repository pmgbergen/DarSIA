"""Read-only admission of saved labels and models from explicit metadata DATA.

These helpers never import archived numerical methods or perform inference.
Saved masks are decoded only by explicitly requested display readers. Callers supply original ordered source roles and frozen declarations.
"""
from __future__ import annotations
from collections.abc import Mapping
import ast
import hashlib
import json
from pathlib import Path
import re
import io
import os
import uuid
from copy import deepcopy
import math
import numpy as np
from .phase.metrics import _shape as _rig_shape
from typing import Any
from .project import sha256_file
from .review_storage import resolve_review_path


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_model_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object in {path}")
    return payload


def bound_path(asset, *, base=None, verify=True):
    """Resolve and admit a metadata asset with the original exact SHA contract."""
    if not (isinstance(asset, dict) and isinstance(asset.get("path"), str)
            and bool(asset["path"]) and isinstance(asset.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}\Z", asset["sha256"])):
        raise ValueError("Revision assets require an exact path and SHA256")
    path = Path(asset["path"])
    path = resolve_review_path(path if path.is_absolute() else Path(base or ".") / path).resolve()
    if verify and sha256_file(path) != asset["sha256"]:
        raise ValueError("Rig/camera bound source artifact changed")
    if not verify and not path.is_file():
        raise ValueError("Rig/camera bound source artifact is missing")
    return path


def declared_code_binding(*, parameters: Mapping):
    """Hash passive original source versions in their literal declared name order.

    The original declaration source is byte-admitted first. Neither it nor any
    listed numerical source is imported or executed. Original module identities
    remain keys even if their exact physical bytes later reside in an archive.
    """
    declaration = parameters["code_declaration"]
    path = bound_path(declaration)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == declaration["function_name"]]
    if len(functions) != 1:
        raise ValueError("Saved review source has no unique declared code binding")
    declarations = [node.value for node in functions[0].body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "names"
                            for target in node.targets)]
    if len(declarations) != 1 or not isinstance(declarations[0], ast.Tuple):
        raise ValueError("Saved review code binding must declare a literal name tuple")
    names = []
    for node in declarations[0].elts:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
        elif isinstance(node, ast.Name) and node.id == "__name__":
            names.append(declaration["module_name"])
        else:
            raise ValueError("Saved review source contains a nonliteral module name")
    if names != parameters["declared_code_names"] or set(names) != set(parameters["code_sources"]):
        raise ValueError("Saved review declared source names changed")
    return {name: sha256_file(bound_path(parameters["code_sources"][name])) for name in names}


def canonical_sha256(value: Any) -> str:
    """Bind ordered JSON content using the unchanged compact, finite encoding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def model_identity(model_root: Path, *, parameters: Mapping) -> tuple[str, str]:
    """Bind summary and head roles in their original basename/digest order."""
    summary = read_model_json(model_root / parameters["summary"])
    model_id = str(summary.get("model_id") or model_root.name).strip()
    digest = hashlib.sha256()
    for path in (model_root / relative for relative in parameters["files"]):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return model_id, digest.hexdigest()


def validate_source_inventory(sources, *, parameters: Mapping):
    """Require the original unique whole-campaign run/frame key population."""
    keys = [(s["run"], s["frame_index"]) for s in sources]
    if len(keys) != parameters["expected_frame_count"] or len(set(keys)) != parameters["expected_frame_count"] or len({k[0] for k in keys}) != parameters["expected_run_count"]:
        raise ValueError(parameters["errors"]["inventory"])


def validate_runtime_heads(candidate, marker, heads, *, parameters: Mapping, verified_plan=None):
    """Admit saved placement, calibration and heads without evaluating them."""
    root = Path(candidate).resolve()
    plan = read_json(bound_path(marker["plan"], base=root)) if verified_plan is None else verified_plan
    if (plan.get("contract") != parameters["contract"] or marker.get("contract") != parameters["contract"]
        or plan.get("output_root") != str(root) or plan.get("placement") not in parameters["placements"]
        or marker.get("placement") != plan["placement"] or marker.get("delta") != plan["delta"]
        or marker.get("spec_sha256") != parameters["spec_sha256"]
        or plan.get("model_identity") != list(model_identity(root, parameters=parameters["model_identity"]))
        or heads != plan["binding"]["heads"] or heads != marker.get("heads")
        or plan["job_sha256"] != canonical_sha256(plan["binding"])
        or plan["pipeline_sha256"] != canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"]))):
        raise ValueError(parameters["errors"]["runtime"])
    for key in ("co2", "g3", "regional"): bound_path(heads[key], base=root)
    for value in heads["t4b"].values(): bound_path(value, base=root)
    shift = read_json(bound_path(plan["binding"]["inputs"]["calibrated_shift"]))
    if shift.get("placement") != plan["placement"] or shift.get("delta") != plan["delta"]:
        raise ValueError(parameters["errors"]["delta"])
    return True


def registered_warning(root, registration, *, parameters: Mapping):
    """Return the original review-only warning only for an intact saved seal."""
    try:
        root = Path(root)
        if (registration.get("registration_kind") != parameters["kind"] or registration.get("model_sha256") != model_identity(root, parameters=parameters["model_identity"])[1]
                or registration.get("bottom_mass_allowed") is not False or registration.get("mass_eligible") is not False):
            raise ValueError(parameters["errors"]["registration"])
        seal=read_json(root/parameters["seal"])
        if (seal.get("complete") is not True or (seal.get("run_count"),seal.get("frame_count"))!=(parameters["expected_run_count"],parameters["expected_frame_count"])
                or set(seal.get("blind_hold_runs",[]))!=set(parameters["hidden_runs"])
                or seal.get("bottom_mass_allowed") is not False or seal.get("mass_eligible") is not False):
            raise ValueError(parameters["errors"]["seal"])
        for relative, digest in registration["metadata_sha256"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError(parameters["errors"]["metadata"])
        return parameters["warning"]
    except (OSError, KeyError, ValueError, TypeError):
        return parameters["errors"]["invalid_warning"]


def configured_parameters(section):
    """Select explicit application DATA for a retained saved-reader adapter."""
    from .native.config import require_parameters
    return require_parameters("saved_review_metadata")[section]


def configured_export(section, name, *, allowed_names):
    """Keep original typed metadata exports, rejecting protocol probes first."""
    if name not in allowed_names:
        raise AttributeError(name)
    item = configured_parameters(section)["exports"][name]
    return tuple(item["value"]) if item["python_type"] == "tuple" else item["value"]


def configured_model_identity(root):
    """Use the original model digest roles from explicit saved-reader DATA."""
    return model_identity(root, parameters=configured_parameters("floor")["model_identity"])


def registration_path(model_id, *, parameters, results=None):
    """Locate the immutable registration from an explicit result root."""
    results = Path(parameters["results"]) if results is None else Path(results)
    return results / parameters["directory"] / f"{model_id}.json"


def configured_registration_path(model_id, *, results=None):
    """Keep the retained registration API using real application DATA."""
    return registration_path(model_id, parameters=configured_parameters("registry"), results=results)


class _SavedRigMetadata:
    """Original saved-source predicates; frozen values supplied by application DATA."""
    def __init__(self, parameters):
        self._p = dict(parameters)
        self._p["_SCIENTIFIC_FIELDS"] = tuple(self._p["_SCIENTIFIC_FIELDS"])
        self._sha = re.compile(parameters["_SHA_PATTERN"])

    @property
    def SOURCE_KEY(self):
        """Return the exact required saved-source DATA declaration."""
        return self._p["SOURCE_KEY"]

    @property
    def SOURCE_CONTRACT(self):
        """Return the exact required saved-source DATA declaration."""
        return self._p["SOURCE_CONTRACT"]

    @property
    def INPUT_CONTRACT(self):
        """Return the exact required saved-source DATA declaration."""
        return self._p["INPUT_CONTRACT"]

    @property
    def SCIENTIFIC_CONTRACT(self):
        """Return the exact required saved-source DATA declaration."""
        return self._p["SCIENTIFIC_CONTRACT"]

    def require(self, condition, message):
        if not condition:
            raise ValueError(message)

    def asset(self, value):
        self.require(isinstance(value, dict) and isinstance(value.get('path'), str) and bool(value['path']) and isinstance(value.get('sha256'), str) and self._sha.fullmatch(value['sha256']), 'Revision assets require an exact path and SHA256')
        return deepcopy(value)

    def homography(self, value):
        h = np.asarray(value, np.float64)
        self.require(h.shape == (3, 3) and np.isfinite(h).all() and (abs(np.linalg.det(h)) > 1e-12) and (abs(h[2, 2]) > 1e-12), 'Invalid source-bound rig homography')
        return (h / h[2, 2]).tolist()

    def phase_grid(self, phase_shape_hw=None):
        """Declare the phase raster independently of the unchanged photograph canvas.

        Accepted geometry may append bottom rows. Changing the origin, width or
        source crop would need a different source contract and is rejected here.
        """
        shape = list(self._p['CANVAS_WH'][::-1] if phase_shape_hw is None else phase_shape_hw)
        self.require(len(shape) == 2 and all((type(value) is int for value in shape)) and (shape[0] >= self._p['CANVAS_WH'][1]) and (shape[1] == self._p['CANVAS_WH'][0]), 'Rig phase grid permits explicit bottom padding only')
        return dict(shape_hw=shape, native_origin_xy=deepcopy(self._p['ORIGIN_XY']), photograph_canvas_wh=deepcopy(self._p['CANVAS_WH']), source_native_shape_hw=deepcopy(self._p['SOURCE_SHAPE']), source_content_box_xywh=deepcopy(self._p['SOURCE_BOX']))

    def validate_phase_grid(self, value):
        self.require(isinstance(value, dict) and value == self.phase_grid(value.get('shape_hw')), 'Phase grid must bind its exact raster, origin and unchanged photographic crop')
        return deepcopy(value)

    def validate_rig_descriptor(self, value):
        self.require(isinstance(value, dict) and value.get('contract') == self._p['RIG_CONTRACT'] and (value.get('native_origin_xy') == self._p['ORIGIN_XY']), 'Rig classes/area must declare the full canvas and its exact native origin')
        shape = value.get('phase_shape_hw', self._p['CANVAS_WH'][::-1])
        grid = self.phase_grid(shape)
        self.require(value.get('canvas_wh') == self._p['CANVAS_WH'], 'Rig descriptor changed its fixed photographic canvas')
        self.require(shape == self._p['CANVAS_WH'][::-1] or value.get('phase_shape_hw') == shape, 'Padded rig descriptors need an explicit phase_shape_hw')
        self.require(value.get('fixed_area_mm2') == 0.027053, 'Rig revision changed the historical AC28 area key')
        for name in ('area_map', 'classes'):
            self.asset(value.get(name))
        self.require(value['area_map'].get('array_key', 'pixel_area_mm2') == 'pixel_area_mm2', 'Rig area map has another array key')
        self.require(value.get('mass_eligible') is False, 'Rig area descriptor cannot approve mass')
        return deepcopy(value)

    def validate_frame_binding(self, row, record, *, run, index, reference=False):
        self.require(isinstance(row, dict) and row.get('run') == run and (row.get('frame_index') == index), 'Source binding belongs to another frame')
        self.require(row.get('source_sha256') == record.get('source_sha256') and row.get('source_path') == record.get('source_path') and (float(row.get('elapsed_h', math.nan)) == float(record.get('elapsed_h', math.nan))), 'Source binding changed its camera image or elapsed time')
        self.require(math.isfinite(float(record['elapsed_h'])) and isinstance(record['source_sha256'], str) and self._sha.fullmatch(record['source_sha256']), 'Source timeline has invalid time/SHA')
        self.asset(row.get('geometry_receipt'))
        self.homography(row.get('H_ac28_to_frame_native'))
        if not reference:
            self.homography(row.get('H_v5_B_to_frame_native'))
            self.require(type(row.get('source_changed')) is bool, 'Source binding lacks explicit source-change state')
            old = row.get('parent_source_frame_index')
            self.require(row['source_changed'] and old is None and (row.get('own_source_geometry') is True) or (not row['source_changed'] and type(old) is int and (old >= 0)), 'Changed sources need their own geometry; unchanged sources need an exact parent index')
        grid = self.validate_phase_grid(row.get('phase_grid'))
        self.require(row.get('rig_domain_revision') == record.get('rig_domain_revision'), 'Source and area/class bindings differ')
        self.validate_rig_descriptor(row['rig_domain_revision'])
        self.require(row['rig_domain_revision'].get('phase_shape_hw', self._p['CANVAS_WH'][::-1]) == grid['shape_hw'], 'Source binding and rig descriptor disagree on the phase raster')
        full = record.get('full_native_scientific')
        self.require(full == row.get('full_native_scientific') and isinstance(full, dict) and (full.get('contract') == self._p['SCIENTIFIC_CONTRACT']) and (full.get('source_sha256') == record['source_sha256']) and (full.get('native_shape_hw') == self._p['SOURCE_SHAPE']) and (full.get('unit_denominator') == 65535), 'Full-native scientific source needs its own exact source/correction contract')
        for name in ('log_linear_rgb', 'valid_mask', 'receipt'):
            self.asset(full.get(name))
        extended = record.get('full_extended_scientific')
        self.require(extended == row.get('full_extended_scientific'), 'Source and extended RAW-ring bindings differ')
        if extended is not None:
            self.require(isinstance(extended, dict) and extended.get('contract') == self._p['EXTENDED_SCIENTIFIC_CONTRACT'] and (extended.get('complete') is True) and (extended.get('source_sha256') == record['source_sha256']) and (extended.get('phase_shape_hw') == grid['shape_hw']) and (extended.get('native_origin_xy') == self._p['ORIGIN_XY']) and (extended.get('native_centre') == full) and (extended.get('unit_denominator') == 65535) and (extended.get('storage') == 'exact-native-centre-plus-disjoint-frozen-raw-ring-windows'), 'Extended RAW ring lost its source, raster, correction or exact native centre')
            for window in extended.get('ring_windows', []):
                self.asset(window.get('log_linear_rgb'))
        cache = full.get('parent_cache')
        if reference or row.get('source_changed') is False:
            self.require(isinstance(cache, dict) and cache.get('log_linear_rgb') == dict(path=record['v5_log_linear_rgb_path'], sha256=record['v5_log_linear_rgb_sha256']) and (cache.get('valid_mask') == dict(path=record['v5_log_linear_valid_mask_path'], sha256=record['v5_log_linear_valid_mask_sha256'])), 'Expanded scientific source lost its exact frozen parent cache')
        else:
            self.require(cache is None, "New camera acquisition cannot claim another source's parent cache")
        photos = row.get('photographs') or {}
        self.require(set(photos) == {'natural', 'lift1', 'lift2', 'change', 'raw_detail', 'original'}, 'Source revision requires all six explicit photographic views')
        for mode, value in photos.items():
            self.asset(value)
        self.require(photos['original'] == dict(path=record['source_path'], sha256=record['source_sha256']), 'Original display is not the bound camera JPEG')
        prefix = 'v5_baseline' if reference else 'v5_frame'
        for prefix_key in ('v5_log_linear_rgb', 'v5_log_linear_valid_mask', prefix + '_receipt'):
            self.asset(dict(path=record.get(prefix_key + '_path'), sha256=record.get(prefix_key + '_sha256')))
        self.require((record.get('v5_display_modes') or {}).get('natural', {}).get('jpeg_sha256') == photos['natural']['sha256'], 'Natural photograph differs from the scientific source record')
        for mode in ('natural', 'lift1', 'lift2'):
            declaration = (record.get('v5_display_modes') or {}).get(mode) or {}
            self.require(declaration.get('jpeg_path') == photos[mode]['path'] and declaration.get('jpeg_sha256') == photos[mode]['sha256'], 'Photographic mode differs from its bound source declaration')
        self.require(record.get('v5_raw_detail_jpeg_path') == photos['raw_detail']['path'] and record.get('v5_raw_detail_jpeg_sha256') == photos['raw_detail']['sha256'], 'RAW detail differs from its bound source declaration')
        return deepcopy(row)

    def build_native_metadata(self, parent_native, camera_run, records, bindings, reference_binding, *, input_manifest):
        """Return a new native document; inputs and all old label/source records stay intact."""
        run = str(parent_native.get('run', '')).casefold()
        self.require(run == camera_run.get('run') and run.startswith('ac'), 'Camera plan belongs to another run')
        self.require(parent_native.get('canvas_wh') == self._p['CANVAS_WH'] and parent_native.get('analysis_image_shape_hw') == self._p['SOURCE_SHAPE'] and (parent_native.get('content_box_xywh') == self._p['SOURCE_BOX']), 'Parent source geometry changed')
        selected = camera_run.get('frames') or []
        self.require(len(records) == len(bindings) == len(selected) == camera_run.get('frame_count'), 'Source metadata does not cover the reselected camera timeline')
        result = deepcopy(parent_native)
        for old_marker in ('v5_native_source_alias', 'v5_track1_photo_geometry', 'v5_frame_geometry'):
            result.pop(old_marker, None)
        result['frame_sources'] = deepcopy(records)
        result['frame_count'] = len(records)
        for index, (new, selected_row, binding) in enumerate(zip(records, selected, bindings)):
            self.require(new.get('frame_index') == index and new.get('source_sha256') == selected_row['source_sha256'] and (new.get('source_path') == selected_row['source_path']) and (float(new.get('elapsed_h', math.nan)) == float(selected_row['elapsed_h'])), 'Materialized frame changed the camera-clock selection')
            old_index = selected_row.get('retained_old_frame_index')
            if old_index is not None:
                old = parent_native['frame_sources'][old_index]
                self.require(old['source_sha256'] == new['source_sha256'], 'Retained source is not the same camera image')
                self.require(all((new.get(key) == old.get(key) for key in self._p['_SCIENTIFIC_FIELDS'])), 'Unchanged camera source must retain all frozen scientific/photographic assets')
                self.require(binding.get('parent_source_frame_index') == old_index and binding.get('source_changed') is False, 'Retained source lost its original index')
                self.require(binding.get('temporal_input_policy') == 'retain-parent-predecessor-by-source-sha' and binding.get('temporal_predecessor') == selected_row.get('frozen_temporal_predecessor'), 'Retained source changed its original source-specific temporal predecessor')
            else:
                self.require(binding.get('source_changed') is True, 'New acquisition was incorrectly declared unchanged')
                previous = records[index - 1] if index > 1 else None
                predecessor = binding.get('temporal_predecessor')
                self.require(binding.get('temporal_input_policy') == 'camera-clock-prior-source' and (previous is None and predecessor is None or (isinstance(predecessor, dict) and predecessor.get('source_sha256') == previous['source_sha256'] and (predecessor.get('source_path') == previous['source_path']) and (predecessor.get('full_native_scientific') == previous['full_native_scientific']))), 'New acquisition changed its declared camera-clock temporal predecessor')
            result['frame_sources'][index]['temporal_input_policy'] = binding['temporal_input_policy']
            result['frame_sources'][index]['temporal_predecessor'] = deepcopy(binding['temporal_predecessor'])
            self.validate_frame_binding(binding, new, run=run, index=index)
        reference = deepcopy(parent_native.get('reference') or {})
        reference['rig_domain_revision'] = deepcopy(reference_binding['rig_domain_revision'])
        reference['full_native_scientific'] = deepcopy(reference_binding['full_native_scientific'])
        if 'full_extended_scientific' in reference_binding:
            reference['full_extended_scientific'] = deepcopy(reference_binding['full_extended_scientific'])
        result['reference'] = reference
        self.validate_frame_binding(reference_binding, reference, run=run, index=None, reference=True)
        grids = [self.validate_phase_grid(row.get('phase_grid')) for row in [reference_binding, *bindings]]
        self.require(all((grid == grids[0] for grid in grids)), 'One camera run contains different phase rasters')
        result[self._p['SOURCE_KEY']] = dict(contract=self._p['SOURCE_CONTRACT'], input_manifest=self.asset(input_manifest), source_native_shape_hw=self._p['SOURCE_SHAPE'], source_content_box_xywh=self._p['SOURCE_BOX'], phase_grid=grids[0], labels_source_owned=True, inference_authorized=False)
        return result


def _saved_geometry(row, record, *, base, reference=False):
    metadata = _SavedRigMetadata(configured_parameters('rig_source')['metadata'])
    receipt = read_json(bound_path(row['geometry_receipt'], base=base))
    source_sha = receipt.get('camera_jpeg_sha256', receipt.get('source_sha256'))
    metadata.require(source_sha == record['source_sha256'], 'Geometry belongs to another camera source')
    h = receipt.get('H_ac28_to_frame_native', receipt.get('H_ac28_to_reference_native'))
    metadata.require(metadata.homography(h) == metadata.homography(row['H_ac28_to_frame_native']), 'Frame uses a homography borrowed from another source')
    if not reference:
        metadata.require(metadata.homography(receipt['H_v5_B_to_frame_native']) == metadata.homography(row['H_v5_B_to_frame_native']), 'Frozen B-to-source geometry changed')
    if row.get('source_changed'):
        metadata.require(receipt.get('new_geometry_measured_from_own_source') is True and receipt.get('camera_arw_sha256') == record.get('v5_raw_sha256') and ((receipt.get('source_sand_registration') or {}).get('geometry_pass') is True), 'New source lacks its own source-exact photographic geometry proof')
    return receipt

def _saved_scientific_receipt(record):
    metadata = _SavedRigMetadata(configured_parameters('rig_source')['metadata'])
    full = record['full_native_scientific']
    receipt = read_json(bound_path(full['receipt']))
    metadata.require(receipt.get('contract') == metadata.SCIENTIFIC_CONTRACT and receipt.get('source_sha256') == record['source_sha256'] and (receipt.get('log_linear_rgb_sha256') == full['log_linear_rgb']['sha256']) and (receipt.get('valid_mask_sha256') == full['valid_mask']['sha256']) and (receipt.get('parent_cache') == full['parent_cache']) and (receipt.get('frozen_photometry') is True), 'Full-native scientific correction certificate differs from its source/output/parent cache')
    if full['parent_cache'] is not None:
        metadata.require(receipt.get('parent_valid_bits_preserved') is True, 'Expanded scientific source lacks preserved-parent-bit certification')
    return receipt

def load_native_source(review_root, run, manifest=None, *, verify_geometry=True):
    metadata = _SavedRigMetadata(configured_parameters('rig_source')['metadata'])
    root = resolve_review_path(review_root).resolve()
    run = str(run).casefold()
    native_path = root / 'native_frames' / run / 'manifest.json'
    native = read_json(native_path) if manifest is None else manifest
    marker = native.get(metadata.SOURCE_KEY)
    metadata.require(isinstance(marker, dict) and marker.get('contract') == metadata.SOURCE_CONTRACT and (marker.get('inference_authorized') is False), 'Invalid explicit rig/camera native source contract')
    metadata.validate_phase_grid(marker.get('phase_grid'))
    metadata.require(not any((key in native for key in ('v5_native_source_alias', 'v5_track1_photo_geometry', 'v5_frame_geometry'))), 'Rig/camera timeline cannot pretend to be an old mask-only alias')
    input_path = bound_path(marker['input_manifest'], base=native_path.parent)
    inputs = read_json(input_path)
    metadata.require(inputs.get('contract') == metadata.INPUT_CONTRACT and inputs.get('spec_sha256') == configured_parameters('rig_source')['spec_sha256'], 'Rig/camera source certificate binds another specification')
    parent = read_json(bound_path(inputs['parent_native_manifest'], base=input_path.parent))
    camera = read_json(bound_path(inputs['camera_timeline_run'], base=input_path.parent))
    frames = inputs.get('frames') or []
    expected = metadata.build_native_metadata(parent, camera, native['frame_sources'], frames, inputs['reference'], input_manifest=marker['input_manifest'])
    metadata.require(native == expected and native.get('run') == run, 'Native source metadata differs from its frozen source/camera contract')
    if verify_geometry:
        for row, record in zip(frames, native['frame_sources']):
            _saved_geometry(row, record, base=input_path.parent)
            _saved_scientific_receipt(record)
        _saved_geometry(inputs['reference'], native['reference'], base=input_path.parent, reference=True)
        _saved_scientific_receipt(native['reference'])
    return dict(root=root, native_path=native_path, native=native, inputs=inputs, input_path=input_path)

def native_frame_assets(review_root, run, frame_index=None, *, reference=False, manifest=None):
    metadata = _SavedRigMetadata(configured_parameters('rig_source')['metadata'])
    state = load_native_source(review_root, run, manifest, verify_geometry=False)
    if reference:
        record = state['native']['reference']
        row = state['inputs']['reference']
    else:
        rows = state['native']['frame_sources']
        metadata.require(type(frame_index) is int and 0 <= frame_index < len(rows), 'Rig/camera frame is outside its timeline')
        record = rows[frame_index]
        row = state['inputs']['frames'][frame_index]
    _saved_geometry(row, record, base=state['input_path'].parent, reference=reference)
    _saved_scientific_receipt(record)
    return (state, record, row)


def _rig_class_asset(asset, base_dir):
    if not isinstance(asset, Mapping) or not asset.get('path') or (not asset.get('sha256')):
        raise ValueError('Pixel-area asset requires an exact path and SHA256')
    path = Path(asset['path'])
    if not path.is_absolute():
        path = Path(base_dir) / path
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != asset['sha256']:
        raise ValueError('Pixel-area asset SHA256 mismatch')
    return (path, data, digest)


def _rig_array_asset(asset, base_dir, key):
    path, data, digest = _rig_class_asset(asset, base_dir)
    loaded = np.load(io.BytesIO(data), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as archive:
            value = archive[asset.get('array_key', key)].copy()
    else:
        value = loaded
    return (value, dict(path=str(path), sha256=digest))


def rig_phase_shape(revision):
    """Return the explicit analysis grid, independently of photograph height.

    A new rig revision may add rows below the photographic canvas. Historical
    revisions keep their original common photograph/analysis shape.
    """
    photograph = _rig_shape(tuple(revision['canvas_wh'])[::-1])
    shape = _rig_shape(revision.get('phase_shape_hw', photograph))
    if shape[1] != photograph[1] or shape[0] < photograph[0]:
        raise ValueError('Rig phase grid may extend only below its photographic canvas')
    declared = revision.get('photograph_canvas_wh')
    if declared is not None and tuple(declared) != photograph[::-1]:
        raise ValueError('Rig photograph canvas bindings disagree')
    return shape


def load_rig_classes(revision, *, base_dir='.', shape_hw=None, origin_xy=(0, 0)):
    """Read SHA-bound domain/fill flags, checking their disjoint partition."""
    if not isinstance(revision, Mapping) or revision.get('contract') != configured_parameters('display_views')['rig_contract']:
        raise ValueError('Unsupported rig-domain class revision')
    path, data, digest = _rig_class_asset(revision['classes'], base_dir)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in ('modelled', 'filled_observed', 'filled_unobserved')}
        domain = archive['rig_domain'].copy() if 'rig_domain' in archive.files else None
    if domain is None:
        domain, _ = _rig_array_asset(revision['rig_domain'], base_dir, 'rig_domain')
    full_shape = rig_phase_shape(revision)
    if any((v.dtype != bool or v.shape != full_shape for v in (*fields.values(), domain))):
        raise ValueError('Rig-domain classes must be exact bool masks on the declared canvas')
    total = sum((v.astype(np.uint8) for v in fields.values()))
    if np.any(total > 1) or not np.array_equal(total == 1, domain):
        raise ValueError('Rig-domain modelled/filled classes do not form a disjoint complete partition')
    fields['rig_domain'] = domain
    if shape_hw is not None:
        fields = {k: _crop_rig_grid(v, revision['native_origin_xy'], shape_hw, origin_xy) for k, v in fields.items()}
    for v in fields.values():
        v.flags.writeable = False
    return (fields, dict(path=str(path), sha256=digest, contract=configured_parameters('display_views')['rig_contract']))


def _crop_rig_grid(array, source_origin_xy, shape_hw, origin_xy=(0, 0)):
    """Select an exact native rectangle; no interpolation or silent padding."""
    shape = _rig_shape(shape_hw)
    offset = np.asarray(origin_xy, np.float64) - np.asarray(source_origin_xy, np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all() or not np.array_equal(offset, np.rint(offset)):
        raise ValueError("Physical grid crop needs exact integer native offsets")
    x, y = map(int, offset)
    if x < 0 or y < 0 or x + shape[1] > array.shape[1] or y + shape[0] > array.shape[0]:
        raise ValueError("Physical grid crop is outside the recorded review canvas")
    from .phase.domain import crop_native_canvas
    return crop_native_canvas(array, (x, y, shape[1], shape[0]))


def require_frozen_gas_audit(audit):
    if (audit.get('contract') != configured_parameters('identity_report')['contract']
            or audit.get('complete') is not True
            or audit.get('frozen_gas_no_production_reads') is not True
            or audit.get('legacy_reconstruction_unreachable') is not True
            or audit.get('frozen_gas_compare_and_report_only') is not True):
        raise ValueError('No exact source audit permits diagnostic-only frozen_gas')


from urllib.parse import unquote, urlparse


def local_path(value):
    if str(value).startswith('file:///'):
        return Path(unquote(urlparse(str(value)).path).lstrip('/'))
    return Path(value)


def result_paths(path):
    payload = _read(path)
    if payload.get('complete') and 'selected_inputs' in payload and 'frames' in payload:
        return [Path(path).resolve()]
    if isinstance(payload.get('experiments'), list):
        return [local_path(r['result']).resolve() for r in payload['experiments'] if r.get('result')]
    if payload.get('stage') == 'complete' and payload.get('result'):
        return [local_path(payload['result']).resolve()]
    raise ValueError('Select a completed result.json, status.json or delivery.json')


def inspect_result(path, result=None):
    result = read_result(path) if result is None else result
    if not result.get('complete') or not result.get('frames'):
        raise ValueError('A complete numerical result is required')
    candidate = result['selected_inputs']['candidate']
    return dict(path=str(Path(path).resolve()), dataset=candidate['dataset'], run=candidate['run'],
                model=candidate['candidate_id'], frames=len(result['frames']),
                created_utc=result.get('created_utc', ''), method=result['method']['settings'])


def read_result(path):
    result=_read(path);folder=Path(path).resolve().parent
    for row in result.get('frames',[]):
        for key in ('preview','fields'):
            if row.get(key) and not Path(row[key]).is_absolute():row[key]=str(folder/row[key])
    for key in ('mass_curve','overflow_curve','series_calibration','result_source_manifest'):
        if result.get(key) and not Path(result[key]).is_absolute():result[key]=str(folder/result[key])
    return result


def references_for(run, times, path):
    from darsia.fluidflower.overflow_phase_reference import load_overflow_phase_constraint
    constraint = load_overflow_phase_constraint(path, run=run, parameters=configured_parameters("overflow_reference"))
    records = []
    for time in times:
        target = constraint.target_at(float(time))
        records.append(None if target is None else dict(mass_g=target.gas_mass_kg*1000,
            sigma_g=target.phase_mass_sigma_kg*1000, time_h=target.time_h,
            mode=constraint.constraint_mode, source=target.target_source))
    return records, constraint.to_dict()

def _read(path):
    """Read an original metadata document through the existing locator resolver."""
    return read_json(resolve_review_path(path))


def _asset(path):
    """Return the original two-field saved-asset identity, without numerical imports."""
    path = resolve_review_path(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _write(path, value):
    """Preserve the original GUI-owned metadata publication protocol."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


read = _read
write = _write


def resolve_photograph(review_root, run, frame_index=None, *, mode="natural", reference=False, verify=True):
    """Resolve one declared saved photographic mode after source/geometry admission."""
    state, _record, row = native_frame_assets(review_root, run, frame_index, reference=reference)
    metadata = _SavedRigMetadata(configured_parameters("rig_source")["metadata"])
    metadata.require(mode in row["photographs"], "Unknown rig/camera photographic mode")
    return bound_path(row["photographs"][mode], base=state["input_path"].parent, verify=verify)

from functools import lru_cache
from numbers import Real
from .review_storage import resolve_review_path
_SAVED_MISSING = object()

class _SavedRevisionGuardBase:

    def __init__(self, parameters):
        self._parameters = parameters

    @staticmethod
    def _typed(entry):
        value, kind = (entry['value'], entry['python_type'])
        if kind == 'tuple':
            return tuple(value)
        if kind == 'frozenset':
            return frozenset(value)
        if kind == 'set':
            return set(value)
        expected = {'str': str, 'bool': bool, 'int': int, 'float': float, 'dict': dict, 'list': list, 'NoneType': type(None)}
        if kind not in expected or type(value) is not expected[kind]:
            raise TypeError('Saved metadata parameter changed its original type')
        return value

    def _literal(self, method, index):
        return self._typed(self._parameters['literals'][method][index])

    def _export(self, name):
        return self._typed(self._parameters['exports'][name])

    def _saved_combined_floor_read(self, path):
        value = json.loads(Path(path).read_text(encoding=self._literal('_saved_combined_floor_read', 0)))
        if not isinstance(value, dict):
            raise ValueError(self._literal('_saved_combined_floor_read', 1))
        return value

    def _saved_combined_floor_samefile(self, left, right):
        return resolve_review_path(Path(left)).samefile(right)

    def _saved_combined_floor_require(self, value, message):
        if not value:
            raise ValueError(message)

    @lru_cache(maxsize=16)
    def _saved_combined_floor_admitted(self, root_text, registration_text, stamps):
        from .phase_saved_registration import registration_warning
        root, registration_path = (Path(root_text), Path(registration_text))
        registration = self._read(registration_path)
        self._require(registration.get(self._literal('_saved_combined_floor_admitted', 0)) == self._export('KIND') and registration_warning(root) == self._export('WARNING'), self._literal('_saved_combined_floor_admitted', 1))
        seal_path, plan_path = (root / self._export('SEAL'), root / self._export('PLAN'))
        seal, plan, policy = (self._read(seal_path), self._read(plan_path), self._read(root / self._literal('_saved_combined_floor_admitted', 2)))
        identity = list(configured_model_identity(root))
        plan_sha, seal_sha = (sha256_file(plan_path), sha256_file(seal_path))
        pipeline = plan.get(self._literal('_saved_combined_floor_admitted', 3))
        self._require(registration.get(self._literal('_saved_combined_floor_admitted', 4)) == self._literal('_saved_combined_floor_admitted', 5) and (registration.get(self._literal('_saved_combined_floor_admitted', 6)), registration.get(self._literal('_saved_combined_floor_admitted', 7))) == (self._literal('_saved_combined_floor_admitted', 8), self._literal('_saved_combined_floor_admitted', 9)) and all((registration.get(key) is self._literal('_saved_combined_floor_admitted', 10) for key in (self._literal('_saved_combined_floor_admitted', 11), self._literal('_saved_combined_floor_admitted', 12), self._literal('_saved_combined_floor_admitted', 13), self._literal('_saved_combined_floor_admitted', 14), self._literal('_saved_combined_floor_admitted', 15)))) and (registration.get(self._literal('_saved_combined_floor_admitted', 16)) == identity[self._literal('_saved_combined_floor_admitted', 17)]) and (registration.get(self._literal('_saved_combined_floor_admitted', 18)) == identity[self._literal('_saved_combined_floor_admitted', 19)]), self._literal('_saved_combined_floor_admitted', 20))
        metadata = registration.get(self._literal('_saved_combined_floor_admitted', 21)) or {}
        self._require(metadata.get(self._export('PLAN')) == plan_sha == registration.get(self._literal('_saved_combined_floor_admitted', 22)) and metadata.get(self._export('SEAL')) == seal_sha == registration.get(self._literal('_saved_combined_floor_admitted', 23)) and (metadata.get(self._literal('_saved_combined_floor_admitted', 24)) == sha256_file(root / self._literal('_saved_combined_floor_admitted', 25))), self._literal('_saved_combined_floor_admitted', 26))
        self._require(seal.get(self._literal('_saved_combined_floor_admitted', 27)) == plan.get(self._literal('_saved_combined_floor_admitted', 28)) == self._export('CONTRACT') and seal.get(self._literal('_saved_combined_floor_admitted', 29)) is self._literal('_saved_combined_floor_admitted', 30) and ((seal.get(self._literal('_saved_combined_floor_admitted', 31)), seal.get(self._literal('_saved_combined_floor_admitted', 32))) == (self._literal('_saved_combined_floor_admitted', 33), self._literal('_saved_combined_floor_admitted', 34))) and (seal.get(self._literal('_saved_combined_floor_admitted', 35)) == plan.get(self._literal('_saved_combined_floor_admitted', 36)) == identity) and isinstance(pipeline, str) and re.fullmatch(self._literal('_saved_combined_floor_admitted', 37), pipeline) and (seal.get(self._literal('_saved_combined_floor_admitted', 38)) == pipeline) and (seal.get(self._literal('_saved_combined_floor_admitted', 39)) == plan_sha) and self._samefile(seal[self._literal('_saved_combined_floor_admitted', 40)][self._literal('_saved_combined_floor_admitted', 41)], plan_path) and (seal[self._literal('_saved_combined_floor_admitted', 42)][self._literal('_saved_combined_floor_admitted', 43)] == plan_sha) and (plan.get(self._literal('_saved_combined_floor_admitted', 44)) == canonical_sha256(plan[self._literal('_saved_combined_floor_admitted', 45)])) and (pipeline == canonical_sha256(dict(binding=plan[self._literal('_saved_combined_floor_admitted', 46)], model_identity=identity))), self._literal('_saved_combined_floor_admitted', 47))
        inputs = plan[self._literal('_saved_combined_floor_admitted', 48)][self._literal('_saved_combined_floor_admitted', 49)]
        self._require(plan[self._literal('_saved_combined_floor_admitted', 50)][self._literal('_saved_combined_floor_admitted', 51)][self._literal('_saved_combined_floor_admitted', 52)] == self._export('SPEC_SHA256') and all((seal.get(key) is self._literal('_saved_combined_floor_admitted', 53) for key in (self._literal('_saved_combined_floor_admitted', 54), self._literal('_saved_combined_floor_admitted', 55), self._literal('_saved_combined_floor_admitted', 56), self._literal('_saved_combined_floor_admitted', 57), self._literal('_saved_combined_floor_admitted', 58), self._literal('_saved_combined_floor_admitted', 59), self._literal('_saved_combined_floor_admitted', 60)))) and all((inputs.get(key) is self._literal('_saved_combined_floor_admitted', 61) for key in (self._literal('_saved_combined_floor_admitted', 62), self._literal('_saved_combined_floor_admitted', 63), self._literal('_saved_combined_floor_admitted', 64), self._literal('_saved_combined_floor_admitted', 65), self._literal('_saved_combined_floor_admitted', 66)))) and (set(seal.get(self._literal('_saved_combined_floor_admitted', 67), [])) == set(inputs.get(self._literal('_saved_combined_floor_admitted', 68), [])) == self._export('HIDDEN')) and (set(policy.get(self._literal('_saved_combined_floor_admitted', 69), [])) == self._export('HIDDEN')) and (policy.get(self._literal('_saved_combined_floor_admitted', 70)) == identity[self._literal('_saved_combined_floor_admitted', 71)]) and (policy.get(self._literal('_saved_combined_floor_admitted', 72)) is self._literal('_saved_combined_floor_admitted', 73)) and (policy.get(self._literal('_saved_combined_floor_admitted', 74)) is self._literal('_saved_combined_floor_admitted', 75)) and (policy.get(self._literal('_saved_combined_floor_admitted', 76)) is self._literal('_saved_combined_floor_admitted', 77)), self._literal('_saved_combined_floor_admitted', 78))
        return dict(registration=registration, pipeline_sha256=pipeline, model_identity=identity)

    def _saved_combined_floor_saved_review(self, model_root):
        """Return the checked registration/pipeline, or raise without changing files."""
        root = resolve_review_path(Path(model_root)).resolve()
        registration = resolve_review_path(configured_registration_path(root.name)).resolve()
        metadata = self._read(registration).get(self._literal('_saved_combined_floor_saved_review', 0)) or {}
        paths = sorted({registration, root / self._export('PLAN'), root / self._export('SEAL'), root / self._literal('_saved_combined_floor_saved_review', 1), *(root / name for name in metadata)}, key=str)
        stamps = (tuple(configured_model_identity(root)), tuple(((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)))
        return self._admitted(str(root), str(registration), stamps)

    def _saved_combined_floor_verify_heads(self, heads, expected, root=_SAVED_MISSING, relatives=_SAVED_MISSING):
        if root is _SAVED_MISSING:
            root = self._literal('_saved_combined_floor_verify_heads', 0)
        if relatives is _SAVED_MISSING:
            relatives = self._literal('_saved_combined_floor_verify_heads', 1)
        if set(heads) != set(expected):
            raise ValueError(self._literal('_saved_combined_floor_verify_heads', 2))
        for key, digest in expected.items():
            path = bound_path(heads[key], base=root)
            if heads[key][self._literal('_saved_combined_floor_verify_heads', 3)] != digest:
                raise ValueError(self._literal('_saved_combined_floor_verify_heads', 4))
            if relatives and path != (Path(root) / relatives[key]).resolve():
                raise ValueError(self._literal('_saved_combined_floor_verify_heads', 5))

    def _saved_combined_floor_validate_runtime_heads(self, candidate, marker, heads):
        root = Path(candidate).resolve()
        plan = read_json(bound_path(marker[self._literal('_saved_combined_floor_validate_runtime_heads', 0)], base=root))
        if marker.get(self._literal('_saved_combined_floor_validate_runtime_heads', 1)) != self._export('CONTRACT') or marker.get(self._literal('_saved_combined_floor_validate_runtime_heads', 2)) != self._export('SPEC_SHA256') or marker.get(self._literal('_saved_combined_floor_validate_runtime_heads', 3)) != self._export('BASE_HEAD_SHA')[self._literal('_saved_combined_floor_validate_runtime_heads', 4)] or (plan.get(self._literal('_saved_combined_floor_validate_runtime_heads', 5)) != self._export('CONTRACT')) or (plan.get(self._literal('_saved_combined_floor_validate_runtime_heads', 6)) != str(root)) or (plan.get(self._literal('_saved_combined_floor_validate_runtime_heads', 7)) != list(configured_model_identity(root))) or (plan.get(self._literal('_saved_combined_floor_validate_runtime_heads', 8)) != canonical_sha256(plan[self._literal('_saved_combined_floor_validate_runtime_heads', 9)])) or (plan.get(self._literal('_saved_combined_floor_validate_runtime_heads', 10)) != canonical_sha256(dict(binding=plan[self._literal('_saved_combined_floor_validate_runtime_heads', 11)], model_identity=plan[self._literal('_saved_combined_floor_validate_runtime_heads', 12)]))) or (marker.get(self._literal('_saved_combined_floor_validate_runtime_heads', 13)) != heads) or (heads != plan[self._literal('_saved_combined_floor_validate_runtime_heads', 14)][self._literal('_saved_combined_floor_validate_runtime_heads', 15)]) or (marker.get(self._literal('_saved_combined_floor_validate_runtime_heads', 16)) != plan[self._literal('_saved_combined_floor_validate_runtime_heads', 17)][self._literal('_saved_combined_floor_validate_runtime_heads', 18)]) or (plan[self._literal('_saved_combined_floor_validate_runtime_heads', 19)][self._literal('_saved_combined_floor_validate_runtime_heads', 20)][self._literal('_saved_combined_floor_validate_runtime_heads', 21)] != self._export('SPEC_SHA256')):
            raise ValueError(self._literal('_saved_combined_floor_validate_runtime_heads', 22))
        self._verify_heads(heads, self._export('BASE_HEAD_SHA'), root, dict(co2=self._literal('_saved_combined_floor_validate_runtime_heads', 23), g3=self._literal('_saved_combined_floor_validate_runtime_heads', 24), regional=self._literal('_saved_combined_floor_validate_runtime_heads', 25)))
        self._verify_heads(marker[self._literal('_saved_combined_floor_validate_runtime_heads', 26)], self._export('T2_HEAD_SHA'), root, dict(co2=self._literal('_saved_combined_floor_validate_runtime_heads', 27), g3=self._literal('_saved_combined_floor_validate_runtime_heads', 28), regional=self._literal('_saved_combined_floor_validate_runtime_heads', 29)))
        return self._literal('_saved_combined_floor_validate_runtime_heads', 30)

    def _saved_combined_floor_registered_warning(self, root, registration):
        try:
            root = Path(root)
            if registration.get(self._literal('_saved_combined_floor_registered_warning', 0)) != self._export('KIND') or registration.get(self._literal('_saved_combined_floor_registered_warning', 1)) != configured_model_identity(root)[self._literal('_saved_combined_floor_registered_warning', 2)] or registration.get(self._literal('_saved_combined_floor_registered_warning', 3)) is not self._literal('_saved_combined_floor_registered_warning', 4) or (registration.get(self._literal('_saved_combined_floor_registered_warning', 5)) is not self._literal('_saved_combined_floor_registered_warning', 6)):
                raise ValueError(self._literal('_saved_combined_floor_registered_warning', 7))
            for relative, digest in registration[self._literal('_saved_combined_floor_registered_warning', 8)].items():
                if sha256_file(root / relative) != digest:
                    raise ValueError(self._literal('_saved_combined_floor_registered_warning', 9))
            return self._export('WARNING')
        except (OSError, KeyError, ValueError, TypeError):
            return self._literal('_saved_combined_floor_registered_warning', 10)

    @lru_cache(maxsize=16)
    def _saved_trained_floor_admitted(self, root_text, registration_text, stamps):
        from .phase_rig_bottom_review_policy import load_policy
        from .phase_saved_registration import registration_warning
        root, registration_path = (Path(root_text), Path(registration_text))
        registration = self._read(registration_path)
        plan_path, seal_path = (root / self._export('PLAN'), root / self._export('SEAL'))
        plan, seal = (self._read(plan_path), self._read(seal_path))
        identity = list(configured_model_identity(root))
        plan_sha, seal_sha = (sha256_file(plan_path), sha256_file(seal_path))
        metadata = registration.get(self._literal('_saved_trained_floor_admitted', 0)) or {}
        policy = load_policy(root)
        self._require(registration.get(self._literal('_saved_trained_floor_admitted', 1)) == self._export('KIND') and registration_warning(root) == self._export('WARNING') and (registration.get(self._literal('_saved_trained_floor_admitted', 2)) == self._literal('_saved_trained_floor_admitted', 3)) and ((registration.get(self._literal('_saved_trained_floor_admitted', 4)), registration.get(self._literal('_saved_trained_floor_admitted', 5))) == (self._literal('_saved_trained_floor_admitted', 6), self._literal('_saved_trained_floor_admitted', 7))) and (registration.get(self._literal('_saved_trained_floor_admitted', 8)) == identity[self._literal('_saved_trained_floor_admitted', 9)]) and (registration.get(self._literal('_saved_trained_floor_admitted', 10)) == identity[self._literal('_saved_trained_floor_admitted', 11)]) and all((registration.get(key) is self._literal('_saved_trained_floor_admitted', 12) for key in (self._literal('_saved_trained_floor_admitted', 13), self._literal('_saved_trained_floor_admitted', 14), self._literal('_saved_trained_floor_admitted', 15), self._literal('_saved_trained_floor_admitted', 16), self._literal('_saved_trained_floor_admitted', 17)))), self._literal('_saved_trained_floor_admitted', 18))
        self._require(metadata.get(self._export('PLAN')) == plan_sha == registration.get(self._literal('_saved_trained_floor_admitted', 19)) and metadata.get(self._export('SEAL')) == seal_sha == registration.get(self._literal('_saved_trained_floor_admitted', 20)) and (metadata.get(self._literal('_saved_trained_floor_admitted', 21)) == sha256_file(root / self._literal('_saved_trained_floor_admitted', 22))), self._literal('_saved_trained_floor_admitted', 23))
        self._require(plan.get(self._literal('_saved_trained_floor_admitted', 24)) == seal.get(self._literal('_saved_trained_floor_admitted', 25)) == self._export('CONTRACT') and seal.get(self._literal('_saved_trained_floor_admitted', 26)) is self._literal('_saved_trained_floor_admitted', 27) and ((seal.get(self._literal('_saved_trained_floor_admitted', 28)), seal.get(self._literal('_saved_trained_floor_admitted', 29))) == (self._literal('_saved_trained_floor_admitted', 30), self._literal('_saved_trained_floor_admitted', 31))) and (seal.get(self._literal('_saved_trained_floor_admitted', 32)) == plan.get(self._literal('_saved_trained_floor_admitted', 33)) == identity) and (seal.get(self._literal('_saved_trained_floor_admitted', 34)) == plan.get(self._literal('_saved_trained_floor_admitted', 35))) and (seal.get(self._literal('_saved_trained_floor_admitted', 36)) == plan_sha) and (plan.get(self._literal('_saved_trained_floor_admitted', 37)) == canonical_sha256(plan[self._literal('_saved_trained_floor_admitted', 38)])) and (plan.get(self._literal('_saved_trained_floor_admitted', 39)) == canonical_sha256(dict(binding=plan[self._literal('_saved_trained_floor_admitted', 40)], model_identity=identity))) and all((seal.get(key) is self._literal('_saved_trained_floor_admitted', 41) for key in (self._literal('_saved_trained_floor_admitted', 42), self._literal('_saved_trained_floor_admitted', 43), self._literal('_saved_trained_floor_admitted', 44), self._literal('_saved_trained_floor_admitted', 45), self._literal('_saved_trained_floor_admitted', 46)))), self._literal('_saved_trained_floor_admitted', 47))
        heads = plan[self._literal('_saved_trained_floor_admitted', 48)].get(self._literal('_saved_trained_floor_admitted', 49)) or {}
        self._require((heads.get(self._literal('_saved_trained_floor_admitted', 50)) or {}).get(self._literal('_saved_trained_floor_admitted', 51)) == self._export('CO2_SHA256'), self._literal('_saved_trained_floor_admitted', 52))
        inputs = plan[self._literal('_saved_trained_floor_admitted', 53)][self._literal('_saved_trained_floor_admitted', 54)]
        policy_plan = policy.get(self._literal('_saved_trained_floor_admitted', 55)) if policy else {}
        self._require((inputs.get(self._literal('_saved_trained_floor_admitted', 56)) or {}).get(self._literal('_saved_trained_floor_admitted', 57)) == self._export('SPEC_SHA256'), self._literal('_saved_trained_floor_admitted', 58))
        self._require(inputs.get(self._literal('_saved_trained_floor_admitted', 59)) is self._literal('_saved_trained_floor_admitted', 60) and inputs.get(self._literal('_saved_trained_floor_admitted', 61)) is self._literal('_saved_trained_floor_admitted', 62) and (inputs.get(self._literal('_saved_trained_floor_admitted', 63)) is self._literal('_saved_trained_floor_admitted', 64)) and (inputs.get(self._literal('_saved_trained_floor_admitted', 65)) == dict(regional_threshold=self._literal('_saved_trained_floor_admitted', 66), pixel_threshold=self._literal('_saved_trained_floor_admitted', 67), minimum_mm2=self._literal('_saved_trained_floor_admitted', 68), connectivity=self._literal('_saved_trained_floor_admitted', 69))) and (set(inputs.get(self._literal('_saved_trained_floor_admitted', 70)) or ()) == set(self._export('T3_BLIND_RUNS'))) and (set(seal.get(self._literal('_saved_trained_floor_admitted', 71)) or ()) == set(self._export('T3_BLIND_RUNS'))) and (policy is not self._literal('_saved_trained_floor_admitted', 72)) and (policy.get(self._literal('_saved_trained_floor_admitted', 73)) == self._export('T3_PURPOSE')) and (tuple(policy[self._literal('_saved_trained_floor_admitted', 74)]) == self._export('T3_BLIND_RUNS')) and (policy_plan.get(self._literal('_saved_trained_floor_admitted', 75)) == plan_sha) and resolve_review_path(policy_plan.get(self._literal('_saved_trained_floor_admitted', 76), self._literal('_saved_trained_floor_admitted', 77))).resolve().samefile(plan_path) and (policy.get(self._literal('_saved_trained_floor_admitted', 78)) is self._literal('_saved_trained_floor_admitted', 79)) and (policy.get(self._literal('_saved_trained_floor_admitted', 80)) is self._literal('_saved_trained_floor_admitted', 81)), self._literal('_saved_trained_floor_admitted', 82))
        return dict(registration=registration, pipeline_sha256=plan[self._literal('_saved_trained_floor_admitted', 83)], model_identity=identity)

    def _saved_trained_floor_saved_review(self, model_root):
        root = resolve_review_path(model_root).resolve()
        registration = resolve_review_path(configured_registration_path(root.name)).resolve()
        metadata = self._read(registration).get(self._literal('_saved_trained_floor_saved_review', 0)) or {}
        paths = sorted({registration, root / self._export('PLAN'), root / self._export('SEAL'), root / self._literal('_saved_trained_floor_saved_review', 1), *(root / name for name in metadata)}, key=str)
        stamps = (tuple(configured_model_identity(root)), tuple(((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)))
        return self._admitted(str(root), str(registration), stamps)

    def _saved_trained_floor_validate_head_revision(self, parent, pixel, regional):
        parent = Path(parent)
        if sha256_file(parent / self._literal('_saved_trained_floor_validate_head_revision', 0)) != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_trained_floor_validate_head_revision', 1))
        old, new = (read_json(parent / self._literal('_saved_trained_floor_validate_head_revision', 2)), read_json(regional))
        for value in (old, new):
            value = value[self._literal('_saved_trained_floor_validate_head_revision', 3)].pop(self._literal('_saved_trained_floor_validate_head_revision', 4))
        if old != new:
            raise ValueError(self._literal('_saved_trained_floor_validate_head_revision', 5))
        self.verify_pixel_model(pixel, sha256_file(pixel))

    def _saved_trained_floor_validate_runtime_heads(self, candidate, marker, heads):
        """Dedicated opt-in; original rig profiles keep their original exact hashes."""
        root = Path(candidate)
        if marker.get(self._literal('_saved_trained_floor_validate_runtime_heads', 0)) != self._export('CONTRACT') or marker.get(self._literal('_saved_trained_floor_validate_runtime_heads', 1)) != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_trained_floor_validate_runtime_heads', 2))
        plan = read_json(bound_path(marker[self._literal('_saved_trained_floor_validate_runtime_heads', 3)], base=root))
        if plan.get(self._literal('_saved_trained_floor_validate_runtime_heads', 4)) != self._export('CONTRACT') or plan.get(self._literal('_saved_trained_floor_validate_runtime_heads', 5)) != str(root.resolve()) or plan.get(self._literal('_saved_trained_floor_validate_runtime_heads', 6)) != list(configured_model_identity(root)) or (plan.get(self._literal('_saved_trained_floor_validate_runtime_heads', 7)) != canonical_sha256(plan[self._literal('_saved_trained_floor_validate_runtime_heads', 8)])) or (plan.get(self._literal('_saved_trained_floor_validate_runtime_heads', 9)) != canonical_sha256(dict(binding=plan[self._literal('_saved_trained_floor_validate_runtime_heads', 10)], model_identity=plan[self._literal('_saved_trained_floor_validate_runtime_heads', 11)]))):
            raise ValueError(self._literal('_saved_trained_floor_validate_runtime_heads', 12))
        if plan[self._literal('_saved_trained_floor_validate_runtime_heads', 13)][self._literal('_saved_trained_floor_validate_runtime_heads', 14)][self._literal('_saved_trained_floor_validate_runtime_heads', 15)] != marker[self._literal('_saved_trained_floor_validate_runtime_heads', 16)] or marker[self._literal('_saved_trained_floor_validate_runtime_heads', 17)] != self._export('SPEC_SHA256'):
            raise ValueError(self._literal('_saved_trained_floor_validate_runtime_heads', 18))
        if marker[self._literal('_saved_trained_floor_validate_runtime_heads', 19)] != heads or plan[self._literal('_saved_trained_floor_validate_runtime_heads', 20)][self._literal('_saved_trained_floor_validate_runtime_heads', 21)] != heads:
            raise ValueError(self._literal('_saved_trained_floor_validate_runtime_heads', 22))
        for key, relative in ((self._literal('_saved_trained_floor_validate_runtime_heads', 23), self._literal('_saved_trained_floor_validate_runtime_heads', 24)), (self._literal('_saved_trained_floor_validate_runtime_heads', 25), self._literal('_saved_trained_floor_validate_runtime_heads', 26)), (self._literal('_saved_trained_floor_validate_runtime_heads', 27), self._literal('_saved_trained_floor_validate_runtime_heads', 28))):
            if bound_path(heads[key], base=root) != (root / relative).resolve():
                raise ValueError(self._literal('_saved_trained_floor_validate_runtime_heads', 29))
        self.validate_head_revision(plan[self._literal('_saved_trained_floor_validate_runtime_heads', 30)][self._literal('_saved_trained_floor_validate_runtime_heads', 31)], root / self._literal('_saved_trained_floor_validate_runtime_heads', 32), root / self._literal('_saved_trained_floor_validate_runtime_heads', 33))
        return self._literal('_saved_trained_floor_validate_runtime_heads', 34)

    def _saved_trained_floor_validate_source_inventory(self, sources):
        keys = [(s[self._literal('_saved_trained_floor_validate_source_inventory', 0)], s[self._literal('_saved_trained_floor_validate_source_inventory', 1)]) for s in sources]
        if len(keys) != self._literal('_saved_trained_floor_validate_source_inventory', 2) or len(set(keys)) != self._literal('_saved_trained_floor_validate_source_inventory', 3) or len({k[self._literal('_saved_trained_floor_validate_source_inventory', 4)] for k in keys}) != self._literal('_saved_trained_floor_validate_source_inventory', 5):
            raise ValueError(self._literal('_saved_trained_floor_validate_source_inventory', 6))

    def _saved_trained_floor_registered_warning(self, root, registration):
        try:
            root = Path(root)
            if registration.get(self._literal('_saved_trained_floor_registered_warning', 0)) != self._export('KIND') or registration.get(self._literal('_saved_trained_floor_registered_warning', 1)) != configured_model_identity(root)[self._literal('_saved_trained_floor_registered_warning', 2)] or registration.get(self._literal('_saved_trained_floor_registered_warning', 3)) is not self._literal('_saved_trained_floor_registered_warning', 4) or (registration.get(self._literal('_saved_trained_floor_registered_warning', 5)) is not self._literal('_saved_trained_floor_registered_warning', 6)):
                raise ValueError(self._literal('_saved_trained_floor_registered_warning', 7))
            seal = read_json(root / self._export('SEAL'))
            if seal.get(self._literal('_saved_trained_floor_registered_warning', 8)) is not self._literal('_saved_trained_floor_registered_warning', 9) or (seal.get(self._literal('_saved_trained_floor_registered_warning', 10)), seal.get(self._literal('_saved_trained_floor_registered_warning', 11))) != (self._literal('_saved_trained_floor_registered_warning', 12), self._literal('_saved_trained_floor_registered_warning', 13)) or set(seal.get(self._literal('_saved_trained_floor_registered_warning', 14), [])) != self._export('HIDDEN') or (seal.get(self._literal('_saved_trained_floor_registered_warning', 15)) is not self._literal('_saved_trained_floor_registered_warning', 16)) or (seal.get(self._literal('_saved_trained_floor_registered_warning', 17)) is not self._literal('_saved_trained_floor_registered_warning', 18)):
                raise ValueError(self._literal('_saved_trained_floor_registered_warning', 19))
            for relative, digest in registration[self._literal('_saved_trained_floor_registered_warning', 20)].items():
                if sha256_file(root / relative) != digest:
                    raise ValueError(self._literal('_saved_trained_floor_registered_warning', 21))
            return self._export('WARNING')
        except (OSError, KeyError, ValueError, TypeError):
            return self._literal('_saved_trained_floor_registered_warning', 22)

    def _saved_trained_floor_verify_pixel_model(self, path, expected_sha256):
        return self._legacy_verify_pixel_model(path, expected_sha256, training_revision=self._literal('_saved_trained_floor_verify_pixel_model', 0))

    def _saved_trained_floor_legacy_verify_pixel_model(self, path, expected_sha256, *, training_revision=_SAVED_MISSING):
        if training_revision is _SAVED_MISSING:
            training_revision = self._literal('_saved_trained_floor_legacy_verify_pixel_model', 0)
        if sha256_file(Path(path)) != expected_sha256:
            raise ValueError(self._literal('_saved_trained_floor_legacy_verify_pixel_model', 1))
        model = self._read(path)
        if training_revision not in (self._literal('_saved_trained_floor_legacy_verify_pixel_model', 2), self._literal('_saved_trained_floor_legacy_verify_pixel_model', 3)) or model.get(self._literal('_saved_trained_floor_legacy_verify_pixel_model', 4)) != self._literal('_saved_trained_floor_legacy_verify_pixel_model', 5) or model.get(self._literal('_saved_trained_floor_legacy_verify_pixel_model', 6)) != training_revision or (model.get(self._literal('_saved_trained_floor_legacy_verify_pixel_model', 7)) != list(self._export('FLATFIELD_FEATURE_NAMES'))) or (model.get(self._literal('_saved_trained_floor_legacy_verify_pixel_model', 8), {}).get(self._literal('_saved_trained_floor_legacy_verify_pixel_model', 9)) is not self._literal('_saved_trained_floor_legacy_verify_pixel_model', 10)):
            raise ValueError(self._literal('_saved_trained_floor_legacy_verify_pixel_model', 11))
        return model

    def _saved_epoch_floor_signature(self, path):
        return metadata_file_stamp(path)

    @lru_cache(maxsize=16)
    def _saved_epoch_floor_plan_dependencies(self, root_text, plan_signature, expected_plan_sha):
        """Retain only small asset bindings, keyed by the current registered plan."""
        plan = self._read(Path(root_text) / self._export('PLAN'))
        heads = plan[self._literal('_saved_epoch_floor_plan_dependencies', 0)][self._literal('_saved_epoch_floor_plan_dependencies', 1)]
        assets = [heads[key] for key in (self._literal('_saved_epoch_floor_plan_dependencies', 2), self._literal('_saved_epoch_floor_plan_dependencies', 3), self._literal('_saved_epoch_floor_plan_dependencies', 4))]
        assets += [asset for pair in heads[self._literal('_saved_epoch_floor_plan_dependencies', 5)].values() for asset in pair.values()]
        assets.append(plan[self._literal('_saved_epoch_floor_plan_dependencies', 6)][self._literal('_saved_epoch_floor_plan_dependencies', 7)][self._literal('_saved_epoch_floor_plan_dependencies', 8)])
        return tuple(((asset[self._literal('_saved_epoch_floor_plan_dependencies', 9)], asset[self._literal('_saved_epoch_floor_plan_dependencies', 10)]) for asset in assets))

    @lru_cache(maxsize=16)
    def _saved_epoch_floor_admitted(self, root_text, registration_text, stamps):
        from .phase_saved_registration import registration_warning
        root, registration_path = (Path(root_text), Path(registration_text))
        registration, plan, seal = (self._read(registration_path), self._read(root / self._export('PLAN')), self._read(root / self._export('SEAL')))
        identity = list(configured_model_identity(root))
        plan_sha, seal_sha = (sha256_file(root / self._export('PLAN')), sha256_file(root / self._export('SEAL')))
        metadata = registration.get(self._literal('_saved_epoch_floor_admitted', 0)) or {}
        self._require(registration.get(self._literal('_saved_epoch_floor_admitted', 1)) == self._export('KIND') and registration_warning(root) == self._export('WARNING') and (registration.get(self._literal('_saved_epoch_floor_admitted', 2)) == self._literal('_saved_epoch_floor_admitted', 3)) and ((registration.get(self._literal('_saved_epoch_floor_admitted', 4)), registration.get(self._literal('_saved_epoch_floor_admitted', 5))) == (self._literal('_saved_epoch_floor_admitted', 6), self._literal('_saved_epoch_floor_admitted', 7))) and (registration.get(self._literal('_saved_epoch_floor_admitted', 8)) == identity[self._literal('_saved_epoch_floor_admitted', 9)]) and (registration.get(self._literal('_saved_epoch_floor_admitted', 10)) == identity[self._literal('_saved_epoch_floor_admitted', 11)]) and all((registration.get(key) is self._literal('_saved_epoch_floor_admitted', 12) for key in (self._literal('_saved_epoch_floor_admitted', 13), self._literal('_saved_epoch_floor_admitted', 14), self._literal('_saved_epoch_floor_admitted', 15), self._literal('_saved_epoch_floor_admitted', 16), self._literal('_saved_epoch_floor_admitted', 17)))), self._literal('_saved_epoch_floor_admitted', 18))
        self._require(metadata.get(self._export('PLAN')) == plan_sha == registration.get(self._literal('_saved_epoch_floor_admitted', 19)) and metadata.get(self._export('SEAL')) == seal_sha == registration.get(self._literal('_saved_epoch_floor_admitted', 20)), self._literal('_saved_epoch_floor_admitted', 21))
        self._require(plan.get(self._literal('_saved_epoch_floor_admitted', 22)) == seal.get(self._literal('_saved_epoch_floor_admitted', 23)) == self._export('CONTRACT') and seal.get(self._literal('_saved_epoch_floor_admitted', 24)) is self._literal('_saved_epoch_floor_admitted', 25) and ((seal.get(self._literal('_saved_epoch_floor_admitted', 26)), seal.get(self._literal('_saved_epoch_floor_admitted', 27))) == (self._literal('_saved_epoch_floor_admitted', 28), self._literal('_saved_epoch_floor_admitted', 29))) and (seal.get(self._literal('_saved_epoch_floor_admitted', 30)) == plan.get(self._literal('_saved_epoch_floor_admitted', 31)) == identity) and (seal.get(self._literal('_saved_epoch_floor_admitted', 32)) == plan.get(self._literal('_saved_epoch_floor_admitted', 33))) and (seal.get(self._literal('_saved_epoch_floor_admitted', 34)) == plan_sha) and (plan.get(self._literal('_saved_epoch_floor_admitted', 35)) == canonical_sha256(plan[self._literal('_saved_epoch_floor_admitted', 36)])) and (plan.get(self._literal('_saved_epoch_floor_admitted', 37)) == canonical_sha256(dict(binding=plan[self._literal('_saved_epoch_floor_admitted', 38)], model_identity=identity))) and all((seal.get(key) is self._literal('_saved_epoch_floor_admitted', 39) for key in (self._literal('_saved_epoch_floor_admitted', 40), self._literal('_saved_epoch_floor_admitted', 41), self._literal('_saved_epoch_floor_admitted', 42), self._literal('_saved_epoch_floor_admitted', 43), self._literal('_saved_epoch_floor_admitted', 44)))), self._literal('_saved_epoch_floor_admitted', 45))
        inputs = plan[self._literal('_saved_epoch_floor_admitted', 46)][self._literal('_saved_epoch_floor_admitted', 47)]
        self._require((plan[self._literal('_saved_epoch_floor_admitted', 48)][self._literal('_saved_epoch_floor_admitted', 49)].get(self._literal('_saved_epoch_floor_admitted', 50)) or {}).get(self._literal('_saved_epoch_floor_admitted', 51)) == self._export('CO2_SHA256'), self._literal('_saved_epoch_floor_admitted', 52))
        self._require((inputs.get(self._literal('_saved_epoch_floor_admitted', 53)) or {}).get(self._literal('_saved_epoch_floor_admitted', 54)) == self._export('SPEC_SHA256') and inputs.get(self._literal('_saved_epoch_floor_admitted', 55)) is self._literal('_saved_epoch_floor_admitted', 56) and (inputs.get(self._literal('_saved_epoch_floor_admitted', 57)) is self._literal('_saved_epoch_floor_admitted', 58)) and (inputs.get(self._literal('_saved_epoch_floor_admitted', 59)) is self._literal('_saved_epoch_floor_admitted', 60)) and (inputs.get(self._literal('_saved_epoch_floor_admitted', 61)) == self._export('FIXED_CONFIGURATION')) and (inputs.get(self._literal('_saved_epoch_floor_admitted', 62)) == seal.get(self._literal('_saved_epoch_floor_admitted', 63)) == []) and (inputs.get(self._literal('_saved_epoch_floor_admitted', 64)) == []) and (not (root / self._literal('_saved_epoch_floor_admitted', 65)).exists()), self._literal('_saved_epoch_floor_admitted', 66))
        self.validate_source_inventory(plan[self._literal('_saved_epoch_floor_admitted', 67)])
        self._require(inputs.get(self._literal('_saved_epoch_floor_admitted', 68)) == canonical_sha256(plan[self._literal('_saved_epoch_floor_admitted', 69)]), self._literal('_saved_epoch_floor_admitted', 70))
        profile = self._read(root / self._literal('_saved_epoch_floor_admitted', 71))
        self.validate_runtime_heads(root, profile[self._literal('_saved_epoch_floor_admitted', 72)], profile[self._literal('_saved_epoch_floor_admitted', 73)], verified_plan=plan)
        self._require(profile[self._literal('_saved_epoch_floor_admitted', 74)] == seal.get(self._literal('_saved_epoch_floor_admitted', 75)), self._literal('_saved_epoch_floor_admitted', 76))
        return dict(registration=registration, pipeline_sha256=plan[self._literal('_saved_epoch_floor_admitted', 77)], model_identity=identity, light_epochs=plan[self._literal('_saved_epoch_floor_admitted', 78)], blind_hold_runs=[])

    def _saved_epoch_floor_saved_review(self, model_root):
        root = resolve_review_path(model_root).resolve()
        registration = resolve_review_path(configured_registration_path(root.name)).resolve()
        current_registration = self._read(registration)
        metadata = current_registration.get(self._literal('_saved_epoch_floor_saved_review', 0)) or {}
        expected_plan_sha = metadata.get(self._export('PLAN'))
        self._require(isinstance(expected_plan_sha, str) and expected_plan_sha == current_registration.get(self._literal('_saved_epoch_floor_saved_review', 1)), self._literal('_saved_epoch_floor_saved_review', 2))
        assets = self._plan_dependencies(str(root), self._signature(root / self._export('PLAN')), expected_plan_sha)
        paths = sorted({registration, root / self._export('PLAN'), root / self._export('SEAL'), *(root / name for name in metadata), *(bound_path(dict(path=path, sha256=sha), base=root) for path, sha in assets)}, key=str)
        bottom_policy = root / self._literal('_saved_epoch_floor_saved_review', 3)
        stamps = (tuple(configured_model_identity(root)), expected_plan_sha, tuple(((str(path), self._signature(path)) for path in paths)), self._signature(bottom_policy) if bottom_policy.exists() else self._literal('_saved_epoch_floor_saved_review', 4))
        return self._admitted(str(root), str(registration), stamps)

    def _saved_epoch_floor_validate_head_revision(self, parent, pixel, regional, epoch):
        if sha256_file(Path(parent) / self._literal('_saved_epoch_floor_validate_head_revision', 0)) != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_epoch_floor_validate_head_revision', 1))
        old, new = (read_json(Path(parent) / self._literal('_saved_epoch_floor_validate_head_revision', 2)), read_json(regional))
        for value in (old, new):
            value[self._literal('_saved_epoch_floor_validate_head_revision', 3)].pop(self._literal('_saved_epoch_floor_validate_head_revision', 4))
        if old != new:
            raise ValueError(self._literal('_saved_epoch_floor_validate_head_revision', 5))
        self.verify_pixel_model(pixel, sha256_file(pixel), light_epoch_id=epoch)

    def _saved_epoch_floor_validate_runtime_heads(self, candidate, marker, heads, *, verified_plan=_SAVED_MISSING):
        if verified_plan is _SAVED_MISSING:
            verified_plan = self._literal('_saved_epoch_floor_validate_runtime_heads', 0)
        root = Path(candidate).resolve()
        plan_path = bound_path(marker[self._literal('_saved_epoch_floor_validate_runtime_heads', 1)], base=root)
        plan = read_json(plan_path) if verified_plan is self._literal('_saved_epoch_floor_validate_runtime_heads', 2) else verified_plan
        if plan_path != root / self._export('PLAN') or marker.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 3)) != self._export('CONTRACT') or marker.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 4)) != self._export('CO2_SHA256') or (plan.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 5)) != self._export('CONTRACT')) or (plan.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 6)) != str(root)) or (plan.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 7)) != list(configured_model_identity(root))) or (plan.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 8)) != canonical_sha256(plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 9)])) or (plan.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 10)) != canonical_sha256(dict(binding=plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 11)], model_identity=plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 12)]))):
            raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 13))
        if plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 14)][self._literal('_saved_epoch_floor_validate_runtime_heads', 15)][self._literal('_saved_epoch_floor_validate_runtime_heads', 16)] != marker[self._literal('_saved_epoch_floor_validate_runtime_heads', 17)] or marker[self._literal('_saved_epoch_floor_validate_runtime_heads', 18)] != self._export('SPEC_SHA256'):
            raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 19))
        if heads != marker.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 20)) or heads != plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 21)][self._literal('_saved_epoch_floor_validate_runtime_heads', 22)] or heads[self._literal('_saved_epoch_floor_validate_runtime_heads', 23)][self._literal('_saved_epoch_floor_validate_runtime_heads', 24)] != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 25))
        for key, relative in ((self._literal('_saved_epoch_floor_validate_runtime_heads', 26), self._literal('_saved_epoch_floor_validate_runtime_heads', 27)), (self._literal('_saved_epoch_floor_validate_runtime_heads', 28), self._literal('_saved_epoch_floor_validate_runtime_heads', 29)), (self._literal('_saved_epoch_floor_validate_runtime_heads', 30), self._literal('_saved_epoch_floor_validate_runtime_heads', 31))):
            if bound_path(heads[key], base=root) != (root / relative).resolve():
                raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 32))
            if heads[key][self._literal('_saved_epoch_floor_validate_runtime_heads', 33)] != {self._literal('_saved_epoch_floor_validate_runtime_heads', 34): self._export('CO2_SHA256'), self._literal('_saved_epoch_floor_validate_runtime_heads', 35): self._export('G3_SHA256'), self._literal('_saved_epoch_floor_validate_runtime_heads', 36): self._export('REFIT_SHA256')}[key]:
                raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 37))
        table = self.load_light_epochs(bound_path(plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 38)][self._literal('_saved_epoch_floor_validate_runtime_heads', 39)][self._literal('_saved_epoch_floor_validate_runtime_heads', 40)]))
        if table[self._literal('_saved_epoch_floor_validate_runtime_heads', 41)] != plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 42)] or set(heads[self._literal('_saved_epoch_floor_validate_runtime_heads', 43)]) != set(table[self._literal('_saved_epoch_floor_validate_runtime_heads', 44)]):
            raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 45))
        if marker.get(self._literal('_saved_epoch_floor_validate_runtime_heads', 46)) != plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 47)][self._literal('_saved_epoch_floor_validate_runtime_heads', 48)][self._literal('_saved_epoch_floor_validate_runtime_heads', 49)]:
            raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 50))
        for epoch, pair in heads[self._literal('_saved_epoch_floor_validate_runtime_heads', 51)].items():
            pixel, regional = (bound_path(pair[self._literal('_saved_epoch_floor_validate_runtime_heads', 52)], base=root), bound_path(pair[self._literal('_saved_epoch_floor_validate_runtime_heads', 53)], base=root))
            if pixel != (root / f"{self._literal('_saved_epoch_floor_validate_runtime_heads', 54)}{epoch}{self._literal('_saved_epoch_floor_validate_runtime_heads', 55)}").resolve() or regional != (root / f"{self._literal('_saved_epoch_floor_validate_runtime_heads', 56)}{epoch}{self._literal('_saved_epoch_floor_validate_runtime_heads', 57)}").resolve():
                raise ValueError(self._literal('_saved_epoch_floor_validate_runtime_heads', 58))
            self.validate_head_revision(plan[self._literal('_saved_epoch_floor_validate_runtime_heads', 59)][self._literal('_saved_epoch_floor_validate_runtime_heads', 60)], pixel, regional, epoch)
        return self._literal('_saved_epoch_floor_validate_runtime_heads', 61)

    def _saved_epoch_floor_verify_pixel_model(self, path, expected_sha256, *, light_epoch_id=_SAVED_MISSING):
        if light_epoch_id is _SAVED_MISSING:
            light_epoch_id = self._literal('_saved_epoch_floor_verify_pixel_model', 0)
        if sha256_file(Path(path)) != expected_sha256:
            raise ValueError(self._literal('_saved_epoch_floor_verify_pixel_model', 1))
        model = read_json(path)
        if model.get(self._literal('_saved_epoch_floor_verify_pixel_model', 2)) != self._literal('_saved_epoch_floor_verify_pixel_model', 3) or model.get(self._literal('_saved_epoch_floor_verify_pixel_model', 4)) != self._literal('_saved_epoch_floor_verify_pixel_model', 5) or model.get(self._literal('_saved_epoch_floor_verify_pixel_model', 6)) != list(self._export('FLATFIELD_FEATURE_NAMES')) or (model.get(self._literal('_saved_epoch_floor_verify_pixel_model', 7), {}).get(self._literal('_saved_epoch_floor_verify_pixel_model', 8)) is not self._literal('_saved_epoch_floor_verify_pixel_model', 9)) or (not model.get(self._literal('_saved_epoch_floor_verify_pixel_model', 10))) or (light_epoch_id is not self._literal('_saved_epoch_floor_verify_pixel_model', 11) and model[self._literal('_saved_epoch_floor_verify_pixel_model', 12)] != light_epoch_id):
            raise ValueError(self._literal('_saved_epoch_floor_verify_pixel_model', 13))
        return model

    def _saved_epoch_floor_load_light_epochs(self, path):
        """Load source-declared epoch membership; never infer it from run numbers."""
        path = resolve_review_path(path).resolve()
        table = self.read(path)
        if table.get(self._literal('_saved_epoch_floor_load_light_epochs', 0)) != self._literal('_saved_epoch_floor_load_light_epochs', 1) or table.get(self._literal('_saved_epoch_floor_load_light_epochs', 2)) != self._literal('_saved_epoch_floor_load_light_epochs', 3) or table.get(self._literal('_saved_epoch_floor_load_light_epochs', 4)) != [self._literal('_saved_epoch_floor_load_light_epochs', 5), self._literal('_saved_epoch_floor_load_light_epochs', 6)] or (table.get(self._literal('_saved_epoch_floor_load_light_epochs', 7), {}).get(self._literal('_saved_epoch_floor_load_light_epochs', 8)) != self._export('SPEC_SHA256')):
            raise ValueError(self._literal('_saved_epoch_floor_load_light_epochs', 9))
        mapping = {}
        for row in table[self._literal('_saved_epoch_floor_load_light_epochs', 10)]:
            run, epoch = (row[self._literal('_saved_epoch_floor_load_light_epochs', 11)].casefold(), row[self._literal('_saved_epoch_floor_load_light_epochs', 12)])
            if run in mapping or epoch not in table[self._literal('_saved_epoch_floor_load_light_epochs', 13)] or (not run.startswith(self._literal('_saved_epoch_floor_load_light_epochs', 14))):
                raise ValueError(self._literal('_saved_epoch_floor_load_light_epochs', 15))
            mapping[run] = epoch
        if not mapping or set(mapping.values()) != set(table[self._literal('_saved_epoch_floor_load_light_epochs', 16)]):
            raise ValueError(self._literal('_saved_epoch_floor_load_light_epochs', 17))
        return dict(epochs=list(table[self._literal('_saved_epoch_floor_load_light_epochs', 18)]), run_to_epoch=mapping, receipt=dict(contract=table[self._literal('_saved_epoch_floor_load_light_epochs', 19)], path=str(path), sha256=sha256_file(path), spec_sha256=self._export('SPEC_SHA256'), run_to_epoch=mapping, epochs=list(table[self._literal('_saved_epoch_floor_load_light_epochs', 20)]), provenance=table[self._literal('_saved_epoch_floor_load_light_epochs', 21)]))

    def _saved_epoch_floor_read(self, path):
        return json.loads(resolve_review_path(path).read_text(encoding=self._literal('_saved_epoch_floor_read', 0)))

    @lru_cache(maxsize=16)
    def _saved_joint_floor_plan_dependencies(self, root_text, plan_signature, expected_plan_sha):
        """Retain only small asset bindings, keyed by the current registered plan."""
        plan = self._read(Path(root_text) / self._export('PLAN'))
        heads = plan[self._literal('_saved_joint_floor_plan_dependencies', 0)][self._literal('_saved_joint_floor_plan_dependencies', 1)]
        assets = [heads[key] for key in (self._literal('_saved_joint_floor_plan_dependencies', 2), self._literal('_saved_joint_floor_plan_dependencies', 3), self._literal('_saved_joint_floor_plan_dependencies', 4))]
        assets += list(heads[self._literal('_saved_joint_floor_plan_dependencies', 5)].values())
        assets += list(plan[self._literal('_saved_joint_floor_plan_dependencies', 6)][self._literal('_saved_joint_floor_plan_dependencies', 7)][self._literal('_saved_joint_floor_plan_dependencies', 8)].values())
        assets.append(plan[self._literal('_saved_joint_floor_plan_dependencies', 9)][self._literal('_saved_joint_floor_plan_dependencies', 10)][self._literal('_saved_joint_floor_plan_dependencies', 11)])
        assets.append(plan[self._literal('_saved_joint_floor_plan_dependencies', 12)][self._literal('_saved_joint_floor_plan_dependencies', 13)][self._literal('_saved_joint_floor_plan_dependencies', 14)])
        return tuple(((asset[self._literal('_saved_joint_floor_plan_dependencies', 15)], asset[self._literal('_saved_joint_floor_plan_dependencies', 16)]) for asset in assets))

    def _saved_joint_floor_validate_head_revision(self, parent, pixel, regional):
        if sha256_file(Path(parent) / self._literal('_saved_joint_floor_validate_head_revision', 0)) != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_joint_floor_validate_head_revision', 1))
        old, new = (read_json(Path(parent) / self._literal('_saved_joint_floor_validate_head_revision', 2)), read_json(regional))
        for value in (old, new):
            value[self._literal('_saved_joint_floor_validate_head_revision', 3)].pop(self._literal('_saved_joint_floor_validate_head_revision', 4))
        if old != new:
            raise ValueError(self._literal('_saved_joint_floor_validate_head_revision', 5))
        self.verify_pixel_model(pixel, sha256_file(pixel))
        self.verify_regional_model(regional, sha256_file(regional), parent)

    def _saved_joint_floor_validate_runtime_heads(self, candidate, marker, heads, *, verified_plan=_SAVED_MISSING):
        if verified_plan is _SAVED_MISSING:
            verified_plan = self._literal('_saved_joint_floor_validate_runtime_heads', 0)
        root = Path(candidate).resolve()
        plan_path = bound_path(marker[self._literal('_saved_joint_floor_validate_runtime_heads', 1)], base=root)
        plan = read_json(plan_path) if verified_plan is self._literal('_saved_joint_floor_validate_runtime_heads', 2) else verified_plan
        if plan_path != root / self._export('PLAN') or marker.get(self._literal('_saved_joint_floor_validate_runtime_heads', 3)) != self._export('CONTRACT') or marker.get(self._literal('_saved_joint_floor_validate_runtime_heads', 4)) != self._export('CO2_SHA256') or (plan.get(self._literal('_saved_joint_floor_validate_runtime_heads', 5)) != self._export('CONTRACT')) or (plan.get(self._literal('_saved_joint_floor_validate_runtime_heads', 6)) != str(root)) or (plan.get(self._literal('_saved_joint_floor_validate_runtime_heads', 7)) != list(configured_model_identity(root))) or (plan.get(self._literal('_saved_joint_floor_validate_runtime_heads', 8)) != canonical_sha256(plan[self._literal('_saved_joint_floor_validate_runtime_heads', 9)])) or (plan.get(self._literal('_saved_joint_floor_validate_runtime_heads', 10)) != canonical_sha256(dict(binding=plan[self._literal('_saved_joint_floor_validate_runtime_heads', 11)], model_identity=plan[self._literal('_saved_joint_floor_validate_runtime_heads', 12)]))):
            raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 13))
        if plan[self._literal('_saved_joint_floor_validate_runtime_heads', 14)][self._literal('_saved_joint_floor_validate_runtime_heads', 15)][self._literal('_saved_joint_floor_validate_runtime_heads', 16)] != marker[self._literal('_saved_joint_floor_validate_runtime_heads', 17)] or marker[self._literal('_saved_joint_floor_validate_runtime_heads', 18)] != self._export('SPEC_SHA256'):
            raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 19))
        if heads != marker.get(self._literal('_saved_joint_floor_validate_runtime_heads', 20)) or heads != plan[self._literal('_saved_joint_floor_validate_runtime_heads', 21)][self._literal('_saved_joint_floor_validate_runtime_heads', 22)] or heads[self._literal('_saved_joint_floor_validate_runtime_heads', 23)][self._literal('_saved_joint_floor_validate_runtime_heads', 24)] != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 25))
        for key, relative in ((self._literal('_saved_joint_floor_validate_runtime_heads', 26), self._literal('_saved_joint_floor_validate_runtime_heads', 27)), (self._literal('_saved_joint_floor_validate_runtime_heads', 28), self._literal('_saved_joint_floor_validate_runtime_heads', 29)), (self._literal('_saved_joint_floor_validate_runtime_heads', 30), self._literal('_saved_joint_floor_validate_runtime_heads', 31))):
            if bound_path(heads[key], base=root) != (root / relative).resolve():
                raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 32))
            if heads[key][self._literal('_saved_joint_floor_validate_runtime_heads', 33)] != {self._literal('_saved_joint_floor_validate_runtime_heads', 34): self._export('CO2_SHA256'), self._literal('_saved_joint_floor_validate_runtime_heads', 35): self._export('G3_SHA256'), self._literal('_saved_joint_floor_validate_runtime_heads', 36): self._export('REFIT_SHA256')}[key]:
                raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 37))
        table = self.load_light_epochs(bound_path(plan[self._literal('_saved_joint_floor_validate_runtime_heads', 38)][self._literal('_saved_joint_floor_validate_runtime_heads', 39)][self._literal('_saved_joint_floor_validate_runtime_heads', 40)]))
        if table[self._literal('_saved_joint_floor_validate_runtime_heads', 41)] != plan[self._literal('_saved_joint_floor_validate_runtime_heads', 42)] or self._literal('_saved_joint_floor_validate_runtime_heads', 43) in heads or set(heads.get(self._literal('_saved_joint_floor_validate_runtime_heads', 44), {})) != {self._literal('_saved_joint_floor_validate_runtime_heads', 45), self._literal('_saved_joint_floor_validate_runtime_heads', 46)}:
            raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 47))
        if marker.get(self._literal('_saved_joint_floor_validate_runtime_heads', 48)) != plan[self._literal('_saved_joint_floor_validate_runtime_heads', 49)][self._literal('_saved_joint_floor_validate_runtime_heads', 50)][self._literal('_saved_joint_floor_validate_runtime_heads', 51)]:
            raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 52))
        pair = heads[self._literal('_saved_joint_floor_validate_runtime_heads', 53)]
        pixel, regional = (bound_path(pair[self._literal('_saved_joint_floor_validate_runtime_heads', 54)], base=root), bound_path(pair[self._literal('_saved_joint_floor_validate_runtime_heads', 55)], base=root))
        if pixel != (root / self._literal('_saved_joint_floor_validate_runtime_heads', 56)).resolve() or regional != (root / self._literal('_saved_joint_floor_validate_runtime_heads', 57)).resolve():
            raise ValueError(self._literal('_saved_joint_floor_validate_runtime_heads', 58))
        self.validate_head_revision(plan[self._literal('_saved_joint_floor_validate_runtime_heads', 59)][self._literal('_saved_joint_floor_validate_runtime_heads', 60)], pixel, regional)
        return self._literal('_saved_joint_floor_validate_runtime_heads', 61)

    def _saved_joint_floor_verify_pixel_model(self, path, expected_sha256):
        if sha256_file(Path(path)) != expected_sha256:
            raise ValueError(self._literal('_saved_joint_floor_verify_pixel_model', 0))
        model = read_json(path)
        fit = model.get(self._literal('_saved_joint_floor_verify_pixel_model', 1), {})
        if model.get(self._literal('_saved_joint_floor_verify_pixel_model', 2)) != self._literal('_saved_joint_floor_verify_pixel_model', 3) or model.get(self._literal('_saved_joint_floor_verify_pixel_model', 4)) != self._literal('_saved_joint_floor_verify_pixel_model', 5) or model.get(self._literal('_saved_joint_floor_verify_pixel_model', 6)) != list(self._export('FEATURE_NAMES')) or (model.get(self._literal('_saved_joint_floor_verify_pixel_model', 7)) is not self._literal('_saved_joint_floor_verify_pixel_model', 8)) or (fit.get(self._literal('_saved_joint_floor_verify_pixel_model', 9)) is not self._literal('_saved_joint_floor_verify_pixel_model', 10)) or (fit.get(self._literal('_saved_joint_floor_verify_pixel_model', 11)) != self._literal('_saved_joint_floor_verify_pixel_model', 12)) or any((np.asarray(fit.get(k, [])).shape != (self._literal('_saved_joint_floor_verify_pixel_model', 13),) for k in (self._literal('_saved_joint_floor_verify_pixel_model', 14), self._literal('_saved_joint_floor_verify_pixel_model', 15), self._literal('_saved_joint_floor_verify_pixel_model', 16)))) or (not all((np.isfinite(fit[k]).all() for k in (self._literal('_saved_joint_floor_verify_pixel_model', 17), self._literal('_saved_joint_floor_verify_pixel_model', 18), self._literal('_saved_joint_floor_verify_pixel_model', 19))))) or np.any(np.asarray(fit[self._literal('_saved_joint_floor_verify_pixel_model', 20)]) <= self._literal('_saved_joint_floor_verify_pixel_model', 21)) or (not np.isfinite(fit[self._literal('_saved_joint_floor_verify_pixel_model', 22)])):
            raise ValueError(self._literal('_saved_joint_floor_verify_pixel_model', 23))
        return model

    def _saved_joint_floor_load_light_epochs(self, path):
        return self._base_load_light_epochs(self.checked(path, self._export('LIGHT_EPOCHS_SHA256')))

    def _saved_joint_floor_checked(self, path, expected):
        path = resolve_review_path(path)
        if sha256_file(path) != expected:
            raise ValueError(f"{self._literal('_saved_joint_floor_checked', 0)}{path}")
        return path

    def _saved_joint_floor_verify_regional_model(self, path, expected_sha256, parent):
        model = self._base_verify_regional_model(path, expected_sha256, parent)
        if model[self._literal('_saved_joint_floor_verify_regional_model', 0)] != list(self._export('regional.FEATURE_NAMES')) or len(model[self._literal('_saved_joint_floor_verify_regional_model', 1)]) != self._literal('_saved_joint_floor_verify_regional_model', 2):
            raise ValueError(self._literal('_saved_joint_floor_verify_regional_model', 3))
        return model

    def _saved_joint_floor_base_verify_regional_model(self, path, expected_sha256, parent):
        if sha256_file(Path(path)) != expected_sha256:
            raise ValueError(self._literal('_saved_joint_floor_base_verify_regional_model', 0))
        frozen = read_json(Path(parent) / self._literal('_saved_joint_floor_base_verify_regional_model', 1))
        model = read_json(path)
        self._validate_refit(frozen, model)
        before, after = (deepcopy(frozen), deepcopy(model))
        before[self._literal('_saved_joint_floor_base_verify_regional_model', 2)].pop(self._literal('_saved_joint_floor_base_verify_regional_model', 3))
        after[self._literal('_saved_joint_floor_base_verify_regional_model', 4)].pop(self._literal('_saved_joint_floor_base_verify_regional_model', 5))
        if before != after:
            raise ValueError(self._literal('_saved_joint_floor_base_verify_regional_model', 6))
        if model[self._literal('_saved_joint_floor_base_verify_regional_model', 7)][self._literal('_saved_joint_floor_base_verify_regional_model', 8)].get(self._literal('_saved_joint_floor_base_verify_regional_model', 9)) is not self._literal('_saved_joint_floor_base_verify_regional_model', 10):
            raise ValueError(self._literal('_saved_joint_floor_base_verify_regional_model', 11))
        return model

    def _saved_joint_floor_validate_refit(self, frozen, refit):
        if frozen[self._literal('_saved_joint_floor_validate_refit', 0)] != refit[self._literal('_saved_joint_floor_validate_refit', 1)]:
            raise ValueError(self._literal('_saved_joint_floor_validate_refit', 2))
        if frozen.get(self._literal('_saved_joint_floor_validate_refit', 3)) != refit.get(self._literal('_saved_joint_floor_validate_refit', 4)):
            raise ValueError(self._literal('_saved_joint_floor_validate_refit', 5))
        if set(frozen[self._literal('_saved_joint_floor_validate_refit', 6)]) != set(refit[self._literal('_saved_joint_floor_validate_refit', 7)]):
            raise ValueError(self._literal('_saved_joint_floor_validate_refit', 8))
        for name in frozen[self._literal('_saved_joint_floor_validate_refit', 9)]:
            if name != self._export('HEAD') and frozen[self._literal('_saved_joint_floor_validate_refit', 10)][name] != refit[self._literal('_saved_joint_floor_validate_refit', 11)][name]:
                raise ValueError(self._literal('_saved_joint_floor_validate_refit', 12))

    @lru_cache(maxsize=16)
    def _saved_neighbourhood_floor_admitted(self, root_text, registration_text, stamps):
        from .phase_rig_bottom_review_policy import load_policy
        from .phase_saved_registration import registration_warning
        root, registration_path = (Path(root_text), Path(registration_text))
        registration = self._read(registration_path)
        plan_path, seal_path = (root / self._export('PLAN'), root / self._export('SEAL'))
        plan, seal = (self._read(plan_path), self._read(seal_path))
        identity = list(configured_model_identity(root))
        plan_sha, seal_sha = (sha256_file(plan_path), sha256_file(seal_path))
        metadata = registration.get(self._literal('_saved_neighbourhood_floor_admitted', 0)) or {}
        policy = load_policy(root)
        self._require(registration.get(self._literal('_saved_neighbourhood_floor_admitted', 1)) == self._export('KIND') and registration_warning(root) == self._export('WARNING') and (registration.get(self._literal('_saved_neighbourhood_floor_admitted', 2)) == self._literal('_saved_neighbourhood_floor_admitted', 3)) and ((registration.get(self._literal('_saved_neighbourhood_floor_admitted', 4)), registration.get(self._literal('_saved_neighbourhood_floor_admitted', 5))) == (self._literal('_saved_neighbourhood_floor_admitted', 6), self._literal('_saved_neighbourhood_floor_admitted', 7))) and (registration.get(self._literal('_saved_neighbourhood_floor_admitted', 8)) == identity[self._literal('_saved_neighbourhood_floor_admitted', 9)]) and (registration.get(self._literal('_saved_neighbourhood_floor_admitted', 10)) == identity[self._literal('_saved_neighbourhood_floor_admitted', 11)]) and all((registration.get(key) is self._literal('_saved_neighbourhood_floor_admitted', 12) for key in (self._literal('_saved_neighbourhood_floor_admitted', 13), self._literal('_saved_neighbourhood_floor_admitted', 14), self._literal('_saved_neighbourhood_floor_admitted', 15), self._literal('_saved_neighbourhood_floor_admitted', 16), self._literal('_saved_neighbourhood_floor_admitted', 17)))), self._literal('_saved_neighbourhood_floor_admitted', 18))
        self._require(metadata.get(self._export('PLAN')) == plan_sha == registration.get(self._literal('_saved_neighbourhood_floor_admitted', 19)) and metadata.get(self._export('SEAL')) == seal_sha == registration.get(self._literal('_saved_neighbourhood_floor_admitted', 20)) and (metadata.get(self._literal('_saved_neighbourhood_floor_admitted', 21)) == sha256_file(root / self._literal('_saved_neighbourhood_floor_admitted', 22))), self._literal('_saved_neighbourhood_floor_admitted', 23))
        self._require(plan.get(self._literal('_saved_neighbourhood_floor_admitted', 24)) == seal.get(self._literal('_saved_neighbourhood_floor_admitted', 25)) == self._export('CONTRACT') and seal.get(self._literal('_saved_neighbourhood_floor_admitted', 26)) is self._literal('_saved_neighbourhood_floor_admitted', 27) and ((seal.get(self._literal('_saved_neighbourhood_floor_admitted', 28)), seal.get(self._literal('_saved_neighbourhood_floor_admitted', 29))) == (self._literal('_saved_neighbourhood_floor_admitted', 30), self._literal('_saved_neighbourhood_floor_admitted', 31))) and (seal.get(self._literal('_saved_neighbourhood_floor_admitted', 32)) == plan.get(self._literal('_saved_neighbourhood_floor_admitted', 33)) == identity) and (seal.get(self._literal('_saved_neighbourhood_floor_admitted', 34)) == plan.get(self._literal('_saved_neighbourhood_floor_admitted', 35))) and (seal.get(self._literal('_saved_neighbourhood_floor_admitted', 36)) == plan_sha) and (plan.get(self._literal('_saved_neighbourhood_floor_admitted', 37)) == canonical_sha256(plan[self._literal('_saved_neighbourhood_floor_admitted', 38)])) and (plan.get(self._literal('_saved_neighbourhood_floor_admitted', 39)) == canonical_sha256(dict(binding=plan[self._literal('_saved_neighbourhood_floor_admitted', 40)], model_identity=identity))) and all((seal.get(key) is self._literal('_saved_neighbourhood_floor_admitted', 41) for key in (self._literal('_saved_neighbourhood_floor_admitted', 42), self._literal('_saved_neighbourhood_floor_admitted', 43), self._literal('_saved_neighbourhood_floor_admitted', 44), self._literal('_saved_neighbourhood_floor_admitted', 45), self._literal('_saved_neighbourhood_floor_admitted', 46)))), self._literal('_saved_neighbourhood_floor_admitted', 47))
        heads = plan[self._literal('_saved_neighbourhood_floor_admitted', 48)].get(self._literal('_saved_neighbourhood_floor_admitted', 49)) or {}
        self._require((heads.get(self._literal('_saved_neighbourhood_floor_admitted', 50)) or {}).get(self._literal('_saved_neighbourhood_floor_admitted', 51)) == self._export('CO2_SHA256') and (heads.get(self._literal('_saved_neighbourhood_floor_admitted', 52)) or {}).get(self._literal('_saved_neighbourhood_floor_admitted', 53)) == self._export('G3_SHA256') and ((heads.get(self._literal('_saved_neighbourhood_floor_admitted', 54)) or {}).get(self._literal('_saved_neighbourhood_floor_admitted', 55)) == self._export('REFIT_SHA256')), self._literal('_saved_neighbourhood_floor_admitted', 56))
        inputs = plan[self._literal('_saved_neighbourhood_floor_admitted', 57)][self._literal('_saved_neighbourhood_floor_admitted', 58)]
        policy_plan = policy.get(self._literal('_saved_neighbourhood_floor_admitted', 59)) if policy else {}
        spec_sha = (inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 60)) or {}).get(self._literal('_saved_neighbourhood_floor_admitted', 61))
        try:
            settings = self.approved_spec_settings(spec_sha)
        except (TypeError, ValueError) as error:
            raise ValueError(self._literal('_saved_neighbourhood_floor_admitted', 62)) from error
        bound_spec = plan[self._literal('_saved_neighbourhood_floor_admitted', 63)].get(self._literal('_saved_neighbourhood_floor_admitted', 64))
        self._require(bound_spec is self._literal('_saved_neighbourhood_floor_admitted', 65) or (isinstance(bound_spec, dict) and bound_spec.get(self._literal('_saved_neighbourhood_floor_admitted', 66)) == spec_sha), self._literal('_saved_neighbourhood_floor_admitted', 67))
        self._require(inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 68)) is self._literal('_saved_neighbourhood_floor_admitted', 69) and inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 70)) is self._literal('_saved_neighbourhood_floor_admitted', 71) and (inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 72)) is self._literal('_saved_neighbourhood_floor_admitted', 73)) and (inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 74)) is self._literal('_saved_neighbourhood_floor_admitted', 75)) and (inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 76)) == dict(activation_h=self._literal('_saved_neighbourhood_floor_admitted', 77), radius_px=self._literal('_saved_neighbourhood_floor_admitted', 78), box_hw=[self._literal('_saved_neighbourhood_floor_admitted', 79), self._literal('_saved_neighbourhood_floor_admitted', 80)], gas_fraction=settings[self._literal('_saved_neighbourhood_floor_admitted', 81)], iterations=self._literal('_saved_neighbourhood_floor_admitted', 82))) and (inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 83)) is self._literal('_saved_neighbourhood_floor_admitted', 84)) and (inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 85)) is self._literal('_saved_neighbourhood_floor_admitted', 86)) and (inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 87)) == dict(regional_threshold=self._literal('_saved_neighbourhood_floor_admitted', 88), pixel_threshold=self._literal('_saved_neighbourhood_floor_admitted', 89), minimum_mm2=self._literal('_saved_neighbourhood_floor_admitted', 90), connectivity=self._literal('_saved_neighbourhood_floor_admitted', 91))) and (set(inputs.get(self._literal('_saved_neighbourhood_floor_admitted', 92)) or ()) == set(self._export('T3_BLIND_RUNS'))) and (set(seal.get(self._literal('_saved_neighbourhood_floor_admitted', 93)) or ()) == set(self._export('T3_BLIND_RUNS'))) and (policy is not self._literal('_saved_neighbourhood_floor_admitted', 94)) and (policy.get(self._literal('_saved_neighbourhood_floor_admitted', 95)) == self._export('T3_PURPOSE')) and (tuple(policy[self._literal('_saved_neighbourhood_floor_admitted', 96)]) == self._export('T3_BLIND_RUNS')) and (policy_plan.get(self._literal('_saved_neighbourhood_floor_admitted', 97)) == plan_sha) and resolve_review_path(policy_plan.get(self._literal('_saved_neighbourhood_floor_admitted', 98), self._literal('_saved_neighbourhood_floor_admitted', 99))).resolve().samefile(plan_path) and (policy.get(self._literal('_saved_neighbourhood_floor_admitted', 100)) is self._literal('_saved_neighbourhood_floor_admitted', 101)) and (policy.get(self._literal('_saved_neighbourhood_floor_admitted', 102)) is self._literal('_saved_neighbourhood_floor_admitted', 103)), self._literal('_saved_neighbourhood_floor_admitted', 104))
        return dict(registration=registration, pipeline_sha256=plan[self._literal('_saved_neighbourhood_floor_admitted', 105)], model_identity=identity, rule_settings=settings)

    def _saved_neighbourhood_floor_validate_head_revision(self, parent, pixel, regional):
        root = Path(parent)
        if sha256_file(root / self._literal('_saved_neighbourhood_floor_validate_head_revision', 0)) != self._export('CO2_SHA256') or sha256_file(pixel) != self._export('G3_SHA256') or sha256_file(regional) != self._export('REFIT_SHA256'):
            raise ValueError(self._literal('_saved_neighbourhood_floor_validate_head_revision', 1))

    def _saved_neighbourhood_floor_validate_runtime_heads(self, candidate, marker, heads, *, verified_plan=_SAVED_MISSING):
        """Dedicated opt-in; original rig profiles keep their original exact hashes."""
        if verified_plan is _SAVED_MISSING:
            verified_plan = self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 0)
        root = Path(candidate)
        if marker.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 1)) != self._export('CONTRACT') or marker.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 2)) != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 3))
        plan_path = bound_path(marker[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 4)], base=root)
        if verified_plan is self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 5):
            plan = read_json(plan_path)
        else:
            if plan_path != (root / self._export('PLAN')).resolve():
                raise ValueError(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 6))
            plan = verified_plan
        if plan.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 7)) != self._export('CONTRACT') or plan.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 8)) != str(root.resolve()) or plan.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 9)) != list(configured_model_identity(root)) or (plan.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 10)) != canonical_sha256(plan[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 11)])) or (plan.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 12)) != canonical_sha256(dict(binding=plan[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 13)], model_identity=plan[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 14)]))):
            raise ValueError(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 15))
        settings = self.plan_rule_settings(plan)
        if settings[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 16)] != marker[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 17)] or marker.get(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 18), self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 20) if marker[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 19)] == self._export('SPEC_SHA256') else self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 21)) != settings[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 22)]:
            raise ValueError(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 23))
        if marker[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 24)] != heads or plan[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 25)][self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 26)] != heads:
            raise ValueError(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 27))
        for key, relative in ((self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 28), self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 29)), (self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 30), self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 31)), (self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 32), self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 33))):
            if bound_path(heads[key], base=root) != (root / relative).resolve():
                raise ValueError(self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 34))
        self.validate_head_revision(plan[self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 35)][self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 36)], root / self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 37), root / self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 38))
        return self._literal('_saved_neighbourhood_floor_validate_runtime_heads', 39)

    def _saved_neighbourhood_floor_plan_rule_settings(self, plan):
        """Read the approved rule variant from its immutable specification binding."""
        binding = plan[self._literal('_saved_neighbourhood_floor_plan_rule_settings', 0)]
        settings = self.approved_spec_settings(binding[self._literal('_saved_neighbourhood_floor_plan_rule_settings', 1)][self._literal('_saved_neighbourhood_floor_plan_rule_settings', 2)])
        inputs = binding[self._literal('_saved_neighbourhood_floor_plan_rule_settings', 3)]
        threshold = inputs[self._literal('_saved_neighbourhood_floor_plan_rule_settings', 4)][self._literal('_saved_neighbourhood_floor_plan_rule_settings', 5)]
        if type(threshold) not in (int, float) or threshold != settings[self._literal('_saved_neighbourhood_floor_plan_rule_settings', 6)] or inputs[self._literal('_saved_neighbourhood_floor_plan_rule_settings', 7)][self._literal('_saved_neighbourhood_floor_plan_rule_settings', 8)] != settings[self._literal('_saved_neighbourhood_floor_plan_rule_settings', 9)]:
            raise ValueError(self._literal('_saved_neighbourhood_floor_plan_rule_settings', 10))
        return settings

    def _saved_neighbourhood_floor_approved_spec_settings(self, spec_sha256):
        """Derive the fixed settings only from one of the two approved source SHAs."""
        if not isinstance(spec_sha256, str):
            raise ValueError(self._literal('_saved_neighbourhood_floor_approved_spec_settings', 0))
        if spec_sha256 == self._export('SPEC_SHA256'):
            return self.approved_settings(self._literal('_saved_neighbourhood_floor_approved_spec_settings', 1))
        if spec_sha256 == self._export('F05_SPEC_SHA256'):
            return self.approved_settings(self._literal('_saved_neighbourhood_floor_approved_spec_settings', 2))
        raise ValueError(self._literal('_saved_neighbourhood_floor_approved_spec_settings', 3))

    def _saved_neighbourhood_floor_approved_settings(self, gas_fraction_threshold=_SAVED_MISSING):
        """Bind either approved gas fraction to its exact specification, never tune it."""
        if gas_fraction_threshold is _SAVED_MISSING:
            gas_fraction_threshold = self._export('GAS_FRACTION_THRESHOLD')
        if isinstance(gas_fraction_threshold, (bool, np.bool_)) or not isinstance(gas_fraction_threshold, Real):
            raise ValueError(self._literal('_saved_neighbourhood_floor_approved_settings', 0))
        threshold = float(gas_fraction_threshold)
        if threshold == self._literal('_saved_neighbourhood_floor_approved_settings', 1):
            spec_sha256 = self._export('SPEC_SHA256')
        elif threshold == self._literal('_saved_neighbourhood_floor_approved_settings', 2):
            spec_sha256 = self._export('F05_SPEC_SHA256')
        else:
            raise ValueError(self._literal('_saved_neighbourhood_floor_approved_settings', 3))
        return dict(gas_fraction_threshold=threshold, spec_sha256=spec_sha256)

    def _saved_legacy_floor_validate_head_revision(self, parent, pixel, regional):
        parent = Path(parent)
        if sha256_file(parent / self._literal('_saved_legacy_floor_validate_head_revision', 0)) != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_legacy_floor_validate_head_revision', 1))
        old, new = (read_json(parent / self._literal('_saved_legacy_floor_validate_head_revision', 2)), read_json(regional))
        for value in (old, new):
            value = value[self._literal('_saved_legacy_floor_validate_head_revision', 3)].pop(self._literal('_saved_legacy_floor_validate_head_revision', 4))
        if old != new:
            raise ValueError(self._literal('_saved_legacy_floor_validate_head_revision', 5))
        if not read_json(pixel):
            raise ValueError(self._literal('_saved_legacy_floor_validate_head_revision', 6))

    def _saved_legacy_floor_validate_runtime_heads(self, candidate, marker, heads):
        """Dedicated opt-in; original rig profiles keep their original exact hashes."""
        root = Path(candidate)
        if marker.get(self._literal('_saved_legacy_floor_validate_runtime_heads', 0)) != self._export('CONTRACT') or marker.get(self._literal('_saved_legacy_floor_validate_runtime_heads', 1)) != self._export('CO2_SHA256'):
            raise ValueError(self._literal('_saved_legacy_floor_validate_runtime_heads', 2))
        plan = read_json(bound_path(marker[self._literal('_saved_legacy_floor_validate_runtime_heads', 3)], base=root))
        if plan.get(self._literal('_saved_legacy_floor_validate_runtime_heads', 4)) != self._export('CONTRACT') or plan.get(self._literal('_saved_legacy_floor_validate_runtime_heads', 5)) != str(root.resolve()) or plan.get(self._literal('_saved_legacy_floor_validate_runtime_heads', 6)) != list(configured_model_identity(root)) or (plan.get(self._literal('_saved_legacy_floor_validate_runtime_heads', 7)) != canonical_sha256(plan[self._literal('_saved_legacy_floor_validate_runtime_heads', 8)])) or (plan.get(self._literal('_saved_legacy_floor_validate_runtime_heads', 9)) != canonical_sha256(dict(binding=plan[self._literal('_saved_legacy_floor_validate_runtime_heads', 10)], model_identity=plan[self._literal('_saved_legacy_floor_validate_runtime_heads', 11)]))):
            raise ValueError(self._literal('_saved_legacy_floor_validate_runtime_heads', 12))
        if plan[self._literal('_saved_legacy_floor_validate_runtime_heads', 13)][self._literal('_saved_legacy_floor_validate_runtime_heads', 14)][self._literal('_saved_legacy_floor_validate_runtime_heads', 15)] != marker[self._literal('_saved_legacy_floor_validate_runtime_heads', 16)]:
            raise ValueError(self._literal('_saved_legacy_floor_validate_runtime_heads', 17))
        if marker[self._literal('_saved_legacy_floor_validate_runtime_heads', 18)] != heads or plan[self._literal('_saved_legacy_floor_validate_runtime_heads', 19)][self._literal('_saved_legacy_floor_validate_runtime_heads', 20)] != heads:
            raise ValueError(self._literal('_saved_legacy_floor_validate_runtime_heads', 21))
        for key, relative in ((self._literal('_saved_legacy_floor_validate_runtime_heads', 22), self._literal('_saved_legacy_floor_validate_runtime_heads', 23)), (self._literal('_saved_legacy_floor_validate_runtime_heads', 24), self._literal('_saved_legacy_floor_validate_runtime_heads', 25)), (self._literal('_saved_legacy_floor_validate_runtime_heads', 26), self._literal('_saved_legacy_floor_validate_runtime_heads', 27))):
            if bound_path(heads[key], base=root) != (root / relative).resolve():
                raise ValueError(self._literal('_saved_legacy_floor_validate_runtime_heads', 28))
        self.validate_head_revision(plan[self._literal('_saved_legacy_floor_validate_runtime_heads', 29)][self._literal('_saved_legacy_floor_validate_runtime_heads', 30)], root / self._literal('_saved_legacy_floor_validate_runtime_heads', 31), root / self._literal('_saved_legacy_floor_validate_runtime_heads', 32))
        return self._literal('_saved_legacy_floor_validate_runtime_heads', 33)

class _SavedCombinedBindings(_SavedRevisionGuardBase):
    _read = _SavedRevisionGuardBase._saved_combined_floor_read
    _samefile = _SavedRevisionGuardBase._saved_combined_floor_samefile
    _require = _SavedRevisionGuardBase._saved_combined_floor_require
    _admitted = _SavedRevisionGuardBase._saved_combined_floor_admitted
    saved_review = _SavedRevisionGuardBase._saved_combined_floor_saved_review
    _verify_heads = _SavedRevisionGuardBase._saved_combined_floor_verify_heads
    validate_runtime_heads = _SavedRevisionGuardBase._saved_combined_floor_validate_runtime_heads
    registered_warning = _SavedRevisionGuardBase._saved_combined_floor_registered_warning

class _SavedTrainedBindings(_SavedRevisionGuardBase):
    _read = _SavedRevisionGuardBase._saved_combined_floor_read
    _require = _SavedRevisionGuardBase._saved_combined_floor_require
    _admitted = _SavedRevisionGuardBase._saved_trained_floor_admitted
    saved_review = _SavedRevisionGuardBase._saved_trained_floor_saved_review
    validate_head_revision = _SavedRevisionGuardBase._saved_trained_floor_validate_head_revision
    validate_runtime_heads = _SavedRevisionGuardBase._saved_trained_floor_validate_runtime_heads
    validate_source_inventory = _SavedRevisionGuardBase._saved_trained_floor_validate_source_inventory
    registered_warning = _SavedRevisionGuardBase._saved_trained_floor_registered_warning
    verify_pixel_model = _SavedRevisionGuardBase._saved_trained_floor_verify_pixel_model
    _legacy_verify_pixel_model = _SavedRevisionGuardBase._saved_trained_floor_legacy_verify_pixel_model

class _SavedEpochBindings(_SavedRevisionGuardBase):
    _read = _SavedRevisionGuardBase._saved_combined_floor_read
    _require = _SavedRevisionGuardBase._saved_combined_floor_require
    _signature = _SavedRevisionGuardBase._saved_epoch_floor_signature
    _plan_dependencies = _SavedRevisionGuardBase._saved_epoch_floor_plan_dependencies
    _admitted = _SavedRevisionGuardBase._saved_epoch_floor_admitted
    saved_review = _SavedRevisionGuardBase._saved_epoch_floor_saved_review
    validate_head_revision = _SavedRevisionGuardBase._saved_epoch_floor_validate_head_revision
    validate_runtime_heads = _SavedRevisionGuardBase._saved_epoch_floor_validate_runtime_heads
    validate_source_inventory = _SavedRevisionGuardBase._saved_trained_floor_validate_source_inventory
    registered_warning = _SavedRevisionGuardBase._saved_trained_floor_registered_warning
    verify_pixel_model = _SavedRevisionGuardBase._saved_epoch_floor_verify_pixel_model
    load_light_epochs = _SavedRevisionGuardBase._saved_epoch_floor_load_light_epochs
    read = _SavedRevisionGuardBase._saved_epoch_floor_read

class _SavedJointBindings(_SavedRevisionGuardBase):
    _read = _SavedRevisionGuardBase._saved_combined_floor_read
    _require = _SavedRevisionGuardBase._saved_combined_floor_require
    _signature = _SavedRevisionGuardBase._saved_epoch_floor_signature
    _plan_dependencies = _SavedRevisionGuardBase._saved_joint_floor_plan_dependencies
    _admitted = _SavedRevisionGuardBase._saved_epoch_floor_admitted
    saved_review = _SavedRevisionGuardBase._saved_epoch_floor_saved_review
    validate_head_revision = _SavedRevisionGuardBase._saved_joint_floor_validate_head_revision
    validate_runtime_heads = _SavedRevisionGuardBase._saved_joint_floor_validate_runtime_heads
    validate_source_inventory = _SavedRevisionGuardBase._saved_trained_floor_validate_source_inventory
    registered_warning = _SavedRevisionGuardBase._saved_trained_floor_registered_warning
    verify_pixel_model = _SavedRevisionGuardBase._saved_joint_floor_verify_pixel_model
    load_light_epochs = _SavedRevisionGuardBase._saved_joint_floor_load_light_epochs
    _base_load_light_epochs = _SavedRevisionGuardBase._saved_epoch_floor_load_light_epochs
    read = _SavedRevisionGuardBase._saved_epoch_floor_read
    checked = _SavedRevisionGuardBase._saved_joint_floor_checked
    verify_regional_model = _SavedRevisionGuardBase._saved_joint_floor_verify_regional_model
    _base_verify_regional_model = _SavedRevisionGuardBase._saved_joint_floor_base_verify_regional_model
    _validate_refit = _SavedRevisionGuardBase._saved_joint_floor_validate_refit

class _SavedNeighbourhoodBindings(_SavedRevisionGuardBase):
    _read = _SavedRevisionGuardBase._saved_combined_floor_read
    _require = _SavedRevisionGuardBase._saved_combined_floor_require
    _admitted = _SavedRevisionGuardBase._saved_neighbourhood_floor_admitted
    saved_review = _SavedRevisionGuardBase._saved_trained_floor_saved_review
    validate_head_revision = _SavedRevisionGuardBase._saved_neighbourhood_floor_validate_head_revision
    validate_runtime_heads = _SavedRevisionGuardBase._saved_neighbourhood_floor_validate_runtime_heads
    validate_source_inventory = _SavedRevisionGuardBase._saved_trained_floor_validate_source_inventory
    registered_warning = _SavedRevisionGuardBase._saved_trained_floor_registered_warning
    plan_rule_settings = _SavedRevisionGuardBase._saved_neighbourhood_floor_plan_rule_settings
    approved_spec_settings = _SavedRevisionGuardBase._saved_neighbourhood_floor_approved_spec_settings
    approved_settings = _SavedRevisionGuardBase._saved_neighbourhood_floor_approved_settings

class _SavedLegacyFloorBindings(_SavedRevisionGuardBase):
    validate_head_revision = _SavedRevisionGuardBase._saved_legacy_floor_validate_head_revision
    validate_runtime_heads = _SavedRevisionGuardBase._saved_legacy_floor_validate_runtime_heads
    registered_warning = _SavedRevisionGuardBase._saved_combined_floor_registered_warning
_SAVED_REVISION_INSTANCES = {}

def saved_revision_bindings(section):
    """Return only the configured read/hash guards for a historical saved view."""
    parameters = configured_parameters(section)
    classes = {'combined_floor': _SavedCombinedBindings, 'trained_floor': _SavedTrainedBindings, 'epoch_floor': _SavedEpochBindings, 'joint_floor': _SavedJointBindings, 'neighbourhood_floor': _SavedNeighbourhoodBindings, 'legacy_floor': _SavedLegacyFloorBindings}
    if section not in classes or parameters['class_key'] != section:
        raise ValueError('Unknown saved review metadata section')
    identity = (section, canonical_sha256(parameters))
    if identity not in _SAVED_REVISION_INSTANCES:
        _SAVED_REVISION_INSTANCES[identity] = classes[section](parameters)
    return _SAVED_REVISION_INSTANCES[identity]


# Saved-profile DTOs/admission only; no numerical sequence is exported.
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace as _SavedProfileNamespace
import cv2


def _saved_profile_typed(item):
    value = item["value"]
    if item["python_type"] == "tuple":
        return tuple(value)
    if item["python_type"] == "bytes":
        return bytes.fromhex(value)
    if item["python_type"] == "set":
        return set(value)
    if type(value).__name__ != item["python_type"]:
        raise TypeError("Saved-profile DATA changed its original Python type")
    return deepcopy(value)


def _producer_sha256(role):
    """Read the exact historical metadata-producer digest from source-bound DATA.

    Runtime payloads retain their original producer identity; the actual current
    implementation is independently checked by saved_physics_profile. Historical
    source code is neither required on disk nor imported or executed.
    """
    entry = configured_parameters("physics_profiles")["roles"][role]["producer_identity"]
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        raise ValueError("Saved-profile producer identity is not explicitly bound")
    digest = entry["sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Saved-profile producer identity is not a SHA256")
    return digest


_SAVED_PROFILE_CACHE = {}


def saved_physics_profile(role):
    """Return explicit-DATA saved-profile validators and DTOs, never inference.

    Passive producer hashes remain saved-result inputs. The actual current
    helper is admitted separately; no module __file__ value is substituted.
    """
    parameters = configured_parameters("physics_profiles")
    current = parameters["current_code_source"]
    if not isinstance(current, dict) or Path(current["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("Saved-profile current implementation is not explicitly bound")
    if sha256_file(Path(__file__)) != current["sha256"] or Path(__file__).stat().st_size != current["size_bytes"]:
        raise ValueError("Saved-profile current implementation changed")
    if role not in _SAVED_PROFILE_FACTORIES:
        raise ValueError("Unknown saved-profile metadata role")
    settings = parameters["roles"][role]
    key = (role, canonical_sha256(settings), current["sha256"])
    if key not in _SAVED_PROFILE_CACHE:
        _SAVED_PROFILE_CACHE[key] = _SAVED_PROFILE_FACTORIES[role](settings)
    return _SAVED_PROFILE_CACHE[key]


def _saved_profile_decision_dispatch(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    PROFILE_FILE = saved_physics_profile('decision').PROFILE_FILE
    PROFILE_KEY = saved_physics_profile('decision').PROFILE_KEY

    def implementation(spec):
        if spec.get(_literal('implementation:12:16')) == saved_physics_profile('decision').VERSION:
            return saved_physics_profile('decision')
        age = saved_physics_profile('decision_age')
        if spec.get(_literal('implementation:14:16')) == saved_physics_profile('decision_age').VERSION:
            return saved_physics_profile('decision_age')
        regional = saved_physics_profile('regional_decision')
        if spec.get(_literal('implementation:16:16')) == saved_physics_profile('regional_decision').VERSION:
            return saved_physics_profile('regional_decision')
        raise ValueError(_literal('implementation:17:21'))

    def attach_profile(root, inherited):
        path = Path(root) / PROFILE_FILE
        if not path.is_file():
            return inherited
        return implementation(json.loads(path.read_text(encoding=_literal('attach_profile:23:61')))).attach_profile(root, inherited)

    def runtime_identity(inherited_identity, spec):
        return implementation(spec).runtime_identity(inherited_identity, spec)
    return _SavedProfileNamespace(PROFILE_FILE=PROFILE_FILE, PROFILE_KEY=PROFILE_KEY, attach_profile=attach_profile, implementation=implementation, runtime_identity=runtime_identity)

def _saved_profile_completion(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_FILE = _export('PROFILE_FILE')
    PROFILE_KEY = _export('PROFILE_KEY')
    WORKING_SPACING_M = _export('WORKING_SPACING_M')
    PATCH = _export('PATCH')
    STRIDE = _export('STRIDE')

    def _sha(path):
        digest = hashlib.sha256()
        with Path(path).open(_literal('_sha:26:25')) as stream:
            for block in iter(lambda: stream.read(_literal('_sha:27:46') * _literal('_sha:27:53')), _literal('_sha:27:60')):
                digest.update(block)
        return digest.hexdigest()

    def attach_profile(root, inherited):
        """Load a portable sidecar without modifying the inherited profile file."""
        root = Path(root).resolve()
        path = root / PROFILE_FILE
        if not path.is_file():
            return inherited
        if inherited is _literal('attach_profile:38:20') or PROFILE_KEY in inherited:
            raise ValueError(_literal('attach_profile:39:25'))
        specification = json.loads(path.read_text(encoding=_literal('attach_profile:40:55')))
        if set(specification) != {_literal('attach_profile:41:30'), _literal('attach_profile:41:40'), _literal('attach_profile:41:54')}:
            raise ValueError(_literal('attach_profile:42:25'))
        if specification[_literal('attach_profile:43:21')] != VERSION:
            raise ValueError(_literal('attach_profile:44:25'))
        checkpoint = (root / specification[_literal('attach_profile:45:39')]).resolve()
        if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
            raise ValueError(_literal('attach_profile:47:25'))
        if _sha(checkpoint) != specification[_literal('attach_profile:48:41')]:
            raise ValueError(_literal('attach_profile:49:25'))
        return {**inherited, PROFILE_KEY: {**specification, _literal('attach_profile:50:56'): str(checkpoint)}}

    def split_profile(profile):
        specification = profile[PROFILE_KEY]
        if specification.get(_literal('split_profile:55:25')) != VERSION:
            raise ValueError(_literal('split_profile:56:25'))
        return ({k: v for k, v in profile.items() if k != PROFILE_KEY}, specification)

    def runtime_identity(inherited_identity, specification):
        public = {k: v for k, v in specification.items() if not k.startswith(_literal('runtime_identity:62:73'))}
        payload = {_literal('runtime_identity:63:15'): inherited_identity, _literal('runtime_identity:63:56'): public, _literal('runtime_identity:64:15'): _producer_sha256('completion')}
        return hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:65:56')).encode()).hexdigest()
    return _SavedProfileNamespace(PATCH=PATCH, PROFILE_FILE=PROFILE_FILE, PROFILE_KEY=PROFILE_KEY, STRIDE=STRIDE, VERSION=VERSION, WORKING_SPACING_M=WORKING_SPACING_M, _sha=_sha, attach_profile=attach_profile, runtime_identity=runtime_identity, split_profile=split_profile)

def _saved_profile_local_support(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_FILE = _export('PROFILE_FILE')
    PROFILE_KEY = _export('PROFILE_KEY')

    @dataclass(frozen=True)
    class Settings:
        working_spacing_m: float = _literal('Settings:26:31')
        smoothing_m: float = _literal('Settings:27:25')
        search_radius_m: float = _literal('Settings:28:29')
        minimum_chromatic_change: float = _literal('Settings:29:38')
        noise_multiplier: float = _literal('Settings:30:30')
        minimum_direction_cosine: float = _literal('Settings:31:38')
        minimum_support_area_m2: float = _literal('Settings:32:37')
        minimum_controls: int = _literal('Settings:33:28')
        minimum_seeds: int = _literal('Settings:34:25')
        minimum_model_aqueous_probability: float = _literal('Settings:35:47')
        use_model_negative_veto: bool = _literal('Settings:36:36')
        continuity_radius_m: float = _literal('Settings:37:33')
        contour_smoothing_m: float = _literal('Settings:38:33')

        def __post_init__(self):
            for key, value in asdict(self).items():
                if key == _literal('__post_init__:42:22'):
                    if not isinstance(value, bool):
                        raise ValueError(_literal('__post_init__:43:65'))
                    continue
                if key in {_literal('__post_init__:45:23'), _literal('__post_init__:45:46')} and value == _literal('__post_init__:45:82'):
                    continue
                if not np.isfinite(value) or value <= _literal('__post_init__:47:50'):
                    raise ValueError(f'Positive finite local-support setting required: {key}')
            if self.minimum_direction_cosine > _literal('__post_init__:49:43'):
                raise ValueError(_literal('__post_init__:50:29'))
            if self.minimum_model_aqueous_probability > _literal('__post_init__:51:52'):
                raise ValueError(_literal('__post_init__:52:29'))

    def attach_profile(root, inherited):
        path = Path(root) / PROFILE_FILE
        if not path.exists():
            return inherited
        spec = json.loads(path.read_text(encoding=_literal('attach_profile:59:46')))
        if inherited is _literal('attach_profile:60:20') or set(spec) != {_literal('attach_profile:60:42'), _literal('attach_profile:60:52')} or spec[_literal('attach_profile:60:72')] != VERSION:
            raise ValueError(_literal('attach_profile:61:25'))
        Settings(**spec[_literal('attach_profile:62:20')])
        return {**inherited, PROFILE_KEY: spec}

    def split_profile(profile):
        return ({k: v for k, v in profile.items() if k != PROFILE_KEY}, profile[PROFILE_KEY])

    def runtime_identity(inherited, specification):
        source = _producer_sha256('local_support')
        return hashlib.sha256(json.dumps({_literal('runtime_identity:72:38'): inherited, _literal('runtime_identity:72:62'): specification, _literal('runtime_identity:73:37'): source}, sort_keys=_literal('runtime_identity:73:73')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_FILE=PROFILE_FILE, PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, attach_profile=attach_profile, runtime_identity=runtime_identity, split_profile=split_profile)

def _saved_profile_temporal_support(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    @dataclass(frozen=True)
    class Settings:
        format: str = VERSION
        working_spacing_m: float = _literal('Settings:33:31')
        weak_optical_probability: float = _literal('Settings:34:38')
        independent_seed_probability: float = _literal('Settings:35:42')
        weak_chromatic_change: float = _literal('Settings:36:35')
        strong_chromatic_change: float = _literal('Settings:37:37')
        independent_seed_change: float = _literal('Settings:38:37')
        history_distance_mm: float = _literal('Settings:39:33')
        source_distance_mm: float = _literal('Settings:40:32')
        maximum_history_gap_h: float = _literal('Settings:41:35')
        minimum_suspicion: float = _literal('Settings:42:31')
        maximum_score_penalty: float = _literal('Settings:43:35')
        maximum_withdrawn_fraction: float = _literal('Settings:44:40')

        def __post_init__(self):
            if self.format != VERSION:
                raise ValueError(_literal('__post_init__:48:29'))
            numeric = {key: float(value) for key, value in asdict(self).items() if key != _literal('__post_init__:49:86')}
            if any((not np.isfinite(value) or value <= _literal('__post_init__:50:50') for value in numeric.values())):
                raise ValueError(_literal('__post_init__:51:29'))
            if not (_literal('__post_init__:52:16') < self.working_spacing_m <= _literal('__post_init__:52:46') and _literal('__post_init__:53:20') < self.independent_seed_probability < self.weak_optical_probability < _literal('__post_init__:53:92') and (_literal('__post_init__:54:20') < self.weak_chromatic_change < self.independent_seed_change < self.strong_chromatic_change < _literal('__post_init__:54:115')) and (_literal('__post_init__:55:20') < self.minimum_suspicion < _literal('__post_init__:55:49')) and (_literal('__post_init__:56:20') < self.maximum_score_penalty <= _literal('__post_init__:56:54')) and (_literal('__post_init__:57:20') < self.maximum_withdrawn_fraction <= _literal('__post_init__:57:59')) and (self.history_distance_mm <= _literal('__post_init__:58:48')) and (self.source_distance_mm <= _literal('__post_init__:59:47')) and (self.maximum_history_gap_h <= _literal('__post_init__:60:50'))):
                raise ValueError(_literal('__post_init__:61:29'))

    def runtime_identity(inherited: str, specification: dict) -> str:
        settings = Settings(**specification)
        payload = dict(inherited=inherited, settings=asdict(settings), code_sha256=_producer_sha256('temporal_support'), opencv=cv2.__version__)
        return hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:68:56')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, runtime_identity=runtime_identity)

def _saved_profile_source_history(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    @dataclass(frozen=True)
    class Settings:
        format: str = VERSION
        working_spacing_m: float = _literal('Settings:35:31')
        independent_seed_probability: float = _literal('Settings:36:42')
        independent_seed_change: float = _literal('Settings:37:37')
        source_anchor_radius_mm: float = _literal('Settings:38:37')
        growth_slack_mm: float = _literal('Settings:39:29')
        growth_mm_per_sqrt_h: float = _literal('Settings:40:34')
        history_retention_h: float = _literal('Settings:41:33')
        maximum_observation_gap_h: float = _literal('Settings:42:39')
        history_distance_mm: float = _literal('Settings:43:33')
        source_distance_mm: float = _literal('Settings:44:32')
        current_optical_protection_probability: float = _literal('Settings:45:52')
        current_chromatic_protection: float = _literal('Settings:46:42')
        optical_weakness_weight: float = _literal('Settings:47:37')
        minimum_suspicion: float = _literal('Settings:48:31')
        maximum_score_penalty: float = _literal('Settings:49:35')
        maximum_withdrawn_fraction: float = _literal('Settings:50:40')

        def __post_init__(self):
            if self.format != VERSION:
                raise ValueError(_literal('__post_init__:54:29'))
            numeric = [float(value) for key, value in asdict(self).items() if key != _literal('__post_init__:55:81')]
            if any((not np.isfinite(value) or value <= _literal('__post_init__:56:50') for value in numeric)):
                raise ValueError(_literal('__post_init__:57:29'))
            if not (self.working_spacing_m <= _literal('__post_init__:59:38') and _literal('__post_init__:60:16') <= self.independent_seed_probability < self.current_optical_protection_probability < _literal('__post_init__:60:104') and (_literal('__post_init__:61:16') < self.independent_seed_change < self.current_chromatic_protection < _literal('__post_init__:61:87')) and (self.source_anchor_radius_mm <= _literal('__post_init__:62:48')) and (self.growth_slack_mm <= _literal('__post_init__:63:40')) and (self.growth_mm_per_sqrt_h <= _literal('__post_init__:64:45')) and (self.history_retention_h <= _literal('__post_init__:65:44')) and (self.maximum_observation_gap_h <= _literal('__post_init__:66:50')) and (self.history_distance_mm <= _literal('__post_init__:67:44')) and (self.source_distance_mm <= _literal('__post_init__:68:43')) and (_literal('__post_init__:69:16') < self.optical_weakness_weight < _literal('__post_init__:69:51')) and (_literal('__post_init__:70:16') < self.minimum_suspicion < _literal('__post_init__:70:45')) and (self.maximum_score_penalty <= _literal('__post_init__:71:46')) and (self.maximum_withdrawn_fraction <= _literal('__post_init__:72:51'))):
                raise ValueError(_literal('__post_init__:74:29'))

    def runtime_identity(inherited: str, specification: dict) -> str:
        settings = Settings(**specification)
        payload = dict(inherited=inherited, settings=asdict(settings), code_sha256=_producer_sha256('source_history'), shared_geometry_code_sha256=_producer_sha256('temporal_support'), opencv=cv2.__version__, numpy=np.__version__)
        return hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:85:56')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, runtime_identity=runtime_identity)

def _saved_profile_boundary(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    from . import phase_presence_context as presence
    from .project import sha256_file

    def split_profile(profile):
        """Validate the explicit revision and return a non-mutating parent view."""
        from . import phase_reference_support, phase_reviewed_reference
        from . import phase_facies_reference, phase_native_reference
        supported = {presence.VERSION, phase_reference_support.VERSION, phase_reviewed_reference.VERSION, phase_facies_reference.VERSION, phase_native_reference.VERSION}
        if profile.get(_literal('split_profile:31:19')) not in supported:
            raise ValueError(_literal('split_profile:32:25'))
        specification = profile.get(_literal('split_profile:33:32'))
        if not isinstance(specification, dict) or specification.get(_literal('split_profile:34:64')) != VERSION:
            raise ValueError(_literal('split_profile:35:25'))
        if set(specification) - {_literal('split_profile:36:29'), _literal('split_profile:36:39')}:
            raise ValueError(_literal('split_profile:37:25'))
        values = specification.get(_literal('split_profile:38:31'), {})
        if not isinstance(values, dict):
            raise ValueError(_literal('split_profile:40:25'))
        settings = saved_physics_profile('boundary_core').AqueousBoundarySettings(**values)
        inherited = {key: value for key, value in profile.items() if key != _literal('split_profile:42:72')}
        return (inherited, settings)

    def runtime_identity(inherited_identity, settings):
        """Bind the inherited runtime, canonical settings, and both new modules."""
        payload = {_literal('runtime_identity:48:15'): VERSION, _literal('runtime_identity:48:34'): inherited_identity, _literal('runtime_identity:49:15'): asdict(settings)}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:50:58'), separators=(_literal('runtime_identity:50:76'), _literal('runtime_identity:50:81'))).encode())
        for producer_digest in (_producer_sha256('boundary_core'), _producer_sha256('boundary')):
            digest.update(producer_digest.encode())
        return digest.hexdigest()
    return _SavedProfileNamespace(VERSION=VERSION, runtime_identity=runtime_identity, split_profile=split_profile)

def _saved_profile_boundary_core(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')

    @dataclass(frozen=True)
    class AqueousBoundarySettings:
        sigma_m: float = _literal('AqueousBoundarySettings:23:21')
        maximum_band_m: float = _literal('AqueousBoundarySettings:24:28')
        threshold: float = _literal('AqueousBoundarySettings:25:23')

        def __post_init__(self):
            if not all((np.isfinite(v) for v in asdict(self).values())):
                raise ValueError(_literal('__post_init__:29:29'))
            if not _literal('__post_init__:30:15') < self.sigma_m <= self.maximum_band_m:
                raise ValueError(_literal('__post_init__:31:29'))
            if self.maximum_band_m > _literal('__post_init__:32:33'):
                raise ValueError(_literal('__post_init__:33:29'))
            if not _literal('__post_init__:34:15') < self.threshold < _literal('__post_init__:34:36'):
                raise ValueError(_literal('__post_init__:35:29'))
    return _SavedProfileNamespace(AqueousBoundarySettings=AqueousBoundarySettings, VERSION=VERSION)

def _saved_profile_boundary_extended(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = saved_physics_profile('boundary_extended_core').VERSION
    from .project import sha256_file

    def split_profile(profile):
        specification = profile.get(_literal('split_profile:17:32'))
        if not isinstance(specification, dict) or specification.get(_literal('split_profile:18:64')) != VERSION:
            raise ValueError(_literal('split_profile:19:25'))
        if set(specification) - {_literal('split_profile:20:29'), _literal('split_profile:20:39')} or not isinstance(specification.get(_literal('split_profile:20:87'), {}), dict):
            raise ValueError(_literal('split_profile:21:25'))
        settings = saved_physics_profile('boundary_extended_core').AqueousBoundarySettings(**specification.get(_literal('split_profile:22:68'), {}))
        inherited, _ = saved_physics_profile('boundary').split_profile({**profile, _literal('split_profile:23:52'): dict(format=saved_physics_profile('boundary').VERSION, settings=asdict(saved_physics_profile('boundary_extended_core').legacy_settings(settings)))})
        return (inherited, settings)

    def runtime_identity(inherited_identity, settings):
        payload = dict(format=VERSION, inherited_runtime=inherited_identity, legacy_runtime=saved_physics_profile('boundary').runtime_identity(inherited_identity, saved_physics_profile('boundary_extended_core').legacy_settings(settings)), settings=asdict(settings))
        digest = hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:32:58')).encode())
        for producer_digest in (_producer_sha256('boundary_extended'), _producer_sha256('boundary_extended_core')):
            digest.update(producer_digest.encode())
        return digest.hexdigest()
    return _SavedProfileNamespace(VERSION=VERSION, runtime_identity=runtime_identity, split_profile=split_profile)

def _saved_profile_boundary_extended_core(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')

    @dataclass(frozen=True)
    class AqueousBoundarySettings(saved_physics_profile('boundary_core').AqueousBoundarySettings):
        fringe_sigma_m: float = _literal('AqueousBoundarySettings:23:28')
        refinement_band_m: float = _literal('AqueousBoundarySettings:24:31')
        maximum_island_diameter_m: float = _literal('AqueousBoundarySettings:25:39')
        minimum_finger_length_m: float = _literal('AqueousBoundarySettings:26:37')
        maximum_fringe_occupancy: float = _literal('AqueousBoundarySettings:27:38')
        fringe_optical_threshold: float = _literal('AqueousBoundarySettings:28:38')
        strong_optical_threshold: float = _literal('AqueousBoundarySettings:29:38')
        gap_sigma_m: float = _literal('AqueousBoundarySettings:30:25')
        minimum_gap_occupancy: float = _literal('AqueousBoundarySettings:31:35')
        enable_gap_completion: bool = _literal('AqueousBoundarySettings:32:34')

        def __post_init__(self):
            if type(self.enable_gap_completion) is not bool:
                raise ValueError(_literal('__post_init__:36:29'))
            super().__post_init__()
            if not _literal('__post_init__:38:15') < self.fringe_sigma_m <= self.maximum_island_diameter_m <= self.refinement_band_m <= _literal('__post_init__:38:102'):
                raise ValueError(_literal('__post_init__:39:29'))
            if not _literal('__post_init__:40:15') < self.minimum_finger_length_m <= self.refinement_band_m:
                raise ValueError(_literal('__post_init__:41:29'))
            if not _literal('__post_init__:42:15') < self.maximum_fringe_occupancy < _literal('__post_init__:42:51'):
                raise ValueError(_literal('__post_init__:43:29'))
            if not _literal('__post_init__:44:15') < self.fringe_optical_threshold <= _literal('__post_init__:44:52'):
                raise ValueError(_literal('__post_init__:45:29'))
            if not _literal('__post_init__:46:15') <= self.strong_optical_threshold <= _literal('__post_init__:46:54'):
                raise ValueError(_literal('__post_init__:47:29'))
            if not self.fringe_sigma_m <= self.gap_sigma_m <= self.refinement_band_m:
                raise ValueError(_literal('__post_init__:49:29'))
            if not _literal('__post_init__:50:15') < self.minimum_gap_occupancy <= _literal('__post_init__:50:50'):
                raise ValueError(_literal('__post_init__:51:29'))

    def legacy_settings(settings):
        return saved_physics_profile('boundary_core').AqueousBoundarySettings(**{key: asdict(settings)[key] for key in saved_physics_profile('boundary_core').AqueousBoundarySettings.__dataclass_fields__})
    return _SavedProfileNamespace(AqueousBoundarySettings=AqueousBoundarySettings, VERSION=VERSION, legacy_settings=legacy_settings)

def _saved_profile_boundary_continuity(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    from .project import sha256_file

    @dataclass(frozen=True)
    class ContinuitySettings:
        sigma_m: float = _literal('ContinuitySettings:26:21')
        maximum_recovery_m: float = _literal('ContinuitySettings:27:32')
        cleanup_band_m: float = _literal('ContinuitySettings:28:28')
        history_optical_minimum: float = _literal('ContinuitySettings:29:37')
        connected_optical_minimum: float = _literal('ContinuitySettings:30:39')
        cleanup_optical_maximum: float = _literal('ContinuitySettings:31:37')
        history_signal_minimum: float = _literal('ContinuitySettings:32:36')
        connected_signal_minimum: float = _literal('ContinuitySettings:33:38')

        def __post_init__(self):
            if not all((np.isfinite(v) and v > _literal('__post_init__:36:42') for v in asdict(self).values())):
                raise ValueError(_literal('__post_init__:37:29'))
            if not self.sigma_m <= self.cleanup_band_m <= self.maximum_recovery_m <= _literal('__post_init__:38:81'):
                raise ValueError(_literal('__post_init__:39:29'))
            if not self.history_optical_minimum < self.cleanup_optical_maximum < self.connected_optical_minimum < _literal('__post_init__:40:110'):
                raise ValueError(_literal('__post_init__:41:29'))
            if not _literal('__post_init__:42:15') <= self.history_signal_minimum <= self.connected_signal_minimum:
                raise ValueError(_literal('__post_init__:43:29'))

    def split_profile(profile):
        specification = profile.get(_literal('split_profile:47:32'), {})
        if specification.get(_literal('split_profile:48:25')) != VERSION or set(specification) - {_literal('split_profile:48:71'), _literal('split_profile:48:81')}:
            raise ValueError(_literal('split_profile:49:25'))
        inherited, _ = saved_physics_profile('boundary').split_profile({**profile, _literal('split_profile:51:52'): {_literal('split_profile:52:8'): saved_physics_profile('boundary').VERSION, _literal('split_profile:52:34'): {}}})
        return (inherited, ContinuitySettings(**specification.get(_literal('split_profile:53:61'), {})))

    def runtime_identity(inherited_identity, settings):
        payload = dict(format=VERSION, inherited_runtime=inherited_identity, settings=asdict(settings))
        digest = hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:58:58')).encode())
        for producer_digest in (_producer_sha256('boundary_continuity'), _producer_sha256('boundary_core'), _producer_sha256('boundary')):
            digest.update(producer_digest.encode())
        return digest.hexdigest()
    return _SavedProfileNamespace(ContinuitySettings=ContinuitySettings, VERSION=VERSION, runtime_identity=runtime_identity, split_profile=split_profile)

def _saved_profile_decision(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    PROFILE_FILE = _export('PROFILE_FILE')
    PROFILE_KEY = _export('PROFILE_KEY')
    VERSION = _export('VERSION')

    def attach_profile(root, inherited):
        path = Path(root) / PROFILE_FILE
        if not path.is_file():
            return inherited
        spec = json.loads(path.read_text(encoding=_literal('attach_profile:27:44')))
        if set(spec) != {_literal('attach_profile:28:19'), _literal('attach_profile:28:28')} or spec[_literal('attach_profile:28:48')] != VERSION:
            raise ValueError(_literal('attach_profile:29:25'))
        saved_physics_profile('decision_core').Settings(**spec[_literal('attach_profile:30:25')])
        if inherited is _literal('attach_profile:31:20') or saved_physics_profile('completion').PROFILE_KEY not in inherited:
            raise ValueError(_literal('attach_profile:32:25'))
        if _literal('attach_profile:33:7') in inherited:
            raise ValueError(_literal('attach_profile:34:25'))
        return {**inherited, PROFILE_KEY: spec}

    def split_profile(profile):
        return ({k: v for k, v in profile.items() if k != PROFILE_KEY}, profile[PROFILE_KEY])

    def runtime_identity(inherited_identity, specification):
        return hashlib.sha256(json.dumps(dict(format=VERSION, provider_identity=inherited_identity, specification=specification, decision_identity=saved_physics_profile('decision_core').identity(saved_physics_profile('decision_core').Settings(**specification[_literal('runtime_identity:44:98')])), adapter_sha256=_producer_sha256('decision')), sort_keys=_literal('runtime_identity:45:90')).encode()).hexdigest()
    core = saved_physics_profile('decision_core')
    return _SavedProfileNamespace(PROFILE_FILE=PROFILE_FILE, PROFILE_KEY=PROFILE_KEY, VERSION=VERSION, attach_profile=attach_profile, core=core, runtime_identity=runtime_identity, split_profile=split_profile)

def _saved_profile_decision_core(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    MINIMUM_FREE_GAS_AREA_M2 = _export('MINIMUM_FREE_GAS_AREA_M2')

    @dataclass(frozen=True)
    class Settings:
        working_spacing_m: float = _literal('Settings:25:31')
        reference_smoothing_m: float = _literal('Settings:26:35')
        minimum_chromatic_change: float = _literal('Settings:27:38')
        noise_multiplier: float = _literal('Settings:28:30')
        minimum_direction_cosine: float = _literal('Settings:29:38')
        reference_search_m: float = _literal('Settings:30:32')
        maximum_reference_evidence_weight: float = _literal('Settings:31:47')
        minimum_controls: int = _literal('Settings:32:28')
        minimum_direction_samples: int = _literal('Settings:33:37')
        independent_observation_score: float = _literal('Settings:34:43')
        minimum_observation_area_m2: float = _literal('Settings:35:41')
        contour_smoothing_m: float = _literal('Settings:36:33')
        contour_outward_allowance_m: float = _literal('Settings:37:41')
        contour_band_m: float = _literal('Settings:38:28')
        contour_reference_floor: float = _literal('Settings:39:37')
        secure_core_distance_m: float = _literal('Settings:40:36')

        def __post_init__(self):
            for name, value in asdict(self).items():
                if not np.isfinite(value) or value <= _literal('__post_init__:44:50'):
                    raise ValueError(f'Positive finite decision setting required: {name}')
            if self.minimum_direction_cosine > _literal('__post_init__:46:43'):
                raise ValueError(_literal('__post_init__:47:29'))
            if self.contour_band_m < self.contour_outward_allowance_m:
                raise ValueError(_literal('__post_init__:49:29'))
            for key in (_literal('__post_init__:50:20'), _literal('__post_init__:50:40')):
                if int(getattr(self, key)) != getattr(self, key):
                    raise ValueError(f'Integer sample count required: {key}')

    def identity(settings=Settings()):
        return hashlib.sha256(json.dumps(dict(format=VERSION, settings=asdict(settings), source_sha256=_producer_sha256('decision_core')), sort_keys=_literal('identity:86:90')).encode()).hexdigest()
    return _SavedProfileNamespace(MINIMUM_FREE_GAS_AREA_M2=MINIMUM_FREE_GAS_AREA_M2, Settings=Settings, VERSION=VERSION, identity=identity)

def _saved_profile_decision_age(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    PROFILE_FILE = saved_physics_profile('decision').PROFILE_FILE
    PROFILE_KEY = saved_physics_profile('decision').PROFILE_KEY
    VERSION = _export('VERSION')

    def attach_profile(root, inherited):
        path = Path(root) / PROFILE_FILE
        if not path.is_file():
            return inherited
        spec = json.loads(path.read_text(encoding=_literal('attach_profile:20:44')))
        if set(spec) != {_literal('attach_profile:21:19'), _literal('attach_profile:21:28')} or spec[_literal('attach_profile:21:48')] != VERSION:
            raise ValueError(_literal('attach_profile:22:25'))
        saved_physics_profile('decision_age_core').Settings(**spec[_literal('attach_profile:23:25')])
        if inherited is _literal('attach_profile:24:20') or saved_physics_profile('completion').PROFILE_KEY not in inherited:
            raise ValueError(_literal('attach_profile:25:25'))
        if _literal('attach_profile:26:7') in inherited:
            raise ValueError(_literal('attach_profile:27:25'))
        return {**inherited, PROFILE_KEY: spec}

    def runtime_identity(inherited_identity, specification):
        return hashlib.sha256(json.dumps(dict(format=VERSION, provider_identity=inherited_identity, specification=specification, decision_identity=saved_physics_profile('decision_age_core').identity(saved_physics_profile('decision_age_core').Settings(**specification[_literal('runtime_identity:33:98')])), adapter_sha256=_producer_sha256('decision_age'), shared_adapter_sha256=_producer_sha256('decision')), sort_keys=_literal('runtime_identity:35:104')).encode()).hexdigest()
    core = saved_physics_profile('decision_age_core')
    return _SavedProfileNamespace(PROFILE_FILE=PROFILE_FILE, PROFILE_KEY=PROFILE_KEY, VERSION=VERSION, attach_profile=attach_profile, core=core, runtime_identity=runtime_identity)

def _saved_profile_decision_age_core(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')

    @dataclass(frozen=True)
    class Settings(saved_physics_profile('decision_core').Settings):
        mature_local_age_h: float = _literal('Settings:22:32')
        young_reference_threshold: float = _literal('Settings:23:39')
        young_weak_fringe_m: float = _literal('Settings:24:33')
        strong_reference_threshold: float = _literal('Settings:25:40')
        young_contour_reference_floor: float = _literal('Settings:26:43')
        young_outward_allowance_m: float = _literal('Settings:27:39')

        def __post_init__(self):
            super().__post_init__()
            if self.young_reference_threshold < _literal('__post_init__:31:44') or self.strong_reference_threshold < self.young_reference_threshold:
                raise ValueError(_literal('__post_init__:32:29'))
            if self.young_contour_reference_floor < self.contour_reference_floor:
                raise ValueError(_literal('__post_init__:34:29'))
            if self.young_outward_allowance_m > self.contour_outward_allowance_m:
                raise ValueError(_literal('__post_init__:36:29'))

    def identity(settings=Settings()):
        return hashlib.sha256(json.dumps(dict(format=VERSION, settings=asdict(settings), source_sha256=_producer_sha256('decision_age_core'), common_source_sha256=_producer_sha256('decision_core')), sort_keys=_literal('identity:73:101')).encode()).hexdigest()
    return _SavedProfileNamespace(Settings=Settings, VERSION=VERSION, identity=identity)

def _saved_profile_regional_decision(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_FILE = saved_physics_profile('decision').PROFILE_FILE
    PROFILE_KEY = saved_physics_profile('decision').PROFILE_KEY

    @dataclass(frozen=True)
    class RegionalSettings:
        spacing_m: float = _literal('RegionalSettings:34:23')

        def __post_init__(self):
            if not np.isfinite(self.spacing_m) or self.spacing_m <= _literal('__post_init__:37:64'):
                raise ValueError(_literal('__post_init__:38:29'))

    def validate(spec):
        if set(spec) != {_literal('validate:42:21'), _literal('validate:42:31'), _literal('validate:42:43')} or spec[_literal('validate:42:63')] != VERSION:
            raise ValueError(_literal('validate:43:25'))
        RegionalSettings(**spec[_literal('validate:44:28')])
        age = saved_physics_profile('decision_age')
        if spec[_literal('validate:47:12')].get(_literal('validate:47:28')) not in (saved_physics_profile('decision').VERSION, saved_physics_profile('decision_age').VERSION):
            raise ValueError(_literal('validate:48:25'))
        impl = saved_physics_profile('decision_dispatch').implementation(spec[_literal('validate:49:41')])
        if set(spec[_literal('validate:50:16')]) != {_literal('validate:50:33'), _literal('validate:50:43')}:
            raise ValueError(_literal('validate:51:25'))
        impl.core.Settings(**spec[_literal('validate:52:30')][_literal('validate:52:42')])
        return spec

    def attach_profile(root, inherited):
        path = Path(root) / PROFILE_FILE
        if not path.is_file():
            return inherited
        spec = validate(json.loads(path.read_text(encoding=_literal('attach_profile:70:55'))))
        if inherited is _literal('attach_profile:71:20') or saved_physics_profile('completion').PROFILE_KEY not in inherited:
            raise ValueError(_literal('attach_profile:72:25'))
        if _literal('attach_profile:73:7') in inherited:
            raise ValueError(_literal('attach_profile:74:25'))
        return {**inherited, PROFILE_KEY: spec}

    def runtime_identity(inherited_identity, spec):
        validate(spec)
        payload = dict(format=VERSION, specification=spec, parent_identity=inherited_identity, final_identity=saved_physics_profile('decision_dispatch').runtime_identity(inherited_identity, spec[_literal('runtime_identity:82:75')]), adapter_sha256=_producer_sha256('regional_decision'))
        return hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:84:56')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_FILE=PROFILE_FILE, PROFILE_KEY=PROFILE_KEY, RegionalSettings=RegionalSettings, VERSION=VERSION, attach_profile=attach_profile, runtime_identity=runtime_identity, validate=validate)

def _saved_profile_free_gas(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    @dataclass(frozen=True)
    class Settings:
        format: str = VERSION
        minimum_area_mm2: float = _literal('Settings:20:30')

        def __post_init__(self):
            if self.format != VERSION or not np.isfinite(self.minimum_area_mm2) or (not _literal('__post_init__:24:23') < self.minimum_area_mm2 <= _literal('__post_init__:24:52')):
                raise ValueError(_literal('__post_init__:25:29'))

    def runtime_identity(inherited, specification):
        return hashlib.sha256(json.dumps(dict(inherited=inherited, settings=asdict(Settings(**specification)), code_sha256=_producer_sha256('free_gas'), opencv=cv2.__version__), sort_keys=_literal('runtime_identity:31:43')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, runtime_identity=runtime_identity)

def _saved_profile_region_guard(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    @dataclass(frozen=True)
    class Settings:
        format: str = VERSION
        minimum_island_area_mm2: float = _literal('Settings:25:37')

        def __post_init__(self):
            if self.format != VERSION or not np.isfinite(self.minimum_island_area_mm2) or (not _literal('__post_init__:29:23') < self.minimum_island_area_mm2 <= _literal('__post_init__:29:59')):
                raise ValueError(_literal('__post_init__:30:29'))

    def runtime_identity(inherited, specification):
        settings = Settings(**specification)
        return hashlib.sha256(json.dumps(dict(inherited=inherited, settings=asdict(settings), code_sha256=_producer_sha256('region_guard'), opencv=cv2.__version__), sort_keys=_literal('runtime_identity:36:84')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, runtime_identity=runtime_identity)

def _saved_profile_bottom_gas(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    @dataclass(frozen=True)
    class Settings:
        format: str = VERSION
        maximum_depth_mm: float = _literal('Settings:27:30')
        boundary_contact_mm: float = _literal('Settings:28:33')
        activation_after_last_injection_h: float = _literal('Settings:29:47')
        source_protection_radius_mm: float = _literal('Settings:30:41')
        source_protection_after_end_h: float = _literal('Settings:31:43')

        def __post_init__(self):
            if self.format != VERSION:
                raise ValueError(_literal('__post_init__:35:29'))
            for name in (_literal('__post_init__:36:21'), _literal('__post_init__:36:41'), _literal('__post_init__:37:21'), _literal('__post_init__:37:58'), _literal('__post_init__:38:21')):
                value = getattr(self, name)
                if not np.isfinite(value) or value <= _literal('__post_init__:40:50'):
                    raise ValueError(f'Invalid bottom-gas guard {name}')
            if self.maximum_depth_mm > _literal('__post_init__:42:36') or self.boundary_contact_mm > self.maximum_depth_mm or self.source_protection_radius_mm > _literal('__post_init__:44:54') or (self.activation_after_last_injection_h > _literal('__post_init__:45:60')) or (self.source_protection_after_end_h > _literal('__post_init__:46:56')):
                raise ValueError(_literal('__post_init__:47:29'))

    def runtime_identity(inherited, specification):
        return hashlib.sha256(json.dumps(dict(inherited=inherited, settings=asdict(Settings(**specification)), code_sha256=_producer_sha256('bottom_gas'), opencv=cv2.__version__), sort_keys=_literal('runtime_identity:53:43')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, runtime_identity=runtime_identity)

def _saved_profile_region_edge(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    @dataclass(frozen=True)
    class Settings:
        format: str = VERSION
        smoothing_sigma_mm: float = _literal('Settings:27:32')
        maximum_changed_area_fraction: float = _literal('Settings:28:43')

        def __post_init__(self):
            if self.format != VERSION:
                raise ValueError(_literal('__post_init__:32:29'))
            if not np.isfinite(self.smoothing_sigma_mm) or not _literal('__post_init__:34:23') < self.smoothing_sigma_mm <= _literal('__post_init__:34:54'):
                raise ValueError(_literal('__post_init__:35:29'))
            if not np.isfinite(self.maximum_changed_area_fraction) or not _literal('__post_init__:37:23') < self.maximum_changed_area_fraction <= _literal('__post_init__:37:65'):
                raise ValueError(_literal('__post_init__:38:29'))

    def runtime_identity(inherited, specification):
        """Version the setting and implementation without changing inherited runs."""
        settings = Settings(**specification)
        payload = dict(inherited=inherited, settings=asdict(settings), code_sha256=_producer_sha256('region_edge'), opencv=cv2.__version__)
        return hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:46:56')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, runtime_identity=runtime_identity)

def _saved_profile_regional_gas(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    @dataclass(frozen=True)
    class Settings:
        format: str = VERSION
        threshold: float = _literal('Settings:27:23')

        def __post_init__(self):
            if self.format != VERSION or not np.isfinite(self.threshold) or (not _literal('__post_init__:30:76') < self.threshold < _literal('__post_init__:30:97')):
                raise ValueError(_literal('__post_init__:31:29'))

    def runtime_identity(inherited, specification):
        return hashlib.sha256(json.dumps(dict(inherited=inherited, settings=asdict(Settings(**specification)), code_sha256=_producer_sha256('regional_gas'), opencv=cv2.__version__), sort_keys=_literal('runtime_identity:37:43')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, Settings=Settings, VERSION=VERSION, runtime_identity=runtime_identity)

def _saved_profile_approved_gas(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    G3_SHA256 = _export('G3_SHA256')
    CO2_SHA256 = _export('CO2_SHA256')
    REFIT_SHA256 = _export('REFIT_SHA256')
    from .project import sha256_file

    @dataclass(frozen=True)
    class ApprovedG3Settings:
        format: str = VERSION
        pixel_threshold: float = _literal('ApprovedG3Settings:33:29')
        regional_mode: str = _literal('ApprovedG3Settings:34:25')
        regional_threshold: float = _literal('ApprovedG3Settings:35:32')
        minimum_area_mm2: float = _literal('ApprovedG3Settings:36:30')
        connectivity: int = _literal('ApprovedG3Settings:37:24')

        def __post_init__(self):
            if self.format != VERSION or self.pixel_threshold not in (_literal('__post_init__:40:67'), _literal('__post_init__:40:71'), _literal('__post_init__:40:75')) or self.regional_mode not in (_literal('__post_init__:41:46'), _literal('__post_init__:41:56'), _literal('__post_init__:41:65')) or (self.regional_threshold != _literal('__post_init__:42:46')) or (self.minimum_area_mm2 not in (_literal('__post_init__:42:82'), _literal('__post_init__:42:86'), _literal('__post_init__:42:90'))) or (self.connectivity != _literal('__post_init__:43:40')):
                raise ValueError(_literal('__post_init__:44:29'))

    def load_approved_profile(root, profile):
        """Validate an explicit opt-in without replacing the legacy generic head."""
        root = Path(root).resolve()
        marker = profile[PROFILE_KEY]
        required = {_literal('load_approved_profile:212:16'), _literal('load_approved_profile:212:26'), _literal('load_approved_profile:212:38'), _literal('load_approved_profile:212:44'), _literal('load_approved_profile:212:56'), _literal('load_approved_profile:212:63')}
        if not isinstance(marker, dict) or set(marker) != required or marker[_literal('load_approved_profile:213:74')] != VERSION or (ApprovedG3Settings(**marker[_literal('load_approved_profile:214:43')]) != APPROVED_SETTINGS) or (marker[_literal('load_approved_profile:215:22')] != _literal('load_approved_profile:215:53')):
            raise ValueError(_literal('load_approved_profile:216:25'))
        if any((name in profile for name in (_literal('load_approved_profile:217:40'), _literal('load_approved_profile:217:57'), _literal('load_approved_profile:217:77'), _literal('load_approved_profile:218:39')))):
            raise ValueError(_literal('load_approved_profile:219:25'))
        paths = {}
        for name, expected in ((_literal('load_approved_profile:221:28'), G3_SHA256), (_literal('load_approved_profile:221:47'), REFIT_SHA256), (_literal('load_approved_profile:221:75'), CO2_SHA256)):
            entry = marker[name]
            if not isinstance(entry, dict) or set(entry) != {_literal('load_approved_profile:223:57'), _literal('load_approved_profile:223:65')}:
                raise ValueError(_literal('load_approved_profile:224:29'))
            path = (root / entry[_literal('load_approved_profile:225:29')]).resolve()
            if not path.is_relative_to(root) or entry[_literal('load_approved_profile:226:50')] != expected or sha256_file(path) != expected:
                raise ValueError(_literal('load_approved_profile:227:29'))
            paths[name] = str(path)
        if marker[_literal('load_approved_profile:229:14')] != profile[_literal('load_approved_profile:229:37')]:
            raise ValueError(_literal('load_approved_profile:230:25'))
        if marker[_literal('load_approved_profile:231:14')][_literal('load_approved_profile:231:21')] != _literal('load_approved_profile:231:32'):
            raise ValueError(_literal('load_approved_profile:232:25'))
        runtime = dict(root=str(root), artifact_paths=paths, frame_bindings_directory=str(root / marker[_literal('load_approved_profile:234:62')]))
        return dict(profile, _approved_g3_runtime=runtime)

    def approved_runtime_identity(inherited_identity, marker):
        value = dict(inherited_identity=inherited_identity, marker=marker, code_sha256=_producer_sha256('approved_gas'), numpy=np.__version__, opencv=cv2.__version__)
        return hashlib.sha256(json.dumps(value, sort_keys=_literal('approved_runtime_identity:241:54')).encode()).hexdigest()
    APPROVED_SETTINGS = ApprovedG3Settings()
    return _SavedProfileNamespace(APPROVED_SETTINGS=APPROVED_SETTINGS, ApprovedG3Settings=ApprovedG3Settings, CO2_SHA256=CO2_SHA256, G3_SHA256=G3_SHA256, PROFILE_KEY=PROFILE_KEY, REFIT_SHA256=REFIT_SHA256, VERSION=VERSION, approved_runtime_identity=approved_runtime_identity, load_approved_profile=load_approved_profile)

def _saved_profile_top_revision(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    from .project import sha256_file

    def _read_bound(entry, label, *, root=None):
        if not isinstance(entry, dict) or set(entry) != {_literal('_read_bound:24:54'), _literal('_read_bound:24:62')} or (not isinstance(entry[_literal('_read_bound:25:36')], str)) or (len(entry[_literal('_read_bound:25:65')]) != _literal('_read_bound:25:79')):
            raise ValueError(f'Top revision {label} lacks an exact artifact binding')
        path = Path(entry[_literal('_read_bound:27:22')])
        path = (Path(root) / path).resolve() if root is not _literal('_read_bound:28:56') else path.resolve()
        if sha256_file(path) != entry[_literal('_read_bound:29:34')]:
            raise ValueError(f'Top revision {label} changed')
        return path

    def load_top_revision_profile(root, profile):
        """Validate the new revision without changing the inherited G3 profile."""
        marker = profile[PROFILE_KEY]
        if _literal('load_top_revision_profile:37:8') not in profile or not profile.get(_literal('load_top_revision_profile:37:60')) or (not isinstance(marker, dict)) or (set(marker) != {_literal('load_top_revision_profile:39:16'), _literal('load_top_revision_profile:39:26'), _literal('load_top_revision_profile:39:38'), _literal('load_top_revision_profile:39:51'), _literal('load_top_revision_profile:39:74')}) or (marker[_literal('load_top_revision_profile:40:22')] != VERSION) or (marker[_literal('load_top_revision_profile:41:22')] != _literal('load_top_revision_profile:41:38')) or (marker[_literal('load_top_revision_profile:42:22')] != _literal('load_top_revision_profile:42:48')) or (marker[_literal('load_top_revision_profile:43:22')] != _literal('load_top_revision_profile:43:50')):
            raise ValueError(_literal('load_top_revision_profile:44:25'))
        candidate = Path(root).resolve()
        manifest_path = _read_bound(marker[_literal('load_top_revision_profile:46:39')], _literal('load_top_revision_profile:46:52'), root=candidate)
        if not manifest_path.is_relative_to(candidate):
            raise ValueError(_literal('load_top_revision_profile:48:25'))
        manifest = json.loads(manifest_path.read_text(encoding=_literal('load_top_revision_profile:49:59')))
        frames = manifest.get(_literal('load_top_revision_profile:50:26'))
        if manifest.get(_literal('load_top_revision_profile:51:21'), manifest.get(_literal('load_top_revision_profile:51:46'))) != VERSION or not isinstance(frames, list) or (not frames):
            raise ValueError(_literal('load_top_revision_profile:53:25'))
        rows = {}
        for row in frames:
            run, index = (str(row.get(_literal('load_top_revision_profile:56:33'), _literal('load_top_revision_profile:56:40'))).casefold(), row.get(_literal('load_top_revision_profile:56:65')))
            if not run or not isinstance(index, int) or index < _literal('load_top_revision_profile:57:61') or ((run, index) in rows) or (f'ffac/{run}' not in profile[_literal('load_top_revision_profile:58:48')]) or (not np.isfinite(float(row.get(_literal('load_top_revision_profile:59:49'), np.nan)))):
                raise ValueError(_literal('load_top_revision_profile:60:29'))
            rows[run, index] = row
        return dict(profile, _photo_sand_top_runtime={_literal('load_top_revision_profile:62:50'): str(candidate), _literal('load_top_revision_profile:62:74'): rows, _literal('load_top_revision_profile:63:49'): marker[_literal('load_top_revision_profile:63:75')][_literal('load_top_revision_profile:63:87')]})

    def top_revision_runtime_identity(inherited_identity, marker):
        value = dict(inherited_identity=inherited_identity, revision=marker, code_sha256=_producer_sha256('top_revision'), numpy=np.__version__)
        return hashlib.sha256(json.dumps(value, sort_keys=_literal('top_revision_runtime_identity:167:54')).encode()).hexdigest()
    return _SavedProfileNamespace(PROFILE_KEY=PROFILE_KEY, VERSION=VERSION, _read_bound=_read_bound, load_top_revision_profile=load_top_revision_profile, top_revision_runtime_identity=top_revision_runtime_identity)

def _saved_profile_barrier(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    PROFILE_KEY = _export('PROFILE_KEY')
    SPEC_SHA256 = _export('SPEC_SHA256')
    OPERATION = _export('OPERATION')
    from .project import sha256_file

    def bound_path(asset, *, root=None):
        if not isinstance(asset, dict) or set(asset) != {_literal('bound_path:21:53'), _literal('bound_path:21:61')}:
            raise ValueError(_literal('bound_path:22:25'))
        path = Path(asset[_literal('bound_path:23:22')])
        path = (Path(root) / path).resolve() if root is not _literal('bound_path:24:56') else path.resolve()
        if sha256_file(path) != asset[_literal('bound_path:25:34')]:
            raise ValueError(_literal('bound_path:26:25'))
        return path

    def load_barrier_profile(root, profile):
        marker = profile[PROFILE_KEY]
        if not profile.get(_literal('load_barrier_profile:63:24')) or not isinstance(marker, dict) or set(marker) != {_literal('load_barrier_profile:64:31'), _literal('load_barrier_profile:64:41'), _literal('load_barrier_profile:64:53'), _literal('load_barrier_profile:64:68'), _literal('load_barrier_profile:64:81')} or (marker[_literal('load_barrier_profile:65:22')] != VERSION) or (marker[_literal('load_barrier_profile:65:53')] != SPEC_SHA256) or (marker[_literal('load_barrier_profile:66:22')] != OPERATION):
            raise ValueError(_literal('load_barrier_profile:67:25'))
        candidate = Path(root).resolve()
        manifest_path = bound_path(marker[_literal('load_barrier_profile:69:38')], root=candidate)
        derivation_path = bound_path(marker[_literal('load_barrier_profile:70:40')], root=candidate)
        if not manifest_path.is_relative_to(candidate) or not derivation_path.is_relative_to(candidate):
            raise ValueError(_literal('load_barrier_profile:72:25'))
        manifest = json.loads(manifest_path.read_text(encoding=_literal('load_barrier_profile:73:59')))
        if manifest.get(_literal('load_barrier_profile:74:21')) != VERSION or manifest.get(_literal('load_barrier_profile:74:60')) != SPEC_SHA256 or manifest.get(_literal('load_barrier_profile:75:28')) != marker[_literal('load_barrier_profile:75:67')][_literal('load_barrier_profile:75:89')] or (not isinstance(manifest.get(_literal('load_barrier_profile:76:43')), list)) or (not manifest[_literal('load_barrier_profile:76:76')]):
            raise ValueError(_literal('load_barrier_profile:77:25'))
        frames = {}
        for row in manifest[_literal('load_barrier_profile:79:24')]:
            run, index = (str(row.get(_literal('load_barrier_profile:80:33'), _literal('load_barrier_profile:80:40'))).casefold(), row.get(_literal('load_barrier_profile:80:65')))
            if type(index) is not int or index < _literal('load_barrier_profile:81:46') or (run, index) in frames or (_literal('load_barrier_profile:82:19') + run not in profile[_literal('load_barrier_profile:82:48')]) or (not np.isfinite(float(row.get(_literal('load_barrier_profile:83:49'), np.nan)))):
                raise ValueError(_literal('load_barrier_profile:84:29'))
            frames[run, index] = row
        return dict(profile, _gas_barrier_runtime=dict(root=str(candidate), frames=frames))

    def barrier_runtime_identity(inherited_identity, marker):
        core = saved_physics_profile('barrier_core_source')
        value = dict(inherited_identity=inherited_identity, marker=marker, code_sha256=_producer_sha256('barrier'), core_sha256=_producer_sha256('barrier_core_source'), numpy=np.__version__)
        return hashlib.sha256(json.dumps(value, sort_keys=_literal('barrier_runtime_identity:130:54')).encode()).hexdigest()
    return _SavedProfileNamespace(OPERATION=OPERATION, PROFILE_KEY=PROFILE_KEY, SPEC_SHA256=SPEC_SHA256, VERSION=VERSION, barrier_runtime_identity=barrier_runtime_identity, bound_path=bound_path, load_barrier_profile=load_barrier_profile)

def _saved_profile_barrier_core_source(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    VERSION = _export('VERSION')
    GEOMETRY_VERSION = _export('GEOMETRY_VERSION')
    SPEC_SHA256 = _export('SPEC_SHA256')
    return _SavedProfileNamespace(GEOMETRY_VERSION=GEOMETRY_VERSION, SPEC_SHA256=SPEC_SHA256, VERSION=VERSION)

def _saved_profile_rig_camera(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    SPEC_SHA256 = _export('SPEC_SHA256')
    PROFILE_FILE = _export('PROFILE_FILE')
    from .project import sha256_file
    from .review_storage import resolve_review_path

    def read(path):
        return json.loads(Path(path).read_text(encoding=_literal('read:24:52')))

    def bound_path(asset, *, base=None, verify=True):
        saved_physics_profile('rig_metadata').asset(asset)
        path = Path(asset[_literal('bound_path:29:20')])
        path = resolve_review_path(path if path.is_absolute() else Path(base or _literal('bound_path:30:74')) / path).resolve()
        if verify and sha256_file(path) != asset[_literal('bound_path:31:43')]:
            raise ValueError(_literal('bound_path:32:25'))
        if not verify and (not path.is_file()):
            raise ValueError(_literal('bound_path:34:25'))
        return path

    def load_profile(root):
        candidate = resolve_review_path(root).resolve()
        path = candidate / PROFILE_FILE
        profile = read(path)
        saved_physics_profile('rig_metadata').require(profile.get(_literal('load_profile:166:33')) == saved_physics_profile('rig_metadata').PROFILE_FORMAT and profile.get(_literal('load_profile:166:84')) == SPEC_SHA256 and (profile.get(_literal('load_profile:167:24')) == saved_physics_profile('rig_metadata').CONFIGURATION) and (profile.get(_literal('load_profile:168:24')) == _literal('load_profile:168:46')) and all((profile.get(key) is _literal('load_profile:169:36') for key in (_literal('load_profile:170:12'), _literal('load_profile:170:28'), _literal('load_profile:170:52'), _literal('load_profile:170:73'), _literal('load_profile:170:96'), _literal('load_profile:170:107')))), _literal('load_profile:171:8'))
        marker = profile.get(_literal('load_profile:172:23')) or {}
        saved_physics_profile('rig_metadata').require(marker.get(_literal('load_profile:173:32')) == saved_physics_profile('rig_metadata').REVISION_CONTRACT, _literal('load_profile:174:21'))
        phase_grid = saved_physics_profile('rig_metadata').validate_phase_grid(marker.get(_literal('load_profile:175:55')))
        manifest_path = bound_path(marker[_literal('load_profile:176:36')], base=candidate)
        saved_physics_profile('rig_metadata').require(manifest_path.is_relative_to(candidate), _literal('load_profile:177:61'))
        manifest = read(manifest_path)
        saved_physics_profile('rig_metadata').validate_campaign_bindings(manifest[_literal('load_profile:179:49')])
        saved_physics_profile('rig_metadata').require(manifest.get(_literal('load_profile:180:34')) == saved_physics_profile('rig_metadata').REVISION_CONTRACT and manifest.get(_literal('load_profile:180:91')) == SPEC_SHA256, _literal('load_profile:181:21'))
        saved_physics_profile('rig_metadata').require(manifest.get(_literal('load_profile:182:34'), phase_grid) == phase_grid, _literal('load_profile:183:21'))
        for source in manifest[_literal('load_profile:184:27')]:
            for row in source[_literal('load_profile:185:26')]:
                saved_physics_profile('rig_metadata').require(row.get(_literal('load_profile:186:37')) == phase_grid, _literal('load_profile:187:29'))
        acceptance = read(bound_path(manifest[_literal('load_profile:188:40')], base=manifest_path.parent))
        saved_physics_profile('rig_metadata').validate_geometry_acceptance(acceptance, geometry_sha256=manifest[_literal('load_profile:189:78')][_literal('load_profile:189:99')], spec_sha256=SPEC_SHA256)
        for name in (_literal('load_profile:190:17'), _literal('load_profile:190:41')):
            bound_path(acceptance[name], base=manifest_path.parent)
        bound_path(manifest[_literal('load_profile:192:24')], base=manifest_path.parent)
        if _literal('load_profile:193:7') in profile:
            saved_physics_profile('rig_metadata').require(not {_literal('load_profile:194:30'), _literal('load_profile:194:54'), _literal('load_profile:195:30'), _literal('load_profile:195:54'), _literal('load_profile:196:30')}.intersection(profile), _literal('load_profile:197:25'))
            validate_runtime_heads = saved_revision_bindings('epoch_floor').validate_runtime_heads
            validate_runtime_heads(candidate, profile[_literal('load_profile:199:50')], profile[_literal('load_profile:199:84')])
        elif _literal('load_profile:200:9') in profile:
            saved_physics_profile('rig_metadata').require(not {_literal('load_profile:201:30'), _literal('load_profile:201:65'), _literal('load_profile:202:30'), _literal('load_profile:202:55')}.intersection(profile), _literal('load_profile:203:25'))
            validate_runtime_heads = saved_revision_bindings('trained_floor').validate_runtime_heads
            validate_runtime_heads(candidate, profile[_literal('load_profile:205:50')], profile[_literal('load_profile:205:83')])
        elif _literal('load_profile:206:9') in profile:
            saved_physics_profile('rig_metadata').require(not {_literal('load_profile:207:30'), _literal('load_profile:207:54'), _literal('load_profile:207:79')}.intersection(profile), _literal('load_profile:208:25'))
            validate_runtime_heads = saved_revision_bindings('neighbourhood_floor').validate_runtime_heads
            validate_runtime_heads(candidate, profile[_literal('load_profile:210:50')], profile[_literal('load_profile:210:94')])
        elif _literal('load_profile:211:9') in profile:
            saved_physics_profile('rig_metadata').require(not {_literal('load_profile:212:30'), _literal('load_profile:212:55')}.intersection(profile), _literal('load_profile:213:25'))
            validate_runtime_heads = saved_revision_bindings('joint_floor').validate_runtime_heads
            validate_runtime_heads(candidate, profile[_literal('load_profile:215:50')], profile[_literal('load_profile:215:83')])
        elif _literal('load_profile:216:9') in profile:
            saved_physics_profile('rig_metadata').require(_literal('load_profile:217:25') not in profile, _literal('load_profile:218:25'))
            validate_runtime_heads = saved_revision_bindings('combined_floor').validate_runtime_heads
            validate_runtime_heads(candidate, profile[_literal('load_profile:220:50')], profile[_literal('load_profile:220:84')])
        elif _literal('load_profile:221:9') in profile:
            validate_runtime_heads = saved_revision_bindings('legacy_floor').validate_runtime_heads
            validate_runtime_heads(candidate, profile[_literal('load_profile:223:50')], profile[_literal('load_profile:223:83')])
        else:
            CO2_SHA256, G3_SHA256, REFIT_SHA256 = (saved_physics_profile('approved_gas').CO2_SHA256, saved_physics_profile('approved_gas').G3_SHA256, saved_physics_profile('approved_gas').REFIT_SHA256)
            for key, expected in ((_literal('load_profile:226:30'), saved_physics_profile('approved_gas').CO2_SHA256), (_literal('load_profile:226:49'), saved_physics_profile('approved_gas').G3_SHA256), (_literal('load_profile:226:66'), saved_physics_profile('approved_gas').REFIT_SHA256)):
                saved_physics_profile('rig_metadata').require(profile[_literal('load_profile:227:37')][key][_literal('load_profile:227:58')] == expected, _literal('load_profile:227:78'))
                bound_path(profile[_literal('load_profile:228:31')][key], base=candidate)
        return dict(profile, _rig_camera_runtime=dict(root=str(candidate), manifest=manifest))

    def runtime_identity(profile):
        payload = dict(profile=saved_physics_profile('rig_metadata').public_profile(profile), reader_sha256=_producer_sha256('rig_camera'), metadata_sha256=_producer_sha256('rig_metadata'), numpy=np.__version__)
        return hashlib.sha256(json.dumps(payload, sort_keys=_literal('runtime_identity:235:55'), allow_nan=_literal('runtime_identity:235:70')).encode()).hexdigest()

    def require_dedicated_inference():
        raise ValueError(_literal('require_dedicated_inference:239:21'))
    return _SavedProfileNamespace(PROFILE_FILE=PROFILE_FILE, SPEC_SHA256=SPEC_SHA256, bound_path=bound_path, load_profile=load_profile, read=read, require_dedicated_inference=require_dedicated_inference, runtime_identity=runtime_identity)

def _saved_profile_rig_metadata(parameters):

    def _literal(key):
        return _saved_profile_typed(parameters['literals'][key])

    def _export(key):
        return _saved_profile_typed(parameters['exports'][key])
    SOURCE_KEY = _export('SOURCE_KEY')
    SOURCE_CONTRACT = _export('SOURCE_CONTRACT')
    INPUT_CONTRACT = _export('INPUT_CONTRACT')
    PROFILE_FORMAT = _export('PROFILE_FORMAT')
    REVISION_CONTRACT = _export('REVISION_CONTRACT')
    QA_CONTRACT = _export('QA_CONTRACT')
    ACCEPTANCE_CONTRACT = _export('ACCEPTANCE_CONTRACT')
    RIG_CONTRACT = _export('RIG_CONTRACT')
    SCIENTIFIC_CONTRACT = _export('SCIENTIFIC_CONTRACT')
    EXTENDED_SCIENTIFIC_CONTRACT = _export('EXTENDED_SCIENTIFIC_CONTRACT')
    SOURCE_SHAPE = _export('SOURCE_SHAPE')
    SOURCE_BOX = _export('SOURCE_BOX')
    CANVAS_WH = _export('CANVAS_WH')
    ORIGIN_XY = _export('ORIGIN_XY')
    CONFIGURATION = _export('CONFIGURATION')
    _metadata = _SavedRigMetadata(configured_parameters('rig_source')['metadata'])
    require, asset = (_metadata.require, _metadata.asset)
    phase_grid, validate_phase_grid = (_metadata.phase_grid, _metadata.validate_phase_grid)

    def validate_campaign_bindings(sources):
        """Require all 43 runs/864 unique contiguous selected frame positions."""
        require(isinstance(sources, list) and len(sources) == _literal('validate_campaign_bindings:244:55'), _literal('validate_campaign_bindings:244:58'))
        keys = set()
        runs = set()
        for source in sources:
            run = source.get(_literal('validate_campaign_bindings:247:23'))
            require(run not in runs and isinstance(run, str) and re.fullmatch(_literal('validate_campaign_bindings:248:73'), run), _literal('validate_campaign_bindings:248:87'))
            runs.add(run)
            for index, row in enumerate(source.get(_literal('validate_campaign_bindings:250:46')) or []):
                require(row.get(_literal('validate_campaign_bindings:251:28')) == run and row.get(_literal('validate_campaign_bindings:251:52')) == index, _literal('validate_campaign_bindings:251:74'))
                keys.add((run, index))
        require(len(keys) == _literal('validate_campaign_bindings:253:23'), _literal('validate_campaign_bindings:253:27'))
        return keys

    def validate_geometry_acceptance(value, *, geometry_sha256, spec_sha256):
        require(value.get(_literal('validate_geometry_acceptance:280:22')) == ACCEPTANCE_CONTRACT and value.get(_literal('validate_geometry_acceptance:280:69')) is _literal('validate_geometry_acceptance:280:84') and (value.get(_literal('validate_geometry_acceptance:281:26')) == geometry_sha256) and (value.get(_literal('validate_geometry_acceptance:282:26')) == spec_sha256) and (value.get(_literal('validate_geometry_acceptance:282:68')) == _literal('validate_geometry_acceptance:282:82')) and (value.get(_literal('validate_geometry_acceptance:283:26')) == _literal('validate_geometry_acceptance:283:42')) and (value.get(_literal('validate_geometry_acceptance:283:60')) is _literal('validate_geometry_acceptance:283:91')) and (type(value.get(_literal('validate_geometry_acceptance:284:31'))) is bool) and (value.get(_literal('validate_geometry_acceptance:285:26')) is _literal('validate_geometry_acceptance:285:60')) and (value.get(_literal('validate_geometry_acceptance:286:26')) in (_literal('validate_geometry_acceptance:286:61'), _literal('validate_geometry_acceptance:286:71'), _literal('validate_geometry_acceptance:286:86'))), _literal('validate_geometry_acceptance:287:12'))
        asset(value.get(_literal('validate_geometry_acceptance:288:20')))
        asset(value.get(_literal('validate_geometry_acceptance:289:20')))
        return deepcopy(value)

    def public_profile(profile):
        """Serialize public physics only; private tuple-key caches are derived at read time."""
        return {key: deepcopy(value) for key, value in profile.items() if not key.startswith(_literal('public_profile:259:87'))}
    return _SavedProfileNamespace(ACCEPTANCE_CONTRACT=ACCEPTANCE_CONTRACT, CANVAS_WH=CANVAS_WH, CONFIGURATION=CONFIGURATION, EXTENDED_SCIENTIFIC_CONTRACT=EXTENDED_SCIENTIFIC_CONTRACT, INPUT_CONTRACT=INPUT_CONTRACT, ORIGIN_XY=ORIGIN_XY, PROFILE_FORMAT=PROFILE_FORMAT, QA_CONTRACT=QA_CONTRACT, REVISION_CONTRACT=REVISION_CONTRACT, RIG_CONTRACT=RIG_CONTRACT, SCIENTIFIC_CONTRACT=SCIENTIFIC_CONTRACT, SOURCE_BOX=SOURCE_BOX, SOURCE_CONTRACT=SOURCE_CONTRACT, SOURCE_KEY=SOURCE_KEY, SOURCE_SHAPE=SOURCE_SHAPE, asset=asset, phase_grid=phase_grid, public_profile=public_profile, require=require, validate_campaign_bindings=validate_campaign_bindings, validate_geometry_acceptance=validate_geometry_acceptance, validate_phase_grid=validate_phase_grid)

_SAVED_PROFILE_FACTORIES = {
    'decision_dispatch': _saved_profile_decision_dispatch,
    'completion': _saved_profile_completion,
    'local_support': _saved_profile_local_support,
    'temporal_support': _saved_profile_temporal_support,
    'source_history': _saved_profile_source_history,
    'boundary': _saved_profile_boundary,
    'boundary_core': _saved_profile_boundary_core,
    'boundary_extended': _saved_profile_boundary_extended,
    'boundary_extended_core': _saved_profile_boundary_extended_core,
    'boundary_continuity': _saved_profile_boundary_continuity,
    'decision': _saved_profile_decision,
    'decision_core': _saved_profile_decision_core,
    'decision_age': _saved_profile_decision_age,
    'decision_age_core': _saved_profile_decision_age_core,
    'regional_decision': _saved_profile_regional_decision,
    'free_gas': _saved_profile_free_gas,
    'region_guard': _saved_profile_region_guard,
    'bottom_gas': _saved_profile_bottom_gas,
    'region_edge': _saved_profile_region_edge,
    'regional_gas': _saved_profile_regional_gas,
    'approved_gas': _saved_profile_approved_gas,
    'top_revision': _saved_profile_top_revision,
    'barrier': _saved_profile_barrier,
    'barrier_core_source': _saved_profile_barrier_core_source,
    'rig_camera': _saved_profile_rig_camera,
    'rig_metadata': _saved_profile_rig_metadata,
}

def metadata_file_stamp(path):
    """Return the exact size and three filesystem identity stamps for saved guards."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def typed_saved_data_value(item):
    """Decode the original narrow saved-DATA type declaration without coercion."""
    value = item["value"]
    if item["python_type"] == "tuple":
        return tuple(value)
    if type(value).__name__ != item["python_type"]:
        raise TypeError("Saved metadata DATA changed its original Python type")
    return value
