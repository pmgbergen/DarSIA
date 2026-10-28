"""External, immutable display-file receipts for complete rig RAW photographs.

These files are presentation derivatives. No candidate, scientific sidecar,
phase map, label, or session is written by this module.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from functools import lru_cache
import json
import os
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image

CONTRACT = "ffac-rig-raw-display-prerender-v1"
_ROOTS = OrderedDict()
_JSON = OrderedDict()
_JSON_SIZES = {}
_POINTERS = OrderedDict()
_LOCK = RLock()
_PREFETCH = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rig-display-prefetch")
_PENDING = set()
_PREFETCH_GENERATION = 0
_PREFETCH_LIMIT = 4
_PREFETCH_SELECTION = None

# This previously frozen renderer differs only in path/metadata/cache handling.
# Accept its saved pixels only while the actual numerical implementation still
# has the same AST fingerprint. Future pixel changes cannot reuse this allowance.


def _key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def candidate_root(review_root):
    from .review_storage import resolve_review_path
    from .phase_rig_raw_display import _path_configuration, _remember
    path = resolve_review_path(review_root)
    key = (_key(path), _path_configuration())
    with _LOCK:
        if key in _ROOTS:
            return _ROOTS[key]
    for root in [path, *list(path.parents)[:10]]:
        if any((root / name).is_file() for name in ("p1_final3_plan.json", "p1_final2_plan.json", "p1_final_plan.json")):
            with _LOCK:
                _remember(_ROOTS, key, root, limit=128)
            return root
    return None


def pointer_path(root, *, registry_dir=None):
    from .phase_saved_registration import registration_path
    from .phase_rig_raw_display import _path_configuration, _remember
    key = (_key(root), str(registry_dir), _path_configuration())
    with _LOCK:
        if key in _POINTERS:
            _POINTERS.move_to_end(key)
            return _POINTERS[key]
    parent = Path(registry_dir) if registry_dir is not None else registration_path(Path(root).name).parent
    path = parent / "studio_display_renders" / (Path(root).name + ".json")
    with _LOCK:
        _remember(_POINTERS, key, path, limit=128)
    return path


def source_binding(state, record, binding, *, mode, reference):
    manifest = state["native"]
    return dict(renderer_contract="ffac-rig-extended-raw-display-v1", renderer_code=_renderer_code(),
                mode=mode, reference=reference,
                source_sha256=record["source_sha256"], source_record=record, frame_binding=binding,
                reference_record=manifest["reference"] if mode == "change" else None,
                content_box_xywh=manifest["content_box_xywh"],
                analysis_image_shape_hw=manifest["analysis_image_shape_hw"])


@lru_cache(maxsize=16)
def _code_sha(path, stamp):
    return sha256(Path(path).read_bytes()).hexdigest()



def _renderer_code():
    """Admit current code separately from immutable saved producer identities."""
    from .phase_saved_review_metadata import configured_parameters, bound_path
    parameters = configured_parameters("display_renderer")
    base = Path(__file__).parent
    for asset in parameters["current_execution_sources"].values():
        path = bound_path(asset, base=base)
        if path != (base / asset["path"]).resolve() or path.stat().st_size != asset["size_bytes"]:
            raise ValueError("Current display execution source binding differs")
    result = {}
    for name, asset in parameters["producer_code_sources"].items():
        path = bound_path(asset)
        stat = path.stat()
        if stat.st_size != asset["size_bytes"]:
            raise ValueError("Saved display producer source size differs")
        result[name] = _code_sha(str(path), (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return result


@lru_cache(maxsize=16)
def _pixel_fingerprint(path, stamp):
    fields = {"CONTRACT", "BACKGROUND", "MISSING_DARK", "MISSING_LIGHT"}
    functions = {"rig_raw_display_available", "compose_display_raw", "render_raw_detail",
                 "mark_missing_raw", "_prepared_reference", "_render_rig_raw_display_rgb",
                 "_asset_array", "_raw_assets", "load_rig_raw_display_rgb"}
    return _function_fingerprint(path, fields, functions)


@lru_cache(maxsize=16)
def _grid_fingerprint(path, stamp):
    return _function_fingerprint(path, {"RIG_CONTRACT"}, {
        "_shape", "_asset", "_array_asset", "crop_native_grid", "rig_phase_shape", "load_rig_classes"})


def _function_fingerprint(path, fields, functions):
    selected = []
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in fields for target in node.targets):
            selected.append(node)
    code = ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)
    return sha256(code.encode()).hexdigest()


def _compatible_binding(expected, actual):
    """Allow the audited cache-only revision; retain every source/grid binding."""
    if actual == expected:
        return True
    if not isinstance(actual, dict):
        return False
    from .phase_saved_review_metadata import configured_parameters, bound_path
    from .phase_rig_raw_display import _stamp
    parameters = configured_parameters("display_renderer")
    name = parameters["raw_renderer_name"]
    codes = actual.get("renderer_code")
    if not isinstance(codes, dict) or codes.get(name) != parameters["compatible_raw_renderer"]:
        return False
    adjusted = dict(codes, **{name: expected["renderer_code"][name]})
    if adjusted != expected["renderer_code"]:
        return False
    path = bound_path(parameters["pixel_fingerprint_source"])
    if _pixel_fingerprint(str(path), _stamp(path)) != parameters["compatible_raw_pixels"]:
        return False
    grid_path = bound_path(parameters["grid_fingerprint_source"])
    if _grid_fingerprint(str(grid_path), _stamp(grid_path)) != parameters["compatible_grid_pixels"]:
        return False
    return dict(actual, renderer_code=adjusted) == expected


def _token(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_cached(path, key):
    from .phase_rig_raw_display import _cache_budget_bytes
    with _LOCK:
        if key in _JSON:
            _JSON.move_to_end(key)
            return _JSON[key]
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        # JSON metadata contains many Python floats/lists. Allow four times the
        # file size and cap separately from full-resolution decoded pixels.
        size = key[1][0] * 4
        budget = min(128 << 20, _cache_budget_bytes() // 8)
        if size <= budget:
            _JSON[key] = value
            _JSON_SIZES[key] = size
            while len(_JSON) > 256 or sum(_JSON_SIZES.values()) > budget:
                oldest, _ = _JSON.popitem(last=False)
                _JSON_SIZES.pop(oldest)
        return value


def _read_verified(asset):
    from .phase_rig_raw_display import _verified_display_identity
    path, key = _verified_display_identity(asset)
    return _json_cached(path, key)


def _pointer(path):
    from .phase_rig_raw_display import _stamp
    key = (_key(path), _stamp(path))
    return _json_cached(path, key)


def load_prerendered_rgb(review_root, state, record, binding, run, frame_index, *, mode, reference,
                         copy=True):
    """Return a verified display file, or fall back on missing/mismatched binding."""
    from .phase_rig_raw_display import _verified_display_path, _cached
    from .review_storage import resolve_review_path
    root = candidate_root(review_root)
    if root is None:
        return None
    path = pointer_path(root)
    try:
        pointer = _pointer(path)
        if (pointer.get("contract") != CONTRACT or pointer.get("display_only") is not True
                or _key(resolve_review_path(pointer["candidate_root"])) != _key(root)):
            return None
        manifest = _read_verified(pointer["manifest"])
        if manifest.get("contract") != CONTRACT or manifest.get("candidate_id") != root.name:
            return None
        rows = _read_verified(manifest["runs"][str(run).casefold()])["records"]
        row = rows["reference" if reference else f"frame_{frame_index:03d}"][mode]
        expected = source_binding(state, record, binding, mode=mode, reference=reference)
        receipt = _read_verified(row["receipt"])
        if (receipt.get("contract") != CONTRACT
                or row["binding_sha256"] != _token(receipt.get("source_binding"))
                or not _compatible_binding(expected, receipt.get("source_binding"))
                or receipt.get("display") != row["display"] or receipt.get("masks") != row["masks"]):
            return None
        display_path = _verified_display_path(row["display"])
        masks_path = _verified_display_path(row["masks"])
        def decode():
            with Image.open(display_path) as photo:
                rgb = np.asarray(photo.convert("RGB"), dtype=np.uint8).copy()
            with np.load(masks_path, allow_pickle=False) as arrays:
                shape = tuple(int(v) for v in arrays["shape_hw"])
                masks = {name: np.unpackbits(arrays[name], bitorder="little")[:np.prod(shape)].reshape(shape).astype(bool)
                         for name in ("support", "missing", "rig_domain")}
            if rgb.shape != (*shape, 3) or list(shape) != record["rig_domain_revision"]["phase_shape_hw"]:
                raise ValueError("Pre-rendered RAW photograph has another display grid")
            audit = dict(receipt["live_audit"], pre_rendered=True, display_file=row["display"],
                         display_receipt=row["receipt"], presentation_backend={"backend": "saved-file"})
            return rgb, masks, audit
        key = _token(dict(display=row["display"], masks=row["masks"], receipt=row["receipt"]))
        rgb, masks, audit = _cached("prerender", key, decode)
        if not copy:
            # A prefetch only warms the bounded cache; avoid allocating another
            # 134 MB result that the speculative request immediately discards.
            return True
        return rgb.copy(), {name: value.copy() for name, value in masks.items()}, deepcopy(audit)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        # Failed derivative QA never authorises another source's photograph.
        return None


def prefetch_neighbours(review_root, run, frame_index, *, mode, frame_count):
    global _PREFETCH_GENERATION, _PREFETCH_SELECTION
    if type(frame_index) is not int:
        return
    from .phase_rig_raw_display import _native_frame_assets_cached
    selection = (_key(review_root), str(run).casefold(), frame_index)
    with _LOCK:
        if selection != _PREFETCH_SELECTION:
            _PREFETCH_SELECTION = selection
            _PREFETCH_GENERATION += 1
        generation = _PREFETCH_GENERATION
    other_mode = "change" if mode == "raw_detail" else "raw_detail"
    for index, selected_mode in ((frame_index, other_mode), (frame_index-1, mode), (frame_index+1, mode)):
        if not 0 <= index < frame_count:
            continue
        key = (generation, selection[:2], index, selected_mode)
        with _LOCK:
            if key in _PENDING or len(_PENDING) >= _PREFETCH_LIMIT:
                continue
            _PENDING.add(key)
        def load(index=index, selected_mode=selected_mode, key=key):
            try:
                with _LOCK:
                    if generation != _PREFETCH_GENERATION:
                        return
                state, record, binding = _native_frame_assets_cached(review_root, run, index)
                with _LOCK:
                    if generation != _PREFETCH_GENERATION:
                        return
                # Missing/invalid derivatives do not start speculative RAW
                # reconstruction and compete with the selected photograph.
                load_prerendered_rgb(review_root, state, record, binding, run, index,
                                     mode=selected_mode, reference=False, copy=False)
            except Exception:
                pass  # A foreground request will report the real source error.
            finally:
                with _LOCK:
                    _PENDING.discard(key)
        _PREFETCH.submit(load)


def clear_session_cache():
    global _PREFETCH_GENERATION, _PREFETCH_SELECTION
    with _LOCK:
        _PREFETCH_GENERATION += 1
        _PREFETCH_SELECTION = None
        _ROOTS.clear()
        _JSON.clear()
        _JSON_SIZES.clear()
        _POINTERS.clear()
