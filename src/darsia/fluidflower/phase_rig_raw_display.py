"""Source-bound RAW/flat-field photographs on the complete rig display grid.

Only transient presentation pixels are produced. Scientific sidecars, phase
maps, labels, and annotation coordinates are never written or transformed.
Missing RAW/reference support has an explicit neutral hatch inside the rig.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from weakref import WeakValueDictionary

import numpy as np

from .phase_ffac_rig_camera_metadata import EXTENDED_SCIENTIFIC_CONTRACT
from .phase_saved_review_metadata import bound_path, native_frame_assets, load_rig_classes, rig_phase_shape

CONTRACT = "ffac-rig-extended-raw-display-v1"
BACKGROUND = (19, 22, 27)
MISSING_DARK = (83, 88, 94)
MISSING_LIGHT = (131, 137, 144)
_REFERENCES = OrderedDict()
_REFERENCE_LOCK = RLock()
_VERIFIED_ASSETS = OrderedDict()
_DISPLAY_CACHE = OrderedDict()
_KEY_LOCKS = WeakValueDictionary()
_CACHE_LIMIT = None
_NATIVE_STATES = OrderedDict()
_FRAME_STATES = OrderedDict()
_ASSET_PATHS = OrderedDict()
_CACHE_GENERATION = 0
_METADATA_LIMIT = 8192


def _cache_budget_bytes():
    global _CACHE_LIMIT
    if _CACHE_LIMIT is None:
        try:
            import psutil
            available = psutil.virtual_memory().available
        except (ImportError, OSError):
            available = 4 << 30
        _CACHE_LIMIT = max(256 << 20, min(4 << 30, int(available) // 16))
    return _CACHE_LIMIT


def _key_lock(key):
    with _REFERENCE_LOCK:
        return _KEY_LOCKS.setdefault(key, RLock())


def _remember(cache, key, value, *, limit=_METADATA_LIMIT):
    """Keep small presentation metadata bounded as users browse campaigns."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _path_configuration():
    from .review_storage import _configuration_stamp
    return _configuration_stamp(os.environ.get("DARSIA_REVIEW_ALIAS_FILES"))


def _stamp(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            stat.st_dev, stat.st_ino)


def _asset_identity(asset, *, base=None):
    # Resolves configured path aliases, validates the descriptor and checks
    # existence without streaming the NAS asset. Scientific readers are unchanged.
    # Resolving an unchanged UNC path opens it repeatedly on Windows. Preserve
    # the checked path while rechecking its file stamp on every selected read.
    key = (str(asset.get("path")), str(base), asset.get("sha256"), _path_configuration())
    with _REFERENCE_LOCK:
        path = _ASSET_PATHS.get(key)
        if path is not None:
            _ASSET_PATHS.move_to_end(key)
    if path is None:
        path = bound_path(asset, base=base, verify=False)
        with _REFERENCE_LOCK:
            _remember(_ASSET_PATHS, key, path)
    return path, (os.path.normcase(str(path)), _stamp(path), asset["sha256"])


def _verified_display_identity(asset, *, base=None):
    with _key_lock(("sha", str(asset.get("path")), str(base))):
        path, identity = _asset_identity(asset, base=base)
        before = identity[1]
        with _REFERENCE_LOCK:
            saved = _VERIFIED_ASSETS.get(identity[0])
            if saved is not None:
                _VERIFIED_ASSETS.move_to_end(identity[0])
        if saved is not None and saved[0] == before:
            if saved[1] != asset["sha256"]:
                raise ValueError("Rig/camera bound source artifact changed")
            return path, identity
        verified = bound_path(asset, base=base, verify=True)
        if _stamp(verified) != before:
            raise ValueError("Rig RAW display asset changed while verifying")
        with _REFERENCE_LOCK:
            _remember(_VERIFIED_ASSETS, identity[0], (before, asset["sha256"]))
        return verified, identity


def _verified_display_path(asset, *, base=None):
    return _verified_display_identity(asset, base=base)[0]


def _token(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _raw_assets(record):
    descriptor = record["full_extended_scientific"]
    centre = descriptor["native_centre"]
    return [centre["log_linear_rgb"], centre["valid_mask"],
            *[row["log_linear_rgb"] for row in descriptor["ring_windows"]]]


def _raw_identity(record, manifest):
    stamps = []
    for asset in _raw_assets(record):
        _, identity = _verified_display_identity(asset)
        stamps.append(identity)
    return _token(dict(record=record, stamps=stamps,
                       content_box=manifest["content_box_xywh"],
                       analysis_shape=manifest["analysis_image_shape_hw"]))


def _memory_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_memory_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_memory_bytes(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return sum(_memory_bytes(getattr(value, name)) for name in value.__dataclass_fields__)
    return 0


def _cached(kind, key, create):
    identity = (kind, key)
    with _key_lock(identity):
        with _REFERENCE_LOCK:
            if identity in _DISPLAY_CACHE:
                _DISPLAY_CACHE.move_to_end(identity)
                return _DISPLAY_CACHE[identity][0]
            generation = _CACHE_GENERATION
        value = create()  # Independent frames can verify/render concurrently.
        size = _memory_bytes(value)
        with _REFERENCE_LOCK:
            limit = _cache_budget_bytes()
            if size <= limit and generation == _CACHE_GENERATION:
                _DISPLAY_CACHE[identity] = (value, size)
                # Retain active runs' prepared baseline before disposable frame
                # derivatives. On small CPUs keep at least one baseline if it fits.
                refs = [item for item in _DISPLAY_CACHE if item[0] == "reference"]
                while len(refs) > 1 and sum(_DISPLAY_CACHE[item][1] for item in refs) > limit // 2:
                    _DISPLAY_CACHE.pop(refs.pop(0))
                while sum(item[1] for item in _DISPLAY_CACHE.values()) > limit:
                    victim = next((item for item in _DISPLAY_CACHE if item[0] != "reference"),
                                  next(iter(_DISPLAY_CACHE)))
                    _DISPLAY_CACHE.pop(victim)
        return value


def clear_display_session_cache():
    """Discard presentation reuse between sessions/tests; never alter a source."""
    global _CACHE_GENERATION
    with _REFERENCE_LOCK:
        _CACHE_GENERATION += 1
        _REFERENCES.clear()
        _DISPLAY_CACHE.clear()
        _VERIFIED_ASSETS.clear()
        _NATIVE_STATES.clear()
        _FRAME_STATES.clear()
        _ASSET_PATHS.clear()
    from .phase_rig_display_prerender import clear_session_cache
    clear_session_cache()


def display_session_cache_info():
    with _REFERENCE_LOCK:
        return dict(verified_assets=len(_VERIFIED_ASSETS),
                    cache_bytes=sum(item[1] for item in _DISPLAY_CACHE.values()),
                    cache_limit_bytes=_cache_budget_bytes(), entries=len(_DISPLAY_CACHE),
                    resolved_paths=len(_ASSET_PATHS), key_locks=len(_KEY_LOCKS),
                    native_states=len(_NATIVE_STATES), frame_states=len(_FRAME_STATES))


def _native_frame_assets_cached(review_root, run, frame_index=None, *, reference=False):
    """Presentation-only reuse of fully checked manifest/receipt metadata.

    The native/source manifests are revalidated on a stamp change. Selected
    geometry and scientific receipts are verified once per version in this
    session. The strict scientific reader itself remains unchanged.
    """
    from . import phase_saved_review_metadata as runtime
    from .review_storage import resolve_review_path
    root = resolve_review_path(review_root)
    path = root / "native_frames" / str(run).casefold() / "manifest.json"
    key = (os.path.normcase(str(path)), str(run).casefold())
    with _key_lock(("metadata", key)):
        with _REFERENCE_LOCK:
            saved = _NATIVE_STATES.get(key)
            if saved is not None:
                _NATIVE_STATES.move_to_end(key)
            generation = _CACHE_GENERATION
        if saved is None and not path.is_file():
            # Keep standalone adapters and alternate injected readers supported.
            return native_frame_assets(review_root, run, frame_index, reference=reference)
        fresh = saved is not None and all(_stamp(p) == stamp for p, stamp in saved[1])
        if not fresh:
            state = runtime.load_native_source(root, run, verify_geometry=False)
            deps = [path, state["input_path"]]
            for name in ("parent_native_manifest", "camera_timeline_run"):
                deps.append(_verified_display_path(state["inputs"][name], base=state["input_path"].parent))
            saved = (state, [(p, _stamp(p)) for p in deps])
            with _REFERENCE_LOCK:
                if generation == _CACHE_GENERATION:
                    # Full-native metadata includes camera feature coordinates;
                    # sixteen recent runs keep those multi-MB documents bounded.
                    _remember(_NATIVE_STATES, key, saved, limit=16)
        state = saved[0]
        if reference:
            record, binding = state["native"]["reference"], state["inputs"]["reference"]
        else:
            rows = state["native"]["frame_sources"]
            if type(frame_index) is not int or not 0 <= frame_index < len(rows):
                raise ValueError("Rig/camera frame is outside its timeline")
            record, binding = rows[frame_index], state["inputs"]["frames"][frame_index]
        receipts = [_verified_display_identity(binding["geometry_receipt"], base=state["input_path"].parent)[1],
                    _verified_display_identity(record["full_native_scientific"]["receipt"])[1]]
        check_key = _token(dict(record=record, binding=binding,
                               stamps=receipts))
        with _REFERENCE_LOCK:
            checked = check_key in _FRAME_STATES
        if not checked:
            runtime._saved_geometry(binding, record, base=state["input_path"].parent, reference=reference)
            runtime._saved_scientific_receipt(record)
            with _REFERENCE_LOCK:
                if generation == _CACHE_GENERATION:
                    _remember(_FRAME_STATES, check_key, True)
        return state, record, binding


def rig_raw_display_available(manifest, *, frame_index=None, reference=False):
    """Inspect declarations only; no scientific reads on the GUI thread."""
    if reference:
        record = manifest.get("reference") or {}
    else:
        rows = manifest.get("frame_sources") or []
        if type(frame_index) is not int or not 0 <= frame_index < len(rows):
            return False
        record = rows[frame_index]
    extended = record.get("full_extended_scientific") or {}
    revision = record.get("rig_domain_revision") or {}
    return (extended.get("contract") == EXTENDED_SCIENTIFIC_CONTRACT
            and extended.get("complete") is True
            and revision.get("contract") == "ffac-rig-domain-revision-v1"
            and extended.get("phase_shape_hw") == revision.get("phase_shape_hw")
            and extended.get("source_sha256") == record.get("source_sha256"))


def _asset_array(asset):
    return np.load(_verified_display_path(asset), mmap_mode="r", allow_pickle=False)


def compose_display_raw(record, manifest):
    """Read exact centre/ring assets with an explicit scientific support mask."""
    descriptor = record["full_extended_scientific"]
    revision = record["rig_domain_revision"]
    shape = rig_phase_shape(revision)
    origin = tuple(revision["native_origin_xy"])
    if (descriptor.get("contract") != EXTENDED_SCIENTIFIC_CONTRACT
            or descriptor.get("complete") is not True
            or descriptor.get("storage") != "exact-native-centre-plus-disjoint-frozen-raw-ring-windows"
            or tuple(descriptor.get("phase_shape_hw", ())) != shape
            or tuple(descriptor.get("native_origin_xy", ())) != origin
            or descriptor.get("source_sha256") != record["source_sha256"]):
        raise ValueError("Extended RAW display belongs to another source/grid")
    centre = descriptor["native_centre"]
    if (centre.get("source_sha256") != record["source_sha256"]
            or centre["log_linear_rgb"] != record["full_native_scientific"]["log_linear_rgb"]
            or centre["valid_mask"] != record["full_native_scientific"]["valid_mask"]):
        raise ValueError("Extended RAW centre differs from its bound scientific source")
    x, y, width, height = tuple(manifest["content_box_xywh"])
    if (any(type(value) is not int for value in (x, y, width, height))
            or (x, y) != (-origin[0], -origin[1])
            or (height, width) != tuple(manifest["analysis_image_shape_hw"])
            or min(x, y) < 0 or x+width > shape[1] or y+height > shape[0]):
        raise ValueError("Extended RAW centre has another annotation origin")
    values, valid = _asset_array(centre["log_linear_rgb"]), _asset_array(centre["valid_mask"])
    if values.dtype != np.float32 or values.shape != (3, height, width) or valid.dtype != bool or valid.shape != (height, width):
        raise ValueError("Extended RAW centre has another native encoding")
    full = np.full((3, *shape), np.nan, np.float32)
    support = np.zeros(shape, bool)
    covered = np.zeros(shape, bool)
    full[:, y:y+height, x:x+width] = values
    support[y:y+height, x:x+width] = valid & np.isfinite(values).all(0)
    covered[y:y+height, x:x+width] = True
    for window in descriptor["ring_windows"]:
        a, b, c, d = window["canvas_xyxy"]
        if (any(type(value) is not int for value in (a, b, c, d))
                or not (0 <= a < c <= shape[1] and 0 <= b < d <= shape[0])
                or covered[b:d, a:c].any()):
            raise ValueError("Extended RAW windows overlap or leave their recorded display grid")
        values = _asset_array(window["log_linear_rgb"])
        if values.dtype != np.float32 or values.shape != (3, d-b, c-a):
            raise ValueError("Extended RAW window has another float32 native encoding")
        full[:, b:d, a:c] = values
        support[b:d, a:c] = np.isfinite(values).all(0)
        covered[b:d, a:c] = True
    full[:, ~support] = np.nan  # Transient display copy only.
    return full, support


def render_raw_detail(signal, support, *, tile_rows=128):
    """The existing campaign-wide log-channel transfer, rendered in tiles."""
    if signal.dtype != np.float32 or signal.shape != (3, *support.shape) or support.dtype != bool:
        raise ValueError("RAW detail requires native float32 RGB and bool support")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("tile_rows must be a positive integer")
    low, high = np.log(2.), np.log(65535.)
    output = np.empty((*support.shape, 3), np.uint8)
    for first in range(0, support.shape[0], tile_rows):
        last = min(first+tile_rows, support.shape[0])
        block = np.nan_to_num(signal[:, first:last], nan=low)
        normalized = np.clip((block-low)/(high-low), 0, 1)
        output[first:last] = np.rint(255*np.moveaxis(np.power(normalized, .75), 0, -1)).astype(np.uint8)
    output[~support] = BACKGROUND
    return output


def mark_missing_raw(rgb, domain, support):
    """Mark unavailable rig data without inventing a RAW colour or ratio."""
    if rgb.dtype != np.uint8 or rgb.shape != (*domain.shape, 3) or domain.dtype != bool or support.dtype != bool or support.shape != domain.shape:
        raise ValueError("Missing-data display flags require one native RGB/bool grid")
    missing = domain & ~support
    rgb[missing] = MISSING_DARK
    x = np.arange(domain.shape[1], dtype=np.int32)[None, :]
    y = np.arange(domain.shape[0], dtype=np.int32)[:, None]
    rgb[missing & (((x+y) % 16) < 4)] = MISSING_LIGHT
    return missing


def _prepared_reference(record, manifest):
    from .native.display_modes import DisplayFlatfieldReference, prepare_display_flatfield_reference
    from .native.config import require_parameters
    parameters = {"display_modes": require_parameters("display_modes"),
                  "change_display": require_parameters("change_display")}
    key = _raw_identity(record, manifest)
    def create():
        signal, support = compose_display_raw(record, manifest)
        prepared = prepare_display_flatfield_reference(signal, source_valid_mask=support,
            tile_rows=parameters["change_display"]["tile_rows"], parameters=parameters)
        # The old display renderer repairs tiny nonfinite pixels. A rig RAW
        # view must disclose those pixels as lacking source support instead.
        luminance = prepared.luminance.copy()
        luminance[~support] = 0
        luminance.setflags(write=False)
        support &= prepared.valid
        support.setflags(write=False)
        return DisplayFlatfieldReference(luminance, support, prepared.invalid_cleanup)
    return _cached("reference", key, create)


def load_rig_raw_display_rgb(review_root, run, frame_index=None, *, mode="raw_detail", reference=False,
                             use_prerender=True, backend="auto", cache_signal=False):
    """Display adapter; use bound saved photographs before optional live CUDA."""
    if mode not in ("raw_detail", "change"):
        raise ValueError("Extended rig RAW display supports RAW detail or Change only")
    state, record, binding = _native_frame_assets_cached(review_root, run, frame_index, reference=reference)
    manifest = state["native"]
    if not rig_raw_display_available(manifest, frame_index=frame_index, reference=reference):
        raise ValueError("Extended RAW display is unavailable for this source")
    if use_prerender:
        from .phase_rig_display_prerender import load_prerendered_rgb
        saved = load_prerendered_rgb(review_root, state, record, binding, run, frame_index,
                                    mode=mode, reference=reference)
        if saved is not None:
            return saved
    raw_key = _raw_identity(record, manifest)
    reference_key = _raw_identity(manifest["reference"], manifest) if mode == "change" else None
    revision = record["rig_domain_revision"]
    base = state["native_path"].parent
    _, classes_stamp = _asset_identity(revision["classes"], base=base)
    fallback_stamp = _asset_identity(revision["rig_domain"], base=base)[1] if revision.get("rig_domain") else None
    classes_key = _token(dict(revision=revision, classes=classes_stamp, fallback=fallback_stamp))
    key = _token(dict(contract=CONTRACT, raw=raw_key, reference_raw=reference_key, classes=classes_key,
                      binding=binding, run=run, frame_index=frame_index, reference=reference, mode=mode,
                      backend=backend))
    def render():
        return _render_rig_raw_display_rgb(state, record, binding, run, frame_index,
            mode=mode, reference=reference, reference_key=reference_key, classes_key=classes_key,
            backend=backend, raw_key=raw_key, cache_signal=cache_signal)
    rgb, masks, audit = _cached("render", key, render)
    # Keep callers' mutable presentation buffers separate from shared session data.
    return rgb.copy(), {name: value.copy() for name, value in masks.items()}, deepcopy(audit)


def _render_rig_raw_display_rgb(state, record, binding, run, frame_index, *, mode, reference,
                                reference_key, classes_key, backend="auto", raw_key=None, cache_signal=False):
    manifest = state["native"]
    if cache_signal:
        signal, current_support = _cached("source-raw", raw_key, lambda: compose_display_raw(record, manifest))
    else:
        signal, current_support = compose_display_raw(record, manifest)
    def classes():
        fields, asset = load_rig_classes(record["rig_domain_revision"], base_dir=state["native_path"].parent)
        return fields["rig_domain"], asset
    domain, classes_asset = _cached("domain", classes_key, classes)
    mode_support = current_support
    rendering = {}
    reference_sha256 = None
    if mode == "raw_detail":
        from .phase_rig_display_acceleration import raw_detail, backend_description
        rgb = raw_detail(signal, current_support, backend=backend)
        rgb[~current_support] = BACKGROUND
        backend_audit = backend_description() if backend != "cpu" else {"backend": "cpu"}
    else:
        from .native.display_modes import (
            register_display_flatfield_reference, render_change_flatfield, display_box_fully_valid)
        from .native.config import require_parameters
        parameters = {"display_modes": require_parameters("display_modes"),
                      "change_display": require_parameters("change_display")}
        prepared = _prepared_reference(manifest["reference"], manifest)
        reference_sha256 = manifest["reference"]["source_sha256"]
        homography = np.eye(3) if reference else np.asarray(binding["H_v5_B_to_frame_native"], np.float64)
        origin = tuple(record["rig_domain_revision"]["native_origin_xy"])
        source_origin = tuple(manifest["reference"]["rig_domain_revision"]["native_origin_xy"])
        registered_key = _token(dict(reference=reference_key, homography=homography.tolist(),
            shape=list(domain.shape), source_origin=source_origin, target_origin=origin))
        registered = _cached("registered", registered_key, lambda: register_display_flatfield_reference(
            prepared, homography, target_shape_hw=domain.shape,
            source_origin_xy=source_origin, target_origin_xy=origin, parameters=parameters))
        from .phase_rig_display_acceleration import render_lift1, backend_description
        lift1 = render_lift1(signal, domain, backend=backend)
        rgb, rendering = render_change_flatfield(signal, registered, domain, lift1_rgb=lift1,
            tile_rows=parameters["change_display"]["tile_rows"], parameters=parameters)
        backend_audit = backend_description() if backend != "cpu" else {"backend": "cpu"}
        mode_support = (display_box_fully_valid(current_support, parameters=parameters) & display_box_fully_valid(registered.valid, parameters=parameters)
                        & (registered.luminance_box5 >= parameters["change_display"]["minimum_reference_luminance"]))
    missing = mark_missing_raw(rgb, domain, mode_support)
    support = domain & mode_support
    extended_domain = domain.copy()
    x, y, width, height = manifest["content_box_xywh"]
    extended_domain[y:y+height, x:x+width] = False
    descriptor = record["full_extended_scientific"]
    audit = dict(contract=CONTRACT, mode=mode, run=run, frame_index=frame_index, reference=reference,
        source_sha256=record["source_sha256"], display_only=True, cpu_only=backend_audit["backend"] == "cpu",
        presentation_backend=backend_audit,
        source_extended_raw=dict(contract=descriptor["contract"],
            descriptor_sha256=sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            native_centre=descriptor["native_centre"], ring_windows=[dict(
                canvas_xyxy=row["canvas_xyxy"], log_linear_rgb=row["log_linear_rgb"])
                for row in descriptor["ring_windows"]]),
        reference_source_sha256=reference_sha256, rig_classes=classes_asset,
        phase_shape_hw=list(domain.shape), native_origin_xy=record["rig_domain_revision"]["native_origin_xy"],
        source_supported_rig_pixels=int((domain & current_support).sum()),
        mode_supported_rig_pixels=int(support.sum()), missing_rig_pixels=int(missing.sum()),
        outside_native_rig_pixels=int(extended_domain.sum()),
        outside_native_source_supported_pixels=int((extended_domain & current_support).sum()),
        outside_native_mode_supported_pixels=int((extended_domain & support).sum()),
        outside_native_missing_pixels=int((extended_domain & missing).sum()),
        missing_data_presentation="neutral diagonal hatch; no categorical or colour extrapolation",
        phase_maps_written=False, scientific_sidecars_written=False, annotation_coordinates_changed=False,
        rendering=rendering)
    return rgb, dict(support=support, missing=missing, rig_domain=domain), audit


def load_rig_raw_display_image(review_root, run, frame_index=None, *, mode="raw_detail", reference=False):
    """Qt worker adapter; retain support metadata with the decoded photograph."""
    from PySide6.QtGui import QImage
    rgb, masks, audit = load_rig_raw_display_rgb(review_root, run, frame_index, mode=mode, reference=reference)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
    image._photo_sand_domain = masks["rig_domain"]
    image._raw_display_support_mask = masks["support"]
    image._raw_display_missing_mask = masks["missing"]
    image._display_support_audit = audit
    if not reference and type(frame_index) is int:
        from .phase_rig_display_prerender import prefetch_neighbours
        state, _, _ = _native_frame_assets_cached(review_root, run, frame_index)
        prefetch_neighbours(review_root, run, frame_index, mode=mode,
                            frame_count=len(state["native"]["frame_sources"]))
    return image


__all__ = ["CONTRACT", "rig_raw_display_available", "load_rig_raw_display_rgb", "load_rig_raw_display_image",
           "clear_display_session_cache", "display_session_cache_info"]
