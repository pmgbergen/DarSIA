"""Source-bound photograph appearance on the existing review grid, display only.

The JPEG of the exact RAW exposure already uses the calibrated camera-U
coordinates. Only its display RGB is mapped with existing measured matrices.
Ordinary image changes read only the JPEG. The explicit water-line reader
retains the original RAW reader's basis and proposal; the matching JPEG changes
only their display appearance. No registration is fitted, review changed or
water line confirmed here. Transfer review uses the complete original RAW
field rectified onto expanded camera-U, preserving the saved overlay coordinates.
"""
from copy import deepcopy
from collections import OrderedDict
import hashlib
import io
import os
from pathlib import Path
from threading import RLock

import numpy as np

from . import painted_review_comparison as comparison

MAX_CAMERA_JPEG_BYTES = 128 * 1024 * 1024
_TRANSFER_LAYOUTS = OrderedDict()
_TRANSFER_LAYOUT_LOCK = RLock()
_TRANSFER_LAYOUT_PID = os.getpid()


def _companion(raw):
    """Select only one exact same-directory/stem camera JPEG, never another shot."""
    raw = comparison._descriptor(raw)
    path = Path(raw["path"])
    found = {}
    for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            found[os.path.normcase(str(candidate.resolve()))] = candidate
    if len(found) != 1:
        raise ValueError("The selected exposure needs one same-stem camera JPG beside its RAW")
    jpeg = next(iter(found.values()))
    if jpeg.parent.resolve() != path.parent.resolve() or jpeg.stem.casefold() != path.stem.casefold():
        raise ValueError("The camera JPG belongs to another exposure")
    if jpeg.stat().st_size > MAX_CAMERA_JPEG_BYTES:
        raise ValueError("The selected camera JPG exceeds the display read bound")
    return comparison._bound(jpeg)


def _camera_rgb(raw, geometry, certificate):
    """Read stored JPEG pixel coordinates, matching original photograph_io U."""
    from PIL import Image
    from .rig_point_alignment import model_photo_path
    physical = dict(raw, path=str(model_photo_path(geometry, raw)))
    binding = _companion(physical)
    with Path(binding["path"]).open("rb") as stream:
        payload = stream.read(MAX_CAMERA_JPEG_BYTES + 1)
    if len(payload) > MAX_CAMERA_JPEG_BYTES:
        raise ValueError("The selected camera JPG exceeds the display read bound")
    if len(payload) != binding["size_bytes"] or hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        raise ValueError("The camera JPG changed while reading")
    conventions = comparison._camera_u_context(geometry, certificate)
    with Image.open(io.BytesIO(payload)) as photograph:
        if photograph.format not in {"JPEG", "MPO"}:
            raise ValueError("The camera JPG is not a supported JPEG container")
        photograph.seek(0)
        if list(photograph.size) != conventions["camera_wh"]:
            raise ValueError("The camera JPG differs from the existing camera-U grid")
        # Like photograph_io.load_reduced_rgb8, do not apply an EXIF transpose:
        # the measured U convention refers to stored camera JPEG pixel centres.
        rgb = np.array(photograph.convert("RGB"), copy=True)
    comparison._checked(binding)
    return rgb, binding


def _mapped(row, geometry, certificate, raw, transform):
    rgb, binding = _camera_rgb(raw, geometry, certificate)
    image, support = comparison._display_warp(rgb, transform, row["gap"]["shape_hw"])
    comparison._checked(binding)
    return dict(image_rgb=image, image_encoding="rgb_uint8", valid_support=support,
                native_to_photo_h=np.eye(3).tolist(), display_grid=comparison._grid(row),
                camera_jpeg=deepcopy(binding), source_raw=deepcopy(raw),
                display_transform=np.asarray(transform).tolist(), camera_jpg_display_only=True)


def read_gap_preview(review, run_id, component_id=None):
    """Keep original gap/overlay admission and substitute only display RGB."""
    review = comparison._review(review)
    rows = [row for row in review["runs"] if row["run_id"] == run_id]
    if len(rows) != 1:
        raise ValueError("The selected run is not in this exact reviewed cohort")
    row = rows[0]
    geometry, native, certificate = comparison._model(row)
    mapped = _mapped(row, geometry, certificate, comparison._descriptor(geometry.reference_image), native)
    packet = comparison.read_gap_preview(review, run_id, component_id,
                                         display_rgb=mapped["image_rgb"])
    packet.update(mapped)
    packet["image_role"] = "current"
    return packet


def _transfer_canvas(models):
    """One native-scale display extent containing every complete photograph.

    Only the display origin changes. Every run keeps its original measured
    camera-to-native transform and all native masks keep their coordinates.
    """
    points, largest = [], 0
    for geometry, native, row in models:
        width, height = geometry.model['camera_profile']['camera_wh']
        corners = np.array([[0., 0., 1.], [width-1., 0., 1.],
                            [width-1., height-1., 1.], [0., height-1., 1.]])
        projected = corners @ np.asarray(native, np.float64).T
        denominator = projected[:, 2]
        if (not np.isfinite(projected).all() or np.any(np.abs(denominator) < 1e-12)
                or not (np.all(denominator > 0) or np.all(denominator < 0))):
            raise ValueError('The complete photograph crosses its display projection horizon')
        points.append(projected[:, :2] / denominator[:, None])
        native_height, native_width = row['gap']['shape_hw']
        points.append(np.array([[0., 0.], [native_width-1., native_height-1.]]))
        if 'own_frame_native_xy' in row:
            frame = np.asarray(row['own_frame_native_xy'], np.float64)
            if frame.shape != (4, 2) or not np.isfinite(frame).all():
                raise ValueError('The transfer display needs its exact finite rig frame')
            points.append(frame)
        largest = max(largest, width*height, native_width*native_height)
    if not points:
        raise ValueError('Choose a saved transfer review with photographs')
    all_points = np.concatenate(points)
    padding = max(2, int(np.ceil(np.sqrt(largest) * .01)))
    low = np.floor(all_points.min(axis=0)).astype(np.int64)
    high = np.ceil(all_points.max(axis=0)).astype(np.int64)
    width, height = (high-low+1).tolist()
    if width < 1 or height < 1 or width*height > largest*4:
        raise ValueError('The complete photographs exceed their bounded common display extent')
    low -= padding
    high += padding
    width, height = (high-low+1).tolist()
    offset = np.array([[1., 0., -float(low[0])],
                       [0., 1., -float(low[1])], [0., 0., 1.]])
    return (height, width), offset


def _review_crop_xywh(models, camera_u_to_photo_h):
    """One display-only viewport containing every cohort frame at a fixed scale."""
    points = []
    for _, native, row in models:
        frame = np.asarray(row['own_frame_native_xy'], np.float64)
        if frame.shape != (4, 2) or not np.isfinite(frame).all():
            raise ValueError('The fixed rig viewport needs each exact saved rig frame')
        transform = np.asarray(camera_u_to_photo_h) @ np.linalg.inv(native)
        projected = np.column_stack((frame, np.ones(4))) @ transform.T
        denominator = projected[:, 2]
        if (not np.isfinite(projected).all() or np.any(np.abs(denominator) < 1e-12)
                or not (np.all(denominator > 0) or np.all(denominator < 0))):
            raise ValueError('The saved rig frame crosses its display projection horizon')
        points.append(projected[:, :2] / denominator[:, None])
    points = np.concatenate(points)
    low, high = points.min(axis=0), points.max(axis=0)
    margin = max(8, int(np.ceil(np.max(high-low) * .02)))
    low = np.floor(low).astype(np.int64) - margin
    high = np.ceil(high).astype(np.int64) + margin
    return [int(low[0]), int(low[1]), int(high[0]-low[0]+1), int(high[1]-low[1]+1)]


def _layout_point_inputs(models, files):
    """Capture the exact alignment metadata and photo versions used by models."""
    from . import rig_point_alignment as alignment
    from . import rig_point_review as points
    from .map_placement import _model_path
    seen, photographs = set(), []
    for geometry, _, _ in models:
        bound = (getattr(geometry, 'entry', None) or {}).get(alignment.ENTRY_KEY)
        key = None if bound is None else (bound['path'], bound['sha256'], bound['size_bytes'])
        if bound is None or key in seen:
            continue
        seen.add(key)
        package, base, _ = alignment._load_package(bound)
        files.read(bound['path'], limit=alignment.MAX_BYTES)
        files.read(package['base_index']['path'], limit=alignment.MAX_BYTES)
        for run, entry in alignment._entries(base):
            path = _model_path(Path(package['base_index']['path']), run, entry.get('reference_model'))
            files.read(path, limit=alignment.MAX_BYTES)
        for run in package['reviewed_runs']:
            photo = package['review']['runs'][run]['photo']
            photographs.append(points.photo_binding(photo['path'], photo.get('image_size')))
    return photographs


def _layout_photos_unchanged(photographs):
    from .rig_point_review import photo_binding
    return all(photo_binding(value['path'], value.get('image_size')) == value for value in photographs)


def _transfer_layout(review):
    """Reuse at most two exact all-run layouts, never RAW or mask arrays."""
    global _TRANSFER_LAYOUT_PID
    from .painted_adoption_table import _FILES
    from .native.raw_full_display import mapping_from_parameters, full_raw_display_extent
    from .native.raw_photograph import _parameters
    key = comparison.display_context_digest(review)
    cacheable = isinstance(review.get('review_binding'), dict) and not _FILES.requires_fresh()
    with _TRANSFER_LAYOUT_LOCK:
        if _TRANSFER_LAYOUT_PID != os.getpid():
            _TRANSFER_LAYOUTS.clear()
            _TRANSFER_LAYOUT_PID = os.getpid()
        cached = _TRANSFER_LAYOUTS.get(key) if cacheable else None
    if cached is not None:
        if _FILES.unchanged(cached['dependencies']) and _layout_photos_unchanged(cached['photographs']):
            return deepcopy(cached['models']), cached['extent']
        with _TRANSFER_LAYOUT_LOCK:
            _TRANSFER_LAYOUTS.pop(key, None)
    with _FILES.capture() as dependencies:
        models, mappings = [], {}
        for row in review['runs']:
            geometry, native, _ = comparison._model(row)
            models.append((geometry, native, row))
            profile = geometry.model['camera_profile']
            profile_key = (profile['path'], profile['sha256'], profile['profile_id'])
            if profile_key not in mappings:
                parameters = _parameters(*profile_key)
                mappings[profile_key] = mapping_from_parameters(parameters, profile['profile_id'])
        extent = full_raw_display_extent(list(mappings.values()))
        photographs = _layout_point_inputs(models, _FILES)
    if cacheable:
        if not _FILES.unchanged(dependencies) or not _layout_photos_unchanged(photographs):
            raise ValueError('The exact transfer display inputs changed while preparing its layout')
        with _TRANSFER_LAYOUT_LOCK:
            _TRANSFER_LAYOUTS[key] = dict(models=deepcopy(models), extent=extent,
                dependencies=dependencies, photographs=photographs)
            while len(_TRANSFER_LAYOUTS) > 2:
                _TRANSFER_LAYOUTS.popitem(last=False)
    return models, extent


def read_transfer_preview(review, run_id):
    """Show the complete original RAW field on one common expanded camera-U grid.

    The already calibrated G lens model only rectifies display RGB. Saved
    native masks and rig corners use their unchanged native-to-U homography.
    """
    from .native.raw_full_display import read_full_raw_display
    from .painted_analysis_exclusions import _read_admitted_gap_preview
    review = comparison._review(review)
    rows = [row for row in review['runs'] if row['run_id'] == run_id]
    if len(rows) != 1:
        raise ValueError('The selected run is absent from this transfer review')
    models, extent = _transfer_layout(review)
    row = rows[0]
    geometry, native, _ = next(value for value in models if value[2]['run_id'] == run_id)
    certificate = comparison._checked(geometry.model['baseline_native_grid']['renderer'], metadata=True)
    # Keep the ordinary exact native-photo/gap admission. No JPEG supplies FOV.
    packet = _read_admitted_gap_preview(review, run_id, include_native_rgb=False)
    raw = comparison._descriptor(geometry.reference_image)
    profile = geometry.model['camera_profile']
    conventions = comparison._camera_u_context(geometry, certificate)
    from .rig_point_alignment import model_photo_path
    display = read_full_raw_display(model_photo_path(geometry), profile_path=profile['path'],
        profile_sha256=profile['sha256'], profile_id=profile['profile_id'],
        expected_sha256=raw['sha256'], expected_orientation=conventions['raw_orientation'], extent=extent)
    translation = np.asarray(display['camera_u_to_photo_h'], np.float64)
    native_to_photo = translation @ np.linalg.inv(native)
    crop = _review_crop_xywh(models, translation)
    packet.update(image_rgb=display['image_rgb'], image_encoding='rgb_uint8', image_role='current',
        valid_support=display['valid_support'], native_to_photo_h=native_to_photo.tolist(),
        display_transform=translation.tolist(), camera_u_to_photo_h=translation.tolist(),
        source_raw=deepcopy(raw), source_profile=deepcopy(profile), camera_jpg_display_only=False,
        full_camera_photograph=True, full_raw_sensor_field=True,
        raw_sensor_wh=display['sensor_wh'], camera_u_wh=display['camera_u_wh'],
        expanded_u_extent_xywh=display['extent_u_xywh'],
        review_crop_xywh=crop,
        display_grid=dict(format='darsia-transfer-full-raw-camera-u-display-grid-v1',
            shape_hw=list(display['image_rgb'].shape[:2]), native_to_photo_h=native_to_photo.tolist(),
            camera_u_to_photo_h=translation.tolist(),
            review_crop_xywh=crop,
            runs=[item['run_id'] for _, _, item in models],
            scope='display only; complete original RAW field, saved camera-U/native coordinates unchanged'))
    comparison._model(row)
    return packet


def read_target_paint(review, run_id, component_id):
    """Keep the original target masks and basis with verified JPEG appearance."""
    from .painted_analysis_exclusions import read_target_paint as original
    return original(review, run_id, component_id, preview_reader=read_gap_preview)


def _water_surface_on_jpg(review, row, packet, original):
    """Join only the original unconfirmed proposal on this exact native plane."""
    water = original.get("water_surface")
    if (original.get("image_role") != "clearwater" or not isinstance(water, dict)
            or not comparison._same(original.get("source_raw"), packet["source_raw"])
            or not comparison._same((water.get("basis") or {}).get("photo"), packet["source_raw"])):
        raise ValueError("The water-line proposal and camera JPG need the exact same selected exposure")
    grid = comparison._grid(row)
    shape = tuple(row["gap"]["shape_hw"])
    for value in (packet, original):
        for key, dtype, expected_shape in (("image_rgb", np.uint8, (*shape, 3)),
                                           ("valid_support", bool, shape)):
            array = value.get(key)
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype) or array.shape != expected_shape:
                raise ValueError("The water-line proposal and camera JPG need the exact same native display grid")
    if (not comparison._same(packet["display_grid"], grid)
            or not comparison._same(original.get("display_grid"), grid)
            or not comparison._same(water.get("display_grid"), grid)
            or not comparison._same(original.get("native_to_photo_h"), packet["native_to_photo_h"])
            or not comparison._same(water.get("photo_to_display_h"), packet["native_to_photo_h"])
            or not comparison._same(original.get("display_transform"), packet["display_transform"])
            or not np.array_equal(original.get("valid_support"), packet["valid_support"])):
        raise ValueError("The water-line proposal and camera JPG need the exact same native display grid")
    if (not comparison._same(water.get("basis"), comparison.expected_water_surface_basis(review, row["run_id"]))
            or water.get("action") is not None or water.get("measurements") is not None):
        raise ValueError("Keep the original current water-line basis and unconfirmed proposal")
    return deepcopy(water)


def read_comparison_preview(review, run_id, component_id, image_id, *, include_water_surface=False):
    """Lazy one-exposure display; original role/series/model rules remain active."""
    review = comparison._review(review)
    context, rows, row = comparison._state(review, run_id)
    target, native, certificate = comparison._model(row)
    if image_id in {"current", "first-experiment"}:
        if image_id == "first-experiment":
            comparison._first_role(context, review, row, target, certificate)
        packet = _mapped(row, target, certificate, comparison._descriptor(target.reference_image), native)
    elif image_id == "clearwater":
        selection = comparison._clear_role(context, row)
        packet = _mapped(row, target, certificate, selection["selected_raw"], native)
        if selection.get("camera_jpeg") is not None:
            from .clearwater_source_choice import _same_asset
            if not _same_asset(packet["camera_jpeg"], selection["camera_jpeg"]):
                raise ValueError("The recorded AC45 companion JPG changed")
        packet["water_surface"] = None
        if include_water_surface:
            original = comparison.read_comparison_preview(review, run_id, component_id, image_id)
            packet["water_surface"] = _water_surface_on_jpg(review, row, packet, original)
    elif image_id == "previous-run":
        order = comparison._order(context["declared_run_order"])
        index = order.index(run_id)
        if index == 0 or order[index - 1] not in rows:
            raise ValueError("The immediate declared previous run has no accepted review; no run is skipped")
        previous = rows[order[index - 1]]
        source, _source_native, source_certificate = comparison._model(previous)
        if not comparison._profile_equal(target, certificate, source, source_certificate):
            raise ValueError("Previous and current photographs have different exact camera-U profiles")
        transform = native @ target.metric_to_grid @ np.linalg.inv(source.metric_to_grid)
        packet = _mapped(row, source, source_certificate, comparison._descriptor(source.reference_image), transform)
        packet["source_run_id"] = previous["run_id"]
        comparison._model(previous)
    else:
        raise ValueError("The requested camera JPG role is not supported")
    comparison.validate_display_context(context, review)
    comparison._model(row)
    packet["image_role"] = image_id
    return packet


def read_water_surface_comparison_preview(review, run_id, component_id, image_id):
    """Request the original RAW water basis only for the measurement step."""
    return read_comparison_preview(review, run_id, component_id, image_id,
                                   include_water_surface=True)
