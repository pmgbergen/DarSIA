"""Frozen, source-bound camera radiometry with explicit configuration.

Image equations are lifted from the original source processor without
refitting calibration or changing the registered native domain.
"""
from __future__ import annotations
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Any
import cv2
import numpy as np
from .io import sha256_file, resolve_asset_path
from . import photometry as obs
from .photometry import correct_frame_native
from .registration import registered_edge_aware_sensor_linear, register_chart
from .support import source_photo_grid, project_source_photo_safe_core, photo_safe_core_policy_metadata, fill_isolated_invalid, invalid_policy_contract as isolated_invalid_contract, validate_invalid_policy as validate_isolated_invalid_policy
from .display_modes import photographic_display, photographic_contract, validate_display_mode, render_display_modes, display_contract as modes_contract
from .sidecar import CorrectedFrame, raw_detail_display

def _path(value, *, parameters):
    return resolve_asset_path(value, parameters['source']['asset_aliases'])

def verify_execution_identity(actual: Mapping, expected: Mapping, required_keys) -> None:
    """Require each configured executable identity to be present and exact.

    Historical producer hashes remain provenance. This binding describes
    the current implementation and cannot accept absent or null hashes.
    """
    if (
        not required_keys
        or len(required_keys) != len(set(required_keys))
        or set(expected) != set(required_keys)
        or any((not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None for value in expected.values()))
    ):
        raise ValueError('An exact current source execution identity is required')
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError('Current RAW preprocessing code differs from its execution binding: ' + key)

def _marker_policy(marker: dict[str, Any], *, parameters: Mapping) -> str:
    """Preserve the source-bound camera processing contract with explicit parameters."""
    policy = marker.get('enclosed_hole_policy', parameters['source']['policy_ids']['keep'])
    expected = photo_safe_core_policy_metadata(policy, parameters=parameters['support'])
    for key, value in expected.items():
        if key in marker and marker[key] != value:
            raise ValueError('Track1 photo cohort support policy metadata changed')
    if policy != parameters['source']['policy_ids']['keep'] and (not all((key in marker for key in expected))):
        raise ValueError('Track1 filled photo cohort lacks its explicit support contract')
    return policy

def _read(path: Path, *, parameters: Mapping) -> dict[str, Any]:
    """Preserve the source-bound camera processing contract with explicit parameters."""
    return json.loads(path.read_text(encoding='utf-8-sig'))

def _check(path: Any, digest: Any, label: str, *, parameters: Mapping) -> Path:
    """Preserve the source-bound camera processing contract with explicit parameters."""
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f'Track1 {label} lacks exact source path/SHA')
    source = _path(path, parameters=parameters)
    if not source.is_file() or sha256_file(source) != digest:
        raise ValueError(f'Track1 {label} changed: {source}')
    return source

def _same_path(first: Any, second: Any, *, parameters: Mapping) -> bool:
    """Preserve the source-bound camera processing contract with explicit parameters."""
    return _path(str(first), parameters=parameters).resolve() == _path(str(second), parameters=parameters).resolve()

def _same_h(actual: np.ndarray, expected: np.ndarray, *, parameters: Mapping) -> bool:
    """Preserve the source-bound camera processing contract with explicit parameters."""
    return bool(np.allclose(actual, expected, rtol=0.0, atol=parameters['source']['homography_match_atol']))

class PhotoCohort:
    """Verify each acquisition and its recorded temporal camera transform."""

    def __init__(self, native_manifest_path: str | Path, *, parameters: Mapping) -> None:
        """Preserve the source-bound camera processing contract with explicit parameters."""
        self.parameters = parameters
        native_path = _path(native_manifest_path, parameters=self.parameters)
        native = _read(native_path, parameters=self.parameters)
        marker = native.get('v5_track1_photo_geometry') or {}
        if marker.get('contract') != self.parameters['source']['cohort_contract']:
            raise ValueError('Track1 native manifest lacks its photo-only cohort')
        self.enclosed_hole_policy = _marker_policy(marker, parameters=self.parameters)
        source_path = _check(
            marker.get('original_source_manifest_path'),
            marker.get('original_source_manifest_sha256'),
            'original native source manifest',
            parameters=self.parameters,
        )
        source = _read(source_path, parameters=self.parameters)
        count = source.get('frame_count')
        rows = marker.get('frames')
        if (
            native.get('run') != source.get('run')
            or native.get('frame_count') != count
            or tuple(source.get('analysis_image_shape_hw') or ()) != tuple(self.parameters['source']['native_shape_hw'])
            or (native.get('v5_photometry') or {}).get('source_manifest_sha256') != marker['original_source_manifest_sha256']
            or not isinstance(rows, list)
            or len(rows) != count
            or [row.get('frame_index') for row in rows] != list(range(count))
        ):
            raise ValueError('Track1 photo cohort does not match its native revision')
        self.native_manifest_path = native_path
        self.native_manifest_sha256 = sha256_file(_path(native_path, parameters=self.parameters))
        self.source_manifest_path = source_path
        self.source_manifest_sha256 = marker['original_source_manifest_sha256']
        self.source = source
        self.rows = rows
        self.shape_hw = tuple(self.parameters['source']['native_shape_hw'])

    def frame(self, index: int) -> dict[str, Any]:
        """Live-hash the receipt and exact frame/photo sources before use."""
        if not isinstance(index, int) or not 0 <= index < len(self.rows):
            raise ValueError('Track1 frame index is outside its source timeline')
        row = self.rows[index]
        receipt_path = _check(row.get('receipt_path'), row.get('receipt_sha256'), 'frame transport receipt', parameters=self.parameters)
        receipt = _read(receipt_path, parameters=self.parameters)
        original = self.source['frame_sources'][index]
        native_source = _path(original.get('frame_path') or self.source_manifest_path.parent / f'frame_{index:03d}.jpg', parameters=self.parameters)
        if (
            receipt.get('contract') != self.parameters['source']['photo_contract']
            or receipt.get('original_source_manifest_sha256') != self.source_manifest_sha256
            or not _same_path(receipt.get('original_source_manifest_path'), self.source_manifest_path, parameters=self.parameters)
            or receipt.get('frame_index') != index
            or receipt.get('native_shape_hw') != list(self.shape_hw)
            or receipt.get('validation_only') is not True
            or receipt.get('mass_eligible') is not False
            or receipt.get('photo_sand_transport_qa_pass') is not True
            or receipt.get('temporal_registration_pass') is not True
            or receipt.get('optical_geometry_qa_pass') is not True
            or receipt.get('photo_sand_qa_scope') != self.parameters['source']['photo_scope']
            or receipt.get('photo_sand_qa_pass') is not True
            or receipt.get('fixed_native_mask_erosion_px') != self.parameters['source']['cell_scale']
            or receipt.get('training_label_eligibility_certified') is not False
            or receipt.get('phase_map_eligibility_certified') is not False
            or receipt.get('rig_metal_certified') is not False
            or receipt.get('dynamic_overflow_certified') is not False
            or not _same_path(receipt.get('native_review_jpeg_path'), native_source, parameters=self.parameters)
            or receipt.get('native_review_jpeg_sha256') != (original.get('native_jpeg_sha256') or original.get('frame_sha256'))
        ):
            raise ValueError('Track1 frame receipt or source revision is inconsistent')
        if not (self.source['run'] == self.parameters['source']['symbols']['prepared_source_id'] and index == 0):
            if (
                not _same_path(receipt.get('camera_jpeg_path'), original.get('source_path'), parameters=self.parameters)
                or receipt.get('camera_jpeg_sha256') != (original.get('source_sha256') or original.get('source_raw_sha256'))
            ):
                raise ValueError('Track1 camera JPEG differs from exact source frame')
        else:
            _check(original.get('source_path'), original.get('source_sha256') or original.get('source_raw_sha256'), self.parameters['source']['symbols']['prepared_source_label'], parameters=self.parameters)
        _check(receipt['native_review_jpeg_path'], receipt['native_review_jpeg_sha256'], 'native review frame', parameters=self.parameters)
        _check(receipt.get('camera_jpeg_path'), receipt.get('camera_jpeg_sha256'), 'camera JPEG acquisition', parameters=self.parameters)
        _check(receipt.get('v5_B_native_jpeg_path'), receipt.get('v5_B_native_jpeg_sha256'), 'V5 B native review', parameters=self.parameters)
        _check(receipt.get('v5_B_registration_certificate_path'), receipt.get('v5_B_registration_certificate_sha256'), 'RAW-to-B registration certificate', parameters=self.parameters)
        _check(receipt.get('v5_B_arw_path'), receipt.get('v5_B_arw_sha256'), 'V5 B RAW acquisition', parameters=self.parameters)
        _check(receipt.get('v5_B_camera_jpeg_path'), receipt.get('v5_B_camera_jpeg_sha256'), 'V5 B camera JPEG', parameters=self.parameters)
        temporal_path = _check(
            receipt.get('temporal_geometry_receipt_path'),
            receipt.get('temporal_geometry_receipt_sha256'),
            'temporal source registration',
            parameters=self.parameters,
        )
        temporal = _read(temporal_path, parameters=self.parameters)
        frame_evidence = temporal.get('frames') or []
        if (
            index >= len(frame_evidence)
            or frame_evidence[index].get('frame_index') != index
            or not _same_h(normalize_homography(frame_evidence[index].get('H_timeline0_to_frame_native'), parameters=self.parameters), normalize_homography(receipt.get('H_timeline0_to_frame_native'), parameters=self.parameters), parameters=self.parameters)
        ):
            raise ValueError('Track1 temporal receipt differs from this frame H')
        if frame_evidence[index].get('temporal_registration_pass') is not True:
            exception_path = _check(
                receipt.get(self.parameters['source']['symbols']['temporal_exception_path_key']),
                receipt.get(self.parameters['source']['symbols']['temporal_exception_sha_key']),
                self.parameters['source']['symbols']['temporal_exception_label'],
                parameters=self.parameters,
            )
            exception = _read(exception_path, parameters=self.parameters)
            if (
                self.source.get('run') != self.parameters['source']['symbols']['temporal_exception_source_id']
                or index != self.parameters['source']['temporal_exception_frame_index']
                or exception.get('contract') != self.parameters['source']['symbols']['temporal_exception_contract']
                or exception.get('source_manifest_sha256') != self.source_manifest_sha256
                or exception.get('original_camera_frame_jpeg_sha256') != receipt['camera_jpeg_sha256']
                or exception.get('frame_native_sha256') != receipt['native_review_jpeg_sha256']
                or exception.get('original_primary_gate_pass') is not False
                or exception.get('fallback_decision_pass') is not True
                or exception.get('validation_only') is not True
                or exception.get('mass_eligible') is not False
                or not _same_h(normalize_homography(exception.get('selected_H_timeline0_to_frame_native'), parameters=self.parameters), normalize_homography(receipt['H_timeline0_to_frame_native'], parameters=self.parameters), parameters=self.parameters)
            ):
                raise ValueError('Track1 temporal exception is absent or mis-scoped')
        photo_path = _check(
            receipt.get('photo_sand_reference_receipt_path'),
            receipt.get('photo_sand_reference_receipt_sha256'),
            'B-photo source acceptance',
            parameters=self.parameters,
        )
        photo = _read(photo_path, parameters=self.parameters)
        mask_path = _check(
            receipt.get('photo_sand_reference_mask_path'),
            receipt.get('photo_sand_reference_mask_sha256'),
            'B-photo source mask',
            parameters=self.parameters,
        )
        if (
            photo.get('contract') != self.parameters['source']['photo_core_contract']
            or photo.get('source_manifest_sha256') != self.source_manifest_sha256
            or photo.get('B_native_review_jpeg_sha256') != receipt['v5_B_native_jpeg_sha256']
            or photo.get('B_arw_sha256') != receipt['v5_B_arw_sha256']
            or photo.get('B_camera_jpeg_sha256') != receipt['v5_B_camera_jpeg_sha256']
            or photo.get('B_registration_certificate_sha256') != receipt['v5_B_registration_certificate_sha256']
            or photo.get('mask_npz_sha256') != receipt['photo_sand_reference_mask_sha256']
            or not _same_path(photo.get('mask_npz_path'), mask_path, parameters=self.parameters)
            or photo.get('photo_gate_pass') is not True
            or photo.get('photo_sand_qa_scope') != self.parameters['source']['photo_scope']
            or photo.get('per_frame_transport_qa_pass') is not False
            or photo.get('training_label_eligibility_certified') is not False
            or photo.get('phase_map_eligibility_certified') is not False
            or photo.get('rig_metal_certified') is not False
            or photo.get('dynamic_overflow_certified') is not False
            or photo.get('mass_eligible') is not False
        ):
            raise ValueError('Track1 B-photo acceptance was promoted beyond its source scope')
        h_time = normalize_homography(receipt.get('H_timeline0_to_frame_native'), parameters=self.parameters)
        h_b0 = normalize_homography(receipt.get('H_v5_B_to_timeline0_native'), parameters=self.parameters)
        h_bf = normalize_homography(receipt.get('H_v5_B_to_frame_native'), parameters=self.parameters)
        h_ref0 = normalize_homography(receipt.get(self.parameters['source']['symbols']['reference_to_timeline_h_key']), parameters=self.parameters)
        h_reff = normalize_homography(receipt.get(self.parameters['source']['symbols']['reference_to_frame_h_key']), parameters=self.parameters)
        if (
            not _same_h(h_bf, normalize_homography(h_time @ h_b0, parameters=self.parameters), parameters=self.parameters)
            or not _same_h(h_reff, normalize_homography(h_time @ h_ref0, parameters=self.parameters), parameters=self.parameters)
        ):
            raise ValueError('Track1 per-frame B/photo homography composition changed')
        receipt['receipt_path'] = str(receipt_path)
        receipt['receipt_sha256'] = row['receipt_sha256']
        receipt.update(photo_safe_core_policy_metadata(self.enclosed_hole_policy, parameters=self.parameters['support']))
        return receipt

    def photo(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return the manifest's explicit B-sand support and its complement."""
        receipt = self.frame(index)
        sand = source_photo_grid(
            _path(receipt['photo_sand_reference_mask_path'], parameters=self.parameters),
            receipt['photo_sand_reference_mask_sha256'],
            parameters=self.parameters['support'],
        )
        (safe, support) = project_source_photo_safe_core(
            sand,
            receipt[self.parameters['source']['symbols']['reference_to_frame_h_key']],
            shape_hw=self.shape_hw,
            enclosed_hole_policy=self.enclosed_hole_policy,
            return_metadata=True,
            parameters=self.parameters['support'],
            guard_px=self.parameters['source']['mask_guard_px'],
        )
        support['photo_sand_support_mask_sha256'] = sha256(np.packbits(safe, bitorder='little').tobytes()).hexdigest()
        support['source_transport_receipt_sha256'] = receipt['receipt_sha256']
        support['native_manifest_sha256'] = self.native_manifest_sha256
        support['frame_index'] = index
        receipt.update(support)
        receipt['photo_sand_support_receipt_sha256'] = sha256(json.dumps(support, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return (safe, ~safe, receipt)

    def baseline_to_frame(self, index: int) -> np.ndarray:
        """Return the recorded baseline-to-frame transform after receipt validation."""
        return normalize_homography(self.frame(index)['H_v5_B_to_frame_native'], parameters=self.parameters)

    def previous_to_frame(self, previous_index: int, index: int) -> np.ndarray:
        """Compose the current baseline transform with the inverse previous transform, without measuring new registration."""
        return normalize_homography(self.baseline_to_frame(index) @ np.linalg.inv(self.baseline_to_frame(previous_index)), parameters=self.parameters)

def bind_photo_marker(source_manifest_path: str | Path, row: dict[str, Any], *, enclosed_hole_policy: str=None, parameters: Mapping) -> dict[str, Any]:
    """Validate a complete source timeline before radiometry. Every acquisition, baseline and temporal receipt remains content-bound; the returned marker does not promote its scientific eligibility."""
    enclosed_hole_policy = parameters['source']['default_policies']['enclosed_hole_policy'] if enclosed_hole_policy is None else enclosed_hole_policy
    source_path = _path(source_manifest_path, parameters=parameters)
    policy = photo_safe_core_policy_metadata(enclosed_hole_policy, parameters=parameters['support'])
    source_sha = sha256_file(source_path)
    source = _read(source_path, parameters=parameters)
    frames = row.get('frames')
    if (
        row.get('source_manifest_sha256') != source_sha
        or not _same_path(row.get('source_manifest_path'), source_path, parameters=parameters)
        or row.get('run') != source.get('run')
        or not isinstance(frames, list)
        or len(frames) != int(source.get('frame_count') or -1)
        or [r.get('frame_index') for r in frames] != list(range(len(frames)))
    ):
        raise ValueError('Track1 photo aggregate does not cover this exact source revision')
    probe = PhotoCohort.__new__(PhotoCohort)
    probe.parameters = parameters
    probe.source_manifest_path = source_path
    probe.source_manifest_sha256 = source_sha
    probe.source = source
    probe.rows = frames
    probe.shape_hw = tuple(source['analysis_image_shape_hw'])
    probe.enclosed_hole_policy = enclosed_hole_policy
    for index in range(len(frames)):
        probe.frame(index)
    marker = dict(
        contract=parameters['source']['cohort_contract'],
        original_source_manifest_path=str(source_path),
        original_source_manifest_sha256=source_sha,
        frames=[dict(frame_index=index, receipt_path=str(item['receipt_path']), receipt_sha256=str(item['receipt_sha256'])) for index, item in enumerate(frames)],
    )
    if enclosed_hole_policy != parameters['source']['policy_ids']['keep']:
        marker.update(policy)
    return marker

def _validate_output_policy(enclosed_hole_policy: str, display_mode: str, isolated_invalid_policy: str=None, *, parameters: Mapping) -> None:
    """Preserve the source-bound camera processing contract with explicit parameters."""
    isolated_invalid_policy = parameters['source']['default_policies']['isolated_invalid_policy'] if isolated_invalid_policy is None else isolated_invalid_policy
    if enclosed_hole_policy not in (parameters['source']['policy_ids']['keep'], parameters['source']['policy_ids']['filled_support']):
        raise ValueError(f'Unknown Track1 enclosed-hole policy: {enclosed_hole_policy!r}')
    if display_mode != parameters['source']['policy_ids']['natural_display']:
        validate_display_mode(display_mode, parameters=parameters['photographic_display'])
    validate_isolated_invalid_policy(isolated_invalid_policy, parameters=parameters['isolated_invalid'])
    if (
        isolated_invalid_policy != parameters['source']['policy_ids']['keep']
        and enclosed_hole_policy != parameters['source']['policy_ids']['filled_support']
    ):
        raise ValueError('Isolated-invalid repair requires the explicit filled-sand policy')

def output_display_contract(mode: str, *, parameters: Mapping) -> dict:
    """Return the explicit display-only renderer contract and current executable provenance for the selected presentation mode."""
    if mode == parameters['source']['policy_ids']['natural_display']:
        return modes_contract(parameters=parameters['display_modes'])
    return photographic_contract(parameters=parameters['photographic_display']) if mode == parameters['source']['policy_ids']['photographic_display'] else {'contract': 'existing-fixed-log-camera-channel-view'}

def source_outputs(corrected: np.ndarray, safe: np.ndarray, *, display_mode: str, isolated_invalid_policy: str=None, return_auxiliary: bool=False, parameters: Mapping):
    """Separate full photographic presentation from scientific sand support."""
    isolated_invalid_policy = parameters['source']['default_policies']['isolated_invalid_policy'] if isolated_invalid_policy is None else isolated_invalid_policy
    finite = np.isfinite(corrected).all(axis=0)
    effective = safe & finite
    scientific = corrected.copy()
    scientific[:, ~effective] = np.nan
    auxiliary = dict(
        filled_invalid_mask=None,
        isolated_invalid_fill=None,
        display_modes=None,
        photo_sand_mask=None,
        display_modes_metadata=None,
        display_invalid_mask=None,
    )
    if isolated_invalid_policy != parameters['source']['policy_ids']['keep']:
        scientific, filled, metadata = fill_isolated_invalid(scientific, safe, parameters=parameters['isolated_invalid'])
        effective = safe & np.isfinite(scientific).all(axis=0)
        auxiliary.update(filled_invalid_mask=filled, isolated_invalid_fill=metadata)
    if display_mode == parameters['source']['policy_ids']['natural_display']:
        modes, metadata = render_display_modes(corrected, safe, tile_rows=parameters['source']['tile_rows'], parameters=parameters['display_modes'])
        rgb = modes['natural']
        auxiliary.update(display_modes=modes, display_modes_metadata=metadata, photo_sand_mask=safe.copy(), display_invalid_mask=~finite)
    else:
        rgb = photographic_display(corrected, tile_rows=parameters['source']['tile_rows'], parameters=parameters['photographic_display']) if display_mode == parameters['source']['policy_ids']['photographic_display'] else raw_detail_display(scientific, parameters=parameters['sidecar'])
    output = (rgb, scientific, effective)
    return (*output, auxiliary) if return_auxiliary else output

def source_homography(value: Any, *, parameters: Mapping) -> np.ndarray:
    """Normalize a source transform by its recorded homogeneous coordinate, preserving the original float64 arithmetic and scale limit."""
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(matrix[2, 2]) < parameters['source']['homography_min_scale']
    ):
        raise ValueError('Source-bound homography is invalid')
    return matrix / matrix[2, 2]

def chart_baseline_coordinates(fit: Any, grid: Any, h_ref_to_baseline: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Transport frozen chart centres from area-averaged reference cells into baseline camera cells. Cell scale and centre offsets are explicit, and point operations retain float32 ordering."""
    xy = np.asarray(fit.xy, dtype=np.float32)
    if xy.shape != tuple(parameters['source']['chart_shape']):
        raise ValueError('V4-R5 ColorChecker lattice is incomplete')
    native_ref = np.column_stack(((xy[:, 0] + grid.origin_cell_xy[0]) * parameters['source']['cell_scale'] + parameters['source']['cell_center'], (xy[:, 1] + grid.origin_cell_xy[1]) * parameters['source']['cell_scale'] + parameters['source']['cell_center'])).astype(np.float32)
    native_baseline = cv2.perspectiveTransform(native_ref[None], h_ref_to_baseline)[0]
    return (native_baseline - parameters['source']['cell_center']) / parameters['source']['cell_scale']

class SourceProcessor:
    """Apply frozen camera radiometry to an exact, certified source revision."""

    def __init__(self, source_row: dict[str, Any], *, calibration_manifest_path: Path, photo_aggregate_path: Path, enclosed_hole_policy: str=None, display_mode: str=None, isolated_invalid_policy: str=None, parameters: Mapping) -> None:
        """Preserve the source-bound camera processing contract with explicit parameters."""
        from darsia._opencv_runtime import scientific_thread
        scientific_thread("SourceProcessor:setup")
        self.parameters = parameters
        enclosed_hole_policy = self.parameters['source']['default_policies']['enclosed_hole_policy'] if enclosed_hole_policy is None else enclosed_hole_policy
        display_mode = self.parameters['source']['default_policies']['display_mode'] if display_mode is None else display_mode
        isolated_invalid_policy = self.parameters['source']['default_policies']['isolated_invalid_policy'] if isolated_invalid_policy is None else isolated_invalid_policy
        _validate_output_policy(enclosed_hole_policy, display_mode, isolated_invalid_policy, parameters=self.parameters)
        self.enclosed_hole_policy = enclosed_hole_policy
        self.display_mode = display_mode
        self.isolated_invalid_policy = isolated_invalid_policy
        self.row = source_row
        self.run = str(source_row['run'])
        self.source_path = _path(source_row['source_manifest_path'], parameters=self.parameters)
        self.source = _read(self.source_path, parameters=self.parameters)
        self.calibration_path = _path(calibration_manifest_path, parameters=self.parameters)
        self.photo_aggregate_path = _path(photo_aggregate_path, parameters=self.parameters)
        calibration = _read(self.calibration_path, parameters=self.parameters)
        if calibration.get('contract') != self.parameters['source']['symbols']['calibration_contract']:
            if not str(calibration.get('contract') or '').startswith(self.parameters['source']['symbols']['calibration_contract_prefix']):
                raise ValueError('Not a V4-R5 calibration manifest')
        marker = bind_photo_marker(self.source_path, source_row, enclosed_hole_policy=self.enclosed_hole_policy, parameters=self.parameters)
        cohort = PhotoCohort.__new__(PhotoCohort)
        cohort.parameters = self.parameters
        cohort.source_manifest_path = self.source_path
        cohort.source_manifest_sha256 = source_row['source_manifest_sha256']
        cohort.source = self.source
        cohort.rows = marker['frames']
        cohort.shape_hw = tuple(self.source['analysis_image_shape_hw'])
        cohort.enclosed_hole_policy = self.enclosed_hole_policy
        self.cohort = cohort
        fit_path = self.calibration_path.parent / f'{self.run}_baseline.npz'
        sidecar_path = fit_path.with_suffix('.json')
        self.fit, self.grid = obs.load_baseline(fit_path, parameters=self.parameters['photometry'])
        self.reference, ref_grid = obs.load_baseline(self.calibration_path.parent / 'virtual_reference_baseline.npz', parameters=self.parameters['photometry'])
        if self.grid != ref_grid:
            raise ValueError('V4-R5 run fit and reference use different grids')
        fit_source = _read(sidecar_path, parameters=self.parameters)['source']
        frame0 = cohort.frame(0)
        if (
            fit_source['source_manifest_sha256'] != source_row['source_manifest_sha256']
            or fit_source['source_arw_sha256'] != frame0['v5_B_arw_sha256']
            or fit_source['source_review_sha256'] != frame0['v5_B_native_jpeg_sha256']
        ):
            raise ValueError('Current source revision differs from its V4-R5 B fit')
        self.old_field = None
        self.old_field_sha = None
        for item in calibration.get('old_B_local_residual_fields') or []:
            if item['run'] != self.run:
                continue
            path = _path(item['field_path'], parameters=self.parameters)
            if not path.is_absolute():
                path = self.calibration_path.parents[3] / path
            if sha256_file(_path(path, parameters=self.parameters)) != item['field_sha256']:
                raise ValueError('Frozen old-light field changed')
            with np.load(_path(path, parameters=self.parameters), allow_pickle=False) as archive:
                self.old_field = archive['field'].astype(np.float32)
            self.old_field_sha = item['field_sha256']
            break
        self.identity_value = dict(
            contract=self.parameters['source']['processor_contract'],
            calibration_manifest_path=str(self.calibration_path),
            calibration_manifest_sha256=sha256_file(_path(self.calibration_path, parameters=self.parameters)),
            photo_source_aggregate_path=str(self.photo_aggregate_path),
            photo_source_aggregate_sha256=sha256_file(_path(self.photo_aggregate_path, parameters=self.parameters)),
            processor_module_sha256=sha256_file(_path(__file__, parameters=self.parameters)),
            renderer_module_sha256=sha256_file(_path(registered_edge_aware_sensor_linear.__code__.co_filename, parameters=self.parameters)),
            photometry_module_sha256=sha256_file(_path(obs.__file__, parameters=self.parameters)),
            pilot_math_module_sha256=sha256_file(_path(correct_frame_native.__code__.co_filename, parameters=self.parameters)),
            validation_only=True,
            scientific_mass_eligible=False,
            whole_run_B_H_pass=True,
            donor_tile_cv_pass=False,
            late_yellow_gate_pass=False,
            weak_phase_gate_status='pending-diagnostic',
        )
        if (
            enclosed_hole_policy != self.parameters['source']['policy_ids']['keep']
            or display_mode != self.parameters['source']['policy_ids']['raw_display']
        ):
            self.identity_value.update(processor_variant_contract=self.parameters['source']['filled_processor_contract'], enclosed_hole_policy=enclosed_hole_policy, enclosed_hole_max_area_px=self.parameters['source']['enclosed_hole_max_area_px'], enclosed_hole_vertical_gap_max_px=self.parameters['source']['enclosed_hole_vertical_gap_max_px'] if enclosed_hole_policy == self.parameters['source']['policy_ids']['filled_support'] else 0, photo_sand_mask_module_sha256=sha256_file(_path(project_source_photo_safe_core.__code__.co_filename, parameters=self.parameters)), photo_sand_support_policy=photo_safe_core_policy_metadata(enclosed_hole_policy, parameters=self.parameters['support']), display_mode=display_mode, display_transfer=output_display_contract(display_mode, parameters=self.parameters), full_frame_correction_before_model_mask=True)
        if isolated_invalid_policy != self.parameters['source']['policy_ids']['keep']:
            self.identity_value.update(processor_variant_contract=self.parameters['source']['isolated_processor_contract'], isolated_invalid_policy=isolated_invalid_policy, isolated_invalid_fill=isolated_invalid_contract(parameters=self.parameters['isolated_invalid']), isolated_invalid_module_sha256=sha256_file(_path(fill_isolated_invalid.__code__.co_filename, parameters=self.parameters)))

    def identity(self) -> dict[str, Any]:
        """Preserve the source-bound camera processing contract with explicit parameters."""
        return dict(self.identity_value)

    def correct(self, *, run: str, frame_id: str, camera_jpeg: Path, camera_arw: Path, review_jpeg: Path, source_manifest_sha256: str, content_shape_hw: tuple[int, int], content_box_xywh: tuple[int, int, int, int]) -> CorrectedFrame:
        """Preserve the source-bound camera processing contract with explicit parameters."""
        from darsia._opencv_runtime import scientific_thread, completed_frame
        ipp_before = scientific_thread("SourceProcessor.correct:before")
        if (
            run != self.run
            or source_manifest_sha256 != self.row['source_manifest_sha256']
            or tuple(content_shape_hw) != tuple(self.parameters['source']['native_shape_hw'])
            or tuple(content_box_xywh) != tuple(self.parameters['source']['content_box_xywh'])
        ):
            raise ValueError('V4-R5 source/geometry binding changed')
        reference = frame_id == 'pre_injection_reference'
        if not reference and (not frame_id.startswith('frame_')):
            raise ValueError('Unknown V4-R5 source frame ID')
        index = 0 if reference else int(frame_id.split('_')[1])
        frame = self.cohort.frame(index)
        if reference:
            if (
                sha256_file(_path(camera_arw, parameters=self.parameters)) != frame['v5_B_arw_sha256']
                or sha256_file(_path(camera_jpeg, parameters=self.parameters)) != frame['v5_B_camera_jpeg_sha256']
                or sha256_file(_path(review_jpeg, parameters=self.parameters)) != frame['v5_B_native_jpeg_sha256']
            ):
                raise ValueError('Reference source differs from frozen V4-R5 B')
        elif sha256_file(_path(camera_jpeg, parameters=self.parameters)) != frame['camera_jpeg_sha256'] or sha256_file(_path(review_jpeg, parameters=self.parameters)) != frame['native_review_jpeg_sha256']:
            raise ValueError('Timeline source differs from photo/H receipt')
        safe_grid = source_photo_grid(
            _path(frame['photo_sand_reference_mask_path'], parameters=self.parameters),
            frame['photo_sand_reference_mask_sha256'],
            parameters=self.parameters['support'],
        )
        h_ref_frame = source_homography(frame[self.parameters['source']['symbols']['reference_to_frame_h_key']], parameters=self.parameters)
        h_b_frame = source_homography(frame['H_v5_B_to_frame_native'], parameters=self.parameters)
        h_ref_b = source_homography(np.linalg.inv(h_b_frame) @ h_ref_frame, parameters=self.parameters)
        h_for_source = h_ref_b if reference else h_ref_frame
        (safe, mask_metadata) = project_source_photo_safe_core(
            safe_grid,
            h_for_source,
            shape_hw=tuple(self.parameters['source']['native_shape_hw']),
            enclosed_hole_policy=self.enclosed_hole_policy,
            return_metadata=True,
            parameters=self.parameters['support'],
            guard_px=self.parameters['source']['mask_guard_px'],
        )
        expected = dict(
            arw_sha256=sha256_file(_path(camera_arw, parameters=self.parameters)),
            camera_jpeg_sha256=sha256_file(_path(camera_jpeg, parameters=self.parameters)),
            review_native_jpeg_sha256=sha256_file(_path(review_jpeg, parameters=self.parameters)),
        )
        (linear, renderer) = registered_edge_aware_sensor_linear(
            camera_arw,
            camera_jpeg,
            review_jpeg,
            expected_source_hashes=expected,
            content_box=content_box_xywh,
            parameters=self.parameters['registration'],
            source_id=self.parameters['source']['renderer_source_ids'][self.run],
            profile_id=self.parameters['source']['profiles'][self.run],
        )
        exact_b = expected['arw_sha256'] == frame['v5_B_arw_sha256'] and expected['camera_jpeg_sha256'] == frame['v5_B_camera_jpeg_sha256']
        if reference and (not exact_b):
            raise ValueError('Reference correction did not use the exact B acquisition')
        chart_certificate = None
        xy = usable = None
        if not exact_b:
            bcert = _read(_path(frame['v5_B_registration_certificate_path'], parameters=self.parameters), parameters=self.parameters)
            if bcert.get('sources', {}).get('review_native_jpeg_sha256') != frame['v5_B_native_jpeg_sha256']:
                raise ValueError('Baseline chart registration belongs to another B')
            (xy, usable, chart_certificate) = register_chart(
                baseline_camera_jpeg=_path(frame['v5_B_camera_jpeg_path'], parameters=self.parameters),
                baseline_camera_sha256=frame['v5_B_camera_jpeg_sha256'],
                frame_camera_jpeg=camera_jpeg,
                frame_camera_sha256=expected['camera_jpeg_sha256'],
                baseline_chart_xy8=chart_baseline_coordinates(self.fit, self.grid, h_ref_b, parameters=self.parameters),
                baseline_chart_log_rgb=self.fit.CB,
                baseline_review_to_camera_h=np.asarray(bcert['review_to_camera_h'], np.float64),
                frame_review_to_camera_h=np.asarray(renderer['review_to_camera_h'], np.float64),
                frame_native_linear_rgb=linear,
                parameters=self.parameters['registration'],
            )
        (corrected, info) = correct_frame_native(
            linear,
            self.fit,
            self.reference,
            self.grid,
            h_for_source,
            np.ones_like(safe) if self.enclosed_hole_policy != self.parameters['source']['policy_ids']['keep'] or self.display_mode in (self.parameters['source']['policy_ids']['photographic_display'], self.parameters['source']['policy_ids']['natural_display']) else safe,
            exact_baseline_source=exact_b,
            frame_chart_xy8=xy,
            frame_chart_usable=usable,
            old_field_ref8=self.old_field,
            parameters=self.parameters['photometry'],
        )
        finite = np.isfinite(corrected).all(axis=0)
        (rgb, scientific, effective, auxiliary) = source_outputs(
            corrected,
            safe,
            display_mode=self.display_mode,
            isolated_invalid_policy=getattr(self, 'isolated_invalid_policy', self.parameters['source']['policy_ids']['keep']),
            return_auxiliary=True,
            parameters=self.parameters,
        )
        if not effective.any():
            raise ValueError('V4-R5 has no source-supported corrected sand')
        if np.nanmax(scientific) > np.log(self.parameters['source']['signal_denominator']):
            raise ValueError('V4-R5 corrected linear signal exceeds fixed denominator')
        certificate = dict(
            transform_sha256=sha256(json.dumps(renderer, sort_keys=True, default=str).encode()).hexdigest(),
            camera_jpeg_sha256=expected['camera_jpeg_sha256'],
            camera_arw_sha256=expected['arw_sha256'],
            original_review_jpeg_sha256=expected['review_native_jpeg_sha256'],
            native_shape_hw=list(self.parameters['source']['native_shape_hw']),
            renderer_certificate=renderer,
            chart_certificate=chart_certificate,
            v4r5_native_info=info,
            source_photo_receipt_sha256=frame['photo_sand_reference_receipt_sha256'],
            source_frame_transport_receipt_sha256=frame['receipt_sha256'],
            fit_baseline_sidecar_sha256=sha256_file(_path(self.calibration_path.parent / f'{self.run}_baseline.json', parameters=self.parameters)),
            old_field_sha256=self.old_field_sha,
            required_valid_mask_sha256=sha256(np.packbits(effective, bitorder='little').tobytes()).hexdigest(),
            source_safe_pixels=int(safe.sum()),
            source_safe_corrected_pixels=int(effective.sum()),
            source_safe_unresolved_pixels=int(np.count_nonzero(safe & ~effective)),
            source_safe_corrected_fraction=float(effective.sum() / safe.sum()),
            provisional_gate_status=dict(whole_run_B_H_pass=True, donor_tile_cv_pass=False, late_yellow_gate_pass=False, weak_phase='pending-diagnostic'),
        )
        if (
            self.enclosed_hole_policy != self.parameters['source']['policy_ids']['keep']
            or self.display_mode != self.parameters['source']['policy_ids']['raw_display']
        ):
            certificate.update(mask_metadata)
            certificate.update(display_mode=self.display_mode, photographic_display=output_display_contract(self.display_mode, parameters=self.parameters) if self.display_mode in (self.parameters['source']['policy_ids']['photographic_display'], self.parameters['source']['policy_ids']['natural_display']) else None, photographic_full_frame_finite_pixels=int(finite.sum()), scientific_sidecar_domain='source-safe-filled-and-finite', full_frame_correction_before_model_mask=True)
        if auxiliary['isolated_invalid_fill'] is not None:
            certificate.update(isolated_invalid_fill=auxiliary['isolated_invalid_fill'], photo_sand_support_contract=auxiliary['isolated_invalid_fill']['contract'])
        if auxiliary['display_modes_metadata'] is not None:
            certificate['display_modes_metadata'] = auxiliary['display_modes_metadata']
        certificate["opencv_ipp"] = completed_frame(ipp_before, "SourceProcessor.correct:after")
        return CorrectedFrame(rgb=rgb, log_linear_rgb=scientific, certificate=certificate, required_valid_mask=effective, filled_invalid_mask=auxiliary['filled_invalid_mask'], display_modes=auxiliary['display_modes'], photo_sand_mask=auxiliary['photo_sand_mask'], display_invalid_mask=auxiliary['display_invalid_mask'])

def read_bound(path, expected, *, parameters: Mapping):
    """Read an unchanged, digest-bound source document through explicit asset relocations."""
    path = _path(path, parameters=parameters)
    if not expected or sha256_file(path) != expected:
        raise ValueError('Source artifact changed: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))

def verify_calibration_baselines(calibration_manifest_path, expected_manifest_sha256, run, *, parameters: Mapping):
    """Bind both loaded fits and their sidecars to the sealed calibration.

    The upstream fit loader checks a NPZ against its sidecar. This additional
    check prevents a mutually consistent, but changed, NPZ/sidecar pair from
    silently replacing the frozen run or virtual-reference parameters.
    """
    manifest_path = _path(calibration_manifest_path, parameters=parameters).resolve()
    calibration = read_bound(manifest_path, expected_manifest_sha256, parameters=parameters)
    if not isinstance(run, str) or re.fullmatch(parameters['source']['source_id_pattern'], run) is None:
        raise ValueError('Frozen calibration requires an exact source identity')
    rows = calibration.get('results')
    if not isinstance(rows, list):
        raise ValueError('Frozen calibration has no parameter asset inventory')
    assets = []
    for identity in (run, 'virtual_reference'):
        matching = [row for row in rows if isinstance(row, dict) and row.get('run') == identity]
        if len(matching) != 1:
            raise ValueError('Frozen calibration needs one parameter binding for ' + identity)
        row = matching[0]
        expected_path = (manifest_path.parent / (identity + '_baseline.npz')).resolve()
        if not row.get('baseline_path'):
            raise ValueError('Frozen calibration lacks its recorded parameter path')
        declared = _path(row['baseline_path'], parameters=parameters)
        historical_manifest = Path(parameters['source'].get('historical_calibration_manifest_path', str(manifest_path)))
        candidates = [_path(declared.resolve(), parameters=parameters)] if declared.is_absolute() else [_path((base / declared).resolve(), parameters=parameters) for base in (historical_manifest.parent, *historical_manifest.parent.parents)]
        if expected_path not in candidates:
            raise ValueError('Frozen calibration parameter path differs from the loaded fit')
        for kind, path, key in (('baseline', expected_path, 'baseline_sha256'), ('sidecar', expected_path.with_suffix('.json'), 'sidecar_sha256')):
            digest = row.get(key)
            if (
                not isinstance(digest, str)
                or re.fullmatch('[0-9a-f]{64}', digest) is None
                or not path.is_file()
                or sha256_file(path) != digest
            ):
                raise ValueError('Frozen calibration parameter asset changed: ' + str(path))
            assets.append(dict(run=identity, kind=kind, path=str(path), sha256=digest))
    return dict(contract=parameters['source']['symbols']['baseline_assets_contract'], calibration_manifest_path=str(manifest_path), calibration_manifest_sha256=expected_manifest_sha256, run=run, assets=assets, verified=True)

def retain_parent_scientific_pixels(full, parent, parent_valid, *, parameters: Mapping):
    """Explicit exact cache reuse, including the previous isolated-pixel repair."""
    full, parent, valid = (np.asarray(full), np.asarray(parent), np.asarray(parent_valid))
    if (
        full.dtype != np.float32
        or parent.dtype != np.float32
        or full.shape != parent.shape
        or full.shape[0] != 3
        or valid.dtype != bool
        or valid.shape != full.shape[1:]
        or not np.isfinite(parent[:, valid]).all()
    ):
        raise ValueError('Parent scientific cache does not share exact native float geometry')
    freshly_finite = np.isfinite(full).all(0) & valid
    differences = np.abs(full[:, freshly_finite].astype(np.float64) - parent[:, freshly_finite])
    audit = dict(
        parent_supported_pixels=int(valid.sum()),
        freshly_decoded_max_abs_log_difference=float(np.max(differences, initial=0)),
        freshly_invalid_parent_supported_pixels=int(np.count_nonzero(valid & ~freshly_finite)),
        parent_float_bytes_preserved=True,
    )
    full[:, valid] = parent[:, valid]
    return (full, audit)

class RigSourceProcessor:
    """Correct registered raw acquisitions while preserving finite parent pixels."""

    def __init__(self, native_manifest_path, *, parameters: Mapping):
        """Preserve the source-bound camera processing contract with explicit parameters."""
        from darsia._opencv_runtime import scientific_thread
        scientific_thread("RigSourceProcessor:setup")
        self.parameters = parameters
        self.native_path = _path(native_manifest_path, parameters=self.parameters)
        self.native = json.loads(self.native_path.read_text(encoding='utf-8-sig'))
        binding = self.native['v5_photometry']
        self.source_path = _path(binding['source_manifest'], parameters=self.parameters)
        read_bound(self.source_path, binding['source_manifest_sha256'], parameters=self.parameters)
        processor = binding['processor']
        for path_key, sha_key in (('calibration_manifest_path', 'calibration_manifest_sha256'), ('photo_source_aggregate_path', 'photo_source_aggregate_sha256')):
            read_bound(processor[path_key], processor[sha_key], parameters=self.parameters)
        self.calibration_baseline_binding = verify_calibration_baselines(
            processor['calibration_manifest_path'],
            processor['calibration_manifest_sha256'],
            self.native['run'],
            parameters=self.parameters,
        )
        aggregate = read_bound(processor['photo_source_aggregate_path'], processor['photo_source_aggregate_sha256'], parameters=self.parameters)
        matching = [row for row in aggregate['rows'] if row['run'] == self.native['run'] and row['source_manifest_sha256'] == binding['source_manifest_sha256']]
        if len(matching) != 1:
            raise ValueError('Frozen photo transport does not bind this exact source version')
        self.processor = SourceProcessor(
            matching[0],
            calibration_manifest_path=_path(processor['calibration_manifest_path'], parameters=self.parameters),
            photo_aggregate_path=_path(processor['photo_source_aggregate_path'], parameters=self.parameters),
            enclosed_hole_policy=self.parameters['source']['source_policies']['enclosed_hole_policy'],
            display_mode=self.parameters['source']['source_policies']['display_mode'],
            isolated_invalid_policy=self.parameters['source']['source_policies']['isolated_invalid_policy'],
            parameters=self.parameters,
        )
        verify_execution_identity(self.processor.identity(), self.parameters['source']['execution_identity'], self.parameters['source']['required_execution_identity_keys'])
        self.historical_processor_identity = dict(processor)
        self.identity = dict(
            contract=self.parameters['source']['rig_contract'],
            parent_native_manifest_path=str(self.native_path),
            parent_native_manifest_sha256=sha256_file(_path(self.native_path, parameters=self.parameters)),
            frozen_photometry=self.processor.identity(),
            calibration_baseline_binding=self.calibration_baseline_binding,
            full_native_shape_hw=list(self.parameters['source']['native_shape_hw']),
            code_sha256=sha256_file(_path(__file__, parameters=self.parameters)),
            cpu_only=True,
            training=False,
            photographic_input=False,
            mass_eligible=False,
        )

    def parent_geometry(self, index=None):
        """Return the certified acquisition geometry. A missing index selects the true pre-injection baseline, never the first timeline acquisition."""
        row = dict(self.processor.cohort.frame(0 if index is None else index))
        if index is None:
            row.update(camera_jpeg_path=row['v5_B_camera_jpeg_path'], camera_jpeg_sha256=row['v5_B_camera_jpeg_sha256'], native_review_jpeg_path=row['v5_B_native_jpeg_path'], native_review_jpeg_sha256=row['v5_B_native_jpeg_sha256'], camera_arw_path=row['v5_B_arw_path'], camera_arw_sha256=row['v5_B_arw_sha256'])
            h = np.linalg.inv(np.asarray(row['H_v5_B_to_frame_native'])) @ np.asarray(row[self.parameters['source']['symbols']['reference_to_frame_h_key']])
            row[self.parameters['source']['symbols']['reference_to_frame_h_key']] = (h / h[2, 2]).tolist()
            row['H_v5_B_to_frame_native'] = np.eye(3).tolist()
        else:
            frame = self.native['frame_sources'][index]
            row['camera_arw_path'], row['camera_arw_sha256'] = (frame['v5_raw_path'], frame['v5_raw_sha256'])
        return row

    def correct(self, geometry_frame, *, parent_frame_index=None, parent_reference=False):
        """Preserve the source-bound camera processing contract with explicit parameters."""
        from darsia._opencv_runtime import scientific_thread, completed_frame
        ipp_before = scientific_thread("RigSourceProcessor.correct:before")
        row, p = (geometry_frame, self.processor)
        if parent_reference and parent_frame_index is not None:
            raise ValueError('Reference and timeline-frame reuse are distinct source bindings')
        required = ('camera_jpeg', 'camera_arw', 'native_review_jpeg')
        sources = {}
        for name in required:
            path, expected = (_path(row[name + '_path'], parameters=self.parameters), row[name + '_sha256'])
            if sha256_file(_path(path, parameters=self.parameters)) != expected:
                raise ValueError('New rig RAW input differs from its source receipt: ' + name)
            sources[name] = dict(path=str(path), sha256=expected)
        h_ref = source_homography(row[self.parameters['source']['symbols']['reference_to_frame_h_key']], parameters=self.parameters)
        h_b = source_homography(row['H_v5_B_to_frame_native'], parameters=self.parameters)
        h_ref_b = source_homography(np.linalg.inv(h_b) @ h_ref, parameters=self.parameters)
        (linear, renderer) = registered_edge_aware_sensor_linear(
            sources['camera_arw']['path'],
            sources['camera_jpeg']['path'],
            sources['native_review_jpeg']['path'],
            content_box=self.parameters['source']['content_box_xywh'],
            expected_source_hashes=dict(arw_sha256=sources['camera_arw']['sha256'], camera_jpeg_sha256=sources['camera_jpeg']['sha256'], review_native_jpeg_sha256=sources['native_review_jpeg']['sha256']),
            parameters=self.parameters['registration'],
            source_id=self.parameters['source']['renderer_source_ids'][p.run],
            profile_id=self.parameters['source']['profiles'][p.run],
        )
        exact_b = sources['camera_arw']['sha256'] == row['v5_B_arw_sha256'] and sources['camera_jpeg']['sha256'] == row['v5_B_camera_jpeg_sha256']
        xy = usable = chart = None
        if not exact_b:
            certificate_path = _path(row['v5_B_registration_certificate_path'], parameters=self.parameters)
            certificate = json.loads(certificate_path.read_text(encoding='utf-8-sig'))
            if certificate['sources']['review_native_jpeg_sha256'] != row['v5_B_native_jpeg_sha256']:
                raise ValueError('Frozen B chart registration belongs to another reference')
            (xy, usable, chart) = register_chart(
                baseline_camera_jpeg=_path(row['v5_B_camera_jpeg_path'], parameters=self.parameters),
                baseline_camera_sha256=row['v5_B_camera_jpeg_sha256'],
                frame_camera_jpeg=_path(sources['camera_jpeg']['path'], parameters=self.parameters),
                frame_camera_sha256=sources['camera_jpeg']['sha256'],
                baseline_chart_xy8=chart_baseline_coordinates(p.fit, p.grid, h_ref_b, parameters=self.parameters),
                baseline_chart_log_rgb=p.fit.CB,
                baseline_review_to_camera_h=np.asarray(certificate['review_to_camera_h']),
                frame_review_to_camera_h=np.asarray(renderer['review_to_camera_h']),
                frame_native_linear_rgb=linear,
                parameters=self.parameters['registration'],
            )
        (corrected, info) = correct_frame_native(
            linear,
            p.fit,
            p.reference,
            p.grid,
            h_ref,
            np.ones(self.parameters['source']['native_shape_hw'], bool),
            exact_baseline_source=exact_b,
            frame_chart_xy8=xy,
            frame_chart_usable=usable,
            old_field_ref8=p.old_field,
            parameters=self.parameters['photometry'],
        )
        del linear
        cache_audit = None
        source_identity = dict(kind='same-stem-camera-acquisition', camera_member_sha256=sources['camera_jpeg']['sha256'])
        if parent_frame_index is not None:
            old = self.native['frame_sources'][parent_frame_index]
            if old['source_sha256'] != sources['camera_jpeg']['sha256']:
                if p.run != self.parameters['source']['symbols']['prepared_source_id'] or parent_frame_index != 0:
                    raise ValueError('Cannot reuse parent pixels for a different acquisition')
                original = p.source['frame_sources'][0]
                transport = p.cohort.frame(0)
                if (
                    original['source_sha256'] != old['source_sha256']
                    or sha256_file(_path(original['source_path'], parameters=self.parameters)) != original['source_sha256']
                    or transport['camera_jpeg_sha256'] != sources['camera_jpeg']['sha256']
                    or old['v5_raw_sha256'] != sources['camera_arw']['sha256']
                ):
                    raise ValueError(self.parameters['source']['symbols']['prepared_source_error'])
                source_identity = dict(
                    kind='historical-three-ARW-prepared-source',
                    prepared_source_path=original['source_path'],
                    prepared_source_sha256=original['source_sha256'],
                    camera_member=sources['camera_jpeg'],
                    raw_member=sources['camera_arw'],
                    original_transport_receipt_sha256=transport['receipt_sha256'],
                    label_identity_preserved=True,
                    prepared_source_reinterpreted=False,
                    new_radiometry='same SHA-bound member renderer as the frozen parent; no new composite reconstruction',
                )
            for path_key, sha_key in (('v5_log_linear_rgb_path', 'v5_log_linear_rgb_sha256'), ('v5_log_linear_valid_mask_path', 'v5_log_linear_valid_mask_sha256')):
                if sha256_file(_path(old[path_key], parameters=self.parameters)) != old[sha_key]:
                    raise ValueError('Parent float cache changed')
            parent = np.load(_path(old['v5_log_linear_rgb_path'], parameters=self.parameters), mmap_mode='r', allow_pickle=False)
            valid = np.load(_path(old['v5_log_linear_valid_mask_path'], parameters=self.parameters), mmap_mode='r', allow_pickle=False)
            corrected, cache_audit = retain_parent_scientific_pixels(corrected, parent, valid, parameters=self.parameters)
        elif parent_reference:
            reference = self.native['reference']
            state = reference['v5_reference_state']
            if (
                not exact_b
                or state['camera_jpeg_sha256'] != sources['camera_jpeg']['sha256']
                or state['camera_arw_sha256'] != sources['camera_arw']['sha256']
            ):
                raise ValueError('Pre-injection reuse differs from its original reference acquisition')
            for path_key, sha_key in (('v5_log_linear_rgb_path', 'v5_log_linear_rgb_sha256'), ('v5_log_linear_valid_mask_path', 'v5_log_linear_valid_mask_sha256'), ('v5_baseline_receipt_path', 'v5_baseline_receipt_sha256')):
                if sha256_file(_path(reference[path_key], parameters=self.parameters)) != reference[sha_key]:
                    raise ValueError('Parent pre-injection float reference changed')
            parent = np.load(_path(reference['v5_log_linear_rgb_path'], parameters=self.parameters), mmap_mode='r', allow_pickle=False)
            valid = np.load(_path(reference['v5_log_linear_valid_mask_path'], parameters=self.parameters), mmap_mode='r', allow_pickle=False)
            corrected, cache_audit = retain_parent_scientific_pixels(corrected, parent, valid, parameters=self.parameters)
            cache_audit['reference_source_binding_preserved'] = True
        return (corrected, dict(contract=self.parameters['source']['rig_contract'], processor=self.identity, sources=sources, renderer=renderer, chart=chart, photometry=info, exact_B=exact_b, **{self.parameters['source']['symbols']['reference_to_frame_h_key']: h_ref.tolist()}, H_v5_B_to_frame_native=h_b.tolist(), full_native_corrected_finite_pixels=int(np.isfinite(corrected).all(0).sum()), source_identity=source_identity, parent_cache=cache_audit, cpu_only=True, training=False, opencv_ipp=completed_frame(ipp_before, "RigSourceProcessor.correct:after")))

def normalize_homography(value: Any, *, parameters: Mapping) -> np.ndarray:
    """Validate and scale a recorded camera homography. The determinant and homogeneous-coordinate limits come from the receipt policy; no transform is fitted."""
    h = np.asarray(value, np.float64)
    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
        or abs(float(np.linalg.det(h))) < parameters['source']['receipt_homography_min_scale']
        or abs(float(h[2, 2])) < parameters['source']['receipt_homography_min_scale']
    ):
        raise ValueError('V5 per-frame homography is invalid')
    return h / h[2, 2]
