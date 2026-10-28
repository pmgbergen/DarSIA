"""Reject mixed frozen/live registration grids before phase-to-mass work."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


class GeometryRevisionMismatch(ValueError):
    """Images, phase masks and references do not describe one geometry revision."""


def _sha(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise GeometryRevisionMismatch(f'Registration {field} is missing or invalid')
    return value


def registration_revision(correction_contract):
    """Absent registration is legacy; malformed declared registration is not."""
    if correction_contract is None:
        correction_contract = {}
    if not isinstance(correction_contract, Mapping):
        raise GeometryRevisionMismatch('Correction contract must be an object')
    registration = correction_contract.get('rig_registration')
    if registration is None:
        return None
    if not isinstance(registration, Mapping) or registration.get('active') is not True:
        raise GeometryRevisionMismatch('Declared rig registration must be explicitly active')
    identifier = registration.get('geometry_revision_id')
    if not isinstance(identifier, str) or not identifier.strip():
        raise GeometryRevisionMismatch('Registration geometry_revision_id is missing')
    digest = _sha(registration.get('geometry_revision_sha256'), 'geometry_revision_sha256')
    return {'geometry_revision_id': identifier, 'geometry_revision_sha256': digest}


def require_matching_geometry(frozen_manifest, source_correction_contract, *, context='phase-to-mass preparation'):
    """Allow legacy↔legacy or the same verified revision; never infer equivalence."""
    frozen = registration_revision(frozen_manifest.get('correction_contract'))
    live = registration_revision(source_correction_contract)
    declared = frozen_manifest.get('geometry_revision_id')
    if declared is not None and (frozen is None or declared != frozen['geometry_revision_id']):
        raise GeometryRevisionMismatch(f'{context}: frozen manifest geometry ID disagrees with its correction contract')
    if frozen != live:
        old_id = 'legacy geometry (no sand registration)' if frozen is None else frozen['geometry_revision_id']
        new_id = 'legacy geometry (no sand registration)' if live is None else live['geometry_revision_id']
        raise GeometryRevisionMismatch(
            f'{context}: frozen phase/RGB geometry [{old_id}] differs from live reference geometry [{new_id}]. '
            'Prepare a separate revision with matching photographs, phase masks and references. '
            'Old phase masks cannot be combined with newly registered clearwater or current images.'
        )
    return {'matched': True, 'registration_revision': deepcopy(live),
            'mode': 'legacy' if live is None else 'registered', 'identity_inferred_from_image_similarity': False}


def frame_geometry_metadata(correction_contract, correction_rows, shape_hw, *, active_mask=None):
    """Bind per-frame valid bounds to this run's active registration revision."""
    revision = registration_revision(correction_contract)
    rows = [row for row in correction_rows if row.get('type') == 'RigRegistrationCorrection']
    if revision is None:
        if rows:
            raise GeometryRevisionMismatch('Registration receipt exists without an active source geometry contract')
        return None
    if len(rows) != 1 or not isinstance(rows[0].get('receipt'), Mapping):
        raise GeometryRevisionMismatch('Registered frame requires exactly one nonempty geometry receipt')
    receipt = rows[0]['receipt']
    if receipt.get('accepted') is not True or receipt.get('status') not in ('accepted', 'baseline_identity'):
        raise GeometryRevisionMismatch('Registered frame geometry was not accepted')
    if receipt.get('host_geometry_revision_sha256') != revision['geometry_revision_sha256']:
        raise GeometryRevisionMismatch('Frame registration receipt belongs to a different geometry revision')
    if list(receipt.get('shape_hw') or []) != list(shape_hw):
        raise GeometryRevisionMismatch('Frame receipt and numerical field grids differ')
    h, w = map(int, shape_hw)
    bounds = receipt.get('valid_bounds_xyxy')
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or any(type(v) is not int for v in bounds):
        raise GeometryRevisionMismatch('Registered frame needs integer valid_bounds_xyxy')
    left, top, right, bottom = bounds
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise GeometryRevisionMismatch('Registered frame valid bounds lie outside its grid')
    if active_mask is not None:
        import numpy as np
        mask = np.asarray(active_mask)
        if mask.shape != (h, w) or mask.dtype != np.bool_:
            raise GeometryRevisionMismatch('Active mass domain must be a boolean frame grid')
        if mask[:top].any() or mask[bottom:].any() or mask[:, :left].any() or mask[:, right:].any():
            raise GeometryRevisionMismatch('Active mass domain includes invalid registration border pixels')
    return {**revision, 'valid_bounds_xyxy': list(bounds),
            'validity': receipt.get('validity'), 'sampling_displacement_xy_native_px': deepcopy(receipt.get('shift_xy_native_px')),
            'geometry_receipt': deepcopy(dict(receipt)),
            'numerical_field_grid': 'same registered native grid for current RGB and freshly inferred phase probabilities'}


def mass_revision_identifiers(correction_contract):
    revision = registration_revision(correction_contract)
    if revision is None:
        return {'image_correction_revision_id': 'legacy-uniform24-import',
                'geometry_revision_id': 'legacy-darsia-rig-import'}
    profile = str(correction_contract.get('profile') or 'configured-colour-correction')
    return {'image_correction_revision_id': f"{profile}:{revision['geometry_revision_id']}",
            'geometry_revision_id': revision['geometry_revision_id']}
