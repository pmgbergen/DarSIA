"""Exact masked and temporal native representations from explicit source arrays.

These adapters never select an acquisition or load a model. Photographic
support is supplied independently of phase labels; inherited support cannot be
promoted by a larger scientific rig sidecar.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskedNativeInput:
    """Float32 log RGB and its exact finite support, with source provenance."""
    log: np.ndarray
    valid: np.ndarray
    provenance: dict


@dataclass(frozen=True)
class TemporalNativeInput:
    """A native core or its explicitly declared, source-backed temporal ring."""
    log: np.ndarray
    valid: np.ndarray
    origin_xy: tuple[int, int]
    provenance: dict


def validate_native_input(log, valid):
    """Require the original CHW float32 representation and exact finite mask."""
    if (log.dtype != np.float32 or log.ndim != 3 or log.shape[0] != 3
            or valid.dtype != bool or valid.shape != log.shape[1:]
            or not np.array_equal(valid, np.isfinite(log).all(axis=0))):
        raise ValueError("Historical native input requires exact float32 finite/support agreement")


def original_native_input(log, valid, *, provenance, scientific_native_log=None,
                          parameters: Mapping):
    """Retain an acquisition's own masked source and check its supported bits.

    A supplied rig sidecar is compared through uint32 views only on the
    original valid pixels. Its additional finite pixels do not become model
    support. The loader supplies the exact original descriptors in provenance.
    """
    validate_native_input(log, valid)
    if scientific_native_log is not None:
        current = np.asarray(scientific_native_log)
        if (current.dtype != np.float32 or current.shape != log.shape
                or not np.array_equal(current[:, valid].view(np.uint32), log[:, valid].view(np.uint32))):
            raise ValueError("Rig RAW differs from original native source on historical valid pixels")
    return MaskedNativeInput(log, valid, dict(provenance,
        contract=parameters["contract"], kind=parameters["kinds"]["original_native"],
        support_promoted=False, scientific_sidecar_changed=False))


def new_native_input(log, valid, photo_sand, *, source_binding,
                     photo_mask_binding, parameters: Mapping):
    """Mask a new source transiently using separately certified photo geometry.

    Phase predictions cannot define this support. Unsupported values become
    NaN in a copy, while the full scientific input is left untouched.
    """
    current, supplied = np.asarray(log), np.asarray(valid)
    validate_native_input(current, supplied)
    photo = np.asarray(photo_sand)
    if (photo.dtype != bool or photo.shape != supplied.shape
            or not isinstance(source_binding, dict) or not source_binding.get("source_sha256")
            or not isinstance(photo_mask_binding, dict) or not photo_mask_binding.get("sha256")
            or photo_mask_binding.get("phase_labels_used") is not False):
        raise ValueError("New native input needs a source-bound geometric photo mask, independent of phase labels")
    supported = supplied & photo
    masked = current.copy()
    masked[:, ~supported] = np.nan
    masked.setflags(write=False)
    supported.setflags(write=False)
    return MaskedNativeInput(masked, supported, dict(contract=parameters["contract"],
        kind=parameters["kinds"]["new_native"], source=source_binding,
        photo_mask=photo_mask_binding, support_promoted=False,
        scientific_sidecar_changed=False, model_or_photometry_fit=False))


def temporal_policy(reference_row, *, verified_ring=None, parameters: Mapping):
    """Retain a declared reference ring; absent metadata means zero padding.

    A partial ring declaration fails. The bound loader supplies verified_ring
    only after checking its original bytes. No implicit ring size is inferred.
    """
    keys = tuple(parameters["fields"]["temporal"])
    present = [reference_row.get(key) is not None for key in keys]
    if not any(present):
        return dict(contract=parameters["contract"], pad_px=0, origin_xy=[0, 0],
                    source=parameters["policy_sources"]["no_ring"])
    if not all(present):
        raise ValueError("Historical temporal ring metadata is partial")
    pad = reference_row[keys[2]]
    origin = tuple(reference_row[keys[3]])
    if type(pad) is not int or pad <= 0 or origin != (-pad, -pad):
        raise ValueError("Historical temporal ring has invalid native origin/padding")
    if verified_ring is None:
        raise ValueError("A declared temporal ring needs its exact verified source descriptor")
    return dict(contract=parameters["contract"], pad_px=pad, origin_xy=list(origin),
                source=parameters["policy_sources"]["ring"], ring=dict(verified_ring))


def original_temporal_input(native, *, policy, assembled_log=None,
                            parameters: Mapping):
    """Use the original native core or an unchanged source-backed ring assembly.

    The loader assembles declared strips with the existing strict sidecar
    routine. Core pixels retain the masked native representation.
    """
    if not isinstance(native, MaskedNativeInput):
        raise TypeError("Original temporal input requires a masked native source")
    if policy["pad_px"] == 0:
        return TemporalNativeInput(native.log, native.valid, (0, 0),
            dict(contract=parameters["contract"], kind=parameters["kinds"]["original_core"], policy=policy))
    if assembled_log is None:
        raise ValueError("Original temporal ring requires its verified strip assembly")
    log = assembled_log
    valid = np.isfinite(log).all(0)
    return TemporalNativeInput(log, valid, tuple(policy["origin_xy"]),
        dict(contract=parameters["contract"], kind=parameters["kinds"]["original_ring"], policy=policy))


def new_temporal_input(native, full_rig_log, *, native_box_xywh,
                       frozen_policy, parameters: Mapping):
    """Crop a new source's own ring under the inherited reference convention.

    The transient ring crop is copied from this source, then its native centre
    is replaced by the masked native log. Larger unmasked rig support cannot
    replace the native model input.
    """
    if not isinstance(native, MaskedNativeInput) or frozen_policy.get("contract") != parameters["contract"]:
        raise ValueError("New temporal input requires an explicit historical input policy")
    pad = frozen_policy.get("pad_px")
    if type(pad) is not int or pad < 0 or tuple(frozen_policy.get("origin_xy", ())) != (-pad, -pad):
        raise ValueError("New temporal padding must equal the bound reference convention")
    if pad == 0:
        return TemporalNativeInput(native.log, native.valid, (0, 0),
            dict(contract=parameters["contract"], kind=parameters["kinds"]["new_core"], policy=frozen_policy))
    if not frozen_policy.get("ring", {}).get("sha256"):
        raise ValueError("A new temporal ring needs the original bound ring-policy evidence")
    full = np.asarray(full_rig_log)
    x, y, w, h = map(int, native_box_xywh)
    if (full.dtype != np.float32 or full.ndim != 3 or full.shape[0] != 3
            or native.log.shape != (3, h, w) or min(x - pad, y - pad) < 0
            or x + w + pad > full.shape[2] or y + h + pad > full.shape[1]):
        raise ValueError("Bound temporal ring escapes the exact full rig source")
    log = full[:, y - pad:y + h + pad, x - pad:x + w + pad].copy()
    log[:, pad:pad + h, pad:pad + w] = native.log
    valid = np.isfinite(log).all(0)
    log.setflags(write=False)
    valid.setflags(write=False)
    return TemporalNativeInput(log, valid, (-pad, -pad),
        dict(contract=parameters["contract"], kind=parameters["kinds"]["new_ring"], policy=frozen_policy,
             rig_scientific_input_changed=False, masked_core_preserved=True))


def original_photo_sand(packed_masks, metadata, descriptor, *, parameters: Mapping):
    """Decode only the original producer's certified photographic sand mask.

    Mask order, bit order and shape remain bound by the original receipt.
    Predicted labels are not read or substituted for this geometry.
    """
    names = tuple(parameters["mask_names"])
    bitorder = parameters["bitorder"]
    if metadata.get("masks_order") != list(names) or metadata.get("bitorder") != bitorder:
        raise ValueError("Original photo domain lacks its frozen packed-mask contract")
    shape = tuple(metadata["native_shape_hw"])
    rows = packed_masks
    if rows.shape != (len(names), (int(np.prod(shape)) + 7) // 8):
        raise ValueError("Original photo mask does not share the frozen native grid")
    mask = np.unpackbits(rows[0], bitorder=bitorder, count=int(np.prod(shape))).reshape(shape).astype(bool)
    mask.setflags(write=False)
    return mask, dict(contract=parameters["contract"], original_probability_masks=descriptor,
                     array_key=parameters["photo_mask_name"], phase_labels_used=False)
