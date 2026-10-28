"""Optional, in-memory RAW clipping evidence; never a correction or fit policy.

The original uint16 OpenCV EA path retains the measured colour at each interior
CFA photosite. Recovering those samples does not need a second RAW decode. The
outermost photosites cannot be recovered after OpenCV's replicated RGB border
and are explicitly unknown. This module is outside native producer identities.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from typing import Any, Mapping

import cv2
import numpy as np


CONTRACT = "darsia-optional-cfa-clipping-evidence-v1"
FIT_VALIDITY_POLICY = "darsia-exact-cfa-candidate-channel-validity-v1"
_OPENCV_VERSION = "4.13.0"
_EA_SOURCE = (
    "https://github.com/opencv/opencv/blob/4.13.0/"
    "modules/imgproc/src/demosaicing.cpp#L1599-L1741"
)
# The candidate rectangles below must track the original sampler, not silently
# turn into a different ColorChecker measurement if that implementation changes.
_PATCH_SAMPLER_SHA256 = "0c98107c2679a49e2431e7c1597395b73a433a1715d077ef377612250b60fbed"


@dataclass(frozen=True)
class CfaClipping:
    """Ephemeral arrays, deliberately separate from JSON-serializable metadata."""

    mosaic_dn: np.ndarray
    flags: np.ndarray  # bit 0: proven clipped; bit 1: original sample unknown
    phase_rgb: np.ndarray
    basis: Mapping[str, Any]


def recover_cfa_clipping(sensor, parameters: Mapping) -> CfaClipping:
    """Recover exact original>=LibRaw white-level flags at known CFA sites.

    Input is the existing ``raw_positioning.Sensor`` and its registration
    parameters. No file IO, RAW decode, normalization or RGB cutoff occurs.
    """
    if cv2.__version__ != _OPENCV_VERSION:
        raise ValueError("CFA recovery is verified only for OpenCV 4.13.0 uint16 EA")
    rgb = np.asarray(sensor.rgb)
    policy = parameters["common"]["sensor"]
    pattern = np.asarray(policy["cfa_pattern"], dtype=int)
    description = policy["color_description"]
    if (rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 5 or pattern.shape != (2, 2)
            or list(rgb.shape[1::-1]) != list(policy["raw_wh"])):
        raise ValueError("CFA evidence requires the declared uint16 full-sensor RGB grid")
    try:
        phase = np.array([["RGB".index(description[index]) for index in row]
                          for row in pattern], dtype=np.int8)
    except (IndexError, ValueError, TypeError) as error:
        raise ValueError("CFA colour identities must describe a two-green RGB Bayer array") from error
    # Bayer's equal diagonal must be green; R and B form the other diagonal.
    if not ((phase[0, 0] == phase[1, 1] == 1
             and sorted([int(phase[0, 1]), int(phase[1, 0])]) == [0, 2])
            or (phase[0, 1] == phase[1, 0] == 1
                and sorted([int(phase[0, 0]), int(phase[1, 1])]) == [0, 2])):
        raise ValueError("Only RGB Bayer CFA layouts have verified EA support")
    code = parameters["edge_aware"]["demosaic_code"]
    if not isinstance(code, str) or not code.endswith("_EA"):
        raise ValueError("CFA evidence requires the declared edge-aware demosaicer")
    # Independent colour-site fixture protects the sometimes surprising OpenCV
    # Bayer-code names; its samples never come from the acquisition being tested.
    fixture = np.empty((8, 10), np.uint16)
    for row in range(2):
        for col in range(2):
            fixture[row::2, col::2] = (3000, 2000, 1000)[phase[row, col]]
    decoded = cv2.demosaicing(fixture, getattr(cv2, code))
    if not np.array_equal(decoded[2, 2], [3000, 2000, 1000]):
        raise ValueError("Declared CFA and EA code do not retain RGB colour identity")
    black, white = (int(sensor.metadata[key]) for key in ("black_level", "white_level"))
    if (black != int(policy["black_level"]) or not 0 <= black < white
            or white - black >= 2**14
            or any(int(v) != black for v in policy["black_levels"])):
        raise ValueError("CFA evidence requires the existing common-black integer sensor contract")
    source = deepcopy(dict(sensor.source))
    digest = source.get("sha256", "")
    if not source.get("path") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("CFA evidence requires an exact original RAW source binding")
    mosaic = np.empty(rgb.shape[:2], np.uint16)
    for row in range(2):
        for col in range(2):
            mosaic[row::2, col::2] = rgb[row::2, col::2, phase[row, col]]
    flags = (mosaic >= white - black).astype(np.uint8)
    flags[[0, -1], :] = 2
    flags[:, [0, -1]] = 2
    total = int(sensor.metadata["sensor_saturated_count"])
    known_clipped = int(np.count_nonzero(flags & 1))
    unknown = int(np.count_nonzero(flags & 2))
    if not known_clipped <= total <= known_clipped + unknown:
        raise ValueError("Recovered CFA clipping disagrees with the original RAW count")
    basis = dict(contract=CONTRACT, source=source, black_level=black,
        declared_raw_white_level=white, clipping_criterion="original CFA DN >= LibRaw white_level",
        cfa_pattern=pattern.tolist(), color_description=description,
        phase_rgb=phase.tolist(), sensor_shape_hw=list(mosaic.shape),
        original_sensor_clipped_count=total, recovered_interior_clipped_count=known_clipped,
        unknown_border_photosites=unknown, known_region="rows 1:H-1, columns 1:W-1",
        recovered_clipped_by_cfa_phase=[
            [int(np.count_nonzero(flags[row::2, col::2] & 1)) for col in range(2)]
            for row in range(2)],
        profile_id=sensor.profile_id,
        registration_parameters_sha256=sha256(json.dumps(parameters, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        demosaic_code=code, opencv_version=cv2.__version__, ea_source=_EA_SOURCE)
    for array in (mosaic, flags, phase):
        array.setflags(write=False)
    return CfaClipping(mosaic, flags, phase, basis)


def ea_clipping_support(cfa: CfaClipping, *, tile_rows: int = 128) -> np.ndarray:
    """Map measured CFA flags through the actual uint16 EA sampling branches.

    A bit in the returned sensor RGB grid means at least one *selected numeric*
    source sample has that bit. EA's direction-comparison-only samples are not
    counted as numeric contributors. If its branch cannot be recovered because
    it needs a lost border sample, the result is explicitly unknown.
    """
    mosaic, flags, phase = cfa.mosaic_dn, cfa.flags, cfa.phase_rgb
    height, width = mosaic.shape
    if tile_rows < 1:
        raise ValueError("A positive diagnostic tile size is required")
    output = np.zeros((height, width, 3), np.uint8)
    for y0 in range(1, height - 1, tile_rows):
        y1 = min(y0 + tile_rows, height - 1)
        centre = flags[y0:y1, 1:-1]
        left, right = flags[y0:y1, :-2], flags[y0:y1, 2:]
        up, down = flags[y0 - 1:y1 - 1, 1:-1], flags[y0 + 1:y1 + 1, 1:-1]
        horizontal, vertical = left | right, up | down
        diagonal = (flags[y0 - 1:y1 - 1, :-2] | flags[y0 - 1:y1 - 1, 2:]
                    | flags[y0 + 1:y1 + 1, :-2] | flags[y0 + 1:y1 + 1, 2:])
        dx = np.abs(mosaic[y0:y1, :-2].astype(np.int32) - mosaic[y0:y1, 2:])
        dy = np.abs(mosaic[y0 - 1:y1 - 1, 1:-1].astype(np.int32)
                    - mosaic[y0 + 1:y1 + 1, 1:-1])
        green = np.where(dx > dy, vertical, horizontal).astype(np.uint8)
        branch_unknown = ((horizontal | vertical) & 2) != 0
        # Only clipping shared by both possible branches can be proved when
        # the original edge-gradient comparison itself is unavailable.
        green[branch_unknown] = 2 | (horizontal[branch_unknown] & vertical[branch_unknown] & 1)
        rows, cols = np.ogrid[y0:y1, 1:width - 1]
        colours = phase[rows % 2, cols % 2]
        neighbour_colour = phase[rows % 2, (cols + 1) % 2]
        for channel in range(3):
            values = np.where(colours == channel, centre,
                np.where(colours == 1,
                    np.where(neighbour_colour == channel, horizontal, vertical),
                    green if channel == 1 else diagonal))
            output[y0:y1, 1:-1, channel] = values
    # Exactly the RGB border replication in OpenCV's ushort EA implementation.
    output[1:-1, 0] = output[1:-1, 1]
    output[1:-1, -1] = output[1:-1, -2]
    output[0] = output[1]
    output[-1] = output[-2]
    return output


def _patch_candidates(values, roi_rc):
    """Replay the original geometric sampler up to its 24 k-means inputs."""
    import darsia
    sampler = darsia.CustomColorChecker._extract_from_image
    source = inspect.getsource(sampler).replace("\r\n", "\n").encode()
    if sha256(source).hexdigest() != _PATCH_SAMPLER_SHA256:
        raise ValueError("The original ColorChecker candidate sampler changed; support needs reverification")
    extractor = darsia.ColorCorrection(config={"roi": np.asarray(roi_rc).tolist(),
                                             "active": True, "clip": False})
    restricted = extractor._restrict_to_roi(values)
    warped = darsia.extract_quadrilateral_ROI(restricted, pts_src=None, width=27.3, height=17.8)
    height, width = warped.shape[:2]
    resized = cv2.resize(warped, (500, int(height / width * 500)))
    return [[resized[row:row + 50, col:col + 50] for col in (12, 95, 177, 260, 344, 427)]
            for row in (12, 93, 175, 255)]


def chart_clipping_diagnostics(sensor, parameters: Mapping, roi_rc) -> dict[str, Any]:
    """Return small JSON evidence for the actual 24 candidate sampling regions.

    Counts are affected patch *pixels*, separately per RGB channel, not unique
    CFA photosites, selected k-means members or saturation-based fit exclusions.
    """
    from .raw_positioning import sample_rgb_values

    cfa = recover_cfa_clipping(sensor, parameters)
    roi = np.asarray(roi_rc)
    camera_width, camera_height = parameters["common"]["sensor"]["camera_wh"]
    if (roi.shape != (4, 2) or not np.isfinite(roi).all()
            or np.any(roi < 0) or np.any(roi[:, 0] >= camera_height)
            or np.any(roi[:, 1] >= camera_width) or not np.array_equal(roi, roi.astype(int))):
        raise ValueError("Clipping evidence requires the actual four integer camera-U ROI points")
    support = ea_clipping_support(cfa)
    counts = {}
    for bit, name in ((1, "clipped_source_affected_pixels"), (2, "unknown_source_affected_pixels")):
        values = ((support & bit) != 0).astype(np.uint8)
        sampled = sample_rgb_values(sensor, values, np.eye(3), shape_hw=(camera_height, camera_width),
                                    parameters=parameters)
        # A nonfinite lens sample has no proven complete source support.
        finite = np.isfinite(sampled)
        sampled = np.where(finite, sampled, 1. if bit == 2 else 0.)
        del values
        patches = _patch_candidates(np.moveaxis(sampled, 0, -1), roi)
        counts[name] = [[np.count_nonzero(patch > 0, axis=(0, 1)).tolist() for patch in row]
                        for row in patches]
        del sampled, patches
    return dict(status="available", **deepcopy(dict(cfa.basis)), roi_rc=roi.astype(int).tolist(), **counts,
        patch_shape_hw=[50, 50], patch_axis_order="row, column, RGB channel",
        sampling_scope="All 2500 candidate pixels per patch, before dominant-cluster selection",
        evidence_scope="Selected numeric CFA sources through EA, per-channel lens/CA and original ROI/resize",
        candidate_sampler_sha256=_PATCH_SAMPLER_SHA256,
        unique_cfa_photosite_counts_available=False, dominant_cluster_support_available=False,
        correction_or_fit_policy_changed=False)


def chart_fit_validity(sensor, parameters: Mapping, roi_rc, *, camera_rgb=None):
    """Explicit conservative fit input, bound to this already decoded RAW.

    Any clipped or unproven numeric source excludes its whole candidate
    patch/channel. This is deliberately stricter than dominant-cluster-only
    exclusion. A global original-CFA count of zero proves that even the lost
    border has no clipped samples, and avoids dense EA/lens reconstruction.
    """
    if camera_rgb is not None and sensor.metadata.get("sensor_saturated_count") == 0:
        values = np.asarray(camera_rgb)
        expected = parameters["common"]["sensor"]["camera_wh"]
        roi = np.asarray(roi_rc)
        source = deepcopy(dict(sensor.source))
        digest = source.get("sha256", "")
        if (values.shape != (3, int(expected[1]), int(expected[0]))
                or roi.shape != (4, 2) or not np.isfinite(roi).all()
                or not np.array_equal(roi, roi.astype(int)) or np.any(roi < 0)
                or np.any(roi[:, 0] > int(expected[1])) or np.any(roi[:, 1] > int(expected[0]))
                or not source.get("path") or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("CFA validity requires this RAW's exact complete camera-U basis")
        black, white = (int(sensor.metadata[key]) for key in ("black_level", "white_level"))
        if not 0 <= black < white:
            raise ValueError("CFA validity needs the original positive sensor white range")
        # Replay only the small chart crop/warps, directly from the already
        # rendered photograph. No extra full-sensor support maps or RAW IO.
        patches = _patch_candidates(np.moveaxis(values, 0, -1), roi_rc)
        unknown = [[np.count_nonzero(~np.isfinite(patch), axis=(0, 1)).tolist()
                    for patch in row] for row in patches]
        evidence = dict(contract=CONTRACT, status="available", source=source,
            black_level=black, declared_raw_white_level=white,
            clipping_criterion="original CFA DN >= LibRaw white_level",
            original_sensor_clipped_count=0,
            cfa_pattern=deepcopy(parameters["common"]["sensor"]["cfa_pattern"]),
            color_description=parameters["common"]["sensor"]["color_description"],
            registration_parameters_sha256=sha256(json.dumps(parameters, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            roi_rc=np.asarray(roi_rc).tolist(), patch_shape_hw=[50, 50],
            clipped_source_affected_pixels=np.zeros((4, 6, 3), int).tolist(),
            unknown_source_affected_pixels=unknown,
            support_method="Original sensor has zero clipped CFA sites; actual candidate finiteness checked",
            sampling_scope="All 2500 candidate pixels per patch, before dominant-cluster selection",
            unique_cfa_photosite_counts_available=False, dominant_cluster_support_available=False)
    else:
        evidence = chart_clipping_diagnostics(sensor, parameters, roi_rc)
    clipped = np.asarray(evidence["clipped_source_affected_pixels"])
    unknown = np.asarray(evidence["unknown_source_affected_pixels"])
    validity = (clipped == 0) & (unknown == 0)
    if validity.shape != (4, 6, 3):
        raise ValueError("CFA validity must have four rows, six columns and RGB channels")
    evidence.update(fit_validity_policy=FIT_VALIDITY_POLICY,
        exclusion_scope="Entire candidate patch/channel if ANY clipped or unknown numeric source contributes",
        valid_patch_channels=validity.tolist())
    return validity, evidence
