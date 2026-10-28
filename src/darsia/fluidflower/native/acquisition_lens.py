"""Lens tables that a camera records in each RAW acquisition.

Some cameras record their lens-correction tables per exposure, so a small zoom
or focus change gives an acquisition other tables than the calibrated profile.
Those recorded tables then describe that acquisition. The project configuration
still defines where the tables are stored, their structure and the profile's
camera-U coefficients; the configured reader verifies every table.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import tifffile

from . import registration
from ..pipeline.job_queue import retry_transient


@dataclass(frozen=True)
class AcquisitionLens:
    """Registration parameters bound to one acquisition's recorded tables."""

    parameters: Mapping
    provenance: Mapping


def recorded_lens_tables(path: str | Path, lens: Mapping) -> tuple[tuple[int, ...], ...]:
    """Return the distortion and channel tables recorded in one RAW file.

    Only the channel tag is read here, to give the configured reader its
    expectation; that reader then verifies both tables' structure.
    """
    with tifffile.TiffFile(path) as tiff:
        sub = tiff.pages[lens["main_page"]].pages[lens["sub_page"]]
        channel = tuple(int(v) for v in sub.tags[lens["ca_tag"]].value)
    if len(channel) < lens["ca_min_length"] or channel[0] != lens["ca_header"]:
        raise ValueError("Unexpected sensor lens-table structure")
    expected = dict(lens, ca_red=list(channel[slice(*lens["red_slice"])]),
                    ca_blue=list(channel[slice(*lens["blue_slice"])]))
    return registration.read_lens_tables(path, parameters=expected)


def acquisition_lens(path: str | Path, parameters: Mapping, *,
                     profile_id: str) -> AcquisitionLens | None:
    """Bind registration parameters to the tables recorded in ``path``.

    Returns ``None`` when the recorded tables equal the configured profile, so
    callers keep the unchanged configuration. Otherwise only the expected
    distortion table of ``profile_id`` and the expected channel tables differ
    in the returned private copy; coefficients and sensor checks are unchanged.
    """
    common = parameters["common"]
    lens, profile = common["lens"], common["profiles"][profile_id]
    distortion, red, blue = retry_transient(lambda: recorded_lens_tables(path, lens))
    if (distortion == tuple(profile["distortion"]) and red == tuple(lens["ca_red"])
            and blue == tuple(lens["ca_blue"])):
        return None
    own = dict(parameters, common=deepcopy(common))
    own["common"]["lens"].update(ca_red=list(red), ca_blue=list(blue))
    own["common"]["profiles"][profile_id]["distortion"] = list(distortion)
    return AcquisitionLens(own, dict(origin="recorded in this acquisition",
        profile_id=profile_id, distortion=list(distortion), ca_red=list(red),
        ca_blue=list(blue)))
