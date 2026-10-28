"""Connect verified stage-1 colour coordinates to the existing stage-2 API.

The caller supplies its calibration, geometry, masks and cycle budget inputs.
This module reads only the recorded colour-coordinate artifact; it does not
open photographs, infer field names, fit a calibration or publish results.
Geometry may come from the caller's existing stored-product reader. Pore volume
is explicitly in m3, as required by :class:`FrameInputs`.

The original ``process_frame`` and ``process_cycle`` perform all titration,
censoring, background subtraction and budget arithmetic. Nothing installs this
optional post-processing route into a stage-1 pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike

import numpy as np

from darsia.fluidflower.stage2_postprocessing import (
    BlueFractionAdapter,
    BudgetTerms,
    CycleResult,
    FrameInputs,
    FrameResult,
    Stage2Config,
    blue_fraction_from_colour,
    process_cycle,
    process_frame,
    read_colour_coordinate,
)

__all__ = [
    "StoredCoordinateFrame",
    "read_coordinate_inputs",
    "process_coordinate_frame",
    "process_coordinate_cycle",
]


@dataclass(frozen=True, kw_only=True, eq=False)
class StoredCoordinateFrame:
    """A recorded coordinate and the caller's explicitly selected frame fields.

    Every role is required. Pass None for an unavailable optional mask or pixel
    area, exactly as for ``FrameInputs``. ``frame_id`` is the caller's frame key;
    ``frame_directory`` contains the stage-1 result.json and the coordinate it
    records. The existing reader verifies that artifact's SHA, size and grid.
    Array validation runs when the frame is read, through ``FrameInputs``.
    """

    frame_id: str
    frame_directory: str | PathLike
    labels: np.ndarray
    pore_volume_m3: np.ndarray
    valid_colour_mask: np.ndarray | None
    gas_mask: np.ndarray | None
    outside_mask: np.ndarray | None
    pixel_area_m2: np.ndarray | None


def read_coordinate_inputs(
    frame: StoredCoordinateFrame, *, adapter: BlueFractionAdapter
) -> FrameInputs:
    """Read the verified coordinate once and apply the caller's calibration.

    A calibration must retain the stored coordinate's grid and invalid support.
    It may invalidate additional colour, but may not turn a missing coordinate
    into a quantified value. The original blue-fraction and FrameInputs checks
    enforce the remaining range, dtype, mask and geometry contracts.
    """
    if not isinstance(frame, StoredCoordinateFrame):
        raise ValueError("frame must be a StoredCoordinateFrame")
    colour = read_colour_coordinate(frame.frame_directory)
    shape = colour.shape
    invalid = ~np.isfinite(colour)
    fraction = blue_fraction_from_colour(colour, adapter)
    if fraction.shape != shape:
        raise ValueError("the calibration must retain the stored coordinate grid")
    if np.isfinite(fraction[invalid]).any():
        raise ValueError("the calibration must retain invalid coordinate support")
    return FrameInputs(
        frame_id=frame.frame_id,
        blue_fraction=fraction,
        labels=frame.labels,
        pore_volume_m3=frame.pore_volume_m3,
        valid_colour_mask=frame.valid_colour_mask,
        gas_mask=frame.gas_mask,
        outside_mask=frame.outside_mask,
        pixel_area_m2=frame.pixel_area_m2,
    )


def process_coordinate_frame(
    config: Stage2Config,
    frame: StoredCoordinateFrame,
    *,
    adapter: BlueFractionAdapter,
    regions: Mapping[str, np.ndarray] | None = None,
) -> FrameResult:
    """Evaluate one recorded coordinate with the original stage-2 method."""
    return process_frame(
        config, read_coordinate_inputs(frame, adapter=adapter), regions=regions
    )


def process_coordinate_cycle(
    config: Stage2Config,
    *,
    cycle_run: str,
    baseline: StoredCoordinateFrame,
    frames: Iterable[tuple[StoredCoordinateFrame, BudgetTerms]],
    adapter: BlueFractionAdapter,
    regions: Mapping[str, np.ndarray] | None = None,
) -> CycleResult:
    """Read a baseline and stream stored frames into the original cycle budget.

    ``frames`` may be a generator. The baseline is read first; subsequent colour
    arrays are read one at a time when the original ``process_cycle`` requests
    them. Only its usual baseline maps and frame summaries are retained. Terms,
    including unknown scale/overflow values, are passed through unchanged.
    """

    def read_frames():
        for item in frames:
            if not (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], StoredCoordinateFrame)
                and isinstance(item[1], BudgetTerms)
            ):
                raise ValueError(
                    "frames must yield (StoredCoordinateFrame, BudgetTerms) pairs"
                )
            yield read_coordinate_inputs(item[0], adapter=adapter), item[1]

    return process_cycle(
        config,
        cycle_run=cycle_run,
        baseline=read_coordinate_inputs(baseline, adapter=adapter),
        frames=read_frames(),
        regions=regions,
    )
