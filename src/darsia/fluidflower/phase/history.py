"""Chronological acquisition history and fixed parent gas-envelope timing."""

import numpy as np


def envelope_ordinal(elapsed_h, first_parent_time_h, second_parent_time_h):
    """Count parent gas envelopes whose camera time strictly precedes a frame.

    Equality does not activate an envelope. The supplied parent camera times
    remain fixed when a selected timeline contains additional acquisitions.
    """
    if (
        not all(np.isfinite([elapsed_h, first_parent_time_h, second_parent_time_h]))
        or second_parent_time_h < first_parent_time_h
    ):
        raise ValueError("Gas-envelope camera times must be finite and ordered")
    return int(elapsed_h > first_parent_time_h) + int(elapsed_h > second_parent_time_h)


def chronological_history(rows, *, policy):
    """Bind each distinct selected acquisition to its camera-time predecessor.

    Source digests own the history; frame numbers and review-anchor status do
    not determine adjacency. The descriptive policy text is supplied as data.
    """
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise ValueError("History must contain each selected acquisition exactly once")
    runs = {row["run"] for row in rows}
    if len(runs) != 1:
        raise ValueError("A chronological history belongs to exactly one run")
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["elapsed_h"]),
            row["camera_datetime"],
            row["source_sha256"],
        ),
    )
    if any(
        float(a["elapsed_h"]) >= float(b["elapsed_h"])
        for a, b in zip(ordered, ordered[1:])
    ):
        raise ValueError(
            "Distinct selected acquisitions require strictly increasing camera times"
        )
    result = []
    previous = None
    for row in ordered:
        result.append(
            dict(
                row,
                history_policy=policy,
                previous_source_sha256=(
                    None if previous is None else previous["source_sha256"]
                ),
            )
        )
        previous = row
    return result


def advance_previous(current, previous, *, selected, baseline):
    """Advance the predecessor for a selected frame or the baseline only.

    A review probe may be inferred against the existing predecessor but must
    leave that predecessor intact for the next selected camera acquisition.
    """
    if selected or baseline:
        return current
    return previous
