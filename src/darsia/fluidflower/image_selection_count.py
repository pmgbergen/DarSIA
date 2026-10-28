"""Extend a saved image selection to an exact count using protocol metadata."""

from bisect import bisect_left, bisect_right, insort
from copy import deepcopy
import math

from darsia.fluidflower.image_selection_proposal import (
    _key,
    _quality_classes,
    eligible_frames,
    first_usable_rule,
)


def propose_image_count(metadata, retained_paths, count=80, *, quality=None):
    """Retain every saved image and fill equally weighted original intervals.

    Each original interval has length one in selection coordinates, regardless
    of its duration. Repeatedly choose the available image farthest from an
    already selected image in those coordinates, with earlier times winning
    ties. The first and last retained images bound the entire proposal.
    Unrecorded quality remains explicitly unrecorded. No image file is read.
    """
    if type(count) is not int or count < 1:
        raise ValueError("Image count must be a positive integer.")
    if retained_paths is None or isinstance(retained_paths, (str, bytes)):
        raise ValueError("Provide the saved selection as a list of image paths.")
    retained_keys = [_key(path) for path in retained_paths]
    if not retained_keys:
        raise ValueError("A saved image selection is required before extending it.")
    if len(set(retained_keys)) != len(retained_keys):
        raise ValueError("The saved selection contains duplicate source paths.")
    if count < len(retained_keys):
        raise ValueError("Image count cannot be smaller than the saved selection.")

    rows = eligible_frames(metadata)
    classes = _quality_classes(metadata, quality)
    available = {_key(row["source_path"]): row for row in rows}
    retained = []
    for key in retained_keys:
        if key not in available:
            raise ValueError(
                "A retained image is missing, blacklisted or unavailable: " + key
            )
        if classes.get(key, {}).get("class") == "reject":
            raise ValueError("A retained image has saved quality reject: " + key)
        retained.append(available[key])
    retained.sort(key=lambda row: (row["elapsed_h"], row["input_index"]))
    times = [row["elapsed_h"] for row in retained]
    if len(set(times)) != len(times):
        raise ValueError("Retained images must have distinct acquisition times.")
    first_usable_rule(metadata, retained, quality)

    retained_keys = set(retained_keys)
    chosen = {row["elapsed_h"]: row for row in retained}
    candidates = {}
    for row in rows:
        time = row["elapsed_h"]
        if (
            times[0] < time < times[-1]
            and time not in chosen
            and classes.get(_key(row["source_path"]), {}).get("class") != "reject"
        ):
            # Protocol order resolves distinct images with the same camera time.
            candidates.setdefault(time, row)
    if len(chosen) + len(candidates) < count:
        raise ValueError(
            f"Only {len(chosen) + len(candidates)} eligible distinct acquisition "
            f"times exist within the saved selection; {count} images are required."
        )

    positions = list(range(len(times)))
    located = []
    for time, row in candidates.items():
        interval = bisect_right(times, time) - 1
        position = interval + (time - times[interval]) / (
            times[interval + 1] - times[interval]
        )
        located.append((position, row))
    while len(chosen) < count:
        best_index, best_distance = None, -1.0
        for index, (position, row) in enumerate(located):
            right = bisect_left(positions, position)
            distance = min(position - positions[right - 1], positions[right] - position)
            # Equal fractional positions can round differently across intervals.
            # Traversal is in time order, so retain the earlier candidate on ties.
            if distance > best_distance and not math.isclose(
                distance, best_distance, rel_tol=0.0, abs_tol=1e-12
            ):
                best_index, best_distance = index, distance
        position, row = located.pop(best_index)
        chosen[row["elapsed_h"]] = row
        insort(positions, position)

    selected = []
    for time in sorted(chosen):
        row = deepcopy(chosen[time])
        key = _key(row["source_path"])
        row.update(
            retained=key in retained_keys,
            quality=(
                deepcopy(classes[key])
                if key in classes
                else {"class": "Not recorded", "recorded": False}
            ),
        )
        selected.append(row)
    return dict(
        selected_paths=[row["source_path"] for row in selected],
        retained_paths=[row["source_path"] for row in retained],
        count=len(selected),
        selected=selected,
        unrecorded_quality=sum(
            _key(row["source_path"]) not in classes for row in selected
        ),
    )
