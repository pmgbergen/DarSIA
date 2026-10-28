"""The nested raw-image tiers of a series (S50 inside S100 inside S200).

A series declares its tiers in a JSON file (format ``darsia-published-tiers-v1``,
by default ``image_selection/published_tiers.json`` in the series folder)::

    {"format": "darsia-published-tiers-v1",
     "tiers": {"S50": {"runs": {"<run>": ["<raw file name or SHA-256>", ...]}},
               "S100": {...}, "S200": {...}}}

Under a time-ordered history policy every selected image builds the history,
so a frame's result depends on the frames before it. A tier is therefore
reproducible only from the job that ran exactly its frames: the builder
refuses a tier whose frames differ from its job's, and tiers that do not nest.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from .layout import TIER_ORDER, require_tier

TIERS_FORMAT = "darsia-published-tiers-v1"
DEFAULT_TIERS_FILE = "image_selection/published_tiers.json"


def read_tiers(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Read a tier file: {tier: {run: [frame keys in the listed order]}}."""
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or document.get("format") != TIERS_FORMAT:
        raise ValueError(f"A {TIERS_FORMAT} tier file is required: {path}")
    tiers = document.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError(f"The tier file declares no tiers: {path}")
    result = {}
    for tier, value in tiers.items():
        require_tier(tier)
        runs = (value or {}).get("runs") if isinstance(value, dict) else None
        if not isinstance(runs, dict) or not runs:
            raise ValueError(f"Tier {tier} lists no runs")
        result[tier] = {}
        for run, frames in runs.items():
            if (not isinstance(frames, list) or not frames
                    or any(not isinstance(frame, str) or not frame.strip() for frame in frames)):
                raise ValueError(f"Tier {tier}, run {run}: a nonempty list of frames is required")
            if len({frame.casefold() for frame in frames}) != len(frames):
                raise ValueError(f"Tier {tier}, run {run} lists a frame twice")
            result[tier][str(run)] = [frame.strip() for frame in frames]
    return result


def frame_key(value: str, frames: list[dict]) -> int | None:
    """The index of the job frame a tier entry names (by SHA-256 or file name), else None."""
    if re.fullmatch(r"[0-9a-f]{64}", value):
        matches = [i for i, frame in enumerate(frames) if frame["raw"]["sha256"] == value]
    else:
        matches = [i for i, frame in enumerate(frames)
                   if frame["raw"]["name"].casefold() == value.casefold()]
    return matches[0] if len(matches) == 1 else None


def nesting_problems(tiers: dict[str, dict[str, list[str]]]) -> list[str]:
    """Why declared tiers do not nest (S50 inside S100 inside S200); empty when they do."""
    problems = []
    declared = [tier for tier in TIER_ORDER if tier in tiers]
    for smaller, larger in zip(declared, declared[1:]):
        for run, frames in tiers[smaller].items():
            outer = {frame.casefold() for frame in tiers[larger].get(run, ())}
            if run not in tiers[larger]:
                problems.append(f"Tier {smaller} has run {run}, which {larger} lacks")
                continue
            missing = [frame for frame in frames if frame.casefold() not in outer]
            if missing:
                problems.append(f"Tier {smaller}, run {run}: {len(missing)} frame(s) not in "
                                f"{larger}, first {missing[0]}")
    return problems


def tier_job_problems(tier: str, tiers: dict, job_frames: dict[str, list[dict]]) -> list[str]:
    """Why a tier's frames differ from the frames of the job that made its results.

    ``job_frames`` maps each run of the job to its frames in time order, each
    with ``raw`` = {name, sha256}. The runs and, per run, the frames must be
    exactly the same.
    """
    if tier not in tiers:
        return [f"The tier file does not declare {tier}"]
    problems = []
    declared = tiers[tier]
    if set(declared) != set(job_frames):
        missing, extra = sorted(set(job_frames) - set(declared)), sorted(set(declared) - set(job_frames))
        if missing:
            problems.append(f"Tier {tier} lacks the job's runs: {', '.join(missing)}")
        if extra:
            problems.append(f"Tier {tier} has runs the job did not run: {', '.join(extra)}")
    for run in sorted(set(declared) & set(job_frames)):
        frames = job_frames[run]
        indices = [frame_key(value, frames) for value in declared[run]]
        unknown = [value for value, index in zip(declared[run], indices) if index is None]
        if unknown:
            problems.append(f"Tier {tier}, run {run}: {len(unknown)} frame(s) the job did not "
                            f"analyse, first {unknown[0]}")
        if sorted(index for index in indices if index is not None) != list(range(len(frames))):
            problems.append(f"Tier {tier}, run {run}: {len(declared[run])} frame(s) listed, the "
                            f"job analysed {len(frames)}; the expected results need exactly the job's frames")
    return problems
