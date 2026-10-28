"""Read saved native phase-area summaries for review-list ordering.

The review UI uses these small, model-bound reports instead of decoding phase
map PNGs or running inference while an experiment list is being sorted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def peak_phase_area_percent(candidate: Any) -> dict[str, float] | None:
    """Return peak visible-domain area (%) over a run's native frames.

    Mixed pixels, where present, support both phase categories. A missing or
    incomplete report is unknown rather than zero. The folder is taken from
    the selected candidate, so switching models switches the area source.
    """
    phase_dir = getattr(candidate, "phase_map_output_dir", None)
    dataset = str(getattr(candidate, "dataset", "") or "").casefold()
    run = str(getattr(candidate, "run", "") or "").casefold()
    if phase_dir is None or not dataset or not run:
        return None
    from darsia.fluidflower.phase_blind_review_policy import predictions_withheld
    if predictions_withheld(phase_dir, run):
        return None
    report_path = Path(phase_dir).parent / "physics_reports" / f"{dataset}_{run}.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(report, dict):
        return None
    denominator = report.get("phase_area_denominator") or {}
    if isinstance(denominator, dict) and denominator.get("kind") == "frame-native-rig-domain":
        return _weighted_peak(report, candidate)
    if not isinstance(denominator, dict) or denominator.get("kind") not in {
            "porous-domain", "frame-native-porous-domain",
            "frame-native-observation-domain"}:
        return None
    moving_domain = denominator["kind"] in {
        "frame-native-porous-domain", "frame-native-observation-domain"}
    frame_area_field = ("observation_domain_pixels"
                        if denominator["kind"] == "frame-native-observation-domain" else
                        "porous_domain_pixels")
    try:
        domain_pixels = None if moving_domain else int(denominator["pixels"])
        frames = report["frames"]
        expected_count = int(candidate.frame_count)
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    if ((domain_pixels is not None and domain_pixels <= 0) or
            not isinstance(frames, list) or len(frames) != expected_count or not frames):
        return None
    peaks = {"gas": 0.0, "aqueous": 0.0}
    seen: set[int] = set()
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("phase_area_counts"), dict):
            return None
        try:
            index = int(frame["frame_index"])
            counts = frame["phase_area_counts"]
            gas = int(counts["gas"])
            aqueous = int(counts["aqueous"])
            mixed = int(counts.get("mixed", 0))
            frame_domain_pixels = (int(frame[frame_area_field])
                                   if moving_domain else domain_pixels)
        except (KeyError, ValueError, TypeError):
            return None
        if index in seen or not 0 <= index < expected_count:
            return None
        seen.add(index)
        if (frame_domain_pixels <= 0 or min(gas, aqueous, mixed) < 0 or
                gas + aqueous + mixed > frame_domain_pixels):
            return None
        peaks["gas"] = max(peaks["gas"], 100.0 * (gas + mixed) / frame_domain_pixels)
        peaks["aqueous"] = max(peaks["aqueous"], 100.0 * (aqueous + mixed) / frame_domain_pixels)
    if len(seen) != expected_count or any(not math.isfinite(value) for value in peaks.values()):
        return None
    return peaks


def _weighted_peak(report, candidate):
    """Read physical-area fractions without substituting a pixel-count ratio."""
    try:
        count = int(candidate.frame_count)
        frames = report["frames"]
        if not isinstance(frames, list) or len(frames) != count or not count:
            return None
        peaks, seen = {"gas": 0., "aqueous": 0.}, set()
        for frame in frames:
            index = int(frame["frame_index"])
            values = frame["phase_area_cm2"]
            gas, aqueous, mixed = (float(values.get(k, 0)) for k in ("gas", "aqueous", "mixed"))
            domain = float(frame["rig_domain_area_cm2"])
            if (index in seen or not 0 <= index < count or domain <= 0
                    or not all(math.isfinite(v) for v in (gas, aqueous, mixed, domain))
                    or min(gas, aqueous, mixed) < 0 or gas + aqueous + mixed > domain * (1 + 1e-9)):
                return None
            seen.add(index)
            for key, value in (("gas", gas), ("aqueous", aqueous)):
                peaks[key] = max(peaks[key], 100. * (value + mixed) / domain)
        return peaks
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def _run_number(run: str) -> tuple[int, str]:
    """Preserve stable acquisition ordering by numeric digits then identifier."""
    digits = "".join(character for character in run if character.isdigit())
    return (int(digits) if digits else 10 ** 9, run)


def sorted_phase_area_runs(
    entries: list[tuple[str, str, str, int]],
    areas: dict[str, dict[str, dict[str, float] | None]],
    *,
    side: str,
    phase: str,
    largest_first: bool,
) -> list[tuple[str, str, str, int]]:
    """Sort known reports first, unknown reports last, then stable run number."""

    def run_number(run: str) -> tuple[int, str]:
        return _run_number(run)

    def key(entry: tuple[str, str, str, int]) -> tuple[bool, float, tuple[int, str]]:
        report = areas.get(entry[0], {}).get(side)
        value = None if report is None else report.get(phase)
        return (
            value is None,
            -(value or 0.0) if largest_first else (value or 0.0),
            run_number(entry[0]),
        )

    return sorted(entries, key=key)
