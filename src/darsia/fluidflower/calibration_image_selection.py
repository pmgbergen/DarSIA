"""Mass-fit observations, separate from phase history and optical-path samples."""
from copy import deepcopy
import math
from pathlib import Path
import tomllib

DEFAULT_IMAGES_PER_RUN = 30
SELECTION_METHOD = "common-nominal-log-targets-v1"


def calibration_image_count(document):
    value = document.get("calibration", {}).get("mass", {}).get("image_selection", {}).get(
        "max_images_per_run", DEFAULT_IMAGES_PER_RUN)
    if type(value) is not int or not 3 <= value <= 1000:
        raise ValueError("Calibration needs between 3 and 1000 target images per run.")
    return value


def _targets(times, count, *, required=()):
    """Choose existing nominal targets on one shared log-time grid."""
    times = sorted(set(map(float, times)))
    if not times:
        return []
    if len(times) <= count:
        return times
    positive = [time for time in times if time > 0]
    first = min(positive, default=1.)
    coordinate = lambda time: math.log1p(max(0., time) / first)
    chosen = list(dict.fromkeys(time for time in required if time in times))[:count]
    # Endpoints and stage representatives are fixed; then spread observations
    # over elapsed time. Fill duplicates by the largest remaining log-time gap.
    for index in range(count):
        target = coordinate(times[0]) + index / (count - 1) * (
            coordinate(times[-1]) - coordinate(times[0]))
        nearest = min(times, key=lambda time: (abs(coordinate(time) - target), time))
        if nearest not in chosen and len(chosen) < count:
            chosen.append(nearest)
    while len(chosen) < count:
        remaining = [time for time in times if time not in chosen]
        chosen.append(max(remaining, key=lambda time: (
            min(abs(coordinate(time) - coordinate(other)) for other in chosen), -time)))
    return sorted(chosen)


def calibration_statistics_selection(statistics, rules, count=DEFAULT_IMAGES_PER_RUN):
    """Subset fit rows, retaining each run's original anchor as a support row.

    P1's chronological source sequence and files are never edited. The rules'
    target list restricts the numerical objective; an extra anchor row supports
    dilution only and receives no objective weight unless also a chosen target.
    """
    calibration_image_count({"calibration": {"mass": {"image_selection": {"max_images_per_run": count}}}})
    allowed = rules.get("target_times_h")
    low, high = rules["stage1_window_h"]
    late = rules["stage2_start_h"]
    def eligible(time):
        return (allowed is None or time in allowed) and (low <= time <= high or time >= late)
    times = []
    anchors = {}
    for run, entries in statistics.items():
        if not entries:
            raise ValueError("No eligible calibration statistics for " + run)
        for entry in entries:
            time = entry.get("selected_target_h")
            if type(time) not in {int, float} or not math.isfinite(time):
                raise ValueError("Calibration statistics need finite recorded nominal times.")
            if eligible(time):
                times.append(time)
        anchors[run] = min(entries, key=lambda entry: (
            abs(entry.get("elapsed_h", entry["selected_target_h"]) - rules["anchor_target_h"]),
            entry["frame_index"]))
    times = sorted(set(times))
    early = [time for time in times if low <= time <= high]
    post = [time for time in times if time >= late]
    required = ([early[0]] if early else []) + ([post[0], post[-1]] if post else [])
    targets = _targets(times, count, required=required)
    if not targets:
        raise ValueError("No saved calibration images match the selected fit windows.")
    selected, counts = {}, {}
    for run, entries in statistics.items():
        observations = [entry for entry in entries if entry["selected_target_h"] in targets]
        if not observations:
            raise ValueError("No selected calibration target is available for " + run)
        support = anchors[run]
        rows = list(observations)
        if support["frame_index"] not in {entry["frame_index"] for entry in rows}:
            rows.append(support)
        selected[run] = deepcopy(sorted(rows, key=lambda entry: entry["frame_index"]))
        counts[run] = dict(observations=len(observations), anchor_support=int(len(rows) > len(observations)))
    chosen_rules = deepcopy(rules)
    chosen_rules["target_times_h"] = targets
    record = dict(method=SELECTION_METHOD, max_images_per_run=count, target_hours=targets,
        runs=counts, history_policy="Keep the complete original chronological phase sequence.",
        anchor_policy="Retain the original injection-end anchor; extra support rows do not fit.")
    return selected, chosen_rules, record


def save_calibration_image_selection(root, count, *, expected_bytes, context_current=None):
    from .series_method import ManifestChangedError, _write_manifest
    from .series_calibration_approval import require_unlocked
    path = Path(root) / "series.toml"
    original = path.read_bytes()
    if original != expected_bytes:
        raise ManifestChangedError("The series changed; reload calibration image selection before saving.")
    document = tomllib.loads(original.decode("utf-8-sig"))
    require_unlocked(document)
    updated = deepcopy(document)
    updated.setdefault("calibration", {}).setdefault("mass", {}).setdefault(
        "image_selection", {})["max_images_per_run"] = count
    calibration_image_count(updated)
    _write_manifest(path, original, updated, context_current=context_current)
    return path.read_bytes()


def calibration_selection_preview(root, cancelled=lambda: False):
    """Only the manifest's recorded image times; no image or result reads."""
    from .pipeline.current_calibration_selection import _saved_target
    root = Path(root)
    path = root / "series.toml"
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("Series metadata exceeds the image-selection preview limit.")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    count = calibration_image_count(document)
    excluded = document.get("runs", {}).get("excluded", {})
    candidates, rows = {}, {}
    for run in document.get("runs", {}).get("ids", []):
        if run in excluded or cancelled():
            continue
        saved = document.get("image_selection", {}).get("runs", {}).get(run, {})
        frames = saved.get("selected", [])
        times = [_saved_target(frame, saved) for frame in frames]
        candidates[run] = [time for time in times if time is not None]
        rows[run] = dict(source_images=len(frames), eligible=len(candidates[run]),
                         available_targets=candidates[run])
    targets = _targets([time for values in candidates.values() for time in values], count)
    for run in rows:
        rows[run]["targets"] = [time for time in targets if time in candidates[run]]
    if path.read_bytes() != raw:
        raise ValueError("The series changed while reading calibration image selection.")
    return dict(root=root, raw=raw, count=count, runs=rows, targets=targets)
