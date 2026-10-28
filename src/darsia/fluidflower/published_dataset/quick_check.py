"""Whether the first frames of a run can be checked on their own (the quick check).

The quick check reproduces the first frames of one run, from its baseline. It
is valid only when no step uses a run's later frames. The pipelines below
compute each frame from the frame itself, its predecessor (the history), the
run's baseline and frozen inputs; a run resumes from a verified prefix of
committed frames, which relies on exactly that. A frame's configuration may
still name other frames; this check reports every such reference and finds
the frames that name a later frame.
"""

from __future__ import annotations

# Pipelines whose frames depend only on earlier frames, the baseline and frozen
# inputs (their runner resumes from a verified prefix of committed frames).
PREFIX_PIPELINES = frozenset({"native-phase-n1", "lattice-phase-n1", "n1"})
DEFAULT_QUICK_FRAMES = 3


def _strings(value):
    """Every string (keys and values) of a nested configuration."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def frame_dependencies(config: dict) -> list[dict]:
    """For each frame, the other frames of its run that its configuration names.

    A frame names another frame by that frame's acquisition SHA-256 or case
    id. The result lists, per frame, the earlier and the later frames named.
    """
    frames = config["frames"]
    by_value = {}
    for index, spec in enumerate(frames):
        by_value[spec["source"]["source_sha256"]] = index
        by_value[spec["case_id"]] = index
    rows = []
    for index, spec in enumerate(frames):
        named = {by_value[value] for value in _strings(spec) if value in by_value} - {index}
        rows.append({"case_id": spec["case_id"],
                     "earlier": sorted(frames[i]["case_id"] for i in named if i < index),
                     "later": sorted(frames[i]["case_id"] for i in named if i > index)})
    return rows


def history_chain(config: dict) -> list[str]:
    """Problems with the run's time order and predecessor chain; empty when intact."""
    problems = []
    frames = config["frames"]
    predecessor = ((config.get("phase") or {}).get("initial_previous") or {}).get("source", {}).get(
        "source_sha256")
    last = None
    for spec in frames:
        source = spec["source"]
        if predecessor is not None and source.get("previous_source_sha256") != predecessor:
            problems.append(f"{spec['case_id']}: its predecessor is not the previous frame")
        key = (spec.get("elapsed_h"), spec.get("frame_index"))
        if last is not None and not (key[0] > last[0] and key[1] > last[1]):
            problems.append(f"{spec['case_id']}: not after the previous frame in time")
        predecessor, last = source["source_sha256"], key
    return problems


def quick_check_validity(pipeline: str, configs: list[dict]) -> dict:
    """Decide whether the quick check is valid for these run configurations.

    Returns ``valid`` and a report: the pipeline's rule, the predecessor chain
    of each run, and every frame that names another frame of its run.
    """
    reasons = []
    if pipeline not in PREFIX_PIPELINES:
        reasons.append(f"Pipeline {pipeline!r} is not known to compute frames in time order only")
    runs = []
    for config in configs:
        chain = history_chain(config)
        dependencies = frame_dependencies(config)
        later = [row for row in dependencies if row["later"]]
        for row in later:
            reasons.append(f"Run {config['run']}, {row['case_id']} names later frame(s): "
                           + ", ".join(row["later"]))
        reasons.extend(f"Run {config['run']}: {problem}" for problem in chain)
        runs.append({"run": config["run"], "history": config.get("history_policy"),
                     "frames": len(config["frames"]), "chain_problems": chain,
                     "frames_naming_earlier_frames": sum(bool(row["earlier"]) for row in dependencies),
                     "frames_naming_later_frames": [row["case_id"] for row in later]})
    steps = [
        {"step": "native image preparation", "uses": "the frame's own raw image and the run's baseline"},
        {"step": "phase labels", "uses": "the frame, its predecessor (history) and the baseline"},
        {"step": "mass (N1)", "uses": "the frame's phase labels and frozen calibration"},
        {"step": "stored coarse layer", "uses": "the frame's own native products"},
        {"step": "frozen run inputs", "uses": "model, calibration, envelopes and geometry as published; "
                                             "a frozen input built from a whole run is read as it is"},
    ]
    return {"valid": not reasons, "reasons": reasons, "pipeline": pipeline,
            "rule": "each frame uses only itself, earlier frames, the baseline and frozen inputs",
            "steps": steps, "runs": runs, "default_frames": DEFAULT_QUICK_FRAMES}
