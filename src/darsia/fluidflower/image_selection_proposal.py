"""Q6 Default20 proposal and explicit historical metadata import; no pixels.

This is a UI rule, distinct from the historical hand-picked Studio20. All
classes are read from existing source-bound metadata; no classifier is run.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

from darsia.fluidflower.image_selection_plan import _eligible_frames
from darsia.fluidflower.series_method import ManifestChangedError
from darsia.fluidflower.series_paths import series_key
from darsia.fluidflower.storage_paths import storage_path_remapper

RULE = "Default 20 (1 + 8 + 11)"
QUALITY_RULE = "28sep-frozen-raw-chart-pillar-v1"


def _key(value):
    return str(value).replace("\\", "/").casefold()


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be finite.")
    return float(value)


def _metadata_file(path, expected=None):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Selection metadata exceeds the 32 MiB metadata limit.")
    raw = path.read_bytes()
    bound = dict(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )
    if expected is not None and expected != bound:
        raise ManifestChangedError("Imported selection metadata changed: " + str(path))
    doc = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(doc, dict):
        raise ValueError("Selection metadata must be a JSON object.")
    return doc, bound


def eligible_frames(metadata):
    rows, omitted = _eligible_frames(
        list(metadata.frames), check_files=False, source_root=None
    )
    if any(
        row["reason"] == "duplicate_source_path_conflicting_time" for row in omitted
    ):
        raise ValueError("A source image has conflicting protocol times.")
    if not rows:
        raise ValueError("The saved protocol has no eligible images.")
    return rows


def _quality_classes(metadata, quality):
    if quality is None:
        return {}
    if (
        quality.get("run_id") != metadata.run_id
        or Path(quality.get("series_root", "")).resolve()
        != Path(metadata.series_root).resolve()
    ):
        raise ValueError("Saved quality belongs to another series/run.")
    fresh = load_saved_quality(
        quality["source"]["path"], metadata, expected=quality["source"]
    )
    if fresh["classes"] != quality["classes"]:
        raise ValueError(
            "Saved quality classes differ from their exact source receipt."
        )
    return fresh["classes"]


def first_usable_rule(metadata, selected, quality=None):
    """Keep a usable pre frame; only a genuine absent-pre case may use first."""
    classes = _quality_classes(metadata, quality)
    usable = [
        row
        for row in eligible_frames(metadata)
        if classes.get(_key(row["source_path"]), {}).get("class") != "reject"
    ]
    if not usable:
        raise ValueError(
            "The stored quality metadata marks every eligible image reject."
        )
    has_pre = any(row["elapsed_h"] < metadata.injection_start_h for row in usable)
    if has_pre:
        if not any(
            row["elapsed_h"] < metadata.injection_start_h
            and classes.get(_key(row["source_path"]), {}).get("class") != "reject"
            for row in selected
        ):
            raise ValueError(
                "The first selected frame must precede injection when a usable pre-injection image exists."
            )
    elif not selected or _key(selected[0]["source_path"]) != _key(
        usable[0]["source_path"]
    ):
        raise ValueError(
            "No usable pre-injection image exists; include the first usable image explicitly."
        )
    return not has_pre


def default20_proposal(metadata, *, d1_h=0.25, quality=None):
    """Exact Q6 grid, nearest camera time/tie earlier, bounded saved-class repair.

    An out-of-tolerance image is retained with a flag. A collision advances to
    the next distinct acquisition; bounded reject repair only picks saved ok.
    Every finite candidate list is visited at most once per target.
    """
    d1_h = _number(d1_h, "d1")
    if d1_h <= 0:
        raise ValueError("d1 must be greater than zero.")
    rows = eligible_frames(metadata)
    classes = _quality_classes(metadata, quality)
    usable = [
        row
        for row in rows
        if classes.get(_key(row["source_path"]), {}).get("class") != "reject"
    ]
    if not usable:
        raise ValueError(
            "The stored quality metadata marks every eligible image reject."
        )
    t0 = _number(metadata.injection_start_h, "Injection start")
    end = _number(metadata.injection_end_h, "Injection end")
    if end <= t0:
        raise ValueError("The last injection end must follow the first start.")
    pre = [row for row in usable if row["elapsed_h"] < t0]
    initial = pre[-1] if pre else usable[0]
    last = usable[-1]
    t_last = last["elapsed_h"]
    targets = [initial["elapsed_h"]] + [t0 + k * (end - t0) / 8 for k in range(1, 9)]
    issues = []
    if not pre:
        issues.append(
            "No usable pre-injection image; frame 0 is the first usable image."
        )
    d11 = t_last - end
    if d11 > 0:
        targets += [end + d1_h * (d11 / d1_h) ** (k / 10) for k in range(11)]
        targets[-1] = t_last  # Last target is exactly the stored last usable time.
        if d11 < d1_h:
            issues.append(
                "d1 exceeds the available post-injection delay; late targets descend and may collide."
            )
    else:
        issues.append(
            "No usable image after the last injection; eleven late targets cannot be proposed."
        )
    selected, used, matches = [], set(), []
    positions = {_key(row["source_path"]): i for i, row in enumerate(rows)}
    for index, target in enumerate(targets):
        tolerance = 15.0 if target - t0 <= 3 else 150.0
        replacement_bound = 60.0 if target - t0 <= 3 else 600.0
        nearest = (
            initial
            if index == 0
            else min(
                rows,
                key=lambda row: (
                    abs(row["elapsed_h"] - target),
                    row["elapsed_h"],
                    row["input_index"],
                ),
            )
        )
        chosen = nearest
        flags = []
        if _key(chosen["source_path"]) in used:
            chosen = next(
                (
                    row
                    for row in rows[positions[_key(nearest["source_path"])] + 1 :]
                    if _key(row["source_path"]) not in used
                ),
                None,
            )
            flags.append("duplicate target: next distinct image")
        record = dict(
            target_index=index,
            target_time_h=target,
            tolerance_seconds=tolerance,
            replacement_tolerance_seconds=replacement_bound,
            flags=flags,
            selected=False,
        )
        if chosen is None:
            record["flags"].append(
                "no next distinct image; proposal has fewer than 20 frames"
            )
            matches.append(record)
            continue
        q = classes.get(_key(chosen["source_path"]), {})
        if q.get("class") == "reject":
            record["rejected_image"] = dict(
                source_path=chosen["source_path"],
                elapsed_h=chosen["elapsed_h"],
                frame_list_class="reject",
            )
            candidates = [
                row
                for row in rows
                if _key(row["source_path"]) not in used
                and classes.get(_key(row["source_path"]), {}).get("class") == "ok"
                and abs(row["elapsed_h"] - target) * 3600 <= replacement_bound + 1e-8
            ]
            chosen = (
                min(
                    candidates,
                    key=lambda row: (
                        abs(row["elapsed_h"] - target),
                        row["elapsed_h"],
                        row["input_index"],
                    ),
                )
                if candidates
                else None
            )
            if chosen is None:
                record["flags"].append(
                    "reject without a saved ok replacement within the bound"
                )
                matches.append(record)
                continue
            record["flags"].append("reject replaced by nearest saved ok image")
            q = classes[_key(chosen["source_path"])]
        offset = (chosen["elapsed_h"] - target) * 3600
        if abs(offset) > tolerance + 1e-8:
            record["flags"].append("no image within tolerance")
        used.add(_key(chosen["source_path"]))
        selected.append(deepcopy(chosen))
        record.update(
            selected=True,
            chosen_image={
                key: chosen[key] for key in ("source_path", "image_id", "frame_index")
            },
            camera_elapsed_h=chosen["elapsed_h"],
            offset_seconds=offset,
            frame_list_class=q.get("class", "Not recorded"),
        )
        matches.append(record)
    selected.sort(key=lambda row: (row["elapsed_h"], row["input_index"]))
    if len(selected) < 20:
        issues.append(
            f"Proposal contains {len(selected)} distinct images, fewer than 20."
        )
    return dict(
        format="fluidflower-default20-ui-proposal-v1",
        rule=RULE,
        run_id=metadata.run_id,
        series_root=metadata.series_root,
        parameters=dict(
            t0_h=t0,
            t_end_h=end,
            t_last_h=t_last,
            d1_h=d1_h,
            t_last_class=classes.get(_key(last["source_path"]), {}).get(
                "class", "Not recorded"
            ),
        ),
        no_preinjection_image=not bool(pre),
        target_matches=matches,
        selected=selected,
        selected_frame_count=len(selected),
        issues=issues,
        quality_source={} if quality is None else deepcopy(quality["source"]),
        source_metadata=deepcopy(list(metadata.sources)),
        scientific_computation=False,
    )


def legacy_stored_plan_status(root):
    """Follow only the manifest's explicit Studio pointer; never discover models."""
    root = Path(root)
    raw = (root / "series.toml").read_bytes()
    doc = tomllib.loads(raw.decode("utf-8-sig"))
    pointer = (doc.get("legacy") or {}).get("studio_project")
    if not pointer:
        return dict(
            available=False,
            label="Stored plan (Studio 20)",
            note="No Studio project pointer is stored. Import its source-bound native manifest explicitly.",
        )
    path = Path(storage_path_remapper()(str(pointer)))
    path = path if path.is_absolute() else root / path
    project, bound = _metadata_file(path.resolve())
    from darsia.fluidflower.project import PROJECT_FORMAT

    if project.get("format") != PROJECT_FORMAT or project.get("schema") != 1:
        raise ValueError(
            "The stored Studio project has an unsupported original format."
        )
    if series_key(project.get("series_id", "")) != series_key(doc["series_id"]):
        raise ValueError("The stored Studio project belongs to another series.")
    # The original project schema stores source mapping, not an active model or
    # selected native timeline. Never pick a model/catalog/QSettings as a proxy.
    return dict(
        available=False,
        label="Stored plan (Studio 20)",
        studio_project=bound,
        note="The Studio project stores its source mapping but no bound Studio20 native manifest. Import that manifest explicitly; no historical model was guessed.",
    )


def load_stored_native_plan(path, metadata, *, series_id, expected=None):
    """Retain a bound native timeline's own clock beside current protocol times."""
    doc, bound = _metadata_file(path, expected)
    if (
        series_key(doc.get("dataset", "")) != series_key(series_id)
        or str(doc.get("run", "")).casefold() != metadata.run_id
    ):
        raise ValueError("The imported native manifest belongs to another series/run.")
    native = doc.get("frame_sources")
    if not isinstance(native, list) or not native:
        raise ValueError("Choose an existing native manifest with frame_sources.")
    available = {_key(row["source_path"]): row for row in eligible_frames(metadata)}
    selected, seen, identities = [], set(), []
    # A receipt keeps its historical paths; the storage aliases (if any) name
    # where this machine finds the same images.
    remap = storage_path_remapper()
    for row in native:
        key = _key(remap(str(row.get("source_path", ""))))
        digest = row.get("source_sha256")
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError(
                "Every native selected image needs its original source SHA-256."
            )
        if key not in available or key in seen:
            raise ValueError(
                "Native selected source identity is absent or duplicated in the saved protocol."
            )
        historical = _number(row.get("elapsed_h"), "Native camera time")
        current = available[key]
        choice = deepcopy(current)
        choice["historical_elapsed_h"] = historical
        choice["protocol_elapsed_h"] = current["elapsed_h"]
        choice["clock_difference"] = historical != current["elapsed_h"]
        identity = dict(
            source_path=row["source_path"],
            source_sha256=digest,
            elapsed_h=historical,
            historical_elapsed_h=historical,
            native_frame_index=row.get("frame_index"),
            protocol_source_path=current["source_path"],
            protocol_frame_index=current["frame_index"],
            protocol_elapsed_h=current["elapsed_h"],
            data_time_h=current["elapsed_h"],
            clock_difference=choice["clock_difference"],
        )
        if "target_time_h" in row:
            identity["historical_target_time_h"] = _number(
                row["target_time_h"], "Native target time"
            )
            choice["historical_target_time_h"] = identity["historical_target_time_h"]
        selected.append(choice)
        seen.add(key)
        identities.append(identity)
    return dict(
        label=(
            "Stored plan (Studio 20)"
            if len(selected) == 20
            else "Stored plan (imported native manifest)"
        ),
        source=bound,
        run_id=metadata.run_id,
        series_root=metadata.series_root,
        selected=selected,
        native_source_identities=identities,
        readonly=True,
        time_basis="Original native-manifest elapsed/target values are historical; protocol camera/data_time values are current saved protocol times.",
        note="Original saved source SHA values and historical times are retained; no image pixels were opened or rehashed. Stored and protocol clock values are shown separately; a difference does not change either clock.",
    )


def load_saved_quality(path, metadata, *, expected=None):
    """Import an original quality-grid receipt, never run the 28 Sep classifier."""
    doc, bound = _metadata_file(path, expected)
    if doc.get("format") != "fluidflower-quality-camera-grid-v1":
        raise ValueError("Choose an existing source-bound quality-camera-grid receipt.")
    available = {_key(row["source_path"]): row for row in eligible_frames(metadata)}
    classes = {}
    rows = list(doc.get("selected") or [])
    for match in doc.get("target_matches") or []:
        for field in ("replacement", "rejected_image"):
            if isinstance(match.get(field), dict):
                rows.append(match[field])
    remap = storage_path_remapper()  # Historical receipt paths, as read here.
    for row in rows:
        key = _key(remap(str(row.get("source_path", ""))))
        quality = row.get("quality") or {}
        if (
            quality.get("class_measurement", doc.get("class_measurement"))
            != QUALITY_RULE
        ):
            raise ValueError(
                "The saved quality metadata does not identify the 28 September rule; no new classification is run."
            )
        if key not in available or row.get("elapsed_h") != available[key]["elapsed_h"]:
            raise ValueError(
                "Saved quality source/time differs from this run's protocol."
            )
        if quality.get("class") not in {"ok", "minor", "reject"}:
            raise ValueError("Saved quality must explicitly say ok, minor or reject.")
        digest = row.get("source_sha256")
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("Saved quality needs its original image source SHA-256.")
        item = dict(quality, source_sha256=digest)
        if key in classes and classes[key] != item:
            raise ValueError("Conflicting saved classifications for one image.")
        classes[key] = item
    if not classes:
        raise ValueError("The existing receipt contains no source-bound saved classes.")
    return dict(
        source=bound,
        classes=classes,
        run_id=metadata.run_id,
        series_root=metadata.series_root,
        classification_run=False,
    )
