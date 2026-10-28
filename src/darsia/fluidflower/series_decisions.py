"""Series decision settings: who decides, and which recorded choices apply.

The code checks only that a recorded decision exists and is unchanged. Who
decided, and which choices were made, is read from the series configuration:
the ``[decisions]`` table of the series' ``series.toml``. A feature that needs
a setting the series does not have is unavailable and names the missing key;
nothing is accepted in its place.

Documented keys (all optional)::

    [decisions]
    decider = "..."                  # actor recorded for approvals and saves
    gap_review_decision_ref = "..."  # reference written with gap approvals

    [decisions.calibration_protocol]  # the recorded calibration protocol
    sha256 = "..."                    # SHA-256 of its exact bytes
    size_bytes = 0

    [decisions.accepted_pressure]     # existing short-gap pressure means
    record_id = "..."                 # the decision record in the record file
    record_file = { sha256 = "...", size_bytes = 0 }
    actor = "..."
    confirmed_utc = "YYYY-MM-DDTHH:MM:SSZ"
    reason = "..."
    runs = ["..."]
    source_index = { sha256 = "...", size_bytes = 0 }
    source_manifest = { sha256 = "...", size_bytes = 0 }
    data_builder = { sha256 = "...", size_bytes = 0 }

    [decisions.overflow_method]       # the recorded overflow method
    protocol = { sha256 = "...", size_bytes = 0 }
    addendum = { sha256 = "...", size_bytes = 0 }
    overflow_review = { sha256 = "...", size_bytes = 0 }
    records = { "<record id>" = "<canonical record SHA-256>" }
    [[decisions.overflow_method.authorizations]]
    actor = "..."
    confirmed_utc = "..."
    reason = "..."                    # plus optional conditional_anchor_rule

    [decisions.clearwater_source]     # one recorded water photograph
    record_id = "..."
    record_sha256 = "..."             # canonical SHA-256 of the record
    actor = "..."
    confirmed_utc = "..."
    run_id = "..."
    reason = "..."
    raw = { path = "...", sha256 = "...", size_bytes = 0 }
    copied_raw = { path = "...", sha256 = "...", size_bytes = 0 }
    camera_jpeg = { path = "...", sha256 = "...", size_bytes = 0 }
    current_model = { sha256 = "...", size_bytes = 0 }
    current_profile = { sha256 = "...", size_bytes = 0, profile_id = "...", camera_wh = [0, 0] }
    instrument = { profile_id = "...", intrinsics_source_profile_id = "...", distortion = [],
                   ca_red = [], ca_blue = [], camera_wh = [0, 0], output_grid = "..." }

Record files and photographs stay where they are; these settings only name
them and pin their exact bytes.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import tomllib

SECTION = "decisions"
MAX_SERIES_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DecisionSettingUnavailable(ValueError):
    """The series configuration does not name a setting a feature needs."""


def series_decisions_from_bytes(raw) -> dict:
    """Return ``[decisions]`` from the exact bytes of a series manifest (empty when absent)."""
    if not isinstance(raw, bytes) or len(raw) > MAX_SERIES_BYTES:
        raise ValueError("The series manifest exceeds its read limit")
    value = tomllib.loads(raw.decode("utf-8-sig")).get(SECTION, {})
    if not isinstance(value, dict):
        raise ValueError("The series manifest [decisions] entry must be a table")
    return value


def read_series_decisions(series_root) -> dict:
    """Return a copy of ``[decisions]`` from ``series.toml`` (empty when absent)."""
    path = Path(series_root) / "series.toml"
    with path.open("rb") as stream:
        raw = stream.read(MAX_SERIES_BYTES + 1)
    return deepcopy(series_decisions_from_bytes(raw))


def decision_setting(decisions, key):
    """Return ``decisions[key]`` or say which series setting is missing."""
    if not isinstance(decisions, dict) or key not in decisions:
        raise DecisionSettingUnavailable(
            f"This series does not name '{SECTION}.{key}' in series.toml; "
            "add it before using this feature.")
    return deepcopy(decisions[key])


def text_setting(decisions, key):
    """Return a nonempty text setting or say which series setting is missing."""
    value = decision_setting(decisions, key)
    if not isinstance(value, str) or not value.strip():
        raise DecisionSettingUnavailable(f"The series setting '{SECTION}.{key}' must be nonempty text.")
    return value


def configured_decider(series_root):
    """The decider this series names, or None when it names none or has no manifest."""
    try:
        decisions = read_series_decisions(series_root)
    except FileNotFoundError:
        return None
    value = decisions.get("decider")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DecisionSettingUnavailable(f"The series setting '{SECTION}.decider' must be nonempty text.")
    return value


def gap_review_decision_ref(series_root):
    """The decision reference this series writes with gap approvals, or None for the default."""
    if series_root is None:
        return None
    try:
        decisions = read_series_decisions(series_root)
    except FileNotFoundError:
        return None
    if "gap_review_decision_ref" not in decisions:
        return None
    return text_setting(decisions, "gap_review_decision_ref")


def project_series_root(project):
    """The series folder a project belongs to: its linked series, else its own folder."""
    from collections.abc import Mapping
    from .series_geometry import native_series_link
    manifest = getattr(project, "manifest", None)
    linked = native_series_link(project).get("series_root") if isinstance(manifest, Mapping) else None
    root = linked or getattr(project, "root", None)
    return Path(str(root)) if root else None


def series_actor(series_root):
    """The actor recorded for an application action: the series decider, else the signed-in user."""
    from .series_geometry import default_actor
    actor = (configured_decider(series_root) if series_root is not None else None) or default_actor()
    if not isinstance(actor, str) or not actor.strip():
        raise DecisionSettingUnavailable(
            f"Name the series decider ('{SECTION}.decider' in series.toml) before saving.")
    return actor


def checked_pin(value, where="recorded file"):
    """Validate an exact ``{sha256, size_bytes}`` pin and return a plain copy."""
    if (not isinstance(value, dict) or set(value) != {"sha256", "size_bytes"}
            or not isinstance(value["sha256"], str) or _SHA256.fullmatch(value["sha256"]) is None
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 1):
        raise DecisionSettingUnavailable(f"The {where} needs an exact sha256 and positive size_bytes pin.")
    return dict(sha256=value["sha256"], size_bytes=value["size_bytes"])


def matches_pin(raw, pin):
    """True when the bytes have exactly the pinned size and SHA-256."""
    pin = checked_pin(pin)
    return (isinstance(raw, bytes) and len(raw) == pin["size_bytes"]
            and hashlib.sha256(raw).hexdigest() == pin["sha256"])


def utc_text(value, where="decision time"):
    """Validate an explicit UTC time written as text and return it unchanged."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"The {where} must be an explicit UTC time.") from error
    if not isinstance(value, str) or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"The {where} must be an explicit UTC time.")
    return value


def recorded_decision(record, where="decision"):
    """Check that a decision record exists: its id, decider and time are named.

    The record's bytes are pinned by its caller; this never compares who
    decided with a fixed name.
    """
    if (not isinstance(record, dict)
            or any(not isinstance(record.get(key), str) or not record[key].strip()
                   for key in ("id", "decided_by", "utc"))):
        raise ValueError(f"The recorded {where} must name its id, decider and time.")
    return record


def record_digest(record):
    """Canonical SHA-256 of one JSON decision record."""
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def find_decision_record(rows, record_id, *, digest=None, where="decision"):
    """Return the one record with ``record_id``; optionally pin its canonical digest.

    ``rows`` is one record or a list of records, such as an append-only
    answer file.
    """
    rows = rows if isinstance(rows, list) else [rows]
    found = [row for row in rows if isinstance(row, dict) and row.get("id") == record_id]
    if len(found) != 1 or (digest is not None and record_digest(found[0]) != digest):
        raise ValueError(f"Choose the unchanged recorded {where}.")
    return deepcopy(recorded_decision(found[0], where))
