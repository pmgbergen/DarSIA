"""Short, provenance-backed names for saved phase models.

Display names are presentation only. The model ID and saved root remain the
identity used for loading, review corrections, and all provenance checks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


_TWO_ARM_FORMAT = "ffac-two-arm-optical-regional-v1"
_RAW_V5_FORMAT = "ffac-v5-raw-full-native-training-v1"
_RAW_V5_ARM = "raw_ac28_v5"
_RAW_V5_CHEM_ARM = "raw_ac28_v5_chemistry"
_ARM_NAMES = {
    "within_run": "FF_AC · Normalisering per forsøk",
    "global": "FF_AC · Global ColorChecker-normalisering",
    _RAW_V5_ARM: "FF_AC · RAW-normalisering med AC28 som lysreferanse (v5)",
    _RAW_V5_CHEM_ARM: "FF_AC · RAW-normalisering og AC28-kjemijustering (v5)",
}


def _arm(document: Mapping[str, object]) -> str:
    settings = document.get("settings")
    if not isinstance(settings, Mapping):
        return ""
    comparison = settings.get("two_arm_photometry")
    if isinstance(comparison, Mapping) and comparison.get("format") == _TWO_ARM_FORMAT:
        arm = str(comparison.get("arm") or "")
        return arm if arm in {"within_run", "global"} else ""
    raw = settings.get("v5_raw_photometry")
    if (isinstance(raw, Mapping) and raw.get("format") == _RAW_V5_FORMAT
            and raw.get("full_native_resolution") is True
            and raw.get("ac28_light_anchor") is True):
        chemistry = raw.get("chemistry_alignment")
        if (isinstance(chemistry, Mapping)
                and chemistry.get("mode") == "ac28-chroma-curve-v1"
                and isinstance(chemistry.get("bundle_sha256"), str)
                and len(chemistry["bundle_sha256"]) == 64
                and isinstance(chemistry.get("native_qa_receipt_sha256"), str)
                and len(chemistry["native_qa_receipt_sha256"]) == 64):
            return _RAW_V5_CHEM_ARM
        return _RAW_V5_ARM
    return ""


def recorded_photometry_provenance(
    root: Path, request: Mapping[str, object], summary: Mapping[str, object]
) -> tuple[str, str]:
    """Identify the optical parent or fitted regional stage from saved evidence."""
    regional = (summary.get("artifact_kind") == "optical-region-training-candidate"
                and summary.get("regional_training_performed") is True)
    if not regional:
        identifier = str(request.get("model_id") or "")
        saved_identifier = str(summary.get("campaign_id") or summary.get("model_id") or "")
        arm = _arm(request)
        if arm and identifier and saved_identifier == identifier and not summary.get("artifact_kind"):
            return arm, "optical_parent"
        return "", ""
    if arm := _arm(request):
        return arm, "regional_model"
    inherited = str(summary.get("inherited_optical_model_id") or "")
    if not inherited:
        return "", ""
    settings = request.get("settings")
    parent = str(settings.get("parent_model") or "") if isinstance(settings, Mapping) else ""
    if Path(parent).name != inherited:
        return "", ""
    path = root / "parent_provenance" / "training_request.json"
    try:
        parent_request = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "", ""
    if not isinstance(parent_request, Mapping):
        return "", ""
    if parent_request.get("model_id") != inherited:
        return "", ""
    arm = _arm(parent_request)
    return (arm, "regional_model") if arm else ("", "")


def recorded_photometry_arm(
    root: Path, request: Mapping[str, object], summary: Mapping[str, object]
) -> str:
    return recorded_photometry_provenance(root, request, summary)[0]


def _t4c_display_name(name: str) -> str:
    """Repair known separator encodings in titles without changing saved evidence."""
    if not any(f"G3-T4c {placement} gas" in name for placement in ("P1", "P2", "P3")):
        return name
    # UTF-8 decoded once or twice as Windows-1252/Latin-1. Replace only these
    # complete separator sequences; legitimate Unicode words remain untouched.
    for separator in ("\u00c3\u201a\u00c2\u00b7", "\u00c3\u0082\u00c2\u00b7", "\u00c2\u00b7"):
        name = name.replace(separator, "\u00b7")
    return name


def friendly_model_name(entry: Mapping[str, object]) -> str:
    """Return a readable name only when the arm is established by model metadata."""
    if (entry.get("p1_final_registered_complete") is True
            or entry.get("p1_final2_registered_complete") is True
            or entry.get("p1_final3_registered_complete") is True):
        # The history inspector sets this only after exact saved admission. The
        # copied approved-G3 provenance describes the parent, not this revision.
        name = str(entry.get("display_name") or "").strip()
        if name:
            return name
    if entry.get("area_only_revision") is True:
        name = str(entry.get("display_name") or "").strip()
        if name:
            return _t4c_display_name(name)
    approved = entry.get("approved_g3_production")
    if isinstance(approved, Mapping) and approved.get("configuration") == {
            "regional_threshold": .6, "pixel_threshold": .7,
            "minimum_area_mm2": 2.5, "connectivity": 4}:
        label = "FF_AC · G3 gas · Regional refit 0.6 · Pixel 0.7 · Minimum 2.5 mm²"
        if (approved.get("domain_revision") or {}).get("format") == "ffac-photo-sand-top-revision-v1":
            label += " · Top surface corrected"
        if (approved.get("barrier_rule") or {}).get("format") == "ffac-gas-barrier-rule-b-v1":
            label += " · Barrier rule B"
        return label
    arm = _ARM_NAMES.get(str(entry.get("photometry_arm") or ""), "")
    stage = str(entry.get("photometry_stage") or "")
    if arm and stage in {"optical_parent", "regional_model"}:
        if entry.get("photometry_arm") in {_RAW_V5_ARM, _RAW_V5_CHEM_ARM}:
            return f"{arm} · {'Optisk mellommodell' if stage == 'optical_parent' else 'Regional fasekandidat for validering'}"
        return f"{arm} · {'Optisk mellommodell' if stage == 'optical_parent' else 'Ferdig regional fasemodell'}"
    return arm


def created_utc_text(entry: Mapping[str, object]) -> str:
    """Render the recorded creation timestamp in UTC for identity details."""
    raw = str(entry.get("created_utc") or "").strip()
    if not raw:
        return "time not recorded"
    try:
        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
