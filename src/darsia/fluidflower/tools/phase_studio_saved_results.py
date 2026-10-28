"""Fast, read-only browser index for immutable series concentration deliveries.

The index is a local display cache, never a scientific source. A chosen result
is checked against its NAS delivery receipt before Studio uses its contents.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from darsia.fluidflower.tools.phase_studio_mass_comparison import result_variant_label
from darsia.fluidflower.tools.phase_studio_mass_inputs import IDENTITY_KEYS


_FORMAT = "studio-mass-variant-delivery-v1"
_CACHE_FORMAT = "studio-saved-series-browser-v4"


def _identity(candidate):
    return tuple(candidate.get(key) for key in IDENTITY_KEYS)


def model_browser_metadata(result, result_path):
    """Summarize saved model *results* without reading numerical field arrays.

    Peak free gas is the largest modeled gas mass among this experiment's saved
    images. It is a mass in grams, not a gas saturation or a cross-run maximum.
    """
    calibration = result.get("calibration") or {}
    scope = str(calibration.get("scope") or "unknown").casefold()
    score = (calibration.get("metrics") or {}).get("mean_run_nmae")
    if score is None:
        score = (calibration.get("metrics") or {}).get("nmae")
    try:
        score = float(score) if score is not None else None
        if score is not None and not math.isfinite(score):
            score = None
    except (TypeError, ValueError):
        score = None

    gas_masses = []
    for frame in result.get("frames") or []:
        try:
            gas_mass = float(frame["gas_mass_g"])
            if math.isfinite(gas_mass) and gas_mass >= 0:
                gas_masses.append(gas_mass)
        except (KeyError, TypeError, ValueError):
            continue
    peak_gas_mass_g = max(gas_masses) if gas_masses else None

    mode = str(result.get("overflow_mode") or calibration.get("overflow_mode") or "unknown").casefold()
    if mode not in {"off", "compare", "constrain", "hard"}:
        mode = "unknown"
    scope_label = {"shared": "Shared", "series": "Shared", "individual": "Individual"}.get(
        scope, "Scope unknown" if scope == "unknown" else scope.title())
    overflow_label = {"off": "No overflow", "compare": "Overflow comparison",
                      "constrain": "Overflow used in fit", "hard": "Overflow hard constraint",
                      "unknown": "Overflow status unknown"}[mode]
    peak_label = (f" · peak free gas {peak_gas_mass_g:.3f} g"
                  if peak_gas_mass_g is not None else "")
    score_label = f" · fit NMAE {100 * score:.2f}%" if score is not None else ""
    reference = result.get("series_calibration")
    if isinstance(reference, str) and reference:
        reference = Path(reference)
        if not reference.is_absolute():
            reference = Path(result_path).resolve().parent / reference
        identity = str(reference.resolve())
        fallback_name = "Saved calibration: " + reference.name
    else:
        identity = str(Path(result_path).resolve())
        fallback_name = "Unnamed saved calibration"
    # Literal saved metadata only: a matching label or parameter vector never
    # proves that two run results use the same calibration.
    recorded_name = calibration.get("display_name") or calibration.get("name")
    chemistry = calibration.get("chemistry_model") or result.get("method", {}).get("settings", {}).get("chemistry")
    chemistry_label = {"titration": "Titration", "darsia_titration_flash": "Titration",
                       "simple_flash": "Simple flash", "darsia_simple_flash": "Simple flash"}.get(chemistry)
    digests = {mapping[key] for mapping, key in (
        (result.get("bindings") or {}, "calibration_sha256"),
        (result.get("scalar_result") or {}, "calibration_sha256"),
        (calibration, "sha256"), (result, "calibration_sha256"))
        if isinstance(mapping, dict) and isinstance(mapping.get(key), str) and mapping[key]}
    return {
        "review_calibration_id": identity,
        "review_calibration_sha256": next(iter(digests)) if len(digests) == 1 else None,
        "review_calibration_binding_conflict": len(digests) > 1,
        "review_calibration_name": str(recorded_name) if recorded_name else fallback_name,
        "review_chemistry": chemistry_label or (str(chemistry) if chemistry else "Chemistry not recorded"),
        "calibration_scope": scope,
        "mass_fit_nmae": score,
        "peak_free_gas_mass_g": peak_gas_mass_g,
        "overflow_usage": mode,
        "label": (f"{scope_label} · {overflow_label}{peak_label} · "
                  f"{result_variant_label(result, result_path)}{score_label}"),
    }


def _entry(result_path, result, receipt_path, receipt):
    candidate = receipt["candidate"]
    selected = result.get("selected_inputs", {}).get("candidate", {})
    if (result.get("complete") is not True or not result.get("frames")
            or _identity(selected) != _identity(candidate)
            or result.get("source_bindings", {}) != receipt.get("source_bindings", {})):
        return None
    return {
        "path": str(result_path),
        "dataset": candidate["dataset"],
        "run": candidate["run"],
        "model": candidate["candidate_id"],
        "frames": len(result["frames"]),
        "created_utc": result.get("created_utc", ""),
        "method": result["method"]["settings"],
        **model_browser_metadata(result, result_path),
        "receipt_path": str(receipt_path),
        "result_sha256": receipt["result_sha256"],
    }


def discover_saved_series_entries(dataset, catalog_root, cache_root, *, cancelled=None):
    """List display metadata with one NAS read per result on a cold cache.

    File name, size and modification time invalidate a changed receipt. Result
    bytes are verified during a cold build and again at use; warm display
    listings need only the small receipt directory and its metadata.
    """
    dataset = str(dataset).casefold()
    catalog_root = Path(catalog_root).resolve()
    cache_root = Path(cache_root)
    # Browser organization is a separate mutable index, not a scientific
    # delivery receipt. Moving a model into history must not invalidate and
    # rebuild hundreds of immutable result summaries over NAS.
    receipts = []
    for path in catalog_root.glob("*.json"):
        if cancelled is not None and cancelled():
            return []
        if path.name != "browser_archive.json":
            receipts.append(path)
    receipts.sort(key=lambda path: path.name)
    stamps = []
    for path in receipts:
        if cancelled is not None and cancelled():
            return []
        stamp = path.stat()
        stamps.append((path.name, stamp.st_size, stamp.st_mtime_ns))
    cache_key = hashlib.sha256(str(catalog_root).encode("utf-8")).hexdigest()[:16]
    cache_path = cache_root / f"{dataset}-{cache_key}.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (cached.get("format") == _CACHE_FORMAT and cached.get("dataset") == dataset
                and cached.get("catalog_root") == str(catalog_root)
                and cached.get("receipts") == [list(row) for row in stamps]):
            return cached["entries"]
    except (OSError, ValueError, TypeError, KeyError):
        pass

    entries, seen = [], set()
    cacheable = True
    for receipt_path in receipts:
        if cancelled is not None and cancelled():
            return []
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            if receipt.get("format") != _FORMAT:
                continue
            candidate = receipt.get("candidate") or {}
            if str(candidate.get("dataset") or "").casefold() != dataset:
                continue
            result_path = Path(receipt["result"]).resolve()
            if result_path in seen:
                continue
            if cancelled is not None and cancelled():
                return []
            data = result_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != receipt.get("result_sha256"):
                continue
            result = json.loads(data.decode("utf-8-sig"))
            entry = _entry(result_path, result, receipt_path, receipt)
            if entry is not None:
                entries.append(entry)
                seen.add(result_path)
        except OSError:
            # A temporary NAS fault must be retried on the next visit.
            cacheable = False
            continue
        except (ValueError, KeyError, TypeError, UnicodeError):
            continue
    payload = {"format": _CACHE_FORMAT, "dataset": dataset,
                "catalog_root": str(catalog_root), "receipts": stamps, "entries": entries}
    if cancelled is not None and cancelled():
        return []
    if cacheable:
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            os.replace(temporary, cache_path)
        except OSError:
            # A read-only local cache must not hide NAS results.
            pass
    return entries


def verify_saved_series_entry(entry):
    """Verify cached display metadata against the authoritative NAS receipt."""
    receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8-sig"))
    if (receipt.get("format") != _FORMAT
            or Path(receipt["result"]).resolve() != Path(entry["path"]).resolve()
            or receipt.get("result_sha256") != entry.get("result_sha256")):
        raise ValueError("Saved concentration delivery changed; refresh its catalog")
    data = Path(entry["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt["result_sha256"]:
        raise ValueError("Saved concentration result differs from its delivery receipt")
    result = json.loads(data.decode("utf-8-sig"))
    if _entry(Path(entry["path"]).resolve(), result, Path(entry["receipt_path"]), receipt) is None:
        raise ValueError("Saved concentration result is no longer bound to its delivery")
    return result
