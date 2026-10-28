"""Immutable acceptance of existing, flagged pressure DATA by a recorded decision.

This is a decision/provenance bridge, not a pressure producer. Original rows,
observations, quality flags and gap records are retained without recomputation.
Only the decision-state flag changes, after the recorded authority and original
short-gap recipe have been verified. No active pointer or execution is changed.

Who decided, which runs and which pinned index, manifest and recipe are read
from the series configuration (``[decisions.accepted_pressure]``; see
series_decisions). A published authorization is later checked to be complete
and consistent with the evidence it accompanies.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

from .pipeline.job_inputs import MAX_METADATA_BYTES
from .pipeline.job_queue import descriptor, _immutable_bytes, _publish_bytes

FORMAT = "darsia-accepted-pressure-policy-v1"
POLICY = "ffac-existing-short-gap-mean-pm30min"
AUTHORIZATION_FORMAT = "darsia-pressure-policy-authorization-v1"
FIELD = "accepted_pressure_policy"
SETTING = "accepted_pressure"
_SETTING_KEYS = {"record_id", "record_file", "actor", "confirmed_utc", "reason", "runs",
                 "source_index", "source_manifest", "data_builder"}
_AUTHORIZATION_KEYS = {"format", "actor", "confirmed_utc", "reason", "series", "runs",
                       "source_index", "source_manifest"}
_SOURCES = ("authority", "source_index", "source_atmosphere", "source_manifest", "series_manifest")


def pressure_settings(decisions):
    """Validate the series' recorded pressure decision settings and return a plain copy."""
    from .series_decisions import DecisionSettingUnavailable, checked_pin, decision_setting, utc_text
    value = decision_setting(decisions, SETTING)
    if not isinstance(value, dict) or set(value) != _SETTING_KEYS:
        raise DecisionSettingUnavailable("decisions.accepted_pressure needs exactly: " + ", ".join(sorted(_SETTING_KEYS)))
    runs = value["runs"]
    if (any(not isinstance(value[key], str) or not value[key].strip() for key in ("record_id", "actor", "reason"))
            or not isinstance(runs, list) or not runs
            or any(not isinstance(run, str) or not run for run in runs) or len(set(runs)) != len(runs)):
        raise DecisionSettingUnavailable("decisions.accepted_pressure needs its record, decider, reason and distinct runs")
    utc_text(value["confirmed_utc"], "recorded pressure decision time")
    return dict(record_id=value["record_id"], record_file=checked_pin(value["record_file"], "pressure decision record file"),
        actor=value["actor"], confirmed_utc=value["confirmed_utc"], reason=value["reason"], runs=list(runs),
        source_index=checked_pin(value["source_index"], "recorded pressure index"),
        source_manifest=checked_pin(value["source_manifest"], "recorded GFI source manifest"),
        data_builder=checked_pin(value["data_builder"], "recorded pressure index producer"))


def human_authorization(settings, series):
    """Ordinary human decision metadata from the series settings; coordination records stay private."""
    return dict(format=AUTHORIZATION_FORMAT, actor=settings["actor"],
        confirmed_utc=settings["confirmed_utc"], reason=settings["reason"],
        series=series, runs=sorted(settings["runs"]),
        source_index=dict(sha256=settings["source_index"]["sha256"], size_bytes=settings["source_index"]["size_bytes"]),
        source_manifest=dict(sha256=settings["source_manifest"]["sha256"],
                             size_bytes=settings["source_manifest"]["size_bytes"]))


def gap_authority(authorization):
    """The gap-default authority text written with accepted pressure: who decided and why."""
    return authorization["actor"] + ": " + authorization["reason"]


def _bound(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value["path"], str) or not value["path"]
            or re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")) is None
            or type(value["size_bytes"]) is not int
            or not 0 < value["size_bytes"] <= MAX_METADATA_BYTES):
        raise ValueError("Pressure policy requires exact bounded source descriptors")
    return value


def _raw(bound, *, catalog=None, use="pressure_policy"):
    _bound(bound)
    actual = bound
    if catalog is not None:
        catalog.add(bound, "configuration", use)
        actual = catalog.physical(bound)
        raw = catalog.metadata_bytes(actual["path"])
    else:
        with Path(actual["path"]).open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
    if (not isinstance(raw, bytes) or len(raw) != bound["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != bound["sha256"]):
        raise ValueError("Pressure policy source SHA/size changed: " + bound["path"])
    return raw


def _json(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate pressure policy metadata key: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite pressure policy metadata: " + value)
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)


def _same(left, right):
    # JSON numeric types are significant: 1, 1.0 and True are not interchangeable.
    encode = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False, separators=(",", ":"))
    return encode(left) == encode(right)


def _authority(raw, settings):
    """The chosen record file is the pinned one and holds the named decision."""
    from .series_decisions import find_decision_record, matches_pin
    if not matches_pin(raw, settings["record_file"]):
        raise ValueError("Choose the unchanged recorded pressure decision")
    find_decision_record(_json(raw), settings["record_id"], where="pressure decision")


def _checked_authorization(raw, proof):
    """A published authorization names who decided, when and why, and matches its evidence."""
    from .series_decisions import utc_text
    try:
        value = _json(raw)
        utc_text(value.get("confirmed_utc") if isinstance(value, dict) else None)
        index, manifest = proof["source_index"]["original"], proof["source_manifest"]["original"]
        valid = (isinstance(value, dict) and set(value) == _AUTHORIZATION_KEYS
                 and value["format"] == AUTHORIZATION_FORMAT
                 and all(isinstance(value[key], str) and value[key].strip() for key in ("actor", "reason"))
                 and value["series"] == proof["series"]
                 and isinstance(value["runs"], list) and value["runs"] == sorted(set(value["runs"]))
                 and all(isinstance(run, str) and run for run in value["runs"])
                 and proof["run"] in value["runs"]
                 and _same(value["source_index"], dict(sha256=index["sha256"], size_bytes=index["size_bytes"]))
                 and _same(value["source_manifest"], dict(sha256=manifest["sha256"], size_bytes=manifest["size_bytes"])))
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        valid = False
    if not valid:
        raise ValueError("The recorded pressure authorization changed")
    return value


def _registered(raw):
    """The registered series id and distinct run order of a series manifest."""
    value = tomllib.loads(raw.decode("utf-8-sig"))
    runs = (value.get("runs") or {}).get("ids")
    series = value.get("series_id")
    if (value.get("schema_version") != 1 or not isinstance(series, str) or not series
            or not isinstance(runs, list) or not runs
            or any(not isinstance(run, str) or not run for run in runs)
            or len(set(runs)) != len(runs)):
        raise ValueError("The pressure decision needs the registered series and distinct run order")
    return series, runs


def _recorded_index(raw, pin):
    # The decision covers the existing flagged gaps, not arbitrary replacement
    # DATA produced later with the same method name. The original index pins
    # every atmosphere and the source manifest transitively, including sizes.
    from .series_decisions import matches_pin
    if not matches_pin(raw, pin):
        raise ValueError("Choose the unchanged recorded pressure index")
    return _json(raw)


def _original(index, atmosphere, run, *, series, runs, source_atmosphere, source_manifest, source_document,
              data_builder=None):
    """Check the original flagged recipe/evidence, without producing any values.

    ``data_builder`` pins the recorded index producer when the series settings
    are at hand; a published index is already pinned through its authorization.
    """
    from .round5_pressure import utc, bind_running_atmosphere
    if (not isinstance(index, dict)
            or index.get("format") != "darsia-running-gfi-pressure-index-v1"
            or FIELD in index or index.get("source_manifest") != source_manifest):
        raise ValueError("The pressure decision needs the original bound running-pressure index and producer")
    # This unused historical code locator remains literal provenance. Its exact
    # declared identity identifies the recorded recipe; it is never opened/executed
    # or substituted with a later working-tree source file. The active verifier
    # below checks the original DATA and recipe evidence itself.
    builder = _bound(index["data_builder"])
    if data_builder is not None and (builder["sha256"], builder["size_bytes"]) != (
            data_builder["sha256"], data_builder["size_bytes"]):
        raise ValueError("The pressure decision needs the original recorded producer identity")
    entry = index["runs"][series][run]
    if (run not in runs or entry.get("run") != run or entry.get("complete") is not True
            or entry.get("provisional_gap_default") is not True
            or entry.get("atmosphere_series") != source_atmosphere
            or atmosphere.get("format") != "darsia-running-gfi-pressure-v1"
            or atmosphere.get("series") != series or atmosphere.get("run") != run
            or atmosphere.get("provisional_gap_default") is not True
            or atmosphere.get("spans_all_captures") is not True
            or atmosphere.get("extrapolation_allowed") is not False
            or atmosphere.get("adopted") is not False or FIELD in atmosphere
            or atmosphere.get("source_manifest") != source_manifest
            or entry.get("row_count") != len(atmosphere["rows"])
            or not _same(entry.get("gaps"), atmosphere.get("gaps"))):
        raise ValueError("Pressure policy differs from the original flagged source")
    # Reuse the unchanged numerical consumer's finite/order/completeness checks.
    bind_running_atmosphere({"mass": {"template": {}}, "frames": []}, atmosphere)
    rows = atmosphere["rows"]
    if any(row.get("quality") not in {"observed", "provisional_short_gap_mean_pm30min"} for row in rows):
        raise ValueError("The pressure decision cannot accept a different pressure filling method")
    if (not isinstance(source_document, dict)
            or source_document.get("format") != "gfi-source-manifest-v1"
            or not isinstance(source_document.get("gfi_source_files"), list)):
        raise ValueError("Pressure observations need the original bound GFI source manifest")
    source_files = set()
    for item in source_document["gfi_source_files"]:
        if "sha256" in item and "size_bytes" in item:
            for path in item.get("paths", []):
                bound = _bound(dict(path=path, sha256=item["sha256"], size_bytes=item["size_bytes"]))
                source_files.add((bound["path"], bound["sha256"], bound["size_bytes"]))
    def observation_sources(observations):
        if not isinstance(observations, list) or not observations:
            raise ValueError("Original pressure observation has no source rows")
        for observation in observations:
            bound = _bound(observation["source"])
            if ((bound["path"], bound["sha256"], bound["size_bytes"]) not in source_files
                    or type(observation.get("sheet_row")) is not int or observation["sheet_row"] < 1):
                raise ValueError("Pressure observation is outside its bound GFI source manifest")
    for row in rows:
        if row["quality"] == "observed":
            observation_sources(row["observations"])
    timestamps = [utc(row["utc"]) for row in rows]
    if len(set(timestamps)) != len(timestamps) or timestamps != sorted(timestamps):
        raise ValueError("Original pressure timestamps must be distinct and ordered")
    filled = {utc(row["utc"]): row for row in rows
              if row.get("quality") == "provisional_short_gap_mean_pm30min"}
    expected = set()
    for gap in atmosphere["gaps"]:
        count = gap.get("missing_stamp_count")
        first, last = utc(gap["first_utc"]), utc(gap["last_utc"])
        if (type(count) is not int or count not in (1, 2)
                or gap.get("filled_count") != count or gap.get("resolved_provisionally") is not True
                or last - first != timedelta(minutes=10 * (count - 1))):
            raise ValueError("The pressure decision covers only the original one/two-stamp flagged gaps")
        expected.update(first + timedelta(minutes=10 * offset) for offset in range(count))
    if not expected or expected != set(filled):
        raise ValueError("Original filled rows do not match the flagged short gaps")
    original_observations = {utc(row["utc"]): row for row in rows if row.get("quality") == "observed"}
    for stamp, row in filled.items():
        observations = row.get("observations")
        if not isinstance(observations, list) or len(observations) < 4:
            raise ValueError("Original short-gap mean lacks four bound observations")
        moments = [utc(item["utc"]) for item in observations]
        if (len(set(moments)) != len(moments) or moments != sorted(moments)
                or not min(moments) < stamp < max(moments)
                or any(not stamp - timedelta(minutes=30) <= moment < stamp + timedelta(minutes=30)
                       for moment in moments)):
            raise ValueError("Original short-gap observations differ from the recorded half-open averaging window")
        available = {moment for moment in original_observations
                     if stamp - timedelta(minutes=30) <= moment < stamp + timedelta(minutes=30)}
        if not available.issubset(set(moments)):
            raise ValueError("The short-gap mean omits original observations in its window")
        for moment, observation in zip(moments, observations):
            # The producer retains original observations with their own source
            # descriptors; filled values can never become a neighbouring input.
            if observation.get("quality") == "provisional_short_gap_mean_pm30min":
                raise ValueError("A filled pressure value cannot feed another mean")
            observation_sources(observation["observations"])
            if moment in original_observations:
                actual = original_observations[moment]
                expected_observation = {key: actual[key] for key in
                                        ("utc", "station_pressure_hpa", "observations")}
                if not _same(observation, expected_observation):
                    raise ValueError("Short-gap source observation differs from its original row")
        mean = math.fsum(item["station_pressure_hpa"] for item in observations) / len(observations)
        if (not _same(row["station_pressure_hpa"], mean)
                or not _same(row["pressure_bar"], mean * atmosphere["height_term"]["factor"] / 1000)):
            raise ValueError("The existing short-gap values differ from the original mean recipe")


def verify_accepted_pressure(atmosphere, *, catalog=None, use="pressure_policy"):
    """Verify a policy-bearing ABI document; legacy documents are untouched.

    Catalogue mode admits every active evidence copy as a named rooted input.
    The historical original locators are never opened in place of those copies.
    """
    if FIELD not in atmosphere:
        return
    proof = atmosphere[FIELD]
    if (not isinstance(proof, dict)
            or set(proof) != {"format", "policy", "run", "series", *_SOURCES}
            or proof["format"] != FORMAT or proof["policy"] != POLICY
            or not isinstance(proof["series"], str) or not proof["series"]
            or proof["run"] != atmosphere.get("run")):
        raise ValueError("Unknown or incomplete accepted-pressure policy")
    raw = {}
    for key in _SOURCES:
        pair = proof[key]
        if key == "authority":
            raw[key] = _raw(_bound(pair), catalog=catalog, use=use + "." + key)
            continue
        if not isinstance(pair, dict) or set(pair) != {"original", "copy"}:
            raise ValueError("Pressure policy needs exact original and copied source bindings")
        original, copied = _bound(pair["original"]), _bound(pair["copy"])
        if any(original[field] != copied[field] for field in ("sha256", "size_bytes")):
            raise ValueError("Pressure evidence copy changes original bytes")
        raw[key] = _raw(copied, catalog=catalog, use=use + "." + key)
    authorization = _checked_authorization(raw["authority"], proof)
    series, registered = _registered(raw["series_manifest"])
    if proof["series"] != series or proof["run"] not in registered:
        raise ValueError("Accepted pressure run is outside its registered series")
    source_document = _json(raw["source_manifest"])
    original = _json(raw["source_atmosphere"])
    _original(_recorded_index(raw["source_index"], authorization["source_index"]), original, proof["run"],
        series=series, runs=authorization["runs"], source_document=source_document,
        **{key: proof[key]["original"] for key in ("source_atmosphere", "source_manifest")})
    expected = deepcopy(original)
    expected["provisional_gap_default"] = False
    expected["gap_default_authority"] = gap_authority(authorization)
    expected[FIELD] = deepcopy(proof)
    if not _same(atmosphere, expected):
        raise ValueError("Accepted pressure changes original DATA or its decision provenance")


def pressure_policy_sources_path(destination):
    """The publisher's one fresh evidence namespace; also used by the GUI."""
    destination = Path(destination)
    return destination.with_name(destination.stem + "_policy_sources")


def prepare_accepted_pressure_index(pressure_index, destination, *, authority, series_root,
                                    source_aliases=()):
    """Publish a fresh inert index from the series' recorded pressure decision.

    The series manifest names the decision record, who decided, the runs and
    the pinned index, manifest and recipe (``[decisions.accepted_pressure]``).
    """
    from .pipeline.alias_job import _manifest
    from .pipeline.job_inputs import InputCatalog
    from .series_decisions import series_decisions_from_bytes
    destination = Path(destination)
    folder = pressure_policy_sources_path(destination)
    if (destination.exists() or destination.is_symlink() or folder.exists() or folder.is_symlink()
            or destination.suffix.casefold() != ".json"):
        raise FileExistsError("Choose a fresh accepted-pressure JSON and evidence namespace")
    destination = destination.resolve()
    folder = folder.resolve()
    manifest = descriptor(Path(series_root) / "series.toml")
    _, _, roots = _manifest(manifest)  # Original schema/root admission remains authoritative.
    # Use the original same-byte alias contract for relocated source metadata.
    # These aliases never change the index/table/manifest identities and are
    # unnecessary on the original host. Published proof copies are self-bound.
    if not isinstance(source_aliases, (list, tuple)):
        raise ValueError("Pressure source aliases must use the original explicit same-byte contract")
    resolver = InputCatalog({}, source_aliases)
    def read_source(bound):
        actual = resolver.physical(_bound(bound))
        _bound(actual)
        if actual["size_bytes"] != bound["size_bytes"]:
            raise ValueError("Pressure source alias changes original size")
        return _raw(actual)
    originals = dict(authority=deepcopy(authority), source_index=deepcopy(pressure_index),
                     series_manifest=manifest)
    raw = {key: read_source(value) for key, value in originals.items()}
    settings = pressure_settings(series_decisions_from_bytes(raw["series_manifest"]))
    _authority(raw["authority"], settings)
    series, registered = _registered(raw["series_manifest"])
    index = _recorded_index(raw["source_index"], settings["source_index"])
    for key in ("source_manifest",):
        originals[key] = deepcopy(index[key])
        raw[key] = read_source(originals[key])
    if (_bound(originals["source_manifest"])["sha256"], originals["source_manifest"]["size_bytes"]) != (
            settings["source_manifest"]["sha256"], settings["source_manifest"]["size_bytes"]):
        raise ValueError("The recorded pressure index names another GFI source manifest")
    source_document = _json(raw["source_manifest"])
    copies, payloads = {}, {}
    def copied(key, bound, data):
        path = folder / key
        value = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        payloads[path] = data
        return dict(original=deepcopy(bound), copy=value)
    for key in originals:
        if key != "authority":
            copies[key] = copied(key + (".toml" if key == "series_manifest" else ".json"),
                                 originals[key], raw[key])
    authorization = human_authorization(settings, series)
    authorization_data = _immutable_bytes(authorization)
    authorization_path = folder / "authorization.json"
    payloads[authorization_path] = authorization_data
    copies["authority"] = dict(path=str(authorization_path),
        sha256=hashlib.sha256(authorization_data).hexdigest(), size_bytes=len(authorization_data))
    updated, changed = deepcopy(index), []
    for run in registered:
        entry = index["runs"][series].get(run)
        if not isinstance(entry, dict) or entry.get("complete") is not True:
            raise ValueError("The selected pressure index does not cover registered run " + run)
        if entry.get("provisional_gap_default") is not True:
            continue
        original_bound = entry["atmosphere_series"]
        source_raw = read_source(original_bound)
        atmosphere = _json(source_raw)
        _original(index, atmosphere, run, series=series, runs=settings["runs"],
                  data_builder=settings["data_builder"], source_atmosphere=original_bound,
                  source_manifest=originals["source_manifest"], source_document=source_document)
        proof = dict(format=FORMAT, policy=POLICY, run=run, series=series, **deepcopy(copies))
        proof["source_atmosphere"] = copied(run + "_original.json", original_bound, source_raw)
        accepted = deepcopy(atmosphere)
        accepted["provisional_gap_default"] = False
        accepted["gap_default_authority"] = gap_authority(authorization)
        accepted[FIELD] = proof
        data = _immutable_bytes(accepted)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("Accepted pressure metadata exceeds the original byte limit")
        path = folder / (run + "_accepted.json")
        payloads[path] = data
        updated["runs"][series][run]["atmosphere_series"] = dict(
            path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        updated["runs"][series][run]["provisional_gap_default"] = False
        changed.append(run)
    if not changed:
        raise ValueError("No registered provisional short gaps need this recorded policy")
    updated[FIELD] = dict(format=FORMAT, policy=POLICY, runs=changed,
        source_index=deepcopy(copies["source_index"]), authority=deepcopy(copies["authority"]),
        series_manifest=deepcopy(copies["series_manifest"]))
    data = _immutable_bytes(updated)
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError("Accepted pressure index exceeds the original byte limit")
    # Refuse unusable output locators before creating anything. The later real
    # queue repeats rooted admission and SHA checks on these same named copies.
    from .store_map import input_reference
    for path, content in [(destination, data), *payloads.items()]:
        input_reference(path, roots, sha256=hashlib.sha256(content).hexdigest(),
                        size_bytes=len(content), role="configuration")
    # Repeat every selected original binding before the first publication.
    for key, bound in originals.items():
        if read_source(bound) != raw[key]:
            raise ValueError("Pressure source changed during preparation")
    for run in changed:
        bound = index["runs"][series][run]["atmosphere_series"]
        if read_source(bound) != payloads[folder / (run + "_original.json")]:
            raise ValueError("Original atmosphere changed during preparation")
    folder.mkdir(parents=False, exist_ok=False)
    for path, content in payloads.items():
        _publish_bytes(path, content, create_parents=False)
    _publish_bytes(destination, data, create_parents=False)  # Index published last.
    return dict(pressure_index=descriptor(destination), runs=changed,
                covered_runs=registered, plan_prepared=False, worker_started=False)
