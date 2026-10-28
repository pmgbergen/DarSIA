"""The recorded overflow method, separated from private coordination answers.

This metadata does not admit a worker, a fit, source payloads or production.
An owned preparation must additionally replay its completed original P1 inputs,
measurement sources and derived statistics before the separate F fit is allowed.

Who authorized the method, and which pinned protocol, addendum and review it
uses, are read from the series configuration (``[decisions.overflow_method]``;
see series_decisions). A saved method record names them itself and is checked
to be complete and unchanged.
"""
from copy import deepcopy
import hashlib

FORMAT = "darsia-overflow-steering-method-v1"
SETTING = "overflow_method"
_SOURCE_KEYS = (("protocol", "protocol"), ("addendum", "addendum"), ("overflow_review", "overflow review"))
_AUTHORIZATION_KEYS = ("actor", "confirmed_utc", "reason", "conditional_anchor_rule")
USABLE_RUNS = ("ac19", "ac21", "ac22", "ac25", "ac28", "ac30", "ac33", "ac34", "ac35", "ac39",
    "ac41", "ac42", "ac43", "ac44", "ac49", "ac52", "ac54", "ac55", "ac56", "ac57", "ac58", "ac59", "ac60", "ac61")
GROUP_CANDIDATES = ("ac28", "ac30", "ac33", "ac34", "ac39", "ac42", "ac43", "ac49", "ac54", "ac55", "ac57", "ac58", "ac59", "ac61")
NO_USABLE_OVERFLOW = ("ac17", "ac18", "ac23", "ac26", "ac29", "ac31", "ac32", "ac38", "ac40", "ac45", "ac46", "ac47", "ac48", "ac50", "ac53")
CONDITIONAL_ANCHOR_RUNS = ("ac34", "ac41")
HELDOUT_RUNS = ("ac19", "ac25", "ac35", "ac41", "ac47", "ac52", "ac56", "ac60", "ac21", "ac22", "ac32", "ac44")
EXCLUDED_RUNS = ("ac20", "ac24", "ac27", "ac51")

def human_authorizations(value):
    """Validate plain human authorizations: who, when and why, with an optional anchor rule."""
    from ..series_decisions import utc_text
    if not isinstance(value, list) or not value:
        raise ValueError("The overflow method needs its recorded human authorizations")
    result = []
    for row in value:
        if (not isinstance(row, dict) or not {"actor", "confirmed_utc", "reason"} <= set(row)
                or not set(row) <= set(_AUTHORIZATION_KEYS)
                or any(not isinstance(row[key], str) or not row[key].strip()
                       for key in ("actor", "reason", "conditional_anchor_rule") if key in row)):
            raise ValueError("Each overflow authorization needs who decided, when and why")
        utc_text(row["confirmed_utc"], "overflow authorization time")
        result.append({key: row[key] for key in _AUTHORIZATION_KEYS if key in row})
    return result


def method_settings(decisions):
    """Validate the series' recorded overflow method settings and return a plain copy."""
    from ..series_decisions import DecisionSettingUnavailable, checked_pin, decision_setting
    value = decision_setting(decisions, SETTING)
    keys = {"protocol", "addendum", "overflow_review", "records", "authorizations"}
    if not isinstance(value, dict) or set(value) != keys:
        raise DecisionSettingUnavailable("decisions.overflow_method needs exactly: " + ", ".join(sorted(keys)))
    records = value["records"]
    if (not isinstance(records, dict) or not records
            or any(not isinstance(key, str) or not key or not isinstance(digest, str) or len(digest) != 64
                   or any(character not in "0123456789abcdef" for character in digest)
                   for key, digest in records.items())):
        raise DecisionSettingUnavailable("decisions.overflow_method.records maps record ids to canonical SHA-256")
    try:
        authorizations = human_authorizations(value["authorizations"])
    except ValueError as error:
        raise DecisionSettingUnavailable("decisions.overflow_method.authorizations: " + str(error)) from error
    return dict({key: checked_pin(value[key], name) for key, name in _SOURCE_KEYS},
                records=dict(records), authorizations=authorizations)


def check_recorded_decisions(value, records):
    """Privately check the series' named existing decisions; write nothing.

    ``records`` maps each record id to its canonical SHA-256 (the series'
    ``decisions.overflow_method.records``). Record ids and text never enter
    the product record.
    """
    from ..series_decisions import find_decision_record
    for identifier, expected in records.items():
        find_decision_record(value, identifier, digest=expected, where="overflow decisions")


def _source(value, name, expected=None):
    """An exact bound source descriptor; with ``expected`` also its pinned SHA-256 and size."""
    from ..pipeline.staged_data import _bound
    _bound(value, where=name)
    if expected is not None and {key: value[key] for key in expected} != expected:
        raise ValueError("The overflow method needs its exact accepted " + name)
    return deepcopy(value)


def method_document(*, protocol, addendum, review, settings):
    """Plain immutable DATA only, never a source/admission success receipt.

    ``settings`` are the series' recorded method settings (method_settings):
    the pinned sources and the human authorizations.
    """
    sources = dict(protocol=protocol, addendum=addendum, overflow_review=review)
    return _document({key: _source(sources[key], name, settings[key]) for key, name in _SOURCE_KEYS},
                     settings["authorizations"])


def _document(sources, authorizations):
    """The canonical method record for checked sources and human authorizations."""
    return dict(format=FORMAT, authorizations=deepcopy(authorizations),
        protocol=sources["protocol"], addendum=sources["addendum"], overflow_review=sources["overflow_review"],
        feed="Original feed C anchored at the last surviving pre-injection reviewed reading",
        ratio_window_h=[1., 4.], minimum_model_gas_volume_ml=3.,
        uncertainty="Original feed-model spread/2 plus late residual SD; median per-frame ratio uncertainty; no division by square-root frame count",
        group="Precision-weighted admitted fit-run mean; method-of-moments between-run variance",
        historical_group_candidates=list(GROUP_CANDIDATES),
        conditional_anchor_runs=list(CONDITIONAL_ANCHOR_RUNS),
        level="Shrink the run ratio to the common fit group; cap the physical run level at one",
        redistribution="s_g=(1-Swi)*min(1,p/p_star), using the stored decision-used float32 pixel probability",
        certain_gas_probability=1., unreachable_target="cap_to_available_capacity_and_report",
        pressure_in_level=False, phase_labels_modified=False, physical_geometry_modified=False,
        fit_gas_gain=False, recompute_group_per_fold=False,
        report=["gas_after_minus_phase_model_gas_mass_and_share",
                "gas_on_original_non_gas_pixels_mass_and_share", "capped_amount"],
        reference_without_overflow="Separate fit/freeze and mass tables; no second pixel-map set",
        calibration_admitted=False, production_admitted=False)


def validate_method(value, *, settings=None):
    """Exact canonical record; source byte reads remain with checked_method.

    The record names its own sources and human authorizations. With the
    series' ``settings`` it must also name exactly those.
    """
    from ..pipeline.staged_data import _same_json
    if type(value) is not dict:
        raise ValueError("An explicit recorded overflow method is required")
    try:
        sources = {key: _source(value[key], name, None if settings is None else settings[key])
                   for key, name in _SOURCE_KEYS}
        authorizations = (human_authorizations(value["authorizations"]) if settings is None
                          else settings["authorizations"])
    except KeyError as exc:
        raise ValueError("The recorded overflow method lacks its accepted sources") from exc
    if not _same_json(_document(sources, authorizations), value):
        raise ValueError("The recorded overflow method or human authorization changed")
    return deepcopy(value)


def _source_reader(reader):
    """Keep the original byte verifier while following declared storage moves."""
    from .n1_pipeline import DataReader
    from ..pipeline.current_inputs import CurrentInputReader, current_input_reader
    if type(reader) is DataReader:
        return current_input_reader(reader)
    if type(reader) is CurrentInputReader and type(reader.reader) is DataReader:
        return reader
    raise TypeError("An original DATA reader is required")


def checked_metadata(reader, bound, *, limit):
    """Bounded strict JSON through the original content-verifying DATA reader."""
    from .n1_pipeline import _strict_json
    from ..pipeline.staged_data import _bound
    reader = _source_reader(reader)
    if type(limit) is not int or limit < 1:
        raise TypeError("An original DATA reader and explicit metadata limit are required")
    _bound(bound, where="overflow metadata")
    if bound["size_bytes"] > limit:
        raise ValueError("Overflow metadata exceeds its explicit byte limit")
    path = reader.path(bound)
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    if len(raw) != bound["size_bytes"] or hashlib.sha256(raw).hexdigest() != bound["sha256"]:
        raise ValueError("Overflow metadata differs from its bound original bytes")
    value = _strict_json(raw)
    reader.path(bound)
    return value


def checked_method(reader, bound, *, settings=None):
    """Check the clean method plus its three actual original source files."""
    reader = _source_reader(reader)
    value = validate_method(checked_metadata(reader, bound, limit=64*1024), settings=settings)
    for field, _name in _SOURCE_KEYS:
        source = value[field]
        path = reader.path(source)
        # The two specifications are raw Markdown, never guessed to be JSON.
        with path.open("rb") as stream:
            raw = stream.read(source["size_bytes"]+1)
        if len(raw) != source["size_bytes"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("An accepted overflow method source changed")
        reader.path(source)
    return value


def validate_roles(roles, analysis_runs, *, cohort=None):
    """Retain the accepted cohort and fit/held-out separation without reordering."""
    def run_list(value):
        return (type(value) is list and all(type(run) is str and run for run in value)
                and len(set(value)) == len(value))
    if (type(roles) is not dict or set(roles) != {"fit", "heldout", "descriptive", "excluded"}
            or not run_list(analysis_runs) or not all(run_list(value) for value in roles.values())):
        raise ValueError("Explicit distinct original run roles and analysis order are required")
    if cohort is not None:
        if (type(cohort) is not dict or cohort.get("format") != "darsia-current-mass-cohort-v1"
                or cohort.get("roles") != roles or not run_list(cohort.get("analysis_runs"))
                or set(cohort["analysis_runs"]) != set(analysis_runs) or not roles["fit"]):
            raise ValueError("The current overflow fit needs its exact saved calibration cohort")
        sets = [set(roles[key]) for key in ("fit", "heldout", "descriptive", "excluded")]
        if (any(left & right for index, left in enumerate(sets) for right in sets[index+1:])
                or set(analysis_runs) != sets[0] | sets[1] | sets[2]):
            raise ValueError("Fit, holdout, descriptive and excluded runs must remain disjoint")
        return deepcopy(roles)
    cohort = set(USABLE_RUNS) | set(NO_USABLE_OVERFLOW)
    fit = cohort - set(HELDOUT_RUNS) - {"ac38"}
    if (set(analysis_runs) != cohort or set(roles["fit"]) != fit
            or set(roles["heldout"]) != set(HELDOUT_RUNS)
            or roles["descriptive"] != ["ac38"] or set(roles["excluded"]) not in (set(), set(EXCLUDED_RUNS))):
        raise ValueError("The recorded overflow method needs its unchanged accepted run roles")
    if set(GROUP_CANDIDATES) != set(roles["fit"]) & set(USABLE_RUNS):
        raise ValueError("The declared historical group candidates differ from the admitted fit roles")
    return deepcopy(roles)
