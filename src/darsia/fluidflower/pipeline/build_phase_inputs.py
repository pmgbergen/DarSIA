"""Assemble existing phase-only DATA from exact original acquisition declarations.

Explicit original RAW-preparation reference/history specs and complete selected
sources are caller inputs. Replacement geometry/area/static/transform DATA comes
from its genuine producers. This assembler reads metadata only, writes nothing,
and neither adapts a grid nor grants scientific eligibility.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..mass.build_native_bridge import BuildNativeBridge
from ..mass.n1_pipeline import DataReader
from ..mass.native_optical_source import _bound, _same
from .build_raw_phase import FORMAT, _ROLES, _specification


@dataclass(frozen=True)
class BuildPhaseReplacement:
    """Explicit source-bound products; no inferred geometry or default transforms."""
    raw_acquisition: Mapping
    geometry: Mapping
    area: Mapping
    static: Mapping
    current_h: Sequence
    reference_to_current_h: Sequence


def assemble_build_phase_inputs(
    bridge: BuildNativeBridge, old_run_configuration: Mapping,
    reference_spec: Mapping, initial_previous_spec: Mapping,
    selected_sources: Sequence[Mapping], replacements: Sequence[BuildPhaseReplacement],
    *, phase_configuration: Mapping, temporal_pad_px: int, reader=None,
) -> dict:
    """Join literal old roles to supplied current products in the original format.

    The selected cohort must be the complete old cohort, in its existing order.
    Reference/previous specs are original RAW-preparation acquisition declarations;
    this function does not discover or synthesize them. Grid compatibility remains
    the original collector's responsibility. Historical tasks, mass inputs and
    positional caches never become current inputs.
    """
    if not isinstance(bridge, BuildNativeBridge):
        raise TypeError("The original completed-Build native bridge is required")
    reader = DataReader() if reader is None else reader
    pins = []

    def document(binding):
        bound = deepcopy(_bound(dict(binding)))
        pins.append(bound)
        return reader.json(bound)

    old = document(old_run_configuration)
    phase_binding = deepcopy(_bound(dict(phase_configuration)))
    before_phase = document(old["phase"]["configuration"])
    after_phase = document(phase_binding)
    native_binding = deepcopy(bridge.context.provenance["native_configuration"])
    before_native = document(old["native"]["configuration"])
    after_native = document(native_binding)
    if (old["run"] != bridge.context.run
            or not _same(before_phase["chain_assets"]["floor"], after_phase["chain_assets"]["floor"])
            or not _same(before_phase["parameters"]["geometry"]["floor_source_origin_xy"],
                         after_phase["parameters"]["geometry"]["floor_source_origin_xy"])):
        raise ValueError("The original run/floor source or its source origin changed")
    if not _same(before_native["support"], after_native["support"]):
        raise ValueError("Original photographic support configuration changed")
    frames = old["frames"]
    if (not frames or len(selected_sources) != len(frames)
            or any(not _same(source, spec["source"]) for source, spec in zip(selected_sources, frames))):
        raise ValueError("The complete original selected source cohort/order must remain exact")
    reference, previous = old["phase"]["reference"], old["phase"]["initial_previous"]
    if (reference_spec["source"]["source_sha256"] != reference["source_sha256"]
            or reference["source_sha256"] != reference["full"]["source_sha256"]
            or not _same(initial_previous_spec["source"], previous["source"])
            or previous["source"]["source_sha256"] != previous["full"]["source_sha256"]):
        raise ValueError("Original reference/history acquisition declarations differ")

    def identity(source):
        raw = source["source_binding"]["raw_acquisition"]
        if raw["sha256"] != source["source_sha256"]:
            raise ValueError("The declared RAW identity differs from its acquisition source SHA")
        return raw["path"], raw["sha256"]

    originals = [reference_spec, initial_previous_spec, *frames]
    identities = {identity(spec["source"]) for spec in originals}
    products = {}
    for item in replacements:
        if not isinstance(item, BuildPhaseReplacement):
            raise TypeError("Explicit typed Build phase replacements are required")
        raw = _bound(dict(item.raw_acquisition))
        key = raw["path"], raw["sha256"]
        if key in products:
            raise ValueError("Each original RAW acquisition needs exactly one replacement record")
        products[key] = item
    if set(products) != identities:
        raise ValueError("Replacement RAW path/SHA coverage differs from the original acquisitions")
    baseline = products[identity(reference_spec["source"])].raw_acquisition
    if not _same(dict(baseline), bridge.context.provenance["raw_acquisition"]):
        raise ValueError("The original reference differs from the completed Build baseline RAW")

    def assemble(spec, *, selected):
        item = products[identity(spec["source"])]
        roles = {key: deepcopy(spec["phase_roles"][key]) for key in _ROLES}
        roles.update(area=deepcopy(dict(item.area)), static=deepcopy(dict(item.static)),
                     current_h=deepcopy(item.current_h), reference_to_current_h=deepcopy(item.reference_to_current_h),
                     retained_source_cache=None, encoding_contract=None)
        result = {key: deepcopy(spec[key]) for key in ("case_id", "frame_index", "elapsed_h", "source")}
        result.update(raw_acquisition=deepcopy(dict(item.raw_acquisition)),
                      geometry=deepcopy(dict(item.geometry)), phase_roles=roles)
        for key in ("provenance", "context_acceptance"):
            if key in spec:
                result[key] = deepcopy(spec[key])
        if "raw_acquisition" in spec and any(
                spec["raw_acquisition"][key] != result["raw_acquisition"][key] for key in ("path", "sha256")):
            raise ValueError("Replacement RAW path/SHA differs from its original acquisition")
        _specification(result, old["run"], old["selection_index_field"], selected=selected)
        return result

    native = deepcopy(old["native"])
    native["configuration"] = native_binding
    result = dict(format=FORMAT, run=old["run"], native=native,
                  phase=dict(configuration=phase_binding), history_policy=deepcopy(old["history_policy"]),
                  selection_index_field=old["selection_index_field"], temporal_pad_px=temporal_pad_px,
                  reference_spec=assemble(reference_spec, selected=False),
                  initial_previous_spec=assemble(initial_previous_spec, selected=False),
                  frames=[assemble(spec, selected=True) for spec in frames])
    for key in ("data_aliases", "provenance"):
        if key in old:
            result[key] = deepcopy(old[key])
    for binding in pins:
        reader.path(binding)
    return result


def assemble_current_build_phase_inputs(
    bridge: BuildNativeBridge, base_run_configuration: Mapping,
    reference_spec: Mapping, initial_previous_spec: Mapping,
    frame_specs: Sequence[Mapping], replacements: Sequence[BuildPhaseReplacement],
    *, phase_configuration: Mapping, temporal_pad_px: int,
    selection_record: Mapping, reader=None,
) -> dict:
    """Assemble new local selection DATA without relabelling an old cohort.

    The separately saved selection record owns acquisition identities and
    history. The old configuration supplies only explicitly retained scientific
    roles. Actual Build/RAW geometry, area, static and transform checks remain
    the original collector's responsibility; this assembler writes nothing.
    """
    if not isinstance(bridge, BuildNativeBridge):
        raise TypeError("The original completed-Build native bridge is required")
    reader = DataReader() if reader is None else reader
    pins = []

    def document(binding):
        bound = deepcopy(_bound(dict(binding)))
        pins.append(bound)
        return reader.json(bound)

    selection_bound = deepcopy(_bound(dict(selection_record)))
    selection = document(selection_bound)
    decision = selection.get("decision", {})
    if (selection.get("format") != "ffac-local-p1-selection-data-v1"
            or not isinstance(decision.get("actor"), str) or not decision["actor"].strip()
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
            or not isinstance(selection.get("history_policy"), str)
            or not selection["history_policy"].strip()):
        raise ValueError("Current phase DATA requires an explicit operator and selection record")
    current_series = _bound(dict(selection["input_authority"]["current_series"]))
    reuse = bridge.context.provenance.get("local_build_reuse")
    if reuse is not None and (not isinstance(reuse, dict)
            or not _same(reuse.get("current_series_manifest"), current_series)):
        raise ValueError("Current selection differs from the explicit local Build reuse")
    # A reuse receipt can bind this configuration only after it has been saved.
    # Do not put that future receipt in the configuration's own hash dependency.
    pins.append(deepcopy(current_series))
    reader.path(current_series)
    rows = selection.get("runs")
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or not isinstance(row.get("run"), str) for row in rows)
            or len({row["run"] for row in rows}) != len(rows)):
        raise ValueError("The current selection must name distinct complete run records")
    selected = [row for row in rows if row["run"] == bridge.context.run]
    if len(selected) != 1:
        raise ValueError("The current selection does not name this exact Build run")
    selected = selected[0]
    frames = deepcopy(list(frame_specs))
    reference, previous = deepcopy(dict(reference_spec)), deepcopy(dict(initial_previous_spec))
    declared = selected.get("selected_frames")
    if not isinstance(declared, list) or not declared or len(declared) != len(frames):
        raise ValueError("Current frame specifications must cover the complete selected order")
    old = document(base_run_configuration)
    phase_binding = deepcopy(_bound(dict(phase_configuration)))
    before_phase, after_phase = document(old["phase"]["configuration"]), document(phase_binding)
    native_binding = deepcopy(bridge.context.provenance["native_configuration"])
    before_native, after_native = document(old["native"]["configuration"]), document(native_binding)
    if (old["run"] != bridge.context.run
            or not _same(before_phase["chain_assets"]["floor"], after_phase["chain_assets"]["floor"])
            or not _same(before_phase["parameters"]["geometry"]["floor_source_origin_xy"],
                         after_phase["parameters"]["geometry"]["floor_source_origin_xy"])):
        raise ValueError("The original run/floor source or its source origin changed")
    if not _same(before_native["support"], after_native["support"]):
        raise ValueError("Original photographic support configuration changed")
    if type(temporal_pad_px) is not int or temporal_pad_px < 0:
        raise ValueError("The existing temporal padding convention must remain explicit")
    index_field = old["selection_index_field"]

    def matches(spec, row, *, is_selected):
        _specification(spec, old["run"], index_field, selected=is_selected)
        raw, source = spec["raw_acquisition"], spec["source"]
        if (raw["path"] != row.get("raw_path") or raw["sha256"] != row.get("raw_sha256")
                or not _same(spec["elapsed_h"], row.get("elapsed_h"))
                or not _same(spec["frame_index"], row.get("current_frame_index"))
                or source.get("previous_source_sha256") != row.get("previous_raw_sha256")
                or source.get("selected") is not is_selected):
            raise ValueError("Current frame differs from its bound RAW/index/time/history selection")
        bound_raw = source["source_binding"].get("raw_acquisition")
        if bound_raw is None or not _same(bound_raw, raw):
            raise ValueError("Current RAW source requires its exact acquisition descriptor")

    baseline = selected.get("baseline", {})
    matches(reference, baseline, is_selected=False)
    matches(previous, baseline, is_selected=False)
    if (not _same(reference, previous)
            or selected.get("initial_previous_raw_sha256") != baseline.get("raw_sha256")
            or baseline.get("previous_raw_sha256") is not None
            or not _same(reference["raw_acquisition"], bridge.context.provenance["raw_acquisition"])):
        raise ValueError("The current reference/history must retain the exact Build baseline")
    prior, last = baseline["raw_sha256"], (reference["elapsed_h"], reference["frame_index"])
    seen, cases = {prior}, {reference["case_id"]}
    for spec, row in zip(frames, declared):
        matches(spec, row, is_selected=True)
        digest, position = spec["source"]["source_sha256"], (spec["elapsed_h"], spec["frame_index"])
        if (digest in seen or spec["case_id"] in cases or row["previous_raw_sha256"] != prior
                or position[0] <= last[0] or position[1] <= last[1]):
            raise ValueError("Current selection changed its distinct chronological predecessor chain")
        seen.add(digest)
        cases.add(spec["case_id"])
        prior, last = digest, position

    def identity(spec):
        raw = spec["raw_acquisition"]
        return raw["path"], raw["sha256"]

    identities = {identity(spec) for spec in (reference, previous, *frames)}
    products = {}
    for item in replacements:
        if not isinstance(item, BuildPhaseReplacement):
            raise TypeError("Explicit typed Build phase replacements are required")
        raw = _bound(dict(item.raw_acquisition))
        key = raw["path"], raw["sha256"]
        if key in products:
            raise ValueError("Each current RAW acquisition needs exactly one replacement record")
        products[key] = item
    if set(products) != identities:
        raise ValueError("Replacement RAW path/SHA coverage differs from the current acquisitions")

    def assemble(spec, *, is_selected):
        item = products[identity(spec)]
        if not _same(dict(item.raw_acquisition), spec["raw_acquisition"]):
            raise ValueError("Replacement RAW descriptor differs from its current acquisition")
        result = deepcopy(spec)
        roles = {key: deepcopy(spec["phase_roles"][key]) for key in _ROLES}
        roles.update(area=deepcopy(dict(item.area)), static=deepcopy(dict(item.static)),
                     current_h=deepcopy(item.current_h), reference_to_current_h=deepcopy(item.reference_to_current_h),
                     retained_source_cache=None, encoding_contract=None)
        result.update(raw_acquisition=deepcopy(dict(item.raw_acquisition)),
                      geometry=deepcopy(dict(item.geometry)), phase_roles=roles)
        _specification(result, old["run"], index_field, selected=is_selected)
        return result

    native = deepcopy(old["native"])
    native["configuration"] = native_binding
    result = dict(format=FORMAT, run=old["run"], native=native,
                  phase=dict(configuration=phase_binding), history_policy=selection["history_policy"],
                  selection_index_field=index_field, temporal_pad_px=temporal_pad_px,
                  reference_spec=assemble(reference, is_selected=False),
                  initial_previous_spec=assemble(previous, is_selected=False),
                  frames=[assemble(spec, is_selected=True) for spec in frames],
                  provenance=dict(local_current_selection=dict(
                      original_base_configuration=deepcopy(dict(base_run_configuration)),
                      selection_record=selection_bound,
                      current_series_manifest=deepcopy(current_series), accepted_by=decision["actor"],
                      reason=decision["reason"], historical_configuration_relabelled=False,
                      scientific_qualification_claimed=False)))
    if "data_aliases" in old:
        result["data_aliases"] = deepcopy(old["data_aliases"])
    for binding in pins:
        reader.path(binding)
    return result
