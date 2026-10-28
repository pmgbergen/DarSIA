"""Couple original RAW/phase-one replay to an unchanged sealed N1 task.

This module runs no science and changes no input identity or runtime guard.
The caller first executes the original phase-one entry point in its admitted
RunIOBinding. Both complete receipt chains are then content verified. Fresh
binary scientific inputs replace physical locations only after exact SHA/size
equality; original receipts remain historical evidence. Statistics gzip files
retain their original container after exact decompressed-byte comparison,
because the original writer records a fresh gzip timestamp on each run.
Declared phase QA retains both original and replayed top/floor execution
timings while every other scientific field remains subject to exact comparison.
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import math
from pathlib import Path

FORMAT = "darsia-staged-reproduction-coupling-v1"

# The direct top/floor dictionaries written by phase.rules and retained by
# phase.infer in the frame's declared phase_qa.json. Unknown producer schemas
# refuse; a timing-like key anywhere else remains an exact scientific field.
_PHASE_TIMINGS = ("cpu_seconds", "wall_seconds", "gpu_seconds")
_PHASE_RULE_QA_FIELDS = {
    "top": frozenset("""
        removed_top_components excluded_columns acting_columns top_median_row
        all_pass after_a_co2_union_equals_base after_a_gas_equals_p1
        changed_pixels_exactly_top_flags removed_components_closed
        removed_components_within_fixed_band labels_outside_masks_identical
        rig_domain_unchanged removed_floor_pixels removed_top_pixels
        removed_floor_cm2 removed_top_aqueous_cm2 removed_top_gas_cm2
        nominal_removed_top_aqueous_cm2 nominal_removed_top_gas_cm2
        remaining_co2_60mm_cm2 remaining_co2_90mm_cm2 retained_60mm_reason
        backend gpu_seconds cpu_seconds wall_seconds training inference
    """.split()),
    "floor": frozenset("""
        floor_median_row acting_columns excluded_columns co2_connectivity_groups
        floor_connectivity_removed_groups
        floor_connectivity_removed_group_min_distance_to_kept_co2_mm
        floor_connectivity_distance_status floor_connectivity_distance_median_mm
        floor_connectivity_distance_max_mm floor_connectivity_removed_gas_pixels
        fill_reason_a_columns fill_reason_b_columns carryover_mask_known
        filled_under_gas_pixels fill_invalid_trigger_columns fill_unsupported_pixels
        preinjection_fill_pixels backend requested_backend cuda_fallback_reason
        step_pixels step_area_cm2 step_previous_aqueous_cm2 step_previous_gas_cm2
        fill_reason_a_cm2 fill_reason_b_cm2 mask_overlaps_pixels
        mask_overlaps_on_changed_pixels unchanged_stage_restoration_pixels
        changed_pixels flag_not_final_changed_pixels changed_unflagged_pixels
        labels_outside_masks_identical gas_changed_outside_a_d_pixels
        co2_union_changed_outside_a_b_c_pixels domain_changed_outside_a_pixels
        floor_boundary_matches fill_flags_are_new_aqueous gas_flags_are_gas_to_aqueous
        size_filter_reapplied training inference cpu_seconds gpu_seconds wall_seconds all_pass
    """.split()),
}


def _phase_rule_timing_values(value, rule):
    """Validate only the known producer's finite elapsed-time observations."""
    if type(value) is not dict or set(value) != _PHASE_RULE_QA_FIELDS[rule]:
        raise ValueError("Unrecognized phase_qa.json " + rule + " producer schema")
    result = {field: value[field] for field in _PHASE_TIMINGS}
    # Both original producers initialize GPU time to 0.0 and obtain CPU/wall
    # differences and CUDA event durations as floats. None/bool/int are not
    # observations from this schema, including on its CPU/no-CUDA path.
    if any(type(item) is not float or not math.isfinite(item) or item < 0
           for item in result.values()):
        raise ValueError("Invalid phase_qa.json " + rule + " execution timing observation")
    return result


def _identity(bound):
    return bound["path"], bound["sha256"], bound["size_bytes"]


def _same(left, right, label):
    from ..pipeline.staged_data import _same_json

    if not _same_json(left, right):
        raise ValueError("Staged reproduction differs: " + label)


def _relocated(value, replacements):
    """Normalize only previously byte-proven descriptor locators for comparison."""
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"} <= value.keys():
            previous = replacements.get(_identity(value))
            if previous is not None:
                return dict(value, path=previous["path"])
        return {key: _relocated(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocated(item, replacements) for item in value]
    return value


def _metadata_relocated(value, replacements):
    """Normalize a generated descriptor only after its own comparison closed."""
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"} <= value.keys():
            previous = replacements.get(_identity(value))
            if previous is not None:
                return dict(value, **{key: previous[key] for key in ("path", "sha256", "size_bytes")})
        return {key: _metadata_relocated(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_relocated(item, replacements) for item in value]
    return value


def _frame_metadata(old, new, original_reader, replay_reader, statistics):
    """Compare every declared output, preserving genuine runtime observations.

    Metadata may point to other generated metadata whose container bytes changed
    with its locators, executing-thread identities or known phase-rule timings.
    Follow only paired declared inventory descriptors, verify that dependency
    first, and retain both actual
    descriptors in the receipt. No rewritten document is supplied to science.
    """
    from darsia._opencv_runtime import validate_observation

    initial, fresh = old["files"], new["files"]
    _same(sorted(initial), sorted(fresh), "complete generated frame inventory")
    declared = {_identity(bound): name for name, bound in fresh.items()}
    proven, visiting, observations, input_paths, files = {}, set(), [], [], []
    phase_timings = []
    stats_key = _identity(statistics["replayed"])

    def observation(value):
        validate_observation(value)
        if "before" in value:
            return dict(value, before=observation(value["before"]), after=observation(value["after"]))
        return {key: item for key, item in value.items()
                if key not in {"pid", "thread_id", "native_thread_id"}}

    def compare(before, after, label, *, phase_qa=False, field_path=()):
        if isinstance(before, dict) and isinstance(after, dict):
            _same(sorted(before), sorted(after), label + " fields")
            timing_fields = ()
            if phase_qa and field_path in (("top",), ("floor",)):
                rule = field_path[0]
                original_timings = _phase_rule_timing_values(before, rule)
                replayed_timings = _phase_rule_timing_values(after, rule)
                phase_timings.append(dict(
                    where=label, producer="phase.rules." + (
                        "apply_top_rule" if rule == "top" else "apply_floor_round"),
                    original=original_timings, replayed=replayed_timings,
                    original_file=deepcopy(initial["phase_qa.json"]),
                    replayed_file=deepcopy(fresh["phase_qa.json"])))
                timing_fields = _PHASE_TIMINGS
            if {"path", "sha256"} <= before.keys():
                key = _identity(after) if "size_bytes" in after else None
                name = declared.get(key)
                if name is not None:
                    _same({field: before[field] for field in ("path", "sha256", "size_bytes")},
                          initial[name], label + " declared original descriptor")
                    verify(name)
                    normalized = dict(after, **{field: before[field]
                        for field in ("path", "sha256", "size_bytes")})
                    _same(normalized, before, label)
                    return
                if before["path"] != after["path"]:
                    _same(dict(after, path=before["path"]), before, label + " input identity")
                    # The genuine P1 binding, not matching content alone, must
                    # identify this exact new input location as admitted.
                    binding = replay_reader.io_binding
                    if binding is None:
                        raise ValueError("Metadata input relocation requires genuine P1 input admission")
                    admitted = Path(binding._descriptor_location(after)).resolve()
                    original_path = Path(original_reader.path(before)).resolve()
                    replayed_path = Path(replay_reader.path(after)).resolve()
                    if admitted != replayed_path:
                        raise ValueError("Metadata input relocation differs from its admitted P1 input")
                    input_paths.append(dict(original=deepcopy(before), replayed=deepcopy(after),
                        original_verified_path=str(original_path), admitted_path=str(admitted)))
                    return
            for key in before:
                if key in timing_fields:
                    continue
                if key == "opencv_ipp":
                    _same(observation(before[key]), observation(after[key]), label + ".opencv_ipp policy")
                    observations.append(dict(where=label + ".opencv_ipp", original=deepcopy(before[key]),
                                             replayed=deepcopy(after[key])))
                else:
                    compare(before[key], after[key], label + "." + key,
                            phase_qa=phase_qa, field_path=field_path + (key,))
        elif isinstance(before, list) and isinstance(after, list):
            _same(len(before), len(after), label + " length")
            for index, (left, right) in enumerate(zip(before, after)):
                compare(left, right, label + "." + str(index),
                        phase_qa=phase_qa, field_path=field_path + (index,))
        else:
            _same(before, after, label)

    def verify(name):
        before, after = initial[name], fresh[name]
        key = _identity(after)
        if key in proven:
            return
        if name in visiting:
            raise ValueError("Generated metadata has a cyclic descriptor dependency")
        visiting.add(name)
        original_reader.path(before)
        replay_reader.path(after)
        if key == stats_key:
            _same(before, statistics["original"], "declared original statistics")
            scope = "exact decompressed JSON bytes; original gzip retained"
        elif Path(name).suffix.lower() == ".json":
            phase_qa = name == "phase_qa.json" and (
                old.get("phase", {}).get("qa") == before
                and new.get("phase", {}).get("qa") == after)
            before_value, after_value = original_reader.json(before), replay_reader.json(after)
            if phase_qa:
                for value in (before_value, after_value):
                    if type(value) is not dict or not {"top", "floor"} <= value.keys():
                        raise ValueError("Declared phase_qa.json omits its top/floor producer QA")
                    for rule in ("top", "floor"):
                        _phase_rule_timing_values(value[rule], rule)
            compare(before_value, after_value, name, phase_qa=phase_qa)
            scope = "exact JSON fields after verified descriptors and executing-thread identities"
            if phase_qa:
                scope += "; declared top/floor execution timings retained separately"
        else:
            _same([before[field] for field in ("sha256", "size_bytes")],
                  [after[field] for field in ("sha256", "size_bytes")], name + " complete file bytes")
            scope = "exact complete file SHA256 and size"
        visiting.remove(name)
        proven[key] = deepcopy(before)
        files.append(dict(name=name, original=deepcopy(before), replayed=deepcopy(after), comparison=scope))

    for name in initial:
        verify(name)
    return proven, dict(files=files, input_relocations=input_paths, opencv_observations=observations,
        phase_rule_timing_observations=phase_timings,
        differing_runtime_fields=["pid", "thread_id", "native_thread_id"] + [
            row["where"] + "." + field for row in phase_timings for field in _PHASE_TIMINGS],
        original_metadata_retained=True, rewritten_metadata_supplied_to_science=False)


def _payload_roles(record, reader):
    """Known original producer fields, never basename or directory inference."""
    full = record["native"]["full"]
    result = {"current.centre.log": full["native_centre"]["log_linear_rgb"]}
    if "valid_mask" in full["native_centre"]:
        result["current.centre.valid"] = full["native_centre"]["valid_mask"]
    for index, window in enumerate(full["ring_windows"]):
        result["current.ring." + str(index)] = window["log_linear_rgb"]
    for role in ("labels", "classes", "area", "operator_masks"):
        result[role] = record["phase"][role]
    history = reader.json(record["phase"]["history_input"])
    for role in ("temporal_log", "temporal_valid"):
        result["history." + role] = history[role]
    for role in ("before_rules", "photograph"):
        if record.get(role, {}).get("state") == "complete":
            result[role + ".png"] = record[role]["png"]
    if "probabilities" in record:
        result["probabilities.archive"] = record["probabilities"]["archive"]
    return result


def _statistics_bytes(reader, bound):
    """Read exact scientific JSON bytes, bounded like the original catalogue."""
    from ..pipeline.job_inputs import MAX_METADATA_BYTES, _json_bytes

    path = reader.path(bound)
    with Path(path).open("rb") as raw:
        if raw.read(2) != b"\x1f\x8b":
            raise ValueError("Original phase-one statistics require their gzip container")
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw) as stream:
            decoded = stream.read(MAX_METADATA_BYTES + 1)
    if len(decoded) > MAX_METADATA_BYTES:
        raise ValueError("Replayed statistics exceed the original metadata limit")
    _json_bytes(decoded)
    reader.path(bound)
    return decoded


def _configuration(rows, reader, expected):
    from ..pipeline.store_runtime import ResolvedInput

    if not isinstance(rows, (list, tuple)) or any(type(row) is not ResolvedInput for row in rows):
        raise TypeError("Original resolved phase-two input rows are required")
    sources = [row for row in rows if "run.configuration" in row.uses]
    if len(sources) != 1:
        raise ValueError("Exactly one original phase-two configuration consumer is required")
    row = sources[0]
    bound = dict(path=str(row.path), sha256=row.reference["sha256"], size_bytes=row.reference["bytes"])
    _same(reader.json(bound), expected, "the actual sealed phase-two configuration bytes")
    return bound


def _replay_owned(reader, completion_bound, configuration_bound, expected):
    from ..pipeline.run_io import RunIOBinding

    binding = reader.io_binding
    if (type(binding) is not RunIOBinding or binding._replayed is not True
            or binding.task.get("pipeline") != "native-phase-statistics"
            or binding.task.get("run") != expected["run"]
            or binding.task.get("precision") != expected["precision"]
            or binding.task.get("commit") != expected["phase1"]["code_commit"]
            or binding.task.get("ordered_case_ids") != [frame["case_id"] for frame in expected["frames"]]):
        raise ValueError("A completed original phase-one replay needs its actual admitted RunIOBinding")
    path = Path(reader.path(completion_bound)).resolve()
    binding.require_output(path.parent)
    if path.name != "phase1_completed.json":
        raise ValueError("The replay must use its original owned phase-one completion")
    _same(configuration_bound, binding.configuration_descriptor, "replayed configuration admission")
    return path.parent


def _owned_record(reader, root, item):
    for bound in (item["commit"], *item["record"]["files"].values()):
        path = Path(reader.path(bound)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Replayed phase-one artifact escapes its actual owned output")


def verify_coupling(original_config, original_completed_binding, replay_completion_binding,
                    original_reader, replay_reader, phase2_resolved_rows):
    """Return unchanged-reference rows with proven fresh payload locations and a receipt.

    ``original_config`` is authenticated against the unique ``run.configuration``
    row. ``replay_reader`` must have been constructed inside the genuine original
    P1 RunIOBinding. The caller must still pass the returned rows through unchanged
    ``verify_inputs`` and ``RunIOBinding.replay`` before executing original N1.
    This receipt is evidence of coupling, never production/qualification admission.
    """
    from ..mass.n1_pipeline import DataReader
    from ..pipeline import staged_data as staged
    from ..pipeline.series import bind_mass_frame
    from ..pipeline.store_runtime import ResolvedInput

    if type(original_reader) is not DataReader or type(replay_reader) is not DataReader:
        raise TypeError("Both original content-verifying DataReaders are required")
    phase2_bound = _configuration(phase2_resolved_rows, original_reader, original_config)
    # This helper executes inside the original P1 interpreter, whose P2 schema
    # may predate the producing P2 commit. Authenticate the complete sealed
    # bytes here; never strip/upgrade them or run the older P2 schema against
    # them. The actual P2 interpreter must still execute its original complete
    # RunIOBinding.replay consumer walk before any N1 scientific call.
    source = original_config["phase1"]
    _same(original_completed_binding, source["completion"], "original phase-one completion binding")
    original = staged.verify_phase1_completion(original_completed_binding, source["configuration"],
        expected_commit=source["code_commit"], reader=original_reader)
    replay_core = replay_reader.json(replay_completion_binding)
    replay_configuration = replay_core.get("configuration")
    output = _replay_owned(replay_reader, replay_completion_binding, replay_configuration, original_config)
    replayed = staged.verify_phase1_completion(replay_completion_binding, replay_configuration,
        expected_commit=source["code_commit"], reader=replay_reader)
    _same(original["configuration"], replayed["configuration"], "phase-one input DATA")
    if any(source["configuration"][key] != replay_configuration[key] for key in ("sha256", "size_bytes")):
        raise ValueError("Replayed phase-one configuration has different original bytes")
    order = [frame["case_id"] for frame in original_config["frames"]]
    _same(order, original["configuration"]["ordered_case_ids"], "selected phase-one case order")
    _same(order, replayed["completion"]["ordered_case_ids"], "replayed selected case order")
    for key, expected in original["configuration"]["statistics"].items():
        if key == "format":
            continue
        actual = original_config.get(key)
        if key == "provenance" and isinstance(actual, dict) and "common_frozen_refit" in actual:
            actual = {field: value for field, value in actual.items() if field != "common_frozen_refit"}
        _same(actual, expected, "phase-two statistics template " + key)

    replacements, by_original_path, statistics, expected_uses = {}, {}, [], {}
    case_receipts = []
    phase1 = original["configuration"]
    for selected, initial, old_item, new_item in zip(original_config["frames"], phase1["frames"],
                                                    original["records"], replayed["records"]):
        case = selected["case_id"]
        old, new = old_item["record"], new_item["record"]
        _owned_record(replay_reader, output, new_item)
        refs = source["frames"][case]
        _same(refs["commit"], old_item["commit"], "sealed phase-one case commit")
        for field in ("statistics", "before_rules", "photograph"):
            _same(refs.get(field), old.get(field), "sealed case evidence " + field)
        old_roles, new_roles = _payload_roles(old, original_reader), _payload_roles(new, replay_reader)
        _same(list(old_roles), list(new_roles), "generated payload roles")
        payloads = []
        for role, old_bound in old_roles.items():
            new_bound = new_roles[role]
            if any(old_bound[key] != new_bound[key] for key in ("sha256", "size_bytes")):
                raise ValueError("Regenerated scientific payload bytes differ: " + case + "." + role)
            old_path = Path(original_reader.path(old_bound)).resolve()
            new_path = Path(replay_reader.path(new_bound)).resolve()
            if old_path == new_path or not new_path.is_relative_to(output):
                raise ValueError("Fresh phase-one scientific payload must belong to the new owned attempt")
            key = _identity(new_bound)
            if key in replacements and replacements[key] != old_bound:
                raise ValueError("Ambiguous descriptor relocation between original phase-one roles")
            replacements[key] = deepcopy(old_bound)
            pair = dict(case_id=case, role=role, original=deepcopy(old_bound), replayed=deepcopy(new_bound))
            prior = by_original_path.get(old_path)
            if prior is not None and prior[0] != new_path:
                raise ValueError("One original input maps to conflicting fresh scientific payloads")
            by_original_path[old_path] = (new_path, pair)
            payloads.append(pair)
            if role.startswith("current.") or role in {"labels", "classes", "area"}:
                expected_uses["frame." + case + ".mass." + role] = old_path
            elif role == "before_rules.png":
                expected_uses["frame." + case + ".mass.phase1.before_png"] = old_path
            elif role == "photograph.png":
                expected_uses["frame." + case + ".mass.phase1.photo_png"] = old_path
        old_stats = _statistics_bytes(original_reader, old["statistics"])
        new_stats = _statistics_bytes(replay_reader, new["statistics"])
        if old_stats != new_stats:
            raise ValueError("Regenerated phase-one statistics differ byte for byte: " + case)
        statistic = dict(case_id=case, original=deepcopy(old["statistics"]),
            replayed=deepcopy(new["statistics"]), comparison="exact decompressed JSON bytes",
            decoded_sha256=hashlib.sha256(new_stats).hexdigest(), decoded_size_bytes=len(new_stats),
            consumed_container="original sealed gzip; its original SHA admission remains required")
        statistics.append(statistic)
        metadata_replacements, metadata_evidence = _frame_metadata(old, new, original_reader, replay_reader, statistic)
        expected_frame = bind_mass_frame(initial, old["native"], old["phase"], baseline=phase1["phase"]["reference"]["full"])
        actual_frame = bind_mass_frame(initial, new["native"], new["phase"], baseline=phase1["phase"]["reference"]["full"])
        selected_core = {key: value for key, value in selected.items()
                         if key not in {"pixel_domain_geometry", "frozen_response_basis"}}
        _same(selected_core, expected_frame, "original sealed N1 frame " + case)
        _same(_metadata_relocated(actual_frame, metadata_replacements), expected_frame,
              "regenerated N1 frame geometry/source " + case)
        old_history = original_reader.json(old["phase"]["history_input"])
        new_history = replay_reader.json(new["phase"]["history_input"])
        _same(_metadata_relocated(new_history, metadata_replacements), old_history,
              "regenerated acquisition history " + case)
        if "probabilities" in old:
            old_probabilities = original_reader.json(old["probabilities"]["receipt"])
            new_probabilities = replay_reader.json(new["probabilities"]["receipt"])
            _same(_relocated(new_probabilities, replacements), old_probabilities,
                  "regenerated decision probability evidence " + case)
        for field in ("before_rules", "photograph"):
            _same(old.get(field, {}).get("state"), new.get(field, {}).get("state"), field + " state")
            if field in old:
                before = original_reader.json(old[field]["receipt"])
                after = replay_reader.json(new[field]["receipt"])
                _same(_relocated(after, replacements), before, field + " scientific evidence")
        case_receipts.append(dict(case_id=case, original_commit=deepcopy(old_item["commit"]),
                                 replayed_commit=deepcopy(new_item["commit"]), payloads=payloads,
                                 metadata=metadata_evidence))

    seen_uses, substituted, rows = {}, [], []
    for row in phase2_resolved_rows:
        old_path = Path(row.path).resolve()
        for use in row.uses:
            if use in seen_uses:
                raise ValueError("Repeated original phase-two consumer: " + use)
            seen_uses[use] = old_path
        pair = by_original_path.get(old_path)
        if pair is None:
            rows.append(row)
            continue
        new_path, evidence = pair
        bound = evidence["original"]
        if row.reference["sha256"] != bound["sha256"] or row.reference["bytes"] != bound["size_bytes"]:
            raise ValueError("Consumed original catalogue identity differs from phase-one payload")
        rows.append(ResolvedInput(deepcopy(row.reference), new_path, tuple(row.uses)))
        substituted.append(dict(reference=deepcopy(row.reference), uses=list(row.uses), **deepcopy(evidence)))
    if any(seen_uses.get(use) != path for use, path in expected_uses.items()):
        raise ValueError("Phase-two catalogue omits or changes an original scientific payload consumer")
    receipt = dict(format=FORMAT, complete=True, run=original_config["run"],
        precision=original_config["precision"], ordered_case_ids=order,
        phase2_configuration=phase2_bound, original_phase1_completion=deepcopy(original_completed_binding),
        replayed_phase1_completion=deepcopy(replay_completion_binding),
        original_phase1_configuration=deepcopy(source["configuration"]),
        replayed_phase1_configuration=deepcopy(replay_configuration),
        phase1_code_commit=source["code_commit"], cases=case_receipts, substitutions=substituted,
        statistics=statistics, original_metadata_retained=True,
        phase2_runtime_verification_required=True, scientific_qualification=False, production=False)
    return tuple(rows), receipt
