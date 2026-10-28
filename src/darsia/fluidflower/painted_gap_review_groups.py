"""Compact, descriptive deviation groups from already verified replay pixels.

Groups reduce repeated drawing; every destination retains its original human
confirmation and typed validators. This DATA cannot resolve a finding.
"""
from copy import deepcopy

import numpy as np


FORMAT = "darsia-painted-deviation-groups-data-v1"
_IDENTITY = ("run_id", "component_id", "review_row_sha256")


def _key(value):
    return tuple(value[name] for name in _IDENTITY)


def _item(row, component):
    return dict(run_id=row["run_id"], component_id=component["component_id"],
                review_row_sha256=component["review_row_sha256"])


def deviation_groups_from_replay(replay):
    """Use only this replay's masks and exact review, with no second read.

    Largest components are considered first to avoid choosing a short strip as
    representative of a longer one. Ties keep literal review/component order.
    A destination joins only when its full original component is supported.
    """
    from .painted_analysis_exclusions import _identity, _same
    from .painted_gap_region_transfer import component_transfer_coverage

    review = replay["review"]
    native = {row["run_id"]: row for row in replay["runs"]}
    if len(native) != len(replay["runs"]) or set(native) != {row["run_id"] for row in review["runs"]}:
        raise ValueError("Deviation grouping needs this exact replayed cohort")
    entries = []
    for row in review["runs"]:
        verified = native[row["run_id"]]
        actual = {component["component_id"]: (component, mask)
                  for component, mask in verified["components"]}
        if (len(actual) != len(verified["components"])
                or set(actual) != {component["component_id"] for component in row["components"]}):
            raise ValueError("Deviation grouping needs the exact current residual components")
        for component in row["components"]:
            original, mask = actual[component["component_id"]]
            if (not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 2 or not mask.any()
                    or list(mask.shape) != row["gap"]["shape_hw"]
                    or not _same(_identity(mask), component["identity"])
                    or not _same(original, {key: value for key, value in component.items()
                                            if key != "review_row_sha256"})):
                raise ValueError("A descriptive group cannot replace an original component binding")
            entries.append(dict(item=_item(row, component), mask=mask,
                                placement=row["native_to_painting_h"], pixels=component["pixels"]))
    remaining = sorted(range(len(entries)), key=lambda index: (-entries[index]["pixels"], index))
    groups, failed_pairs = [], 0
    while remaining:
        representative, *candidates = remaining
        source = entries[representative]
        members, unmatched = [representative], []
        for candidate in candidates:
            target = entries[candidate]
            try:
                coverage = component_transfer_coverage(source["mask"], source["placement"],
                    target["mask"], target["placement"])
            except (ValueError, np.linalg.LinAlgError):
                failed_pairs += 1
                unmatched.append(candidate)
                continue
            if coverage["remaining_pixels"] == 0 and coverage["matched_pixels"] == target["pixels"]:
                members.append(candidate)
            else:
                unmatched.append(candidate)
        groups.append(dict(representative=deepcopy(source["item"]),
            members=[deepcopy(entries[index]["item"]) for index in members]))
        remaining = unmatched
    result = dict(format=FORMAT, review=deepcopy(review["review_binding"]), groups=groups,
        summary=dict(deviation_count=len(entries), classification_groups=len(groups),
                     transferred_confirmations=len(entries)-len(groups), failed_pairs=failed_pairs))
    return validate_deviation_groups(result, review)


def validate_deviation_groups(value, review):
    """Check compact worker DATA against every literal current review row.

    This does not authenticate geometry or grant approval. Physical matching
    belongs to the already verified worker; the GUI keeps all original rows.
    """
    from .painted_analysis_exclusions import _same

    expected = {_key(_item(row, component))
                for row in review["runs"] for component in row["components"]}
    if (not isinstance(value, dict) or set(value) != {"format", "review", "groups", "summary"}
            or value["format"] != FORMAT or not _same(value["review"], review["review_binding"])
            or not isinstance(value["groups"], list)):
        raise ValueError("The deviation groups belong to another reviewed source")
    seen = set()
    for group in value["groups"]:
        if (not isinstance(group, dict) or set(group) != {"representative", "members"}
                or not isinstance(group["members"], list) or not group["members"]
                or group["representative"] != group["members"][0]):
            raise ValueError("A deviation group needs its literal representative first")
        for member in group["members"]:
            if not isinstance(member, dict) or set(member) != set(_IDENTITY):
                raise ValueError("A deviation group needs exact original row identities")
            key = _key(member)
            if key not in expected or key in seen:
                raise ValueError("A deviation cannot be omitted, repeated or rebound by grouping")
            seen.add(key)
    summary = value["summary"]
    if (seen != expected or not isinstance(summary, dict)
            or set(summary) != {"deviation_count", "classification_groups", "transferred_confirmations", "failed_pairs"}
            or any(type(number) is not int or number < 0 for number in summary.values())
            or summary["deviation_count"] != len(expected)
            or summary["classification_groups"] != len(value["groups"])
            or summary["transferred_confirmations"] != len(expected)-len(value["groups"])):
        raise ValueError("The descriptive counts must cover every original deviation exactly once")
    return deepcopy(value)
