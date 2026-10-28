"""Optional lossless copies of the probabilities used by frozen decisions.

This module is an output side channel. It never predicts, resizes, reconstructs
scores from labels, or supplies values to a scientific decision.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

REQUEST_FORMAT = "darsia-decision-probability-storage-request-v1"
FORMAT = "darsia-decision-probabilities-v3"
ENCODING = "float32-exact-decision-values"
ARRAYS = ("pixel_probability", "regional_probability", "branch_support", "later_rule_changes")
CODES = {"unscored": 0, "centre": 1, "edge": 2, "placement": 3}

RULE_NAMES = ("centre_minimum", "edge_minimum", "centre_B", "edge_B",
              "seam_minimum", "seam_B", "placement_minimum", "placement_top",
              "placement_B", "post_top", "post_floor", "post_contact")
RULE_BITS = {name: 1 << index for index, name in enumerate(RULE_NAMES)}
RULE_BITS_ALL = sum(RULE_BITS.values())
RULE_BRANCHES = {name: ((1,) if name.startswith("centre_") else
                        (2,) if name.startswith("edge_") else
                        (1, 2) if name.startswith("seam_") else
                        (3,) if name.startswith("placement_") else (1, 2, 3))
                 for name in RULE_NAMES}


def changed_rule(capture, name, before, after):
    """Retain only the actual changed pixels of a named eligible later rule."""
    if name not in RULE_BITS or name in capture.get("rules", {}):
        raise ValueError("Unknown or duplicate actual later-rule observation")
    a, b = np.asarray(before), np.asarray(after)
    if (a.ndim != 2 or a.shape != b.shape or a.dtype != b.dtype
            or a.dtype not in (np.dtype(bool), np.dtype(np.uint8))):
        raise ValueError("Actual same-grid categorical/bool rule operands required")
    changed = a != b
    capture.setdefault("rules", {})[name] = dict(
        mask=changed, before=identity(a), after=identity(b), changed=identity(changed),
        raw_changed_pixels=int(np.count_nonzero(changed)))


def capture_seam_population(capture, values):
    """Copy the literal original commit and score-origin population, no feedback."""
    capture["seam_commit"] = values["commit"].copy()
    capture["seam_inherited"] = (values["domain"] & ~values["ring"]).copy()


def parse_request(config):
    if "gas_probability_storage" not in config:
        return None
    request = config["gas_probability_storage"]
    if (type(request) is not dict or set(request) != {"format", "encoding"}
            or request != dict(format=REQUEST_FORMAT, encoding=ENCODING)):
        raise ValueError("Explicit exact-float32 decision probability request required")
    return deepcopy(request)


def same_metadata(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, allow_nan=False, separators=(",", ":"))


def identity(array):
    value = np.asarray(array)
    digest = hashlib.sha256()
    for chunk in np.nditer(value, flags=["external_loop", "buffered", "zerosize_ok"],
                          op_flags=["readonly"], order="C", buffersize=65536):
        digest.update(chunk.tobytes(order="C"))
    return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=digest.hexdigest())


def transition(capture, name, before, after):
    """Count actual ordered transitions, including overlaps across operators."""
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape:
        raise ValueError("Decision transition grids differ")
    transitions = {}
    for old in np.unique(a):
        for new in np.unique(b):
            count = int(np.count_nonzero((a == old) & (b == new)))
            if count:
                transitions[str(int(old)) + "->" + str(int(new))] = count
    capture.setdefault("transitions", []).append(dict(
        operator=name, before=identity(a), after=identity(b),
        changed_pixels=int(np.count_nonzero(a != b)), transitions=transitions))


def pair(capture, pixel, regional, support, co2, selected, *, branch,
         pixel_threshold, regional_threshold):
    """Copy the actual native arrays immediately after their original use."""
    p, r = np.asarray(pixel), np.asarray(regional)
    valid, union, chosen = (np.asarray(v) for v in (support, co2, selected))
    if (p.dtype != np.float32 or r.dtype != np.float32 or p.ndim != 2
            or r.shape != p.shape or any(v.dtype != bool or v.shape != p.shape
                                      for v in (valid, union, chosen))):
        raise ValueError("Decision-used native float32 pair and bool supports required")
    valid = valid & np.isfinite(p) & np.isfinite(r)
    reproduced = union & valid & (p >= pixel_threshold) & (r >= regional_threshold)
    if not np.array_equal(reproduced, chosen):
        raise ValueError("Captured pair does not reproduce the actual pre-minimum decision")
    capture.update(pixel=p.copy(), regional=r.copy(), support=valid.copy(), branch=branch,
                   decision=dict(pixel=identity(p), regional=identity(r), support=identity(valid),
                                 frozen_co2=identity(union), selected=identity(chosen),
                                 selected_pixels=int(chosen.sum()),
                                 pixel_threshold=float(pixel_threshold),
                                 regional_threshold=float(regional_threshold),
                                 reproduced_pre_minimum_exact=True))


def capture_centre(capture, values):
    if values["source_shape"] is None:
        raise ValueError("Probability pair storage requires an actually used regional head")
    g3, scores = values["g3"], values["scores"]
    settings = values["settings"]
    pair(capture, g3, scores, np.isfinite(g3) & np.isfinite(scores), values["co2"],
         values["regional_gas"], branch="centre", pixel_threshold=settings.pixel_threshold,
         regional_threshold=settings.regional_threshold)
    transition(capture, "centre-minimum", values["regional_gas"], values["gas"])
    changed_rule(capture, "centre_minimum", values["regional_gas"], values["gas"])


def capture_centre_selection(capture, values):
    pixel, native, parameters = values["pixel"], values["native"], values["parameters"]
    pt, rt = (parameters["centre_selection"][key] for key in ("pixel_threshold", "regional_threshold"))
    pair(capture, pixel, native, np.isfinite(pixel) & np.isfinite(native), np.ones(pixel.shape, bool),
         (pixel >= pt) & (native >= rt), branch="centre", pixel_threshold=pt, regional_threshold=rt)


def require_pair_mode(placement, parameters):
    if placement == parameters["placement"]["placement_modes"]["constrained_neighbourhood"]:
        raise ValueError("P2 has no decision-used regional probability pair; optional pair storage is unavailable")


def capture_placement(capture, values):
    support, masks, params = values["support"], values["threshold_masks"], values["parameters"]
    pair(capture, values["pixel"], values["regional_score"], support,
         np.isin(values["labels"], (1, 2)), masks["pre_minimum_gas"] & support,
         branch="placement", pixel_threshold=params["literal_values"]["replace_gas_from_scores_float_0_7"],
         regional_threshold=params["literal_values"]["replace_gas_from_scores_float_0_6"])
    capture["placement_mode"] = values["placement"]
    capture["light_epoch_id"] = values["context"].light_epoch_id
    transition(capture, "placement-minimum-including-declared-fill",
               masks["pre_minimum_gas"], masks["post_minimum_gas"])
    changed_rule(capture, "placement_minimum", masks["pre_minimum_gas"], masks["post_minimum_gas"])
    transition(capture, "placement-threshold-and-minimum", values["labels"], values["result"])
    state(capture, "placement-minimum-labels", values["result"])


def state(capture, name, labels, *, eligible_rule=None):
    value = np.asarray(labels)
    if "last_labels" in capture:
        transition(capture, name, capture["last_labels"], value)
        if eligible_rule is not None:
            changed_rule(capture, eligible_rule, capture["last_labels"], value)
    elif eligible_rule is not None:
        raise ValueError("Eligible rule requires its actual preceding label state")
    capture["last_labels"] = value.copy()


def spatial_proof(arrays, labels, thresholds):
    """Independently replay thresholds and spatial memberships on saved arrays."""
    validate_arrays(arrays, labels=labels)
    codes, bits = arrays["branch_support"], arrays["later_rule_changes"]
    if type(thresholds) is not dict or set(thresholds) != {"1", "2", "3"}:
        raise ValueError("Exact per-branch decision thresholds required")
    selected = np.zeros(codes.shape, bool)
    for code in (1, 2, 3):
        threshold = thresholds[str(code)]
        if threshold is None:
            if np.any(codes == code):
                raise ValueError("Scored branch has no actual threshold provenance")
            continue
        if (type(threshold) is not dict or set(threshold) != {"pixel", "regional"}
                or any(type(v) is not float or not np.isfinite(v) or not 0 <= v <= 1
                       for v in threshold.values())):
            raise ValueError("Actual finite literal threshold pair required")
        selected |= ((codes == code) & (arrays["pixel_probability"] >= threshold["pixel"])
                     & (arrays["regional_probability"] >= threshold["regional"]))
    inverse = (codes != 0) & (np.asarray(labels) == 2) & ~selected
    if inverse.any():
        raise ValueError("Later removal rules cannot explain below-threshold final gas: " + str(int(inverse.sum())))
    disagreement = (codes != 0) & (selected != (np.asarray(labels) == 2))
    memberships = np.zeros(codes.shape, np.uint8)
    rows = {}
    for name, bit in RULE_BITS.items():
        mask = (bits & bit) != 0
        if np.any(mask & ~np.isin(codes, RULE_BRANCHES[name])):
            raise ValueError("Later-rule proof borrows an inapplicable decision branch")
        memberships += mask
        rows[name] = dict(applicable_pixels=int(mask.sum()), applicable_mask=identity(mask),
                          disagreement_pixels=int(np.count_nonzero(mask & disagreement)),
                          disagreement_intersection=identity(mask & disagreement))
    unexplained = disagreement & (bits == 0)
    if unexplained.any():
        raise ValueError("Stored-pair final decision has unexplained spatial disagreement: "
                         + str(int(unexplained.sum())))
    return dict(format="darsia-decision-spatial-membership-v3", rules=rows,
                threshold_disagreement_pixels=int(disagreement.sum()),
                disagreement_mask=identity(disagreement),
                applicable_union_pixels=int(np.count_nonzero(memberships)),
                overlapping_pixels=int(np.count_nonzero(memberships > 1)),
                total_memberships=int(memberships.sum()),
                membership_histogram={str(n): int(np.count_nonzero(memberships == n))
                                      for n in range(len(RULE_NAMES) + 1)},
                below_threshold_final_gas_pixels=0, unexplained_pixels=0)


def finish(capture, labels, classes, source_box_xywh):
    """Compose the exact population used by each decision, then prove Gate 2."""
    labels = np.asarray(labels)
    shape = labels.shape
    p, r = (np.full(shape, np.nan, np.float32) for _ in range(2))
    codes = np.zeros(shape, np.uint8)
    target = None
    thresholds = {str(code): None for code in (1, 2, 3)}

    def assign(part, mask, code, destination=None, *, clear=False):
        if not part or "pixel" not in part:
            raise ValueError("Actual selected decision pair is missing")
        dest = (slice(None), slice(None)) if destination is None else destination
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != part["support"].shape:
            raise ValueError("Actual pair population/support grids differ")
        if clear:
            p[dest][mask], r[dest][mask], codes[dest][mask] = np.nan, np.nan, 0
        keep = mask & part["support"]
        p[dest][keep], r[dest][keep], codes[dest][keep] = part["pixel"][keep], part["regional"][keep], code
        decision = part["decision"]
        threshold = dict(pixel=decision["pixel_threshold"], regional=decision["regional_threshold"])
        previous = thresholds[str(code)]
        if previous is not None and previous != threshold:
            raise ValueError("One stored branch cannot conflate different actual threshold pairs")
        thresholds[str(code)] = threshold

    placement = capture.get("placement", {})
    if placement.get("unsupported"):
        raise ValueError(placement["unsupported"])
    active_placement = "pixel" in placement
    composition = capture.get("composition", {})
    seam_commit = np.zeros(shape, bool)
    release = np.zeros(shape, bool)
    if active_placement:
        assign(placement, placement["support"], CODES["placement"])
    else:
        edge = capture["edge"]
        assign(edge, edge["support"], CODES["edge"])
        if "central_support" in composition:
            x, y, width, height = source_box_xywh
            target = (slice(y, y + height), slice(x, x + width))
            centre = capture["centre"]
            assign(centre, composition["central_support"], CODES["centre"], target, clear=True)
            release = np.asarray(composition["release"])
            seam_commit = np.asarray(composition["seam_commit"])
            inherited = np.asarray(composition["seam_inherited"])
            if any(v.dtype != bool or v.shape != shape for v in (release, seam_commit, inherited)):
                raise ValueError("Exact full-grid seam populations required")
            assign(edge, release, CODES["edge"], clear=True)
            centre_seam = seam_commit & inherited
            if np.any(centre_seam) and "seam_pair" not in centre:
                raise ValueError("Actual centre seam pair is missing")
            if "seam_pair" in centre:
                assign(centre["seam_pair"], centre_seam[target], CODES["centre"], target, clear=True)
            assign(edge, seam_commit & ~inherited, CODES["edge"], clear=True)
            covered = np.zeros(shape, bool)
            covered[target] = True
            if np.any(centre_seam & ~covered):
                raise ValueError("Centre seam population escapes its original native grid")
    scored = (codes != CODES["unscored"]) & np.asarray(classes["modelled"], bool)
    for name in ("filled_observed", "filled_unobserved"):
        scored &= ~np.asarray(classes[name], bool)
    codes[~scored] = CODES["unscored"]
    p[~scored], r[~scored] = np.nan, np.nan
    before_mask = dict(pixel=identity(p), regional=identity(r))
    outside = scored & ~np.isin(labels, (1, 2))
    p[outside], r[outside] = np.float32(0), np.float32(0)
    bits = np.zeros(shape, np.uint16)
    raw_rules = {}
    for name in RULE_NAMES:
        observed = []
        for part_name in ("edge", "centre", "composition", "placement", "postprocessing"):
            rule = capture.get(part_name, {}).get("rules", {}).get(name)
            if rule is not None:
                observed.append(rule)
        if len(observed) > 1:
            raise ValueError("Duplicate actual rule source across decision branches")
        if not observed:
            raw_rules[name] = dict(observed=False, raw_changed_pixels=0, before=None, after=None, changed=None)
            continue
        rule = observed[0]
        raw_rules[name] = {key: deepcopy(rule[key]) for key in ("raw_changed_pixels", "before", "after", "changed")}
        raw_rules[name]["observed"] = True
        if active_placement and not name.startswith(("placement_", "post_")):
            continue
        raw = rule["mask"]
        if raw.shape != shape:
            if name not in ("centre_minimum", "centre_B") or target is None or raw.shape != codes[target].shape:
                raise ValueError("Actual rule mask cannot be placed on the original decision grid")
            mask = np.zeros(shape, bool)
            mask[target] = raw
        else:
            mask = raw
        applicable = mask & np.isin(codes, RULE_BRANCHES[name])
        if name.startswith(("centre_", "edge_")):
            applicable &= ~seam_commit
            if name == "edge_B":
                applicable &= ~release
        elif name == "seam_minimum":
            applicable &= seam_commit
        bits[applicable] |= RULE_BITS[name]
    arrays = dict(pixel_probability=p, regional_probability=r, branch_support=codes, later_rule_changes=bits)
    proof = spatial_proof(arrays, labels, thresholds)
    decisions, transitions = {}, []
    for name in ("edge", "centre", "composition", "placement", "postprocessing"):
        part = capture.get(name, {})
        if "decision" in part:
            decisions[name] = deepcopy(part["decision"])
        if "encoding" in part:
            decisions[name + "_encoding"] = deepcopy(part["encoding"])
        if "placement_mode" in part:
            decisions[name + "_mode"] = dict(placement=part["placement_mode"], light_epoch_id=part["light_epoch_id"])
        if "seam_pair" in part:
            seam = part["seam_pair"]
            if identity(seam["actual_returned_selection"]) != seam["decision"]["selected"]:
                raise ValueError("Captured seam pair differs from the original returned decision")
            decisions[name + "_seam"] = deepcopy(seam["decision"])
        transitions.extend(deepcopy(part.get("transitions", [])))
    capture["product"] = dict(arrays=arrays, evidence=dict(
        decisions=decisions, ordered_transitions=transitions,
        decision_thresholds=thresholds, raw_later_rules=raw_rules, spatial_proof=proof,
        population_composition={name: identity(composition[name]) for name in
                                ("central_support", "release", "seam_commit", "seam_inherited")
                                if name in composition},
        pre_storage_mask_pair=before_mask, scored_outside_final_co2_zeroed=int(outside.sum()),
        branch_counts={name: int(np.count_nonzero(codes == code)) for name, code in CODES.items()},
        unscored_policy="NaN takes precedence over outside-CO2 zero; no invented score",
        score_feedback=False, score_reconstruction=False, quantization=False))


def validate_arrays(arrays, *, labels=None):
    if type(arrays) is not dict or set(arrays) != set(ARRAYS):
        raise ValueError("Exactly two probabilities, branch/support and actual later-rule proof required")
    p, r, codes, bits = (np.asarray(arrays[name]) for name in ARRAYS)
    if (p.dtype != np.float32 or r.dtype != np.float32 or codes.dtype != np.uint8
            or bits.dtype != np.uint16 or p.ndim != 2 or not p.size
            or r.shape != p.shape or codes.shape != p.shape or bits.shape != p.shape
            or not np.isin(codes, tuple(CODES.values())).all()):
        raise ValueError("Exact float32/uint8/uint16 probability and spatial proof grid required")
    scored = codes != CODES["unscored"]
    if np.any(bits & np.uint16(65535 ^ RULE_BITS_ALL)) or np.any(bits[~scored] != 0):
        raise ValueError("Unknown later-rule bits or rule proof on unscored pixels")
    if (not np.isnan(p[~scored]).all() or not np.isnan(r[~scored]).all()
            or not np.isfinite(p[scored]).all() or not np.isfinite(r[scored]).all()
            or np.any((p[scored] < 0) | (p[scored] > 1))
            or np.any((r[scored] < 0) | (r[scored] > 1))):
        raise ValueError("Probability values and explicit unscored support disagree")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != p.shape:
            raise ValueError("Probability labels/grid differ")
        outside = scored & ~np.isin(labels, (1, 2))
        if np.any(p[outside] != 0) or np.any(r[outside] != 0):
            raise ValueError("Scored probabilities outside final CO2 must be exactly zero")
    return arrays


def _head_bindings(models):
    if models.get("family") == "fresh-native-logistic":
        selected = models["selected"]
        result = dict(selected=dict(selected["heads"],
            decision_policy=selected["decision_policy"], image_basis=selected["image_basis"]))
        validate_heads(result)
        return result
    result = {name: deepcopy(models[name]["documents"]) for name in ("native", "edge")}
    placement = models["placement"]
    result["placement"] = {name: deepcopy(placement[name])
                           for name in ("pixel_model", "regional_model")}
    if placement.get("projected_models") is not None:
        result["placement"]["projected_models"] = deepcopy(placement["projected_models"])
    validate_heads(result)
    return result


def validate_heads(heads):
    if type(heads) is not dict or set(heads) not in ({"native", "edge", "placement"}, {"selected"}):
        raise ValueError("Original centre, edge and placement head bindings required")
    def visit(value):
        if type(value) is not dict or not value:
            raise ValueError("Exact original head descriptors required")
        if "path" in value:
            if (not isinstance(value["path"], str) or not value["path"]
                    or re.fullmatch("[0-9a-f]{64}", str(value.get("sha256"))) is None):
                raise ValueError("Head source path/SHA required; no redumped model identity")
        else:
            for child in value.values():
                visit(child)
    visit(heads)


def capture_selected(capture, co2, conditional_gas, support, *, thresholds, labels, classes, source_box):
    """Store the exact CO2/conditional-gas pair used by the selected fresh heads.

    The two existing probability columns retain an explicit named role in the
    evidence. No regional model, score reconstruction or old decision is used.
    """
    from .domain import embed_native_canvas
    if capture:
        raise ValueError("A fresh probability capture is required")
    shape = labels.shape
    scored = embed_native_canvas(np.asarray(support, bool), source_box_xywh=source_box,
        phase_shape_hw=shape, fill=False) & np.asarray(classes["modelled"], bool)
    p = embed_native_canvas(np.asarray(co2, np.float32), source_box_xywh=source_box,
        phase_shape_hw=shape, fill=np.nan)
    r = embed_native_canvas(np.asarray(conditional_gas, np.float32), source_box_xywh=source_box,
        phase_shape_hw=shape, fill=np.nan)
    p[~scored], r[~scored] = np.nan, np.nan
    before = dict(pixel=identity(p), regional=identity(r))
    outside = scored & ~np.isin(labels, (1, 2))
    p[outside], r[outside] = 0, 0
    codes = np.where(scored, CODES["centre"], 0).astype(np.uint8)
    arrays = dict(pixel_probability=p, regional_probability=r, branch_support=codes,
        later_rule_changes=np.zeros(shape, np.uint16))
    literal = {"1": dict(pixel=float(thresholds[0]), regional=float(thresholds[1])), "2": None, "3": None}
    proof = spatial_proof(arrays, labels, literal)
    raw = {name: dict(observed=False, raw_changed_pixels=0, before=None, after=None, changed=None)
           for name in RULE_NAMES}
    capture["product"] = dict(arrays=arrays, evidence=dict(
        decisions=dict(selected=dict(family="fresh-native-logistic", pixel_role="co2_presence",
            regional_role="gas_within_co2", legacy_models_consumed=False)),
        ordered_transitions=[], pre_storage_mask_pair=before,
        scored_outside_final_co2_zeroed=int(outside.sum()),
        branch_counts={name: int(np.count_nonzero(codes == code)) for name, code in CODES.items()},
        unscored_policy="NaN takes precedence over outside-CO2 zero; no invented score",
        score_feedback=False, score_reconstruction=False, quantization=False,
        decision_thresholds=literal, raw_later_rules=raw, spatial_proof=proof,
        population_composition=dict(native_support=identity(scored))))


def write_product(stage, spec, capture, *, model_bindings, phase_configuration, labels, label_descriptor):
    from ..mass.n1_pipeline import _write_json, _write_npz
    product = capture["product"]
    arrays = validate_arrays(product["arrays"], labels=labels)
    if not same_metadata(spatial_proof(arrays, labels, product["evidence"]["decision_thresholds"]),
                         product["evidence"]["spatial_proof"]):
        raise ValueError("Probability arrays changed after actual spatial proof")
    archive = _write_npz(Path(stage) / "phase_probabilities.npz", arrays)
    logical = sum(value.nbytes for value in arrays.values())
    retained = {}
    def count_arrays(value):
        if isinstance(value, np.ndarray):
            retained[id(value)] = value.nbytes
        elif isinstance(value, dict):
            for child in value.values():
                count_arrays(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                count_arrays(child)
    count_arrays(capture)
    receipt = dict(format=FORMAT, complete=True, source=deepcopy(spec["source"]),
        case_id=spec["case_id"], encoding=ENCODING, branch_codes=CODES, later_rule_bits=RULE_BITS,
        phase_configuration=deepcopy(phase_configuration), head_bindings=_head_bindings(model_bindings),
        labels=deepcopy(label_descriptor), final_labels=identity(labels),
        decision_sources={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in (("infer.py", "rules.py", "domain.py", "probability_storage.py")
                              + (("selected_model.py",) if model_bindings.get("family") == "fresh-native-logistic" else ()))},
        arrays={name: identity(value) for name, value in arrays.items()}, archive=archive,
        evidence=deepcopy(product["evidence"]), logical_bytes=logical,
        captured_retained_array_bytes=sum(retained.values()),
        memory_accounting_scope="unique retained capture arrays only; not RSS, transient peak, or admission",
        compressed_bytes=archive["size_bytes"], projected_frames=3090,
        projected_3090_compressed_bytes=archive["size_bytes"] * 3090,
        projection_basis="this actual frame only; not an admitted job budget or measured maximum",
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    saved = _write_json(Path(stage) / "phase_probabilities_receipt.json", receipt)
    return dict(archive=archive, receipt=saved)


def validate_receipt(receipt, roles, *, source, case_id, shape=None):
    required = {"format", "complete", "source", "case_id", "encoding", "branch_codes",
                "phase_configuration", "head_bindings", "labels", "final_labels", "decision_sources", "later_rule_bits",
                "arrays", "archive", "evidence",
                "logical_bytes", "captured_retained_array_bytes", "memory_accounting_scope",
                "compressed_bytes", "projected_frames",
                "projected_3090_compressed_bytes", "projection_basis", "producer_sha256"}
    if (type(receipt) is not dict or set(receipt) != required or receipt["format"] != FORMAT
            or receipt["complete"] is not True or not same_metadata(receipt["source"], source)
            or receipt["case_id"] != case_id or receipt["encoding"] != ENCODING
            or not same_metadata(receipt["branch_codes"], CODES)
            or not same_metadata(receipt["later_rule_bits"], RULE_BITS)
            or not same_metadata(receipt["archive"], roles["archive"])
            or set(receipt["arrays"]) != set(ARRAYS)):
        raise ValueError("Probability receipt/source/archive contract differs")
    validate_heads(receipt["head_bindings"])
    for descriptor in (receipt["labels"], receipt["phase_configuration"]):
        if (type(descriptor) is not dict or not isinstance(descriptor.get("path"), str)
                or re.fullmatch("[0-9a-f]{64}", str(descriptor.get("sha256"))) is None):
            raise ValueError("Original phase configuration and final label descriptors required")
    expected_sources = {"infer.py", "rules.py", "domain.py", "probability_storage.py"}
    if set(receipt["head_bindings"]) == {"selected"}:
        expected_sources.add("selected_model.py")
    if (type(receipt["decision_sources"]) is not dict
            or set(receipt["decision_sources"]) != expected_sources
            or any(re.fullmatch("[0-9a-f]{64}", str(v)) is None for v in receipt["decision_sources"].values())):
        raise ValueError("Actual decision/capture source identities required")
    for name, row in receipt["arrays"].items():
        if (type(row) is not dict or set(row) != {"dtype", "shape", "sha256"}
                or row["dtype"] != ("|u1" if name == "branch_support" else "<u2" if name == "later_rule_changes" else "<f4")
                or len(row["shape"]) != 2 or any(type(v) is not int or v <= 0 for v in row["shape"])
                or re.fullmatch("[0-9a-f]{64}", str(row["sha256"])) is None
                or shape is not None and row["shape"] != list(shape)):
            raise ValueError("Probability array dtype/grid/identity differs")
    dimensions = receipt["arrays"][ARRAYS[0]]["shape"]
    if (any(row["shape"] != dimensions for row in receipt["arrays"].values())
            or type(receipt["logical_bytes"]) is not int
            or receipt["logical_bytes"] != dimensions[0] * dimensions[1] * 11
            or type(receipt["captured_retained_array_bytes"]) is not int
            or receipt["captured_retained_array_bytes"] < receipt["logical_bytes"]
            or receipt["memory_accounting_scope"] != "unique retained capture arrays only; not RSS, transient peak, or admission"
            or type(receipt["compressed_bytes"]) is not int or receipt["compressed_bytes"] <= 0
            or receipt["compressed_bytes"] != roles["archive"]["size_bytes"]
            or type(receipt["projected_frames"]) is not int or receipt["projected_frames"] != 3090
            or receipt["projection_basis"] != "this actual frame only; not an admitted job budget or measured maximum"
            or receipt["projected_3090_compressed_bytes"] != receipt["compressed_bytes"] * 3090
            or re.fullmatch("[0-9a-f]{64}", str(receipt["producer_sha256"])) is None):
        raise ValueError("Probability byte accounting differs")
    final = receipt["final_labels"]
    if (type(final) is not dict or set(final) != {"dtype", "shape", "sha256"}
            or final["dtype"] != "|u1" or final["shape"] != dimensions
            or re.fullmatch("[0-9a-f]{64}", str(final["sha256"])) is None):
        raise ValueError("Actual final label array identity differs")
    evidence = receipt["evidence"]
    if (type(evidence) is not dict
            or set(evidence) != {"decisions", "ordered_transitions", "pre_storage_mask_pair",
                                 "scored_outside_final_co2_zeroed", "branch_counts", "unscored_policy",
                                 "score_feedback", "score_reconstruction", "quantization", "decision_thresholds",
                                 "raw_later_rules", "spatial_proof", "population_composition"}
            or type(evidence["branch_counts"]) is not dict or set(evidence["branch_counts"]) != set(CODES)
            or any(type(v) is not int or v < 0 for v in evidence["branch_counts"].values())
            or sum(evidence["branch_counts"].values()) != dimensions[0] * dimensions[1]
            or type(evidence["ordered_transitions"]) is not list or not isinstance(evidence["decisions"], dict)
            or type(evidence["scored_outside_final_co2_zeroed"]) is not int
            or not 0 <= evidence["scored_outside_final_co2_zeroed"] <= dimensions[0] * dimensions[1]):
        raise ValueError("Explicit actual branch/support and ordered transition evidence required")
    if any(evidence.get(key) is not False for key in ("score_feedback", "score_reconstruction", "quantization")):
        raise ValueError("Probability evidence must retain no-feedback/no-reconstruction contract")
    raw = evidence["raw_later_rules"]
    proof = evidence["spatial_proof"]
    pixels = dimensions[0] * dimensions[1]
    proof_keys = {"format", "rules", "threshold_disagreement_pixels", "disagreement_mask",
                  "applicable_union_pixels", "overlapping_pixels", "total_memberships",
                  "membership_histogram", "below_threshold_final_gas_pixels", "unexplained_pixels"}
    if (type(raw) is not dict or set(raw) != set(RULE_NAMES)
            or type(proof) is not dict or set(proof) != proof_keys
            or proof["format"] != "darsia-decision-spatial-membership-v3"
            or type(proof["rules"]) is not dict or set(proof["rules"]) != set(RULE_NAMES)
            or type(proof["unexplained_pixels"]) is not int or proof["unexplained_pixels"] != 0
            or type(proof["below_threshold_final_gas_pixels"]) is not int
            or proof["below_threshold_final_gas_pixels"] != 0):
        raise ValueError("Actual spatial later-rule proof required")
    for key in ("threshold_disagreement_pixels", "applicable_union_pixels", "overlapping_pixels"):
        if type(proof[key]) is not int or not 0 <= proof[key] <= pixels:
            raise ValueError("Spatial proof pixel accounting differs")
    for row in proof["rules"].values():
        if (type(row) is not dict or set(row) != {"applicable_pixels", "disagreement_pixels", "applicable_mask", "disagreement_intersection"}
                or type(row["applicable_pixels"]) is not int or not 0 <= row["applicable_pixels"] <= pixels
                or type(row["disagreement_pixels"]) is not int
                or not 0 <= row["disagreement_pixels"] <= min(row["applicable_pixels"], proof["threshold_disagreement_pixels"])):
            raise ValueError("Exact actual per-rule membership counts required")
        for key in ("applicable_mask", "disagreement_intersection"):
            bound = row[key]
            if (type(bound) is not dict or set(bound) != {"dtype", "shape", "sha256"}
                    or bound["dtype"] != "|b1" or bound["shape"] != dimensions
                    or re.fullmatch("[0-9a-f]{64}", str(bound["sha256"])) is None):
                raise ValueError("Actual spatial rule/intersection identities required")
    histogram = proof["membership_histogram"]
    if (type(histogram) is not dict or set(histogram) != {str(n) for n in range(13)}
            or any(type(n) is not int or n < 0 for n in histogram.values())
            or sum(histogram.values()) != pixels
            or proof["applicable_union_pixels"] != pixels - histogram["0"]
            or proof["overlapping_pixels"] != sum(histogram[str(n)] for n in range(2, 13))
            or type(proof["total_memberships"]) is not int
            or proof["total_memberships"] != sum(n * histogram[str(n)] for n in range(13))
            or proof["total_memberships"] != sum(row["applicable_pixels"] for row in proof["rules"].values())
            or proof["threshold_disagreement_pixels"] > proof["applicable_union_pixels"]):
        raise ValueError("Spatial membership union/overlap accounting differs")
    mismatch_identity = proof["disagreement_mask"]
    if (type(mismatch_identity) is not dict or set(mismatch_identity) != {"dtype", "shape", "sha256"}
            or mismatch_identity["dtype"] != "|b1" or mismatch_identity["shape"] != dimensions
            or re.fullmatch("[0-9a-f]{64}", str(mismatch_identity["sha256"])) is None):
        raise ValueError("Actual threshold disagreement mask identity required")
    thresholds = evidence["decision_thresholds"]
    if type(thresholds) is not dict or set(thresholds) != {"1", "2", "3"}:
        raise ValueError("Exact per-branch decision thresholds required")
    for name, code in CODES.items():
        if not code:
            continue
        threshold = thresholds[str(code)]
        if threshold is None:
            if evidence["branch_counts"][name]:
                raise ValueError("Scored branch has no actual threshold provenance")
        elif (type(threshold) is not dict or set(threshold) != {"pixel", "regional"}
              or any(type(v) is not float or not np.isfinite(v) or not 0 <= v <= 1
                     for v in threshold.values())):
            raise ValueError("Actual finite literal threshold pair required")
    populations = evidence["population_composition"]
    population_names = {"central_support", "release", "seam_commit", "seam_inherited"}
    allowed_populations = (set(), population_names)
    if set(receipt["head_bindings"]) == {"selected"}:
        allowed_populations = ({"native_support"},)
    if type(populations) is not dict or set(populations) not in allowed_populations:
        raise ValueError("Exact original composition population identities required")
    for row in populations.values():
        if (type(row) is not dict or set(row) != {"dtype", "shape", "sha256"}
                or row["dtype"] != "|b1" or len(row["shape"]) != 2
                or any(type(n) is not int or n <= 0 for n in row["shape"])
                or re.fullmatch("[0-9a-f]{64}", str(row["sha256"])) is None):
            raise ValueError("Typed original composition population identity required")
    for name, row in raw.items():
        if (type(row) is not dict or set(row) != {"observed", "raw_changed_pixels", "before", "after", "changed"}
                or type(row["observed"]) is not bool or type(row["raw_changed_pixels"]) is not int
                or row["raw_changed_pixels"] < proof["rules"][name]["applicable_pixels"]):
            raise ValueError("Actual raw/applicable later-rule provenance differs")
        if not row["observed"]:
            if row != dict(observed=False, raw_changed_pixels=0, before=None, after=None, changed=None):
                raise ValueError("Unobserved rule cannot supply a proof mask")
        else:
            for key in ("before", "after", "changed"):
                bound = row[key]
                if (type(bound) is not dict or set(bound) != {"dtype", "shape", "sha256"}
                        or bound["dtype"] not in ("|b1", "|u1") or len(bound["shape"]) != 2
                        or any(type(n) is not int or n <= 0 for n in bound["shape"])
                        or re.fullmatch("[0-9a-f]{64}", str(bound["sha256"])) is None):
                    raise ValueError("Typed original later-rule operand identities required")
            if (row["changed"]["dtype"] != "|b1" or row["before"]["shape"] != row["after"]["shape"]
                    or row["changed"]["shape"] != row["before"]["shape"]
                    or row["raw_changed_pixels"] > row["changed"]["shape"][0] * row["changed"]["shape"][1]):
                raise ValueError("Original later-rule raw mask/count differs")
    return receipt


def read_product(roles, *, reader, source, case_id, labels=None):
    """Use the existing exact reader (including its indexed resolver), no fallback."""
    from ..mass.n1_pipeline import DataReader
    if type(reader) is not DataReader:
        raise TypeError("The existing exact DataReader is required")
    if type(roles) is not dict or set(roles) != {"archive", "receipt"}:
        raise ValueError("Explicit probability archive and receipt roles required")
    receipt = validate_receipt(reader.json(roles["receipt"]), roles, source=source,
                               case_id=case_id, shape=None if labels is None else labels.shape)
    if labels is None:
        from PIL import Image
        label_path = reader.path(receipt["labels"])
        with Image.open(label_path) as image:
            labels = np.asarray(image).copy()
        reader.path(receipt["labels"])
    if identity(labels) != receipt["final_labels"]:
        raise ValueError("Probability storage mask is not the actual bound final label product")
    path = reader.path(roles["archive"])
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAYS):
            raise ValueError("Undeclared probability archive members")
        arrays = {name: archive[name].copy() for name in ARRAYS}
    reader.path(roles["archive"])
    validate_arrays(arrays, labels=labels)
    if {name: identity(value) for name, value in arrays.items()} != receipt["arrays"]:
        raise ValueError("Stored probabilities differ from decision-used byte identities")
    counts = {name: int(np.count_nonzero(arrays["branch_support"] == code)) for name, code in CODES.items()}
    if counts != receipt["evidence"]["branch_counts"]:
        raise ValueError("Actual stored branch counts differ from their receipt")
    proof = spatial_proof(arrays, labels, receipt["evidence"]["decision_thresholds"])
    if not same_metadata(proof, receipt["evidence"]["spatial_proof"]):
        raise ValueError("Saved spatial rule memberships differ from their actual receipt")
    return dict(arrays=arrays, receipt=receipt)
