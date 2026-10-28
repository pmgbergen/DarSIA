"""Pure proposal geometry from authenticated saved final categorical content.

The caller owns source/model admission and publication checks. This selector
reads no files and cannot infer labels, reconstruct scores or apply phase rules.
Rule masks must be the caller's authenticated actual stage-change masks, not
operator extents or inferred differences. Domain membership alone determines
eligibility; supplied filled pixels are neither reclassified nor removed.
"""

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np


SELECTION_ALGORITHM = "saved-categorical-boundaries-and-rule-changes-v1"
COORDINATE_SPACE = "normalized_native_review_canvas_16x9"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _geometry(canvas_wh, content_box_xywh, shape):
    for value, length in ((canvas_wh, 2), (content_box_xywh, 4)):
        if (not isinstance(value, (tuple, list)) or len(value) != length
                or any(type(item) is not int for item in value)):
            raise ValueError("Exact integer canvas and content geometry required")
    cw, ch = canvas_wh
    x, y, width, height = content_box_xywh
    if (min(cw, ch, width, height) <= 0 or min(x, y) < 0
            or x + width > cw or y + height > ch or shape != (height, width)):
        raise ValueError("Saved content shape must exactly match its canvas placement")
    return cw, ch, x, y, width, height


def _iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def _existing_boxes(rows, *, run_id, frame_index, geometry):
    """Accept original canvas rows or the original helper's local exclusions."""
    cw, ch, cx, cy, width, height = geometry
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Existing exclusions must be proposal records")
        run = row.get("run_id", row.get("run"))
        if run is not None and _text(run, "existing run").casefold() != run_id:
            continue
        frame = row.get("frame_index")
        if frame is not None and _integer(frame, "existing frame", 0) != frame_index:
            continue
        try:
            values = tuple(_finite(row[key], f"existing {key}")
                           for key in ("x", "y", "width", "height"))
        except KeyError as error:
            raise ValueError("Existing exclusion lacks its original bounds") from error
        if min(values[2:]) <= 0:
            raise ValueError("Existing exclusions need positive width and height")
        space = row.get("coordinate_space")
        if space == COORDINATE_SPACE:
            local = ((values[0] * cw - cx) / width,
                     (values[1] * ch - cy) / height,
                     values[2] * cw / width, values[3] * ch / height)
        elif space is None or space == "normalized_corrected_visible_2d":
            local = values
        else:
            raise ValueError("Existing exclusion has an unknown coordinate space")
        pixels = row.get("bbox_xywh_px")
        if pixels is not None:
            if (not isinstance(pixels, (tuple, list)) or len(pixels) != 4
                    or any(type(value) is not int for value in pixels)
                    or min(pixels[:2]) < 0 or min(pixels[2:]) <= 0
                    or pixels[0] + pixels[2] > width or pixels[1] + pixels[3] > height):
                raise ValueError("Existing exclusion has invalid content pixel bounds")
            exact = (pixels[0] / width, pixels[1] / height,
                     pixels[2] / width, pixels[3] / height)
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                   for a, b in zip(local, exact)):
                raise ValueError("Existing canvas and content pixel bounds differ")
            local = exact
        result.append(local)
    return result


def _overlaps(box, previous, maximum_iou):
    # Even an explicitly permissive IoU threshold cannot repeat an exact box.
    return any(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                   for a, b in zip(box, other)) or _iou(box, other) > maximum_iou
               for other in previous)


def _starts(length, size):
    values = list(range(0, length - size + 1, max(2, size // 2)))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def _boundary_pairs(labels, horizontal, vertical):
    pairs = set()
    for a, b, selected in ((labels[:, :-1], labels[:, 1:], horizontal),
                           (labels[:-1, :], labels[1:, :], vertical)):
        first, second = a[selected], b[selected]
        pairs.update(int(value) for value in np.unique(
            np.minimum(first, second) * 3 + np.maximum(first, second)))
    return pairs


def propose_saved_category_regions(
    labels, *, domain, rule_masks, run_id, frame_index, elapsed_h, model_id,
    candidate_id, fingerprint, canvas_wh, content_box_xywh, count,
    existing_regions=(), facies_labels=None, box_size_px=96,
    minimum_domain_fraction=.95, maximum_iou=.18,
):
    """Return blind original-schema boxes without changing any supplied array.

    Arrays are already the native CONTENT crop, never the padded canvas. Only
    horizontal/vertical neighbouring distinct domain classes seed boundaries;
    transitions to 255 do not. Named rule masks can independently seed boxes.
    Half-box sampling, square geometry, domain-fraction and IoU policies follow
    the original proposal geometry. The last position on each axis is included
    explicitly so the original native right/bottom edges are not lost.

    Ranking balances observed boundary pairs/rule names, facies when supplied,
    and the original 3-by-4 spatial bins. These are selection mechanics, not
    confidence values or human answers. IDs bind the original selection
    identities, algorithm and native-content pixel box, independently of batch
    size or previously selected boxes.
    """
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.uint8
            or labels.ndim != 2 or min(labels.shape) < 2
            or not np.isin(labels, (0, 1, 2, 255)).all()):
        raise ValueError("Saved labels must be exact 2D uint8 categories 0, 1, 2, 255")
    if (not isinstance(domain, np.ndarray) or domain.dtype != bool
            or domain.shape != labels.shape or not np.array_equal(domain, labels != 255)):
        raise ValueError("Exact bool domain must equal saved labels != 255")
    if not isinstance(rule_masks, Mapping):
        raise ValueError("Named actual saved rule-change masks required")
    for name, mask in rule_masks.items():
        if (not isinstance(name, str) or not name or name != name.strip()
                or not isinstance(mask, np.ndarray) or mask.dtype != bool
                or mask.shape != labels.shape):
            raise ValueError("Rule-change masks need names and exact bool content grids")
    if facies_labels is not None and (
            not isinstance(facies_labels, np.ndarray) or facies_labels.shape != labels.shape
            or facies_labels.dtype.kind not in "iu"):
        raise ValueError("Facies labels must be supplied integer content-grid labels")
    geometry = _geometry(canvas_wh, content_box_xywh, labels.shape)
    requested = _integer(count, "count", 0)
    frame_index = _integer(frame_index, "frame_index", 0)
    box_size_px = _integer(box_size_px, "box_size_px", 1)
    run_id = _text(run_id, "run_id").casefold()
    model_id, candidate_id = _text(model_id, "model_id"), _text(candidate_id, "candidate_id")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(value not in "0123456789abcdef" for value in fingerprint)):
        raise ValueError("Exact lowercase SHA-256 parameter fingerprint required")
    elapsed_h = None if elapsed_h is None else _finite(elapsed_h, "elapsed_h")
    minimum_domain_fraction = _finite(minimum_domain_fraction, "minimum_domain_fraction")
    maximum_iou = _finite(maximum_iou, "maximum_iou")
    if not 0 <= minimum_domain_fraction <= 1 or not 0 <= maximum_iou <= 1:
        raise ValueError("Domain fraction and maximum IoU must lie between zero and one")
    excluded = _existing_boxes(existing_regions, run_id=run_id,
                               frame_index=frame_index, geometry=geometry)
    if requested == 0 or not domain.any():
        return []
    cw, ch, cx, cy, width, height = geometry
    size = min(max(4, box_size_px), height, width)
    horizontal = domain[:, :-1] & domain[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    vertical = domain[:-1, :] & domain[1:, :] & (labels[:-1, :] != labels[1:, :])
    candidates = []
    for y in _starts(height, size):
        for x in _starts(width, size):
            region = np.s_[y:y + size, x:x + size]
            patch_domain = domain[region]
            fraction = float(np.count_nonzero(patch_domain)) / (size * size)
            if fraction < minimum_domain_fraction:
                continue
            h = horizontal[y:y + size, x:x + size - 1]
            v = vertical[y:y + size - 1, x:x + size]
            pairs = _boundary_pairs(labels[region], h, v)
            rules = tuple(name for name in sorted(rule_masks)
                          if np.any(rule_masks[name][region] & patch_domain))
            if not pairs and not rules:
                continue
            local = (x / width, y / height, size / width, size / height)
            if _overlaps(local, excluded, maximum_iou):
                continue
            facies = None
            if facies_labels is not None:
                values, counts = np.unique(facies_labels[region][patch_domain], return_counts=True)
                facies = int(values[int(np.argmax(counts))])
            seed = dict(run_id=run_id, frame_index=frame_index, model_id=model_id,
                        candidate_id=candidate_id, params_sha256=fingerprint,
                        selection_algorithm=SELECTION_ALGORITHM,
                        bbox_xywh_px=[x, y, size, size])
            candidates.append(dict(seed=seed, digest=_digest(seed), local=local,
                fraction=fraction, facies=facies, pairs=pairs, rules=rules,
                reasons=tuple([f"boundary:{pair}" for pair in sorted(pairs)]
                              + [f"rule:{name}" for name in rules]),
                spatial=(min(2, int(3 * (y + size / 2) / height)),
                         min(3, int(4 * (x + size / 2) / width)))))
    selected, output = [], []
    reason_counts, facies_counts, spatial_counts = {}, {}, {}
    while candidates and len(output) < requested:
        def rank(row):
            return (sum(1 / (1 + reason_counts.get(reason, 0)) for reason in row["reasons"]),
                    1 / (1 + spatial_counts.get(row["spatial"], 0)),
                    0 if row["facies"] is None else 1 / (1 + facies_counts.get(row["facies"], 0)),
                    row["digest"])
        chosen = max(candidates, key=rank)
        candidates.remove(chosen)
        if _overlaps(chosen["local"], selected, maximum_iou):
            continue
        selected.append(chosen["local"])
        for reason in chosen["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        spatial_counts[chosen["spatial"]] = spatial_counts.get(chosen["spatial"], 0) + 1
        if chosen["facies"] is not None:
            facies_counts[chosen["facies"]] = facies_counts.get(chosen["facies"], 0) + 1
        x, y, _, _ = chosen["seed"]["bbox_xywh_px"]
        nx, ny, nw, nh = (cx + x) / cw, (cy + y) / ch, size / cw, size / ch
        basis = ("saved_final_phase_boundaries_and_rule_changes" if chosen["pairs"] and chosen["rules"]
                 else "saved_final_phase_boundaries" if chosen["pairs"] else "saved_rule_changes")
        row = dict(chosen["seed"], proposal_id="phase-saved-" + chosen["digest"][:20],
                   elapsed_h=elapsed_h, candidate="A", shape="box",
                   coordinate_space=COORDINATE_SPACE, analysis_domain="registered_porous_medium_2d",
                   selection_basis=basis, model_suggestion_hidden_until_answer=True,
                   domain_fraction=chosen["fraction"], x=nx, y=ny, width=nw, height=nh,
                   points=[[nx, ny], [nx + nw, ny], [nx + nw, ny + nh], [nx, ny + nh]],
                   focus_box=dict(x=max(0., nx - nw), y=max(0., ny - nh),
                                  width=min(1., 3 * nw), height=min(1., 3 * nh)))
        if chosen["facies"] is not None:
            row["facies_id"] = chosen["facies"]
        output.append(row)
    return output
