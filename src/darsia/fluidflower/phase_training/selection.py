"""Opt-in fixed-display anchor checks for selecting optical model complexity."""
from collections import defaultdict

import numpy as np

from .logistic import pixel_target


ANCHOR_POLICY = "fixed-display-phase-anchors-v1"
LEGACY_POLICY = "regional-oof-v1"


def display_anchors(probabilities, owners, regions, *, head, run):
    """Evaluate pure labeled regions in a held-out run, not pseudo ground truth.

    Every region contributes equally within its run/class. Missing positive
    controls remain missing; absence-only annotations cannot establish recall.
    """
    threshold = .6 if head == "gas_within_co2" else .5
    counts = np.bincount(owners, minlength=len(regions))
    positive = np.bincount(owners, weights=np.asarray(probabilities) >= threshold,
                           minlength=len(regions))
    groups = defaultdict(list)
    for index, region in enumerate(regions):
        target = pixel_target(region.phase_class, head)
        if region.run != run or target is None or counts[index] == 0:
            continue
        fraction = positive[index] / counts[index]
        groups[int(target)].append(float(fraction if target else 1-fraction))
    return {"run": run, "threshold": threshold,
            "classes": {str(key): {"regions": len(values), "correct_fraction": float(np.mean(values))}
                        for key, values in sorted(groups.items())}}


def summarize_anchors(folds):
    means = {key: [row["classes"][key]["correct_fraction"] for row in folds if key in row["classes"]]
             for key in ("0", "1")}
    if any(not values for values in means.values()):
        raise ValueError("Fixed-display selection requires positive and negative pure-phase anchors")
    recall, specificity = float(np.mean(means["1"])), float(np.mean(means["0"]))
    return {"positive_recall": recall, "negative_specificity": specificity,
            "balanced_accuracy": (recall+specificity)/2,
            "missing_positive_runs": [r["run"] for r in folds if "1" not in r["classes"]],
            "folds": folds}


def select_anchor_record(records, *, recall_tolerance=.03):
    """Do not trade a large loss of gas recall for cleaner negative regions.

    The tolerance is absolute relative to the best candidate on the *same*
    held-out, current human labels, never relative to old predicted labels.
    """
    best = max(row["display_anchors"]["positive_recall"] for row in records)
    admissible = [row for row in records
                  if row["display_anchors"]["positive_recall"] >= best-recall_tolerance]
    return max(admissible, key=lambda row: (
        row["display_anchors"]["balanced_accuracy"],
        row["display_anchors"]["negative_specificity"],
        row["mean_global_threshold_balanced_accuracy"],
        -float(row["regularization"])))
