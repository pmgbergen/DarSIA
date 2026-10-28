"""Compare a reproduction with a dataset's expected results, frame by frame.

Three verdicts per frame:

* IDENTICAL: every expected mass and every array of the stored layer is
  bit-identical;
* WITHIN EXPECTED VARIATION: within the manifest's cross-machine limits;
* OUTSIDE: beyond at least one limit.

The limits (the manifest's rule) apply per frame:

* labels: the changed CO2 area (dissolved and gas) and the changed gas area,
  each at most max(relative limit x reference area, absolute floor), in cm2;
* masses (g) and areas (cm2): |a - b| <= relative x |a| + absolute;
* maps: per cell, |a - b| <= relative x the frame's largest reference value of
  that array, with at most a small fraction of the cells above. Cells whose
  labels differ are exempt from the map limit; they count under the label limit.

On the stored layer the labels are area fractions per cell, so the changed area
is the sum over cells of the change in a class's area. Every check reports the
share of its limit used. A run and the dataset take their worst frame verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import numpy as np

from ..phase.metrics import array_pair_identity

IDENTICAL = "IDENTICAL"
WITHIN = "WITHIN EXPECTED VARIATION"
OUTSIDE = "OUTSIDE"
VERDICTS = (IDENTICAL, WITHIN, OUTSIDE)
MASSES = ("total", "aqueous", "gas")
CELL_AREA = "static_pixel_area_m2_sum"
LABEL_PREFIX, LABEL_SUFFIX = "phase_", "_area_fraction"
CM2_PER_M2 = 1e4


@dataclass
class Check:
    """One quantity checked against its limit, with the share of the limit used."""

    quantity: str
    kind: str
    difference: float
    limit: float
    passed: bool
    detail: str = ""
    share: float = 0.0


def _share(difference: float, limit: float) -> float:
    """The share of a limit used: 0 when nothing differs, infinite against a zero limit."""
    if difference == 0:
        return 0.0
    return math.inf if limit == 0 else difference / limit


def _check(quantity, kind, difference, limit, detail="") -> Check:
    """A check that passes when its difference is within its limit."""
    return Check(quantity, kind, difference, limit, difference <= limit, detail, _share(difference, limit))


def _finite(value):
    """A JSON-ready number: None for infinity or NaN."""
    return None if value is None or not math.isfinite(value) else value


@dataclass
class FrameComparison:
    """The verdict of one frame, its checks and the first check that failed."""

    run: str
    case_id: str
    verdict: str
    checks: list = field(default_factory=list)
    first_failure: dict | None = None

    def as_dict(self) -> dict:
        """The comparison as a JSON-ready dictionary."""
        rows = []
        for check in self.checks:
            row = asdict(check)
            for key in ("difference", "limit", "share"):
                row[key] = _finite(row[key])
            rows.append(row)
        return {"run": self.run, "case_id": self.case_id, "verdict": self.verdict, "checks": rows,
                "first_failure": self.first_failure}


def load_layer(path: str | Path) -> dict[str, np.ndarray]:
    """The arrays of a stored layer file (NumPy archive), loaded without pickles."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _same(a, b) -> bool:
    """Two scalars are the same value (NaN equals NaN)."""
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def masses_identical(expected: dict, reproduced: dict) -> bool:
    """True when the three frame masses are the same floating-point values."""
    return all(_same(expected.get(key), reproduced.get(key)) for key in MASSES)


def mass_checks(expected: dict, reproduced: dict, rule: dict) -> list[Check]:
    """The frame masses (total, aqueous, gas) against |a - b| <= relative x |a| + absolute."""
    limits = rule["masses"]
    checks = []
    for key in MASSES:
        a, b = expected.get(key), reproduced.get(key)
        if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)):
            passed = _same(a, b)
            checks.append(Check(f"{key} mass", "mass", 0.0 if passed else math.inf, 0.0, passed,
                                "missing or not finite", 0.0 if passed else math.inf))
            continue
        difference = abs(float(a) - float(b))
        limit = limits["relative"] * abs(float(a)) + limits["absolute"]
        checks.append(_check(f"{key} mass", "mass", difference, limit, f"{a!r} vs {b!r} g"))
    return checks


def label_fractions(arrays: dict) -> dict[str, np.ndarray]:
    """The layer's label area fractions by class name (water, dissolved, gas, excluded)."""
    return {name[len(LABEL_PREFIX):-len(LABEL_SUFFIX)]: np.asarray(value, np.float64)
            for name, value in arrays.items()
            if name.startswith(LABEL_PREFIX) and name.endswith(LABEL_SUFFIX)}


def _equal_cells(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cell-wise equality where NaN equals NaN."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        left, right = a.astype(np.float64), b.astype(np.float64)
        return (left == right) | (np.isnan(left) & np.isnan(right))
    return a == b


def label_checks(expected: dict, reproduced: dict, rule: dict) -> tuple[list[Check], np.ndarray | None]:
    """The changed CO2 area and the changed gas area against the label limits.

    Areas per class follow the labels, so the label limits cover them (not the
    mass and area formula). Returns the checks and the cells whose labels
    differ (exempt from the map limit), or no checks and None when the layer
    holds no labels.
    """
    a_fractions, b_fractions = label_fractions(expected), label_fractions(reproduced)
    if not a_fractions or CELL_AREA not in expected:
        return [], None
    if set(a_fractions) != set(b_fractions) or any(
            a_fractions[name].shape != b_fractions[name].shape for name in a_fractions):
        return [Check("labels", "label", math.inf, 0.0, False, "label arrays missing or of another shape",
                      math.inf)], None
    area = np.asarray(expected[CELL_AREA], np.float64) * CM2_PER_M2
    changed_cells = np.zeros(area.shape, bool)
    for name in a_fractions:
        changed_cells |= ~_equal_cells(a_fractions[name], b_fractions[name])
    labels = rule["labels"]
    checks = []
    for phase, classes, relative in (("CO2", ("dissolved", "gas"), labels["co2_relative"]),
                                     ("gas", ("gas",), labels["gas_relative"])):
        a = sum(np.nan_to_num(a_fractions[name]) for name in classes if name in a_fractions)
        b = sum(np.nan_to_num(b_fractions[name]) for name in classes if name in b_fractions)
        reference = float(np.sum(a * area, dtype=np.float64))
        reproduced_area = float(np.sum(b * area, dtype=np.float64))
        changed = float(np.sum(np.abs(a - b) * area, dtype=np.float64))
        checks.append(_check(f"changed {phase} area", "label", changed,
                             max(relative * reference, labels["absolute_cm2"]),
                             f"{reference:.6g} cm2 expected, {reproduced_area:.6g} cm2 reproduced"))
    return checks, changed_cells


def map_check(name: str, expected: np.ndarray, reproduced, rule: dict, exempt: np.ndarray | None) -> Check:
    """One array of the stored layer against the map limit; label-changed cells are exempt.

    The share used is the larger of the largest difference over the per-cell
    limit and the fraction of cells above it over the allowed fraction.
    """
    maps = rule["maps"]
    a = np.asarray(expected)
    if reproduced is None or np.asarray(reproduced).shape != a.shape:
        return Check(name, "map", math.inf, 0.0, False, "missing or of another shape", math.inf)
    b = np.asarray(reproduced)
    if array_pair_identity(b, a)["bit_identical"]:
        return Check(name, "map", 0.0, 0.0, True, "bit-identical", 0.0)
    left, right = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(left)
    per_cell = maps["relative_to_frame_maximum"] * (float(np.max(np.abs(left[finite]))) if finite.any() else 0.0)
    considered = ~_equal_cells(left, right)
    if exempt is not None and exempt.shape == considered.shape:
        considered &= ~exempt
    with np.errstate(invalid="ignore"):
        difference = np.abs(left - right)
    unbounded = considered & ~np.isfinite(difference)
    above = considered & (unbounded | (difference > per_cell))
    largest = math.inf if unbounded.any() else float(np.max(np.where(considered, difference, 0.0)))
    fraction = float(np.count_nonzero(above)) / max(1, above.size)
    allowed = maps["allowed_fraction_above"]
    share = max(_share(largest, per_cell), _share(fraction, allowed))
    return Check(name, "map", largest, per_cell, fraction <= allowed,
                 f"{np.count_nonzero(above)} of {above.size} cells above the per-cell limit", share)


def compare_frame(run: str, case_id: str, expected_masses: dict, reproduced_masses: dict,
                  expected_layer: dict, reproduced_layer: dict, rule: dict) -> FrameComparison:
    """One frame's verdict under the rule, with every check and the first failure."""
    identical = (masses_identical(expected_masses, reproduced_masses)
                 and set(expected_layer) == set(reproduced_layer)
                 and all(array_pair_identity(reproduced_layer[name], expected_layer[name])["bit_identical"]
                         for name in expected_layer))
    labels, exempt = label_checks(expected_layer, reproduced_layer, rule)
    maps = [map_check(name, expected_layer[name], reproduced_layer.get(name), rule, exempt)
            for name in sorted(expected_layer)
            if not (name.startswith(LABEL_PREFIX) and name.endswith(LABEL_SUFFIX))]
    checks = labels + mass_checks(expected_masses, reproduced_masses, rule) + maps
    failed = [check for check in checks if not check.passed]
    verdict = IDENTICAL if identical else (WITHIN if not failed else OUTSIDE)
    first = None if not failed else {"quantity": failed[0].quantity, "kind": failed[0].kind,
                                     "detail": failed[0].detail}
    return FrameComparison(run, case_id, verdict, checks, first)


def worst(verdicts) -> str:
    """The worst of some verdicts (IDENTICAL < WITHIN EXPECTED VARIATION < OUTSIDE)."""
    values = list(verdicts)
    if not values:
        raise ValueError("No verdict to combine")
    return max(values, key=VERDICTS.index)


def summarize(frames: list[FrameComparison]) -> dict:
    """Run and dataset verdicts, the largest share of each kind of limit, and the first divergence.

    Frames are taken in the order given (time order within each run). The
    first divergence is the first frame whose verdict is OUTSIDE, with its
    first failing quantity.
    """
    runs = {}
    for frame in frames:
        runs.setdefault(frame.run, []).append(frame.verdict)
    margins = {}
    for frame in frames:
        for check in frame.checks:
            margins[check.kind] = max(margins.get(check.kind, 0.0), check.share)
    first = next((frame for frame in frames if frame.verdict == OUTSIDE), None)
    run_verdicts = {run: worst(values) for run, values in runs.items()}
    dataset = worst(run_verdicts.values())
    return {
        "dataset": dataset,
        "reproduced": dataset in (IDENTICAL, WITHIN),
        "runs": run_verdicts,
        "frames": {verdict: sum(frame.verdict == verdict for frame in frames) for verdict in VERDICTS},
        "largest_share_of_limit": {key: _finite(value) for key, value in sorted(margins.items())},
        "first_divergence": None if first is None else {"run": first.run, "case_id": first.case_id,
                                                        **first.first_failure},
    }
