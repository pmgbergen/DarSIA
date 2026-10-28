"""Shared training primitives for cross-series optical phase models.

The module keeps head-specific pixel supervision and regional phase-presence
supervision separate. Explicit absence of gas supervises every sampled pixel
as gas-negative without specifying aqueous CO2 or water. Mixed phase presence
does not imply that every pixel belongs to either named phase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PHASE_WATER = "water"
PHASE_AQUEOUS = "aqueous"
PHASE_GAS = "gas"
PHASE_MIXED_GAS_AQUEOUS = "mixed_gas_aqueous"
PHASE_MIXED_AQUEOUS_WATER = "mixed_aqueous_water"
PHASE_MIXED_ALL = "mixed_gas_aqueous_water"
PHASE_CO2_PRESENT_REGION = "co2_present_region"
PHASE_CO2_ABSENT_REGION = "co2_absent_region"
PHASE_GAS_PRESENT_REGION = "gas_present_region"
PHASE_AQUEOUS_PRESENT_REGION = "aqueous_present_region"
PHASE_GAS_ABSENT_REGION = "gas_absent_region"
PHASE_AQUEOUS_ABSENT_REGION = "aqueous_absent_region"
PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION = "aqueous_present_gas_absent_region"
PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT = "current_experiment_co2_artifact"
PHASE_IGNORED = "ignored"

# V4 changes only how new fits interpret review records. Historical review
# payloads and the models already trained from them remain immutable.
MANUAL_TRAINING_LABEL_POLICY_VERSION = "visible-phase-manual-v4"
LEGACY_CANDIDATE_GAS_ISSUE = "unsupported_free_gas_classification"
LEGACY_CANDIDATE_AQUEOUS_ISSUE = "unsupported_aqueous_co2_classification"
CANDIDATE_ONLY_UNSUPPORTED_PHASE_ISSUES = frozenset(
    (LEGACY_CANDIDATE_GAS_ISSUE, LEGACY_CANDIDATE_AQUEOUS_ISSUE))
EXPLICIT_GAS_FREE_AREA = "confirmed_no_free_gas_in_selected_area"
EXPLICIT_AQUEOUS_FREE_AREA = "confirmed_no_aqueous_co2_in_selected_area"

# These labels explicitly exclude gas but leave the per-pixel aqueous/water
# distinction unknown. The closed aqueous+water mixture has the same exclusion.
EXPLICIT_GAS_ABSENCE_PHASES = frozenset(
    (PHASE_GAS_ABSENT_REGION, PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION,
     PHASE_MIXED_AQUEOUS_WATER)
)

PURE_PHASES = frozenset((PHASE_WATER, PHASE_AQUEOUS, PHASE_GAS))
MIXED_PHASES = frozenset(
    (PHASE_MIXED_GAS_AQUEOUS, PHASE_MIXED_AQUEOUS_WATER, PHASE_MIXED_ALL)
)
REGIONAL_EVIDENCE_PHASES = frozenset(
    (
        PHASE_CO2_PRESENT_REGION,
        PHASE_CO2_ABSENT_REGION,
        PHASE_GAS_PRESENT_REGION,
        PHASE_AQUEOUS_PRESENT_REGION,
        PHASE_GAS_ABSENT_REGION,
        PHASE_AQUEOUS_ABSENT_REGION,
        PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION,
        PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT,
    )
)

MANUAL_PHASE_CLASSES: Mapping[str, str] = {
    "confirmed_no_co2": PHASE_WATER,
    "confirmed_only_aqueous_co2": PHASE_AQUEOUS,
    "confirmed_predominantly_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "confirmed_aqueous_co2": PHASE_AQUEOUS,
    "confirmed_predominantly_free_gas": PHASE_MIXED_GAS_AQUEOUS,
    "confirmed_only_free_gas": PHASE_GAS,
    "confirmed_free_gas": PHASE_GAS,
    "confirmed_mixed_free_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "confirmed_mixed_aqueous_water": PHASE_MIXED_AQUEOUS_WATER,
    "confirmed_mixed_free_aqueous_water": PHASE_MIXED_ALL,
    # Candidate attribution records which model exposed the problem. Only
    # source-image claims that apply to the whole marked area become labels.
    "missed_visible_plume": PHASE_CO2_PRESENT_REGION,
    "missed_visible_co2": PHASE_CO2_PRESENT_REGION,
    "missed_visible_gas": PHASE_GAS_PRESENT_REGION,
    # Reviewer clarification, 2026-09-20: this correction labels the entire
    # marked area as aqueous, including historical boxes with this category.
    "missed_visible_aqueous_co2": PHASE_AQUEOUS,
    "isolated_noise": PHASE_CO2_ABSENT_REGION,
    "disconnected_noise": PHASE_CO2_ABSENT_REGION,
    # A box around a candidate error may also contain real gas. Its legacy
    # stored training_phase_class must not turn the entire box gas-negative.
    LEGACY_CANDIDATE_GAS_ISSUE: PHASE_IGNORED,
    EXPLICIT_GAS_FREE_AREA: PHASE_GAS_ABSENT_REGION,
    LEGACY_CANDIDATE_AQUEOUS_ISSUE: PHASE_IGNORED,
    EXPLICIT_AQUEOUS_FREE_AREA: PHASE_AQUEOUS_ABSENT_REGION,
    "likely_dense_aqueous_co2_misclassified_as_gas": (
        PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION
    ),
    "incomplete_wash_indicator_carryover": PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT,
    "phase_not_assessable": PHASE_IGNORED,
    "phase_signal_artifact": PHASE_IGNORED,
}

QUIZ_PHASE_CLASSES: Mapping[str, str] = {
    "water_or_no_visible_co2": PHASE_WATER,
    "only_aqueous_co2": PHASE_AQUEOUS,
    "predominantly_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "predominantly_free_gas": PHASE_MIXED_GAS_AQUEOUS,
    "free_gas_only": PHASE_GAS,
    "mixed_free_and_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "mixed_aqueous_co2_and_water": PHASE_MIXED_AQUEOUS_WATER,
    "mixed_free_aqueous_co2_and_water": PHASE_MIXED_ALL,
    "co2_present_phase_uncertain": PHASE_CO2_PRESENT_REGION,
    "not_assessable": PHASE_IGNORED,
    "phase_signal_artifact": PHASE_IGNORED,
    "ignore_region": PHASE_IGNORED,
}

CONFIDENCE_WEIGHTS: Mapping[str, float] = {
    "high": 1.0,
    "medium": 0.55,
    "low": 0.0,
}

SOURCE_WEIGHTS: Mapping[str, float] = {
    "manual": 1.0,
    "quiz": 0.8,
}


@dataclass(frozen=True)
class ReviewedRegion:
    """One reviewed visible-2D region and its traceable provenance."""

    region_id: str
    run: str
    dataset: str
    frame_index: int
    elapsed_h: float
    phase_class: str
    confidence: str
    source: str
    reviewer: str
    payload: Mapping[str, Any]
    reviewer_id: str = ""

    @property
    def supervision_weight(self) -> float:
        return float(CONFIDENCE_WEIGHTS.get(self.confidence, 0.0)) * float(
            SOURCE_WEIGHTS.get(self.source, 0.0)
        )


@dataclass(frozen=True)
class LogisticFit:
    """Serializable result from deterministic weighted logistic IRLS."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    backend: str
    precision: str
    iterations: int
    converged: bool
    objective: float
    regularization: float
    convergence_certificate: Mapping[str, Any] | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        standardized = (matrix - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.scale, dtype=np.float32
        )
        logits = standardized @ np.asarray(self.coefficients, dtype=np.float32)
        logits += np.float32(self.intercept)
        logits = np.clip(logits, -30.0, 30.0)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)


@dataclass(frozen=True)
class _PackedCudaBags:
    """Padded weak-region tensors for fused CUDA MIL evaluation."""

    features: Any
    valid: Any
    counts: Any
    normalized_weights: Any
    selected_counts: Any | None
    target: int
    region_indices: tuple[int, ...]
    total_rows: int
    row_weights: Any | None = None
    selected_mask: Any | None = None


@dataclass(frozen=True)
class _PackedCpuBags:
    """Contiguous float64 weak rows with offsets instead of padded bags."""

    features: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    normalized_weights: np.ndarray
    selected_counts: np.ndarray | None
    target: int
    total_rows: int


def canonical_phase_class(item: Mapping[str, Any], source: str) -> str | None:
    """Map one UI record to the shared visible-phase vocabulary."""

    if source == "manual":
        category = str(item.get("category") or "").strip().lower()
        mapped = MANUAL_PHASE_CLASSES.get(category)
        if mapped is not None:
            return mapped
        declared = str(item.get("training_phase_class") or "").strip().lower()
        if declared in PURE_PHASES | MIXED_PHASES | REGIONAL_EVIDENCE_PHASES:
            return declared
        return None
    if source == "quiz":
        choice = str(item.get("human_choice") or "").strip().lower()
        schema = str(item.get("phase_label_schema") or "").strip().lower()
        # V1/V2 combined "mostly" and "entirely" labels. Preserve their
        # stored canonical class instead of silently applying V3 semantics.
        if schema and schema != "fluidflower-visible-phase-v3":
            declared = str(item.get("phase_class") or "").strip().lower()
            if declared in (
                PURE_PHASES | MIXED_PHASES | REGIONAL_EVIDENCE_PHASES | {PHASE_IGNORED}
            ):
                return declared
        mapped = QUIZ_PHASE_CLASSES.get(choice)
        if mapped is not None:
            return mapped
        declared = str(item.get("phase_class") or "").strip().lower()
        if declared in (
            PURE_PHASES | MIXED_PHASES | REGIONAL_EVIDENCE_PHASES | {PHASE_IGNORED}
        ):
            return declared
        return None
    raise ValueError(f"Unsupported review source {source!r}")


def assert_safe_training_regions(regions: Sequence[ReviewedRegion | Mapping[str, Any]]) -> None:
    """Reject frozen legacy evidence that bypassed the V4 category mapping."""

    unsafe = []
    for region in regions:
        if isinstance(region, Mapping):
            payload = region.get("payload") or {}
            phase = region.get("phase_class")
            region_id = str(region.get("region_id") or "unknown")
        else:
            payload = getattr(region, "payload", {}) or {}
            phase = getattr(region, "phase_class", None)
            region_id = str(getattr(region, "region_id", "unknown"))
        if (str(payload.get("category") or "").strip().lower()
                in CANDIDATE_ONLY_UNSUPPORTED_PHASE_ISSUES
                and phase != PHASE_IGNORED):
            unsafe.append(region_id)
    if unsafe:
        raise ValueError(
            f"{len(unsafe)} legacy candidate-specific phase issue regions still carry training labels; "
            "rebuild evidence with visible-phase-manual-v4: " + ", ".join(unsafe[:5]))


def pixel_target(phase_class: str, head: str) -> int | None:
    """Return a hard pixel target only where the review supports one."""

    if head in ("gas", "gas_within_co2") and phase_class in EXPLICIT_GAS_ABSENCE_PHASES:
        return 0
    if head == "gas":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "aqueous":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            # A gas-labelled region may also contain aqueous CO2 around or
            # behind the visible gas in the camera-facing slice.
            PHASE_GAS: None,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "co2_presence":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            PHASE_GAS: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "gas_within_co2":
        return {
            PHASE_WATER: None,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
        }.get(phase_class)
    raise ValueError(f"Unsupported phase head {head!r}")


def regional_presence_target(phase_class: str, head: str) -> int | None:
    """Return whether a reviewed region contains evidence for one phase."""

    if head == "gas":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 0,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: None,
            PHASE_CO2_ABSENT_REGION: 0,
            PHASE_GAS_PRESENT_REGION: 1,
            PHASE_AQUEOUS_PRESENT_REGION: None,
            PHASE_GAS_ABSENT_REGION: 0,
            PHASE_AQUEOUS_ABSENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 0,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "aqueous":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            PHASE_GAS: None,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 1,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: None,
            PHASE_CO2_ABSENT_REGION: None,
            PHASE_GAS_PRESENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_REGION: 1,
            PHASE_GAS_ABSENT_REGION: None,
            PHASE_AQUEOUS_ABSENT_REGION: 0,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "co2_presence":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            PHASE_GAS: 1,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 1,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: 1,
            PHASE_CO2_ABSENT_REGION: 0,
            PHASE_GAS_PRESENT_REGION: 1,
            PHASE_AQUEOUS_PRESENT_REGION: 1,
            PHASE_GAS_ABSENT_REGION: None,
            PHASE_AQUEOUS_ABSENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "gas_within_co2":
        return {
            PHASE_WATER: None,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 0,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: None,
            PHASE_CO2_ABSENT_REGION: None,
            PHASE_GAS_PRESENT_REGION: 1,
            PHASE_AQUEOUS_PRESENT_REGION: 0,
            # This head-specific absence is also a direct pixel target. The
            # regional target remains available for evaluation; it does not
            # assign a per-pixel aqueous/water class.
            PHASE_GAS_ABSENT_REGION: 0,
            PHASE_AQUEOUS_ABSENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 0,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: None,
        }.get(phase_class)
    raise ValueError(f"Unsupported phase head {head!r}")


def balanced_pixel_view(
    features: np.ndarray,
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a run/class-balanced view of all supported pixel targets.

    Every reviewed region first receives a total confidence/source weight
    independent of its pixel area.  Each ``(run, target)`` stratum is then
    normalized to equal total weight, preventing a heavily reviewed water
    class in one run from dominating the cross-series model.
    """

    matrix = np.asarray(features, dtype=np.float32)
    owners = np.asarray(region_indices, dtype=np.int32)
    if owners.shape != (matrix.shape[0],):
        raise ValueError("Region indices do not match feature rows")

    selected_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    weights_chunks: list[np.ndarray] = []
    owner_chunks: list[np.ndarray] = []
    stratum_chunks: list[np.ndarray] = []
    stratum_codes: dict[tuple[str, int], int] = {}
    # Stable sorting groups equal owners without changing the order of that
    # region's rows. One sort replaces a full owner-vector scan per region.
    owner_order = np.argsort(owners, kind="stable")
    sorted_owners = owners[owner_order]
    region_ids = np.arange(len(regions), dtype=np.int32)
    region_starts = np.searchsorted(sorted_owners, region_ids, side="left")
    region_stops = np.searchsorted(sorted_owners, region_ids, side="right")
    for index, region in enumerate(regions):
        if runs is not None and region.run not in runs:
            continue
        target = pixel_target(region.phase_class, head)
        total_weight = region.supervision_weight
        if target is None or total_weight <= 0.0:
            continue
        rows = owner_order[region_starts[index]:region_stops[index]]
        if rows.size == 0:
            continue
        selected_chunks.append(rows)
        labels_chunks.append(np.full(rows.size, target, dtype=np.int8))
        weights_chunks.append(
            np.full(rows.size, total_weight / rows.size, dtype=np.float64)
        )
        owner_chunks.append(np.full(rows.size, index, dtype=np.int32))
        stratum = (region.run, int(target))
        code = stratum_codes.setdefault(stratum, len(stratum_codes))
        stratum_chunks.append(np.full(rows.size, code, dtype=np.int32))

    if not selected_chunks:
        raise ValueError(f"No pixel supervision is available for {head}")
    selected = np.concatenate(selected_chunks)
    labels = np.concatenate(labels_chunks)
    weights = np.concatenate(weights_chunks)
    selected_owners = np.concatenate(owner_chunks)
    codes = np.concatenate(stratum_chunks)
    # Stable order retains the old selected-row summation order within each
    # stratum, including strata split across nonadjacent reviewed regions.
    stratum_order = np.argsort(codes, kind="stable")
    sorted_codes = codes[stratum_order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1]
    stops = np.r_[starts[1:], codes.size]
    for start, stop in zip(starts, stops):
        positions = stratum_order[start:stop]
        total = float(np.sum(weights[positions]))
        if total > 0.0:
            weights[positions] /= total
    weights *= float(weights.size) / max(float(np.sum(weights)), 1e-12)
    return matrix[selected], labels, weights, selected_owners


def weighted_standardization(
    features: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one shared float64 reference normalization."""

    matrix = np.asarray(features, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Training weights must have positive finite mass")
    mean = np.sum(matrix * weights[:, None], axis=0) / total
    centered = matrix - mean
    variance = np.sum(centered * centered * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale = np.maximum(scale, 1e-6)
    return mean, scale


def _objective_numpy(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    regularization: float,
) -> float:
    logits = matrix @ coefficients + intercept
    loss = _binary_logistic_loss(logits, labels, np)
    return float(
        np.sum(weights * loss) / np.sum(weights)
        + 0.5 * regularization * np.dot(coefficients, coefficients)
    )


def _binary_logistic_loss(logits: Any, labels: Any, array_module: Any) -> Any:
    """Stable binary cross-entropy, including saturated float32 logits.

    Subtracting ``label * logit`` from softplus(logit) loses the entire
    positive-class loss once both terms round to the same float32 number.
    The signed-logit form retains that small but relevant loss for IRLS line
    search at weak regularization.
    """

    signed_logits = array_module.where(labels > 0.5, -logits, logits)
    return array_module.logaddexp(0.0, signed_logits)


def _fit_cpu64(
    standardized: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, float, int, bool, float]:
    matrix = np.asarray(standardized, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weight))
    dimension = matrix.shape[1]
    coefficients = np.zeros(dimension, dtype=np.float64)
    positive = float(np.sum(weight * target) / weight_sum)
    intercept = float(
        np.log(np.clip(positive, 1e-6, 1 - 1e-6) / np.clip(1 - positive, 1e-6, 1))
    )
    identity = np.eye(dimension, dtype=np.float64)
    converged = False
    objective = _objective_numpy(
        matrix, target, weight, coefficients, intercept, regularization
    )

    for iteration in range(1, max_iter + 1):
        logits = np.clip(matrix @ coefficients + intercept, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = weight * (probability - target) / weight_sum
        curvature = (
            weight * np.maximum(probability * (1.0 - probability), 1e-7) / weight_sum
        )
        gradient = np.concatenate(
            (matrix.T @ residual + regularization * coefficients, [np.sum(residual)])
        )
        weighted_matrix = matrix * curvature[:, None]
        hessian = np.empty((dimension + 1, dimension + 1), dtype=np.float64)
        hessian[:dimension, :dimension] = (
            matrix.T @ weighted_matrix + regularization * identity
        )
        cross = matrix.T @ curvature
        hessian[:dimension, dimension] = cross
        hessian[dimension, :dimension] = cross
        hessian[dimension, dimension] = float(np.sum(curvature))
        hessian.flat[:: dimension + 2] += 1e-8
        step = np.linalg.solve(hessian, gradient)

        factor = 1.0
        accepted = False
        for _ in range(12):
            candidate_coefficients = coefficients - factor * step[:dimension]
            candidate_intercept = intercept - factor * float(step[dimension])
            candidate_objective = _objective_numpy(
                matrix,
                target,
                weight,
                candidate_coefficients,
                candidate_intercept,
                regularization,
            )
            if candidate_objective <= objective + 1e-12:
                coefficients = candidate_coefficients
                intercept = candidate_intercept
                objective = candidate_objective
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(factor * step))) < tolerance:
            converged = True
            break
    return coefficients, intercept, iteration, converged, objective


def _certify_cuda32_pure_irls_plateau(
    standardized: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    regularization: float,
    gpu_objective: float,
) -> dict[str, Any]:
    """Verify a stalled GPU fit at its unchanged float32 coefficients.

    Strongly correlated features can leave the float32 Newton *parameter*
    step above its tolerance even when the convex objective has essentially
    reached its minimum. This separate terminal certificate uses the exact
    standardized float32 inputs in float64 reductions; it never fits a CPU
    model or changes the coefficients returned by the GPU.
    """

    matrix = np.asarray(standardized, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    beta = np.asarray(coefficients, dtype=np.float64)
    weight_sum = float(np.sum(weight))
    dimension = matrix.shape[1]
    limits = {
        "gradient_linf": 1e-7,
        "newton_decrement_half": 1e-10,
        "objective_abs_difference": max(5e-8, 5e-7 * abs(float(gpu_objective))),
    }
    result: dict[str, Any] = {
        "type": "cuda32-pure-irls-flat-direction-cpu64-certificate-v1",
        "certified": False,
        "original_parameter_step_converged": False,
        "limits": limits,
    }
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(beta))
        or not np.isfinite(intercept)
        or not np.isfinite(gpu_objective)
        or not np.isfinite(weight_sum)
        or weight_sum <= 0.0
    ):
        result["reason"] = "nonfinite_inputs"
        return result

    logits = matrix @ beta + float(intercept)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    residual = weight * (probability - target) / weight_sum
    gradient = np.concatenate((
        matrix.T @ residual + regularization * beta,
        [float(np.sum(residual))],
    ))
    curvature = weight * probability * (1.0 - probability) / weight_sum
    hessian = np.empty((dimension + 1, dimension + 1), dtype=np.float64)
    hessian[:dimension, :dimension] = matrix.T @ (matrix * curvature[:, None])
    hessian[:dimension, :dimension] += regularization * np.eye(dimension)
    cross = matrix.T @ curvature
    hessian[:dimension, dimension] = cross
    hessian[dimension, :dimension] = cross
    hessian[dimension, dimension] = float(np.sum(curvature))
    objective64 = _objective_numpy(
        matrix, target, weight, beta, float(intercept), regularization
    )
    objective_difference = abs(objective64 - float(gpu_objective))
    gradient_linf = float(np.max(np.abs(gradient)))
    result.update(
        objective_gpu32=float(gpu_objective),
        objective_cpu64=float(objective64),
        objective_abs_difference=float(objective_difference),
        gradient_linf=gradient_linf,
    )
    if not (np.all(np.isfinite(hessian)) and np.all(np.isfinite(gradient))):
        result["reason"] = "nonfinite_derivatives"
        return result
    try:
        np.linalg.cholesky(hessian)
        newton_step = np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        result["reason"] = "hessian_not_positive_definite"
        return result
    decrement = float(0.5 * gradient @ newton_step)
    result.update(
        hessian_positive_definite=True,
        newton_step_linf=float(np.max(np.abs(newton_step))),
        newton_decrement_half=decrement,
    )
    result["certified"] = bool(
        np.isfinite(decrement)
        and decrement >= 0.0
        and gradient_linf <= limits["gradient_linf"]
        and decrement <= limits["newton_decrement_half"]
        and objective_difference <= limits["objective_abs_difference"]
    )
    if not result["certified"]:
        result["reason"] = "convex_optimality_bounds_not_met"
    return result


def _fit_cuda32(
    standardized: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, float, int, bool, float, Mapping[str, Any] | None]:
    import cupy as cp

    matrix = cp.asarray(standardized, dtype=cp.float32)
    target = cp.asarray(labels, dtype=cp.float32)
    weight = cp.asarray(weights, dtype=cp.float32)
    weight_sum = cp.sum(weight, dtype=cp.float32)
    dimension = matrix.shape[1]
    coefficients = cp.zeros(dimension, dtype=cp.float32)
    positive = cp.sum(weight * target, dtype=cp.float32) / weight_sum
    positive = cp.clip(positive, cp.float32(1e-6), cp.float32(1 - 1e-6))
    intercept = cp.log(positive / (cp.float32(1.0) - positive))
    identity = cp.eye(dimension, dtype=cp.float32)

    def objective_value(beta: Any, bias: Any) -> Any:
        logits = matrix @ beta + bias
        loss = _binary_logistic_loss(logits, target, cp)
        return cp.sum(weight * loss, dtype=cp.float32) / weight_sum + cp.float32(
            0.5 * regularization
        ) * cp.dot(beta, beta)

    objective = objective_value(coefficients, intercept)
    converged = False
    for iteration in range(1, max_iter + 1):
        logits = cp.clip(matrix @ coefficients + intercept, -30.0, 30.0)
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-logits))
        residual = weight * (probability - target) / weight_sum
        curvature = (
            weight
            * cp.maximum(
                probability * (cp.float32(1.0) - probability),
                cp.float32(1e-7),
            )
            / weight_sum
        )
        gradient = cp.concatenate(
            (
                matrix.T @ residual + cp.float32(regularization) * coefficients,
                cp.sum(residual, dtype=cp.float32).reshape(1),
            )
        )
        weighted_matrix = matrix * curvature[:, None]
        hessian = cp.empty((dimension + 1, dimension + 1), dtype=cp.float32)
        hessian[:dimension, :dimension] = (
            matrix.T @ weighted_matrix + cp.float32(regularization) * identity
        )
        cross = matrix.T @ curvature
        hessian[:dimension, dimension] = cross
        hessian[dimension, :dimension] = cross
        hessian[dimension, dimension] = cp.sum(curvature, dtype=cp.float32)
        hessian += cp.float32(1e-6) * cp.eye(dimension + 1, dtype=cp.float32)
        step = cp.linalg.solve(hessian, gradient)

        factor = cp.float32(1.0)
        accepted = False
        for _ in range(12):
            candidate_coefficients = coefficients - factor * step[:dimension]
            candidate_intercept = intercept - factor * step[dimension]
            candidate_objective = objective_value(
                candidate_coefficients, candidate_intercept
            )
            if bool((candidate_objective <= objective + cp.float32(1e-7)).item()):
                coefficients = candidate_coefficients
                intercept = candidate_intercept
                objective = candidate_objective
                accepted = True
                break
            factor *= cp.float32(0.5)
        if not accepted:
            break
        if float(cp.max(cp.abs(factor * step)).item()) < tolerance:
            converged = True
            break

    result = cp.asnumpy(coefficients).astype(np.float64)
    result_intercept = float(intercept.item())
    result_objective = float(objective.item())
    certificate = None
    if not converged:
        certificate = _certify_cuda32_pure_irls_plateau(
            standardized, labels, weights, result, result_intercept,
            regularization=regularization, gpu_objective=result_objective,
        )
        converged = bool(certificate["certified"])
    del coefficients
    cp.get_default_memory_pool().free_all_blocks()
    return (
        result,
        result_intercept,
        iteration,
        converged,
        result_objective,
        certificate,
    )


def cuda_available() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - optional runtime and driver failures vary
        return False


def fit_weighted_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    regularization: float,
    backend: str = "auto",
    max_iter: int = 40,
    tolerance: float = 1e-5,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> LogisticFit:
    """Fit one deterministic weighted logistic model on CPU or CUDA."""

    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int8)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("Training matrix and labels differ")
    if weights.shape != (target.size,):
        raise ValueError("Training weights and labels differ")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("Logistic training requires both target classes")
    finite = np.all(np.isfinite(matrix), axis=1) & np.isfinite(weights) & (weights > 0)
    matrix = matrix[finite]
    target = target[finite]
    weights = weights[finite]
    mean, scale = (
        weighted_standardization(matrix, weights)
        if normalization is None
        else (
            np.asarray(normalization[0], dtype=np.float64),
            np.asarray(normalization[1], dtype=np.float64),
        )
    )

    requested = backend.strip().lower()
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"
    if requested == "cuda":
        standardized = (matrix - mean.astype(np.float32)) / scale.astype(np.float32)
        coefficients, intercept, iterations, converged, objective, certificate = _fit_cuda32(
            standardized,
            target,
            weights,
            regularization=float(regularization),
            max_iter=int(max_iter),
            tolerance=float(tolerance),
        )
        precision = "float32"
    elif requested == "cpu":
        certificate = None
        standardized = (matrix.astype(np.float64) - mean) / scale
        coefficients, intercept, iterations, converged, objective = _fit_cpu64(
            standardized,
            target,
            weights,
            regularization=float(regularization),
            max_iter=int(max_iter),
            tolerance=float(tolerance),
        )
        precision = "float64"
    else:
        raise ValueError(f"Unsupported logistic backend {backend!r}")

    return LogisticFit(
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        intercept=float(intercept),
        backend=requested,
        precision=precision,
        iterations=int(iterations),
        converged=bool(converged),
        objective=float(objective),
        regularization=float(regularization),
        convergence_certificate=certificate,
    )


def _weak_region_bags(
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None,
) -> list[tuple[np.ndarray, int, float, int]]:
    """Return run/class-balanced weak bags not used as hard pixel labels."""

    owners = np.asarray(region_indices, dtype=np.int32)
    pending: list[tuple[np.ndarray, int, float, int, tuple[str, int]]] = []
    for index, region in enumerate(regions):
        if runs is not None and region.run not in runs:
            continue
        if pixel_target(region.phase_class, head) is not None:
            continue
        target = regional_presence_target(region.phase_class, head)
        weight = region.supervision_weight
        if target is None or weight <= 0.0:
            continue
        rows = np.flatnonzero(owners == index)
        if rows.size == 0:
            continue
        dominance = regional_phase_dominance(region)
        pending.append(
            (rows, int(target), float(weight), index, (region.run, int(target)))
        )
        if head == "gas_within_co2" and dominance == "aqueous":
            # Retain the assertion that some gas is visible, and separately
            # require a majority with low gas score. Neither bag fixes which
            # native pixels carry which phase.
            pending.append((rows, 0, float(weight), index, (region.run, 0)))

    totals: dict[tuple[str, int], float] = {}
    for _rows, _target, weight, _index, stratum in pending:
        totals[stratum] = totals.get(stratum, 0.0) + weight
    bags: list[tuple[np.ndarray, int, float, int]] = []
    for rows, target, weight, index, stratum in pending:
        total = totals[stratum]
        if total > 0.0:
            bags.append((rows, target, weight / total, index))
    return bags


MIL_NORMALIZATION_POLICY = "mil-active-support-v1"
MIL_SUPERVISION_POLICY = "phase-supervision-explicit-gas-absence-v2"


def pixel_supervision_metadata(
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None,
) -> dict[str, Any]:
    """Record active direct supervision independently of weak-bag strength.

    Counts describe sampled valid feature rows, not full native box areas.
    Held-out, zero-weight and unsampled regions cannot enter these counts.
    """
    owners, counts = np.unique(np.asarray(region_indices, dtype=np.int32), return_counts=True)
    active = []
    for owner, count in zip(owners, counts, strict=True):
        region = regions[int(owner)]
        target = pixel_target(region.phase_class, head)
        if (target is None or region.supervision_weight <= 0
                or (runs is not None and region.run not in runs)):
            continue
        active.append(dict(region_id=region.region_id, region_index=int(owner),
            phase_class=region.phase_class, target=int(target), sampled_rows=int(count)))
    explicit = [record for record in active
                if head in ("gas", "gas_within_co2")
                and record["phase_class"] in EXPLICIT_GAS_ABSENCE_PHASES]
    return dict(policy=MIL_SUPERVISION_POLICY, head=head,
        direct_regions=len(active), direct_sampled_rows=sum(row["sampled_rows"] for row in active),
        negative_regions=sum(row["target"] == 0 for row in active),
        positive_regions=sum(row["target"] == 1 for row in active),
        explicit_gas_absence_regions=len(explicit),
        explicit_gas_absence_sampled_rows=sum(row["sampled_rows"] for row in explicit),
        explicit_gas_absence_constraints=explicit,
        absence_assigns_aqueous_or_water=False)


def regional_phase_dominance(region: ReviewedRegion) -> str | None:
    """Interpret dominance only for the canonical two-phase mixed class.

    Older schemas sometimes used 'predominantly' for canonical pure labels;
    those historical labels retain their stored pixel-supervision contract.
    """
    if region.phase_class != PHASE_MIXED_GAS_AQUEOUS:
        return None
    aliases = {
        "confirmed_predominantly_free_gas": "gas", "predominantly_free_gas": "gas",
        "confirmed_predominantly_aqueous_co2": "aqueous", "predominantly_aqueous_co2": "aqueous",
    }
    claims = {aliases[value] for key in ("category", "human_choice")
              if (value := str(region.payload.get(key) or "").strip().lower()) in aliases}
    declared = region.payload.get("phase_dominance")
    if declared is not None and str(declared).strip():
        value = str(declared).strip().lower()
        if value not in {"gas", "aqueous", "none"}:
            raise ValueError(f"Invalid mixed-region dominance in {region.region_id}: {value}")
        claims.add(value)
    if len(claims) > 1:
        raise ValueError(f"Contradictory mixed-region dominance in {region.region_id}")
    value = next(iter(claims), "none")
    return None if value == "none" else value


def _dominance_bag_selection(regions, bags, *, head):
    """Mark dynamic strict-majority constraints without changing bag tuples."""
    selections = {}
    if head == "gas_within_co2":
        for _rows, target, _weight, owner in bags:
            dominance = regional_phase_dominance(regions[owner])
            if (dominance == "gas" and target == 1) or (dominance == "aqueous" and target == 0):
                selections[(int(owner), int(target))] = "strict-majority"
    return selections


def _bag_selected_count(count, target, owner, positive_top_fraction, bag_selection):
    if (bag_selection or {}).get((int(owner), int(target))) == "strict-majority":
        return int(count) // 2 + 1
    return max(1, round(float(positive_top_fraction) * int(count))) if int(target) else int(count)


def _mil_supervision_metadata(regions, bags, selections):
    constraints = [dict(region_id=regions[owner].region_id, region_index=int(owner),
        dominant_phase=regional_phase_dominance(regions[owner]), target=int(target),
        sampled_rows=len(rows), selected_rows=len(rows)//2 + 1,
        selection="highest-current-gas-scores" if target else "lowest-current-gas-scores")
        for rows, target, _weight, owner in bags if (int(owner), int(target)) in selections]
    return dict(policy=MIL_SUPERVISION_POLICY, majority_fraction=.5,
        strict_majority=True, fixed_pixel_labels_created=False,
        selection_updated_each_optimizer_step=True,
        gas_dominant_regions=sum(row["target"] == 1 for row in constraints),
        aqueous_dominant_regions=sum(row["target"] == 0 for row in constraints),
        majority_constraints=constraints)


def _mil_normalization(
    pure_matrix: np.ndarray,
    pure_weights: np.ndarray,
    matrix: np.ndarray,
    bags: Sequence[tuple[np.ndarray, int, float, int]],
    *,
    weak_strength: float,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """Normalize only the support admitted to this head's training loss.

    Pure rows carry total mass one. Weak bags carry total mass weak_strength,
    using the same balanced bag weights as MIL and uniform mass within a bag.
    This describes feature support; it does not turn regional evidence into
    hard pixel labels or depend on the changing positive top-k selection.

    The caller has already excluded held-out runs, ignored regions and zero
    confidence evidence. A feature constant in pure rows can still vary in
    weak bags, so pure-only normalization is unsafe for their shared model.
    """
    pure_total = float(np.sum(pure_weights))
    bag_total = float(sum(bag[2] for bag in bags))
    if not np.isfinite(weak_strength) or weak_strength <= 0:
        raise ValueError("Active MIL normalization requires positive finite weak strength")
    if not np.isfinite(pure_total) or pure_total <= 0 or not np.isfinite(bag_total) or bag_total <= 0:
        raise ValueError("MIL normalization requires positive finite training mass")

    def chunks():
        for first in range(0, len(pure_matrix), 65536):
            last = min(first + 65536, len(pure_matrix))
            yield (np.asarray(pure_matrix[first:last], dtype=np.float64),
                   np.asarray(pure_weights[first:last], dtype=np.float64) / pure_total)
        for rows, _target, weight, _owner in bags:
            row_mass = float(weak_strength) * float(weight) / bag_total / len(rows)
            for first in range(0, len(rows), 65536):
                indices = rows[first:first + 65536]
                yield (np.asarray(matrix[indices], dtype=np.float64),
                       np.full(len(indices), row_mass, dtype=np.float64))

    total = 1.0 + float(weak_strength)
    mean = np.zeros(matrix.shape[1], dtype=np.float64)
    for values, weights in chunks():
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Nonfinite features or invalid weights in active MIL support")
        mean += np.sum(values * weights[:, None], axis=0)
    mean /= total
    variance = np.zeros_like(mean)
    for values, weights in chunks():
        centered = values - mean
        variance += np.sum(centered * centered * weights[:, None], axis=0)
    variance /= total
    constant = variance <= 1e-12
    # Unit scale is neutral for a constant feature and avoids amplifying a
    # later nonconstant observation by the reciprocal of an epsilon floor.
    scale = np.where(constant, 1.0, np.sqrt(np.maximum(variance, 0.0)))
    metadata = {"policy": MIL_NORMALIZATION_POLICY,
                "scope": "active pure and weak training support; held-out runs excluded",
                "pure_mass": 1.0, "weak_mass": float(weak_strength),
                "pure_rows": len(pure_matrix), "weak_rows": sum(len(bag[0]) for bag in bags),
                "constant_feature_indices": np.flatnonzero(constant).tolist()}
    return (mean, scale), metadata


def _pack_weak_bags_cuda(
    matrix: np.ndarray,
    bags: Sequence[tuple[np.ndarray, int, float, int]],
    *,
    target: int,
    mean: np.ndarray,
    scale: np.ndarray,
    weight_sum: float,
    positive_top_fraction: float,
    bag_selection: Mapping[tuple[int, int], str] | None = None,
) -> _PackedCudaBags | None:
    """Pack equal-target bags once instead of transferring every bag per step."""

    selected = [bag for bag in bags if int(bag[1]) == int(target)]
    if not selected:
        return None

    import cupy as cp

    dimension = int(matrix.shape[1])
    maximum_rows = max(int(rows.size) for rows, *_rest in selected)
    host_features = np.zeros(
        (len(selected), maximum_rows, dimension),
        dtype=np.float32,
    )
    host_valid = np.zeros((len(selected), maximum_rows), dtype=bool)
    host_counts = np.empty(len(selected), dtype=np.int32)
    host_weights = np.empty(len(selected), dtype=np.float32)
    host_selected_counts = (
        np.empty(len(selected), dtype=np.int32)
        if int(target) == 1 or any((bag[3], int(target)) in (bag_selection or {}) for bag in selected) else None
    )
    region_ids: list[int] = []
    total_rows = 0
    mean32 = np.asarray(mean, dtype=np.float32)
    scale32 = np.asarray(scale, dtype=np.float32)
    for bag_index, (rows, _target, bag_weight, region_index) in enumerate(selected):
        count = int(rows.size)
        host_features[bag_index, :count] = (
            np.asarray(matrix[rows], dtype=np.float32) - mean32
        ) / scale32
        host_valid[bag_index, :count] = True
        host_counts[bag_index] = count
        host_weights[bag_index] = np.float32(float(bag_weight) / float(weight_sum))
        if host_selected_counts is not None:
            host_selected_counts[bag_index] = _bag_selected_count(
                count, target, region_index, positive_top_fraction, bag_selection)
        region_ids.append(int(region_index))
        total_rows += count

    device_counts = cp.asarray(host_counts, dtype=cp.int32)
    device_weights = cp.asarray(host_weights, dtype=cp.float32)
    device_selected_counts = (
        None if host_selected_counts is None
        else cp.asarray(host_selected_counts, dtype=cp.int32)
    )
    # These depend only on bag sizes and weights, never on model parameters.
    if device_selected_counts is None:
        row_weights = device_weights[:, None] / device_counts[:, None]
        selected_mask = None
    else:
        positions = cp.arange(maximum_rows, dtype=cp.int32)[None, :]
        selected_mask = (
            positions >= (cp.int32(maximum_rows) - device_selected_counts[:, None])
            if target else positions < device_selected_counts[:, None]
        )
        row_weights = None
    return _PackedCudaBags(
        features=cp.asarray(host_features, dtype=cp.float32),
        valid=cp.asarray(host_valid),
        counts=device_counts,
        normalized_weights=device_weights,
        selected_counts=device_selected_counts,
        target=int(target),
        region_indices=tuple(region_ids),
        total_rows=int(total_rows),
        row_weights=row_weights,
        selected_mask=selected_mask,
    )


def _packed_weak_terms_cuda(
    packed: _PackedCudaBags,
    coefficients: Any,
    intercept: Any,
) -> tuple[Any, Any, Any]:
    """Evaluate one target group with a bounded number of CUDA operations."""

    import cupy as cp

    bag_count, maximum_rows, dimension = packed.features.shape
    flat_features = packed.features.reshape(bag_count * maximum_rows, dimension)
    logits = (flat_features @ coefficients).reshape(bag_count, maximum_rows)
    logits = cp.clip(logits + intercept, -30.0, 30.0)

    if packed.target == 0 and packed.selected_counts is None:
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-logits))
        row_weights = packed.row_weights
        if row_weights is None:
            raise RuntimeError("Ordinary negative MIL bags require row weights")
        residual = cp.where(
            packed.valid,
            probability * row_weights,
            cp.float32(0.0),
        )
        safe_logits = cp.where(packed.valid, logits, cp.float32(0.0))
        per_bag_loss = (
            cp.sum(
                cp.where(
                    packed.valid,
                    cp.logaddexp(cp.float32(0.0), safe_logits),
                    cp.float32(0.0),
                ),
                axis=1,
                dtype=cp.float32,
            )
            / packed.counts
        )
        loss = cp.sum(
            packed.normalized_weights * per_bag_loss,
            dtype=cp.float32,
        )
    else:
        if packed.selected_counts is None:
            raise RuntimeError("Selected MIL bags require selected row counts")
        masked_logits = cp.where(packed.valid, logits, -cp.inf if packed.target else cp.inf)
        order = cp.argsort(masked_logits, axis=1)
        ordered_logits = cp.take_along_axis(masked_logits, order, axis=1)
        selected_mask = packed.selected_mask
        if selected_mask is None:
            raise RuntimeError("Selected MIL bags require a selection mask")
        safe_logits = cp.where(selected_mask, ordered_logits, cp.float32(0.0))
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-safe_logits))
        ordered_residual = cp.where(
            selected_mask,
            (probability - cp.float32(packed.target))
            * packed.normalized_weights[:, None]
            / packed.selected_counts[:, None],
            cp.float32(0.0),
        )
        residual = cp.empty_like(ordered_residual)
        cp.put_along_axis(residual, order, ordered_residual, axis=1)
        per_bag_loss = (
            cp.sum(
                cp.where(
                    selected_mask,
                    cp.logaddexp(cp.float32(0.0), -safe_logits if packed.target else safe_logits),
                    cp.float32(0.0),
                ),
                axis=1,
                dtype=cp.float32,
            )
            / packed.selected_counts
        )
        loss = cp.sum(
            packed.normalized_weights * per_bag_loss,
            dtype=cp.float32,
        )

    flat_residual = residual.reshape(bag_count * maximum_rows)
    gradient = flat_features.T @ flat_residual
    bias_gradient = cp.sum(flat_residual, dtype=cp.float32)
    return gradient, bias_gradient, loss


def _mil_objective_cuda(
    pure_matrix: Any,
    pure_labels: Any,
    pure_weights: Any,
    coefficients: Any,
    intercept: Any,
    *,
    positive_payload: _PackedCudaBags | None,
    negative_payload: _PackedCudaBags | None,
    weak_strength: float,
    regularization: float,
) -> float:
    """Evaluate the unchanged float32 MIL objective at proposed parameters."""

    import cupy as cp

    logits = cp.clip(pure_matrix @ coefficients + intercept, -30.0, 30.0)
    pure_loss = cp.sum(
        pure_weights * (
            cp.logaddexp(cp.float32(0.0), logits) - pure_labels * logits
        ), dtype=cp.float32,
    )
    weak_loss = cp.float32(0.0)
    for packed in (positive_payload, negative_payload):
        if packed is not None:
            weak_loss += _packed_weak_terms_cuda(packed, coefficients, intercept)[2]
    return float((
        pure_loss + cp.float32(weak_strength) * weak_loss
        + cp.float32(0.5 * regularization) * cp.dot(coefficients, coefficients)
    ).item())


def _pack_weak_bags_cpu(
    matrix: np.ndarray,
    bags: Sequence[tuple[np.ndarray, int, float, int]],
    *,
    target: int,
    mean: np.ndarray,
    scale: np.ndarray,
    weight_sum: float,
    positive_top_fraction: float,
    bag_selection: Mapping[tuple[int, int], str] | None = None,
) -> _PackedCpuBags | None:
    """Standardize CPU bags once without allocating maximum-size padding."""

    selected = [bag for bag in bags if int(bag[1]) == int(target)]
    if not selected:
        return None
    counts = np.asarray([rows.size for rows, *_rest in selected], dtype=np.intp)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    features = np.empty((int(offsets[-1]), matrix.shape[1]), dtype=np.float64)
    mean64, scale64 = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    for index, (rows, _target, _weight, _region) in enumerate(selected):
        features[offsets[index]:offsets[index + 1]] = (
            np.asarray(matrix[rows], dtype=np.float64) - mean64
        ) / scale64
    return _PackedCpuBags(
        features=features,
        counts=counts,
        offsets=offsets,
        normalized_weights=np.asarray(
            [float(bag[2]) / float(weight_sum) for bag in selected], dtype=np.float64
        ),
        selected_counts=(
            np.asarray([_bag_selected_count(len(rows), target, owner, positive_top_fraction, bag_selection)
                        for rows, _target, _weight, owner in selected], dtype=np.intp)
            if int(target) == 1 or any((bag[3], int(target)) in (bag_selection or {}) for bag in selected) else None
        ),
        target=int(target),
        total_rows=int(offsets[-1]),
    )


def _packed_weak_terms_cpu(
    packed: _PackedCpuBags,
    coefficients: np.ndarray,
    intercept: float,
) -> tuple[np.ndarray, float, float]:
    """Use CUDA's clipped-logit loss/gradient convention in CPU float64.

    Positive bags select the highest-scoring requested rows at each step;
    negative majority bags select the lowest scores; ordinary negatives use
    every row. Stable sorting makes exact CPU ties
    deterministic. Gradients follow the selected rows, as in the CUDA path.
    """

    logits = np.clip(packed.features @ coefficients + intercept, -30.0, 30.0)
    if packed.target == 0 and packed.selected_counts is None:
        row_weights = np.repeat(packed.normalized_weights / packed.counts, packed.counts)
        residual = row_weights / (1.0 + np.exp(-logits))
        loss = float(np.dot(row_weights, np.logaddexp(0.0, logits)))
    else:
        if packed.selected_counts is None:
            raise RuntimeError("Selected MIL bags require selected row counts")
        residual = np.zeros_like(logits)
        loss = 0.0
        for index, selected_count in enumerate(packed.selected_counts):
            start, stop = packed.offsets[index:index + 2]
            bag_logits = logits[start:stop]
            order = np.argsort(bag_logits, kind="stable")
            selected = order[-selected_count:] if packed.target else order[:selected_count]
            selected_logits = bag_logits[selected]
            row_weight = packed.normalized_weights[index] / selected_count
            residual[start + selected] = row_weight * (
                1.0 / (1.0 + np.exp(-selected_logits)) - packed.target
            )
            loss += float(row_weight * np.sum(np.logaddexp(0.0,
                -selected_logits if packed.target else selected_logits)))
    return packed.features.T @ residual, float(np.sum(residual)), float(loss)


def _mil_objective_gradient_cpu(
    pure_matrix: np.ndarray,
    pure_labels: np.ndarray,
    pure_weights: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    positive_payload: _PackedCpuBags | None,
    negative_payload: _PackedCpuBags | None,
    weak_strength: float,
    regularization: float,
) -> tuple[np.ndarray, float, float]:
    """Pure loss plus normalized weak-bag loss and coefficient-only L2."""

    logits = np.clip(pure_matrix @ coefficients + intercept, -30.0, 30.0)
    probability = 1.0 / (1.0 + np.exp(-logits))
    residual = pure_weights * (probability - pure_labels)
    gradient = pure_matrix.T @ residual + regularization * coefficients
    bias_gradient = float(np.sum(residual))
    loss = float(np.dot(pure_weights, np.logaddexp(0.0, logits) - pure_labels * logits))
    loss += 0.5 * regularization * float(np.dot(coefficients, coefficients))
    for packed in (positive_payload, negative_payload):
        if packed is not None:
            weak_gradient, weak_bias_gradient, weak_loss = _packed_weak_terms_cpu(
                packed, coefficients, intercept
            )
            gradient += weak_strength * weak_gradient
            bias_gradient += weak_strength * weak_bias_gradient
            loss += weak_strength * weak_loss
    return gradient, float(bias_gradient), float(loss)


def _mil_trace_iteration(iteration: int, max_iter: int) -> bool:
    """Bounded optimizer telemetry without changing the stopping decision."""

    return iteration in (1, 8, 64, 256) or iteration % 512 == 0 or iteration == max_iter


def _mil_stationary_step_fraction(
    *,
    iteration: int,
    max_iter: int,
    gradient_linf: float,
    bias_gradient_abs: float,
    proposed_maximum_step: float,
    tolerance: float,
    objective_before_step: float,
    objective_at_fraction: Any,
) -> tuple[float, int, bool]:
    """Damp a late Adam step only after a near-stationary, descent check.

    The regular MIL update and its stopping rule are left alone unless its
    proposed step remains large near the iteration limit while both gradients
    are already below the *existing* tolerance. Backtracking uses the same
    objective, with weak-bag selection recomputed at every trial. A failed
    descent check returns the unmodified Adam step and no convergence credit.
    """

    if (iteration < max(1024, max_iter - 2048)
            or not np.isfinite(objective_before_step)
            or max(gradient_linf, bias_gradient_abs) > tolerance
            or proposed_maximum_step <= tolerance):
        return 1.0, 0, False
    fraction = min(1.0, 0.5 * tolerance / proposed_maximum_step)
    for attempts in range(1, 10):
        trial = float(objective_at_fraction(fraction))
        if np.isfinite(trial) and trial <= objective_before_step:
            return fraction, attempts, True
        fraction *= 0.5
    return 1.0, 9, False


def _fit_weighted_logistic_mil_cpu(
    matrix: np.ndarray,
    pure_matrix: np.ndarray,
    pure_labels: np.ndarray,
    pure_weights: np.ndarray,
    initial: LogisticFit,
    bags: Sequence[tuple[np.ndarray, int, float, int]],
    *,
    regularization: float,
    weak_strength: float,
    positive_top_fraction: float,
    max_iter: int,
    learning_rate: float,
    tolerance: float,
    bag_selection: Mapping[tuple[int, int], str] | None = None,
) -> tuple[LogisticFit, dict[str, Any]]:
    """Apply the existing MIL Adam update schedule using only NumPy."""

    if int(max_iter) < 1:
        raise ValueError("CPU MIL max_iter must be positive")
    x_pure = (np.asarray(pure_matrix, dtype=np.float64) - initial.mean) / initial.scale
    y_pure = np.asarray(pure_labels, dtype=np.float64)
    w_pure = np.asarray(pure_weights, dtype=np.float64)
    w_pure = w_pure / np.sum(w_pure)
    weak_weight_sum = max(sum(value[2] for value in bags), 1e-12)
    payloads = tuple(_pack_weak_bags_cpu(
        matrix, bags, target=target, mean=initial.mean, scale=initial.scale,
        weight_sum=weak_weight_sum, positive_top_fraction=positive_top_fraction,
        bag_selection=bag_selection,
    ) for target in (1, 0))
    coefficients = np.asarray(initial.coefficients, dtype=np.float64).copy()
    intercept = float(initial.intercept)
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients)
    bias_first = bias_second = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-7
    stable_steps = 0
    terms = dict(positive_payload=payloads[0], negative_payload=payloads[1],
                 weak_strength=float(weak_strength), regularization=float(regularization))
    convergence_trace: list[dict[str, float | int]] = []
    maximum_step = float("nan")
    stationary_damping_attempts = 0
    stationary_damping_accepted = 0
    stationary_damping_rejected = 0
    stationary_backtracks = 0
    iterations_done = 0
    for iteration in range(1, int(max_iter) + 1):
        gradient, bias_gradient, step_objective = _mil_objective_gradient_cpu(
            x_pure, y_pure, w_pure, coefficients, intercept, **terms
        )
        gradient_linf = float(np.max(np.abs(gradient)))
        bias_gradient_abs = abs(float(bias_gradient))
        # Eight small updates only qualify for stopping once the gradient at
        # those updated parameters meets the same final certificate.
        if (stable_steps >= 8 and np.isfinite(step_objective)
                and gradient_linf <= float(tolerance)
                and bias_gradient_abs <= float(tolerance)):
            break
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > 10.0:
            gradient *= 10.0 / gradient_norm
            bias_gradient = float(np.clip(bias_gradient, -10.0, 10.0))
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * np.square(gradient)
        bias_first = beta1 * bias_first + (1.0 - beta1) * bias_gradient
        bias_second = beta2 * bias_second + (1.0 - beta2) * bias_gradient ** 2
        correction1, correction2 = 1.0 - beta1 ** iteration, 1.0 - beta2 ** iteration
        rate = float(learning_rate) / np.sqrt(1.0 + iteration / 80.0)
        coefficient_step = rate * (first_moment / correction1) / (
            np.sqrt(second_moment / correction2) + epsilon
        )
        intercept_step = rate * (bias_first / correction1) / (
            np.sqrt(bias_second / correction2) + epsilon
        )
        proposed_maximum_step = max(
            float(np.max(np.abs(coefficient_step))), abs(float(intercept_step)))
        step_fraction = 1.0
        if (iteration >= max(1024, int(max_iter) - 2048)
                and proposed_maximum_step > float(tolerance)):
            step_fraction, attempts, accepted = _mil_stationary_step_fraction(
                iteration=iteration,
                max_iter=int(max_iter),
                gradient_linf=gradient_linf,
                bias_gradient_abs=bias_gradient_abs,
                proposed_maximum_step=proposed_maximum_step,
                tolerance=float(tolerance),
                objective_before_step=float(step_objective),
                objective_at_fraction=lambda fraction: _mil_objective_gradient_cpu(
                    x_pure, y_pure, w_pure,
                    coefficients - fraction * coefficient_step,
                    intercept - fraction * intercept_step,
                    **terms,
                )[2],
            )
            if attempts:
                stationary_damping_attempts += 1
                stationary_backtracks += attempts - 1
                if accepted:
                    stationary_damping_accepted += 1
                else:
                    stationary_damping_rejected += 1
        coefficients -= step_fraction * coefficient_step
        intercept -= float(step_fraction * intercept_step)
        iterations_done = iteration
        maximum_step = proposed_maximum_step * step_fraction
        stable_steps = stable_steps + 1 if maximum_step < float(tolerance) else 0
        if _mil_trace_iteration(iteration, int(max_iter)) or stable_steps == 8:
            convergence_trace.append({
                "iteration": int(iteration),
                "objective_before_step": float(step_objective),
                "gradient_linf_before_step": gradient_linf,
                "bias_gradient_abs_before_step": bias_gradient_abs,
                "maximum_step": float(maximum_step),
                "step_fraction": float(step_fraction),
                "stable_steps_after_step": int(stable_steps),
            })
    # Re-evaluate at the returned parameters; do not mix pre-update losses
    # with a post-update regularizer in the CPU reference diagnostics.
    final_gradient, final_bias_gradient, objective = _mil_objective_gradient_cpu(
        x_pure, y_pure, w_pure, coefficients, intercept, **terms
    )
    final_gradient_linf = float(np.max(np.abs(final_gradient)))
    final_bias_gradient_abs = abs(float(final_bias_gradient))
    converged = bool(
        stable_steps >= 8 and np.isfinite(objective)
        and final_gradient_linf <= float(tolerance)
        and final_bias_gradient_abs <= float(tolerance)
    )
    result = LogisticFit(
        mean=np.asarray(initial.mean, dtype=np.float64),
        scale=np.asarray(initial.scale, dtype=np.float64),
        coefficients=coefficients, intercept=float(intercept), backend="cpu", precision="float64",
        iterations=int(iterations_done), converged=bool(converged), objective=objective,
        regularization=float(regularization),
    )
    diagnostics = {
        "mode": "pure-irls-plus-mixed-region-mil",
        "weak_bags": len(bags),
        "positive_weak_bags": sum(value[1] == 1 for value in bags),
        "negative_weak_bags": sum(value[1] == 0 for value in bags),
        "cpu_bag_evaluation": "ragged-float64-v1",
        "positive_weak_rows": 0 if payloads[0] is None else payloads[0].total_rows,
        "negative_weak_rows": 0 if payloads[1] is None else payloads[1].total_rows,
        "weak_strength": float(weak_strength),
        "positive_top_fraction": float(positive_top_fraction),
        "iterations": int(iterations_done),
        "converged": bool(converged),
        "objective": objective,
        "objective_evaluated_at": "final-parameters",
        "final_gradient_linf": final_gradient_linf,
        "final_bias_gradient_abs": final_bias_gradient_abs,
        "final_maximum_step": float(maximum_step),
        "stopping_tolerance": float(tolerance),
        "required_stable_steps": 8,
        "stationary_damping": {
            "policy": "late-conditional-descent-float64-mil-v1",
            "attempts": stationary_damping_attempts,
            "accepted": stationary_damping_accepted,
            "rejected": stationary_damping_rejected,
            "backtracks": stationary_backtracks,
            "gradient_tolerance": float(tolerance),
            "max_searches_per_attempt": 9,
        },
        "convergence_trace": convergence_trace,
    }
    return result, diagnostics


def fit_weighted_logistic_mil(
    features: np.ndarray,
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None,
    regularization: float,
    backend: str = "auto",
    weak_strength: float = 0.35,
    positive_top_fraction: float = 0.10,
    max_iter: int = 160,
    learning_rate: float = 0.035,
    tolerance: float = 2e-5,
) -> tuple[LogisticFit, dict[str, Any]]:
    """Fine-tune a pure-region logistic model with mixed-region bag losses.

    Positive mixed bags require a high-scoring subset, while negative bags
    constrain every sampled pixel. This lets pure regions anchor the optical
    classes without assigning one contradictory hard label to every pixel in
    a mixed region.
    """

    assert_safe_training_regions(regions)
    if not 0.0 < float(positive_top_fraction) <= 1.0:
        raise ValueError("positive_top_fraction must be in (0, 1]")
    if not np.isfinite(weak_strength) or float(weak_strength) < 0.0:
        raise ValueError("weak_strength must be finite and non-negative")
    requested = backend.strip().lower()
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported logistic MIL backend {backend!r}")

    matrix = np.asarray(features, dtype=np.float32)
    pure_matrix, pure_labels, pure_weights, _ = balanced_pixel_view(
        matrix,
        region_indices,
        regions,
        head=head,
        runs=runs,
    )
    bags = _weak_region_bags(
        region_indices,
        regions,
        head=head,
        runs=runs,
    )
    bag_selection = _dominance_bag_selection(regions, bags, head=head)
    supervision_metadata = _mil_supervision_metadata(regions, bags, bag_selection)
    supervision_metadata["pixel_supervision"] = pixel_supervision_metadata(
        region_indices, regions, head=head, runs=runs)
    normalization = None
    normalization_metadata = {"policy": "pure-weighted-standardization-v1"}
    if bags and float(weak_strength) > 0.0:
        normalization, normalization_metadata = _mil_normalization(
            pure_matrix, pure_weights, matrix, bags, weak_strength=float(weak_strength))
    initial = fit_weighted_logistic(
        pure_matrix,
        pure_labels,
        pure_weights,
        regularization=float(regularization),
        backend=requested,
        normalization=normalization,
        max_iter=40,
    )
    standard_initial = initial
    if not initial.converged:
        # The same deterministic objective and initialization get more Newton
        # steps only when the usual 40-step solve is uncertified. Never change
        # an already converged pure fit or its downstream MIL initialization.
        initial = fit_weighted_logistic(
            pure_matrix,
            pure_labels,
            pure_weights,
            regularization=float(regularization),
            backend=requested,
            normalization=normalization,
            max_iter=120,
        )
    pure_irls_retry = {
        "standard_max_iter": 40,
        "standard_iterations": int(standard_initial.iterations),
        "standard_converged": bool(standard_initial.converged),
        "retry_from_scratch": not standard_initial.converged,
        "retry_max_iter": 120 if not standard_initial.converged else None,
        "final_iterations": int(initial.iterations),
        "final_converged": bool(initial.converged),
    }
    if initial.convergence_certificate is not None:
        pure_irls_retry["convergence_certificate"] = initial.convergence_certificate
    if not bags or float(weak_strength) == 0.0:
        return initial, {
            "mode": "pure-region-irls",
            "weak_bags": len(bags),
            "iterations": 0,
            "converged": initial.converged,
            "pure_irls_retry": pure_irls_retry,
            "normalization": normalization_metadata,
            "regional_supervision": supervision_metadata,
        }

    if requested == "cpu":
        fit, diagnostics = _fit_weighted_logistic_mil_cpu(
            matrix, pure_matrix, pure_labels, pure_weights, initial, bags,
            regularization=float(regularization), weak_strength=float(weak_strength),
            positive_top_fraction=float(positive_top_fraction), max_iter=int(max_iter),
            learning_rate=float(learning_rate), tolerance=float(tolerance),
            bag_selection=bag_selection,
        )
        diagnostics["normalization"] = normalization_metadata
        diagnostics["regional_supervision"] = supervision_metadata
        diagnostics["pure_irls_retry"] = pure_irls_retry
        return fit, diagnostics

    import cupy as cp

    mean32 = np.asarray(initial.mean, dtype=np.float32)
    scale32 = np.asarray(initial.scale, dtype=np.float32)
    pure_standardized = (pure_matrix - mean32) / scale32
    x_pure = cp.asarray(pure_standardized, dtype=cp.float32)
    y_pure = cp.asarray(pure_labels, dtype=cp.float32)
    w_pure = cp.asarray(pure_weights, dtype=cp.float32)
    w_pure /= cp.sum(w_pure, dtype=cp.float32)
    coefficients = cp.asarray(initial.coefficients, dtype=cp.float32)
    intercept = cp.float32(initial.intercept)
    first_moment = cp.zeros_like(coefficients)
    second_moment = cp.zeros_like(coefficients)
    bias_first = cp.float32(0.0)
    bias_second = cp.float32(0.0)
    beta1 = cp.float32(0.9)
    beta2 = cp.float32(0.999)
    epsilon = cp.float32(1e-7)
    weak_weight_sum = max(sum(value[2] for value in bags), 1e-12)
    positive_payload = _pack_weak_bags_cuda(
        matrix,
        bags,
        target=1,
        mean=mean32,
        scale=scale32,
        weight_sum=weak_weight_sum,
        positive_top_fraction=float(positive_top_fraction),
        bag_selection=bag_selection,
    )
    negative_payload = _pack_weak_bags_cuda(
        matrix,
        bags,
        target=0,
        mean=mean32,
        scale=scale32,
        weight_sum=weak_weight_sum,
        positive_top_fraction=float(positive_top_fraction),
        bag_selection=bag_selection,
    )
    stable_steps = 0
    convergence_trace: list[dict[str, float | int]] = []
    maximum_step = float("nan")
    stationary_damping_attempts = 0
    stationary_damping_accepted = 0
    stationary_damping_rejected = 0
    stationary_backtracks = 0
    iterations_done = 0

    for iteration in range(1, int(max_iter) + 1):
        pure_logits = cp.clip(x_pure @ coefficients + intercept, -30.0, 30.0)
        pure_probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-pure_logits))
        pure_residual = w_pure * (pure_probability - y_pure)
        gradient = x_pure.T @ pure_residual
        bias_gradient = cp.sum(pure_residual, dtype=cp.float32)
        pure_loss = cp.sum(
            w_pure
            * (cp.logaddexp(cp.float32(0.0), pure_logits) - y_pure * pure_logits),
            dtype=cp.float32,
        )

        weak_loss = cp.float32(0.0)
        weak_gradient = cp.zeros_like(coefficients)
        weak_bias_gradient = cp.float32(0.0)
        for packed in (positive_payload, negative_payload):
            if packed is None:
                continue
            packed_gradient, packed_bias_gradient, packed_loss = (
                _packed_weak_terms_cuda(packed, coefficients, intercept)
            )
            weak_gradient += packed_gradient
            weak_bias_gradient += packed_bias_gradient
            weak_loss += packed_loss

        gradient += cp.float32(weak_strength) * weak_gradient
        bias_gradient += cp.float32(weak_strength) * weak_bias_gradient
        gradient += cp.float32(regularization) * coefficients
        trace_now = _mil_trace_iteration(iteration, int(max_iter))
        stationary_window = iteration >= max(1024, int(max_iter) - 2048)
        if trace_now or stationary_window or stable_steps >= 8:
            step_objective = float((
                pure_loss + cp.float32(weak_strength) * weak_loss
                + cp.float32(0.5 * regularization) * cp.dot(coefficients, coefficients)
            ).item())
            gradient_linf = float(cp.max(cp.abs(gradient)).item())
            bias_gradient_abs = abs(float(bias_gradient.item()))
        # Check the full, unclipped gradient at the parameters reached by the
        # preceding small steps before allowing an early stop.
        if (stable_steps >= 8 and np.isfinite(step_objective)
                and gradient_linf <= float(tolerance)
                and bias_gradient_abs <= float(tolerance)):
            break
        gradient_norm = cp.linalg.norm(gradient)
        if float(gradient_norm.item()) > 10.0:
            gradient *= cp.float32(10.0) / gradient_norm
            bias_gradient = cp.clip(bias_gradient, -10.0, 10.0)

        first_moment = beta1 * first_moment + (cp.float32(1.0) - beta1) * gradient
        second_moment = beta2 * second_moment + (cp.float32(1.0) - beta2) * cp.square(
            gradient
        )
        bias_first = beta1 * bias_first + (cp.float32(1.0) - beta1) * bias_gradient
        bias_second = beta2 * bias_second + (cp.float32(1.0) - beta2) * cp.square(
            bias_gradient
        )
        correction1 = cp.float32(1.0) - beta1**iteration
        correction2 = cp.float32(1.0) - beta2**iteration
        rate = cp.float32(learning_rate / np.sqrt(1.0 + iteration / 80.0))
        coefficient_step = (
            rate
            * (first_moment / correction1)
            / (cp.sqrt(second_moment / correction2) + epsilon)
        )
        intercept_step = (
            rate
            * (bias_first / correction1)
            / (cp.sqrt(bias_second / correction2) + epsilon)
        )
        proposed_maximum_step = max(
            float(cp.max(cp.abs(coefficient_step)).item()),
            abs(float(intercept_step.item())),
        )
        step_fraction = 1.0
        if stationary_window and proposed_maximum_step > float(tolerance):
            step_fraction, attempts, accepted = _mil_stationary_step_fraction(
                iteration=iteration,
                max_iter=int(max_iter),
                gradient_linf=gradient_linf,
                bias_gradient_abs=bias_gradient_abs,
                proposed_maximum_step=proposed_maximum_step,
                tolerance=float(tolerance),
                objective_before_step=step_objective,
                objective_at_fraction=lambda fraction: _mil_objective_cuda(
                    x_pure, y_pure, w_pure,
                    coefficients - cp.float32(fraction) * coefficient_step,
                    intercept - cp.float32(fraction) * intercept_step,
                    positive_payload=positive_payload,
                    negative_payload=negative_payload,
                    weak_strength=float(weak_strength),
                    regularization=float(regularization),
                ),
            )
            if attempts:
                stationary_damping_attempts += 1
                stationary_backtracks += attempts - 1
                if accepted:
                    stationary_damping_accepted += 1
                else:
                    stationary_damping_rejected += 1
        coefficients -= cp.float32(step_fraction) * coefficient_step
        intercept -= cp.float32(step_fraction) * intercept_step
        iterations_done = iteration
        maximum_step = proposed_maximum_step * step_fraction
        stable_steps = stable_steps + 1 if maximum_step < float(tolerance) else 0
        if trace_now or stable_steps == 8:
            convergence_trace.append({
                "iteration": int(iteration),
                "objective_before_step": step_objective,
                "gradient_linf_before_step": gradient_linf,
                "bias_gradient_abs_before_step": bias_gradient_abs,
                "maximum_step": float(maximum_step),
                "step_fraction": float(step_fraction),
                "stable_steps_after_step": int(stable_steps),
            })

    # Re-evaluate all loss terms at the returned parameters. The iteration
    # losses above precede the Adam step and cannot be mixed with a post-step
    # regularizer when reporting a final objective.
    final_logits = cp.clip(x_pure @ coefficients + intercept, -30.0, 30.0)
    final_probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-final_logits))
    final_residual = w_pure * (final_probability - y_pure)
    final_gradient = x_pure.T @ final_residual
    final_bias_gradient = cp.sum(final_residual, dtype=cp.float32)
    final_pure_loss = cp.sum(
        w_pure * (cp.logaddexp(cp.float32(0.0), final_logits)
                  - y_pure * final_logits), dtype=cp.float32)
    final_weak_loss = cp.float32(0.0)
    for packed in (positive_payload, negative_payload):
        if packed is None:
            continue
        packed_gradient, packed_bias_gradient, packed_loss = (
            _packed_weak_terms_cuda(packed, coefficients, intercept)
        )
        final_gradient += cp.float32(weak_strength) * packed_gradient
        final_bias_gradient += cp.float32(weak_strength) * packed_bias_gradient
        final_weak_loss += packed_loss
    final_gradient += cp.float32(regularization) * coefficients
    objective = float((
        final_pure_loss + cp.float32(weak_strength) * final_weak_loss
        + cp.float32(0.5 * regularization) * cp.dot(coefficients, coefficients)
    ).item())
    final_gradient_linf = float(cp.max(cp.abs(final_gradient)).item())
    final_bias_gradient_abs = abs(float(final_bias_gradient.item()))
    # Tiny updates can also come from a vanishing learning rate or float32
    # rounding. They never certify convergence without a final full-gradient
    # check, regardless of whether damping was used.
    converged = bool(
        stable_steps >= 8 and np.isfinite(objective)
        and final_gradient_linf <= float(tolerance)
        and final_bias_gradient_abs <= float(tolerance)
    )

    result = LogisticFit(
        mean=np.asarray(initial.mean, dtype=np.float64),
        scale=np.asarray(initial.scale, dtype=np.float64),
        coefficients=cp.asnumpy(coefficients).astype(np.float64),
        intercept=float(intercept.item()),
        backend="cuda",
        precision="float32",
        iterations=int(iterations_done),
        converged=bool(converged),
        objective=float(objective),
        regularization=float(regularization),
    )
    diagnostics = {
        "mode": "pure-irls-plus-mixed-region-mil",
        "weak_bags": len(bags),
        "positive_weak_bags": sum(value[1] == 1 for value in bags),
        "negative_weak_bags": sum(value[1] == 0 for value in bags),
        "cuda_bag_evaluation": "padded-fused-v1",
        "normalization": normalization_metadata,
        "regional_supervision": supervision_metadata,
        "pure_irls_retry": pure_irls_retry,
        "positive_weak_rows": (
            0 if positive_payload is None else positive_payload.total_rows
        ),
        "negative_weak_rows": (
            0 if negative_payload is None else negative_payload.total_rows
        ),
        "weak_strength": float(weak_strength),
        "positive_top_fraction": float(positive_top_fraction),
        "iterations": int(iterations_done),
        "converged": bool(converged),
        "objective": float(objective),
        "objective_evaluated_at": "final-parameters",
        "final_gradient_linf": final_gradient_linf,
        "final_bias_gradient_abs": final_bias_gradient_abs,
        "final_maximum_step": float(maximum_step),
        "stopping_tolerance": float(tolerance),
        "required_stable_steps": 8,
        "stationary_damping": {
            "policy": "late-conditional-descent-float32-mil-v1",
            "attempts": stationary_damping_attempts,
            "accepted": stationary_damping_accepted,
            "rejected": stationary_damping_rejected,
            "backtracks": stationary_backtracks,
            "gradient_tolerance": float(tolerance),
            "max_searches_per_attempt": 9,
        },
        "convergence_trace": convergence_trace,
    }
    del x_pure, y_pure, w_pure, coefficients, first_moment, second_moment
    del positive_payload, negative_payload
    cp.get_default_memory_pool().free_all_blocks()
    return result, diagnostics


@dataclass(frozen=True)
class _CudaPredictionFeatureBuffer:
    """One unchanged float32 host matrix retained on CUDA for CV predictions."""

    source: np.ndarray
    device: Any


def _try_cuda_prediction_feature_buffer(
    features: np.ndarray, *, backend: str,
) -> _CudaPredictionFeatureBuffer | None:
    """Fall back to chunk uploads unless the matrix fits with ample VRAM left."""

    requested = backend.strip().lower()
    if requested == "cpu" or (requested == "auto" and not cuda_available()):
        return None
    if requested not in {"auto", "cuda"}:
        return None
    matrix = np.asarray(features, dtype=np.float32)
    import cupy as cp

    try:
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception:
        # The cache is optional; preserve the original chunked prediction path.
        return None
    reserve_bytes = max(4 * 1024**3, 2 * matrix.nbytes)
    if free_bytes < matrix.nbytes + reserve_bytes:
        return None
    try:
        device = cp.asarray(matrix, dtype=cp.float32)
    except cp.cuda.memory.OutOfMemoryError:
        return None
    return _CudaPredictionFeatureBuffer(source=matrix, device=device)


def predict_logistic(
    fit: LogisticFit,
    features: np.ndarray,
    *,
    backend: str = "auto",
    chunk_rows: int = 250_000,
    feature_buffer: _CudaPredictionFeatureBuffer | None = None,
) -> np.ndarray:
    """Predict in bounded chunks, retaining GPU acceleration when available."""

    matrix = np.asarray(features, dtype=np.float32)
    requested = backend.strip().lower()
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"
    if requested == "cpu":
        return fit.predict(matrix)
    if requested != "cuda":
        raise ValueError(f"Unsupported prediction backend {backend!r}")

    import cupy as cp

    mean = cp.asarray(fit.mean, dtype=cp.float32)
    scale = cp.asarray(fit.scale, dtype=cp.float32)
    coefficients = cp.asarray(fit.coefficients, dtype=cp.float32)
    output = np.empty(matrix.shape[0], dtype=np.float32)
    if feature_buffer is not None and feature_buffer.source is not matrix:
        raise ValueError("Prediction feature buffer does not match input matrix")
    step = max(1, int(chunk_rows))
    for start in range(0, matrix.shape[0], step):
        stop = min(matrix.shape[0], start + step)
        block = (cp.asarray(matrix[start:stop], dtype=cp.float32)
                 if feature_buffer is None else feature_buffer.device[start:stop])
        logits = ((block - mean) / scale) @ coefficients
        logits += cp.float32(fit.intercept)
        logits = cp.clip(logits, -30.0, 30.0)
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-logits))
        output[start:stop] = cp.asnumpy(probability)
    del mean, scale, coefficients
    cp.get_default_memory_pool().free_all_blocks()
    return output


def region_probability_scores(
    probabilities: np.ndarray,
    region_indices: np.ndarray,
    region_count: int,
    *,
    quantile: float = 0.90,
) -> np.ndarray:
    """Aggregate pixel probabilities to robust phase-presence scores."""

    values = np.asarray(probabilities, dtype=np.float64)
    owners = np.asarray(region_indices, dtype=np.int32)
    scores = np.full(int(region_count), np.nan, dtype=np.float64)
    if owners.size:
        # Keep the input order within each region while grouping just once.
        order = np.argsort(owners, kind="stable")
        grouped_owners = owners[order]
        starts = np.r_[0, np.flatnonzero(grouped_owners[1:] != grouped_owners[:-1]) + 1]
        stops = np.r_[starts[1:], owners.size]
        for start, stop in zip(starts, stops):
            scores[int(grouped_owners[start])] = float(
                np.quantile(values[order[start:stop]], quantile))
    return scores


def regional_evaluation_view(
    scores: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an equal-run/class regional presence evaluation view."""

    score_rows: list[float] = []
    targets: list[int] = []
    weights: list[float] = []
    indices: list[int] = []
    strata: list[tuple[str, int]] = []
    for index, region in enumerate(regions):
        if runs is not None and region.run not in runs:
            continue
        target = regional_presence_target(region.phase_class, head)
        value = float(scores[index])
        weight = region.supervision_weight
        if target is None or weight <= 0.0 or not np.isfinite(value):
            continue
        score_rows.append(value)
        targets.append(int(target))
        weights.append(weight)
        indices.append(index)
        strata.append((region.run, int(target)))
    if not score_rows:
        raise ValueError(f"No regional evaluation labels for {head}")
    score_array = np.asarray(score_rows, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    for stratum in sorted(set(strata)):
        mask = np.fromiter(
            (value == stratum for value in strata), dtype=bool, count=len(strata)
        )
        total = float(np.sum(weight_array[mask]))
        if total > 0.0:
            weight_array[mask] /= total
    return score_array, target_array, weight_array, np.asarray(indices, dtype=np.int32)


def weighted_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    recalls: list[float] = []
    for value in (0, 1):
        mask = labels == value
        total = float(np.sum(weights[mask]))
        if total > 0.0:
            recalls.append(
                float(np.sum(weights[mask] * (predictions[mask] == value)) / total)
            )
    return float(np.mean(recalls)) if recalls else float("nan")


def tune_presence_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Tune a regional presence threshold by weighted balanced accuracy."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.02, 0.98, 97),
                np.quantile(score_array, np.linspace(0.02, 0.98, 97)),
            )
        )
    )
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        accuracy = weighted_balanced_accuracy(
            label_array,
            (score_array >= threshold).astype(np.int8),
            weight_array,
        )
        candidate = (float(accuracy), -abs(float(threshold) - 0.5))
        incumbent = (best[0], -abs(best[1] - 0.5))
        if candidate > incumbent:
            best = (float(accuracy), float(threshold))
    return best[1], best[0]


def regional_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Return weighted, interpretable regional binary metrics."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    predicted = (score_array >= float(threshold)).astype(np.int8)
    result = {
        "balanced_accuracy": weighted_balanced_accuracy(
            label_array, predicted, weight_array
        ),
        "threshold": float(threshold),
        "regions": int(label_array.size),
        "positive_fraction": float(
            np.sum(weight_array * label_array) / np.sum(weight_array)
        ),
    }
    if set(np.unique(label_array).tolist()) == {0, 1}:
        result["roc_auc"] = float(
            roc_auc_score(label_array, score_array, sample_weight=weight_array)
        )
        result["average_precision"] = float(
            average_precision_score(
                label_array, score_array, sample_weight=weight_array
            )
        )
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")
    return result
