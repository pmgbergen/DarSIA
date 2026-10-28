"""Derive F summaries through the original saved-native N1 preparation.

This module writes nothing and creates no job. Its caller must first verify
the exact completed P1 closure and named frame/label/probability/statistics
descriptors. Worker queue ownership, when applicable, belongs to that caller.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from .phase_expert_knowledge import expert_knowledge_of

STATISTICS_FORMAT = "darsia-overflow-adjusted-n1-estimates-v1"


def baseline_volume_from_saved_frame(pipeline, mass_spec):
    """First-pass gas volume without repeating the optical interpretation.

    load_frame retains the original source/H/area/physical-support guards. The
    geometry construction and float32 Swi subtraction below are literal
    operations from NativeAnalysis and the original N1 gas-field provider.
    A frame-loading-only context is sufficient; no optical model is needed.
    """
    import darsia
    from .n1_pipeline import N1Pipeline

    if type(pipeline) is not N1Pipeline or pipeline.math.precision != "float32":
        raise ValueError("The original float32 saved-native frame context is required")
    frame, *_ = pipeline.load_frame(mass_spec)
    shape = frame.domain.shape
    domain = frame.known_physical_domain
    depth = np.broadcast_to(np.asarray(frame.depth_m, np.float64), shape).copy()
    pore = np.broadcast_to(np.asarray(frame.porosity, np.float64), shape).copy()
    depth = np.where(domain, depth, 0.)
    pore = np.where(domain, pore, 0.)
    geometry = darsia.ExtrudedPorousGeometry(porosity=pore, depth=depth,
        space_dim=2, num_voxels=shape, dimensions=list(frame.dimensions_m))
    nominal_volume = geometry.voxel_volume.copy()
    nominal_area = float(np.prod(frame.dimensions_m))/np.prod(shape)
    geometry.voxel_volume = nominal_volume*(frame.pixel_area_m2/nominal_area)
    geometry.cached_voxel_volume = geometry.voxel_volume.copy()
    swi = np.full(shape, np.nan, np.float32)
    for label, value in pipeline.config['shared']['swi_by_facies'].items():
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Measured Swi must be in [0,1]")
        swi[frame.facies == int(label)] = np.float32(value)
    keep = frame.mass_eligible_mask & domain & (expert_knowledge_of(frame).free_gas) & np.isfinite(swi)
    gas = np.zeros(shape, np.float32)
    gas[keep] = np.float32(1)-swi[keep]
    geometry._prepare_cached_voxel_volume(list(shape))
    if np.asarray(geometry.cached_voxel_volume).dtype != np.float64:
        raise TypeError("Original native integration weights must remain float64")
    return dict(baseline_gas_volume_m3=float(geometry.integrate(gas)),
                case_id=mass_spec['case_id'], frame_index=frame.frame_index,
                elapsed_h=frame.elapsed_h, source_binding=deepcopy(mass_spec['source_binding']),
                raw_decoded=False, optical_model_evaluated=False)


@dataclass(frozen=True)
class SteeringFrame:
    pipeline: object
    prepared: dict
    population: object
    probability_roles: dict
    statistics_descriptor: dict
    phase_source: dict
    case_id: str
    opencv_ipp: dict


def frame_population(reader, prepared, *, probability_roles, phase_source, case_id, labels):
    """One prepared frame's admitted gas population, from its stored probabilities.

    ``prepared`` is the N1 pipeline's prepared frame. The stored decision
    probabilities must name this frame's acquisition, case and phase-label
    product (``labels``). The pore volumes are the original native geometry
    weights and the maximum gas saturation is 1 - Swi.
    """
    from .n1_arithmetic import main_partition
    from .overflow_steering import gas_population
    from ..phase.probability_storage import read_product, same_metadata
    frame, response = prepared["frame"], prepared["response"]
    stored = read_product(probability_roles, reader=reader, source=phase_source,
                          case_id=case_id, labels=frame.phase_labels)
    if not same_metadata(stored["receipt"]["labels"], labels):
        raise ValueError("Stored gas probabilities identify another original label product")
    partition = main_partition(response)
    keep = (frame.mass_eligible_mask & frame.known_physical_domain
            & (partition["aq_known"] | partition["gas_known"]))
    geometry = response["analysis"].geometry
    geometry._prepare_cached_voxel_volume(list(frame.domain.shape))
    weights = np.asarray(geometry.cached_voxel_volume)
    if weights.dtype != np.float64:
        raise TypeError("Original native integration weights must remain float64")
    # The exact original Geometry weights are reused, not independently
    # recomputed as area*depth*porosity in another multiplication order.
    volume = np.broadcast_to(weights, frame.domain.shape)
    swi = np.broadcast_to(np.asarray(response["swi"], np.float64), frame.domain.shape)
    return gas_population(stored["arrays"], expert_knowledge_of(frame), keep, volume, swi)


def prepare_frame(pipeline, mass_spec, *, probability_roles, statistics_descriptor,
                  phase_source, case_id):
    """Replay unchanged source/optics/physical guards before deriving any F data.

    Both the original summary and the exact decision-used probability product
    must reproduce their already-completed P1 identities. No RAW decoder,
    phase inference, new optical fit or geometry estimator is called here.
    """
    from darsia._opencv_runtime import scientific_thread, completed_frame
    from .n1_pipeline import N1Pipeline
    from ..phase.probability_storage import same_metadata
    from ..pipeline.refit_runner import read_statistics

    if (type(pipeline) is not N1Pipeline or not pipeline.statistics_only
            or pipeline.frame_loading_only or pipeline.math.precision != "float32"):
        raise ValueError("The original float32 statistics-only native context is required")
    if (not isinstance(phase_source, dict) or not isinstance(case_id, str) or not case_id
            or mass_spec.get("case_id") != case_id
            or phase_source.get("run") != pipeline.config["run"]
            or not same_metadata(mass_spec["source_binding"], phase_source.get("source_binding"))):
        raise ValueError("Exact original selected acquisition and mass-frame binding required")
    expected = read_statistics(pipeline.reader, statistics_descriptor)
    before = scientific_thread("overflow-statistics:original-preparation:before")
    prepared = pipeline._prepare_statistics(mass_spec)
    observation = completed_frame(before, "overflow-statistics:original-preparation:after")
    if not same_metadata(prepared["statistics"], expected):
        raise ValueError("Original native summary differs from the completed P1 statistics")
    population = frame_population(pipeline.reader, prepared, probability_roles=probability_roles,
        phase_source=phase_source, case_id=case_id, labels=mass_spec["labels"])
    return SteeringFrame(pipeline, prepared, population, deepcopy(probability_roles),
                          deepcopy(statistics_descriptor), deepcopy(phase_source), case_id,
                          observation)


def _integrate(frame, field, mask):
    """Original Geometry.integrate multiply and repeated axis-zero reductions."""
    geometry = frame.prepared["response"]["analysis"].geometry
    if field.shape != mask.shape or mask.dtype != bool:
        raise ValueError("Exact original native field and boolean integration support required")
    geometry._prepare_cached_voxel_volume(list(field.shape))
    if np.asarray(geometry.cached_voxel_volume).dtype != np.float64:
        raise TypeError("Original native integration weights must remain float64")
    return float(geometry.integrate(np.where(mask, field, 0.)))


def baseline_volume(frame):
    """The original phase-labelled gas volume; independent of gain and beta."""
    if not isinstance(frame, SteeringFrame):
        raise TypeError("An original checked steering frame is required")
    return _integrate(frame, frame.population.baseline_saturation, frame.population.eligible)


def derive_statistics(frame, *, level, uncapped_level):
    """Change only quantified main gas/water terms and its aqueous histogram.

    Original E/U_rest estimates, labels, source bindings, selected frames and
    dilution anchors stay literal. F is restricted to admitted main physical
    support; it does not invent physics in excluded/unknown pixels.
    """
    if (not isinstance(frame, SteeringFrame) or type(level) not in (int, float)
            or type(uncapped_level) not in (int, float)
            or not np.isfinite([level, uncapped_level]).all()
            or not 0 <= level <= 1 or uncapped_level < level
            or level != min(1., uncapped_level)):
        raise ValueError("An explicitly bound physical run level and its uncapped value are required")
    volume = baseline_volume(frame)
    return _derive_target_statistics(frame, target_volume_m3=level*volume,
        target_metadata=dict(level=float(level), uncapped_level=float(uncapped_level)),
        level_capped_volume_m3=(uncapped_level-level)*volume)


def derive_phase_split_statistics(frame, target):
    """Apply one measured/explicitly estimated volume without fitting total mass."""
    from .overflow_phase_split import TARGET_FORMAT, POLICY
    if (not isinstance(frame, SteeringFrame) or not isinstance(target, dict)
            or target.get("format") != TARGET_FORMAT or target.get("policy") != POLICY
            or target.get("kind") not in {"own_measurement", "pressure_matched_estimate"}
            or target.get("run") != frame.prepared["frame"].run_id
            or target.get("elapsed_h") != frame.prepared["frame"].elapsed_h):
        raise ValueError("The phase-split target must belong to this exact run and image time")
    volume = target.get("target_volume_m3")
    if type(volume) not in (int, float) or not np.isfinite(volume) or volume < 0:
        raise ValueError("A finite nonnegative phase-split gas volume is required")
    return _derive_target_statistics(frame, target_volume_m3=volume,
        target_metadata=dict(phase_split_target=deepcopy(target)), level_capped_volume_m3=0.)


def _derive_target_statistics(frame, *, target_volume_m3, target_metadata, level_capped_volume_m3):
    from .n1_arithmetic import main_partition
    from .n1_statistics import _native_histogram
    from .overflow_steering import apply_gas_fields, redistribute_gas

    pipeline, response = frame.pipeline, frame.prepared["response"]
    original = frame.prepared["statistics"]
    population = frame.population
    volume = baseline_volume(frame)
    if "phase_split_target" in target_metadata:
        from .overflow_phase_split import distribute_target
        distribution = distribute_target(population, target_volume_m3)
    else:
        distribution = redistribute_gas(population, target_volume_m3=target_volume_m3)
    distribution.update(actual_volume_m3=_integrate(
        frame, distribution["saturation_free_gas"], population.eligible),
        baseline_volume_m3=volume, **deepcopy(target_metadata),
        level_capped_volume_m3=level_capped_volume_m3)
    distribution["float32_volume_residual_m3"] = distribution["actual_volume_m3"]-distribution["target_volume_m3"]
    partition = main_partition(response)
    math, physics = pipeline.math, response["physics"]
    # Literal original undiluted baseline, never a fitted F gas parameter.
    maps = math.selected_n1_fields(response["prepared"].aqueous_relative,
        gain=float(pipeline.config["summary"]["baseline_gain"]), factor=1.,
        swi=response["swi"], aqueous_known=partition["aq_known"],
        gas_known=partition["gas_known"], solubility=physics.solubility_co2,
        gas_density=physics.density_gaseous_co2)
    sol, rho = math.pixels(physics.solubility_co2), math.pixels(physics.density_gaseous_co2)
    changed = apply_gas_fields(maps, distribution, population=population,
                               solubility=sol, gas_density=rho)
    main_aq = population.eligible & (expert_knowledge_of(response["frame"]).dissolved)
    # Retain the original histogram's physical-capacity multiplication order,
    # then account for the remaining water in originally aqueous pixels.
    capacity = partition["volume"] * np.asarray(physics.solubility_co2, np.float64) * 1000
    capacity *= (np.float32(1)-distribution["saturation_free_gas"]).astype(np.float64)
    row = deepcopy(original)
    # A derived row cannot masquerade as an unchanged P1 extraction at the
    # original ordinary-refit capacity/contract boundary.
    row["contract"] = STATISTICS_FORMAT
    row["main_aqueous_histogram"] = _native_histogram(
        response["prepared"].aqueous_relative, capacity, main_aq, "cpu")
    row["baseline_undiluted_main_mass_g"] = _integrate(frame, changed["mass_total"], population.eligible)*1000
    row["main_gas_mass_g"] = _integrate(frame, changed["mass_gas"], population.eligible)*1000
    row["main_gas_water_mass_g"] = _integrate(frame, changed["mass_aqueous"], population.phase_gas)*1000
    original_gas = _integrate(frame, maps["mass_gas"], population.eligible)*1000
    added_on_aqueous = _integrate(frame, changed["mass_gas"], main_aq)*1000
    gas = row["main_gas_mass_g"]
    report = dict(baseline_gas_mass_g=original_gas, gas_mass_g=gas,
        gas_change_g=gas-original_gas,
        gas_change_share=None if gas == 0 else (gas-original_gas)/gas,
        gas_on_non_gas_label_g=added_on_aqueous,
        gas_on_non_gas_label_share=None if gas == 0 else added_on_aqueous/gas,
        capped_volume_m3=distribution["capped_volume_m3"]+distribution["level_capped_volume_m3"])
    row["overflow_steering"] = dict(format="darsia-overflow-statistics-derivation-v1",
        original_statistics=deepcopy(frame.statistics_descriptor),
        probability_roles=deepcopy(frame.probability_roles),
        case_id=frame.case_id, **deepcopy(target_metadata),
        threshold=distribution["threshold"],
        baseline_gas_volume_m3=volume, actual_gas_volume_m3=distribution["actual_volume_m3"],
        labels_modified=False, physical_geometry_modified=False,
        excluded_estimates_modified=False)
    # The immutable operational producer adds the actual preparation descriptor
    # and row/array/source identities before a refit may admit this derived row.
    from .gas_contribution_review import capture_gas_contributions
    gas_contribution_maps = capture_gas_contributions(frame, maps["mass_gas"], changed["mass_gas"])
    return dict(statistics=row, distribution=distribution, report=report,
                gas_contribution_maps=gas_contribution_maps,
                original_statistics=deepcopy(frame.statistics_descriptor),
                probability_roles=deepcopy(frame.probability_roles),
                phase_source=deepcopy(frame.phase_source), case_id=frame.case_id,
                opencv_ipp=deepcopy(frame.opencv_ipp), raw_decoded=False,
                phase_inference_run=False, optical_model_training_run=False,
                source_admission_or_job_created=False)
