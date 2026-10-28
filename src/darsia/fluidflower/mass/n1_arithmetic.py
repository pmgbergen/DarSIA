"""Selected native N1 fields and original float64 Geometry integration."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np
from .phase_expert_knowledge import expert_knowledge_of
from .concentration_joint import dilution_factor

FIELDS = {"c_aq_g_l", "c_aq_relative", "s_g", "phase_labels", "unquantified_phase_mask", "main_mass_eligible_mask", "mass_density_g_l", "aqueous_mass_density_g_l", "gas_mass_density_g_l"}

ANALYSIS_GAP = "approved_analysis_gap"
_MASS_SCALARS = ("total_mass_g", "main_mass_g", "aqueous_mass_g", "gas_mass_g",
                 "gas_water_mass_g", "main_aqueous_nongas_mass_g")
_INCOMPLETE_SCALARS = ("interval_lower_g", "interval_upper_g", "U_rest_g", "W_g",
                       "W_over_d", "signed_interval_residual", "delta")


def analysis_gap_scope(frame):
    """Account for an already approved mask; grant no physical/source approval."""
    mask = (frame.excluded_masks or {}).get(ANALYSIS_GAP)
    if mask is None:
        return None, None
    unknown = getattr(frame, "unknown_physical_mask", None)
    if (not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != frame.domain.shape
            or not isinstance(unknown, np.ndarray) or unknown.dtype != bool or unknown.shape != mask.shape
            or np.any(mask & ~unknown)
            or np.any(mask & ~frame.domain) or np.any(mask & frame.mass_eligible_mask)
            or np.any(mask & frame.known_physical_domain)):
        raise ValueError("Approved analysis gaps must remain excluded and physically unquantified")
    # JSON detaches the complete existing producer record, including its
    # negative scope flags; no source identity is inferred from pixel values.
    provenance = json.loads(json.dumps(frame.provenance.get(ANALYSIS_GAP),
        sort_keys=True, separators=(",", ":"), allow_nan=False))
    identity = dict(dtype=mask.dtype.str, shape=list(mask.shape),
        sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes(order="C")).hexdigest())
    if (not isinstance(provenance, dict) or provenance.get("run_id") != frame.run_id
            or provenance.get("mask_identity") != identity
            or provenance.get("meaning") != "outside-analysis-unquantified"
            or provenance.get("physical_frame_unchanged") is not True
            or provenance.get("labels_unchanged") is not True
            or not isinstance(provenance.get("approval"), dict)
            or not isinstance(provenance.get("native_mask"), dict)
            or not isinstance(provenance.get("transport"), dict)):
        raise ValueError("Approved analysis gaps require their unchanged bound producer provenance")
    if not mask.any():
        return None, None
    return mask, dict(status="quantified_subtotal_with_unquantified_remainder",
        approved_analysis_gap_pixels=int(mask.sum()),
        phase255_pixels=int((mask & (expert_knowledge_of(frame).unknown)).sum()),
        area_cm2=None, unknown_mass_g=None, whole_rig_mass_complete=False,
        area_status="unavailable_without_separately_verified_area_evidence",
        provenance=provenance)


def scoped_result(value, scope):
    """Retain internal subtotals while refusing a complete whole-rig interval."""
    if scope is None:
        return value
    result = dict(value, analysis_scope=deepcopy(scope))
    for name in _MASS_SCALARS:
        if name in result:
            result["quantified_" + name] = result[name]
    for name in _INCOMPLETE_SCALARS:
        if name in result:
            # A whole-rig injected dose is not the dose of the quantified
            # subtotal. Do not retain a seemingly usable partial-dose score.
            if name not in {"W_over_d", "signed_interval_residual", "delta"}:
                result["quantified_" + name] = result[name]
            result[name] = None
    return result


def public_analysis_scope(value):
    """Publish explicit subtotals and null unqualified whole-domain masses."""
    if value.get("analysis_scope") is None:
        return value
    result = dict(value)
    for name in _MASS_SCALARS:
        if name in result:
            result.setdefault("quantified_" + name, result[name])
            result[name] = None
    if "U_g" in result:
        result.setdefault("quantified_U_g", result["U_g"])
        result["U_g"] = None
    for name in ("scalar_result", "prediction", "pixel_domain"):
        if isinstance(result.get(name), dict):
            result[name] = public_analysis_scope(result[name])
    if isinstance(result.get("domains"), dict):
        result["domains"] = {name: public_analysis_scope(row) for name, row in result["domains"].items()}
    return result

def validate_parameters(parameters):
    """Require the frozen positive N1 gain and nonnegative dilution beta."""
    if (parameters.get("candidate") != "N1" or not np.isfinite(parameters.get("gain", 0))
            or parameters.get("gain",0) <= 0 or not np.isfinite(parameters.get("beta",0))
            or parameters.get("beta",0) < 0 or any(k in parameters for k in ("gas_scale","gas_saturation","gas_gain"))):
        raise ValueError("Frozen N1 gain/beta and an unfitted gas branch are required")

def main_partition(response):
    """Original measured main support/volume, without discarded bound products."""
    import numpy as np
    frame, physics = response["frame"], response["physics"]
    shape = frame.domain.shape
    approved_gap, scope = analysis_gap_scope(frame)
    positive = frame.domain & expert_knowledge_of(frame).co2
    if approved_gap is not None:
        positive &= ~approved_gap
    known = frame.known_physical_domain
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), shape)
    depth = np.broadcast_to(np.asarray(frame.depth_m, np.float64), shape)
    phi = np.broadcast_to(np.asarray(frame.porosity, np.float64), shape)
    volume = np.where(known, area*depth*phi, 0.)
    aq_known = known & response["quantifiable"] & (expert_knowledge_of(frame).dissolved)
    gas_known = known & np.isfinite(response["swi"]) & (expert_knowledge_of(frame).free_gas)
    sol = np.asarray(physics.solubility_co2, np.float64)
    density = np.asarray(physics.density_gaseous_co2, np.float64)
    if (not np.isfinite(sol[positive]).all() or not np.isfinite(density[positive]).all()
            or np.any(sol[positive] < 0) or np.any(density[positive] < 0)):
        raise ValueError("Native thermodynamic capacity must be finite across the locked phase domain")
    result = dict(volume=volume, aq_known=aq_known, gas_known=gas_known,
                  affected_pore_volume_m3=float(np.sum(volume[positive])))
    return scoped_result(result, scope)

def evaluate_main(response, anchor, *, parameters, pixel_math, calibration_sha256, domain_masks, kept_fields=FIELDS,
                  gas_distribution=None):
    """Evaluate original typed N1 fields and native geometry reductions.

    ``gas_distribution`` (its ``population`` and ``distribution``) is an
    image's overflow-governed free gas. It replaces only the gas saturation
    of the unchanged N1 fields, before every reduction; None keeps them.
    """
    math = pixel_math
    validate_parameters(parameters)
    if response.get("precision") != math.precision or set(domain_masks) != {"full_rig", "box1", "box2"}:
        raise ValueError("Exact working precision and all three native domains are required")
    if set(kept_fields)-FIELDS:
        raise ValueError("Unknown retained field")
    digest = calibration_sha256
    frame = response["frame"]
    approved_gap, scope = analysis_gap_scope(frame)
    for name, mask in domain_masks.items():
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != frame.domain.shape:
            raise ValueError("Frozen domain mask differs from the native full-resolution grid")
        if name == "full_rig" and not np.array_equal(mask, frame.domain):
            raise ValueError("Full-rig domain must be the exact original frozen phase domain")
    analysis, physics = response["analysis"], response["physics"]
    partition = main_partition(response)  # original finite/positive physics guards
    factor = dilution_factor(frame.elapsed_h, partition["affected_pore_volume_m3"],
                                      anchor, parameters.get("beta", 0.))
    maps = math.selected_n1_fields(response["prepared"].aqueous_relative,
        gain=parameters["gain"], factor=factor, swi=response["swi"],
        aqueous_known=partition["aq_known"], gas_known=partition["gas_known"],
        solubility=physics.solubility_co2, gas_density=physics.density_gaseous_co2)
    if gas_distribution is not None:
        from .overflow_steering import apply_gas_fields
        maps = apply_gas_fields(maps, gas_distribution["distribution"],
            population=gas_distribution["population"], solubility=math.pixels(physics.solubility_co2),
            gas_density=math.pixels(physics.density_gaseous_co2))
    eligible = frame.mass_eligible_mask & frame.known_physical_domain
    def integrate(value, mask):
        # Original Geometry.integrate itself remains unchanged; its weights
        # are genuinely float64 before multiply and repeated axis-zero sums.
        """Use original float64 native geometry weights and axis order."""
        analysis.geometry._prepare_cached_voxel_volume(list(value.shape[:analysis.geometry.space_dim]))
        if np.asarray(analysis.geometry.cached_voxel_volume).dtype != np.float64:
            raise TypeError("Geometry weights must be float64 before reduction")
        return float(analysis.geometry.integrate(np.where(mask, value, 0.)))*1000
    def masses(mask):
        """Report separate total, aqueous and gas native integrals."""
        return dict(total_mass_g=integrate(maps["mass_total"], mask),
                    aqueous_mass_g=integrate(maps["mass_aqueous"], mask),
                    gas_mass_g=integrate(maps["mass_gas"], mask))
    domains = {name: masses(eligible & mask) for name, mask in domain_masks.items()}
    excluded, covered = {}, np.zeros(frame.domain.shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        covered |= mask
        excluded[name] = (dict(total_mass_g=None, aqueous_mass_g=None, gas_mass_g=None,
            pixel_count=scope["approved_analysis_gap_pixels"], area_cm2=None,
            unknown_mass_g=None, meaning="outside-analysis-unquantified")
            if name == ANALYSIS_GAP and scope is not None else masses(mask))
    excluded["other_not_mass_eligible"] = masses(frame.domain & ~frame.mass_eligible_mask & ~covered)
    fields = {}
    if "c_aq_g_l" in kept_fields or "unquantified_phase_mask" in kept_fields or "s_g" in kept_fields:
        unknown = frame.domain & expert_knowledge_of(frame).co2 & ~(
            partition["aq_known"] | partition["gas_known"])
        if approved_gap is not None:
            unknown |= approved_gap
        if "c_aq_g_l" in kept_fields:
            concentration = maps["concentration"]
            concentration[unknown] = np.nan
            fields["c_aq_g_l"] = concentration
        if "unquantified_phase_mask" in kept_fields:
            fields["unquantified_phase_mask"] = unknown
        if "s_g" in kept_fields:
            gas = maps["gas"]
            gas[unknown & (expert_knowledge_of(frame).free_gas)] = np.nan
            if approved_gap is not None:
                gas[approved_gap] = np.nan
            fields["s_g"] = gas
    if "c_aq_relative" in kept_fields: fields["c_aq_relative"] = maps["aqueous"]
    if "phase_labels" in kept_fields: fields["phase_labels"] = frame.phase_labels.copy()
    if "main_mass_eligible_mask" in kept_fields: fields["main_mass_eligible_mask"] = eligible
    for name, value in (("mass_density_g_l", "mass_total"),
                        ("aqueous_mass_density_g_l", "mass_aqueous"),
                        ("gas_mass_density_g_l", "mass_gas")):
        if name in kept_fields: fields[name] = maps[value]
    if approved_gap is not None:
        fields[ANALYSIS_GAP + "_mask"] = approved_gap.copy()
        for name in ("c_aq_relative", "mass_density_g_l", "aqueous_mass_density_g_l", "gas_mass_density_g_l"):
            if name in fields:
                if name != "c_aq_relative":
                    fields["quantified_" + name] = fields[name]
                fields[name] = fields[name].copy()
                fields[name][approved_gap] = np.nan
        domains = {name: scoped_result(row, dict(scope,
            approved_analysis_gap_pixels=int((approved_gap & domain_masks[name]).sum()),
            phase255_pixels=int((approved_gap & domain_masks[name] & (expert_knowledge_of(frame).unknown)).sum()), domain=name))
            for name, row in domains.items()}
    result = dict(contract="darsia-n1-fields-v1", parameters=parameters,
        calibration_sha256=digest, scalar_reference="native_final_calibration_fields", run=frame.run_id,
        frame_index=frame.frame_index, elapsed_h=frame.elapsed_h, dilution_factor=factor,
        **domains["full_rig"], domains=domains, excluded_mass=excluded,
        affected_pore_volume_m3=partition["affected_pore_volume_m3"], fields=fields,
        precision=math.precision, pixel_backend=math.facts()["backend"],
        pixel_backend_facts=dict(math.facts()),
        input_storage_dtype="float32-log-RGB/frozen-clearwater", working_pixel_dtype=math.precision,
        parameter_dtype="float64", reduction_dtype="float64",
        scientific_eligibility=False, production_permission_granted=False)
    return scoped_result(result, scope)

def main_products(result, pixel_math):
    """Original retained export/mask/units, with truthful chosen mass dtype."""
    import numpy as np
    if result.get("precision") != pixel_math.precision:
        raise ValueError("Products differ from the declared working precision")
    fields = result["fields"]
    required = {"c_aq_g_l", "s_g", "phase_labels", "main_mass_eligible_mask",
                "mass_density_g_l", "aqueous_mass_density_g_l", "gas_mass_density_g_l"}
    if not required.issubset(fields):
        raise ValueError("Every retained native §3 product and the main mask must be present")
    mask = fields["main_mass_eligible_mask"]
    approved_gap = fields.get(ANALYSIS_GAP + "_mask")
    if not isinstance(mask, np.ndarray) or mask.dtype != bool:
        raise ValueError("Main product mask must retain the original boolean grid")
    products = dict(concentration_aqueous=fields["c_aq_g_l"], saturation_free_gas=fields["s_g"],
                    phase_labels=fields["phase_labels"])
    for name, native in (("mass_total", "mass_density_g_l"),
                         ("mass_aqueous", "aqueous_mass_density_g_l"),
                         ("mass_free_gas", "gas_mass_density_g_l")):
        value = fields[native]
        if value.dtype != pixel_math.dtype or value.shape != mask.shape:
            raise ValueError("Retained mass map must keep chosen precision and exact grid")
        products[name] = np.where(mask, value, 0.)
        if approved_gap is not None:
            products[name][approved_gap] = np.nan
    if approved_gap is not None:
        products[ANALYSIS_GAP + "_mask"] = approved_gap.copy()
    exported = dict(products=products, units={"concentration_aqueous": "g/L", "saturation_free_gas": "1",
        "mass_total": "g/L", "mass_aqueous": "g/L", "mass_free_gas": "g/L", "phase_labels": "category"},
        required_transport_weight_units="L pore volume per pixel", native_mass_scalar_units="g",
        main_mask_applied=True, raw_native_fields_retained=True, precision=pixel_math.precision,
        pixel_backend=result["pixel_backend"], scientific_eligibility=False, production_permission_granted=False)
    if approved_gap is not None:
        exported["quantified_products"] = {name: np.where(mask, fields["quantified_" + native], 0.)
            for name, native in (("mass_total", "mass_density_g_l"),
                                 ("mass_aqueous", "aqueous_mass_density_g_l"),
                                 ("mass_free_gas", "gas_mass_density_g_l"))}
    return scoped_result(exported, result.get("analysis_scope"))
