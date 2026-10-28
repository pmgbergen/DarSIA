"""Context-free measured, column-estimated, and unknown pixel masses."""

from __future__ import annotations

import numpy as np

from ..geometry.pixels import column_co2_donors


def _mask(value, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
        raise TypeError(name + " must be an existing boolean array")
    if value.shape != shape:
        raise ValueError(name + " must share the exact pixel grid")
    return value


def _scalar(value, name, *, positive, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(name + " must be an explicit finite scalar")
    number = float(value)
    if (
        not np.isfinite(number)
        or number < 0
        or (positive and number == 0)
        or (upper is not None and number > upper)
    ):
        raise ValueError(name + " is outside its physical range")
    return number


def _sum_axis_zero(value):
    for _ in range(value.ndim):
        value = np.sum(value, axis=0, dtype=np.float64)
    return float(value)


def hidden_mass_by_adjacency(*, hidden_mask, adjacency, elapsed_h,
                             round3_capacity_g=None):
    """Apply N16 to an explicit original capacity, without creating a formula.

    The hidden flag is always retained. A non-adjacent plume has U=0 by the
    named physical rule even if no capacity was calculated. An adjacent plume
    requires a supplied finite nonnegative capacity map covering every hidden
    pixel; no old scalar, phase inference or geometry maximum substitutes it.
    The production caller must establish that map's original round-3 binding.
    """
    hidden = np.asarray(hidden_mask)
    if hidden.dtype != np.bool_ or hidden.ndim != 2 or not all(hidden.shape):
        raise ValueError("N16 hidden support must be an explicit native boolean mask")
    if (not isinstance(adjacency, dict) or type(adjacency.get("co2_adjacent")) is not bool
            or adjacency.get("distance_limit_mm") != 20.0):
        raise ValueError("N16 needs the original 20 mm hidden-boundary decision")
    elapsed = _scalar(elapsed_h, "Frame time", positive=False)
    touching = _scalar(adjacency.get("touching_length_mm"), "Touching length", positive=False)
    mass = np.zeros(hidden.shape, np.float64)
    if adjacency["co2_adjacent"]:
        capacity = np.asarray(round3_capacity_g)
        if (capacity.shape != hidden.shape or capacity.dtype.kind not in "fiu"
                or not np.isfinite(capacity[hidden]).all() or np.any(capacity[hidden] < 0)):
            raise ValueError("Adjacent hidden sand requires the explicitly bound round-3 capacity map")
        mass[hidden] = capacity[hidden]
        status = "upper bound from original round-3 capacity"
    else:
        if touching != 0:
            raise ValueError("A zero-by-adjacency decision cannot have a touching length")
        status = "zero by adjacency rule"
    total = _sum_axis_zero(mass)
    return dict(hidden_mask=hidden.copy(), U_per_pixel_g=mass, U_g=total,
                U_layer=None if total == 0 else mass,
                receipt=dict(adjacency, status=status, elapsed_h=elapsed,
                             touching_length_mm=touching),
                capacity_formula_created=False)


def evaluate_pixels(
    *,
    domain,
    visible,
    geometry_known,
    colour_available,
    phase_labels,
    relative_aqueous,
    swi,
    solubility,
    gas_density,
    transport_weight_l,
    y_m,
    max_distance_m,
    gain,
    frozen_dilution_factor,
    force_estimated,
    unknown_without_capacity,
    pixel_math,
):
    """Evaluate one physical pixel rule without loading or fitting any model.

    All masks and the immutable uint8 phase labels share one grid. Optical
    relative response, water saturation, solubility and gas density already
    have the provider's declared pixel precision. ``transport_weight_l`` is
    float64 pore volume in litres per target pixel, based on target measured
    depth and porosity; this function never transports a donor's geometry.

    Visible aqueous CO2 with a colour path uses the unchanged N1 provider
    with explicit ``gain`` and ``frozen_dilution_factor``. Other aqueous CO2
    may borrow the actual concentration of the nearest measured visible CO2
    above in its column, within ``max_distance_m``. Estimates and forced
    estimates never donate, and borrowed concentration is not calibrated a
    second time. ``force_estimated`` excludes a pixel's own optical response.

    Gas retains the provider's target-facies one-minus-Swi saturation and
    local thermodynamics. Visible, unforced finite-Swi gas is measured; other
    finite-Swi gas is explicitly estimated. No donor gas saturation is copied.

    Unresolved positive phase and in-domain phase 255 have unknown
    concentration. Capacity uses local solubility for aqueous phase and the
    larger of local solubility/gas density for gas or unknown phase. Unknown
    geometry makes aggregate U None, never an invented zero or maximum
    geometry. ``unknown_without_capacity`` explicitly identifies a subset of
    in-domain frozen phase 255 for which no CO2 or capacity is assigned. It
    remains unknown with NaN CO2 fields, including when geometry is undefined;
    this exception never applies to other unknown pixels. The caller derives
    this mask from its physical policy, without changing phase labels.
    Frozen phase-zero water remains zero mass. Density maps keep
    pixel precision; weights and repeated-axis-zero mass reductions are
    float64. Litres-weighted maps can round differently from an older m3
    integration followed by a final multiplication by 1000; no equivalence
    tolerance or replacement of the actual result is applied here.
    """
    if (
        not isinstance(domain, np.ndarray)
        or domain.dtype != np.bool_
        or domain.ndim != 2
        or not all(domain.shape)
    ):
        raise TypeError("Domain must be a nonempty two-dimensional boolean array")
    shape = domain.shape
    visible = _mask(visible, shape, "Visibility")
    geometry_known = _mask(geometry_known, shape, "Known geometry")
    colour_available = _mask(colour_available, shape, "Colour availability")
    force_estimated = _mask(force_estimated, shape, "Forced estimate")
    unknown_without_capacity = _mask(
        unknown_without_capacity, shape, "Unknown without capacity"
    )
    if (
        not isinstance(phase_labels, np.ndarray)
        or phase_labels.dtype != np.uint8
        or phase_labels.shape != shape
        or not np.isin(phase_labels, (0, 1, 2, 255)).all()
    ):
        raise TypeError("Frozen phase labels must be uint8 categories on the exact grid")
    if np.any(unknown_without_capacity & ~(domain & (phase_labels == 255))):
        raise ValueError("Unknown without capacity must be a subset of in-domain phase 255")
    dtype = np.dtype(pixel_math.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or (
        pixel_math.precision != dtype.name
    ):
        raise TypeError("The provider must declare float32 or float64 pixel precision")
    for name, value in (
        ("Relative aqueous response", relative_aqueous),
        ("Water saturation", swi),
        ("Solubility", solubility),
        ("Gas density", gas_density),
    ):
        pixel_math.require_pixels(value)
        if value.shape != shape:
            raise ValueError(name + " must share the exact pixel grid")
    if (
        not isinstance(transport_weight_l, np.ndarray)
        or transport_weight_l.dtype != np.float64
        or transport_weight_l.shape != shape
    ):
        raise TypeError("Transport weights must be float64 litres on the exact pixel grid")
    gain = _scalar(gain, "Frozen gain", positive=True)
    factor = _scalar(
        frozen_dilution_factor, "Frozen dilution factor", positive=False, upper=1
    )
    physical = domain & geometry_known
    for name, value in (("Solubility", solubility), ("Gas density", gas_density)):
        if not np.isfinite(value[physical]).all() or np.any(value[physical] < 0):
            raise ValueError(name + " must be finite and nonnegative on known geometry")
    if not np.isfinite(transport_weight_l[physical]).all() or np.any(
        transport_weight_l[physical] <= 0
    ):
        raise ValueError("Known geometry requires positive finite transport weights")
    aqueous_phase = domain & (phase_labels == 1)
    gas_phase = domain & (phase_labels == 2)
    positive = aqueous_phase | gas_phase
    phase_unknown = domain & (phase_labels == 255)
    water = domain & (phase_labels == 0)
    finite_swi = np.isfinite(swi)
    if np.any(physical & gas_phase & finite_swi & ((swi < 0) | (swi > 1))):
        raise ValueError("Finite gas water saturation must lie in [0,1]")
    own_aqueous = (
        aqueous_phase & physical & visible & colour_available & ~force_estimated
    )
    if not np.isfinite(relative_aqueous[own_aqueous]).all():
        raise ValueError("Measured aqueous pixels need a finite optical response")
    known_gas = gas_phase & physical & finite_swi
    measured_gas = known_gas & visible & ~force_estimated
    donors = own_aqueous | measured_gas
    donor_rows, column_available, donor_distance = column_co2_donors(
        donors, y_m, max_distance_m=max_distance_m
    )
    borrowed = aqueous_phase & physical & ~own_aqueous & column_available
    estimated_gas = known_gas & ~measured_gas
    estimated = borrowed | estimated_gas
    measured = water | own_aqueous | measured_gas
    unknown = (positive | phase_unknown) & ~(measured | estimated)

    native = pixel_math.selected_n1_fields(
        relative_aqueous,
        gain=gain,
        factor=factor,
        swi=swi,
        aqueous_known=own_aqueous,
        gas_known=known_gas,
        solubility=solubility,
        gas_density=gas_density,
    )
    names = ("concentration", "gas", "mass_total", "mass_aqueous", "mass_gas")
    maps = {}
    quantified = own_aqueous | known_gas
    for name in names:
        value = native[name]
        pixel_math.require_pixels(value)
        if value.shape != shape or not np.isfinite(value[quantified]).all():
            raise ValueError("Native provider returned invalid quantified fields")
        maps[name] = np.where(quantified, value, dtype.type(0))
    columns = np.arange(shape[1])[None, :]
    rows = np.maximum(donor_rows, 0)
    donor_concentration = native["concentration"][rows, columns]
    for name in ("concentration", "mass_total", "mass_aqueous"):
        maps[name][borrowed] = donor_concentration[borrowed]
    for name in ("concentration", "mass_total", "mass_aqueous", "mass_gas"):
        maps[name][unknown] = np.nan
    maps["gas"][phase_unknown | (gas_phase & ~known_gas)] = np.nan

    def mass_map(density, selection):
        result = np.zeros(shape, np.float64)
        np.multiply(
            np.asarray(density, np.float64), transport_weight_l,
            out=result, where=selection & positive,
        )
        return result

    measured_mass = mass_map(maps["mass_total"], measured)
    edge_mass = mass_map(maps["mass_total"], estimated)
    unknown_capacity = np.zeros(shape, np.float64)
    capacity_density = np.where(
        aqueous_phase, solubility, np.maximum(solubility, gas_density)
    )
    np.multiply(
        np.asarray(capacity_density, np.float64), transport_weight_l,
        out=unknown_capacity, where=unknown & geometry_known & ~unknown_without_capacity,
    )
    undefined_capacity = unknown & ~geometry_known & ~unknown_without_capacity
    unknown_capacity[undefined_capacity] = np.nan
    return dict(
        maps=dict(
            concentration_aqueous=maps["concentration"],
            saturation_free_gas=maps["gas"],
            mass_total=maps["mass_total"],
            mass_aqueous=maps["mass_aqueous"],
            mass_free_gas=maps["mass_gas"],
            phase_labels=phase_labels.copy(),
        ),
        masks=dict(
            measured=measured,
            concentration_estimated=estimated,
            concentration_unknown=unknown,
            phase_unknown=phase_unknown,
            unknown_without_capacity=unknown_without_capacity.copy(),
            geometry_unknown=domain & ~geometry_known,
            water=water,
            measured_aqueous=own_aqueous,
            measured_gas=measured_gas,
            estimated_gas=estimated_gas,
            column_estimated_aqueous=borrowed,
            donors=donors,
        ),
        donor_rows=np.where(borrowed, donor_rows, -1).astype(np.int32),
        donor_distance_m=np.where(borrowed, donor_distance, np.nan),
        per_pixel_mass_g=dict(
            measured=measured_mass, edge=edge_mass, unknown_capacity=unknown_capacity
        ),
        M_measured_g=_sum_axis_zero(measured_mass),
        M_edge_g=_sum_axis_zero(edge_mass),
        U_g=None if undefined_capacity.any() else _sum_axis_zero(unknown_capacity),
        units=dict(
            concentration_aqueous="g/L",
            saturation_free_gas="1",
            mass_total="g/L", mass_aqueous="g/L", mass_free_gas="g/L",
            transport_weight_l="L pore volume per target pixel",
            per_pixel_mass="g per target pixel", scalar_mass="g", donor_distance="m",
        ),
        audit=dict(
            working_pixel_dtype=dtype.name, transport_weight_dtype="float64",
            reduction_dtype="float64", gain=gain, frozen_dilution_factor=factor,
            phase_labels_modified=False, calibration_refitted=False,
            estimated_pixels_used_as_donors=False,
            donor_concentration_recalibrated=False,
            gas_saturation_borrowed=False,
            gas_saturation_policy="target facies one minus Swi; local thermodynamics",
            visible_unforced_finite_swi_gas_is_measured=True,
            other_finite_swi_gas_is_estimated=True,
            phase_zero_water_is_zero_mass=True,
            unknown_geometry_capacity_bounded=False,
            unknown_without_capacity_pixels=int(np.count_nonzero(unknown_without_capacity)),
            unknown_without_capacity_policy="explicit frozen phase 255; no CO2 or capacity assigned",
        ),
    )


def round3_hidden_solubility_capacity(*, hidden_mask, physical_properties_valid_mask,
                                     pixel_area_m2, depth_m, porosity, solubility):
    """Q85 A: the literal R3 volume * solubility * 1000 expression for hidden255.

    Inputs retain their original producer precision. The original R3 capacity
    operands/reduction use float64; no gas-density branch or maximum geometry
    is introduced. This creates no observed concentration or measured mass.
    """
    hidden = np.asarray(hidden_mask)
    if hidden.dtype != np.bool_ or hidden.ndim != 2 or not all(hidden.shape):
        raise ValueError("Explicit native hidden sand mask required")
    valid = _mask(physical_properties_valid_mask, hidden.shape, "Physical properties")
    fields = []
    for name, value in (("area", pixel_area_m2), ("depth", depth_m),
                        ("porosity", porosity), ("solubility", solubility)):
        value = np.asarray(value)
        if value.shape != hidden.shape or value.dtype.kind not in "fiu":
            raise ValueError("Original capacity " + name + " must share the native grid")
        if not np.isfinite(value[hidden]).all() or np.any(value[hidden] <= 0):
            raise ValueError("Positive actual hidden capacity " + name + " required")
        fields.append(np.asarray(value, np.float64))
    if np.any(hidden & ~valid) or np.any(fields[2][hidden] > 1):
        raise ValueError("Every hidden pixel requires its actual full physical geometry")
    area, depth, phi, sol = fields
    volume = np.where(valid, area * depth * phi, 0.)
    original_capacity = volume * sol * 1000
    return np.where(hidden, original_capacity, 0.).astype(np.float64)


def measured_component_maps(pixel, phase_labels, transport_weight_l):
    """Expose independent original N15 contribution maps without export recasts.

    The original measured total map is reused directly. The two component
    maps use exactly the original mass_map density*float64-litres operation
    and original measured-positive support. Native/export and stored product
    precision conversions happen later under their existing policies.
    """
    labels = np.asarray(phase_labels)
    weights = np.asarray(transport_weight_l)
    measured = pixel["masks"]["measured"] & np.isin(labels, (1, 2))
    if weights.dtype != np.float64 or weights.shape != measured.shape:
        raise ValueError("Original float64 target transport weights required")
    result = dict(total=pixel["per_pixel_mass_g"]["measured"])
    for component, field in (("aqueous", "mass_aqueous"), ("gas", "mass_free_gas")):
        contribution = np.zeros(measured.shape, np.float64)
        np.multiply(np.asarray(pixel["maps"][field], np.float64), weights,
                    out=contribution, where=measured)
        result[component] = contribution
    if any(value.dtype != np.float64 or not np.isfinite(value).all()
           or np.any(value < 0) or np.any(value[~measured] != 0) for value in result.values()):
        raise ValueError("Original independent measured contribution maps differ")
    return result
