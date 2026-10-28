"""Frozen 230e2349 summary for the independent exact-output memory oracle."""

def summarize_round3_response(response, geometry, *, own_colour, photo_support,
        no_outflow_evidence, light_epoch, source_binding=None, distances_m=None,
        backend="cpu", scope="frozen_evaluation", pixel_math, summary_settings):
    """Extract native main and correlated excluded estimates once per frame."""
    frame, physics = response["frame"], response["physics"]
    run=frame.run_id
    partition=physical_partition(response,max_depth_m=summary_settings["max_depth_m"],max_porosity=summary_settings["max_porosity"])
    parameters=dict(candidate="N1",gain=float(summary_settings["baseline_gain"]),beta=0.)
    anchor=dict(elapsed_h=float(summary_settings.get("anchor_target_h",2.25)),affected_pore_volume_m3=max(partition["affected_pore_volume_m3"],1e-20))
    main=baseline_main(response,parameters,anchor,pixel_math)
    shape = frame.domain.shape
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), shape)
    sol, density = partition["solubility"], partition["density"]
    volume = np.where(geometry.valid, area*geometry.depth_m*geometry.porosity, 0.)
    capacity = volume*sol*1000
    missing = partition["missing"]
    is_gas = frame.phase_labels == 2
    gas_available = missing & is_gas & geometry.valid & np.isfinite(geometry.swi)
    fixed_gas = volume*density*(1-geometry.swi)*1000
    fixed_water = volume*sol*geometry.swi*1000
    gas_constant = np.where(gas_available, fixed_gas+fixed_water, 0.)
    own_fraction = np.asarray(own_colour["aqueous_relative"],np.float64)
    own_available = missing & ~is_gas & own_colour["available"] & np.asarray(photo_support, bool) & geometry.valid
    main_positive = partition["main_aq"] | partition["main_gas"]
    donor, column_available, column_distance = nearest_main_above(main_positive, geometry.y_m, max_distance_m=summary_settings["max_column_distance_m"])
    cols = np.arange(shape[1])[None, :]
    donor = np.maximum(donor, 0)
    source_gas = partition["main_gas"][donor, cols]
    source_relative = response["prepared"].aqueous_relative[donor,cols].astype(np.float64)
    source_sol = sol[donor,cols]
    solubility_ratio = np.divide(source_sol,sol,out=np.zeros(shape),where=sol>0)
    column_fraction = source_relative
    column_available &= missing & ~is_gas & geometry.valid & np.isfinite(source_relative) & np.isfinite(solubility_ratio)
    source_gas &= column_available
    pair_mask = missing & ~is_gas & (own_available | column_available)
    own_fraction = np.where(own_available, own_fraction, 0.)
    column_fraction = np.where(column_available, column_fraction, 0.)
    neither = missing & ~(gas_available | own_available | column_available)
    undefined = neither & ~geometry.valid
    urest = volume* np.where(is_gas, np.maximum(sol, density), sol)*1000
    parts = _parts(frame, missing)
    part_records = {}
    distance_arrays = {name: np.broadcast_to(np.asarray(value, np.float64), shape)
        for name, value in (distances_m or {}).items()}
    for name, mask in parts.items():
        labels = {}
        for label in np.unique(frame.facies[mask]):
            selection = mask & (frame.facies == label)
            labels[str(int(label))] = dict(co2_area_cm2=float(area[selection].sum()*10000),
                without_main_geometry_area_cm2=float(area[selection & ~frame.known_physical_domain].sum()*10000),
                estimate_geometry_undefined_area_cm2=float(area[selection & ~geometry.valid].sum()*10000),
                capacity_g=None if (selection & ~geometry.valid).any() else float(urest[selection].sum()),
                own_colour_histogram=_native_histogram(own_fraction,capacity,selection & own_available,backend),
                fixed_gas_mass_g=float(fixed_gas[selection & gas_available].sum()),
                fixed_gas_water_mass_g=float(fixed_water[selection & gas_available].sum()))
        part_records[name] = dict(by_facies=labels, co2_area_cm2=float(area[mask].sum()*10000),
            without_main_geometry_area_cm2=float(area[mask & ~frame.known_physical_domain].sum()*10000),
            geometry_extended_area_cm2=float(area[mask & geometry.valid & ~frame.known_physical_domain].sum()*10000),
            depth_nearest_outside_hull_area_cm2=float(area[mask & geometry.depth_nearest_outside_hull].sum()*10000),
            porosity_column_extended_area_cm2=float(area[mask & geometry.porosity_from_column].sum()*10000),
            capacity_g=None if (mask & ~geometry.valid).any() else float(urest[mask].sum()),
            own_colour_histogram=_native_histogram(own_fraction,capacity,mask & own_available,backend),
            fixed_gas_mass_g=float(fixed_gas[mask & gas_available].sum()),
            fixed_gas_water_mass_g=float(fixed_water[mask & gas_available].sum()),
            urest_g=None if (mask & undefined).any() else float(urest[mask & neither].sum()),
            median_distances_mm={key: None if (value := _weighted_median(array[mask],area[mask])) is None
                else value*1000 for key,array in distance_arrays.items()})
    row = dict(contract=summary_settings.get("contract", CONTRACT), run=run, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        light_epoch=light_epoch, injected_mass_g=frame.nominal_injected_mass_g,
        source_binding=dict(source_binding or frame.provenance), geometry_source_binding=geometry.source_binding,
        no_outflow_evidence=dict(no_outflow_evidence), scope=scope,
        main_aqueous_histogram=_native_histogram(response["prepared"].aqueous_relative,
            partition["volume"]*sol*1000,partition["main_aq"],backend),
        baseline_undiluted_main_mass_g=main["total_mass_g"],main_gas_mass_g=main["gas_mass_g"],
        main_gas_water_mass_g=main["gas_water_mass_g"], affected_pore_volume_m3=partition["affected_pore_volume_m3"],
        estimate_fixed_gas_mass_g=float(fixed_gas[gas_available].sum()),
        estimate_fixed_gas_water_mass_g=float(fixed_water[gas_available].sum()),
        estimate_gas_constant_g=float(gas_constant[gas_available].sum()),
        urest_g=None if undefined.any() else float(urest[neither].sum()),
        estimate_geometry_undefined_co2_pixels=int(undefined.sum()),
        estimate_geometry_extended_area_cm2=float(area[missing & geometry.valid & ~frame.known_physical_domain].sum()*10000),
        excluded_parts=part_records, own_colour_audit=own_colour["audit"],
        own_estimate_pixels=int(own_available.sum()), column_estimate_pixels=int(column_available.sum()),
        both_estimates_pixels=int((own_available & column_available).sum()),
        fixed_gas_estimate_pixels=int(gas_available.sum()), neither_estimate_pixels=int(neither.sum()),
        column_distance_median_mm=None if not column_available.any() else float(np.nanmedian(column_distance[column_available])*1000),
        main_geometry_modified=False, main_labels_modified=False, gas_branch_fitted=False,
        paired_estimates=_pairs(own_fraction,column_fraction,solubility_ratio,source_gas,
            own_available,column_available,capacity,pair_mask,backend))
    return row
