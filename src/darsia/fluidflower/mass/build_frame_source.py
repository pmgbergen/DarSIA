"""Collect source-bound physical frames from the original RAW phase producers.

The caller supplies the measured rig/photo/phase inputs and state. This adapter
does not fit them, reuse old physical corrections, or launch a production job.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .build_native_bridge import _fresh
from .native_optical_source import _bound, _same
from .n1_models import FrameData
from .n1_pipeline import _atmosphere, _write_json, _write_npz
from .pooled_optical_paths import recipe_from_document, _source_contract
from .production_precision import PixelMath


class _ScientificInputs:
    """Retain bridge late checks while using its original array reader."""

    def __init__(self, bound_inputs):
        self.bound = bound_inputs

    def path(self, value):
        return self.bound.path(value)

    def json(self, value):
        return self.bound.json(value)

    def array(self, value):
        self.bound.path(value)
        result = self.bound.reader.array(value)
        self.bound.path(value)
        return result


def _roles(context, reader):
    """Read the actual adopted full CSV and join saved labels to its facies."""
    from ..geometry_raster_sources import _full_csv

    adopted = reader.json(context.provenance["adopted_revision"])
    configuration = adopted["configuration"]
    table = _bound(adopted["artifacts"]["source_mode_files"]["facies_properties"])
    columns, records, ids = _full_csv(reader.path(table), strict=True)
    names = {name.strip().casefold(): name for name in columns}
    roles = [dict(facies_id=value, name=row[names["name"]],
                  analysis_role=row[names["analysis_role"]])
             for value, row in zip(ids, records)]
    if configuration["grouping_mode"] != "shared_facies":
        raise ValueError("The original pooled paths require the adopted shared facies")
    mapping, enabled = {}, []
    for row in configuration["regions"]:
        label, facies = row["label_id"], row["facies_id"]
        if type(label) is not int or type(facies) is not int or label in mapping:
            raise ValueError("Adopted labels require unique explicit integer facies")
        mapping[label] = facies
        if type(row.get("enabled", True)) is not bool:
            raise ValueError("Adopted region activation must retain its explicit boolean type")
        if row.get("enabled", True):
            enabled.append(label)
    from ..painted_analysis_exclusions import configuration_exclusion
    exclusion = configuration_exclusion(configuration, context.run, labels=context.labels)
    mapped = np.ones(context.labels.shape, bool) if exclusion is None else ~exclusion
    for label in np.unique(context.labels[mapped]):
        if int(label) not in mapping or np.any(
                context.facies[mapped & (context.labels == label)] != mapping[int(label)]):
            raise ValueError("Saved Build labels/facies differ from the actual adoption")
    declared = {row["facies_id"] for row in roles}
    if not set(map(int, np.unique(context.facies[mapped]))).issubset(declared):
        raise ValueError("Saved Build facies are absent from the adopted table")
    porous = [row["facies_id"] for row in roles if row["analysis_role"] == "porous_medium"]
    source_domain = mapped & np.isin(context.facies, porous) & np.isin(context.labels, enabled)
    if not source_domain.any():
        raise ValueError("The actual Build has no adopted porous physical support")
    return roles, source_domain, table


class BuildFrameSource:
    """One explicit run's phase, clearwater and source-bound numeric Build fields.

    Configuration is the DATA accepted by BuildRawPhaseCollector. The separate
    state document binds measured atmosphere/chemistry to the Build protocols.
    Missing measured geometry or phase inputs remain required producer inputs.
    This producer establishes identities and numeric transport, not independent
    geometry/scale qualification. Its frames retain empty mass eligibility;
    the original pooled-path sampler refuses this unqualified context.
    """

    def __init__(self, bridge, configuration, clearwater, state, optical_recipe, output):
        from ..pipeline.build_raw_phase import BuildRawPhaseCollector

        self.bridge, self.reader = bridge, _ScientificInputs(bridge.reader)
        self.math = PixelMath("float32")
        self.configuration_binding = deepcopy(_bound(configuration))
        self.state_binding = deepcopy(_bound(state))
        self.recipe_binding = deepcopy(_bound(optical_recipe))
        config = self.reader.json(configuration)
        self.state = self.reader.json(state)
        self.recipe = recipe_from_document(self.reader.json(optical_recipe))
        self.recipe.require_inputs()
        self.document = self.recipe.document
        self.run = bridge.context.run
        proof = self.reader.json(bridge.proof)
        selected = self.document["runs"].get(self.run)
        if selected is None or config["run"] != self.run:
            raise ValueError("The collector belongs to another selected optical run")
        expected = dict(adopted_revision=bridge.context.provenance["adopted_revision"],
                        build_context=bridge.context.provenance["operation_result"],
                        phase_configuration=config["phase"]["configuration"],
                        native_configuration=bridge.context.provenance["native_configuration"])
        if not _same(self.document["adopted_revision"], expected["adopted_revision"]):
            raise ValueError("The optical recipe differs from the actual Build/phase/native inputs")
        for role in ("build_context", "phase_configuration", "native_configuration"):
            self._run_input(self.document[role], role, expected[role])
        self.inputs = expected
        if (not _same(selected["baseline"], bridge.context.provenance["raw_acquisition"])
                or not _same(selected["clearwater"], _bound(clearwater))
                or not _same(selected["profile"], proof["photometry_source"])):
            raise ValueError("The optical RAW reference selections differ from their actual producers")
        self.roles, self.source_domain, self.table = _roles(bridge.context, self.reader)
        if not _same(sorted(self.roles, key=lambda row: row["facies_id"]),
                     sorted(self.document["roles"], key=lambda row: row["facies_id"])):
            raise ValueError("The optical roles differ from the actual adopted full facies table")
        self._state_inputs()
        frames = selected["frames"]
        actual = config["frames"]
        if (len(frames) != len(actual) or any(
                any(not _same(want[key], got[key]) for key in ("case_id", "frame_index", "elapsed_h"))
                or not _same(want["source"], got["raw_acquisition"])
                for want, got in zip(frames, actual))):
            raise ValueError("The actual chronological selection differs from the optical recipe")
        if set(self.states) != {row["case_id"] for row in frames}:
            raise ValueError("Measured state must cover exactly the selected optical frames")
        self.output = _fresh(output, bridge.forbidden)
        self.clear = bridge.frame(clearwater, self.output / "clearwater")
        self.clear_receipt = self.reader.json(self.clear.receipt)
        self.clear_rgb, self.clear_valid = self._centre(self.clear)
        self.phase = BuildRawPhaseCollector(bridge, config, self.output / "phase")
        self._iterator = iter(self.phase)
        self._closed = False

    def _run_input(self, bound, role, expected):
        """Admit a shared exact input or an explicit run member of a bound bundle."""
        if _same(bound, expected):
            self.reader.path(_bound(bound))
            return
        bundle = self.reader.json(_bound(bound))
        if (set(bundle) != {"format", "role", "runs"}
                or bundle["format"] != "darsia-pooled-optical-run-inputs-v1"
                or bundle["role"] != role or not isinstance(bundle["runs"], dict)
                or set(bundle["runs"]) != set(self.document["fit_runs"])):
            raise ValueError("The optical input bundle differs from its exact role/run cohort")
        for descriptor in bundle["runs"].values():
            _bound(descriptor)
        if not _same(bundle["runs"][self.run], expected):
            raise ValueError("The optical input bundle differs from this run's actual producer")
        self.reader.path(expected)

    def _state_inputs(self):
        """Require the measured state source identities of this original Build."""
        value = self.state
        if (value.get("format") != "darsia-build-frame-state-v1"
                or value.get("run") != self.run
                or set(value) != {"format", "run", "recipe", "protocol_inputs", "frames"}):
            raise ValueError("Explicit source-bound measured frame state is required")
        request = self.reader.json(self.bridge.context.provenance["raw_build_request"])
        protocols = [row for row in request["protocol_inputs"]
                     if row["role"] in {"pressure_temperature", "injection"}]
        if (not protocols or not any(row["role"] == "pressure_temperature" for row in protocols)
                or not _same(value["protocol_inputs"], protocols)):
            raise ValueError("Measured state differs from the Build's actual original protocols")
        from ..protocol_content import current_source
        for row in protocols:
            self.reader.path(_bound(current_source(row, check=lambda pin: self.reader.path(_bound(pin)))))
        chemistry = value["recipe"]
        if (set(chemistry) != {"alkalinity_M", "btb_M"}
                or any(type(x) not in {int, float} or not np.isfinite(x) or x < 0
                       for x in chemistry.values())):
            raise ValueError("Explicit finite measured titration chemistry is required")
        rows = value["frames"]
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(row, dict) or set(row) != {
                    "case_id", "frame_index", "elapsed_h", "atmosphere", "nominal_injected_mass_g"}
                       for row in rows)
                or len({row["case_id"] for row in rows}) != len(rows)):
            raise ValueError("Measured state requires distinct exact frame identities")
        self.states = {row["case_id"]: row for row in rows}
        for row in rows:
            dose = row["nominal_injected_mass_g"]
            if dose is not None and (type(dose) not in {int, float}
                    or not np.isfinite(dose) or dose < 0):
                raise ValueError("Measured dose must be None or an explicit finite nonnegative number")

    def _centre(self, frame):
        """Decode actual bridge log storage with the original scientific decoder."""
        from .n1_geometry import compose_raw
        shape = self.bridge.context.grid["num_voxels"]
        centre = dict(native_shape_hw=shape,
            log_linear_rgb=dict(frame.products, array_key="log_linear_rgb"),
            valid_mask=dict(frame.products, array_key="valid_mask"))
        return compose_raw(dict(phase_shape_hw=shape, native_origin_xy=[0, 0],
                                native_centre=centre, ring_windows=[]), self.reader, self.math)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        row = next(self._iterator)
        frame = self._collect(row)
        self.bridge._revalidate()
        self.phase.commit(row)
        return frame

    def _collect(self, row):
        """Join actual producer buffers without promoting source QA or scope."""
        from .build_frame_fields import registered_build_fields, registered_scientific_reference
        from .n1_geometry import compose_raw

        spec, inferred = row["spec"], row["inferred"]
        phase_frame, native = row["frame"], row["native"]
        phase_proof = self.reader.json(_bound(row["phase_receipt"]))
        if (phase_proof.get("format") != "darsia-build-raw-phase-frame-v1"
                or phase_proof.get("complete") is not True
                or not _same(phase_proof.get("source"), spec["source"])
                or not _same(phase_proof.get("phase_configuration"),
                             self.inputs["phase_configuration"])
                or not _same(phase_proof.get("bridge_frame"), row["bridge_frame"].receipt)
                or not _same(phase_proof.get("products"), row["phase_products"])):
            raise ValueError("The phase frame differs from its actual completed producer")
        for product in row["phase_products"].values():
            if isinstance(product, dict) and "path" in product:
                self.reader.path(product)
        state = self.states.get(spec["case_id"])
        if state is None or any(not _same(state[key], spec[key])
                                for key in ("frame_index", "elapsed_h")):
            raise ValueError("Measured state belongs to another source frame/time")
        labels = inferred["labels"]
        shape = labels.shape
        domain = inferred["classes"]["rig_domain"]
        if (domain.dtype != bool or domain.shape != shape
                or not np.array_equal(domain, labels != 255)):
            raise ValueError("Original phase categories and rig domain disagree")
        analysis_domain = domain
        current_descriptor = native["record"]["full"]
        origin = current_descriptor["native_origin_xy"]
        current, valid = compose_raw(current_descriptor, self.reader, self.math)
        baseline_descriptor = self.phase.reference_record["full"]
        baseline, baseline_valid = compose_raw(baseline_descriptor, self.reader, self.math)
        receipt = self.reader.json(row["bridge_frame"].receipt)
        baseline_receipt = self.reader.json(self.phase.reference_bridge_frame.receipt)
        h = receipt["build_to_frame_native_h"]
        gp = self.phase.owner.phase["geometry_parameters"]
        from ..painted_analysis_exclusions import transport_build_exclusion
        approved_gap, gap_provenance = transport_build_exclusion(
            self.bridge.context, self.reader, h, origin, shape, gp)
        if approved_gap is not None:
            if (not np.array_equal(approved_gap, inferred["classes"].get("approved_analysis_gap"))
                    or np.any(approved_gap & analysis_domain)):
                raise ValueError("The phase producer changed its approved analysis-gap exclusion")
            physical_domain = inferred["classes"].get("physical_rig_domain")
            if (not isinstance(physical_domain, np.ndarray) or physical_domain.dtype != bool
                    or not np.array_equal(physical_domain, analysis_domain | approved_gap)):
                raise ValueError("Physical domain lost approved unquantified pixels")
            domain = physical_domain
        baseline = registered_scientific_reference(baseline, baseline_valid,
            baseline_receipt["build_to_frame_native_h"], baseline_descriptor["native_origin_xy"],
            h, origin, shape, self.math, geometry_parameters=gp)
        clear = registered_scientific_reference(self.clear_rgb, self.clear_valid,
            self.clear_receipt["build_to_frame_native_h"], [0, 0], h, origin, shape,
            self.math, geometry_parameters=gp, require_positive=True)
        physical = registered_build_fields(self.bridge.context, h, origin, shape, domain,
            self.math, gp, source_domain=self.source_domain)
        optical = (valid & baseline["valid"] & clear["valid"]
                   & np.isfinite(current).all(-1) & (current >= 0).all(-1))
        known = physical["physical_supported_mask"]
        if approved_gap is not None:
            known &= ~approved_gap
        supported = domain & known & optical
        area = inferred["pixel_area_mm2"]
        if (area.dtype != np.float32 or area.shape != shape or not np.isfinite(area).all()
                or np.any(area <= 0) or not np.array_equal(area, phase_frame["pixel_area_mm2"])):
            raise ValueError("Actual phase areas differ from their original measured geometry")
        selected = next(item for item in self.document["runs"][self.run]["frames"]
                        if item["case_id"] == spec["case_id"])
        if (not _same(selected["source"], spec["raw_acquisition"])
                or spec["source"]["source_sha256"] != selected["source"]["sha256"]):
            raise ValueError("The actual phase acquisition differs from its optical source")
        source_contract = _source_contract(self.document, self.run, selected)
        context_evidence = spec.get("context_acceptance")
        if context_evidence is not None:
            self.reader.path(_bound(context_evidence))
        provenance = dict(physical_context_verified=False, optical_source_contract=source_contract,
            build=self.bridge.context.provenance, bridge=self.bridge.proof,
            current_bridge=row["bridge_frame"].receipt, phase=row["phase_receipt"],
            baseline_bridge=self.phase.reference_bridge_frame.receipt,
            clearwater_bridge=self.clear.receipt, adopted_facies_table=self.table,
            configuration=self.configuration_binding, measured_state=self.state_binding,
            measured_state_verification_scope="explicit bound measured DATA; protocol identities checked, values not reparsed",
            optical_recipe=self.recipe_binding,
            run_inputs=deepcopy(self.inputs),
            geometry=deepcopy(spec["geometry"]), context_evidence=deepcopy(context_evidence),
            physical_transport=physical["provenance"],
            baseline_transport=baseline["provenance"], clearwater_transport=clear["provenance"],
            physical_support="actual adopted porous Build fields, complete original interpolation coverage",
            eligibility_scope="numeric source transport only; independent geometry, photographic transport and physical-scale qualification remain required",
            source_binding_verification_grants_scientific_approval=False)
        atmosphere = _atmosphere({}, state)
        if (not isinstance(atmosphere, dict) or set(atmosphere) - {
                "pressure_bar", "temperature_c", "pressure_gradient_bar_per_m",
                "temperature_gradient_c_per_m"}
                or not {"pressure_bar", "temperature_c"} <= set(atmosphere)
                or any(type(x) not in {int, float} or not np.isfinite(x)
                       for x in atmosphere.values())):
            raise ValueError("Measured atmosphere must retain finite explicit scalar values")
        if approved_gap is not None:
            provenance["approved_analysis_gap"] = gap_provenance
        frame = FrameData(run_id=self.run, frame_index=spec["frame_index"],
            elapsed_h=spec["elapsed_h"], current_rgb=current, baseline_rgb=baseline["rgb"],
            clear_water_rgb=clear["rgb"], clear_water_supported_mask=clear["valid"],
            phase_labels=labels, facies=physical["facies"], domain=domain,
            supported_mask=supported, mass_eligible_mask=np.zeros(shape, bool),
            depth_m=physical["depth_m"], porosity=physical["porosity"],
            pixel_area_m2=area.astype(np.float64) / 1e6,
            dimensions_m=tuple(self.bridge.context.grid["dimensions"]),
            pressure_bar=atmosphere["pressure_bar"], temperature_c=atmosphere["temperature_c"],
            pressure_gradient_bar_per_m=atmosphere.get("pressure_gradient_bar_per_m", 0.),
            temperature_gradient_c_per_m=atmosphere.get("temperature_gradient_c_per_m", 0.),
            nominal_injected_mass_g=state["nominal_injected_mass_g"], recipe=deepcopy(self.state["recipe"]),
            provenance=provenance, unknown_physical_mask=domain & ~known,
            **({"excluded_masks": {"approved_analysis_gap": approved_gap}} if approved_gap is not None else {}))
        self.bridge._revalidate()
        self.phase.revalidate()
        # A collected frame is committed before the original acquisition history advances.
        target = _fresh(self.output / spec["case_id"], self.bridge.forbidden)
        context_fields = dict(
            facies=frame.facies, depth_m=frame.depth_m, porosity=frame.porosity,
            domain=domain, supported_mask=supported, mass_eligible_mask=frame.mass_eligible_mask,
            unknown_physical_mask=frame.unknown_physical_mask,
            pixel_area_m2=frame.pixel_area_m2, clearwater_supported_mask=clear["valid"])
        if approved_gap is not None:
            context_fields["approved_analysis_gap"] = approved_gap
        products = _write_npz(target / "context.npz", context_fields)
        _write_json(target / "context.json", dict(format="darsia-build-frame-context-v1",
            run=self.run, case_id=spec["case_id"], frame_index=spec["frame_index"],
            elapsed_h=spec["elapsed_h"], products=products, provenance=provenance))
        return frame

    def close(self):
        """Release only the owned chronological readers and decoded references."""
        if not self._closed:
            self.phase.close()
            self.clear_rgb = self.clear_valid = None
            self._closed = True
