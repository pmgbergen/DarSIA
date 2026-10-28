"""Dispatch opt-in physics revisions without changing legacy runtime hashes."""
from .phase_saved_review_metadata import saved_physics_profile
from . import phase_physics, phase_temporal, phase_plume, phase_plume_envelope, phase_presence_context
from . import phase_reference_support
from . import phase_reviewed_reference
from . import phase_facies_reference
from . import phase_native_reference


def load_physics_profile(root):
    from pathlib import Path
    if (Path(root) / "phase_selected_policy_a.json").exists():
        if any((Path(root) / name).exists() for name in ("phase_rig_camera.json", "phase_optical_regions.json")):
            raise ValueError("Selected Policy A review requires its own source-bound profile")
        from .phase_selected_policy_review import load_profile
        return load_profile(root)
    if (Path(root) / "phase_rig_camera.json").exists():
        if (Path(root) / "phase_optical_regions.json").exists():
            raise ValueError("Rig/camera revision requires its own master profile, not a stacked legacy optical profile")
        load_profile = saved_physics_profile('rig_camera').load_profile
        return load_profile(root)
    from . import phase_optical_regions_runtime as optical_regions
    regional = optical_regions.load_profile(root)
    if regional is not None:
        from pathlib import Path
        track1_scope = regional.get("v5_track1_review_scope")
        if track1_scope is not None:
            if (track1_scope != {
                    "contract": "ffac-v5-track1-photo-only-review-context-v1",
                    "facies_ids_used": False,
                    "physical_scale_qa_pass": False,
                    "absolute_position_rules_enabled": False,
                    "mass_analysis_allowed": False,
                } or any(key in regional for key in (
                    "bottom_gas_guard", "region_edge_guard", "output_guards",
                    "aqueous_temporal_support"))):
                raise ValueError("Track1 photo-only review cannot use physical or temporal guards")
        nominal_review = regional.get("v5_nominal_review_gas_minimum")
        if nominal_review is not None:
            from dataclasses import asdict
            GasSettings = saved_physics_profile('free_gas').Settings
            expected = {
                "contract": "ffac-v5-nominal-review-gas-minimum-v1",
                "minimum_area_mm2": 2.5,
                "physical_scale_qa_pass": False,
                "mass_analysis_allowed": False,
            }
            if (nominal_review != expected
                    or regional.get("free_gas_cleanup") != asdict(GasSettings())
                    or any(key in regional for key in (
                        "bottom_gas_guard", "output_guards", "region_edge_guard"))):
                raise ValueError("Invalid V5 nominal review-only gas minimum")
        if "output_guards" in regional:
            Settings = saved_physics_profile('region_guard').Settings
            Settings(**regional["output_guards"])
        if "free_gas_cleanup" in regional:
            Settings = saved_physics_profile('free_gas').Settings
            Settings(**regional["free_gas_cleanup"])
        if "bottom_gas_guard" in regional:
            Settings = saved_physics_profile('bottom_gas').Settings
            Settings(**regional["bottom_gas_guard"])
        if "region_edge_guard" in regional:
            Settings = saved_physics_profile('region_edge').Settings
            Settings(**regional["region_edge_guard"])
        if "aqueous_temporal_support" in regional:
            runtime = _aqueous_temporal_runtime(regional["aqueous_temporal_support"])
            runtime.Settings(**regional["aqueous_temporal_support"])
        if "regional_gas" in regional:
            Settings = saved_physics_profile('regional_gas').Settings
            Settings(**regional["regional_gas"])
        if "approved_g3_gas" in regional:
            load_approved_profile = saved_physics_profile('approved_gas').load_approved_profile
            regional = load_approved_profile(root, regional)
        if "photo_sand_top_revision" in regional:
            load_top_revision_profile = saved_physics_profile('top_revision').load_top_revision_profile
            regional = load_top_revision_profile(root, regional)
        if "gas_barrier_rule_b" in regional:
            load_barrier_profile = saved_physics_profile('barrier').load_barrier_profile
            regional = load_barrier_profile(root, regional)
        if any((Path(root) / name).exists() for name in (
                "phase_native_reference.json", "phase_facies_reference.json", "phase_reviewed_reference.json",
                "phase_reference_support.json", "phase_presence_context.json", "phase_physics.json",
                "phase_plume_support.json", "phase_aqueous_context_completion.json")):
            raise ValueError("The optical region model cannot silently stack legacy context completion")
        return regional
    attach_decision = saved_physics_profile('decision_dispatch').attach_profile
    attach_profile = saved_physics_profile('completion').attach_profile
    attach_local = saved_physics_profile('local_support').attach_profile
    return attach_decision(root, attach_local(root, attach_profile(root, _load_base_physics_profile(root))))


def _load_base_physics_profile(root):
    native = phase_native_reference.load_profile(root)
    if native is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_facies_reference.json", "phase_reviewed_reference.json", "phase_reference_support.json",
                "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Native reference contains its inherited support profiles")
        return native
    facies = phase_facies_reference.load_profile(root)
    if facies is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_reviewed_reference.json", "phase_reference_support.json", "phase_presence_context.json",
                "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Facies reference contains its inherited support profiles")
        return facies
    reviewed = phase_reviewed_reference.load_profile(root)
    if reviewed is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_reference_support.json", "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Reviewed reference contains its inherited support profiles")
        return reviewed
    reference = phase_reference_support.load_profile(root)
    if reference is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Reference support already contains its inherited presence and gas profiles")
        return reference
    presence = phase_presence_context.load_profile(root)
    if presence is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in ("phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Presence profile already contains its inherited gas profile")
        return presence
    legacy = phase_physics.load_physics_profile(root)
    plume = phase_plume_envelope.load_profile(root)
    if plume is None:
        plume = phase_plume.load_profile(root)
    if legacy is not None and plume is not None:
        raise ValueError("Choose one phase-support profile; legacy and plume rules cannot be stacked")
    return plume if plume is not None else legacy


def _new(profile):
    return profile is not None and profile.get("format") in (phase_plume.VERSION, phase_plume_envelope.VERSION)


def _aqueous_temporal_runtime(specification):
    if not isinstance(specification, dict):
        raise ValueError("Aqueous temporal-support profile must be an object")
    first = saved_physics_profile('temporal_support')

    if specification.get("format") == first.VERSION:
        return first
    second = saved_physics_profile('source_history')

    if specification.get("format") == second.VERSION:
        return second
    raise ValueError("Unknown aqueous temporal-support format")


def _boundary_runtime(profile):
    phase_aqueous_boundary_runtime = saved_physics_profile('boundary')
    specification = profile.get("boundary_refinement")
    if isinstance(specification, dict) and specification.get("format") == "native-aqueous-boundary-v3":
        phase_aqueous_boundary_v3_runtime = saved_physics_profile('boundary_extended')
        return phase_aqueous_boundary_v3_runtime
    if isinstance(specification, dict) and specification.get("format") == "signal-backed-aqueous-continuity-v3":
        phase_aqueous_continuity = saved_physics_profile('boundary_continuity')
        return phase_aqueous_continuity
    return phase_aqueous_boundary_runtime


def physics_runtime_identity(profile):
    if profile is not None and profile.get("format") == "ffac-selected-policy-a-review-physics-v1":
        from .phase_selected_policy_review import runtime_identity
        return runtime_identity(profile)
    if profile is not None and profile.get("format") == "ffac-rig-camera-physics-v1":
        runtime_identity = saved_physics_profile('rig_camera').runtime_identity
        return runtime_identity(profile)
    if profile is not None and profile.get("format") == "baseline-relative-optical-regions-v1":
        from . import phase_optical_regions_runtime as optical_regions
        identity = optical_regions.runtime_identity(profile)
        if "approved_g3_gas" in profile:
            approved_runtime_identity = saved_physics_profile('approved_gas').approved_runtime_identity
            identity = approved_runtime_identity(identity, profile["approved_g3_gas"])
            if "photo_sand_top_revision" in profile:
                top_revision_runtime_identity = saved_physics_profile('top_revision').top_revision_runtime_identity
                identity = top_revision_runtime_identity(identity, profile["photo_sand_top_revision"])
            if "gas_barrier_rule_b" in profile:
                barrier_runtime_identity = saved_physics_profile('barrier').barrier_runtime_identity
                identity = barrier_runtime_identity(identity, profile["gas_barrier_rule_b"])
            return identity
        if "regional_gas" in profile:
            runtime_identity = saved_physics_profile('regional_gas').runtime_identity
            identity = runtime_identity(identity, profile["regional_gas"])
        if "output_guards" in profile:
            runtime_identity = saved_physics_profile('region_guard').runtime_identity
            identity = runtime_identity(identity, profile["output_guards"])
        if "free_gas_cleanup" in profile:
            runtime_identity = saved_physics_profile('free_gas').runtime_identity
            identity = runtime_identity(identity, profile["free_gas_cleanup"])
        if "bottom_gas_guard" in profile:
            runtime_identity = saved_physics_profile('bottom_gas').runtime_identity
            identity = runtime_identity(identity, profile["bottom_gas_guard"])
        if "region_edge_guard" in profile:
            runtime_identity = saved_physics_profile('region_edge').runtime_identity
            identity = runtime_identity(identity, profile["region_edge_guard"])
        if "aqueous_temporal_support" in profile:
            runtime = _aqueous_temporal_runtime(profile["aqueous_temporal_support"])
            identity = runtime.runtime_identity(identity, profile["aqueous_temporal_support"])
        return identity
    if profile is not None and "unified_phase_decision" in profile:
        decision = saved_physics_profile('decision_dispatch')
        inherited,specification=decision.split_profile(profile)
        return decision.runtime_identity(physics_runtime_identity(inherited),specification)
    if profile is not None and "local_aqueous_support" in profile:
        local = saved_physics_profile('local_support')
        inherited, specification = local.split_profile(profile)
        return local.runtime_identity(physics_runtime_identity(inherited), specification)
    if profile is not None and "aqueous_context_completion" in profile:
        completion = saved_physics_profile('completion')
        inherited, specification = completion.split_profile(profile)
        return completion.runtime_identity(physics_runtime_identity(inherited), specification)
    if profile is not None and "boundary_refinement" in profile:
        runtime = _boundary_runtime(profile)
        inherited, settings = runtime.split_profile(profile)
        return runtime.runtime_identity(physics_runtime_identity(inherited), settings)
    if profile is not None and profile.get("format") == phase_native_reference.VERSION:
        return phase_native_reference.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_facies_reference.VERSION:
        return phase_facies_reference.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_reviewed_reference.VERSION:
        return phase_reviewed_reference.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_reference_support.VERSION:
        return phase_reference_support.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_presence_context.VERSION:
        return phase_presence_context.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_plume_envelope.VERSION:
        return phase_plume_envelope.runtime_identity(profile)
    return phase_plume.runtime_identity(profile) if _new(profile) else phase_temporal.physics_runtime_identity(profile)


def load_run_context(root, profile, dataset, run, *, native_manifest_sha256):
    if profile is not None and profile.get("format") == "ffac-selected-policy-a-review-physics-v1":
        from .phase_selected_review import require_dedicated_inference
        require_dedicated_inference()
    if profile is not None and profile.get("format") == "ffac-rig-camera-physics-v1":
        require_dedicated_inference = saved_physics_profile('rig_camera').require_dedicated_inference
        require_dedicated_inference()
    if profile is not None and profile.get("format") == "baseline-relative-optical-regions-v1":
        return phase_plume.load_context(root, profile, dataset, run,
                                       native_manifest_sha256=native_manifest_sha256)
    if profile is not None and "unified_phase_decision" in profile:
        split_profile = saved_physics_profile('decision').split_profile
        profile,_=split_profile(profile)
    if profile is not None and "local_aqueous_support" in profile:
        split_profile = saved_physics_profile('local_support').split_profile
        profile, _ = split_profile(profile)
    if profile is not None and "aqueous_context_completion" in profile:
        split_profile = saved_physics_profile('completion').split_profile
        profile, _ = split_profile(profile)
    if profile is not None and "boundary_refinement" in profile:
        profile, _ = _boundary_runtime(profile).split_profile(profile)
    if profile is not None and profile.get("format") in (phase_presence_context.VERSION, phase_reference_support.VERSION, phase_reviewed_reference.VERSION, phase_facies_reference.VERSION, phase_native_reference.VERSION):
        return phase_plume.load_context(root, profile, dataset, run, native_manifest_sha256=native_manifest_sha256)
    loader = phase_plume.load_context if _new(profile) else phase_physics.load_run_context
    return loader(root, profile, dataset, run, native_manifest_sha256=native_manifest_sha256)
