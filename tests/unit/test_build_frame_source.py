"""Local numeric integration only, with explicit nonqualifying RAW/phase doubles.

JSON/CSV/NPZ identities, scientific decoding, field transport and FrameData are
the original implementations. No camera, classifier, Build or fit is executed.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass import build_frame_source as subject
from darsia.fluidflower.mass.build_native_bridge import BuildNativeFrame, _Inputs
from darsia.fluidflower.mass.n1_models import FrameData
from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json, _write_npz, asset
from darsia.fluidflower.mass.native_optical_source import BuildPhysicalContext
from darsia.fluidflower.mass.pooled_optical_paths import sample_frame_for_common_paths
from darsia.fluidflower.phase.inputs import TemporalNativeInput
from darsia.fluidflower.pipeline import build_raw_phase
from darsia.fluidflower.pipeline.runner import _phase_outputs


def shift(x, y=0):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


class NumericBridge:
    """Produce explicit fixture buffers only; never represent RAW qualification."""

    def __init__(self, world):
        self.world, self.context = world, world.context
        self.reader, self.forbidden, self.calls = _Inputs(DataReader()), [], []
        self.proof = world.document("bridge", dict(
            format="darsia-build-native-bridge-v1", photometry_source=world.profile,
            scope="numeric producer double; no scientific qualification"))

    def _revalidate(self):
        self.reader.revalidate()

    def frame(self, acquisition, output):
        self.reader.path(acquisition)
        self.calls.append(deepcopy(acquisition))
        kind = self.world.kind[acquisition["sha256"]]
        output = Path(output)
        output.mkdir(parents=True)
        rgb = np.full((3, 4, 3), {"baseline": .2, "clearwater": .4, "current": .1}[kind], np.float32)
        log = np.moveaxis(np.log(rgb * np.float32(65535)).astype(np.float32), -1, 0)
        products = _write_npz(output / "native.npz", dict(
            log_linear_rgb=log, valid_mask=np.ones((3, 4), bool)))
        h = self.world.current_h if kind == "current" else np.eye(3)
        geometry = _write_json(output / "geometry.json", dict(
            acquisition=acquisition, reference_to_native_h=self.world.photometry_h.tolist(),
            scope="numeric geometry double; distinct photometry coordinates"))
        receipt = _write_json(output / "frame.json", dict(
            format="darsia-build-native-frame-v1", bridge=self.proof,
            products=products, native_frame=geometry, build_to_frame_native_h=h.tolist(),
            scope="numeric corrected-buffer double; no scientific qualification"))
        return BuildNativeFrame(dict(kind="raw_rig", geometry=geometry,
            processor=dict(raw_binding=self.world.profile)), products, receipt)


class NumericPhase:
    """Yield saved original product formats and acknowledge collection explicitly."""

    def __init__(self, bridge, configuration, output):
        self.world, self.bridge = bridge.world, bridge
        self.world.phase = self
        self.owner = SimpleNamespace(phase=dict(geometry_parameters=dict(determinant_floor=1e-12)))
        output = Path(output)
        output.mkdir()
        self.reference_bridge_frame = bridge.frame(
            bridge.context.provenance["raw_acquisition"], output / "baseline")
        self.reference_record = dict(full=self.full(self.reference_bridge_frame))
        self.rows, self.committed, self.closed = [], [], False
        for original in configuration["frames"]:
            spec = deepcopy(original)
            target = output / spec["case_id"]
            target.mkdir()
            native_frame = bridge.frame(spec["raw_acquisition"], target / "native")
            full = self.full(native_frame)
            area = np.full((5, 6), .75, np.float32)
            log = bridge.reader.reader.array(dict(native_frame.products, array_key="log_linear_rgb"))
            temporal = TemporalNativeInput(log, np.isfinite(log).all(0), (0, 0),
                dict(scope="numeric temporal double"))
            frame = dict(pixel_area_mm2=area, temporal_input=temporal,
                current_h=self.world.photometry_h.copy(), scientific=dict(full=full))
            modelled = np.zeros((5, 6), bool)
            modelled[1:4, 1:5] = True
            inferred = dict(labels=np.ones((5, 6), np.uint8), pixel_area_mm2=area.copy(),
                classes=dict(rig_domain=np.ones((5, 6), bool), modelled=modelled,
                    filled_observed=np.zeros((5, 6), bool), filled_unobserved=~modelled),
                qa=dict(scope="numeric phase double; no scientific qualification"),
                seam_masks={}, post_masks={})
            products = _phase_outputs(target, spec, frame, inferred)
            row = dict(spec=spec, inferred=inferred, frame=frame,
                native=dict(record=dict(full=full)), bridge_frame=native_frame,
                phase_products=products)
            row["phase_receipt"] = self.world.phase_receipt(row)
            self.rows.append(row)

    @staticmethod
    def full(frame):
        return dict(phase_shape_hw=[5, 6], native_origin_xy=[-1, -1], ring_windows=[],
            native_centre=dict(native_shape_hw=[3, 4],
                log_linear_rgb=dict(frame.products, array_key="log_linear_rgb"),
                valid_mask=dict(frame.products, array_key="valid_mask")))

    def __iter__(self):
        return iter(self.rows)

    def commit(self, row):
        # Read the actual context products before allowing predecessor advancement.
        saved = self.world.output / row["spec"]["case_id"] / "context.json"
        record = self.bridge.reader.json(asset(saved))
        with np.load(self.bridge.reader.path(record["products"]), allow_pickle=False) as archive:
            assert archive["supported_mask"].any()
            assert not archive["mass_eligible_mask"].any()
            np.testing.assert_array_equal(archive["domain"], row["inferred"]["classes"]["rig_domain"])
        self.committed.append(row["spec"]["case_id"])

    def revalidate(self):
        self.bridge.reader.revalidate()

    def close(self):
        self.closed = True


class NumericWorld:
    def __init__(self, root):
        self.root, self.output, self.sequence, self.phase = root, root / "collected", 0, None
        self.current_h, self.photometry_h = np.eye(3), shift(30, 20)
        self.kind = {}
        self.raw = {kind: self.raw_file(kind) for kind in ("baseline", "clearwater", "current")}
        self.profile = self.document("profile", dict(scope="numeric photometry double"))
        self.native_configuration = self.document("native", dict(scope="numeric native parameters"))
        self.phase_configuration = self.document("phase", dict(scope="numeric phase parameters"))
        table_path = root / "facies.csv"
        table_path.write_text("id,name,analysis_role,porosity\n0,Outside,frame,0\n"
                              "1,Sand,porous_medium,0.3\n3,Other sand,porous_medium,0.4\n", encoding="utf-8")
        self.table = asset(table_path)
        self.adopted = dict(configuration=dict(grouping_mode="shared_facies",
            regions=[dict(label_id=10, facies_id=1, enabled=True),
                     dict(label_id=20, facies_id=3, enabled=True)]),
            artifacts=dict(source_mode_files=dict(facies_properties=self.table)))
        adopted = self.document("adopted", self.adopted)
        protocols = [dict(role=role, source=self.document(role, dict(scope="private numeric protocol")))
                     for role in ("pressure_temperature", "injection")]
        request = self.document("request", dict(protocol_inputs=protocols))
        operation = self.document("operation", dict(scope="numeric Build context double"))
        facies = np.tile(np.array([1, 3, 1, 3], np.uint16), (3, 1))
        self.context = BuildPhysicalContext(run="sample", baseline_rgb=np.full((3, 4, 3), .9, np.float32),
            labels=np.where(facies == 1, 10, 20).astype(np.uint16), facies=facies,
            depth_m=np.tile(np.array([.01, .02, .03, .04], np.float32), (3, 1)),
            porosity=np.where(facies == 1, .3, .4).astype(np.float32),
            grid=dict(num_voxels=[3, 4], dimensions=[.3, .4], origin=[0., .3], voxel_size=[.1, .1]),
            provenance=dict(adopted_revision=adopted, operation_result=operation,
                native_configuration=self.native_configuration, raw_acquisition=self.raw["baseline"],
                raw_build_request=request))
        self.bridge = NumericBridge(self)
        source = dict(run="sample", frame_index=1, elapsed_h=.5, selected=True,
            baseline=False, studio_anchor=False, previous_source_sha256=self.raw["baseline"]["sha256"],
            source_sha256=self.raw["current"]["sha256"],
            source_binding=dict(source_sha256=self.raw["current"]["sha256"]))
        geometry = self.document("rig-geometry", dict(scope="numeric geometry double"))
        photo = _write_npz(root / "photo.npz", dict(sand=np.ones((3, 4), bool)))
        components = _write_npz(root / "components.npz", dict(
            open_components=np.ones((3, 4), np.uint8), barrier_mask=np.zeros((3, 4), bool),
            reference_domain=np.ones((3, 4), bool)))
        roles = dict(area=dict(role="selected", factor=1., revision=dict(scope="numeric area revision")),
            current_h=self.photometry_h.tolist(), reference_to_current_h=self.photometry_h.tolist(),
            static=dict(components=components, source_ids=[1], unchanged_source=False),
            retained_source_cache=None, encoding_contract=None, light_epoch_id="numeric",
            inherited_spacing_m=[.001, .001], probability_spacing_m=[.001, .001], placement_metadata={},
            photo_sand_reference=photo, photo_proof=dict(sha256=photo["sha256"], phase_labels_used=False))
        spec = dict(case_id="sample-1", frame_index=1, elapsed_h=.5, source=source,
            raw_acquisition=self.raw["current"], geometry=geometry, phase_roles=roles)
        reference = dict(case_id="baseline", frame_index=0, elapsed_h=0.,
            raw_acquisition=self.raw["baseline"], geometry=geometry, phase_roles=deepcopy(roles),
            source=dict(source, frame_index=0, elapsed_h=0., selected=False, baseline=True,
                source_sha256=self.raw["baseline"]["sha256"], previous_source_sha256=None,
                source_binding=dict(source_sha256=self.raw["baseline"]["sha256"])))
        self.configuration = dict(format="darsia-build-raw-phase-inputs-v1", run="sample", frames=[spec],
            native=dict(configuration=self.native_configuration), phase=dict(configuration=self.phase_configuration),
            history_policy="selected acquisitions in camera order", selection_index_field="frame_index",
            reference_spec=reference, initial_previous_spec=deepcopy(reference), temporal_pad_px=0)
        self.state = dict(format="darsia-build-frame-state-v1", run="sample",
            recipe=dict(alkalinity_M=.01, btb_M=.001), protocol_inputs=protocols,
            frames=[dict(case_id="sample-1", frame_index=1, elapsed_h=.5,
                atmosphere=dict(pressure_bar=1.1, temperature_c=21.), nominal_injected_mass_g=.2)])
        selected = dict(case_id="sample-1", frame_index=1, elapsed_h=.5, target_h=.5,
                        source=self.raw["current"])
        self.recipe = dict(format="darsia-pooled-native-optical-recipe-v1",
            method="equal-run-native-relative-anchors-v1", fit_runs=["sample", "other"], excluded_runs=[],
            roles=[dict(facies_id=i, name=n, analysis_role=r) for i, n, r in
                ((0, "Outside", "frame"), (1, "Sand", "porous_medium"), (3, "Other sand", "porous_medium"))],
            adopted_revision=adopted, build_context=operation, phase_configuration=self.phase_configuration,
            native_configuration=self.native_configuration,
            runs={run: dict(baseline=self.raw["baseline"], clearwater=self.raw["clearwater"],
                            profile=self.profile, frames=[deepcopy(selected)]) for run in ("sample", "other")},
            sampling=dict(stride_px=1, interior_radius_px=0, min_samples=1, min_runs=2),
            flash_bounds=dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.),
            path_name_prefix="numeric-only")

    def document(self, name, value):
        self.sequence += 1
        return _write_json(self.root / f"{name}-{self.sequence}.json", value)

    def raw_file(self, kind):
        path = self.root / (kind + ".arw")
        path.write_bytes(("numeric producer double only: " + kind).encode("ascii"))
        bound = asset(path)
        self.kind[bound["sha256"]] = kind
        return bound

    def phase_receipt(self, row, **changes):
        proof = dict(format="darsia-build-raw-phase-frame-v1", complete=True,
            source=deepcopy(row["spec"]["source"]), phase_configuration=self.phase_configuration,
            bridge_frame=row["bridge_frame"].receipt, products=row["phase_products"],
            scope="numeric phase producer double; no scientific qualification")
        proof.update(changes)
        return self.document("phase-frame", proof)

    def create(self):
        adopted = self.document("adopted", self.adopted)
        self.bridge.context = replace(self.context, provenance=dict(
            self.context.provenance, adopted_revision=adopted))
        self.recipe["adopted_revision"] = adopted
        return subject.BuildFrameSource(self.bridge, self.document("configuration", self.configuration),
            self.raw["clearwater"], self.document("state", self.state),
            self.document("recipe", self.recipe), self.output)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(build_raw_phase, "BuildRawPhaseCollector", NumericPhase)
    value = NumericWorld(tmp_path)
    yield value
    if value.phase is not None:
        value.phase.close()


def test_collect_decodes_original_buffers_and_saves_context_before_history_acknowledgement(world):
    collector = world.create()
    frame = next(collector)
    assert isinstance(frame, FrameData)
    np.testing.assert_allclose(frame.current_rgb[1:4, 1:5], .1, rtol=2e-6)
    np.testing.assert_allclose(frame.baseline_rgb[1:4, 1:5], .2, rtol=2e-6)
    np.testing.assert_allclose(frame.clear_water_rgb[1:4, 1:5], .4, rtol=2e-6)
    assert not np.allclose(frame.baseline_rgb[1:4, 1:5], world.context.baseline_rgb)
    assert frame.supported_mask.sum() == 12
    assert not frame.mass_eligible_mask.any()
    np.testing.assert_array_equal(frame.unknown_physical_mask, ~frame.supported_mask)
    np.testing.assert_allclose(frame.pixel_area_m2, .75e-6)
    assert frame.pressure_bar == 1.1 and frame.temperature_c == 21.
    assert world.phase.committed == ["sample-1"]
    collector.close()
    assert world.phase.closed


def test_physical_fields_follow_build_coordinates_instead_of_distinct_photometry_reference(world):
    world.current_h = shift(1)
    frame = next(world.create())
    np.testing.assert_array_equal(frame.facies[1:4, 2:6], world.context.facies)
    np.testing.assert_array_equal(frame.depth_m[1:4, 2:6], world.context.depth_m)
    assert frame.unknown_physical_mask[1:4, 1].all()
    assert frame.supported_mask[1:4, 2:5].all()
    assert not frame.supported_mask[:, 5].any()
    assert world.phase.committed == ["sample-1"]


def test_fractional_source_coverage_never_enters_known_or_optical_support(world):
    world.current_h = shift(.25)
    frame = next(world.create())
    assert frame.unknown_physical_mask[1:4, 1].all()
    assert not frame.supported_mask[1:4, 1].any()
    assert not frame.clear_water_supported_mask[1:4, 1].any()
    assert frame.supported_mask[1:4, 2:5].all()


def test_missing_measured_depth_is_preserved_without_imputation(world):
    world.context.depth_m[1, 1] = 0
    frame = next(world.create())
    assert frame.depth_m[2, 2] == 0 and frame.unknown_physical_mask[2, 2]
    assert not frame.supported_mask[2, 2] and not frame.mass_eligible_mask[2, 2]
    assert world.context.depth_m[1, 1] == 0


@pytest.mark.parametrize("slot", ["build_context", "clearwater", "profile", "source"])
def test_exact_optical_input_mismatch_refuses_before_any_bridge_or_phase_work(world, slot):
    other = world.document("different-input", dict(scope="unrelated numeric input"))
    if slot == "build_context":
        world.recipe[slot] = other
    elif slot == "source":
        world.recipe["runs"]["sample"]["frames"][0][slot] = other
    else:
        world.recipe["runs"]["sample"][slot] = other
    with pytest.raises(ValueError, match="optical recipe|input bundle|reference selections|chronological selection"):
        world.create()
    assert not world.bridge.calls and world.phase is None and not world.output.exists()


@pytest.mark.parametrize("fault", ["literal-roles", "label-facies", "grouping"])
def test_adopted_full_table_and_saved_labels_must_agree(world, fault):
    if fault == "literal-roles":
        world.recipe["roles"][1]["name"] = "Different sand"
    elif fault == "label-facies":
        world.adopted["configuration"]["regions"][1]["facies_id"] = 1
    else:
        world.adopted["configuration"]["grouping_mode"] = "separate_regions"
    with pytest.raises(ValueError, match="roles differ|labels/facies|shared facies"):
        world.create()
    assert not world.bridge.calls and world.phase is None


def test_state_protocol_mismatch_refuses_before_production_or_history(world):
    world.state["protocol_inputs"] = deepcopy(world.state["protocol_inputs"])
    world.state["protocol_inputs"][0]["source"] = world.profile
    with pytest.raises(ValueError, match="actual original protocols"):
        world.create()
    assert not world.bridge.calls and world.phase is None


def test_disabled_adopted_porous_region_remains_unknown_and_outside_supported_fields(world):
    world.adopted["configuration"]["regions"][1]["enabled"] = False
    frame = next(world.create())
    assert frame.supported_mask[1:4, [1, 3]].all()
    assert frame.unknown_physical_mask[1:4, [2, 4]].all()
    assert not frame.supported_mask[1:4, [2, 4]].any()
    np.testing.assert_array_equal(frame.facies[1:4, 1:5], world.context.facies)


def run_bundles(world):
    for role, actual in (
        ("build_context", world.context.provenance["operation_result"]),
        ("phase_configuration", world.phase_configuration),
        ("native_configuration", world.native_configuration),
    ):
        other = world.document("other-" + role, dict(scope="distinct other-run numeric input"))
        world.recipe[role] = world.document("bundle-" + role, dict(
            format="darsia-pooled-optical-run-inputs-v1", role=role,
            runs=dict(sample=actual, other=other)))


def test_distinct_run_bundles_preserve_global_recipe_and_actual_phase_producer_binding(world):
    run_bundles(world)
    frame = next(world.create())
    contract = frame.provenance["optical_source_contract"]
    assert contract["build_context"] == world.recipe["build_context"]
    assert contract["phase_configuration"] == world.recipe["phase_configuration"]
    assert frame.provenance["build"]["operation_result"] == world.context.provenance["operation_result"]
    phase = world.bridge.reader.json(frame.provenance["phase"])
    assert phase["phase_configuration"] == world.phase_configuration
    assert phase["phase_configuration"] != contract["phase_configuration"]
    assert world.phase.committed == ["sample-1"]


@pytest.mark.parametrize("role", ["build_context", "phase_configuration"])
def test_bundle_cannot_substitute_another_runs_producer_for_this_run(world, role):
    run_bundles(world)
    bundle = world.bridge.reader.json(world.recipe[role])
    bundle["runs"]["sample"] = bundle["runs"]["other"]
    world.recipe[role] = world.document("wrong-member-" + role, bundle)
    with pytest.raises(ValueError, match="this run's actual producer"):
        world.create()
    assert not world.bridge.calls and world.phase is None


def test_state_time_mismatch_cannot_advance_history_or_publish_a_context(world):
    world.state["frames"][0]["elapsed_h"] = .6
    collector = world.create()
    with pytest.raises(ValueError, match="source frame/time"):
        next(collector)
    assert not world.phase.committed and not (world.output / "sample-1").exists()


def test_changed_phase_source_refuses_even_when_its_producer_receipt_agrees(world):
    collector = world.create()
    row = world.phase.rows[0]
    row["spec"]["raw_acquisition"] = world.raw["clearwater"]
    row["spec"]["source"]["source_sha256"] = world.raw["clearwater"]["sha256"]
    row["spec"]["source"]["source_binding"]["source_sha256"] = world.raw["clearwater"]["sha256"]
    row["phase_receipt"] = world.phase_receipt(row)
    with pytest.raises(ValueError, match="actual phase acquisition"):
        next(collector)
    assert not world.phase.committed and not (world.output / "sample-1").exists()


def test_phase_configuration_receipt_mismatch_refuses_before_history(world):
    collector = world.create()
    row = world.phase.rows[0]
    row["phase_receipt"] = world.phase_receipt(row, phase_configuration=world.profile)
    with pytest.raises(ValueError, match="actual completed producer"):
        next(collector)
    assert not world.phase.committed


def test_changed_saved_phase_npz_is_rejected_by_original_data_reader(world):
    collector = world.create()
    product = world.phase.rows[0]["phase_products"]["classes"]
    Path(product["path"]).write_bytes(b"changed private numeric NPZ fixture")
    with pytest.raises(ValueError, match="Data size differs|data changed|SHA or identity"):
        next(collector)
    assert not world.phase.committed and not (world.output / "sample-1").exists()


@pytest.mark.parametrize("field,value", [
    ("pressure_gradient_bar_per_m", True),
    ("temperature_gradient_c_per_m", "unknown"),
])
def test_unmeasured_atmosphere_scalar_cannot_publish_or_advance_history(world, field, value):
    world.state["frames"][0]["atmosphere"][field] = value
    collector = world.create()
    with pytest.raises(ValueError, match="finite explicit scalar"):
        next(collector)
    assert not world.phase.committed and not (world.output / "sample-1").exists()


def test_cached_clearwater_still_requires_unchanged_original_npz_before_context_publication(world):
    collector = world.create()
    Path(collector.clear.products["path"]).write_bytes(b"changed private cached-reference fixture")
    with pytest.raises(ValueError, match="Data size differs|data changed|SHA or identity"):
        next(collector)
    assert not world.phase.committed and not (world.output / "sample-1").exists()


def test_numeric_source_support_cannot_enter_original_pooled_optical_sampling(world):
    collector = world.create()
    frame = next(collector)
    assert frame.supported_mask.any() and not frame.mass_eligible_mask.any()
    with pytest.raises(ValueError, match="independently source-verified"):
        sample_frame_for_common_paths(frame, collector.recipe, case_id="sample-1")


def test_bound_source_uncertainty_is_preserved_and_an_index_grants_no_approval(world):
    original = dict(physical_scale_approved=False, edge_gate=False,
                    unobserved_wall_uncertainty=True)
    geometry = world.document("unqualified-geometry", original)
    index = world.document("context-index", dict(original_geometry=geometry,
        scope="numeric index of existing source evidence; no scientific approval"))
    world.configuration["frames"][0].update(geometry=geometry, context_acceptance=index)
    collector = world.create()
    frame = next(collector)
    assert frame.provenance["geometry"] == geometry
    assert world.bridge.reader.json(frame.provenance["geometry"]) == original
    assert frame.provenance["context_evidence"] == index
    assert world.bridge.reader.json(frame.provenance["context_evidence"])["original_geometry"] == geometry
    assert not frame.mass_eligible_mask.any()
    with pytest.raises(ValueError, match="independently source-verified"):
        sample_frame_for_common_paths(frame, collector.recipe, case_id="sample-1")


@pytest.mark.parametrize("dose", [True, -.1, "0.2"])
def test_dose_requires_explicit_finite_nonnegative_numeric_data_before_production(world, dose):
    world.state["frames"][0]["nominal_injected_mass_g"] = dose
    with pytest.raises(ValueError, match="Measured dose must be None or an explicit finite nonnegative"):
        world.create()
    assert not world.bridge.calls and world.phase is None and not world.output.exists()


@pytest.mark.parametrize("dose", [None, 0, .25])
def test_missing_or_measured_numeric_dose_retains_its_original_value(world, dose):
    world.state["frames"][0]["nominal_injected_mass_g"] = dose
    frame = next(world.create())
    assert frame.nominal_injected_mass_g == dose
    assert not frame.mass_eligible_mask.any()
