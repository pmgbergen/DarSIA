"""Assignment metadata distinguishes absent properties from physical zero."""

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.facies_props import FaciesProps


def test_assignment_membership_preserves_known_zero_and_original_numeric_defaults():
    facies = darsia.Image(np.array([[0, 2, 7]], dtype=np.uint8), dimensions=[1, 3])
    properties = FaciesProps(
        facies, porosity={0: 0.0, 2: 0.44}, permeability={0: 0.0, 2: 473.0}
    )

    assert properties.porosity_labels == frozenset({0, 2})
    assert properties.permeability_labels == frozenset({0, 2})
    np.testing.assert_array_equal(properties.porosity.img, [[0.0, 0.44, 0.0]])
    np.testing.assert_array_equal(properties.permeability.img, [[0.0, 473.0, 0.0]])
    assert 0 in properties.porosity_labels
    assert 7 not in properties.porosity_labels


def test_constant_properties_have_no_unassigned_label_support():
    """Retain original full_like dtype arithmetic while marking every label assigned."""
    facies = darsia.Image(np.array([[0, 7]], dtype=np.uint8), dimensions=[1, 2])
    properties = FaciesProps(facies, porosity=0.0, permeability=2.5)

    assert properties.porosity_labels is None
    assert properties.permeability_labels is None
    np.testing.assert_array_equal(properties.porosity.img, [[0.0, 0.0]])
    # Original scalar properties inherit the label dtype and truncate this value.
    np.testing.assert_array_equal(properties.permeability.img, [[2, 2]])
    assert properties.permeability.img.dtype == facies.img.dtype


def test_loaded_assignment_metadata_comes_from_the_same_parsed_property_rows(tmp_path):
    source = tmp_path / "properties.csv"
    source.write_text("id,porosity,permeability\n0,0,0\n3,0.44,44\n")
    facies = darsia.Image(np.array([[0, 3, 9]], dtype=np.uint8), dimensions=[1, 3])

    properties = FaciesProps.load(facies, source)

    assert properties.porosity_labels == frozenset({0, 3})
    assert properties.permeability_labels == frozenset({0, 3})
    np.testing.assert_array_equal(properties.porosity.img, [[0.0, 0.44, 0.0]])
    np.testing.assert_array_equal(properties.permeability.img, [[0.0, 44.0, 0.0]])


@pytest.mark.parametrize("native_product", [None, {"selected": "native-without-gap"}])
def test_setup_proves_native_inputs_before_skipping_shape_corrections(monkeypatch, tmp_path, native_product):
    """Fully painted native input needs no fabricated exclusion approval."""
    _setup_until_properties(monkeypatch, tmp_path, native_product=native_product)


def test_failed_native_proof_never_loads_labels_or_facies(monkeypatch, tmp_path):
    """A same-sized but foreign native input cannot enter corrected loading."""
    _setup_until_properties(
        monkeypatch, tmp_path, native_product={"selected": "foreign-native"}, fail_proof=True
    )


def test_new_undeclared_setup_does_not_reuse_previous_domain(monkeypatch, tmp_path):
    """Reusing a Rig for legacy input cannot retain a prior native exclusion."""
    _setup_until_properties(monkeypatch, tmp_path, native_product=None, prior_domain=True)


def test_native_contract_requires_raw_context_before_reading_or_writing(monkeypatch, tmp_path):
    """A native product cannot fall through the original JPEG reading route."""
    _setup_until_properties(
        monkeypatch, tmp_path, native_product={"selected": "native"}, raw_context=False
    )


def test_rejected_native_setup_preserves_previous_domain_and_camera(monkeypatch, tmp_path):
    """A rejected setup cannot turn an existing native Rig into a legacy save."""
    _setup_until_properties(
        monkeypatch, tmp_path, native_product={"selected": "native"},
        raw_context=False, prior_domain=True,
    )


def _setup_until_properties(
    monkeypatch, tmp_path, *, native_product, fail_proof=False, prior_domain=False, raw_context=True
):
    """Exercise original setup ordering with the external proof mocked explicitly."""
    import sys
    from types import ModuleType, SimpleNamespace

    from darsia.fluidflower import rig_raw_input
    from darsia.presets.workflows.rig import Rig

    events = []
    camera = {"run_id": "ac17"} if raw_context else None
    monkeypatch.setattr(rig_raw_input, "camera_binding", lambda value: value)
    monkeypatch.setattr(rig_raw_input, "require_raw_rebuild", lambda value: None)

    def proof(rig, contract, *, labels_path, facies_path, native_product):
        assert contract is None
        assert labels_path == tmp_path / "labels.npz"
        assert facies_path == tmp_path / "facies.npz"
        events.append("proof")
        if fail_proof:
            raise ValueError("The bound native baseline differs")

    helper = ModuleType("darsia.fluidflower.rig_analysis_domain")
    helper.prepare_analysis_domain = proof
    monkeypatch.setitem(sys.modules, helper.__name__, helper)

    class ReachedProperties(Exception):
        pass

    class SetupProbe:
        def setup_reading(self, *args, **kwargs):
            if prior_domain:
                assert "_analysis_domain" not in self.__dict__
            events.append("reading")

        def setup_depth(self, *args, **kwargs):
            events.append("depth")

        def setup_labels(self, *args, **kwargs):
            events.append(("labels", kwargs["apply_corrections"]))

        def setup_inner_labels(self, **kwargs):
            pass

        def setup_facies(self, *args, **kwargs):
            events.append(("facies", kwargs["apply_corrections"]))

        def setup_facies_props(self, *args, **kwargs):
            raise ReachedProperties

    build = SimpleNamespace(
        camera=camera,
        raw_source={"sha256": "a" * 64},
        operation_id="reserved-operation",
        begin_setup=lambda rig, log: None,
    )
    probe = SetupProbe()
    old_domain = object()
    old_camera = {"run_id": "ac18"}
    if prior_domain:
        probe._analysis_domain = old_domain
        probe.raw_camera = old_camera
    error = ValueError if fail_proof or not raw_context else ReachedProperties
    with pytest.raises(error):
        Rig.setup(
            probe,
            experiment=SimpleNamespace(experiment_start=0),
            baseline_path=tmp_path / "baseline.ARW",
            depth_map_path=tmp_path / "depth.npz",
            labels_path=tmp_path / "labels.npz",
            facies_path=tmp_path / "facies.npz",
            raw_camera=camera,
            raw_baseline_sha256="a" * 64,
            raw_build_id="reserved-operation",
            raw_build=build,
            native_product=native_product,
        )
    if prior_domain:
        assert probe._analysis_domain is old_domain
        assert probe.raw_camera is old_camera
    if not raw_context:
        assert events == []
    elif fail_proof:
        assert events == ["reading", "depth", "proof"]
    else:
        expected = ["reading", "depth"]
        if native_product is not None:
            expected.append("proof")
        expected.extend([("labels", native_product is None), ("facies", native_product is None)])
        assert events == expected


@pytest.mark.parametrize("failure", ["ownership", "native_proof", "properties", None])
@pytest.mark.parametrize("prior_native", [False, True])
def test_native_replacement_keeps_previous_rig_coherent_until_success(
    monkeypatch, tmp_path, failure, prior_native
):
    """Keep actual old pixels and guarded accounting after late setup failures.

    RAW ownership and scientific admission are named synthetic boundaries here.
    The original setup ordering and native integral implementation are exercised.
    """
    import sys
    from inspect import signature
    from types import ModuleType, SimpleNamespace

    from darsia.fluidflower import rig_raw_input
    from darsia.fluidflower.native_analysis_domain import AnalysisDomainGeometry
    from darsia.presets.workflows.rig import Rig

    rig = Rig()
    old_pixels = np.array([[2.0, 9.0]])
    old_baseline = darsia.Image(old_pixels.copy(), dimensions=[1, 2])
    old_geometry = darsia.Geometry(2, (1, 2), dimensions=[1, 2])
    old_provenance = {"source": "previous-native-fixture", "physical_qualification": False}
    if prior_native:
        rig._analysis_domain = SimpleNamespace(proof=old_provenance)
        rig.geometry = AnalysisDomainGeometry(
            old_geometry, np.array([[False, True]]), old_provenance
        )
        before_total = rig.geometry.integrate_quantified(old_pixels)
    else:
        rig.geometry = old_geometry
        before_total = rig.geometry.integrate(old_pixels)
    rig.baseline = old_baseline
    rig.shape_corrected_baseline = old_baseline.copy()
    rig.labels = darsia.Image(np.array([[1, 2]], dtype=np.uint8), dimensions=[1, 2])
    rig.facies = rig.labels.copy()
    rig.depth = darsia.Image(np.array([[0.01, 0.02]]), dimensions=[1, 2])
    rig.porosity = darsia.Image(np.array([[0.25, 0.5]]), dimensions=[1, 2])
    rig.permeability = darsia.Image(np.array([[10.0, 20.0]]), dimensions=[1, 2])
    rig.shape_corrections = [{"old_mutable_array": old_pixels.copy()}]
    rig.color_corrections = [{"old_mutable_list": [1, 2]}]
    rig.raw_camera = {"run_id": "ac18"}
    rig._raw_build = object()
    previous = rig.__dict__
    old_fields = dict(previous)
    image_fields = ("baseline", "shape_corrected_baseline", "labels", "facies",
                    "depth", "porosity", "permeability")
    old_image_bytes = {name: getattr(rig, name).img.tobytes() for name in image_fields}
    old_correction_bytes = rig.shape_corrections[0]["old_mutable_array"].tobytes()
    camera = {"run_id": "ac17"}
    attempt = SimpleNamespace(rig=None, retained_markers=[])
    new_domain = SimpleNamespace(proof={"source": "replacement-fixture"})
    monkeypatch.setattr(rig_raw_input, "camera_binding", lambda value: value)

    def begin_setup(subject, log):
        """Model the original identity binding while retaining failed attempt state."""
        assert subject is rig
        assert subject.__dict__ is not previous
        assert "geometry" not in subject.__dict__
        attempt.rig = subject
        attempt.retained_markers.append("setup-started")
        subject._raw_build = build
        subject.raw_baseline_sha256 = "a" * 64
        subject._raw_build_id = "replacement-operation"
        if failure == "ownership":
            raise ValueError("Synthetic ownership refusal")

    build = SimpleNamespace(camera=camera, raw_source={"sha256": "a" * 64},
        operation_id="replacement-operation", begin_setup=begin_setup)

    def reading(subject, *args, **kwargs):
        """Install fresh replacements without retaining any previous mutable field."""
        assert "_analysis_domain" not in subject.__dict__
        subject.shape_corrections = [{"new_mutable_array": np.array([5.0])}]
        subject.color_corrections = []
        subject.baseline = darsia.Image(np.array([[5.0, 6.0]]), dimensions=[1, 2])
        subject.shape_corrected_baseline = subject.baseline.copy()

    def depth(subject, *args, **kwargs):
        """A later proof refusal follows actual replacement state mutation."""
        subject.depth = darsia.Image(np.array([[0.03, 0.04]]), dimensions=[1, 2])

    def proof(subject, *args, **kwargs):
        """Keep the external native source/grid proof as an explicit unit boundary."""
        assert subject.baseline is not old_baseline
        assert subject.depth is not old_fields["depth"]
        if failure == "native_proof":
            raise ValueError("Synthetic native grid refusal")
        subject._analysis_domain = new_domain

    def labels(subject, *args, **kwargs):
        assert kwargs["apply_corrections"] is False
        subject.labels = darsia.Image(np.array([[3, 4]], dtype=np.uint8), dimensions=[1, 2])

    def facies(subject, *args, **kwargs):
        assert kwargs["apply_corrections"] is False
        subject.facies = subject.labels.copy()

    def properties(subject, *args, **kwargs):
        subject.porosity = darsia.Image(np.array([[0.4, 0.6]]), dimensions=[1, 2])
        subject.permeability = darsia.Image(np.array([[30.0, 40.0]]), dimensions=[1, 2])
        if failure == "properties":
            raise ValueError("Synthetic later property refusal")

    def geometry(subject):
        subject.geometry = darsia.Geometry(2, (1, 2), dimensions=[1, 2])

    def image_porosity(subject, *args, **kwargs):
        subject.image_porosity = darsia.Image(np.ones((1, 2)), dimensions=[1, 2])

    helper = ModuleType("darsia.fluidflower.rig_analysis_domain")
    helper.prepare_analysis_domain = proof
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    for name, method in {"setup_reading": reading, "setup_depth": depth,
                        "setup_labels": labels, "setup_facies": facies,
                        "setup_facies_props": properties, "setup_geometry": geometry,
                        "setup_image_porosity": image_porosity}.items():
        monkeypatch.setattr(Rig, name, method)
    monkeypatch.setattr(Rig, "setup_inner_labels", lambda *args, **kwargs: None)
    monkeypatch.setattr(Rig, "setup_boolean_image_porosity", lambda *args, **kwargs: None)
    monkeypatch.setattr(Rig, "setup_color_corrections", lambda *args, **kwargs: None)
    monkeypatch.setattr(darsia, "porosity_based_averaging",
                        lambda *args: darsia.ClipModel(min_value=0.0))
    kwargs = dict(experiment=SimpleNamespace(experiment_start=0),
        baseline_path=tmp_path / "baseline.ARW", depth_map_path=tmp_path / "depth.npz",
        labels_path=tmp_path / "labels.npz", facies_path=tmp_path / "facies.npz",
        raw_camera=camera, raw_baseline_sha256="a" * 64,
        raw_build_id="replacement-operation", raw_build=build,
        native_product={"selected": "synthetic-native-proof-boundary"})
    if prior_native:
        run_setup = lambda: rig.setup(**kwargs)
    else:
        bound = signature(Rig.setup).bind(rig, **kwargs)
        bound.apply_defaults()
        assert "native_product" not in bound.kwargs
        run_setup = lambda: Rig.setup(*bound.args, **bound.kwargs)
    if failure is None:
        run_setup()
        assert rig.__dict__ is not previous
        assert rig._analysis_domain is new_domain
        assert rig.baseline is not old_baseline
        assert rig._raw_build is build
        assert rig.geometry is not old_fields["geometry"]
        assert hasattr(rig, "upscaling")
    else:
        with pytest.raises(ValueError, match="Synthetic"):
            run_setup()
        assert rig.__dict__ is previous
        assert set(rig.__dict__) == set(old_fields)
        assert all(rig.__dict__[key] is value for key, value in old_fields.items())
        assert {name: getattr(rig, name).img.tobytes() for name in image_fields} == old_image_bytes
        assert rig.shape_corrections[0]["old_mutable_array"].tobytes() == old_correction_bytes
        assert rig.color_corrections[0]["old_mutable_list"] == [1, 2]
        if prior_native:
            with pytest.raises(ValueError, match="incomplete"):
                rig.geometry.integrate(old_pixels)
            after_total = rig.geometry.integrate_quantified(old_pixels)
            assert after_total.provenance == before_total.provenance == old_provenance
            assert after_total.unquantified_pixels == before_total.unquantified_pixels == 1
            assert after_total.full_value is None
            assert np.asarray(after_total.quantified_value).tobytes() == np.asarray(before_total.quantified_value).tobytes()
        else:
            assert np.asarray(rig.geometry.integrate(old_pixels)).tobytes() == np.asarray(before_total).tobytes()
    assert attempt.rig is rig
    assert attempt.retained_markers == ["setup-started"]
