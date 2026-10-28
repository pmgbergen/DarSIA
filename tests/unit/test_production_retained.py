"""Tiny synthetic N1 products against unchanged native calculation ASTs."""
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

SOURCES = Path(__file__).resolve().parents[2] / "src/darsia/fluidflower"
from darsia.fluidflower.mass import n1_arithmetic as retained
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset
from darsia.fluidflower.mass.production_precision import PixelMath

BASE_COMMIT = "6dfca622f5ed8ebe5c3ea6474025dcf5bf28023d"


def original_function(relative, name, namespace, *, class_name=None):
    """Use one pure historical math codeobject; never import its old runner."""
    repo = Path(__file__).resolve().parents[2]
    raw = subprocess.check_output(
        ["git", "show", BASE_COMMIT + ":src/darsia/fluidflower/" + relative],
        cwd=repo).decode("utf-8")
    tree = ast.parse(raw)
    body = tree.body if class_name is None else next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    context = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 relative, "exec"), context)
    return context[name]



def definitions(relative, names, namespace):
    tree = ast.parse((SOURCES / relative).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCES / relative), "exec"), namespace)
    return namespace


@pytest.fixture(params=["float32", "float64"])
def synthetic(request):
    shape = (3, 4)
    phase = np.array([[1, 2, 1, 0], [2, 1, 1, 2], [1, 0, 2, 255]], np.uint8)
    domain = phase != 255
    known = domain.copy()
    known[1, 2] = False
    bottom = np.zeros(shape, bool)
    bottom[0, 0] = True
    eligible = domain & ~bottom
    frame = SimpleNamespace(domain=domain, phase_labels=phase, known_physical_domain=known,
        mass_eligible_mask=eligible, depth_m=np.full(shape, .008), porosity=np.full(shape, .41),
        pixel_area_m2=np.full(shape, 1.e-6), excluded_masks={"bottom_strip": bottom},
        run_id="synthetic", frame_index=1, elapsed_h=9., nominal_injected_mass_g=1.,
        recipe={}, provenance={})
    value = np.arange(12).reshape(shape)
    physics = SimpleNamespace(solubility_co2=1.83+value*.031,
                              density_gaseous_co2=1.93+value*.065)
    parameters = dict(candidate="N1", gain=24.253536351793787, beta=.6448739280294578)
    response = dict(frame=frame, physics=physics, quantifiable=known.copy(),
        prepared=SimpleNamespace(aqueous_relative=np.asarray((value+.51)*.003, np.dtype(request.param))),
        linear_relative=np.zeros(shape, np.float32), swi=np.full(shape, .231),
        frozen_parameters=parameters, precision=request.param, pixel_backend="numpy_cpu",
        input_storage_dtype="float32-log-RGB/frozen-clearwater")
    from darsia.fluidflower.mass.concentration_joint import dilution_factor
    math = PixelMath(request.param)
    volume = np.where(known, frame.pixel_area_m2*frame.depth_m*frame.porosity, 0.)
    geometry_tree = ast.parse((SOURCES.parent / "measure/integration.py").read_text(encoding="utf-8"))
    geometry_class = next(node for node in geometry_tree.body if isinstance(node, ast.ClassDef) and node.name == "Geometry")
    integration = next(node for node in geometry_class.body if isinstance(node, ast.FunctionDef) and node.name == "integrate")
    geometry_ns = dict(np=np, darsia=SimpleNamespace(ExtensiveImage=type("Extensive", (), {})), __name__=__name__)
    annotation = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[annotation, integration], type_ignores=[])),
                 "unchanged native geometry integration", "exec"), geometry_ns)
    geometry = SimpleNamespace(space_dim=2, cached_voxel_volume=volume,
        _prepare_cached_voxel_volume=lambda shape: None)
    geometry.integrate = lambda array: geometry_ns["integrate"](geometry, array)
    def scalar(array, elapsed_h):
        return SimpleNamespace(img=np.asarray(array, np.float32), name="synthetic", date=None, time=elapsed_h)
    response["analysis"] = SimpleNamespace(geometry=geometry, _scalar=scalar)
    mass_tree = ast.parse((SOURCES.parent / "multiphase/mass_analysis.py").read_text(encoding="utf-8"))
    mass_class = next(node for node in mass_tree.body if isinstance(node, ast.ClassDef) and node.name == "CO2MassAnalysis")
    mass_method = next(node for node in mass_class.body if isinstance(node, ast.FunctionDef) and node.name == "mass_analysis")
    mass_ns = dict(np=np, darsia=SimpleNamespace(SimpleMassAnalysisResults=SimpleNamespace,
        full_like=lambda image, values: SimpleNamespace(img=np.full_like(image.img, values))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[annotation, mass_method], type_ignores=[])),
                 "unchanged native mass analysis", "exec"), mass_ns)
    physics.mass_analysis = lambda a, g: mass_ns["mass_analysis"](physics, a, g)
    partition = original_function("production_retained.py", "_main_partition", {})
    reference_retained = SimpleNamespace(
        _main_partition=partition, FIELDS=retained.FIELDS,
        _selected_n1=lambda *args: (dict(parameters), "a"*64))
    context = SimpleNamespace(
        modules={"engine": SimpleNamespace(r2=SimpleNamespace(dilution_factor=dilution_factor))},
        asset_reader=None)
    adapter = SimpleNamespace(context=context, math=math, precision=request.param,
                              retained=reference_retained, _facts=math.facts)
    original_evaluate = original_function(
        "production_precision_adapter.py", "evaluate_main", {},
        class_name="PrecisionAdapter")
    anchor = dict(elapsed_h=2.247, affected_pore_volume_m3=1.e-8)
    box1 = domain & (np.indices(shape)[0] < 1)
    domains = dict(full_rig=domain, box1=box1, box2=domain & ~box1)
    return SimpleNamespace(response=response, math=math, anchor=anchor,
        domains=domains, parameters=parameters,
        old=lambda: original_evaluate(
            adapter, response, anchor, selected_calibration={},
            domain_masks=domains, kept_fields=retained.FIELDS))


def evaluate(f, **kwargs):
    return retained.evaluate_main(f.response, f.anchor, parameters=f.parameters,
        pixel_math=f.math, calibration_sha256="a"*64, domain_masks=f.domains, **kwargs)


def test_native_fields_casts_and_domain_masses_are_bit_identical(synthetic):
    f = synthetic
    old = f.old()
    result = evaluate(f, kept_fields=retained.FIELDS)
    for name in retained.FIELDS:
        first, second = old["fields"][name], result["fields"][name]
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes(order="C") == second.tobytes(order="C")
    for name in ("total_mass_g", "aqueous_mass_g", "gas_mass_g", "dilution_factor", "affected_pore_volume_m3"):
        assert np.float64(old[name]).tobytes() == np.float64(result[name]).tobytes()
    for name, domain in f.domains.items():
        for mass, field in (("total_mass_g", "mass_density_g_l"),
                            ("aqueous_mass_g", "aqueous_mass_density_g_l"),
                            ("gas_mass_g", "gas_mass_density_g_l")):
            reference = float(f.response["analysis"].geometry.integrate(np.where(
                old["fields"]["main_mass_eligible_mask"] & domain, old["fields"][field], 0.)))*1000
            assert np.float64(reference).tobytes() == np.float64(result["domains"][name][mass]).tobytes()
    for name, part in result["excluded_mass"].items():
        for mass in ("total_mass_g", "aqueous_mass_g", "gas_mass_g"):
            assert np.float64(old["excluded_mass"][name][mass]).tobytes() == np.float64(part[mass]).tobytes()
    assert result["scalar_reference"] == "native_final_calibration_fields"
    assert result["calibration_sha256"] == "a"*64
    assert result["scientific_eligibility"] is False


def test_only_declared_maps_are_materialized_and_no_inputs_modified(synthetic):
    f = synthetic
    original = {name: array.tobytes() for name, array in vars(f.response["frame"]).items()
                if isinstance(array, np.ndarray)}
    prepared = f.response["prepared"].aqueous_relative.tobytes()
    result = evaluate(f, kept_fields={"c_aq_g_l", "s_g", "phase_labels"})
    assert set(result["fields"]) == {"c_aq_g_l", "s_g", "phase_labels"}
    assert not ({"E_lo_g", "E_hi_g", "U_rest_g", "paired_estimates", "excluded_parts"} & set(result))
    assert original == {name: array.tobytes() for name, array in vars(f.response["frame"]).items()
                        if isinstance(array, np.ndarray)}
    assert prepared == f.response["prepared"].aqueous_relative.tobytes()


@pytest.mark.parametrize("kind", ["field", "domain_missing", "domain_dtype", "domain_shape", "full_domain"])
def test_wrong_products_domains_or_freeze_fail_closed(synthetic, kind):
    kwargs = {}
    if kind == "field": kwargs["kept_fields"] = {"inferred_new_product"}
    if kind == "domain_missing": synthetic.domains.pop("box1")
    if kind == "domain_dtype": synthetic.domains["box1"] = synthetic.domains["box1"].astype(np.uint8)
    if kind == "domain_shape": synthetic.domains["box1"] = np.zeros((1, 1), bool)
    if kind == "full_domain": synthetic.domains["full_rig"] = np.zeros((3, 4), bool)
    with pytest.raises(ValueError): evaluate(synthetic, **kwargs)


def test_retained_path_has_no_secondary_or_summary_scientific_calls():
    source = (SOURCES / "mass/n1_arithmetic.py").read_text(encoding="utf-8")
    compile(source, str(SOURCES / "mass/n1_arithmetic.py"), "exec")
    tree = ast.parse(source)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert not calls & {"_joint_signal", "SimpleFlash", "summarize_round3_response", "_native_histogram",
                        "_pairs", "distances_for_part", "build_estimate_geometry", "prepare_own_colour_response"}








def test_main_export_zeroes_excluded_only_and_preserves_raw_native_maps(synthetic):
    result = evaluate(synthetic, kept_fields=retained.FIELDS)
    before = {name: value.tobytes() for name, value in result["fields"].items()}
    exported = retained.main_products(result, synthetic.math)
    assert set(exported["products"]) == {"concentration_aqueous", "saturation_free_gas", "phase_labels",
                                         "mass_total", "mass_aqueous", "mass_free_gas"}
    assert exported["products"]["concentration_aqueous"] is result["fields"]["c_aq_g_l"]
    mask = result["fields"]["main_mass_eligible_mask"]
    for name, native in (("mass_total", "mass_density_g_l"), ("mass_aqueous", "aqueous_mass_density_g_l"),
                         ("mass_free_gas", "gas_mass_density_g_l")):
        projected = exported["products"][name]
        raw = result["fields"][native]
        assert projected.dtype == synthetic.math.dtype
        assert np.all(projected[~mask] == 0) and projected[mask].tobytes() == raw[mask].tobytes()
        assert projected.tobytes() == np.where(mask, raw, 0.).tobytes()
    assert result["fields"]["mass_density_g_l"][0, 0] > 0  # Real excluded mass stays in the raw view.
    assert before == {name: value.tobytes() for name, value in result["fields"].items()}
    assert exported["units"]["mass_total"] == "g/L"
    assert exported["required_transport_weight_units"] == "L pore volume per pixel"
    assert exported["scientific_eligibility"] is False


def test_separate_total_cast_is_not_sum_of_cast_components(synthetic):
    # An exact half-ULP witness: gas rounds to 1, water to 2**-24, while
    # their float64 sum rounds to the next float32 above 1.
    synthetic.response["swi"][0, 1] = .5
    synthetic.response["physics"].density_gaseous_co2[0, 1] = 2. + 2.**-23
    synthetic.response["physics"].solubility_co2[0, 1] = 2.**-23
    result = evaluate(synthetic, kept_fields=retained.FIELDS)
    fields = result["fields"]
    physics = synthetic.response["physics"]
    aqueous = fields["c_aq_relative"].astype(np.float32)
    gas = fields["s_g"].astype(np.float32)
    gas[~np.isfinite(gas)] = 0
    raw_g = physics.density_gaseous_co2*gas
    raw_aq = physics.solubility_co2*aqueous*np.clip(1-gas, 0, None)
    # The canonical CO2MassAnalysis performs the original separate float32
    # image casts after its float64 thermodynamic operands. The typed N1
    # provider separately keeps the explicitly chosen working-map precision.
    analysis = synthetic.response["analysis"]
    mass = physics.mass_analysis(
        analysis._scalar(aqueous, synthetic.response["frame"].elapsed_h),
        analysis._scalar(gas, synthetic.response["frame"].elapsed_h))
    assert mass.mass.img.dtype == np.float32
    assert mass.mass.img.tobytes() == (raw_g+raw_aq).astype(np.float32).tobytes()
    assert mass.mass.img.tobytes() != (mass.mass_aq.img+mass.mass_g.img).tobytes()

@pytest.mark.parametrize("kind", [
    "zero_gain", "negative_gain", "nonfinite_gain", "negative_beta",
    "gas_parameter", "other_candidate", "sha", "size"])
def test_verified_calibration_parameters_fail_closed(tmp_path, kind):
    """Use canonical metadata verification and scientific N1 admission."""
    parameters = dict(candidate="N1", gain=23.882294164545023, beta=.6143584288103799)
    if kind == "zero_gain": parameters["gain"] = 0.
    if kind == "negative_gain": parameters["gain"] = -1.
    if kind == "nonfinite_gain": parameters["gain"] = float("inf")
    if kind == "negative_beta": parameters["beta"] = -1.
    if kind == "gas_parameter": parameters["gas_scale"] = 1.
    if kind == "other_candidate": parameters["candidate"] = "N2"
    path = tmp_path / "calibration.json"
    # Nonfinite scientific parameters are rejected before JSON publication.
    if kind == "nonfinite_gain":
        with pytest.raises(ValueError):
            retained.validate_parameters(parameters)
        return
    path.write_text(json.dumps({"parameters": parameters}), encoding="utf-8")
    bound = asset(path)
    if kind == "sha": bound["sha256"] = "f"*64
    if kind == "size": bound["size_bytes"] += 1
    with pytest.raises(ValueError):
        selected = DataReader().json(bound)["parameters"]
        retained.validate_parameters(selected)


def test_calibration_gain_is_read_from_verified_data_not_baseline_constant(tmp_path):
    """A genuine calibration descriptor supplies the selected positive pair."""
    parameters = dict(candidate="N1", gain=23.882294164545023, beta=.6143584288103799)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"parameters": parameters}), encoding="utf-8")
    bound = asset(path)
    selected = DataReader().json(bound)["parameters"]
    retained.validate_parameters(selected)
    assert selected == parameters
    assert selected["gain"] != 24.253536351793787
