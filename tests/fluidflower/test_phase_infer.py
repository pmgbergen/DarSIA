"""Tiny exact comparisons with the pre-extraction native and rig call bodies.

Only model/file admission and imports are replaced in the isolated reference.
The original numerical statements, casts, branches and matrix products execute
on synthetic data; no campaign input or saved scientific array is read.
"""

from __future__ import annotations

import ast
import copy
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from scipy.special import expit
from threadpoolctl import threadpool_limits

from darsia.fluidflower.native.support import compute_feature_support
from darsia.fluidflower.phase import features, heads, infer, metrics, regional, rules
from test_phase_features_heads_regional import (
    PARAMETERS,
    assert_exact,
    models,
    original_namespace,
    original_source,
)
from test_phase_rules import originals, parameters as rule_parameters
from test_phase_domain import parameters as domain_parameters


class _InjectImports(ast.NodeTransformer):
    """Bind original numerical dependencies without importing campaign loaders."""

    def __init__(self, alignment=False):
        self.alignment = alignment

    def visit_ImportFrom(self, node):
        if self.alignment and any(
            a.name == "align_temporal_inputs" for a in node.names
        ):
            return ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id="align_temporal_inputs", ctx=ast.Store())],
                    value=ast.Attribute(
                        value=ast.Name(id="temporal", ctx=ast.Load()),
                        attr="align_temporal_inputs",
                        ctx=ast.Load(),
                    ),
                ),
                node,
            )
        return None


def _definitions(filename, names, env, *, alignment=False):
    selected = [
        copy.deepcopy(n)
        for n in ast.parse(original_source(filename)).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    tree = ast.Module(selected, type_ignores=[])
    tree = _InjectImports(alignment).visit(tree)
    exec(compile(ast.fix_missing_locations(tree), filename, "exec"), env)
    return SimpleNamespace(**env)


def _data(rule_parameters):
    p = copy.deepcopy(PARAMETERS)
    p["regional"]["pooling"]["review_denominator"] = 255
    p["decisions"] = copy.deepcopy(rule_parameters)
    p["feature_support"] = dict(
        homography_epsilon=1e-9, coverage_tolerance=1e-6, texture_footprint_hw=[5, 5]
    )
    p["native"] = dict(
        threads=1,
        maximum_threads=16,
        block_rows=3,
        input_encoding="v5-unit-float32",
        gas_feature_count=4,
        flatfield_source_origin_xy=[0, 0],
        flatfield_target_origin_xy=[0, 0],
        square_metres_to_square_millimetres=1e6,
        provenance={},
    )
    p["edge"] = dict(
        block_rows=128,
        input_encoding="v5-unit-float32",
        pixel_threshold=0.7,
        regional_threshold=0.6,
        minimum_area_mm2=2.5,
        gas_connectivity=4,
        provenance={},
    )
    return p


def _bundle(p):
    model, oldmodel, oldfeatures = models(p)
    # Make both partitions populated while retaining nonconstant optical weights.
    for head, old in zip(
        (model.co2_presence, model.gas_within_co2),
        (oldmodel.co2_presence, oldmodel.gas_within_co2),
    ):
        object.__setattr__(
            head, "intercept", 2.0 if head is model.co2_presence else 1.2
        )
        old.intercept = head.intercept
    rp = p["regional"]
    fit = dict(
        mean=np.linspace(-0.02, 0.03, 20).tolist(),
        scale=np.linspace(0.7, 1.2, 20).tolist(),
        coefficients=np.linspace(-0.12, 0.15, 20).tolist(),
        intercept=1.5,
    )
    regional_model = dict(
        format=rp["model_format"],
        feature_names=rp["feature_names"],
        settings=rp["settings"],
        heads={"co2_presence": fit, "gas_within_co2": fit},
    )
    refit = copy.deepcopy(regional_model)
    refit["heads"]["gas_within_co2"]["intercept"] = 0.65
    gas = dict(
        variant="G3",
        feature_names=p["heads"]["flat_feature_names"],
        fit=dict(
            mean=[0.01, -0.02, 0.03, -0.01],
            scale=[0.5, 0.7, 0.9, 1.1],
            coefficients=[0.3, -0.2, 0.1, 0.4],
            intercept=0.9,
        ),
    )
    profile = {
        rp["profile_keys"][key]: value for key, value in rp["guard_defaults"].items()
    }
    grid = p["decisions"]["gas_configuration"]
    settings = rules.GasDecisionSettings(
        grid["format"], 0.7, "refit", 0.6, 2.5, 4, parameters=p["decisions"]
    )
    return (
        dict(
            original_model=model,
            gas_head=gas,
            regional_original=regional_model,
            regional_refit=refit,
            regional_edge=copy.deepcopy(refit),
            profile=profile,
            gas_decision_settings=settings,
            bindings={},
        ),
        oldmodel,
        oldfeatures,
    )


def _original_math(p, oldmodel, oldfeatures, originals):
    """Assemble unchanged old pure dependencies and call bodies in one test scope."""
    fp, rp = p["features"], p["regional"]
    backend = original_namespace(
        "tools/phase_inference_benchmark_backends.py",
        {"packed_parameters", "CudaFeaturePredictor", "CpuFeaturePredictor"},
        OpticalFeatureContract=oldfeatures.OpticalFeatureContract,
        relative_bright_grain_columns=oldfeatures.relative_bright_grain_columns,
        GENERIC_FEATURE_VERSION=oldfeatures.GENERIC_FEATURE_VERSION,
        RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION=oldfeatures.RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION,
        BASELINE_RELATIVE_FEATURE_VERSION=oldfeatures.BASELINE_RELATIVE_FEATURE_VERSION,
        SPATIAL_CONTEXT_FRACTION=oldfeatures.SPATIAL_CONTEXT_FRACTION,
    )
    temporal = original_namespace("tools/ffac_v5_temporal_alignment.py")
    oldregional = original_namespace(
        "phase_optical_regions.py",
        VERSION=rp["model_format"],
        FEATURE_NAMES=tuple(rp["feature_names"]),
    )
    context = original_namespace(
        "phase_gas_regional_context.py",
        regional=oldregional,
        GAS_DEPENDENT_COLUMNS=(1, 11),
    )
    display = SimpleNamespace(
        M_NATURAL=np.asarray(fp["flatfield"]["natural_matrix"], np.float64)
    )
    flat = original_namespace(
        "phase_gas_flatfield.py",
        display=display,
        _h=temporal._h,
        BOX_SIZE=5,
        DENOMINATOR_FLOOR=np.float32(1e-6),
        _BOX_KERNEL=np.ones((5, 5), np.uint8),
        _LUMINANCE=np.asarray(fp["flatfield"]["luminance"], np.float32),
    )
    gas_head = original_namespace(
        "tools/train_ffac_gas_flatfield_variants.py", {"gas_probability"}
    )
    profile_refs = {}
    for key, filename, function in (
        ("gas", "phase_regional_gas.py", "confirm_gas"),
        ("output", "phase_region_output_guards.py", "guard_labels"),
        ("free_gas", "phase_free_gas_output_guard.py", "clean_labels"),
    ):
        profile_refs[key] = original_namespace(
            filename,
            {"Settings", function},
            VERSION=rp["formats"][key],
            PROFILE_KEY=rp["profile_keys"][key],
        )
    env = dict(
        np=np,
        cv2=cv2,
        expit=expit,
        Path=Path,
        time=time,
        threadpool_limits=threadpool_limits,
        temporal=temporal,
        regional=oldregional,
        flat=flat,
        CpuFeaturePredictor=backend.CpuFeaturePredictor,
        build_shared_regional_context=context.build_shared_regional_context,
        prepare_conditional_gas=context.prepare_conditional_gas,
        gas_probability=gas_head.gas_probability,
        gas=profile_refs["gas"],
        guards=profile_refs["output"],
        cleanup=profile_refs["free_gas"],
        compute_feature_support=lambda **kw: compute_feature_support(
            **kw, parameters=p["feature_support"]
        ),
        apply_g3_configuration=originals["phase_g3_production"].apply_g3_configuration,
        probability_encoding_contract=lambda doc, shape: doc,
        NativeCentreResult=infer.NativeCentreResult,
        sha256_file=lambda path: "synthetic-code-provenance",
        __file__="synthetic-original",
        CONTRACT="synthetic-native-contract",
    )
    native = _definitions(
        "phase_ffac_selected_native_centre.py",
        {
            "decode_native_log",
            "encode_decode_probabilities",
            "original_co2_labels",
            "preminimum_gas_selection",
            "NativeCentreEvaluator",
        },
        env,
        alignment=True,
    )
    # Use the original method on supplied models; its file-admission constructor
    # cannot accept a deliberately synthetic model fixture.
    evaluator = native.NativeCentreEvaluator.__new__(native.NativeCentreEvaluator)
    evaluator.original = oldmodel
    evaluator.threads, evaluator.block_rows = 1, 3
    evaluator.binding = {}
    evaluator._flat_key = None
    evaluator._fixed_baseline_decode_cache = SimpleNamespace(
        decode=lambda key, log, valid: native.decode_native_log(log, valid),
        validate_fixed=lambda *args: None,
        receipt=lambda: {},
    )
    # RequiredConventionCentre dynamically replaces only its two original seams.
    # Access through this namespace lets its unmodified body affect the method.
    native_proxy = SimpleNamespace(original_co2_labels=env["original_co2_labels"])
    original_co2 = env["original_co2_labels"]
    env["original_co2_labels"] = lambda *args, **kw: native_proxy.original_co2_labels(
        *args, **kw
    )
    required = _definitions(
        "phase_ffac_selected_policy_a_required_inference.py",
        {"RequiredConventionCentre"},
        dict(
            np=np,
            ExitStack=ExitStack,
            patch=patch,
            native=native_proxy,
            temporal=temporal,
            array_identity=metrics.array_pair_identity,
            bound_json=lambda descriptor: descriptor,
        ),
    )
    rig_env = dict(
        np=np,
        cv2=cv2,
        expit=expit,
        flat=flat,
        M_NATURAL=display.M_NATURAL,
        CpuFeaturePredictor=backend.CpuFeaturePredictor,
        SPATIAL_CONTEXT_FRACTION=oldfeatures.SPATIAL_CONTEXT_FRACTION,
        regional=oldregional,
        gas_probability=gas_head.gas_probability,
        build_shared_regional_context=context.build_shared_regional_context,
        prepare_conditional_gas=context.prepare_conditional_gas,
        validate_pixel_area=originals["phase_pixel_area"].validate_pixel_area,
        area_filtered_gas=originals["phase_pixel_area"].area_filtered_gas,
        require_frozen_heads=lambda candidate: None,
        json=json,
        CONTRACT="synthetic-rig-contract",
    )
    rig = _definitions(
        "phase_rig_domain_inference.py",
        {
            "log_to_unit_rgb",
            "normalized_gaussian",
            "MaskedCpuFeaturePredictor",
            "masked_flatfield_features",
            "frozen_native_partition",
            "require_categorical_feature_support",
            "weighted_co2_output_guard",
            "infer_native",
        },
        rig_env,
    )
    return SimpleNamespace(
        native=evaluator,
        wrapper=required.RequiredConventionCentre,
        rig=rig,
        rig_env=rig_env,
        old_co2=original_co2,
        temporal=temporal,
    )


def _logs(*, previous=True, extended=False):
    rng = np.random.default_rng(197)
    shape = (11, 13)
    baseline = np.log(rng.uniform(120.0, 2200.0, (3, *shape))).astype(np.float32)
    current = (baseline + np.float32(0.015) * rng.normal(size=baseline.shape)).astype(
        np.float32
    )
    current[:, 5, 6] = np.nan
    valid = np.isfinite(current).all(0)
    baseline_valid = np.isfinite(baseline).all(0)
    predecessor = (
        (baseline + np.float32(0.003)).astype(np.float32) if previous else None
    )
    origin = (0, 0)
    temporal_base = baseline
    temporal_valid = baseline_valid
    if extended:
        origin = (-1, -1)
        temporal_base = np.pad(baseline, ((0, 0), (1, 1), (1, 1)), mode="edge")
        temporal_valid = np.ones(temporal_base.shape[1:], bool)
        if predecessor is not None:
            predecessor = np.pad(predecessor, ((0, 0), (1, 1), (1, 1)), mode="edge")
    photo = np.ones(shape, bool)
    photo[:2, :3] = False
    return dict(
        current_log=current,
        current_valid=valid,
        source_baseline_log=temporal_base,
        baseline_valid=temporal_valid,
        current_h=np.eye(3),
        photo_sand_mask=photo,
        probability_spacing_m=(0.0004, 0.0006),
        inherited_spacing_m=(0.001, 0.0012),
        source_origin_xy=origin,
        flat_baseline_log=baseline,
        flat_baseline_valid=baseline_valid,
        previous_log=predecessor,
        previous_valid=None if predecessor is None else np.isfinite(predecessor).all(0),
        previous_h=None if predecessor is None else np.eye(3),
        previous_origin_xy=origin,
        input_provenance={"fixture": "synthetic"},
    )


def test_fixed_reference_cache_preserves_native_arrays_and_rejects_changed_key(
    rule_parameters,
):
    """Cold, warm and changed temporal inputs retain the fresh numerical path."""
    from functools import partial
    from darsia.fluidflower.pipeline.phase_inputs import FixedBaselineDecodeCache

    p = _data(rule_parameters)
    bundle, _, _ = _bundle(p)
    args = _logs(previous=True, extended=True)
    current = args.pop("current_log")
    ip = json.loads(
        (
            Path(__file__).parent / "fixtures" / "phase_input_parameters.json"
        ).read_bytes()
    )
    cp = ip["fixed_baseline_cache"]
    p["native"].update(
        temporal_cache_role="fixed_temporal_baseline",
        flat_cache_role=cp["validation_role"],
    )
    cache = FixedBaselineDecodeCache(max_bytes=32768, parameters=cp)
    cached_bundle = dict(
        bundle,
        fixed_baseline_decode_cache=cache,
        fixed_baseline_decoder=partial(
            features.decode_native_log, parameters=p["features"]
        ),
        flatfield_reference=features.prepare_flatfield_reference(
            args["flat_baseline_log"],
            source_valid_mask=args["flat_baseline_valid"],
            tile_rows=p["native"]["block_rows"],
            parameters=p["features"],
        ),
    )
    outputs = (
        "labels",
        "feature_support",
        "raw_feature_support",
        "domain",
        "preminimum_gas",
        "g3_pixel",
        "regional_refit",
        "flatfield_valid",
    )
    fresh = infer.infer_native_centre(
        current,
        **args,
        labelled_probability_encoding=False,
        model_bundle=bundle,
        parameters=p,
    )
    for _ in range(2):
        actual = infer.infer_native_centre(
            current,
            **args,
            labelled_probability_encoding=False,
            model_bundle=cached_bundle,
            parameters=p,
        )
        for name in outputs:
            assert_exact(getattr(actual, name), getattr(fresh, name))
    receipt = actual.provenance["fixed_baseline_decode_cache"]
    assert (
        receipt["misses"],
        receipt["hits"],
        receipt["validation_misses"],
        receipt["validation_hits"],
    ) == (1, 1, 1, 1)
    changed = args["source_baseline_log"].copy()
    changed[:, 3, 3] += np.float32(0.001)
    args["source_baseline_log"] = changed
    fresh = infer.infer_native_centre(
        current,
        **args,
        labelled_probability_encoding=False,
        model_bundle=bundle,
        parameters=p,
    )
    actual = infer.infer_native_centre(
        current,
        **args,
        labelled_probability_encoding=False,
        model_bundle=cached_bundle,
        parameters=p,
    )
    for name in outputs:
        assert_exact(getattr(actual, name), getattr(fresh, name))
    assert actual.provenance["fixed_baseline_decode_cache"]["misses"] == 2


@pytest.mark.parametrize(
    "previous,labelled,extended",
    [
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_native_executes_original_two_spacing_pipeline(
    rule_parameters, originals, previous, labelled, extended, monkeypatch
):
    p = _data(rule_parameters)
    bundle, oldmodel, oldfeatures = _bundle(p)
    old = _original_math(p, oldmodel, oldfeatures, originals)
    old.native.regional, old.native.refit = (
        bundle["regional_original"],
        bundle["regional_refit"],
    )
    old.native.g3, old.native.profile = bundle["gas_head"], bundle["profile"]
    data = _logs(previous=previous, extended=extended)
    stage_calls, product_calls = [], []
    co2, shared = regional.original_co2_labels, regional.build_shared_regional_context
    packed, presence = heads.score_packed_heads, heads.score_packed_presence

    def inherited(*args, **kwargs):
        stage_calls.append(("inherited", tuple(args[5])))
        return co2(*args, **kwargs)

    def gas_context(*args, **kwargs):
        stage_calls.append(("gas", tuple(args[4])))
        return shared(*args, **kwargs)

    def both(*args):
        product_calls.append("two")
        return packed(*args)

    def single(*args):
        product_calls.append("single")
        return presence(*args)

    monkeypatch.setattr(regional, "original_co2_labels", inherited)
    monkeypatch.setattr(regional, "build_shared_regional_context", gas_context)
    monkeypatch.setattr(heads, "score_packed_heads", both)
    monkeypatch.setattr(heads, "score_packed_presence", single)
    actual = infer.infer_native_centre(
        **data,
        labelled_probability_encoding=labelled,
        model_bundle=bundle,
        parameters=p,
    )
    wrapped = old.wrapper(
        old.native,
        {"spacing_m": data["inherited_spacing_m"]},
        {"metadata": {"spacing_m": data["probability_spacing_m"]}},
    )
    named = dict(data)
    log = named.pop("current_log")
    named["H_B_current"] = named.pop("current_h")
    named["H_B_previous"] = named.pop("previous_h")
    named.pop("probability_spacing_m")
    named.pop("inherited_spacing_m")
    # Use axes-only fallback for the compact case, never inject old probabilities.
    if labelled:
        named["probability_receipt"] = dict(
            layout="native-float16",
            metadata={"spacing_m": data["probability_spacing_m"]},
        )
    expected = wrapped.infer(log, **named, labelled_probability_encoding=labelled)
    for field in (
        "labels",
        "feature_support",
        "raw_feature_support",
        "domain",
        "preminimum_gas",
        "g3_pixel",
        "regional_refit",
        "flatfield_valid",
    ):
        assert_exact(getattr(actual, field), getattr(expected, field))
    for key in (
        "co2_context",
        "gas_configuration",
        "probability_encoding",
        "feature_support_pixels",
        "domain_pixels",
        "flat_valid_pixels",
        "nominal_pixel_area_mm2",
        "temporal_alignment",
    ):
        assert actual.provenance[key] == expected.provenance[key]
    assert stage_calls == [
        ("inherited", data["inherited_spacing_m"]),
        ("gas", data["probability_spacing_m"]),
    ]
    assert product_calls == ["two", "single"] * 4
    assert actual.provenance["nominal_pixel_area_mm2"] == 0.24
    assert actual.provenance["independent_alignment_audit"][
        "previous_rgb_bit_identical"
    ]
    assert actual.preminimum_gas.dtype == bool
    assert not np.array_equal(actual.feature_support, actual.raw_feature_support)


class _ModelFiles:
    """Read model mappings in place of original file admission, with no IO."""

    def __init__(self, models, key=""):
        self.models, self.key = models, key

    def __truediv__(self, key):
        return _ModelFiles(self.models, key)

    def read_text(self):
        return json.dumps(self.models[self.key])


@pytest.mark.parametrize("previous", [False, True])
def test_rig_executes_original_masked_pipeline_and_measured_area(
    rule_parameters, originals, previous, monkeypatch
):
    p = _data(rule_parameters)
    bundle, oldmodel, oldfeatures = _bundle(p)
    old = _original_math(p, oldmodel, oldfeatures, originals)
    data = _logs(previous=previous)
    current, baseline = data["current_log"], data["flat_baseline_log"]
    domain = np.ones(current.shape[1:], bool)
    domain[[0, -1]] = False
    domain[:, [0, -1]] = False
    # A varying measured map makes substitution of nominal area observable.
    area = np.linspace(0.3, 1.0, domain.size).reshape(domain.shape)
    files = {
        "regional_model.json": bundle["regional_edge"],
        "models/g3/gas_head.json": bundle["gas_head"],
    }
    old.rig_env.update(
        Path=lambda candidate: _ModelFiles(files),
        HierarchicalPhaseModel=SimpleNamespace(load=lambda root: oldmodel),
    )
    observed_rgb, joint_inputs, gas_masks = [], [], []
    decode, conditional, area_filter = (
        features.log_to_unit_rgb,
        regional.prepare_conditional_gas,
        infer.area_filtered_gas,
    )

    def decoded(*args, **kwargs):
        result = decode(*args, **kwargs)
        observed_rgb.append(result[0].copy())
        return result

    def prepared(context, values, **kwargs):
        joint_inputs.append(values.copy())
        return conditional(context, values, **kwargs)

    def filtered(mask, actual_area, **kwargs):
        assert actual_area is area
        assert kwargs == dict(minimum_area_mm2=2.5, connectivity=4)
        gas_masks.append(mask.copy())
        return area_filter(mask, actual_area, **kwargs)

    monkeypatch.setattr(features, "log_to_unit_rgb", decoded)
    monkeypatch.setattr(regional, "prepare_conditional_gas", prepared)
    monkeypatch.setattr(infer, "area_filtered_gas", filtered)
    args = dict(
        aligned_previous_log=data["previous_log"],
        spacing_m=(0.001, 0.0012),
        pixel_area_mm2=area,
        source_baseline_log=baseline,
        baseline_to_current_h=np.eye(3),
        source_baseline_valid=data["flat_baseline_valid"],
        feature_shape_hw=(2979, 5472),
    )
    actual = infer.infer_rig_edge(
        current, baseline, domain, **args, model_bundle=bundle, parameters=p
    )
    oldargs = dict(args)
    oldargs["h_baseline_to_current"] = oldargs.pop("baseline_to_current_h")
    expected = old.rig.infer_native(
        "synthetic-models",
        current,
        baseline,
        domain,
        **oldargs,
        return_decision_layers=True,
    )
    assert_exact(actual[0], expected[0])
    assert_exact(actual[1], expected[1])
    for key in (
        "co2",
        "co2_islands",
        "gas_minimum",
        "categorical_feature_support",
        "optical_spatial_sigma_px",
        "optical_context_native_shape_hw",
    ):
        assert actual[2][key] == expected[2][key]
    for key in actual[3]:
        assert_exact(actual[3][key], expected[3][key])
    valid = domain & np.isfinite(current).all(0) & np.isfinite(baseline).all(0)
    assert_exact(observed_rgb[0], old.rig.log_to_unit_rgb(current, valid)[0])
    native_decoder = original_namespace(
        "phase_ffac_selected_native_centre.py", {"decode_native_log"}
    )
    native = native_decoder.decode_native_log(current, np.isfinite(current).all(0))
    assert np.any(observed_rgb[0][valid] != native[valid])
    assert len(joint_inputs) == 2 and not np.array_equal(
        joint_inputs[0], joint_inputs[1]
    )
    assert np.array_equal(gas_masks[0], actual[3]["pre_minimum_gas"])
    assert actual[2]["optical_spatial_sigma_px"] == 4.4685
    assert (actual[0][~actual[1]] == 255).all()


def test_independent_alignment_masks_remain_separate_and_require_bit_identity(
    monkeypatch,
):
    p = _data({})
    rgb = np.full((7, 9, 3), 0.2, np.float32)
    valid = np.ones((7, 9), bool)
    valid[3, 4] = False
    calls = []
    align = features.align_temporal_inputs

    def observed(**kwargs):
        calls.append(
            (kwargs["baseline_source_valid_mask"], kwargs["previous_source_valid_mask"])
        )
        return align(**kwargs)

    monkeypatch.setattr(features, "align_temporal_inputs", observed)
    args = dict(
        baseline=rgb,
        current=rgb,
        previous=rgb,
        h_baseline_to_current=np.eye(3),
        h_baseline_to_previous=np.eye(3),
        required_valid_mask=valid,
        baseline_origin_xy=(0, 0),
        previous_origin_xy=(0, 0),
        baseline_source_valid_mask=valid,
        previous_source_valid_mask=valid,
    )
    infer._independent_alignments(args, feature_parameters=p["features"])
    assert calls[0] == (None, None)
    assert calls[1][0] is valid and calls[1][1] is None
    count = 0

    def changed(**kwargs):
        nonlocal count
        result = align(**kwargs)
        count += 1
        if count == 2:
            result[0][0, 0, 0] = np.nextafter(result[0][0, 0, 0], np.float32(1))
        return result

    monkeypatch.setattr(features, "align_temporal_inputs", changed)
    with pytest.raises(ValueError, match="sharing scores is forbidden"):
        infer._independent_alignments(args, feature_parameters=p["features"])


def test_native_tile_mismatch_and_rig_missing_area_fail_before_decisions(
    rule_parameters, monkeypatch
):
    p = _data(rule_parameters)
    bundle, _, _ = _bundle(p)

    def wrong_tiles(log, registered, **kwargs):
        shape = log.shape[1:]
        yield 1, 4, np.zeros((3, shape[1], 4), np.float32), np.ones((3, shape[1]), bool)

    monkeypatch.setattr(features, "iter_flatfield_feature_tiles", wrong_tiles)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid admission reached a decision stage")

    monkeypatch.setattr(regional, "original_co2_labels", forbidden)
    with pytest.raises(ValueError, match="tile rows differ"):
        infer.infer_native_centre(
            **_logs(previous=False),
            labelled_probability_encoding=False,
            model_bundle=bundle,
            parameters=p,
        )
    with pytest.raises(ValueError, match="physical area"):
        infer.infer_rig_edge(
            None,
            None,
            None,
            aligned_previous_log=None,
            spacing_m=None,
            pixel_area_mm2=None,
            source_baseline_log=None,
            baseline_to_current_h=None,
            source_baseline_valid=None,
            feature_shape_hw=None,
            model_bundle=bundle,
            parameters=p,
        )


def _full_inputs(rule_parameters, domain_parameters):
    """A real same-grid acquisition with an empty phase and identity lamp rule."""
    p = _data(rule_parameters)
    p["domain"] = copy.deepcopy(domain_parameters)
    shape = (11, 13)
    p["geometry"] = dict(
        source_box_xywh=[0, 0, 13, 11],
        native_shape_hw=list(shape),
        source_origin_xy=[0, 0],
        unit_denominator=65535.0,
        curve_homography_floor=1e-12,
        floor_source_origin_xy=[0, 0],
        rig_origin_xy=[0, 0],
        lower_zone_id=4,
        mask_border_value=0,
    )
    p["placement"] = dict(
        light_epoch_values=["rig_light", "new_lamps"],
        identity_epoch="rig_light",
        base_model_key="P1",
        projected_model_key="P3",
        regional_feature_count=20,
        projected_applied_audit_key="projection_applied",
    )
    p["chain"] = dict(
        rig_spacing_m=[0.001, 0.0012],
        probability_spacing_m=[0.0004, 0.0006],
        empty_previous_origin_xy=[0, 0],
        unresolved_label=255,
        source_digest_length=64,
        placement_mode=rule_parameters["placement"]["placement_modes"]["regional_base"],
        first_envelope_ordinal=1,
        second_envelope_ordinal=2,
        contact_step="B",
        rule_backend="cpu",
        delta=1.14,
        square_millimetres_per_square_centimetre=100.0,
        postprocessing_provenance={},
        changed_source_provenance={},
        changed_source_seam_provenance={},
    )
    bundle, _, _ = _bundle(p)
    for head in (
        bundle["original_model"].co2_presence,
        bundle["original_model"].gas_within_co2,
    ):
        object.__setattr__(head, "coefficients", np.zeros_like(head.coefficients))
        object.__setattr__(head, "intercept", -100.0)
    bundle["gas_head"]["fit"].update(coefficients=[0.0] * 4, intercept=-100.0)
    for name in ("regional_original", "regional_refit", "regional_edge"):
        for fit in bundle[name]["heads"].values():
            fit.update(coefficients=[0.0] * 20, intercept=-100.0)
    full = np.log(np.full((3, *shape), 300.0, np.float32)).astype(np.float32)
    valid = np.ones(shape, bool)
    native = SimpleNamespace(log=full, valid=valid, origin_xy=(0, 0))
    context = SimpleNamespace(
        rig_domain=valid,
        hardware_excluded=np.zeros(shape, bool),
        parameters={"fixture_geometry": "same admitted grid"},
    )
    source = dict(
        previous_source_sha256=None,
        source_sha256="a" * 64,
        studio_anchor=False,
        parent_source={"source_changed": False},
        elapsed_h=0.0,
        run="synthetic",
        selected=True,
    )
    frame = dict(
        source=source,
        full_log=full,
        full_valid=valid,
        context=context,
        native_input=native,
        current_h=np.eye(3),
        temporal_input=native,
        photo_sand=valid,
        inherited_spacing_m=[0.001, 0.0012],
        probability_spacing_m=[0.0004, 0.0006],
        labelled_probability_encoding=True,
        encoding_contract=None,
        input_provenance={"synthetic": True},
        observed=valid,
        original_area=np.ones(shape, np.float64),
        pixel_area_mm2=np.ones(shape, np.float64),
        prior_cache=None,
        components=dict(
            open_components=np.ones(shape, np.uint8), barrier_mask=np.zeros(shape, bool)
        ),
        source_component_ids=[1],
        light_epoch_id="rig_light",
        placement_metadata={},
        reference_to_current_h=np.eye(3),
        top_keep=valid,
        history_policy="ordered unique acquisitions",
        barrier_proof={"synthetic": True},
        physical_scale_revision={"synthetic": True},
        scientific={"synthetic": True},
    )
    all_models = dict(
        native=bundle,
        edge=bundle,
        baseline=dict(
            full_log=full, full_valid=valid, native_input=native, temporal_input=native
        ),
        placement=dict(
            original_model=bundle["original_model"],
            pixel_model=None,
            regional_model=None,
        ),
        floor_curve=np.full(shape[1], shape[0] - 1, np.float64),
        zones=np.full(shape, 4, np.uint8),
        envelopes=dict(
            env1=np.zeros(shape, bool),
            env2=np.zeros(shape, bool),
            u1_elapsed_h=0.5,
            u2_elapsed_h=1.0,
        ),
    )
    return frame, all_models, p


def test_full_infer_labels_runs_actual_empty_phase_math(
    rule_parameters, domain_parameters
):
    frame, bundle, p = _full_inputs(rule_parameters, domain_parameters)
    before = frame["full_log"].copy()
    result = infer.infer_labels(frame, None, bundle, p)
    assert_exact(result["labels"], np.zeros((11, 13), np.uint8))
    assert_exact(frame["full_log"], before)
    populations = result["classes"]
    for name in (
        "modelled",
        "filled_observed",
        "filled_unobserved",
        "rig_domain",
        "hardware_excluded",
        "below_sand_floor",
    ):
        assert populations[name].dtype == bool and populations[name].shape == (11, 13)
    assert np.array_equal(populations["rig_domain"], np.ones((11, 13), bool))
    assert not populations["hardware_excluded"].any()
    assert not populations["below_sand_floor"].any()
    partition = sum(
        populations[name].astype(np.uint8)
        for name in ("modelled", "filled_observed", "filled_unobserved")
    )
    assert (partition == 1).all()
    assert result["pixel_area_mm2"] is frame["pixel_area_mm2"]
    assert result["native_inference"]["inference_performed"] is True
    assert result["qa"]["edge_inference"]["inference_performed"] is True
    assert result["qa"]["phase_area"]["co2_cm2"] == 0.0
    assert result["qa"]["phase_area"]["gas_cm2"] == 0.0
    assert result["qa"]["top"]["all_pass"] is True
    assert result["qa"]["floor"]["all_pass"] is True
    assert result["qa"]["source_history"]["previous_source_sha256"] is None
    assert all(mask.dtype == bool for mask in result["post_masks"].values())
    assert not any(mask.any() for mask in result["post_masks"].values())


def test_full_infer_labels_rejects_wrong_predecessor_before_pixels(
    rule_parameters, domain_parameters
):
    frame, bundle, p = _full_inputs(rule_parameters, domain_parameters)
    frame["source"]["previous_source_sha256"] = "b" * 64
    frame["full_log"] = None  # Any optical call would fail differently.
    with pytest.raises(ValueError, match="exact declared predecessor"):
        infer.infer_labels(frame, None, bundle, p)
    with pytest.raises(ValueError, match="another predecessor"):
        infer.infer_labels(frame, {"source": {"source_sha256": "c" * 64}}, bundle, p)


@pytest.mark.parametrize("anchor", [False, True])
def test_series_photo_geometry_reaches_actual_native_and_rig_inference(
    monkeypatch, rule_parameters, domain_parameters, anchor
):
    """The admitted geometric mask must reach inference in both source routes.

    Only source asset reads are substituted. Masking, temporal composition,
    measured-area recipe, static-component transport and all phase maths execute.
    """
    from darsia.fluidflower.native import support
    from darsia.fluidflower.phase import inputs
    from darsia.fluidflower.pipeline import phase_inputs as loaders
    from darsia.fluidflower.pipeline.series import SeriesInputs

    seed, bundle, p = _full_inputs(rule_parameters, domain_parameters)
    shape = seed["full_valid"].shape
    full = seed["full_log"].copy()
    valid = seed["full_valid"].copy()
    photo = valid.copy()
    photo[0, 0] = False
    input_data = json.loads(
        (Path(__file__).parent / "fixtures" / "phase_input_parameters.json").read_bytes()
    )
    p["inputs"] = input_data["inputs"]
    p["retained"] = input_data["retained"]
    policy = dict(contract=p["inputs"]["contract"], pad_px=0, origin_xy=[0, 0])
    support_parameters = dict(
        keep_policy="keep", fill_policy="fill", keep_contract="keep",
        fill_contract="fill", hole_max_area_px=0, connectivity=8,
        guard_px=0, vertical_gap_max_px=1, vertical_gap_rule="bounded",
        grid_shape_hw=list(shape), grid_archive_key="sand",
        cell_to_reference_native=np.eye(3).tolist(), homography_epsilon=1e-9,
        coverage_tolerance=1e-6, texture_footprint_hw=[3, 3],
    )
    projected = support.project_source_photo_safe_core(
        photo, np.eye(3), shape_hw=shape, guard_px=0,
        enclosed_hole_policy="fill", parameters=support_parameters,
    )
    expected_photo = photo if anchor else projected
    proof = dict(sha256="d" * 64, phase_labels_used=False)
    source = copy.deepcopy(seed["source"])
    source.update(studio_anchor=anchor, previous_source_sha256="b" * 64)
    source["source_binding"] = dict(source_sha256=source["source_sha256"])
    previous_native = inputs.TemporalNativeInput(
        full, valid, (0, 0), {"fixture_predecessor": True}
    )
    previous = dict(
        source=dict(source_sha256="b" * 64), full_log=full,
        full_valid=valid, current_h=np.eye(3), temporal_input=previous_native,
    )
    rp = p["retained"]
    probability = dict(
        contract=rp["contracts"]["PROBABILITY_CONTRACT"], complete=True,
        binding=dict(labelled=True, provisional=False,
            g3_model_sha256=rp["expected_models"]["g3"],
            regional_refit_sha256=rp["expected_models"]["regional_refit"]),
        metadata=dict(storage_layout=rp["encoding"]["labelled_layout"],
            native_shape_hw=list(shape), regional_working_shape_hw=[5, 6],
            co2_head_sha256=rp["expected_models"]["co2"],
            bitorder="little", masks_order=rp["mask_names"],
            resize_rule=rp["encoding"]["resize_rule"],
            g3_compact_rule=rp["encoding"]["compact_rule"],
            spacing_m=seed["probability_spacing_m"]),
        files={name:dict(dtype="float16",shape=list(shape))
            for name in rp["encoding"]["probability_files"]},
    )
    encoding = loaders.probability_encoding_contract(
        probability, shape, parameters=rp
    ) if anchor else None
    original_area = np.linspace(.5, 1.5, valid.size, dtype=np.float32).reshape(shape)
    measured = (original_area.astype(np.float64) * .5).astype(np.float32)
    read_roles = []

    class AssetReader:
        def json(self, descriptor):
            read_roles.append(descriptor["role"])
            return probability if descriptor["role"] == "probability" else {}
        def path(self, descriptor):
            read_roles.append(descriptor["role"])
            return Path("synthetic_photo.npz")
        def array(self, descriptor):
            read_roles.append(descriptor["role"])
            return original_area if descriptor["role"] == "historical_area" else measured
        def archive(self, descriptor, keys):
            read_roles.append(descriptor["role"])
            return dict(open_components=np.ones(shape, np.uint8),
                barrier_mask=np.zeros(shape, bool), reference_domain=valid)

    def admitted_original_native(row, *, scientific_native_log, parameters, aliases):
        masked = full.copy()
        masked[:, ~photo] = np.nan
        return inputs.original_native_input(masked, photo,
            provenance={"fixture_original": True},
            scientific_native_log=scientific_native_log, parameters=parameters)

    monkeypatch.setattr(loaders, "original_photo_sand",
        lambda receipt, **kwargs: (photo, proof))
    monkeypatch.setattr(loaders, "original_native_input", admitted_original_native)
    monkeypatch.setattr(loaders, "original_temporal_input",
        lambda native, row, **kwargs: inputs.original_temporal_input(
            native, policy=policy, parameters=kwargs["parameters"]))
    monkeypatch.setattr(support, "source_photo_grid", lambda *args, **kwargs: photo)
    owner = SeriesInputs.__new__(SeriesInputs)
    owner.reader, owner.aliases, owner.previous = AssetReader(), {}, previous
    owner.parameters = p
    owner.native_parameters = dict(support=support_parameters, extended_ring={})
    owner.temporal_policy = policy
    owner.config = dict(run="synthetic", history_policy="ordered unique acquisitions")
    owner.phase = dict(area_parameters=dict(maximum_relative_rounding_error=1e-7),
        barrier_parameters=dict(contract="synthetic static geometry"))
    roles = dict(
        native_row={"role":"native"} if anchor else None,
        probability_receipt={"role":"probability"},
        photo_sand_reference=dict(role="photo_reference",sha256="d"*64),
        photo_proof=proof, area=dict(role="measured_parent" if anchor else "selected",
            historical={"role":"historical_area"}, measured={"role":"measured_area"},
            factor=.5, revision={"synthetic":True}),
        static=dict(components={"role":"static_geometry"},source_ids=[1],
            unchanged_source=anchor),
        encoding_contract=encoding, probability_spacing_m=seed["probability_spacing_m"],
        inherited_spacing_m=seed["inherited_spacing_m"],
        current_h=np.eye(3).tolist(),reference_to_current_h=np.eye(3).tolist(),
        light_epoch_id="rig_light",placement_metadata={},
    )
    native = dict(log=full, valid=valid, full_log=full, full_valid=valid,
        context=SimpleNamespace(rig_domain=valid,hardware_excluded=np.zeros(shape,bool),
            pixel_area_mm2=original_area,parameters={"fixture_geometry":True}),
        observed=valid,top=valid,record={"synthetic_source":True},
        record_asset={"role":"actual_native"},audit={"parent_cache":None})
    assembled = owner.frame(dict(source=source,phase_roles=roles), native)
    assert np.array_equal(assembled["photo_sand"], expected_photo)
    assert assembled["photo_sand"].dtype == bool
    assert np.array_equal(assembled["native_input"].valid, expected_photo)
    assert np.array_equal(assembled["native_input"].log[:,expected_photo],full[:,expected_photo])
    assert np.isnan(assembled["native_input"].log[:,~expected_photo]).all()
    assert assembled["full_valid"] is valid and assembled["context"].rig_domain is valid
    assert np.array_equal(full, seed["full_log"])
    assert not {"labels","classes","expected_phase"}.intersection(read_roles)
    actual = infer.infer_labels(assembled, previous, bundle, p)
    assert np.array_equal(actual["labels"], np.zeros(shape,np.uint8))
    assert actual["native_inference"]["inference_performed"] is True
    assert actual["qa"]["edge_inference"]["inference_performed"] is True
    assert actual["qa"]["phase_area"]["co2_cm2"] == 0.0
    assert actual["qa"]["phase_area"]["gas_cm2"] == 0.0
    assert actual["qa"]["source_history"]["previous_source_sha256"] == "b"*64
    assert actual["pixel_area_mm2"] is assembled["pixel_area_mm2"]
