"""Synthetic parity with immutable rule code objects from before extraction."""
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy import ndimage
from scipy.special import expit, logit

from darsia.fluidflower.phase import rules
from darsia.fluidflower.phase import metrics


BASE = "f0d338aa2be5e6ee8be6a2db0bd8f6fb70e9bd5a"
REPO = Path(__file__).resolve().parents[2]
MAPPING = {
    "phase_ffac_p1_final": {
        "select_backend": "_select_backend", "_top_mask": "_top_mask",
        "apply_p1_final": "apply_top_rule"},
    "phase_ffac_p1_final2": {
        "_bool": "_floor_bool", "_labels": "_floor_labels", "_modules": "_modules",
        "_numpy": "_numpy", "_is_gpu_failure": "_is_gpu_failure", "_edt": "_edt",
        "_dilate": "_dilate", "_label": "_label", "apply_floor_boundary": "apply_floor_boundary",
        "_envelopes_core": "_envelopes_core", "build_envelopes": "build_envelopes",
        "_core": "_floor_core", "apply_final_round": "apply_floor_round"},
    "phase_ffac_p1_final3": {"remove_contact_fill_b": "remove_contact_fill_b"},
    "phase_gas_barrier_rule": {"array_identity": "array_identity",
        "_validate_components": "_validate_components", "apply_barrier_rule_b": "apply_barrier_rule_b"},
    "phase_l1_neighbourhood_gas": {"frozen_rule_b_forbidden_mask": "frozen_rule_b_forbidden_mask"},
    "phase_gas_t4c_placements": {"_labels": "_placement_labels", "_bool_grid": "_placement_bool",
        "base_gas_fraction": "base_gas_fraction", "_cached_fraction": "_cached_fraction",
        "prepare_p2_allowed": "prepare_p2_allowed", "_minimum_additions": "_minimum_additions",
        "evaluate_placement": "evaluate_placement"},
    "phase_gas_t2_regional_inputs": {"replace_gas_from_scores": "replace_gas_from_scores"},
    "phase_ffac_selected_native_centre": {"preminimum_gas_selection": "preminimum_gas_selection"},
    "phase_g3_production": {"configuration_identity": "gas_configuration_identity",
        "_mask_identity": "_packed_mask_identity", "_check_scores": "_check_scores",
        "apply_g3_configuration": "apply_gas_configuration"},
    "phase_rig_domain_inference": {"frozen_native_partition": "frozen_native_partition",
        "require_categorical_feature_support": "require_categorical_feature_support"},
}


def git_tree(short):
    source = subprocess.check_output(["git", "show", BASE +
        ":src/darsia/fluidflower/" + short + ".py"], cwd=REPO)
    return ast.parse(source)


@pytest.fixture
def parameters():
    return json.loads((Path(__file__).parent / "fixtures" /
                       "phase_rules_parameters.json").read_bytes())


@pytest.fixture(scope="module")
def originals():
    """Execute only the original pure definitions, without campaign imports."""
    modules = {}
    for short in (*MAPPING, "phase_pixel_area"):
        module = ModuleType("_preserved_rules_" + short)
        sys.modules[module.__name__] = module
        module.__dict__.update(np=np, cv2=cv2, cpu_ndimage=ndimage, ndimage=ndimage,
            dataclass=dataclass, field=field, asdict=asdict, json=json, time=time, hashlib=hashlib,
            logit=logit, expit=expit)
        tree = git_tree(short)
        # Original top-level literals and derived physical distances are DATA.
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                try:
                    exec(compile(ast.Module([node], type_ignores=[]), BASE, "exec"), module.__dict__)
                except (NameError, AttributeError):
                    pass
        names = set(MAPPING.get(short, {}))
        if short == "phase_ffac_p1_final": names.add("FinalResult")
        if short == "phase_ffac_p1_final2": names.add("FinalRoundResult")
        if short == "phase_gas_t4c_placements": names.update(("ShiftContext", "shifted_probability"))
        if short == "phase_g3_production":
            settings = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ApprovedG3Settings")
            exec(compile(ast.Module([settings], type_ignores=[]), BASE, "exec"), module.__dict__)
            module.APPROVED_SETTINGS = module.ApprovedG3Settings()
        if short == "phase_pixel_area":
            names.update(("_shape", "validate_pixel_area", "filter_components_by_area", "area_filtered_gas"))
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                       and n.name in names]
        exec(compile(ast.Module(definitions, type_ignores=[]), BASE, "exec"), module.__dict__)
        modules[short] = module
    top, floor = modules["phase_ffac_p1_final"], modules["phase_ffac_p1_final2"]
    floor.select_backend = top.select_backend
    barrier, frozen = modules["phase_gas_barrier_rule"], modules["phase_l1_neighbourhood_gas"]
    frozen.RULE_B_CONTRACT, frozen.array_identity = barrier.VERSION, barrier.array_identity
    placement, replacement, area = (modules[name] for name in
        ("phase_gas_t4c_placements", "phase_gas_t2_regional_inputs", "phase_pixel_area"))
    placement.array_identity = barrier.array_identity
    placement.frozen_rule_b_forbidden_mask = frozen.frozen_rule_b_forbidden_mask
    placement.apply_barrier_rule_b = barrier.apply_barrier_rule_b
    placement.FIXED_AC28_MM2 = area.FIXED_AC28_MM2
    placement.validate_pixel_area = area.validate_pixel_area
    placement.shared = replacement
    replacement.validate_pixel_area, replacement.area_filtered_gas = area.validate_pixel_area, area.area_filtered_gas
    yield modules
    for module in modules.values(): sys.modules.pop(module.__name__, None)


def same(actual, expected):
    """Exact scientific arrays/scalars; exclude only actual execution timings."""
    if isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key not in {"cpu_seconds", "wall_seconds"}: same(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): same(a, b)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


def same_result(actual, expected):
    assert actual.__dataclass_fields__.keys() == expected.__dataclass_fields__.keys()
    for key in actual.__dataclass_fields__: same(getattr(actual, key), getattr(expected, key))


def small_floor(parameters, original):
    parameters["nominal_mm_per_pixel"] = original.MM_PER_PIXEL = 1.0
    parameters["floor"]["floor_band_pixels"] = original.FLOOR_BAND_PIXELS = 2
    parameters["floor"]["column_gate_pixels"] = original.COLUMN_GATE_PIXELS = 2
    parameters["contact"]["short_fill_limit_pixels"] = original.SHORT_FILL_PIXELS = 2


def test_required_parameter_types_and_all_arithmetic_ast(parameters, originals):
    assert type(parameters["top"]["band_mm"]) is int
    assert type(parameters["floor"]["floor_band_mm"]) is float
    assert type(parameters["literal_values"]["apply_top_rule_int_100"]) is int
    assert type(parameters["literal_values"]["apply_floor_round_float_100_0"]) is float
    with pytest.raises(TypeError): rules.base_gas_fraction(np.zeros((2, 3), np.uint8))
    with pytest.raises(KeyError): rules.base_gas_fraction(np.zeros((2, 3), np.uint8), parameters={})

    rename = {new: old for names in MAPPING.values() for old, new in names.items()}
    rename["GasDecisionSettings"] = "ApprovedG3Settings"
    globals_by_module = {
        "phase_ffac_p1_final": {"NOMINAL_AREA_MM2", "MM_PER_PIXEL", "BAND_PIXELS", "COLUMN_GATE_PIXELS"},
        "phase_ffac_p1_final2": {"MM_PER_PIXEL", "COLUMN_GATE_PIXELS", "FLOOR_BAND_PIXELS", "SHORT_FILL_PIXELS"},
        "phase_gas_barrier_rule": {"VERSION"}, "phase_l1_neighbourhood_gas": {"RULE_B_CONTRACT"},
        "phase_gas_t4c_placements": {"PIXEL_THRESHOLD", "P2_FRACTION_KERNEL", "MINIMUM_AREA_MM2", "FIXED_AC28_MM2", "CONTRACT"},
        "phase_gas_t2_regional_inputs": {"CONTRACT"},
        "phase_g3_production": {"VERSION"},
    }

    class Normalize(ast.NodeTransformer):
        def __init__(self, old): self.old = old
        def visit_Name(self, node):
            if node.id == "NOMINAL_BARRIER_PIXEL_AREA_MM2": node.id = "FIXED_AC28_MM2"
            if node.id in globals_by_module.get(self.old.__name__.removeprefix("_preserved_rules_"), set()):
                return ast.parse(repr(getattr(self.old, node.id)), mode="eval").body
            node.id = rename.get(node.id, node.id)
            return node
        def visit_Subscript(self, node):
            path, head = [], node
            while isinstance(head, ast.Subscript) and isinstance(head.slice, ast.Constant):
                path.insert(0, head.slice.value); head = head.value
            if isinstance(head, ast.Name) and head.id == "parameters":
                value = parameters
                for key in path: value = value[key]
                return ast.parse(repr(value), mode="eval").body
            return self.generic_visit(node)
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "tuple" and len(node.args) == 1 and isinstance(node.args[0], ast.List):
                return ast.Tuple(node.args[0].elts, ast.Load())
            node.keywords = [k for k in node.keywords if k.arg not in {"parameters", "heads_parameters", "regional_parameters"}]
            return node
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id in {"heads", "regional", "shared"}:
                return ast.Name(node.attr, ast.Load())
            return self.generic_visit(node)

    # Only the reviewed optional side-channel presence comparison is excluded.
    # Every arithmetic expression INSIDE those blocks is still walked below.
    capture_blocks = {'evaluate_placement': ['48e844a7f1d14d17075b43dcc741594f24d11fd1b9c33259b659f9a06099e3b9', '2822f60641fa58870c5907e182f8ef0e94dcd9a70c7a2ac05d5691a5713ac2c0', '8cadf234a928030aa46ca43e146a9f8e4485dcf755c3f4419a27f6220bd69265', '0d82a191cd842d04b5dd3fc401d92a68a104b646aa655c9566e81f5987b4b860', 'c6224601cea392725a9d2be428e279e228e5a0ba1b6190443d1361ffac447c23'], 'preminimum_gas_selection': ['c5c5df8c7bdfb8813c12572f91633916d594a0dc96a550d9415a7f8193d6296f'], 'apply_gas_configuration': ['2d915aef59e47748e852142271b0f2e41c73fc3bb155628f206573e92087127a']}
    for function_name, expected_blocks in capture_blocks.items():
        body = ast.parse(inspect.getsource(getattr(rules, function_name))).body[0]
        presence = ast.dump(ast.parse("output_capture is not None", mode="eval").body, include_attributes=False)
        actual_blocks = [hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                         for node in ast.walk(body) if isinstance(node, ast.If)
                         and ast.dump(node.test, include_attributes=False) == presence]
        assert actual_blocks == expected_blocks
    from darsia.fluidflower.phase import probability_storage
    helper_ast = ast.dump(ast.parse(inspect.getsource(probability_storage)), include_attributes=False)
    assert hashlib.sha256(helper_ast.encode()).hexdigest() == "739ccb57a0154cd3d414df28000a244b85f426c0f964d237904f5c48c36f6d39"
    checked = 0
    for module, names in MAPPING.items():
        original_nodes = {n.name: n for n in git_tree(module).body if isinstance(n, ast.FunctionDef)}
        for old_name, new_name in names.items():
            old = Normalize(originals[module]).visit(original_nodes[old_name])
            function = getattr(metrics if new_name == "array_identity" else rules, new_name)
            new = Normalize(originals[module]).visit(ast.parse(inspect.getsource(function)).body[0])
            def expressions(tree, current):
                output = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                        if current and isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "backend" and len(node.ops) == 1 and isinstance(node.ops[0], ast.Is) and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                            continue
                        if (current and new_name in capture_blocks and isinstance(node, ast.Compare)
                                and ast.dump(node, include_attributes=False) == presence):
                            continue
                        output.append(ast.dump(node, include_attributes=False))
                return output
            assert expressions(new, True) == expressions(old, False), new_name
            checked += len(expressions(old, False))
    assert checked > 200


def test_top_component_depth_closure_and_floor_restoration(parameters, originals):
    old = originals["phase_ffac_p1_final"]
    # Test geometry is small; the supplied DATA and original namespace agree.
    parameters["top"]["band_pixels"] = old.BAND_PIXELS = 4
    parameters["top"]["column_gate_pixels"] = old.COLUMN_GATE_PIXELS = 2
    base = np.zeros((14, 12), np.uint8)
    base[1:3, 1:3] = 1          # Detached inside-band component.
    base[2:9, 7:9] = 2          # Connected below-band gas must all remain.
    base[:, -1] = 255
    p1, floor = base.copy(), np.zeros_like(base, bool)
    p1[12, 3] = 1; floor[12, 3] = True
    area = np.linspace(.02, .04, base.size).reshape(base.shape)
    actual = rules.apply_top_rule(p1, base, floor, area, backend="cpu", parameters=parameters)
    expected = old.apply_p1_final(p1, base, floor, area, backend="cpu")
    same_result(actual, expected)
    assert actual.removed_top[1:3, 1:3].all()
    assert np.array_equal(actual.labels[2:9, 7:9], base[2:9, 7:9])
    assert actual.labels[12, 3] == 0


def test_floor_boundary_and_float64_envelopes(parameters, originals):
    old = originals["phase_ffac_p1_final2"]
    parameters["nominal_mm_per_pixel"] = old.MM_PER_PIXEL = 1.0
    labels = np.zeros((15, 19), np.uint8); labels[2, 3] = 2
    below = np.zeros_like(labels, bool); below[-2:] = True
    same(rules.apply_floor_boundary(labels, below, parameters=parameters), old.apply_floor_boundary(labels, below))
    first = np.zeros_like(labels, bool); second = first.copy(); second[7, 9] = True
    actual = rules.build_envelopes(first, second, "cpu", parameters=parameters)
    expected = old.build_envelopes(first, second, "cpu")
    same(actual, expected)
    assert not actual[0].any()
    assert actual[1][7, 14] and not actual[1][7, 15]
    assert rules._edt(~second, np, ndimage, parameters=parameters).dtype == np.float64


@pytest.mark.parametrize("elapsed,index", [(0.0, 2), (1.0, 2), (1.0, 3), (1.0, 5)])
def test_floor_contact_precedence_and_strict_envelope_ordinals(parameters, originals, elapsed, index):
    old = originals["phase_ffac_p1_final2"]; small_floor(parameters, old)
    labels = np.zeros((16, 10), np.uint8)
    labels[4:14, 1] = 1         # Supported short contact run.
    labels[4:12, 4] = 1         # Long carryover run, distinct reason.
    labels[4:14, 7] = 2         # Gas trigger cannot fill contact.
    zero = np.zeros_like(labels, bool); support = np.ones_like(labels, bool)
    carry = zero.copy(); carry[11:, 4] = True
    l1 = np.ones_like(zero); env1 = zero.copy(); env2 = zero.copy()
    env1[:, :6] = True; env2[:8] = True
    args = (labels, zero, support, carry, l1, env1, env2, elapsed, index, 2, 4, .1)
    actual = rules.apply_floor_round(*args, backend="cpu", parameters=parameters)
    expected = old.apply_final_round(*args, backend="cpu")
    same_result(actual, expected)
    assert not actual.fill_reason_a[:, 7].any()
    assert not np.any(actual.fill_reason_a & actual.fill_reason_b)
    if elapsed == 0: assert not (actual.fill_reason_a | actual.fill_reason_b).any()
    else:
        assert actual.fill_reason_a[14:, 1].all()
        assert actual.fill_reason_b[12:, 4].all()
    if index == 2: assert not actual.step_masks["D"].any()
    else: assert actual.step_masks["D"][:, 7].any()


def test_contact_reason_b_only_and_partition_rejection(parameters, originals):
    old = originals["phase_ffac_p1_final3"]
    labels = np.zeros((5, 7), np.uint8); labels[3, 1:4] = 1
    a = np.zeros_like(labels, bool); a[3, 1] = True
    b = np.zeros_like(labels, bool); b[3, 2:4] = True
    same(rules.remove_contact_fill_b(labels, a, b, a | b, parameters=parameters),
         old.remove_contact_fill_b(labels, a, b, a | b))
    with pytest.raises(ValueError, match="disjoint"):
        rules.remove_contact_fill_b(labels, a, a, a, parameters=parameters)


def barrier_fixture():
    components = np.zeros((8, 13), np.uint8)
    components[:, :4] = 1; components[:, 5:9] = 2; components[:, 10:] = 3
    barrier = components == 0
    labels = np.zeros_like(components)
    labels[2, 2:7] = 1          # CO2 bridge reaches open region 2.
    labels[5, 6] = 2            # Isolated gas in reached region stays.
    labels[5, 11] = 2           # Unreached gas is downgraded, even if tiny.
    return labels, components, barrier


def test_barrier_closure_frozen_receipt_and_array_identity(parameters, originals):
    old = originals["phase_gas_barrier_rule"]
    frozen = originals["phase_l1_neighbourhood_gas"]
    labels, components, barrier = barrier_fixture()
    actual = rules.apply_barrier_rule_b(labels, components, [1], pixel_area_mm2=.027053,
                                      barrier_mask=barrier, parameters=parameters)
    expected = old.apply_barrier_rule_b(labels, components, [1], pixel_area_mm2=.027053, barrier_mask=barrier)
    same(actual, expected)
    assert actual[0][5, 6] == 2 and actual[0][5, 11] == 1
    assert actual[1]["allowed_component_ids"] == [1, 2]
    assert actual[1]["gas_minimum_reapplied"] is False
    same(rules.frozen_rule_b_forbidden_mask(components, barrier, actual[1], parameters=parameters),
         frozen.frozen_rule_b_forbidden_mask(components, barrier, expected[1]))
    same(rules.array_identity(components), old.array_identity(components))
    assert rules.array_identity(components) != rules.array_identity(components.astype(np.int16))
    changed = components.copy(); changed[0, 0] = 2
    with pytest.raises(ValueError, match="different component geometry"):
        rules.frozen_rule_b_forbidden_mask(changed, barrier, actual[1], parameters=parameters)


def test_fraction_and_constrained_candidates_exact(parameters, originals):
    old = originals["phase_gas_t4c_placements"]
    labels, components, barrier = barrier_fixture()
    same(rules.base_gas_fraction(labels, parameters=parameters), old.base_gas_fraction(labels))
    assert rules.base_gas_fraction(labels, parameters=parameters).dtype == np.float32
    _, receipt = rules.apply_barrier_rule_b(labels, components, [1], pixel_area_mm2=.027053, barrier_mask=barrier, parameters=parameters)
    zones = np.ones_like(labels); zones[0] = 2; zones[2] = 3
    same(rules.prepare_p2_allowed(labels, zones, barrier, components, receipt, parameters=parameters),
         old.prepare_p2_allowed(labels, zones, barrier, components, receipt))


def test_old_light_bypasses_all_optical_and_spatial_evaluation(parameters, originals, monkeypatch):
    old = originals["phase_gas_t4c_placements"]
    labels = np.array([[0, 1, 2, 255]], np.uint8)
    def forbidden(*args, **kwargs): raise AssertionError("old light must bypass this stage")
    monkeypatch.setattr(rules, "_cached_fraction", forbidden)
    monkeypatch.setattr(rules, "frozen_rule_b_forbidden_mask", forbidden)
    monkeypatch.setattr(rules.heads, "shifted_probability", forbidden)
    context = rules.PlacementContext(parameters["placement"]["epoch_modes"]["bypass"], np.zeros_like(labels, bool))
    previous = old.ShiftContext(context.light_epoch_id, context.domain)
    kwargs = dict(top_keep=None, open_components=None, barrier_mask=None, frozen_base_rule_b=None)
    same(rules.evaluate_placement(context, "P1", .5, labels, None, None,
                                 parameters=parameters, heads_parameters={}, regional_parameters={}, **kwargs),
         old.evaluate_placement(previous, "P1", .5, labels, None, None, **kwargs))


def test_neighbourhood_additions_protect_base_gas_and_clip_new_top(parameters, originals):
    old = originals["phase_gas_t4c_placements"]
    labels = np.ones((9, 12), np.uint8); labels[4, 4] = 2
    components = np.ones_like(labels); barrier = np.zeros_like(labels, bool)
    _, receipt = rules.apply_barrier_rule_b(labels, components, [1], pixel_area_mm2=.027053, barrier_mask=barrier, parameters=parameters)
    allowed = np.zeros_like(barrier); allowed[4, 5] = True; allowed[1, 9] = True
    top = np.ones_like(barrier); top[4, 5] = False
    area = np.full(labels.shape, .1); area[1, 9] = 3.0
    probability = np.full(labels.shape, .999999, np.float32)
    context = rules.PlacementContext("new_lamps", ~barrier, ~barrier, probability)
    previous = old.ShiftContext("new_lamps", ~barrier, ~barrier, probability)
    kwargs = dict(top_keep=top, open_components=components, barrier_mask=barrier,
                  frozen_base_rule_b=receipt, p2_allowed=allowed)
    actual = rules.evaluate_placement(context, "P2", 10., labels, ~barrier, area,
                                     parameters=parameters, heads_parameters={}, regional_parameters={}, **kwargs)
    expected = old.evaluate_placement(previous, "P2", 10., labels, ~barrier, area, **kwargs)
    same(actual, expected)
    assert actual[0][4, 4] == 2 and actual[0][4, 5] == 1 and actual[0][1, 9] == 2
    assert actual[1]["gas_minimum"]["base_gas_removed_pixels"] == 0
    different = labels.copy(); different[0, 0] = 2
    with pytest.raises(ValueError, match="another frozen base gas mask"):
        rules._cached_fraction(context, different, parameters=parameters)


def test_score_partition_inherited_fill_variable_area_before_barrier(parameters, originals):
    old = originals["phase_gas_t2_regional_inputs"]
    labels = np.ones((9, 11), np.uint8); labels[0, 0] = 255; labels[8, 10] = 2
    support = np.zeros_like(labels, bool); support[2:4, 2:4] = True; support[6, 7] = True
    filled = np.zeros_like(support); filled[4, 2:4] = True
    pixel, score = np.full(labels.shape, .8, np.float32), np.full(labels.shape, .7, np.float32)
    area = np.full(labels.shape, .5); area[6, 7] = .1
    actual = rules.replace_gas_from_scores(labels, pixel, score, support, area, filled_support=filled, parameters=parameters)
    expected = old.replace_gas_from_scores(labels, pixel, score, support, area, filled_support=filled)
    same(actual, expected)
    assert (actual[0][2:5, 2:4] == 2).all()
    assert actual[0][6, 7] == 1 and actual[0][8, 10] == 2
    assert actual[1]["gas_minimum_before_rule_b"] is True


def test_preminimum_evidence_before_union_with_float32_linear_resize(parameters, originals):
    old = originals["phase_ffac_selected_native_centre"]
    pixel = np.array([[.8, .9, .1, np.nan], [.8, .9, .7, .8]], np.float32)
    regional_score = np.array([[.8, .2]], np.float32)
    actual = rules.preminimum_gas_selection(pixel, regional_score, parameters=parameters)
    same(actual, old.preminimum_gas_selection(pixel, regional_score))
    assert actual.dtype == np.bool_ and actual[0, 0]
    assert not actual[0, 3]
    with pytest.raises(ValueError, match="float32"):
        rules.preminimum_gas_selection(pixel.astype(np.float64), regional_score, parameters=parameters)


def test_frozen_native_hierarchical_partition_and_feature_support(parameters, originals):
    old = originals["phase_rig_domain_inference"]
    presence = np.array([[.49, .5, .9, np.nan], [.8, .8, .1, .7]], np.float32)
    gas = np.array([[.99, .59, .6, .99], [np.nan, .9, .9, .7]], np.float32)
    valid = np.array([[True, True, True, True], [True, False, True, True]])
    actual = rules.frozen_native_partition(presence, gas, valid, parameters=parameters)
    same(actual, old.frozen_native_partition(presence, gas, valid))
    assert actual.tolist() == [[0, 1, 2, 0], [1, 0, 0, 2]]
    optical, gas_valid = valid.copy(), np.isfinite(gas)
    supported = rules.require_categorical_feature_support(actual, optical, gas_valid, parameters=parameters)
    same(supported, old.require_categorical_feature_support(actual, optical, gas_valid))
    assert supported[0][1, 0] == 255
    assert supported[2]["gas_feature_invalid_optical_supported_pixels"] == 1
    with pytest.raises(ValueError, match="share the native"):
        rules.require_categorical_feature_support(actual, optical[:, :2], gas_valid, parameters=parameters)


@pytest.mark.parametrize("regional_mode,minimum", [("refit", 2.5), ("frozen", 1.0), ("off", 0.0)])
def test_nominal_gas_configuration_and_original_hashes(parameters, originals, regional_mode, minimum):
    old = originals["phase_g3_production"]
    values = dict(parameters["gas_configuration"]["selected_settings"],
                  regional_mode=regional_mode, minimum_area_mm2=minimum)
    settings = rules.GasDecisionSettings(**values, parameters=parameters)
    previous = old.ApprovedG3Settings(**values)
    same(rules.gas_configuration_identity(settings), old.configuration_identity(previous))
    assert asdict(settings) == asdict(previous)
    labels = np.zeros((9, 13), np.uint8)
    labels[2:4, 2:4] = 1; labels[5:7, 8:11] = 2; labels[0, 0] = 255
    pixel = np.full(labels.shape, .8, np.float16); pixel[5, 8] = np.nan
    regional_score = np.full((3, 4), .7, np.float64) if regional_mode != "off" else None
    actual = rules.apply_gas_configuration(labels, pixel, regional_score, .5, settings, parameters=parameters)
    expected = old.apply_g3_configuration(labels, pixel, regional_score, .5, previous)
    same(actual, expected)
    assert np.array_equal(np.isin(actual[0], (1, 2)), np.isin(labels, (1, 2)))
    assert actual[0][0, 0] == 255 and actual[0][5, 8] == 1
    if minimum == 2.5:
        assert not (actual[0][2:4, 2:4] == 2).any()
        assert np.count_nonzero(actual[0][5:7, 8:11] == 2) == 5
    same(rules._packed_mask_identity(labels == 2), old._mask_identity(labels == 2))
    with pytest.raises(ValueError, match="configuration grid"):
        rules.GasDecisionSettings(**dict(values, connectivity=8), parameters=parameters)
    with pytest.raises(ValueError, match="configuration grid"):
        old.ApprovedG3Settings(**dict(values, connectivity=8))


def test_nominal_gas_probability_guards_preserved(parameters, originals):
    old = originals["phase_g3_production"]
    settings = rules.GasDecisionSettings(**parameters["gas_configuration"]["selected_settings"], parameters=parameters)
    labels = np.ones((3, 5), np.uint8)
    for bad in (np.full(labels.shape, np.inf), np.full(labels.shape, 1.1), np.zeros(labels.shape, np.uint8)):
        with pytest.raises(ValueError):
            rules.apply_gas_configuration(labels, bad, np.ones((2, 2), np.float32), .1, settings, parameters=parameters)
        with pytest.raises(ValueError):
            old.apply_g3_configuration(labels, bad, np.ones((2, 2), np.float32), .1)


@pytest.mark.parametrize("placement", ["P1", "P3"])
def test_regional_placement_threshold_minimum_top_then_barrier(parameters, originals, monkeypatch, placement):
    old = originals["phase_gas_t4c_placements"]
    labels, components, barrier = barrier_fixture()
    labels[3:5, 1:4] = 1; labels[3:5, 10:13] = 1
    _, receipt = rules.apply_barrier_rule_b(labels, components, [1], pixel_area_mm2=.027053, barrier_mask=barrier, parameters=parameters)
    valid = np.ones_like(barrier); top = valid.copy(); top[3, 1] = False
    logits = np.full(labels.shape, 3., np.float64)
    events = []
    def prepare(context, gas, **kwargs):
        events.append("prepare"); assert gas.dtype == np.float32
        return {"features": gas}
    def probability(features, fit, **kwargs):
        events.append("regional"); return np.full(features.shape, .8, np.float32)
    monkeypatch.setattr(rules.regional, "prepare_conditional_gas", prepare)
    monkeypatch.setattr(rules.regional, "probability", probability)
    old.prepare_conditional_gas = prepare; old.regional = SimpleNamespace(probability=probability)
    model = {"heads": {"gas_within_co2": {}}}
    context = rules.PlacementContext("new_lamps", valid, valid, np.full(labels.shape,.8,np.float32),
                                     {placement: logits}, {placement: model}, object())
    previous = old.ShiftContext("new_lamps", valid, valid, context.base_pixel,
                                {placement: logits}, {placement: model}, context.shared_regional)
    kwargs = dict(top_keep=top, open_components=components, barrier_mask=barrier, frozen_base_rule_b=receipt)
    actual = rules.evaluate_placement(context, placement, .2, labels, valid, 1.,
                                     parameters=parameters, heads_parameters={}, regional_parameters={}, **kwargs)
    expected = old.evaluate_placement(previous, placement, .2, labels, valid, 1., **kwargs)
    same(actual, expected)
    assert events == ["prepare", "regional", "prepare", "regional"]
    assert actual[0][3, 1] == 1 and actual[0][3, 11] == 1
    assert actual[1]["gas_minimum_before_rule_b"] is True
    assert actual[1]["rule_b_application"]["gas_minimum_reapplied"] is False
