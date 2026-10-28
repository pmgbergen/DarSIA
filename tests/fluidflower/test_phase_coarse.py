"""Exact coarse observational tables against preserved original Git functions."""
import ast
import inspect
from pathlib import Path
import subprocess

import numpy as np
import pytest

from darsia.fluidflower.phase import metrics


BASE = "f0d338aa2be5e6ee8be6a2db0bd8f6fb70e9bd5a"
PARAMETERS = {
    "contract": "ffac-selected-policy-a-required-coarse-spatial-qa-v2",
    "acceptance_backend": "CPU float64 measured pixel-area sums",
    "allowed_labels": [0, 1, 2, 255],
    "co2_relative_limit": .01,
    "gas_relative_limit": .02,
    "absolute_floor_cm2": .5,
    "square_mm_per_cm2": 100.,
}


def original():
    source = subprocess.check_output(["git", "show", BASE +
        ":src/darsia/fluidflower/phase_ffac_selected_policy_a_baseline_qa.py"],
        cwd=Path(__file__).resolve().parents[2])
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in {"_grids", "coarse_studio_gate"}]
    scope = dict(np=np, CONTRACT=PARAMETERS["contract"])
    exec(compile(ast.Module(nodes, type_ignores=[]), BASE, "exec"), scope)
    return scope


def test_coarse_complete_tables_and_original_arithmetic_order():
    old = original()
    before = np.zeros((9, 13), np.uint8)
    before[1:8, 1:10] = 2; before[:, -1] = 255
    after = before.copy(); after[2, 2] = 1; after[4, 10] = 1
    area = np.full(before.shape, 100., np.float64); area[4, 10] = 50.
    actual = metrics.categorical_coarse_gate(before, after, area, parameters=PARAMETERS)
    assert actual == old["coarse_studio_gate"](before, after, area)
    assert actual["gas"]["relative_limit"] == .02
    assert actual["co2"]["acceptance_limit_cm2"] == .63
    assert actual["all_pass"] is True
    # Validate expression order as well as one synthetic numerical table.
    source = inspect.getsource(metrics.categorical_coarse_gate)
    class Normalize(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "parameters" and isinstance(node.slice, ast.Constant):
                return ast.parse(repr(PARAMETERS[node.slice.value]), mode="eval").body
            return self.generic_visit(node)
    new = Normalize().visit(ast.parse(source))
    original_source = subprocess.check_output(["git", "show", BASE +
        ":src/darsia/fluidflower/phase_ffac_selected_policy_a_baseline_qa.py"],
        cwd=Path(__file__).resolve().parents[2])
    previous = next(n for n in ast.parse(original_source).body if isinstance(n, ast.FunctionDef) and n.name == "coarse_studio_gate")
    expressions = lambda tree: [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
        if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare))]
    assert expressions(new) == expressions(previous)


def test_coarse_zero_parent_floor_and_exact_grid_rejection():
    old = original()
    before = np.zeros((3, 5), np.uint8); after = before.copy(); after[1, 2] = 2
    area = np.full(before.shape, 1., np.float64); area[1, 2] = 50.
    actual = metrics.categorical_coarse_gate(before, after, area, parameters=PARAMETERS)
    assert actual == old["coarse_studio_gate"](before, after, area)
    assert actual["gas"]["relative_xor"] is None and actual["all_pass"] is True
    area[1, 2] = np.nextafter(50., np.inf)
    assert metrics.categorical_coarse_gate(before, after, area, parameters=PARAMETERS) == old["coarse_studio_gate"](before, after, area)
    assert not metrics.categorical_coarse_gate(before, after, area, parameters=PARAMETERS)["all_pass"]
    for a, b, weights in ((before.astype(np.int16), after, area), (before, after[:, :4], area),
                           (before, after, -area), (before, after, np.full(before.shape, np.nan))):
        with pytest.raises(ValueError): metrics.categorical_coarse_gate(a, b, weights, parameters=PARAMETERS)
        with pytest.raises(ValueError): old["coarse_studio_gate"](a, b, weights)
