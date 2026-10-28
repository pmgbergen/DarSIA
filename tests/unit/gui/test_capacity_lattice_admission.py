"""Claim-time capacity admits a lattice task through its own ESTIMATE; native routes are unchanged."""
import ast
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import capacity as cap, lattice_capacity as lc
from .test_capacity_lattice import lattice_run


def test_claim_capacity_gives_a_lattice_task_its_estimate_before_any_profile_read(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("A lattice task reads no measured or lean profile")
    monkeypatch.setattr(cap, "settings", refused)
    monkeypatch.setattr(cap, "lean_estimate", refused)
    task, config = lattice_run()
    for value in (task, dict(task, storage_profile="lean")):
        estimate = cap.load_estimate(None, "S1", value, config, repository=None)
        assert isinstance(estimate, lc.LatticeEstimate) and estimate.case_count == 2


def test_a_native_task_never_reaches_the_lattice_estimate(monkeypatch):
    class Reached(Exception):
        pass

    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(cap, "settings", reached)
    monkeypatch.setattr(lc, "lattice_estimate", lambda *args, **kwargs: pytest.fail("Native task took the lattice route"))
    task, config = lattice_run()
    with pytest.raises(Reached):
        cap.load_estimate(None, "S1", dict(task, pipeline="native-phase-n1"), config, repository=None)


def test_the_capacity_module_reaches_lattice_code_only_through_its_guarded_branch():
    source = Path(cap.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and "lattice-phase-n1" in ast.dump(node.test)]
    assert len(branches) == 1
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and "lattice" in (node.module or "")]
    assert len(imports) == 1 and any(imports[0] is node for node in ast.walk(branches[0]))
    lines = [number for number, line in enumerate(source.splitlines(), 1) if "lattice" in line]
    assert lines and all(branches[0].lineno <= number <= branches[0].end_lineno for number in lines)
