"""Fake-owner and local-file parity for exact SDK protocol helpers."""
from __future__ import annotations
import ast
import copy
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

BASE = "65c0e85015700f764b2c4f5fed6bb558490ef7f0"
STORE_CASES = [
    ("calibration.py", "CalibrationPlanStore"),
    ("calibration_review.py", "CalibrationResultReviewStore"),
    ("indicator_response.py", "IndicatorResponseModelStore"),
    ("mass_analysis.py", "MassAnalysisPlanStore"),
    ("post_analysis.py", "PostAnalysisPlanStore"),
    ("reporting.py", "ReportingPlanStore"),
    ("wasserstein.py", "WassersteinPlanStore"),
    ("phase_interpretation.py", "PhaseInterpretationRevisionStore"),
]


def _node(relative, name, class_name=None, original=False):
    repo = Path(__file__).resolve().parents[2]
    path = "src/darsia/fluidflower/" + relative
    text = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=repo).decode("utf8") if original else (repo / path).read_text(encoding="utf8")
    tree = ast.parse(text)
    parent = tree if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return copy.deepcopy(next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name))


def _compile(node, globals=None):
    node.returns = None
    node.decorator_list = []
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    namespace = dict(json=json, deepcopy=deepcopy, hashlib=hashlib, math=math, Path=Path)
    namespace.update(globals or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "source-bound helper", "exec"), namespace)
    return namespace[node.name]


def _forward(relative, name, target, class_name=None):
    node = _node(relative, name, class_name)
    first = 1 if ast.get_docstring(node) else 0
    route = node.body.pop(first)
    assert isinstance(route, ast.ImportFrom) and len(route.names) == 1
    return _compile(node, {route.names[0].asname or route.names[0].name: target})


@pytest.mark.parametrize("relative,class_name", STORE_CASES)
@pytest.mark.parametrize("mode", ["dict", "list", "missing", "empty_history", "malformed"])
def test_current_store_protocol_preserves_original_read_fallback_copy_and_errors(tmp_path, relative, class_name, mode):
    canonical = _compile(_node("analysis_domains.py", "current", "AnalysisDomainStore"))
    actual = _forward(relative, "current", SimpleNamespace(current=canonical), class_name)
    original = _compile(_node(relative, "current", class_name, original=True))
    path = tmp_path / "current.json"
    rows = [{"nested": {"items": [1, 2]}, "revision": 2}]
    if mode == "dict":
        path.write_bytes(b"\xef\xbb\xbf" + '{"unicode":"β","nan":NaN}'.encode("utf8"))
    elif mode == "list":
        path.write_text("[]", encoding="utf8")
    elif mode == "malformed":
        path.write_text("{broken", encoding="utf8")
    if mode == "empty_history":
        rows = []
    calls = []
    owner = SimpleNamespace(current_path=path, revisions=lambda: (calls.append(True), rows)[1])
    if mode == "malformed":
        for function in (original, actual):
            with pytest.raises(json.JSONDecodeError):
                function(owner)
        return
    expected = original(owner)
    original_calls = len(calls)
    result = actual(owner)
    assert len(calls) == original_calls * 2
    assert json.dumps(result, sort_keys=True) == json.dumps(expected, sort_keys=True)
    if mode in ("list", "missing"):
        result["nested"]["items"].append(3)
        assert rows[0]["nested"]["items"] == [1, 2]
    if mode == "dict":
        assert calls == []


def test_domain_and_mass_sdk_protocols_preserve_selection_and_detached_configuration():
    from darsia.fluidflower.analysis_domains import ANALYSIS_DOMAIN_REVISION_FORMAT
    revision = {"format": ANALYSIS_DOMAIN_REVISION_FORMAT,
                "configuration": {"domains": [{"enabled": True, "nested": [1]}, {"enabled": False}, {"nested": [2]}]}}
    fake_store = lambda project: SimpleNamespace(current=lambda: revision)
    globals = dict(AnalysisDomainStore=fake_store, ANALYSIS_DOMAIN_REVISION_FORMAT=ANALYSIS_DOMAIN_REVISION_FORMAT, deepcopy_settings=deepcopy)
    canonical = _compile(_node("post_analysis.py", "_current_domain_state"), globals)
    original = _compile(_node("wasserstein.py", "_current_domain_state", original=True), globals)
    actual = _forward("wasserstein.py", "_current_domain_state", canonical)
    expected_revision, expected_configuration = original(object())
    result_revision, result_configuration = actual(object())
    assert result_revision is expected_revision is revision
    assert result_configuration == expected_configuration
    result_configuration["domains"][0]["nested"].append(9)
    assert revision["configuration"]["domains"][0]["nested"] == [1]
    row_owner = _compile(_node("pages/post_analysis/page.py", "_post_domain_rows", "PostAnalysisPage"), globals)
    row_original = _compile(_node("pages/wasserstein/page.py", "_analysis_domain_rows", "WassersteinPage", original=True), globals)
    row_actual = _forward("pages/wasserstein/page.py", "_analysis_domain_rows", SimpleNamespace(_post_domain_rows=row_owner), "WassersteinPage")
    owner = SimpleNamespace(project=object())
    assert row_actual(owner) == row_original(owner)
    assert len(row_actual(owner)) == 2
    run_owner = _compile(_node("post_analysis.py", "_mass_run_ids"))
    run_original = _compile(_node("wasserstein.py", "_mass_run_ids", original=True))
    run_actual = _forward("wasserstein.py", "_mass_run_ids", run_owner)
    for plan in ({}, {"configuration": {"run_bindings": [{"run_id": "r10"}, {"run_id": "r2"}, {"run_id": ""}, {"run_id": 3}]}}):
        assert run_actual(plan) == run_original(plan)
    revision["format"] = "not the declared domain format"
    assert actual(object()) == original(object()) == ({}, {})
    assert row_actual(owner) == row_original(owner) == []


def test_alpha_variants_reuse_exact_json_number_sha_and_queue_descriptor(tmp_path):
    json_owner = _compile(_node("analysis_domains.py", "_json_digest"))
    json_original = _compile(_node("indicator_response.py", "_json_digest", original=True))
    json_actual = _forward("indicator_response.py", "_json_digest", json_owner)
    for value in ({"unicode": "β", "nan": float("nan")}, [None, True, 2], None):
        assert json_actual(value) == json_original(value)
    number_owner = _compile(_node("mass_inputs.py", "_number"))
    number_original = _compile(_node("layer_painting.py", "_number", original=True))
    number_actual = _forward("layer_painting.py", "_number", number_owner)
    for value in (None, True, "2.5", "bad", float("nan"), float("inf")):
        assert number_actual(value) == number_original(value)
    path = tmp_path / "metadata.bin"
    path.write_bytes(b"local descriptor fixture\x00\xff")
    sha_owner = _compile(_node("project.py", "_sha256_file_uncached"))
    sha_original = _compile(_node("overflow.py", "_sha256_file", original=True))
    sha_actual = _forward("overflow.py", "_sha256_file", sha_owner)
    assert sha_actual(path) == sha_original(path)
    from darsia.fluidflower.pipeline.job_queue import retry_transient
    # The queue descriptor waits out a transient SMB read error (N111).
    descriptor_owner = _compile(_node("pipeline/job_queue.py", "descriptor"),
                                dict(retry_transient=retry_transient))
    descriptor_original = _compile(_node("pipeline/checkpoint.py", "_descriptor", original=True))
    descriptor_actual = _forward("pipeline/checkpoint.py", "_descriptor", descriptor_owner)
    assert descriptor_actual(path) == descriptor_original(path)
    for function in (descriptor_original, descriptor_actual):
        with pytest.raises(FileNotFoundError):
            function(tmp_path / "absent")
