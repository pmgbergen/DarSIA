"""Metadata-only source tests; no scientific imports, RAW reads or producers.

The private tree is an overlay. Execute exact AST bodies from its assembler and
the original metadata guards, with only bridge construction, descriptor IO and
link inspection replaced. This is not package/import or numerical integration QA.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Mapping, Sequence

import pytest


PRIVATE = Path(__file__).resolve().parents[2] / "src/darsia/fluidflower"
ORIGINAL = PRIVATE  # The original job-queue name helper of this repository.


def load_bodies(path, names, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    body = [node for node in tree.body if getattr(node, "name", None) in names
            or isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def subject():
    class Bridge:
        """Class-valid stand-in; no heavy BuildNativeBridge constructor."""
    ns = dict(__name__=__name__, deepcopy=deepcopy, dataclass=dataclass,
              Mapping=Mapping, Sequence=Sequence, Path=Path, math=math, json=json,
              re=re, BuildNativeBridge=Bridge, _no_links=lambda path: None)
    load_bodies(PRIVATE / "mass/native_optical_source.py", {"_bound", "_same"}, ns)
    load_bodies(ORIGINAL / "pipeline/job_queue.py", {"name"}, ns)
    load_bodies(PRIVATE / "pipeline/build_raw_phase.py",
                {"FORMAT", "_SPEC", "_ROLES", "_keys", "_specification"}, ns)
    load_bodies(PRIVATE / "pipeline/build_phase_inputs.py",
                {"BuildPhaseReplacement", "assemble_current_build_phase_inputs"}, ns)
    return SimpleNamespace(**ns)


@pytest.fixture
def inputs(tmp_path, subject):
    def bound(label, index):
        return dict(path=str(tmp_path / label), sha256=f"{index:064x}", size_bytes=17)
    docs = {}
    def document(label, value):
        pin = bound(label, 100 + len(docs))
        docs[pin["path"]] = deepcopy(value)
        return pin
    class Reader:
        def __init__(self):
            self.reads = []
        def path(self, pin):
            assert pin["path"] in docs, "Assembler must not read RAW/products"
            self.reads.append(deepcopy(pin))
            return Path(pin["path"])
        def json(self, pin):
            self.path(pin)
            return deepcopy(docs[pin["path"]])
    raws = [bound(f"raw{i}.ARW", i + 1) for i in range(5)]
    photo, floor = bound("photo.npz", 50), bound("floor.npy", 51)
    native = document("native.json", dict(support={"original": True}))
    old_native = document("old_native.json", dict(support={"original": True}))
    phase_doc = dict(chain_assets=dict(floor=floor),
                     parameters=dict(geometry=dict(floor_source_origin_xy=[-416, -496])))
    phase = document("phase.json", phase_doc)
    old_phase = document("old_phase.json", phase_doc)
    base = document("base.json", dict(run="ac17", native=dict(configuration=old_native),
        phase=dict(configuration=old_phase), selection_index_field="frame_index",
        frames=[{"historical_cohort_not_used": True}]))
    matrix = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    times, indices = [-.25, .125, .247, .375, .497], [0, 7, 12, 20, 31]
    specs, rows, replacements = [], [], []
    for i, raw in enumerate(raws):
        previous = None if i == 0 else raws[i - 1]["sha256"]
        geometry = bound(f"geometry{i}.json", 60 + i)
        area = dict(role="selected", factor=1., revision={"measured": True})
        static = dict(components=bound(f"static{i}.npz", 70 + i))
        source = dict(run="ac17", frame_index=indices[i], elapsed_h=times[i],
            source_sha256=raw["sha256"], previous_source_sha256=previous, selected=i != 0,
            source_binding=dict(source_sha256=raw["sha256"], raw_acquisition=deepcopy(raw)))
        roles = dict(area=area, current_h=matrix, reference_to_current_h=matrix, static=static,
            retained_source_cache=None, encoding_contract=None, light_epoch_id="saved",
            inherited_spacing_m=[.001, .001], probability_spacing_m=[.001, .001],
            placement_metadata={}, photo_sand_reference=photo,
            photo_proof=dict(sha256=photo["sha256"], phase_labels_used=False))
        specs.append(dict(case_id=f"frame{i}", frame_index=indices[i], elapsed_h=times[i],
            source=source, raw_acquisition=deepcopy(raw), geometry=geometry, phase_roles=roles))
        rows.append(dict(raw_path=raw["path"], raw_sha256=raw["sha256"], elapsed_h=times[i],
            current_frame_index=indices[i], previous_raw_sha256=previous, baseline=i == 0,
            fit_candidate=i == 4, nominal_fit_target_h=.5 if i == 4 else None))
        replacements.append(subject.BuildPhaseReplacement(raw, geometry, area, static, matrix, matrix))
    series = document("series.toml", {"fixture": True})
    selection = document("selection.json", dict(format="ffac-local-p1-selection-data-v1",
        decision=dict(actor="Reviewer", reason="Explicit current selection"),
        history_policy="baseline plus all selected acquisitions in original chronological order",
        input_authority=dict(current_series=series), runs=[dict(run="ac17", baseline=rows[0],
            initial_previous_raw_sha256=raws[0]["sha256"], selected_frames=rows[1:])]))
    bridge = subject.BuildNativeBridge()
    bridge.context = SimpleNamespace(run="ac17", provenance=dict(native_configuration=native,
        raw_acquisition=raws[0]))
    reader = Reader()
    args = dict(bridge=bridge, base_run_configuration=base, reference_spec=specs[0],
        initial_previous_spec=deepcopy(specs[0]), frame_specs=specs[1:], replacements=replacements,
        phase_configuration=phase, temporal_pad_px=0, selection_record=selection, reader=reader)
    return SimpleNamespace(args=args, docs=docs, rows=rows, raws=raws, series=series)


def test_exact_current_selection_roundtrip_no_hash_cycle(inputs, subject):
    args = inputs.args
    before = deepcopy({k: v for k, v in args.items() if k not in {"reader", "bridge"}})
    result = subject.assemble_current_build_phase_inputs(**args)
    assert json.loads(json.dumps(result)) == result
    assert {k: v for k, v in args.items() if k not in {"reader", "bridge"}} == before
    assert result["reference_spec"] == result["initial_previous_spec"]
    assert result["reference_spec"]["elapsed_h"] == -.25
    assert [r["frame_index"] for r in result["frames"]] == [7, 12, 20, 31]
    assert [r["elapsed_h"] for r in result["frames"]] == [.125, .247, .375, .497]
    assert len(result["frames"]) == 4  # three early history outputs and one fit candidate
    assert "local_build_reuse" not in json.dumps(result)
    assert all(not Path(raw["path"]).exists() for raw in inputs.raws)
    result["frames"][0]["source"]["elapsed_h"] = 999
    assert args["frame_specs"][0]["source"]["elapsed_h"] == .125


@pytest.mark.parametrize("change", ["order", "missing_history", "skip_history",
    "clock", "baseline_time", "initial_previous", "renumber", "partial", "duplicate",
    "floor", "origin", "native", "reuse_series", "photo_labels", "cache"])
def test_refuses_changed_history_clock_bindings_and_roles(inputs, subject, change):
    a = inputs.args
    if change == "order":
        a["frame_specs"].reverse()
    elif change == "missing_history":
        del a["frame_specs"][1]
    elif change == "skip_history":
        a["frame_specs"][0]["source"]["previous_source_sha256"] = inputs.raws[2]["sha256"]
    elif change in {"clock", "renumber"}:
        field, value = ("elapsed_h", .5) if change == "clock" else ("frame_index", 4)
        a["frame_specs"][-1][field] = value
        a["frame_specs"][-1]["source"][field] = value
    elif change == "baseline_time":
        for key in ("reference_spec", "initial_previous_spec"):
            a[key]["elapsed_h"] = a[key]["source"]["elapsed_h"] = 0.
    elif change == "initial_previous":
        a["initial_previous_spec"] = deepcopy(a["frame_specs"][0])
    elif change == "partial":
        a["replacements"].pop()
    elif change == "duplicate":
        a["replacements"].append(a["replacements"][0])
    elif change in {"floor", "origin"}:
        doc = inputs.docs[a["phase_configuration"]["path"]]
        if change == "floor":
            doc["chain_assets"]["floor"]["sha256"] = "f" * 64
        else:
            doc["parameters"]["geometry"]["floor_source_origin_xy"] = [0, 0]
    elif change == "native":
        pin = a["bridge"].context.provenance["native_configuration"]
        inputs.docs[pin["path"]]["support"] = {"changed": True}
    elif change == "reuse_series":
        a["bridge"].context.provenance["local_build_reuse"] = {
            "current_series_manifest": dict(inputs.series, sha256="f" * 64)}
    elif change == "photo_labels":
        a["frame_specs"][0]["phase_roles"]["photo_proof"]["phase_labels_used"] = True
    else:
        a["frame_specs"][0]["phase_roles"]["retained_source_cache"] = {"old": True}
    before = deepcopy({k: v for k, v in a.items() if k not in {"reader", "bridge"}})
    with pytest.raises(ValueError):
        subject.assemble_current_build_phase_inputs(**a)
    assert {k: v for k, v in a.items() if k not in {"reader", "bridge"}} == before
    assert all(not Path(raw["path"]).exists() for raw in inputs.raws)


def test_optional_matching_reuse_does_not_embed_receipt(inputs, subject):
    inputs.args["bridge"].context.provenance["local_build_reuse"] = {
        "current_series_manifest": inputs.series}
    result = subject.assemble_current_build_phase_inputs(**inputs.args)
    assert "local_build_reuse" not in json.dumps(result)
    assert result["provenance"]["local_current_selection"]["selection_record"] == inputs.args["selection_record"]
