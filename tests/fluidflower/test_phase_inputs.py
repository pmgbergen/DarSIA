"""Strict synthetic input admission without NAS or inference calls."""
from copy import deepcopy
from collections import OrderedDict
import hashlib
from dataclasses import dataclass
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower.native.io import sha256_file
from darsia.fluidflower.native import sidecar
from darsia.fluidflower.phase import inputs
from darsia.fluidflower.phase.metrics import validate_pixel_area
from darsia.fluidflower.pipeline import phase_inputs as loader


BASE = "f0d338aa2be5e6ee8be6a2db0bd8f6fb70e9bd5a"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def parameters():
    return json.loads((Path(__file__).parent / "fixtures" / "phase_input_parameters.json").read_bytes())


@pytest.fixture(scope="module")
def originals():
    result = {}
    for short in ("phase_ffac_selected_native_inputs_corrected", "phase_retained_source_inference"):
        module = ModuleType("_preserved_inputs_" + short)
        module.__package__ = "darsia.fluidflower"
        sys.modules[module.__name__] = module
        module.__dict__.update(np=np, Path=Path, dataclass=dataclass, json=json,
            re=re, sha256=sha256, sha256_file=sha256_file, Image=Image,
            validate_pixel_area=validate_pixel_area)
        source = subprocess.check_output(["git", "show", BASE + ":src/darsia/fluidflower/" + short + ".py"], cwd=REPO)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                try: exec(compile(ast.Module([node], type_ignores=[]), BASE, "exec"), module.__dict__)
                except NameError: pass
        selected = {"MaskedNativeInput", "TemporalNativeInput", "_source_array", "_validate",
            "original_native_input", "new_native_input", "temporal_policy", "original_temporal_input",
            "new_temporal_input", "original_photo_sand", "RetainedSourceInferenceCache", "_asset", "_json",
            "_identity", "_mask_digest", "probability_encoding_contract", "prepare_retained_source_cache"}
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
        exec(compile(ast.Module(definitions, type_ignores=[]), BASE, "exec"), module.__dict__)
        result[short] = module
    masked, retained = result.values()
    masked.bound_path = lambda item: retained._asset(item, "test input")
    masked.MASK_NAMES = retained.MASK_NAMES
    parameter_data = json.loads((Path(__file__).parent / "fixtures" / "phase_input_parameters.json").read_bytes())
    retained.CO2_SHA256 = parameter_data["retained"]["expected_models"]["co2"]
    retained.G3_SHA256 = parameter_data["retained"]["expected_models"]["g3"]
    retained.REFIT_SHA256 = parameter_data["retained"]["expected_models"]["regional_refit"]
    yield masked, retained
    for module in result.values(): sys.modules.pop(module.__name__, None)


def same(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b, equal_nan=True)
        assert a.flags.writeable == b.flags.writeable
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert type(a) is type(b) and a == b


def same_result(a, b):
    assert a.__dataclass_fields__.keys() == b.__dataclass_fields__.keys()
    for key in a.__dataclass_fields__: same(getattr(a, key), getattr(b, key))


def asset(path):
    return dict(path=str(path), sha256=sha256_file(path), size_bytes=path.stat().st_size)


def json_asset(folder, name, value):
    path = folder / name
    path.write_text(json.dumps(value, allow_nan=False))
    return asset(path)


def probability_receipt(parameters, shape=(4, 5), labelled=False):
    p = parameters["retained"]
    layout = p["encoding"]["labelled_layout" if labelled else "compact_layout"]
    working = [2, 3]
    files = {name: dict(dtype="float16" if labelled or name != "g3_pixel.npy" else "uint8",
                       shape=list(shape) if labelled or name == "g3_pixel.npy" else working)
             for name in p["encoding"]["probability_files"]}
    return dict(contract=p["contracts"]["PROBABILITY_CONTRACT"], complete=True,
        binding=dict(provisional=False, labelled=labelled,
                     g3_model_sha256=p["expected_models"]["g3"], regional_refit_sha256=p["expected_models"]["regional_refit"]),
        metadata=dict(storage_layout=layout, native_shape_hw=list(shape), regional_working_shape_hw=working,
                      co2_head_sha256=p["expected_models"]["co2"], bitorder="little",
                      masks_order=list(p["mask_names"]), resize_rule=p["encoding"]["resize_rule"],
                      g3_compact_rule=p["encoding"]["compact_rule"]), files=files)


def test_original_and_new_native_masked_bits_and_invalid_support(parameters, originals, tmp_path):
    old, _ = originals
    log = np.ones((3, 4, 5), np.float32); log[:, 0, 0] = np.nan
    valid = np.isfinite(log).all(0)
    lp, vp = tmp_path / "log.npy", tmp_path / "valid.npy"
    np.save(lp, log); np.save(vp, valid)
    fields = parameters["inputs"]["fields"]["native"]
    row = dict(zip(fields, (str(lp), sha256_file(lp), str(vp), sha256_file(vp))))
    row["source_sha256"] = "a" * 64
    current = log.copy(); current[:, 0, 0] = 2
    actual = loader.original_native_input(row, scientific_native_log=current, parameters=parameters["inputs"])
    same_result(actual, old.original_native_input(row, scientific_native_log=current))
    current[0, 1, 1] = -0.0; log[0, 1, 1] = 0.0
    with pytest.raises(ValueError, match="historical valid"):
        inputs.original_native_input(log, valid, provenance={}, scientific_native_log=current, parameters=parameters["inputs"])
    photo = valid.copy(); photo[-1] = False
    bindings = dict(source_binding={"source_sha256": "a" * 64}, photo_mask_binding={"sha256": "b" * 64, "phase_labels_used": False})
    same_result(inputs.new_native_input(log, valid, photo, parameters=parameters["inputs"], **bindings),
                old.new_native_input(log, valid, photo, **bindings))
    assert np.isfinite(log[:, -1]).all()
    with pytest.raises(ValueError, match="finite/support"):
        inputs.validate_native_input(log, ~valid)


def test_temporal_absence_partial_origin_and_source_owned_crop(parameters, originals, tmp_path):
    old, _ = originals
    p = parameters["inputs"]
    same(inputs.temporal_policy({}, parameters=p), old.temporal_policy({}))
    with pytest.raises(ValueError, match="partial"):
        inputs.temporal_policy({p["fields"]["temporal"][0]: "ring"}, parameters=p)
    log = np.ones((3, 4, 5), np.float32); log[:, 1, 2] = np.nan
    native = inputs.MaskedNativeInput(log, np.isfinite(log).all(0), {})
    prior = old.MaskedNativeInput(log, native.valid, {})
    policy = inputs.temporal_policy({}, parameters=p)
    same_result(inputs.original_temporal_input(native, policy=policy, parameters=p), old.original_temporal_input(prior, {}))
    full = np.arange(3 * 10 * 12, dtype=np.float32).reshape(3, 10, 12)
    frozen = dict(contract=p["contract"], pad_px=1, origin_xy=[-1, -1], ring={"sha256": "a" * 64})
    kwargs = dict(native_box_xywh=(3, 3, 5, 4), frozen_policy=frozen)
    same_result(inputs.new_temporal_input(native, full, parameters=p, **kwargs), old.new_temporal_input(prior, full, **kwargs))
    with pytest.raises(ValueError, match="escapes"):
        inputs.new_temporal_input(native, full, native_box_xywh=(0, 0, 5, 4), frozen_policy=frozen, parameters=p)


def test_original_ring_loader_uses_strict_existing_sidecar(parameters, originals, tmp_path, monkeypatch):
    old, _ = originals
    p = parameters["inputs"]
    native_log = np.ones((3, 3, 4), np.float32); native_log[:, 0, 0] = np.nan
    native = inputs.MaskedNativeInput(native_log, np.isfinite(native_log).all(0), {})
    pad = 1; rp = {"contract": "fixture-ring-contract"}
    ring = tmp_path / "ring.npz"
    np.savez(ring, top=np.full((3, 1, 6), 2., np.float32), bottom=np.full((3, 1, 6), 3., np.float32),
             left=np.full((3, 3, 1), 4., np.float32), right=np.full((3, 3, 1), 5., np.float32),
             pad_px=pad, contract=rp["contract"])
    fields = p["fields"]["temporal"]
    row = dict(zip(fields, (str(ring), sha256_file(ring), pad, [-pad, -pad])))
    adapter = ModuleType("darsia.fluidflower.tools.ffac_v5_extended_reference")
    adapter.assemble_extended_log = lambda *a, **kw: sidecar.assemble_extended_log(*a, parameters=rp, **kw)
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    actual = loader.original_temporal_input(native, row, parameters=p, ring_parameters=rp)
    expected = old.original_temporal_input(old.MaskedNativeInput(native.log, native.valid, {}), row)
    same_result(actual, expected)
    assert np.array_equal(actual.log[:, pad:-pad, pad:-pad], native.log, equal_nan=True)
    with pytest.raises(ValueError, match="padding"):
        loader.temporal_policy(dict(row, **{fields[3]: [0, 0]}), parameters=p)


@pytest.mark.parametrize("labelled", [False, True])
def test_probability_encoding_exact_contract_and_controls(parameters, originals, labelled):
    _, old = originals
    receipt = probability_receipt(parameters, labelled=labelled)
    same(loader.probability_encoding_contract(receipt, (4, 5), parameters=parameters["retained"]),
         old.probability_encoding_contract(receipt, (4, 5)))
    for key, value in (("bitorder", "big"), ("masks_order", []), ("regional_working_shape_hw", [True, 3]),
                       ("resize_rule", "threshold before resize")):
        bad = deepcopy(receipt); bad["metadata"][key] = value
        with pytest.raises(ValueError): loader.probability_encoding_contract(bad, (4, 5), parameters=parameters["retained"])
        with pytest.raises(ValueError): old.probability_encoding_contract(bad, (4, 5))


def test_photo_mask_order_member_grid_and_content_bound_alias(parameters, originals, tmp_path):
    old, _ = originals
    p = parameters["inputs"]; mask = np.zeros((4, 5), bool); mask[1:3, 1:4] = True
    packed = np.array([np.packbits(mask.ravel(), bitorder="little")] * len(p["mask_names"]))
    path = tmp_path / "masks.npz"; np.savez(path, masks=packed)
    descriptor = asset(path); receipt = dict(metadata={"native_shape_hw": list(mask.shape), "masks_order": p["mask_names"], "bitorder": "little"}, files={p["mask_file"]: descriptor})
    same(loader.original_photo_sand(receipt, parameters=p), old.original_photo_sand(receipt))
    logical = str(tmp_path / "relocated-masks.npz")
    alias_receipt = deepcopy(receipt); alias_receipt["files"][p["mask_file"]]["path"] = logical
    got, provenance = loader.original_photo_sand(alias_receipt, parameters=p, aliases={logical: str(path)})
    assert np.array_equal(got, mask) and provenance["original_probability_masks"]["path"] == logical
    assert not got.flags.writeable
    with pytest.raises(ValueError, match="packing|grid"):
        inputs.original_photo_sand(packed[:, :1], receipt["metadata"], descriptor, parameters=p)
    assert loader.load_bound_array(dict(descriptor, array_key="masks")).dtype == np.uint8
    with pytest.raises(ValueError, match="Changed"):
        loader.bound_path(dict(descriptor, sha256="0" * 64))


def cache_fixture(tmp_path, parameters):
    shape = (4, 5); source = "a" * 64; raw = "b" * 64
    current = np.ones((3, *shape), np.float32); valid = np.ones(shape, bool)
    current[:, 0, 0] = np.nan; valid[0, 0] = False
    lp, vp = tmp_path / "parent.npy", tmp_path / "valid.npy"
    np.save(lp, current); np.save(vp, valid)
    frame = dict(source_sha256=source, v5_raw_sha256=raw, v5_log_linear_rgb_path=str(lp),
        v5_log_linear_rgb_sha256=sha256_file(lp), v5_log_linear_valid_mask_path=str(vp), v5_log_linear_valid_mask_sha256=sha256_file(vp))
    manifest = json_asset(tmp_path, "native.json", dict(run="ac01", analysis_image_shape_hw=list(shape), frame_sources=[frame]))
    names = parameters["retained"]["mask_names"]
    masks = {name: np.ones(shape, bool) for name in names}
    masks["photo_sand_domain"][0, 0] = False
    masks["feature_support"][0, 0] = False
    masks["regional_native_domain"] = masks["photo_sand_domain"] & masks["feature_support"]
    masks["flatfield_valid"][2, 2] = False; masks["frozen_unresolved"][:] = False
    packed = np.array([np.packbits(masks[n].ravel(), bitorder="little") for n in names])
    mp = tmp_path / "masks.npz"; np.savez(mp, masks=packed)
    labels = np.ones(shape, np.uint8); labels[0, 0] = 255; labels[1, 1] = 2
    labels_path = tmp_path / "labels.png"; Image.fromarray(labels).save(labels_path)
    receipt = probability_receipt(parameters, shape)
    receipt["binding"].update(run="ac01", frame_index=0, native_manifest_sha256=manifest["sha256"])
    receipt["metadata"].update(native_manifest_path=manifest["path"], pixel_area_mm2=.027053)
    receipt["files"]["extra_masks.packbits.npz"] = asset(mp)
    export = json_asset(tmp_path, "probability.json", receipt)
    categorical = json_asset(tmp_path, "categorical.json", dict(contract=parameters["retained"]["contracts"]["FRAME_CONTRACT"], complete=True,
        binding={"run": "ac01", "frame_index": 0, "native_manifest_sha256": manifest["sha256"]}, labels_map=asset(labels_path)))
    binding = dict(contract=parameters["retained"]["contracts"]["CONTRACT"], run="ac01", frame_index=0,
                   parent_native_manifest=manifest, parent_frame_receipt=categorical, parent_probability_receipt=export)
    audit = dict(cpu_only=True, training=False, parent_cache={"parent_float_bytes_preserved": True},
        processor={"parent_native_manifest_sha256": manifest["sha256"]}, sources={"camera_jpeg": {"sha256": source}, "camera_arw": {"sha256": raw}})
    args = dict(run="ac01", frame_index=0, current_source_sha256=source, current_log=current,
                allowed_domain=np.ones(shape, bool), producer_audit=audit)
    return binding, args


def test_retained_cache_all_original_proofs_populations_and_no_release(parameters, originals, tmp_path):
    _, old = originals
    binding, args = cache_fixture(tmp_path, parameters)
    actual = loader.prepare_retained_source_cache(binding, parameters=parameters["retained"], **args)
    expected = old.prepare_retained_source_cache(binding, **args)
    same_result(actual, expected)
    assert actual.audit["parent_scientific_bits_verified"] is True
    assert actual.prior_fill_eligible[2, 2] and not actual.cached_parent_modelled[2, 2]
    assert not actual.recomputed_old_domain_ring.any()
    assert not actual.labels.flags.writeable
    same_result(loader.prepare_retained_source_cache(None, parameters=parameters["retained"], **args),
                old.prepare_retained_source_cache(None, **args))
    changed = args["current_log"].copy(); changed[0, 1, 1] = -0.0
    for function, kw in ((loader.prepare_retained_source_cache, {"parameters": parameters["retained"]}), (old.prepare_retained_source_cache, {})):
        with pytest.raises(ValueError, match="float bits"):
            function(binding, **dict(args, current_log=changed), **kw)
        with pytest.raises(ValueError, match="changed acquisition|different"):
            function(binding, **dict(args, current_source_sha256="c" * 64), **kw)
    with pytest.raises(ValueError, match="does not enable"):
        loader.prepare_retained_source_cache(binding, **args, parameters=parameters["retained"], recompute_old_domain_ring=True)


def test_model_role_byte_admission_refit_and_converged_grid(parameters, tmp_path):
    frozen = dict(settings={"sigma": .1}, feature_names=["a"], heads={"co2": {"value": 1}, "gas": {"value": 2}})
    refit = deepcopy(frozen); refit["heads"]["gas"] = {"value": 3}
    gas = dict(variant="four", feature_names=["a", "b", "c", "d"], fit={"converged": True})
    bindings = {"original": json_asset(tmp_path, "original.json", frozen), "refit": json_asset(tmp_path, "refit.json", refit), "gas": json_asset(tmp_path, "gas.json", gas)}
    requirements = dict(expected_sha256={k: v["sha256"] for k, v in bindings.items()},
        refit_pair=["original", "refit"], gas_head="gas", gas_variant="four", gas_feature_count=4)
    docs = loader.load_model_documents(bindings, requirements=requirements, heads_parameters={"gas_head_name": "gas"})
    assert docs["gas"] == gas
    wrong = deepcopy(docs); wrong["refit"]["heads"]["co2"] = {"value": 5}
    with pytest.raises(ValueError, match="non-gas"):
        loader.validate_model_documents(wrong, bindings, requirements=requirements, heads_parameters={"gas_head_name": "gas"})
    wrong = deepcopy(docs); wrong["gas"]["fit"]["converged"] = 1
    with pytest.raises(ValueError, match="converged"):
        loader.validate_model_documents(wrong, bindings, requirements=requirements, heads_parameters={"gas_head_name": "gas"})
    with pytest.raises(ValueError, match="[Cc]hanged"):
        loader.load_bound_json(dict(bindings["gas"], sha256="0" * 64))


def original_baseline_cache():
    source = subprocess.check_output(["git", "show", BASE +
        ":src/darsia/fluidflower/phase_ffac_selected_fixed_baseline_decode_cache.py"], cwd=REPO)
    namespace = dict(np=np, hashlib=hashlib, OrderedDict=OrderedDict)
    exec(compile(source, BASE + ":fixed-baseline-cache", "exec"), namespace)
    return namespace


def test_fixed_baseline_cache_exact_keys_budget_mutability_and_validation(parameters):
    original = original_baseline_cache()
    p = parameters["fixed_baseline_cache"]
    cache = loader.FixedBaselineDecodeCache(max_bytes=1024, parameters=p)
    previous = original["FixedBaselineDecodeCache"](max_bytes=1024)
    log = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    valid = np.ones((4, 5), bool)
    calls = []
    def decode(log, valid):
        calls.append(1)
        return np.moveaxis(log.copy(), 0, -1)
    role = p["fixed_roles"][0]
    for _ in range(2):
        actual = cache.decode(role, log, valid, decoder=decode)
        expected = previous.decode(role, log, valid, decoder=decode)
        same(actual, expected)
        assert actual.flags.writeable
        actual[:] = -1; expected[:] = -1
    assert cache.receipt()["hits"] == 1 and len(calls) == 2
    changed = log.copy(); changed[0, 0, 0] = -0.0
    same(cache.decode(role, changed, valid, decoder=decode), previous.decode(role, changed, valid, decoder=decode))
    same(cache.decode("current", log, valid, decoder=decode), previous.decode("current", log, valid, decoder=decode))
    same(cache.decode(role, log.astype(np.float64), valid, decoder=decode), previous.decode(role, log.astype(np.float64), valid, decoder=decode))
    for _ in range(2):
        same(cache.validate_fixed(p["validation_role"], log, valid, decoder=decode),
             previous.validate_fixed(p["validation_role"], log, valid, decoder=decode))
    same(cache.receipt(), previous.receipt())
    for budget in (-1, True, p["maximum_bytes"] + 1):
        with pytest.raises(ValueError): loader.FixedBaselineDecodeCache(max_bytes=budget, parameters=p)
        with pytest.raises(ValueError): original["FixedBaselineDecodeCache"](max_bytes=budget)
    tiny = loader.FixedBaselineDecodeCache(max_bytes=1, parameters=p)
    tiny.decode(role, log, valid, decoder=decode)
    assert tiny.receipt()["oversized_results"] == 1 and tiny.receipt()["cached_entries"] == 0
    strided = log[:, ::2, ::2]
    assert loader._array_sha256(strided, buffer_elements=p["array_hash_buffer_elements"]) == original["_array_sha256"](strided)
    cache.clear(); previous.clear(); same(cache.receipt(), previous.receipt())


def test_sparse_scientific_reconstruction_original_windows_and_selected_wrapper(parameters, tmp_path, monkeypatch):
    p = dict(parameters["scientific"], photograph_canvas_wh=[9, 7], native_shape_hw=[3, 4], native_box_xywh=[2, 2, 4, 3])
    centre = tmp_path / "centre.npy"; valid = tmp_path / "valid.npy"; ring = tmp_path / "ring.npy"
    np.save(centre, np.full((3, 3, 4), 1., np.float32)); np.save(valid, np.ones((3, 4), bool))
    np.save(ring, np.full((3, 2, 4), 2., np.float32))
    descriptor = dict(contract=p["contract"], complete=True, storage=p["storage"], phase_shape_hw=[8, 9],
        native_centre={"log_linear_rgb": asset(centre)},
        ring_windows=[{"canvas_xyxy": [2, 0, 6, 2], "log_linear_rgb": asset(ring)}])
    source = subprocess.check_output(["git", "show", BASE + ":src/darsia/fluidflower/tools/run_ffac_rig_camera_inference.py"], cwd=REPO)
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "compose_extended_scientific")
    metadata_source = subprocess.check_output(["git", "show", BASE + ":src/darsia/fluidflower/phase_ffac_rig_camera_metadata.py"], cwd=REPO)
    grid_nodes = [n for n in ast.parse(metadata_source).body if isinstance(n, ast.FunctionDef) and n.name in {"phase_grid", "require"}]
    grid_scope = dict(CANVAS_WH=p["photograph_canvas_wh"], SOURCE_SHAPE=p["native_shape_hw"], SOURCE_BOX=p["native_box_xywh"], ORIGIN_XY=[-2, -2], deepcopy=deepcopy)
    exec(compile(ast.Module(grid_nodes, type_ignores=[]), BASE, "exec"), grid_scope)
    namespace = dict(np=np, load_array=loader.load_bound_array,
                     metadata=SimpleNamespace(SOURCE_SHAPE=p["native_shape_hw"], SOURCE_BOX=p["native_box_xywh"], phase_grid=grid_scope["phase_grid"]))
    exec(compile(ast.Module([original], type_ignores=[]), BASE, "exec"), namespace)
    external = ModuleType("darsia.fluidflower.phase_ffac_rig_raw_extension"); external.CONTRACT = p["contract"]
    monkeypatch.setitem(sys.modules, external.__name__, external)
    same(loader.compose_extended_scientific(descriptor, parameters=p), namespace["compose_extended_scientific"](descriptor))
    receipt = dict(native_scientific={"log_linear_rgb": asset(centre), "valid_mask": asset(valid)}, full_extended_scientific=descriptor)
    selected_source = subprocess.check_output(["git", "show", BASE + ":src/darsia/fluidflower/phase_ffac_selected_sources.py"], cwd=REPO)
    selected = next(n for n in ast.parse(selected_source).body if isinstance(n, ast.FunctionDef) and n.name == "load_selected_scientific")
    scope = dict(np=np, bound_path=loader.bound_path, rig=SimpleNamespace(metadata=namespace["metadata"], compose_extended_scientific=namespace["compose_extended_scientific"]))
    exec(compile(ast.Module([selected], type_ignores=[]), BASE, "exec"), scope)
    same(loader.load_selected_scientific(receipt, parameters=p), scope["load_selected_scientific"](receipt))
    for window in ([2, 2, 6, 4], [-1, 0, 3, 2], [2., 0, 6, 2]):
        bad = deepcopy(descriptor); bad["ring_windows"][0]["canvas_xyxy"] = window
        with pytest.raises(ValueError): loader.compose_extended_scientific(bad, parameters=p)
        with pytest.raises(ValueError): namespace["compose_extended_scientific"](bad)
    bad = deepcopy(descriptor); bad["phase_shape_hw"] = [6, 9]
    with pytest.raises(ValueError, match="bottom padding"):
        loader.compose_extended_scientific(bad, parameters=p)
    same(loader.load_bound_array(dict(asset(centre), array_key=None)), np.load(centre, mmap_mode="r"))
