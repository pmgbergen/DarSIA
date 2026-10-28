"""Small scientific/IO tests; no campaign input or remote numeric run."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import darsia

from darsia.fluidflower import n1_arithmetic as arithmetic
from darsia.fluidflower import n1_geometry as geometry
from darsia.fluidflower import n1_models as models
from darsia.fluidflower import n1_pipeline as pipeline
from darsia.fluidflower import n1_statistics as statistics
from darsia.fluidflower.concentration_joint import dilution_factor
from darsia.fluidflower.production_precision import PixelMath

BASE_COMMIT = "6dfca622f5ed8ebe5c3ea6474025dcf5bf28023d"


def original_function(filename, name, namespace, *, class_name=None):
    """Load a historical function codeobject from unchanged Git source."""
    repo = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output(["git", "show", BASE_COMMIT+":src/darsia/fluidflower/"+filename],cwd=repo).decode("utf-8")
    tree = ast.parse(raw)
    body = tree.body if class_name is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n,ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node],type_ignores=[])
    context = dict(namespace)
    exec(compile(ast.fix_missing_locations(module),filename,"exec"),context)
    return context[name]


def assert_same(first, second):
    """Compare complete numerical structures, including native NaN patterns."""
    if isinstance(first,np.ndarray):
        assert first.dtype == second.dtype
        np.testing.assert_array_equal(first,second)
    elif isinstance(first,dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same(first[key],second[key])
    elif isinstance(first,(list,tuple)):
        assert len(first) == len(second)
        for left,right in zip(first,second):
            assert_same(left,right)
    else:
        assert first == second


@pytest.fixture
def tiny_config(tmp_path):
    """Build a genuine tiny saved-model dataset with no fitted parameters."""
    def save(name, value, kind="json"):
        path=tmp_path/name
        if kind == "json":
            path.write_text(json.dumps(value),encoding="utf-8")
        elif kind == "npy":
            np.save(path,value,allow_pickle=False)
        elif kind == "npz":
            np.savez(path,**value)
        elif kind == "png":
            from PIL import Image
            Image.fromarray(value).save(path)
        return pipeline.asset(path)
    shape=(3,3)
    domain=np.ones(shape,bool)
    facies=np.ones(shape,np.uint8)
    facies[:,2]=2
    labels=np.ones(shape,np.uint8)
    labels[1,1]=2
    labels[2,0]=0
    physical=save("physical.npz",dict(facies=facies,domain=domain,porosity=np.full(shape,.3),depth_m=np.full(shape,.01)),"npz")
    clear=save("clear.npy",np.ones((*shape,3),np.float32),"npy")
    valid=save("valid.npy",domain,"npy")
    phase=save("phase.png",labels,"png")
    old_domain=save("old.png",np.zeros(shape,np.uint8),"png")
    classes=save("classes.npz",dict(rig_domain=domain,modelled=domain,filled_observed=domain,filled_unobserved=np.zeros(shape,bool)),"npz")
    area=save("area.npz",dict(pixel_area_mm2=np.ones(shape)),"npz")
    top=save("top.npz",dict(keep=domain),"npz")
    floor=save("floor.npy",np.full(3,2.),"npy")
    def raw(name, rgb):
        value=np.broadcast_to(np.asarray(rgb,np.float32),(*shape,3)).copy()
        logs=np.moveaxis(np.log(value*np.float32(65535)),2,0).astype(np.float32)
        bound=save(name,logs,"npy")
        return dict(phase_shape_hw=list(shape),native_origin_xy=[0,0],
            native_centre=dict(native_shape_hw=list(shape),log_linear_rgb=bound,valid_mask=valid),
            ring_windows=[],binding=dict(geometry={"H_ac28_to_frame_native":np.eye(3).tolist()}))
    current,baseline=raw("current.npy",[.8,.7,.6]),raw("baseline.npy",[1.,1.,1.])
    colors,signals={},{}
    flash_bounds=dict(min_value_aq=0.,max_value_aq=1.,min_value_g=0.,max_value_g=1.)
    for label in (1,2):
        base=tmp_path/("color_"+str(label))
        path=darsia.ColorPath(colors=[np.array([1.,1.,1.]),np.array([.8,.7,.6])])
        darsia.ColorPathInterpolation(path,darsia.ColorMode.RELATIVE,[0.,1.]).save(base)
        colors[str(label)]=pipeline.asset(base.with_suffix(".json"))
        base=tmp_path/("signal_"+str(label))
        darsia.PWTransformation([0.,1.],[0.,1.]).save(base)
        signals[str(label)]=pipeline.asset(base.with_suffix(".csv"))
    base=tmp_path/"flash"
    darsia.SimpleFlash(**flash_bounds).save(base)
    metadata=save("optics.json",dict(label_ids=[1,2],ignore_labels=[2],basis="facies",flash_bounds=flash_bounds))
    config=dict(format=pipeline.CONFIG_FORMAT,run="experiment",frames=[],
        shared=dict(physical_reference=physical,reference_matrix=[[1.,0.,0.],[0.,-1.,3.],[0.,0.,1.]],
            floor_curve=floor,floor_origin_xy=[0,0],depth_points=[[0,1,.01],[3,1,.01],[0,3,.01],[3,3,.01]],
            porosity_by_facies={"1":.3,"2":.3},swi_by_facies={"1":.2,"2":.25},unquantified_facies=[2],
            clearwater=dict(rgb=clear,valid=valid,H_reference_to_grid=np.eye(3).tolist()),
            optics=dict(metadata=metadata,color_paths=colors,signals=signals,flash=pipeline.asset(base.with_suffix(".json")))),
        calibration=save("calibration.json",dict(parameters=dict(candidate="N1",gain=1.2,beta=.3))),
        physical_correction=save("correction.json",dict(correction_matrix=np.eye(3).tolist())),
        anchor=dict(elapsed_h=.5,affected_pore_volume_m3=1e-8),recipe=dict(alkalinity_M=.001,btb_M=.0001),
        dimensions_m=[1.,1.],boxes=dict(box1=[[0.,0.],[1.,1.]],box2=[[0.,0.],[.5,.5]]),
        summary=dict(baseline_gain=1.1,intervals=32768,max_column_distance_m=.02,max_depth_m=.0155,max_porosity=.45),
        bindings=dict(amendment_sha256="a"*64,round4_spec_sha256="b"*64),
        contracts=dict(result="saved-products",sandmap="saved-sandmap"))
    for index in (1,3):
        config["frames"].append(dict(case_id="sample_"+str(index),frame_index=index,elapsed_h=float(index),selected_target_h=float(index),
            source_binding=dict(source_sha256="c"*64),current=current,baseline=baseline,labels=phase,classes=classes,
            area=area,photo_top=top,original_domain=dict(parent_labels_map=old_domain,native_shape_hw=list(shape),
                H_ac28_to_frame_native=np.eye(3).tolist(),frame_index=0),inner_polygon_native=[[0,0],[2,0],[2,2],[0,2]],
            atmosphere=dict(pressure_bar=1.,temperature_c=20.,pressure_gradient_bar_per_m=0.,temperature_gradient_c_per_m=0.),
            nominal_injected_mass_g=1.,light_epoch="measured"))
    return config,save("config.json",config)


def test_real_tiny_native_path_and_main_storage(tiny_config,tmp_path):
    """Run saved native models through all scalar/array/commit operations."""
    config,bound=tiny_config
    events=[]
    (tmp_path/"out").mkdir()  # The queue caller owns and creates the attempt.
    result=pipeline.run_n1(dict(run=config["run"],configuration=bound,commit=BASE_COMMIT),tmp_path/"out","cpu","float32",events.append)
    assert result["complete"] and result["worker_references_closed"]
    assert result["ordered_case_ids"] == ["sample_1","sample_3"]
    assert result["frames"] == 2 and result["wall_seconds"] > 0 and result["images_per_hour"] > 0
    manifest=json.loads(Path(result["manifest"]["path"]).read_bytes())
    runtime=manifest["settings"]["thread_runtime"]
    assert runtime["requested_threads"] == 1 and runtime["cv2_threads"] == 1
    assert all(pool["num_threads"] == 1 for pool in runtime["threadpools"])
    assert len(events) == 4
    for publication in result["records"]:
        root=Path(publication["nas_directory"])
        record=json.loads((root/"result.json").read_bytes())
        assert record["contract"] == "saved-products"
        assert record["scalar_result"]["reduction_dtype"] == "float64"
        with np.load(root/"products.npz",allow_pickle=False) as archive:
            for name in ("concentration_aqueous","saturation_free_gas","mass_total","mass_aqueous","mass_free_gas"):
                assert archive[name].dtype == np.float32
            assert archive["phase_labels"].dtype == np.uint8
        with np.load(root/record["static"]["asset"]["path"],allow_pickle=False) as archive:
            assert archive["transport_weight_l"].dtype == np.float64
        for name,value in publication["files"].items():
            actual=pipeline.asset(root/name)
            assert actual["sha256"] == value["sha256"]
            assert actual["size_bytes"] == value["size_bytes"]


@pytest.mark.parametrize("precision",["float32","float64"])
def test_selected_math_matches_original_codeobject(tiny_config,precision):
    """Compare complete native fields/scalars against frozen original math."""
    config,_=tiny_config
    math=PixelMath(precision)
    native=pipeline.N1Pipeline(config,precision=precision,pixel_math=math)
    frame,*_=native.load_frame(config["frames"][0])
    response=models.prepare_response(frame,native.models,math,config["shared"]["swi_by_facies"])
    response.update(frozen_parameters=native.parameters,pixel_backend="numpy_cpu",input_storage_dtype="float32-log-RGB/frozen-clearwater")
    original_partition=original_function("production_retained.py","_main_partition",{})
    retained=SimpleNamespace(_main_partition=original_partition,FIELDS=arithmetic.FIELDS,
        _selected_n1=lambda *args:(native.parameters,config["calibration"]["sha256"]))
    context=SimpleNamespace(modules={"engine":SimpleNamespace(r2=SimpleNamespace(dilution_factor=dilution_factor))},asset_reader=None)
    adapter=SimpleNamespace(context=context,math=math,precision=precision,retained=retained,_facts=math.facts)
    masks={"full_rig":frame.domain,"box1":frame.domain,"box2":np.zeros(frame.domain.shape,bool)}
    old=original_function("production_precision_adapter.py","evaluate_main",{},class_name="PrecisionAdapter")
    expected=old(adapter,response,config["anchor"],selected_calibration={},domain_masks=masks,kept_fields=arithmetic.FIELDS)
    actual=arithmetic.evaluate_main(response,config["anchor"],parameters=native.parameters,pixel_math=math,
        calibration_sha256=config["calibration"]["sha256"],domain_masks=masks)
    actual["contract"]=expected["contract"]
    assert_same(expected,actual)


def test_physical_geometry_matches_original_codeobjects(tiny_config):
    """Exercise filled facies, linear/nearest depth and corrected exclusions."""
    config,_=tiny_config
    native=pipeline.N1Pipeline(config,precision="float32",pixel_math=PixelMath("float32"))
    frame,x,y,photo,report=native.load_frame(config["frames"][0])
    labels=frame.facies.copy();labels[1,1]=0
    names=("_mask","_points","fill_to_edges","measured_geometry")
    namespace=dict(np=np)
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree
    from scipy.ndimage import distance_transform_edt
    namespace.update(LinearNDInterpolator=LinearNDInterpolator,cKDTree=cKDTree,distance_transform_edt=distance_transform_edt)
    for name in names:
        namespace[name]=original_function("production_sandmap.py",name,namespace)
    expected=namespace["fill_to_edges"](labels,frame.domain,sand_labels=config["shared"]["porosity_by_facies"])
    actual=geometry.fill_to_edges(labels,frame.domain,sand_labels=config["shared"]["porosity_by_facies"])
    assert_same(expected,actual)
    kwargs=dict(porosity_by_facies=config["shared"]["porosity_by_facies"],swi_by_facies=config["shared"]["swi_by_facies"],chunk_pixels=2)
    assert_same(namespace["measured_geometry"](x,y,actual[0],native.depth_points,**kwargs),geometry.measured_geometry(x,y,actual[0],native.depth_points,**kwargs))


def test_data_alias_and_changed_identity_fail_closed(tmp_path):
    """One consumed payload verification cannot silently adopt later bytes."""
    path=tmp_path/"data.json";path.write_bytes(b'{"value":1}')
    bound=pipeline.asset(path);logical=dict(bound,path="logical/source.json")
    reader=pipeline.DataReader([dict(logical=logical,physical=bound)])
    assert reader.json(logical) == {"value":1}
    path.write_bytes(b'{"value":2}')
    with pytest.raises(ValueError,match="changed"):
        reader.json(logical)
    wrong=dict(bound,sha256="0"*64)
    with pytest.raises(ValueError,match="SHA"):
        pipeline.DataReader([dict(logical=logical,physical=wrong)])


def test_json_rejects_changed_bytes_with_preserved_size_and_mtime(tmp_path):
    """The consumed payload must match its SHA even if metadata is unchanged."""
    import os

    path = tmp_path / "data.json"
    path.write_bytes(b'{"value":1}')
    bound = pipeline.asset(path)
    logical = dict(bound, path="logical/source.json")
    reader = pipeline.DataReader([dict(logical=logical, physical=bound)])
    assert reader.json(logical) == {"value": 1}
    before = path.stat()
    path.write_bytes(b'{"value":2}')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino) == (
        before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() != bound["sha256"]
    with pytest.raises(ValueError, match="changed"):
        reader.json(logical)


def test_existing_partial_attempt_is_preserved(tiny_config,tmp_path):
    """An attempt with existing bytes cannot be replayed or silently resumed."""
    config,bound=tiny_config
    output=tmp_path/"partial";output.mkdir()
    existing=output/"preserved.partial";existing.write_bytes(b"original evidence")
    events=[]
    with pytest.raises(FileExistsError):
        pipeline.run_n1(dict(run=config["run"],configuration=bound,commit=BASE_COMMIT),output,"cpu","float32",events.append)
    assert existing.read_bytes() == b"original evidence" and events == []


def test_production_reader_rejects_local_or_mapped_alias_targets(tmp_path):
    """Logical aliases cannot bypass direct UNC production input reads."""
    path=tmp_path/"data.json";path.write_bytes(b'{}')
    bound=pipeline.asset(path)
    logical=dict(bound,path=r"C:\logical\historic.json")
    for locator in (str(path),"Z:/mapped/data.json"):
        physical=dict(bound,path=locator)
        reader=pipeline.DataReader([dict(logical=logical,physical=physical)],unc_only=True)
        with pytest.raises(ValueError,match="UNC"):
            reader.path(logical)
    assert pipeline._is_unc(r"\\server\share\file.json")
    assert not pipeline._is_unc("Z:/file.json")


@pytest.mark.parametrize("raw",[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'])
def test_strict_metadata_rejects_ambiguous_numbers(raw):
    """Data parsing keeps duplicate/nonfinite failures explicit."""
    with pytest.raises(ValueError):
        pipeline._strict_json(raw)


def test_sparse_order_and_no_hidden_anchor(tiny_config,tmp_path):
    """The config needs an existing anchor and never renumbers sparse IDs."""
    config,bound=tiny_config
    assert pipeline.load_n1_run(bound)["frames"][1]["frame_index"] == 3
    config["frames"].reverse()
    path=tmp_path/"wrong.json";path.write_text(json.dumps(config),encoding="utf-8")
    with pytest.raises(ValueError,match="chronological"):
        pipeline.load_n1_run(pipeline.asset(path))
    config["frames"].reverse();config["anchor"]={}
    path.write_text(json.dumps(config),encoding="utf-8")
    with pytest.raises(ValueError,match="anchor"):
        pipeline.load_n1_run(pipeline.asset(path))


def test_npz_roundtrip_preserves_nan_bits_and_refuses_overwrite(tmp_path):
    """Standard NPZ serialization changes no scientific dtype/bit bytes."""
    value=np.array([1.,np.nan,-0.],np.float32)
    path=tmp_path/"arrays.npz"
    pipeline._write_npz(path,{"value":value})
    with np.load(path,allow_pickle=False) as archive:
        assert archive["value"].tobytes() == value.tobytes()
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        pipeline._write_npz(path,{"value":value})
    assert path.read_bytes() == before


def test_histogram_reduction_and_interval_boundary():
    """Duplicate codes aggregate in sorted CPU64 order; intervals close."""
    unique,sums,audit=statistics.reduce_weighted_codes(np.array([2,1,2],np.int64),np.array([.5,.25,.75]))
    np.testing.assert_array_equal(unique,[1,2]);np.testing.assert_array_equal(sums,[.25,1.25])
    assert audit["backend"] == "cpu-float64"
    assert statistics.interval_residual(1.,1.5,.5) == 0.
    assert statistics.interval_residual(1.,2.,.5) == -.25


def test_new_pipeline_has_no_old_execution_or_external_code_imports():
    """The science closure cannot accidentally reconnect old control code."""
    root=Path(pipeline.__file__).parent
    forbidden=("production_round4","production_native","production_job","production_inputs","production_retained","phase_ffac","tools.","validation","operations","runtime")
    policy = Path(darsia.__file__).with_name("_opencv_runtime.py")
    paths = [root/name for name in ("n1_pipeline.py","n1_models.py","n1_geometry.py","n1_arithmetic.py","n1_statistics.py")] + [policy]
    for path in paths:
        tree=ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                text=ast.unparse(node)
                if isinstance(node,ast.ImportFrom) and node.module == "darsia._opencv_runtime":
                    assert node.level == 0 and {item.name for item in node.names} <= {"scientific_thread", "completed_frame", "require_ipp_off"}
                else:
                    assert not any(token in text for token in forbidden),text
                if path == policy:
                    if isinstance(node,ast.Import):
                        assert {item.name for item in node.names} <= {"os", "threading", "cv2"}
                    else:
                        assert node.level == 0 and node.module == "contextlib" and {item.name for item in node.names} == {"contextmanager"}
            assert not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {"eval","exec","__import__"})


def test_qualified_cuda_provider_is_unchanged():
    """Carry the complete tested host ABI/RawKernels byte-for-byte."""
    path=Path(pipeline.__file__).with_name("mass_cuda.py")
    assert len(path.read_bytes()) == 26135
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "ebc695edf36bc308b7f994d589c0de74660c1f692c8d09b25d764ba351a9e5ea"


def test_thread_limits_apply_inside_run_and_restore(monkeypatch,tmp_path):
    """Measure real OpenCV/BLAS limits inside the body, then restore them."""
    import cv2
    import scipy.linalg
    from threadpoolctl import threadpool_info
    previous_cv2=cv2.getNumThreads()
    before={pool["filepath"]:pool["num_threads"] for pool in threadpool_info()}
    seen={}
    def body(task,output,backend,precision,on_progress,*,thread_runtime):
        """Observe actual runtime configuration at the scientific boundary."""
        seen.update(thread_runtime)
        assert cv2.getNumThreads() == 1
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        return {"complete":True}
    monkeypatch.setattr(pipeline,"_run_n1",body)
    assert pipeline.run_n1({"threads":1},tmp_path/"not-created","cpu","float32")["complete"]
    assert seen["requested_threads"] == 1 and seen["cv2_threads"] == 1
    assert cv2.getNumThreads() == previous_cv2
    assert {pool["filepath"]:pool["num_threads"] for pool in threadpool_info()} == before
    assert not (tmp_path/"not-created").exists()


@pytest.mark.parametrize("threads",[0,-1,True,1.0,"1",None])
def test_invalid_thread_request_rejected_before_outputs(threads,tmp_path):
    """Bad resource metadata cannot be silently coerced or create an output."""
    output=tmp_path/"not-created"
    with pytest.raises(ValueError,match="positive integer"):
        pipeline.run_n1({"threads":threads},output,"cpu","float32")
    assert not output.exists()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_frame_loading_only_matches_full_native_preparation(tiny_config, precision):
    """Read real saved fields with no optical package and preserve every output."""
    from copy import deepcopy
    from dataclasses import fields
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline

    config, _ = tiny_config
    config = dict(config, geometry_fields=dict(pipeline._LEGACY_GEOMETRY_FIELDS))
    original = N1Pipeline(config, precision=precision, pixel_math=PixelMath(precision))
    expected = original.load_frame(config["frames"][0])
    inputs = deepcopy(config)
    # These genuine files need no invented bootstrap replacement. An absent
    # descriptor must not be touched by frame loading, even in statistics-only
    # setups whose old constructor would need saved optical models.
    for key in ("calibration", "boxes", "bindings", "anchor", "summary"):
        inputs.pop(key)
    inputs["shared"].pop("optics")
    loader = N1Pipeline(inputs, precision=precision, pixel_math=PixelMath(precision),
                        frame_loading_only=True)
    actual = loader.load_frame(inputs["frames"][0])
    for field in fields(models.FrameData):
        assert_same(getattr(expected[0], field.name), getattr(actual[0], field.name))
    for old, new in zip(expected[1:], actual[1:]):
        assert_same(old, new)
    assert not hasattr(loader, "models")
    assert not hasattr(loader, "parameters")
    assert not hasattr(loader, "boxes")
    assert "physical_context_verified" not in actual[0].provenance
    assert "optical_source_contract" not in actual[0].provenance


@pytest.mark.parametrize("entrypoint", ["_prepare_statistics", "extract_statistics", "evaluate"])
def test_frame_loading_only_refuses_optical_statistics_and_mass(tiny_config, entrypoint):
    """The input adapter cannot accidentally call absent/fake optical models."""
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline

    config, _ = tiny_config
    loader = N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
        geometry_fields=pipeline._LEGACY_GEOMETRY_FIELDS, frame_loading_only=True)
    # A bad source would fail loading if the guard were reached only afterwards.
    with pytest.raises(ValueError, match="frame-loading-only context cannot"):
        getattr(loader, entrypoint)({"case_id": "must-not-read"})


@pytest.mark.parametrize("mode", [1, "yes", None])
def test_frame_loading_only_requires_literal_boolean(tiny_config, mode):
    """The new model-free route is an explicit positive caller choice."""
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline

    config, _ = tiny_config
    with pytest.raises(TypeError, match="Frame-loading-only mode must be explicit"):
        N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
            geometry_fields=pipeline._LEGACY_GEOMETRY_FIELDS, frame_loading_only=mode)


def test_frame_loading_only_is_separate_from_optical_statistics(tiny_config):
    """Contradictory mode declarations cannot imply optical qualification."""
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline

    config, _ = tiny_config
    with pytest.raises(ValueError, match="separate modes"):
        N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
            geometry_fields=pipeline._LEGACY_GEOMETRY_FIELDS,
            frame_loading_only=True, statistics_only=True)


def test_frame_loading_only_keeps_original_physical_source_sha_check(tiny_config):
    """A changed measured physical field never enters the model-free adapter."""
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline

    config, _ = tiny_config
    physical = Path(config["shared"]["physical_reference"]["path"])
    physical.write_bytes(physical.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Data size differs"):
        N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
            geometry_fields=pipeline._LEGACY_GEOMETRY_FIELDS, frame_loading_only=True)


def test_frame_loading_only_keeps_original_phase_domain_check(tiny_config):
    """Real categorical input reading still checks the original native domain."""
    from copy import deepcopy
    from PIL import Image
    from darsia.fluidflower.mass.n1_pipeline import N1Pipeline

    config, _ = tiny_config
    spec = deepcopy(config["frames"][0])
    phase_path = Path(spec["labels"]["path"])
    with Image.open(phase_path) as image:
        labels = np.asarray(image).copy()
    labels[0, 0] = 255
    Image.fromarray(labels).save(phase_path)
    spec["labels"] = pipeline.asset(phase_path)
    loader = N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
        geometry_fields=pipeline._LEGACY_GEOMETRY_FIELDS, frame_loading_only=True)
    with pytest.raises(ValueError, match="Fixed labels and original rig domain disagree"):
        loader.load_frame(spec)


def test_overflow_phase_split_replaces_only_gas_saturation_and_is_off_by_default(tiny_config):
    """An image's overflow-governed free gas enters before every reduction; None keeps today's bytes."""
    from darsia.fluidflower.mass.overflow_phase_split import distribute_target
    from darsia.fluidflower.mass.overflow_steering import gas_population
    from darsia.fluidflower.mass.phase_expert_knowledge import expert_knowledge_of
    from darsia.fluidflower.phase.probability_storage import CODES
    config, _ = tiny_config
    math = PixelMath("float32")
    native = pipeline.N1Pipeline(config, precision="float32", pixel_math=math)
    frame, *_ = native.load_frame(config["frames"][0])
    response = models.prepare_response(frame, native.models, math, config["shared"]["swi_by_facies"])
    response.update(frozen_parameters=native.parameters, pixel_backend="numpy_cpu",
                    input_storage_dtype="float32-log-RGB/frozen-clearwater")
    masks = {"full_rig": frame.domain, "box1": frame.domain, "box2": np.zeros(frame.domain.shape, bool)}
    arguments = dict(parameters=native.parameters, pixel_math=math,
                     calibration_sha256=config["calibration"]["sha256"], domain_masks=masks)
    plain = arithmetic.evaluate_main(response, config["anchor"], **arguments)
    assert_same(plain, arithmetic.evaluate_main(response, config["anchor"], gas_distribution=None, **arguments))
    labels = np.asarray(frame.phase_labels)
    co2 = np.isin(labels, (1, 2))
    probability = np.where(co2, .5, 0).astype(np.float32)
    arrays = dict(pixel_probability=probability, regional_probability=probability.copy(),
                  branch_support=np.full(labels.shape, CODES["centre"], np.uint8),
                  later_rule_changes=np.zeros(labels.shape, np.uint16))
    partition = arithmetic.main_partition(response)
    keep = (frame.mass_eligible_mask & frame.known_physical_domain
            & (partition["aq_known"] | partition["gas_known"]))
    geometry_weights = response["analysis"].geometry
    geometry_weights._prepare_cached_voxel_volume(list(frame.domain.shape))
    volume = np.broadcast_to(np.asarray(geometry_weights.cached_voxel_volume), frame.domain.shape)
    swi = np.broadcast_to(np.asarray(response["swi"], np.float64), frame.domain.shape)
    population = gas_population(arrays, expert_knowledge_of(frame), keep, volume, swi)
    capacity = float(np.sum((population.pore_volume_m3 * population.maximum_saturation)[population.eligible]))
    assert capacity > 0
    distribution = distribute_target(population, capacity / 4)
    split = arithmetic.evaluate_main(response, config["anchor"],
        gas_distribution=dict(population=population, distribution=distribution), **arguments)
    assert split["gas_mass_g"] != plain["gas_mass_g"]
    assert split["affected_pore_volume_m3"] == plain["affected_pore_volume_m3"]  # Labels are unchanged.
    assert split["total_mass_g"] == pytest.approx(split["aqueous_mass_g"] + split["gas_mass_g"], rel=1e-6)


def test_worker_reports_the_image_overflow_split_only_under_the_plan_switch():
    """Without the plan's switch nothing changes; an image without a reference keeps the model split."""
    from darsia.fluidflower.mass.overflow_phase_split import POLICY
    off = SimpleNamespace(config={}, reader=None)
    assert pipeline.N1Pipeline._overflow_phase_split(off, dict(case_id="c1"), None) is None
    block = dict(policy=POLICY, frames={"c1": dict(target=None, reason="No comparable donor",
                                                   probability_roles={}, phase_source={})})
    on = SimpleNamespace(config=dict(overflow_phase_split=block), reader=None)
    assert pipeline.N1Pipeline._overflow_phase_split(on, dict(case_id="c1"), None) == dict(report=dict(
        policy=POLICY, kind="none", reason="No comparable donor", labels_modified=False))
