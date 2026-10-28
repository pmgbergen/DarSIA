"""Physical-reader tests on genuine small saved Image archives.

The JSON documents are explicit synthetic producer-contract DATA, not a real
RAW decode, renderer certificate, scientific validation or training release.
Ownership and save records use the original bounded output writer.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import darsia
import numpy as np
import pytest

from darsia.fluidflower.mass import native_optical_source as source
from darsia.fluidflower.rig_raw_build import _descriptor, _grid
from darsia.fluidflower.rig_raw_write import OwnedRawOutput


def _json(path, value):
    """Write exact private fixture DATA and return its actual file descriptor."""
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _descriptor(path)


@pytest.fixture
def saved(tmp_path, native_parameters=None, run="sample"):
    """Save original image NPZ buffers and genuine owned-save metadata."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    raw = inputs / "baseline.ARW"
    raw.write_bytes(b"contract fixture; no RAW decode claimed")
    configuration = inputs / "run.toml"
    configuration.write_text("# private producer-contract fixture\n", encoding="utf-8")
    native = _json(inputs / "native.json", native_parameters if native_parameters is not None
                   else {"source": "explicit native fixture"})
    adopted = _json(inputs / "adopted.json", {"revision_id": "saved-revision"})
    camera = dict(configuration=native["path"], sha256=native["sha256"],
                  profile_id="camera", run_id=run)
    basis = _json(inputs / "basis.json", {"artifact_kind": "raw_shape_basis"})
    manifest = inputs / "series.toml"
    manifest.write_text("# private series context\n", encoding="utf-8")
    root = tmp_path / "products"
    root.mkdir()
    destination = root / "fresh"
    authority = dict(destination=str(destination), intended_root=str(root),
        forbidden_raw_roots=[str(inputs)], maximum_output_bytes=2_000_000,
        maximum_file_bytes=500_000)
    request = dict(format=source.REQUEST_FORMAT, run_id=run,
        configuration_files=[_descriptor(configuration)],
        series_context=dict(root=str(inputs), manifest=_descriptor(manifest)),
        baseline=_descriptor(raw), profile=native, operation_overrides={},
        protocol_inputs=[], shape_basis_certificate=basis, output=authority)
    request_path = inputs / "request.json"
    request_bound = _json(request_path, request)
    owner = OwnedRawOutput(destination, authority, operation_id="saved-operation", request=request_bound)
    rig = object()
    owner.begin_setup(rig, destination / "figures")
    owner.begin_save(rig, destination)
    baseline = darsia.OpticalImage(np.full((2, 3, 3), .5, np.float32),
        color_space="RGB", dimensions=[.02, .03], origin=[0., .02])
    scalar = dict(space_dim=2, indexing="ij", dimensions=[.02, .03], origin=[0., .02])
    images = dict(shape_corrected_baseline=baseline,
        labels=darsia.ScalarImage(np.ones((2, 3), np.uint8), **scalar),
        facies=darsia.ScalarImage(np.ones((2, 3), np.uint8), **scalar),
        depth=darsia.ScalarImage(np.full((2, 3), .01, np.float64), **scalar),
        porosity=darsia.ScalarImage(np.full((2, 3), .3, np.float64), **scalar))
    for name, image in images.items():
        owner.image(image, destination / (name + ".npz"))
    owner.finish_save()
    filenames = [name + ".npz" for name in images] + [
        "raw_build_owner.json", "raw_setup_started.json", "raw_save_started.json", "raw_save_completed.json"]
    plan = dict(format="darsia-series-setup-plan-v1", adopted_revision=adopted,
        series_manifest=request["series_context"]["manifest"], build_plans=[dict(run_id=run,
            native_parameters=native, configuration_paths=[str(configuration)],
            raw_setup_settings=dict(run_id=run, raw_camera=camera, rig_path=str(destination)),
            request_template={key: value for key, value in request.items() if key != "configuration_files"})])
    plan_path = inputs / "plan.json"
    certificate = dict(format=source.FORMAT, artifact_kind="raw_rig_baseline", run_id=run,
        complete_rig_model=False, production_qualification_claimed=False,
        producer=dict(operation_id=owner.operation_id),
        inputs=dict(operation_request=request_bound, configuration_files=request["configuration_files"],
            series_context=request["series_context"], raw=request["baseline"], profile=native,
            camera_binding=deepcopy(camera),
            protocol_inputs=[], operation_overrides={}),
        corrected_baseline=dict(grid=_grid(baseline)),
        output_ownership=dict(operation_id=owner.operation_id, request=request_bound, authority=authority,
            directory_identity=list(owner.directory_identity), intended_root_identity=list(owner.root_identity)))
    result = dict(format="darsia-series-setup-result-v1", pipeline="series-setup", run=run,
        complete=True, worker_references_closed=True, operation="build", raw_build_request=request_bound,
        configuration_files=request["configuration_files"])
    result_path = inputs / "operation_result.json"

    def seal():
        """Bind fixture documents to the exact current saved-file bytes."""
        certificate["outputs"] = [_descriptor(destination / name) for name in filenames]
        certificate["corrected_baseline"]["archive"] = _descriptor(destination / "shape_corrected_baseline.npz")
        for role, filename in [("owner", "raw_build_owner.json"), ("setup_started", "raw_setup_started.json"),
                               ("save_started", "raw_save_started.json"), ("save_completed", "raw_save_completed.json")]:
            certificate["output_ownership"][role] = _descriptor(destination / filename)
        result["setup_plan"] = _json(plan_path, plan)
        result["baseline_renderer"] = _json(destination / source.CERTIFICATE_NAME, certificate)
        result["produced_files"] = deepcopy(certificate["outputs"])
        return _json(result_path, result)

    return SimpleNamespace(bound=seal(), seal=seal, result=result, plan=plan, certificate=certificate,
        destination=destination, images=images, filenames=filenames, adopted=adopted)


def test_original_saved_buffers_and_grid_are_returned_without_context_qualification(saved):
    actual = source.load_build_physical_context(saved.bound)
    assert actual.run == "sample"
    for field, name in [("baseline_rgb", "shape_corrected_baseline"), ("depth_m", "depth"),
                        ("porosity", "porosity"), ("labels", "labels"), ("facies", "facies")]:
        np.testing.assert_array_equal(getattr(actual, field), saved.images[name].img)
        assert getattr(actual, field).dtype == saved.images[name].img.dtype
        assert not getattr(actual, field).flags.writeable
    assert actual.grid["dimensions"] == np.asarray(saved.images["depth"].dimensions).tolist()
    assert actual.provenance["adopted_revision"] == saved.adopted
    assert actual.provenance["operation_result"] == saved.bound
    assert actual.provenance["camera_binding"] == saved.certificate["inputs"]["camera_binding"]
    assert actual.provenance["raw_acquisition"] == saved.certificate["inputs"]["raw"]
    assert "physical_context_verified" not in actual.provenance
    assert saved.certificate["complete_rig_model"] is False


@pytest.mark.parametrize("configuration", [
    pytest.param(None, id="absent-configuration"),
    pytest.param({}, id="empty-configuration"),
    pytest.param({"opaque_legacy": {"path": "unused-legacy-data", "value": [None, False]}},
                 id="opaque-non-opted-configuration"),
])
def test_non_opted_revision_preserves_original_buffers_grid_and_provenance(saved, monkeypatch, configuration):
    """Absent optional data never activates old opaque configuration or native replay."""
    revision = {"revision_id": "saved-revision"}
    if configuration is not None:
        revision["configuration"] = deepcopy(configuration)
    saved.adopted = _json(Path(saved.adopted["path"]), revision)
    saved.plan["adopted_revision"] = saved.adopted
    bound = saved.seal()

    def undeclared_replay(*args, **kwargs):
        """Fail if this explicit non-opted unit fixture enters optional source replay."""
        pytest.fail("A non-opted saved Build entered optional native replay")

    for module_name, functions in (
        ("darsia.fluidflower.painted_analysis_exclusions",
         ("configuration_exclusion", "exclusion_provenance", "exclusion_sources")),
        ("darsia.fluidflower.painted_target_integration", ("target_sources",)),
    ):
        module = ModuleType(module_name)
        for name in functions:
            setattr(module, name, undeclared_replay)
        monkeypatch.setitem(sys.modules, module_name, module)

    decoded = {}
    original_imread = darsia.imread

    def remember(path, *args, **kwargs):
        """Retain genuine decoded image objects to check original buffer identity."""
        image = original_imread(path, *args, **kwargs)
        decoded[Path(path).stem] = image
        return image

    monkeypatch.setattr(darsia, "imread", remember)
    actual = source.load_build_physical_context(bound)
    for field, name in (("baseline_rgb", "shape_corrected_baseline"), ("labels", "labels"),
                        ("facies", "facies"), ("depth_m", "depth"), ("porosity", "porosity")):
        array = getattr(actual, field)
        assert array is decoded[name].img
        assert array.dtype == saved.images[name].img.dtype
        assert array.shape == saved.images[name].img.shape
        assert array.tobytes() == saved.images[name].img.tobytes()
        assert not array.flags.writeable
    baseline = saved.images["shape_corrected_baseline"]
    assert actual.grid == {name: np.asarray(getattr(baseline, name)).tolist()
                           for name in ("num_voxels", "dimensions", "origin", "voxel_size")}
    outputs = {Path(row["path"]).name: row for row in saved.certificate["outputs"]}
    assert actual.provenance == dict(
        operation_result=bound, setup_plan=saved.result["setup_plan"], adopted_revision=saved.adopted,
        native_configuration=saved.plan["build_plans"][0]["native_parameters"],
        raw_build_request=saved.result["raw_build_request"], baseline_renderer=saved.result["baseline_renderer"],
        camera_binding=saved.certificate["inputs"]["camera_binding"],
        raw_acquisition=saved.certificate["inputs"]["raw"],
        physical_archives={name: outputs[name + ".npz"]
                           for name in ("shape_corrected_baseline", "labels", "facies", "depth", "porosity")},
    )
    assert "physical_context_verified" not in actual.provenance
    assert "approved_analysis_gap" not in actual.provenance


def test_missing_physical_archive_refuses_even_with_consistent_output_descriptors(saved):
    saved.filenames.remove("porosity.npz")
    with pytest.raises(ValueError, match="lacks.*porosity"):
        source.load_build_physical_context(saved.seal())


def test_changed_saved_bytes_refuse_before_original_archive_loading(saved, monkeypatch):
    (saved.destination / "depth.npz").write_bytes(b"changed")
    monkeypatch.setattr(darsia, "imread", lambda *args: pytest.fail("Changed archive reached decoder"))
    with pytest.raises(ValueError, match="size differs|SHA or identity differs"):
        source.load_build_physical_context(saved.bound)


def test_an_independently_valid_archive_on_another_grid_cannot_be_relabelled(saved):
    changed = saved.images["depth"].copy()
    changed.origin[0] += .001
    changed.save(saved.destination / "depth.npz")
    with pytest.raises(ValueError, match="another original grid: depth"):
        source.load_build_physical_context(saved.seal())


@pytest.mark.parametrize("field,value", [("complete", False), ("operation", "apply"),
                                         ("worker_references_closed", False)])
def test_incomplete_or_other_operation_cannot_supply_physical_context(saved, field, value):
    saved.result[field] = value
    with pytest.raises(ValueError, match="completed Setup Build"):
        source.load_build_physical_context(saved.seal())


def test_a_shape_basis_certificate_cannot_be_used_as_a_full_build(saved):
    saved.certificate["artifact_kind"] = "raw_shape_basis"
    with pytest.raises(ValueError, match="original full RAW Build"):
        source.load_build_physical_context(saved.seal())


def test_native_parameters_cannot_differ_from_the_original_request(saved, tmp_path):
    saved.plan["build_plans"][0]["native_parameters"] = _json(tmp_path / "other-native.json", {"other": True})
    with pytest.raises(ValueError, match="request differs"):
        source.load_build_physical_context(saved.seal())


def test_adopted_revision_bytes_are_read_and_verified(saved):
    Path(saved.adopted["path"]).write_bytes(b"changed revision")
    with pytest.raises(ValueError, match="size differs|SHA or identity differs"):
        source.load_build_physical_context(saved.bound)


@pytest.mark.parametrize("record_name,change", [
    ("raw_build_owner.json", {"operation_id": "another-operation"}),
    ("raw_save_completed.json", {"complete": False}),
])
def test_rebound_ownership_or_incomplete_save_cannot_authorize_buffers(saved, record_name, change):
    path = saved.destination / record_name
    record = json.loads(path.read_bytes())
    record.update(change)
    _json(path, record)
    with pytest.raises(ValueError, match="another operation|save is not complete"):
        source.load_build_physical_context(saved.seal())


def test_rebound_camera_binding_cannot_replace_the_original_saved_build_settings(saved):
    saved.certificate["inputs"]["camera_binding"]["profile_id"] = "another-camera"
    with pytest.raises(ValueError, match="camera binding differs"):
        source.load_build_physical_context(saved.seal())
