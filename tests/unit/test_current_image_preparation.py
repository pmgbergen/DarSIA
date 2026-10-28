from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import image_normalisation as choices
from darsia.fluidflower.pipeline.current_image_preparation import (
    load_image_context, prepare_image_photometry, proposal_from_operations, stable_support,
    preview_from_operation, _registered_build_operations,
    render_preparation_display,
    colour_fit_preview_from_operations,
)


def inputs(tmp_path):
    table = tmp_path / "materials.csv"
    table.write_text("id,name,analysis_role\n12,Sand,porous_medium\n4,Water,free_water\n")
    adopted = {"artifacts": {"source_mode_files": {"facies_properties": {"path": str(table)}}},
               "configuration": {"regions": [{"label_id": 31, "facies_id": 12},
                                                 {"label_id": 17, "facies_id": 4}]}}
    class Reader:
        def json(self, bound):
            assert bound["path"] == "adopted"
            return adopted
        def path(self, bound):
            return Path(bound["path"])
    def context(run):
        labels = np.full((24, 24), 31)
        facies = np.full((24, 24), 12)
        labels[:4], facies[:4] = 17, 4
        return SimpleNamespace(run=run, labels=labels, facies=facies,
            baseline_rgb=np.full((24, 24, 3), .3),
            grid={"voxel_size": [.001, .001], "num_voxels": [24, 24], "dimensions": [.024, .024]},
            provenance={"adopted_revision": {"path": "adopted"},
                        "physical_archives": {"labels": {"path": "labels"}, "facies": {"path": "facies"}}})
    return Reader(), context


def display_fixture(context, *, reader):
    return dict(rgb=np.full((*context.labels.shape, 3), 128, np.uint8), display=dict(display_only=True))


def test_layers_display_is_projected_through_exact_build_crop_without_touching_science():
    raw = dict(path="original.ARW", sha256="a"*64, size_bytes=12)
    profile = dict(path="camera.json", sha256="b"*64, size_bytes=24)
    native_to_u = [[1., 0., 1.], [0., 1., 2.], [0., 0., 1.]]
    context = SimpleNamespace(grid=dict(num_voxels=[3, 4]), raw_orientation=1,
        baseline_rgb=np.full((3, 4, 3), [.02, .2, .01]),
        provenance=dict(raw_acquisition=raw, native_configuration=profile,
                        baseline_renderer=dict(path="certificate.json", sha256="c"*64)),
        display_renderer=dict(source_kind="raw", raw_acquisition=raw, profile_id="camera-test",
            native_shape_hw=[3, 4], camera_u_wh=[10, 8], native_to_camera_u_h=native_to_u))
    scientific = context.baseline_rgb.copy()
    photographic = np.arange(10*12*3, dtype=np.uint8).reshape(10, 12, 3)
    calls = []
    class Reader:
        def physical_descriptor(self, pin):
            return pin
    def display(path, **options):
        calls.append((path, options))
        return dict(image_rgb=photographic, image_encoding="rgb_uint8", display_only=True,
            source_raw=raw, profile=dict(profile, profile_id="camera-test"), raw_orientation=1,
            camera_u_wh=[10, 8], camera_u_to_photo_h=[[1., 0., 2.], [0., 1., 1.], [0., 0., 1.]],
            display_conversion=dict(producer="rawpy.RawPy.postprocess", use_camera_wb=True,
                output_bps=8, half_size=False, user_flip=0))
    result = render_preparation_display(context, reader=Reader(), display_reader=display)
    np.testing.assert_array_equal(result["rgb"], photographic[3:6, 3:7])
    np.testing.assert_array_equal(context.baseline_rgb, scientific)
    assert result["display"]["corrections_applied"] is False
    assert result["display"]["raw_acquisition"] == raw
    assert calls[0][1]["expected_sha256"] == raw["sha256"]
    assert calls[0][1]["expected_orientation"] == 1
    context.display_renderer["raw_acquisition"] = dict(raw, sha256="d"*64)
    with pytest.raises(ValueError, match="same certified acquisition"):
        render_preparation_display(context, reader=Reader(), display_reader=display)
    assert len(calls) == 1


def test_material_roles_define_band_without_assuming_facies_ids(tmp_path):
    reader, context = inputs(tmp_path)
    support = stable_support(context("r1"), 2., reader=reader)
    assert not support["mask"][:5, 5:19].any()  # Water-facing top is not the frame.
    assert support["mask"][10, :3].all()
    assert not support["mask"][10, 4:-4].any()


def test_preview_reuses_exact_baseline_and_material_grid_when_band_changes(tmp_path):
    reader, context = inputs(tmp_path)
    loaded, cache = [], {}
    original_json = reader.json
    reader.json = lambda bound: {} if bound["path"].endswith("build-r2") else original_json(bound)
    archives = {}
    for role in ("labels", "facies"):
        path = tmp_path / role
        path.write_bytes(b"original")
        archives[role] = dict(path=str(path))
    def load(operation, **kwargs):
        loaded.append(operation)
        value = context("r2")
        value.provenance["physical_archives"] = archives
        return value
    first = preview_from_operation({"path": "build-r2"}, 2., reader=reader,
        context_loader=load, context_cache=cache, display_loader=display_fixture)
    second = preview_from_operation({"path": "build-r2"}, 5., reader=reader,
        context_loader=load, context_cache=cache, display_loader=display_fixture)
    assert loaded == [{"path": "build-r2"}]
    assert first["rgb"] is second["rgb"]
    assert second["stable_mask"].sum() > first["stable_mask"].sum()
    assert not second["stable_mask"][:5, 8:16].any()
    assert second["source_pins"]["source_labels"] == archives["labels"]
    assert second["grid"] == context("r2").grid
    Path(archives["labels"]["path"]).write_bytes(b"changed")
    preview_from_operation({"path": "build-r2"}, 2., reader=reader,
        context_loader=load, context_cache=cache, display_loader=display_fixture)
    assert len(loaded) == 2  # Modified data must be read/verified again.
    preview_from_operation({"path": "revised-build-r2"}, 2., reader=reader,
        context_loader=load, context_cache=cache, display_loader=display_fixture)
    assert len(loaded) == 3


def test_preview_missing_or_empty_band_shows_only_real_baseline(tmp_path):
    reader, context = inputs(tmp_path)
    baseline = context("r1")
    shown = preview_from_operation({}, 0., reader=reader,
        context_loader=lambda *a, **kw: baseline, display_loader=display_fixture)
    assert shown["stable_mask"] is None and shown["rgb"].shape == (24, 24, 3)
    assert "positive stable-band" in shown["note"]
    baseline.facies[:] = 4  # No mapped sand; no guessed rectangular mask.
    shown = preview_from_operation({}, 3., reader=reader,
        context_loader=lambda *a, **kw: baseline, display_loader=display_fixture)
    assert shown["stable_mask"] is None and "no mapped sand" in shown["stable_area_error"]


@pytest.mark.parametrize("complete_geometry", [True, False])
@pytest.mark.parametrize("image_state", ["Waiting", "Failed", "Metadata error"])
def test_preview_uses_completed_geometry_while_image_preparation_is_pending(
        tmp_path, monkeypatch, complete_geometry, image_state):
    import json
    from darsia.fluidflower import series_geometry, series_native_context, geometry_facies
    from darsia.fluidflower.mass.n1_pipeline import asset
    adopted = tmp_path / "revision.json"
    adopted.write_text(json.dumps({"geometry": "measured-current"}))
    plan_path = tmp_path / "setup-controls" / "plan.json"
    plan_path.parent.mkdir()
    plan_path.write_text(json.dumps(dict(adopted_revision=asset(adopted), build_plans=[
        dict(run_id=run, native_parameters=dict(sha256="current-camera")) for run in ("r1", "r2")])))
    plan_pin = asset(plan_path)
    snapshot = dict(root=str(tmp_path), complete=False,
        errors=['image-only metadata error'] if image_state == 'Metadata error' else [],
        job_facts=dict(pipeline="series-setup", series_id="sample", setup_plan=plan_pin),
        runs=[dict(run=run, operation="build", state="Complete" if complete_geometry else "Waiting",
                   result=dict(files={"operation_result.json": {"path": run}})) for run in ("r1", "r2")]
             + [dict(run="setup-image-preparation", operation="image-preparation", state=image_state,
                     error='image-only metadata error' if image_state == 'Metadata error' else '')])
    monkeypatch.setattr(series_native_context, "read_series_native_binding", lambda root: SimpleNamespace(sha256="current-camera"))
    monkeypatch.setattr(series_geometry, "load_series_geometry_project", lambda root: "current")
    geometry = SimpleNamespace(current_path=adopted)
    monkeypatch.setattr(geometry_facies, "GeometryRevisionStore", lambda project: geometry)
    if complete_geometry:
        reads, cache = [], {}
        registration = dict(root=str(tmp_path), job=dict(sha256="original-job"))
        def read(row):
            reads.append(row)
            return snapshot
        for _ in range(2):
            assert _registered_build_operations(tmp_path, "sample", [registration],
                operations_cache=cache, snapshot_reader=read) == {run: {"path": run} for run in ("r1", "r2")}
        assert len(reads) == 1  # Band-only changes do not replay all run metadata.
        registration["job"]["sha256"] = "updated-job"
        _registered_build_operations(tmp_path, "sample", [registration],
            operations_cache=cache, snapshot_reader=read)
        assert len(reads) == 2
        snapshot['errors'].append('unrelated metadata error')
        with pytest.raises(ValueError, match="Complete the geometry Build"):
            _registered_build_operations(tmp_path, "sample", [registration], snapshot_reader=read)
        snapshot['errors'].pop()
        new_geometry = tmp_path / "new-revision.json"
        new_geometry.write_text(json.dumps({"geometry": "new-crop"}))
        geometry.current_path = new_geometry
        with pytest.raises(ValueError, match="Complete the geometry Build"):
            _registered_build_operations(tmp_path, "sample", [registration],
                operations_cache=cache, snapshot_reader=read)
    else:
        with pytest.raises(ValueError, match="Complete the geometry Build"):
            _registered_build_operations(tmp_path, "sample", ["registered"], snapshot_reader=lambda row: snapshot)


@pytest.mark.parametrize('rejected', [True, False])
def test_colour_fit_preview_uses_original_full_photo_producer_for_only_two_baselines(tmp_path, rejected):
    import darsia
    from darsia.fluidflower.image_correction import srgb_to_linear
    swatches = darsia.ColorCheckerAfter2014().swatches_rgb
    srgb = np.repeat(np.repeat(swatches,80,axis=0),84,axis=1)
    reference = np.moveaxis(srgb_to_linear(srgb,dtype=np.float32),-1,0)
    target = reference.copy()
    if rejected:
        target[1,-80:,:] = np.repeat(np.array([.999,.986,.63,.35,.19,.07]),84)
    roi = [[0,0],[0,504],[320,504],[320,0]]
    loaded, collected = [], []
    def load(operation, *, reader, include_baseline):
        assert include_baseline is False
        loaded.append(operation)
        return SimpleNamespace(run=operation)
    def collect(context, reader, *, require_chart):
        assert require_chart
        collected.append(context.run)
        return dict(run=context.run, rgb_full=(reference if context.run == 'reference' else target)*1000,
                    white_range_dn=1000, chart_rgb=srgb_to_linear(swatches,dtype=np.float32),
                    chart_roi_rc=roi, source_pins=dict(raw_acquisition=dict(path=context.run+'.raw')))
    choice = choices.choice_record('other', ['reference','target','unrelated'], choices.MATCH_REFERENCE,
                                   'reference', color_method=choices.COLOR_UNIFORM24, stable_band_mm=40.)
    result = colour_fit_preview_from_operations(
        {run:run for run in ('reference','target','unrelated')}, 'target', choice,
        context_loader=load, collector=collect)
    assert loaded == collected == ['reference','target']
    assert result['accepted'] is not rejected and result['complete'] is False
    assert result['choice'] == choice and not list(tmp_path.iterdir())
    if rejected:
        assert result['failed_run'] == 'target'
        assert result['diagnostics']['channel'] == 1
        assert np.shape(result['diagnostics']['source_swatches_srgb']) == (4,6,3)
        assert result['diagnostics']['source_shape'] == [320,504,3]
        assert result['sensor_clipping']['status'] == 'unavailable'
    else:
        assert result['diagnostics']['producer'] == 'fit_uniform24_to_reference_swatches'


@pytest.mark.parametrize('provider_fails', [False, True])
def test_optional_sensor_evidence_reuses_decoded_target_and_never_accepts_rejected_fit(monkeypatch, provider_fails):
    import sys
    from darsia.fluidflower.image_correction import NeutralToneFitError
    from darsia.fluidflower.native import full_photometry
    sensor = SimpleNamespace(metadata=dict(white_level=1023, black_level=0))
    roi, registration, calls = [[0,0],[0,6],[4,6],[4,0]], {'camera':'known'}, []
    def produce(unit, *args):
        if unit[0,0,0] == 2:
            raise NeutralToneFitError('unchanged physical rejection',dict(roi_rc=roi,channel=1))
        return unit, {}
    def collect(context, reader, **options):
        return dict(run=context.run,rgb_full=np.full((3,4,6),1 if context.run=='reference' else 2),
            white_range_dn=1,chart_rgb=np.full((4,6,3),.3),chart_roi_rc=roi,source_pins={},
            sensor=sensor,parameters=dict(registration=registration))
    def evidence(actual_sensor, actual_parameters, actual_roi):
        calls.append((actual_sensor,actual_parameters,actual_roi))
        if provider_fails:
            raise ValueError('clipping support could not be proved')
        return dict(status='available',criterion='actual CFA white level',counts=[1])
    monkeypatch.setattr(full_photometry,'_colour_full',produce)
    monkeypatch.setitem(sys.modules,'darsia.fluidflower.native.cfa_clipping_diagnostics',
                        SimpleNamespace(chart_clipping_diagnostics=evidence))
    result = colour_fit_preview_from_operations({'reference':'reference','target':'target'}, 'target',
        dict(method='off',color_method='uniform24',reference_run='reference'),
        context_loader=lambda operation,**kw:SimpleNamespace(run=operation),collector=collect)
    assert len(calls)==1 and calls[0][0] is sensor and calls[0][1] is registration and calls[0][2]==roi
    assert result['accepted'] is False and result['error']=='unchanged physical rejection'
    assert result['sensor_clipping']['status']==('unavailable' if provider_fails else 'available')


def test_legacy_does_not_read_or_rebuild_any_optical_source(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Historical records must keep their existing producer")
    choice = choices.choice_record("other", ["r1"], choices.MATCH_REFERENCE, "r1")
    assert prepare_image_photometry({"r1": {}}, choice, {}, tmp_path / "unused",
        context_loader=forbidden, collector=forbidden, producer=forbidden) is None
    assert not (tmp_path / "unused").exists()


def test_new_cohort_fits_once_and_loads_each_baseline_only_once(tmp_path):
    reader, context = inputs(tmp_path)
    loaded, collected = [], []
    operations = {"r1": {"run": "r1"}, "r2": {"run": "r2"}}
    def load(operation, *, reader):
        loaded.append(operation["run"])
        return context(operation["run"])
    def collect(value, reader, *, require_chart):
        assert not require_chart  # Light-only must not require a ColorChecker.
        collected.append(value.run)
        return {"run": value.run}
    def produce(baselines, choice, supports, output, *, choice_authority):
        result = {}
        for run, lazy in baselines.items():
            assert lazy()["run"] == run
            assert supports.get(run)["mask"].any()
            assert run not in supports
            result[run] = {"source_binding": {"path": "fresh-" + run}}
        return {"runs": result}
    choice = choices.choice_record("other", list(operations), choices.MATCH_REFERENCE, "r1",
        color_method=choices.COLOR_OFF, stable_band_mm=2.)
    result = prepare_image_photometry(operations, choice, {"path": "choice"}, tmp_path / "fresh",
        reader=reader, context_loader=load, collector=collect, producer=produce)
    assert loaded == collected == ["r1", "r2"]
    assert set(result["runs"]) == set(operations)


def test_new_lighting_requires_user_stable_width_before_source_reads(tmp_path):
    choice = choices.choice_record("other", ["r1"], choices.MATCH_REFERENCE, "r1",
                                   color_method=choices.COLOR_OFF)
    with pytest.raises(ValueError, match="stable sand-band"):
        prepare_image_photometry({"r1": {}}, choice, {}, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_baseline_prefetch_uses_selected_cpu_budget_and_stays_bounded(tmp_path, monkeypatch):
    import threading
    from darsia.fluidflower import execution
    reader, context = inputs(tmp_path)
    monkeypatch.setattr(execution, "detect_host_resources", lambda: dict(
        cpu=dict(physical_cores=8), memory=dict(available_bytes=48 * 1024**3)))
    lock, gate = threading.Lock(), threading.Barrier(3)
    active, maximum, calls = 0, 0, []
    def collect(value, reader, *, require_chart):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(value.run)
            index = len(calls)
        if index <= 3:
            gate.wait(timeout=3)
        with lock:
            active -= 1
        return dict(run=value.run)
    def produce(baselines, choice, supports, output, *, choice_authority):
        first = baselines[choice["reference_run"]]()
        result = {choice["reference_run"]: first}
        for run, loader in baselines.items():
            if run != choice["reference_run"]:
                result[run] = loader()
        return result
    operations = {"r" + str(index): {"run": "r" + str(index)} for index in range(4)}
    choice = choices.choice_record("other", list(operations), "off", None, color_method="off")
    # Reference r0 is a data-load anchor when both corrections are off.
    choice["reference_run"] = "r0"
    result = prepare_image_photometry(operations, choice, {}, tmp_path / "fresh", reader=reader,
        context_loader=lambda row, **kw: context(row["run"]), collector=collect, producer=produce, workers=3)
    assert set(result) == set(operations) and sorted(calls) == sorted(operations)
    assert maximum == 3


@pytest.mark.parametrize("preview_only", [False, True])
def test_optical_context_reads_no_depth_porosity_or_protocol_payloads(monkeypatch, preview_only):
    import darsia
    files = [{"path": name + ".npz"} for name in
             ("shape_corrected_baseline", "labels", "facies", "depth", "porosity")]
    documents = {
        "operation": dict(operation="build", complete=True, worker_references_closed=True,
            run="sample", raw_build_request={"path": "request"}, setup_plan={"path": "plan"},
            baseline_renderer={"path": "certificate"}, produced_files=files),
        "request": dict(run_id="sample", baseline={"path": "raw"}, profile={"path": "native"}),
        "plan": dict(adopted_revision={"path": "adopted"}),
        "certificate": dict(run_id="sample", outputs=files,
            renderer=dict(source_grid=dict(num_voxels=[4, 6], voxel_size=[.01, .01])),
            inputs=dict(raw={"path": "raw"}, profile={"path": "native"}, camera_binding={"profile_id": "camera"})),
    }
    opened = []
    class Reader:
        def json(self, bound):
            return documents[bound["path"]]
        def path(self, bound):
            opened.append(bound["path"])
            return bound["path"]
    def read(path):
        shape = (4, 6, 3) if path.startswith("shape_corrected") else (4, 6)
        return SimpleNamespace(space_dim=2, indexing="ij", series=False,
            num_voxels=np.array([4, 6]), dimensions=np.array([.04, .06]),
            origin=np.array([0., .04]), voxel_size=np.array([.01, .01]), img=np.zeros(shape))
    monkeypatch.setattr(darsia, "imread", read)
    value = load_image_context({"path": "operation"}, reader=Reader(), include_baseline=not preview_only)
    assert opened == (["labels.npz", "facies.npz"] if preview_only else
                      ["shape_corrected_baseline.npz", "labels.npz", "facies.npz"])
    assert value.run == "sample" and value.grid["num_voxels"] == [4, 6]
