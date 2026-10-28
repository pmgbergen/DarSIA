"""Separate run/facies colour packages without a phase head or mass objective."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from darsia.fluidflower.mass import pooled_optical_paths as optical
from darsia.fluidflower.mass.n1_pipeline import asset, _write_json
from darsia.fluidflower.pipeline import setup_colour_paths as subject
from darsia.fluidflower.pipeline.current_optical_preparation import save_samples
from .test_pooled_optical_paths import recipe, _frame


def prepared_recipe(original):
    value = original.document
    value.update(method=optical.SETUP_METHOD, phase_configuration=None)
    return optical.recipe_from_document(value)


def optical_frame(frame, recipe):
    value = recipe.document
    selected = value["runs"][frame.run_id]["frames"][0]
    return optical.PreparedFaciesFrame(frame.run_id, frame.frame_index, frame.elapsed_h,
        frame.current_rgb, frame.baseline_rgb, frame.clear_water_rgb, frame.clear_water_supported_mask,
        frame.facies, frame.known_physical_domain, frame.supported_mask,
        dict(optical_context_verified=True,
             optical_source_contract=optical._source_contract(value, frame.run_id, selected)))


def test_setup_sampler_keeps_exact_original_signal_and_only_removes_phase_filter(recipe):
    current = prepared_recipe(recipe)
    for run in recipe.document["fit_runs"]:
        legacy = _frame(recipe, run)
        expected = optical.sample_frame_for_common_paths(legacy, recipe, case_id=run + "-1")
        prepared = optical_frame(legacy, current)
        sampled = optical.sample_frame_for_common_paths(prepared, current, case_id=run + "-1")
        for facies in sampled.samples:
            np.testing.assert_array_equal(sampled.samples[facies], expected.samples[facies])
        # A colour path exists before any model can assign water/CO2 labels.
        legacy.phase_labels[:] = 0
        old = optical.sample_frame_for_common_paths(legacy, recipe, case_id=run + "-1")
        assert all(not len(value) for value in old.samples.values())
        assert any(len(value) for value in sampled.samples.values())
        assert not hasattr(prepared, "phase_labels")


def test_setup_has_same_knots_and_flash_when_original_phase_support_is_full(recipe):
    current = prepared_recipe(recipe)
    old = [optical.sample_frame_for_common_paths(_frame(recipe, run), recipe, case_id=run + "-1")
           for run in recipe.document["fit_runs"]]
    new = [optical.sample_frame_for_common_paths(optical_frame(_frame(recipe, run), current),
           current, case_id=run + "-1") for run in current.document["fit_runs"]]
    _paths, before, _sources = optical.fit_pooled_color_paths(recipe, old)
    _paths, after, _sources = optical.fit_pooled_color_paths(current, new)
    assert before == after
    assert current.document["flash_bounds"] == recipe.document["flash_bounds"]


def test_clearwater_and_phase_contract_cannot_be_silently_changed(recipe):
    value = prepared_recipe(recipe).document
    value["phase_configuration"] = recipe.document["phase_configuration"]
    with pytest.raises(ValueError, match="phase-model"):
        optical.recipe_from_document(value)
    value = prepared_recipe(recipe).document
    value["runs"]["alpha"]["clearwater"] = None
    with pytest.raises(ValueError, match="clearwater"):
        optical.recipe_from_document(value).require_inputs()


def test_selection_does_not_hash_raw_or_require_phase_history(tmp_path):
    class Reader:
        resolve_path = staticmethod(Path)
        def path(self, pin):
            pytest.fail("Metadata-only selection should not open undeclared files")
    selected = {run: {"selected": [dict(source_path=run + str(index) + ".ARW",
        frame_index=index, elapsed_h=float(index)) for index in range(50)]} for run in ("a", "b")}
    document = dict(image_selection=dict(runs=selected, color_paths=dict(max_images_per_run=15)))
    value = subject._selection(document, dict.fromkeys(selected), tmp_path, Reader())
    assert value["max_images_per_run"] == 15
    assert all(1 <= len(rows) <= 15 for rows in value["runs"].values())
    assert value["target_hours"][-1] == 49


def test_existing_signal_registration_matches_baseline_and_clearwater_grids():
    raw = np.log(np.float32(65535.) * np.full((3, 4, 4), .4, np.float32))
    current = dict(log_linear_rgb=raw, valid_mask=np.ones((4, 4), bool),
        geometry=dict(baseline_to_native_h=np.eye(3).tolist(), reference_to_native_h=np.eye(3).tolist()),
        build_to_frame_native_h=np.eye(3).tolist(), correction={})
    bridge = SimpleNamespace(context=SimpleNamespace(facies=np.ones((4, 4), np.int32)), proof={})
    pin = dict(path="C:/source", sha256="a" * 64, size_bytes=1)
    row = dict(case_id="sample", frame_index=1, elapsed_h=1., target_h=1., source=pin)
    run = dict(baseline=pin, clearwater=pin, profile=pin, frames=[row])
    recipe = subject._recipe({"one": run}, [dict(facies_id=1, name="sand", analysis_role="porous_medium")],
        pin, pin, pin, dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.))
    frame = subject._frame(bridge, current, current, np.full((4, 4, 3), .2, np.float32),
        current, np.full((4, 4, 3), .5, np.float32), np.ones((4, 4), bool), "one", row, recipe)
    result = optical.sample_frame_for_common_paths(frame, recipe, case_id="sample")
    expected = subject._rgb(current) / np.float32(.5) - np.float32(.2) / np.float32(.5)
    np.testing.assert_array_equal(result.samples[1], expected[::8, ::8].reshape(-1, 3))


@pytest.fixture
def setup_run_inputs(tmp_path, monkeypatch):
    runs = ("first", "second", "heldout")
    rows = {run: {"selected": [dict(source_path=str(tmp_path / (run + str(i) + ".ARW")),
        frame_index=i, elapsed_h=float(i)) for i in range(35)]} for run in runs}
    for value in rows.values():
        for image in value["selected"]:
            Path(image["source_path"]).write_bytes(image["source_path"].encode())
    authority = tmp_path / "series.toml"
    import toml
    authority.write_text(toml.dumps(dict(image_selection=dict(runs=rows,
        color_paths=dict(max_images_per_run=15)))), encoding="utf-8")
    pin = _write_json(tmp_path / "source.json", {})
    operations = {run: pin for run in runs}
    basis = dict(runs={run: dict(source_binding=pin, baseline=pin) for run in runs})
    bounds = dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)
    from darsia.fluidflower.pipeline import current_statistics_recipe, setup_image_preparation
    monkeypatch.setattr(current_statistics_recipe, "setup_optical_sources",
        lambda authority, run, **kwargs: dict(clearwater_source=dict(raw=pin), flash_bounds=bounds))
    monkeypatch.setattr(setup_image_preparation, "retained_series_aliases", lambda *args: [])
    calls = []
    def produce(arguments):
        run, operation, image_basis, selected, output, _aliases, sources = arguments
        calls.append(run)
        frames = [dict(case_id=run + str(i), frame_index=target["frame"]["current_frame_index"],
            elapsed_h=target["frame"]["elapsed_h"], target_h=target["target_h"],
            source=asset(target["frame"]["raw_path"])) for i, target in enumerate(selected)]
        selection = {run: dict(baseline=pin, clearwater=pin, profile=pin, frames=frames)}
        recipe = subject._recipe(selection, [dict(facies_id=1, name="sand", analysis_role="porous_medium")],
            pin, operation, pin, sources["flash_bounds"])
        samples = []
        for i, row in enumerate(frames):
            shape = (40, 40)
            frame = optical.PreparedFaciesFrame(run, row["frame_index"], row["elapsed_h"],
                np.full((*shape, 3), .2 + i / 100 * (runs.index(run) + 1), np.float32),
                np.full((*shape, 3), .2, np.float32), np.ones((*shape, 3), np.float32),
                np.ones(shape, bool), np.ones(shape, np.int32), np.ones(shape, bool), np.ones(shape, bool),
                dict(optical_context_verified=True,
                    optical_source_contract=optical._source_contract(recipe.document, run, row)))
            samples.append(save_samples(Path(output) / (str(i).zfill(4) + "-samples"),
                optical.sample_frame_for_common_paths(frame, recipe, case_id=row["case_id"])))
        return dict(run=run, recipe=recipe.document, samples=samples)
    return SimpleNamespace(runs=runs, operations=operations, basis=basis,
        authority=asset(authority), produce=produce, calls=calls)


def test_prepare_writes_reusable_actual_paths_for_every_included_run(tmp_path, setup_run_inputs):
    setup = setup_run_inputs
    operations, basis, runs, calls, produce = (setup.operations, setup.basis, setup.runs, setup.calls, setup.produce)
    receipt = subject.prepare_setup_colour_paths(operations, basis, setup.authority, tmp_path / "output",
                                                run_producer=produce)
    value = json.loads(Path(receipt["path"]).read_text())
    assert calls == list(runs)
    assert value["runs"] == list(runs) and value["complete"]
    assert value["image_count_per_run"] == 15
    assert all(1 <= count <= 15 for count in value["sampled_frame_counts"].values())
    assert value["scope"] == "per-run" and "package" not in value
    assert set(value["packages_by_run"]) == set(runs)
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    packages = subject.colour_path_packages(value, DataReader(), require_per_run=True)
    knots = {}
    for run, package in packages.items():
        assert set(package) == {"color_paths", "signals", "flash", "metadata"}
        metadata = json.loads(Path(package["metadata"]["path"]).read_text())
        assert metadata["recipe"]["fit_runs"] == [run]
        assert set(metadata["source_bindings"]) == {run}
        assert metadata["path_audit"]["1"]["run_count"] == 1
        knots[run] = metadata["path_audit"]["1"]["knots"]
    assert knots["first"] != knots["second"] != knots["heldout"]
    axis = value["signal_axis"]
    assert axis["source_cohort"] == list(runs) and axis["frozen_for_production"] is True
    import darsia
    interpreted, endpoints = [], []
    for run in runs:
        model = darsia.ColorPathInterpolation.load(Path(packages[run]["color_paths"]["1"]["path"]))
        endpoints.append(model.values[-1])
        interpreted.append(model(np.full((1, 3), .03, np.float32)))
    assert endpoints[0] < endpoints[1] < endpoints[2] == 1.
    # Equal measured RGB displacement has the same signal coordinate even
    # though each fitted path has its own different observed endpoint.
    np.testing.assert_allclose(interpreted, np.repeat(interpreted[:1], 3, axis=0), rtol=1e-6)
    assert all(value[key] is False for key in
        ("phase_inference_performed", "model_training_performed", "mass_fit_performed"))
    calls.clear()
    assert subject.prepare_setup_colour_paths(operations, basis, setup.authority, tmp_path / "output",
                                               run_producer=produce) == receipt
    assert not calls


def _legacy_completed_receipt(tmp_path, setup):
    """The old genuine writer's package/audits and real per-frame samples."""
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    from darsia.fluidflower.pipeline.current_optical_preparation import load_samples
    reader = DataReader()
    original = subject.prepare_setup_colour_paths(setup.operations, setup.basis, setup.authority,
        tmp_path / "original", run_producer=setup.produce)
    current = reader.json(original)
    rows = [reader.json(current["prepared_runs"][run])["result"] for run in setup.runs]
    recipe = subject._pooled_recipe(rows, tmp_path / "original" / "legacy_recipe")
    frames = (subject._rebind_sample(load_samples(pin, reader), row, recipe, reader)
              for row in rows for pin in row["samples"])
    package = optical.save_common_path_package(tmp_path / "original" / "legacy_package", recipe, frames)
    legacy = deepcopy(current)
    for key in ("scope", "signal_axis", "prepared_runs", "packages_by_run", "reused_from"):
        legacy.pop(key, None)
    legacy.update(format=subject.LEGACY_FORMAT, package=package)
    bound = _write_json(tmp_path / "original" / "legacy_receipt.json", legacy)
    return bound, rows, current


def test_completed_legacy_samples_fit_own_paths_without_opening_raw_again(tmp_path, setup_run_inputs):
    setup = setup_run_inputs
    legacy, rows, before = _legacy_completed_receipt(tmp_path, setup)
    for row in rows:
        for frame in row["recipe"]["runs"][row["run"]]["frames"]:
            Path(frame["source"]["path"]).unlink()
    setup.calls.clear()
    result = subject.prepare_setup_colour_paths(setup.operations, setup.basis, setup.authority,
        tmp_path / "retry", reuse_receipt=legacy,
        run_producer=lambda args: pytest.fail("A completed corrected run must not be rendered again"))
    after = json.loads(Path(result["path"]).read_text())
    assert after["scope"] == "per-run" and after["reused_from"] == legacy
    assert after["signal_axis"] == before["signal_axis"]
    for run in setup.runs:
        old_meta = json.loads(Path(before["packages_by_run"][run]["metadata"]["path"]).read_text())
        new_meta = json.loads(Path(after["packages_by_run"][run]["metadata"]["path"]).read_text())
        assert new_meta["path_audit"] == old_meta["path_audit"]
        assert new_meta["recipe"]["fit_runs"] == [run]
    assert not setup.calls


@pytest.mark.parametrize("changed", ["recipe", "npz", "audit"])
def test_legacy_reuse_rejects_changed_saved_measurements(tmp_path, setup_run_inputs, changed):
    setup = setup_run_inputs
    legacy, rows, _before = _legacy_completed_receipt(tmp_path, setup)
    sidecar = Path(rows[0]["samples"][0]["path"])
    document = json.loads(sidecar.read_text())
    if changed == "npz":
        with Path(document["arrays"]["path"]).open("ab") as stream:
            stream.write(b"not the saved measurement")
    else:
        if changed == "recipe":
            document["recipe_sha256"] = "0" * 64
        else:
            document["audit"]["sample_counts"]["1"] += 1
        sidecar.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="(SHA|size|identity|recipe|audit|another corrected run)"):
        subject.prepare_setup_colour_paths(setup.operations, setup.basis, setup.authority,
            tmp_path / "retry", reuse_receipt=legacy,
            run_producer=lambda args: pytest.fail("Changed reuse inputs must not trigger silent rendering"))


def test_native_packages_keep_the_frozen_axis_and_own_run(tmp_path, setup_run_inputs):
    setup = setup_run_inputs
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    receipt = subject.prepare_setup_colour_paths(setup.operations, setup.basis, setup.authority,
        tmp_path / "output", run_producer=setup.produce)
    value = DataReader().json(receipt)
    assert list(subject.colour_path_packages(value, DataReader(), require_per_run=True, run="first")) == ["first"]
    wrong = deepcopy(value)
    wrong["packages_by_run"]["first"] = wrong["packages_by_run"]["second"]
    with pytest.raises(ValueError, match="different optical sources"):
        subject.colour_path_packages(wrong, DataReader(), require_per_run=True, run="first")
    wrong = deepcopy(value)
    wrong["signal_axis"]["scales_by_facies"]["1"] *= 2
    with pytest.raises(ValueError, match="different shared signal scale"):
        subject.colour_path_packages(wrong, DataReader(), require_per_run=True, run="first")


def test_new_per_run_checkpoint_reuses_saved_samples_without_producer(tmp_path, setup_run_inputs):
    setup = setup_run_inputs
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    first = subject.prepare_setup_colour_paths(setup.operations, setup.basis, setup.authority,
        tmp_path / "first", run_producer=setup.produce)
    before = DataReader().json(first)
    second = subject.prepare_setup_colour_paths(setup.operations, setup.basis, setup.authority,
        tmp_path / "second", reuse_receipt=first,
        run_producer=lambda args: pytest.fail("Completed run checkpoints must reuse saved samples"))
    after = DataReader().json(second)
    assert after["signal_axis"] == before["signal_axis"] and after["reused_from"] == first
    for run in setup.runs:
        original = DataReader().json(before["prepared_runs"][run])["result"]["samples"]
        reused = DataReader().json(after["prepared_runs"][run])["result"]["samples"]
        assert reused == original


def test_changed_image_selection_cannot_reuse_a_completed_run_set(tmp_path, setup_run_inputs):
    setup = setup_run_inputs
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    first = subject.prepare_setup_colour_paths(setup.operations, setup.basis, setup.authority,
        tmp_path / "first", run_producer=setup.produce)
    authority = Path(setup.authority["path"])
    authority.write_text(authority.read_text().replace("max_images_per_run = 15", "max_images_per_run = 14"))
    with pytest.raises(ValueError, match="another image basis or selection"):
        subject.prepare_setup_colour_paths(setup.operations, setup.basis, asset(authority),
            tmp_path / "second", reuse_receipt=first,
            run_producer=lambda args: pytest.fail("Changed selection must require an explicit fresh preparation"))


@pytest.fixture
def failed_sample_inputs(tmp_path, setup_run_inputs):
    """An actual failed publication with measured per-frame sample sidecars."""
    from darsia.fluidflower.pipeline.current_inputs import current_input_reader
    from darsia.fluidflower.setup_partial_build import snapshot
    setup, reader = setup_run_inputs, current_input_reader()
    root = tmp_path / ("f" * 32)
    root.mkdir()
    properties = tmp_path / "facies.csv"
    properties.write_text("facies_id,name,analysis_role\n1,sand,porous_medium\n")
    adopted = _write_json(tmp_path / "adopted.json", dict(
        artifacts=dict(source_mode_files=dict(facies_properties=asset(properties))),
        configuration=dict(regions=[dict(label_id=1)])))
    native = _write_json(tmp_path / "native.json", {})
    plan = _write_json(tmp_path / "plan.json", dict(adopted_revision=adopted, native_parameters=native))
    preflight = subject.preflight_setup_colour_paths(setup.authority, list(setup.runs))
    rows, operations = {}, {}
    def publish(run, folder):
        selection, sources = preflight["frame_selection"]["runs"][run], preflight["optical_sources"][run]
        baseline = setup.basis["runs"][run]["baseline"]
        if run not in operations:
            request = _write_json(tmp_path / (run + "-request.json"), dict(baseline=baseline))
            operations[run] = _write_json(tmp_path / (run + "-build.json"), dict(setup_plan=plan, raw_build_request=request))
        operation = operations[run]
        (folder / "bridge").mkdir(parents=True)
        profile = _write_json(folder / "bridge/raw_source_binding.json", dict(baseline_raw=baseline))
        bridge = _write_json(folder / "bridge/bridge.json", dict(format="darsia-build-native-bridge-v1",
            operation_result=operation, photometry_source=setup.basis["runs"][run]["source_binding"],
            native_configuration=native, raw_source_binding=profile))
        frames = subject._selected_source_frames(run, selection, reader)
        recipe = subject._recipe({run:dict(baseline=baseline, clearwater=sources["clearwater_source"]["raw"],
            profile=profile, frames=frames)}, [dict(facies_id=1,name="sand",analysis_role="porous_medium")],
            adopted, operation, native, sources["flash_bounds"])
        pins = []
        for index, selected in enumerate(frames):
            shape = (40,40)
            frame = optical.PreparedFaciesFrame(run, selected["frame_index"], selected["elapsed_h"],
                np.full((*shape,3), .2 + index/100, np.float32), np.full((*shape,3), .2,np.float32),
                np.ones((*shape,3),np.float32), np.ones(shape,bool),np.ones(shape,np.int32),
                np.ones(shape,bool),np.ones(shape,bool),dict(optical_context_verified=True,bridge=bridge,
                    optical_source_contract=optical._source_contract(recipe.document,run,selected)))
            pins.append(save_samples(folder / f"{index:04d}-samples",
                optical.sample_frame_for_common_paths(frame,recipe,case_id=selected["case_id"])))
        return dict(run=run, recipe=recipe.document, samples=pins)
    rows = {run:publish(run,root/"colour-paths"/run) for run in setup.runs}
    failure = _write_json(tmp_path / "failed.json", dict(run="setup-image-preparation",
        complete=False, attempt_id=root.name, failed_utc="2026-10-27T05:39:02Z"))
    retired = _write_json(tmp_path / "retirement.json", dict(run="setup-image-preparation",
        attempt_id=root.name,complete=True,all_absent=True,worker_os_references_closed=True,
        retirement_basis="owned_process_incarnations_absent"))
    document = dict(format=subject.PARTIAL_REUSE_FORMAT, complete=False, source_root=str(root),
        operations=operations, image_basis=setup.basis, **preflight,
        failed_attempt=failure, retirement=dict(retirement=retired), file_versions=snapshot(root)["files"])
    return SimpleNamespace(setup=setup,root=root,operations=operations,rows=rows,document=document,publish=publish,
        save=lambda:_write_json(tmp_path/"partial-reuse.json",document))


def test_retired_failed_samples_reuse_without_completion_or_raw(tmp_path, failed_sample_inputs):
    data = failed_sample_inputs
    for path in tmp_path.glob("*.ARW"):
        path.unlink()
    source = data.save()
    result = subject.prepare_setup_colour_paths(data.operations,data.setup.basis,data.setup.authority,
        tmp_path/"retry",reuse_partial=source,
        run_producer=lambda args:pytest.fail("All genuinely measured samples must be reused without RAW"))
    receipt = json.loads(Path(result["path"]).read_text())
    assert receipt["reused_from_failed_attempt"] == source
    assert receipt["reused_from"] is None and receipt["complete"]
    assert set(receipt["packages_by_run"]) == set(data.setup.runs)
    for run in data.setup.runs:
        checkpoint = json.loads(Path(receipt["prepared_runs"][run]["path"]).read_text())
        assert checkpoint["result"]["samples"] == data.rows[run]["samples"]
    assert json.loads(Path(source["path"]).read_text())["complete"] is False


@pytest.fixture
def chained_failed_samples(tmp_path,failed_sample_inputs):
    import shutil
    from darsia.fluidflower.pipeline.current_inputs import current_input_reader
    from darsia.fluidflower.setup_partial_build import snapshot
    data=failed_sample_inputs
    first,second,third=data.setup.runs
    # The first attempt really has only the first run. The second completed
    # another run and retained the first run's genuine original sample pins.
    for run in (second,third):
        shutil.move(str(data.root/"colour-paths"/run),str(tmp_path/("unused-"+run)))
    data.document["file_versions"]=snapshot(data.root)["files"]
    prior=data.save()
    root=tmp_path/("e"*32)
    root.mkdir()
    newer=data.publish(second,root/"colour-paths"/second)
    reader=current_input_reader()
    for run,row in ((first,data.rows[first]),(second,newer)):
        args=(run,data.operations[run],data.setup.basis["runs"][run],
            data.document["frame_selection"]["runs"][run],str(root/"colour-paths"/run),[],
            data.document["optical_sources"][run])
        subject._prepared_checkpoint(args,row,reader=reader)
    failure=_write_json(tmp_path/"latest-failed.json",dict(run="setup-image-preparation",
        complete=False,attempt_id=root.name,failed_utc="2026-10-27T06:21:19Z"))
    retirement=_write_json(tmp_path/"latest-retirement.json",dict(run="setup-image-preparation",
        complete=True,attempt_id=root.name,all_absent=True,worker_os_references_closed=True,
        retirement_basis="owned_process_incarnations_absent"))
    document=dict(deepcopy(data.document),source_root=str(root),file_versions=snapshot(root)["files"],
        failed_attempt=failure,retirement=dict(retirement=retirement),prior_reuse=prior)
    def save():
        return _write_json(tmp_path/"latest-reuse.json",document)
    return SimpleNamespace(data=data,root=root,document=document,save=save,newer=newer)


def test_retry_chain_keeps_old_and_new_runs_and_only_prepares_missing_run(tmp_path,chained_failed_samples):
    chain=chained_failed_samples
    data=chain.data
    first,second,third=data.setup.runs
    for run in (first,second):
        for path in tmp_path.glob(run+"*.ARW"):
            path.unlink()
    calls=[]
    def producer(args):
        calls.append(args[0])
        assert args[0]==third and len(args)==7
        return data.publish(third,Path(args[4]))
    source=chain.save()
    result=subject.prepare_setup_colour_paths(data.operations,data.setup.basis,data.setup.authority,
        tmp_path/"final-retry",reuse_partial=source,run_producer=producer)
    receipt=json.loads(Path(result["path"]).read_text())
    assert calls==[third] and receipt["complete"]
    assert receipt["reused_from_failed_attempt"]==source
    for run,expected in ((first,data.rows[first]),(second,chain.newer)):
        checkpoint=json.loads(Path(receipt["prepared_runs"][run]["path"]).read_text())
        assert checkpoint["result"]["samples"]==expected["samples"]


@pytest.mark.parametrize("changed",["raw_identity","checkpoint","overlapping_recipe"])
def test_retry_chain_rejects_different_inputs_or_sample_ancestry(tmp_path,chained_failed_samples,changed):
    from darsia.fluidflower.setup_partial_build import snapshot
    chain=chained_failed_samples
    data=chain.data
    first=data.setup.runs[0]
    if changed=="raw_identity":
        chain.document["raw_execution_identity"]={"changed":True}
    elif changed=="checkpoint":
        path=chain.root/"colour-paths"/first/"prepared_run.json"
        checkpoint=json.loads(path.read_text())
        checkpoint["result"]["samples"][0]=chain.newer["samples"][0]
        path.write_text(json.dumps(checkpoint))
    else:
        data.publish(first,chain.root/"colour-paths"/first)
    chain.document["file_versions"]=snapshot(chain.root)["files"]
    with pytest.raises(ValueError,match="(ancestry|recipe)"):
        subject.prepare_setup_colour_paths(data.operations,data.setup.basis,data.setup.authority,
            tmp_path/"retry",reuse_partial=chain.save(),
            run_producer=lambda args:pytest.fail("Changed ancestry must not silently recompute"))


def test_retry_chain_cycle_is_rejected_before_sample_or_raw_reads(tmp_path,failed_sample_inputs):
    from darsia.fluidflower.pipeline.current_inputs import CurrentInputReader,current_input_reader
    data=failed_sample_inputs
    pin=data.save()
    class Reader(CurrentInputReader):
        def json(self,bound):
            if bound==pin:
                return dict(data.document,prior_reuse=pin)
            return super().json(bound)
    identity=dict(operations=data.operations,image_basis=data.setup.basis["runs"],
        frame_selection=data.document["frame_selection"],optical_sources=data.document["optical_sources"])
    with pytest.raises(ValueError,match="bounded retry chain"):
        subject._reused_partial_runs(pin,identity,reader=Reader(current_input_reader()))


def test_retired_partial_run_only_requests_missing_samples(tmp_path, failed_sample_inputs):
    data = failed_sample_inputs
    run = data.setup.runs[0]
    missing = Path(data.rows[run]["samples"][2]["path"])
    # The failed attempt really never published this frame. Preserve the
    # removed synthetic sample separately solely as the test decoder result.
    recovered = json.loads(missing.read_text())
    relocated = _write_json(tmp_path/"missing-frame.json", recovered)
    for key in list(data.document["file_versions"]):
        if key.startswith(f"colour-paths/{run}/0002-samples/"):
            del data.document["file_versions"][key]
    calls=[]
    def resume(args):
        calls.append(args[0])
        retained=args[7]
        assert set(retained["samples_by_index"]) == {str(i) for i in range(len(data.rows[run]["samples"])) if i!=2}
        assert retained["recipe"] == data.rows[run]["recipe"]
        return dict(retained,samples=[relocated if i==2 else retained["samples_by_index"][str(i)]
            for i in range(len(data.rows[run]["samples"]))])
    result=subject.prepare_setup_colour_paths(data.operations,data.setup.basis,data.setup.authority,
        tmp_path/"retry",reuse_partial=data.save(),run_producer=resume)
    assert calls==[run]
    assert json.loads(Path(result["path"]).read_text())["complete"]


def test_normal_run_producer_decodes_only_missing_retained_frames(tmp_path,failed_sample_inputs,monkeypatch):
    data=failed_sample_inputs
    run=data.setup.runs[0]
    old=data.rows[run]
    recipe=old["recipe"]
    selected=data.document["frame_selection"]["runs"][run]
    calls=[]
    from darsia.fluidflower.mass import build_native_bridge
    class Bridge:
        def __init__(self,operation,binding,output,**kwargs):
            folder=Path(output)
            folder.mkdir()
            self.context=SimpleNamespace(labels=np.ones((40,40),np.int32),facies=np.ones((40,40),np.int32),
                provenance=dict(raw_acquisition=recipe["runs"][run]["baseline"],
                    adopted_revision=recipe["adopted_revision"],native_configuration=recipe["native_configuration"]))
            original=json.loads(Path(recipe["runs"][run]["profile"]["path"]).read_text())
            self.binding=_write_json(folder/"raw_source_binding.json",original)
            self.proof=_write_json(folder/"bridge.json",dict(operation_result=operation,raw_source_binding=self.binding))
        def corrected_frame(self,source,**kwargs):
            calls.append((source,kwargs.get("source_purpose")))
            return dict(valid_mask=np.ones((40,40),bool),
                log_linear_rgb=np.full((3,40,40),np.log(np.float32(65535.)*.4),np.float32),
                geometry=dict(baseline_to_native_h=np.eye(3).tolist(),reference_to_native_h=np.eye(3).tolist()),
                build_to_frame_native_h=np.eye(3).tolist(),correction={})
        def _revalidate(self):
            pass
    monkeypatch.setattr(build_native_bridge,"BuildNativeBridge",Bridge)
    retained=dict(old,samples_by_index={str(i):pin for i,pin in enumerate(old["samples"]) if i!=2})
    arguments=(run,data.operations[run],data.setup.basis["runs"][run],selected,str(tmp_path/"retry-run"),[],
        data.document["optical_sources"][run],retained)
    result=subject._prepare_run(arguments)
    assert len(calls)==3  # baseline, clear water, and the one missing acquisition
    assert calls[-1][0]==recipe["runs"][run]["frames"][2]["source"]
    for i,pin in enumerate(result["samples"]):
        assert (pin==old["samples"][i]) is (i!=2)
    assert result["recipe"]==recipe
    saved=json.loads(Path(result["samples"][2]["path"]).read_text())
    assert saved["recipe_sha256"]==optical.recipe_from_document(recipe).sha256
    assert saved["audit"]["frame_provenance"]["retained_sampling_profile"]["identical_binding_bytes"]


def test_parallel_failure_stops_new_submissions_and_names_run(tmp_path,setup_run_inputs,monkeypatch):
    from concurrent.futures import Future
    import concurrent.futures
    from darsia.fluidflower import execution
    setup=setup_run_inputs
    submitted=[]
    class Pool:
        def __init__(self,**kwargs):
            assert kwargs["max_workers"]==2
        def __enter__(self):
            return self
        def __exit__(self,*args):
            assert submitted[1][1].cancelled()
        def submit(self,producer,args):
            future=Future()
            submitted.append((args[0],future))
            if len(submitted)==1:
                future.set_exception(ValueError("wash registration failed"))
            return future
    monkeypatch.setattr(concurrent.futures,"ProcessPoolExecutor",Pool)
    monkeypatch.setattr(execution,"detect_host_resources",lambda:dict(
        cpu=dict(physical_cores=2),memory=dict(available_bytes=100*1024**3)))
    progress=[]
    with pytest.raises(RuntimeError,match="failed for first"):
        subject.prepare_setup_colour_paths(setup.operations,setup.basis,setup.authority,
            tmp_path/"output",workers=2,on_progress=progress.append)
    assert [row[0] for row in submitted]==list(setup.runs[:2])
    assert progress[-1]["state"]=="colour_paths_failed"
    assert "first: wash registration failed" in progress[-1]["message"]


def test_parallel_failure_names_every_run_that_failed_while_in_progress(tmp_path,setup_run_inputs,monkeypatch):
    from concurrent.futures import Future
    import concurrent.futures
    from darsia.fluidflower import execution
    setup=setup_run_inputs
    submitted=[]
    class Pool:
        def __init__(self,**kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
        def submit(self,producer,args):
            future=Future()
            submitted.append((args[0],future))
            if len(submitted)==1:
                future.set_exception(ValueError("wash registration failed"))
            else:
                # Already running: it cannot be cancelled and fails later.
                future.set_running_or_notify_cancel()
                future.set_exception(ValueError("frame registration failed"))
            return future
    monkeypatch.setattr(concurrent.futures,"ProcessPoolExecutor",Pool)
    monkeypatch.setattr(execution,"detect_host_resources",lambda:dict(
        cpu=dict(physical_cores=2),memory=dict(available_bytes=100*1024**3)))
    progress=[]
    with pytest.raises(RuntimeError) as raised:
        subject.prepare_setup_colour_paths(setup.operations,setup.basis,setup.authority,
            tmp_path/"output",workers=2,on_progress=progress.append)
    first,second=(row[0] for row in submitted)
    assert first in str(raised.value) and second in str(raised.value)
    assert [row["state"] for row in progress[-2:]]==["colour_paths_failed"]*2
    assert "wash registration failed" in progress[-1]["message"]
    assert "frame registration failed" in progress[-1]["message"]


@pytest.mark.parametrize("changed", ["version", "npz", "recipe", "audit", "basis", "failure"])
def test_retired_failed_samples_reject_changed_inputs(tmp_path, failed_sample_inputs, changed):
    data = failed_sample_inputs
    sidecar=Path(data.rows[data.setup.runs[0]]["samples"][0]["path"])
    value=json.loads(sidecar.read_text())
    if changed=="version":
        sidecar.write_text(sidecar.read_text()+" ")
    elif changed=="npz":
        arrays=Path(value["arrays"]["path"])
        arrays.write_bytes(arrays.read_bytes()+b"changed sample bytes")
    elif changed in {"recipe","audit"}:
        if changed=="recipe":
            value["recipe_sha256"]="0"*64
        else:
            value["audit"]["source_binding"]["phase_configuration"]={"different":True}
        sidecar.write_text(json.dumps(value))
        from darsia.fluidflower.setup_partial_build import snapshot
        data.document["file_versions"]=snapshot(data.root)["files"]
    elif changed=="basis":
        data.document["image_basis"]=deepcopy(data.document["image_basis"])
        data.document["image_basis"]["runs"][data.setup.runs[0]]["baseline"]={"different":True}
    else:
        data.document["failed_attempt"]=_write_json(tmp_path/"other-failure.json",dict(
            run="setup-image-preparation",attempt_id="0"*32,failed_utc="2026-10-27T05:39:02Z"))
    with pytest.raises(ValueError,match="(Retired|recipe|source|audit)"):
        subject.prepare_setup_colour_paths(data.operations,data.setup.basis,data.setup.authority,
            tmp_path/"retry",reuse_partial=data.save(),
            run_producer=lambda args:pytest.fail("Changed partials must not silently render replacements"))
