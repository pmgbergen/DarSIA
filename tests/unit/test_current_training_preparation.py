from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json
from darsia.fluidflower.pipeline import current_training_preparation as subject
from darsia.fluidflower.storage_paths import only_added_storage_aliases


def test_training_selection_retains_actual_raw_history_and_baseline_time():
    base = dict(decision=dict(actor="operator", reason="prepared images"), input_authority={})
    baseline = dict(path="baseline.ARW", sha256="b", size_bytes=10)
    acquisitions = [dict(frame_index=None, elapsed_h=None, acquisition=baseline),
        dict(frame_index=10, elapsed_h=2., acquisition=dict(path="second.ARW", sha256="d")),
        dict(frame_index=9, elapsed_h=1., acquisition=dict(path="first.ARW", sha256="c"))]
    selected = subject._training_selection(base, "sample", baseline, acquisitions, baseline_elapsed_h=-.125)
    row = selected["runs"][0]
    assert row["baseline"]["elapsed_h"] == -.125
    assert [value["raw_sha256"] for value in row["selected_frames"]] == ["c", "d"]
    assert [value["previous_raw_sha256"] for value in row["selected_frames"]] == ["b", "c"]


def test_runtime_alias_targets_are_relocated_without_rewriting_native_parameters():
    from darsia.fluidflower.pipeline.current_phase_recipe import _runtime_parameters
    native = dict(source=dict(asset_aliases={"original": "previous"}), photometry=dict(scale=17))
    before = deepcopy(native)
    reader = SimpleNamespace(json=lambda binding: native,
        resolve_path=lambda path: Path("current") if path == "previous" else Path(path))
    actual = _runtime_parameters({}, reader)
    assert actual["source"]["asset_aliases"] == {"original": "current", "previous": "current"}
    assert actual["photometry"] == native["photometry"]
    assert native == before


def test_only_new_storage_aliases_are_positive_manifest_compatibility(tmp_path):
    before = dict(schema_version=1, image_normalisation=dict(method='match_reference_lighting',
        reference_run='r2'), storage=dict(path_aliases={'old': 'retained'}))
    after = deepcopy(before)
    after['storage']['path_aliases']['moved'] = 'current'
    assert only_added_storage_aliases(before, after, tmp_path)
    changed = deepcopy(after)
    changed['image_normalisation']['reference_run'] = 'r3'
    assert not only_added_storage_aliases(before, changed, tmp_path)
    changed = deepcopy(after)
    changed['storage']['path_aliases']['old'] = 'another'
    assert not only_added_storage_aliases(before, changed, tmp_path)
    assert not only_added_storage_aliases(before, before, tmp_path)
    assert only_added_storage_aliases({'schema_version': 1},
        dict(schema_version=1, storage=dict(path_aliases={'old': 'new'})), tmp_path)


def test_aliases_moved_to_a_pinned_table_with_new_rows_are_only_added(tmp_path):
    import hashlib
    before = dict(schema_version=1, storage=dict(path_aliases={'old': 'retained'}))
    content = b'schema_version = 1\n[path_aliases]\nold = "retained"\nmoved = "current"\n'
    (tmp_path / "relocations.toml").write_bytes(content)
    table = dict(path="relocations.toml", sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    after = dict(schema_version=1, storage=dict(path_alias_tables=[table]))
    assert only_added_storage_aliases(before, after, tmp_path)
    changed = dict(schema_version=1, storage=dict(path_alias_tables=[dict(table, sha256="0" * 64)]))
    with pytest.raises(ValueError, match="pinned bytes"):
        only_added_storage_aliases(before, changed, tmp_path)


def test_completed_build_checks_retained_series_when_aliases_are_added(tmp_path, monkeypatch):
    from darsia.fluidflower import project, geometry_facies, series_native_context
    build, series = tmp_path/'build', tmp_path/'series'
    controls = build/'setup-controls'
    controls.mkdir(parents=True)
    series.mkdir()
    manifest = series/'series.toml'
    before = dict(schema_version=1, image_normalisation=dict(method='match_reference_lighting',
        reference_run='sample'), runs=dict(ids=['sample']),
        storage=dict(path_aliases={str(tmp_path/'kept'): str(tmp_path/'kept-copy')}))
    manifest.write_text(toml.dumps(before))
    original_series = asset(manifest)
    (controls/'series-manifest.toml').write_bytes(manifest.read_bytes())
    adopted = _write_json(tmp_path/'adopted.json', dict(revision_id='same-geometry'))
    native = _write_json(tmp_path/'native.json', dict(camera='same-camera'))
    config = tmp_path/'run.toml'
    config.write_text('geometry="same"\n')
    config_pin = asset(config)
    plan = _write_json(controls/'plan.json', dict(project_root='project',
        series_manifest=original_series, adopted_revision=adopted, native_parameters=native))
    request = _write_json(tmp_path/'request.json', dict(
        baseline=dict(sha256='same-raw'), profile=native))
    operation = _write_json(tmp_path/'operation.json', dict(
        raw_build_request=request, configuration_files=[config_pin]))
    snapshot = dict(root=str(build), complete=True, errors=[], job_facts=dict(
        pipeline='series-setup', series_id='series', series=original_series,
        setup_plan=dict(sha256=plan['sha256'], bytes=plan['size_bytes'])),
        runs=[dict(run='sample', operation='build', state='Complete',
            result=dict(files={'operation_result.json': operation}))])
    monkeypatch.setattr(project.FluidFlowerProject, 'load', lambda *args: object())
    monkeypatch.setattr(geometry_facies, 'GeometryRevisionStore',
                        lambda *args: SimpleNamespace(current_path=Path(adopted['path'])))
    monkeypatch.setattr(series_native_context, 'read_series_native_binding',
                        lambda *args: SimpleNamespace(sha256=native['sha256']))
    selection = dict(input_authority=dict(current_series=None),
        runs=[dict(run='sample', baseline=dict(raw_sha256='same-raw'))])

    def current(document):
        manifest.write_text(toml.dumps(document))
        selection['input_authority']['current_series'] = asset(manifest)
        return selection

    after = deepcopy(before)
    after['storage']['path_aliases'][str(tmp_path/'old')] = str(tmp_path/'moved')
    assert subject.find_completed_build(series, 'series', [snapshot], current(after)) is snapshot
    # A later lighting choice after added rows keeps the Build, as without added rows;
    # the Setup image basis checks the lighting choice where it is used.
    after['image_normalisation']['reference_run'] = 'another'
    assert subject.find_completed_build(series, 'series', [snapshot], current(after)) is snapshot
    for rows in ({str(tmp_path/'kept'): str(tmp_path/'elsewhere'), str(tmp_path/'old'): str(tmp_path/'moved')},
                 {str(tmp_path/'old'): str(tmp_path/'moved')}):
        # An earlier row changed or removed is not a relocation.
        with pytest.raises(ValueError, match='Complete Apply and Build'):
            subject.find_completed_build(series, 'series', [snapshot],
                                         current(dict(after, storage=dict(path_aliases=rows))))
    current(after)
    (controls/'series-manifest.toml').write_text('schema_version=2\n')
    with pytest.raises(ValueError, match='Complete Apply and Build'):
        subject.find_completed_build(series, 'series', [snapshot], selection)


def test_model_choice_changes_do_not_discard_a_completed_unchanged_build(tmp_path, monkeypatch):
    from darsia.fluidflower import project, geometry_facies, series_native_context
    build = tmp_path / "build"
    (build / "setup-controls").mkdir(parents=True)
    original = _write_json(tmp_path / "adopted.json", dict(revision_id="measured-quad"))
    current_path = tmp_path / "current.json"
    current_path.write_bytes(Path(original["path"]).read_bytes())
    native = _write_json(tmp_path / "native.json", dict(camera="current"))
    configuration = tmp_path / "run.toml"
    configuration.write_text('geometry="measured"\n')
    config_pin = asset(configuration)
    plan = _write_json(build / "setup-controls" / "plan.json", dict(project_root="project",
        adopted_revision=original, native_parameters=native))
    request = _write_json(tmp_path / "request.json", dict(baseline=dict(sha256="actual-raw"), profile=native))
    operation = _write_json(tmp_path / "operation.json", dict(raw_build_request=request,
        configuration_files=[config_pin]))
    snapshot = dict(root=str(build), complete=True, errors=[], job_facts=dict(
        pipeline="series-setup", series_id="series", series=dict(sha256="before-model-choice"),
        setup_plan=dict(sha256=plan["sha256"], bytes=plan["size_bytes"])),
        runs=[dict(run="sample", operation="build", state="Complete",
            result=dict(files={"operation_result.json": operation}))])
    selection = dict(input_authority=dict(current_series=dict(sha256="after-model-choice")),
        runs=[dict(run="sample", baseline=dict(raw_sha256="actual-raw"))])
    monkeypatch.setattr(project.FluidFlowerProject, "load", lambda *a: object())
    monkeypatch.setattr(geometry_facies, "GeometryRevisionStore", lambda *a: SimpleNamespace(current_path=current_path))
    monkeypatch.setattr(series_native_context, "read_series_native_binding", lambda *a: SimpleNamespace(sha256=native["sha256"]))
    assert subject.find_completed_build(tmp_path, "series", [snapshot], selection) is snapshot
    current_path.write_text('{"revision_id":"new-geometry"}')
    with pytest.raises(ValueError, match="Complete Apply and Build"):
        subject.find_completed_build(tmp_path, "series", [snapshot], selection)
    current_path.write_bytes(Path(original["path"]).read_bytes())
    configuration.write_text('geometry="changed"\n')
    with pytest.raises(ValueError, match="Complete Apply and Build"):
        subject.find_completed_build(tmp_path, "series", [snapshot], selection)


@pytest.mark.parametrize("independent", [False, True])
def test_preparation_reuses_corrected_sources_and_keeps_old_labels_unchanged(tmp_path, monkeypatch, independent):
    from darsia.fluidflower.pipeline import current_calibration_selection, current_build_job
    from darsia.fluidflower import series_native_context
    root = tmp_path / "series"
    root.mkdir()
    baseline = tmp_path / "baseline.ARW"
    baseline.write_bytes(b"synthetic baseline source")
    raw = tmp_path / "image.ARW"
    raw.write_bytes(b"synthetic measured source")
    baseline_pin, raw_pin = asset(baseline), asset(raw)
    series_path = root / "series.toml"
    series_document = dict(image_selection=dict(runs=dict(sample=dict(
        selected=[dict(source_path=str(baseline), elapsed_h=-.125)]))))
    if independent:
        from darsia.fluidflower.image_normalisation import choice_record
        series_document.update(series_id="series", runs=dict(ids=["sample"]), method=dict(active="phase_model"),
            image_normalisation=choice_record("series", ["sample"], "off", None, color_method="off"))
    series_path.write_text(toml.dumps(series_document))
    series = asset(series_path)
    phase = _write_json(tmp_path / "old_phase.json", {})
    selection = dict(decision=dict(actor="operator", reason="explicit GUI start"),
        input_authority=dict(current_series=series, phase_plan=phase),
        runs=[dict(run="sample", baseline=dict(raw_sha256=baseline_pin["sha256"]))])
    native = _write_json(tmp_path / "native.json", {})
    monkeypatch.setattr(current_calibration_selection, "current_calibration_selection", lambda *a, **k: deepcopy(selection))
    monkeypatch.setattr(series_native_context, "read_series_native_binding", lambda *a: SimpleNamespace(
        path=Path(native["path"]), sha256=native["sha256"]))
    original_decisions = _write_json(tmp_path / "original_decisions.json", dict(decisions=dict(sample=dict(
        annotations=[dict(id="human-label", frame_index=2, elapsed_h=1., category="confirmed_free_gas")]))))
    original_combined = _write_json(tmp_path / "original_combined.json", dict(jobs=[dict(
        run="sample", candidate_id="old-candidate", dataset="series")]))
    original = {path: path.read_bytes() for path in (Path(original_decisions["path"]), Path(original_combined["path"]))}
    template = dict(root=str(tmp_path / "models" / "old-model"), request=phase,
        settings={}, bundles=[dict(decisions=original_decisions, combined_manifest=original_combined)])
    monkeypatch.setattr(subject, "selected_training_template", lambda *a, **k: deepcopy(template))
    request = _write_json(tmp_path / "build_request.json", dict(baseline=baseline_pin))
    operation = _write_json(tmp_path / "operation.json", dict(raw_build_request=request))
    snapshot = dict(root=str(tmp_path / "build"), complete=True, errors=[], job_facts=dict(
        pipeline="series-setup", series_id="series", series=series), runs=[dict(run="sample",
        operation="build", state="Complete", result=dict(files={"operation_result.json": operation}))])
    if independent:
        from darsia.fluidflower.pipeline import current_image_preparation
        monkeypatch.setattr(current_image_preparation, "prepare_image_photometry", lambda *a, **k:
                            pytest.fail("Completed Setup photometry must not be fitted again during training"))
        setup_runs = dict(sample=dict(source_binding=phase, native_configuration=native, baseline=phase))
        prep = _write_json(tmp_path / "setup_preparation.json", dict(
            format="darsia-setup-image-preparation-v1", complete=True,
            choice=series_document["image_normalisation"], choice_authority=series,
            operations=dict(sample=operation), runs=setup_runs))
        snapshot["runs"].append(dict(run="setup-image-preparation", operation="image-preparation", state="Complete",
                                     result=dict(image_preparation=prep)))
    monkeypatch.setattr(current_build_job, "_metric_for_build", lambda op, target, reader:
        _write_json(target, dict(area_factor=1.)))
    metadata_reads = []
    read_text, read_bytes = Path.read_text, Path.read_bytes
    def observed_text(path, *args, **kwargs):
        if path == Path(native["path"]): metadata_reads.append("text")
        return read_text(path, *args, **kwargs)
    def observed_bytes(path, *args, **kwargs):
        if path == Path(native["path"]): metadata_reads.append("bytes")
        return read_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", observed_text)
    monkeypatch.setattr(Path, "read_bytes", observed_bytes)
    def collector(decisions, combined, **kwargs):
        for _ in range(100):
            kwargs["reader"].json(native)
        return dict(runs=dict(sample=dict(source_manifest=phase, acquisitions=[
            dict(frame_index=None, elapsed_h=None, acquisition=baseline_pin, roles=["baseline"]),
            dict(frame_index=2, elapsed_h=1., acquisition=raw_pin, roles=["annotation"])])))
    calls = []
    def materialize(op, metric, choice, destination, review, **kwargs):
        calls.append((op, choice))
        if independent:
            assert kwargs["prepared_photometry"] == setup_runs["sample"]
        saved = DataReader().json(choice)
        assert saved["runs"][0]["baseline"]["elapsed_h"] == -.125
        (review / "native_frames" / "sample").mkdir(parents=True)
        manifest = _write_json(review / "native_frames" / "sample" / "manifest.json", dict(unit_fixture=True))
        return dict(manifest=manifest, operation_result=op, selection=choice, phase_inference_performed=False)
    def projector(decisions, combined, targets, **kwargs):
        assert kwargs["source_native_configuration"] == native
        for _ in range(100):
            kwargs["reader"].json(native)
        return dict(decisions=DataReader().json(decisions), combined_manifest=DataReader().json(combined),
            provenance=dict(source=decisions, target=targets["sample"]))
    produced = []
    def producer(**kwargs):
        produced.append(kwargs)
        output = kwargs["output_parent"] / "new-model"
        output.mkdir(parents=True)
        document = dict(output_root=str(output), settings=kwargs["settings"], bundles=[dict(
            decisions=dict(snapshot_path=str(pair[0])), combined_manifest=dict(snapshot_path=str(pair[1])))
            for pair in kwargs["bundles"]])
        pin = _write_json(output / "training_request.json", document)
        return dict(document, request_path=pin["path"])
    arguments = dict(series_root=root, series_id="series", output=tmp_path / "preparation", actor="operator",
        materializer=materialize, request_producer=producer, source_collector=collector, projector=projector)
    first = subject.prepare_training_from_completed_build(snapshot, **arguments)
    assert len(calls) == len(produced) == 1
    assert metadata_reads == ["text"]  # One verified parse across all region checks and projection.
    assert first["source_bundles"] == [(Path(original_decisions["path"]), Path(original_combined["path"]))]
    assert DataReader().json(asset(first["bundles"][0][1]))["jobs"][0]["candidate_id"] == "old-candidate"
    again = subject.prepare_training_from_completed_build(snapshot, **arguments)
    assert again["reused_source_preparation"]
    assert len(calls) == len(produced) == 1
    assert again["request"]["request_path"] == first["request"]["request_path"]
    assert all(path.read_bytes() == value for path, value in original.items())
    series_path.write_text(series_path.read_text() + '\nchoice="changed"\n')
    selection["input_authority"]["current_series"] = asset(series_path)
    snapshot["job_facts"]["series"] = selection["input_authority"]["current_series"]
    with pytest.raises(ValueError, match="other inputs"):
        subject.prepare_training_from_completed_build(snapshot, **arguments)


def test_source_reading_keeps_order_across_threads_and_stops_on_failure():
    import threading
    import time
    names = set()

    def slow(index, value):
        names.add(threading.current_thread().name)
        time.sleep(0.02 * (5 - index))
        return index, value * 2

    items = list(enumerate(range(5)))
    assert list(subject._in_order(slow, items, 4)) == [(index, 2 * index) for index in range(5)]
    assert len(names) > 1
    assert list(subject._in_order(slow, items, 1)) == [(index, 2 * index) for index in range(5)]

    def fail(index, _value):
        if index == 1:
            raise ValueError("run 1 failed")
        return index

    results = subject._in_order(fail, items, 4)
    assert next(results) == 0
    with pytest.raises(ValueError, match="run 1 failed"):
        next(results)
    assert subject._reading_threads(1) == 1 and 1 <= subject._reading_threads(0) <= 16


def test_preparation_workers_use_logical_processors_and_measured_memory():
    def host(physical, logical, memory_gib):
        return dict(cpu=dict(physical_cores=physical, logical_cores=logical),
                    memory=dict(available_bytes=memory_gib * 1024**3))

    # Automatic allows the logical processors less a reserve of two.
    assert subject.preparation_workers(0, host(16, 32, 210), 39) == 30
    assert subject.preparation_workers(0, host(4, 8, 64), 39) == 6
    # Memory allows one worker per 6 GiB available.
    assert subject.preparation_workers(0, host(16, 32, 30), 39) == 5
    assert subject.preparation_workers(0, host(16, 32, 2), 39) == 1
    # An explicit count is kept, within memory and the number of runs.
    assert subject.preparation_workers(28, host(16, 32, 210), 39) == 28
    assert subject.preparation_workers(28, host(16, 32, 210), 3) == 3
    # A host that reports only physical cores still gets a count.
    assert subject.preparation_workers(0, dict(cpu=dict(physical_cores=8),
                                               memory=dict(available_bytes=64 * 1024**3)), 39) == 6


def test_memory_margins_scale_with_the_host_up_to_fixed_amounts():
    gib = 1024**3
    assert subject.memory_margins(254 * gib) == (8 * gib, 4 * gib)
    reserve, stop = subject.memory_margins(16 * gib)
    assert reserve == int(16 * gib * 0.15) and stop == int(16 * gib * 0.06)


def test_frame_cycles_count_only_frames_at_one_concurrency():
    cycles = subject._FrameCycles()
    start = dict(state="raw_preparation", run="ac17")
    assert cycles.cycle(start, 0.0, 16) is None  # first frame start of the run
    assert cycles.cycle(dict(state="raw_prepared", run="ac17"), 50.0, 16) is None
    assert cycles.cycle(start, 120.0, 16) == (120.0, 16)
    assert cycles.cycle(start, 250.0, 18) is None  # the concurrency changed during the frame
    assert cycles.cycle(start, 370.0, 18) == (120.0, 18)
    assert cycles.cycle(dict(state="raw_preparation", run="ac18"), 380.0, 18) is None
