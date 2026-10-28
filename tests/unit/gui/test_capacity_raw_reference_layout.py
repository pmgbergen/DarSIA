"""TESTONLY: RAW-only frames get one layout on every capacity route.

The released reference build, the bound first qualification and the ordinary
measured profiles all call the one layout reader. It reads the native grid of
RAW-only frames from their one RAW source binding and names that binding, so a
RAW layout never equals a rig layout and a profile of one kind never applies to
the other. Synthetic JSON only; no arrays, profiles, services or claims.
"""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity as cap
from .test_capacity import bound_output_profile_TESTONLY, measured, native_layout, native_metadata

TASK = dict(run="ac52", pipeline="native-phase-n1", precision="float32")


def raw_metadata_TESTONLY(tmp_path, *, binding=None, frame=None, split_binding=False):
    """The rig fixture with every frame switched to RAW frames on one binding."""
    config, rows, rig_layout = native_metadata(tmp_path)
    rows = [row for row in rows if not row.uses[0].endswith(".native_geometry")]

    def put(stem, value, uses):
        path = tmp_path / (stem + ".json")
        raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        rows.append(SimpleNamespace(path=path, reference=dict(sha256=digest, bytes=len(raw)), uses=tuple(uses)))
        return dict(path=str(path), sha256=digest, size_bytes=len(raw))

    cases = [row["case_id"] for row in config["frames"]]
    value = dict(contract="darsia-raw-rig-source-v1", run="ac52", native_shape_hw=[2, 4], profile_id="TESTONLY")
    value.update(binding or {})
    if split_binding:
        bound = [put("raw_binding_" + case, dict(value, profile_id=case), ["frame." + case + ".native.raw_binding"])
                 for case in cases]
    else:
        bound = [put("raw_binding", value, ["frame." + case + ".native.raw_binding" for case in cases])] * len(cases)
    for row, descriptor in zip(config["frames"], bound):
        case = row["case_id"]
        native = dict(contract="darsia-raw-native-frame-v1", run="ac52", acquisition={"TESTONLY": case})
        native.update(frame or {})
        row["native_task"] = dict(kind="raw_rig", processor=dict(raw_binding=descriptor),
                                  geometry=put("frame." + case + ".raw_native", native, ["frame." + case + ".native_geometry"]))
    expected = dict(rig_layout, native_task_kind="raw_rig", raw_source_binding_sha256=bound[0]["sha256"])
    return config, tuple(rows), expected, rig_layout


def raw_layout_TESTONLY():
    """The tiny literal layout of test_capacity, as RAW-only frames name it."""
    return dict(native_layout(), native_task_kind="raw_rig", raw_source_binding_sha256="e" * 64)


def test_every_route_reads_the_raw_binding_grid_and_names_the_binding(tmp_path):
    config, rows, expected, rig_layout = raw_metadata_TESTONLY(tmp_path)
    actual = cap._native_layout_for_task(None, "FF_AC", TASK, config, rows)
    assert actual == expected
    assert actual != rig_layout


def test_rig_frames_keep_their_original_ten_field_layout(tmp_path):
    config, rows, layout = native_metadata(tmp_path)
    actual = cap._native_layout_for_task(None, "FF_AC", TASK, config, rows)
    assert actual == layout and len(actual) == 10 and "native_task_kind" not in actual


@pytest.mark.parametrize("kwargs", [
    dict(binding=dict(contract="darsia-raw-rig-source-v0")),
    dict(frame=dict(contract="darsia-raw-native-frame-v0")),
    dict(frame=dict(run="ac53")),
    dict(binding=dict(run="ac53"), frame=dict(run="ac53")),  # Another run's binding.
    dict(binding=dict(native_shape_hw=[3, 4])),
    dict(split_binding=True),
])
def test_raw_frames_need_their_exact_binding_frame_contract_and_grid(tmp_path, kwargs):
    config, rows, _, _ = raw_metadata_TESTONLY(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        cap._native_layout_for_task(None, "FF_AC", TASK, config, rows)


def test_mixed_rig_and_raw_frames_are_refused(tmp_path):
    config, rows, _, _ = raw_metadata_TESTONLY(tmp_path)
    config["frames"][1] = native_metadata(tmp_path / "rig")[0]["frames"][1]
    with pytest.raises(ValueError, match="Unmeasured native processor"):
        cap._native_layout_for_task(None, "FF_AC", TASK, config, rows)


def test_a_raw_measured_layout_admits_exactly_the_same_raw_layout():
    cap._check_measured_native_layout({"measured_native_layout": raw_layout_TESTONLY()}, raw_layout_TESTONLY())


@pytest.mark.parametrize("change", [
    lambda value: value.update(native_task_kind="rig"),
    lambda value: value.update(native_task_kind="source"),
    lambda value: value.update(native_task_kind=None),
    lambda value: value.update(raw_source_binding_sha256="E" * 64),
    lambda value: value.update(raw_source_binding_sha256="e" * 63),
    lambda value: value.update(raw_source_binding_sha256=int("1" * 64)),
    lambda value: value.pop("raw_source_binding_sha256"),
    lambda value: value.update(TESTONLY="an unknown field"),
    lambda value: value.pop("native_task_kind"),  # A binding SHA without its kind.
])
def test_equal_but_malformed_raw_layouts_do_not_validate_each_other(change):
    value = raw_layout_TESTONLY()
    change(value)
    with pytest.raises(ValueError, match="closed"):
        cap._check_measured_native_layout({"measured_native_layout": value}, deepcopy(value))


@pytest.mark.parametrize("measured_raw", [True, False])
def test_a_layout_of_one_kind_never_matches_the_other(measured_raw):
    raw, rig = raw_layout_TESTONLY(), native_layout()
    measured_layout, requested = (raw, rig) if measured_raw else (rig, raw)
    with pytest.raises(ValueError, match="native processor kind differs"):
        cap._check_measured_native_layout({"measured_native_layout": measured_layout}, requested)


def test_another_raw_binding_is_another_layout():
    requested = dict(raw_layout_TESTONLY(), raw_source_binding_sha256="f" * 64)
    with pytest.raises(ValueError, match="differs from measured profile"):
        cap._check_measured_native_layout({"measured_native_layout": raw_layout_TESTONLY()}, requested)


def ordinary_profile_TESTONLY(tmp_path, monkeypatch, *, layout):
    """An ordinary store map that names one direct measured profile on disk."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _, _, profile, measurement, completion = measured()

    def put(name, value):
        path = tmp_path / name
        raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    writers = [dict(relative_path="src/darsia/fluidflower/" + path, sha256="d" * 64) for path in (
        "mass/n1_pipeline.py", "mass/n1_statistics.py", "pipeline/series.py", "native/sidecar.py")]
    profile.update(measured_native_layout=layout, writers=writers, orchestration=[dict(TESTONLY="runner")],
                   measurement=put("measurement.json", measurement), completion=put("completion.json", completion),
                   layout_receipt=None)
    # The writer and runner identity checks have their own tests in test_capacity.
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_orchestration_identity", lambda *args: None)
    return SimpleNamespace(document=dict(capacity=dict(state_directory=str(tmp_path / "state"),
                                                       profiles={"FF_AC": put("profile.json", profile)})))


def test_the_ordinary_route_estimates_a_raw_task_from_a_raw_profile_as_for_rig(tmp_path, monkeypatch):
    task = measured()[0]
    config, rows, raw_layout, _ = raw_metadata_TESTONLY(tmp_path / "raw")
    raw = cap.load_estimate(ordinary_profile_TESTONLY(tmp_path / "raw_profile", monkeypatch, layout=raw_layout),
                            "FF_AC", task, config, repository=tmp_path, metadata_rows=rows)
    rig_config, rig_rows, rig_layout = native_metadata(tmp_path / "rig")
    rig = cap.load_estimate(ordinary_profile_TESTONLY(tmp_path / "rig_profile", monkeypatch, layout=rig_layout),
                            "FF_AC", task, rig_config, repository=tmp_path, metadata_rows=rig_rows)
    assert (raw.native_expected_bytes, raw.transient_estimated_bytes) == (250, 140)
    assert (raw.native_expected_bytes, raw.transient_estimated_bytes) == (rig.native_expected_bytes,
                                                                          rig.transient_estimated_bytes)


@pytest.mark.parametrize("task_kind,profile_kind", [("raw", "rig"), ("rig", "raw")])
def test_the_ordinary_route_never_estimates_one_kind_from_the_other(tmp_path, monkeypatch, task_kind, profile_kind):
    raw_config, raw_rows, raw_layout, _ = raw_metadata_TESTONLY(tmp_path / "raw")
    rig_config, rig_rows, rig_layout = native_metadata(tmp_path / "rig")
    config, rows = (raw_config, raw_rows) if task_kind == "raw" else (rig_config, rig_rows)
    stores = ordinary_profile_TESTONLY(tmp_path / "profile", monkeypatch,
                                       layout=raw_layout if profile_kind == "raw" else rig_layout)
    with pytest.raises(ValueError, match="native processor kind differs"):
        cap.load_estimate(stores, "FF_AC", measured()[0], config, repository=tmp_path, metadata_rows=rows)


def test_the_ordinary_route_never_estimates_another_raw_binding(tmp_path, monkeypatch):
    config, rows, raw_layout, _ = raw_metadata_TESTONLY(tmp_path / "raw")
    stores = ordinary_profile_TESTONLY(tmp_path / "profile", monkeypatch,
                                       layout=dict(raw_layout, raw_source_binding_sha256="e" * 64))
    with pytest.raises(ValueError, match="differs from measured profile"):
        cap.load_estimate(stores, "FF_AC", measured()[0], config, repository=tmp_path, metadata_rows=rows)


def test_a_bound_raw_profile_estimates_its_qualified_raw_task_through_the_shared_reader(monkeypatch):
    task, config, wrapped = bound_output_profile_TESTONLY()
    wrapped["result_profile"]["measured_native_layout"] = raw_layout_TESTONLY()
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    calls = []

    def layout(*args, **kwargs):
        calls.append((args, kwargs))
        return raw_layout_TESTONLY()

    monkeypatch.setattr(cap, "_native_layout_for_task", layout)
    value = cap._bound_result_estimate("STORES", "FF_AC", task, config, wrapped, {},
                                       repository="TESTONLY", metadata_rows="ROWS")
    assert calls == [(("STORES", "FF_AC", task, config, "ROWS"), {})]
    assert value.case_count == 39
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: native_layout())
    with pytest.raises(ValueError, match="native processor kind differs"):
        cap._bound_result_estimate("STORES", "FF_AC", task, config, wrapped, {},
                                   repository="TESTONLY", metadata_rows="ROWS")


def bound_first_runtime_TESTONLY(tmp_path, monkeypatch, *, reference_kind):
    """A bound first qualification of RAW frames whose all-stage reference names a layout."""
    from darsia_cache import first_qualification as fq
    config, rows, raw_layout, rig_layout = raw_metadata_TESTONLY(tmp_path)
    cases = [frame["case_id"] for frame in config["frames"]]
    task = dict(TASK, backend="cpu", commit="c" * 40, ordered_case_ids=cases, attempt_id="a" * 32)

    def pin(name):
        return dict(path="C:/TESTONLY/" + name, sha256=hashlib.sha256(name.encode()).hexdigest(), size_bytes=1)

    writers = [dict(relative_path="src/darsia/fluidflower/" + path, sha256="d" * 64) for path in (
        "mass/n1_pipeline.py", "mass/n1_statistics.py", "pipeline/series.py", "native/sidecar.py",
        "pipeline/runner.py")]
    reference = dict(requested_native_layout=raw_layout if reference_kind == "raw" else rig_layout,
                     output_writers=writers, reference_commit="b" * 40)
    plan = dict(format=fq.BOUND_OUTPUT_ADMINISTRATION,
                administrative_output_entitlement_bytes=fq.FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES,
                watched_output_scope=dict(path="C:/TESTONLY/results"),
                all_stage_reference=pin("all_stage_reference.json"), scope=dict(TESTONLY="scope"))
    first = SimpleNamespace(bound_mode=True, output_entitlement=plan, provider_binding=pin("provider.json"),
                            binding=pin("request.json"), request=dict(output_entitlement_plan=pin("plan.json")),
                            require_task=lambda lease, actual: actual is task)
    runtime = SimpleNamespace(cache_adapter=SimpleNamespace(lease=SimpleNamespace(first_qualification=first)),
                              stores=SimpleNamespace(roots=lambda series: {"results": "C:/TESTONLY/results"}),
                              job={"series_id": "FF_AC"}, repository="C:/TESTONLY/source")
    monkeypatch.setattr(cap, "settings", lambda *args: (None, cap.DEFAULT_RESERVE_BYTES, first.provider_binding))
    original = cap.read_bound
    monkeypatch.setattr(cap, "read_bound", lambda bound: (reference, b"") if bound == plan["all_stage_reference"]
                        else original(bound))
    # The all-stage reference reader has its own tests in test_cache_first_qualification.
    monkeypatch.setattr(fq, "checked_all_stage_reference", lambda *args, **kwargs: reference)
    commits = []
    monkeypatch.setattr(cap, "_writer_identity", lambda repository, commit, row: commits.append(commit))
    return runtime, task, config, rows, commits


def test_the_bound_first_qualification_admits_raw_frames_against_a_raw_reference(tmp_path, monkeypatch):
    runtime, task, config, rows, commits = bound_first_runtime_TESTONLY(tmp_path, monkeypatch, reference_kind="raw")
    value = cap.first_output_entitlement(runtime, task, config, rows)
    assert value.maximum_output_bytes == 65728995330 and value.case_count == 2
    assert commits == ["b" * 40, "c" * 40] * 5


def test_the_bound_first_qualification_never_admits_raw_frames_against_a_rig_reference(tmp_path, monkeypatch):
    runtime, task, config, rows, _ = bound_first_runtime_TESTONLY(tmp_path, monkeypatch, reference_kind="rig")
    with pytest.raises(ValueError, match="layout differs from the actual all-stage reference"):
        cap.first_output_entitlement(runtime, task, config, rows)


def test_the_n43_entitlement_reads_the_one_shared_layout(monkeypatch):
    from .test_n43_reference_administration import entitlement_runtime_TESTONLY
    runtime, task, config, value, _ = entitlement_runtime_TESTONLY(monkeypatch)
    calls = []

    def layout(*args, **kwargs):
        calls.append((args, kwargs))
        return value["requested_native_layout"]

    monkeypatch.setattr(cap, "_native_layout_for_task", layout)
    cap.n43_entitlement(runtime, task, config, ())
    assert calls == [((runtime.stores, "TESTONLY", task, config, ()), {})]
