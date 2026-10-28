"""Measured GPU device choice with synthetic device lists and timings.

Receipts are written under pytest's tmp_path only. The real child-process
test times the CPU path on a tiny frame; no GPU is used.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from darsia.fluidflower import device_choice as dc
from darsia.fluidflower import execution
from darsia.fluidflower import mass_acceleration as acceleration

from test_mass_acceleration_device import FakeDevice, FakePlatform, fake_cupy, fake_opencl

GIB = 1024**3
SHAPE = (400, 600)
RUNTIME = {
    "python": "3",
    "numpy": "2",
    "implementation_sha256": "a" * 64,
    "cuda": "",
    "opencl": "fake",
}


def device(index, name, *, family="opencl", driver="1.0", fp64=True, free=8 * GIB,
           memory=8 * GIB, largest=None):
    return {
        "family": family,
        "device_id": str(index),
        "name": name,
        "vendor": "Vendor",
        "platform": "Platform",
        "platform_version": "OpenCL 3.0",
        "driver": driver,
        "display_driver": "",
        "fp64": fp64,
        "memory_bytes": memory,
        "max_allocation_bytes": largest or memory,
        "unified_memory": False,
        "pci_bus_id": "",
        "free_bytes": free,
        "free_source": "test",
    }


def inventory(*devices):
    return {"devices": list(devices), "runtime": dict(RUNTIME), "notes": {}}


class FakeMeasure:
    """Measure stub: fixed seconds per device; logs requests and calls."""

    def __init__(self, seconds, *, peak=GIB, cpu=2.0, failures=()):
        self.seconds, self.peak, self.cpu, self.failures = seconds, peak, cpu, failures
        self.calls, self.requests = [], []

    def __call__(self, inventory, *, precision, shape):
        self.calls.append((precision, tuple(shape)))
        return dc.measure_devices(
            inventory, precision=precision, shape=shape, runner=self.runner
        )

    def runner(self, request):
        self.requests.append(request)
        if request["family"] == "cpu":
            return {"status": "measured", "seconds_per_frame_median": self.cpu,
                    "peak_device_bytes": None}
        key = f"{request['family']}:{request['device_id']}"
        if key in self.failures:
            return {"status": "failed", "error": "driver fault"}
        return {"status": "measured", "seconds_per_frame_median": self.seconds[key],
                "peak_device_bytes": self.peak}


def choose(tmp_path, devices, measure, **options):
    return dc.choose_device(
        family=options.pop("family", "opencl"),
        precision=options.pop("precision", "float32"),
        shape=options.pop("shape", SHAPE),
        receipt_path=tmp_path / "receipt.json",
        inventory=devices,
        measure=measure,
        **options,
    )


@pytest.fixture(autouse=True)
def no_device_environment(monkeypatch):
    monkeypatch.delenv("DARSIA_OPENCL_DEVICE", raising=False)
    monkeypatch.delenv("DARSIA_CUDA_DEVICE", raising=False)


def test_two_devices_where_index_zero_is_slower_choose_the_faster(tmp_path):
    devices = inventory(device(0, "Slow integrated"), device(1, "Fast discrete"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1})
    record = choose(tmp_path, devices, measure)
    assert record["basis"] == "measured" and record["device_id"] == "1"
    assert record["device"]["name"] == "Fast discrete"
    assert [row["eligible"] for row in record["candidates"]] == [True, True]
    assert record["benchmark"]["seconds_per_frame"] == {"opencl:0": 0.5, "opencl:1": 0.1}
    assert record["benchmark"]["cpu_seconds_per_frame"] == 2.0
    assert record["benchmark"]["reused"] is False
    assert dc.load_receipt(tmp_path / "receipt.json")["measurements"][0]["shape_rc"] == [400, 600]
    assert measure.calls == [("float32", SHAPE)]


def test_equal_times_go_to_the_lower_index(tmp_path):
    devices = inventory(device(0, "A"), device(1, "B"))
    record = choose(tmp_path, devices, FakeMeasure({"opencl:0": 0.2, "opencl:1": 0.2}))
    assert record["device_id"] == "0"


def test_explicit_device_and_environment_override_the_measurement(tmp_path, monkeypatch):
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1})
    record = choose(tmp_path, devices, measure, explicit="0", explicit_source="GUI")
    assert (record["basis"], record["source"], record["device_id"]) == ("explicit", "GUI", "0")
    assert record["device"]["name"] == "Slow" and record["benchmark"] is None
    monkeypatch.setenv("DARSIA_OPENCL_DEVICE", "0")
    record = choose(tmp_path, devices, measure)
    assert record["source"] == "environment:DARSIA_OPENCL_DEVICE"
    assert record["device_id"] == "0"
    assert choose(tmp_path, devices, measure, explicit="1")["device_id"] == "1"
    assert measure.calls == []


def test_explicit_choice_shows_stored_times_without_measuring(tmp_path):
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    choose(tmp_path, devices, FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1}))
    record = choose(tmp_path, devices, None, explicit="0")
    assert record["benchmark"]["seconds_per_frame"]["opencl:0"] == 0.5


def test_float64_skips_a_device_without_fp64(tmp_path):
    devices = inventory(device(0, "Fast single", fp64=False), device(1, "Double capable"))
    measure = FakeMeasure({"opencl:0": 0.05, "opencl:1": 0.3})
    record = choose(tmp_path, devices, measure, precision="float64")
    assert record["device_id"] == "1"
    skipped = record["candidates"][0]
    assert not skipped["eligible"] and "no float64 support" in skipped["reasons"][0]
    assert [r["device_id"] for r in measure.requests] == ["1", "cpu"]
    assert choose(tmp_path, devices, measure)["device_id"] == "0"


def test_receipt_is_reused_and_invalidated_by_a_driver_change(tmp_path):
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1})
    first = choose(tmp_path, devices, measure)
    second = choose(tmp_path, devices, measure)
    assert len(measure.calls) == 1 and second["benchmark"]["reused"] is True
    assert second["benchmark"]["receipt_sha256"] == first["benchmark"]["receipt_sha256"]
    updated = inventory(device(0, "Slow"), device(1, "Fast", driver="2.0"))
    third = choose(tmp_path, updated, measure)
    assert len(measure.calls) == 2 and third["benchmark"]["reused"] is False
    assert third["benchmark"]["fingerprint"] != first["benchmark"]["fingerprint"]
    removed = inventory(device(0, "Slow"), device(1, "Fast", driver="2.0"), device(2, "New"))
    measure.seconds["opencl:2"] = 0.01
    assert choose(tmp_path, removed, measure)["device_id"] == "2"
    assert len(measure.calls) == 3


def test_another_precision_or_frame_size_adds_a_measurement(tmp_path):
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1})
    choose(tmp_path, devices, measure)
    choose(tmp_path, devices, measure, precision="float64")
    choose(tmp_path, devices, measure, shape=(100, 200))
    choose(tmp_path, devices, measure, shape=(100, 200))
    receipt = dc.load_receipt(tmp_path / "receipt.json")
    assert len(measure.calls) == 3 and len(receipt["measurements"]) == 3


def test_free_memory_and_largest_allocation_rules_skip_a_device(tmp_path):
    peak = 3 * GIB
    devices = inventory(device(0, "Roomy"), device(1, "Tight", free=5 * GIB))
    measure = FakeMeasure({"opencl:0": 0.4, "opencl:1": 0.1}, peak=peak)
    assert choose(tmp_path, devices, measure)["device_id"] == "1"
    record = choose(tmp_path, devices, measure, workers=2)
    assert record["device_id"] == "0" and "worker(s)" in record["candidates"][1]["reasons"][0]
    small = inventory(device(0, "Roomy"), device(1, "Small blocks", largest=1024))
    record = choose(tmp_path / "other", small, measure)
    assert record["device_id"] == "0"
    assert "largest allocation" in record["candidates"][1]["reasons"][0]


def test_unreported_free_memory_is_checked_against_total_memory(tmp_path):
    devices = inventory(device(0, "A", free=None, memory=2 * GIB), device(1, "B", free=None))
    measure = FakeMeasure({"opencl:0": 0.1, "opencl:1": 0.2}, peak=3 * GIB)
    record = choose(tmp_path, devices, measure)
    assert record["device_id"] == "1"
    assert record["candidates"][1]["free_source"].startswith("total memory")


def test_no_eligible_device_names_every_reason(tmp_path):
    devices = inventory(device(0, "Broken"), device(1, "No double", fp64=False))
    measure = FakeMeasure({"opencl:0": 0.1, "opencl:1": 0.1}, failures=("opencl:0",))
    with pytest.raises(RuntimeError) as raised:
        choose(tmp_path, devices, measure, precision="float64")
    text = str(raised.value)
    assert "driver fault" in text and "no float64 support" in text
    assert "DARSIA_OPENCL_DEVICE" in text


def test_a_failed_device_measurement_only_fails_that_device(tmp_path):
    devices = inventory(device(0, "Fast but faulty"), device(1, "Working"))
    measure = FakeMeasure({"opencl:0": 0.01, "opencl:1": 0.2}, failures=("opencl:0",))
    record = choose(tmp_path, devices, measure)
    assert record["device_id"] == "1"
    assert record["candidates"][0]["reasons"] == ["not measured: driver fault"]


def test_single_device_needs_no_measurement(tmp_path):
    devices = inventory(device(0, "Only OpenCL"), device(0, "A CUDA device", family="cuda"))
    record = choose(tmp_path, devices, FakeMeasure({}))
    assert record["basis"] == "single_device" and record["device_id"] == "0"
    assert record["device"]["name"] == "Only OpenCL" and record["benchmark"] is None
    assert not (tmp_path / "receipt.json").exists()
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        choose(tmp_path, inventory(device(0, "Only OpenCL")), None, family="cuda")


def test_choose_device_reads_the_environment_variable_of_its_family(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_CUDA_DEVICE", "1")
    devices = inventory(device(0, "A"), device(1, "B"))
    record = choose(tmp_path, devices, FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1}))
    assert record["basis"] == "measured"
    with pytest.raises(ValueError, match="No device choice"):
        choose(tmp_path, devices, None, family="metal")


def test_measure_devices_skips_float64_and_times_the_cpu():
    devices = inventory(device(0, "No double", fp64=False), device(1, "Double"))
    measure = FakeMeasure({"opencl:1": 0.3})
    result = dc.measure_devices(devices, precision="float64", shape=SHAPE, runner=measure.runner)
    assert [row["status"] for row in result["devices"]] == ["skipped", "measured"]
    assert result["cpu"]["seconds_per_frame_median"] == 2.0
    assert result["protocol"]["warmup_frames"] == 1
    assert result["protocol"]["statistic"] == "median"


def test_child_measurement_times_the_cpu_without_loading_darsia():
    result = dc.run_measurement_child(
        {"family": "cpu", "device_id": "cpu", "precision": "float32", "shape_rc": [16, 24]}
    )
    assert result["status"] == "measured", result
    assert result["darsia_package_loaded"] is False
    assert dc.MINIMUM_REPEATS <= result["repeats"] <= dc.MAXIMUM_REPEATS
    assert result["seconds_per_frame_median"] == float(np.median(result["seconds_per_frame"]))
    failed = dc.run_measurement_child(
        {"family": "opencl", "device_id": "99", "precision": "float32", "shape_rc": [4, 4]}
    )
    assert failed["status"] == "failed" and failed["darsia_package_loaded"] is False


def test_child_measurement_reports_a_changed_kernel_file(tmp_path):
    copy = tmp_path / "kernel.py"
    copy.write_bytes(Path(acceleration.__file__).read_bytes())
    result = dc.child_measurement(
        {"implementation": str(copy), "implementation_sha256": "0" * 64, "family": "cpu",
         "device_id": "cpu", "precision": "float32", "shape_rc": [4, 4]}
    )
    assert result["status"] == "failed" and "changed" in result["error"]


def test_a_hung_or_crashed_child_only_fails_its_device(monkeypatch):
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(dc.subprocess, "run", hang)
    request = {"family": "opencl", "device_id": "0", "precision": "float32", "shape_rc": [4, 4]}
    assert dc.run_measurement_child(request, timeout=5)["error"] == "no result within 5 s"

    def crash(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 3, stdout="", stderr="driver fault\n")

    monkeypatch.setattr(dc.subprocess, "run", crash)
    failed = dc.run_measurement_child(request)
    assert failed["status"] == "failed" and "exit code 3" in failed["error"]
    assert "driver fault" in failed["error"]


def test_store_and_load_receipt_detect_tampering(tmp_path):
    path = tmp_path / "nested" / "receipt.json"
    sealed = dc.store_receipt(path, {"format": dc.RECEIPT_FORMAT, "measurements": []})
    assert dc.load_receipt(path) == sealed
    damaged = json.loads(path.read_text(encoding="utf-8"))
    damaged["measurements"] = [{"precision": "float32"}]
    path.write_text(json.dumps(damaged), encoding="utf-8")
    assert dc.load_receipt(path) is None
    assert dc.load_receipt(tmp_path / "missing.json") is None
    assert [p.name for p in path.parent.iterdir()] == ["receipt.json"]


def test_unwritable_receipt_is_reported_never_fatal(tmp_path, monkeypatch):
    def refuse(path, receipt):
        raise PermissionError("read-only")

    monkeypatch.setattr(dc, "store_receipt", refuse)
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    record = choose(tmp_path, devices, FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1}))
    assert record["device_id"] == "1"
    assert "read-only" in record["benchmark"]["store_error"]


def test_find_measurement_and_ensure_measurement_force(tmp_path):
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1})
    path = tmp_path / "receipt.json"
    receipt, measurement, reused = dc.ensure_measurement(
        devices, precision="float32", shape=SHAPE, path=path, measure=measure)
    assert not reused and dc.find_measurement(receipt, "float32", SHAPE) == measurement
    assert dc.find_measurement(receipt, "float64", SHAPE) is None
    dc.ensure_measurement(devices, precision="float32", shape=SHAPE, path=path,
                          measure=measure, force=True)
    assert len(measure.calls) == 2
    assert len(dc.load_receipt(path)["measurements"]) == 1


def test_measure_and_store_replaces_the_same_measurement(tmp_path):
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1})
    for _ in range(2):
        receipt = dc.measure_and_store(precision="float32", shape=(8, 8),
                                       receipt_path=tmp_path / "r.json",
                                       inventory=devices, measure=measure)
    assert len(measure.calls) == 2 and len(receipt["measurements"]) == 1


def test_concurrent_choices_measure_once(tmp_path):
    devices = inventory(device(0, "Slow"), device(1, "Fast"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1})
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(choose(tmp_path, devices, measure)))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(measure.calls) == 1 and {r["device_id"] for r in results} == {"1"}


def test_receipt_lock_excludes_a_second_holder(tmp_path):
    path = tmp_path / "lock"
    first, second = open(path, "a+b"), open(path, "a+b")
    try:
        assert dc._try_lock(first)
        assert not dc._try_lock(second)
        dc._unlock(first)
        assert dc._try_lock(second)
        dc._unlock(second)
    finally:
        first.close()
        second.close()


def test_device_memory_fits_is_the_advisors_vram_rule():
    assert dc.device_memory_fits(4 * GIB, 2 * GIB, 2)
    assert not dc.device_memory_fits(4 * GIB - 1, 2 * GIB, 2)
    assert dc.device_memory_fits(4 * GIB, 2 * GIB, 0)
    assert not dc.device_memory_fits(4 * GIB, 0, 1)


def test_backend_family_and_device_identity():
    assert [dc.backend_family(b) for b in ("opencl-float32", "cuda-float64", "metal-float32",
                                           "cpu-float32")] == ["opencl", "cuda", None, None]
    identity = dc.device_identity(device(0, "A"))
    assert "free_bytes" not in identity and identity["name"] == "A"
    assert set(identity) == set(dc.IDENTITY_FIELDS)


def test_fingerprint_ignores_free_memory_but_not_runtime():
    base = inventory(device(0, "A"))
    busier = inventory(device(0, "A", free=GIB))
    assert dc.device_fingerprint(base) == dc.device_fingerprint(busier)
    newer = dict(base, runtime=dict(RUNTIME, opencl="newer"))
    assert dc.device_fingerprint(base) != dc.device_fingerprint(newer)


def test_committed_device_reads_old_and_new_run_records():
    old = {"device_id": "1", "device_name": "GPU B", "device_driver": "2.0"}
    new = dict(old, device_choice={"basis": "measured", "device": {"name": "GPU B"}})
    assert dc.committed_device([], "opencl") is None
    pinned = dc.committed_device([old, new], "opencl")
    assert (pinned["device_id"], pinned["name"], pinned["driver"]) == ("1", "GPU B", "2.0")
    assert pinned["choice"] is None
    with pytest.raises(RuntimeError, match="different devices"):
        dc.committed_device([old, dict(old, device_id="0")], "opencl")


def test_resumed_job_keeps_its_device_and_stops_when_it_is_gone(tmp_path):
    devices = inventory(device(0, "Fast now"), device(1, "Used before", driver="2.0"))
    measure = FakeMeasure({"opencl:0": 0.01, "opencl:1": 0.5})
    committed = [{"device_id": "1", "device_name": "Used before", "device_driver": "2.0"}]
    options = dict(backend="opencl-float32", shape=SHAPE, receipt_path=tmp_path / "r.json",
                   inventory=devices, measure=measure, committed=committed)
    record = dc.resolve_job_device(**options)
    assert (record["basis"], record["device_id"]) == ("resumed", "1") and measure.calls == []
    explicit = dc.resolve_job_device(**dict(options, explicit="0", explicit_source="GUI"))
    assert (explicit["basis"], explicit["device_id"]) == ("explicit", "0")
    gone = dict(options, inventory=inventory(device(0, "Fast now"), device(1, "Used before",
                                                                           driver="3.0")))
    with pytest.raises(RuntimeError, match="not available now"):
        dc.resolve_job_device(**gone)
    fresh = dc.resolve_job_device(**dict(options, committed=[]))
    assert (fresh["basis"], fresh["device_id"]) == ("measured", "0")
    assert dc.resolve_job_device(**dict(options, backend="cpu-float32")) is None
    assert dc.resolve_job_device(**dict(options, backend="metal-float32")) is None


def test_verify_session_device_refuses_another_device():
    choice = {"device_id": "1", "device": {"name": "GPU B", "driver": "2.0"}}
    dc.verify_session_device({"device_id": "1", "device_name": "GPU B ",
                              "device_driver": "2.0"}, choice)
    for opened in ({"device_id": "0", "device_name": "GPU B", "device_driver": "2.0"},
                   {"device_id": "1", "device_name": "GPU A", "device_driver": "2.0"},
                   {"device_id": "1", "device_name": "GPU B", "device_driver": "2.1"}):
        with pytest.raises(RuntimeError, match="another device"):
            dc.verify_session_device(opened, choice)
    dc.verify_session_device({"device_id": "3", "device_name": "X"},
                             {"device_id": "3", "device": {"name": None, "driver": None}})


def test_task_device_request_separates_explicit_from_automatic():
    assert dc.task_device_request({}) == ("", "")
    assert dc.task_device_request({"adapter": {"settings": {"device_id": "2"}},
                                   "resources": {"device_id": "1"}}) == ("2", "task adapter setting")
    assert dc.task_device_request({"resources": {"device_id": "1"}}) == ("1", "task resources")
    for automatic in ("single_device", "measured"):
        assert dc.task_device_request(
            {"resources": {"device_id": "0", "device_choice": automatic}}) == ("", "")


def hardware(*names, backend="opencl-float32"):
    return {"accelerators": [{"backend": backend, "supported_backends": [backend],
                              "device_id": str(index), "name": name}
                             for index, name in enumerate(names)]}


def test_requested_device_resources_and_backend_devices():
    two = hardware("A", "B")
    assert [row["name"] for row in dc.backend_devices(two, "opencl-float32")] == ["A", "B"]
    assert dc.backend_devices(two, "cuda-float32") == []
    assert dc.requested_device_resources(two, "opencl-float32") == {
        "device_id": "", "device_name": "Fastest of 2 devices, measured at job start",
        "device_choice": "measured"}
    assert dc.requested_device_resources(two, "opencl-float32", "1") == {
        "device_id": "1", "device_name": "B", "device_choice": "explicit"}
    assert dc.requested_device_resources(hardware("Only"), "opencl-float32") == {
        "device_id": "0", "device_name": "Only", "device_choice": "single_device"}
    with pytest.raises(ValueError, match="index 7"):
        dc.requested_device_resources(two, "opencl-float32", "7")
    with pytest.raises(ValueError, match="No device provides"):
        dc.requested_device_resources(two, "cuda-float32")


def test_recommend_resources_leaves_several_devices_to_the_measurement():
    def host(accelerators):
        return {"cpu": {"logical_cores": 8, "physical_cores": 4},
                "memory": {"available_bytes": 16 * GIB}, **accelerators}

    several = execution.recommend_resources("mass_analysis", host(hardware("A", "B")))
    assert (several["device_id"], several["device_choice"]) == ("", "measured")
    single = execution.recommend_resources("mass_analysis", host(hardware("Only")))
    assert (single["device_id"], single["device_name"], single["device_choice"]) == (
        "0", "Only", "single_device")
    cpu = execution.recommend_resources("mass_analysis", host({"accelerators": []}))
    assert (cpu["device_id"], cpu["device_name"]) == ("", "CPU") and "device_choice" not in cpu


def test_receipt_rows_show_the_cpu_and_the_fastest_of_each_backend(tmp_path):
    devices = inventory(device(0, "Slow"), device(1, "Fast"), device(0, "Card", family="cuda"))
    measure = FakeMeasure({"opencl:0": 0.5, "opencl:1": 0.1, "cuda:0": 0.2},
                          failures=("cuda:0",))
    choose(tmp_path, devices, measure)
    rows = dc.receipt_rows(dc.load_receipt(tmp_path / "receipt.json"))
    by_device = {row["device"]: row for row in rows}
    assert by_device["OpenCL 1: Fast"]["fastest"] and not by_device["OpenCL 0: Slow"]["fastest"]
    assert by_device["CUDA 0: Card"]["status"] == "Failed: driver fault"
    cpu = by_device["CPU (NumPy, same mass algebra)"]
    assert (cpu["family"], cpu["seconds_per_frame"], cpu["fastest"]) == ("cpu", 2.0, False)
    assert {row["frame"] for row in rows} == {"400x600"}
    assert dc.receipt_rows(None) == []


def test_synthetic_frame_and_benchmark_configuration_are_valid():
    inputs, active, voxel = dc.synthetic_frame((5, 7))
    assert inputs["acidification"].shape == (5, 7) and inputs["acidification"].dtype == np.float64
    assert np.all(inputs["gas_probability"] <= inputs["co2_probability"])
    assert active.dtype == np.uint8 and voxel.shape == (5, 7)
    again, _, _ = dc.synthetic_frame((5, 7))
    assert np.array_equal(again["acidification"], inputs["acidification"])
    acceleration.MassKernelConfiguration(**dc.benchmark_configuration()).validate()


def test_session_device_bytes_reads_buffers_or_the_memory_pool():
    from types import SimpleNamespace

    buffer = SimpleNamespace(size=10)
    opencl = SimpleNamespace(active_buffer=buffer, voxel_buffer=buffer, lut_signal_buffer=buffer,
                             lut_concentration_buffer=buffer, dynamic_buffers=[buffer] * 5,
                             output_buffers=[buffer] * 6)
    assert dc.session_device_bytes(opencl) == 150
    pool = SimpleNamespace(total_bytes=lambda: 4096)
    cuda = SimpleNamespace(cp=SimpleNamespace(get_default_memory_pool=lambda: pool))
    assert dc.session_device_bytes(cuda) == 4096
    assert dc.session_device_bytes(object()) is None


def test_enumerate_devices_reads_both_runtimes_and_free_memory(monkeypatch):
    platforms = [
        FakePlatform("NVIDIA CUDA", [FakeDevice("Card", nv_bus=1, driver="600.1")]),
        FakePlatform("AMD", [FakeDevice("Integrated", unified=True, amd_free_kib=1024)]),
        FakePlatform("Other", [FakeDevice("Shared", unified=True), FakeDevice("Plain")]),
    ]
    monkeypatch.setitem(sys.modules, "pyopencl", fake_opencl(platforms))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy(["Card"]))
    monkeypatch.setattr(dc, "_nvidia_smi", lambda: {
        "00000000:01:00": {"driver": "600.1", "free_bytes": 5 * GIB}})
    monkeypatch.setattr(execution, "_memory_bytes", lambda: (32 * GIB, 7 * GIB))
    found = dc.enumerate_devices()
    rows = {(row["family"], row["device_id"]): row for row in found["devices"]}
    assert rows[("cuda", "0")]["display_driver"] == "600.1"
    assert rows[("cuda", "0")]["free_source"] == "nvidia-smi"
    assert rows[("opencl", "0")]["free_bytes"] == 5 * GIB
    assert rows[("opencl", "1")]["free_bytes"] == 1024 * 1024
    assert rows[("opencl", "2")]["free_bytes"] == 7 * GIB
    assert rows[("opencl", "3")]["free_bytes"] is None
    assert rows[("opencl", "3")]["free_source"] == "not reported"
    assert found["runtime"]["opencl"] == "fake-opencl" and found["runtime"]["cuda"] == "fake-cupy"
    monkeypatch.setitem(sys.modules, "cupy", None)
    without = dc.enumerate_devices()
    assert "cuda" in without["notes"] and all(r["family"] == "opencl" for r in without["devices"])


def test_nvidia_smi_rows_are_keyed_by_bus(monkeypatch):
    output = "00000000:01:00.0, 617.42, 13270\n0000:02:00.0, 617.42, x\nbad line\n"
    monkeypatch.setattr(dc.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout=output, stderr=""))
    assert dc._nvidia_smi() == {
        "00000000:01:00": {"driver": "617.42", "free_bytes": 13270 * 1024 * 1024}}

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(dc.subprocess, "run", missing)
    assert dc._nvidia_smi() == {}


def test_native_worker_route_loads_none_of_the_device_choice_modules():
    """The CPU route of the native pipeline never imports the GPU choice code."""
    source = Path(dc.__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "import darsia.fluidflower.pipeline.worker, darsia.fluidflower.pipeline.runner\n"
        "import darsia.fluidflower.mass.n1_pipeline\n"
        "loaded = [m for m in ('darsia.fluidflower.device_choice', "
        "'darsia.fluidflower.mass_acceleration', 'darsia.fluidflower.mass_materialization', "
        "'darsia.fluidflower.execution') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=dict(os.environ, PYTHONPATH=str(source)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_receipt_path_is_host_local():
    path = dc.default_receipt_path()
    assert path.parent.parent == execution.default_execution_cache_root().parent
    assert path.name == "receipt.json" and path.parent.name == "gpu_benchmark"
