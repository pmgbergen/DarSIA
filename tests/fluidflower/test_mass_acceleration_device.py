"""Device choice inside the accelerated mass sessions, on fake GPU runtimes.

The fake PyOpenCL and CuPy modules compute the fused kernel with NumPy and log
the device of every frame. No GPU, driver or user profile is touched.
"""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from darsia.fluidflower import device_choice
from darsia.fluidflower import mass_acceleration as acceleration

GIB = 1024**3


class FakeLogicError(Exception):
    """Raised by vendor queries a fake device does not answer."""


class FakeDevice:
    def __init__(
        self,
        name,
        *,
        driver="1.0",
        fp64=True,
        gpu=True,
        vendor="Vendor",
        memory=8 * GIB,
        max_allocation=None,
        unified=False,
        amd_free_kib=None,
        nv_bus=None,
    ):
        self.name, self.vendor, self.driver_version = name, vendor, driver
        self.version = "OpenCL 3.0"
        self.extensions = "cl_khr_fp64" if fp64 else "cl_khr_byte_addressable_store"
        self.double_fp_config = 63 if fp64 else 0
        self.type = 4 if gpu else 2
        self.global_mem_size = memory
        self.max_mem_alloc_size = max_allocation or memory // 4
        self.host_unified_memory = unified
        self._amd_free_kib, self._nv_bus = amd_free_kib, nv_bus

    @property
    def global_free_memory_amd(self):
        if self._amd_free_kib is None:
            raise FakeLogicError("INVALID_VALUE")
        return [self._amd_free_kib]

    @property
    def pci_bus_id_nv(self):
        if self._nv_bus is None:
            raise FakeLogicError("INVALID_VALUE")
        return self._nv_bus

    @property
    def pci_slot_id_nv(self):
        return self.pci_bus_id_nv * 0

    @property
    def pci_domain_id_nv(self):
        return self.pci_bus_id_nv * 0


class FakePlatform:
    def __init__(self, name, devices, version="OpenCL 3.0 Fake"):
        self.name, self.version, self.devices = name, version, list(devices)

    def get_devices(self, device_type=None):
        return [
            device
            for device in self.devices
            if device_type is None or device.type & device_type
        ]


class FakeBuffer:
    def __init__(self, context, flags, size=0, hostbuf=None):
        self.data = None if hostbuf is None else np.array(hostbuf).ravel()
        self.size = int(size) if hostbuf is None else self.data.nbytes


def fused_kernel(arguments, dtype):
    """Evaluate the OpenCL/CUDA mass kernel with NumPy, element for element."""
    (acid, co2, gas, active, voxel, solubility, density, lut_x, lut_y, _size,
     titration, smoothstep, acid_low, acid_high, co2_onset, co2_high, gas_onset,
     gas_high, max_saturation, diffusion) = arguments
    one = dtype.type(1)

    def transfer(values, low, high):
        scaled = np.clip((values - low) / (high - low), 0, one).astype(dtype)
        return scaled * scaled * (3 - 2 * scaled) if smoothstep else scaled

    fraction = transfer(acid, acid_low, acid_high)
    concentration = (
        np.interp(fraction, lut_x, lut_y).astype(dtype) if titration else fraction
    )
    gate = transfer(co2, co2_onset, co2_high)
    saturation = (max_saturation * transfer(gas, gas_onset, gas_high)).astype(dtype)
    concentration = np.clip(concentration * gate * diffusion, 0, one).astype(dtype)
    concentration[active == 0] = 0
    saturation[active == 0] = 0
    volume = voxel * dtype.type(1000)
    gas_mass = density * saturation * volume
    aqueous = solubility * concentration * np.maximum(one - saturation, 0) * volume
    return [
        np.asarray(values, dtype=dtype)
        for values in (concentration, saturation, gate, aqueous, gas_mass, aqueous + gas_mass)
    ]


def fake_opencl(platforms):
    """Return a PyOpenCL stand-in whose kernel logs the device of every frame."""
    cl = SimpleNamespace(
        VERSION_TEXT="fake-opencl",
        device_type=SimpleNamespace(GPU=4, CPU=2),
        mem_flags=SimpleNamespace(READ_ONLY=4, WRITE_ONLY=2, COPY_HOST_PTR=32),
        Buffer=FakeBuffer,
        frames=[],
        contexts=[],
    )
    cl.get_platforms = lambda: list(platforms)

    def context(devices):
        value = SimpleNamespace(devices=list(devices))
        cl.contexts.append(value)
        return value

    class Program:
        def __init__(self, context, source):
            self.context, self.source = context, source

        def build(self):
            return self

    def kernel(program, name):
        def run(queue, global_size, local_size, *arguments, wait_for=None):
            cl.frames.append(queue.context.devices[0].name)
            inputs = [
                value.data if isinstance(value, FakeBuffer) else value
                for value in arguments[:20]
            ]
            for buffer, values in zip(
                arguments[20:], fused_kernel(inputs, arguments[0].data.dtype)
            ):
                buffer.data = values
            return object()

        return run

    def enqueue_copy(queue, destination, source, wait_for=None, is_blocking=True):
        if isinstance(destination, FakeBuffer):
            destination.data = np.array(source).ravel()
        else:
            np.copyto(destination, source.data.reshape(destination.shape))
        return object()

    cl.Context = context
    cl.Program = Program
    cl.CommandQueue = lambda context: SimpleNamespace(context=context, finish=lambda: None)
    cl.Kernel = kernel
    cl.enqueue_copy = enqueue_copy
    cl.wait_for_events = lambda events: None
    return cl


def fake_cupy(names, *, driver=12040):
    """Return a CuPy stand-in for CUDA sessions; the kernel logs the current device."""
    state = SimpleNamespace(current=None, frames=[])

    class Device:
        def __init__(self, index):
            self.id = index

        def use(self):
            state.current = self.id

        def synchronize(self):
            pass

    runtime = SimpleNamespace(
        getDeviceCount=lambda: len(names),
        driverGetVersion=lambda: driver,
        runtimeGetVersion=lambda: 12000,
        getDeviceProperties=lambda index: {
            "name": names[index].encode(),
            "totalGlobalMem": 8 * GIB,
            "pciDomainID": 0,
            "pciBusID": index + 1,
            "pciDeviceID": 0,
            "integrated": 0,
        },
    )

    def elementwise(input_signature, output_signature, operation, name=""):
        dtype = np.dtype("float32" if name.endswith("float32") else "float64")

        def run(*arguments):
            state.frames.append(state.current)
            return fused_kernel(arguments, dtype)

        return run

    pool = SimpleNamespace(total_bytes=lambda: 3 * GIB, free_all_blocks=lambda: None)
    return SimpleNamespace(
        __version__="fake-cupy",
        cuda=SimpleNamespace(runtime=runtime, Device=Device),
        ElementwiseKernel=elementwise,
        asarray=np.asarray,
        asnumpy=np.asarray,
        get_default_memory_pool=lambda: pool,
        state=state,
    )


@pytest.fixture
def opencl(monkeypatch):
    """Install a fake PyOpenCL with an empty program cache; return an installer."""

    def install(platforms):
        cl = fake_opencl(platforms)
        monkeypatch.setitem(sys.modules, "pyopencl", cl)
        monkeypatch.setattr(acceleration, "_OPENCL_PROGRAM_CACHE", {})
        return cl

    monkeypatch.delenv("DARSIA_OPENCL_DEVICE", raising=False)
    monkeypatch.delenv("DARSIA_CUDA_DEVICE", raising=False)
    return install


def configuration():
    return acceleration.MassKernelConfiguration(**device_choice.benchmark_configuration())


def frame(shape=(4, 6)):
    inputs, active, voxel = device_choice.synthetic_frame(shape)
    return inputs, active, voxel


def session(backend, *, device_id="", shape=(4, 6)):
    _, active, voxel = frame(shape)
    return acceleration.create_mass_compute_session(
        backend=backend,
        active=active,
        voxel_volume_m3=voxel,
        configuration=configuration(),
        device_id=device_id,
    )


def two_gpus():
    return [
        FakePlatform("Platform A", [FakeDevice("CPU A", gpu=False), FakeDevice("GPU A")]),
        FakePlatform("Platform B", [FakeDevice("GPU B", driver="2.0")]),
    ]


def test_opencl_gpu_devices_lists_gpus_in_platform_order(opencl):
    cl = opencl(two_gpus())
    pairs = acceleration.opencl_gpu_devices(cl)
    assert [(platform.name, device.name) for platform, device in pairs] == [
        ("Platform A", "GPU A"),
        ("Platform B", "GPU B"),
    ]


def test_opencl_supports_float64_reads_the_config_or_the_extension():
    assert acceleration.opencl_supports_float64(FakeDevice("A", fp64=True))
    assert not acceleration.opencl_supports_float64(FakeDevice("B", fp64=False))
    extension_only = FakeDevice("C", fp64=False)
    extension_only.extensions = "cl_khr_fp64"
    assert acceleration.opencl_supports_float64(extension_only)


def test_cuda_driver_version_names_the_driver_api():
    assert acceleration.cuda_driver_version(fake_cupy(["G"], driver=13040)) == (
        "CUDA driver API 13040"
    )


def test_resolve_device_index_explicit_environment_single_and_measured(monkeypatch):
    calls = []

    def choose(**options):
        calls.append(options)
        return {"basis": "measured", "device_id": "2", "device": {}}

    monkeypatch.setattr(device_choice, "choose_device", choose)
    monkeypatch.delenv("DARSIA_OPENCL_DEVICE", raising=False)
    resolve = acceleration.resolve_device_index
    common = dict(family="opencl", count=3, precision="float32", shape=(4, 6))
    assert resolve(requested="1", **common) == (
        1,
        {"basis": "explicit", "source": "device_id"},
    )
    monkeypatch.setenv("DARSIA_OPENCL_DEVICE", "0")
    assert resolve(requested="", **common)[1] == {
        "basis": "explicit",
        "source": "environment:DARSIA_OPENCL_DEVICE",
    }
    assert resolve(requested="2", **common)[0] == 2
    monkeypatch.delenv("DARSIA_OPENCL_DEVICE")
    assert resolve(requested="", **dict(common, count=1)) == (
        0,
        {"basis": "single_device"},
    )
    assert calls == []
    index, choice = resolve(requested="", **common)
    assert index == 2 and choice["basis"] == "measured"
    assert calls == [{"family": "opencl", "precision": "float32", "shape": (4, 6)}]


def test_one_opencl_device_is_used_as_before_and_recorded(opencl, monkeypatch):
    opencl([FakePlatform("Only", [FakeDevice("Lone GPU")])])
    monkeypatch.setattr(
        device_choice,
        "choose_device",
        lambda **_: pytest.fail("one device needs no measurement"),
    )
    value = session("opencl-float32")
    assert value.device_index == 0
    provenance = value.provenance()
    assert provenance["device_name"] == "Lone GPU"
    assert provenance["device_choice"] == {"basis": "single_device"}


def test_opencl_session_uses_the_measured_device_and_never_switches(opencl, monkeypatch):
    cl = opencl(two_gpus())
    answers = iter(["1", "0", "0"])
    calls = []

    def choose(**options):
        calls.append(options)
        device_id = next(answers)
        name, driver = ("GPU B", "2.0") if device_id == "1" else ("GPU A", "1.0")
        return {
            "basis": "measured",
            "device_id": device_id,
            "device": {"name": name, "driver": driver},
        }

    monkeypatch.setattr(device_choice, "choose_device", choose)
    value = session("opencl-float32")
    inputs, _, _ = frame()
    for _ in range(3):
        result = value.compute_frame(**inputs)
    assert value.device_index == 1 and len(calls) == 1
    assert cl.frames == ["GPU B"] * 3
    assert value.provenance()["device_choice"]["device_id"] == "1"
    assert np.all(np.isfinite(result.mass_total))


def test_explicit_opencl_device_wins_without_a_measurement(opencl, monkeypatch):
    cl = opencl(two_gpus())
    monkeypatch.setattr(
        device_choice,
        "choose_device",
        lambda **_: pytest.fail("an explicit device needs no measurement"),
    )
    value = session("opencl-float32", device_id="0")
    value.compute_frame(**frame()[0])
    monkeypatch.setenv("DARSIA_OPENCL_DEVICE", "1")
    other = session("opencl-float32")
    other.compute_frame(**frame()[0])
    assert cl.frames == ["GPU A", "GPU B"]
    assert other.provenance()["device_choice"]["source"] == (
        "environment:DARSIA_OPENCL_DEVICE"
    )


def test_opencl_session_refuses_a_device_list_that_changed(opencl, monkeypatch):
    opencl(two_gpus())
    monkeypatch.setattr(
        device_choice,
        "choose_device",
        lambda **_: {
            "basis": "measured",
            "device_id": "1",
            "device": {"name": "Another GPU", "driver": "2.0"},
        },
    )
    with pytest.raises(RuntimeError, match="device list changed"):
        session("opencl-float32")


def test_explicit_float64_on_a_device_without_fp64_still_fails(opencl):
    opencl([FakePlatform("Only", [FakeDevice("Half", fp64=False)])])
    with pytest.raises(RuntimeError, match="does not support float64"):
        session("opencl-float64")


def test_cuda_session_records_device_driver_and_choice(monkeypatch):
    cp = fake_cupy(["GPU Zero", "GPU One"])
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.delenv("DARSIA_CUDA_DEVICE", raising=False)
    monkeypatch.setattr(
        device_choice,
        "choose_device",
        lambda **_: {
            "basis": "measured",
            "device_id": "1",
            "device": {"name": "GPU One", "driver": "CUDA driver API 12040"},
        },
    )
    value = session("cuda-float32")
    for _ in range(2):
        value.compute_frame(**frame()[0])
    provenance = value.provenance()
    assert cp.state.frames == [1, 1]
    assert value._device_name() == "GPU One"
    assert value._device_driver() == "CUDA driver API 12040"
    assert provenance["device_driver"] == "CUDA driver API 12040"
    assert provenance["device_choice"]["basis"] == "measured"
    single = fake_cupy(["Only GPU"])
    monkeypatch.setitem(sys.modules, "cupy", single)
    assert session("cuda-float64").provenance()["device_choice"] == {
        "basis": "single_device"
    }
