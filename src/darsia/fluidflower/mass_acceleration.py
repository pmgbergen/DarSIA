"""Backend-explicit phase and mass algebra for FluidFlower analysis."""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

ACCELERATED_MASS_BACKENDS = (
    "cuda-float32",
    "cuda-float64",
    "opencl-float32",
    "opencl-float64",
    "metal-float32",
)
NONREFERENCE_MASS_BACKENDS = ("cpu-float32", *ACCELERATED_MASS_BACKENDS)
DEVICE_ENVIRONMENT_VARIABLES = {
    "cuda": "DARSIA_CUDA_DEVICE",
    "opencl": "DARSIA_OPENCL_DEVICE",
}


def opencl_gpu_devices(cl: Any) -> list[tuple[Any, Any]]:
    """Return every OpenCL GPU as ``(platform, device)`` in platform order.

    The position in this list is the OpenCL device index that DarSIA records
    and that ``device_id`` and ``DARSIA_OPENCL_DEVICE`` select.
    """

    return [
        (platform, device)
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]


def opencl_supports_float64(device: Any) -> bool:
    """Return whether an OpenCL device offers double precision (cl_khr_fp64)."""

    return (
        int(getattr(device, "double_fp_config", 0)) != 0
        or "cl_khr_fp64" in str(device.extensions).split()
    )


def resolve_device_index(
    *,
    family: str,
    requested: str,
    count: int,
    precision: str,
    shape: tuple[int, ...],
) -> tuple[int, dict[str, Any]]:
    """Return the device index of one accelerator family and how it was chosen.

    An explicit ``device_id`` wins, then the family's environment variable. One
    device is used as before. With several devices the measured choice of
    :mod:`darsia.fluidflower.device_choice` picks the fastest device that can
    run the session, instead of silently taking index 0.
    """

    variable = DEVICE_ENVIRONMENT_VARIABLES[family]
    if requested:
        return int(requested), {"basis": "explicit", "source": "device_id"}
    if os.environ.get(variable, ""):
        return int(os.environ[variable]), {
            "basis": "explicit",
            "source": f"environment:{variable}",
        }
    if count == 1:
        return 0, {"basis": "single_device"}
    from .device_choice import choose_device

    choice = choose_device(
        family=family, precision=precision, shape=tuple(int(n) for n in shape)
    )
    return int(choice["device_id"]), choice


def cuda_driver_version(cp: Any) -> str:
    """Return the CUDA driver version that DarSIA records for a CUDA device."""

    return f"CUDA driver API {int(cp.cuda.runtime.driverGetVersion())}"


def _require_chosen_device(choice: dict[str, Any], name: str, driver: str) -> None:
    """Refuse a session whose device is not the one the measured choice named."""

    device = choice.get("device") or {}
    if choice.get("basis") == "measured" and (
        str(device.get("name")).strip() != name.strip()
        or str(device.get("driver")).strip() != driver.strip()
    ):
        raise RuntimeError(
            "The GPU device list changed between the measured choice and the "
            f"session: chose {device.get('name')!r}, found {name!r}"
        )


@dataclass(frozen=True)
class MassKernelConfiguration:
    """Immutable scalar and lookup-table inputs shared by every backend."""

    model: str
    transfer_shape: str
    acid_low: float
    acid_high: float
    co2_onset: float
    co2_high: float
    gas_onset: float
    gas_high: float
    max_gas_saturation: float
    lut_signal: np.ndarray
    lut_concentration: np.ndarray

    def validate(self) -> None:
        if self.model not in {"simple_flash", "titration"}:
            raise ValueError(f"Unsupported accelerated chemistry model: {self.model}")
        if self.transfer_shape not in {"linear", "smoothstep"}:
            raise ValueError(f"Unsupported transfer shape: {self.transfer_shape}")
        for low, high, label in (
            (self.acid_low, self.acid_high, "acid"),
            (self.co2_onset, self.co2_high, "CO2"),
            (self.gas_onset, self.gas_high, "gas"),
        ):
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(f"Invalid {label} transfer interval [{low}, {high}]")
        if not np.isfinite(self.max_gas_saturation) or not (
            0.0 <= self.max_gas_saturation <= 1.0
        ):
            raise ValueError("Maximum gas saturation must lie in [0, 1]")
        x = np.asarray(self.lut_signal, dtype=np.float64)
        y = np.asarray(self.lut_concentration, dtype=np.float64)
        if (
            x.ndim != 1
            or y.ndim != 1
            or x.size < 2
            or x.shape != y.shape
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0.0)
        ):
            raise ValueError(
                "Chemistry LUT must contain finite, increasing coordinates"
            )


@dataclass(frozen=True)
class MassFrameResult:
    """Host arrays returned by one CPU or accelerator frame evaluation."""

    concentration_aqueous: np.ndarray
    saturation_free_gas: np.ndarray
    co2_gate: np.ndarray
    mass_aqueous: np.ndarray
    mass_free_gas: np.ndarray
    mass_total: np.ndarray

    def fields(self) -> dict[str, np.ndarray]:
        return {
            "concentration_aqueous": self.concentration_aqueous,
            "saturation_free_gas": self.saturation_free_gas,
            "co2_gate": self.co2_gate,
            "mass_aqueous": self.mass_aqueous,
            "mass_free_gas": self.mass_free_gas,
            "mass_total": self.mass_total,
        }


def _dtype_for_backend(backend: str) -> np.dtype[Any]:
    if backend.endswith("-float32"):
        return np.dtype(np.float32)
    if backend.endswith("-float64"):
        return np.dtype(np.float64)
    raise ValueError(f"Backend does not encode a supported precision: {backend}")


def _transfer_numpy(
    values: np.ndarray,
    low: float,
    high: float,
    shape: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    scalar = dtype.type
    scaled = np.clip(
        (np.asarray(values, dtype=dtype) - scalar(low)) / scalar(high - low),
        scalar(0.0),
        scalar(1.0),
    )
    if shape == "linear":
        return scaled
    if shape == "smoothstep":
        return scaled * scaled * (scalar(3.0) - scalar(2.0) * scaled)
    raise ValueError(f"Unsupported transfer shape: {shape}")


class MassComputeSession(ABC):
    """One fixed-grid backend session used for a complete run."""

    backend: str
    dtype: np.dtype[Any]

    def __init__(
        self,
        *,
        backend: str,
        active: np.ndarray,
        voxel_volume_m3: np.ndarray,
        configuration: MassKernelConfiguration,
        device_id: str = "",
    ) -> None:
        configuration.validate()
        self.backend = backend
        self.dtype = _dtype_for_backend(backend)
        self.configuration = configuration
        self.device_id = str(device_id)
        self.active = np.ascontiguousarray(np.asarray(active, dtype=np.uint8))
        self.voxel_volume_m3 = np.ascontiguousarray(
            np.asarray(voxel_volume_m3, dtype=self.dtype)
        )
        if self.active.ndim != 2 or self.voxel_volume_m3.shape != self.active.shape:
            raise ValueError(
                "Accelerated active and voxel-volume fields must share one 2D grid"
            )
        if not np.all(np.isfinite(self.voxel_volume_m3)):
            raise ValueError("Accelerated voxel-volume field is non-finite")
        self.shape = self.active.shape
        self.size = int(self.active.size)

    def _dynamic_field(self, values: np.ndarray, label: str) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(values, dtype=self.dtype))
        if array.shape != self.shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must be a finite field on {self.shape}")
        return array

    @abstractmethod
    def compute_frame(
        self,
        *,
        acidification: np.ndarray,
        co2_probability: np.ndarray,
        gas_probability: np.ndarray,
        solubility_co2: np.ndarray,
        density_gaseous_co2: np.ndarray,
        diffusion_factor: float,
    ) -> MassFrameResult:
        """Evaluate one frame and return all scientific fields on the host."""

    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return backend and device metadata for the immutable result."""

    def close(self) -> None:
        """Release optional backend resources."""


class NumpyMassComputeSession(MassComputeSession):
    """Deterministic NumPy implementation for CPU-float32 and contract tests."""

    def compute_frame(
        self,
        *,
        acidification: np.ndarray,
        co2_probability: np.ndarray,
        gas_probability: np.ndarray,
        solubility_co2: np.ndarray,
        density_gaseous_co2: np.ndarray,
        diffusion_factor: float,
    ) -> MassFrameResult:
        dtype = self.dtype
        scalar = dtype.type
        config = self.configuration
        acid = self._dynamic_field(acidification, "acidification")
        co2 = self._dynamic_field(co2_probability, "CO2 probability")
        gas = self._dynamic_field(gas_probability, "gas probability")
        solubility = self._dynamic_field(solubility_co2, "CO2 solubility")
        density = self._dynamic_field(density_gaseous_co2, "gaseous CO2 density")
        acid_fraction = _transfer_numpy(
            acid, config.acid_low, config.acid_high, config.transfer_shape, dtype
        )
        if config.model == "titration":
            concentration = np.interp(
                acid_fraction,
                np.asarray(config.lut_signal, dtype=dtype),
                np.asarray(config.lut_concentration, dtype=dtype),
            ).astype(dtype, copy=False)
        else:
            concentration = acid_fraction
        co2_gate = _transfer_numpy(
            co2, config.co2_onset, config.co2_high, config.transfer_shape, dtype
        )
        concentration = np.clip(
            concentration * co2_gate * scalar(diffusion_factor),
            scalar(0.0),
            scalar(1.0),
        )
        saturation = np.clip(
            scalar(config.max_gas_saturation)
            * _transfer_numpy(
                gas,
                config.gas_onset,
                config.gas_high,
                config.transfer_shape,
                dtype,
            ),
            scalar(0.0),
            scalar(1.0),
        )
        active = self.active.astype(bool, copy=False)
        concentration = np.where(active, concentration, scalar(0.0))
        saturation = np.where(active, saturation, scalar(0.0))
        mass_gas = density * saturation * self.voxel_volume_m3 * scalar(1000.0)
        mass_aqueous = (
            solubility
            * concentration
            * np.maximum(scalar(1.0) - saturation, scalar(0.0))
            * self.voxel_volume_m3
            * scalar(1000.0)
        )
        return MassFrameResult(
            concentration_aqueous=np.asarray(concentration, dtype=dtype),
            saturation_free_gas=np.asarray(saturation, dtype=dtype),
            co2_gate=np.asarray(co2_gate, dtype=dtype),
            mass_aqueous=np.asarray(mass_aqueous, dtype=dtype),
            mass_free_gas=np.asarray(mass_gas, dtype=dtype),
            mass_total=np.asarray(mass_aqueous + mass_gas, dtype=dtype),
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "runtime": "numpy-cpu",
            "numpy": np.__version__,
            "device_id": "cpu",
            "dtype": self.dtype.name,
            "fused_phase_mass_kernel": False,
        }


CUDA_OPERATION = r"""
T acid_x = acidification;
T co2_x = co2_probability;
T gas_x = gas_probability;
T acid_fraction = (acid_x - acid_low) / (acid_high - acid_low);
acid_fraction = acid_fraction < (T)0 ? (T)0 : (acid_fraction > (T)1 ? (T)1 : acid_fraction);
if (smoothstep != 0) acid_fraction = acid_fraction * acid_fraction * ((T)3 - (T)2 * acid_fraction);
T concentration_x = acid_fraction;
if (titration != 0) {
    if (acid_fraction <= lut_signal[0]) {
        concentration_x = lut_concentration[0];
    } else if (acid_fraction >= lut_signal[lut_size - 1]) {
        concentration_x = lut_concentration[lut_size - 1];
    } else {
        int lo = 0;
        int hi = lut_size - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (lut_signal[mid] <= acid_fraction) lo = mid; else hi = mid;
        }
        T weight = (acid_fraction - lut_signal[lo]) / (lut_signal[hi] - lut_signal[lo]);
        concentration_x = lut_concentration[lo] + weight * (lut_concentration[hi] - lut_concentration[lo]);
    }
}
T gate_x = (co2_x - co2_onset) / (co2_high - co2_onset);
gate_x = gate_x < (T)0 ? (T)0 : (gate_x > (T)1 ? (T)1 : gate_x);
if (smoothstep != 0) gate_x = gate_x * gate_x * ((T)3 - (T)2 * gate_x);
T saturation_x = (gas_x - gas_onset) / (gas_high - gas_onset);
saturation_x = saturation_x < (T)0 ? (T)0 : (saturation_x > (T)1 ? (T)1 : saturation_x);
if (smoothstep != 0) saturation_x = saturation_x * saturation_x * ((T)3 - (T)2 * saturation_x);
saturation_x = max_gas_saturation * saturation_x;
concentration_x = concentration_x * gate_x * diffusion_factor;
concentration_x = concentration_x < (T)0 ? (T)0 : (concentration_x > (T)1 ? (T)1 : concentration_x);
if (active == 0) {
    concentration_x = (T)0;
    saturation_x = (T)0;
}
T volume_x = voxel_volume * (T)1000;
T gas_mass_x = density_gaseous_co2 * saturation_x * volume_x;
T aqueous_fraction = (T)1 - saturation_x;
aqueous_fraction = aqueous_fraction > (T)0 ? aqueous_fraction : (T)0;
T aqueous_mass_x = solubility_co2 * concentration_x * aqueous_fraction * volume_x;
concentration_aqueous = concentration_x;
saturation_free_gas = saturation_x;
co2_gate = gate_x;
mass_aqueous = aqueous_mass_x;
mass_free_gas = gas_mass_x;
mass_total = aqueous_mass_x + gas_mass_x;
"""


class CudaMassComputeSession(MassComputeSession):
    """One fused CuPy elementwise kernel with explicit float32/float64 inputs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CUDA mass analysis requires the optional CuPy dependency"
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError(
                "CUDA mass analysis requested, but no CUDA device exists"
            )
        count = int(cp.cuda.runtime.getDeviceCount())
        device_id, self.device_choice = resolve_device_index(
            family="cuda",
            requested=self.device_id,
            count=count,
            precision=self.dtype.name,
            shape=self.shape,
        )
        if not 0 <= device_id < count:
            raise ValueError(f"CUDA device index is out of range: {device_id}")
        self.cp = cp
        self.device_index = device_id
        self.device = cp.cuda.Device(device_id)
        self.device.use()
        _require_chosen_device(
            self.device_choice, self._device_name(), self._device_driver()
        )
        signature_type = "float32" if self.dtype == np.dtype(np.float32) else "float64"
        type_name = "float" if self.dtype == np.dtype(np.float32) else "double"
        input_signature = ", ".join(
            (
                f"{signature_type} acidification",
                f"{signature_type} co2_probability",
                f"{signature_type} gas_probability",
                "uint8 active",
                f"{signature_type} voxel_volume",
                f"{signature_type} solubility_co2",
                f"{signature_type} density_gaseous_co2",
                f"raw {signature_type} lut_signal",
                f"raw {signature_type} lut_concentration",
                "int32 lut_size",
                "int32 titration",
                "int32 smoothstep",
                f"{signature_type} acid_low",
                f"{signature_type} acid_high",
                f"{signature_type} co2_onset",
                f"{signature_type} co2_high",
                f"{signature_type} gas_onset",
                f"{signature_type} gas_high",
                f"{signature_type} max_gas_saturation",
                f"{signature_type} diffusion_factor",
            )
        )
        output_signature = ", ".join(
            f"{signature_type} {name}"
            for name in (
                "concentration_aqueous",
                "saturation_free_gas",
                "co2_gate",
                "mass_aqueous",
                "mass_free_gas",
                "mass_total",
            )
        )
        operation = CUDA_OPERATION.replace("T ", f"{type_name} ").replace(
            "(T)", f"({type_name})"
        )
        self.kernel = cp.ElementwiseKernel(
            input_signature,
            output_signature,
            operation,
            name=f"darsia_mass_{self.dtype.name}",
        )
        self.active_device = cp.asarray(self.active.ravel())
        self.voxel_device = cp.asarray(self.voxel_volume_m3.ravel())
        self.lut_signal_device = cp.asarray(
            np.asarray(self.configuration.lut_signal, dtype=self.dtype)
        )
        self.lut_concentration_device = cp.asarray(
            np.asarray(self.configuration.lut_concentration, dtype=self.dtype)
        )
        self.device.synchronize()

    def compute_frame(self, **values: Any) -> MassFrameResult:
        self.device.use()
        scalar = self.dtype.type
        config = self.configuration
        acid = self.cp.asarray(
            self._dynamic_field(values["acidification"], "acidification").ravel()
        )
        co2 = self.cp.asarray(
            self._dynamic_field(values["co2_probability"], "CO2 probability").ravel()
        )
        gas = self.cp.asarray(
            self._dynamic_field(values["gas_probability"], "gas probability").ravel()
        )
        solubility = self.cp.asarray(
            self._dynamic_field(values["solubility_co2"], "CO2 solubility").ravel()
        )
        density = self.cp.asarray(
            self._dynamic_field(
                values["density_gaseous_co2"], "gaseous CO2 density"
            ).ravel()
        )
        outputs = self.kernel(
            acid,
            co2,
            gas,
            self.active_device,
            self.voxel_device,
            solubility,
            density,
            self.lut_signal_device,
            self.lut_concentration_device,
            np.int32(self.lut_signal_device.size),
            np.int32(config.model == "titration"),
            np.int32(config.transfer_shape == "smoothstep"),
            scalar(config.acid_low),
            scalar(config.acid_high),
            scalar(config.co2_onset),
            scalar(config.co2_high),
            scalar(config.gas_onset),
            scalar(config.gas_high),
            scalar(config.max_gas_saturation),
            scalar(values["diffusion_factor"]),
        )
        self.device.synchronize()
        host = [self.cp.asnumpy(array).reshape(self.shape) for array in outputs]
        return MassFrameResult(*host)

    def _device_name(self) -> str:
        """Return the CUDA device name reported by the runtime."""

        properties = self.cp.cuda.runtime.getDeviceProperties(self.device_index)
        raw_name = properties.get("name", "CUDA device")
        return raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)

    def _device_driver(self) -> str:
        """Return the CUDA driver version of this session's device."""

        return cuda_driver_version(self.cp)

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "runtime": "cupy-elementwise",
            "cupy": self.cp.__version__,
            "device_id": str(self.device_index),
            "device_name": self._device_name(),
            "device_driver": self._device_driver(),
            "device_choice": dict(self.device_choice),
            "dtype": self.dtype.name,
            "fused_phase_mass_kernel": True,
        }

    def close(self) -> None:
        self.device.use()
        self.device.synchronize()
        self.cp.get_default_memory_pool().free_all_blocks()


OPENCL_SOURCE = r"""
{fp64_extension}
typedef {real_type} real;

inline real clamp01(real value) {{
    return value < (real)0 ? (real)0 : (value > (real)1 ? (real)1 : value);
}}

inline real transfer_value(real value, real low, real high, int smoothstep) {{
    real scaled = clamp01((value - low) / (high - low));
    return smoothstep ? scaled * scaled * ((real)3 - (real)2 * scaled) : scaled;
}}

inline real interpolate_lut(
    real value,
    __global const real *x,
    __global const real *y,
    int size
) {{
    if (value <= x[0]) return y[0];
    if (value >= x[size - 1]) return y[size - 1];
    int lo = 0;
    int hi = size - 1;
    while (hi - lo > 1) {{
        int mid = (lo + hi) / 2;
        if (x[mid] <= value) lo = mid; else hi = mid;
    }}
    real weight = (value - x[lo]) / (x[hi] - x[lo]);
    return y[lo] + weight * (y[hi] - y[lo]);
}}

__kernel void mass_kernel(
    __global const real *acidification,
    __global const real *co2_probability,
    __global const real *gas_probability,
    __global const uchar *active,
    __global const real *voxel_volume,
    __global const real *solubility_co2,
    __global const real *density_gaseous_co2,
    __global const real *lut_signal,
    __global const real *lut_concentration,
    int lut_size,
    int titration,
    int smoothstep,
    real acid_low,
    real acid_high,
    real co2_onset,
    real co2_high,
    real gas_onset,
    real gas_high,
    real max_gas_saturation,
    real diffusion_factor,
    __global real *concentration_aqueous,
    __global real *saturation_free_gas,
    __global real *co2_gate,
    __global real *mass_aqueous,
    __global real *mass_free_gas,
    __global real *mass_total
) {{
    int i = get_global_id(0);
    real acid_fraction = transfer_value(
        acidification[i], acid_low, acid_high, smoothstep
    );
    real concentration = titration
        ? interpolate_lut(acid_fraction, lut_signal, lut_concentration, lut_size)
        : acid_fraction;
    real gate = transfer_value(co2_probability[i], co2_onset, co2_high, smoothstep);
    real saturation = max_gas_saturation * transfer_value(
        gas_probability[i], gas_onset, gas_high, smoothstep
    );
    concentration = clamp01(concentration * gate * diffusion_factor);
    if (active[i] == 0) {{
        concentration = (real)0;
        saturation = (real)0;
    }}
    real volume = voxel_volume[i] * (real)1000;
    real gas_mass = density_gaseous_co2[i] * saturation * volume;
    real aqueous_mass = solubility_co2[i] * concentration
        * max((real)1 - saturation, (real)0) * volume;
    concentration_aqueous[i] = concentration;
    saturation_free_gas[i] = saturation;
    co2_gate[i] = gate;
    mass_aqueous[i] = aqueous_mass;
    mass_free_gas[i] = gas_mass;
    mass_total[i] = aqueous_mass + gas_mass;
}}
"""


_OPENCL_PROGRAM_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}
_OPENCL_PROGRAM_LOCK = threading.Lock()


class OpenCLMassComputeSession(MassComputeSession):
    """Fused PyOpenCL mass kernel with strict backend-wide precision."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            import pyopencl as cl
        except ImportError as exc:
            raise RuntimeError(
                "OpenCL mass analysis requires the optional PyOpenCL dependency"
            ) from exc
        devices = opencl_gpu_devices(cl)
        if not devices:
            raise RuntimeError(
                "OpenCL mass analysis requested, but no GPU device exists"
            )
        device_id, self.device_choice = resolve_device_index(
            family="opencl",
            requested=self.device_id,
            count=len(devices),
            precision=self.dtype.name,
            shape=self.shape,
        )
        if not 0 <= device_id < len(devices):
            raise ValueError(f"OpenCL device index is out of range: {device_id}")
        self.cl = cl
        self.device_index = device_id
        self.platform, self.device = devices[device_id]
        _require_chosen_device(
            self.device_choice, str(self.device.name), str(self.device.driver_version)
        )
        if self.dtype == np.dtype(np.float64) and not opencl_supports_float64(
            self.device
        ):
            raise RuntimeError(
                f"OpenCL device {self.device.name!r} does not support float64"
            )
        source = OPENCL_SOURCE.format(
            fp64_extension=(
                "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                if self.dtype == np.dtype(np.float64)
                else ""
            ),
            real_type="double" if self.dtype == np.dtype(np.float64) else "float",
        )
        cache_key = (
            str(self.platform.name).strip(),
            str(self.platform.version).strip(),
            str(self.device.name).strip(),
            str(self.device.vendor).strip(),
            str(self.device.driver_version).strip(),
            self.dtype.name,
        )
        with _OPENCL_PROGRAM_LOCK:
            cached = _OPENCL_PROGRAM_CACHE.get(cache_key)
            if cached is None:
                context = cl.Context([self.device])
                program = cl.Program(context, source).build()
                cached = (context, program)
                _OPENCL_PROGRAM_CACHE[cache_key] = cached
        self.context, self.program = cached
        self.queue = cl.CommandQueue(self.context)
        # Kernel argument state is mutable. Keep one instance per session even
        # though the compiled program and context are shared process-wide.
        self.kernel = cl.Kernel(self.program, "mass_kernel")
        flags = cl.mem_flags
        self.active_buffer = cl.Buffer(
            self.context,
            flags.READ_ONLY | flags.COPY_HOST_PTR,
            hostbuf=self.active.ravel(),
        )
        self.voxel_buffer = cl.Buffer(
            self.context,
            flags.READ_ONLY | flags.COPY_HOST_PTR,
            hostbuf=self.voxel_volume_m3.ravel(),
        )
        self.lut_signal = np.ascontiguousarray(
            np.asarray(self.configuration.lut_signal, dtype=self.dtype)
        )
        self.lut_concentration = np.ascontiguousarray(
            np.asarray(self.configuration.lut_concentration, dtype=self.dtype)
        )
        self.lut_signal_buffer = cl.Buffer(
            self.context,
            flags.READ_ONLY | flags.COPY_HOST_PTR,
            hostbuf=self.lut_signal,
        )
        self.lut_concentration_buffer = cl.Buffer(
            self.context,
            flags.READ_ONLY | flags.COPY_HOST_PTR,
            hostbuf=self.lut_concentration,
        )
        field_bytes = self.size * self.dtype.itemsize
        self.dynamic_buffers = [
            cl.Buffer(self.context, flags.READ_ONLY, field_bytes) for _ in range(5)
        ]
        self.output_buffers = [
            cl.Buffer(self.context, flags.WRITE_ONLY, field_bytes) for _ in range(6)
        ]

    def compute_frame(self, **values: Any) -> MassFrameResult:
        scalar = self.dtype.type
        config = self.configuration
        host_inputs = [
            self._dynamic_field(values["acidification"], "acidification").ravel(),
            self._dynamic_field(values["co2_probability"], "CO2 probability").ravel(),
            self._dynamic_field(values["gas_probability"], "gas probability").ravel(),
            self._dynamic_field(values["solubility_co2"], "CO2 solubility").ravel(),
            self._dynamic_field(
                values["density_gaseous_co2"], "gaseous CO2 density"
            ).ravel(),
        ]
        copy_events = [
            self.cl.enqueue_copy(self.queue, buffer, host)
            for buffer, host in zip(self.dynamic_buffers, host_inputs, strict=True)
        ]
        kernel_event = self.kernel(
            self.queue,
            (self.size,),
            None,
            self.dynamic_buffers[0],
            self.dynamic_buffers[1],
            self.dynamic_buffers[2],
            self.active_buffer,
            self.voxel_buffer,
            self.dynamic_buffers[3],
            self.dynamic_buffers[4],
            self.lut_signal_buffer,
            self.lut_concentration_buffer,
            np.int32(self.lut_signal.size),
            np.int32(config.model == "titration"),
            np.int32(config.transfer_shape == "smoothstep"),
            scalar(config.acid_low),
            scalar(config.acid_high),
            scalar(config.co2_onset),
            scalar(config.co2_high),
            scalar(config.gas_onset),
            scalar(config.gas_high),
            scalar(config.max_gas_saturation),
            scalar(values["diffusion_factor"]),
            *self.output_buffers,
            wait_for=copy_events,
        )
        host_outputs = [np.empty(self.size, dtype=self.dtype) for _ in range(6)]
        events = [
            self.cl.enqueue_copy(
                self.queue, host, buffer, wait_for=[kernel_event], is_blocking=False
            )
            for host, buffer in zip(host_outputs, self.output_buffers, strict=True)
        ]
        self.cl.wait_for_events(events)
        return MassFrameResult(
            *(np.asarray(array).reshape(self.shape) for array in host_outputs)
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "runtime": str(self.platform.version),
            "pyopencl": getattr(self.cl, "VERSION_TEXT", "unknown"),
            "device_id": str(self.device_index),
            "device_name": str(self.device.name).strip(),
            "device_driver": str(self.device.driver_version),
            "device_choice": dict(self.device_choice),
            "dtype": self.dtype.name,
            "fused_phase_mass_kernel": True,
        }

    def close(self) -> None:
        self.queue.finish()


class MetalMassComputeSession(MassComputeSession):
    """Vectorized PyTorch MPS implementation for Apple Silicon."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.dtype != np.dtype(np.float32):
            raise ValueError("Metal mass analysis currently requires float32")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Metal mass analysis requires the optional PyTorch dependency"
            ) from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError("Metal mass analysis requested, but MPS is unavailable")
        self.torch = torch
        self.device = torch.device("mps")
        self.torch_dtype = torch.float32
        self.active_device = torch.as_tensor(
            self.active.astype(bool), dtype=torch.bool, device=self.device
        )
        self.voxel_device = torch.as_tensor(
            self.voxel_volume_m3, dtype=self.torch_dtype, device=self.device
        )
        self.lut_signal_device = torch.as_tensor(
            np.asarray(self.configuration.lut_signal, dtype=np.float32),
            dtype=self.torch_dtype,
            device=self.device,
        )
        self.lut_concentration_device = torch.as_tensor(
            np.asarray(self.configuration.lut_concentration, dtype=np.float32),
            dtype=self.torch_dtype,
            device=self.device,
        )
        torch.mps.synchronize()

    def _tensor(self, values: np.ndarray, label: str) -> Any:
        return self.torch.as_tensor(
            self._dynamic_field(values, label),
            dtype=self.torch_dtype,
            device=self.device,
        )

    def _transfer(self, values: Any, low: float, high: float) -> Any:
        result = self.torch.clamp((values - low) / (high - low), 0.0, 1.0)
        if self.configuration.transfer_shape == "smoothstep":
            result = result * result * (3.0 - 2.0 * result)
        return result

    def compute_frame(self, **values: Any) -> MassFrameResult:
        torch = self.torch
        config = self.configuration
        with torch.inference_mode():
            acid = self._tensor(values["acidification"], "acidification")
            co2 = self._tensor(values["co2_probability"], "CO2 probability")
            gas = self._tensor(values["gas_probability"], "gas probability")
            solubility = self._tensor(values["solubility_co2"], "CO2 solubility")
            density = self._tensor(values["density_gaseous_co2"], "gaseous CO2 density")
            acid_fraction = self._transfer(acid, config.acid_low, config.acid_high)
            if config.model == "titration":
                indices = torch.searchsorted(
                    self.lut_signal_device, acid_fraction.contiguous()
                ).clamp(1, self.lut_signal_device.numel() - 1)
                lower = indices - 1
                x0 = self.lut_signal_device[lower]
                x1 = self.lut_signal_device[indices]
                y0 = self.lut_concentration_device[lower]
                y1 = self.lut_concentration_device[indices]
                concentration = y0 + (acid_fraction - x0) * (y1 - y0) / (x1 - x0)
            else:
                concentration = acid_fraction
            gate = self._transfer(co2, config.co2_onset, config.co2_high)
            saturation = config.max_gas_saturation * self._transfer(
                gas, config.gas_onset, config.gas_high
            )
            concentration = torch.clamp(
                concentration * gate * float(values["diffusion_factor"]), 0.0, 1.0
            )
            concentration = torch.where(
                self.active_device, concentration, torch.zeros_like(concentration)
            )
            saturation = torch.where(
                self.active_device, saturation, torch.zeros_like(saturation)
            )
            volume = self.voxel_device * 1000.0
            mass_gas = density * saturation * volume
            mass_aqueous = (
                solubility
                * concentration
                * torch.clamp(1.0 - saturation, min=0.0)
                * volume
            )
            outputs = (
                concentration,
                saturation,
                gate,
                mass_aqueous,
                mass_gas,
                mass_aqueous + mass_gas,
            )
            torch.mps.synchronize()
            host = [tensor.detach().cpu().numpy() for tensor in outputs]
        return MassFrameResult(*host)

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "runtime": "pytorch-mps",
            "torch": self.torch.__version__,
            "device_id": "0",
            "device_name": "Apple Silicon GPU",
            "dtype": self.dtype.name,
            "fused_phase_mass_kernel": False,
        }

    def close(self) -> None:
        self.torch.mps.synchronize()
        self.torch.mps.empty_cache()


def create_mass_compute_session(
    *,
    backend: str,
    active: np.ndarray,
    voxel_volume_m3: np.ndarray,
    configuration: MassKernelConfiguration,
    device_id: str = "",
) -> MassComputeSession:
    """Create a strict non-reference session without silent backend fallback."""

    kwargs = {
        "backend": backend,
        "active": active,
        "voxel_volume_m3": voxel_volume_m3,
        "configuration": configuration,
        "device_id": device_id,
    }
    if backend == "cpu-float32":
        return NumpyMassComputeSession(**kwargs)
    if backend.startswith("cuda-"):
        return CudaMassComputeSession(**kwargs)
    if backend.startswith("opencl-"):
        return OpenCLMassComputeSession(**kwargs)
    if backend == "metal-float32":
        return MetalMassComputeSession(**kwargs)
    raise ValueError(f"No accelerated mass session exists for {backend}")


def compare_mass_frame_results(
    candidate: MassFrameResult,
    reference: MassFrameResult,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Return field-wise parity diagnostics against CPU float64."""

    if absolute_tolerance <= 0.0 or relative_tolerance <= 0.0:
        raise ValueError("Parity tolerances must be positive")
    fields: dict[str, dict[str, Any]] = {}
    passed = True
    for name, reference_values in reference.fields().items():
        candidate_values = np.asarray(candidate.fields()[name], dtype=np.float64)
        reference_array = np.asarray(reference_values, dtype=np.float64)
        if candidate_values.shape != reference_array.shape:
            fields[name] = {
                "passed": False,
                "reason": "shape_mismatch",
                "candidate_shape": list(candidate_values.shape),
                "reference_shape": list(reference_array.shape),
            }
            passed = False
            continue
        difference = np.abs(candidate_values - reference_array)
        denominator = np.maximum(np.abs(reference_array), absolute_tolerance)
        field_passed = bool(
            np.all(np.isfinite(candidate_values))
            and np.allclose(
                candidate_values,
                reference_array,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        )
        fields[name] = {
            "passed": field_passed,
            "maximum_absolute_error": float(np.max(difference, initial=0.0)),
            "maximum_relative_error": float(
                np.max(difference / denominator, initial=0.0)
            ),
            "normalized_mean_absolute_error": float(
                np.sum(difference, dtype=np.float64)
                / max(
                    np.sum(np.abs(reference_array), dtype=np.float64),
                    absolute_tolerance,
                )
            ),
        }
        passed = passed and field_passed
    return {
        "passed": passed,
        "reference_backend": "cpu-float64",
        "candidate_backend": "accelerator",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "fields": fields,
    }


__all__ = [
    "ACCELERATED_MASS_BACKENDS",
    "DEVICE_ENVIRONMENT_VARIABLES",
    "MassComputeSession",
    "MassFrameResult",
    "MassKernelConfiguration",
    "NumpyMassComputeSession",
    "NONREFERENCE_MASS_BACKENDS",
    "compare_mass_frame_results",
    "create_mass_compute_session",
    "cuda_driver_version",
    "opencl_gpu_devices",
    "opencl_supports_float64",
    "resolve_device_index",
]
