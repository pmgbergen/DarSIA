"""Optional, bounded OpenCL execution of the original selected N1 fields.

Geometry, colour interpretation, frozen parameters and float64 reductions stay
with their existing providers. A run keeps the CPU result of its first call;
only a faster device with two bit-identical whole-call replays is used later.
This local decision does not replace the complete run's acceptance comparison.
"""
from __future__ import annotations

import time
from threading import Lock

import numpy as np


FORMAT = "darsia-opencl-n1-v1"
TILE_PIXELS = 262144
FIELDS = ("aqueous", "gas", "concentration", "mass_total", "mass_aqueous", "mass_gas")


class _ProofMismatch(ArithmeticError):
    """A released common proof must fail rather than conceal changed pixels."""


from darsia_cache.compute_control import optional_n1_configuration as configuration


SOURCE = r"""
FP64
#pragma OPENCL FP_CONTRACT OFF
typedef REAL real;
inline real clip01(real x) {
    return x < (real)0 ? (real)0 : (x > (real)1 ? (real)1 : x);
}
__kernel void selected_n1(
    __global const real *prepared, __global const real *swi,
    __global const real *sol, __global const real *rho,
    __global const uchar *aq_known, __global const uchar *gas_known,
    real gain, real factor,
    __global real *aqueous, __global real *gas, __global real *concentration,
    __global real *total, __global real *aqueous_mass, __global real *gas_mass) {
    int i = get_global_id(0);
    volatile real scaled = prepared[i] * gain;
    volatile real fraction = clip01(scaled) * factor;
    volatile real aq = aq_known[i] ? fraction : (real)0;
    volatile real g = gas_known[i] ? (real)1 - swi[i] : (real)0;
    if (gas_known[i]) aq = (real)1;
    volatile real raw_g = rho[i] * g;
    volatile real dissolved = sol[i] * aq;
    volatile real remaining = (real)1 - g;
    volatile real raw_aq = dissolved * (remaining < (real)0 ? (real)0 : remaining);
    aqueous[i] = aq;
    gas[i] = g;
    concentration[i] = aq * sol[i];
    total[i] = raw_g + raw_aq;
    aqueous_mass[i] = raw_aq;
    gas_mass[i] = raw_g;
}
"""


def _devices(precision, requested):
    """Enumerate devices from the optional runtime only after explicit selection."""
    from ..mass_acceleration import opencl_gpu_devices
    import pyopencl as cl

    pairs = opencl_gpu_devices(cl)
    return [(cl, index, platform, device) for index, (platform, device) in enumerate(pairs)
            if requested is None or requested == index]


def _fp_configuration(entry, precision):
    """Require the IEEE behavior used by the unchanged NumPy operation."""
    cl, _, _, device = entry
    bits = int(getattr(device, "double_fp_config" if precision == "float64" else "single_fp_config"))
    required = {name: bool(bits & int(getattr(cl.device_fp_config, flag)))
                for name, flag in (("round_to_nearest", "ROUND_TO_NEAREST"),
                                   ("denormals", "DENORM"), ("inf_nan", "INF_NAN"))}
    return dict(precision=precision, bits=bits, **required), all(required.values())


def device_identity(entry):
    """The observed device/driver identity bound by the common proof profile."""
    _, index, platform, device = entry
    return dict(device_id=index, platform=str(platform.name).strip(),
        platform_version=str(platform.version).strip(), name=str(device.name).strip(),
        driver=str(device.driver_version).strip(), global_memory_bytes=int(device.global_mem_size))


class _Session:
    """One device's fixed-size tile buffers; no full-frame device allocation."""

    def __init__(self, entry, precision, size):
        """Create the selected context, queue and uncached program under its resource scope."""
        cl, index, platform, device = entry
        self.cl, self.dtype = cl, np.dtype(precision)
        self.context = self.queue = self.program = self.kernel = None
        self.inputs, self.outputs = [], []
        self.capacity = min(size, TILE_PIXELS)
        field_bytes = self.capacity * self.dtype.itemsize
        self.buffer_bytes = field_bytes * 10 + self.capacity * 2
        if (field_bytes > int(device.max_mem_alloc_size)
                or self.buffer_bytes > min(64 * 1024**2, int(device.global_mem_size) // 4)):
            raise RuntimeError("Bounded N1 tile exceeds the device allocation allowance")
        self.device = device_identity(entry)
        try:
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context)
            source = SOURCE.replace("FP64", "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                                    if precision == "float64" else "")
            source = source.replace("REAL", "double" if precision == "float64" else "float")
            self.program = cl.Program(self.context, source).build(options=["-cl-opt-disable"], cache_dir=False)
            self.kernel = cl.Kernel(self.program, "selected_n1")
            for size_bytes in [field_bytes] * 4 + [self.capacity] * 2:
                self.inputs.append(cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size_bytes))
            for _ in FIELDS:
                self.outputs.append(cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, field_bytes))
        except Exception:
            self.close()
            raise

    def tiles(self, values, gain, factor):
        """Upload, execute and download each tile, including all transfer costs."""
        size = values[0].size
        for start in range(0, size, self.capacity):
            count = min(self.capacity, size - start)
            host = [np.ascontiguousarray(value.flat[start:start + count]) for value in values]
            for buffer, value in zip(self.inputs, host, strict=True):
                self.cl.enqueue_copy(self.queue, buffer, value, is_blocking=True)
            self.kernel(self.queue, (count,), None, *self.inputs,
                        self.dtype.type(gain), self.dtype.type(factor), *self.outputs)
            result = {}
            for name, buffer in zip(FIELDS, self.outputs, strict=True):
                array = np.empty(count, dtype=self.dtype)
                self.cl.enqueue_copy(self.queue, array, buffer, is_blocking=True)
                result[name] = array
            yield start, result
            del host, result, array  # Retire tile staging before allocating the next tile.

    def close(self):
        """Release the current explicit application buffers and runtime handles."""
        try:
            if self.queue is not None:
                self.queue.finish()
        finally:
            # Release this session's own buffers even after a queue failure.
            for buffer in (*self.inputs, *self.outputs):
                try:
                    buffer.release()
                except Exception:
                    pass
            self.inputs, self.outputs = [], []
            self.kernel = self.program = self.queue = self.context = None


class OptionalN1:
    """First-call exactness and timing decision, followed by a fixed device."""

    def __init__(self, precision, control, *, qualified_devices=None, verify_every_call=False):
        """Create one locked optional backend with the declared precision and guard rules."""
        from copy import deepcopy
        self.precision, self.control = precision, dict(control)
        self.qualified_devices = deepcopy(qualified_devices)
        self.verify_every_call = verify_every_call
        self._lock = Lock()
        self.session = None
        self.probed = False
        self.record = dict(format=FORMAT, control=dict(control), used=False,
            admitted_devices=deepcopy(qualified_devices),
            selected_device=None, candidates=[], cpu_reference_ns=None,
            initialization_ns=0, probe_ns=0, calls=0, accelerated_calls=0, accelerated_ns=0,
            last_backend="cpu",
            verification_required=verify_every_call, cpu_comparison_calls=0,
            cpu_comparison_ns=0, verification_ns=0, comparison_failures=0,
            probe_verified_calls=0, probe_verified_replays=0,
            byte_verified_accelerated_calls=0, guarded_accelerated_calls=0, nonfinite_cpu_calls=0,
            peak_proof_duplicate_host_bytes=0, peak_probe_duplicate_host_bytes=0,
            peak_operand_cast_host_bytes=0, tile_host_staging_bound_bytes=0,
            application_device_buffer_bytes=0, peak_application_device_buffer_bytes=0,
            device_allocation_failure_peak_unknown=False, peak_process_memory_bytes=None,
            peak_process_memory_scope="unavailable",
            acceptance="Whole-run comparison remains required")

    def _values(self, prepared, kwargs):
        # Match the CPU's declared operand conversion and broadcasting.
        """Convert each original operand to the declared working precision."""
        values = [prepared]
        cast_bytes = 0
        for key in ("swi", "solubility", "gas_density"):
            original = np.asarray(kwargs[key])
            working = np.asarray(original, dtype=prepared.dtype)
            if not np.shares_memory(original, working):
                cast_bytes += working.nbytes
            values.append(np.broadcast_to(working, prepared.shape))
        values += [np.broadcast_to(kwargs[key], prepared.shape)
                   for key in ("aqueous_known", "gas_known")]
        self.record["peak_operand_cast_host_bytes"] = max(self.record["peak_operand_cast_host_bytes"], cast_bytes)
        return values

    @staticmethod
    def _identical(reference, result):
        """Compare output fields byte for byte without changing their scalar arithmetic."""
        for name in FIELDS:
            left, right = reference[name], result[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            chunks = np.nditer([left, right], flags=["external_loop", "buffered", "zerosize_ok"],
                               op_flags=[["readonly"], ["readonly"]], order="C", buffersize=65536)
            if any(a.tobytes() != b.tobytes() for a, b in chunks):
                return False
        return True

    @staticmethod
    def _finite(prepared, kwargs):
        # Check converted operands in bounded chunks, before any kernel call.
        """Check every converted floating operand before a kernel call."""
        for value in (prepared, *(kwargs[key] for key in ("swi", "solubility", "gas_density")),
                      kwargs["gain"], kwargs["factor"]):
            chunks = np.nditer(np.asarray(value), flags=["external_loop", "buffered", "zerosize_ok"],
                               op_flags=["readonly"], op_dtypes=[prepared.dtype],
                               casting="unsafe", buffersize=65536)
            if any(not np.isfinite(chunk).all() for chunk in chunks):
                return False
        return True

    def _operation(self, session, prepared, kwargs):
        """The identical complete GPU operation timed during probing and reuse."""
        if not self._finite(prepared, kwargs):
            return None
        values = self._values(prepared, kwargs)
        outputs = {name: np.empty_like(prepared) for name in FIELDS}
        for start, result in session.tiles(values, kwargs["gain"], kwargs["factor"]):
            for name, array in result.items():
                outputs[name].flat[start:start + len(array)] = array
            del result, array
        return outputs

    def _allocate(self, entry, size):
        """Allocate one bounded session and record initialization and buffers."""
        begin = time.perf_counter_ns()
        try:
            session = _Session(entry, self.precision, size)
        except Exception:
            self.record["device_allocation_failure_peak_unknown"] = True
            raise
        finally:
            self.record["initialization_ns"] += time.perf_counter_ns() - begin
        self.record["peak_application_device_buffer_bytes"] = max(
            self.record["peak_application_device_buffer_bytes"], session.buffer_bytes)
        self.record["tile_host_staging_bound_bytes"] = max(
            self.record["tile_host_staging_bound_bytes"], session.buffer_bytes)
        return session

    def _probe(self, prepared, kwargs, reference, cpu_ns):
        """Accept the selected implementation only after its original CPU comparison."""
        begin = time.perf_counter_ns()
        try:
            entries = _devices(self.precision, self.control["device_id"])
        except Exception as error:
            self.record["fallback"] = type(error).__name__
            self.record["initialization_ns"] += time.perf_counter_ns() - begin
            return
        self.record["initialization_ns"] += time.perf_counter_ns() - begin
        best, best_entry = cpu_ns, None
        for entry in entries:
            session = None
            candidate = dict(device_id=entry[1], identical=False, replay_ns=[])
            self.record["candidates"].append(candidate)
            try:
                observed = device_identity(entry)
                candidate["device"] = observed
                if self.qualified_devices is not None and observed not in self.qualified_devices:
                    candidate["fallback"] = "device_not_in_qualified_profile"
                    continue
                candidate["fp_configuration"], supported = _fp_configuration(entry, self.precision)
                if not supported:
                    candidate["fallback"] = "required_floating_point_behavior_unavailable"
                    continue
                session = self._allocate(entry, prepared.size)
                candidate.update(device=dict(session.device),
                    application_device_buffer_bytes=session.buffer_bytes,
                    operation_ns=[], verification_ns=[])
                for _ in range(2):
                    begin = time.perf_counter_ns()
                    outputs = self._operation(session, prepared, kwargs)
                    operation_ns = time.perf_counter_ns() - begin
                    self.record["peak_probe_duplicate_host_bytes"] = max(
                        self.record["peak_probe_duplicate_host_bytes"], sum(value.nbytes for value in outputs.values()))
                    verification_begin = time.perf_counter_ns()
                    same = self._identical(reference, outputs)
                    verification_ns = time.perf_counter_ns() - verification_begin
                    del outputs
                    if not same:
                        self.record["comparison_failures"] += 1
                        if self.verify_every_call:
                            raise _ProofMismatch("N1 OpenCL pixels differ from CPU bytes")
                        raise ArithmeticError("N1 OpenCL pixels differ from CPU bytes")
                    candidate["replay_ns"].append(time.perf_counter_ns() - begin)
                    candidate["operation_ns"].append(operation_ns)
                    candidate["verification_ns"].append(verification_ns)
                    self.record["probe_verified_replays"] += 1
                candidate["identical"] = True
                self.record["probe_verified_calls"] = 1
                # Matched allocation, casts, finite guard, transfers and scatter.
                # Verification and initialization are separate proof overheads.
                candidate_ns = max(candidate["operation_ns"])
                if candidate_ns < best:
                    best_entry, best = entry, candidate_ns
            except _ProofMismatch:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass  # Cleanup cannot replace the original pixel mismatch.
                    session = None
                raise
            except Exception as error:
                candidate["fallback"] = type(error).__name__
            finally:
                if session is not None:
                    session.close()
        # No two candidate contexts/buffer sets coexist. Rebuild only the winner.
        if best_entry is not None:
            self.session = self._allocate(best_entry, prepared.size)
        if self.session is not None:
            self.record.update(selected_device=dict(self.session.device),
                application_device_buffer_bytes=self.session.buffer_bytes)
        else:
            self.record["fallback"] = "no_faster_identical_device" if entries else "no_usable_device"

    def compute(self, cpu, prepared, **kwargs):
        """Run one guarded call and retain distinct proof and fallback counters."""
        with self._lock:
            return self._compute(cpu, prepared, **kwargs)

    def _compute(self, cpu, prepared, **kwargs):
        """Keep the CPU probe result and guard or verify every later call."""
        self.record["calls"] += 1
        self.record["last_backend"] = "cpu"
        if not self.probed and not self._finite(prepared, kwargs):
            self.record["nonfinite_cpu_calls"] += 1
            return cpu(prepared, **kwargs)
        if not self.probed:
            begin = time.perf_counter_ns()
            reference = cpu(prepared, **kwargs)
            elapsed = time.perf_counter_ns() - begin
            self.record["cpu_reference_ns"] = elapsed
            self.probed = True
            if any(value.shape != prepared.shape or value.dtype != prepared.dtype
                   for value in reference.values()):
                self.record["fallback"] = "unsupported_broadcast_output"
            elif prepared.size:
                probe_begin = time.perf_counter_ns()
                try:
                    self._probe(prepared, kwargs, reference, elapsed)
                except _ProofMismatch:
                    self._close()
                    raise
                except Exception as error:
                    self.record["fallback"] = type(error).__name__
                    self._close()
                finally:
                    self.record["probe_ns"] += time.perf_counter_ns() - probe_begin
            # Keep the original CPU arrays; record complete probe output duplicates.
            return reference
        if self.session is None:
            return cpu(prepared, **kwargs)
        begin = time.perf_counter_ns()
        try:
            outputs = self._operation(self.session, prepared, kwargs)
        except Exception as error:
            self.record["fallback"] = type(error).__name__
            self._close()
            return cpu(prepared, **kwargs)
        if outputs is None:
            self.record["nonfinite_cpu_calls"] += 1
            return cpu(prepared, **kwargs)
        self.record["accelerated_calls"] += 1
        self.record["accelerated_ns"] += time.perf_counter_ns() - begin
        self.record["used"] = True
        self.record["last_backend"] = "opencl"
        if self.verify_every_call:
            begin = time.perf_counter_ns()
            reference = cpu(prepared, **kwargs)
            self.record["cpu_comparison_ns"] += time.perf_counter_ns() - begin
            self.record["cpu_comparison_calls"] += 1
            self.record["peak_proof_duplicate_host_bytes"] = max(
                self.record["peak_proof_duplicate_host_bytes"], sum(value.nbytes for value in reference.values()))
            begin = time.perf_counter_ns()
            same = self._identical(reference, outputs)
            self.record["verification_ns"] += time.perf_counter_ns() - begin
            del reference
            if not same:
                self.record["comparison_failures"] += 1
                self._close()
                raise _ProofMismatch("Common N1 OpenCL proof differs from CPU bytes")
            self.record["byte_verified_accelerated_calls"] += 1
        else:
            self.record["guarded_accelerated_calls"] += 1
        return outputs

    def facts(self):
        """Report the observed backend, comparison counters and application allocations."""
        with self._lock:
            return self._facts()

    def _facts(self):
        """Copy execution facts with an explicitly scoped process peak."""
        from copy import deepcopy
        import os
        record = deepcopy(self.record)
        if os.name == "nt":
            try:
                import psutil
                peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
                if peak is not None:
                    record["peak_process_memory_bytes"] = int(peak)
                    record["peak_process_memory_scope"] = "current-process-kernel-lifetime-peak-working-set-at-facts-snapshot"
            except Exception:
                pass
        return record

    def close(self):
        """Close the optional session while preserving locked lifecycle transitions."""
        with self._lock:
            self._close()

    def _close(self):
        """Retire the optional session without preventing CPU fallback."""
        if self.session is not None:
            session, self.session = self.session, None
            try:
                session.close()
            except Exception:
                pass  # A failed optional device cannot prevent the CPU path.
