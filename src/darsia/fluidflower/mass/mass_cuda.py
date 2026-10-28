"""UNQUALIFIED CUDA pixel kernels for the original retained native method.

Scratch proposal only: no production permission, implicit fallback, model fit,
input decoder, labels inference, reduction or publication. Host inputs are
verified/decoded by the original reader. Tables and fit parameters stay frozen
on the CPU; local pixel operands use one declared precision. CUDA results must
pass addendum8v2 against CPU at that same precision before use.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
import numpy as np


# Independently expressed OpenCV INTER_LINEAR semantics: fractional coordinate
# table has 32 entries, nearest-even coordinate rounding, constant-zero border.
# Geometry uses the original CPU float64 inverse. No fast-math/FMA contraction.
_WARP = r'''
extern "C" __global__ void warp_tab32(
 const REAL* s, REAL* out, const double* m,
 int sh, int sw, int dh, int dw, int nc) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;
 if(q>=dh*dw) return;
 int y=q/dw,x=q%dw;
 double den=m[6]*x+(m[7]*y+m[8]);
 double scale=den!=0.0 ? 32.0/den : 0.0;
 double fx=(m[0]*x+(m[1]*y+m[2]))*scale;
 double fy=(m[3]*x+(m[4]*y+m[5]))*scale;
 fx=fmin(2147483647.0,fmax(-2147483648.0,fx));
 fy=fmin(2147483647.0,fmax(-2147483648.0,fy));
 int X=__double2int_rn(fx),Y=__double2int_rn(fy);
 int ix=max(-32768,min(32767,X>>5));
 int iy=max(-32768,min(32767,Y>>5));
 float ax=(X&31)*(1.0f/32.0f),ay=(Y&31)*(1.0f/32.0f);
 REAL w0=(REAL)((1.0f-ay)*(1.0f-ax));
 REAL w1=(REAL)((1.0f-ay)*ax);
 REAL w2=(REAL)(ay*(1.0f-ax));
 REAL w3=(REAL)(ay*ax);
 for(int c=0;c<nc;c++) {
  REAL v0=0,v1=0,v2=0,v3=0;
  if(iy>=0&&iy<sh&&ix>=0&&ix<sw) v0=s[(iy*sw+ix)*nc+c];
  if(iy>=0&&iy<sh&&ix+1>=0&&ix+1<sw) v1=s[(iy*sw+ix+1)*nc+c];
  if(iy+1>=0&&iy+1<sh&&ix>=0&&ix<sw) v2=s[((iy+1)*sw+ix)*nc+c];
  if(iy+1>=0&&iy+1<sh&&ix+1>=0&&ix+1<sw) v3=s[((iy+1)*sw+ix+1)*nc+c];
  out[q*nc+c]=((v0*w0+v1*w1)+v2*w2)+v3*w3;
 }
}
'''


class _PixelMath:
    """Shared mathematical expression for CUDA and tiny CPU-only unit fixtures."""
    def __init__(self, xp, precision):
        if precision not in {"float32", "float64"}:
            raise ValueError("One explicit pixel precision required")
        self.xp, self.precision = xp, precision
        self.dtype = getattr(xp, precision)

    def array(self, value):
        return self.xp.asarray(value, dtype=self.dtype)

    def decode_log_rgb(self, storage):
        # Conversion precedes exp: original storage remains untouched float32.
        return self.xp.exp(self.array(storage)) / self.dtype(65535)

    def reference_normalized(self, current, baseline, clear, support):
        xp = self.xp
        current, baseline, clear = map(self.array, (current, baseline, clear))
        support = xp.asarray(support, dtype=xp.bool_)
        if current.shape != baseline.shape or current.shape != clear.shape or current.shape[:-1] != support.shape:
            raise ValueError("Native RGB/support grid differs")
        if current.ndim != 3 or current.shape[-1] != 3:
            raise ValueError("Native HWC RGB required")
        if bool(xp.any(support & (~xp.isfinite(clear).all(-1) | (clear <= 0).any(-1)))):
            raise ValueError("Invalid measured clear-water support")
        a, b = xp.zeros_like(current), xp.zeros_like(baseline)
        # CuPy 14.1.1 ufuncs do not accept NumPy's `where` kwarg. Evaluate
        # only the same original measured support, keeping other pixels zero.
        a[support] = current[support]/clear[support]
        b[support] = baseline[support]/clear[support]
        if bool(xp.any(~xp.isfinite(a[support]))) or bool(xp.any(~xp.isfinite(b[support]))):
            raise ValueError("Nonfinite normalized scientific signal")
        return a, b, support

    def relative_rgb(self, current, baseline, *, option="positive"):
        xp = self.xp
        current, baseline = self.array(current), self.array(baseline)
        if current.shape != baseline.shape:
            raise ValueError("Current/baseline grid differs")
        if option == "positive": return xp.maximum(current-baseline, self.dtype(0))
        if option == "negative": return xp.maximum(baseline-current, self.dtype(0))
        if option == "plain": return current-baseline
        if option == "absolute": return xp.abs(current-baseline)
        raise ValueError("Unknown original background subtraction option")

    def _path_table(self, supports, distances):
        # Validation/copy of small frozen metadata; original arrays never edited.
        s = np.asarray(supports)
        d = np.asarray(distances)
        if s.ndim != 2 or s.shape[1] != 3 or d.shape != (len(s),) or len(s) < 2:
            raise ValueError("RGB path/support shape differs")
        if not np.isfinite(s).all() or not np.isfinite(d).all() or np.any(np.diff(d) <= 0):
            raise ValueError("Finite strictly increasing original path required")
        if np.any(np.sum(np.diff(s, axis=0)**2, axis=1) <= 0):
            raise ValueError("Original colour path contains a zero-length segment")
        sw, dw = self.array(s), self.array(d)
        if bool(self.xp.any(self.xp.diff(dw) <= 0)) or bool(self.xp.any(self.xp.sum(self.xp.diff(sw, axis=0)**2, axis=1) <= 0)):
            raise ValueError("Frozen path collapses in selected pixel precision")
        return sw, dw

    def fit_color_path(self, colors, supports, distances, *, tile_pixels=262144):
        """Original projection/clip/L1 nearest-first-tie algorithm, bounded tile."""
        if type(tile_pixels) is not int or tile_pixels <= 0:
            raise ValueError("Positive bounded colour tile required")
        xp = self.xp
        colors = self.array(colors)
        if colors.ndim < 1 or colors.shape[-1] != 3:
            raise ValueError("RGB path samples required")
        supports, distances = self._path_table(supports, distances)
        flat = colors.reshape(-1, 3)
        output = xp.empty(len(flat), dtype=self.dtype)
        for start in range(0, len(flat), tile_pixels):
            c = flat[start:start+tile_pixels]
            best = xp.zeros(len(c), dtype=self.dtype)
            closest = xp.full(len(c), xp.inf, dtype=self.dtype)
            for segment in range(len(supports)-1):
                delta = supports[segment+1]-supports[segment]
                length = distances[segment+1]-distances[segment]
                diff = c-supports[segment]
                dot = (diff[:, 0]*delta[0]+diff[:, 1]*delta[1])+diff[:, 2]*delta[2]
                norm = (delta[0]*delta[0]+delta[1]*delta[1])+delta[2]*delta[2]
                t = distances[segment]+length*dot/norm
                if segment == 0: t = xp.minimum(t, distances[segment+1])
                elif segment == len(supports)-2: t = xp.maximum(t, distances[segment])
                else: t = xp.clip(t, distances[segment], distances[segment+1])
                interpretation = supports[segment]+((t-distances[segment])/length)[:, None]*delta
                absolute = xp.abs(c-interpretation)
                residual = (absolute[:, 0]+absolute[:, 1])+absolute[:, 2]
                choose = residual < closest  # same first-segment tie rule
                best = xp.where(choose, t, best)
                closest = xp.where(choose, residual, closest)
            output[start:start+len(c)] = xp.where(xp.isnan(best), self.dtype(0), best)
        return output.reshape(colors.shape[:-1])

    def interpolate_color_path(self, parameter, values, distances):
        xp = self.xp
        d, v = np.asarray(distances), np.asarray(values)
        if (d.ndim != 1 or v.shape != d.shape or len(d) < 2 or np.any(np.diff(d) <= 0)
                or not np.isfinite(d).all() or not np.isfinite(v).all()):
            raise ValueError("Frozen path interpolation table differs")
        d, v = self.array(d), self.array(v)
        if bool(xp.any(xp.diff(d) <= 0)):
            raise ValueError("Frozen interpolation supports collapse in selected precision")
        p = self.array(parameter)
        result = xp.zeros_like(p)
        for i in range(len(d)-1):
            if i == 0: mask = p <= d[i+1]
            elif i == len(d)-2: mask = p >= d[i]
            else: mask = (p >= d[i]) & (p <= d[i+1])
            value = v[i]+((p-d[i])/(d[i+1]-d[i]))*(v[i+1]-v[i])
            result = xp.where(mask, value, result)
        return result

    def piecewise_linear(self, signal, supports, values, *, clamp=False):
        """Selected-precision linear interpolation; explicit endpoint policy."""
        xp = self.xp
        s, v = np.asarray(supports), np.asarray(values)
        if (s.ndim != 1 or v.shape != s.shape or len(s) < 2
                or not np.isfinite(s).all() or not np.isfinite(v).all() or np.any(np.diff(s) <= 0)):
            raise ValueError("Frozen signal/LUT table differs")
        x, s, v = self.array(signal), self.array(s), self.array(v)
        if bool(xp.any(xp.diff(s) <= 0)):
            raise ValueError("Frozen interpolation supports collapse in selected precision")
        if not clamp and bool(xp.any((x < s[0]) | (x > s[-1]))):
            raise ValueError("Signal is outside original interp1d bounds")
        index = xp.clip(xp.searchsorted(s, x, side="left"), 1, len(s)-1)
        # scipy interp1d computes slope first, then delta * slope + low.
        value = ((v[index]-v[index-1])/(s[index]-s[index-1]))*(x-s[index-1])+v[index-1]
        if clamp:
            value = xp.where(x <= s[0], v[0], xp.where(x >= s[-1], v[-1], value))
        return value

    def titration(self, signal, flash):
        """Reuse the original CPU-built chemistry LUT; no chemistry refit."""
        if flash.restoration is not None:
            raise ValueError("Restoration needs an explicitly implemented CUDA route")
        xp, x = self.xp, self.array(signal)
        denom = (flash.max_value_aq-flash.min_value_aq) or 1.0
        y = xp.clip((x-self.dtype(flash.min_value_aq))/self.dtype(denom), 0, 1)
        physical = self.piecewise_linear(y, flash._lut_y, flash._lut_caq, clamp=True)
        caq = self.dtype(1-flash.aqueous_blend)*physical+self.dtype(flash.aqueous_blend)*y
        gas = (xp.clip(x, self.dtype(flash.min_value_g), self.dtype(flash.max_value_g))
               -self.dtype(flash.min_value_g))/self.dtype(flash.max_value_g-flash.min_value_g)
        return caq, gas

    def hydrostatic_fields(self, *, temperature, height, top_pressure,
                           water_density_20, water_density_23, density_20,
                           density_23, solubility_20, solubility_23):
        temp, height, top = map(self.array, (temperature, height, top_pressure))
        water = (self.dtype(water_density_20)*(self.dtype(23)-temp)/self.dtype(3)
                 + self.dtype(water_density_23)*(temp-self.dtype(20))/self.dtype(3))
        pressure = top+water*self.dtype(9.81)*height*self.dtype(1e-5)
        d20 = self.piecewise_linear(pressure, *density_20)
        d23 = self.piecewise_linear(pressure, *density_23)
        s20 = self.piecewise_linear(pressure, *solubility_20)
        s23 = self.piecewise_linear(pressure, *solubility_23)
        density = d20*(self.dtype(23)-temp)/self.dtype(3)+d23*(temp-self.dtype(20))/self.dtype(3)
        solubility = s20*(self.dtype(23)-temp)/self.dtype(3)+s23*(temp-self.dtype(20))/self.dtype(3)
        return dict(pressure=pressure, density=density, solubility=solubility)

    def selected_n1_fields(self, prepared, *, gain, factor, swi, aqueous_known,
                           gas_known, solubility, gas_density):
        xp = self.xp
        x, swi, sol, rho = map(self.array, (prepared, swi, solubility, gas_density))
        aqmask, gasmask = xp.asarray(aqueous_known), xp.asarray(gas_known)
        if aqmask.dtype != xp.bool_ or gasmask.dtype != xp.bool_:
            raise TypeError("Existing native boolean supports required")
        if any(v.shape != x.shape for v in (swi, sol, rho, aqmask, gasmask)):
            raise ValueError("Selected N1 pixel grids differ")
        fraction = xp.clip(x*self.dtype(gain), self.dtype(0), self.dtype(1))*self.dtype(factor)
        aqueous = xp.where(aqmask, fraction, self.dtype(0))
        gas = xp.zeros_like(x)
        gas[gasmask] = self.dtype(1)-swi[gasmask]
        aqueous[gasmask] = self.dtype(1)
        raw_gas = rho*gas
        raw_aqueous = sol*aqueous*xp.clip(self.dtype(1)-gas, self.dtype(0), None)
        return dict(aqueous=aqueous, gas=gas, concentration=aqueous*sol,
                    mass_total=raw_gas+raw_aqueous,
                    mass_aqueous=raw_aqueous, mass_gas=raw_gas)


class CudaPixelKernels(_PixelMath):
    """Explicit CUDA only. Missing runtime/OOM/unsupported operation raises."""
    def __init__(self, precision, *, device_id=0, pool_limit_bytes):
        if type(device_id) is not int or device_id < 0:
            raise ValueError("Explicit device index required")
        if type(pool_limit_bytes) is not int or pool_limit_bytes <= 0:
            raise ValueError("Explicit measured device budget required")
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= device_id:
            raise RuntimeError("Requested CUDA device is unavailable")
        super().__init__(cp, precision)
        self.device = cp.cuda.Device(device_id)
        self.device.use()
        free, total = cp.cuda.runtime.memGetInfo()
        if pool_limit_bytes >= free or total-free+pool_limit_bytes >= int(total*.9):
            raise RuntimeError("Device budget violates observed free memory/90 percent ceiling")
        self.pool = cp.cuda.MemoryPool()
        self.pool.set_limit(size=pool_limit_bytes)
        self.pool_limit_bytes = pool_limit_bytes
        self._active = False
        self.events = []
        typename = "float" if precision == "float32" else "double"
        self._warp = cp.RawKernel(_WARP.replace("REAL", typename), "warp_tab32",
                                 options=("--fmad=false",))

    @contextmanager
    def scope(self, stage):
        """Timed synchronised device boundary; transfer costs are included."""
        cp = self.xp
        if self._active:
            raise RuntimeError("CUDA stage cannot be nested")
        with self.device, cp.cuda.using_allocator(self.pool.malloc):
            completed = False
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                other_used = total-free-self.pool.total_bytes()
                if other_used+self.pool_limit_bytes >= int(total*.9):
                    raise RuntimeError("Other VRAM use invalidates the preserved device budget")
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                wall = time.perf_counter_ns()
                begin.record()
                self._active = True
                try:
                    yield self
                finally:
                    end.record()
                    end.synchronize()
                    free, total = cp.cuda.runtime.memGetInfo()
                    reserved = self.pool.total_bytes()
                    self.events.append(dict(stage=stage, wall_ns=time.perf_counter_ns()-wall,
                        cuda_elapsed_ms=cp.cuda.get_elapsed_time(begin, end),
                        pool_used_bytes=self.pool.used_bytes(), pool_reserved_bytes=reserved,
                        device_free_bytes=free, device_total_bytes=total,
                        device_global_used_bytes=total-free,
                        external_and_nonpool_used_bytes=total-free-reserved))
                # Actual driver usage includes external owners and this pool's
                # reserved blocks. Check after synchronization, before any host
                # result can leave this context; no fallback or alternate math.
                if total-free >= int(total*.9):
                    raise RuntimeError("Observed global VRAM use reached the 90 percent ceiling")
                completed = True
            finally:
                self._active = False
                if not completed:
                    # Host bridge drops device outputs before context exit, so
                    # unused blocks can actually be released on a failed stage.
                    # Never free another owner's allocation or retry the stage.
                    try:
                        self.device.synchronize()
                    finally:
                        self.pool.free_all_blocks()

    def array(self, value):
        if not self._active:
            raise RuntimeError("Every CUDA operation requires its scoped allocation budget")
        return super().array(value)

    def warp_continuous(self, source, matrix, dsize, *, inverse_map=False,
                        cv2_module=None):
        """Float image only; original CPU geometry inversion, CUDA TAB32 sampling."""
        if cv2_module is None:
            import cv2 as cv2_module
        cp = self.xp
        if type(inverse_map) is not bool or len(dsize) != 2 or any(type(v) is not int or v <= 0 for v in dsize):
            raise ValueError("Explicit positive OpenCV (width,height) size required")
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (3, 3) or not np.isfinite(m).all():
            raise ValueError("Original finite geometry matrix required")
        if not inverse_map:
            _, m = cv2_module.invert(m)
        src = cp.ascontiguousarray(self.array(source))
        if src.ndim not in (2, 3) or src.ndim == 3 and src.shape[2] not in (1, 3):
            raise ValueError("Continuous scalar/RGB warp only; never categorical labels")
        channels = 1 if src.ndim == 2 else src.shape[2]
        width, height = dsize
        if max(height*width*channels, src.size) >= 2**31:
            raise ValueError("Warp exceeds its explicit int32 indexing bound")
        shape = (height, width) if src.ndim == 2 else (height, width, channels)
        out = cp.empty(shape, dtype=self.dtype)
        self._warp(((height*width+255)//256,), (256,),
                   (src, out, cp.asarray(m.ravel()), np.int32(src.shape[0]), np.int32(src.shape[1]),
                    np.int32(height), np.int32(width), np.int32(channels)))
        return out

    def upload(self, value):
        return self.array(value)

    def download(self, value):
        return self.xp.asnumpy(value, blocking=True)

    def facts(self):
        p = self.xp.cuda.runtime.getDeviceProperties(self.device.id)
        name = p['name'].decode() if isinstance(p['name'], bytes) else p['name']
        return dict(backend="cuda", precision=self.precision, cupy_version=self.xp.__version__,
            cuda_runtime=self.xp.cuda.runtime.runtimeGetVersion(), device_id=self.device.id,
            device_name=name, pool_limit_bytes=self.pool_limit_bytes,
            events=list(self.events), cpu_decode=True, gpu_decode=False,
            cpu_matrix_inversion=True, cpu_float64_reducers=True,
            scientific_eligibility=False, production_permission_granted=False)

    def close(self):
        with self.device:
            self.device.synchronize()
            self.pool.free_all_blocks()


class CudaHostPixelMath:
    """CPU-host ABI for PrecisionAdapter, with explicit measured CUDA stages.

    Source/native guards, categorical facies masks and geometry/reducers remain
    on the CPU. Every accelerated stage uploads its actual operands and returns
    a synchronously downloaded result. All such transfers are included in the
    timed scope. This initial implementation deliberately has no shared-device
    residency cache and never presents CPU decoding/scatter/reduction as CUDA.
    """
    def __init__(self, precision, *, device_id=0, pool_limit_bytes):
        self.device_math = CudaPixelKernels(precision, device_id=device_id,
                                          pool_limit_bytes=pool_limit_bytes)
        self.precision = precision
        self.dtype = getattr(np, precision)

    def pixels(self, value):
        return np.asarray(value, dtype=self.dtype)

    def require_pixels(self, value):
        if not isinstance(value, np.ndarray) or value.dtype != self.dtype:
            raise TypeError("Pixel input must already have the declared working dtype")
        return value

    def _download_tree(self, value):
        if isinstance(value, dict):
            return {key: self._download_tree(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._download_tree(item) for item in value)
        return self.device_math.download(value)

    def _run(self, stage, function, *args, **kwargs):
        with self.device_math.scope(stage):
            result = None
            try:
                result = function(*args, **kwargs)
                return self._download_tree(result)
            except BaseException as error:
                # Failed kernel/download tracebacks can retain device arrays.
                # Keep the same exception and callsites, release only inactive
                # frame locals before scope cleanup releases unused own blocks.
                import traceback
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                del result  # Release device outputs before postguard/pool cleanup.

    def decode_log_rgb(self, storage):
        if not isinstance(storage, np.ndarray) or storage.dtype != np.float32:
            raise TypeError("Frozen log-linear RGB storage must remain float32")
        return self._run("decode_exp_upload_download", self.device_math.decode_log_rgb, storage)

    def warp_continuous(self, values, matrix, shape_hw, *, cv2_module):
        # Match CPU PixelMath: original continuous coverage is bool and stored
        # continuous references are float32. Convert BEFORE any warp arithmetic.
        # Categorical facies/phase transport never enters this method.
        working = self.pixels(values)
        if len(shape_hw) != 2 or any(type(v) is not int or v <= 0 for v in shape_hw):
            raise ValueError("Explicit positive native (height,width) size required")
        height, width = shape_hw
        return self._run("warp_tab32_upload_download", self.device_math.warp_continuous,
                         working, matrix, (width, height), cv2_module=cv2_module)

    def reference_normalized(self, current, baseline, clear, support):
        for value in (current, baseline, clear):
            self.require_pixels(value)
        if not isinstance(support, np.ndarray) or support.dtype != bool:
            raise TypeError("Original CPU boolean normalization support required")
        a, b, _ = self._run("photometry_upload_download",
                           self.device_math.reference_normalized,
                           current, baseline, clear, support)
        return a, b

    def relative_rgb(self, current, baseline, *, option="plain"):
        self.require_pixels(current)
        self.require_pixels(baseline)
        return self._run("relative_rgb_upload_download", self.device_math.relative_rgb,
                         current, baseline, option=option)

    def per_facies(self, signal, labels, models, *, ignore_labels=()):
        self.require_pixels(signal)
        if (not isinstance(labels, np.ndarray) or labels.shape != signal.shape[:2]
                or not np.issubdtype(labels.dtype, np.integer)):
            raise ValueError("Exact original CPU categorical facies grid required")
        output = np.zeros(signal.shape[:2], dtype=self.dtype)
        for label in np.unique(labels):
            if label in ignore_labels:
                continue
            mask = labels == label
            values = self.require_pixels(models[label](signal[mask]))
            if values.shape != (int(np.count_nonzero(mask)),):
                raise ValueError("Native per-facies model output grid differs")
            output[mask] = values
        return output

    def fit_color_path(self, colors, supports, path_distances):
        self.require_pixels(colors)
        return self._run("colour_projection_upload_download", self.device_math.fit_color_path,
                         colors, supports, path_distances)

    def interpret_color_path(self, parametrization, path_distances, values):
        self.require_pixels(parametrization)
        return self._run("colour_interpolation_upload_download",
                         self.device_math.interpolate_color_path,
                         parametrization, values, path_distances)

    def piecewise_linear(self, signal, supports, values, *, endpoint="error"):
        self.require_pixels(signal)
        if endpoint not in {"error", "clamp"}:
            raise ValueError("Explicit native endpoint rule required")
        return self._run("signal_interpolation_upload_download", self.device_math.piecewise_linear,
                         signal, supports, values, clamp=endpoint == "clamp")

    def titration(self, signal, flash):
        self.require_pixels(signal)
        return self._run("titration_upload_download", self.device_math.titration, signal, flash)

    def hydrostatic_fields(self, **kwargs):
        return self._run("hydrostatic_upload_download", self.device_math.hydrostatic_fields, **kwargs)

    def selected_n1_fields(self, prepared, **kwargs):
        self.require_pixels(prepared)
        return self._run("selected_n1_upload_download", self.device_math.selected_n1_fields,
                         prepared, **kwargs)

    @staticmethod
    def integrate(value, weight, mask, *, space_dim=2):
        if mask.dtype != bool or mask.shape != value.shape[:space_dim]:
            raise ValueError("Original native domain mask required")
        weighted = np.asarray(weight, np.float64)*np.asarray(np.where(mask, value, 0.), np.float64)
        for _ in range(space_dim):
            weighted = np.sum(weighted, axis=0, dtype=np.float64)
        return weighted

    def facts(self):
        return dict(self.device_math.facts(), host_api="PixelMath",
                    cpu_facies_scatter=True, cpu_float64_reducers=True,
                    transfers_in_each_stage=True, cached_scientific_arrays=False)

    def close(self):
        self.device_math.close()
