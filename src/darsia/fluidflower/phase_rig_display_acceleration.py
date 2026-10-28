"""Optional CUDA presentation kernels with deterministic CPU fallback.

The scientific input is never modified. Integer chroma filtering is exactly
OpenCV's filter; floating RAW encoding repairs rounding-boundary pixels with
the canonical NumPy expression so CUDA and CPU produce the same RGB8.
"""
from __future__ import annotations

import os
import threading
import cv2
import numpy as np

_LOCK = threading.RLock()
_CUDA = None
_CUDA_REASON = None
_KERNEL = None
_GPU_SECONDS = 0.0


def gpu_seconds():
    with _LOCK:
        return _GPU_SECONDS


def _record_gpu(start, stop):
    global _GPU_SECONDS
    stop.synchronize()
    with _LOCK:
        _GPU_SECONDS += float(cuda_backend().cuda.get_elapsed_time(start, stop)) / 1000.


def cuda_backend():
    global _CUDA, _CUDA_REASON
    with _LOCK:
        if _CUDA is None:
            try:
                if os.environ.get("FFAC_DISPLAY_BACKEND", "auto").casefold() == "cpu":
                    raise RuntimeError("CPU backend explicitly selected")
                import cupy as cp
                if not cp.cuda.runtime.getDeviceCount():
                    raise RuntimeError("No CUDA device")
                cp.zeros(1, cp.uint8)
                _CUDA = cp
            except Exception as exc:
                _CUDA = False
                _CUDA_REASON = str(exc)
        return _CUDA or None


def backend_description():
    cp = cuda_backend()
    if cp is None:
        return {"backend": "cpu", "cuda_fallback_reason": _CUDA_REASON}
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    return {"backend": "cuda", "device": name.decode() if isinstance(name, bytes) else str(name)}


def _cuda_failure(exc):
    """Treat device/runtime unavailability as a presentation CPU fallback."""
    global _CUDA, _CUDA_REASON
    if not type(exc).__module__.startswith(("cupy", "cuda")):
        return False
    with _LOCK:
        _CUDA, _CUDA_REASON = False, str(exc)
    return True


def median_chroma(rgb8, *, backend="auto"):
    """Same RGB8/YCrCb/5x5 chroma transfer, CUDA integer median if available."""
    try:
        return _median_chroma(rgb8, backend=backend)
    except Exception as exc:
        if backend == "cpu" or not _cuda_failure(exc):
            raise
        return _median_chroma(rgb8, backend="cpu")


def _median_chroma(rgb8, *, backend):
    global _KERNEL
    cp = None if backend == "cpu" else cuda_backend()
    ycc = cv2.cvtColor(np.ascontiguousarray(rgb8), cv2.COLOR_RGB2YCrCb)
    if cp is None:
        for channel in (1, 2):
            ycc[..., channel] = cv2.medianBlur(ycc[..., channel], 5)
    else:
        with _LOCK:
            if _KERNEL is None:
                _KERNEL = cp.RawKernel(r'''
                extern "C" __global__ void chroma_median(
                    const unsigned char* a, unsigned char* b, int h, int w) {
                    int p = blockDim.x * blockIdx.x + threadIdx.x;
                    if (p >= h*w*2) return;
                    int c = p % 2 + 1, q = p/2, y=q/w, x=q%w;
                    unsigned char v[25]; int n=0;
                    for(int dy=-2;dy<=2;dy++) for(int dx=-2;dx<=2;dx++) {
                        int yy=max(0,min(h-1,y+dy)),xx=max(0,min(w-1,x+dx));
                        unsigned char t=a[(yy*w+xx)*3+c]; int j=n++;
                        while(j>0 && v[j-1]>t) {v[j]=v[j-1];j--;}
                        v[j]=t;
                    }
                    b[q*2+c-1]=v[12];
                }''', "chroma_median")
            gpu = cp.asarray(ycc)
            result = cp.empty((*ycc.shape[:2], 2), cp.uint8)
            size = result.size
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            _KERNEL(((size+255)//256,), (256,), (gpu, result, ycc.shape[0], ycc.shape[1]))
            stop.record()
            _record_gpu(start, stop)
            ycc[..., 1:] = result.get()
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def raw_detail(signal, support, *, backend="auto", tile_rows=128):
    try:
        return _raw_detail(signal, support, backend=backend, tile_rows=tile_rows)
    except Exception as exc:
        if backend == "cpu" or not _cuda_failure(exc):
            raise
        return _raw_detail(signal, support, backend="cpu", tile_rows=tile_rows)


def _raw_detail(signal, support, *, backend, tile_rows):
    cp = None if backend == "cpu" else cuda_backend()
    low, high = np.log(2.), np.log(65535.)
    output = np.empty((*support.shape, 3), np.uint8)
    for start in range(0, support.shape[0], tile_rows):
        end = min(start+tile_rows, support.shape[0])
        block = np.nan_to_num(signal[:, start:end], nan=low)
        if cp is None:
            encoded = 255*np.power(np.clip((block-low)/(high-low), 0, 1), .75)
            quantized = np.rint(encoded).astype(np.uint8)
        else:
            # NumPy's scalar promotion for this float32 array is intentionally
            # retained; only transcendental last-bit differences are repaired.
            gpu = cp.asarray(block)
            start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
            start_event.record()
            normalized = cp.clip((gpu-np.float32(low))/np.float32(high-low), 0, 1)
            encoded = (cp.float32(255)*cp.power(normalized, cp.float32(.75))).get()
            stop_event.record()
            _record_gpu(start_event, stop_event)
            quantized = np.rint(encoded).astype(np.uint8)
            uncertain = np.abs(encoded-np.floor(encoded)-np.float32(.5)) < np.float32(.002)
            if uncertain.any():
                values = block[uncertain]
                exact = 255*np.power(np.clip((values-low)/(high-low), 0, 1), .75)
                quantized[uncertain] = np.rint(exact).astype(np.uint8)
        output[start:end] = np.moveaxis(quantized, 0, -1)
    return output



def render_lift1(signal, sand, *, backend="auto", tile_rows=None):
    """Forward the configured lift mode through the single display implementation."""
    from .native.config import require_parameters
    from .native.display_modes import render_display_modes
    display = require_parameters("display_modes")
    change = require_parameters("change_display")
    rows = change["tile_rows"] if tile_rows is None else tile_rows
    mode = change["lift_mode"]
    modes, _ = render_display_modes(
        signal, sand, tile_rows=rows, parameters=display,
        selected_modes=(mode,),
        chroma_filter=lambda image: median_chroma(image, backend=backend))
    return np.rint(modes[mode] * 255).astype(np.uint8)
