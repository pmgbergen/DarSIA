"""Native concentration, statistics and mass methods.

Attach runtime help text to the byte-locked optional provider. Importing its
class definitions does not import CuPy, allocate a device, or run pixel math.
"""
from . import mass_cuda as _cuda_provider

_cuda_method_docs = {
    "CudaPixelKernels": {
        "__init__": "Create selected-precision CUDA kernels for an explicit device and private allocation budget in bytes; existing live-VRAM admission remains mandatory.",
        "array": "Convert operands to the declared device floating dtype inside the active private-allocation scope.",
        "upload": "Upload host operands to the active device scope in the declared working precision.",
        "download": "Synchronously return device operands as host arrays, retaining their working dtype and shape.",
        "facts": "Report the actual device, precision, allocation budget and stage timings; metadata grants no scientific or production eligibility.",
        "close": "Synchronize this device and release unused blocks from its own private memory pool.",
    },
    "CudaHostPixelMath": {
        "__init__": "Create the host-array interface to a selected-precision CUDA device with an explicit private-pool budget in bytes.",
        "pixels": "Convert a host operand to the declared NumPy working dtype without allocating on the GPU.",
        "require_pixels": "Require an existing host ndarray in the declared working dtype and return that same array without conversion.",
        "decode_log_rgb": "Decode unchanged float32 log-linear RGB storage with a scoped device exponential and return RGB in working precision.",
        "warp_continuous": "Warp continuous host fields using the supplied homography and height-width grid; categorical labels do not enter this route.",
        "reference_normalized": "Normalize current and baseline RGB by measured clear-water RGB on the original boolean support, returning two host arrays.",
        "relative_rgb": "Subtract the bound baseline using the explicit plain, positive, negative or absolute background convention.",
        "per_facies": "Apply supplied per-facies models and scatter their results on the CPU categorical grid while preserving the working dtype.",
        "fit_color_path": "Project host RGB samples onto the supplied frozen colour path and return its distance parameter with the original nearest-first tie rule.",
        "interpret_color_path": "Evaluate the supplied colour-path response table at host parameters with linear end-segment extension.",
        "piecewise_linear": "Interpolate the supplied signal-response table with the explicit error or clamp endpoint policy.",
        "titration": "Return the supplied chemistry-LUT aqueous response and normalized gas response; table values define aqueous units and are not refitted.",
        "hydrostatic_fields": "Evaluate hydrostatic pressure in bar from temperatures in degrees Celsius and heights in metres, then interpolate supplied density and solubility tables.",
        "selected_n1_fields": "Apply frozen gain and factor on original aqueous and gas boolean supports; concentration and mass fields inherit the supplied solubility and density units.",
        "integrate": "Integrate scalar native fields with explicit domain weights and boolean masks on the CPU using float64 reductions.",
        "facts": "Report device-stage facts and the host interface responsibilities without claiming a scientific qualification or GPU-only pipeline.",
        "close": "Close the owned device interface, synchronize it and release its unused private-pool blocks.",
    },
}
for _class_name, _method_docs in _cuda_method_docs.items():
    _class = getattr(_cuda_provider, _class_name)
    for _method_name, _doc_text in _method_docs.items():
        _method = getattr(_class, _method_name)
        if _method.__doc__ is None:
            _method.__doc__ = _doc_text

del _cuda_provider, _cuda_method_docs, _class_name, _method_docs
del _class, _method_name, _doc_text, _method
