"""Byte budget of the gas probability a full native-phase-n1 task declares.

The lean_capacity module also uses these codec envelopes.

A qualified bound profile measures the full native-phase-n1 core of its head
and route. A task on that head and route may declare the per-pixel gas
probability (``gas_probability_storage``, the phase stage's decision
probabilities). It adds its own byte budget per frame to the measured core.
The storage layer and the pixel-domain mass route belong to the phase-two (n1)
route, and ``probability_storage`` is a key no writer reads: all three are
refused here. The capacity rule refuses the colour-path coordinate unless the
qualified profile measured it.

The budget is an upper bound, never a measurement of the task. Each array the
writer declares counts at its exact dtype, uncompressed, on the grid it is
written on: the label grid, which is the task layout's phase canvas. Stated
margins cover the NPY and ZIP headers, deflate's worst-case expansion and the
JSON receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: The optional outputs a budget can cover on this route.
OPTIONS = ("gas_probability_storage",)
#: Optional outputs refused on this route, each with its reason.
REFUSED = {
    "storage": "belongs to the phase-two n1 route",
    "pixel_domain": "belongs to the phase-two n1 route",
    "probability_storage": "is no output of this route; no writer reads it",
}
#: The probability archive's arrays and exact dtypes (probability_storage.ARRAYS
#: and validate_arrays), all on the label grid.
PROBABILITY_ARRAYS = {"pixel_probability": "<f4", "regional_probability": "<f4",
                      "branch_support": "|u1", "later_rule_changes": "<u2"}
#: Bound on one 2-D array's NPY header. Format 1.0 pads the header to a multiple
#: of 64 bytes, and a 2-D header is 128 bytes.
NPY_HEADER_BYTES = 256
#: Bound on one ZIP member's records, beside twice its name: local header 30,
#: its ZIP64 extra 20, data descriptor 24, central header 46, its ZIP64 extra 28.
ZIP_MEMBER_BYTES = 148
#: End of central directory 22, ZIP64 end record 56 and ZIP64 locator 20.
ZIP_END_BYTES = 98
#: Bound on deflate's expansion of one stream, by the library that compresses it.
#: Classic zlib's deflateBound for a raw stream is about 0.031 % plus 7 bytes;
#: these allow more than three times that. zlib-ng's own deflateBound allows
#: 9-bit literals, so a zlib-ng stream counts an eighth more.
DEFLATE_FRACTION = 1 / 1000
DEFLATE_STREAM_BYTES = 64
ZLIBNG_FRACTION = 1 / 8
#: The JSON receipt beside the archive, and the frame record's growth.
RECEIPT_BYTES = 1 << 20
# Each serialization owns a new file. A failed flush can retain the whole file.
SERIALIZATION_ATTEMPTS = 4
OPTION_MODEL = ("UPPER BOUND, not a measurement: the qualified core per frame plus the "
                "declared option's arrays at their exact dtypes, uncompressed on the label "
                "grid, with stated header, deflate and receipt margins")
WRITERS = {"gas_probability_storage": ("src/darsia/fluidflower/phase/probability_storage.py",)}


def retained_serialization_bytes(case_count, frame_bytes, *, run_bytes=0):
    """Reserve every failed serialization until the whole task retires.

    This is additional to committed products and the ordinary in-flight frame.
    Prefix progress must never release this allowance: partial files survive it.
    """
    from .job_queue import TRANSIENT_RETRY_DELAYS
    if len(TRANSIENT_RETRY_DELAYS) + 1 != SERIALIZATION_ATTEMPTS:
        raise ValueError("The output retry count differs from its capacity envelope")
    if (type(case_count) is not int or case_count < 1
            or any(type(value) is not int or value < 0 for value in (frame_bytes, run_bytes))):
        raise ValueError("Exact nonnegative serialization byte allowances required")
    return (SERIALIZATION_ATTEMPTS - 1) * (case_count * frame_bytes + run_bytes)


def run_file_bytes(inventory, case_ids):
    """Measured files outside all frame directories, counted once per run."""
    from pathlib import PureWindowsPath
    cases = set(case_ids)
    return sum(row["bytes"] for row in inventory
               if not cases.intersection(PureWindowsPath(row["relative_path"]).parts))


def _shape(value, where):
    """An explicit positive (height, width) grid shape."""
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(size) is not int or size < 1 for size in value)):
        raise ValueError("Explicit positive grid shape required: " + where)
    return tuple(value)


def python_zlib_ng():
    """Whether Python's zlib module, which the NPZ and gzip writers use, is zlib-ng."""
    import zlib
    return hasattr(zlib, "ZLIBNG_VERSION")


def deflate_bound(size, *, zlib_ng=None):
    """Upper bound on the deflated size of a stream of size bytes, by its library.

    By default the library is Python's own zlib, the NPZ writers' deflate.
    """
    if python_zlib_ng() if zlib_ng is None else zlib_ng:
        return size + math.ceil(size * ZLIBNG_FRACTION) + DEFLATE_STREAM_BYTES
    return math.ceil(size * (1 + DEFLATE_FRACTION)) + DEFLATE_STREAM_BYTES


def npz_bound(payloads):
    """Upper bound on an NPZ archive of {array name: payload bytes}, as _write_npz writes it."""
    return ZIP_END_BYTES + sum(
        deflate_bound(NPY_HEADER_BYTES + size) + ZIP_MEMBER_BYTES + 2 * len((name + ".npy").encode())
        for name, size in payloads.items())


def probability_budget(config, *, canvas_hw, precision):
    """Per-frame bytes of the gas probability archive and its receipt on the label grid."""
    probability_request(config)
    pixels = canvas_hw[0] * canvas_hw[1]
    payloads = {name: pixels * np.dtype(dtype).itemsize for name, dtype in PROBABILITY_ARRAYS.items()}
    return dict(frame_bytes=npz_bound(payloads) + RECEIPT_BYTES, run_bytes=0,
                arrays={name: dict(dtype=PROBABILITY_ARRAYS[name], payload_bytes=size)
                        for name, size in payloads.items()},
                source="phase_probabilities.npz on the label grid at exact dtypes; receipt allowance")


def probability_request(config):
    """The gas probability request, checked by the writer's own validator."""
    from ..phase.probability_storage import parse_request
    return parse_request(config)


#: The budget of each option that has one. An option without an entry is refused.
BUDGETS = {"gas_probability_storage": probability_budget}
#: The writer's own request validator of each option.
VALIDATORS = {"gas_probability_storage": probability_request}


def declared_options(config):
    """The optional outputs a run configuration declares, in OPTIONS order.

    An optional output in REFUSED is refused when its entry is non-null. The
    gas probability is declared by its key.
    """
    for key, reason in REFUSED.items():
        if config.get(key) is not None:
            raise ValueError("Optional output " + key + " " + reason)
    return tuple(key for key in OPTIONS if key in config)


def validate_requests(config):
    """The declared options, each request checked by its writer's own validator.

    This reads only the run configuration, so it runs before any profile or
    layout read. An option without a budget is refused.
    """
    options = declared_options(config)
    for key in options:
        if key not in BUDGETS:
            raise ValueError("Optional output " + key + " has no budget; it needs its own measured profile")
        VALIDATORS[key](config)
    return options


@dataclass(frozen=True)
class OptionBudget:
    """The per-frame and per-run bytes of a task's declared optional outputs."""
    options: tuple
    frame_bytes: int
    run_bytes: int
    writers: tuple
    parts: dict
    model: str = OPTION_MODEL


def option_budget(config, *, layout, precision):
    """The budget of every option the run configuration declares, on its own layout.

    The label grid is the layout's phase canvas. An option without a budget
    in BUDGETS is refused, never counted as zero.
    """
    options = declared_options(config)
    if not options:
        raise ValueError("No declared optional output to budget")
    canvas_hw = _shape(layout.get("phase_shape_hw"), "phase canvas")
    parts = {}
    for key in options:
        if key not in BUDGETS:
            raise ValueError("Optional output " + key + " has no budget; it needs its own measured profile")
        parts[key] = BUDGETS[key](config, canvas_hw=canvas_hw, precision=precision)
    writers = tuple(path for key in options for path in WRITERS[key])
    return OptionBudget(options, sum(part["frame_bytes"] for part in parts.values()),
                        sum(part["run_bytes"] for part in parts.values()), writers, parts)
