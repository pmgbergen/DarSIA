"""Conservative phase-two lean byte bounds, independent of compression samples.

The existing writer may retain every static field and mass_total. Accordingly
these bounds include the full fallback archives and the existing run store.
Only source-file identities and NPY headers are read; no numerical payload is
loaded or evaluated. A bound is DATA, never qualification or production GO.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from . import n1_capacity as n1
from .option_capacity import npz_bound, _shape
from .io_ops import LEAN_METADATA_BYTES
from .. import storage_profile as storage


FORMAT = "darsia-phase2-lean-byte-bound-v1"
STORE_BOUND_FORMAT = "darsia-lean-run-store-byte-bound-v1"
# The ordinary publisher always adds these masks to the six exported maps.
PLAIN_PRODUCTS = dict(concentration_aqueous="map", saturation_free_gas="map",
    mass_total="map", mass_aqueous="map", mass_free_gas="map", phase_labels=1,
    main_mass_eligible_mask=1, unquantified_facies_mask=1,
    bottom_strip_mask=1, below_photo_filled_mask=1)
STORE_ROLES = ("mass.physical_reference", "mass.floor_curve", "mass.depth_points")
MODEL = ("Conservative logical-byte bound: full native fallbacks plus run-static store; "
         "no compression gain or successful omission is assumed. Retained retry partials "
         "are counted separately. Free-space reserve remains an operational allowance.")
# result, compressed statistics, commit and the two original operation receipts.
FRAME_METADATA_FILES = 5


def _integer(value, where, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(where + " must be an explicit nonnegative integer")
    return value


def _pin(value):
    if not isinstance(value, dict) or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None:
        raise ValueError("Run-store input needs an exact source SHA-256")
    size = value.get("size_bytes", value.get("bytes"))
    _integer(size, "Run-store source bytes")
    if "size_bytes" in value and "bytes" in value and value["size_bytes"] != value["bytes"]:
        raise ValueError("Run-store source sizes disagree")
    return value["sha256"], size


def _version(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _array_metadata(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Run-store input needs the producer's numeric NPY header format")
    dtype = np.dtype(dtype)
    if (dtype.hasobject or dtype.fields is not None or dtype.kind not in "biuf"
            or dtype.itemsize not in (1, 2, 4, 8) or len(shape) > 4
            or any(type(size) is not int or size < 0 for size in shape)):
        raise ValueError("Unsupported numeric run-store array header")
    # The existing NPZ bound includes a 256-byte NPY header allowance. Refuse
    # a source shape/dtype whose output header could exceed that closed schema.
    from .option_capacity import NPY_HEADER_BYTES
    header = repr(dict(descr=dtype.str, fortran_order=False, shape=shape))
    if len(header.encode("latin1")) + 80 > NPY_HEADER_BYTES:
        raise ValueError("Run-store array exceeds the existing NPY header allowance")
    return dict(dtype=dtype.str, shape=list(shape), payload_bytes=math.prod(shape) * dtype.itemsize)


def _header(path, *, name=None):
    # Explicit current-host aliases may have another filename extension. The
    # numerical reader identifies NPY/NPZ bytes, so the header reader does too.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(set(names)) != len(names):
                raise ValueError("Run-store archive has duplicate members")
            if name is None:
                if len(members) != 1 or not members[0].filename.endswith(".npy"):
                    raise ValueError("Run-store array needs its explicit NPZ array key")
                member = members[0]
            else:
                member = archive.getinfo(str(name) + ".npy")
            with archive.open(member) as stream:
                result = _array_metadata(stream)
                if stream.tell() + result["payload_bytes"] != member.file_size:
                    raise ValueError("Run-store NPY header and member byte size disagree")
                return result
    with path.open("rb") as stream:
        result = _array_metadata(stream)
        if stream.tell() + result["payload_bytes"] != path.stat().st_size:
            raise ValueError("Run-store NPY header and file byte size disagree")
        return result


def metadata_allowance(config):
    """One file's exact ceiling, enforced by the lean publication boundary.

    The argument is retained so callers cannot confuse this with a projected
    measured size. No input length or guessed regeneration-error length can
    replace the actual guard. Queue/control history remains operational space.
    """
    if storage.resolve(config).get("profile") != "lean":
        raise ValueError("The metadata ceiling belongs only to the lean profile")
    return LEAN_METADATA_BYTES


def run_store_budget(config, *, path_for_use):
    """Bound the existing run store using only explicitly resolved input roles.

    ``path_for_use(use, descriptor)`` must return that role's accessible path
    from the existing rooted input catalogue. Original/foreign locators are
    never probed here. This function independently verifies its actual bytes
    and before/after file version. The callback grants no execution authority.
    """
    if storage.resolve(config) != dict(profile="lean", colour_intermediates="keep"):
        raise ValueError("Run-store budgeting requires explicit lean and KEEP")
    n1.validate_requests(config)
    if not callable(path_for_use):
        raise ValueError("Run-store budgeting requires the existing explicit input-role resolver")
    shared = config["shared"]
    checked, sources = {}, {}

    def source(use, bound):
        sha, size = _pin(bound)
        path = Path(path_for_use(use, bound))
        key = str(path), sha, size
        if key not in checked:
            before = _version(path)
            if before[2] != size:
                raise ValueError("Run-store input byte size changed")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != sha or _version(path) != before:
                raise ValueError("Run-store source SHA or version changed")
            checked[key] = before
        sources[use] = deepcopy(bound)
        return path

    physical = source(STORE_ROLES[0], shared["physical_reference"])
    arrays = {"physical_" + name: _header(physical, name=name)
              for name in ("facies", "domain", "porosity", "depth_m")}
    shapes = [row["shape"] for row in arrays.values()]
    if len(shapes[0]) != 2 or any(shape != shapes[0] for shape in shapes):
        raise ValueError("Physical run-store inputs need one common two-dimensional grid")
    floor = shared["floor_curve"]
    arrays["floor_curve"] = _header(source(STORE_ROLES[1], floor), name=floor.get("array_key"))
    depth = shared["depth_points"]
    if isinstance(depth, dict):
        path = source(STORE_ROLES[2], depth)
        # Each CSV row must contain at least one byte. Three float64 values
        # per row therefore need no more than 24 times the source byte size.
        rows = path.stat().st_size
        arrays["depth_points"] = dict(dtype=np.dtype(np.float64).str, shape_bound=[rows, 3],
            payload_bytes=24 * rows, method="CSV source-byte row-count upper bound")
    else:
        if (not isinstance(depth, list) or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in depth)):
            raise ValueError("Inline depth measurements need explicit triples")
        arrays["depth_points"] = dict(dtype=np.dtype(np.float64).str, shape=[len(depth), 3], payload_bytes=24 * len(depth))
    for name in ("reference_matrix", "correction_matrix"):
        arrays[name] = dict(dtype=np.dtype(np.float64).str, shape=[3, 3], payload_bytes=72)
    for (path, _sha, _size), before in checked.items():
        if _version(Path(path)) != before:
            raise ValueError("Run-store input changed while reading its headers")
    archive = npz_bound({name: row["payload_bytes"] for name, row in arrays.items()})
    metadata = metadata_allowance(config)
    return dict(format=STORE_BOUND_FORMAT, archive_bytes=archive, metadata_bytes=metadata,
        committed_bytes=archive + metadata, arrays=arrays, source_roles=sources,
        numerical_payload_loaded=False, production_permission_granted=False)


def fallback_archives(config, *, canvas_hw, precision):
    """The full products/static archives, including every legal lean fallback."""
    height, width = _shape(canvas_hw, "Lean phase-two canvas")
    if precision not in ("float32", "float64") or config.get("precision") != precision:
        raise ValueError("Lean budget and configuration precision differ")
    declared = n1.validate_requests(config)  # Probability remains phase-one only.
    products = dict(n1.PIXEL_PRODUCTS if "pixel_domain" in declared else PLAIN_PRODUCTS)
    # Adoption/Build can supply this optional mask without an N1 config flag.
    # Reserve its lossless fallback even when a historical frame omits it.
    products["approved_analysis_gap_mask"] = 1
    geometry = dict(n1.PIXEL_GEOMETRY)
    if "pixel_domain" in declared and "n16" in config["pixel_domain"]:
        products.update(n1.N16_PRODUCTS)
        geometry.update(n1.N16_GEOMETRY)
    pixels = height * width
    product_payloads = n1._payloads(products, pixels, precision)
    product_payloads.update({name: n1.SCALAR_BYTES for name in n1.PRODUCT_SCALARS})
    geometry_payloads = n1._payloads(geometry, pixels, precision)
    return dict(products_bytes=npz_bound(product_payloads), geometry_bytes=npz_bound(geometry_payloads),
        product_payloads=product_payloads, geometry_payloads=geometry_payloads,
        declared=list(declared), includes_mass_total=True, includes_all_static_fields=True)


def phase2_lean_budget(config, *, canvas_hw, store_bound):
    """Whole-run bound with KEEP, every fallback and retained retry partials.

    These numbers are separate from any measured projected lean use. Existing
    reservation arithmetic can keep ``admitted_run_bytes`` in its transient
    term for the run's lifetime, with one admitted frame also in flight.
    """
    setting = storage.resolve(config)
    if setting != dict(profile="lean", colour_intermediates="keep"):
        raise ValueError("Production lean budgeting requires explicit lean and KEEP")
    frames = config.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Lean budgeting requires the explicit nonempty frame list")
    if (not isinstance(store_bound, dict) or store_bound.get("format") != STORE_BOUND_FORMAT
            or set(store_bound.get("arrays") or {}) != set(storage._STORE_ARRAYS)):
        raise ValueError("Lean budgeting requires the complete checked existing run-store bound")
    shared = config["shared"]
    sources = {STORE_ROLES[0]: shared["physical_reference"], STORE_ROLES[1]: shared["floor_curve"]}
    if isinstance(shared["depth_points"], dict):
        sources[STORE_ROLES[2]] = shared["depth_points"]
    if store_bound.get("source_roles") != sources:
        raise ValueError("Run-store bound belongs to different configuration input roles")
    archive = npz_bound({name: _integer(row.get("payload_bytes"), "Run-store array payload")
                         for name, row in store_bound["arrays"].items()})
    if (store_bound.get("archive_bytes") != archive
            or store_bound.get("metadata_bytes") != LEAN_METADATA_BYTES
            or store_bound.get("committed_bytes") != archive + LEAN_METADATA_BYTES
            or store_bound.get("numerical_payload_loaded") is not False
            or store_bound.get("production_permission_granted") is not False):
        raise ValueError("Run-store bound arithmetic or authority identity differs")
    archives = fallback_archives(config, canvas_hw=canvas_hw, precision=config.get("precision"))
    parts = dict(products=archives["products_bytes"], static_fallback=archives["geometry_bytes"],
                 records=FRAME_METADATA_FILES * metadata_allowance(config))
    declared = archives["declared"]
    if "storage" in declared:
        parts["storage_layer"] = n1.storage_bytes(config, canvas_hw=canvas_hw,
            precision=config["precision"], extensive="pixel_domain" in declared and "n16" in config["pixel_domain"])
        # The accepted storage bound already includes its old 1 MiB receipt
        # allowances. Add the larger enforced ceiling for all three possible
        # newly serialized receipts; copied source receipts already have exact
        # descriptor byte sizes in storage_bytes.
        parts["storage_metadata_ceiling"] = 3 * LEAN_METADATA_BYTES
    if storage.stores_colour_coordinate(config):
        height, width = _shape(canvas_hw, "Colour-coordinate canvas")
        parts["colour_coordinate"] = npz_bound({storage.COLOUR_COORDINATE_ARRAY: height * width * 4})
    from .job_queue import TRANSIENT_RETRY_DELAYS
    copies = 1 + len(TRANSIENT_RETRY_DELAYS)
    committed_frame = sum(parts.values())
    # Original run records are individually guarded, including the conditional
    # scalar table; their serialized bytes cannot grow beyond the same ceiling.
    run_files = storage.run_record_files(config)
    run_records = len(set(run_files) - {"store_archive", "store_index"})
    committed_run = (_integer(store_bound.get("committed_bytes"), "Run-store bound")
                     + metadata_allowance(config) * run_records)
    admitted_frame, admitted_run = committed_frame * copies, committed_run * copies
    return dict(format=FORMAT, profile="lean", colour_intermediates="keep", pipeline="n1",
        precision=config["precision"], frames=len(frames), parts=parts, fallback_archives=archives,
        store=deepcopy(store_bound), committed_frame_bytes=committed_frame,
        committed_run_bytes=committed_run, serialization_attempts=copies,
        admitted_frame_bytes=admitted_frame, admitted_run_bytes=admitted_run,
        admitted_job_bytes=admitted_frame * len(frames) + admitted_run,
        inflight_bytes=admitted_frame, model=MODEL, measured=False,
        compression_gain_assumed=False, successful_omission_assumed=False,
        metadata_file_ceiling_bytes=LEAN_METADATA_BYTES,
        run_metadata_files=run_records, run_record_files=list(run_files.values()),
        scientific_artifact_bound=True, includes_queue_control_history=False,
        queue_control_history_policy="existing operational ledger, reserve and fresh guards",
        qualification_completed=False, production_permission_granted=False)
