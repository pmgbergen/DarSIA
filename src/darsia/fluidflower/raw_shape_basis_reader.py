"""Read a newly released RAW shape basis without rebuilding or registering it.

The explicit model authority joins the reviewed source, the request fixed by
Root before the operation, and that exact Root release. Historical renderer
documents and corrected NPZ files do not enter this branch implicitly. All
image/correction reads belong in the existing transfer/preparation workers.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .pipeline.io_ops import retry_io

FORMAT = "darsia-raw-rig-baseline-build-v1"
REQUEST_FORMAT = "darsia-raw-shape-basis-request-v1"
_MAX_JSON = 4 * 1024**2
_IDENTITY_KEYS = {"path", "sha256", "size_bytes"}
_GRID_KEYS = {"metadata", "voxel_size", "num_voxels", "array_axes",
              "homography_coordinates", "pixel_centres", "indexing",
              "array_row_direction", "array_column_direction"}
_RETAINED_SOURCES = ContextVar("raw_shape_basis_retained_sources", default=None)
_PHOTO_RELOCATIONS = ContextVar("raw_shape_basis_photo_relocations", default=None)
RETAINED_SOURCE_FORMAT = "darsia-retained-shape-basis-sources-v1"


@contextmanager
def project_retained_source_context(project):
    """Capture the project's explicit historical copy selection for a worker."""
    from .series_geometry import native_series_link
    value = None if project is None else native_series_link(project).get("retained_shape_basis_sources")
    manager = nullcontext()
    if value is not None:
        _keys(value, {"format", "folder", "aliases"}, "Retained shape-basis source selection")
        if (value["format"] != RETAINED_SOURCE_FORMAT or not isinstance(value["folder"], str)
                or not Path(value["folder"]).is_absolute() or not isinstance(value["aliases"], list)):
            raise ValueError("Retained shape-basis sources need an explicit folder and exact copy bindings")
        manager = retained_source_context(deepcopy(value["aliases"]), folder=value["folder"])
    with manager, reviewed_photo_context(_project_photo_relocations(project)):
        yield


def _project_photo_relocations(project):
    from . import rig_point_review as review
    document = None if project is None else review.load_review(project)
    if document is None:
        return []
    review._check_document(document)
    result = []
    for row in document["runs"].values():
        relocation = row.get("proposal_binding", {}).get("photograph_relocation")
        if relocation is None or row.get("status") != "approved":
            continue
        review._check_approval(document, row)
        review.check_photo_relocation(relocation, row["photo"])
        result.append(deepcopy(relocation))
    return result


@contextmanager
def reviewed_photo_context(relocations):
    """Keep exact same-byte photograph moves separate from metadata copies."""
    from .rig_point_review import verify_photo_relocation
    bound = {}
    for relocation in relocations:
        verified = verify_photo_relocation(relocation["original"], relocation["current"]["path"])
        if not _same(verified, relocation):
            raise ValueError("The reviewed photograph relocation changed")
        key = _key(verified["original"]["path"])
        if key in bound:
            raise ValueError("The reviewed photograph relocation is ambiguous")
        bound[key] = verified
    token = _PHOTO_RELOCATIONS.set(bound)
    try:
        yield
    finally:
        _PHOTO_RELOCATIONS.reset(token)


def reviewed_photo_path(project, original_path):
    """Resolve only a previously approved exact-byte photograph relocation."""
    from .rig_point_review import verify_photo_relocation
    for row in _project_photo_relocations(project):
        if _key(row["original"]["path"]) == _key(original_path):
            checked = verify_photo_relocation(row["original"], row["current"]["path"])
            if not _same(row, checked):
                raise ValueError("The reviewed photograph relocation changed")
            return Path(checked["current"]["path"])
    return Path(original_path)


@contextmanager
def retained_source_context(aliases, *, folder):
    """Use explicitly selected byte-identical historical inputs in one worker.

    This supplies the ordinary reader's existing relative-copy bindings; it
    grants no replacement bytes, RAW alias, model change or current-series
    authority. Replay proofs retain the original descriptors and their copies.
    """
    if not aliases or folder is None:
        raise ValueError("Retained sources need explicit copies and their containing folder")
    rows = deepcopy(list(aliases))
    if any(Path(str(row.get("original", {}).get("path", ""))).suffix.casefold() not in {".toml", ".csv"} for row in rows):
        raise ValueError("Historical source replay only accepts explicitly bound TOML/CSV metadata")
    from .storage_paths import resolve_storage_path
    # A moved project retains its original descriptors. Only the explicitly
    # declared storage relocation changes where its same-byte copies are read.
    root = str(resolve_storage_path(folder).resolve(strict=True))
    checked = _Pins(rows, folder=root)
    for original, _physical in checked.aliases.values():
        checked.add(original, copyable=True)
    token = _RETAINED_SOURCES.set((rows, root))
    try:
        yield
        checked.final()
    finally:
        _RETAINED_SOURCES.reset(token)


def has_retained_source_context():
    """Whether the current worker explicitly selected retained input copies."""
    return _RETAINED_SOURCES.get() is not None


def _same(left, right):
    """Compare JSON identities without equating integers, floats or booleans."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _keys(value, names, label):
    """Require the supported explicit document fields before following paths."""
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(label + " has unsupported or missing fields")
    return value


def _identity(value):
    """Validate an original absolute path, exact SHA and actual byte length."""
    _keys(value, _IDENTITY_KEYS, "Shape-basis file binding")
    if (not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts
            or re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])) is None
            or type(value["size_bytes"]) is not int or value["size_bytes"] < 0):
        raise ValueError("Shape-basis files require absolute path/SHA/byte identities")
    return deepcopy(value)


def _key(path):
    """Use the existing current-host spelling without resolving foreign history."""
    return str(Path(path)).replace("\\", "/").casefold()


def _stamp(path):
    """Capture the existing file's version around a bounded read or stream hash."""
    row = path.stat()
    return tuple(getattr(row, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _json_bytes(raw):
    """Use the original duplicate/nonfinite-rejecting metadata parser."""
    from .pipeline.job_inputs import _json_bytes as read
    return read(raw)


class _Pins:
    """Track original identities and only explicitly bound byte-identical copies."""

    def __init__(self, aliases=(), *, folder=None):
        """Admit only preparation-relative copies with identical SHA and size."""
        from .pipeline.products_publication import _no_links
        if not aliases and folder is None:
            retained = _RETAINED_SOURCES.get()
            if retained is not None:
                aliases, folder = retained
        self.aliases, self.rows, self.documents, self.copyable = {}, {}, {}, {}
        self.metadata_replay_keys = set()
        self._available_photo_relocations = deepcopy(_PHOTO_RELOCATIONS.get() or {})
        self.photo_relocations = {}
        for row in aliases:
            _keys(row, {"original", "copy"}, "Shape-basis source copy")
            original = _identity(row["original"])
            copied = row["copy"]
            if (folder is None or not isinstance(copied, dict) or set(copied) != _IDENTITY_KEYS
                    or not isinstance(copied.get("path"), str) or Path(copied["path"]).is_absolute()
                    or ".." in Path(copied["path"]).parts):
                raise ValueError("Shape-basis replay needs its exact preparation-relative copy")
            _no_links(Path(folder) / copied["path"])
            path = (Path(folder) / copied["path"]).resolve(strict=True)
            if not path.is_relative_to(Path(folder).resolve(strict=True)):
                raise ValueError("Shape-basis source copy leaves its preparation")
            actual = _identity(dict(copied, path=str(path)))
            if any(actual[key] != original[key] for key in ("sha256", "size_bytes")):
                raise ValueError("Shape-basis source copy changes historical bytes")
            name = _key(original["path"])
            if name in self.aliases:
                raise ValueError("Duplicate shape-basis source copy")
            self.aliases[name] = original, actual
        retained = _RETAINED_SOURCES.get()
        if retained is not None:
            for row in retained[0]:
                name = _key(row["original"]["path"])
                if name in self.aliases and _same(self.aliases[name][0], row["original"]):
                    self.metadata_replay_keys.add(name)

    @retry_io
    def add(self, binding, *, metadata=False, copyable=False):
        """Hash the consumed physical file, preserving its original descriptor."""
        from .pipeline.products_publication import _no_links
        original = _identity(binding)
        name = _key(original["path"])
        alias = self.aliases.get(name)
        if alias is not None:
            if not copyable or not _same(alias[0], original):
                raise ValueError("A shape-basis alias has another role or original identity")
            physical = alias[1]
        else:
            physical = original
        relocation = self._available_photo_relocations.get(name)
        if alias is None and relocation is not None:
            from .image_formats import image_family
            from .rig_point_review import photo_binding, check_photo_relocation
            if metadata or copyable or image_family(original["path"]) != "raw":
                raise ValueError("Reviewed photograph relocation cannot replace another source role")
            check_photo_relocation(relocation, photo_binding(relocation["current"]["path"]), original)
            physical = dict(original, path=relocation["current"]["path"])
            self.photo_relocations[name] = deepcopy(relocation)
        elif alias is None:
            from .storage_paths import resolve_storage_path
            # A declared storage move changes a locator only. The original
            # byte count, SHA and source role below remain authoritative.
            physical = dict(original, path=str(resolve_storage_path(original["path"])))
        path = Path(physical["path"])
        _no_links(path)
        before = _stamp(path)
        if before[2] != physical["size_bytes"]:
            raise ValueError("Shape-basis source size changed: " + str(path))
        if metadata and before[2] > _MAX_JSON:
            raise ValueError("Shape-basis metadata exceeds its explicit bound")
        digest = hashlib.sha256()
        content = [] if metadata else None
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
                if content is not None:
                    content.append(block)
        if before != _stamp(path) or digest.hexdigest() != physical["sha256"]:
            raise ValueError("Shape-basis source changed: " + str(path))
        prior = self.rows.get(name)
        if prior is not None and not _same(prior["original"], original):
            raise ValueError("One shape-basis locator has conflicting identities")
        self.rows[name] = dict(original=original, physical=deepcopy(physical))
        if copyable:
            self.copyable[name] = original
        if metadata:
            document = _json_bytes(b"".join(content))
            self.documents[name] = document
            return deepcopy(document)
        return original

    def final(self):
        """Rehash every consumed physical file without recapturing its identity."""
        rows = list(self.rows.values())
        for row in rows:
            self.add(row["original"], copyable=_key(row["original"]["path"]) in self.copyable)


def _time(value, label):
    """Require an explicit timezone-aware operation timestamp."""
    if not isinstance(value, str):
        raise ValueError(label + " must be a recorded UTC timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(label + " lacks its timezone")
    return result


def _matrix(value):
    """Validate the actual finite nonsingular native-to-camera transform."""
    result = np.asarray(value, np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-12:
        raise ValueError("Shape-basis registration needs a finite nonsingular 3x3 matrix")
    return result


def _pixels(value, *, shape=None, rgb=True):
    """Validate a noncasting C-order array identity and its exact byte count."""
    _keys(value, {"dtype", "shape", "order", "sha256", "size_bytes"}, "Shape-basis pixel identity")
    try:
        dtype = np.dtype(value["dtype"])
    except (TypeError, ValueError) as error:
        raise ValueError("Shape-basis pixel dtype is invalid") from error
    if (dtype.hasobject or dtype.kind not in "fiu" or dtype.str != value["dtype"]
            or value["order"] != "C" or not isinstance(value["shape"], list)
            or len(value["shape"]) != (3 if rgb else 2) or rgb and value["shape"][-1] != 3
            or any(type(n) is not int or n <= 0 for n in value["shape"])
            or re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])) is None
            or type(value["size_bytes"]) is not int
            or value["size_bytes"] != int(np.prod(value["shape"], dtype=object)) * dtype.itemsize
            or shape is not None and value["shape"] != ([*shape, 3] if rgb else list(shape))):
        raise ValueError("Shape-basis pixel identity, shape or byte count differs")
    return dtype


def _grid_document(value, shape):
    """Validate complete new RGB grid metadata and explicit ij conventions."""
    _keys(value, _GRID_KEYS, "Shape-basis native grid")
    metadata = value["metadata"]
    if (not isinstance(metadata, dict) or metadata.get("space_dim") != 2
            or metadata.get("indexing") != "ij" or metadata.get("series") is not False
            or metadata.get("scalar") is not False
            or value["indexing"] != "ij" or value["num_voxels"] != list(shape)
            or value["array_axes"] != ["row", "column", "camera-native RGB"]
            or value["homography_coordinates"] != ["column", "row", "one"]
            or value["pixel_centres"] != "integer indices"
            or value["array_row_direction"] != "down" or value["array_column_direction"] != "right"):
        raise ValueError("Shape-basis grid or orientation convention differs")
    for field in ("space_dim", "indexing", "dimensions", "origin", "series", "scalar", "date",
                  "reference_date", "time", "name", "type", "color_space"):
        if field not in metadata:
            raise ValueError("Shape-basis grid lacks complete metadata: " + field)
    for name, row in (("dimensions", metadata["dimensions"]), ("origin", metadata["origin"]),
                      ("voxel_size", value["voxel_size"])):
        array = np.asarray(row)
        if array.shape != (2,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("Shape-basis grid lacks finite spatial " + name)
        if name != "origin" and np.any(array <= 0):
            raise ValueError("Shape-basis grid spatial scales must be positive")


def _authority(pins, authority, request_binding, implementation, started):
    """Join the exact source/request to its owner's release and retained evidence.

    New releases need no separate reviewer identity. A previously bound
    full_review record remains literal historical evidence when present.
    """
    fields = {"source_closure", "operation_request", "root_release"}
    legacy = isinstance(authority, dict) and "full_review" in authority
    _keys(authority, fields | ({"full_review"} if legacy else set()), "Explicit shape-basis authority")
    source = authority["source_closure"]
    _keys(source, {"base_commit", "tree", "files"}, "Released shape-basis source closure")
    if any(re.fullmatch(r"[0-9a-f]{40}", str(source[k])) is None for k in ("base_commit", "tree")):
        raise ValueError("Shape-basis source release lacks its exact base commit/tree")
    if not isinstance(source["files"], list) or not source["files"] or not _same(source["files"], implementation):
        raise ValueError("Shape-basis producer implementation differs from the released source")
    if len({_key(_identity(row)["path"]) for row in implementation}) != len(implementation):
        raise ValueError("Shape-basis source closure contains duplicate paths")
    review = authority.get("full_review")
    if legacy:
        _keys(review, {"entry", "id", "answer_sha256"}, "Retained source-validation binding")
        entry = pins.add(review["entry"], metadata=True, copyable=True)
        if (not isinstance(entry, dict) or not isinstance(review["id"], str) or not review["id"]
                or entry.get("id") != review["id"] or not isinstance(entry.get("answer"), str)
                or hashlib.sha256(entry["answer"].encode("utf-8")).hexdigest() != review["answer_sha256"]):
            raise ValueError("Shape-basis retained validation record identity changed")
    expected = _identity(authority["operation_request"])
    if not _same(expected, request_binding):
        raise ValueError("Shape-basis certificate has another bound operation request")
    released = pins.add(authority["root_release"], metadata=True, copyable=True)
    if (not _same(released.get("source_closure"), source)
            or not _same(released.get("full_review"), review)
            or not _same(released.get("operation_request"), expected)
            or _time(released.get("released_utc"), "Root release") > started):
        raise ValueError("Root release does not bind these sources/request before the build")
    for row in implementation:
        pins.add(row, copyable=True)


def _request(pins, document, run):
    """Verify the exact original request and all duplicated source metadata."""
    inputs = document["inputs"]
    request = pins.add(inputs["operation_request"], metadata=True, copyable=True)
    _keys(request, {"format", "run_id", "configuration_files", "series_context", "baseline", "profile",
                    "operation_overrides", "protocol_inputs", "output", "shape_basis_certificate"}, "Shape-basis request")
    if (request["format"] != REQUEST_FORMAT or request["run_id"] != run
            or request["shape_basis_certificate"] is not None
            or inputs["raw_camera_origin"] != "bound operation request override"):
        raise ValueError("The explicit RAW shape-basis request role/run differs")
    for name in ("configuration_files", "series_context", "operation_overrides", "protocol_inputs"):
        if not _same(inputs[name], request[name]):
            raise ValueError("Shape-basis request/input closure differs: " + name)
    if not _same(request["baseline"], inputs["raw"]) or not _same(request["profile"], inputs["profile"]):
        raise ValueError("Shape-basis request binds another RAW acquisition or profile")
    context = request["series_context"]
    _keys(context, {"root", "manifest"}, "Shape-basis series context")
    if not Path(context["root"]).is_absolute() or Path(context["manifest"]["path"]) != Path(context["root"]) / "series.toml":
        raise ValueError("Shape-basis original configuration context differs")
    pins.add(context["manifest"], copyable=True)
    files = request["configuration_files"]
    if not isinstance(files, list) or not files or len({_key(_identity(x)["path"]) for x in files}) != len(files):
        raise ValueError("Shape-basis ordered original configuration closure is missing or duplicate")
    for row in files:
        pins.add(row, copyable=True)
    protocols = request["protocol_inputs"]
    if not isinstance(protocols, list) or not all(any(row.get("role") == k for row in protocols) for k in ("imaging", "injection", "pressure_temperature")):
        raise ValueError("Shape-basis original imaging/injection/pressure protocol closure is missing")
    from .protocol_content import require_content
    for row in protocols:
        _keys({key: row[key] for key in require_content(row)}, {"role", "folder", "sheet", "source"},
              "Shape-basis protocol binding")
        if row["role"] not in {"imaging", "injection", "pressure_temperature", "blacklist"} or any(row[k] is not None and not isinstance(row[k], str) for k in ("folder", "sheet")):
            raise ValueError("Shape-basis protocol role/folder/sheet differs")
        pins.add(row["source"], copyable=True)
    return request


def _owned_outputs(pins, document, request, certificate):
    """Verify the original reservation, all marker joins and exact owned files."""
    from .rig_raw_write import _canonical, _identity as filesystem_identity
    from .job_filesystem import inspect_local_file
    owner = document["output_ownership"]
    _keys(owner, {"operation_id", "request", "authority", "directory_identity", "intended_root_identity",
                  "owner", "setup_started", "save_started", "save_completed"}, "Shape-basis ownership")
    authority = request["output"]
    _keys(authority, {"destination", "intended_root", "forbidden_raw_roots", "maximum_output_bytes", "maximum_file_bytes"}, "Shape-basis output authority")
    if any(type(authority[k]) is not int or authority[k] <= 0 for k in ("maximum_output_bytes", "maximum_file_bytes")) or authority["maximum_file_bytes"] > authority["maximum_output_bytes"]:
        raise ValueError("Shape-basis byte ceilings are invalid")
    destination, root = _canonical(authority["destination"]), _canonical(authority["intended_root"])
    for key in ("directory_identity", "intended_root_identity"):
        if (not isinstance(owner[key], list) or len(owner[key]) != 2
                or any(type(value) is not int for value in owner[key])):
            raise ValueError("Shape-basis filesystem ownership needs exact native integer identities")
    if (destination == root or not destination.is_relative_to(root) or Path(certificate["path"]).parent != destination
            or not _same(owner["authority"], authority) or not _same(owner["request"], document["inputs"]["operation_request"])
            or owner["operation_id"] != document["producer"]["operation_id"]
            or re.fullmatch(r"[0-9a-f]{32}", str(owner["operation_id"])) is None
            or list(filesystem_identity(destination)) != owner["directory_identity"]
            or list(filesystem_identity(root)) != owner["intended_root_identity"]):
        raise ValueError("Shape-basis request, operation or current filesystem ownership differs")
    if not inspect_local_file(destination / ".raw-shape-reader-location").allowed:
        raise ValueError("Shape-basis output lacks a positive local filesystem identity")
    protected = authority["forbidden_raw_roots"]
    if not isinstance(protected, list) or not protected:
        raise ValueError("Shape-basis output lacks its protected acquisition roots")
    for row in protected:
        other = _canonical(row)
        if destination == other or destination.is_relative_to(other) or other.is_relative_to(destination):
            raise ValueError("Shape-basis output intersects a protected RAW root")
    if not any(_canonical(document["inputs"]["raw"]["path"]).is_relative_to(_canonical(p)) for p in protected):
        raise ValueError("Shape-basis RAW acquisition is not protected by its request")
    files = document["outputs"]
    if not isinstance(files, list) or not files or len({_key(_identity(x)["path"]) for x in files}) != len(files):
        raise ValueError("Shape-basis owned output closure is missing or duplicate")
    paths = []
    for row in files:
        path = _canonical(row["path"])
        if not path.is_relative_to(destination) or path == destination or path == Path(certificate["path"]):
            raise ValueError("Shape-basis output leaves its owned operation")
        pins.add(row)
        paths.append(path)
    from .pipeline.products_publication import _no_links
    actual = []
    for path in destination.rglob("*"):
        _no_links(path)
        if path.is_file() and path != Path(certificate["path"]):
            actual.append(path)
    if set(actual) != set(paths):
        raise ValueError("Shape-basis owned output membership changed")
    if any(x["size_bytes"] > authority["maximum_file_bytes"] for x in [*files, certificate]) or sum(x["size_bytes"] for x in [*files, certificate]) > authority["maximum_output_bytes"]:
        raise ValueError("Shape-basis retained logical bytes exceed its request ceilings")
    for name in ("owner", "setup_started", "save_started", "save_completed"):
        if owner[name] not in files:
            raise ValueError("Shape-basis ownership marker is outside the original output closure")
    marker = pins.add(owner["owner"], metadata=True)
    if (marker.get("format") != "darsia-owned-raw-output-v1" or marker.get("operation_id") != owner["operation_id"]
            or not _same(marker.get("request"), owner["request"]) or marker.get("destination") != str(destination)
            or marker.get("intended_root") != str(root) or marker.get("directory_identity") != owner["directory_identity"]
            or not isinstance(marker.get("local_filesystem"), dict)):
        raise ValueError("Shape-basis owner marker belongs to another operation")
    for name in ("setup_started", "save_started"):
        if pins.add(owner[name], metadata=True) != dict(operation_id=owner["operation_id"]):
            raise ValueError("Shape-basis setup/save marker differs")
    finished = pins.add(owner["save_completed"], metadata=True)
    if (finished.get("format") != "darsia-owned-raw-save-completed-v1"
            or finished.get("operation_id") != owner["operation_id"] or finished.get("destination") != str(destination)
            or not _same(finished.get("request"), owner["request"])
            or finished.get("directory_identity") != owner["directory_identity"]
            or finished.get("no_clobber_save") is not True or finished.get("complete") is not True):
        raise ValueError("Shape-basis original save did not complete under the same ownership")
    metadata = [x for x in files if Path(x["path"]) == destination / "meta_data.json"]
    if len(metadata) != 1:
        raise ValueError("Shape-basis lacks its original saved operation metadata")
    saved = pins.add(metadata[0], metadata=True)
    if (saved.get("raw_build_id") != owner["operation_id"] or not _same(saved.get("raw_camera"), document["inputs"]["camera_binding"])
            or saved.get("raw_baseline_sha256") != document["inputs"]["raw"]["sha256"]
            or saved.get("baseline_path") != document["inputs"]["raw"]["path"]):
        raise ValueError("Shape-basis saved metadata has another RAW, camera or operation")
    return destination, authority


@dataclass
class ReadShapeBasis:
    """Verified explicit source identity plus its finite operation-local guards."""
    certificate: dict
    document: dict
    guards: _Pins
    parent_labels: dict | None = None
    compatibility: dict | None = None

    def revalidate(self):
        """Recheck consumed originals/copies immediately before publication."""
        from .rig_raw_write import _canonical, _identity as filesystem_identity
        from .pipeline.products_publication import _no_links
        ownership = self.document["output_ownership"]
        authority = ownership["authority"]
        destination = _canonical(authority["destination"])
        if (list(filesystem_identity(destination)) != ownership["directory_identity"]
                or list(filesystem_identity(_canonical(authority["intended_root"])))
                   != ownership["intended_root_identity"]):
            raise ValueError("The original RAW-basis output directory identity changed")
        expected = {Path(row["path"]) for row in self.document["outputs"]} | {Path(self.certificate["path"])}
        actual = set()
        for path in destination.rglob("*"):
            _no_links(path)
            if path.is_file():
                actual.add(path)
        if actual != expected:
            raise ValueError("The original RAW-basis output membership changed")
        self.guards.final()

    def sources_to_copy(self):
        """Return immutable original metadata identities for preparation copies."""
        return deepcopy(list(self.guards.copyable.values()))

    def copy_source_rows(self):
        """Retain original identity while exposing only its checked physical bytes."""
        return [deepcopy(self.guards.rows[key]) for key in self.guards.copyable]


@retry_io
def _npz_header(path, *, shape, dtype=None):
    """Check the actual array header before any trusted image metadata decoding."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "array.npy" not in names or "metadata.npy" not in names:
            raise ValueError("Shape-basis image archive lacks an unambiguous array/metadata pair")
        info = archive.getinfo("array.npy")
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=65536)
            elif version == (2, 0):
                actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=65536)
            else:
                raise ValueError("Shape-basis array uses an unsupported NPY header")
            if (tuple(shape) != actual_shape or fortran or actual_dtype.hasobject
                    or actual_dtype.kind not in "fiu" or dtype is not None and actual_dtype != dtype
                    or info.file_size != stream.tell() + int(np.prod(actual_shape, dtype=object)) * actual_dtype.itemsize):
                raise ValueError("Shape-basis NPZ array header differs from its noncasting pixel/grid identity")


def _camera(pins, document, request, run, reference_image, camera_profile):
    """Join the actual RAW bytes, strict override and unique calibrated profile."""
    from .rig_raw_input import camera_binding
    from .image_formats import image_family
    inputs, renderer = document["inputs"], document["renderer"]
    raw, profile = inputs["raw"], inputs["profile"]
    if image_family(raw["path"]) != "raw":
        raise ValueError("A shape basis needs an actual RAW acquisition descriptor")
    pins.add(raw)
    profile_document = pins.add(profile, metadata=True, copyable=True)
    camera = camera_binding(inputs["camera_binding"])
    if (camera["run_id"] != run or camera["configuration"] != profile["path"]
            or camera["sha256"] != profile["sha256"]
            or request["operation_overrides"] != {"data.raw_camera": camera}
            or not _same(renderer["raw_acquisition"], raw) or renderer["profile_id"] != camera["profile_id"]):
        raise ValueError("Shape-basis camera override has another RAW/profile/run identity")
    parameters = profile_document.get("registration", profile_document)
    source = profile_document.get("source") or {}
    common = parameters.get("common") if isinstance(parameters, dict) else None
    assignments = source.get("profiles") if isinstance(source, dict) else None
    if not isinstance(common, dict) or not isinstance(assignments, dict):
        raise ValueError("Shape-basis calibrated profile assignments are missing")
    selected = [value for name, value in assignments.items() if str(name).casefold() == run]
    sensor, profiles = common.get("sensor"), common.get("profiles")
    if (selected != [camera["profile_id"]] or not isinstance(profiles, dict) or camera["profile_id"] not in profiles
            or not isinstance(sensor, dict) or not isinstance(sensor.get("camera_wh"), list)
            or len(sensor["camera_wh"]) != 2 or any(type(n) is not int or n <= 0 for n in sensor["camera_wh"])
            or renderer["camera_u_wh"] != sensor["camera_wh"]):
        raise ValueError("Shape-basis camera profile/run assignment or camera grid differs")
    if reference_image is not None and not _same(_identity(reference_image), raw):
        raise ValueError("Shape basis belongs to another measured reference photograph")
    if camera_profile is not None:
        if (not isinstance(camera_profile, dict)
                or set(camera_profile) != {"path", "sha256", "size_bytes", "profile_id", "camera_wh"}
                or any(camera_profile[k] != profile[k] for k in _IDENTITY_KEYS)
                or camera_profile["profile_id"] != camera["profile_id"]
                or camera_profile["camera_wh"] != sensor["camera_wh"]):
            raise ValueError("Shape basis differs from the measured model's exact camera profile")
    return parameters


def _runtime(pins, document):
    """Verify saved wrapper/source identities without claiming native equality."""
    from darsia._opencv_runtime import validate_observation
    producer = document["producer"]
    if (type(producer.get("pid")) is not int or producer["pid"] <= 0
            or producer.get("entrypoint") != "darsia.fluidflower.rig_raw_basis.build_raw_shape_basis"
            or _time(producer["finished_utc"], "Shape-basis completion") < _time(producer["started_utc"], "Shape-basis start")):
        raise ValueError("Shape-basis original producer identity or chronology differs")
    runtime = producer["runtime"]
    if (not isinstance(runtime, dict) or not isinstance(runtime.get("python"), str) or not runtime["python"]
            or runtime.get("identity_scope") != "Python module wrappers and executable file identities"
            or runtime.get("native_binary_equivalence_claimed") is not False
            or runtime.get("production_qualification_claimed") is not False
            or not isinstance(runtime.get("libraries"), dict) or set(runtime["libraries"]) != {"cv2", "rawpy", "numpy"}):
        raise ValueError("Shape-basis runtime evidence lacks its original limited scope")
    pins.add(runtime["executable"], copyable=True)
    for value in runtime["libraries"].values():
        _keys(value, {"version", "module"}, "Shape-basis runtime wrapper")
        if not isinstance(value["version"], str) or not value["version"]:
            raise ValueError("Shape-basis runtime wrapper version is missing")
        pins.add(value["module"], copyable=True)
    for name in ("setup_scientific_opencv", "reading_scientific_opencv", "scientific_opencv"):
        observed = document[name]
        validate_observation(observed)
        rows = (observed["before"], observed["after"]) if "before" in observed else (observed,)
        if any(row["pid"] != producer["pid"] for row in rows):
            raise ValueError("Shape-basis scientific observation belongs to another producer")


@retry_io
def _saved_payload(pins, document, destination, parameters):
    """Decode and join genuine saved pixels/grid, chain and measured invocation."""
    import darsia
    from .rig_raw_build import _grid, _array_identity, _require_quality, _json_value
    from .native.raw_positioning import load_shape_chain
    renderer, corrected = document["renderer"], document["corrected_baseline"]
    _keys(corrected, {"archive", "pixels", "grid"}, "Shape-basis corrected baseline")
    archive, pixels, grid = corrected["archive"], corrected["pixels"], corrected["grid"]
    shape = document["shape_hw"]
    if (not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)
            or renderer["native_shape_hw"] != shape or Path(archive["path"]) != destination / "shape_corrected_baseline.npz"
            or archive not in document["outputs"] or not _same(renderer["source_npz"], archive)
            or not _same(renderer["source_pixels"], pixels) or not _same(renderer["source_grid"], grid)):
        raise ValueError("Shape-basis corrected NPZ/source duplicates differ")
    dtype = _pixels(pixels, shape=shape)
    _grid_document(grid, shape)
    pins.add(archive)
    _npz_header(archive["path"], shape=[*shape, 3], dtype=dtype)
    saved = darsia.imread(Path(archive["path"]))
    if not _same(_array_identity(saved.img), pixels) or not _same(_grid(saved), grid):
        raise ValueError("Shape-basis decoded NPZ pixels or complete grid differ")
    if saved.color_space != "RGB":
        raise ValueError("Shape-basis saved image is not RGB")
    replay = renderer["saved_shape_chain_replay"]
    _keys(replay, {"input_pixels", "input_grid", "normalization", "correction_assets", "output_pixels", "output_grid"}, "Shape-basis saved-chain replay")
    camera_hw = list(reversed(renderer["camera_u_wh"]))
    _pixels(replay["input_pixels"], shape=camera_hw)
    _grid_document(replay["input_grid"], camera_hw)
    if (renderer["saved_shape_chain_replay_bit_identical"] is not True
            or not _same(replay["input_grid"], document["inputs"]["uncorrected_grid"])
            or not _same(replay["output_pixels"], pixels) or not _same(replay["output_grid"], grid)):
        raise ValueError("Shape-basis saved-chain input/output closure differs")
    assets = renderer["shape_corrections"]
    if not isinstance(assets, list) or not assets or not _same(assets, replay["correction_assets"]):
        raise ValueError("Shape-basis ordered saved correction chain differs")
    for i, asset in enumerate(assets):
        path = Path(asset["path"])
        if (path.parent != destination or re.fullmatch(r"shape_correction_" + str(i) + r"_[a-z0-9]+\.npz", path.name) is None
                or asset not in document["outputs"]):
            raise ValueError("Shape-basis correction order or owned file membership differs")
        pins.add(asset)
    discovered = set(destination.glob("shape_correction_*.npz"))
    if discovered != {Path(row["path"]) for row in assets}:
        raise ValueError("Shape-basis saved correction membership differs")
    chain = load_shape_chain(assets)
    if any(Path(asset["path"]).name != f"shape_correction_{i}_{type(step).__name__.lower()}.npz" for i, (asset, step) in enumerate(zip(assets, chain))):
        raise ValueError("Shape-basis correction type differs from its original numbered asset")
    del chain
    normalization = replay["normalization"]
    _keys(normalization, {"black_subtracted_dn_divisor", "white_balance_applied", "gamma_applied"}, "Shape-basis normalization")
    sensor = renderer["sensor_metadata"]
    if not isinstance(sensor, dict) or any(type(sensor.get(k)) not in (int, float) or not np.isfinite(sensor[k]) for k in ("white_level", "black_level")):
        raise ValueError("Shape-basis recorded sensor range is missing")
    scale = np.float32(sensor["white_level"] - sensor["black_level"])
    if (not np.isfinite(scale) or scale <= 0 or type(sensor.get("orientation")) is not int
            or type(normalization["black_subtracted_dn_divisor"]) not in (int, float)
            or normalization["black_subtracted_dn_divisor"] != float(scale)
            or normalization["white_balance_applied"] is not False or normalization["gamma_applied"] is not False
            or renderer["white_balance_applied"] is not False or renderer["gamma_applied"] is not False
            or renderer["direct_sensor_output_equality_claimed"] is not False):
        raise ValueError("Shape-basis normalization, orientation or declared linear scope differs")
    invocation = renderer["registration_invocation"]
    _keys(invocation, {"callable", "review_green", "camera_green", "review_units", "camera_units", "policies"}, "Shape-basis measured invocation")
    policies = _json_value({key: parameters["common"][key] for key in ("review", "matching")})
    if (invocation["callable"] != "darsia.fluidflower.native.registration.review_to_camera"
            or invocation["review_units"] != "black-subtracted camera DN from saved normalized RAW pixels"
            or invocation["camera_units"] != "black-subtracted camera DN" or not _same(invocation["policies"], policies)
            or renderer["positioning_inputs"] != "actual saved corrected-baseline NPZ green and pinned RAW camera U green"):
        raise ValueError("Shape-basis registration invocation or profile policies differ")
    _pixels(invocation["review_green"], shape=shape, rgb=False)
    _pixels(invocation["camera_green"], shape=camera_hw, rgb=False)
    green = np.asarray(saved.img[..., 1], np.float32) * scale
    if not _same(_array_identity(green), invocation["review_green"]):
        raise ValueError("Shape-basis registration did not bind these saved green pixels")
    if invocation["camera_green"]["dtype"] != np.dtype(np.float32).str:
        raise ValueError("Shape-basis camera-U registration input is not original float32")
    quality = renderer["registration"]
    if (not isinstance(quality, dict) or type(quality.get("fit_n")) is not int
            or any(type(quality.get(k)) not in (int, float) for k in ("fit_inlier_fraction", "sampled_jacobian_min"))):
        raise ValueError("Shape-basis registration metrics have invalid types")
    for metrics in (quality.get("heldout") or {}).values():
        if not isinstance(metrics, dict) or type(metrics.get("n")) is not int or any(type(metrics.get(k)) not in (int, float) for k in ("p50_px", "p95_px", "max_px", "median_match_correlation")):
            raise ValueError("Shape-basis held-out metrics have invalid types")
    _require_quality(quality, parameters["common"]["review"])
    h = _matrix(document["H_rig_corrected_native_to_baseline_camera_u"])
    if (not _same(h.tolist(), renderer["native_to_camera_u_h"])
            or not np.array_equal(_matrix(renderer["camera_u_to_native_h"]), np.linalg.inv(h))):
        raise ValueError("Shape-basis H, duplicate or actual inverse differs")
    convention = document["homography_convention"]
    if convention != dict(source="saved corrected-baseline pixel centres (column,row)",
            target="camera-U pixel centres (column,row)", camera_u_axes="x right, y down", raw_orientation=sensor["orientation"]):
        raise ValueError("Shape-basis coordinate convention or RAW orientation differs")
    return grid


def _spatial_grid(image):
    """Extract full spatial semantics without equating RGB/scalar/date metadata."""
    from .rig_raw_build import _json_value
    if image.space_dim != 2 or image.indexing != "ij" or image.series:
        raise ValueError("The imported parent is not one two-dimensional ij grid")
    values = dict(space_dim=image.space_dim, indexing=image.indexing,
        dimensions=_json_value(image.dimensions), origin=_json_value(image.origin),
        voxel_size=_json_value(image.voxel_size), num_voxels=_json_value(image.num_voxels),
        row_direction="down", column_direction="right", pixel_centres="integer indices")
    for name in ("dimensions", "origin", "voxel_size"):
        data = np.asarray(values[name])
        if data.shape != (2,) or data.dtype.kind not in "fiu" or not np.isfinite(data).all() or name != "origin" and np.any(data <= 0):
            raise ValueError("The imported parent has invalid full spatial metadata")
    return values


@retry_io
def _parent_grid(pins, document, workspace_binding, run, *, request=None,
                 parent_configuration_lineage=None):
    """Prove this exact run's four original fields keep the new spatial basis."""
    import darsia
    from .geometry_workspace_frame import selected_fields
    revision = pins.add(workspace_binding, metadata=True, copyable=True)
    config = revision.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("Shape-basis parent workspace lacks its saved configuration")
    fields = selected_fields(config, run, check_files=True)
    record = (config["frame_binding"]["runs"] or {}).get(run) or {}
    configuration = {k: record.get("configuration", {}).get(k) for k in _IDENTITY_KEYS}
    _identity(configuration)
    lineage_audit = None
    if (parent_configuration_lineage is not None
            and sum(_same(configuration, x) for x in document["inputs"]["configuration_files"]) == 0):
        from .raw_shape_configuration_lineage import _require_shape_configuration_lineage
        lineage_audit = _require_shape_configuration_lineage(
            pins, run=run, parent_configuration=configuration,
            producer_configuration_files=document["inputs"]["configuration_files"],
            request=request, proof=parent_configuration_lineage)
        if not isinstance(lineage_audit, dict) or not lineage_audit:
            raise ValueError("Shape-basis configuration lineage did not return its verified bindings")
    if (sum(_same(configuration, x) for x in document["inputs"]["configuration_files"]) != 1
            and lineage_audit is None):
        raise ValueError("Shape-basis original config chain does not contain this workspace's bound run")
    new = document["corrected_baseline"]["grid"]
    expected = dict(space_dim=new["metadata"]["space_dim"], indexing=new["indexing"],
        dimensions=new["metadata"]["dimensions"], origin=new["metadata"]["origin"],
        voxel_size=new["voxel_size"], num_voxels=new["num_voxels"],
        row_direction=new["array_row_direction"], column_direction=new["array_column_direction"], pixel_centres=new["pixel_centres"])
    sources, label_fields = {}, None
    for name in ("shape_corrected_baseline", "labels", "facies", "depth"):
        field = fields[name]
        binding = {k: field[k] for k in _IDENTITY_KEYS}
        pins.add(binding)
        _npz_header(binding["path"], shape=field["shape"])
        image = darsia.imread(Path(binding["path"]))
        if list(image.img.shape) != field["shape"] or not _same(_spatial_grid(image), expected):
            raise ValueError(f"{run}: RAW shape basis and original {name} full grid differ; STOP, no migration")
        if name == "shape_corrected_baseline" and (image.scalar or image.img.ndim != 3 or image.img.shape[-1] != 3 or getattr(image, "color_space", None) != "RGB"):
            raise ValueError("The original corrected baseline lacks explicit RGB grid orientation")
        if name != "shape_corrected_baseline" and (image.series or image.img.ndim != image.space_dim):
            raise ValueError("The original categorical/depth field lacks its scalar grid")
        if name == "labels":
            label_fields = dict(metadata=image.metadata(), shape_hw=list(image.img.shape), source=binding)
        sources[name] = binding
        del image
    if lineage_audit is not None:
        return label_fields, dict(workspace_revision=deepcopy(workspace_binding), configuration=configuration,
                                 fields=sources, spatial_grid=expected, old_new_pixel_equality_claimed=False,
                                 configuration_lineage=lineage_audit)
    return label_fields, dict(workspace_revision=deepcopy(workspace_binding), configuration=configuration,
                             fields=sources, spatial_grid=expected, old_new_pixel_equality_claimed=False)


def _released_parent_lineage(pins, authority, explicit):
    """Carry a new release's pinned proof through existing reader callers."""
    released = pins.documents[_key(authority["root_release"]["path"])]
    if "parent_configuration_lineage" not in released:
        return explicit
    binding = _identity(released["parent_configuration_lineage"])
    proof = pins.add(binding, metadata=True, copyable=True)
    if not isinstance(proof, dict):
        raise ValueError("Released parent configuration lineage must be one explicit document")
    if explicit is not None and not _same(explicit, proof):
        raise ValueError("Explicit parent configuration lineage differs from the bound release")
    return proof


def read_shape_basis(certificate, *, run_id, authority, parent_workspace,
                     reference_image=None, camera_profile=None, aliases=(), folder=None,
                     parent_configuration_lineage=None):
    """Admit only the externally released operation and compatible parent grid.

    Registration and RAW decoding are not repeated. The producer's replay and
    camera-green identities remain producer evidence, closed by its reviewed
    source and original request. Saved NPZ pixels, full grid, saved-green input,
    ownership, gates and every consumed file are actually checked here.
    """
    if not isinstance(run_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run_id) is None:
        raise ValueError("Shape-basis run must be one explicit safe lowercase identity")
    if authority is None or parent_workspace is None:
        raise ValueError("RAW shape basis needs its external Root source/request/review authority and per-run parent grid")
    if isinstance(authority, dict) and "received_basis" in authority:
        from .raw_shape_basis_copy import read_received_basis
        return read_received_basis(certificate, run_id=run_id, authority=authority,
            parent_workspace=parent_workspace, reference_image=reference_image,
            camera_profile=camera_profile, aliases=aliases, folder=folder,
            parent_configuration_lineage=parent_configuration_lineage)
    pins = _Pins(aliases, folder=folder)
    try:
        document = pins.add(certificate, metadata=True)
    except (OSError, UnicodeError) as error:
        raise ValueError("The exact RAW shape-basis certificate cannot be read") from error
    if (not isinstance(document, dict) or document.get("format") != FORMAT or document.get("artifact_kind") != "raw_shape_basis"
            or document.get("producer_role") != "raw_shape_basis" or document.get("run_id") != run_id
            or document.get("run") != run_id or document.get("complete_rig_model") is not False
            or document.get("production_qualification_claimed") is not False
            or document.get("certificate_scope") != "baseline and ordered shape chain to camera-U registration only"):
        raise ValueError("No genuine explicit RAW shape-basis role/run is bound")
    try:
        started = _time(document["producer"]["started_utc"], "Shape-basis start")
        _authority(pins, authority, document["inputs"]["operation_request"], document["renderer_implementation"], started)
        parent_configuration_lineage = _released_parent_lineage(pins, authority, parent_configuration_lineage)
        request = _request(pins, document, run_id)
        destination, _ = _owned_outputs(pins, document, request, certificate)
        parameters = _camera(pins, document, request, run_id, reference_image, camera_profile)
        _runtime(pins, document)
        _saved_payload(pins, document, destination, parameters)
        consistency = document["shape_basis_consistency"]
        if consistency != dict(required=False, basis_certificate=None, byte_exact_verified=False,
                mismatch_policy="STOP; no automatic placement or adoption changes"):
            raise ValueError("Shape-basis certificate claims another full-build consistency role")
        if parent_configuration_lineage is None:
            labels, compatibility = _parent_grid(pins, document, parent_workspace, run_id)
        else:
            from .raw_shape_basis_copy import _physical_lineage_inputs
            retained_aliases = dict(pins.aliases)
            _physical_lineage_inputs(pins, document, parent_workspace, run_id, request,
                                     parent_configuration_lineage)
            # Keep the retained copies guarded separately from the live loader files.
            for key in retained_aliases.keys() - pins.aliases.keys():
                pins.add(retained_aliases[key][1], copyable=False)
            labels, compatibility = _parent_grid(
                pins, document, parent_workspace, run_id, request=request,
                parent_configuration_lineage=parent_configuration_lineage)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError,
            OSError, zipfile.BadZipFile, EOFError, UnicodeError) as error:
        raise ValueError("Malformed or incomplete RAW shape-basis closure") from error
    result = ReadShapeBasis(deepcopy(certificate), document, pins, labels, compatibility)
    result.revalidate()
    return result


def has_shape_basis(model):
    """Select only the explicit new model field; never infer it from shape/flags."""
    grid = model.get("baseline_native_grid") if isinstance(model, dict) else None
    return isinstance(grid, dict) and "raw_shape_basis_authority" in grid


def require_basis_index(index_path, runs):
    """Refuse missing basis metadata before any painting/photo/model SHA work.

    This bounded presence check grants no authority. The existing worker must
    subsequently perform full descriptor, payload and compositional admission.
    """
    reason = "The rigs need their RAW basis before the map can be adopted"
    try:
        path = Path(index_path)
        with path.open("rb") as stream:
            raw = stream.read(_MAX_JSON + 1)
        if len(raw) > _MAX_JSON:
            raise ValueError(reason)
        index = _json_bytes(raw)
        rows = index.get("runs") if isinstance(index, dict) else None
        if not isinstance(rows, dict):
            raise ValueError(reason)
        for run in {str(value).casefold() for value in runs}:
            row = rows.get(run)
            bound = row.get("reference_model") if isinstance(row, dict) else None
            if not isinstance(bound, dict) or not isinstance(bound.get("path"), str):
                raise ValueError(reason)
            model_path = Path(bound["path"])
            if not model_path.is_absolute():
                model_path = path.parent / model_path
            with model_path.open("rb") as stream:
                raw = stream.read(_MAX_JSON + 1)
            if len(raw) > _MAX_JSON:
                raise ValueError(reason)
            model = _json_bytes(raw)
            if not has_shape_basis(model):
                raise ValueError(reason)
            grid = model["baseline_native_grid"]
            if (not isinstance(grid.get("raw_shape_basis_authority"), dict)
                    or not isinstance(grid.get("parent_workspace_revision"), dict)
                    or not isinstance(grid.get("renderer"), dict)
                    or not Path(grid["renderer"].get("path", "")).is_file()):
                raise ValueError(reason)
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(reason) from error


def guard_output_root(destination, models):
    """Keep NEW transfer/preparation writes outside their original producer data."""
    from .pipeline.products_publication import _no_links
    candidate = Path(destination)
    _no_links(candidate)
    target = candidate.resolve(strict=False)
    for model in models:
        if not has_shape_basis(model):
            continue
        grid = model["baseline_native_grid"]
        protected = [(Path(grid["renderer"]["path"]).parent, True)]
        photograph = model["reference_image"]
        relocation = (_PHOTO_RELOCATIONS.get() or {}).get(_key(photograph["path"]))
        if relocation is not None:
            from .rig_point_review import check_photo_relocation, photo_binding
            check_photo_relocation(relocation, photo_binding(relocation["current"]["path"]), photograph)
            protected.extend([(Path(photograph["path"]).parent, False),
                              (Path(relocation["current"]["path"]).parent, True)])
        else:
            protected.append((Path(photograph["path"]).parent, True))
        protected.extend((Path(row["path"]).parent, True)
            for row in grid["parent_grid_compatibility"]["fields"].values())
        for source, strict in protected:
            _no_links(source)
            physical = source.resolve(strict=strict)
            if target == physical or target.is_relative_to(physical):
                raise ValueError("New painting outputs must stay outside their original RAW, basis and native-field folders")


def read_model_basis(model, *, aliases=(), folder=None, metadata_replay=(), photo_relocations=()):
    """Reconstruct a new measured model's exact basis and original grid proof."""
    if photo_relocations:
        with reviewed_photo_context(photo_relocations):
            return read_model_basis(model, aliases=aliases, folder=folder, metadata_replay=metadata_replay)
    if metadata_replay:
        wanted = {_key(path) for path in metadata_replay}
        selected = [row for row in aliases if _key(row["original"]["path"]) in wanted]
        if len(wanted) != len(metadata_replay) or len(selected) != len(wanted):
            raise ValueError("Historical metadata replay must select exact retained source copies")
        with retained_source_context(selected, folder=folder):
            return read_model_basis(model, aliases=aliases, folder=folder)
    if not has_shape_basis(model):
        raise ValueError("This measured model has no explicit RAW shape-basis authority")
    grid = model["baseline_native_grid"]
    _keys(grid, {"renderer", "shape_hw", "H_rig_corrected_native_to_baseline_camera_u",
        "H_baseline_camera_u_to_rig_corrected_native", "legacy_dimensions_m",
        "physical_metric_is_separate", "raw_shape_basis_authority",
        "parent_workspace_revision", "parent_grid_compatibility"}, "Explicit model native grid")
    checked = read_shape_basis(grid["renderer"], run_id=model["run_id"],
        authority=grid["raw_shape_basis_authority"], parent_workspace=grid.get("parent_workspace_revision"),
        reference_image=model["reference_image"], camera_profile=model["camera_profile"],
        aliases=aliases, folder=folder)
    document = checked.document
    if (grid.get("shape_hw") != document["shape_hw"]
            or grid.get("legacy_dimensions_m") != [.49, .9]
            or grid.get("physical_metric_is_separate") is not True
            or not _same(grid.get("H_rig_corrected_native_to_baseline_camera_u"), document["H_rig_corrected_native_to_baseline_camera_u"])
            or not _same(grid.get("H_baseline_camera_u_to_rig_corrected_native"), document["renderer"]["camera_u_to_native_h"])
            or not _same(grid.get("parent_grid_compatibility"), checked.compatibility)
            or not _same({key:model["baseline_configuration"][key] for key in _IDENTITY_KEYS}, checked.compatibility["configuration"])):
        raise ValueError("The new measured model changed its RAW basis or parent grid proof")
    _model_composition(checked, model)
    checked.revalidate()
    return checked


def _model_composition(checked, model):
    """Rebuild saved metric/bridge algebra without decoding or fitting again."""
    from . import round5_geometry as geometry
    from .painted_adoption_geometry import _saved_frame
    saved = checked.guards.add(_identity(model["source_model"]), metadata=True, copyable=True)
    frame = _saved_frame(saved, model["run_id"])
    checked.guards.add(_identity(model["current_edges"]), copyable=True)
    photograph = _identity(model["model_photograph"])
    checked.guards.add(photograph)
    bridge = model.get("baseline_to_model_registration")
    if bridge is not None:
        checked.guards.add(_identity(bridge), metadata=True, copyable=True)
        # A retained exact copy can be read by the original builder. Its path
        # affects provenance only; every original identity stays in the model.
        bridge = checked.guards.rows[_key(bridge["path"])]["physical"]
    rebuilt = geometry.reference_model(model["run_id"], frame,
        reference_image=model["reference_image"], camera_profile=model["camera_profile"],
        source_model=model["source_model"], edge_review=model["current_edges"],
        baseline_configuration=model["baseline_configuration"],
        model_photograph=photograph, baseline_to_model=bridge)
    for key in ("H_model_photograph_u_to_metric_mm", "H_camera_u_to_metric_mm",
                "H_metric_mm_to_camera_u", "area_factor", "saved_model_certificate"):
        if not _same(model.get(key), rebuilt.get(key)):
            raise ValueError("The measured saved frame or baseline bridge changed (" + key + ")")


def preflight_geometry(index_binding, geometries, runs, *, run_progress=None):
    """Close new Transfer inputs before its first output; leave legacy callers idle."""
    selected = {str(run).casefold() for run in runs if str(run).casefold() in geometries}
    if not any(has_shape_basis(geometries[run].model) for run in selected):
        return {}
    if any(not has_shape_basis(geometries[run].model) for run in selected):
        raise ValueError("A new RAW-basis geometry index cannot mix unproved legacy native grids")
    result = {}
    for number, run in enumerate(sorted(selected), 1):
        if run_progress is not None:
            run_progress(dict(phase="raw_basis", completed=number - 1, total=len(selected), run_id=run))
        geometry = geometries[run]
        binding = geometry.entry.get("reference_model")
        _identity(binding)
        if Path(binding["path"]) != Path(geometry.source):
            raise ValueError("A RAW-basis reference model lost its literal index binding")
        checked = read_model_basis(geometry.model)
        if not _same(checked.guards.add(binding, metadata=True, copyable=True), geometry.model):
            raise ValueError("The RAW-basis measured model changed while it was selected")
        checked.guards.add(index_binding, metadata=True, copyable=True)
        checked.revalidate()
        result[run] = checked
        if run_progress is not None:
            run_progress(dict(phase="raw_basis", completed=number, total=len(selected), run_id=run))
    return result

