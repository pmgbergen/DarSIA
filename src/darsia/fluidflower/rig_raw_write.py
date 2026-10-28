"""Exclusive output ownership for fresh RAW rig operations on local or shared storage."""

from contextlib import contextmanager
import json
import os
from pathlib import Path, PureWindowsPath

from .storage_paths import resolve_storage_path


def _identity(path):
    """Read the filesystem object identity used by this operation."""
    info = path.stat()
    return (info.st_dev, info.st_ino)


def _no_links(path):
    # Check the spelling before resolve can hide an existing reparse ancestor.
    """Reject symbolic or reparse ancestors before path resolution."""
    from .pipeline.products_publication import _no_links as checked
    return checked(path)


def _canonical(path):
    """Resolve identified paths without hiding redirected ancestors."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("RAW output authority requires absolute paths without parent traversal")
    spelling = str(path).replace("/", "\\")
    if spelling.startswith(("\\\\?\\", "\\\\.\\")):
        raise ValueError("RAW output authority cannot use device path aliases")
    _no_links(path)
    if os.name == "nt" and not spelling.startswith("\\\\"):
        from .job_filesystem import _windows_local_facts, windows_mapped_drive_target
        parent = path
        while not parent.exists():
            if parent == parent.parent:
                raise ValueError("RAW output authority has no existing drive or ancestor")
            parent = parent.parent
        facts = _windows_local_facts(parent)
        if facts.local is False:
            drive = PureWindowsPath(path).drive.upper()
            path = Path(windows_mapped_drive_target(drive)) / Path(*PureWindowsPath(path).parts[1:])
            _no_links(path)
        elif facts.local is not True:
            raise ValueError("RAW output authority has no positive filesystem identity")
    return path.resolve(strict=False)


def _payload(value):
    """Encode finite, deterministic ownership metadata."""
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class OwnedRawOutput:
    """One reserved directory, one setup and one no-clobber save.

    Failed operations and their files are retained. No automatic cleanup exists.
    Directory ownership is rechecked at file boundaries. Byte limits are checked
    before every encoder write using the sizes of this owner's open streams.
    """

    def __init__(self, destination, authority, *, operation_id, request):
        """Reserve a fresh local child and bind its byte authority."""
        if not isinstance(authority, dict) or set(authority) != {
                "destination", "intended_root", "forbidden_raw_roots",
                "maximum_output_bytes", "maximum_file_bytes"}:
            raise ValueError("Exact RAW output destination/root/protected roots required")
        if any(type(authority[key]) is not int or authority[key] <= 0
               for key in ("maximum_output_bytes", "maximum_file_bytes")):
            raise ValueError("Explicit positive RAW output/file byte ceilings required")
        if authority["maximum_file_bytes"] > authority["maximum_output_bytes"]:
            raise ValueError("RAW file ceiling cannot exceed the whole operation ceiling")
        self.destination = Path(destination)
        if self.destination != Path(authority["destination"]):
            raise ValueError("RAW destination differs from its bound operation request")
        self.authority = dict(authority)
        self.root = _canonical(authority["intended_root"])
        if not self.root.is_dir():
            raise ValueError("The intended RAW output root must already exist")
        self.root_identity = _identity(self.root)
        roots = authority["forbidden_raw_roots"]
        if not isinstance(roots, list) or not roots:
            raise ValueError("Explicit protected RAW roots are required")
        self.protected = tuple(_canonical(resolve_storage_path(path)) for path in roots)
        target = _canonical(self.destination)
        if target == self.root or not target.is_relative_to(self.root):
            raise ValueError("RAW output must be a fresh child of its intended root")
        if any(target == root or target.is_relative_to(root) or root.is_relative_to(target)
               for root in self.protected):
            raise ValueError("RAW output intersects a protected acquisition root")
        from .job_filesystem import inspect_results_root, valid_results_admission
        proof = inspect_results_root(target)
        if not valid_results_admission(proof, target):
            raise ValueError("RAW rig output requires a supported local or shared filesystem")
        self.target = target
        # Keep the ownership record's historical field; the evidence explicitly
        # distinguishes local storage from an admitted SMB/NFS filesystem.
        self.local_evidence = dict(filesystem=proof.protocol, local=proof.local, evidence=proof.evidence)
        # No exist_ok: two operations cannot both reserve the same empty path.
        self.destination.mkdir()
        self.directory_identity = _identity(self.destination)
        self.operation_id = operation_id
        self.marker = _payload(dict(format="darsia-owned-raw-output-v1",
            operation_id=operation_id, request=request,
            destination=str(self.destination), intended_root=str(self.root),
            directory_identity=list(self.directory_identity),
            local_filesystem=self.local_evidence))
        if len(self.marker) > min(authority["maximum_output_bytes"], authority["maximum_file_bytes"]):
            raise ValueError("The RAW output ceiling cannot cover its ownership marker")
        with (self.destination / "raw_build_owner.json").open("xb") as stream:
            stream.write(self.marker)
        self.setup_started = self.save_started = self.save_finished = False
        self.failed = False
        self.written = set()
        self._open_streams = {}
        self._retained_bytes = len(self.marker)
        self.rig = None
        self.request = dict(request)

    def _measure_output(self):
        """Account for retained files once per file, never once per ZIP chunk."""
        total = 0
        for path in self.destination.rglob("*"):
            _no_links(path)
            if path.is_file():
                size = path.stat().st_size
                opened = self._open_streams.get(path)
                if opened is not None:
                    # Buffered encoder bytes already count even before flush.
                    size = max(size, opened.size)
                if size > self.authority["maximum_file_bytes"]:
                    raise ValueError("RAW output byte ceiling exceeded; partials retained")
                total += size
        if total > self.authority["maximum_output_bytes"]:
            raise ValueError("RAW output byte ceiling exceeded; partials retained")
        self._retained_bytes = total

    def check(self, path=None):
        """Reverify directory, drive and ownership before a write."""
        if self.failed:
            raise ValueError("RAW output operation has a retained write failure")
        _no_links(self.destination / "raw_build_owner.json")
        if (_canonical(self.authority["intended_root"]) != self.root
                or _identity(self.root) != self.root_identity
                or _canonical(self.destination) != self.target
                or _identity(self.destination) != self.directory_identity
                or tuple(_canonical(resolve_storage_path(path)) for path in self.authority["forbidden_raw_roots"]) != self.protected
                or (self.destination / "raw_build_owner.json").read_bytes() != self.marker):
            raise ValueError("RAW output ownership or an ancestor/drive mapping changed")
        path = self.destination if path is None else Path(path)
        actual = _canonical(path)
        if actual != self.target and not actual.is_relative_to(self.target):
            raise ValueError("RAW writer leaves its exclusively reserved operation directory")
        return path

    def bind(self, rig):
        """Bind this output owner to exactly one live rig object."""
        self.check()
        if self.rig is not None and self.rig is not rig:
            raise ValueError("RAW output operation already belongs to another rig")
        self.rig = rig

    def begin_setup(self, rig, log):
        """Record setup ownership before the original setup writes."""
        self.bind(rig)
        if self.setup_started:
            raise ValueError("RAW setup operation has already started")
        if Path(log) != self.destination / "figures":
            raise ValueError("RAW diagnostic output differs from its owned figures directory")
        self.setup_started = True
        with self.open(self.destination / "raw_setup_started.json") as stream:
            stream.write(_payload(dict(operation_id=self.operation_id)))
        self.check(log).mkdir()

    def require_setup(self, rig, log):
        """Require this rig's unclosed, exclusively owned setup context."""
        self.check(log)
        if self.rig is not rig or not self.setup_started or self.save_started:
            raise ValueError("RAW setup has no current exclusive operation ownership")

    def begin_save(self, rig, folder):
        """Record the original rig's single permitted archive save."""
        self.check(folder)
        if (Path(folder) != self.destination or self.rig is not rig
                or not self.setup_started or self.save_started):
            raise ValueError("RAW save requires its original rig, destination and one fresh save")
        self.save_started = True
        with self.open(self.destination / "raw_save_started.json") as stream:
            stream.write(_payload(dict(operation_id=self.operation_id)))

    @contextmanager
    def open(self, path):
        """Create one bounded file exclusively and retain failures."""
        path = self.check(path)
        if path in self.written:
            raise FileExistsError("RAW output was already written by this operation")
        self.written.add(path)
        try:
            self._measure_output()
            with path.open("xb") as stream:
                bounded = _BoundedStream(stream, self)
                self._open_streams[path] = bounded
                try:
                    yield bounded
                    bounded.flush()
                    os.fsync(stream.fileno())
                    bounded.verify(path)
                finally:
                    self._open_streams.pop(path, None)
            self.check(path)
            self._measure_output()
        except BaseException:
            self.failed = True
            raise

    def image(self, image, path):
        """Write the original image NPZ payload through the owned stream."""
        import numpy as np
        with self.open(path) as stream:
            np.savez(stream, array=image.img, metadata=image.metadata())

    def correction(self, correction, path):
        """Run the original correction serializer on a bounded stream."""
        with self.open(path) as stream:
            # Original correction.save implementations accept a NumPy archive
            # stream; curvature additionally asks for the original parent path.
            correction.save(_ArchiveStream(stream, Path(path)))

    def finish_save(self):
        """Record completion only after all owned saves succeeded."""
        self.check()
        if not self.save_started or self.save_finished:
            raise ValueError("RAW save completion is absent or repeated")
        with self.open(self.destination / "raw_save_completed.json") as stream:
            stream.write(_payload(dict(format="darsia-owned-raw-save-completed-v1",
                operation_id=self.operation_id, request=self.request,
                destination=str(self.destination),
                directory_identity=list(self.directory_identity),
                no_clobber_save=True, complete=True)))
        self.save_finished = True


class _ArchiveStream:
    def __init__(self, stream, path):
        """Give the original serializer its bounded stream and path metadata."""
        self.stream, self.path = stream, path
        self.parent, self.parents = path.parent, path.parents

    def __getattr__(self, name):
        """Delegate archive stream operations to the bounded writer."""
        return getattr(self.stream, name)

    def __str__(self):
        """Expose the original path for diagnostics without opening it."""
        return str(self.path)


class _BoundedStream:
    """Reject an over-budget write before bytes reach the reserved output tree."""
    def __init__(self, stream, owner):
        """Bind an encoder stream to the operation's output ceiling."""
        self.stream, self.owner = stream, owner
        info = os.fstat(stream.fileno())
        self.identity = info.st_dev, info.st_ino
        self.size = info.st_size

    def write(self, data):
        """Check growth in memory before writing through the already open handle."""
        if self.owner.failed:
            raise ValueError("RAW output operation has a retained write failure")
        data = memoryview(data)
        position = self.stream.tell()
        size = max(self.size, position + data.nbytes)
        growth = size - self.size
        if (size > self.owner.authority["maximum_file_bytes"]
                or self.owner._retained_bytes + growth > self.owner.authority["maximum_output_bytes"]):
            raise ValueError("RAW output byte ceiling exceeded before write; partials retained")
        # Charge before the write; a partial/error never frees reserved bytes.
        self.owner._retained_bytes += growth
        self.size = size
        count = self.stream.write(data)
        if count != data.nbytes:
            raise OSError("The bounded RAW output write was incomplete; partial retained")
        return count

    def flush(self):
        """Flush only when requested by the encoder or at the file boundary."""
        return self.stream.flush()

    def verify(self, path):
        """Reject outside edits or path replacement before accepting this file."""
        opened, named = os.fstat(self.stream.fileno()), Path(path).stat()
        if ((opened.st_dev, opened.st_ino) != self.identity
                or (named.st_dev, named.st_ino) != self.identity
                or opened.st_size != self.size or named.st_size != self.size):
            raise ValueError("The owned RAW output changed outside its writer")

    def fileno(self):
        # Encoders must use write(), never a native fd write that bypasses caps.
        """Refuse encoder shortcuts that bypass bounded write calls."""
        import io
        raise io.UnsupportedOperation("RAW owned output requires bounded writes")

    def __getattr__(self, name):
        """Expose the remaining stream interface to original serializers."""
        return getattr(self.stream, name)
