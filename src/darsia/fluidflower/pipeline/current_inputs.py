"""Read current Build inputs through the user's declared storage relocations."""
from pathlib import Path
import ntpath

from ..mass.n1_pipeline import DataReader

_SHARED_FOLDERS = {}


def require_shared_input(path):
    """Refuse an input that other machines may not be able to read.

    Shared needs positive evidence: the operating system reports the file's
    folder as an SMB or NFS share (job_filesystem.inspect_shared_folder,
    remembered per folder). A local disk, a synchronization-client folder
    and a folder that cannot be inspected all refuse.
    """
    from ..job_filesystem import inspect_shared_folder
    folder = str(Path(path).parent)
    if folder not in _SHARED_FOLDERS:
        try:
            _SHARED_FOLDERS[folder] = inspect_shared_folder(folder).allowed
        except (OSError, ValueError):
            _SHARED_FOLDERS[folder] = False
    if not _SHARED_FOLDERS[folder]:
        raise ValueError(str(path) + " is not on a shared network folder, so other machines may not read it. "
                         "Copy it into the series project folder and add a storage alias for it, "
                         "or run this job on one machine.")


class CurrentInputReader:
    def __init__(self, reader=None, *, series_roots=(), shared_only=False):
        from ..storage_paths import storage_path_remapper
        self.reader = reader or DataReader()
        self.remap = (storage_path_remapper(series_roots=series_roots)
                      if series_roots else storage_path_remapper())
        # A job that any connected machine may run reads shared inputs only.
        self.shared_only = shared_only

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def resolve_path(self, path):
        return Path(self.remap(str(path)))

    def physical_descriptor(self, bound):
        physical = self._physical(bound)
        if self.shared_only:
            require_shared_input(physical["path"])
        return physical

    def _physical(self, bound):
        binding = getattr(self.reader, "io_binding", None)
        if binding is not None:
            physical = binding.physical_descriptor(bound)
            # Original Setup leaves scientific descriptors outside its native
            # catalogue to the original SHA-checking data reader. Those still
            # need the user's declared storage relocations, just as they do
            # outside a worker. Exact native admission keeps its own locator.
            if (getattr(binding, "series_setup_original_reader", False)
                    and not binding._native_input(bound["path"])):
                physical = dict(physical, path=str(self.resolve_path(physical["path"])))
            return physical
        physical = getattr(self.reader, "aliases", {}).get(
            ntpath.normcase(ntpath.normpath(bound["path"])), bound)
        return dict(bound, **{key: value for key, value in physical.items() if key != "path"},
                    path=str(self.resolve_path(physical["path"])))

    def path(self, bound):
        return self.reader.path(self.physical_descriptor(bound))

    def json(self, bound):
        return self.reader.json(self.physical_descriptor(bound))

    def array(self, bound):
        return self.reader.array(self.physical_descriptor(bound))


def current_input_reader(reader=None, *, shared_only=False):
    """The current input reader; ``shared_only`` reads only from shared network folders."""
    if isinstance(reader, CurrentInputReader):
        return reader
    return CurrentInputReader(reader, shared_only=shared_only)
