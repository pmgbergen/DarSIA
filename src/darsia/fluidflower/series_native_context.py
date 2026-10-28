"""Native UI ownership of a series' explicit parameter document.

No parameter lookup takes place on module import. This adapter reads only the
selected manifest and its SHA-bound JSON; the original configuration API owns
all native parameter validation and numerical interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import os
from pathlib import Path, PureWindowsPath
import re
import tomllib


_ENVIRONMENT_KEYS = ("DARSIA_NATIVE_CONFIGURATION", "DARSIA_NATIVE_CONFIGURATION_SHA256")


@dataclass(frozen=True)
class SeriesNativeBinding:
    series_root: Path
    path: Path
    sha256: str
    document: Mapping
    current_code_bindings: tuple[Mapping, ...] = ()


# These are the retained saved-reader APIs, never numerical source locators.
_SAVED_READER_MODULES = {
    "mass_freeze": "mass_saved_metadata.py",
    "physics_profiles": "phase_saved_review_metadata.py",
    "selected_policy_review": "phase_selected_review.py",
    "selected_review": "phase_selected_review.py",
    "training_frame_context": "phase_training/frame_context.py",
}


def _bind_current_saved_readers(parameters):
    """Relocate only exact byte-bound saved-reader code into this package.

    The series document and all historical/data/source identities stay intact.
    Only a runtime copy's current-code path is adapted, after the local module
    has passed its original SHA and size. No old checkout is read or imported.
    """
    document = deepcopy(parameters)
    saved = document.get("saved_review_metadata", {})
    if not isinstance(saved, Mapping):
        raise ValueError("saved_review_metadata must be a table")
    package = Path(__file__).resolve().parent
    evidence = []
    for section, settings in saved.items():
        if not isinstance(settings, Mapping) or "current_code_source" not in settings:
            continue
        if section not in _SAVED_READER_MODULES:
            raise ValueError("Unsupported saved-reader current-code binding: " + section)
        declared = settings["current_code_source"]
        if not isinstance(declared, Mapping) or set(declared) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Saved-reader current code requires exact path/SHA/size")
        relative = _SAVED_READER_MODULES[section]
        suffix = ("src", "darsia", "fluidflower", *relative.split("/"))
        old_path = declared["path"]
        if (not isinstance(old_path, str)
                or tuple(x.casefold() for x in PureWindowsPath(old_path).parts[-len(suffix):])
                != tuple(x.casefold() for x in suffix)):
            raise ValueError("Saved-reader declared module role differs: " + section)
        digest, size = declared["sha256"], declared["size_bytes"]
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(size) is not int or size < 1):
            raise ValueError("Saved-reader exact code identity is invalid: " + section)
        local = (package / relative).resolve(strict=True)
        if not local.is_relative_to(package) or not local.is_file():
            raise ValueError("Saved-reader current module escapes its package")
        raw = local.read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Saved-reader current implementation changed: " + section)
        settings["current_code_source"] = dict(declared, path=str(local))
        evidence.append(dict(section=section, declared=dict(declared), actual=dict(path=str(local),
            sha256=digest, size_bytes=size), binding="exact-original-byte-identity-current-package-path"))
    return document, tuple(evidence)


@contextmanager
def saved_reader_ui_context():
    """Bind the same exact saved-reader runtime copy in an owned UI entry.

    No document is invented and no process environment or file is rewritten.
    An absent native document stays absent; a malformed explicit pair fails.
    """
    from .native.config import require_configuration, configure_parameters, reset_parameters
    try:
        document = require_configuration()
    except ValueError as error:
        if (str(error) != "An explicit native parameter document is required"
                or any(os.environ.get(key) for key in _ENVIRONMENT_KEYS)):
            raise
        yield ()
        return
    runtime, evidence = _bind_current_saved_readers(document)
    token = configure_parameters(runtime)
    try:
        yield evidence
    finally:
        reset_parameters(token)


def saved_reader_ui_entrypoint(function):
    """Use the current explicit document only for this retained UI invocation."""
    @wraps(function)
    def invoke(*args, **kwargs):
        with saved_reader_ui_context():
            return function(*args, **kwargs)
    return invoke


def read_series_native_binding(series_root: str | Path | None) -> SeriesNativeBinding | None:
    """Load a complete explicit binding, or return None for an absent table.

    A present malformed table/pair/digest is an error, not the no-native case.
    This function has no context or environment side effects.
    """
    found = _series_native_parameters(series_root)
    if found is None:
        return None
    root, path, digest, parameters = found
    runtime_parameters, code_bindings = _bind_current_saved_readers(parameters)
    return SeriesNativeBinding(root, path, digest, runtime_parameters, code_bindings)


def _series_native_parameters(series_root):
    """The series.toml [native] document: (root, path, sha256, parameters), or None."""
    if series_root is None:
        return None
    root = Path(series_root).expanduser().resolve()
    manifest = root / "series.toml"
    if not manifest.is_file():
        return None
    with manifest.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError(f"Series manifest exceeds the metadata limit: {manifest}")
    document = tomllib.loads(raw.decode("utf-8-sig"))
    if "native" not in document:
        return None
    native = document["native"]
    if not isinstance(native, Mapping):
        raise ValueError("series.toml [native] must be a table")
    path_text, digest = native.get("configuration"), native.get("sha256")
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError("series.toml [native].configuration must name a parameter document")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("series.toml [native].sha256 must be an exact SHA-256 digest")
    path = Path(path_text).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    from .native.config import load_parameters

    return root, path, digest, load_parameters(path, digest)


@dataclass(frozen=True)
class CameraUProfile:
    """A run's configured camera-U profile: the SHA-bound document and its id."""

    path: Path
    sha256: str
    profile_id: str
    camera_wh: tuple[int, int]


def camera_u_profile(series_root: str | Path | None, run_id: str) -> CameraUProfile | None:
    """Return the run's camera-U profile from series.toml [native], or None.

    The series' SHA-bound native parameter document declares the camera
    profiles (registration.common.profiles), the size of the camera grid U
    (registration.common.sensor.camera_wh) and each run's profile
    (source.profiles). A series without them, or a run without a profile, has
    no camera-U rendering. A run naming an undeclared profile is an error.
    """
    # Only the SHA-bound document is read; no saved-reader code is bound here.
    found = _series_native_parameters(series_root)
    if found is None:
        return None
    _root, path, digest, document = found
    source = document.get("source")
    registration = document.get("registration", document)
    common = registration.get("common") if isinstance(registration, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(common, Mapping):
        return None
    profiles = source.get("profiles")
    if not isinstance(profiles, Mapping):
        return None
    by_run = {str(key).casefold(): value for key, value in profiles.items()}
    profile_id = by_run.get(str(run_id).casefold())
    if profile_id is None:
        return None
    declared = common.get("profiles")
    if not isinstance(declared, Mapping) or profile_id not in declared:
        raise ValueError(f"Run {run_id} names camera profile {profile_id!r}, which is not declared")
    width, height = common["sensor"]["camera_wh"]
    return CameraUProfile(path, digest, str(profile_id), (int(width), int(height)))


class SeriesNativeContext:
    """One native window's main-context token and owned environment pair.

    Preparation precedes switching, so a refused/invalid target cannot displace
    the currently accepted series. Background catalogue workers copy the main
    context at launch; changing the process pair does not change those copies.
    """

    def __init__(self):
        self.binding = None
        self._token = None
        self._prior_environment = None
        self._installed_environment = None

    def bind(self, series_root):
        return self.apply(read_series_native_binding(series_root))

    def apply(self, binding):
        """Install an already verified binding in the calling GUI context; no I/O."""
        if (binding is not None and self.binding is not None
                and binding.series_root == self.binding.series_root
                and binding.path == self.binding.path
                and binding.sha256 == self.binding.sha256
                and binding.document == self.binding.document):
            return self.binding
        # A bad target never reaches this point: the old accepted binding stays.
        self.close()
        if binding is None:
            return None
        from .native.config import configure_parameters

        self._prior_environment = {key: os.environ.get(key) for key in _ENVIRONMENT_KEYS}
        self._token = configure_parameters(binding.document)
        self._installed_environment = dict(zip(_ENVIRONMENT_KEYS, (str(binding.path), binding.sha256)))
        os.environ.update(self._installed_environment)
        self.binding = binding
        return binding

    def close(self):
        if self._token is not None:
            from .native.config import reset_parameters

            reset_parameters(self._token)
            self._token = None
        if self._prior_environment is not None:
            # Do not overwrite another owner's later environment update.
            if all(os.environ.get(key) == value for key, value in self._installed_environment.items()):
                for key, value in self._prior_environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        self.binding = None
        self._prior_environment = self._installed_environment = None
