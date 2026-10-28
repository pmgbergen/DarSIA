"""Pure pre-import environment derivation for a fixed host launcher.

Importing this standalone module does not import darsia/Numba/Matplotlib, create
directories, reserve space, launch a process, or prove a hard quota. The fixed
host supervisor must first admit/create the exact scope and then pass this
environment to its unchanged standard worker/test command. No per-run program.
"""
from pathlib import PureWindowsPath
import re


WRITER_SUBDIRECTORIES = ("tmp", "numba", "matplotlib", "pycache", "display", "cupy")
WRITER_ENVIRONMENT_KEYS = ("TEMP", "TMP", "TMPDIR", "NUMBA_CACHE_DIR",
    "NUMBA_CACHE_LOCATOR_CLASSES", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX",
    "DARSIA_FLUIDFLOWER_CACHE", "DARSIA_PHASE_DISPLAY_CACHE", "CUPY_CACHE_DIR")


def derive_cache_environment(stores_document, *, purpose, token, scope_root, parent_environment):
    """Return new child environment; metadata alone grants no allocation right."""
    if purpose not in {"jobs", "gates", "tests", "display", "logs"}:
        raise ValueError("A disposable purpose is required; Git clones are separate")
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("Exact admitted process-scope token required")
    cache = stores_document.get("cache")
    if (not isinstance(cache, dict) or type(cache.get("quota_bytes")) is not int
            or cache["quota_bytes"] <= 0 or not isinstance(cache.get("profiles"), dict)
            or purpose not in cache["profiles"]):
        raise ValueError("An explicit measured purpose profile and quota are required")
    root = PureWindowsPath(cache["root"])
    if (not root.is_absolute() or re.fullmatch(r"[A-Za-z]:", root.drive) is None
            or ".." in root.parts or len(root.parts) < 2):
        raise ValueError("Normalized local non-volume-root cache required")
    expected = root / purpose / ("current" if purpose == "tests" else "") / token
    if str(PureWindowsPath(scope_root)).casefold() != str(expected).casefold():
        raise ValueError("Child cache path differs from the exact admitted scope")
    result = dict(parent_environment)
    for key in ("TEMP", "TMP", "TMPDIR"):
        result[key] = str(expected / "tmp")
    result.update(NUMBA_CACHE_DIR=str(expected / "numba"),
                  NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator",
                  MPLCONFIGDIR=str(expected / "matplotlib"),
                  CUPY_CACHE_DIR=str(expected / "cupy"),
                  PYTHONPYCACHEPREFIX=str(expected / "pycache"),
                  DARSIA_FLUIDFLOWER_CACHE=str(expected),
                  DARSIA_PHASE_DISPLAY_CACHE=str(expected / "display"))
    # Do not change HOME/LOCALAPPDATA, JIT, numerical threads or scientific args.
    return result


# Q102: only the explicitly released finite first provider redirects these
# inherited application locations. Ordinary direct-profile behavior above is
# unchanged. These paths are not an allocation cap or a reserve guarantee.
FIRST_WRITER_SUBDIRECTORIES = WRITER_SUBDIRECTORIES + (
    "xdg-cache", "xdg-config", "xdg-data", "joblib",
    "userprofile", "appdata", "localappdata", "home",
)
FIRST_WRITER_ENVIRONMENT_KEYS = WRITER_ENVIRONMENT_KEYS + (
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    "JOBLIB_TEMP_FOLDER", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOME",
)


def first_qualification_environment(environment, *, scope_root):
    """Extend the original exact scope mapping; no mkdir/global environment I/O."""
    root = PureWindowsPath(scope_root)
    if not root.is_absolute() or root.drive.startswith("\\\\") or ".." in root.parts:
        raise ValueError("First writers require the actual owned local namespace")
    if str(PureWindowsPath(environment["DARSIA_FLUIDFLOWER_CACHE"])).casefold() != str(root).casefold():
        raise ValueError("First writer environment differs from the original service scope")
    result = dict(environment)
    result.update(XDG_CACHE_HOME=str(root / "xdg-cache"),
                  XDG_CONFIG_HOME=str(root / "xdg-config"),
                  XDG_DATA_HOME=str(root / "xdg-data"),
                  JOBLIB_TEMP_FOLDER=str(root / "joblib"),
                  USERPROFILE=str(root / "userprofile"),
                  APPDATA=str(root / "appdata"),
                  LOCALAPPDATA=str(root / "localappdata"),
                  HOME=str(root / "home"))
    for key in FIRST_WRITER_ENVIRONMENT_KEYS:
        if key == "NUMBA_CACHE_LOCATOR_CLASSES":
            continue
        path = PureWindowsPath(result[key])
        if not path.is_relative_to(root):
            raise ValueError("A first cache/temp writer escapes its owned service namespace: " + key)
    return result
