"""Facts about a machine and its Python environment, without names.

The installed-environment fingerprint lists every installed distribution with
its version and a SHA-256 over its installed files, so two environments can
be compared file for file. CPU, operating system and instruction-set facts
describe a machine without its host or user name.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import platform
import re

FINGERPRINT_FORMAT = "darsia-environment-fingerprint-v1"
# Installation bookkeeping, not code: it may hold local paths and changes per install.
_SKIPPED_FILES = {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}


def cpu_model() -> str:
    """The CPU model name, or the generic processor identifier when it is unknown."""
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.casefold().startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        import subprocess

        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "Not recorded"


def instruction_sets() -> dict:
    """The CPU's vector instruction sets as NumPy detects them, with a short summary."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:  # pragma: no cover - older NumPy layout
        from numpy.core._multiarray_umath import __cpu_features__ as features
    supported = sorted(name for name, present in features.items() if present)
    if "AVX512F" in supported:
        summary = "AVX-512"
    elif "AVX2" in supported:
        summary = "AVX2"
    elif "ASIMD" in supported:
        summary = "ASIMD (Arm NEON)"
    elif "AVX" in supported:
        summary = "AVX"
    else:
        summary = "SSE or older" if any(name.startswith("SSE") for name in supported) else "Not recorded"
    return {"summary": summary, "features": supported}


def platform_facts() -> dict:
    """Operating system, architecture and Python version of this machine."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def machine_facts() -> dict:
    """The facts a provenance record or report shows about this machine (no names)."""
    from ..execution import detect_host_resources

    resources = detect_host_resources()
    cpu = dict(resources["cpu"])
    cpu.pop("processor", None)
    return {
        "platform": platform_facts(),
        "cpu": {**cpu, "model": cpu_model(), "instruction_sets": instruction_sets()},
        "memory": resources["memory"],
    }


def distribution_fingerprint(*, distributions=None, hash_files: bool = True) -> dict:
    """Every installed distribution's version and a SHA-256 over its installed files.

    Each file contributes its path inside the distribution and the SHA-256 of
    its bytes (with ``hash_files``) or of its recorded RECORD hash (without).
    Compiled ``.pyc`` files and installation bookkeeping are left out.
    """
    rows = []
    for distribution in (importlib.metadata.distributions() if distributions is None else distributions):
        name = distribution.metadata.get("Name") or "unknown"
        entries = []
        for item in distribution.files or ():
            relative = PurePosixPath(str(item).replace("\\", "/"))
            if relative.suffix == ".pyc" or relative.name in _SKIPPED_FILES:
                continue
            if hash_files:
                location = Path(distribution.locate_file(item))
                try:
                    digest = _file_digest(location)
                except OSError:
                    digest = "missing"
            else:
                digest = item.hash.value if item.hash is not None else "not recorded"
            entries.append(f"{relative.as_posix()}\x00{digest}\n")
        entries.sort()
        rows.append({
            "name": re.sub(r"[-_.]+", "-", name).casefold(),
            "version": distribution.version,
            "files": len(entries),
            "sha256": hashlib.sha256("".join(entries).encode("utf-8")).hexdigest(),
        })
    rows.sort(key=lambda row: (row["name"], row["version"]))
    summary = hashlib.sha256(
        "".join(f"{row['name']}\x00{row['version']}\x00{row['sha256']}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    return {"format": FINGERPRINT_FORMAT, "python": platform.python_version(),
            "file_digests": "content" if hash_files else "recorded", "distributions": rows,
            "sha256": summary}


def _file_digest(path: Path) -> str:
    """The SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_fingerprints(expected: dict, actual: dict) -> list[str]:
    """Differences between two fingerprints, one line per distribution."""
    left = {row["name"]: row for row in expected.get("distributions", ())}
    right = {row["name"]: row for row in actual.get("distributions", ())}
    lines = []
    if expected.get("python") != actual.get("python"):
        lines.append(f"Python {expected.get('python')} here {actual.get('python')}")
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a is None:
            lines.append(f"{name}: not in the published environment (installed {b['version']})")
        elif b is None:
            lines.append(f"{name}: not installed here (published {a['version']})")
        elif a["version"] != b["version"]:
            lines.append(f"{name}: version {a['version']} published, {b['version']} here")
        elif a["sha256"] != b["sha256"] and expected.get("file_digests") == actual.get("file_digests"):
            lines.append(f"{name}: same version {a['version']}, other installed files (another build)")
    return lines


def lock_summary(path: str | Path) -> dict:
    """The worker lock file: its SHA-256, the number of exact pins and the Python line."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    pins = [line for line in text.splitlines() if re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==\S+", line.strip())]
    python = re.search(r"Python\s+(\d+\.\d+)", text)
    return {"path": Path(path).name, "sha256": hashlib.sha256(raw).hexdigest(),
            "pins": len(pins), "python": python.group(1) if python else None,
            "hashes": "--hash" in text}
