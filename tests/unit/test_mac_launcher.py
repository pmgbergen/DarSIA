"""The macOS launcher installs once, starts the chosen window and explains failures.

The launcher runs in a POSIX shell (Git's bash on Windows) against a copy
of itself with a stand-in environment Python that only records its calls,
so nothing is installed or started.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "start_darsia.command"


def _bash():
    """A POSIX bash, or None (the Windows system bash runs a separate Linux)."""
    found = shutil.which("bash")
    if found and "system32" not in found.casefold():
        return found
    for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.isfile(candidate):
            return candidate
    return None


def _posix(path):
    return str(path).replace("\\", "/")


@pytest.fixture
def clone(tmp_path):
    """A clone with the launcher and an environment whose Python records its calls."""
    bash = _bash()
    if bash is None:
        pytest.skip("No POSIX bash on this computer")
    root = tmp_path / "clone"
    root.mkdir()
    shutil.copyfile(LAUNCHER, root / "start_darsia.command")
    (root / "pyproject.toml").write_text("[project]\nname = 'stand-in'\n", encoding="utf-8")
    python = root / ".venv-fluidflower" / "bin" / "python"
    python.parent.mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    python.write_bytes((
        "#!/bin/bash\n"
        f'echo "$* | $VIRTUAL_ENV | $DARSIA_FLUIDFLOWER_PYTHON" >> "{_posix(calls)}"\n'
        'if [ "$1 $2" = "-m darsia.fluidflower.native_darsia_app" ] && [ -n "${STAND_IN_FAILS:-}" ]; then exit 3; fi\n'
        "exit 0\n").encode("ascii"))
    python.chmod(0o755)

    def run(*arguments, **environment):
        env = dict(os.environ, HOME=_posix(tmp_path / "home"), **environment)
        env.pop("DARSIA_PYTHON", None)
        result = subprocess.run([bash, _posix(root / "start_darsia.command"), *arguments],
                                env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        calls.unlink(missing_ok=True)
        return result, [line.split(" | ")[0] for line in lines], lines

    return root, run


def test_first_start_installs_then_later_starts_only_open_the_window(clone):
    root, run = clone
    result, calls, lines = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 3 and calls[0] == "-m pip install --upgrade pip"
    assert calls[1].startswith("-m pip install -e ") and calls[1].endswith("/clone rawpy")
    assert calls[2] == "-m darsia.fluidflower.native_darsia_app"
    assert lines[-1].split(" | ")[1].endswith(".venv-fluidflower")
    assert lines[-1].split(" | ")[2].endswith(".venv-fluidflower/bin/python")
    assert (root / ".venv-fluidflower" / ".darsia-install").is_file()
    result, calls, _lines = run("studio")
    assert result.returncode == 0 and calls == ["-m darsia.fluidflower.phase_model_app"]


def test_a_changed_project_file_installs_again(clone):
    root, run = clone
    run()
    (root / "pyproject.toml").write_text("[project]\nname = 'stand-in'\nversion = '2'\n", encoding="utf-8")
    _result, calls, _lines = run("experiments")
    assert calls[0] == "-m pip install --upgrade pip" and calls[-1] == "-m darsia.fluidflower.app"


def test_failures_are_explained(clone):
    _root, run = clone
    result, calls, _lines = run("unknown")
    assert result.returncode == 1 and "unknown window" in result.stdout and calls == []
    result, _calls, _lines = run(STAND_IN_FAILS="1")
    assert result.returncode == 1 and "closed with an error (exit status 3)" in result.stdout
