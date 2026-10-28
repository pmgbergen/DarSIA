"""The historical runtime is isolated and cancelled workers are reaped."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from darsia.fluidflower import normalisation_replay_process as replay


def test_child_uses_historical_ipp_without_changing_parent(monkeypatch):
    monkeypatch.setenv("OPENCV_IPP", "disabled")
    script = """
import cv2
assert cv2.ipp.useIPP()
from darsia.fluidflower import normalisation_replay as calculation
from darsia.fluidflower import normalisation_replay_process as worker
def calculate(request, output, *, progress):
    assert cv2.ipp.useIPP()
    assert cv2.getNumThreads() == 1
    assert request['manifest_bytes'] == 'original æøå'.encode()
    progress('Original runtime ready')
    return output + '/reproduced'
calculation.execute_replay = calculate
raise SystemExit(worker._child())
"""
    monkeypatch.setattr(replay, "_command", lambda: [sys.executable, "-u", "-c", script])
    progress = []
    result = replay.execute_replay(dict(manifest_bytes="original æøå".encode()), "chosen", progress=progress.append)
    assert result == "chosen/reproduced"
    assert progress == ["Original runtime ready"]
    assert os.environ["OPENCV_IPP"] == "disabled"


def test_real_file_bootstrap_reactivates_ipp_after_package_import(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCV_IPP", "disabled")
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_PYTHON", sys.executable)
    import cv2
    before = cv2.ipp.useIPP()
    assert replay._command() == [sys.executable, "-u", str(Path(replay.__file__).resolve())]
    (tmp_path / "series.toml").write_bytes(b"current series")
    # This deliberate mismatch is reached only after the real bootstrap has
    # initialized IPP, imported DarSIA, and successfully re-enabled IPP.
    with pytest.raises(RuntimeError, match="The series changed; prepare the replay again"):
        replay.execute_replay(dict(manifest_bytes=b"previous series", series_root=str(tmp_path)), tmp_path)
    assert cv2.ipp.useIPP() is before
    assert os.environ["OPENCV_IPP"] == "disabled"


def test_real_bootstrap_reads_request_before_importing_opencv(monkeypatch, tmp_path):
    monkeypatch.setenv("DARSIA_FLUIDFLOWER_PYTHON", sys.executable)
    (tmp_path / "cv2.py").write_text("raise AssertionError('OpenCV imported before decoding request')", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    # Malformed JSON must be reported before even attempting a dependency.
    result = subprocess.run(replay._command(), input="{", capture_output=True, text=True)
    assert result.returncode == 1
    assert '"kind": "error"' in result.stdout
    assert "Expecting property name enclosed in double quotes" in result.stdout
    assert "OpenCV imported before decoding request" not in result.stdout
    assert "requires OpenCV IPP" not in result.stdout


def test_cancel_stops_child_without_waiting_for_a_progress_line(monkeypatch):
    script = """
import json, sys, time
json.load(sys.stdin)
print('DARSIA_NORMALISATION_REPLAY ' + json.dumps(dict(kind='progress', value='started')), flush=True)
time.sleep(60)
"""
    monkeypatch.setattr(replay, "_command", lambda: [sys.executable, "-u", "-c", script])
    processes, progress = [], []
    popen = replay.subprocess.Popen
    def start(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(replay.subprocess, "Popen", start)
    def cancelled():
        if progress:
            raise RuntimeError("cancelled by user")
    with pytest.raises(RuntimeError, match="cancelled by user"):
        replay.execute_replay(dict(manifest_bytes=b"original"), "chosen", progress=progress.append, check_cancelled=cancelled)
    assert len(processes) == 1 and processes[0].poll() is not None


def test_child_error_is_returned_without_publishing_success(monkeypatch):
    script = """
import json, sys
json.load(sys.stdin)
print('DARSIA_NORMALISATION_REPLAY ' + json.dumps(dict(kind='error', value='Saved source changed')), flush=True)
raise SystemExit(1)
"""
    monkeypatch.setattr(replay, "_command", lambda: [sys.executable, "-u", "-c", script])
    with pytest.raises(RuntimeError, match="Saved source changed"):
        replay.execute_replay(dict(manifest_bytes=b"original"), "chosen")


@pytest.mark.parametrize("structured", [False, True])
def test_early_child_error_is_not_masked_by_broken_pipe(monkeypatch, structured):
    if structured:
        script = """
import json
print('DARSIA_NORMALISATION_REPLAY ' + json.dumps(dict(kind='error', value='Bootstrap dependency unavailable')), flush=True)
raise SystemExit(1)
"""
    else:
        script = "raise RuntimeError('Bootstrap dependency unavailable')"
    monkeypatch.setattr(replay, "_command", lambda: [sys.executable, "-u", "-c", script])
    # Larger than the pipe buffer: the child exits without reading this input.
    with pytest.raises(RuntimeError, match="Bootstrap dependency unavailable"):
        replay.execute_replay(dict(manifest_bytes=b"x" * (2 * 1024 * 1024)), "chosen")
