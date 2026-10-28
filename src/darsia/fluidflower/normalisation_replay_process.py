"""Run the historical lighting calculation with its own OpenCV runtime."""

import base64
from collections import deque
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread


_PREFIX = "DARSIA_NORMALISATION_REPLAY "


def _command():
    python = os.environ.get("DARSIA_FLUIDFLOWER_PYTHON") or sys.executable
    # Running this file directly keeps darsia.__init__ out of the bootstrap:
    # it disables IPP before OpenCV can initialize when invoked with -m.
    return [python, "-u", str(Path(__file__).resolve())]


def execute_replay(request, output_parent, *, progress=None, check_cancelled=None):
    """Keep historical IPP settings out of the GUI and other calculations."""
    if check_cancelled is not None:
        check_cancelled()
    document = dict(request)
    document["manifest_bytes"] = base64.b64encode(request["manifest_bytes"]).decode("ascii")
    payload = json.dumps(dict(request=document, output_parent=str(output_parent)), allow_nan=False)
    environment = dict(os.environ)
    # IPP must be available at the first OpenCV import. A GUI that started with
    # OPENCV_IPP=disabled cannot enable it later, even with setUseIPP(True).
    environment.pop("OPENCV_IPP", None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        environment[name] = "1"
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source, environment.get("PYTHONPATH"))))
    process = subprocess.Popen(
        _command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    messages, diagnostics = Queue(), deque(maxlen=12)

    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    try:
        input_error = None
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except OSError as error:
            # An import/startup failure can close stdin before a large request
            # is written. Read the child's error instead of hiding it behind
            # BrokenPipeError (or the Windows invalid-pipe OSError).
            input_error = error
            try:
                process.stdin.close()
            except OSError:
                pass
        while True:
            if check_cancelled is not None:
                check_cancelled()
            try:
                line = messages.get(timeout=.1)
            except Empty:
                continue
            if line is None:
                process.wait()
                detail = "".join(diagnostics).strip() or str(input_error or "")
                raise RuntimeError("Normalisation process ended without a result. " + detail)
            if not line.startswith(_PREFIX):
                diagnostics.append(line)
                continue
            message = json.loads(line[len(_PREFIX):])
            if message["kind"] == "progress" and progress is not None:
                progress(message["value"])
            elif message["kind"] == "error":
                raise RuntimeError(message["value"])
            elif message["kind"] == "result":
                if process.wait() != 0:
                    raise RuntimeError("Normalisation process failed after reporting its result.")
                return message["value"]
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=5)
        process.stdout.close()
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass


def _emit(kind, value):
    print(_PREFIX + json.dumps(dict(kind=kind, value=value), ensure_ascii=True), flush=True)


def _child():
    try:
        # Consume the request before dependency imports so their errors can be
        # returned even when the request is larger than the stdin pipe buffer.
        payload = json.load(sys.stdin)
        request = payload["request"]
        request["manifest_bytes"] = base64.b64decode(request["manifest_bytes"], validate=True)
        import cv2
        cv2.setNumThreads(1)
        cv2.ipp.setUseIPP(True)
        if not cv2.ipp.useIPP():
            raise RuntimeError("The original lighting calculation requires OpenCV IPP in its separate process.")
        from darsia.fluidflower.normalisation_replay import execute_replay as calculate
        # Importing darsia applies its current scientific policy. The original
        # fitter needs the historical setting only in this dedicated process.
        cv2.ipp.setUseIPP(True)
        if not cv2.ipp.useIPP():
            raise RuntimeError("The original lighting calculation could not reactivate OpenCV IPP after importing DarSIA.")
        result = calculate(request, payload["output_parent"], progress=lambda text: _emit("progress", text))
        _emit("result", result)
        return 0
    except Exception as error:
        _emit("error", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(_child())
