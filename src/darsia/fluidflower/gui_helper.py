"""The window's helper process: run the reads the window sends and send back their answers.

helper_process.HelperProcess starts this module as ``python -m
darsia.fluidflower.gui_helper`` with pipes as its standard input and output.
Each request names a module function (``"package.module:function"``) and its
arguments; the answer is its value, or the exception's type and message.
Requests run on a few threads at once, because most reads wait on files. The
process ends when the window closes the pipe or asks it to stop.

Requests and answers are pickled and sent with an 8-byte length. The pipes
are private to the window and this process. Anything else the process
writes goes to the null device, so a stray print cannot break an answer.
"""
from __future__ import annotations

import importlib
import os
import pickle
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HEADER = struct.Struct("<Q")
THREADS = 4


def write_message(stream, value) -> None:
    """Send one pickled message with its length."""
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(HEADER.pack(len(data)) + data)
    stream.flush()


def read_message(stream):
    """Read one message; EOFError when the other side has closed the pipe."""
    header = stream.read(HEADER.size)
    if len(header) < HEADER.size:
        raise EOFError("The pipe was closed")
    (size,) = HEADER.unpack(header)
    data = stream.read(size)
    if len(data) < size:
        raise EOFError("The pipe was closed during a message")
    return pickle.loads(data)


def resolve(target: str):
    """The function a request names, ``"package.module:function"``."""
    module, _, name = str(target).partition(":")
    if not module or not name:
        raise ValueError(f"Not a module function: {target!r}")
    value = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{target} is not callable")
    return value


def warm_up(*modules: str) -> None:
    """Import ``modules`` now, so the first read that needs them does not wait."""
    for module in modules:
        importlib.import_module(module)


def module_file(name: str) -> str:
    """The file a module is imported from in the helper, which shows the code it runs."""
    return importlib.import_module(name).__file__


def _machine_map(rows) -> None:
    """Read paths through the window's machine path map; the window sends it with each request."""
    from darsia.fluidflower import storage_paths

    rows = tuple(tuple(row) for row in rows)
    if rows != tuple(storage_paths.machine_path_map()):
        storage_paths.set_machine_path_map(rows)


def serve(source, sink) -> None:
    """Answer requests from ``source`` on ``sink`` until the pipe closes or a stop request."""
    lock = threading.Lock()
    map_lock = threading.Lock()

    def send(value):
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with lock:
            sink.write(HEADER.pack(len(data)) + data)
            sink.flush()

    def run(number, target, args, kwargs, machine_map):
        try:
            with map_lock:
                _machine_map(machine_map)
            value = resolve(target)(*args, **kwargs)
        except BaseException as error:  # noqa: BLE001 - every failure is the request's answer
            send((number, False, f"{type(error).__name__}: {error}"))
            return
        try:
            send((number, True, value))
        except Exception as error:  # noqa: BLE001 - an answer that cannot be pickled
            send((number, False, f"The answer could not be sent back: {type(error).__name__}: {error}"))

    pool = ThreadPoolExecutor(THREADS, thread_name_prefix="darsia-gui-helper")
    try:
        while True:
            try:
                message = read_message(source)
            except (EOFError, OSError):
                break
            if message[0] == "stop":
                break
            _kind, number, target, args, kwargs, machine_map = message
            pool.submit(run, number, target, args, kwargs, machine_map)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    # Keep the pipes for the messages; anything else written goes nowhere.
    source = os.fdopen(os.dup(sys.stdin.fileno()), "rb")
    sink = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, sys.stdin.fileno())
    os.dup2(null, sys.stdout.fileno())
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    serve(source, sink)
    # Reads still running for a closed window must not keep the process alive.
    os._exit(0)


if __name__ == "__main__":
    main()
