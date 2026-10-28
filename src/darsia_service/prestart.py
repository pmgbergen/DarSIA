"""Start a worker service once more when it stops before it has started.

Use::

    python -B -m darsia_service.prestart --service-log LOG [--attempts 2]
        [--delay 30] -- <the unchanged worker command ...>

The worker command runs with this process's environment, working folder and
console. A service that exits with an error before its log shows its
``started`` line has claimed nothing. The same command is therefore started
again after the delay. That restart gets a fresh service namespace (a new
cache token), and the job's queue is untouched (N64, N109).

At most ``--attempts`` starts are made in total. Some failures are never
retried:

- the same error twice;
- a stop after ``started``;
- a ``--once`` run;
- a first-qualification request.

The once-only and reference modes forbid automatic retry by their own rules.

Each failed start leaves one record in ``prestart/`` next to the service log.
A final failure appends one ``startup_failure`` line to that log, so the
launch reservation in the jobs view is released. This module imports no DarSIA
code, so it also covers a failure while DarSIA loads.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RECORD_FORMAT = "darsia-prestart-attempt-v1"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _size(path):
    try:
        return Path(path).stat().st_size
    except FileNotFoundError:
        return 0


def _new_lines(log, offset):
    try:
        with Path(log).open("rb") as stream:
            stream.seek(offset)
            raw = stream.read()
    except FileNotFoundError:
        return []
    return raw.decode("utf-8", errors="replace").splitlines()


def service_events(log, offset):
    """Return the JSON events a service wrote to its log after a byte offset.

    The worker writes ``<utc> <json>`` lines and raw traceback text. Only the
    JSON lines are events.
    """
    events = []
    for line in _new_lines(log, offset):
        _stamp, space, encoded = line.partition(" ")
        if not space:
            continue
        try:
            event = json.loads(encoded)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def failure_summary(log, offset, exit_code):
    """Return one line naming a failed start.

    That is the last line the service logged after the offset, or else its
    exit code.
    """
    lines = [line.strip() for line in _new_lines(log, offset) if line.strip()]
    if lines:
        return lines[-1][-500:]
    return f"exit code {exit_code} before the service wrote its log"


def retry_refusal(command, environment):
    """Return why a failed start must not be retried automatically, or None."""
    if "--once" in command:
        return "--once: a once-only run is never restarted automatically"
    if environment.get("DARSIA_CACHE_FIRST_QUALIFICATION"):
        return "first-qualification request: its own rules forbid automatic retry"
    return None


def _write_record(path, document):
    """Create one record exclusively, never replacing a file; return its descriptor.

    DarSIA's own immutable writer cannot be imported here: this module must run
    while DarSIA itself fails to load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def _append_line(log, event):
    """Append one ``<utc> <json>`` line, as the worker writes its own events."""
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write((_utc() + " " + json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def supervise(command, *, service_log, attempts=2, delay=30.0, run=None, sleep=time.sleep,
              environment=None):
    """Run the worker command, and start it again after a failure before it started.

    Return the exit code of the last start. The final line is appended to the
    service log only after this process's own service has exited. Until then
    the open launch reservation admits no other start of that machine and
    backend.
    """
    if run is None:
        def run(arguments):
            # Explicit standard handles pass this process's console on to the worker.
            return subprocess.call(arguments, stdin=subprocess.DEVNULL)
    environment = os.environ if environment is None else environment
    command = list(command)
    log = Path(service_log)
    stem = f"{log.stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{os.getpid()}"
    previous = None
    for attempt in range(1, attempts + 1):
        offset = _size(log)
        started_utc = _utc()
        code = run(command)
        if code == 0 or any(event.get("kind") == "started"
                            for event in service_events(log, offset)):
            return code  # It started: its watchdog and queue own any later stop.
        error = failure_summary(log, offset, code)
        refusal = retry_refusal(command, environment)
        repeated = previous is not None and error == previous["error"]
        last = attempt == attempts or refusal is not None or repeated
        record = dict(
            format=RECORD_FORMAT, attempt=attempt, max_attempts=attempts, command=command,
            exit_code=code, started_utc=started_utc, ended_utc=_utc(), error=error,
            service_log=str(log), service_log_offset=offset,
            service_started=False, retried=not last, retry_refused=refusal,
            same_error_repeated=repeated,
            previous_record=None if previous is None else previous["record"])
        try:
            receipt = _write_record(log.parent / "prestart" / f"{stem}-attempt-{attempt}.json", record)
        except OSError as failure:
            receipt = None
            print(f"{_utc()} prestart record not written: {failure!r}", flush=True)
        print(f"{_utc()} prestart failure {attempt}/{attempts}: {error}", flush=True)
        if last:
            try:
                _append_line(log, dict(kind="startup_failure", source="darsia_service.prestart",
                                       attempts=attempt, error=error, record=receipt))
            except OSError as failure:
                print(f"{_utc()} startup_failure line not written: {failure!r}", flush=True)
            return code
        previous = dict(error=error, record=receipt)
        sleep(delay)
    raise AssertionError("unreachable")


def main(argv=None):
    """Command line: options, then ``--``, then the unchanged worker command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -B -m darsia_service.prestart",
        description="Start a worker service again when it stops before it has started.")
    parser.add_argument("--service-log", required=True, type=Path,
                        help="The worker's own log, <job>/logs/<machine>-<backend>.log")
    parser.add_argument("--attempts", type=int, default=2, help="Starts in total (default 2)")
    parser.add_argument("--delay", type=float, default=30.0, help="Seconds before a new start")
    if "--" not in argv:
        parser.error("put the worker command after --")
    split = argv.index("--")
    options = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        parser.error("put the worker command after --")
    if options.attempts < 1 or not options.delay >= 0:
        parser.error("--attempts must be positive and --delay nonnegative")
    return supervise(command, service_log=options.service_log, attempts=options.attempts,
                     delay=options.delay)


if __name__ == "__main__":
    sys.exit(main())
