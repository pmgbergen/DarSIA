"""Start CPU worker processes that also work when the parent has no console.

A Windows GUI runs ``pythonw``, and its spawned pool workers run ``pythonw``
too, without standard error. An inherited ``PYTHONFAULTHANDLER`` then stops
each worker at interpreter start, before any task runs, which a process pool
reports as BrokenProcessPool. :func:`process_pool` starts such workers without
that setting and gives each worker without standard error a log file for its
error output and fault reports.

:func:`run_task_processes` runs each task in its own process instead, so the
count can change while running and one task can be stopped without the
others. Its tasks send progress events with :func:`report`;
:class:`ThroughputTuner` and :class:`MemoryGuard` use them to choose the count.
"""
from __future__ import annotations

import contextlib
import faulthandler
import itertools
import os
from pathlib import Path
import pickle
import statistics
import sys
import time

_events = None


def worker_log(folder, events=None):
    """Pool initializer: send a worker's error output to ``folder`` when it has none.

    A worker that already has standard error keeps it. The log file is named
    after the worker's process id and stays open for the worker's lifetime.
    ``events`` is the pool's progress queue, used by :func:`report`.
    """
    global _events
    _events = events
    if sys.stderr is not None or not folder:
        return
    Path(folder).mkdir(parents=True, exist_ok=True)
    stream = open(Path(folder) / f"worker-{os.getpid()}.log", "a", encoding="utf-8", buffering=1)
    sys.stderr = stream
    faulthandler.enable(file=stream, all_threads=True)


def console_less_workers():
    """Return whether spawned workers would start without a console (``pythonw``)."""
    from multiprocessing import spawn
    return Path(spawn.get_executable()).name.lower().startswith("pythonw")


def report(event):
    """Send a progress event from a task process to its runner, if it listens."""
    if _events is not None:
        _events.put(event)


@contextlib.contextmanager
def process_pool(count, *, log_folder=None):
    """Yield a spawn ProcessPoolExecutor with ``count`` workers that can start without a console.

    While the pool exists, ``PYTHONFAULTHANDLER`` is withheld from console-less
    workers; the parent's own fault handler, enabled at its start, is
    unaffected. ``log_folder`` receives the logs of workers without standard
    error; it is created only when such a worker starts.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    saved = os.environ.pop("PYTHONFAULTHANDLER", None) if console_less_workers() else None
    try:
        with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn"),
                                 initializer=worker_log,
                                 initargs=(None if log_folder is None else str(log_folder),)) as pool:
            yield pool
    finally:
        if saved is not None:
            os.environ["PYTHONFAULTHANDLER"] = saved


def _run_task(function, argument, index, results, events, log_folder):
    """Task process entry: run ``function(argument)`` and send the outcome to ``results``."""
    worker_log(log_folder, events)
    try:
        value = function(argument)
    except BaseException as error:  # the runner raises it in the parent
        import traceback
        text = "".join(traceback.format_exception(error))
        try:
            pickle.dumps(error)
        except Exception:
            error = RuntimeError(f"{type(error).__name__}: {error}")
        results.put((index, False, error, text))
        return
    results.put((index, True, value, ""))


def _start(process):
    """Start a spawn process without the console-only fault handler setting."""
    saved = os.environ.pop("PYTHONFAULTHANDLER", None) if console_less_workers() else None
    try:
        process.start()
    finally:
        if saved is not None:
            os.environ["PYTHONFAULTHANDLER"] = saved


def run_task_processes(function, tasks, *, admit, critical_bytes, log_folder=None,
                       on_event=None, available_memory=None, poll_seconds=0.5, settle_seconds=5.0):
    """Yield ``(key, value)`` for each finished task, each run in its own spawn process.

    ``tasks`` are ``(key, argument)`` pairs and ``function`` is a top-level
    function of one argument. ``admit(running)`` returns how many tasks may run
    now. ``on_event(event, running)`` receives the tasks' :func:`report` events.

    When available memory falls below ``critical_bytes``, the most recently
    started task is stopped at once and put first in the queue, and the count
    stays below the count it had; the last running task is never stopped, and
    after a stop the next one waits ``settle_seconds``. ``on_event`` hears each
    stop as ``{"state": "stopped_for_memory", "task": key}``.

    A failed task does not stop the others: every task runs, each finished one
    is yielded as it completes, and the failures are raised together at the
    end as :class:`TaskFailures`. ``available_memory`` returns the available
    bytes (a test seam).
    """
    import queue as queues
    from multiprocessing import get_context
    if available_memory is None:
        import psutil
        available_memory = lambda: psutil.virtual_memory().available
    context = get_context("spawn")
    results, events = context.Queue(), context.Queue()
    folder = None if log_folder is None else str(log_folder)
    waiting, running, numbers = list(tasks), {}, itertools.count()
    finished, failures, ceiling, last_stop = {}, {}, None, None
    try:
        while running or waiting:
            allowed = admit(len(running)) if ceiling is None else min(admit(len(running)), ceiling)
            if not running:
                allowed = max(1, allowed)  # always make progress
            while waiting and len(running) < allowed:
                key, argument = waiting.pop(0)
                index = next(numbers)
                process = context.Process(target=_run_task, name=f"task-{index}",
                                          args=(function, argument, index, results, events, folder))
                _start(process)
                running[index] = (key, argument, process)
            try:
                outcome = results.get(timeout=poll_seconds)
            except queues.Empty:
                outcome = None
            while outcome is not None:
                finished[outcome[0]] = outcome
                try:
                    outcome = results.get_nowait()
                except queues.Empty:
                    outcome = None
            while True:
                try:
                    event = events.get_nowait()
                except queues.Empty:
                    break
                if on_event is not None:
                    on_event(event, len(running))
            for index in [index for index in finished if index in running]:
                _, ok, value, text = finished.pop(index)
                key, _, process = running.pop(index)
                process.join()
                if ok:
                    yield key, value
                else:
                    failures[key] = (value, text)
            for index, (key, _, process) in list(running.items()):
                if process.exitcode is not None and index not in finished:
                    time.sleep(poll_seconds)  # the outcome may still be in the pipe
                    if index not in finished and results.empty():
                        running.pop(index)
                        failures[key] = (RuntimeError(
                            f"The task stopped without a result (exit code {process.exitcode})"), "")
            now = time.monotonic()
            if (len(running) > 1 and available_memory() < critical_bytes
                    and (last_stop is None or now - last_stop >= settle_seconds)):
                index = max(running)
                key, argument, process = running.pop(index)
                process.kill()
                process.join()
                waiting.insert(0, (key, argument))
                ceiling, last_stop = len(running), now
                if on_event is not None:
                    on_event(dict(state="stopped_for_memory", task=key), len(running))
        if failures:
            raise TaskFailures(failures, len(tasks))
    finally:
        for _, _, process in running.values():
            process.kill()
            process.join()
        results.close()
        events.close()


class TaskFailures(RuntimeError):
    """The tasks that failed in :func:`run_task_processes`, raised after all tasks ran.

    ``failures`` maps each failed task's key to its error and traceback text.
    """

    def __init__(self, failures, total):
        self.failures = dict(failures)
        lines = [f"{key}: {type(error).__name__}: {error}" for key, (error, _) in self.failures.items()]
        super().__init__(f"{len(self.failures)} of {total} tasks failed; the others finished.\n"
                         + "\n".join(lines))
        for key, (_, text) in self.failures.items():
            if text:
                self.add_note(f"{key}:\n{text.rstrip()}")


class MemoryGuard:
    """Admit another worker only while available memory covers the largest worker seen.

    The largest resident size of this process's children over the whole job
    stands for what one more worker will need, so a demanding stage that
    has run once keeps limiting later admissions. ``probe`` returns the
    available bytes and the children's resident sizes (a test seam).
    """

    def __init__(self, reserve_bytes, *, probe=None):
        self.reserve = int(reserve_bytes)
        self.peak = 0
        self._probe = probe or _memory_probe

    def allows_another(self):
        """Measure now; return whether one more worker fits beside the reserve."""
        available, sizes = self._probe()
        self.peak = max([self.peak, *sizes])
        return available >= self.reserve + self.peak


def _memory_probe():
    import psutil
    sizes = []
    for child in psutil.Process().children(recursive=True):
        try:
            sizes.append(child.memory_info().rss)
        except psutil.Error:
            continue
    return psutil.virtual_memory().available, sizes


class ThroughputTuner:
    """Choose how many tasks run at once from the measured rate of finished work.

    The rate at a concurrency is that concurrency divided by the median time
    of the work units (for example image frames) that started and finished
    while exactly that many tasks ran. The tuner starts at ``start`` and adds
    ``step`` while each addition raises the rate by at least ``gain``. When an
    addition does not, it returns to the best count and keeps it. A
    concurrency is judged after ``samples`` units, and at least one unit per
    running task, have finished at it.

    A new kind of unit (another stage of the job) starts the search again from
    the current count, and so does a settled search after ``reprobe_seconds``,
    so the count follows changes in the work and in the host.
    """

    def __init__(self, start, maximum, *, step=2, gain=0.03, samples=8, reprobe_seconds=900.0):
        self.maximum = max(1, int(maximum))
        self.target = max(1, min(int(start), self.maximum))
        self.step, self.gain, self.samples = max(1, int(step)), float(gain), max(1, int(samples))
        self.reprobe_seconds = float(reprobe_seconds)
        self.rates = {}
        self._kind = None
        self._restart(None)

    def _restart(self, now):
        self.settled = self.target >= self.maximum
        self._settled_at = now
        self._durations = {}
        self._best = None

    def _settle(self, target, now):
        self.target, self.settled, self._settled_at = target, True, now

    def unit_done(self, seconds, concurrency, *, kind="unit", now=0.0):
        """Record one finished unit that ran while ``concurrency`` tasks ran; return the target.

        ``kind`` names the stage the unit belongs to and ``now`` is a
        monotonic time in seconds.
        """
        if kind != self._kind:
            self._kind = kind
            self._restart(now)
        elif (self.settled and self.target < self.maximum and self._settled_at is not None
                and now - self._settled_at >= self.reprobe_seconds):
            self._restart(now)
        if self.settled or concurrency != self.target or seconds <= 0:
            return self.target
        durations = self._durations.setdefault(concurrency, [])
        durations.append(float(seconds))
        if len(durations) < max(self.samples, concurrency):
            return self.target
        rate = concurrency / statistics.median(durations)
        self.rates[concurrency] = rate
        if self._best is None or rate >= self._best[0] * (1 + self.gain):
            self._best = (rate, concurrency)
            if concurrency >= self.maximum:
                self._settle(concurrency, now)
            else:
                self.target = min(self.maximum, concurrency + self.step)
        else:
            self._settle(self._best[1], now)
        return self.target
