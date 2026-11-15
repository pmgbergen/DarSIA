"""Timing of functions and of named stages of a workflow.

- `timing_decorator` logs the wall-clock time of each call of a function.
- `timed` records named, nested stages (calls, wall-clock time, memory of the process)
  in `timings`, which reports them as a table and stores them per run.

Both only read a clock around the timed code. They use no threads and no profiling or
tracing hooks, so profilers (cProfile, py-spy, line_profiler) work as usual.

"""

import json
import logging
import os
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import perf_counter, time

try:
    import psutil
except ImportError:  # pragma: no cover - optional
    psutil = None

logger = logging.getLogger(__name__)

__all__ = ["timing_decorator", "timed", "timings", "Timings"]


def timing_decorator(func):
    """Decorator logging the wall-clock execution time of ``func``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        # Do not only use the name of the function, but also the class it belongs to
        logger.info(
            f"{func.__module__}.{func.__name__} executed in {end - start:.3f} seconds"
        )
        return result

    return wrapper


class Timings:
    """Record of timed stages: calls, wall-clock time and memory per stage.

    Stages are nested: a stage entered within another one is stored as its child, so
    the record is a tree of paths like ``("calibration", "read images")``. The time of
    a stage includes its children.

    """

    def __init__(self) -> None:
        self.enabled: bool = True
        """Whether stages are recorded (`timed` costs one check otherwise)."""
        self._local = threading.local()
        self._process = None
        self.reset()

    def reset(self) -> None:
        """Forget all recorded stages."""
        self.records: dict[tuple[str, ...], dict] = {}
        """Calls, seconds and memory (GB, largest value at a stage end) per stage."""

    # ! ---- RECORDING ----

    def _stack(self) -> list:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def _memory(self) -> float | None:
        """Memory of the process (resident set size) in GB, if psutil is installed."""
        if psutil is None:
            return None
        if self._process is None:
            self._process = psutil.Process(os.getpid())
        return self._process.memory_info().rss / 1e9

    def enter(self, name: str) -> float:
        """Start a stage; returns the start time for :meth:`exit`."""
        self._stack().append(name)
        return perf_counter()

    def exit(self, start: float) -> None:
        """End the stage started last."""
        seconds = perf_counter() - start
        stack = self._stack()
        path = tuple(stack)
        stack.pop()
        record = self.records.setdefault(
            path, {"calls": 0, "seconds": 0.0, "memory_gb": None}
        )
        record["calls"] += 1
        record["seconds"] += seconds
        memory = self._memory()
        if memory is not None:
            record["memory_gb"] = max(memory, record["memory_gb"] or 0.0)

    # ! ---- OUTPUT ----

    def as_list(self) -> list[dict]:
        """Stages in the order of the tree (parents before children)."""
        # Parents end after their children, so sort by path to list them first, but
        # keep the order in which siblings were first recorded.
        order = {path: index for index, path in enumerate(self.records)}

        def first_seen(path: tuple[str, ...]) -> tuple:
            return tuple(
                min(
                    index
                    for other, index in order.items()
                    if other[:depth] == path[:depth]
                )
                for depth in range(1, len(path) + 1)
            )

        return [
            {
                "stage": "/".join(path),
                "calls": record["calls"],
                "seconds": round(record["seconds"], 4),
                "memory_gb": (
                    None
                    if record["memory_gb"] is None
                    else round(record["memory_gb"], 3)
                ),
            }
            for path, record in sorted(
                self.records.items(), key=lambda item: first_seen(item[0])
            )
        ]

    def report(self) -> str:
        """Table of the recorded stages (children indented)."""
        lines = [f"{'stage':44s} {'calls':>6s} {'time [s]':>10s} {'memory [GB]':>12s}"]
        for entry in self.as_list():
            parts = entry["stage"].split("/")
            name = "  " * (len(parts) - 1) + parts[-1]
            memory = "" if entry["memory_gb"] is None else f"{entry['memory_gb']:.2f}"
            lines.append(
                f"{name[:44]:44s} {entry['calls']:6d} {entry['seconds']:10.2f} "
                f"{memory:>12s}"
            )
        return "\n".join(lines)

    def write(self, path: Path, **info) -> None:
        """Append the recorded stages as one JSON line to a file.

        Parameters
        ----------
        path : Path
            File (JSON lines); one line per run.
        **info
            Description of the run, e.g. workflow and config.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        run = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **info,
            "stages": self.as_list(),
        }
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(run, default=str) + "\n")


timings = Timings()
"""Record of the timed stages of this process."""


class timed:
    """Record a named stage in `timings`; decorator and context manager.

    Examples
    --------
    >>> with darsia.timed("read images"):
    ...     images = [rig.read_image(path) for path in paths]

    >>> @darsia.timed("fluid model")
    ... def build_fluid_model(): ...

    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._starts: list[float | None] = []

    def __enter__(self) -> "timed":
        self._starts.append(timings.enter(self.name) if timings.enabled else None)
        return self

    def __exit__(self, *exc) -> None:
        start = self._starts.pop()
        if start is not None:
            timings.exit(start)

    def __call__(self, func):
        name = self.name

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not timings.enabled:
                return func(*args, **kwargs)
            start = timings.enter(name)
            try:
                return func(*args, **kwargs)
            finally:
                timings.exit(start)

        return wrapper
