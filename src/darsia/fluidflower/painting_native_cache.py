"""Asynchronous exact proposal state for the painting view.

Numeric inputs are shared through read-only views. They must remain immutable
in the owning GUI. All native validation, correction, gap and display-array
work runs in one worker; a later request replaces only its pending successor.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from . import layer_painting as lp
from . import layer_painting_polygons_only as polygons


ASYNC_PIXELS = 1_000_000


@dataclass(frozen=True)
class NativeState:
    key: Any
    classes: np.ndarray | None = None
    decided: np.ndarray | None = None
    base_decided: np.ndarray | None = None
    skipped: tuple[int, ...] = ()
    gap_mask: np.ndarray | None = None
    gap_count: int | None = None
    map_digest: str = ""
    problems: list[str] | None = None
    display_classes: np.ndarray | None = None
    display_decided: np.ndarray | None = None
    display_gaps: np.ndarray | None = None
    base_display_classes: np.ndarray | None = None
    base_display_gaps: np.ndarray | None = None
    error: str = ""


@dataclass(frozen=True)
class _Request:
    key: Any
    proposal: Any
    session: Any
    corrections: Any
    frame: Any
    reference: Any
    layer_ids: frozenset
    display_shape: tuple[int, int]


def _view(array):
    result = np.asarray(array).view()
    result.setflags(write=False)
    return result


def _snapshot(key, proposal, session, corrections, frame, reference, layer_ids, display_shape):
    shape = tuple(display_shape)
    if len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
        raise ValueError("An explicit positive display height and width are required")
    saved = copy(proposal)
    for name, value in vars(proposal).items():
        setattr(saved, name, _view(value) if isinstance(value, np.ndarray) else deepcopy(value))
    return _Request(key, saved, deepcopy(session), deepcopy(tuple(corrections)),
                    deepcopy(frame), deepcopy(reference), frozenset(layer_ids), shape)


def _error(key, error):
    message = f"{type(error).__name__}: {error}"
    return NativeState(key=key, problems=["The exact proposal state is unavailable: " + message], error=message)


def _compute(request):
    import cv2

    proposal = request.proposal
    polygons._validate(proposal, raster=True)
    known = request.layer_ids & set(proposal.source["layer_ids"])
    classes, decided, skipped = polygons.corrected_classes(
        proposal, request.corrections, layer_ids=known)
    gaps = proposal.inside & (classes == lp.OUTSIDE)
    base_gaps = proposal.inside & (proposal.classes == lp.OUTSIDE)
    base_decided = np.zeros(classes.shape, dtype=bool)
    problems = polygons.check_ready_for_revision(request.session, proposal, request.frame,
        request.reference, corrections=request.corrections)
    digest = lp.map_digest(request.session, proposal, request.corrections)
    height, width = request.display_shape

    def display(array):
        boolean = array.dtype == np.bool_
        reduced = cv2.resize(array.astype(np.uint8) if boolean else array,
                            (width, height), interpolation=cv2.INTER_NEAREST)
        return reduced.astype(bool) if boolean else reduced

    arrays = dict(classes=classes, decided=decided, base_decided=base_decided,
        gap_mask=gaps, display_classes=display(classes), display_decided=display(decided),
        display_gaps=display(gaps), base_display_classes=display(proposal.classes),
        base_display_gaps=display(base_gaps))
    for array in arrays.values():
        array.setflags(write=False)
    return NativeState(key=request.key, **arrays, skipped=tuple(skipped),
        gap_count=int(np.count_nonzero(gaps)), map_digest=digest, problems=list(problems))


class _Worker(QObject):
    finished = Signal()

    def __init__(self, request):
        super().__init__()
        self.request = request
        self.result = None

    @Slot()
    def run(self):
        try:
            self.result = _compute(self.request)
        except Exception as error:
            self.result = _error(self.request.key, error)
        self.finished.emit()


class NativeProposalCache(QObject):
    """Two valid exact states, plus the latest explicit failure, for one view.

    ``get`` is passive. ``request`` updates the desired identity and may return
    a cached undo state while an older worker retires. Stale results can enter
    the cache but never change the desired identity. ``changed`` carries no
    state: the owner asks for its current semantic key again on the GUI thread.
    """
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache = OrderedDict()
        self._failure = None
        self._desired = None
        self._pending = None
        self._thread = self._worker = None

    def busy(self):
        return self._thread is not None

    def get(self, key):
        result = self._cache.get(key)
        if result is not None:
            return result
        return self._failure if self._failure is not None and self._failure.key == key else None

    def request(self, key, proposal, session, corrections, frame, reference, *, layer_ids, display_shape):
        hash(key)
        self._desired = key
        state = self.get(key)
        if state is not None:
            self._pending = None
            if key in self._cache:
                self._cache.move_to_end(key)
            return state
        if self._worker is not None and self._worker.request.key == key:
            self._pending = None
            return None
        if self._pending is not None and self._pending.key == key:
            return None
        try:
            self._pending = _snapshot(key, proposal, session, corrections, frame,
                                      reference, layer_ids, display_shape)
        except Exception as error:
            self._pending = None
            self._failure = _error(key, error)
            return self._failure
        self._start_pending()
        return None

    def _start_pending(self):
        if self._thread is not None or self._pending is None:
            return
        request, self._pending = self._pending, None
        if request.key != self._desired or self.get(request.key) is not None:
            return
        thread, worker = QThread(self), _Worker(request)
        self._thread, self._worker = thread, worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # The GUI can wait during shutdown: quitting must not need its event loop.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._thread_finished)
        thread.start()

    @Slot()
    def _thread_finished(self):
        thread = self.sender()
        if thread is self._thread:
            self._retire(thread)

    def _retire(self, thread):
        if thread is not self._thread:
            return
        thread.wait()
        worker = self._worker
        state = worker.result
        if state is None:
            state = _error(worker.request.key, RuntimeError("The proposal worker returned no result"))
        self._thread = self._worker = None
        if state.error:
            self._failure = state
        else:
            self._cache[state.key] = state
            self._cache.move_to_end(state.key)
            while len(self._cache) > 2:
                oldest = next(key for key in self._cache if key != self._desired)
                del self._cache[oldest]
            if self._failure is not None and self._failure.key == state.key:
                self._failure = None
        # Worker ownership is released here, on the GUI thread, after wait.
        del worker
        thread.deleteLater()
        self._start_pending()
        self.changed.emit()

    def wait(self):
        """Finish owned work without cancelling it or needing queued quit events."""
        while self._thread is not None:
            thread = self._thread
            thread.wait()
            self._retire(thread)


__all__ = ["ASYNC_PIXELS", "NativeState", "NativeProposalCache"]
