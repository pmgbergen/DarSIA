"""Current-view approval metadata checks coalesce and cannot outlive the page."""
from threading import Event

import pytest
from PySide6.QtCore import QCoreApplication, QThread, Qt

from darsia.fluidflower import calibration_review_native_approval_ui as module
from .test_calibration_review_panels import _Signal


def test_stale_preparation_cannot_enable_another_run_and_close_cancels_pending(qtbot, monkeypatch):
    workers = []
    class Signal(_Signal):
        def connect(self, callback, connection):
            assert connection == Qt.ConnectionType.QueuedConnection
            super().connect(callback)
    class Read:
        def __init__(self, generation, context, arguments, parent):
            self.generation, self.context, self.arguments = generation, context, arguments
            self.ready, self.finished = Signal(), Signal()
            self.interruptions = self.waits = self.deleted = 0
            workers.append(self)
        def start(self): pass
        def requestInterruption(self): self.interruptions += 1
        def wait(self): self.waits += 1
        def deleteLater(self): self.deleted += 1
    monkeypatch.setattr(module, "_Read", Read)
    lookup, received = module.NativeApprovalLookup(), []
    lookup.ready.connect(lambda *values: received.append(values))
    registration = {"job": {"sha256": "original"}}
    source_registrations = [{"job": {"sha256": "original-source"}}]
    lookup.request("first", registration=registration, source_registrations=source_registrations)
    registration["job"]["sha256"] = "changed-after-request"
    source_registrations[0]["job"]["sha256"] = "changed-source-after-request"
    source_registrations.append({"job": {"sha256": "added-after-request"}})
    assert workers[0].arguments["registration"]["job"]["sha256"] == "original"
    assert workers[0].arguments["source_registrations"] == [{"job": {"sha256": "original-source"}}]
    lookup.request("second", registration={})
    lookup.request("third", registration={})
    assert len(workers) == 1
    workers[0].ready.emit(workers[0].generation, "first", {"old": True}, "")
    assert received == []
    # Manual workers have no QObject sender; simulate the queued GUI slot.
    lookup._finished(workers[0])
    assert len(workers) == 2 and workers[1].context == "third"
    assert workers[0].arguments is None and workers[0].context is None
    assert workers[0].deleted == 1
    workers[1].ready.emit(workers[1].generation, "third", {"current": True}, "")
    assert received == [("third", {"current": True}, "")]
    lookup.request("fourth", registration={})
    lookup.stop()
    assert workers[1].waits == 1 and workers[1].deleted == 1
    assert workers[1].arguments is None and workers[1].context is None
    assert lookup._worker is None and lookup._pending is None
    workers[1].ready.emit(workers[1].generation, "third", {"late": True}, "")
    lookup._finished(workers[1])
    lookup._finished(None)
    lookup.request("after-close", registration={})
    lookup._start()
    assert len(workers) == 2 and len(received) == 1
    assert workers[1].deleted == 1
    lookup.deleteLater()


@pytest.mark.parametrize("failure", [False, True])
def test_real_workers_coalesce_with_gui_delivery_cleanup_and_pending_start(qtbot, monkeypatch, failure):
    from darsia.fluidflower import calibration_review_native_approval as preparation

    app_thread = QCoreApplication.instance().thread()
    entered, release = Event(), Event()
    work_on_gui, created_on_gui, retired_on_gui, received = [], [], [], []
    binding = object()

    def prepare(**arguments):
        label = arguments["registration"]["label"]
        work_on_gui.append((label, QThread.currentThread() is app_thread))
        if label == "first":
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release the first metadata worker")
        if failure and label == "latest":
            raise ValueError("Approval evidence differs")
        return binding

    class Read(module._Read):
        def __init__(self, *args):
            created_on_gui.append(QThread.currentThread() is app_thread)
            super().__init__(*args)

    class Lookup(module.NativeApprovalLookup):
        def _finished(self, worker):
            if worker is not None and worker is self._worker:
                on_gui = QThread.currentThread() is app_thread
                super()._finished(worker)
                retired_on_gui.append((on_gui, worker.context is None, worker.arguments is None))

    monkeypatch.setattr(preparation, "prepare_native_run_approval", prepare)
    monkeypatch.setattr(module, "_Read", Read)
    lookup = Lookup()

    def ready(context, result, error):
        received.append((context, result, error, QThread.currentThread() is app_thread))

    lookup.ready.connect(ready)
    try:
        lookup.request("first-context", registration={"label": "first"})
        qtbot.waitUntil(entered.is_set, timeout=5000)
        lookup.request("superseded-context", registration={"label": "superseded"})
        lookup.request("latest-context", registration={"label": "latest"})
        assert created_on_gui == [True]
        release.set()
        qtbot.waitUntil(lambda: bool(received) and lookup._worker is None, timeout=5000)
        assert work_on_gui == [("first", False), ("latest", False)]
        assert created_on_gui == [True, True]
        assert retired_on_gui == [(True, True, True), (True, True, True)]
        assert received == [("latest-context", None if failure else binding,
                             "Approval evidence differs" if failure else "", True)]
        lookup.stop()
        lookup.request("after-close", registration={"label": "closed"})
        lookup._start()
        assert lookup._worker is None and lookup._pending is None
        assert created_on_gui == [True, True]
    finally:
        release.set()
        lookup.stop()
        lookup.deleteLater()
