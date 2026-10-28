"""Q185: a QThread worker is freed by its owner in the GUI thread, never in its own.

Deleted in its own thread, a worker created from Python needs the GIL while Qt
holds a signal-slot lock (the Python QThread's disconnectNotify), and a GUI
thread that holds the GIL and waits for that lock, or waits in ~QThread,
deadlocks. Owners therefore wait() for the finished thread and then drop the
worker; they never call or connect the worker's deleteLater.

Offscreen Qt with injected providers; no hardware probe, scan or source write.
"""

import ast
from pathlib import Path
import threading

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from shiboken6 import isValid

import darsia

SOURCE_ROOT = Path(darsia.__file__).resolve().parent


def _moved(node):
    """The expression of a ``X.moveToThread(...)`` call, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "moveToThread"
    ):
        return node.func.value
    return None


def _key(expression):
    """'name' for a local, 'self.name' for an attribute of self, else None."""
    if isinstance(expression, ast.Name):
        return expression.id
    if (
        isinstance(expression, ast.Attribute)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "self"
    ):
        return "self." + expression.attr
    return None


def _assignments(function):
    """(target key, value key) of the plain and tuple assignments in a function."""
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                pairs = zip(target.elts, node.value.elts)
            else:
                pairs = [(target, node.value)]
            for left, right in pairs:
                yield _key(left), _key(right)


def worker_deletions(source):
    """(line, text) of every deleteLater aimed at an object moved to a QThread.

    Within one class, an object counts as moved when a method calls
    ``X.moveToThread(...)`` on it, when it is a self attribute assigned from such
    a local, or a local assigned from such an attribute. Its ``deleteLater`` may
    be called or passed to ``connect``.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    scopes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    top_level = [node for node in tree.body if not isinstance(node, ast.ClassDef)]
    scopes.append(ast.Module(body=top_level, type_ignores=[]))
    found = set()
    for scope in scopes:
        functions = [
            node
            for node in ast.walk(scope)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        attributes = set()
        for function in functions:
            local = {_key(_moved(node)) for node in ast.walk(function)} - {None}
            attributes |= {key for key in local if key.startswith("self.")}
            attributes |= {
                left
                for left, right in _assignments(function)
                if right in local and left and left.startswith("self.")
            }
        for function in functions:
            moved = {_key(_moved(node)) for node in ast.walk(function)} - {None}
            moved |= attributes
            moved |= {
                left
                for left, right in _assignments(function)
                if right in attributes and left
            }
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "deleteLater"
                    and _key(node.value) in moved
                ):
                    found.add((node.lineno, lines[node.lineno - 1].strip()))
    return sorted(found)


SAMPLE = """
class Owner:
    def start(self):
        thread = QThread(self)
        worker = Worker()
        worker.moveToThread(thread)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker

    def closed(self):
        self._worker.deleteLater()
        worker = self._worker
        worker.deleteLater()
        self._thread.deleteLater()
"""


def test_the_guard_finds_a_connected_and_a_called_worker_delete_later():
    assert [line for line, _ in worker_deletions(SAMPLE)] == [7, 12, 14]


def test_no_worker_moved_to_a_thread_is_deleted_with_delete_later():
    found = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "moveToThread" not in text:
            continue
        where = path.relative_to(SOURCE_ROOT).as_posix()
        for line, statement in worker_deletions(text):
            found.append(f"{where}:{line}: {statement}")
    assert found == [], "\n".join(found)


def _watcher(worker):
    """A QObject that Qt deletes once the worker's C++ object is deleted."""
    watcher = QObject()
    worker.destroyed.connect(watcher.deleteLater)
    return watcher


def _deleted(watcher):
    """True once the watched worker was deleted, so its destroyed signal ran."""
    QCoreApplication.sendPostedEvents(watcher, QEvent.Type.DeferredDelete)
    return not isValid(watcher)


def test_a_probe_worker_outlives_its_thread_until_its_owner_lets_go(qtbot):
    """The connected form: deleteLater on the worker's own signal ran in its thread."""
    from darsia.fluidflower import resource_advisor_panel as ui

    panel = ui.ResourceAdvisorPanel(
        inventory_provider=lambda: {"host": "LOCAL_FIXTURE"}
    )
    qtbot.addWidget(panel)
    panel.inspect_button.click()
    worker = panel._probe_worker
    qtbot.waitUntil(lambda: panel._probe_thread is None, timeout=5000)
    assert panel._probe_worker is None
    assert isValid(worker), "the probe worker was deleted in its own thread"
    watcher = _watcher(worker)
    del worker  # The last reference: the worker is freed here, in the GUI thread.
    assert _deleted(watcher)


def test_a_discovery_worker_is_never_handed_to_delete_later(
    qtbot, tmp_path, monkeypatch
):
    """The called form: deleteLater from the finished slot posted into its thread."""
    from darsia.fluidflower.pages.setup.steps import image_sources as module

    requested = []
    monkeypatch.setattr(module, "discover_image_source_mapping", lambda *a, **k: [])
    monkeypatch.setattr(
        module.SourceDiscoveryWorker, "deleteLater", lambda self: requested.append(self)
    )
    page = module.ImageSourcesPage()
    qtbot.addWidget(page)
    page.experiment_images.setText(str(tmp_path))
    page.structure.setCurrentIndex(page.structure.findData("single_run"))
    page._discover()
    worker = page._worker
    qtbot.waitUntil(lambda: page._thread is None, timeout=5000)
    assert requested == [], "deleteLater would delete the worker in its own thread"
    assert page._worker is None and isValid(worker)
    watcher = _watcher(worker)
    del worker  # The last reference: the worker is freed here, in the GUI thread.
    assert _deleted(watcher)


def test_saved_transfer_reader_is_released_in_gui_after_native_thread_retirement(
    qtbot, tmp_path, monkeypatch
):
    """Keep a real finite reader alive until its GUI owner drops the last ref."""
    from darsia.fluidflower import painting_saved_transfers_ui as ui

    requested = []
    monkeypatch.setattr(ui._Read, "deleteLater", lambda self: requested.append(self))
    chooser = ui.SavedTransferChooser(tmp_path)
    qtbot.addWidget(chooser, before_close_func=lambda owned: owned.wait_for_workers())
    worker = chooser._worker
    qtbot.waitUntil(lambda: chooser._thread is None, timeout=5000)
    assert chooser._worker is None and not chooser.busy()
    assert requested == [], "The finite reader must be released by its GUI owner"
    assert isValid(worker), "Native retirement must not destroy a retained worker"
    assert worker.thread() is QCoreApplication.instance().thread()
    watcher = _watcher(worker)
    del worker  # The last reference is deliberately released on the GUI thread.
    assert _deleted(watcher)


@pytest.mark.parametrize("owner", ["gap-review", "target-correction"])
def test_painted_review_reader_is_owned_and_freed_on_gui_after_natural_retirement(
    qtbot, monkeypatch, owner
):
    """Both finite dialogs retain a GUI-affinity worker until its owner lets go."""
    from darsia.fluidflower import painting_gap_review_ui as gaps
    from darsia.fluidflower.painting_target_correction_ui import TargetCorrectionDialog
    from .test_painting_gap_review_ui import _Backend, _review
    from .test_painting_target_correction_ui import _Draft, _paint_packet

    gui_thread_ident = threading.get_ident()
    requested = []
    monkeypatch.setattr(gaps._Read, "deleteLater", lambda self: requested.append(self))
    if owner == "gap-review":
        backend = _Backend()
        dialog = gaps.GapReviewDialog(_review(), actor="Synthetic reviewer",
            review_reader=backend.read, preview_reader=backend.preview,
            approval_builder=backend.approve)
    else:
        dialog = TargetCorrectionDialog(_review(), "ac18", "c" * 64,
            actor="Synthetic reviewer", destination="C:/synthetic/unused-correction-output",
            reader=lambda *_: _paint_packet(), draft_factory=_Draft)
    qtbot.addWidget(dialog, before_close_func=lambda owned: owned.wait_for_workers())
    worker, native = dialog.worker, dialog.thread
    dialog.show()
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=5000)
    assert dialog.worker is None and requested == []
    if isValid(native):
        assert not native.isRunning()
        QCoreApplication.sendPostedEvents(native, QEvent.Type.DeferredDelete)
    assert isValid(worker), "Natural retirement must not destroy a retained worker"
    gui = QCoreApplication.instance().thread()
    assert worker.thread() is gui
    destroyed_thread_idents = []
    worker.destroyed.connect(lambda *_: destroyed_thread_idents.append(threading.get_ident()),
                             Qt.ConnectionType.DirectConnection)
    watcher = _watcher(worker)
    del worker
    assert destroyed_thread_idents == [gui_thread_ident]
    assert _deleted(watcher)
