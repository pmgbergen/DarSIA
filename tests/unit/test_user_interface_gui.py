from pathlib import Path
import threading
import time

import pytest

from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.user_interface_gui import (
    abort_thread,
    normalize_paths,
    resolve_rig_class,
)


def test_resolve_rig_class_default() -> None:
    assert resolve_rig_class("") is Rig


def test_resolve_rig_class_explicit() -> None:
    assert resolve_rig_class("darsia.presets.workflows.rig:Rig") is Rig


def test_resolve_rig_class_invalid_spec() -> None:
    with pytest.raises(ValueError):
        resolve_rig_class("darsia.presets.workflows.rig.Rig")


def test_resolve_rig_class_not_subclass() -> None:
    with pytest.raises(ValueError):
        resolve_rig_class("builtins:str")


def test_normalize_paths_deduplicates_and_resolves(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_text("[data]\nfolder='.'\n")
    raw = [str(p), str(p), "", f"  {p}  "]
    normalized = normalize_paths(raw)
    assert normalized == [p.resolve()]


def test_abort_thread_none_returns_false() -> None:
    assert not abort_thread(None)


def test_abort_thread_stops_running_thread() -> None:
    aborted = threading.Event()

    def worker() -> None:
        try:
            while True:
                time.sleep(0.01)
        except SystemExit:
            aborted.set()
            return

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    assert abort_thread(thread)
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert aborted.is_set()
