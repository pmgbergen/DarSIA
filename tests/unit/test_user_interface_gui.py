import multiprocessing as mp
import time
from pathlib import Path

import pytest

from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.user_interface_gui import (
    abort_process,
    enabled_option_labels,
    format_workflow_done_message,
    format_workflow_start_message,
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


def _long_running_task(duration: float) -> None:
    time.sleep(duration)


def test_abort_process_none_returns_false() -> None:
    assert not abort_process(None)


def test_abort_process_stops_running_process() -> None:
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=_long_running_task, args=(5.0,), daemon=True)
    process.start()

    assert abort_process(process)
    assert not process.is_alive()


def test_enabled_option_labels_excludes_entries() -> None:
    labels = enabled_option_labels(
        {"all": True, "save_npz": True, "show": False}, exclude={"all"}
    )
    assert labels == ["save npz"]


def test_format_workflow_start_message() -> None:
    msg = format_workflow_start_message(
        workflow="analysis",
        actions=["mass", "segmentation"],
        config_paths=[Path("/tmp/a.toml"), Path("/tmp/b.toml")],
        rig_spec="",
    )
    assert "Starting analysis workflow." in msg
    assert "Actions: mass, segmentation." in msg
    assert "Configs: /tmp/a.toml, /tmp/b.toml." in msg
    assert "Rig: darsia.presets.workflows.rig:Rig." in msg


def test_format_workflow_done_message() -> None:
    msg = format_workflow_done_message("setup", ["depth", "rig"], 2, 3.2)
    assert msg == "Setup completed. Actions: depth, rig. Configs: 2. Duration: 3.2s."
