import multiprocessing as mp
import queue
import json
import time
from pathlib import Path

import pytest

from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.user_interface_gui import (
    abort_process,
    clear_queue,
    deduplicate_paths,
    default_session_cache_file,
    enabled_option_labels,
    format_workflow_done_message,
    format_workflow_start_message,
    map_conflict_dialog_choice_to_policy,
    normalize_paths,
    publish_latest_queue_item,
    read_session_cache,
    resolve_utils_bundle_defaults,
    resolve_rig_class,
    write_session_cache,
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


def test_deduplicate_paths_preserves_order(tmp_path: Path) -> None:
    p1 = (tmp_path / "a.toml").resolve()
    p2 = (tmp_path / "b.toml").resolve()
    assert deduplicate_paths([p1, p2, p1]) == [p1, p2]


def test_write_and_read_session_cache_roundtrip(tmp_path: Path) -> None:
    p1 = tmp_path / "a.toml"
    p2 = tmp_path / "b.toml"
    p1.write_text("[data]\nfolder='.'\n")
    p2.write_text("[data]\nfolder='.'\n")
    cache_path = tmp_path / "session.json"
    write_session_cache(cache_path, [p1, p2], "x.y:Rig")

    paths, rig_spec = read_session_cache(cache_path)
    assert paths == [p1.resolve(), p2.resolve()]
    assert rig_spec == "x.y:Rig"


def test_read_session_cache_missing_file(tmp_path: Path) -> None:
    paths, rig_spec = read_session_cache(tmp_path / "missing.json")
    assert paths == []
    assert rig_spec == ""


def test_read_session_cache_invalid_json_raises(tmp_path: Path) -> None:
    cache_path = tmp_path / "session.json"
    cache_path.write_text("not-json")
    with pytest.raises(ValueError):
        read_session_cache(cache_path)


def test_read_session_cache_deduplicates_paths(tmp_path: Path) -> None:
    p1 = tmp_path / "a.toml"
    p1.write_text("[data]\nfolder='.'\n")
    cache_path = tmp_path / "session.json"
    cache_path.write_text(
        json.dumps(
            {"config_paths": [str(p1), str(p1), f"  {p1}  ", ""], "rig_spec": "a:b"}
        )
    )
    paths, rig_spec = read_session_cache(cache_path)
    assert paths == [p1.resolve()]
    assert rig_spec == "a:b"


def test_read_session_cache_rejects_unsupported_version(tmp_path: Path) -> None:
    cache_path = tmp_path / "session.json"
    cache_path.write_text(json.dumps({"version": 999, "config_paths": []}))
    with pytest.raises(ValueError):
        read_session_cache(cache_path)


def test_default_session_cache_file_respects_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-home"))
    assert default_session_cache_file() == (
        tmp_path / "cache-home" / "darsia" / "workflows_gui_session.json"
    )


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


def test_publish_latest_queue_item_keeps_only_latest() -> None:
    q: queue.Queue[str] = queue.Queue()
    q.put_nowait("first")
    publish_latest_queue_item(q, "latest")
    assert q.qsize() == 1
    assert q.get_nowait() == "latest"


def test_clear_queue_removes_all_items() -> None:
    q: queue.Queue[str] = queue.Queue()
    q.put_nowait("a")
    q.put_nowait("b")
    clear_queue(q)
    assert q.empty()


def test_map_conflict_dialog_choice_to_policy() -> None:
    assert map_conflict_dialog_choice_to_policy(True) == "overwrite_all"
    assert map_conflict_dialog_choice_to_policy(False) == "skip_all"
    assert map_conflict_dialog_choice_to_policy(None) is None


def test_resolve_utils_bundle_defaults(tmp_path: Path) -> None:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[data]
folder = "{data_folder}"
baseline = "baseline.jpg"
results = "{tmp_path / "results"}"

[utils.calibration]
export_bundle = "{tmp_path / "export.zip"}"
import_bundle = "{tmp_path / "import.zip"}"
"""
    )
    export_bundle, import_bundle = resolve_utils_bundle_defaults([str(config)])
    assert export_bundle == str(tmp_path / "export.zip")
    assert import_bundle == str(tmp_path / "import.zip")
