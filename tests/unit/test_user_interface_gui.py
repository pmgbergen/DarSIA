import json
import multiprocessing as mp
import queue
import sys
import time
import types
from pathlib import Path

import pytest

from darsia.presets.workflows.analysis.progress import normalize_progress_event
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.user_interface_gui import (
    _run_helper_workflow,
    _run_utils_workflow,
    abort_process,
    clear_queue,
    completion_dialog_spec,
    decode_workflow_error_details,
    deduplicate_paths,
    default_session_cache_file,
    enabled_option_labels,
    encode_workflow_error_details,
    estimate_remaining_time_seconds,
    format_batch_monitor_text,
    format_duration_seconds,
    format_error_details_text,
    format_workflow_done_message,
    format_workflow_error_message,
    format_workflow_start_message,
    map_conflict_dialog_choice_to_policy,
    normalize_paths,
    progress_percent,
    publish_latest_queue_item,
    read_session_cache,
    remaining_image_count,
    resolve_rig_class,
    resolve_utils_bundle_defaults,
    rolling_average_runtime,
    suggested_analysis_results_folder,
    suggested_workflow_results_folder,
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


def test_format_workflow_error_message() -> None:
    msg = format_workflow_error_message("analysis", ["mass"], 3)
    assert msg == "ERROR: analysis workflow failed with exit code 3. Actions: mass."


def test_format_error_details_text_without_details() -> None:
    assert format_error_details_text("") == "No workflow error details available."


def test_format_error_details_text_with_details() -> None:
    details = "Traceback...\nOSError: [WinError 64] ..."
    assert format_error_details_text(details) == details


def test_completion_dialog_spec_success() -> None:
    assert completion_dialog_spec("analysis", 0, False) == (
        "info",
        "Done",
        "Analysis workflow completed.",
    )


def test_completion_dialog_spec_failure() -> None:
    assert completion_dialog_spec("analysis", 7, False) == (
        "error",
        "Error",
        "Analysis workflow failed with exit code 7.",
    )


def test_completion_dialog_spec_abort_is_log_only() -> None:
    assert completion_dialog_spec("analysis", 0, True) is None
    assert completion_dialog_spec("analysis", 9, True) is None


def test_encode_decode_workflow_error_details_roundtrip() -> None:
    details = "Traceback...\nOSError: [WinError 64] ..."
    encoded = encode_workflow_error_details(details)
    assert decode_workflow_error_details(encoded) == details


def test_decode_workflow_error_details_non_error_message() -> None:
    assert decode_workflow_error_details("regular log line") is None


def test_suggested_analysis_results_folder_for_cropping(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder([config], ["cropping"])
    assert folder == results / "cropping"


def test_suggested_analysis_results_folder_from_analysis_section(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    seg_folder = tmp_path / "seg"
    config.write_text(
        f"[data]\nresults = '{results}'\n\n"
        f"[analysis.segmentation]\nfolder = '{seg_folder}'\n"
    )

    folder = suggested_analysis_results_folder([config], ["segmentation"])
    assert folder == seg_folder


def test_suggested_analysis_results_folder_defaults_to_results_on_multiple_modes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder([config], ["mass", "volume"])
    assert folder == results


def test_suggested_analysis_results_folder_fallback_for_missing_mode_folder(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder([config], ["fingers"])
    assert folder == results / "fingers"


def test_suggested_analysis_results_folder_thresholding_fallback(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder([config], ["thresholding"])
    assert folder == results / "thresholding"


def test_suggested_workflow_results_folder_setup(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder("setup", [config], ["depth"])
    assert folder == results / "setup" / "depth"


def test_suggested_workflow_results_folder_calibration(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder(
        "calibration", [config], ["default mass", "show"]
    )
    assert folder == results / "calibration"


def test_suggested_workflow_results_folder_comparison_events_default(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder("comparison", [config], ["events"])
    assert folder == results / "events"


def test_suggested_workflow_results_folder_comparison_wasserstein_override(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    wasserstein = tmp_path / "custom-w1"
    config.write_text(
        f"[data]\nresults = '{results}'\n\n"
        f"[wasserstein]\nresults = '{wasserstein}'\n"
    )

    folder = suggested_workflow_results_folder(
        "comparison", [config], ["wasserstein compute"]
    )
    assert folder == wasserstein


def test_suggested_workflow_results_folder_utils_combined_defaults_to_results(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder(
        "utils", [config], ["download", "media", "export calibration"]
    )
    assert folder == results


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


def test_format_duration_seconds() -> None:
    assert format_duration_seconds(None) == "n/a"
    assert format_duration_seconds(float("nan")) == "n/a"
    assert format_duration_seconds(65.2) == "1:05"
    assert format_duration_seconds(3661.0) == "1:01:01"


def test_rolling_average_runtime_limits_to_last_samples() -> None:
    assert rolling_average_runtime([]) is None
    assert rolling_average_runtime([0.0, -1.0]) is None
    assert rolling_average_runtime([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == pytest.approx(4.0)


def test_remaining_image_count_and_progress_percent() -> None:
    assert remaining_image_count(3, 10) == 7
    assert remaining_image_count(12, 10) == 0
    assert progress_percent(0, 0) == 0.0
    assert progress_percent(5, 10) == pytest.approx(50.0)
    assert progress_percent(15, 10) == pytest.approx(100.0)


def test_estimate_remaining_time_seconds_requires_statistics() -> None:
    assert estimate_remaining_time_seconds(None, 5, 10) is None
    assert estimate_remaining_time_seconds(2.0, 1, 10) is None
    assert estimate_remaining_time_seconds(2.0, 5, 10) == pytest.approx(10.0)
    assert estimate_remaining_time_seconds(2.0, 10, 10) == pytest.approx(0.0)


def test_format_batch_monitor_text_contains_requested_fields() -> None:
    text = format_batch_monitor_text(
        step="mass",
        image_path="/tmp/image.png",
        processed=3,
        total=10,
        last_image_seconds=1.2,
        step_elapsed_seconds=7.0,
        overall_elapsed_seconds=12.4,
        eta_seconds=20.0,
    )
    assert "Current analysis step: mass" in text
    assert "Current image path: /tmp/image.png" in text
    assert "Image count: 3/10 (30.0%)" in text
    assert "Last image elapsed:" in text
    assert "Current step elapsed:" in text
    assert "Overall elapsed:" in text
    assert "Estimated remaining:" in text


def test_normalize_progress_event_payloads() -> None:
    valid = normalize_progress_event(
        {
            "event": "image_progress",
            "step": "mass",
            "image_path": "/tmp/img.png",
            "image_index": 2,
            "image_total": 10,
            "image_duration_s": 1.2,
            "step_elapsed_s": 8.4,
        }
    )
    assert valid is not None
    assert valid["event"] == "image_progress"
    assert valid["step"] == "mass"
    assert valid["image_index"] == 2
    assert valid["image_total"] == 10
    assert normalize_progress_event({"event": "invalid", "step": "mass"}) is None
    assert normalize_progress_event("not-a-dict") is None


def test_resolve_utils_bundle_defaults(tmp_path: Path) -> None:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[data]
folder = "{data_folder.as_posix()}"
baseline = "baseline.jpg"
results = "{(tmp_path / "results").as_posix()}"

[utils.calibration]
export_bundle = "{(tmp_path / "export.zip").as_posix()}"
import_bundle = "{(tmp_path / "import.zip").as_posix()}"
"""
    )
    export_bundle, import_bundle = resolve_utils_bundle_defaults([str(config)])
    assert export_bundle == str(tmp_path / "export.zip")
    assert import_bundle == str(tmp_path / "import.zip")


def test_run_utils_workflow_dispatches_download_and_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[Path]]] = []

    def _fake_download(paths, require_confirmation=True):
        calls.append(("download", paths))

    def _fake_media(paths):
        calls.append(("media", paths))

    def _fake_export(paths, bundle):
        del bundle
        calls.append(("export_calibration", paths))

    def _fake_import(paths, bundle, conflict_action):
        del bundle, conflict_action
        calls.append(("import_calibration", paths))

    fake_download_module = types.ModuleType(
        "darsia.presets.workflows.utils.utils_download"
    )
    fake_download_module.download_data = _fake_download
    fake_media_module = types.ModuleType("darsia.presets.workflows.utils.utils_media")
    fake_media_module.build_media = _fake_media
    fake_calibration_module = types.ModuleType(
        "darsia.presets.workflows.utils.calibration_bundle"
    )
    fake_calibration_module.export_calibration_bundle = _fake_export
    fake_calibration_module.import_calibration_bundle = _fake_import
    monkeypatch.setitem(
        sys.modules,
        "darsia.presets.workflows.utils.utils_download",
        fake_download_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "darsia.presets.workflows.utils.utils_media",
        fake_media_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "darsia.presets.workflows.utils.calibration_bundle",
        fake_calibration_module,
    )

    config_path = tmp_path / "run.toml"
    config_path.write_text("[data]\nfolder='.'\n")

    _run_utils_workflow(
        [str(config_path)],
        {
            "download": True,
            "media": True,
            "export_calibration": False,
            "import_calibration": False,
            "export_bundle": "",
            "import_bundle": "",
            "import_conflict_action": "error",
        },
    )

    assert len(calls) == 2
    assert calls[0][0] == "download"
    assert calls[1][0] == "media"
    assert calls[0][1] == [config_path.resolve()]
    assert calls[1][1] == [config_path.resolve()]


def test_run_helper_workflow_dispatches_roi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, object] = {}

    def _fake_run_helper(rig_cls, args):
        called["rig_cls"] = rig_cls
        called["args"] = args

    fake_helper_module = types.ModuleType(
        "darsia.presets.workflows.user_interface_helper"
    )
    fake_helper_module.run_helper = _fake_run_helper
    monkeypatch.setitem(
        sys.modules,
        "darsia.presets.workflows.user_interface_helper",
        fake_helper_module,
    )

    config_path = tmp_path / "run.toml"
    config_path.write_text("[data]\nfolder='.'\n")

    _run_helper_workflow(
        [str(config_path)],
        "darsia.presets.workflows.rig:Rig",
        {"roi": True, "roi_viewer": False, "show": False},
    )

    args = called["args"]
    assert args.roi is True
    assert args.roi_viewer is False
    assert args.results is False
    assert args.show is False
    assert args.info is False
    assert args.config == [config_path.resolve()]


def test_run_helper_workflow_dispatches_roi_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, object] = {}

    def _fake_run_helper(rig_cls, args):
        called["rig_cls"] = rig_cls
        called["args"] = args

    fake_helper_module = types.ModuleType(
        "darsia.presets.workflows.user_interface_helper"
    )
    fake_helper_module.run_helper = _fake_run_helper
    monkeypatch.setitem(
        sys.modules,
        "darsia.presets.workflows.user_interface_helper",
        fake_helper_module,
    )

    config_path = tmp_path / "run.toml"
    config_path.write_text("[data]\nfolder='.'\n")

    _run_helper_workflow(
        [str(config_path)],
        "darsia.presets.workflows.rig:Rig",
        {"roi": False, "roi_viewer": True, "show": False},
    )

    args = called["args"]
    assert args.roi is False
    assert args.roi_viewer is True
    assert args.results is False
    assert args.show is False


def test_run_helper_workflow_dispatches_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, object] = {}

    def _fake_run_helper(rig_cls, args):
        called["rig_cls"] = rig_cls
        called["args"] = args

    fake_helper_module = types.ModuleType(
        "darsia.presets.workflows.user_interface_helper"
    )
    fake_helper_module.run_helper = _fake_run_helper
    monkeypatch.setitem(
        sys.modules,
        "darsia.presets.workflows.user_interface_helper",
        fake_helper_module,
    )

    config_path = tmp_path / "run.toml"
    config_path.write_text("[data]\nfolder='.'\n")

    _run_helper_workflow(
        [str(config_path)],
        "darsia.presets.workflows.rig:Rig",
        {"roi": False, "roi_viewer": False, "results": True, "show": False},
    )

    args = called["args"]
    assert args.roi is False
    assert args.roi_viewer is False
    assert args.results is True
    assert args.show is False
