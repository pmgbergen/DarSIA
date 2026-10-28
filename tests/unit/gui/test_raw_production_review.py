"""Real collection and Studio candidate readers on private published RAW data."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import raw_production_review as registration
from darsia.fluidflower import raw_review_postrun as post
from darsia.fluidflower.published_dataset.job_reader import Product
from darsia.fluidflower.store_map import input_reference
from .test_raw_review_postrun import finished_inputs, inputs  # noqa: F401


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return post._pin(path)


@pytest.fixture
def published(finished_inputs, monkeypatch):
    from darsia.fluidflower import series_paths
    data = finished_inputs
    model = data.series_root / "models/phase-model-source"
    write(model / "summary.json", dict(model_id=model.name,
        inventory=dict(datasets=[data.job.series_id])))
    heads = {name: write(model / "models/generic" / (name + "_signal.json"), dict(coefficients=[1.0]))
             for name in ("co2_presence", "gas_within_co2")}
    from darsia.fluidflower import phase_optical_regions as regional
    regional_bound = write(model / "regional_model.json", dict(format=regional.VERSION,
        feature_names=list(regional.FEATURE_NAMES), settings={}))
    profile = write(model / "phase_optical_regions.json", dict(format=regional.VERSION,
        model=dict(path="regional_model.json", sha256=regional_bound["sha256"]),
        runs={data.job.series_id + "/" + data.job.runs[0]: {}}))
    heads.update(regional=regional_bound, profile=profile)
    config = json.loads(Path(data.configuration["path"]).read_bytes())
    phase_before = config["phase"]["configuration"]
    phase = json.loads(Path(phase_before["path"]).read_bytes())
    phase["model_bundles"] = dict(native=dict(documents=heads))
    phase_bound = write(Path(phase_before["path"]), phase)
    preparation_config = json.loads(Path(data.preparation_config["path"]).read_bytes())
    preparation_config["phase"]["configuration"] = phase_bound
    preparation_config_bound = write(Path(data.preparation_config["path"]), preparation_config)
    acquisitions = json.loads(Path(data.preparation["path"]).read_bytes())
    acquisitions["configuration"] = preparation_config_bound
    data.preparation = write(Path(data.preparation["path"]), acquisitions)
    config["phase"]["configuration"] = phase_bound
    configuration = write(Path(data.configuration["path"]), config)
    task = data.job.tasks[data.job.runs[0]]
    for row in task["input_catalog"]["inputs"]:
        original = row["reference"]
        replacement = (phase_bound if original["sha256"] == phase_before["sha256"] else
                       configuration if "run.configuration" in row["uses"] else None)
        if replacement:
            row["reference"] = input_reference(replacement["path"], {"series": str(data.series_root)},
                sha256=replacement["sha256"], size_bytes=replacement["size_bytes"], role=original["role"])
        if "run.configuration" in row["uses"]:
            task["configuration"] = task["input_catalog"]["configuration"] = row["reference"]
    write(data.root / "results" / data.job.runs[0] / "task.json", task)
    shape = config["phase"]["reference"]["full"]["phase_shape_hw"]
    expected = np.zeros(shape, np.uint8)
    expected[2:5, 3:8] = 1
    expected[5:7, 6:10] = 2
    for index, result in enumerate(data.results.values()):
        labels_path = data.series_root / ("labels-" + str(index) + ".png")
        Image.fromarray(expected).save(labels_path)
        bound = post._pin(labels_path)
        stage = result["prediction"]["pipeline"]
        stage["configuration_sha256"] = configuration["sha256"]
        stage["phase"].update(labels=bound, complete=True, source_sha256=stage["source"]["source_sha256"])
        data.job.products[data.job.runs[0]]["label-" + str(index)] = Product(
            "label-" + str(index), data.stores.machine_id, bound["path"], bound["sha256"],
            bound["size_bytes"], "phase", None)
    (data.series_root / "series.toml").write_text(
        'schema_version = 1\nseries_id = ' + json.dumps(data.job.series_id) + '\n', encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(data.series_root / "registry.toml"))
    monkeypatch.setenv("FLUIDFLOWER_PHASE_COLLECTION_ROOT", str(data.series_root / "collections"))
    series_paths.register_series(data.series_root)
    from darsia.fluidflower.native import display_modes

    def visible_modes(log_rgb, support, *, selected_modes, **kwargs):
        height, width = support.shape
        red = np.broadcast_to(np.linspace(0.15, 0.85, width, dtype=np.float32), (height, width))
        green = np.broadcast_to(np.linspace(0.2, 0.8, height, dtype=np.float32)[:, None], (height, width))
        rgb = np.stack((red, green, np.full_like(red, 0.45)), axis=-1)
        return {name: rgb.copy() for name in selected_modes}, {}

    monkeypatch.setattr(display_modes, "render_display_modes", visible_modes)
    request = post.attach_request(post.build_request(data.root, data.store_file, [data.preparation["path"]],
        data.output, series_root=data.series_root))
    completed = post.run_request(request)
    return NS(data=data, model=model, completion=Path(completed["review_root"]) / "postrun_completed.json",
              labels=expected, config=configuration, phase=phase_bound)


def test_registered_collection_opens_through_real_studio_and_raw_candidate_readers(published):
    from darsia.fluidflower.tools import phase_review_collections as collections
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage as Page
    from darsia.fluidflower import phase_photographic_assets as photographs
    data = published
    original = {str(path): path.read_bytes() for path in data.data.series_root.rglob("*") if path.is_file()}
    binding = registration.register_review(data.completion, data.model)
    assert binding["run_count"] == 1
    assert collections.discover_collection(data.model, saved_comparison=True) == binding
    studio = NS(review_sessions=[], _normal_review_collection=lambda model: Page._normal_review_collection(None, model))
    entry = dict(root=str(data.model), model_id=data.model.name)
    assert Page._validation_binding(studio, entry) == binding
    inventory = review.discover_review_inventory(Path(binding["combined_root"]))
    assert len(inventory.pairs) == 1
    candidate = inventory.pairs[0].candidate_a
    store = review.ReviewStore(data.data.series_root / "new-session", inventory.pairs)
    app_model = review.ReviewApp(inventory, store)
    payload = app_model.pair_payload(candidate.run, "reviewer")
    assert payload["comparison_mode"] == "single_candidate"
    assert len(payload["frame_sources"]) == candidate.frame_count
    assert payload["frame_sources"][0]["case_id"] == data.data.job.frames(candidate.run)[0]
    image = photographs.resolve_display_mode(candidate.review_output_dir, candidate.run, 0, "natural")
    assert image.is_file()
    overlay, kind = review._preferred_phase_overlay_path(Path(binding["combined_root"]), candidate, "A", 0)
    assert kind == "published_production_labels"
    rgba = np.asarray(Image.open(overlay))
    from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA, GAS_RGBA
    assert np.all(rgba[data.labels == 1] == AQUEOUS_RGBA)
    assert np.all(rgba[data.labels == 2] == GAS_RGBA)
    assert np.all(rgba[data.labels == 0] == 0)
    opened = []
    studio._series_batch_blocks_selection = lambda: False
    studio._selected_validation_model = lambda: dict(entry, validation_binding=binding)
    studio._start_review_process = lambda mode, selected: opened.append((mode, selected))
    Page._launch_review(studio, "open_correction")
    assert opened == [("open_correction", binding)]
    assert all(Path(path).read_bytes() == raw for path, raw in original.items())
    assert registration.register_review(data.completion, data.model) == binding


def test_composite_model_origin_refuses_before_registration(published):
    data = published
    phase = json.loads(Path(data.phase["path"]).read_bytes())
    outside = write(data.data.series_root / "another-model/head.json", dict(coefficients=[2.0]))
    phase["model_bundles"]["placement"] = dict(pixel_model=outside)
    with pytest.raises(ValueError, match="multiple model origins"):
        registration._origin(data.model, phase)
    assert not (data.data.series_root / "collections").exists()


def test_an_identical_copy_does_not_replace_the_explicit_model_origin(published):
    data = published
    other = data.model.with_name("phase-model-other")
    shutil.copytree(data.model, other)
    with pytest.raises(ValueError, match="multiple model origins"):
        registration.register_review(data.completion, other)
    assert not (data.data.series_root / "collections").exists()


def test_registered_source_or_published_label_changes_are_refused(published):
    from darsia.fluidflower.tools import phase_review_collections as collections
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    data = published
    binding = registration.register_review(data.completion, data.model)
    manifest = json.loads((Path(binding["combined_root"]) / "postflight_manifest.json").read_bytes())
    candidate = review._candidate_from_job(Path(binding["combined_root"]), manifest["jobs"][0])
    record = json.loads(Path(candidate.quality_flags["production_review"]["path"]).read_bytes())
    label = Path(record["runs"][0]["labels"][0]["path"])
    label.write_bytes(label.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="phase labels changed"):
        registration.phase_overlay(candidate, 0)
    Path(data.config["path"]).write_bytes(b"{}")
    with pytest.raises(collections.InvalidCollectionError, match="source changed"):
        collections.discover_collection(data.model)


def test_actual_review_window_shows_registered_raw_photo_and_published_overlay(published, qtbot, monkeypatch, tmp_path, record_property):
    from PySide6.QtCore import QThreadPool, QTimer, Qt
    from darsia.fluidflower import phase_workflow
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("DARSIA_PHASE_DISPLAY_CACHE", str(tmp_path / "display"))
    monkeypatch.setattr(phase_workflow, "default_phase_review_roots", lambda: (tmp_path / "sessions",))
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    from darsia.fluidflower.tools import calibration_review_desktop as desktop
    data = published
    original = {path: path.read_bytes() for path in data.data.series_root.rglob("*") if path.is_file()}
    binding = registration.register_review(data.completion, data.model)
    combined = Path(binding["combined_root"])
    inventory = review.discover_review_inventory(combined)
    decisions = tmp_path / "sessions/current"
    store = review.ReviewStore(decisions, inventory.pairs)
    app_model = review.ReviewApp(inventory, store)
    workspace = desktop.ReviewWorkspace(key="current_comparison", title="Published RAW review",
        short_label="Published phase labels", description="Private synthetic source and published labels",
        campaign_id=binding["campaign_id"], ffac_root=combined, ffum_root=combined,
        combined_root=combined, decision_dir=decisions, review_surface="open_correction",
        single_model=True, target_model_id=data.model.name, read_only=False,
        selected_candidates=binding["selected_candidates"])
    window = desktop.workspace_window_class(workspace)(app_model, store, tmp_path / "viewer-cache", workspace,
        reviewer_name="reviewer", use_opengl=False)

    def finish(value):
        for timer in value.findChildren(QTimer):
            timer.stop()
        qtbot.waitUntil(lambda: all(pool.activeThreadCount() == 0 for pool in value.findChildren(QThreadPool))
                       and QThreadPool.globalInstance().activeThreadCount() == 0, timeout=20000)

    qtbot.addWidget(window, before_close_func=finish)
    window.resize(1280, 900)
    window.show()
    qtbot.waitUntil(lambda: bool(window.current_run) and not window.viewers["A"].base_item.pixmap().isNull(), timeout=20000)
    window.contours.setChecked(True)
    qtbot.waitUntil(lambda: not window.viewers["A"].overlay_item.pixmap().isNull(), timeout=20000)
    finish(window)
    viewer = window.viewers["A"]
    assert window.isVisible() and window.current_run == inventory.pairs[0].run
    assert viewer.base_item.pixmap().size() == viewer.overlay_item.pixmap().size()
    assert viewer.base_item.pixmap().width() == data.labels.shape[1]
    assert viewer.base_item.pixmap().height() == data.labels.shape[0]
    assert viewer.base_item.isVisible() and viewer.overlay_item.isVisible()
    assert viewer.base_item.sceneTransform() == viewer.overlay_item.sceneTransform()
    assert viewer.base_item.sceneBoundingRect() == viewer.overlay_item.sceneBoundingRect()
    assert max(viewer.base_item.pixmap().toImage().pixelColor(8, 6).getRgb()[:3]) > 64
    assert window.width() == 1280
    options = window.image_controls_layout
    visible_items = [options.itemAt(index) for index in range(options.count())
                     if not options.itemAt(index).isEmpty()]
    assert len({item.geometry().y() for item in visible_items}) > 1
    for item in visible_items:
        assert item.geometry().width() >= item.sizeHint().width()
        assert options.geometry().contains(item.geometry())
        assert "Â" not in getattr(item.widget(), "text", lambda: "")()
    assert "Â" not in window.run_list.item(0).text()
    assert "·" in window.run_list.item(0).text()
    assert "Â" not in window.frame_label.text()
    assert "Â" not in desktop.FrameTimelineLabel.frame_text(0, 2, baseline=True)
    assert "Â" not in desktop.FrameTimelineLabel.frame_text(0, 2, reference=True)
    decision_pane = window.decision_scroll
    assert decision_pane.widget().width() == decision_pane.viewport().width()
    assert decision_pane.horizontalScrollBar().maximum() == 0
    assert not decision_pane.horizontalScrollBar().isVisible()
    assert window.next_unreviewed.width() >= window.next_unreviewed.sizeHint().width()
    for control in (window.next_unreviewed, *window.flag_boxes["A"].values()):
        assert decision_pane.widget().rect().contains(control.geometry())
        assert control.width() >= control.sizeHint().width()
    for key, label in window.FLAG_LABELS:
        assert " ".join(window.flag_boxes["A"][key].text().split()) == label
    qtbot.keyClicks(window.reference_guide_name, "Reference level")
    qtbot.mouseClick(window.horizontal_guide, Qt.MouseButton.LeftButton)
    guide_point = viewer.mapFromScene(viewer.graphics_scene.sceneRect().center())
    assert viewer.viewport().rect().contains(guide_point)
    expected_position = viewer.mapToScene(guide_point).y() / viewer.graphics_scene.sceneRect().height()
    qtbot.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=guide_point)
    assert len(window.reference_guides) == 1
    guide = dict(window.reference_guides[0])
    assert guide["label"] == "Reference level" and guide["orientation"] == "horizontal"
    assert guide["position"] == pytest.approx(expected_position)
    qtbot.mouseClick(window.pan_images, Qt.MouseButton.LeftButton)
    assert window.verdict.isEnabled() and window.verdict.currentData() == ""
    qtbot.keyClick(window.verdict, Qt.Key.Key_Down)
    assert window.verdict.currentData() == "acceptable"
    decision_pane.ensureWidgetVisible(window.save_now)
    qtbot.waitUntil(lambda: decision_pane.viewport().rect().contains(
        window.save_now.mapTo(decision_pane.viewport(), window.save_now.rect().center())))
    qtbot.mouseClick(window.save_now, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._comparison_save_task is None and not window.save_dirty
                   and bool(window._comparison_saved_records.get(window.current_run)), timeout=20000)
    finish(window)
    saved = review.ReviewStore(decisions, inventory.pairs).decisions()[window.current_run]
    assert saved["run"] == inventory.pairs[0].run and saved["reviewer"] == "reviewer"
    assert saved["candidate_a"]["candidate_id"] == inventory.pairs[0].candidate_a.candidate_id
    assert len(saved["frame_choices"]) == 1
    assert saved["frame_choices"][0]["frame_index"] == 0
    assert saved["frame_choices"][0]["choice"] == "acceptable"
    assert saved["frame_choices"][0]["reviewer"] == "reviewer"
    assert saved["reference_guides"] == [guide]
    assert all(path.read_bytes() == raw for path, raw in original.items())
    decision_pane.verticalScrollBar().setValue(0)
    guide_options = window.guide_controls_widget.layout()
    guide_items = [guide_options.itemAt(index) for index in range(guide_options.count())
                   if not guide_options.itemAt(index).isEmpty()]
    assert len({item.geometry().y() for item in guide_items}) > 1
    for item in guide_items:
        assert item.geometry().width() >= item.sizeHint().width()
        assert guide_options.geometry().contains(item.geometry())
    for control in (window.reference_guide_name, window.horizontal_guide, window.vertical_guide,
                    window.reference_guide_list, window.delete_reference_guide):
        assert control.width() >= control.sizeHint().width()
    assert window.reference_guide_name.width() >= window.reference_guide_name.fontMetrics().horizontalAdvance(
        window.reference_guide_name.placeholderText()) + 12
    assert "Reference level" in window.reference_guide_list.currentText()
    assert "Â" not in window.reference_guide_list.currentText()
    screenshot = tmp_path / "registered_raw_production_review.png"
    assert window.grab().save(str(screenshot))
    record_property("actual_review_window_screenshot", str(screenshot))


def test_gui_reader_installation_is_idempotent_and_delegates_existing_routes(monkeypatch):
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    from darsia.fluidflower.tools import phase_review_collections as collections
    calls = []
    monkeypatch.setattr(review, "_candidate_from_job", lambda *args, **kwargs: calls.append("candidate") or "existing-candidate")
    monkeypatch.setattr(review, "_native_phase_map_path", lambda *args, **kwargs: calls.append("native") or "existing-native")
    monkeypatch.setattr(review, "_preferred_phase_overlay_path", lambda *args, **kwargs: calls.append("overlay") or ("existing-overlay", "legacy"))
    monkeypatch.setattr(collections, "_bound_jobs", lambda *args: calls.append("jobs") or [])
    registration.install_readers()
    installed = review._candidate_from_job
    registration.install_readers()
    assert review._candidate_from_job is installed
    assert review._candidate_from_job(Path("collection"), {}) == "existing-candidate"
    candidate = NS(quality_flags={})
    assert review._native_phase_map_path(output_dir=Path("out"), candidate=candidate, label="A", frame_index=0) == "existing-native"
    assert review._preferred_phase_overlay_path(output_dir=Path("out"), candidate=candidate, label="A", frame_index=0, phase="gas") == ("existing-overlay", "legacy")
    assert collections._bound_jobs(Path("manifest"), {}) == []
    assert calls == ["candidate", "native", "overlay", "jobs"]
