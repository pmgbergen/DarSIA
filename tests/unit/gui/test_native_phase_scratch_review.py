"""Scratch labeling keeps original photo evidence, drawing, and persistence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

from PIL import Image, ImageDraw
import pytest
from PySide6.QtCore import QRectF

from darsia.fluidflower import native_phase_scratch_review as subject
from darsia.fluidflower import native_phase_review_session as lifecycle


def make_collection(tmp_path, monkeypatch):
    """Two real lossless photo assets with camera identities; no phase maps."""
    from darsia.fluidflower.tools import calibration_review_desktop as desktop
    from darsia.fluidflower.raw_photographic_review import RAW_REVIEW_FORMAT
    from darsia.fluidflower.phase_workflow import build_phase_review_command
    root = tmp_path / "collection"
    folder = root / "native_frames" / "ac01"
    folder.mkdir(parents=True)
    monkeypatch.setenv("DARSIA_REVIEW_ALIAS_FILES", str(tmp_path / "absent-alias.json"))

    def document(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")
        return subject._document(path)

    def asset(path):
        return dict(subject._document(path), path=path.name)

    config = document(root / "configuration.json", {})
    binding_doc = document(root / "binding.json", {})
    frames = []
    for index in range(2):
        image = Image.new("RGB", (320, 180), (220, 203, 172))
        draw = ImageDraw.Draw(image)
        draw.rectangle((35, 45, 170 + index * 35, 115), fill=(115, 164, 185))
        draw.ellipse((135, 60, 225, 125), fill=(175, 196, 156))
        photo = folder / f"frame_{index}.png"
        image.save(photo)
        mask = folder / f"support_{index}.png"
        Image.new("L", (320, 180), 255).save(mask)
        camera = root / f"DSC{index:05d}.ARW"
        camera.write_bytes(b"synthetic camera acquisition " + str(index).encode())
        frames.append(dict(frame_index=index, elapsed_h=float(index), case_id=f"ac01_f{index}",
            acquisition=subject._document(camera), assets=dict(natural=asset(photo)),
            photo_support=asset(mask)))
    native = dict(format=RAW_REVIEW_FORMAT, source_kind="raw", run="ac01", frame_count=2,
        native_configuration=config, raw_binding=binding_doc, frames=frames,
        reference=deepcopy(frames[0]), native_shape_hw=[180, 320], phase_shape_hw=[180, 320],
        native_origin_xy=[0, 0], canvas_wh=[320, 180], content_box_xywh=[0, 0, 320, 180])
    document(folder / "manifest.json", native)
    timeline = root / "per_frame.csv"
    timeline.write_text("frame_index,elapsed_h,source_path,source_sha256\n" + "".join(
        f"{row['frame_index']},{row['elapsed_h']},{row['acquisition']['path']},{row['acquisition']['sha256']}\n"
        for row in frames), encoding="utf-8")
    source_maps = root / "existing-predictions"
    source_maps.mkdir()
    old_quiz = document(source_maps / "old-quiz.json", {"proposals": [{"proposal_id": "old"}]})
    candidate = desktop.review.Candidate(run="ac01", role="source", family="saved", variant="source",
        candidate_id="actual-source-candidate", grid_order=0, params_sha256="a" * 64,
        normalized_mae=None, mass_plot=root / "old-plot.png", contour_movie=root / "old-movie.mp4",
        per_frame_csv=timeline, summary_path=root / "summary.json", plot_seconds=0, image_seconds=0,
        fps=1, frame_count=2, dataset="ffac", review_output_dir=root,
        phase_map_output_dir=source_maps, phase_quiz_proposals_path=Path(old_quiz["path"]))
    manifest = root / "postflight_manifest.json"
    document(manifest, dict(jobs=[dict(run="ac01", dataset="ffac", review_output_dir=str(root))]))
    previous = root / "manual_pairwise_review"
    previous.mkdir()
    document(previous / "pairwise_review_decisions.json", dict(schema=2, decisions={
        "ac01": dict(reviewer="Reviewer", annotations=[dict(id="previous-label")])}))
    pair = desktop.review.ReviewPair("ac01", candidate, candidate, "single_candidate")
    monkeypatch.setattr(desktop.review, "discover_review_inventory", lambda *a, **kw:
        desktop.review.ReviewInventory((pair,), {}, manifest))
    command = build_phase_review_command(mode="active_learning", reviewer="Reviewer",
        save_root=tmp_path / "reviews", cache_root=tmp_path / "cache", campaign_root=root,
        selected_candidates={"ac01": candidate.candidate_id}, use_gpu=False)
    args = desktop.build_parser().parse_args(command[3:])
    args.phase_workflow_series = "ffac"
    args.phase_workflow_role = "label"
    args.review_surface = "blind_regions"
    args.offline_bundle = None
    workspace = desktop.review_workspaces(args)["training"]
    binding = {name: getattr(workspace, name) for name in (
        "combined_root", "ffac_root", "ffum_root", "selected_candidates", "ffac_runs", "ffum_runs")}
    return NS(desktop=desktop, args=args, binding=binding, source=root, candidate=candidate,
        save_root=tmp_path / "reviews", frames=frames, manifest=manifest)


@pytest.fixture
def collection(tmp_path, monkeypatch):
    data = make_collection(tmp_path, monkeypatch)
    identity = subject.source_identity(data.binding, series="ffac")
    reservation = lifecycle.reserve_review_session(data.save_root, series="ffac", reviewer="Reviewer",
        target_model_id="", review_basis="scratch", source_identity=identity)
    data.args.review_session_id = reservation["review_session_id"]
    return data


def prepare(data):
    return subject.prepare(data.desktop, data.args, "Reviewer", data.save_root)


def test_new_session_is_photo_only_and_reopens_saved_evidence(collection, monkeypatch):
    data = collection
    before = {path: path.read_bytes() for path in data.source.rglob("*") if path.is_file()}
    original = data.desktop.review._load_phase_quiz_payload
    def quiz(output, candidate, label):
        assert candidate.phase_map_output_dir != data.candidate.phase_map_output_dir
        assert candidate.phase_quiz_proposals_path is None
        return original(output, candidate, label)
    monkeypatch.setattr(data.desktop.review, "_load_phase_quiz_payload", quiz)
    workspace, identity, model, store = prepare(data)
    assert not workspace.target_model_id and workspace.seed_review_dir is None
    assert store.decisions() == {}
    payload = model.pair_payload("ac01")
    assert payload["candidate_a"]["phase_quiz_proposals"] == []
    assert not payload["candidate_a"]["phase_fill_overlays_ready"]
    assert payload["frame_sources"][1]["source_sha256"] == data.frames[1]["acquisition"]["sha256"]
    store.save(dict(run="ac01", reviewer="Reviewer", choice="unresolved", annotations=[dict(
        id="new-box", candidate="A", frame_index=1, category="confirmed_only_aqueous_co2",
        shape="rectangle", x=.1, y=.1, width=.2, height=.2, confidence="high", severity="major")]))
    lifecycle.attach_review_manifest(data.save_root, series="ffac", reviewer="Reviewer",
        review_session_id=workspace.review_session_id, session_path=workspace.output_dir / "review_session.json")
    saved = json.loads((workspace.output_dir / "review_session.json").read_text())
    assert saved["review_basis"] == "scratch" and saved["source_identity"] == identity
    assert prepare(data)[3].decisions()["ac01"]["annotations"][0]["id"] == "new-box"
    from darsia.fluidflower.phase_workflow import parse_training_bundles
    from darsia.fluidflower.phase_training.trainer import _load_sources_and_regions, ReviewBundle
    bundles = parse_training_bundles(f"{store.json_path} | {data.manifest}")
    sources, regions, conflicts = _load_sources_and_regions([ReviewBundle(*bundles[0])])
    assert not conflicts and len(regions) == 1
    assert regions[0].frame_index == 1 and regions[0].phase_class == "aqueous"
    assert sources[0].review_root == data.source
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("change", ["timeline", "manifest", "locked", "target", "source", "source_fields", "missing_photo"])
def test_reopen_rejects_changed_sources_or_retired_session(collection, change):
    data = collection
    workspace, _, _, _ = prepare(data)
    if change in {"timeline", "manifest"}:
        path = (data.candidate.per_frame_csv if change == "timeline"
                else data.source / "native_frames/ac01/manifest.json")
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "missing_photo":
        (data.source / "native_frames/ac01/frame_1.png").unlink()
    else:
        path = workspace.output_dir / "review_session.json"
        saved = json.loads(path.read_text())
        if change == "locked":
            saved["editing_locked"] = True
        elif change == "target":
            saved["target_model_id"] = "not-scratch"
        elif change == "source_fields":
            saved["selected_candidates"]["ac01"] = "other"
        else:
            saved["source_identity"]["selected_candidates"]["ac01"] = "other"
        path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        prepare(data)


def test_other_series_collection_is_rejected_before_creating_evidence(collection, monkeypatch):
    data = collection
    candidate = replace(data.candidate, dataset="ffum")
    pair = data.desktop.review.ReviewPair("ac01", candidate, candidate, "single_candidate")
    monkeypatch.setattr(data.desktop.review, "discover_review_inventory", lambda *a, **kw:
        data.desktop.review.ReviewInventory((pair,), {}, data.manifest))
    with pytest.raises(ValueError, match="experiment scope"):
        prepare(data)
    assert not list(data.save_root.rglob("review_session.json"))


def test_original_photo_drawing_and_background_save(collection, qtbot, monkeypatch):
    data = collection
    workspace, _, model, store = prepare(data)
    monkeypatch.setattr(data.desktop, "QuizPreparationTask", lambda *a, **kw: pytest.fail("Model quiz read"))
    monkeypatch.setattr(data.desktop, "_region_phase_image", lambda *a, **kw: pytest.fail("Phase map read"))
    window = subject.create_window(data.desktop, model, store, data.args.cache_root, workspace,
        reviewer_name="Reviewer", use_opengl=False)
    window.show()
    qtbot.waitUntil(lambda: not window.viewer.base_item.pixmap().isNull(), timeout=10000)
    assert not window.pool and not window.annotations and window.quiz_preparation_task is None
    window.annotation_category.setCurrentIndex(window.annotation_category.findData("confirmed_only_free_gas"))
    window.viewer.rectangle_drawn.emit(QRectF(.2, .2, .2, .2))
    window._save_now()
    qtbot.waitUntil(lambda: not window.save_inflight and store.json_path.is_file(), timeout=10000)
    saved = store.decisions()["ac01"]["annotations"]
    assert len(saved) == 1 and saved[0]["category"] == "confirmed_only_free_gas"
    assert saved[0]["elapsed_h"] == 0 and saved[0]["photo_identity"]["source_sha256"] == data.frames[0]["acquisition"]["sha256"]
    assert window.grab().save(str(data.source.parent / "scratch-label.png"))
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=10000)
    assert not window._workflow_busy()
    window.deleteLater()
