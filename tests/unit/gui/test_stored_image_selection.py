"""Local metadata fixtures only; these tests do not load Studio or images."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from darsia.fluidflower.stored_image_selection import loaded_studio_stored_plan


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "registered-series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version=1\nseries_id="test_series"\n', encoding="utf-8"
    )
    model = tmp_path / "explicit-model"
    campaign = model / "validation"
    review = tmp_path / "native-review"
    native_path = review / "native_frames" / "run1" / "manifest.json"
    frames = [
        dict(
            source_path=str(tmp_path / f"never-read-{i}.raw"),
            source_sha256=str(i) * 64,
            elapsed_h=float(i),
            frame_index=i,
        )
        for i in range(7)
    ]
    native = dict(dataset="test_series", run="run1", frame_sources=frames)
    native_sha = write_json(native_path, native)
    binding = dict(
        combined_root=str(campaign),
        selected_candidates={"run1": "candidate-1"},
        pipeline_sha256="p" * 64,
        model_sha256="m" * 64,
        physics_runtime={"saved": "x"},
    )
    job = dict(
        run="run1",
        dataset="test_series",
        candidate_id="candidate-1",
        params_sha256=binding["pipeline_sha256"],
        optical_model_sha256=binding["model_sha256"],
        physics_runtime=binding["physics_runtime"],
        review_output_dir=str(review),
        native_manifest_sha256=native_sha,
    )
    postflight = dict(jobs=[job])
    write_json(campaign / "postflight_manifest.json", postflight)
    write_json(campaign / "validation_binding.json", binding)
    entry = dict(root=str(model), validation_binding=deepcopy(binding))
    selector = NS(currentData=lambda: str(model))
    studio = NS(
        _active_model_series="testseries",
        validation_model=selector,
        _selected_validation_model=lambda: entry,
    )
    steps = NS(
        existing_basis=NS(isChecked=lambda: True),
        series_id="testseries",
        series_root=root,
    )
    page = NS(
        studio=studio, native_steps=steps, series_id="testseries", project_root=root
    )
    workspace = NS(
        studio_page=page, series_id="testseries", series_roots={"testseries": root}
    )
    window = NS(phase_model_workspace=workspace, _method_series_root=lambda: root)
    metadata = NS(series_root=str(root), run_id="run1")
    calls = []

    def verify(parent, source, *, saved_comparison):
        calls.append((Path(parent), Path(source), saved_comparison))
        return deepcopy(binding)

    def resolve(path):
        return Path(path).resolve()

    def load(**kwargs):
        return loaded_studio_stored_plan(
            window,
            metadata,
            verify_binding=kwargs.pop("verify", verify),
            analysis_root=kwargs.pop("analysis", lambda path, run: path),
            resolve_path=resolve,
            **kwargs,
        )

    return NS(**locals())


def test_exact_selected_loaded_basis_uses_saved_comparison_and_preserves_seven_frames(
    saved,
):
    result = saved.load()
    assert result == dict(
        path=str(saved.native_path),
        sha256=saved.native_sha,
        size_bytes=len(saved.native_path.read_bytes()),
    )
    assert saved.calls == [(saved.model, saved.campaign, True)]
    assert len(json.loads(saved.native_path.read_bytes())["frame_sources"]) == 7
    assert not any(saved.tmp_path.glob("*.raw"))


@pytest.mark.parametrize(
    "missing",
    [
        "workspace",
        "page",
        "studio",
        "steps",
        "basis",
        "selection",
        "binding",
        "loading",
    ],
)
def test_missing_existing_ui_context_never_discovers_or_verifies(saved, missing):
    if missing == "workspace":
        saved.window.phase_model_workspace = None
    elif missing == "page":
        saved.workspace.studio_page = None
    elif missing == "studio":
        saved.page.studio = None
    elif missing == "steps":
        saved.page.native_steps = None
    elif missing == "basis":
        saved.steps.existing_basis.isChecked = lambda: False
    elif missing == "selection":
        saved.selector.currentData = lambda: ""
    elif missing == "binding":
        saved.entry.pop("validation_binding")
    elif missing == "loading":
        saved.entry["validation_binding_loading"] = True
    assert saved.load() is None
    assert not saved.calls


@pytest.mark.parametrize(
    "location", ["workspace", "page", "steps", "studio", "root", "selected-root"]
)
def test_stale_other_series_or_other_model_context_does_not_seed(saved, location):
    if location == "workspace":
        saved.workspace.series_id = "different"
    elif location == "page":
        saved.page.series_id = "different"
    elif location == "steps":
        saved.steps.series_root = saved.tmp_path / "elsewhere"
    elif location == "studio":
        saved.studio._active_model_series = "different"
    elif location == "root":
        saved.window._method_series_root = lambda: saved.tmp_path / "elsewhere"
    else:
        saved.selector.currentData = lambda: str(saved.tmp_path / "other-model")
    assert saved.load() is None
    assert not saved.calls


@pytest.mark.parametrize(
    "alter",
    [
        "selection",
        "job",
        "duplicate-job",
        "dataset",
        "params",
        "model",
        "physics",
        "sha",
        "missing-sha",
    ],
)
def test_postflight_does_not_infer_an_unbound_source(saved, alter):
    if alter == "selection":
        saved.binding["selected_candidates"] = {}
    elif alter == "job":
        saved.postflight["jobs"] = []
    elif alter == "duplicate-job":
        saved.postflight["jobs"].append(dict(saved.job))
    elif alter == "dataset":
        saved.job["dataset"] = "different"
    elif alter == "params":
        saved.job["params_sha256"] = "different"
    elif alter == "model":
        saved.job["optical_model_sha256"] = "different"
    elif alter == "physics":
        saved.job["physics_runtime"] = {"different": True}
    elif alter == "sha":
        saved.job["native_manifest_sha256"] = "0" * 64
    else:
        saved.job.pop("native_manifest_sha256")
    saved.entry["validation_binding"] = deepcopy(saved.binding)
    write_json(saved.campaign / "postflight_manifest.json", saved.postflight)
    with pytest.raises(ValueError):
        saved.load()


def test_collection_does_not_trigger_original_full_payload_verifier(saved):
    saved.postflight["format"] = "fluidflower-phase-review-collection-v1"
    write_json(saved.campaign / "postflight_manifest.json", saved.postflight)
    with pytest.raises(ValueError, match="payload audit.*Import"):
        saved.load()
    assert not saved.calls


def test_original_verifier_failure_is_not_hidden(saved):
    def fail(*args, **kwargs):
        raise ValueError("original saved profile mismatch")

    with pytest.raises(ValueError, match="original saved profile mismatch"):
        saved.load(verify=fail)


@pytest.mark.parametrize("changed", ["selection", "postflight", "binding", "series"])
def test_changed_binding_or_selected_ui_during_load_refuses(saved, changed):
    def mutate(*args, **kwargs):
        if changed == "selection":
            saved.steps.existing_basis.isChecked = lambda: False
        elif changed == "postflight":
            write_json(
                saved.campaign / "postflight_manifest.json",
                dict(saved.postflight, changed=True),
            )
        elif changed == "binding":
            write_json(
                saved.campaign / "validation_binding.json",
                dict(saved.binding, changed=True),
            )
        else:
            (saved.root / "series.toml").write_text(
                'schema_version=1\nseries_id="different"\n', encoding="utf-8"
            )
        return deepcopy(saved.binding)

    with pytest.raises(ValueError, match="changed during Load"):
        saved.load(verify=mutate)


def test_original_analysis_lineage_is_used_with_preferred_explicit_root(saved):
    explicit = saved.tmp_path / "explicit-analysis"
    saved.job["analysis_source_review_root"] = str(explicit)
    write_json(saved.campaign / "postflight_manifest.json", saved.postflight)
    calls = []

    def follow(path, run):
        calls.append((path, run))
        return saved.review

    assert saved.load(analysis=follow)["sha256"] == saved.native_sha
    assert calls == [(explicit, "run1")]


def test_loaded_binding_change_requires_explicit_refresh(saved):
    def changed(*args, **kwargs):
        return dict(saved.binding, completed_utc="new")

    with pytest.raises(ValueError, match="binding changed"):
        saved.load(verify=changed)


def test_original_saved_comparison_verifier_with_local_json_model_metadata(
    saved, monkeypatch
):
    from darsia.fluidflower import phase_runtime
    from darsia.fluidflower.phase_validation import _model_identity

    write_json(saved.model / "summary.json", {"model_id": "explicit-local-model"})
    for name in ("co2_presence_signal.json", "gas_within_co2_signal.json"):
        write_json(
            saved.model / "models" / "generic" / name, {"saved_test_metadata": True}
        )
    model_id, identity = _model_identity(saved.model)
    saved.binding.update(
        model_id=model_id,
        model_sha256=identity,
        completed_utc="2026-10-16T00:00:00Z",
        physics_profile={"saved": True},
    )
    saved.postflight.update(
        model_id=model_id,
        model_sha256=identity,
        physics_runtime=saved.binding["physics_runtime"],
        physics_profile={"saved": True},
    )
    saved.job["optical_model_sha256"] = identity
    saved.entry["validation_binding"] = deepcopy(saved.binding)
    write_json(saved.campaign / "postflight_manifest.json", saved.postflight)
    write_json(saved.campaign / "validation_binding.json", saved.binding)
    monkeypatch.setattr(
        phase_runtime, "load_physics_profile", lambda root: {"saved": True}
    )
    assert saved.load(verify=None)["sha256"] == saved.native_sha
    assert not saved.calls


def test_original_radiometry_source_lineage_reads_only_manifests(saved, monkeypatch):
    from darsia.fluidflower.phase_training import radiometry

    monkeypatch.setattr(
        radiometry, "resolve_review_path", lambda path: Path(path).resolve()
    )
    display = saved.tmp_path / "derived-display"
    write_json(
        display / "native_frames" / "run1" / "manifest.json",
        {
            "analysis_source_review_root": str(saved.review),
            "content_box_xywh": [0, 0, 10, 10],
        },
    )
    native = json.loads(saved.native_path.read_bytes())
    native["content_box_xywh"] = [0, 0, 10, 10]
    saved.job["native_manifest_sha256"] = write_json(saved.native_path, native)
    saved.job["review_output_dir"] = str(display)
    write_json(saved.campaign / "postflight_manifest.json", saved.postflight)
    assert saved.load(analysis=None)["path"] == str(saved.native_path)


def test_descriptor_is_consumed_as_actual_seven_frame_native_plan(saved):
    from darsia.fluidflower.image_selection_proposal import load_stored_native_plan

    saved.metadata.frames = tuple(
        dict(row, image_id=f"image-{i}", input_index=i)
        for i, row in enumerate(saved.frames)
    )
    result = saved.load()
    imported = load_stored_native_plan(
        result["path"], saved.metadata, series_id="test_series", expected=result
    )
    assert len(imported["selected"]) == 7
    assert imported["label"] == "Stored plan (imported native manifest)"
    assert [row["source_path"] for row in imported["selected"]] == [
        row["source_path"] for row in saved.frames
    ]


def test_default_provider_is_deferred_until_explicit_dialog_load(
    qtbot, monkeypatch, tmp_path
):
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QComboBox, QWidget
    from darsia.fluidflower import image_selection_ui as ui

    calls = []
    monkeypatch.setattr(
        ui, "loaded_studio_stored_plan", lambda window, metadata: calls.append(metadata)
    )

    class Dialog(QObject):
        saved = Signal(str)

        def __init__(self, root, parent, **kwargs):
            super().__init__(parent)
            self.provider = kwargs["stored_plan_provider"]
            self.run_combo = QComboBox()

        def exec(self):
            return 0

    parent = QWidget()
    parent._method_series_root = lambda: tmp_path
    parent.run_combo = QComboBox()
    workflow = ui.NativeImageSelectionWorkflow(parent, dialog_factory=Dialog)
    assert calls == []
    workflow.open()
    assert calls == []
    metadata = NS(run_id="run1")
    workflow.dialog.provider(metadata)
    assert calls == [metadata]
