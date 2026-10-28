"""Original command builder is unchanged unless this is the native adapter."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from darsia.fluidflower.ui import model_training as original


@pytest.fixture
def command(monkeypatch, tmp_path):
    monkeypatch.setattr(
        original, "default_execution_cache_root", lambda: tmp_path / "cache"
    )
    return original.SharedPhaseModelTrainingPage._review_command


@pytest.fixture
def context(tmp_path):
    campaign = tmp_path / "explicit saved campaign"
    campaign.mkdir()
    (campaign / "postflight_manifest.json").write_text('{"jobs": []}', encoding="utf-8")
    binding = {
        "ffac_root": str(campaign),
        "ffum_root": str(campaign),
        "combined_root": str(campaign),
        "selected_candidates": {"ac52": "candidate-saved-52"},
    }
    model = {
        "model_id": "saved-a",
        "root": str(tmp_path / "model a"),
        "series_scope": "ffac",
        "display_name": "Saved model A",
        "created_timestamp": 2,
        "validation_binding": binding,
    }
    second = {
        "model_id": "saved-b",
        "root": str(tmp_path / "model b"),
        "series_scope": "ffac",
        "created_timestamp": 1,
    }
    foreign = {
        "model_id": "foreign",
        "root": str(tmp_path / "foreign model"),
        "series_scope": "ffum",
        "created_timestamp": 3,
    }
    return NS(
        _selected_validation_model=lambda: model,
        _review_session_id_for=lambda model_id: "existing-session-" + model_id,
        reviewer=NS(text=lambda: "Existing researcher"),
        review_save_root=NS(text=lambda: str(tmp_path / "saved reviews")),
        review_gpu=NS(isChecked=lambda: False),
        _validation_model_label=lambda model: model.get("display_name")
        or model["model_id"],
        model_history=[foreign, second, model],
        _active_model_series="ffac",
    )


def test_standalone_argv_identical_and_native_appends_only_filtered_choices(
    command, context
):
    expected = original.SharedPhaseModelTrainingPage._review_command(
        context, "open_correction"
    )
    assert command(context, "open_correction") == expected
    context._native_step_adapter = None
    assert command(context, "open_correction") == expected
    context._native_step_adapter = object()
    actual = command(context, "open_correction")
    assert actual[: len(expected)] == expected
    added = actual[len(expected) :]
    assert added[:2] == [
        "--review-current-model-root",
        context._selected_validation_model()["root"],
    ]
    assert added[2::2] == ["--review-model-choice", "--review-model-choice"]
    choices = [json.loads(value) for value in added[3::2]]
    assert [row["root"] for row in choices] == [
        context.model_history[2]["root"],
        context.model_history[1]["root"],
    ]
    assert "foreign model" not in json.dumps(choices)
    assert actual.count("--review-session-id") == 1
    assert actual.count("--single-model") == 1
    assert actual.count("--single-candidate") == 1
    assert "--approval-only" not in actual


@pytest.mark.parametrize("mode", ["model_validation", "active_learning"])
def test_native_other_review_purposes_are_exactly_unchanged(command, context, mode):
    expected = command(context, mode)
    context._native_step_adapter = object()
    assert command(context, mode) == expected
    assert "--review-model-choice" not in expected


def test_missing_original_prepared_binding_is_not_bypassed_for_native(command, context):
    context._native_step_adapter = object()
    context._selected_validation_model = lambda: {
        "model_id": "unprepared",
        "root": "one",
    }
    with pytest.raises(ValueError, match="no generated validation phase maps"):
        command(context, "open_correction")
