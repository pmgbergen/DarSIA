from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import toml
from PySide6.QtWidgets import QLineEdit

from darsia.fluidflower.calibration_model_training import require_prepared_normalisation
from darsia.fluidflower import native_phase_steps
from darsia.fluidflower.phase_saved_assets import asset
from darsia.fluidflower.image_normalisation import choice_record, MATCH_REFERENCE
from darsia.fluidflower.phase_training.raw_review_training import FAMILY


def test_current_prepared_choice_admits_fresh_launch_when_historical_import_is_false(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.native.full_photometry import ORDER
    decision = dict(actor="operator", approved_utc="2026-10-26T00:00:00+00:00",
        reason="Use the saved reference lighting", decision_ref="image-normalisation-method-choice")
    choice = choice_record("generic", ["one", "two"], MATCH_REFERENCE, "two", decision=decision)
    document = dict(schema_version=1, series_id="generic", runs=dict(ids=["one", "two"]),
        image_normalisation=choice, phase_normalisation=dict(source_import=False))
    (tmp_path / "series.toml").write_text(toml.dumps(document))
    record = tmp_path / "prepared.json"
    record.write_text(json.dumps(dict(format="darsia-prepared-image-normalisation-v1", choice=choice)))
    binding = tmp_path / "raw.json"
    binding.write_text(json.dumps(dict(correction_order=ORDER,
        normalisation=dict(choice=choice, record=asset(record)))))
    request = dict(model_id="fresh", output_root=str(tmp_path / "fresh"),
        settings=dict(training_family=FAMILY, raw_image_basis=dict(runs=[dict(raw_binding=asset(binding))])))
    calls = []
    steps = SimpleNamespace(_ready=False, model_name=QLineEdit("New version"), series_root=tmp_path,
        studio=SimpleNamespace(reviewer=QLineEdit("operator")), _named_model_context={},
        phase_workflow=SimpleNamespace(before_launch=lambda request: calls.append("original-source-admission")),
        _original_launch=lambda *args: calls.append("actual-process-launch"), refresh=lambda: None)
    monkeypatch.setattr(native_phase_steps, "save_model_ui_name", lambda *args: None)
    native_phase_steps.NativePhaseSteps._launch_named_job(steps, request, [], tmp_path, "ready")
    assert calls == ["original-source-admission", "actual-process-launch"]
    assert steps._ready is False  # Historical readiness was not overwritten.
    document["image_normalisation"] = choice_record("generic", ["one", "two"], "off", decision=decision)
    (tmp_path / "series.toml").write_text(toml.dumps(document))
    with pytest.raises(ValueError, match="another saved image-normalisation choice"):
        require_prepared_normalisation(request, tmp_path)
    with pytest.raises(ValueError, match="Saved normalisation"):
        require_prepared_normalisation(dict(settings={}), tmp_path)
