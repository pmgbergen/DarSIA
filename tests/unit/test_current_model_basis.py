from pathlib import Path
from types import SimpleNamespace
import re

import pytest

from darsia.fluidflower import series_method
from darsia.fluidflower.pipeline import current_build_job, current_model_basis


def test_jobs_p1_preparation_cannot_bypass_current_model_basis_before_creating_files(tmp_path, monkeypatch):
    monkeypatch.setattr(series_method, "read_series_method", lambda root: SimpleNamespace(
        active="phase_model", model_version="newly-trained-model"))
    from darsia.fluidflower import analysis_preparation, store_map
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda *args: pytest.fail("No preparation before model binding"))
    monkeypatch.setattr(store_map, "read_store_map", lambda *args: pytest.fail("No output admission before model binding"))
    before = list(tmp_path.iterdir())
    with pytest.raises(ValueError, match=re.escape(current_model_basis.UNBOUND_MODEL_REASON)):
        current_build_job.prepare_current_build_job(
            dict(complete=True, job_facts={"pipeline": "series-setup"}),
            series_root=tmp_path, series_id="sample")
    assert list(tmp_path.iterdir()) == before


def test_selecting_another_model_does_not_invent_a_producer_binding(tmp_path, monkeypatch):
    for model in ("original-model", "newly-trained-model"):
        monkeypatch.setattr(series_method, "read_series_method", lambda root, model=model:
            SimpleNamespace(active="phase_model", model_version=model))
        with pytest.raises(ValueError, match=re.escape(current_model_basis.UNBOUND_MODEL_REASON)):
            current_model_basis.require_current_model_basis(tmp_path)


def test_missing_model_keeps_the_actionable_model_selection_message(tmp_path, monkeypatch):
    monkeypatch.setattr(series_method, "read_series_method", lambda root: SimpleNamespace(
        active="phase_model", model_version=""))
    with pytest.raises(ValueError, match="Choose and approve a phase model"):
        current_model_basis.require_current_model_basis(tmp_path)
