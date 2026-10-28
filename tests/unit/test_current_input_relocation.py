from pathlib import Path

import pytest

from darsia.fluidflower.mass.n1_pipeline import asset, _write_json
from darsia.fluidflower.pipeline.current_inputs import current_input_reader
from darsia.fluidflower.pipeline.current_statistics_recipe import _sized


def test_relocated_metadata_uses_declared_path_and_preserves_content_identity(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    actual = _write_json(tmp_path / "kept.json", {"retained": True})
    old = dict(actual, path=str(tmp_path / "old" / "kept.json"))
    monkeypatch.setattr(storage_paths, "storage_path_remapper", lambda: lambda path:
                        actual["path"] if str(path) == old["path"] else str(path))
    reader = current_input_reader()
    assert reader.json(old) == {"retained": True}
    assert _sized(old, reader) == actual
    assert not Path(old["path"]).exists()
    Path(actual["path"]).write_text('{"changed":true}')
    with pytest.raises(ValueError):
        reader.json(old)


def test_output_admission_checks_configured_results_root(tmp_path, monkeypatch):
    from darsia.fluidflower import analysis_preparation, job_filesystem, store_map
    from darsia.fluidflower.pipeline import current_build_job
    from types import SimpleNamespace
    from darsia.fluidflower.pipeline import current_model_basis
    monkeypatch.setattr(current_model_basis, "require_current_model_basis", lambda *args: None)
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda *a: {})
    monkeypatch.setattr(store_map, "read_store_map", lambda *a: SimpleNamespace(
        roots=lambda *a: {"results": str(tmp_path / "shared-results")}))
    seen = []
    monkeypatch.setattr(job_filesystem, "inspect_results_root", lambda path:
        seen.append(path) or SimpleNamespace(allowed=False, reason="unsupported transport"))
    monkeypatch.setattr(job_filesystem, "inspect_local_file", lambda *a, **k:
        pytest.fail("Output admission must not use the local settings-file guard"))
    with pytest.raises(ValueError, match="unsupported transport"):
        current_build_job.prepare_current_build_job(dict(complete=True,
            job_facts={"pipeline": "series-setup"}), series_root=tmp_path, series_id="series")
    assert seen == [tmp_path / "shared-results"]


def test_admitted_shared_results_continues_through_original_preparation(tmp_path, monkeypatch):
    from darsia.fluidflower import analysis_preparation, job_filesystem, store_map
    from darsia.fluidflower.pipeline import current_build_job, current_calibration_selection
    from types import SimpleNamespace
    from darsia.fluidflower.pipeline import current_model_basis
    monkeypatch.setattr(current_model_basis, "require_current_model_basis", lambda *args: None)
    monkeypatch.setattr(analysis_preparation, "job_identity", lambda *a: {})
    monkeypatch.setattr(store_map, "read_store_map", lambda *a: SimpleNamespace(
        roots=lambda *a: {"results": str(tmp_path / "shared-results")}))
    monkeypatch.setattr(job_filesystem, "inspect_results_root", lambda path:
        SimpleNamespace(allowed=True, protocol="smb"))
    monkeypatch.setattr(job_filesystem, "inspect_local_file", lambda *a, **k:
        pytest.fail("NAS results must not be rejected by a local-only file check"))
    def original(*args, **kwargs):
        raise RuntimeError("original selection producer reached")
    monkeypatch.setattr(current_calibration_selection, "save_current_calibration_selection", original)
    with pytest.raises(RuntimeError, match="original selection producer reached"):
        current_build_job.prepare_current_build_job(dict(complete=True, root=str(tmp_path / "completed"),
            job_facts={"pipeline": "series-setup"}), series_root=tmp_path, series_id="series")
