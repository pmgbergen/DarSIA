import json

from darsia.fluidflower.series_setup_jobs_ui import current_version_job_root


def test_prepare_after_source_update_keeps_previous_job_and_suggests_fresh_sibling(tmp_path):
    old = tmp_path / "setup-previous"
    old.mkdir()
    path = old / "job.json"
    raw = json.dumps(dict(pipeline="series-setup", series_id="SERIES", commit="1" * 40)).encode()
    path.write_bytes(raw)
    fresh = current_version_job_root(old, "2" * 40, "SERIES")
    assert fresh != old and fresh.parent == old.parent and not fresh.exists()
    assert path.read_bytes() == raw
    assert current_version_job_root(old, "1" * 40, "SERIES") == old
    assert current_version_job_root(old, "2" * 40, "OTHER") == old
