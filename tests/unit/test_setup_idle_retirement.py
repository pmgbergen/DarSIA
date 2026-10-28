"""Historical claim metadata must not keep completed Setup services alive."""
import json

from darsia.fluidflower.pipeline import worker, job_queue


def test_original_publish_decision_retires_idle_hint_without_removing_claim_history(tmp_path):
    directory = tmp_path / "claimed" / "fixture-cpu"
    directory.mkdir(parents=True)
    task = dict(run="setup-apply", attempt_id="a" * 32)
    path = directory / "setup-apply.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    assert worker._setup_operation_active(tmp_path)
    job_queue.decide(tmp_path, task["run"], task["attempt_id"], "publish")
    assert not worker._setup_operation_active(tmp_path)
    assert json.loads(path.read_text()) == task
