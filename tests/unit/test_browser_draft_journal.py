"""Browser snapshot compaction preserves recovery, deletion and later saves."""
from functools import partial
import json

from darsia.fluidflower.review_snapshot_journal import append_bounded_snapshot_journal
from tests.unit.test_phase_model_folder import review_module, _review_pair  # noqa: F401


def test_compacted_browser_draft_restores_latest_and_keeps_delete_tombstone(tmp_path, monkeypatch, review_module):
    monkeypatch.setattr(review_module, "append_bounded_snapshot_journal",
                        partial(append_bounded_snapshot_journal, max_bytes=1))
    pair = _review_pair(review_module, tmp_path)
    store = review_module.ReviewStore(tmp_path / "session", [pair])
    def save(identifier):
        return store.save_browser_draft(dict(run=pair.run, reviewer="Reviewer",
            draft={"annotations": [{"id": identifier, "text": "saved snapshot"}]}))
    first, latest = save("first"), save("latest")
    path = store._browser_draft_path(pair.run, "Reviewer")
    history = store._browser_draft_history_path(pair.run, "Reviewer")
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert len(records) == 1 and records[0]["payload"] == latest
    path.unlink()
    assert store.browser_draft(pair.run, "Reviewer") == latest
    path.write_text(json.dumps(first))
    store.save_browser_draft(dict(run=pair.run, reviewer="Reviewer", delete=True))
    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert len(records) == 1 and records[0]["operation"] == "delete"
    assert not path.exists() and store.browser_draft(pair.run, "Reviewer") is None
    path.write_text(json.dumps(first))  # A stale redundant JSON cannot revive a deletion.
    assert store.browser_draft(pair.run, "Reviewer") is None
    after_delete = save("new draft")
    assert store.browser_draft(pair.run, "Reviewer") == after_delete
