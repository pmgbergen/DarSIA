"""A stopped feature extraction process costs one new attempt, not the training."""
from concurrent.futures.process import BrokenProcessPool

import pytest

from darsia.fluidflower.phase_training.raw_fresh_training import load_features_with_retries


class _Trainer:
    """Stands in for the trainer: fails the first ``failures`` extractions abruptly."""

    def __init__(self, failures):
        self.failures, self.calls, self.messages = failures, [], []

    def _load_feature_caches(self, sources, regions, *, extraction_workers, **kwargs):
        self.calls.append((sources, regions, extraction_workers, kwargs))
        if len(self.calls) <= self.failures:
            raise BrokenProcessPool("A process in the process pool was terminated abruptly")
        return "features"

    def _status(self, message):
        self.messages.append(message)


def test_a_stopped_worker_extracts_again_with_half_the_workers_and_the_same_inputs():
    trainer, used = _Trainer(failures=2), []
    result = load_features_with_retries(trainer, "sources", "regions", extraction_workers=16,
                                        cache_root="cache", used=used)
    assert result == "features" and used == [16, 8, 4]
    assert [call[2] for call in trainer.calls] == [16, 8, 4]
    assert all(call[:2] == ("sources", "regions") and call[3] == {"cache_root": "cache"}
               for call in trainer.calls)
    assert len(trainer.messages) == 2 and "4 workers" in trainer.messages[-1]


def test_the_last_attempt_reports_the_stopped_worker():
    trainer, used = _Trainer(failures=3), []
    with pytest.raises(BrokenProcessPool):
        load_features_with_retries(trainer, "sources", "regions", extraction_workers=2, used=used)
    assert used == [2, 1, 1]


def test_an_ordinary_error_is_not_retried():
    class Failing(_Trainer):
        def _load_feature_caches(self, *args, **kwargs):
            super()._load_feature_caches(*args, **kwargs)
            raise ValueError("Corrupt feature cache")

    trainer = Failing(failures=0)
    with pytest.raises(ValueError, match="Corrupt feature cache"):
        load_features_with_retries(trainer, "sources", "regions", extraction_workers=8)
    assert len(trainer.calls) == 1
