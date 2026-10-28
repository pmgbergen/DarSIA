"""Saved profile roles share one implementation and isolate context state."""
import pytest
from darsia.fluidflower import phase_selected_review as review


def test_saved_reader_role_context_restores_on_success_and_failure(monkeypatch):
    before = review._PARAMETER_SECTION.get()
    monkeypatch.setattr(review, 'public_profile', lambda _: review._PARAMETER_SECTION.get())
    assert review.SelectedReviewBindings('first').public_profile({}) == 'first'
    assert review.SelectedReviewBindings('second').public_profile({}) == 'second'
    assert review._PARAMETER_SECTION.get() == before
    def fail(_):
        assert review._PARAMETER_SECTION.get() == 'first'
        raise ValueError('bound role failed')
    monkeypatch.setattr(review, 'public_profile', fail)
    with pytest.raises(ValueError, match='bound role failed'):
        review.SelectedReviewBindings('first').public_profile({})
    assert review._PARAMETER_SECTION.get() == before


def test_saved_reader_unknown_protocol_name_does_not_load_configuration(monkeypatch):
    def forbidden(*args):
        raise AssertionError('unknown protocol name consulted config')
    monkeypatch.setattr(review, 'configured_parameters', forbidden)
    with pytest.raises(AttributeError):
        getattr(review.SelectedReviewBindings('first'), '__path__')
