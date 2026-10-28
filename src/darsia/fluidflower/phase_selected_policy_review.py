"""Read-only saved policy-profile adapter; all operations share one implementation."""
from .phase_selected_review import SelectedReviewBindings

_BINDINGS = SelectedReviewBindings('selected_policy_review')


def __getattr__(name):
    return getattr(_BINDINGS, name)
