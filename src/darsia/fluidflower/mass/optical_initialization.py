"""The original DarSIA default-mass starting signal windows, before fitting."""


def initial_flash_bounds():
    """Return the unchanged original unfitted colour-to-mass start convention."""
    return dict(min_value_aq=0, max_value_aq=0.75, min_value_g=0.75, max_value_g=1.0)
