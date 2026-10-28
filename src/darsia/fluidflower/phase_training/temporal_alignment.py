"""Training adapters for the canonical source-supported temporal alignment."""
from __future__ import annotations
from ..phase import features as _core
from ..phase_saved_review_metadata import configured_parameters

def warp_reference_to_current(*args, **kwargs):
    return _core.warp_reference_to_current(*args,parameters=configured_parameters("training_temporal_alignment")["features"],**kwargs)

def align_temporal_inputs(**kwargs):
    return _core.align_temporal_inputs(parameters=configured_parameters("training_temporal_alignment")["features"],**kwargs)

__all__=["warp_reference_to_current","align_temporal_inputs"]
