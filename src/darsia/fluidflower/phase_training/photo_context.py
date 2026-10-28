"""Compatibility photo-cohort names over the configured source reader."""
from darsia.fluidflower.native.config import require_configuration
from darsia.fluidflower.native import source as _core

class Track1FramePhotoCohort(_core.PhotoCohort):
    """Keep review-facing source signatures during the method extraction."""
    def __init__(self,native_manifest_path):
        super().__init__(native_manifest_path,parameters=require_configuration())
    H_v5_B_to_frame=_core.PhotoCohort.baseline_to_frame
    H_previous_to_frame=_core.PhotoCohort.previous_to_frame
    def feature_support(self,index,previous_index,linear_input):
        from .frame_context import V5FrameGeometryCohort
        return V5FrameGeometryCohort.feature_support(self,index,previous_index,linear_input)

def bind_photo_only_marker(*args,**kwargs):
    return _core.bind_photo_marker(*args,parameters=require_configuration(),**kwargs)

def __getattr__(name):
    keys={'CONTRACT':'photo_contract','COHORT_CONTRACT':'cohort_contract','PHOTO_CORE_CONTRACT':'photo_core_contract','PHOTO_SCOPE':'photo_scope','SHAPE':'native_shape_hw'}
    if name in keys:
        value=require_configuration()['source'][keys[name]]
        return tuple(value) if isinstance(value,list) else value
    alias={'_read':'_read','_check':'_check','_same_path':'_same_path','_same_h':'_same_h','_marker_policy':'_marker_policy'}
    if name in alias:
        function=getattr(_core,alias[name])
        return lambda *args,**kwargs:function(*args,parameters=require_configuration(),**kwargs)
    if name in ('_source_photo_grid','project_source_photo_safe_core','photo_safe_core_policy_metadata','_normal_h'):
        from . import frame_context
        return getattr(frame_context,name)
    raise AttributeError(name)

__all__=['CONTRACT','COHORT_CONTRACT','Track1FramePhotoCohort','bind_photo_only_marker']
