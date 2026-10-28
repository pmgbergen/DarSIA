"""The scientific OpenCV policy and observations in the executing thread.

OpenCV's IPP switch is thread local. The environment value only supplies the
default when OpenCV initializes; it does not prove the state of existing threads.
Every scientific entry initializes its own thread, and subsequent boundaries
verify without resetting a switch that was unexpectedly enabled.
"""

from contextlib import contextmanager
import os
import threading


POLICY = "darsia-scientific-opencv-ipp-off-v1"
_thread = threading.local()


def _cv2():
    # Process local only. Cache admission must precede importing this module.
    os.environ["OPENCV_IPP"] = "disabled"
    import cv2
    return cv2


def require_ipp_off(stage):
    """Observe this actual thread; never repair unexpected reactivation."""
    cv2 = _cv2()
    enabled = bool(cv2.ipp.useIPP())
    record = dict(policy=POLICY, stage=stage, pid=os.getpid(),
                  thread_id=threading.get_ident(),
                  native_thread_id=threading.get_native_id(),
                  opencv_version=cv2.__version__, use_ipp=enabled,
                  use_ipp_not_exact=bool(cv2.ipp.useIPP_NotExact()),
                  process_environment_opencv_ipp=os.environ.get("OPENCV_IPP"),
                  observation_scope="executing-thread-at-boundary")
    if enabled:
        raise RuntimeError("Scientific OpenCV IPP policy violated at " + str(stage))
    return record


def scientific_thread(stage):
    """Initialize a newly encountered scientific thread, otherwise verify it."""
    cv2 = _cv2()
    if getattr(_thread, "pid", None) != os.getpid():
        cv2.ipp.setUseIPP(False)
        record = require_ipp_off(stage)
        _thread.pid = os.getpid()
        return record
    return require_ipp_off(stage)


def completed_frame(before, stage):
    """Return actual observations only after successful computation."""
    after = require_ipp_off(stage)
    if before["pid"] != after["pid"] or before["thread_id"] != after["thread_id"]:
        raise RuntimeError("Scientific frame changed its executing thread")
    return dict(policy=POLICY, before=before, after=after,
                saved_inputs="original provenance; no retroactive IPP observation")


def validate_observation(record):
    """Validate saved current observations without measuring an old product."""
    if not isinstance(record, dict) or record.get("policy") != POLICY:
        raise ValueError("Scientific OpenCV observation has another policy")
    observations = (record["before"], record["after"]) if "before" in record else (record,)
    for observed in observations:
        if (not isinstance(observed, dict) or observed.get("policy") != POLICY
                or observed.get("use_ipp") is not False
                or observed.get("observation_scope") != "executing-thread-at-boundary"
                or any(type(observed.get(key)) is not int or observed[key] <= 0
                       for key in ("pid", "thread_id", "native_thread_id"))
                or not isinstance(observed.get("opencv_version"), str) or not observed["opencv_version"]
                or not isinstance(observed.get("stage"), str) or not observed["stage"]
                or type(observed.get("use_ipp_not_exact")) is not bool):
            raise ValueError("Scientific OpenCV observation does not verify IPP off")
    if len(observations) == 2 and any(
            observations[0].get(key) != observations[1].get(key)
            for key in ("pid", "thread_id", "native_thread_id", "opencv_version")):
        raise ValueError("Scientific OpenCV frame changed its executing thread")


@contextmanager
def scientific_frame(stage):
    """Preserve original exceptions; verify success before its caller commits."""
    record = dict(policy=POLICY, before=scientific_thread(stage + ":before"))
    yield record
    record.update(completed_frame(record["before"], stage + ":after"))
