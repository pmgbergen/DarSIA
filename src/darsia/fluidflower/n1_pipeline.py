"""Legacy schema facade for darsia.fluidflower.mass.n1_pipeline.

Only the historical dictionary field names are adapted here. Canonical workers
import the mass package directly and receive these names in explicit run DATA.
No mathematical function is duplicated.
"""

from .mass import n1_pipeline as _core

# Preserve the historical public attributes, including its scientific file path.
for _name, _value in vars(_core).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
__file__ = _core.__file__

_LEGACY_GEOMETRY_FIELDS = {
    "source_to_native": "H_ac28_to_frame_native",
    "original_domain_to_native": "H_ac28_to_frame_native",
    "carried_homography_provenance": "original_H_ac28_to_native",
}


def load_n1_run(descriptor, *, unc_only=False):
    """Read an original run using its exact historical field names."""
    return _core.load_n1_run(
        descriptor, unc_only=unc_only, geometry_fields=_LEGACY_GEOMETRY_FIELDS
    )


class N1Pipeline(_core.N1Pipeline):
    """Adapt only the original configuration schema for historical callers."""

    def __init__(self, config, *, precision, pixel_math, unc_only=False):
        super().__init__(
            config,
            precision=precision,
            pixel_math=pixel_math,
            unc_only=unc_only,
            geometry_fields=_LEGACY_GEOMETRY_FIELDS,
        )


def _run_n1(task, output, backend, precision, on_progress=None, *, thread_runtime):
    """Supply historical field names to the single canonical numerical body."""
    return _core._run_n1(
        task,
        output,
        backend,
        precision,
        on_progress,
        thread_runtime=thread_runtime,
        geometry_fields=_LEGACY_GEOMETRY_FIELDS,
    )


def run_n1(task, output, backend, precision, on_progress=None):
    """Retain the public thread-limit hook without copying its implementation."""
    return _core.run_n1(task, output, backend, precision, on_progress, _runner=_run_n1)
