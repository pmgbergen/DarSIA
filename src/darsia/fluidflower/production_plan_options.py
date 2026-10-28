"""Explicit storage choices for fresh production sources, before original sealing."""
from copy import deepcopy

from . import storage_profile as profiles
from .storage_outputs import parse_request
from .storage_resolution import resolution_mm


def phase2_source_options(source, *, storage_profile=None,
                          stored_resolution="saved", precision="float32"):
    """Copy one source and change only its explicitly chosen storage options.

    Original descriptors, fitted anchors and pixel-domain data remain intact.
    This metadata adapter neither writes nor seals; the original phase-two
    sealer and later job/provider admission must still validate the result.
    """
    if not isinstance(source, dict):
        raise ValueError("An explicit phase-two source dictionary is required")
    result = deepcopy(source)
    setting = profiles.resolve(source)
    if storage_profile is not None:
        chosen = profiles.new_job_profile(storage_profile)
        if setting["colour_intermediates"] != "keep":
            raise ValueError("Production profile changes require KEEP; a saved DROP choice cannot be replaced")
        entry = deepcopy(source.get(profiles.SETTING) or {})
        entry.update(profile=chosen, colour_intermediates="keep")
        result[profiles.SETTING] = entry
        profiles.require_run_choice(chosen, result, pipeline=profiles.PHASE_TWO_ROUTE)

    request = source.get("storage_data")
    if request is not None:
        # Validate the original first: changing a choice cannot repair or hide
        # a malformed source request. Never persist the parsed precision field.
        parse_request({"storage": request}, precision)
    if isinstance(stored_resolution, str) and stored_resolution == "saved":
        return result
    step = resolution_mm(stored_resolution)
    if request is None:
        if step is None:
            return result  # Native keeps the original request-free publication.
        raise ValueError("Coarse stored resolution requires saved storage_data with an explicit extent_m")
    changed = deepcopy(request)
    if step is not None and "extent_m" not in changed:
        raise ValueError("Coarse stored resolution requires the saved explicit extent_m")
    changed["resolution_mm"] = "native" if step is None else step
    parse_request({"storage": changed}, precision)
    result["storage_data"] = changed
    return result
