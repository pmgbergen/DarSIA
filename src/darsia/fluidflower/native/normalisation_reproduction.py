"""In-memory qualification hooks for the actual saved observed-sand method.

These hooks neither choose a recipe nor write or adopt an asset. The caller
must supply the exact source-bound RAW-derived grid, support, chart and final
installation inputs identified by the saved producer. Refitting against a
saved installation alone is not proof that the whole RAW/installation rebuild
has been reproduced.
"""

from dataclasses import fields
import struct

import numpy as np

from . import photometry
from ..normalisation_recipe import require_exact_array


def require_exact_fit(actual, saved):
    """Compare every original Baseline member, with no numerical tolerance."""
    if type(actual) is not type(saved):
        raise ValueError("Saved fit type differs from the reproduced fit.")
    for member in fields(saved):
        left, right = getattr(actual, member.name), getattr(saved, member.name)
        if isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray):
                raise ValueError("Saved fit member differs: " + member.name)
            try:
                require_exact_array(left, right)
            except ValueError as error:
                detail = f"fit member {member.name}: {left.shape}/{left.dtype} versus {right.shape}/{right.dtype}"
                if left.shape == right.shape and left.dtype.kind in "biuf" and right.dtype.kind in "biuf":
                    finite = np.isfinite(left) & np.isfinite(right)
                    if np.any(finite):
                        difference = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
                        detail += f", max finite absolute difference {float(difference.max()):.17g}"
                raise ValueError(f"{error} ({detail})") from error
        elif isinstance(right, dict):
            if not isinstance(left, dict) or left.keys() != right.keys():
                raise ValueError("Saved fit support differs: " + member.name)
            for key in right:
                if type(left[key]) is not type(right[key]):
                    raise ValueError("Saved fit scalar type differs: " + member.name)
                if isinstance(right[key], float):
                    equal = struct.pack("!d", left[key]) == struct.pack("!d", right[key])
                else:
                    equal = left[key] == right[key]
                if not equal:
                    raise ValueError("Saved fit support differs: " + member.name)
        elif isinstance(right, float):
            if struct.pack("!d", left) != struct.pack("!d", right):
                raise ValueError("Saved fit scalar differs: " + member.name)
        elif left != right:
            raise ValueError("Saved fit member differs: " + member.name)


def refit_saved_baseline(*, run, log_rgb, fit_support, chart_xy, chart_log_rgb,
                        reference_chart_log_rgb, chart_usable, matrix,
                        final_consensus, final_flat, grid, saved_fit, parameters):
    """Use the original observed-sand fit on explicit frozen final inputs.

    No physical-facies replacement, new consensus fit, mask inference, source
    rendering or chemistry ranking is performed. All of those source/recipe
    bindings must already be checked by the execution caller.
    """
    result = photometry.fit_observed_baseline(
        run=run, logB=log_rgb, facies=fit_support, chart_xy=chart_xy,
        chart_log_rgb=chart_log_rgb, ref_chart_log_rgb=reference_chart_log_rgb,
        chart_usable=chart_usable, matrix=matrix, rig_consensus=final_consensus,
        flat=final_flat, grid=grid, parameters=parameters,
    )
    require_exact_fit(result, saved_fit)
    return result


def qualify_saved_frame(linear_native_rgb, fit, reference, grid,
                        h_reference_to_frame_native, source_eligible_mask, *,
                        exact_baseline_source, saved_log_rgb, saved_valid_mask,
                        frame_chart_xy8=None, frame_chart_usable=None,
                        old_field_ref8=None, parameters):
    """Use the original native operation order, then require actual saved bits.

    The existing kernel applies frame-light/native correction first and old-B
    scalar residual last. No algebraic reordering, NaN canonicalisation,
    tolerance, JPEG input substitution or automatic adoption is allowed here.
    The caller must also reproduce the original RAW renderer/source mask;
    current geometry is not implicitly substituted for the saved geometry.
    """
    corrected, audit = photometry.correct_frame_native(
        linear_native_rgb, fit, reference, grid, h_reference_to_frame_native,
        source_eligible_mask, exact_baseline_source=exact_baseline_source,
        frame_chart_xy8=frame_chart_xy8, frame_chart_usable=frame_chart_usable,
        old_field_ref8=old_field_ref8, parameters=parameters,
    )
    valid = np.all(np.isfinite(corrected), axis=0)
    require_exact_array(corrected, saved_log_rgb)
    require_exact_array(valid, saved_valid_mask)
    return corrected, dict(audit, exact_saved_output_parity=True,
                           full_raw_installation_rebuild_proven=False,
                           adoption_allowed=False)
