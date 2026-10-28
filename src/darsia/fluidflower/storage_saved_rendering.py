"""Display verified saved coarse products; never decode sources or run a model.

This separate branch deliberately has no photograph input yet. The original
registered/pre-light photograph is truthfully missing until an actual saved
product contract is qualified. These previews are not final original-left films.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .storage_saved_products import read_stored_frame

FIELDS = {
    "concentration": ("concentration_aqueous", "concentration_valid_area_fraction", "Dissolved CO2 concentration", "g/L"),
    "saturation": ("saturation_free_gas", "saturation_valid_area_fraction", "Free-gas saturation", "fraction"),
    "before_rules_coarse": ("gas_probability_before_rules", "gas_probability_before_rules_valid_pixel_fraction", "BEFORE rules: saved valid-pixel mean", "fraction"),
    "after_rules_coarse": ("gas_probability_after_rules", None, "AFTER rules: final gas pixel fraction", "fraction"),
    "before_rules_native_pixels": (None, None, "BEFORE rules: saved native-pixel score", "fraction"),
}


def _limit(field, concentration_limit_g_l):
    if field not in FIELDS:
        raise ValueError("Choose an explicit saved-field view")
    if field == "concentration":
        value = concentration_limit_g_l
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Concentration needs its explicit fixed positive g/L limit")
        return float(value)
    if concentration_limit_g_l is not None:
        raise ValueError("A concentration scale does not apply to this explicit field")
    return 1.0


def _figure(stored, field, limit):
    """One display path for both writing and axis/pixel verification."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(13, 4.4), layout="constrained")
    FigureCanvasAgg(figure)
    photo, axis = figure.subplots(1, 2)
    photo.set_title("Original photograph: unavailable")
    photo.text(.5, .5, "No bound saved registered/pre-light photograph.\n"
               "No source image was decoded.", ha="center", va="center",
               transform=photo.transAxes, wrap=True)
    photo.set_axis_off()
    array_name, support_name, title, unit = FIELDS[field]
    pending = field.startswith("before_rules_") and stored["before_rules"]["state"] == "pending"
    image = None
    if pending:
        axis.set_title(title + ": unavailable")
        axis.text(.5, .5, stored["before_rules"]["reason"], ha="center", va="center",
                  transform=axis.transAxes, wrap=True)
        axis.set_axis_off()
    else:
        value = (stored["before_native"] if field == "before_rules_native_pixels"
                 else stored["arrays"][array_name])
        if value is None:
            raise ValueError("The explicit saved BEFORE PNG was not loaded")
        mask = ~np.isfinite(value)
        if support_name is not None:
            support = stored["arrays"][support_name]
            mask = mask | ~np.isfinite(support) | (support <= 0)
        displayed = np.ma.array(value, mask=mask, copy=False)
        axis.set_facecolor("#e6edf2")
        options = dict(vmin=0, vmax=limit, cmap="viridis")
        if field == "before_rules_native_pixels":
            image = axis.imshow(displayed, origin="upper", interpolation="nearest", **options)
            axis.set(xlabel="Native column (px)", ylabel="Native row (px)")
        else:
            x0, x1, y0, y1 = stored["grid"]["extent_m"]
            step = stored["grid"]["resolution_mm"] / 1000.
            rows, columns = displayed.shape
            x_edges = x0 + np.arange(columns + 1, dtype=np.float64) * step
            y_edges = y0 + np.arange(rows + 1, dtype=np.float64) * step
            # Fixed-width saved cells have only a possibly shorter final edge.
            # Uniform image stretching would move every internal cell boundary.
            x_edges[-1], y_edges[-1] = x1, y1
            image = axis.pcolormesh(x_edges, y_edges, displayed, shading="flat",
                                   edgecolors="none", antialiased=False, **options)
            axis.set(xlabel="x (m)", ylabel="y (m)", xlim=(x0, x1), ylim=(y0, y1),
                     aspect="equal")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=.035, label=unit + " - fixed linear scale")
    figure.suptitle(stored["run"] + " / " + stored["case_id"] + " - verified saved storage output")
    return figure, image


def _asset(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def render_stored_frame(destination, native_frame_metadata, *, resolve_asset, field,
                        concentration_limit_g_l=None, expected=None):
    """Write one new preview/receipt from the saved reader, never a final film.

    ``destination`` is a fresh PNG. The sidecar is ``<name>.receipt.json``.
    BEFORE native pixels and coarse metric cells are explicitly different views.
    No photograph argument, recipe, normalization, resampling or fallback exists.
    """
    limit = _limit(field, concentration_limit_g_l)
    destination = Path(destination)
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    if destination.suffix.lower() != ".png":
        raise ValueError("Choose a new PNG preview path")
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("Saved previews and receipts are immutable")
    stored = read_stored_frame(native_frame_metadata, resolve_asset=resolve_asset,
        load_before=field == "before_rules_native_pixels", expected=expected)
    figure, image = _figure(stored, field, limit)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids the replacement semantics of the legacy writer.
        with destination.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=120)
        native_pixels = field == "before_rules_native_pixels"
        pending = field.startswith("before_rules_") and stored["before_rules"]["state"] == "pending"
        document = dict(format="darsia-saved-coarse-render-receipt-v1", complete=True,
            run=stored["run"], case_id=stored["case_id"], precision=stored["precision"], field=field,
            field_state="pending" if pending else "complete", fixed_limits=[0., limit],
            axes=(dict(frame="native pixels", origin="upper", metric_extent=None,
                       mapping_to_coarse_not_inferred=True) if native_pixels else
                  dict(frame="coarse metric cells", origin="lower", extent_m=stored["grid"]["extent_m"],
                       grid=stored["grid"], display="flat cell mesh",
                       edges="fixed resolution; last edge clipped to literal extent")),
            displayed_shape_hw=(None if image is None else list(image.get_array().shape)),
            array_transposed=False, array_regridded=False, per_frame_normalization=False,
            photograph=dict(state="missing", reason="saved_registered_pre_light_product_not_bound",
                            source_image_decoded=False, corrected_fallback_used=False),
            final_original_left_film_ready=False, before_rules=stored["before_rules"],
            storage_receipt=stored["storage_receipt"], loaded_assets=stored["loaded_assets"],
            source=stored["source"], stored_mass_totals_g=stored["mass_totals_g"], preview=_asset(destination))
        if stored.get("analysis_scope") is not None:
            document.update(analysis_scope=stored["analysis_scope"],
                quantified_stored_mass_totals_g=stored["quantified_mass_totals_g"], unknown_mass_g=None)
        with receipt_path.open("xb") as stream:
            stream.write((json.dumps(document, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, indent=2) + "\n").encode("utf8"))
        return dict(preview=document["preview"], receipt=_asset(receipt_path),
                    field_state=document["field_state"], final_original_left_film_ready=False)
    finally:
        release_stored_figure(figure)


def release_stored_figure(figure):
    """Release Agg buffers without depending on cyclic garbage collection."""
    canvas = figure.canvas
    if canvas is None:
        return
    try:
        figure.clear()
    finally:
        if hasattr(canvas, "renderer"):
            canvas.renderer = None
        figure.set_canvas(None)
        canvas.figure = None
