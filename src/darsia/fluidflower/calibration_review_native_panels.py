"""Display original, verified native saved products without publishing files."""
from __future__ import annotations

import math

from PySide6.QtGui import QImage

from .storage_saved_rendering import release_stored_figure


def _figure_image(figure):
    try:
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        return QImage(figure.canvas.buffer_rgba(), width, height, 4 * width,
                      QImage.Format.Format_RGBA8888).copy()
    finally:
        release_stored_figure(figure)


def mass_curve_figure(reader, *, cancelled):
    """Plot saved scalars after each original frame reader verifies their closure.

    The native reader has no scalar-only admission API. Load and release one
    frame at a time; never retain a run of maps or reconstruct a mass scalar.
    This deliberately runs in the panel worker, only on an explicit graph view.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rows = []
    for index in range(len(reader.case_ids)):
        if cancelled():
            raise InterruptedError("Saved graph read cancelled")
        stored = reader.read_frame(index)
        source = stored["native_scalar_reference"]
        rows.append({key: source.get(key) for key in (
            "elapsed_h", "total_mass_g", "aqueous_mass_g", "gas_mass_g", "injected_mass_g",
            "quantified_total_mass_g", "quantified_aqueous_mass_g", "quantified_gas_mass_g",
            "analysis_scope")})
        del stored
    times = [row["elapsed_h"] for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in times):
        raise ValueError("The saved graph has no finite acquisition-time axis.")
    figure = Figure(figsize=(7, 3.8), layout="constrained")
    try:
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        scoped = any(row["analysis_scope"] is not None for row in rows)
        for key, label, colour in (("total_mass_g", "Total", "#175aad"),
                ("aqueous_mass_g", "Dissolved", "#169879"), ("gas_mass_g", "Free gas", "#dc398b"),
                ("injected_mass_g", "Injected", "black")):
            values = [row[key] if row[key] is not None else math.nan for row in rows]
            if any(math.isfinite(value) for value in values):
                axis.plot(times, values, label=label, color=colour, marker="o", markersize=3,
                          linestyle="--" if key == "injected_mass_g" else "-")
            quantified = "quantified_" + key
            if scoped and key != "injected_mass_g":
                values = [row.get(quantified) if row.get(quantified) is not None else math.nan for row in rows]
                if any(math.isfinite(value) for value in values):
                    axis.plot(times, values, label=label + " · quantified subtotal", color=colour,
                              marker="o", markersize=3, linestyle=":")
        axis.set(xlabel="Acquisition time (h)", ylabel="Saved CO₂ mass (g)",
                 title=reader.expected_run + (" · unknown mass remains" if scoped else " · saved mass"))
        axis.legend()
        axis.grid(alpha=.2)
        return figure
    except BaseException:
        release_stored_figure(figure)
        raise


def read_native_panel(reader, stored, content, *, limit_g_per_cell=None, cancelled=lambda: False):
    if content in {"mass_total", "mass_aqueous", "mass_gas"}:
        from .calibration_review_mass_map import mass_figure
        return _figure_image(mass_figure(stored, content.removeprefix("mass_"), limit_g_per_cell))
    if content == "photograph":
        rgba, _evidence = reader.read_photo(stored)
        if rgba.dtype.name != "uint8" or rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
            raise ValueError("The saved original photograph has an unsupported pixel format.")
        height, width, channels = rgba.shape
        image_format = QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
        return QImage(rgba.data, width, height, rgba.strides[0], image_format).copy()
    if content == "mass_curve":
        return _figure_image(mass_curve_figure(reader, cancelled=cancelled))
    raise ValueError("This native attachment does not bind that display. Choose a saved mass map, photograph or mass graph.")
