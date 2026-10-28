"""Read and display recorded local overflow contributions; never recompute them."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

from .storage_saved_rendering import release_stored_figure


GAS_VIEWS = (("All free gas", "total"), ("Original model interpretation", "model"),
             ("Overflow change (+ / −)", "overflow"), ("Model / overflow origin", "origin"))
PALETTES = {"blue_orange": ("#2587c8", "#f2993f", "#b63e8c"),
            "green_violet": ("#299568", "#8955c4", "#d34a4a")}


def gas_summary_text(data):
    summary = data.get("summary")
    if not summary:
        return "Gas contribution unavailable: " + str(data.get("reason") or "before/after values were not saved")
    share = summary.get("overflow_change_share_of_final")
    percent = "undefined (zero final mass)" if share is None else f"{100 * share:+.2f}% of final gas"
    label = "ESTIMATED overflow" if summary.get("estimated") is True else summary.get("reference_label", "Reference not recorded")
    components = ""
    if all(key in summary for key in ("model_retained_g", "overflow_added_g", "overflow_removed_g")):
        added_share = summary.get("added_share_of_final")
        added_percent = "" if added_share is None else f" ({100 * added_share:.2f}% of final gas)"
        components = (f"Retained model: {summary['model_retained_g']:.5g} g · "
                      f"Added by overflow: {summary['overflow_added_g']:.5g} g{added_percent} · "
                      f"Removed by overflow: {summary['overflow_removed_g']:.5g} g. ")
    return (f"{label} · Model before: {summary['model_gas_mass_g']:.5g} g · "
            f"After overflow: {summary['final_gas_mass_g']:.5g} g · "
            f"Net change: {summary['overflow_change_g']:+.5g} g ({percent}). " + components +
            "Quantified support only; excluded/unknown contributions stay unchanged. "
            "Values describe this image; masses from different times are not added.")


def gas_figure(data, view="total", palette="blue_orange"):
    """Use native cell edges and one saved scale for the whole preparation."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.patches import Patch
    if data.get("maps") is None:
        raise ValueError(str(data.get("reason") or "Spatial gas provenance was not saved"))
    if view not in dict((key, label) for label, key in GAS_VIEWS) or palette not in PALETTES:
        raise ValueError("Choose a recorded gas display and palette")
    arrays, grid = data["maps"], data["grid"]
    scale = data.get("display_scale_g_per_cell", {}).get("signed" if view == "overflow" else "positive")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
        raise ValueError("The saved gas contribution has no fixed display scale")
    # An all-zero preparation still has a constant, explicitly labelled scale.
    limit = float(scale) if scale > 0 else 1.
    height, width = grid["native_shape_hw"]
    x0, x1, y0, y1 = grid["extent_m"]
    x = x0 + np.asarray(grid["column_edges_px"]) / width * (x1-x0)
    y = y1 - np.asarray(grid["row_edges_px"]) / height * (y1-y0)
    support = arrays["cell_has_support"]
    model, added, removed = PALETTES[palette]
    figure = Figure(figsize=(7, 4.4), layout=None if view == "origin" else "constrained")
    try:
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        if view == "origin":
            figure.subplots_adjust(left=.1, right=.98, top=.9, bottom=.25)
        axis.set_facecolor("#e6edf2")
        if view == "origin":
            retained, extra = arrays["cell_model_retained_g"], arrays["cell_overflow_added_g"]
            final = retained + extra
            fraction = np.divide(extra, final, out=np.zeros_like(extra), where=final > 0)
            rgb = (1-fraction[..., None])*np.array(to_rgb(model)) + fraction[..., None]*np.array(to_rgb(added))
            rgba = np.concatenate((rgb, np.where(support, np.clip(final/limit, 0, 1), 0)[..., None]), axis=-1)
            axis.pcolormesh(x, y, rgba, shading="flat", antialiased=False)
            cut = np.where(support, arrays["cell_overflow_removed_g"], 0.)
            overlay = np.zeros((*cut.shape, 4))
            overlay[..., :3] = to_rgb(removed)
            overlay[..., 3] = np.clip(cut/limit, 0, 1)
            axis.pcolormesh(x, y, overlay, shading="flat", antialiased=False, edgecolors="none")
            rows, columns = np.nonzero(cut > 0)
            if len(rows):
                from matplotlib.path import Path as PlotPath
                from matplotlib.patches import PathPatch
                polygons = np.empty((len(rows), 4, 2))
                polygons[:, :, 0] = np.column_stack((x[columns], x[columns+1], x[columns+1], x[columns]))
                polygons[:, :, 1] = np.column_stack((y[rows], y[rows], y[rows+1], y[rows+1]))
                axis.add_patch(PathPatch(PlotPath.make_compound_path_from_polys(polygons),
                    facecolor="none", edgecolor=removed, linewidth=0, hatch="///"))
            axis.legend(handles=[Patch(color=model, label="Model mass retained"),
                Patch(color=added, label="Overflow mass added"),
                Patch(facecolor=removed, hatch="///", label="Model mass removed")], loc="upper left", fontsize=7)
            figure.text(.5, .02, f"Fixed opacity scale: 0–{limit:g} g per review cell.\n"
                "Mixed colours show both retained and added mass in a review cell;\n"
                "hatching marks removal. These are contributions, not mutually exclusive pixel classes.",
                ha="center", va="bottom", fontsize=8)
        else:
            key = {"total": "cell_final_gas_g", "model": "cell_model_before_g", "overflow": "cell_overflow_change_g"}[view]
            cmap = LinearSegmentedColormap.from_list("gas_review", [removed, "#f7f7f7", added]
                if view == "overflow" else ["#f7f7f7", model])
            image = axis.pcolormesh(x, y, np.ma.array(arrays[key], mask=~support, copy=True),
                shading="flat", cmap=cmap, vmin=-limit if view == "overflow" else 0, vmax=limit,
                edgecolors="none", antialiased=False)
            figure.colorbar(image, ax=axis, label="g per review cell · fixed across images")
        axis.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="x (m)", ylabel="y (m)", aspect="equal",
                 title=dict((key, label) for label, key in GAS_VIEWS)[view])
        return figure
    except BaseException:
        release_stored_figure(figure)
        raise


class LocalGasReview(dict):
    """One run of the exact frozen local calibration's saved preparation."""

    def __init__(self, frozen, document, preparation, run, *, dataset, reader):
        rows = [deepcopy(row) for row in preparation.get("frames", []) if row.get("run") == run]
        if not rows or len({row["case_id"] for row in rows}) != len(rows):
            raise ValueError("The saved gas review has no unique image list for this run")
        def metadata(row):
            summary = (row.get("gas_contribution") or {}).get("summary") or {}
            return dict(case_id=row["case_id"], frame_index=row.get("frame_index", summary.get("frame_index")),
                        elapsed_h=row.get("elapsed_h", summary.get("elapsed_h")))
        frames = [metadata(row) for row in rows]
        if all(type(row["frame_index"]) is int for row in frames):
            frames.sort(key=lambda row: row["frame_index"])
        super().__init__(complete=True, frames=frames, method={},
            selected_inputs=dict(candidate=dict(dataset=dataset, run=run)), series_calibration=frozen["path"],
            bindings=dict(calibration_sha256=frozen["sha256"]))
        self.frozen, self.document, self.preparation = frozen, document, preparation
        self.run, self.reader, self._phase_core = run, reader, None

    def gas(self, index):
        from .mass.gas_contribution_review import read_gas_contributions
        return read_gas_contributions(self.preparation, self.run, self["frames"][index]["case_id"], reader=self.reader)

    def photograph(self, index):
        case = self["frames"][index]["case_id"]
        if self._phase_core is None:
            bound = self.preparation.get("phase_sources", {}).get(self.run, {}).get("phase1_completion")
            if not bound:
                raise ValueError("No saved photograph source is bound to this calibration")
            self._phase_core = self.reader.json(bound)
        core = self._phase_core
        position = core.get("ordered_case_ids", []).index(case)
        record = self.reader.json(core["frames"][position])
        if record.get("case_id") != case or record.get("run") != self.run:
            raise ValueError("The saved photograph belongs to another acquisition")
        photo = record.get("photograph") or {}
        if photo.get("state") != "complete" or not photo.get("png"):
            raise ValueError("This image has no saved photograph; RAW is not decoded for Review")
        image = QImage(str(self.reader.path(photo["png"])))
        if image.isNull():
            raise ValueError("The saved photograph cannot be displayed")
        return image

    def curve(self):
        """Plot recorded per-image amounts; no map reads or temporal summation."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from .mass.gas_contribution_review import contribution_summary
        rows = {row["case_id"]: row for row in self.preparation["frames"] if row["run"] == self.run}
        points = []
        for frame in self["frames"]:
            row = rows[frame["case_id"]]
            summary = (row.get("gas_contribution") or {}).get("summary") or contribution_summary(
                row.get("report", {}), row.get("phase_split_target"))
            if summary is None or not isinstance(frame.get("elapsed_h"), (int, float)):
                raise ValueError("The saved gas curve has no complete recorded time/amount series")
            points.append((frame["elapsed_h"], summary))
        figure = Figure(figsize=(7, 3.8), layout="constrained")
        try:
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            for key, label, colour in (("model_gas_mass_g", "Model before", "#2587c8"),
                    ("final_gas_mass_g", "After overflow", "#f2993f"),
                    ("overflow_change_g", "Signed overflow change", "#b63e8c")):
                axis.plot([point[0] for point in points], [point[1][key] for point in points],
                    label=label, color=colour, marker="o", markersize=3)
            axis.axhline(0, color="#555555", linewidth=.5)
            axis.set(xlabel="Acquisition time (h)", ylabel="Free-gas CO₂ mass (g)", title=self.run.upper())
            axis.legend()
            axis.grid(alpha=.2)
            return figure
        except BaseException:
            release_stored_figure(figure)
            raise


def local_reviews(frozen, *, dataset, reader=None):
    """Bind the explicit local result; no folder discovery and no numerical replay."""
    from .mass.n1_pipeline import DataReader, asset
    reader = reader or DataReader(unc_only=False)
    bound = deepcopy(frozen) if isinstance(frozen, dict) else asset(Path(frozen))
    document = reader.json(bound)
    if document.get("frozen") is not True or not document.get("overflow_preparation"):
        raise ValueError("This calibration has no saved overflow contribution preparation")
    preparation = reader.json(document["overflow_preparation"])
    runs = list(dict.fromkeys(row["run"] for row in preparation.get("frames", [])))
    if document.get("calibration_scope") == "individual":
        runs = [run for run in runs if run == document.get("calibrated_run")]
    return [LocalGasReview(bound, document, preparation, run, dataset=dataset, reader=reader) for run in runs]
