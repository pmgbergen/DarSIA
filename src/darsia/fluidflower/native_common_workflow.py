"""Fixed native navigation over the existing calculation routes."""

# Logical IDs remain the original owners' IDs. Availability never depends on
# translated labels, header order or the currently edited, unsaved selection.
CALIBRATION_ROWS = (
    ("Image selection", "calibration", "images", "Choose comparable calibration observations; retain the complete phase history. New phase-model training proposes the same images unless its own training set is chosen."),
    ("Calibration Method", "method", "use_for_series", "Choose the calibration method; model selection and fitting follow in their own sections."),
    ("Phase model", "phase_model", "training", "Choose, review and train a saved phase-model version."),
    ("Mass calibration", "calibration", "mass", "Use the saved calculation model to prepare and inspect mass calibration."),
    ("Saved calibrations", "calibration", "saved", "Inspect the saved calibration; opening starts no calculation."),
    ("Interactive calibration", "calibration_controls", "interactive", "Open DarSIA's original manual mass-calibration settings."),
    ("Review", "phase_model", "calibration_review", "View saved concentration, mass and curves; opening starts no calculation."),
    ("Validation", "phase_model", "checks", "View saved held-out validation scores; opening writes nothing."),
)

METHOD_ROWS = {
    ("setup", "finish"): "phase_model",
    ("calibration", "images"): "phase_model",
    ("phase_model", "training"): "phase_model",
    ("phase_model", "calibration_review"): "phase_model",
    ("phase_model", "checks"): "phase_model",
    ("calibration_controls", "interactive"): "colour_path",
    **{("calibration", item): "colour_path" for item in ("default_mass", "mass_optuna", "reset", "delete")},
    **{("analysis", item): "colour_path" for item in
       ("cropping", "mass", "segmentation", "fingers", "thresholding", "volume")},
    ("image_selection", "production"): "phase_model",
    ("mass_analysis", "frozen"): "phase_model",
    ("jobs", "production_results"): "phase_model",
    ("mass_analysis", "review"): "phase_model",
    ("events", "question"): "phase_model",
}

HIDDEN_COMPATIBILITY_ROWS = {
    ("calibration", item) for item in ("default_mass", "mass_optuna", "reset", "delete")
} | {("mass_analysis", "review"), ("analysis", "mass"), ("calibration_controls", "interactive")} | {
    # One visible Wasserstein comparison row (a job); the original in-process
    # compute and assemble steps stay routable.
    ("comparison", "wasserstein_compute"), ("comparison", "wasserstein_assemble"),
    # The Events page answers questions from finished mass analyses; the
    # original events step stays routable.
    ("comparison", "events")}


def availability(logical, selected_method):
    required = METHOD_ROWS.get(logical)
    if not required or not selected_method or required == selected_method:
        return True, ""
    if logical == ("calibration", "images"):
        return False, ("Colour signal + SimpleFlash keeps its existing calibration data-selection settings. "
                       "This page selects observations for Titration + dynamic phase maps.")
    explanation = ("Available with Colour signal + SimpleFlash."
                   if required == "colour_path" else
                   "Available with Titration + dynamic phase guidance.")
    return False, explanation + " Change the method under Calibration Method."
