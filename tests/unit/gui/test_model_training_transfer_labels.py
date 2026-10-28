"""Experiment rows in Phase model transfer name the Setup step Image correction."""

from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage


def test_unprepared_transfer_row_names_image_correction():
    label = SharedPhaseModelTrainingPage._transfer_row_label(
        {"run": "run1", "review_status": "not_reviewed", "preparation_status": "missing"})
    assert label == "RUN1 · Needs image correction · not manually reviewed"
    ready = SharedPhaseModelTrainingPage._transfer_row_label(
        {"run": "run1", "review_status": "reviewed"})
    assert ready == "RUN1 · manually reviewed"
