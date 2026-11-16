from pathlib import Path

import pytest

from darsia.presets.workflows.results_folder import (
    FIELD_MAPS_CSV,
    INTEGRATED_MASS,
    INTEGRATED_VOLUME,
    THRESHOLDING_EXTENT,
    analysis_folder,
    field_map_folder,
    sparse_data_csv,
    suggested_analysis_results_folder,
    suggested_workflow_results_folder,
)


def test_standard_layout_of_the_analysis_results(tmp_path: Path) -> None:
    """The layout functions give the folders the analyses have always written to."""
    results = tmp_path / "results"
    for name in ("cropping", "segmentation", "thresholding", "fingers"):
        assert analysis_folder(results, name) == results / name
    with pytest.raises(ValueError):
        analysis_folder(results, "mass")
    for mode in ("mass", "rescaled_saturation_g", "concentration_aq"):
        assert field_map_folder(results, mode) == results / mode
    assert sparse_data_csv(results, INTEGRATED_MASS) == (
        results / "sparse_data" / "integrated_mass.csv"
    )
    assert sparse_data_csv(results, INTEGRATED_VOLUME) == (
        results / "sparse_data" / "integrated_volume.csv"
    )
    assert sparse_data_csv(results, THRESHOLDING_EXTENT) == (
        results / "sparse_data" / "thresholding_extent.csv"
    )
    assert FIELD_MAPS_CSV == Path("sparse_data") / "integrated_mass.csv"


def test_suggested_analysis_results_folder_for_cropping(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder(config, ["cropping"])
    assert folder == results / "cropping"


def test_suggested_analysis_results_folder_ignores_old_folder_entries(
    tmp_path: Path,
) -> None:
    """The analyses write to standard folders; an old `folder` entry has no effect."""
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    seg_folder = tmp_path / "seg"
    config.write_text(
        f"[data]\nresults = '{results}'\n\n"
        f"[analysis.segmentation]\nfolder = '{seg_folder}'\n"
    )

    folder = suggested_analysis_results_folder(config, ["segmentation"])
    assert folder == results / "segmentation"


def test_suggested_analysis_results_folder_for_volume(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    assert suggested_analysis_results_folder(config, ["volume"]) == (
        results / "sparse_data"
    )
    (results / "concentration_aq").mkdir(parents=True)
    (results / "concentration_aq" / "image.npz").write_bytes(b"")
    assert suggested_analysis_results_folder(config, ["volume"]) == (
        results / "concentration_aq"
    )


def test_suggested_analysis_results_folder_defaults_to_results_on_multiple_modes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder(config, ["mass", "volume"])
    assert folder == results


def test_suggested_analysis_results_folder_fallback_for_missing_mode_folder(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder(config, ["fingers"])
    assert folder == results / "fingers"


def test_suggested_analysis_results_folder_thresholding_fallback(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_analysis_results_folder(config, ["thresholding"])
    assert folder == results / "thresholding"


def test_suggested_workflow_results_folder_setup(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder("setup", config, ["depth"])
    assert folder == results / "setup" / "depth"


def test_suggested_workflow_results_folder_calibration_mass(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(
        f"[data]\nresults = '{results}'\n\n"
        "[[color_path]]\nname = 'color_path'\n\n"
        "[calibration.mass]\nembedding = 'color_path'\n"
    )

    folder = suggested_workflow_results_folder(
        "calibration", config, ["default mass", "show"]
    )
    assert folder == (
        results / "color" / "color_path" / "color_path" / "interpolation" / "mass"
    )


def test_suggested_workflow_results_folder_calibration_mass_no_embedding(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder("calibration", config, ["mass"])
    assert folder is None


def test_suggested_workflow_results_folder_calibration_color_embedding(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(
        f"[data]\nresults = '{results}'\n\n"
        "[[color_path]]\nname = 'color_path'\n\n"
        "[calibration.color]\nembedding = 'color_path'\n"
    )

    folder = suggested_workflow_results_folder(
        "calibration", config, ["color embedding"]
    )
    assert folder == results / "color" / "color_path" / "color_path"


def test_suggested_workflow_results_folder_comparison_events_default(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder("comparison", config, ["events"])
    assert folder == results / "events"


def test_suggested_workflow_results_folder_comparison_wasserstein_override(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    wasserstein = tmp_path / "custom-w1"
    config.write_text(
        f"[data]\nresults = '{results}'\n\n"
        f"[wasserstein]\nresults = '{wasserstein}'\n"
    )

    folder = suggested_workflow_results_folder(
        "comparison", config, ["wasserstein compute"]
    )
    assert folder == wasserstein


def test_suggested_workflow_results_folder_utils_combined_defaults_to_results(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    results = tmp_path / "results"
    config.write_text(f"[data]\nresults = '{results}'\n")

    folder = suggested_workflow_results_folder(
        "utils", config, ["download", "media", "export calibration"]
    )
    assert folder == results
