import json
from pathlib import Path

import pytest

from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
    _load_baseline_color_spectrum_for_color_to_mass,
)


def _write_spectrum_file(folder: Path, label: int):
    folder.mkdir(parents=True, exist_ok=True)
    content = {
        "base_color": [0.0, 0.0, 0.0],
        "spectrum": [[[False]]],
        "histogram": [[[0.0]]],
        "color_range": {
            "min_color": [0.0, 0.0, 0.0],
            "max_color": [1.0, 1.0, 1.0],
            "color_mode": "RELATIVE",
        },
    }
    (folder / f"color_spectrum_{label}.json").write_text(json.dumps(content))


def test_load_baseline_color_spectrum_none_mode_returns_none(tmp_path):
    result = _load_baseline_color_spectrum_for_color_to_mass(
        ignore_mode="none",
        baseline_color_spectrum_folder=tmp_path / "missing",
        required_labels={0},
    )
    assert result is None


def test_load_baseline_color_spectrum_mode_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="requires them"):
        _load_baseline_color_spectrum_for_color_to_mass(
            ignore_mode="baseline",
            baseline_color_spectrum_folder=tmp_path / "missing",
            required_labels={0},
        )


def test_load_baseline_color_spectrum_mode_requires_all_labels(tmp_path):
    folder = tmp_path / "spectra"
    _write_spectrum_file(folder, 0)

    with pytest.raises(FileNotFoundError, match="Missing labels: \\[1\\]"):
        _load_baseline_color_spectrum_for_color_to_mass(
            ignore_mode="expanded",
            baseline_color_spectrum_folder=folder,
            required_labels={0, 1},
        )


def test_load_baseline_color_spectrum_mode_loads_valid_folder(tmp_path):
    folder = tmp_path / "spectra"
    _write_spectrum_file(folder, 0)
    _write_spectrum_file(folder, 1)

    result = _load_baseline_color_spectrum_for_color_to_mass(
        ignore_mode="baseline",
        baseline_color_spectrum_folder=folder,
        required_labels={0, 1},
    )

    assert set(result.keys()) == {0, 1}
