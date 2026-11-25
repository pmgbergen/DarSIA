"""End-to-end run of the FluidFlower workflows on a small synthetic experiment.

Setup (depth, labels, facies, rig, image porosity) -> colour calibration (automatic colour
paths, default mass calibration) -> analysis (mass, thresholding), through the same entry
points as the CLI and the GUI, on 64 x 128 pixel images (`synthetic.py`). The results are
compared with the painted ground truth.

Run alone with `uv run pytest -m workflow`. With the environment variable
`DARSIA_SYNTHETIC_KEEP=<folder>` the experiment and its results are written there (and
kept) instead of a temporary folder, e.g. to open `<folder>/config.toml` in the GUI.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.analysis.analysis_context import (
    prepare_analysis_context,
)
from darsia.presets.workflows.calibration.calibration_color_paths import (
    calibration_color_paths_from_context,
)
from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
    calibration_color_to_mass_analysis_from_context,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_depth import setup_depth_map
from darsia.presets.workflows.setup.setup_facies import setup_facies
from darsia.presets.workflows.setup.setup_image_porosity import setup_image_porosity
from darsia.presets.workflows.setup.setup_labeling import segment_colored_image
from darsia.presets.workflows.setup.setup_rig import setup_rig
from darsia.presets.workflows.user_interface_analysis import run_analysis

from .synthetic import SyntheticExperiment, make_synthetic_experiment

pytestmark = pytest.mark.workflow

KEEP = "DARSIA_SYNTHETIC_KEEP"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else np.count_nonzero(a & b) / union


@pytest.fixture(scope="module", params=["crop", "no_crop"])
def experiment(request, tmp_path_factory) -> SyntheticExperiment:
    """The synthetic experiment (files and ground truth), with a curvature crop of the
    full image, or without (the physical size then comes from `[rig]` alone)."""
    keep = os.environ.get(KEEP)
    folder = (
        Path(keep) / request.param
        if keep
        else tmp_path_factory.mktemp(f"synthetic_{request.param}")
    )
    return make_synthetic_experiment(folder, crop=request.param == "crop")


@pytest.fixture(scope="module")
def results(experiment) -> Path:
    """Setup steps, in the order of `user_interface_setup --all`."""
    config = experiment.config
    setup_depth_map(config, key="depth")
    segment_colored_image(config)
    setup_facies(Rig, config)
    setup_rig(Rig, config)
    setup_image_porosity(Rig, config)
    return experiment.folder / "results"


@pytest.fixture(scope="module")
def calibrated(experiment, results) -> Path:
    """Colour calibration without interaction: automatic colour paths, default mass."""
    ctx = prepare_analysis_context(
        cls=Rig,
        path=experiment.config,
        all=False,
        require_color_to_mass=False,
        section="calibration",
        require_results=False,
    )
    calibration_color_paths_from_context(ctx, False)
    calibration_color_to_mass_analysis_from_context(ctx, default=True)
    return results


@pytest.fixture(scope="module")
def analysed(experiment, calibrated) -> Path:
    """Mass and thresholding analysis of all images (as `user_interface_analysis`)."""
    args = SimpleNamespace(
        config=[experiment.config],
        cropping=False,
        mass=True,
        volume=False,
        segmentation=False,
        fingers=False,
        thresholding=True,
        all=False,
        show=False,
        stream=False,
        stream_cache_dir=None,
        progress=False,
        info=False,
    )
    run_analysis(Rig, args)
    return calibrated


def test_labels_and_facies(experiment, results):
    """Each painted layer is one label; both sand layers form facies 1, the seal 2."""
    labels = darsia.imread(results / "setup" / "rig" / "labels.npz").img
    facies = darsia.imread(results / "setup" / "rig" / "facies.npz").img
    assert labels.shape == experiment.layers.shape
    assert sorted(np.unique(labels).tolist()) == [1, 2, 3]
    for label in (1, 2, 3):
        mask = labels == label
        layer = np.bincount(experiment.layers[mask]).argmax()
        assert _iou(mask, experiment.layers == layer) > 0.95
    for layer, expected in ((0, 1), (1, 2), (2, 1)):
        assert np.unique(facies[experiment.layers == layer]).tolist() == [expected]


def test_rig_round_trip(experiment, results):
    """The stored rig loads with all fields on the grid of the baseline."""
    rig = Rig.load(results / "setup" / "rig")
    shape = rig.baseline.img.shape[:2]
    assert shape == experiment.layers.shape
    # The physical size of `[rig]`, with or without a crop correction
    assert rig.baseline.dimensions == pytest.approx(list(experiment.size))
    for name in ("labels", "facies", "global"):
        regions = rig.regions(name)
        assert regions.img.shape == shape
    assert np.all(rig.regions("global").img == 0)
    assert rig.depth.img.shape == shape


def test_image_porosity(experiment, results):
    """Grains are not porous; most of the clean sand is.

    By construction, colours between the noise (90) and the key percentile (99.5) of the
    local distances score below 1, i.e. about 10 % of the typical pixels, plus the
    boundary band; hence 75 % for the clean sand.
    """
    porosity = darsia.imread(
        results / "setup" / "image_porosity" / "color_statistics" / "image_porosity.npz"
    ).img
    assert np.mean(porosity[experiment.grains] <= 0.9) > 0.95
    assert np.mean(porosity[~experiment.grains] > 0.9) > 0.75


def test_colour_calibration(calibrated):
    """Automatic colour paths and the default mass calibration are stored."""
    folder = calibrated / "color" / "color_path" / "path"
    assert (folder / "color_paths.csv").exists()
    assert (folder / "interpolation" / "mass" / "metadata.json").exists()


def test_thresholding_finds_the_gas_cap(experiment, analysed):
    """The white colour range finds the painted gas cap in every image."""
    for k, gas in enumerate(experiment.gas):
        detected = darsia.imread(
            analysed / "thresholding" / "npz" / "gas" / f"image_{k:03d}.npz"
        ).img.astype(bool)
        assert _iou(detected, gas) > 0.8, f"image {k}"


def test_gas_extent_in_metres(experiment, analysed):
    """The horizontal extent of the gas cap is reported in metres of the rig."""
    extent = pd.read_csv(analysed / "sparse_data" / "thresholding_extent.csv")
    pixel_width = experiment.size[1] / experiment.layers.shape[1]
    for k, gas in enumerate(experiment.gas):
        columns = np.flatnonzero(gas.any(axis=0))
        painted = 0.0 if len(columns) == 0 else len(columns) * pixel_width
        assert extent["gas_extent_x"][k] == pytest.approx(
            painted, abs=2 * pixel_width
        ), f"image {k}"


def test_mass_grows_with_the_injection(experiment, analysed):
    """No mass in the baseline; the detected mass grows with the injected one.

    The default mass calibration is not fitted, so the amount is only checked to the
    order of magnitude (catches unit errors), not quantitatively.
    """
    mass = pd.read_csv(analysed / "sparse_data" / "integrated_mass.csv")
    assert len(mass) == len(experiment.images)
    detected = mass["detected_mass_total"].to_numpy()
    exact = mass["exact_mass_total"].to_numpy()
    assert detected[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(detected) > 0)
    ratio = detected[1:] / exact[1:]
    assert np.all((ratio > 0.2) & (ratio < 5.0)), ratio
