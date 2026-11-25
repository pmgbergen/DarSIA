"""Synthetic FluidFlower experiment for end-to-end workflow tests.

A small rig (default 64 x 128 pixels, 2 m x 1 m) with three layers with curved
interfaces: a blue sand on top, a dark blue seal, and the same blue sand below (3 labels,
2 facies). CO2 is injected below the seal; the injection images show a growing white gas
cap under the seal and a yellow aqueous halo around it, mixed linearly with the sand
colour. Static dark grains, as in real sand, are the same in every image; a little
Gaussian noise differs per image.

Everything is written by `make_synthetic_experiment` (no binary files in the repository):
images (PNG, lossless), the coloured segmentation, the protocols, the depth measurements,
the facies properties and the config TOML. The ground truth (layers, grains, gas and
aqueous masks per image) is returned for the assertions of the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage

SAND = np.array([0.25, 0.45, 0.75])
"""Colour of the sand (both sand layers, facies 1)."""
SEAL = np.array([0.10, 0.18, 0.42])
"""Colour of the seal (facies 2)."""
GRAIN = np.array([0.05, 0.05, 0.08])
"""Colour of the dark grains."""
GAS = np.array([0.95, 0.95, 0.95])
"""Colour of CO2(g) (white)."""
AQUEOUS = np.array([0.90, 0.80, 0.15])
"""Colour of CO2(aq) (yellow)."""

GAS_SATURATION = 0.8
"""Fraction of the gas colour in the gas cap (linear mixing with the sand)."""
AQUEOUS_CONCENTRATION = 0.8
"""Fraction of the aqueous colour in the halo."""

START = datetime(2026, 1, 1, 10, 0, 0)
"""Date of the baseline and start of the injection."""
STEP = timedelta(minutes=10)
"""Time between images."""

# Pure colours of the coloured segmentation (see `darsia.label_image`).
_SEGMENTATION_COLOURS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


@dataclass
class SyntheticExperiment:
    """Files and ground truth of a synthetic experiment."""

    folder: Path
    """Root folder (images, metadata, results, config)."""
    config: Path
    """Config TOML."""
    images: list[Path]
    """Baseline first, then the injection images (time order)."""
    times: list[float]
    """Hours since the start of the injection, per image."""
    layers: np.ndarray
    """Painted layers (H, W): 0 top sand, 1 seal, 2 bottom sand."""
    grains: np.ndarray
    """Dark grains (H, W), bool; the same in every image."""
    gas: list[np.ndarray] = field(default_factory=list)
    """Gas cap per image (H, W), bool; empty for the baseline."""
    aqueous: list[np.ndarray] = field(default_factory=list)
    """Aqueous halo per image (H, W), bool, without the gas cap."""
    size: tuple[float, float] = (1.0, 1.0)
    """Physical size (height, width) of the rig in metres."""


def _interfaces(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the top and bottom interface of the seal per column."""
    height, width = shape
    x = np.arange(width) / width
    top = height * (0.30 + 0.06 * np.sin(2 * np.pi * x) + 0.05 * x)
    bottom = top + height * (0.16 + 0.03 * np.cos(3 * np.pi * x))
    return top, bottom


def _plume(
    shape: tuple[int, int], bottom: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gas cap under the seal and aqueous halo around it after `step` time steps."""
    height, width = shape
    rows, cols = np.mgrid[:height, :width]
    centre = width / 2
    half_width = width * 0.08 * step
    depth = height * 0.10
    below_seal = rows >= bottom[None, :]
    gas = (
        below_seal
        & (rows < bottom[None, :] + depth)
        & (np.abs(cols - centre) < half_width)
    )
    # Halo: within a growing distance of the gas cap, and a rising column above the
    # injection point at the bottom centre.
    distance = scipy.ndimage.distance_transform_edt(~gas)
    column = (np.abs(cols - centre) < width * 0.03) & (rows > bottom[None, :])
    aqueous = below_seal & ((distance < 1.5 + 1.5 * step) | column) & ~gas
    return gas, aqueous


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """8-bit RGB PNG."""
    data = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(data, cv2.COLOR_RGB2BGR))


def _toml_path(path: Path) -> str:
    return path.as_posix()


def make_synthetic_experiment(
    folder: Path,
    shape: tuple[int, int] = (64, 128),
    num_images: int = 3,
    seed: int = 0,
    noise: float = 0.01,
    grain_fraction: float = 0.02,
    crop: bool = True,
) -> SyntheticExperiment:
    """Write a synthetic experiment to `folder`.

    Parameters
    ----------
    folder : Path
        Root folder; created if missing.
    shape : tuple[int, int]
        Image size (rows, cols).
    num_images : int
        Injection images after the baseline.
    seed : int
        Seed of grains and noise.
    noise : float
        Standard deviation of the Gaussian noise (colours in [0, 1]).
    grain_fraction : float
        Fraction of pixels showing a dark grain.
    crop : bool
        Whether the config has a curvature crop of the full image (which sets the
        physical size); without it, the size comes from `[rig]` alone.

    Returns
    -------
    SyntheticExperiment
        Paths and ground truth.

    """
    folder = Path(folder)
    images_folder = folder / "images"
    meta = folder / "meta"
    results = folder / "results"
    for f in (images_folder, meta, results):
        f.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    height, width = shape

    # Layers and the clean background.
    top, bottom = _interfaces(shape)
    rows = np.arange(height)[:, None]
    layers = np.where(rows < top[None, :], 0, np.where(rows < bottom[None, :], 1, 2))
    background = np.where((layers == 1)[..., None], SEAL, SAND)
    grains = rng.random(shape) < grain_fraction
    background[grains] = GRAIN

    # Coloured segmentation (pure colours per layer).
    segmentation = np.zeros((height, width, 3))
    for layer, colour in enumerate(_SEGMENTATION_COLOURS):
        segmentation[layers == layer] = colour
    _write_png(meta / "segmentation.png", segmentation)

    # Images: baseline and injection images.
    experiment = SyntheticExperiment(
        folder=folder,
        config=folder / "config.toml",
        images=[],
        times=[],
        layers=layers,
        grains=grains,
    )
    for k in range(num_images + 1):
        image = background.copy()
        if k == 0:
            gas = np.zeros(shape, dtype=bool)
            aqueous = np.zeros(shape, dtype=bool)
        else:
            gas, aqueous = _plume(shape, bottom, k)
            # Grains stay visible; the fluids fill the pore space around them.
            gas &= ~grains
            aqueous &= ~grains
            image[gas] = (1 - GAS_SATURATION) * image[gas] + GAS_SATURATION * GAS
            image[aqueous] = (1 - AQUEOUS_CONCENTRATION) * image[
                aqueous
            ] + AQUEOUS_CONCENTRATION * AQUEOUS
        image = image + rng.normal(0, noise, image.shape)
        path = images_folder / f"image_{k:03d}.png"
        _write_png(path, image)
        experiment.images.append(path)
        experiment.times.append(k * STEP.total_seconds() / 3600)
        experiment.gas.append(gas)
        experiment.aqueous.append(aqueous)

    # Protocols.
    stamp = "%Y-%m-%d %H:%M:%S"
    imaging = meta / "imaging.csv"
    imaging.write_text(
        "path,datetime\n"
        + "".join(
            f"{p.name},{(START + k * STEP).strftime(stamp)}\n"
            for k, p in enumerate(experiment.images)
        )
    )
    end = START + num_images * STEP
    injection = meta / "injection.csv"
    injection.write_text(
        "id,location_x,location_y,start,end,rate_g/min\n"
        f"1,1.0,0.1,{START.strftime(stamp)},{end.strftime(stamp)},0.1\n"
    )
    pressure_temperature = meta / "pressure_temperature.csv"
    pressure_temperature.write_text(
        "datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,"
        "temperature_gradient_celsius\n"
        f"{START.strftime(stamp)},1.013,20.0,0.0,0.0\n"
    )

    # Rig metadata: constant depth, facies properties.
    rig_width, rig_height = 2.0, 1.0
    depth = meta / "depth.csv"
    depth.write_text(
        "x,y,depth\n"
        + "".join(
            f"{x},{y},0.01\n"
            for x in (0.0, rig_width / 2, rig_width)
            for y in (0.0, rig_height / 2, rig_height)
        )
    )
    facies_props = meta / "facies.csv"
    facies_props.write_text("id,porosity,permeability\n1,0.44,1e-9\n2,0.30,1e-11\n")

    # Config.
    images = _toml_path(images_folder)
    last = f"00:{10 * num_images:02d}:00"
    corrections = (
        f"""
[corrections]
active = ["curvature"]

[corrections.curvature]
active = ["crop"]

[corrections.curvature.crop]
top_left = [0, 0]
bottom_left = [{height - 1}, 0]
bottom_right = [{height - 1}, {width - 1}]
top_right = [0, {width - 1}]
width = {rig_width}
height = {rig_height}
in_meters = true
"""
        if crop
        else ""
    )
    experiment.size = (rig_height, rig_width)
    experiment.config.write_text(
        f"""
[data]
folders = ["{images}"]
format = "png"
baseline = "{_toml_path(experiment.images[0])}"
results = "{_toml_path(results)}"

[[data_interval]]
name = "all"
start = "00:00:00"
end = "{last}"
step = "00:10:00"
tol = "00:01:00"

[[data_interval]]
name = "injection"
start = "00:10:00"
end = "{last}"
step = "00:10:00"
tol = "00:01:00"

[protocols]
injection = "{_toml_path(injection)}"
pressure_temperature = "{_toml_path(pressure_temperature)}"

[protocols.imaging]
"{images}" = "{_toml_path(imaging)}"

[rig]
dim = 2
width = {rig_width}
height = {rig_height}
{corrections}
[depth]
measurements = "{_toml_path(depth)}"

[labeling]
colored_image = "{_toml_path(meta / "segmentation.png")}"

[facies]
props = "{_toml_path(facies_props)}"

[facies.1]
labels = [1, 3]

[facies.2]
labels = [2]

[image_porosity]
method = "color_statistics"
regions = "labels"
boundary_band = 1

[image_porosity.color_statistics]
patches = [2, 4]
min_pixels = 100

[[color_path]]
name = "path"
basis = "facies"
reference_label = 1

[[color_range]]
name = "white"
color_space = "HSV"
basis = "global"
range = [[0.0, 1.0], [0.0, 0.25], [0.7, 1.0]]

[calibration.color]
embedding = "path"
data_selection = ["injection"]

[calibration.color.color_path]
calibration_mode = "auto"

[calibration.mass]
embedding = "path"
data_selection = ["injection"]

[analysis.mass]
color = "path"
data_selection = ["all"]
formats = []

[analysis.thresholding]
data_selection = ["all"]
formats = ["npz"]

[[analysis.thresholding.layers]]
name = "gas"
mode = "white"
threshold_min = 0.5
"""
    )
    return experiment
