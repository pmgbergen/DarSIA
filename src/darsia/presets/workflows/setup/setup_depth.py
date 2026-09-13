"""Standard workflow steps for depth: generate measurements, then interpolate a map."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import darsia
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.sections import (
    list_required_sections,
    required_sections,
)
from darsia.presets.workflows.setup.illustrations import save_scalar_map_illustration

logger = logging.getLogger(__name__)

_MEASUREMENTS_GRID_SHAPE = (5, 5)
"""Fixed (rows, cols) grid size for a generated constant-depth measurements file.

A single point works mathematically for RBF interpolation, but the real
hand-authored example file in this project's data covers the domain with many
points at the same depth, presumably for robustness against sparse-data edge
effects. A small regular grid matches that shape without exposing a new setting.
"""

_MEASUREMENTS_GRID_MARGIN_FRACTION = 0.0
"""Fraction of width/height kept clear at each edge of the generated grid."""


def _build_constant_depth_grid(
    width: float, height: float, depth: float
) -> pd.DataFrame:
    """Build a regular (x, y, depth) grid CSV-ready table at a uniform depth.

    Parameters
    ----------
    width, height : float
        Rig extent in meters ([rig].width/height).
    depth : float
        Uniform depth in meters written at every grid point.

    Returns
    -------
        DataFrame with columns "x", "y", "depth" (column names required by
        ``darsia.interpolate_to_image_from_csv``).
    """
    margin_x = width * _MEASUREMENTS_GRID_MARGIN_FRACTION
    margin_y = height * _MEASUREMENTS_GRID_MARGIN_FRACTION
    rows, cols = _MEASUREMENTS_GRID_SHAPE
    xs = np.linspace(margin_x, width - margin_x, cols)
    ys = np.linspace(margin_y, height - margin_y, rows)
    xx, yy = np.meshgrid(xs, ys)
    return pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "depth": np.full(xx.size, depth),
        }
    )


def preview_depth_measurements_conflict(path: Path | list[Path]) -> list[Path]:
    """Return the depth-measurements target file if it exists and would be
    overwritten by :func:`setup_depth_measurements`.

    Returns an empty list when the mode is 'Load from CSV' (nothing is ever written
    in that mode), the target simply doesn't exist yet, or [depth] isn't
    configured at all — e.g. a wizard user who never visited the Depth step.
    That last case isn't an error here: there being nothing to preview is exactly
    right when there's nothing to generate. `setup_depth_measurements` itself
    still raises its own clear error if actually run without a `[depth]` section.
    Matches the shape of
    :func:`darsia.presets.workflows.setup.setup_protocols.preview_protocol_setup_conflicts`
    so both can feed the same generic overwrite-confirmation dialog.
    """
    try:
        config = FluidFlowerConfig(path, require_data=False, require_results=False)
        config.check("depth")
        assert config.depth is not None
    except Exception:
        return []

    if config.depth.measurements_mode != "constant":
        return []
    if not config.depth.measurements.exists():
        return []
    return [config.depth.measurements]


@required_sections("depth", "rig")
def setup_depth_measurements(
    path: Path | list[Path], *, force: bool = False, show: bool = False
) -> None:
    """Generate a depth-measurements CSV from a constant value, if configured.

    No-ops when [depth].measurements_mode is 'Load from CSV' (the default) — that
    mode means the user provides the file themselves, so this step leaves it
    untouched whether or not it currently exists.

    Parameters
    ----------
    path : Path | list[Path]
        Path to configuration file(s) (needs to comply with FluidFlowerConfig).
    force : bool
        Whether to overwrite an existing measurements file.
    show : bool
        Unused; accepted for interface parity with the other setup routines.
    """
    del show
    logger.info("\033[92mSetting up depth measurements...\033[0m")

    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check(*list_required_sections(setup_depth_measurements))
    assert config.depth is not None
    assert config.rig is not None

    if config.depth.measurements_mode != "constant":
        logger.info(
            "Depth measurements mode is 'Load from CSV'; leaving %s untouched.",
            config.depth.measurements,
        )
        return

    target = config.depth.measurements
    if target.exists() and not force:
        raise FileExistsError(
            f"Depth measurements file already exists: {target}. "
            "Use --force to overwrite."
        )

    grid = _build_constant_depth_grid(
        config.rig.width, config.rig.height, config.depth.constant_depth
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(target, index=False)
    logger.info("Saved constant-depth measurements CSV to %s", target)


@required_sections("depth", "rig")
def setup_depth_map(path: Path | list[Path], key="mean", show: bool = False) -> None:
    """Set up depth map from measurements.

    NOTE: This function stores the depth map in npz format according to the
    specifications in the config file.

    Parameters
    ----------
    path : Path | list[Path]
        Path to configuration file (needs to comply with FluidFlowerConfig).
    key : str, optional
        Column identifier in the csv file to use for interpolation (default: "mean").
    show : bool
        Whether to show the resulting depth map.
    """
    logger.info("\033[92mSetting up depth map from measurements...\033[0m")

    # ! ---- READ CONFIG ---- ! #
    config = FluidFlowerConfig(path, require_data=False, require_results=False)
    config.check(*list_required_sections(setup_depth_map))

    # Mypy type checking
    for c in [
        config.depth,
        config.rig,
    ]:
        assert c is not None

    # Convert to depth map by interpolation
    proxy_image = darsia.Image(
        img=np.zeros(config.depth.target_resolution),
        width=config.rig.width,
        height=config.rig.height,
        scalar=True,
        space_dim=config.rig.dim,
    )
    depth_map = darsia.interpolate_to_image_from_csv(
        csv_file=config.depth.measurements,
        key=key,
        image=proxy_image,
        method="rbf",
    )
    depth_map_path = config.depth.depth_map.with_suffix(".npz")
    depth_map.save(depth_map_path)
    save_scalar_map_illustration(
        depth_map.img,
        config.depth.depth_map.with_suffix(".jpg"),
        title="Depth map",
        colorbar_label="Depth",
    )

    if show:
        depth_map.show(title="Depth map")
