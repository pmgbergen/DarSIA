"""Compare saved image grids without decoding their pixel arrays."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, is_zipfile

import numpy as np


def _archive_grid(path, *, spatial_metadata=None):
    """Use the original image metadata semantics with a zero-storage shape view.

    Only the NPY array header and the small metadata member are consumed here.
    The caller separately verifies the archive's existing byte descriptor.
    Unrecognised formats and image classes retain the ordinary reader route.
    """
    import darsia
    from . import preset_wasserstein_inputs as io

    selected = Path(path)
    if selected.suffix.casefold() != ".npz":
        return None
    physical = io.source(selected)
    if not is_zipfile(physical):
        return None
    with ZipFile(physical) as archive:
        if not {"array.npy", "metadata.npy"}.issubset(archive.namelist()):
            return None
        with archive.open("array.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                return None
        with archive.open("metadata.npy") as stream:
            metadata = np.lib.format.read_array(stream, allow_pickle=True).item()
    if not isinstance(metadata, dict):
        return None
    if spatial_metadata is not None and not any(
            key in metadata for key in ("dimensions", "height", "width", "depth", "origin")):
        metadata = {**spatial_metadata, **metadata}
    classes = {name: getattr(darsia, name) for name in ("Image", "ScalarImage", "OpticalImage")}
    kind = metadata.get("type", darsia.Image)
    if isinstance(kind, str):
        kind = {"darsia." + name: value for name, value in classes.items()}.get(kind)
    if kind not in classes.values():
        return None
    # The original constructors derive grid defaults from shape and metadata;
    # they do not inspect pixels. Broadcasting retains only one scalar of data.
    view = np.broadcast_to(np.zeros((), dtype=dtype), shape)
    return kind(view, **metadata)


def require_archive_grids(sources, *, native_grid=None):
    """Check the original exact-grid contract, returning False for older formats."""
    from .setup_rig import _require_exact_grid_image

    baseline = _archive_grid(sources["shape_corrected_baseline"]["path"])
    inherited = None
    if baseline is not None and (native_grid or {}).get("physical_metric_is_separate") is True:
        if (native_grid.get("shape_hw") != list(baseline.shape[:baseline.space_dim])
                or not np.array_equal(native_grid.get("legacy_dimensions_m"), baseline.dimensions)):
            raise ValueError("The approved native canvas differs from its baseline grid")
        inherited = dict(space_dim=baseline.space_dim, indexing=baseline.indexing,
                         dimensions=list(baseline.dimensions), origin=list(baseline.origin))
    images = dict(shape_corrected_baseline=baseline)
    for name in ("labels", "facies", "depth"):
        images[name] = _archive_grid(sources[name]["path"],
            spatial_metadata=inherited if name in ("labels", "facies") else None)
    if any(image is None for image in images.values()):
        return False
    for name in ("labels", "facies", "depth"):
        _require_exact_grid_image(images[name], images["shape_corrected_baseline"],
                                  Path(sources[name]["path"]), name)
    return True
