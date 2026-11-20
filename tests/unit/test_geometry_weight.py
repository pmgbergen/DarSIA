"""Weight and voxel volume of porous geometries: pinned to the last digit.

The geometry of a rig (`ExtrudedPorousGeometry`) weights each pixel with its pore
volume, pixel area x porosity x depth. These tests fix the stored values, the regions
and the integrals bit for bit, and how many full-size arrays a geometry owns.

"""

import gc
import weakref

import cv2
import numpy as np
import pytest

import darsia

SHAPE = (6, 8)
DIMENSIONS = [0.6, 0.8]
AREA = np.prod([0.6 / 6, 0.8 / 8])


def _field(value: float, nan: bool) -> np.ndarray:
    rng = np.random.default_rng(int(10 * value))
    array = value * (0.5 + rng.random(SHAPE))
    if nan:
        array[2, 3] = np.nan
    return array


def _input(kind: str, value: float, nan: bool = True):
    if kind == "float":
        return value
    array = _field(value, nan)
    if kind == "array":
        return array
    return darsia.Image(array, dimensions=DIMENSIONS, series=False, scalar=True)


def _values(field) -> np.ndarray | float:
    return field.img if isinstance(field, darsia.Image) else field


def _geometry(porosity, depth) -> darsia.ExtrudedPorousGeometry:
    return darsia.ExtrudedPorousGeometry(
        porosity, depth, space_dim=2, num_voxels=SHAPE, dimensions=DIMENSIONS
    )


KINDS = ["float", "array", "image"]


@pytest.mark.parametrize("porosity_kind", KINDS)
@pytest.mark.parametrize("depth_kind", KINDS)
def test_weight_and_voxel_volume(porosity_kind, depth_kind):
    porosity = _input(porosity_kind, 0.3)
    depth = _input(depth_kind, 0.02, nan=False)
    geometry = _geometry(porosity, depth)
    product = np.multiply(_values(porosity), _values(depth))
    if np.ndim(product) == 0:
        assert np.ndim(geometry.weight) == np.ndim(geometry.voxel_volume) == 0
    # Weight without NaN (0 instead), voxel volume with the NaN of the product
    np.testing.assert_array_equal(geometry.weight, np.nan_to_num(product, nan=0.0))
    np.testing.assert_array_equal(geometry.voxel_volume, np.multiply(AREA, product))
    assert geometry.porosity is porosity and geometry.depth is depth


@pytest.mark.parametrize(
    "corners", [[[0.15, 0.12], [0.55, 0.43]], [[0.0, 0.0], [0.8, 0.6]]]
)
def test_subregion(corners):
    porosity, depth = _input("image", 0.3), _input("image", 0.02, nan=False)
    geometry = _geometry(porosity, depth)
    roi = darsia.CoordinateArray(corners)
    sub = geometry.subregion(roi)

    weight = np.nan_to_num(porosity.img * depth.img, nan=0.0)
    expected = (
        darsia.Image(weight, dimensions=geometry.dimensions, space_dim=2)
        .subregion(roi)
        .img
    )
    np.testing.assert_array_equal(sub.weight, expected)
    sub_area = np.prod(sub.voxel_size)
    np.testing.assert_array_equal(sub.voxel_volume, np.multiply(sub_area, expected))
    data = _field(1.0, nan=False)[: expected.shape[0], : expected.shape[1]]
    assert sub.integrate(data) == np.sum(np.sum(sub_area * expected * data, 0), 0)


def test_integrals():
    geometry = _geometry(
        _input("image", 0.3, nan=False), _input("image", 0.02, nan=False)
    )
    voxel_volume = geometry.voxel_volume
    data = _field(1.0, nan=False)
    assert geometry.integrate(data) == np.sum(np.sum(voxel_volume * data, 0), 0)

    # Data of another shape: voxel volumes resized conservatively
    reduced = data[::2, ::2].copy()
    resized = cv2.resize(voxel_volume, (4, 3), interpolation=cv2.INTER_AREA) * 4.0
    np.testing.assert_array_equal(geometry.voxel_volumes(reduced.shape), resized)
    assert geometry.integrate(reduced) == np.sum(np.sum(resized * reduced, 0), 0)


def _owned_arrays(geometry, inputs: list) -> int:
    """Distinct arrays of the geometry not shared with the inputs."""
    owned = {
        id(value)
        for value in vars(geometry).values()
        if isinstance(value, np.ndarray)
        and not any(np.shares_memory(value, field) for field in inputs)
    }
    return len(owned)


def test_owned_arrays():
    porosity, depth = _input("image", 0.3), _input("image", 0.02)
    geometry = _geometry(porosity, depth)
    assert _owned_arrays(geometry, [porosity.img, depth.img]) == 1  # voxel volume
    sub = geometry.subregion(darsia.CoordinateArray([[0.15, 0.12], [0.55, 0.43]]))
    assert _owned_arrays(sub, []) == 1


def test_freed_without_garbage_collection():
    gc.disable()
    try:
        geometry = _geometry(_input("image", 0.3), _input("image", 0.02))
        sub = geometry.subregion(darsia.CoordinateArray([[0.15, 0.12], [0.55, 0.43]]))
        references = [weakref.ref(geometry), weakref.ref(sub)]
        del geometry, sub
        assert all(reference() is None for reference in references)
    finally:
        gc.enable()


@pytest.mark.parametrize("porous", [True, False])
def test_region_does_not_keep_the_parent(porous):
    if porous:
        geometry = _geometry(_input("image", 0.3), _input("image", 0.02))
    else:
        geometry = darsia.WeightedGeometry(
            _field(0.3, nan=True), space_dim=2, num_voxels=SHAPE, dimensions=DIMENSIONS
        )
    gc.disable()
    try:
        sub = geometry.subregion(darsia.CoordinateArray([[0.15, 0.12], [0.55, 0.43]]))
        weight = sub.weight
        parent = weakref.ref(geometry)
        del geometry
        assert parent() is None
        np.testing.assert_array_equal(sub.weight, weight)
    finally:
        gc.enable()


def test_update_is_not_available():
    geometry = _geometry(_input("image", 0.3), _input("image", 0.02))
    with pytest.raises(NotImplementedError):
        geometry.update(0.03)
