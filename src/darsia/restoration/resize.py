"""Module containing wrappers to resize routines from skimage and cv2.

Access is given through objects. Also contains utility routine which equalizes voxel
size lengths.

"""

from __future__ import annotations

from typing import Optional, Union, overload

import cv2
import numpy as np

import darsia


class Resize:
    """Object for resizing 2d arrays.

    Example:

    import darsia
    options = {
        "example resize x": 0.2,
        "example resize y": 0.5,
        "example interpolation": "inter_nearest",
    }
    resizer = darsia.Resize(key = "example ", **options)
    img_small = resizer(img_large)

    """

    def __init__(
        self,
        ref_image: Optional[darsia.Image] = None,
        shape: Optional[tuple[int]] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        interpolation: Optional[str] = None,
        dtype=None,
        key: str = "",
        **kwargs,
    ) -> None:
        """
        Args:
            ref_image (Image, optional): image whose shape is desired
            shape (tuple of int, optional): desired shape (in matrix indexing)
            fx (float, optional): resize factor in x-dimension.
            fy (float, optional): resize factor in y-dimension.
            interpolation (str, optional): interpolation method, default: None, invoking
                the default option in cv2.resize.
            dtype: conversion dtype before resizing; noting happens if None

        """

        # Cache parameters
        self.shape = kwargs.get(key + "resize shape", None) if shape is None else shape
        general_f = kwargs.get(key + "resize", None)
        self.fx = kwargs.get(key + "resize x", general_f) if fx is None else fx
        self.fy = kwargs.get(key + "resize y", general_f) if fy is None else fy
        self.dtype = kwargs.get(key + "resize dtype", None) if dtype is None else dtype

        # Check if reference image is provided
        if ref_image is not None:
            assert self.shape is None, (
                "Provide only reference image or shape (not both)."
            )
            self.shape = ref_image.num_voxels

        # Safety checks - double check resize options
        if self.shape is None:
            self.fx = 1 if self.fx is None else self.fx
            self.fy = 1 if self.fy is None else self.fy
            self.dsize = None
        else:
            # cv2 expects a flipped order
            self.dsize = tuple(reversed(self.shape))

        # Convert to CV2 format
        # NOTE: The default value for interpolation in cv2 is depending on whether
        # the image is shrinked or enlarged. In case of shrinking, cv2.INTER_NEAREST
        # is chosen, as opposed to cv2.INTER_LINEAR in the case of enlarging.
        interpolation_pre = (
            kwargs.get(key + "resize interpolation", None)
            if interpolation is None
            else interpolation
        )
        if interpolation_pre is None:
            self.interpolation = None
        elif interpolation_pre == "inter_area":
            self.interpolation = cv2.INTER_AREA
        elif interpolation_pre == "inter_linear":
            self.interpolation = cv2.INTER_LINEAR
        elif interpolation_pre == "inter_nearest":
            self.interpolation = cv2.INTER_NEAREST
        else:
            raise NotImplementedError(
                f"Interpolation option {interpolation_pre} is not implemented."
            )

        # Check for conservative rescaling
        self.is_conservative = kwargs.get(key + "resize conservative", False)

    @overload
    def __call__(self, img: np.ndarray, overwrite: bool = False) -> np.ndarray: ...

    @overload
    def __call__(self, img: darsia.Image, overwrite: bool = False) -> darsia.Image: ...

    def __call__(
        self, img: Union[np.ndarray, darsia.Image], overwrite: bool = False
    ) -> Union[np.ndarray, darsia.Image]:
        """
        Wrapper to cv2.resize.

        Args:
            img (np.ndarray, or darsia.Image): image

        Returns:
            np.ndarray, or darsia.Image: resized image, same format as input

        """
        input_is_image = isinstance(img, darsia.Image)
        if input_is_image:
            assert img.space_dim == 2

        # Extract original image
        img_array = img.img.copy() if input_is_image else img.copy()
        if self.dtype is not None:
            img_array = img_array.astype(self.dtype)

        # Treat all indices > 2 as channels
        original_shape = img_array.shape
        multi_channel_img_array = np.reshape(img_array, (*original_shape[:2], -1))

        # Split possibly multi-channel image in single channels
        img_channels: tuple[np.ndarray] = cv2.split(multi_channel_img_array)

        # Apply resizing to each channel separately
        resized_channels = []
        for channel in img_channels:
            if self.interpolation is None:
                resized_channels.append(
                    cv2.resize(
                        channel,
                        dsize=self.dsize,
                        fx=self.fx,
                        fy=self.fy,
                    )
                )
            else:
                resized_channels.append(
                    cv2.resize(
                        channel,
                        dsize=self.dsize,
                        fx=self.fx,
                        fy=self.fy,
                        interpolation=self.interpolation,
                    )
                )

        # Merge channels again and create resized image array of original shape (data-wise)
        resized_multi_channel_img_array = cv2.merge(resized_channels)
        resized_shape = *resized_multi_channel_img_array.shape[:2], *original_shape[2:]
        resized_img_array = np.reshape(resized_multi_channel_img_array, resized_shape)

        # Conserve the (weighted) sum
        if self.is_conservative:
            resized_img_array *= np.prod(img_array.shape[:2]) / np.prod(
                resized_img_array.shape[:2]
            )

        # Return resized image
        if input_is_image:
            meta = img.metadata()
            if overwrite:
                img = type(img)(resized_img_array, **meta)
                return img
            else:
                return type(img)(resized_img_array, **meta)
        else:
            if overwrite:
                img = resized_img_array
                return img
            else:
                # Return resized array
                return resized_img_array


def resize(
    image: darsia.Image,
    ref_image: Optional[darsia.Image] = None,
    shape: Optional[tuple[int]] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    interpolation: Optional[str] = None,
    dtype=None,
) -> darsia.Image:
    """Function wrapper to Resize object.

    Args:
        image (darsia.Image): image to be resized
        ref_image (Image, optional): reference image whose shape is desired
        shape (tuple of int, optional): desired shape (in matrix indexing)
        fx (float, optional): resize factor in x-dimension.
        fy (float, optional): resize factor in y-dimension.
        interpolation (str, optional): interpolation method, default: None, invoking
            the default option in cv2.resize.
        dtype: conversion dtype before resizing; noting happens if None

    """
    # Define Resize object
    resizer = Resize(
        ref_image=ref_image,
        shape=shape,
        fx=fx,
        fy=fy,
        interpolation=interpolation,
        dtype=dtype,
    )

    # Return resized image
    return resizer(image)


def equalize_voxel_size(
    image: darsia.Image, voxel_size: Optional[float] = None, **kwargs
) -> darsia.Image:
    """Resize routine which keeps physical dimensions, but unifies the voxel length.

    Args:
        image (darsia.Image): image to be resized
        voxel_size (float, optional): side length, min of the voxel side of the image
            if None.
        keyword arguments:
            interpolation (str): interpolation type used for resize

    Returns:
        darsia.Image: resized image

    """
    # Fetch dimensions to be kept
    dimensions = image.dimensions

    # Determine resulting shape of the resized image
    if voxel_size is None:
        voxel_size = min(image.voxel_size)
    shape = tuple(int(d / voxel_size) for d in dimensions)

    # Perform resize
    interpolation = kwargs.get("interpolation")
    resize = Resize(shape=shape, interpolation=interpolation)
    return resize(image)


def uniform_refinement(image: darsia.Image, levels: int) -> darsia.Image:
    """Uniform refinement.

    Args:
        image (darsia.Image): image
        levels (int): refinement levels, if positive, coarsing levels, if negative.

    Returns:
        darsia.Image: resized image

    """
    # Fetch original data array
    array = image.img.copy()

    for level in range(abs(levels)):
        if levels > 0:
            # Refinement
            for i in range(image.space_dim):
                array = np.repeat(array, 2, axis=i)

        elif levels < 0:
            # Coarsening
            for i in range(image.space_dim):
                # Slices to address each second element along axis i
                def i_slice(i_slice_item):
                    return tuple(
                        i_slice_item if i == j else slice(0, None)
                        for j in range(image.space_dim)
                    )

                slice_0 = i_slice(slice(0, None, 2))
                slice_1 = i_slice(slice(1, None, 2))

                # Determine weight for slice_0 elements
                axis_length = image.img.shape[i]
                weight_0 = 0.5 * np.ones(array[slice_0].shape)
                half_axis_length = int(np.floor(axis_length) / 2)
                double_axis_length = 2 * half_axis_length
                if axis_length % 2 == 1:
                    weight_0[i_slice(slice(double_axis_length, None))] = 1

                # The weight for slice_1 is constant
                weight_1 = 0.5

                # Weighted sum for coarsening
                sub_array_0 = array[slice_0]
                sub_array_1 = array[slice_1]
                array = np.multiply(weight_1, sub_array_0)
                array[i_slice(slice(0, half_axis_length))] += np.multiply(
                    weight_1, sub_array_1
                )

    # Return resized image
    meta = image.metadata()
    return type(image)(array, **meta)
