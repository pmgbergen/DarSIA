"""Module containing BaseCorrection-compliant resize correction."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

import darsia


class ResizeCorrection(darsia.BaseCorrection):
    """Shape correction wrapping darsia.Resize to fit the BaseCorrection interface.

    darsia.Resize.__call__'s overwrite=True path for Image inputs does not
    actually mutate the caller's object — it only rebinds a local variable to
    a new Image, leaving the original stale. This wrapper reuses Resize's
    array-level resize logic (resize_array) but routes Image/overwrite handling
    through BaseCorrection.__call__, which correctly mutates image.img in place.

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
        """Constructor matching darsia.Resize's signature.

        Args:
            ref_image (Image, optional): image whose shape is desired
            shape (tuple of int, optional): desired shape (in matrix indexing)
            fx (float, optional): resize factor in x-dimension.
            fy (float, optional): resize factor in y-dimension.
            interpolation (str, optional): interpolation method.
            dtype: conversion dtype before resizing; nothing happens if None.
            key (str): prefix for kwargs-based configuration.
            **kwargs: additional configuration options.

        """
        self._resize = darsia.Resize(
            ref_image=ref_image,
            shape=shape,
            fx=fx,
            fy=fy,
            interpolation=interpolation,
            dtype=dtype,
            key=key,
            **kwargs,
        )
        self._input_is_extensive_image = False

    def __call__(
        self,
        image: Union[np.ndarray, darsia.Image],
        overwrite: bool = False,
    ) -> Union[np.ndarray, darsia.Image]:
        """Dispatch to BaseCorrection.__call__, capturing Image-type info first.

        Captures whether the input is an ExtensiveImage (unavailable inside
        correct_array, which receives only np.ndarray), then delegates to
        BaseCorrection's correct overwrite/copy/metadata dispatch.

        Args:
            image (array or Image): image to resize.
            overwrite (bool): whether to mutate in place.

        Returns:
            array or Image: resized image, same format as input.

        """
        if isinstance(image, darsia.Image):
            self._input_is_extensive_image = isinstance(image, darsia.ExtensiveImage)
        return super().__call__(image, overwrite=overwrite)

    def correct_array(self, image: np.ndarray) -> np.ndarray:
        """Resize a raw numpy array.

        Args:
            image (np.ndarray): input image array.

        Returns:
            np.ndarray: resized image array.

        """
        return self._resize.resize_array(
            image, is_extensive=self._input_is_extensive_image
        )

    def save(self, path: Path) -> None:
        """Save the resize correction to a file.

        Args:
            path (Path): path to save the parameters to.

        """
        np.savez(
            path,
            class_name=type(self).__name__,  # Reason why not _resize.save.
            shape=self._resize.shape,
            dsize=self._resize.dsize,
            fx=self._resize.fx,
            fy=self._resize.fy,
            interpolation=self._resize.interpolation,
            dtype=self._resize.dtype,
            is_conservative=self._resize.is_conservative,
        )

    def load(self, path: Path) -> None:
        """Load the resize correction from a file.

        Args:
            path (Path): path to load the parameters from.

        """
        self._resize = darsia.Resize()
        self._resize.load(path)
