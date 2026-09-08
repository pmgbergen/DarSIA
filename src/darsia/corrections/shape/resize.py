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

        Parameters
        ----------
        ref_image : Image, optional
            Image whose shape is desired.
        shape : tuple of int, optional
            Desired shape (in matrix indexing).
        fx : float, optional
            Resize factor in x-dimension.
        fy : float, optional
            Resize factor in y-dimension.
        interpolation : str, optional
            Interpolation method.
        dtype
            Conversion dtype before resizing; nothing happens if None.
        key : str
            Prefix for kwargs-based configuration.
        **kwargs
            Additional configuration options.
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

        Parameters
        ----------
        image : array or Image
            Image to resize.
        overwrite : bool
            Whether to mutate in place.

        Returns
        -------
        array or Image
            Resized image, same format as input.
        """
        if isinstance(image, darsia.Image):
            self._input_is_extensive_image = isinstance(image, darsia.ExtensiveImage)
        return super().__call__(image, overwrite=overwrite)

    def correct_array(self, image: np.ndarray) -> np.ndarray:
        """Resize a raw numpy array.

        Parameters
        ----------
        image : np.ndarray
            Input image array.

        Returns
        -------
        np.ndarray
            Resized image array.
        """
        return self._resize.resize_array(
            image, is_extensive=self._input_is_extensive_image
        )

    def save(self, path: Path) -> None:
        """Save the resize correction to a file.

        Parameters
        ----------
        path : Path
            Path to save the parameters to.
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

        Parameters
        ----------
        path : Path
            Path to load the parameters from.
        """
        self._resize = darsia.Resize()
        self._resize.load(path)
