"""Transfer functions"""

import colour
import cv2
import numpy as np


class EOTF:
    """
    Electro-optical transfer function (EOTF),the standard transfer
    function for sRGB, cf. https://en.wikipedia.org/wiki/SRGB
    """

    def __init__(self):
        """Define look up table (LUT), representing the EOTF."""

        # Use the pre-defined EOTF implemented by colour and apply to any possible
        # uint8 value.
        self.table_eotf = colour.cctf_decoding(np.arange(0, 256) / 255.0)

        # Define the inverse of EOTF, which acts on value in [0,1]. Here, the table
        # translates to uint8 values, i.e., for before applying values of images
        # with values in [0,1] have to be transformed accordingly to allow the application.
        self.table_eotf_inverse = (
            colour.cctf_encoding(np.arange(0, 256) / 255.0) * 255.0
        ).astype("uint8")

    def adjust(self, image: np.ndarray) -> np.ndarray:
        """Apply EOTF using the look up table.

        Arguments:
            image (np.ndarray): image in linear RGB (uint8) format.

        Returns:
            np.ndarray: image with tranformed color space, with values in [0,1].
        """
        return cv2.LUT(image, self.table_eotf)

    def inverse_approx(self, image: np.ndarray) -> np.ndarray:
        """Apply approximate of the inverse of EOTF using the look up table.
        In order to apply the exact inverse, the expensive function colout.cctf_encoding
        would have to be applied to all pixels (with values in [0,1]). By restricting the
        allowed input values to uint8 values, allows for faster LUT.

        Arguments:
            image (np.ndarray): image in gamma-corrected RGB format with values in [0,1].

        Returns:
            np.ndarray: image in linear RGB (uint8) format.
        """

        # Need to transform values to uint8 first before applying the LUT.
        return cv2.LUT((255.0 * image).astype("uint8"), self.table_eotf_inverse)
