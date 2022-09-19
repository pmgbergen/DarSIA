from __future__ import annotations

import cv2
import matplotlib as plt
import numpy as np
import daria as da


class CurvatureCorrection():


    def __init__(self, path_to_reference_image: str, width: float = 1, height: float = 1, in_meters: bool = True) -> None:
        self.config: dict() = {}
        self.reference_image = cv2.imread(path_to_reference_image)
        self.current_image = np.copy(self.reference_image)
        self.Ny, self.Nx = self.reference_image.shape[:2]
        self.width = width
        self.height = height

    def pre_bulge_correction(self, **kwargs) -> None:
        """
        Initialize the curvature correction by forcing all stright lines
        to curve inwards and not outwards.

        Arguments:
            kwargs (optional keyword arguments):
                "horizontal_bulge" (float): parameter for the curvature correction related to the
                    horizontal bulge of the image.
                "horizontal_center_offset" (int): offset in terms of pixel of the image center in
                    x-direction, as compared to the numerical center
                vertical_bulge (float): parameter for the curvature correction related to the
                    vertical bulge of the image.
                "vertical_center_offset" (int): offset in terms of pixel of the image center in
                    y-direction, as compared to the numerical center
        """
        self.config["init"] = {
            "horizontal_bulge": kwargs.pop("horizontal_bulge", 0),
            "horizontal_center_offset": kwargs.pop("horizontal_center_offset", 0),
            "vertical_bulge": kwargs.pop("verical_bulge", 0),
            "vertical_center_offset": kwargs.pop("vertical_center_offset", 0),
        }
        self.current_image = da.simple_curvature_correction(self.current_image, **self.config["init"])

    def crop(self, corner_points: list = [[11, 8], [16, 1755], [3165, 1748], [3165, 5]]):


        self.config["crop"] = {
            "pts_src": corner_points,
            "width": self.width,
            "height": self.width,
            "in meters": True,
        }

        self.current_image = da.extract_quadrilateral_ROI(self.current_image, **self.config["crop"])


    def