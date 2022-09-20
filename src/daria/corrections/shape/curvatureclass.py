"""
Curvature correction class.

A class for setup and application of curvature correction.
"""

from __future__ import annotations

import json
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

import daria as da


class CurvatureCorrection:
    """
    Class for curvature correction of curved images.


    """

    def __init__(self, **kwargs) -> None:

        if "image_source" in kwargs:
            self.config: dict() = {}
            im_source = kwargs.pop("image_source")
            if isinstance(im_source, np.ndarray):
                self.reference_image = im_source
            elif isinstance(im_source, str):
                self.reference_image = cv2.imread(im_source)
            else:
                raise Exception(
                    "Invalid image data. Provide either a path to an image or an image array."
                )
            self.current_image = np.copy(self.reference_image)
            self.Ny, self.Nx = self.reference_image.shape[:2]
            self.in_meters = kwargs.pop("in_meters", True)
            self.width = kwargs.pop("width", 1.0)
            self.height = kwargs.pop("height", 1.0)

        elif "config_source" in kwargs:
            with open(kwargs.pop("config_source"), "r") as openfile:
                self.config = json.load(openfile)

        else:
            raise Exception(
                "Please provide either an image as 'image_source' \
                    or a config file as 'config_source'."
            )

    def pre_bulge_correction(self, **kwargs) -> None:
        """
        Initialize the curvature correction by forcing all stright lines
        to curve inwards and not outwards.

        Arguments:
            kwargs (optional keyword arguments):
                "horizontal_bulge" (float): parameter for the curvature correction related to
                    the horizontal bulge of the image.
                "horizontal_center_offset" (int): offset in terms of pixel of the image center
                    in x-direction, as compared to the numerical center
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
        self.current_image = da.simple_curvature_correction(
            self.current_image, **self.config["init"]
        )

    def crop(
        self,
        corner_points: list = [
            [11, 8],
            [16, 1755],
            [3165, 1748],
            [3165, 5],
        ],
    ) -> None:

        self.config["crop"] = {
            "pts_src": corner_points,
            "width": self.width,
            "height": self.height,
            "in meters": self.in_meters,
        }

        self.current_image = da.extract_quadrilateral_ROI(
            self.current_image, **self.config["crop"]
        )

    def bulge_corection(
        self, left: int = 0, right: int = 0, top: int = 53, bottom: int = 57
    ) -> None:
        (
            horizontal_bulge,
            horizontal_bulge_center_offset,
            vertical_bulge,
            vertical_bulge_center_offset,
        ) = da.compute_bulge(
            self.Nx, self.Ny, left=left, right=right, top=top, bottom=bottom
        )

        self.config["bulge"] = {
            "horizontal_bulge": horizontal_bulge,
            "horizontal_center_offset": horizontal_bulge_center_offset,
            "vertical_bulge": vertical_bulge,
            "vertical_center_offset": vertical_bulge_center_offset,
        }

        self.current_image = da.simple_curvature_correction(
            self.current_image, **self.config["bulge"]
        )

    def stretch_correction(
        self,
        point_source: list = [585, 676],
        point_destination: list = [567, 676],
        stretch_center: list = [1476, 1020],
    ) -> None:
        (
            horizontal_stretch,
            horizontal_stretch_center_offset,
            vertical_stretch,
            vertical_stretch_center_offset,
        ) = da.compute_stretch(
            self.Nx,
            self.Ny,
            point_source=point_source,
            point_destination=point_destination,
            stretch_center=stretch_center,
        )

        self.config["stretch"] = {
            "horizontal_stretch": horizontal_stretch,
            "horizontal_center_offset": horizontal_stretch_center_offset,
            "vertical_stretch": vertical_stretch,
            "vertical_center_offset": vertical_stretch_center_offset,
        }

        self.current_image = da.simple_curvature_correction(
            self.current_image, **self.config["stretch"]
        )

    def return_current_image(self) -> da.Image:
        return da.Image(self.current_image, width=self.width, height=self.height)

    def __call__(self, image_source: Union[str, np.ndarray]) -> np.ndarray:

        if isinstance(image_source, np.ndarray):
            image_tmp = image_source
        elif isinstance(image_source, str):
            image_tmp = cv2.imread(image_source)
        else:
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )
        image_tmp = da.simple_curvature_correction(image_tmp, **self.config["init"])
        image_tmp = da.extract_quadrilateral_ROI(image_tmp, **self.config["crop"])
        image_tmp = da.simple_curvature_correction(image_tmp, **self.config["bulge"])
        image_tmp = da.simple_curvature_correction(image_tmp, **self.config["stretch"])
        return image_tmp


    def write_config_to_file(self, path: str) -> None:
        with open(path, "w") as outfile:
            json.dump(self.config, outfile, indent=4)

    def show_current(self):
        plt.imshow(da.BGR2RGB(self.current_image))

    def read_config_from_file(self, path: str) -> None:
        with open(path, "r") as openfile:
            self.config = json.load(openfile)
