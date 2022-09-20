from __future__ import annotations

from typing import Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import daria as da


class CurvatureCorrection:
    def __init__(
        self,
        image_source: np.ndarray,
        width: float = 1,
        height: float = 1,
        in_meters: bool = True,
    ) -> None:
        self.config: dict() = {}
        if isinstance(image_source, np.ndarray):
            self.reference_image = image_source
        elif isinstance(image_source, str):
            self.reference_image = cv2.imread(image_source)
        else:
            raise Exception(
                "Invalid image data. Provide either a path to an image or an image array."
            )
        self.current_image = np.copy(self.reference_image)
        self.Ny, self.Nx = self.reference_image.shape[:2]
        self.in_meters = in_meters
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
        self.current_image = da.simple_curvature_correction(
            self.current_image, **self.config["init"]
        )

    def crop(
        self, corner_points: list =[
        [11, 8],
        [16, 1755],
        [3165, 1748],
        [3165, 5],
    ]
    ) -> None:

        self.config["crop"] = {
            "pts_src": corner_points,
            "width": self.width,
            "height": self.width,
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
        ) = da.compute_bulge(self.Nx, self.Ny, left, right, top, bottom)

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
            self.Nx, self.Ny, point_source, point_destination, stretch_center
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

    def apply_full_correction(self, image_source: Union[str, np.ndarray]) -> da.Image:

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
        return da.Image(image_tmp, width=self.width, height=self.height)

    def write_config_to_file(self, path: str) -> None:
        f = open(path, "w")
        f.write(self.config)
        f.close()

    def show_current(self):
        plt.imshow(da.BGR2RGB(self.current_image))

    def read_config_from_file(self, path: str) -> None:
        f= open(path, "r")
        self.config = f