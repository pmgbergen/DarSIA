"""Modulde for crop assistant.

Main prupose of the assistant is to produce input arguments for the
'crop' option of CurvatureCorrection.

"""

from typing import Optional, Union

import numpy as np

import darsia


class CropAssistant(darsia.PointSelectionAssistant):
    """Graphical assistant for cropping images as part of CurvatureCorrection."""

    def __init__(self, img: darsia.Image, **kwargs) -> None:
        """Constructor.

        Based on PointSelectionAssistant configured to use points in Voxel format.

        Args:
            img (darsia.Image): input image

        """
        super().__init__(img, **kwargs)

        # Initialize containers
        self._reset()

        # Redefine pre-defined attributes
        self.pts: Optional[darsia.VoxelArray] = None
        """Selected corners to define box after cropping (voxels in matrix indexing)."""

        # Prepare further output
        self.finalized_prompt_input = False
        """Flag controlling whether the user has entered the width and height."""

        self.width = None
        """Identified width of the box."""

        self.height = None
        """Identified height of the box."""

    def _reset(self) -> None:
        """Reset list of points."""
        super()._reset()

    # ! ---- Interactive mode ---- ! #

    def __call__(self) -> dict:
        """Run the assistant.

        Returns:
            dict: configuration for the 'crop' option of CurvatureCorrection

        """
        # Prompt a welcome message
        print("Welcome to the CropAssistant!")

        # Run point selection and check number of points is 4
        super().__call__()
        assert len(self.pts) == 4, "Wrong number of points selected"

        # Ask user to enter width and height into prompt
        if not self.finalized_prompt_input:
            self.width = float(input("Enter width of box: "))
            self.height = float(input("Enter height of box: "))
            self.finalized_prompt_input = True

        # Define a dictionary for input of the 'crop' option of CurvatureCorrection
        config = self._define_config()

        return config

    def _define_config(self) -> dict:
        """Define a dictionary for input of the 'crop' option of CurvatureCorrection.

        Returns:
            dict: configuration for the 'crop' option of CurvatureCorrection

        """
        return {
            "crop": {
                "width": self.width,
                "height": self.height,
                "pts_src": self.pts,
            },
        }

    def _print_info(self) -> None:
        """Print out information about the assistant."""
        print(self._define_config())

    # ! ---- Automatic mode ---- ! #

    def from_image(
        self, color: Union[list[float], np.ndarray], width: float, height: float
    ) -> dict:
        """Run the assistant in automatic mode.

        Detect marks and define a box based on them.

        Args:
            color (Union[list[float], np.ndarray]): color of the marks
            width (float): width of the box
            height (float): height of the box

        Returns:
            dict: configuration for the 'crop' option of CurvatureCorrection

        """
        if not isinstance(color, np.ndarray):
            color = np.array(color)
        color = color.astype(float)

        # Find marks in the image
        self.pts = self._find_marks(color)

        # Define width and height of the box
        self.width = width
        self.height = height

        # Define a dictionary for input of the 'crop' option of CurvatureCorrection
        config = self._define_config()

        return config

    def _find_marks(self, color: Union[list[float], np.ndarray]) -> darsia.VoxelArray:
        """Find marks in the image.

        Args:
            color (Union[list[float], np.ndarray]): color of the marks

        Returns:
            darsia.VoxelArray: selected corners to define box after cropping (voxels in matrix indexing)

        """
        # Find all pixels with the specified color
        marked_voxels = darsia.detect_color(self.img, color, tolerance=5e-2)

        # Find the four corners of the box being the four pixels with smallest
        # distance to the image corners
        top_left = darsia.detect_closest_point(marked_voxels, darsia.Voxel([0, 0]))
        top_right = darsia.detect_closest_point(
            marked_voxels, darsia.Voxel([0, self.img.shape[1]])
        )
        bottom_left = darsia.detect_closest_point(
            marked_voxels, darsia.Voxel([self.img.shape[0], 0])
        )
        bottom_right = darsia.detect_closest_point(
            marked_voxels, darsia.Voxel([self.img.shape[0], self.img.shape[1]])
        )

        voxels = darsia.VoxelArray([top_left, bottom_left, bottom_right, top_right])
        return voxels
