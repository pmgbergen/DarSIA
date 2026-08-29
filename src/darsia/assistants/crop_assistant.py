"""Module for crop assistant.

Main purpose of the assistant is to produce input arguments for the
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

        self.corners: dict[str, darsia.Voxel] = {}
        """Named corners as {top_left, bottom_left, bottom_right, top_right}."""

        # Prepare further output
        self.finalized_prompt_input = False
        """Flag controlling whether the user has entered the width and height."""

        self.width = kwargs.get("width", None)
        """Identified width of the box."""

        self.height = kwargs.get("height", None)
        """Identified height of the box."""

    def _reset(self) -> None:
        """Reset list of points."""
        super()._reset()

    def _classify_corners(self, pts: darsia.VoxelArray) -> dict[str, darsia.Voxel]:
        """Classify 4 points into named corners by their position relative to centroid.

        Assigns each point to a corner (top_left, bottom_left, bottom_right,
        top_right) based on its row/col relative to the centroid of all 4 points.
        This makes the assistant robust to click order.

        Args:
            pts: VoxelArray of 4 points in arbitrary order, each [row, col].

        Returns:
            dict with keys "top_left", "bottom_left", "bottom_right", "top_right",
            each mapping to a (row, col) tuple.
        """
        assert len(pts) == 4, "Expected 4 points"
        pts_array = np.array(pts)
        centroid = pts_array.mean(axis=0)

        corners = {}
        for pt in pts:
            row, col = pt
            is_top = row < centroid[0]
            is_left = col < centroid[1]

            if is_top and is_left:
                key = "top_left"
            elif is_top and not is_left:
                key = "top_right"
            elif not is_top and is_left:
                key = "bottom_left"
            else:
                key = "bottom_right"

            corners[key] = pt

        return corners

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

        # Classify clicked points into named corners (robust to click order)
        self.corners = self._classify_corners(self.pts)

        # Ask user to enter width and height into prompt
        if not self.finalized_prompt_input:
            if self.width is None:
                self.width = float(input("Enter width of box: "))
            if self.height is None:
                self.height = float(input("Enter height of box: "))
            self.finalized_prompt_input = True

        # Define a dictionary for input of the 'crop' option of CurvatureCorrection
        config = self._define_config()

        return config

    def _define_config(self) -> dict:
        """Define a dictionary for input of the 'crop' option of CurvatureCorrection.

        Converts internal Voxel-based corners to tuple[int, int] for config layer.

        Returns:
            dict: configuration for the 'crop' option of CurvatureCorrection

        """
        return {
            "crop": {
                "top_left": tuple(int(x) for x in self.corners["top_left"]),
                "bottom_left": tuple(int(x) for x in self.corners["bottom_left"]),
                "bottom_right": tuple(int(x) for x in self.corners["bottom_right"]),
                "top_right": tuple(int(x) for x in self.corners["top_right"]),
                "width": self.width,
                "height": self.height,
            },
        }

    def _print_info(self) -> None:
        """Print out information about the assistant."""
        print(self._define_config())

    # ! ---- Automatic mode ---- ! #

    def from_image(
        self,
        color: Union[list[float], np.ndarray],
        width: Optional[float],
        height: Optional[float],
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

        # Find marks in the image (returns named corners dict)
        self.corners = self._find_marks(color)

        # Define width and height of the box
        if self.width is None:
            assert width is not None, "Width not provided"
            self.width = width
        if self.height is None:
            assert height is not None, "Height not provided"
            self.height = height

        # Define a dictionary for input of the 'crop' option of CurvatureCorrection
        config = self._define_config()

        return config

    def _find_marks(
        self, color: Union[list[float], np.ndarray]
    ) -> dict[str, darsia.Voxel]:
        """Find marks in the image and classify into named corners.

        Args:
            color (Union[list[float], np.ndarray]): color of the marks

        Returns:
            dict with keys "top_left", "bottom_left", "bottom_right", "top_right",
            each mapping to a (row, col) tuple of the detected corner.

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

        return {
            "top_left": top_left,
            "top_right": top_right,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
        }
