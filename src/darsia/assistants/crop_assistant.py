"""Modulde for crop assistant.

Main prupose of the assistant is to produce input arguments for the
'crop' option of CurvatureCorrection.

"""

from typing import Optional

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
