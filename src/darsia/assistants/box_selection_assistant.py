"""Module for defining subregions interactively."""

from typing import Optional

import numpy as np

import darsia


class BoxSelectionAssistant(darsia.PointSelectionAssistant):
    def __init__(self, img: darsia.Image, **kwargs) -> None:
        super().__init__(img)

        self.name = "Box selection assistant"
        """Name of assistant."""

        self.width = kwargs.get("width", 100)
        """Width and height of selected box(es)."""

        self.boxes = None
        """Output of assistant."""

        self.shape = img.shape
        """Image shape."""

        self.dim = img.space_dim
        """Dimension of image."""

        if self.dim != 2:
            raise NotImplementedError

    def _convert_pts(self) -> None:
        """Convert list of points to list of boxes in terms of slices."""

        self.boxes = []
        half_width = self.width / 2
        for pt in self.pts:
            self.boxes.append(
                tuple(
                    [
                        slice(
                            max(int(pt[d] - half_width), 0),
                            min(int(pt[d] + half_width), self.shape[d]),
                        )
                        for d in range(self.dim)
                    ]
                )
            )

    def __call__(self) -> Optional[np.ndarray]:
        """Call the assistant."""

        super().__call__()
        self._convert_pts()
        return self.boxes

    def _print_info(self) -> None:
        """Print info about boxes so far assigned by the assistant."""
        self._convert_pts()
        print(self.boxes)
