from typing import Literal, overload

import numpy as np

import darsia


class ColorPathInterpolation(darsia.Model):
    def __init__(
        self,
        color_path: darsia.ColorPath,
        interpolation: Literal["absolute", "relative"] = "absolute",
    ):
        self.color_path = color_path
        self.interpolation = interpolation

    def update(
        self,
        color_path: darsia.ColorPath | None = None,
        supports: list | None = None,
        values: list | None = None,
        append: bool = False,
    ) -> None:
        if color_path is not None:
            self.color_path = color_path
        else:
            assert supports is not None and values is not None
            if append:
                self.color_path.add(supports, values)
            else:
                self.color_path = darsia.ColorPath(
                    colors=supports, base_color=supports[0], values=values, mode="rgb"
                )

    def update_model_parameters(
        self, parameters: darsia.ColorPath, dofs: str | None = None
    ) -> None:
        raise NotImplementedError(
            "ColorPathInterpolation does not support update_model_parameters."
        )

    def calibrate(self):
        raise NotImplementedError(
            "ColorPathInterpolation does not support calibration."
        )

    @overload
    def __call__(self, img: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, img: darsia.Image) -> darsia.Image: ...

    def __call__(self, img: np.ndarray | darsia.Image) -> np.ndarray | darsia.Image:
        if self.interpolation == "absolute":
            return self.color_path.absolute_inverse(img)
        elif self.interpolation == "relative":
            return self.color_path.relative_inverse(img)
        else:
            raise ValueError
