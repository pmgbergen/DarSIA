"""Module for color path interpolation models."""

import abc
import json
from pathlib import Path
from typing import overload, Literal

import numpy as np

import darsia


class ColorPathFunction(darsia.Model):
    def __init__(
        self,
        color_path: darsia.ColorPath,
        color_mode: darsia.ColorMode,
    ):
        self.color_path = color_path
        """Underlying color path."""
        self.color_mode = color_mode
        """Color mode used for parametrization."""

    @abc.abstractmethod
    def update_model_parameters(
        self,
        parameters: np.ndarray | list[float],
        dofs: list[tuple[int, str]] | Literal["all"] | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def calibrate(self): ...

    @overload
    def __call__(self, img: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, img: darsia.Image) -> darsia.Image: ...

    @abc.abstractmethod
    def __call__(
        self, image: np.ndarray | darsia.Image
    ) -> np.ndarray | darsia.Image: ...


class ColorPathInterpolation(ColorPathFunction):
    def __init__(
        self,
        color_path: darsia.ColorPath,
        color_mode: darsia.ColorMode,
        values: np.ndarray | list[float] | None = None,
        ignore_spectrum: darsia.ColorSpectrum | None = None,
    ):
        self.color_path = color_path
        self.color_mode = color_mode
        self.values = np.array(
            values if values is not None else color_path.equidistant_distances
        )
        assert len(self.values) == self.color_path.num_segments + 1, (
            "Length of values must match number of segments + 1."
        )
        self.ignore_spectrum = ignore_spectrum
        """Spectrum to ignore during parametrization."""

    def __str__(self) -> str:
        return (
            f"ColorPathInterpolation(color_mode={self.color_mode}, "
            f"color_path={self.color_path}, values={self.values})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    # TODO add for calibration
    def update_model_parameters(
        self,
        parameters: np.ndarray | list[float],
        dofs: list[tuple[int, str]] | Literal["all"] | None = None,
    ) -> None:
        """Update the interpolation values of the color path."""
        self.values = np.array(parameters)

    def calibrate(self):
        raise NotImplementedError(
            "ColorPathInterpolation does not support calibration."
        )

    def to_dict(self) -> dict:
        """Convert the ColorPathInterpolation to a dictionary representation.

        Returns:
            dict: Dictionary representation of the ColorPathInterpolation.
        """
        return {
            "color_path": self.color_path.to_dict(),
            "color_mode": self.color_mode,
            "values": self.values.tolist(),
            "ignore_spectrum": (
                self.ignore_spectrum.to_dict() if self.ignore_spectrum else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ColorPathInterpolation":
        """Create a ColorPathInterpolation from a dictionary.

        Args:
            data (dict): Dictionary representation of the ColorPathInterpolation.

        Returns:
            ColorPathInterpolation: The created ColorPathInterpolation instance.

        """
        return cls(
            color_path=darsia.ColorPath.from_dict(data["color_path"]),
            color_mode=data["color_mode"],
            values=np.array(data["values"]),
            ignore_spectrum=(
                darsia.ColorSpectrum.from_dict(data["ignore_spectrum"])
                if data["ignore_spectrum"] is not None
                else None
            ),
        )

    def save(self, path: Path) -> None:
        """Save the ColorPathInterpolation to a file.

        Args:
            path (Path): The path to the file where the ColorPathInterpolation should be saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ColorPathInterpolation":
        """Load the ColorPathInterpolation from a file.

        Args:
            path (Path): The path to the file from which the ColorPathInterpolation should be loaded.

        Returns:
            ColorPathInterpolation: Loaded ColorPathInterpolation instance.
        """
        with open(path.with_suffix(".json"), "r") as f:
            data = json.load(f)
        return ColorPathInterpolation.from_dict(data)

    @overload
    def __call__(self, img: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, img: darsia.Image) -> darsia.Image: ...

    def __call__(self, image: np.ndarray | darsia.Image) -> np.ndarray | darsia.Image:
        """Inverse of the color path to an image defined by the closest color
        representation on the path, and linearly interpolated with piecewise
        defined functions (through values).

        Args:
            image: Input image to be interpreted.

        Returns:
            darsia.Image: Parametrization of the input image in terms of the color path.

        """
        if isinstance(image, np.ndarray):
            return self._parametrize_colors(image)
        if isinstance(image, darsia.Image):
            return darsia.full_like(
                image,
                fill_value=self._parametrize_colors(image.img),
                mode="voxels",
            )
        else:
            raise TypeError(
                """"Input image must be of type np.ndarray or darsia.Image, """
                f"""got {type(image)}."""
            )

    def _parametrize_colors(self, colors: np.ndarray) -> np.ndarray:
        """Parametrize the image in terms of the color path.

        Apply brute-force minimization to find the closest color representation
        on the path for each pixel in the image, and link to values.

        Args:
            colors: Input image to be interpreted.

        Returns:
            np.ndarray: Parametrization of the input image in terms of the color path.

        """
        # Fit in terms of the color path
        import time

        # Determine
        tic = time.time()
        if self.ignore_spectrum is None or colors.ndim == 1:
            parametrization = self.color_path.fit(
                colors=colors, color_mode=self.color_mode, mode="equidistant"
            )
        else:
            tic = time.time()
            color_mask = (
                np.linalg.norm(colors, axis=1) > 1e-1
            )  # self.ignore_spectrum.in_spectrum(colors, self.color_mode)
            parametrization = np.zeros(colors.shape[0])
            print(f"Color mask took {time.time() - tic:.2f} seconds.")
            tic = time.time()
            parametrization[color_mask] = self.color_path.fit(
                colors=colors[color_mask],
                color_mode=self.color_mode,
                mode="equidistant",
            )
            print(colors.shape, np.sum(color_mask))
            print(f"Fitting colors took {time.time() - tic:.2f} seconds.")
            tic = time.time()
            sub_colors = colors[color_mask]
            sub_parametrization = self.color_path.fit(
                colors=sub_colors,
                color_mode=self.color_mode,
                mode="equidistant",
            )
            parametrization[color_mask] = sub_parametrization
            print(colors.shape, np.sum(color_mask))
            print(f"Fitting colors took {time.time() - tic:.2f} seconds.")

        print(f"Fitting colors took {time.time() - tic:.2f} seconds.")

        # Linear interpolation based on values
        result = np.zeros_like(parametrization)
        for i in range(self.color_path.num_segments):
            if i == 0:
                mask = parametrization <= self.color_path.equidistant_distances[i + 1]
            elif i == self.color_path.num_segments - 1:
                mask = parametrization >= self.color_path.equidistant_distances[i]
            else:
                mask = np.logical_and(
                    parametrization >= self.color_path.equidistant_distances[i],
                    parametrization <= self.color_path.equidistant_distances[i + 1],
                )
            segment_distance = (
                parametrization[mask] - self.color_path.equidistant_distances[i]
            )

            segment_length = (
                self.color_path.equidistant_distances[i + 1]
                - self.color_path.equidistant_distances[i]
            )
            weight = segment_distance / segment_length

            result[mask] = self.values[i] + weight * (
                self.values[i + 1] - self.values[i]
            )

        return result
