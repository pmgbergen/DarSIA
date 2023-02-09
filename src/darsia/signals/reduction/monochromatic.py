"""
Module providing dimension reductions to monochromatic/scalar signals.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class MonochromaticReduction(darsia.SignalReduction):
    def __init__(self, **kwargs) -> None:
        self.color = kwargs.get("color", "gray")

        if self.color in ["hsv-after"]:
            self.hue_lower_bound = kwargs.get("hue lower bound", 0.0)
            self.hue_upper_bound = kwargs.get("hue upper bound", 360.0)
            self.saturation_lower_bound = kwargs.get("saturation lower bound", 0.0)
            self.saturation_upper_bound = kwargs.get("saturation upper bound", 1.0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Make a mono-colored image from potentially multi-colored image.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: monochromatic reduction of the array
        """
        if self.color == "hsv-after":

            hsv = skimage.color.rgb2hsv(img)

            # Plot Hue and Saturation channels, allowing to manually tune
            # the concentration analysis.
            if self.verbosity >= 3:
                plt.figure("hue")
                plt.imshow(hsv[:, :, 0])
                plt.figure("saturation")
                plt.imshow(hsv[:, :, 1])

            # Restrict to user-defined thresholded hue and saturation values.
            mask_hue = np.logical_and(
                hsv[:, :, 0] > self.hue_lower_bound,
                hsv[:, :, 0] < self.hue_upper_bound,
            )
            mask_saturation = np.logical_and(
                hsv[:, :, 1] > self.saturation_lower_bound,
                hsv[:, :, 1] < self.saturation_upper_bound,
            )
            mask = np.logical_and(mask_hue, mask_saturation)

            # Consider value
            img_v = hsv[:, :, 2]
            img_v[~mask] = 0
            return img_v

        elif self.color == "gray":
            # Assume RGB input. NOTE: Make sure that the input is in correct
            # format (CV2 requires np.float32).
            return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)

        elif self.color == "red":
            return img[:, :, 0]

        elif self.color == "green":
            return img[:, :, 1]

        elif self.color == "blue":
            return img[:, :, 2]

        elif self.color == "red+green":
            return img[:, :, 0] + img[:, :, 1]

        elif self.color == "negative-key":
            cmy = 1 - img
            key = np.min(cmy, axis=2)
            return 1 - key

        elif callable(self.color):
            return self.color(img)

        elif self.color == "":
            return img

        else:
            raise ValueError(f"Mono-colored space {self.color} not supported.")
