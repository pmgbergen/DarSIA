"""Module to assist in the analysis of color spectra.

"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def hsv_spectrum(img: np.ndarray, roi: list[tuple], bins: int = 100) -> None:
    """
    Plot histograms for all HSV components present in a ROI of an image.

    Args:
        img (np.ndarray): image array in RGB space with matrix indexing
        roi (tuple of slices): slice for y-components, and slice for x-components,
            defining a region of interest, to be cropped from img
    """
    if isinstance(roi, tuple):
        roi = [roi]

    for i, r in enumerate(roi):
        # Retrict to ROI
        img_roi = img[r]

        # Extract H, S, V components
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_RGB2HSV)
        h_img = hsv[:, :, 0]
        s_img = hsv[:, :, 1]
        v_img = hsv[:, :, 2]

        # Extract values
        h_values = np.linspace(np.min(h_img), np.max(h_img), bins)
        s_values = np.linspace(np.min(s_img), np.max(s_img), bins)
        v_values = np.linspace(np.min(v_img), np.max(v_img), bins)

        # Setup histograms
        h_hist = np.histogram(h_img, bins=bins)[0]
        s_hist = np.histogram(s_img, bins=bins)[0]
        v_hist = np.histogram(v_img, bins=bins)[0]

        # Plot
        plt.figure(f"h {i}")
        plt.plot(h_values, h_hist)
        plt.figure(f"s {i}")
        plt.plot(s_values, s_hist)
        plt.figure(f"v {i}")
        plt.plot(v_values, v_hist)

    plt.show()
