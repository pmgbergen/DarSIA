"""Patchwise illumination correction module for images."""

from pathlib import Path

import cv2
import numpy as np

import darsia


class PatchwiseIlluminationCorrection(darsia.BaseCorrection):
    """Class for performing patchwise illumination correction on images."""

    def __init__(
        self,
        image: str | darsia.Image | None = None,
        baseline_images: list[str] | list[darsia.Image] | None = None,
        nw: int = 100,
        limit: int = 1450,
        eps: float = 1e-6,
        show_images: bool = True,
    ):
        """Initialize the PatchwiseIlluminationCorrection class.

        Args:
            image (str | darsia.Image): Input image for correction.
            baseline_images (list[str] | list[darsia.Image]): List of baseline images for
                correction.
            nw (int): Number of patches in width direction for patchwise illumination
                correction. Default is 100.
            limit (int): Limit in pixels to exclude from top of image for patch sampling.
                Default is 1450.
            eps (float): Small constant to avoid division by zero in patchwise illumination
                correction. Default is 1e-6.
            show_images (bool): Flag to control whether to display the calibrated image.

        """
        if image is None or baseline_images is None:
            self.r_diff = None
            self.g_diff = None
            self.b_diff = None
            return

        self.nw = nw
        self.limit = limit
        self.eps = eps

        if isinstance(image, str):
            self.img = cv2.imread(image)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            if self.img is None:
                raise ValueError(f"Image not found : {image}")
        else:
            self.img = image.img

        self.baseline_images = []
        for baseline in baseline_images:
            if isinstance(baseline, str):
                baseline = cv2.imread(baseline)
                baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2RGB)
                if baseline is None:
                    raise ValueError(f"Image not found : {baseline}")
            else:
                baseline = baseline.img
            self.baseline_images.append(baseline)

        n_baseline_images = len(self.baseline_images)

        self.height, self.width = self.img.shape[:2]
        self.nh = int((self.height - self.limit) * self.nw / self.width)
        self.dh = (self.height - self.limit) / self.nh
        self.dw = self.width / self.nw

        r, g, b = [], [], []
        r_mean, g_mean, b_mean = [], [], []

        for i in range(n_baseline_images):
            ri, gi, bi = self.extract_color_values_patches(
                self.baseline_images[i]
            )
            r.append(ri)
            g.append(gi)
            b.append(bi)
            r_m = np.mean(ri)
            g_m = np.mean(gi)
            b_m = np.mean(bi)
            r_mean.append(r_m)
            g_mean.append(g_m)
            b_mean.append(b_m)

        self.r_diff = self.compute_correction(r, r_mean)
        self.g_diff = self.compute_correction(g, g_mean)
        self.b_diff = self.compute_correction(b, b_mean)

        self.r_diff = self.extend_correction_coefficients(self.r_diff)
        self.g_diff = self.extend_correction_coefficients(self.g_diff)
        self.b_diff = self.extend_correction_coefficients(self.b_diff)

        image_calibrated = self.correct_array(self.img)

        if show_images:
            image_calibrated = cv2.cvtColor(image_calibrated, cv2.COLOR_RGB2BGR)
            cv2.imshow("calibrated image", image_calibrated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def extract_color_values_patches(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract RGB values from image patches.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple containing R, G, B matrices.
        """
        nh = self.nh
        limit = self.limit

        r = np.zeros((nh, self.nw), dtype=np.float32)
        g = np.zeros((nh, self.nw), dtype=np.float32)
        b = np.zeros((nh, self.nw), dtype=np.float32)

        for i in range(nh):
            y0 = max(limit + int(round((i - 0.5) * self.dh)), 0)
            y1 = min(limit + int(round((i + 0.5) * self.dh)), self.height)

            for j in range(self.nw):
                x0 = max(int(round((j - 0.5) * self.dw)), 0)
                x1 = min(int(round((j + 0.5) * self.dw)), self.width)

                roi = image[y0:y1, x0:x1]

                mean_color = cv2.mean(roi)
                
                # TODO: Consider what if colors are in int.
                r[i, j] = mean_color[0]
                g[i, j] = mean_color[1]
                b[i, j] = mean_color[2]

        return np.array(r), np.array(g), np.array(b)

    def compute_correction(
        self,
        coefficient_list: list[np.ndarray],
        coefficient_mean_list: list[np.ndarray],
    ) -> np.ndarray:
        """Calculate correction coefficients based on baseline images.

        Args:
            coefficient_list (list[np.ndarray]): List of coefficient matrices for each
                baseline image.
            coefficient_mean_list (list[np.ndarray]): List of mean coefficient values for
                each baseline image.

        Returns:
            np.ndarray: Array of correction coefficients.

        """

        sum_sq = np.sum([r**2 for r in coefficient_list], axis=0)

        correction = np.zeros_like(sum_sq, dtype=float)

        for r, r_m in zip(coefficient_list, coefficient_mean_list):
            weight = (r**2) / (sum_sq + self.eps)
            correction += weight * (r_m / (r + self.eps))

        return 1.0 / (correction + self.eps)

    def extend_correction_coefficients(self, corr: np.ndarray) -> np.ndarray:
        """Extend correction coefficients to the upper part of the image.

        Args:
            corr (np.ndarray): Array of correction coefficients for the lower part of the
                image.

        Returns:
            np.ndarray: Extended array of correction coefficients for the entire image.

        """
        new_corr = np.zeros((int(self.limit / self.dh), self.nw))
        lim = int(self.nh / 3)
        for col in range(self.nw):
            avg_top = np.mean(corr[:lim, col])
            new_corr[:, col] = avg_top
        full_corr = np.vstack((new_corr, corr))
        extended_corr = cv2.resize(full_corr.astype(np.float32), 
        (self.width, self.height),
        interpolation=cv2.INTER_LINEAR,
        )
        return extended_corr

    def correct_array(self, img: np.ndarray) -> np.ndarray:
        """Apply patchwise illumination correction to the input image.

        Args:
            img (np.ndarray): Input image to be corrected.

        Returns:
            np.ndarray: Corrected image after applying patchwise illumination correction.

        """
        if self.r_diff is None or self.g_diff is None or self.b_diff is None:
            raise ValueError(
                "Correction coefficients are not initialized. Please provide baseline "
                "images to compute correction coefficients."
            )

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        r_new = r / self.r_diff
        g_new = g / self.g_diff
        b_new = b / self.b_diff

        image_calib = cv2.merge(
            (r_new, g_new, b_new)
        )

        return image_calib

    def save(self, path: Path) -> None:
        """Save correction coefficients to a file.

        Args:
            path (Path): Path to the file where coefficients will be saved.

        """
        np.savez(
            path,
            class_name=type(self).__name__,
            correction={
                "r_diff": self.r_diff,
                "g_diff": self.g_diff,
                "b_diff": self.b_diff,
                "nh": self.nh,
                "nw": self.nw,
                "limit": self.limit,
                "eps": self.eps,
                "dh": self.dh,
                "dw": self.dw,
                "height": self.height,
                "width": self.width,

            },
        )
        print(f"Correction coefficients saved to {path}")

    def load(self, path: Path) -> None:
        """Load correction coefficients from a file.

        Args:
            path (Path): Path to the file from which coefficients will be loaded.

        """
        data = np.load(path, allow_pickle=True)["correction"].item()
        if "r_diff" not in data or "g_diff" not in data or "b_diff" not in data:
            raise ValueError(
                f"Invalid correction coefficients file: {path}. "
                "Expected keys 'r_diff', 'g_diff', and 'b_diff' not found."
            )
        self.r_diff = data["r_diff"]
        self.g_diff = data["g_diff"]
        self.b_diff = data["b_diff"]
        self.nh = data["nh"]
        self.nw = data["nw"]
        self.limit = data["limit"]
        self.eps = data["eps"]
        self.dh = data["dh"]
        self.dw = data["dw"]
        self.height = data["height"]
        self.width = data["width"]
        print(f"Correction coefficients loaded from {path}")
