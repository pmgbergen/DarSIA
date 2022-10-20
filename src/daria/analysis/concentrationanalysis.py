"""
Module that contains a class which provides the capabilities to
analyze concentrations/saturation profiles based on image comparison.
"""

import copy
import json
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import skimage
from scipy.optimize import bisect
from sklearn.linear_model import RANSACRegressor

import daria


class ConcentrationAnalysis:
    """
    Class providing the capabilities to determine concentration/saturation
    profiles based on image comparison, and tuning of concentration-intensity
    maps.
    """

    # ! ---- Setter methods

    def __init__(
        self,
        base: Union[daria.Image, list[daria.Image]],
        color: Union[str, callable] = "gray",
    ) -> None:
        """
        Constructor of ConcentrationAnalysis.

        Args:
            base (daria.Image or list of such): baseline image(s); if multiple provided,
                these are used to define a cleaning filter.
            color (string or callable): "gray", "red", "blue", "green", "hue", "saturation",
                "value" identifying which monochromatic space should be used for the
                analysis; tailored routine can also be provided.
        """
        # Define mono-colored space
        self.color: Union[str, callable] = (
            color.lower() if isinstance(color, str) else color
        )

        # Extract mono-colored version for baseline image
        if not isinstance(base, list):
            base = [base]
        self.base: daria.Image = base[0].copy()
        self._extract_scalar_information(self.base)

        # Initialize conversion parameters - if mulitple baseline images are provided,
        # define a cleaning filter.
        self.scaling: float = 1.0
        self.offset: float = 0.0
        self.threshold: np.ndarray[float] = np.zeros(
            self.base.img.shape[:2], dtype=float
        )
        if len(base) > 1:
            self.find_cleaning_filter(base)

        # Initialize cache for effective volumes (per pixel). It is assumed that
        # the physical asset does not change, and that simply different versions
        # of the same volumes have to be stored, differing by the shape only.
        # Start with default value, assuming constant volume per pixel.
        self._volumes_cache: Union[float, dict] = 1.0
        self._volumes_are_constant = True

    def update(
        self,
        base: Optional[daria.Image] = None,
        scaling: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> None:
        """
        Update of the baseline image or parameters.

        Args:
            base (daria.Image, optional): image array
            scaling (float, optional): slope
            offset (float, optional): offset
        """
        if base is not None:
            self.base = base.copy()
            self._extract_scalar_information(self.base)
        if scaling is not None:
            self.scaling = scaling
        if offset is not None:
            self.offset = offset

    def update_volumes(self, volumes: Union[float, np.ndarray]) -> None:
        """
        Clear the cache and redefine some reference of the effective pixel volumes.

        Args:
            volumes (float or array): effective pixel volume per pixel
        """
        self._volumes_are_constant = isinstance(volumes, float)
        if self._volumes_are_constant:
            self._volumes_cache = volumes
        else:
            self._volumes_ref_shape = np.squeeze(volumes).shape
            self._volumes_cache = {
                self._volumes_ref_shape: np.squeeze(volumes),
            }

    def read_calibration_from_file(self, config: dict, path: Union[str, Path]) -> None:
        """
        Read calibration information from file.

        Args:
            config (dict): config file for simple data.
            path (str or Path): path to cleaning filter array.
        """
        # Fetch the scaling parameter and threshold mask from file
        self.scaling = config["scaling"]
        self.threshold = np.load(path)

        # Resize threshold mask if unmatching the size of the base image
        base_shape = self.base.img.shape[:2]
        if self.threshold.shape[:2] != base_shape:
            self.threshold = cv2.resize(self.threshold, tuple(reversed(base_shape)))

    def write_calibration_to_file(
        self, path_to_config: Union[str, Path], path_to_filter: Union[str, Path]
    ) -> None:
        """
        Store available calibration information including the
        scaling factor and the cleaning mask.

        Args:
            path_to_config (str or Path): path for storage of config file.
            path_to_filter (str or Path): path for storage of the cleaning filter.
        """
        # Store scaling parameters.
        config = {
            "scaling": self.scaling,
        }
        with open(Path(path_to_config), "w") as outfile:
            json.dump(config, outfile, indent=4)

        # Store cleaning filter array.
        np.save(str(Path(path_to_filter)), self.threshold)

    # ! ---- Main method

    def __call__(self, img: daria.Image) -> daria.Image:
        """Extract concentration based on a reference image and rescaling.

        Args:
            img (daria.Image): probing image

        Returns:
            daria.Image: concentration
        """
        probe_img = copy.deepcopy(img)

        # Extract monochromatic version and take difference wrt the baseline image
        # If requested by the user, extract a monochromatic version before.
        self._extract_scalar_information(probe_img)
        diff = skimage.util.compare_images(probe_img.img, self.base.img, method="diff")
        signal = self._extract_scalar_information_after(diff)

        # Clean signal
        clean_signal = np.clip(signal - self.threshold, 0, None)

        # Post-process the signal
        processed_signal = self.postprocess_signal(clean_signal)

        # Convert from signal to concentration
        concentration = np.clip(self.scaling * processed_signal + self.offset, 0, 1)

        return daria.Image(concentration, img.metadata)

    # ! ---- Pre- and post-processing methods

    def _extract_scalar_information(self, img: daria.Image) -> None:
        """
        Make a mono-colored image from potentially multi-colored image.

        Args:
            img (daria.Image): image
        """
        if self.color == "gray":
            img.toGray()
        elif self.color == "red":
            img.toRed()
        elif self.color == "green":
            img.toGreen()
        elif self.color == "blue":
            img.toBlue()
        elif self.color == "hue":
            img.toHue()
        elif self.color == "saturation":
            img.toSaturation()
        elif self.color == "value":
            img.toValue()
        elif callable(self.color):
            img.img = self.color(img.img)
        else:
            raise ValueError(f"Mono-colored space {self.color} not supported.")

    def _extract_scalar_information_after(self, img: np.ndarray) -> None:
        """
        Make a mono-colored image from potentially multi-colored image.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: monochromatic reduction of the array
        """
        return img

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Empty postprocessing - should be overwritten for practical cases.

        Example for a postprocessing using some noise removal.
        return skimage.restoration.denoise_tv_chambolle(signal, weight=0.1)
        """
        return signal

    # ! ---- Calibration tools for signal to concentration conversion

    def find_cleaning_filter(
        self, baseline_images: list[daria.Image], reset: bool = False
    ) -> None:
        """
        Determine natural noise by studying a series of baseline images.
        The resulting cleaning filter will be used prior to the conversion
        of signal to concentration. The cleaning filter should be understood
        as thresholding mask.

        Args:
            baseline_images (list of daria.Image): series of baseline_images.
            reset (bool): flag whether the cleaning filter shall be reset.
        """
        # Initialize cleaning filter
        if reset:
            self.threshold = np.zeros_like(self.base, dtype=float)

        # Combine the results of a series of images
        for i, img in enumerate(baseline_images):

            # Extract mono-colored version
            probe_img = img.copy()
            self._extract_scalar_information(probe_img)

            # Take (unsigned) difference
            diff = skimage.util.compare_images(
                probe_img.img, self.base.img, method="diff"
            )

            # Consider elementwise max
            self.threshold = np.maximum(self.threshold, diff)

    def calibrate(
        self,
        injection_rate: float,
        images: list[daria.Image],
        initial_guess: Optional[tuple[float]] = None,
        tol: float = 1e-3,
        maxiter: int = 20,
    ) -> None:
        """
        Calibrate the conversion used in __call__ such that the provided
        injection rate is matched for the given set of images.

        Args:
            injection_rate (float): constant injection rate in ml/hrs.
            images (list of daria.Image): images used for the calibration.
            initial_guess (tuple): interval of scaling values to be considered
                in the calibration; need to define lower and upper bounds on
                the optimal scaling parameter.
            tol (float): tolerance for the bisection algorithm.
            maxiter (int): maximal number of bisection iterations used for
                calibration.
        """

        # Define a function which is zero when the conversion parameters are chosen properly.
        def deviation(scaling: float):
            self.scaling = scaling
            # self.offset = None
            return injection_rate - self._estimate_rate(images)[0]

        # Perform bisection
        self.scaling = bisect(deviation, *initial_guess, xtol=tol, maxiter=maxiter)

        print(f"Calibration results in scaling factor {self.scaling}.")

    def _estimate_rate(self, images: list[daria.Image]) -> tuple[float]:
        """
        Estimate the injection rate for the given series of images.

        Args:
            images (list of daria.Image): basis for computing the injection rate.

        Returns:
            float: estimated injection rate.
            float: offset at time 0, useful to determine the actual start time,
                or plot the total concentration over time compared to the expected
                volumes.
        """
        # Conversion constants
        SECONDS_TO_HOURS = 1.0 / 3600.0
        M3_TO_ML = 1e6

        # Define reference time (not important which image serves as basis)
        ref_time = images[0].timestamp

        # For each image, compute the total concentration, based on the currently
        # set tuning parameters, and compute the relative time.
        total_volumes = []
        relative_times = []
        for img in images:

            # Fetch associated time for image, relate to reference time, and store.
            time = img.timestamp
            relative_time = (time - ref_time).total_seconds() * SECONDS_TO_HOURS
            relative_times.append(relative_time)

            # Convert signal image to concentration, compute the total volumetric
            # concentration in ml, and store.
            concentration = self(img)
            total_volume = self._determine_total_volume(concentration) * M3_TO_ML
            total_volumes.append(total_volume)

        # Determine slope in time by linear regression
        relative_times = np.array(relative_times).reshape(-1, 1)
        total_volumes = np.array(total_volumes)
        ransac = RANSACRegressor()
        ransac.fit(relative_times, total_volumes)

        # Extract the slope and convert to
        return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

    def _determine_total_volume(self, concentration: daria.Image) -> float:
        """
        Determine the total concentration of a spatial concentration map.

        Args:
            concentration (daria.Image): concentration data.

        Returns:
            float: The integral over the spatial, weighted concentration map.
        """
        # Fetch pixel volumes from cache, possibly require reshaping if the dimensions
        # do not match.
        if self._volumes_are_constant:
            # Just fetch the constant volume
            volumes = self._volumes_cache
        else:
            shape = np.squeeze(concentration.img).shape
            if shape in self._volumes_cache:
                # Fetch previously cached volume
                volumes = self._volumes_cache[shape]
            else:
                # Need to resize and rescale (to ensure volume conservation).
                # Use the reference volume for all such operations.
                ref_volumes = self._volumes_cache[self._volumes_ref_shape]
                new_shape = tuple(reversed(shape))
                volumes = cv2.resize(
                    ref_volumes, new_shape, interpolation=cv2.INTER_AREA
                )
                volumes *= np.sum(ref_volumes) / np.sum(volumes)
                self._volumes_cache[shape] = volumes

        # Integral of locally weighted concentration values
        return np.sum(np.multiply(np.squeeze(volumes), np.squeeze(concentration.img)))


class BinaryConcentrationAnalysis(ConcentrationAnalysis):
    """
    Special case of ConcentrationAnalysis which generates boolean concentration profiles.
    """

    def __init__(
        self,
        base: Union[daria.Image, list[daria.Image]],
        color: Union[str, callable] = "gray",
        **kwargs,
    ) -> None:
        """
        Constructor for BinaryConcentrationAnalysis.

        Calls the constructor of parent class and fetches
        all tuning parameters for the binary segmentation.

        Args:
            base (daria.Image or list of such): same as in ConcentrationAnalysis.
            color (string or callable): same as in ConcentrationAnalysis.
            kwargs (keyword arguments): interface to all tuning parameters
        """
        super().__init__(base, color)

        # TVD parameters for pre and post smoothing
        self.apply_presmoothing = kwargs.pop("presmoothing", False)
        if self.apply_presmoothing:
            self.presmoothing = {
                "resize": kwargs.pop("presmoothing resize", 1.0),
                "weight": kwargs.pop("presmoothing weight", 1.0),
                "eps": kwargs.pop("presmoothing eps", 1e-5),
                "max_num_iter": kwargs.pop("presmoothing max_num_iter", 1000),
                "method": kwargs.pop("presmoothing method", "chambolle"),
            }

        self.apply_postsmoothing = kwargs.pop("postsmoothing", False)
        if self.apply_postsmoothing:
            self.postsmoothing = {
                "resize": kwargs.pop("postsmoothing resize", 1.0),
                "weight": kwargs.pop("postsmoothing weight", 1.0),
                "eps": kwargs.pop("postsmoothing eps", 1e-5),
                "max_num_iter": kwargs.pop("postsmoothing max_num_iter", 1000),
                "method": kwargs.pop("postsmoothing method", "chambolle"),
            }

        # Thresholding parameters
        self.apply_automatic_threshold: bool = kwargs.pop("threshold auto", False)
        if not self.apply_automatic_threshold:
            self.threshold_value: float = kwargs.pop("threshold value")

        # Parameters to fill holes
        self.area_threshold: int = kwargs.pop("max hole size", 10**2)

        # Parameters for local convex cover
        self.cover_patch_size: int = kwargs.pop("local convex cover patch size", 1)

        # Active ROI
        self.active_roi: np.ndarray = np.ones(self.base.img.shape[:2], dtype=bool)

    def update_active_roi(self, active_roi: np.ndarray) -> None:
        """
        Update the active ROI to be considered in the analysis.
        """
        self.active_roi = active_roi

    def gradient_modulus(self, img, area) -> float:
        dx = daria.forward_diff_x(img)
        dy = daria.forward_diff_y(img)
        dx[np.logical_not(area)] = 0
        dy[np.logical_not(area)] = 0
        grad_mod = np.sqrt(dx**2 + dy**2)

        return grad_mod

    # ! ---- Main method
    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Postprocessing routine, essentially converting a continuous
        signal into a binary concentration and thereby segmentation.
        The post processing consists of presmoothing, thresholding,
        filling holes, local convex covering, and postsmoothing.
        Tuning parameters for this routine have to be set in the
        initialization routine.

        Args:
            signal (np.ndarray): clean continous signal with values
                in the range between 0 and 1.

        Returns:
            np.ndarray: binary concentration
        """
        # Deactive signal in non-active ROI
        signal[np.logical_not(self.active_roi)] = 0

        # Apply presmoothing
        if self.apply_presmoothing:
            # Resize image
            signal = cv2.resize(
                signal.astype(np.float32),
                None,
                fx=self.presmoothing["resize"],
                fy=self.presmoothing["resize"],
            )

            # Apply TVD
            if self.presmoothing["method"] == "chambolle":
                signal = skimage.restoration.denoise_tv_chambolle(
                    signal,
                    weight=self.presmoothing["weight"],
                    eps=self.presmoothing["eps"],
                    max_num_iter=self.presmoothing["max_num_iter"],
                )
            elif self.presmoothing["method"] == "anisotropic bregman":
                signal = skimage.restoration.denoise_tv_bregman(
                    signal,
                    weight=self.presmoothing["weight"],
                    eps=self.presmoothing["eps"],
                    max_num_iter=self.presmoothing["max_num_iter"],
                    isotropic=False,
                )
            else:
                raise ValueError(f"Method {self.presmoothing['method']} not supported.")

            # Resize to original size
            signal = cv2.resize(signal, tuple(reversed(self.base.img.shape[:2])))

        # Apply thresholding to obtain mask
        if self.apply_automatic_threshold:
            # Extract merely signal values in the active ROI
            active_signal_values = np.ravel(signal)[np.ravel(self.active_roi)]
            # Find automatic threshold value using OTSU
            thresh = skimage.filters.threshold_otsu(active_signal_values)
        else:
            # Fetch user-defined threshold value
            thresh = self.threshold_value
        mask = signal > thresh

        # Fill holes
        mask = skimage.morphology.remove_small_holes(
            mask, area_threshold=self.area_threshold
        )

        # Loop through patches and fill up
        covered_mask = np.zeros(mask.shape[:2], dtype=bool)
        size = self.cover_patch_size
        Ny, Nx = mask.shape[:2]
        for row in range(int(Ny / size)):
            for col in range(int(Nx / size)):
                roi = (
                    slice(row * size, (row + 1) * size),
                    slice(col * size, (col + 1) * size),
                )
                covered_mask[roi] = skimage.morphology.convex_hull_image(mask[roi])

        # Update the mask value
        mask = covered_mask

        # Apply postsmoothing
        if self.apply_postsmoothing:
            # Resize image
            resized_mask = cv2.resize(
                covered_mask.astype(np.float32),
                None,
                fx=self.postsmoothing["resize"],
                fy=self.postsmoothing["resize"],
            )

            # Apply TVD
            if self.postsmoothing["method"] == "chambolle":
                smoothed_mask = skimage.restoration.denoise_tv_chambolle(
                    resized_mask,
                    weight=self.postsmoothing["weight"],
                    eps=self.postsmoothing["eps"],
                    max_num_iter=self.postsmoothing["max_num_iter"],
                )
            elif self.postsmoothing["method"] == "anisotropic bregman":
                smoothed_mask = skimage.restoration.denoise_tv_bregman(
                    resized_mask,
                    weight=self.postsmoothing["weight"],
                    eps=self.postsmoothing["eps"],
                    max_num_iter=self.postsmoothing["max_num_iter"],
                    isotropic=False,
                )
            else:
                raise ValueError(
                    f"Method {self.postsmoothing['method']} is not supported."
                )

            # Resize to original size
            large_mask = cv2.resize(
                smoothed_mask.astype(np.float32),
                tuple(reversed(self.base.img.shape[:2])),
            )

            # Apply hardcoded threshold value of 0.5 assuming it is sufficient to turn
            # off small particles and rely on larger marked regions
            thresh = (
                0.5
                if self.postsmoothing["method"] == "chambolle"
                else skimage.filters.threshold_otsu(smoothed_mask)
            )

        final_mask = mask

        return final_mask

    def convert_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Pass the signal, assuming it identifies a binary concentration.

        Args:
            signal (np.ndarray): signal array.

        Returns:
            np.ndarray: signal array.
        """
        return signal
