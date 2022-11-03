"""
Module that contains a class which provides the capabilities to
analyze concentrations/saturation profiles based on image comparison.
"""

import copy
import json
from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
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
        raise PendingDeprecationWarning(
            "Routine deprecated, and will be removed in the future. Instead read scaling"
            "parameter from config and use read_cleaning_filter_from_file()."
        )

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
        raise PendingDeprecationWarning(
            "Routine deprecated, and will be removed in the future. Instead use"
            "write_cleaning_filter_to_file()."
        )

        # Store scaling parameters.
        config = {
            "scaling": self.scaling,
        }
        with open(Path(path_to_config), "w") as outfile:
            json.dump(config, outfile, indent=4)

        # Store cleaning filter array - create folder if does not exist.
        path_to_filter = Path(path_to_filter)
        path_to_filter.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(path_to_filter, self.threshold)

    def read_cleaning_filter_from_file(self, path: Union[str, Path]) -> None:
        """
        Read cleaning filter from file.

        Args:
            path (str or Path): path to cleaning filter array.
        """
        # Fetch the threshold mask from file
        self.threshold = np.load(path)

        # Resize threshold mask if unmatching the size of the base image
        base_shape = self.base.img.shape[:2]
        if self.threshold.shape[:2] != base_shape:
            self.threshold = cv2.resize(self.threshold, tuple(reversed(base_shape)))

    def write_cleaning_filter_to_file(self, path_to_filter: Union[str, Path]) -> None:
        """
        Store cleaning filter to file.

        Args:
            path_to_filter (str or Path): path for storage of the cleaning filter.
        """
        path_to_filter = Path(path_to_filter)
        path_to_filter.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(path_to_filter, self.threshold)

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

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_scalar(probe_img.img)

        diff = skimage.util.compare_images(probe_img.img, self.base.img, method="diff")

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_diff(diff)

        signal = self._extract_scalar_information_after(diff)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_signal(signal)

        # Clean signal
        clean_signal = np.clip(signal - self.threshold, 0, None)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_clean_signal(clean_signal)

        # Post-process the signal
        processed_signal = self.postprocess_signal(clean_signal)

        # Convert from signal to concentration
        concentration = self.convert_signal(processed_signal)

        return daria.Image(concentration, img.metadata)

    # ! ---- Pre- and post-processing methods
    def _inspect_scalar(self, img):
        pass

    def _inspect_diff(self, img):
        pass

    def _inspect_signal(self, img):
        pass

    def _inspect_clean_signal(self, img):
        pass

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

    def convert_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert signal to concentration. This requires in general some calibration.
        Here, homogeneous scaling is assumed. In practical situations this
        routine may be overwritten.

        Args:
            signal (np.ndarray): signal array.

        Returns:
            np.ndarray: concentration array.
        """
        return np.clip(self.scaling * signal + self.offset, 0, 1)

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
            self.threshold = np.zeros(self.base.img.shape[:2], dtype=float)

        # Combine the results of a series of images
        for i, img in enumerate(baseline_images):

            probe_img = img.copy()

            # Extract mono-colored version in case of a monochromatic comparison
            self._extract_scalar_information(probe_img)

            # Take (unsigned) difference
            diff = skimage.util.compare_images(
                probe_img.img, self.base.img, method="diff"
            )

            # Extract mono-colored version (should be only active in case of a
            # multichromatic comparison
            monochromatic_diff = self._extract_scalar_information_after(diff)

            # Consider elementwise max
            self.threshold = np.maximum(self.threshold, monochromatic_diff)

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
            self.threshold_value: float = kwargs.pop("threshold value", 0.0)

        # Parameters to remove small objects
        self.min_size: int = kwargs.pop("min area size", 1)

        # Parameters to fill holes
        self.area_threshold: int = kwargs.pop("max hole size", 0)

        # Parameters for local convex cover
        self.cover_patch_size: int = kwargs.pop("local convex cover patch size", 1)

        # Threshold for posterior analysis based on gradient moduli
        self.apply_posterior = kwargs.pop("posterior", False)
        if self.apply_posterior:
            self.threshold_posterior: float = kwargs.pop(
                "threshold posterior gradient modulus"
            )

        # Mask
        self.mask: np.ndarray = np.ones(self.base.img.shape[:2], dtype=bool)

        # Fetch verbosity. If True, several intermediate results in the
        # postprocessing will be displayed. This allows for simpler tuning
        # of the parameters.
        self.verbosity = kwargs.pop("verbosity", False)

    def update_mask(self, mask: np.ndarray) -> None:
        """
        Update the mask to be considered in the analysis.

        Args:
            mask (np.ndarray): boolean mask, detecting which pixels will
                be considered, all other will be ignored in the analysis.
        """
        self.mask = mask

    # ! ---- Main methods
    def _prior(self, signal: np.ndarray) -> np.ndarray:
        """
        Prior postprocessing routine, essentially converting a continuous
        signal into a binary concentration and thereby segmentation.
        The post processing consists of presmoothing, thresholding,
        filling holes, local convex covering, and postsmoothing.
        Tuning parameters for this routine have to be set in the
        initialization routine.

        In addition, the determined segments are checked. Only segments are
        marked for which the signal has a sufficiently large gradient
        modulus.

        Args:
            signal (np.ndarray): signal

        Returns:
            np.ndarray: prior binary mask
        """

        if self.verbosity:
            plt.figure("Prior: Input signal")
            plt.imshow(signal)

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

        if self.verbosity:
            plt.figure("Prior: TVD smoothed signal")
            plt.imshow(signal)

        # Cache the (smooth) signal for output
        smooth_signal = np.copy(signal)

        if self.verbosity:
            # Extract merely signal values in the mask
            active_signal_values = np.ravel(signal)[np.ravel(self.mask)]
            # Find automatic threshold value using OTSU
            thresh = skimage.filters.threshold_otsu(active_signal_values)
            print("OTSU threshold value", thresh)

        # Apply thresholding to obtain mask
        if self.apply_automatic_threshold:
            # Extract merely signal values in the mask
            active_signal_values = np.ravel(signal)[np.ravel(self.mask)]
            # Find automatic threshold value using OTSU
            thresh = skimage.filters.threshold_otsu(active_signal_values)
        else:
            # Fetch user-defined threshold value
            thresh = self.threshold_value
        mask = signal > thresh

        if self.verbosity:
            plt.figure("Prior: Thresholded mask")
            plt.imshow(mask)

        # Remove small objects
        if self.min_size > 1:
            mask = skimage.morphology.remove_small_objects(mask, min_size=self.min_size)

        # Fill holes
        if self.area_threshold > 0:
            mask = skimage.morphology.remove_small_holes(
                mask, area_threshold=self.area_threshold
            )

        if self.verbosity:
            plt.figure("Prior: Cleaned mask")
            plt.imshow(mask)

        # Loop through patches and fill up
        if self.cover_patch_size > 1:
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

        if self.verbosity:
            plt.figure("Prior: Locally covered mask")
            plt.imshow(mask)

        # Apply postsmoothing
        if self.apply_postsmoothing:
            # Resize image
            resized_mask = cv2.resize(
                mask.astype(np.float32),
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
                else skimage.filters.threshold_otsu(large_mask)
            )
            mask = large_mask > thresh

        if self.verbosity:
            plt.figure("Prior: TVD postsmoothed mask")
            plt.imshow(mask)

        # Finaly cleaning - inactive signal outside mask
        mask[~self.mask] = 0

        if self.verbosity:
            plt.figure("Prior: Final mask after cleaning")
            plt.imshow(mask)
            plt.show()

        return mask, smooth_signal

    def _posterior(self, signal: np.ndarray, mask_prior: np.ndarray) -> np.ndarray:
        """
        Posterior analysis of signal, determining the gradients of
        for marked regions.

        Args:
            signal (np.ndarray): (smoothed) signal
            mask_prior (np.ndarray): boolean mask marking prior regions

        Return:
            np.ndarray: boolean mask of trusted regions.
        """
        # Only continue if necessary
        if not self.apply_posterior:
            return np.ones(signal.shape[:2], dtype=bool)

        # Determien gradient modulus of the smoothed signal
        dx = daria.forward_diff_x(signal)
        dy = daria.forward_diff_y(signal)
        gradient_modulus = np.sqrt(dx**2 + dy**2)

        if self.verbosity:
            plt.figure("Posterior: Gradient modulus")
            plt.imshow(gradient_modulus)

        # Extract concentration map
        mask_posterior = np.zeros(signal.shape, dtype=bool)

        # Label the connected regions first
        labels_prior, num_labels_prior = skimage.measure.label(
            mask_prior, return_num=True
        )

        if self.verbosity:
            plt.figure("Posterior: Labeled regions from prior")
            plt.imshow(labels_prior)
            plt.show()

        # Investigate each labeled region separately; omit label 0, which corresponds
        # to non-marked area.
        for label in range(1, num_labels_prior + 1):

            # Fix one label
            labeled_region = labels_prior == label

            # Determine contour set of labeled region
            contours, _ = cv2.findContours(
                skimage.img_as_ubyte(labeled_region),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # For each part of the contour set, check whether the gradient is sufficiently
            # large at any location
            accept = False
            for c in contours:

                # Extract coordinates of contours - have to flip columns, since cv2 provides
                # reverse matrix indexing, and also 3 components, with the second one
                # single dimensioned.
                c = (c[:, 0, 1], c[:, 0, 0])

                # Identify region as marked if gradient sufficiently large
                if np.max(gradient_modulus[c]) > self.threshold_posterior:
                    accept = True
                    break
            # Collect findings
            if accept:
                mask_posterior[labeled_region] = True

        return mask_posterior

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
        # Determine prior and posterior
        mask_prior, smooth_signal = self._prior(signal)
        mask_posterior = self._posterior(smooth_signal, mask_prior)

        # NOTE: Here the overlay process is obsolete, posterior is active.
        # Yet, it allows to overwrite posterior by inheritance and design
        # other schemes.

        # Overlay prior and posterior
        mask = np.zeros(mask_prior.shape, dtype=bool)
        labels_prior, num_labels_prior = skimage.measure.label(
            mask_prior, return_num=True
        )

        for label in range(1, num_labels_prior + 1):

            # Fix one label
            labeled_region = labels_prior == label

            # Check whether posterior marked in this area
            if np.any(mask_posterior[labeled_region]):
                mask[labeled_region] = True

        return mask

    def convert_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Pass the signal, assuming it identifies a binary concentration.

        Args:
            signal (np.ndarray): signal array.

        Returns:
            np.ndarray: signal array.
        """
        return signal
