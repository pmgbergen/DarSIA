"""
Module that contains a class which provides the capabilities to
analyze concentrations/saturation profiles based on image comparison.
"""

import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional, Union, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sps
import skimage
from scipy.optimize import bisect
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor

import darsia


class ConcentrationAnalysis:
    """
    Class providing the capabilities to determine concentration/saturation
    profiles based on image comparison, and tuning of concentration-intensity
    maps.
    """

    # ! ---- Setter methods

    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        color: Union[str, Callable] = "gray",
        **kwargs,
    ) -> None:
        """
        Constructor of ConcentrationAnalysis.

        Args:
            base (darsia.Image or list of such): baseline image(s); if multiple provided,
                these are used to define a cleaning filter.
            color (string or Callable): "gray", "red", "blue", "green", "hsv-after",
                "negative-key", identifying which monochromatic space should be used
                for the analysis; a tailored routine can also be provided.
            kwargs (keyword arguments): interface to all tuning parameters
        """
        # Define mono-colored space
        self.color: Union[str, Callable] = (
            color.lower() if isinstance(color, str) else color
        )

        # Extra args for hsv color which here is meant has hue thresholded value
        if self.color in ["hsv-after"]:
            self.hue_lower_bound = kwargs.pop("hue lower bound", 0.0)
            self.hue_upper_bound = kwargs.pop("hue upper bound", 360.0)
            self.saturation_lower_bound = kwargs.pop("saturation lower bound", 0.0)
            self.saturation_upper_bound = kwargs.pop("saturation upper bound", 1.0)

        # Extract mono-colored version for baseline image
        if not isinstance(base, list):
            base = [base]
        self.base: darsia.Image = base[0].copy()

        # Initialize conversion parameters - if mulitple baseline images are provided,
        # define a cleaning filter.
        self.scaling: float = 1.0
        self.offset: float = 0.0
        self.threshold: np.ndarray = np.zeros(self.base.img.shape[:2], dtype=float)
        if len(base) > 1:
            self.find_cleaning_filter(base)

        # Initialize cache for effective volumes (per pixel). It is assumed that
        # the physical asset does not change, and that simply different versions
        # of the same volumes have to be stored, differing by the shape only.
        # Start with default value, assuming constant volume per pixel.
        self._volumes_cache: Union[float, dict] = 1.0
        self._volumes_are_constant = True

        # Option for defining differences of images.
        self._diff_option = kwargs.pop("diff option", "absolute")

    def update(
        self,
        base: Optional[darsia.Image] = None,
        scaling: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> None:
        """
        Update of the baseline image or parameters.

        Args:
            base (darsia.Image, optional): image array
            scaling (float, optional): slope
            offset (float, optional): offset
        """
        if base is not None:
            self.base = base.copy()
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
            self._volumes_cache = cast(float, volumes)
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

        # Store cleaning filter array.
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

    def __call__(self, img: darsia.Image) -> darsia.Image:
        """Extract concentration based on a reference image and rescaling.

        Args:
            img (darsia.Image): probing image

        Returns:
            darsia.Image: concentration
        """
        probe_img = copy.deepcopy(img)

        # Remove background image
        diff = self._subtract_background(probe_img)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_diff(diff)

        # Extract monochromatic version and take difference wrt the baseline image
        signal = self._extract_scalar_information(diff)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_signal(signal)

        # Clean signal
        clean_signal = np.clip(signal - self.threshold, 0, None)

        # Provide possibility for tuning and inspection of intermediate results
        self._inspect_clean_signal(clean_signal)

        # Homogenize signal (take into account possible heterogeneous effects)
        homogenized_signal = self._homogenize_signal(clean_signal)

        # Post-process the signal
        processed_signal = self.postprocess_signal(homogenized_signal, diff)

        # Convert from signal to concentration
        concentration = self.convert_signal(processed_signal)

        # Invoke plot
        if self.verbosity >= 1:
            plt.show()

        return darsia.Image(concentration, img.metadata)

    # ! ---- Inspection routines
    def _inspect_diff(self, img: darsia.Image) -> None:
        """
        Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (darsia.Image): image
        """
        pass

    def _inspect_signal(self, img: darsia.Image) -> None:
        """
        Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (darsia.Image): image
        """
        pass

    def _inspect_clean_signal(self, img) -> None:
        """
        Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (darsia.Image): image
        """
        pass

    # ! ---- Pre- and post-processing methods
    def _subtract_background(self, img: darsia.Image) -> darsia.Image:
        """
        Take difference between input image and baseline image, based
        on cached option.

        Args:
            img (darsia.Image): test image.

        Returns:
            darsia.Image: difference with background image
        """

        if self._diff_option == "positive":
            diff = np.clip(img.img - self.base.img, 0, None)
        elif self._diff_option == "negative":
            diff = np.clip(self.base.img - img.img, 0, None)
        elif self._diff_option == "absolute":
            diff = skimage.util.compare_images(img.img, self.base.img, method="diff")
        else:
            raise ValueError(f"Diff option {self._diff_option} not supported")

        return diff

    def _extract_scalar_information(self, img: np.ndarray) -> np.ndarray:
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
            # Assume RGB input
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif self.color == "red":
            return img[:, :, 0]
        elif self.color == "green":
            return img[:, :, 1]
        elif self.color == "blue":
            return img[:, :, 2]
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

    def _homogenize_signal(self, img: np.ndarray) -> np.ndarray:
        """
        Routine responsible for rescaling wrt segments.

        Here, it is assumed that only one segment is present.
        Thus, no rescaling is performed.

        Args:
            img (np.ndarray): image

        Returns:
            np.ndarray: homogenized image
        """
        return img

    def postprocess_signal(self, signal: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Empty postprocessing - should be overwritten for practical cases.

        Example for a postprocessing using some noise removal.
        return skimage.restoration.denoise_tv_chambolle(signal, weight=0.1)

        Include img (which is the diff from __call__) to allow to extract other
        information in the postprocessing, e.g., in the prior/posterior
        analysis.
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
        self, baseline_images: list[darsia.Image], reset: bool = False
    ) -> None:
        """
        Determine natural noise by studying a series of baseline images.
        The resulting cleaning filter will be used prior to the conversion
        of signal to concentration. The cleaning filter should be understood
        as thresholding mask.

        Args:
            baseline_images (list of darsia.Image): series of baseline_images.
            reset (bool): flag whether the cleaning filter shall be reset.
        """
        # Initialize cleaning filter
        if reset:
            self.threshold = np.zeros(self.base.img.shape[:2], dtype=float)

        # Combine the results of a series of images
        for img in baseline_images:

            probe_img = img.copy()

            # Take (unsigned) difference
            diff = self._subtract_background(probe_img)

            # Extract mono-colored version
            monochromatic_diff = self._extract_scalar_information(diff)

            # Consider elementwise max
            self.threshold = np.maximum(self.threshold, monochromatic_diff)

    def calibrate(
        self,
        injection_rate: float,
        images: list[darsia.Image],
        initial_guess: Optional[tuple[float]] = None,
        tol: float = 1e-3,
        maxiter: int = 20,
    ) -> None:
        """
        Calibrate the conversion used in __call__ such that the provided
        injection rate is matched for the given set of images.

        Args:
            injection_rate (float): constant injection rate in ml/hrs.
            images (list of darsia.Image): images used for the calibration.
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

    def _estimate_rate(self, images: list[darsia.Image]) -> tuple[float, float]:
        """
        Estimate the injection rate for the given series of images.

        Args:
            images (list of darsia.Image): basis for computing the injection rate.

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
        ransac = RANSACRegressor()
        ransac.fit(np.array(relative_times).reshape(-1, 1), np.array(total_volumes))

        # Extract the slope and convert to
        return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

    def _determine_total_volume(self, concentration: darsia.Image) -> float:
        """
        Determine the total concentration of a spatial concentration map.

        Args:
            concentration (darsia.Image): concentration data.

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
            assert isinstance(self._volumes_cache, dict)
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
                volumes *= np.sum(ref_volumes) / np.sum(cast(np.ndarray, volumes))
                self._volumes_cache[shape] = volumes

        # Integral of locally weighted concentration values
        return np.sum(
            np.multiply(
                np.squeeze(cast(np.ndarray, volumes)), np.squeeze(concentration.img)
            )
        )


class SegmentedConcentrationAnalysis(ConcentrationAnalysis):
    """
    Special case of ConcentrationAnalysis broadened and tailored to segmented media.

    NOTE! Currently still under development and highly experimental.
    """

    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        labels: np.ndarray,
        color: Union[str, Callable] = "gray",
        **kwargs,
    ) -> None:
        """
        Constructor for SegmentedConcentrationAnalysis.

        Calls the constructor of parent class and fetches
        all tuning parameters for the binary segmentation.

        Args:
            base (darsia.Image or list of such): same as in ConcentrationAnalysis.
            labels (np.ndarray): labeled image identifying different segments.
                Note: label 0 is ignored.
            color (string or callable): same as in ConcentrationAnalysis.
            kwargs (keyword arguments): interface to all tuning parameters
        """
        super().__init__(base, color)

        # Segmentation
        self.labels = labels
        self.labels_set = np.unique(labels)
        self.num_labels = self.labels_set.shape[0]

        # From the labels determine the region properties
        self.regionprops = skimage.measure.regionprops(self.labels)

        # Cache to segmentation_scaling
        self.segmentation_scaling_path = Path(
            kwargs.get("segmentation_scaling", "./cache/segmentation_scaling.npy")
        )
        # Add segmentation-specific scaling parameter
        self.segmentation_scaling = np.ones(self.num_labels, dtype=float)

        # Prepare geometrical information concerning the segmentation of the geometry
        self.setup_path = kwargs.pop("contours_path", "cache")
        contour_thickness = kwargs.pop("contour_thickness", 10)
        contour_overlap_threshold = kwargs.pop("contour_overlap_threshold", 1000)
        # TODO only required if calibrating the segmentation scaling - move!
        self._setup(contour_thickness, contour_overlap_threshold)

        # Fetch verbosity. If True, several intermediate results in the
        # postprocessing will be displayed. This allows for simpler tuning
        # of the parameters.
        self.verbosity: int = kwargs.pop("verbosity", 0)

    def _setup(self, contour_thickness: int, overlap_threshold: int) -> None:
        """
        One-time setup of infrastructure required for the calibration. Will be
        written to file for re-use when rerunning a calibration analysis.

        Defines/fetches values for the attributes self.contour_mask and
        self.label_couplings

        Args:
            contour_thickness (int): thickness of enlarged contour
            overlap_threshold (int): number of pixels required for an overlap
                to be considered as (strong) coupling.
        """
        # Define paths to
        setup_path = Path(self.setup_path)
        setup_path.mkdir(parents=True, exist_ok=True)
        segmentation_contour_path = setup_path / Path("segmentation_contours.npy")

        # Fetch values if existent, otherwise generate them from scratch
        if segmentation_contour_path.exists():

            segmentation_contours = np.load(
                segmentation_contour_path, allow_pickle=True
            ).item()

            self.contour_mask = segmentation_contours["contour_mask"]
            self.label_couplings = segmentation_contours["label_couplings"]
            self.coupling_strength = segmentation_contours["coupling_strength"]

            # As self.label_couplings is an array, and arrays are not hashable
            # convert to a list of tuples (also as constructed below).
            self.label_couplings = [
                tuple(coupling) for coupling in self.label_couplings
            ]

        else:
            # Define thick contours for all labels
            def _labeled_mask_to_contour_mask(
                labeled_mask: np.ndarray, thickness
            ) -> np.ndarray:
                """
                Starting from a boolean array identifying a region, find
                the contours with a user-defined bandwidth.

                Args:
                    labeled_mask (np.ndarray): boolean array identifying a connected region.

                Returns:
                    np.ndarray: boolean array identifying a band width of the contours
                """
                # Determine the contours of the labeled mask
                contours, _ = cv2.findContours(
                    skimage.img_as_ubyte(labeled_mask),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_NONE,
                    # cv2.CHAIN_APPROX_SIMPLE, # TODO which one? test...
                )

                # TODO Uniformly split contours in N segments and determine
                # jump in an integral sense
                # contour_segments = []

                # Extract the contour as mask
                contour_mask = np.zeros(self.base.img.shape[:2], dtype=bool)
                for c in contours:
                    c = (c[:, 0, 1], c[:, 0, 0])
                    contour_mask[c] = True

                # Dilate to generate a thick contour
                contour_mask = skimage.morphology.binary_dilation(
                    contour_mask, np.ones((thickness, thickness), np.uint8)
                )

                # Convert to boolean mask
                contour_mask = skimage.img_as_bool(contour_mask)

                return contour_mask

            # Final assignment of contours
            contour_mask = {}
            for label in self.labels_set:
                contour_mask[label] = _labeled_mask_to_contour_mask(
                    self.labels == label, contour_thickness
                )

            # Determine common intersection of thick contours shared by neighboring segments
            # Find relevant couplings of masked labels.
            coupling_strength = []
            label_couplings = []
            for label1 in self.labels_set:
                for label2 in self.labels_set:

                    # Consider only directed pairs
                    if label1 < label2:

                        # Check if labeled regions share significant part of contour
                        if (
                            np.count_nonzero(
                                np.logical_and(
                                    contour_mask[label1], contour_mask[label2]
                                )
                            )
                            > overlap_threshold
                        ):
                            label_couplings.append((label1, label2))
                            # Determine the coupling strength depending on the size of the
                            # overlap (later adjusted relatively to the global data).
                            # Approximation of the size of the interface in terms of the
                            # number of pixels for the thick contour.
                            coupling_strength.append(
                                np.count_nonzero(
                                    np.logical_and(
                                        contour_mask[label1], contour_mask[label2]
                                    )
                                )
                            )

            # Cache the central objects
            self.contour_mask = contour_mask
            self.label_couplings = label_couplings
            self.coupling_strength = [
                cs / max(coupling_strength) for cs in coupling_strength
            ]

            # Store to file
            np.save(
                segmentation_contour_path,
                {
                    "contour_mask": self.contour_mask,
                    "label_couplings": self.label_couplings,
                    "coupling_strength": self.coupling_strength,
                },
            )

    # ! ---- Main methods

    def _homogenize_signal(self, img: np.ndarray) -> np.ndarray:
        """
        Routine responsible for rescaling wrt segments.

        Here, it is assumed that only one segment is present.
        Thus, no rescaling is performed.
        """
        for l_counter, label in enumerate(self.labels_set):
            img[self.labels == label] *= self.segmentation_scaling[l_counter]

        return img

    def calibrate_segmentation_scaling(
        self,
        images: list[darsia.Image],
        path: Optional[Path] = None,
        median_disk_radius: int = 20,
        mean_thresh: int = 1,
    ) -> None:
        """
        Routine to setup self.segmentation_scaling.

        Using a set of images, the discontinuity modulus is minimized
        using the segment-wise scaling of the signal.

        Args:
            images (list of darsia.Image): list of processed images
            path (Path, optional): path to scaling vector
            median_disk_radius (int): radius used in the computation of medians
            mean_thresh (int): threshold value
        """
        # TODO move input argument to init?

        # If existent, read segmentation scaling vector from file; otherwise
        # construct it by analyzing the discontinuity jump.
        if self.segmentation_scaling_path.exists():

            self.segmentation_scaling = np.load(self.segmentation_scaling_path)

        else:

            # Strategy: Quantify the discontinuity jumpt of the signal at
            # all boundaries between different segments. These are stored
            # in interace_ratio_container. These ratios will be used to
            # define a segment-wise scaling factor. To transfer the infos
            # on interfaces to segments a least-squares problem is solved.
            #

            # Initialize collection of interface ratios with empty lists,
            # as well as flag of trustable information by identifying
            # none of the coupling as trustworthy.
            interface_ratio_container: dict = {}
            trustable_summary = {}

            for coupling in self.label_couplings:
                interface_ratio_container[coupling] = []
                trustable_summary[coupling] = False

            # Find a suitable segmentation_scaling vector for each separate image.
            for img in images:

                # Generate the clean signal, as obtained in __call__, just before
                # applying _homogenize_signal().
                # TODO move somewhere to have a unique version?

                def _image_to_clean_signal(img: darsia.Image) -> np.ndarray:
                    probe_img = img.copy()

                    # Take (unsigned) difference
                    diff = self._subtract_background(probe_img)

                    # Extract monochromatic version
                    signal = self._extract_scalar_information(diff)

                    # Clean signal
                    clean_signal = np.clip(signal - self.threshold, 0, None)

                    return clean_signal

                signal = _image_to_clean_signal(img)

                # Define the segment-wise median
                median = np.zeros(signal.shape[:2], dtype=np.uint8)
                for regionprop in self.regionprops:

                    # Get mask
                    mask = self.labels == regionprop.label

                    # determine bounding box for labels == label, from regionprops
                    bbox = regionprop.bbox

                    # Define the bbox as roi
                    roi = (slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3]))

                    # Extract image and mask in roi
                    mask_roi = mask[roi]
                    signal_roi = signal[roi]

                    # Determine median on roi
                    median_label_roi = skimage.filters.rank.median(
                        skimage.img_as_ubyte(signal_roi),
                        skimage.morphology.disk(median_disk_radius),
                        mask=mask_roi,
                    )

                    # Remove signal outside the mask
                    median_label_roi[~mask_roi] = 0

                    # Exted to full image
                    median[roi] += median_label_roi

                # Make discontinuity analysis and obtain interface ratios for this image.
                # Find ratio of mean median values for each boundary.
                # The idea will be that this ratio will be required
                # for one-sided scaling in order to correct for
                # discontinuities. Store the ratios in a dictionary.
                # To keep track of orientation, consider sorted pairs
                # of combinations. Also keep track which ratios are
                # trustable or not (only if sufficient information
                # provided).

                interface_ratio = {}
                trustable = {}

                for coupling in self.label_couplings:

                    # TODO replace - hardcoded
                    if coupling in [(5, 7), (7, 8), (6, 10), (7, 10), (9, 12)]:
                        continue

                    # Fetch the single labels
                    label1, label2 = coupling

                    # Fetch the common boundary
                    common_boundary = np.logical_and(
                        self.contour_mask[label1], self.contour_mask[label2]
                    )

                    # Restrict the common boundary to the separate subdomains
                    roi1 = np.logical_and(common_boundary, self.labels == label1)
                    roi2 = np.logical_and(common_boundary, self.labels == label2)

                    # Consider the mean of the median of the signal on the separate bands
                    mean1 = np.mean(median[roi1])
                    mean2 = np.mean(median[roi2])

                    # Check whether this result can be trusted - require sufficient signal.
                    trustable[coupling] = min(mean1, mean2) >= mean_thresh

                    # Define the ratio / later scaling - if value not trustable (for
                    # better usability) choose ratio equal to 1
                    interface_ratio[coupling] = (
                        mean1 / mean2 if trustable[coupling] else 1
                    )

                # Keep only trustable information and collect.
                for coupling, ratio in interface_ratio.items():
                    if trustable[coupling]:
                        interface_ratio_container[coupling].append(ratio)
                        trustable_summary[coupling] = True

            # Perform in principle a Least-squares reduction by choosing a unique
            # interface ratio for each interface by simply reducing to the mean.
            summarized_interface_ratio = {}
            for coupling in self.label_couplings:
                if trustable_summary[coupling]:
                    summarized_interface_ratio[coupling] = np.mean(
                        np.array(interface_ratio_container[coupling])
                    )

                else:
                    # Aim at equalizing scaling parameters
                    summarized_interface_ratio[coupling] = 1.0

            # print("trustable summary", trustable_summary)

            # TODO rm

            # Fix the lowest trusted label - requires that the trusted interfaces
            # define a connected graph.
            lowest_trusted_label = np.min(
                [coupling[0] for coupling in trustable_summary.keys()]
            )

            # TODO - rm the definition of untrusted_labels? it should be taken care
            # of in the definition os self.coupling_strength

            # Find labels for which all couplings are untrustable
            trusted_labels = []
            untrusted_labels = []

            for label in self.labels_set:

                # Initialize label as untrusted
                label_is_trusted = False

                # Identify label as trusted if part of trusted interface ratio
                for coupling in self.label_couplings:
                    if label in coupling:
                        label_is_trusted = True

                # Identify label as untrusted
                if not label_is_trusted:
                    untrusted_labels.append(label)
                else:
                    trusted_labels.append(label)

            # TODO need to check whether trusted interfaces define a connected graph
            # print("unstrusted", untrusted_labels)

            # TODO identify connected components.
            # Define undirected graph based on the trusted labels
            graph = np.zeros((self.num_labels, self.num_labels), dtype=int)
            for coupling, trustvalue in trustable_summary.items():
                if trustvalue:
                    graph[coupling] = 1
            sparse_graph = sps.csr_matrix(graph)
            (
                num_connected_components,
                labels_connected_components,
            ) = sps.csgraph.connected_components(
                sparse_graph, directed=False, return_labels=True
            )

            # Identify characteristic label (with minimal label id) for each connected subgraph
            connected_components_in_graph = []
            for comp in range(num_connected_components):
                # Find first label with
                characteristic_label = np.min(
                    np.argwhere(labels_connected_components == comp)
                )
                connected_components_in_graph.append(characteristic_label)
            # TODO rm?
            connected_components_in_graph = []

            # TODO rm unused
            # print("graph analysis")
            # print(num_connected_components)
            # print(labels_connected_components)
            # print(connected_components_in_graph)

            untrusted_labels = []

            # Based on the interface ratios, build a linear (overdetermined) system,
            # which characterizes the optimal scaling.
            matrix = np.zeros((0, self.num_labels), dtype=float)
            rhs = np.zeros((0, 1), dtype=float)
            num_constraints = 0

            # Add untrusted components and a single constraints (to fix the system).
            # TODO choose range
            # for label in untrusted_labels + connected_components_in_graph:
            # +[lowest_trusted_label]:
            for label in [lowest_trusted_label]:
                basis_vector = np.zeros((1, self.num_labels), dtype=float)
                basis_vector[0, label] = 1
                matrix = np.vstack((matrix, basis_vector))
                rhs = np.vstack((rhs, np.array([[1]])))
                num_constraints += 1

            # Add similarity components
            similar_components = [
                [1, 10, 11],
                [2, 3, 4],
                [6, 7, 8],
            ]  # TODO this is hardcoded...
            similarity_weight = 1  # TODO choose another value?
            for components in similar_components:
                if len(components) > 1:
                    for coupling in list(combinations(components, 2)):
                        label1, label2 = coupling
                        similarity_balance = np.zeros((1, self.num_labels), dtype=float)
                        similarity_balance[0, label1] = similarity_weight
                        similarity_balance[0, label2] = -similarity_weight
                        matrix = np.vstack((matrix, similarity_balance))
                        rhs = np.vstack((rhs, np.array([0])))
                        num_constraints += 1

            # Add trusted couplings.
            for coupling in self.label_couplings:
                label1, label2 = coupling
                scaling_balance = np.zeros((1, self.num_labels), dtype=float)
                scaling_balance[0, label1] = summarized_interface_ratio[coupling]
                scaling_balance[0, label2] = -1
                matrix = np.vstack((matrix, scaling_balance))
                rhs = np.vstack((rhs, np.array([0])))

            # Scale matrix and rhs with couplin strength to prioritize significant interfaces
            matrix[num_constraints:, :] = np.matmul(
                np.diag(self.coupling_strength), matrix[num_constraints:, :]
            )
            rhs[num_constraints:] = np.matmul(
                np.diag(self.coupling_strength), rhs[num_constraints:]
            )

            # Determine suitable scaling by solving the overdetermined system using
            # a least-squares approach.
            self.segmentation_scaling = np.linalg.lstsq(
                matrix, np.ravel(rhs), rcond=None
            )[0]

            # TODO rm. Print the solution
            # print("label couplings", self.label_couplings)
            # print("coupling_strength", self.coupling_strength)
            print(f"Computed segmentation scaling: {self.segmentation_scaling}")
            plt.figure()
            plt.imshow(self.labels)
            scaling_image = np.zeros(self.labels.shape[:2], dtype=float)
            for label in range(self.num_labels):
                mask = self.labels == label
                scaling_image[mask] = self.segmentation_scaling[label]
            plt.figure()
            plt.imshow(scaling_image)
            plt.show()

            # Cache the scaling vector
            np.save(self.segmentation_scaling_path, self.segmentation_scaling)


class BinaryConcentrationAnalysis(ConcentrationAnalysis):
    """
    Special case of ConcentrationAnalysis which generates boolean concentration profiles.
    """

    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        color: Union[str, Callable] = "gray",
        **kwargs,
    ) -> None:
        """
        Constructor for BinaryConcentrationAnalysis.

        Calls the constructor of parent class and fetches
        all tuning parameters for the binary segmentation.

        Args:
            base (darsia.Image or list of such): same as in ConcentrationAnalysis.
            color (string or callable): same as in ConcentrationAnalysis.
            kwargs (keyword arguments): interface to all tuning parameters
        """
        super().__init__(base, color, **kwargs)

        # TVD parameters for pre and post smoothing
        self.apply_presmoothing = kwargs.pop("presmoothing", False)
        if self.apply_presmoothing:
            pre_global_resize = kwargs.pop("presmoothing resize", 1.0)
            self.presmoothing = {
                "resize x": kwargs.pop("presmoothing resize x", pre_global_resize),
                "resize y": kwargs.pop("presmoothing resize y", pre_global_resize),
                "weight": kwargs.pop("presmoothing weight", 1.0),
                "eps": kwargs.pop("presmoothing eps", 1e-5),
                "max_num_iter": kwargs.pop("presmoothing max_num_iter", 1000),
                "method": kwargs.pop("presmoothing method", "chambolle"),
            }

        self.apply_postsmoothing = kwargs.pop("postsmoothing", False)
        if self.apply_postsmoothing:
            post_global_resize = kwargs.pop("postsmoothing resize", 1.0)
            self.postsmoothing = {
                "resize x": kwargs.pop("postsmoothing resize x", post_global_resize),
                "resize y": kwargs.pop("postsmoothing resize y", post_global_resize),
                "weight": kwargs.pop("postsmoothing weight", 1.0),
                "eps": kwargs.pop("postsmoothing eps", 1e-5),
                "max_num_iter": kwargs.pop("postsmoothing max_num_iter", 1000),
                "method": kwargs.pop("postsmoothing method", "chambolle"),
            }

        # Thresholding parameters
        self.apply_dynamic_threshold: bool = kwargs.pop("threshold dynamic", False)
        if self.apply_dynamic_threshold:
            # Define global lower and upper bounds for the threshold value
            self.threshold_value_lower_bound: Union[float, np.ndarray] = kwargs.pop(
                "threshold value min", 0.0
            )
            self.threshold_value_upper_bound: Union[float, np.ndarray] = kwargs.pop(
                "threshold value max", 255.0
            )
            self.threshold_conservative = kwargs.pop("threshold conservative", False)
        else:
            # Define fixed global threshold value
            self.threshold_value: float = kwargs.get("threshold value", 0.0)

        # Parameters to remove small objects
        self.min_size: int = kwargs.pop("min area size", 1)

        # Parameters to fill holes
        self.area_threshold: int = kwargs.pop("max hole size", 0)

        # Parameters for local convex cover
        self.cover_patch_size: int = kwargs.pop("local convex cover patch size", 1)

        # Threshold for posterior analysis based on gradient moduli
        self.apply_posterior = kwargs.pop("posterior", False)
        if self.apply_posterior:
            self.posterior_criterion: str = kwargs.pop(
                "posterior criterion", "gradient modulus"
            )
            assert self.posterior_criterion in [
                "gradient modulus",
                "value",
                "relative value",
                "value/gradient modulus",
                "value/value extra color",
            ]
            self.posterior_threshold = kwargs.pop("posterior threshold", 0.0)
            if self.posterior_criterion == "value/value extra color":
                # Allow for a different color in the posterior
                self.posterior_color: Union[str, Callable] = kwargs.pop(
                    "posterior extra color", self.color
                )

        # Mask
        self.mask: np.ndarray = np.ones(self.base.img.shape[:2], dtype=bool)

        # Fetch verbosity. If True, several intermediate results in the
        # postprocessing will be displayed. This allows for simpler tuning
        # of the parameters.
        self.verbosity: int = kwargs.pop("verbosity", 0)

    def update_mask(self, mask: np.ndarray) -> None:
        """
        Update the mask to be considered in the analysis.

        Args:
            mask (np.ndarray): boolean mask, detecting which pixels will
                be considered, all other will be ignored in the analysis.
        """
        self.mask = mask

    # ! ---- Main methods
    def _prior(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prior postprocessing routine, essentially converting a continuous
        signal into a binary concentration and thereby segmentation.
        The post processing consists of presmoothing, thresholding,
        filling holes, local convex covering, and postsmoothing.
        Tuning parameters for this routine have to be set in the
        initialization routine.

        Args:
            signal (np.ndarray): signal

        Returns:
            np.ndarray: prior binary mask
            np.ndarray: smoothed signal, used in postprocessing
        """

        # Prepare the signal by applying presmoothing
        smooth_signal = self.prepare_signal(signal)

        # Apply thresholding to obain a thresholding mask
        mask = self._apply_thresholding(smooth_signal)

        if self.verbosity >= 2:
            plt.figure("Prior: Thresholded mask")
            plt.imshow(mask)

        # Clean mask by removing small objects, filling holes, and applying postsmoothing.
        clean_mask = self.clean_mask(mask)

        if self.verbosity >= 1:
            plt.show()

        return clean_mask, smooth_signal

    def prepare_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply presmoothing.

        Args:
            signal (np.ndarray): input signal

        Return:
            np.ndarray: smooth signal
        """
        if self.verbosity >= 3:
            plt.figure("Prior: Input signal")
            plt.imshow(signal)

        # Apply presmoothing
        if self.apply_presmoothing:
            # Resize image
            signal = cv2.resize(
                signal.astype(np.float32),
                None,
                fx=self.presmoothing["resize x"],
                fy=self.presmoothing["resize y"],
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
            elif self.presmoothing["method"] == "isotropic bregman":
                signal = skimage.restoration.denoise_tv_bregman(
                    signal,
                    weight=self.presmoothing["weight"],
                    eps=self.presmoothing["eps"],
                    max_num_iter=self.presmoothing["max_num_iter"],
                    isotropic=True,
                )
            else:
                raise ValueError(f"Method {self.presmoothing['method']} not supported.")

            # Resize to original size
            signal = cv2.resize(signal, tuple(reversed(self.base.img.shape[:2])))

        if self.verbosity >= 2:
            plt.figure("Prior: TVD smoothed signal")
            plt.imshow(signal)

        return signal

    def _apply_thresholding(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply global thresholding to obtain binary mask.

        Both static and dynamic thresholding is possible.

        Args:
            signal (np.ndarray): signal on entire domain.

        Returns:
            np.ndarray: binary thresholding mask
        """

        # Update the thresholding values if dynamic thresholding is chosen.
        if self.apply_dynamic_threshold:

            # Only continue if mask not empty
            if np.count_nonzero(self.mask) > 0:
                # Extract merely signal values in the effective mask
                active_signal_values = np.ravel(signal)[np.ravel(self.mask)]

                # Define only once
                def determine_threshold(
                    signal_1d: np.ndarray,
                    sigma: float = 10,
                    bins: int = 100,
                    relative_boundary: float = 0.05,
                ) -> tuple[float, bool]:
                    """
                    Find global minimum and check whether it also is a local minimum
                    (sufficient to check whether it is located at the boundary.

                    Args:
                        signal_1d (np.ndarray): 1d signal
                    """

                    # Smooth the histogram of the signal
                    smooth_hist = ndi.gaussian_filter1d(
                        np.histogram(signal_1d, bins=bins)[0], sigma=sigma
                    )

                    # Determine the global minimum (index)
                    global_min_index = np.argmin(smooth_hist)

                    # Determine the global minimum (in terms of signal values),
                    # determining the candidate for the threshold value
                    thresh_global_min = np.min(signal_1d) + global_min_index / float(
                        bins
                    ) * (np.max(signal_1d) - np.min(signal_1d))

                    # As long a the global minimum does not lie close to the boundary,
                    # it is consisted a local minimum.
                    is_local_min = (
                        relative_boundary * bins
                        < global_min_index
                        < (1 - relative_boundary) * bins
                    )

                    return thresh_global_min, is_local_min, smooth_hist

                # Determine the threshold value (global min of the 1d signal)
                updated_threshold, is_local_min, smooth_hist = determine_threshold(
                    active_signal_values
                )

                # Admit the new threshold value if it is also a local min, and clip at
                # provided bounds.
                if self.threshold_method == "local min" and is_local_min:
                    self.threshold_value = np.clip(
                        updated_threshold,
                        self.threshold_value_lower_bound,
                        self.threshold_value_upper_bound,
                    )
                elif self.threshold_method == "conservative":
                    self.threshold_value = np.clip(
                        updated_threshold,
                        self.threshold_value_lower_bound,
                        self.threshold_value_upper_bound,
                    )

                if self.verbosity >= 2:
                    plt.figure("Histogram analysis")
                    plt.plot(
                        np.linspace(
                            np.min(active_signal_values),
                            np.max(active_signal_values),
                            100,
                        ),
                        smooth_hist,
                    )

        if self.verbosity >= 1:
            print("Thresholding value", self.threshold_value)

        # Build the mask segment by segment.
        mask = signal > self.threshold_value

        return mask

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small objects in binary mask, fill holes and apply postsmoothing.

        Args:
            mask (np.ndarray): binary mask

        Returns:
            np.ndarray: cleaned mask
        """
        # Remove small objects
        if self.min_size > 1:
            mask = skimage.morphology.remove_small_objects(mask, min_size=self.min_size)

        # Fill holes
        if self.area_threshold > 0:
            mask = skimage.morphology.remove_small_holes(
                mask, area_threshold=self.area_threshold
            )

        if self.verbosity >= 3:
            plt.figure("Prior: Cleaned mask")
            plt.imshow(mask)

        # NOTE: Currently not used, yet, kept for the moment.
        # # Loop through patches and fill up
        # if self.cover_patch_size > 1:
        #     covered_mask = np.zeros(mask.shape[:2], dtype=bool)
        #     size = self.cover_patch_size
        #     Ny, Nx = mask.shape[:2]
        #     for row in range(int(Ny / size)):
        #         for col in range(int(Nx / size)):
        #             roi = (
        #                 slice(row * size, (row + 1) * size),
        #                 slice(col * size, (col + 1) * size),
        #             )
        #             covered_mask[roi] = skimage.morphology.convex_hull_image(mask[roi])
        #     # Update the mask value
        #     mask = covered_mask
        #
        # if self.verbosity:
        #     plt.figure("Prior: Locally covered mask")
        #     plt.imshow(mask)

        # Apply postsmoothing
        if self.apply_postsmoothing:
            # Resize image
            resized_mask = cv2.resize(
                mask.astype(np.float32),
                None,
                fx=self.postsmoothing["resize x"],
                fy=self.postsmoothing["resize y"],
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
            elif self.postsmoothing["method"] == "isotropic bregman":
                smoothed_mask = skimage.restoration.denoise_tv_bregman(
                    resized_mask,
                    weight=self.postsmoothing["weight"],
                    eps=self.postsmoothing["eps"],
                    max_num_iter=self.postsmoothing["max_num_iter"],
                    isotropic=True,
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
            thresh = 0.5
            mask = large_mask > thresh

        if self.verbosity >= 2:
            plt.figure("Prior: TVD postsmoothed mask")
            plt.imshow(mask)

            plt.show()

        return mask

    def _posterior(
        self, signal: np.ndarray, mask_prior: np.ndarray, img: np.ndarray
    ) -> np.ndarray:
        """
        Posterior analysis of signal, determining the gradients of
        for marked regions.

        Args:
            signal (np.ndarray): (smoothed) signal
            mask_prior (np.ndarray): boolean mask marking prior regions
            img (np.ndarray): original difference of images

        Return:
            np.ndarray: boolean mask of trusted regions.
        """
        # Only continue if necessary
        if not self.apply_posterior:
            return np.ones(signal.shape[:2], dtype=bool)

        # Initialize the output mask
        mask_posterior = np.zeros(signal.shape, dtype=bool)

        # Label the connected regions first
        labels_prior, num_labels_prior = skimage.measure.label(
            mask_prior, return_num=True
        )

        if self.verbosity >= 3:
            plt.figure("Posterior: Labeled regions from prior")
            plt.imshow(labels_prior)
            plt.show()

        # Criterion-specific preparations
        if self.posterior_criterion in ["gradient modulus", "value/gradient modulus"]:
            # Determien gradient modulus of the smoothed signal
            dx = darsia.forward_diff_x(signal)
            dy = darsia.forward_diff_y(signal)
            gradient_modulus = np.sqrt(dx**2 + dy**2)

            if self.verbosity >= 2:
                plt.figure("Posterior: Gradient modulus")
                plt.imshow(gradient_modulus)

        if self.posterior_criterion == "value/value extra color":
            # Restrict image to monochromatic color and roi
            if self.posterior_color == "red":
                img_color = img[:, :, 0]
            elif self.posterior_color == "green":
                img_color = img[:, :, 1]
            elif self.posterior_color == "blue":
                img_color = img[:, :, 2]
            elif self.posterior_color == "red+green":
                img_color = img[:, :, 0] + img[:, :, 1]
            else:
                raise ValueError(f"Color {self.posterior_color} not supported.")

            # Prepare the signal by applying presmoothing
            smooth_img_color = self.prepare_signal(img_color)

        # Investigate each labeled region separately; omit label 0, which corresponds
        # to non-marked area.
        for label in range(1, num_labels_prior + 1):

            # Fix one label
            labeled_region = labels_prior == label

            # Initialize acceptance
            accept = False

            # Check the chosen criterion
            if self.posterior_criterion == "value":
                # Check whether there exist values in the segment, larger
                # than a provided critical value.

                roi = np.logical_and(labeled_region, self.mask)

                if (
                    np.count_nonzero(roi) > 0
                    and np.max(signal[roi]) > self.posterior_threshold
                ):
                    accept = True

                if self.verbosity >= 3:
                    print(
                        f"""Posterior: Label {label},
                        max value: {np.max(signal[roi])}."""
                    )

            elif self.posterior_criterion == "relative value":
                # Check whether there exist values in the segment, larger
                # than a provided critical value, measured relatively to
                # the smallest existing value (the threshold value).

                roi = np.logical_and(labeled_region, self.mask)

                if np.count_nonzero(roi) > 0 and np.max(
                    signal[roi]
                ) > self.posterior_threshold * np.min(signal[roi]):
                    accept = True

                if self.verbosity >= 3:
                    print(
                        f"""Posterior: Label {label},
                        max value: {np.max(signal[roi]) / np.min(signal[roi])}."""
                    )

            elif self.posterior_criterion == "value/value extra color":
                # Check whether there exist values in the segment, larger
                # than a provided critical value, for the specific color
                # provided and based on the original difference of images.

                # Apply posterior analysis on the region of interest
                roi = np.logical_and(labeled_region, self.mask)

                if (
                    np.count_nonzero(roi) > 0
                    and np.max(signal[roi]) > self.posterior_threshold[0]
                    and np.max(smooth_img_color[roi]) > self.posterior_threshold[1]
                ):
                    accept = True

                if np.count_nonzero(roi) > 0:
                    if self.verbosity >= 3:
                        print(
                            f"""Posterior: Label {label},
                            signal max value: {np.max(signal[roi])},
                            extra color max value: {np.max(img_color[roi])},
                            smooth extra color max value: {np.max(smooth_img_color[roi])}."""
                        )

                if self.verbosity >= 3:
                    plt.figure("posterior extra color value analysis")
                    plt.imshow(img_color)
                if self.verbosity >= 2:
                    plt.figure("posterior extra color value analysis - smooth")
                    plt.imshow(smooth_img_color)

            elif self.posterior_criterion == "gradient modulus":
                # Check whether the gradient modulus reaches high values
                # compared on contours to a provided tolerance.

                # Determine contour set of labeled region
                contours, _ = cv2.findContours(
                    skimage.img_as_ubyte(labeled_region),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                # For each part of the contour set, check whether the gradient is sufficiently
                # large at any location
                for c in contours:

                    # Extract coordinates of contours - have to flip columns, since cv2
                    # provides reverse matrix indexing, and also 3 components, with the
                    # second one single dimensioned.
                    c = (c[:, 0, 1], c[:, 0, 0])

                    if self.verbosity >= 2:
                        print(
                            f"""Posterior: Label {label},
                            Grad mod. {np.max(gradient_modulus[c])},
                            pos: {np.mean(c[0])}, {np.mean(c[1])}."""
                        )

                    # Identify region as marked if gradient sufficiently large
                    if np.max(gradient_modulus[c]) > self.posterior_threshold:
                        accept = True
                        break

            elif self.posterior_criterion == "value/gradient modulus":
                # Run both routines and require both to hold
                accept_value = (
                    np.max(signal[labeled_region]) > self.posterior_threshold[0]
                )

                # Check whether the gradient modulus reaches high values
                # compared on contours to a provided tolerance.
                accept_gradient_modulus = False

                # Determine contour set of labeled region
                contours, _ = cv2.findContours(
                    skimage.img_as_ubyte(labeled_region),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                # For each part of the contour set, check whether the gradient is sufficiently
                # large at any location
                for c in contours:

                    # Extract coordinates of contours - have to flip columns, since cv2
                    # provides reverse matrix indexing, and also 3 components, with the
                    # second one single dimensioned.
                    c = (c[:, 0, 1], c[:, 0, 0])

                    if self.verbosity >= 2:
                        print(
                            f"""Posterior: Label {label},
                            Grad mod. {np.max(gradient_modulus[c])},
                            pos: {np.mean(c[0])}, {np.mean(c[1])}."""
                        )

                    # Identify region as marked if gradient sufficiently large
                    if np.max(gradient_modulus[c]) > self.posterior_threshold[1]:
                        accept_gradient_modulus = True
                        break
                accept = accept_value and accept_gradient_modulus

            # Collect findings
            if accept:
                mask_posterior[labeled_region] = True

        return mask_posterior

    def postprocess_signal(self, signal: np.ndarray, img: np.ndarray) -> np.ndarray:
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
            img (np.ndarray): original difference of images, allowing
                to extract new information besides the signal.

        Returns:
            np.ndarray: binary concentration
        """
        # Determine prior and posterior
        mask_prior, smooth_signal = self._prior(signal)
        mask_posterior = self._posterior(smooth_signal, mask_prior, img)

        # NOTE: Here the overlay process is obsolete, posterior is active.
        # Yet, it allows to overwrite posterior by inheritance and design
        # other schemes.

        # Overlay prior and posterior
        mask = np.zeros(mask_prior.shape, dtype=bool)
        # Label the connected regions first
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


class SegmentedBinaryConcentrationAnalysis(BinaryConcentrationAnalysis):
    """
    Special case of ConcentrationAnalysis which generates boolean concentration profiles.
    Different to ConcentrationAnalysis, it takes advantage of apriori knowledge of
    structurally different regions in the domain, provided as a labeled image.
    These different regions are then treated differently.

    This class supports both static and dynamic thresholding.
    """

    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        labels: np.ndarray,
        color: Union[str, Callable] = "gray",
        **kwargs,
    ) -> None:
        """
        Constructor for SegmentedBinaryConcentrationAnalysis.

        # Update the thresholding values if dynamic thresholding is chosen.
        if self.apply_dynamic_threshold:

        Args:
            base (darsia.Image or list of such): same as in ConcentrationAnalysis.
            color (string or callable): same as in ConcentrationAnalysis.
            kwargs (keyword arguments): interface to all tuning parameters
        """
        super().__init__(base, color, **kwargs)

        # Segmentation
        self.labels = labels
        self.labels_set = np.unique(labels)
        self.num_labels = self.labels_set.shape[0]

        # From the labels determine the region properties
        self.regionprops = skimage.measure.regionprops(self.labels)

        # Initialize the threshold values.
        threshold_init_value: Union[float, list] = kwargs.get("threshold value")

        if isinstance(threshold_init_value, list):
            # Allow for heterogeneous initial value.
            assert len(threshold_init_value) == self.num_labels
            self.threshold_value = np.array(threshold_init_value)

        elif isinstance(threshold_init_value, float):
            # Or initialize all labels with the same value
            self.threshold_value = threshold_init_value * np.ones(
                self.num_labels, dtype=float
            )
        else:
            raise ValueError(f"Type {type(threshold_init_value)} not supported.")

        # For the case of dynamic threshold, allow to choose some settings.
        if self.apply_dynamic_threshold:
            # Intialize lower and upper bounds
            threshold_value_lower_bound: Union[float, list] = kwargs.pop(
                "threshold value min", 0.0
            )
            threshold_value_upper_bound: Union[float, list] = kwargs.pop(
                "threshold value max", 255.0
            )

            # For the segmented analysis, allow for heterogeneous parameters
            if isinstance(threshold_value_lower_bound, list):
                assert len(threshold_value_lower_bound) == self.num_labels
                self.threshold_value_lower_bound = np.array(threshold_value_lower_bound)

            elif isinstance(threshold_value_lower_bound, float):
                self.threshold_value_lower_bound = (
                    threshold_value_lower_bound * np.ones(self.num_labels, dtype=float)
                )

            if isinstance(threshold_value_upper_bound, list):
                assert len(threshold_value_upper_bound) == self.num_labels
                self.threshold_value_upper_bound = np.array(threshold_value_upper_bound)

            elif isinstance(threshold_value_upper_bound, float):
                pass
                self.threshold_value_upper_bound = (
                    threshold_value_upper_bound * np.ones(self.num_labels, dtype=float)
                )

            # Define the method how to choose dynamic threshold values (later to be chosen
            # as global minimum of signal histogram). The method "local min" requires
            # the threshold also to be a local minimum, and otherwise picks a cached
            # value, while the "conservative" method always aims at updating and in
            # in doubt choosing the provided upper bound, making a conservative choice.
            self.threshold_method = kwargs.get("threshold method", "local min")
            assert self.threshold_method in [
                "local/global min",
                "conservative global min",
                "first local min",
                "first local min enhanced",
                "first local min enhanced ransac",
                "first local min otsu",
                "otsu",
                "otsu local min",
                "gradient analysis",
            ]

            self.threshold_safety: Optional[str] = kwargs.get(
                "threshold safety", "none"
            )
            assert self.threshold_safety in ["none", "area", "min", "peaks passed"]

            if (
                self.threshold_method == "first local min enhanced"
                and self.threshold_safety == "peaks passed"
            ):
                self.pre_peaks_passed = np.zeros(self.num_labels, dtype=bool)
                self.peaks_passed = np.zeros(self.num_labels, dtype=bool)

        # Initialize cache for collecting threshold values
        self.threshold_cache = {i: [] for i in range(self.num_labels)}
        self.threshold_cache_all = {i: [] for i in range(self.num_labels)}

        # Also for the posterior analysis, allow for heterogeneous threshold
        if self.apply_posterior and self.posterior_criterion == "value":

            if isinstance(self.posterior_threshold, list):
                assert len(self.posterior_threshold) == self.num_labels
                self.posterior_threshold = np.array(self.posterior_threshold)
            elif isinstance(self.posterior_threshold, float):
                pass
            else:
                raise ValueError(f"Posterior threshold has not-supported data type.")

    def _apply_thresholding(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply heterogeneous thresholding to obtain binary mask.

        This routine is tailored to the segmented nature of the image.
        For each labeled segment, a separate thresholding value is possible.
        If dynamic thresholding is activated, a seperate thresholding
        value for each segment is determined.

        Args:
            signal (np.ndarray): signal on entire domain.

        Returns:
            np.ndarray: binary thresholding mask
        """

        # Update the thresholding values if dynamic thresholding is chosen.
        if self.apply_dynamic_threshold:

            for label in range(self.num_labels):

                # Determine mask of interest, i.e., consider single label,
                # the interval of interest and the provided mask.
                label_mask = self.labels == label
                effective_mask = np.logical_and(label_mask, self.mask)

                # Only continue if mask not empty
                if np.count_nonzero(effective_mask) > 0:

                    # Reduce the signal to the effective mask
                    active_signal_values = np.ravel(signal)[np.ravel(effective_mask)]

                    # General preparations for many of the methods

                    # Define tuning parameters for defining histograms,
                    # and smooth them. NOTE: They should be in general chosen
                    # tailored to the situation. However, these values should
                    # also work for most cases.
                    bins = 200
                    sigma = 10

                    # Smooth the histogram of the signal
                    smooth_hist = ndi.gaussian_filter1d(
                        np.histogram(active_signal_values, bins=bins)[0],
                        sigma=sigma,
                    )

                    # And its derivatives
                    smooth_hist_1st_derivative = np.gradient(smooth_hist)
                    smooth_hist_2nd_derivative = np.gradient(smooth_hist_1st_derivative)

                    # For tuning the parameters, plot the histogram and its derivatives
                    if self.verbosity >= 2:
                        plt.figure("Histogram analysis")
                        plt.plot(
                            np.linspace(
                                np.min(active_signal_values),
                                np.max(active_signal_values),
                                smooth_hist.shape[0],
                            ),
                            smooth_hist,
                            label=f"Label {label}",
                        )
                        plt.legend()

                    if self.verbosity >= 3:
                        plt.figure("Histogram analysis - 1st der")
                        plt.plot(
                            np.linspace(
                                np.min(active_signal_values),
                                np.max(active_signal_values),
                                smooth_hist.shape[0],
                            ),
                            smooth_hist_1st_derivative,
                            label=f"Label {label}",
                        )
                        plt.legend()

                        plt.figure("Histogram analysis - 2nd der")
                        plt.plot(
                            np.linspace(
                                np.min(active_signal_values),
                                np.max(active_signal_values),
                                smooth_hist.shape[0],
                            ),
                            smooth_hist_2nd_derivative,
                            label=f"Label {label}",
                        )
                        plt.legend()

                    # Initialize flag
                    have_new_value = False

                    # Choose algorithm based on user-input
                    if self.threshold_method in [
                        "local/global min",
                        "conservative global min",
                    ]:

                        def determine_threshold(
                            signal_1d: np.ndarray,
                            sigma: float = 10,
                            bins: int = 100,
                            relative_boundary: float = 0.05,
                        ) -> tuple[float, bool]:
                            """
                            Find global minimum and check whether it also is a local minimum
                            (sufficient to check whether it is located at the boundary.

                            Args:
                                signal_1d (np.ndarray): 1d signal
                            """

                            # Determine the global minimum (index)
                            global_min_index = np.argmin(smooth_hist)

                            # Determine the global minimum (in terms of signal values),
                            # determining the candidate for the threshold value
                            thresh_global_min = np.min(
                                signal_1d
                            ) + global_min_index / float(bins) * (
                                np.max(signal_1d) - np.min(signal_1d)
                            )

                            # As long a the global minimum does not lie close to the boundary,
                            # it is consisted a local minimum.
                            is_local_min = (
                                relative_boundary * bins
                                < global_min_index
                                < (1 - relative_boundary) * bins
                            )

                            return thresh_global_min, is_local_min

                        # Determine the threshold value (global min of the 1d signal)
                        (
                            updated_threshold,
                            is_local_min,
                        ) = determine_threshold(active_signal_values)

                        # Admit the new threshold value if it is also a local min, and clip at
                        # provided bounds.
                        if self.threshold_method == "local/global min" and is_local_min:
                            new_threshold = updated_threshold
                            have_new_value = True
                        elif self.threshold_method == "conservative global min":
                            new_threshold = updated_threshold
                            have_new_value = True

                    elif self.threshold_method in ["otsu", "otsu local min"]:

                        # Determine the global minimum (index)
                        # thresh_otsu = skimage.filters.threshold_otsu(active_signal_values)
                        thresh_otsu_smooth_pre = skimage.filters.threshold_otsu(
                            hist=smooth_hist
                        )
                        thresh_otsu_smooth = np.min(
                            active_signal_values
                        ) + thresh_otsu_smooth_pre / bins * (
                            np.max(active_signal_values) - np.min(active_signal_values)
                        )

                        if self.threshold_method == "otsu":
                            # Fix the computed OTSU threshold value for considered label
                            new_threshold = thresh_otsu_smooth

                            # Identify the success of the method
                            have_new_value = True

                            if self.verbosity >= 2:
                                print(
                                    f"""Label {label}; OTSU thresh {thresh_otsu_smooth};
                                    {new_threshold}."""
                                )

                        elif self.threshold_method == "otsu local min":
                            # Before accepcting the computed OTSU threshold value, check
                            # first whether it defines a local minimum. Use find_peaks,
                            # which requires some preparation of edge values to also
                            # consider edge extrema as local extrema - add small value.
                            enriched_smooth_hist = np.hstack(
                                (
                                    np.array([np.min(smooth_hist)]),
                                    smooth_hist,
                                    np.array([np.min(smooth_hist)]),
                                )
                            )

                            # Find the local maxima
                            peaks_pre, _ = find_peaks(enriched_smooth_hist)
                            peaks = np.min(active_signal_values) + (
                                peaks_pre + 1
                            ) / bins * (
                                np.max(active_signal_values)
                                - np.min(active_signal_values)
                            )

                            # Identify as local minimum if value lies in between two peaks
                            is_local_min = peaks.shape[0] > 1 and np.min(
                                peaks
                            ) < thresh_otsu_smooth < np.max(peaks)

                            # Update value if it is a local min
                            if is_local_min:
                                new_threshold = thresh_otsu_smooth

                                # Identify the success of the method
                                have_new_value = True

                            if self.verbosity >= 2:
                                print(
                                    f"""Label {label}; OTSU thresh {new_threshold};
                                    Peaks {peaks}; is local min {is_local_min}."""
                                )

                    elif self.threshold_method == "first local min":

                        # Under the assumption that there is a strong separation
                        # between background (concentration < tol) and foreground
                        # (concentration > tol), we are looking for a separation
                        # by a local minimum. However, when one of the above zones
                        # is not well represented so far, a global histogram will
                        # not be able to highlight such zones. Therefore, a relaxed
                        # concept of local minima is used. When the second derivative
                        # is increasing, and the the first derivative is below some
                        # tolerance, all corresponding points are identified as
                        # local min. We choose the smallest of these.
                        # NOTE: This is not applicable if diffusion zones are dominating
                        # and the transition has a long scale and it is smooth.

                        # Restrict to positive 2nd derivative and small 1st derivative
                        max_value = 0.01 * np.max(
                            np.absolute(smooth_hist_1st_derivative)
                        )
                        tol = 1e-6
                        feasible_indices = np.logical_and(
                            -max_value < smooth_hist_1st_derivative,
                            smooth_hist_2nd_derivative > tol,
                        )

                        if np.count_nonzero(feasible_indices) > 0:

                            # Defining moment.
                            min_index = np.min(np.argwhere(feasible_indices))

                            # Map to actual range
                            new_threshold = np.min(
                                active_signal_values
                            ) + min_index / bins * (
                                np.max(active_signal_values)
                                - np.min(active_signal_values)
                            )

                            # Identity the success of the method
                            have_new_value = True

                    elif self.threshold_method == "first local min enhanced":
                        # Prioritize global minima on the interval between the two
                        # largest peaks. If only a single peak exists, continue
                        # as in 'first local min'.

                        # Define tuning parameters for defining histograms,
                        # To allow edge values being peaks as well, add low
                        # numbers to the sides of the smooth histogram.
                        enriched_smooth_hist = np.hstack(
                            (
                                np.array([np.min(smooth_hist)]),
                                smooth_hist,
                                np.array([np.min(smooth_hist)]),
                            )
                        )

                        # Peak analysis.
                        # Find all peaks of the enriched smooth histogram,
                        # allowing end values to be identified as peaks.
                        peaks_pre, _ = find_peaks(enriched_smooth_hist)

                        # Only continue if at least one peak presents
                        if peaks_pre.shape[0] > 0:

                            # Relate the indices with the original histogram
                            # And continue analysis with the original one.
                            peaks_indices = peaks_pre - 1

                            # Cache the peak heights
                            peaks_heights = smooth_hist[peaks_indices]

                            # Check whether peaks have passed
                            if self.threshold_safety == "peaks passed":
                                self.pre_peaks_passed[label] = False
                                if (
                                    peaks_heights[0] < np.max(peaks_heights)
                                    and not self.peaks_passed[label]
                                ):
                                    self.pre_peaks_passed[label] = True

                            # Fetch the modulus of the second derivative for all peaks
                            peaks_2nd_derivative = np.absolute(
                                smooth_hist_2nd_derivative[peaks_indices]
                            )

                            # Track the feasibility of peaks. Initialize all peaks as feasible.
                            # Feasibility is considered only in the presence of multiple peaks.

                            # Determine feasibility. A peak is considered feasible if
                            # it is sufficiently far away from the global minimum.
                            min_height = np.min(smooth_hist)
                            peaks_are_distinct = (
                                peaks_heights - min_height
                                > 0.2 * np.max(peaks_heights - min_height)
                            )

                            # Determine feasibility. A peak is considered feasible if
                            # the modulus of the second derivative is sufficiently large,
                            # relatively to the most prominent peak.
                            peaks_have_large_2nd_der = (
                                peaks_2nd_derivative
                                > 0.2 * np.max(peaks_2nd_derivative)
                            )

                            peaks_are_feasible = np.logical_or(
                                peaks_are_distinct, peaks_have_large_2nd_der
                            )

                            # Cache the number of feasible peaks
                            num_feasible_peaks = np.count_nonzero(peaks_are_feasible)

                            # Determine the two feasible peaks with largest height.
                            # For this, first, restrict peaks to feasible ones.
                            feasible_peaks_indices = peaks_indices[peaks_are_feasible]
                            feasible_peaks_heights = peaks_heights[peaks_are_feasible]

                            # Sort the peak values from large to small, and restrict to
                            # the two largest
                            relative_max_indices = np.flip(
                                np.argsort(feasible_peaks_heights)
                            )[: min(2, num_feasible_peaks)]
                            max_indices = feasible_peaks_indices[relative_max_indices]
                            sorted_max_indices = np.sort(max_indices)

                            # Continue only if there exist two feasible peaks, and the peaks
                            # are of similar size TODO - see first local min enhanced.
                            if num_feasible_peaks > 1:

                                # Consider the restricted histogram
                                restricted_histogram = smooth_hist[
                                    np.arange(*sorted_max_indices)
                                ]

                                # Identify the global minimum as separator of signals
                                restricted_global_min_index = np.argmin(
                                    restricted_histogram
                                )

                                # Map the relative index from the restricted to the full
                                # (not-enriched) histogram.
                                global_min_index = (
                                    sorted_max_indices[0] + restricted_global_min_index
                                )

                                # Check whether the both peaks values actually are sufficiently
                                # different from the min value. Discard the value otherwise.
                                min_value = smooth_hist[global_min_index]
                                peaks_significant = (
                                    smooth_hist[max_indices[1]] - min_value
                                ) > 0.1 * (smooth_hist[max_indices[0]] - min_value)

                                # Determine the global minimum (in terms of signal values),
                                # determining the candidate for the threshold value
                                # Thresh mapped onto range of values
                                if peaks_significant:
                                    new_threshold = np.min(
                                        active_signal_values
                                    ) + global_min_index / bins * (
                                        np.max(active_signal_values)
                                        - np.min(active_signal_values)
                                    )

                                    # Identify success of the method
                                    have_new_value = True

                            # In case the above analysis has not been accepted (peaks not
                            # significant) there exists only one peak, perform an alternative
                            # step.
                            if not have_new_value and num_feasible_peaks == 0:
                                # Situation occurs when no peaks present.
                                pass
                            elif not have_new_value and (
                                num_feasible_peaks == 1 or not (peaks_significant)
                            ):

                                # Restrict analysis to the signal right from the most
                                # significant peak.
                                # restricted_hist = smooth_hist[max_indices[0] :]
                                restricted_hist_1st_derivative = (
                                    smooth_hist_1st_derivative[max_indices[0] :]
                                )
                                restricted_hist_2nd_derivative = (
                                    smooth_hist_2nd_derivative[max_indices[0] :]
                                )

                                # max_peak_value = np.max(restricted_hist)
                                max_peak_derivative = np.max(
                                    np.absolute(restricted_hist_1st_derivative)
                                )

                                # Restrict to positive 2nd derivative and small 1st derivative
                                max_value = 0.01 * max_peak_derivative
                                tol = 1e-6
                                feasible_restricted_indices = np.logical_and(
                                    -max_value < restricted_hist_1st_derivative,
                                    restricted_hist_2nd_derivative > tol,
                                )

                                if np.count_nonzero(feasible_restricted_indices) > 0:

                                    # Pick the first value in the feasible interval
                                    min_restricted_index = np.min(
                                        np.argwhere(feasible_restricted_indices)
                                    )

                                    # Relate to the full signal
                                    min_index = max_indices[0] + min_restricted_index

                                    # Thresh in the mapped onto range of values
                                    new_threshold = np.min(
                                        active_signal_values
                                    ) + min_index / bins * (
                                        np.max(active_signal_values)
                                        - np.min(active_signal_values)
                                    )

                                    # Identify success of the method
                                    have_new_value = True

                    elif self.threshold_method == "first local min otsu":
                        # Prioritize otsu on the interval between the two
                        # largest peaks. If only a single peak exists, continue
                        # otherwise with first local min

                        # To allow edge values being peaks as well, add low
                        # numbers to the sides of the smooth histogram.
                        enriched_smooth_hist = np.hstack(
                            (
                                np.array([np.min(smooth_hist)]),
                                smooth_hist,
                                np.array([np.min(smooth_hist)]),
                            )
                        )

                        # Peak analysis.
                        # Find all peaks of the enriched smooth histogram,
                        # allowing end values to be identified as peaks.
                        peaks_pre, _ = find_peaks(enriched_smooth_hist)

                        # Relate the indices with the original histogram
                        # And continue analysis with the original one.
                        peaks_indices = peaks_pre - 1

                        # Cache the peak heights
                        peaks_heights = smooth_hist[peaks_indices]

                        # Fetch the modulus of the second derivative for all peaks
                        peaks_2nd_derivative = np.absolute(
                            smooth_hist_2nd_derivative[peaks_indices]
                        )

                        # Track the feasibility of peaks. Initialize all peaks as feasible.
                        # Feasibility is considered only in the presence of multiple peaks.

                        # Determine feasibility. A peak is considered feasible if
                        # it is sufficiently far away from the global minimum.
                        min_height = np.min(smooth_hist)
                        peaks_are_distinct = peaks_heights - min_height > 0.2 * np.max(
                            peaks_heights - min_height
                        )

                        # Determine feasibility. A peak is considered feasible if
                        # the modulus of the second derivative is sufficiently large,
                        # relatively to the most prominent peak.
                        peaks_have_large_2nd_der = peaks_2nd_derivative > 0.2 * np.max(
                            peaks_2nd_derivative
                        )

                        peaks_are_feasible = np.logical_or(
                            peaks_are_distinct, peaks_have_large_2nd_der
                        )

                        # Cache the number of feasible peaks
                        num_feasible_peaks = np.count_nonzero(peaks_are_feasible)

                        # Determine the two feasible peaks with largest height.
                        # For this, first, restrict peaks to feasible ones.
                        feasible_peaks_indices = peaks_indices[peaks_are_feasible]
                        feasible_peaks_heights = peaks_heights[peaks_are_feasible]

                        # Sort the peak values from large to small, and restrict to the
                        # two largest
                        relative_max_indices = np.flip(
                            np.argsort(feasible_peaks_heights)
                        )[: min(2, num_feasible_peaks)]
                        max_indices = feasible_peaks_indices[relative_max_indices]
                        sorted_max_indices = np.sort(max_indices)

                        # Continue only if there exist two feasible peaks, and the peaks
                        # are of similar size TODO.
                        if num_feasible_peaks > 1:

                            thresh_otsu_smooth_pre = skimage.filters.threshold_otsu(
                                hist=smooth_hist
                            )
                            thresh_otsu_smooth = np.min(
                                active_signal_values
                            ) + thresh_otsu_smooth_pre / bins * (
                                np.max(active_signal_values)
                                - np.min(active_signal_values)
                            )
                            new_threshold = thresh_otsu_smooth

                            have_new_value = True

                        # In case the above analysis has not been accepted (peaks not
                        # significant) or there exists only one peak, perform an
                        # alternative step.
                        if not have_new_value and num_feasible_peaks == 0:
                            # Situation occurs when no peaks present.
                            pass
                        elif not have_new_value and (
                            num_feasible_peaks == 1 or not (peaks_significant)
                        ):

                            # Restrict analysis to the signal right from the most
                            # significant peak.
                            # restricted_hist = smooth_hist[max_indices[0] :]
                            restricted_hist_1st_derivative = smooth_hist_1st_derivative[
                                max_indices[0] :
                            ]
                            restricted_hist_2nd_derivative = smooth_hist_2nd_derivative[
                                max_indices[0] :
                            ]

                            # max_peak_value = np.max(restricted_hist)
                            max_peak_derivative = np.max(restricted_hist_1st_derivative)

                            # Restrict to positive 2nd derivative and small 1st derivative
                            max_value = 0.01 * max_peak_derivative
                            tol = 1e-6
                            feasible_restricted_indices = np.logical_and(
                                -max_value < restricted_hist_1st_derivative,
                                # np.logical_and(
                                #    -max_value < restricted_hist_1st_derivative,
                                #    restricted_hist_1st_derivative < -tol),
                                restricted_hist_2nd_derivative > tol,
                            )

                            if np.count_nonzero(feasible_restricted_indices) > 0:

                                # Pick the first value in the feasible interval
                                min_restricted_index = np.min(
                                    np.argwhere(feasible_restricted_indices)
                                )

                                # Relate to the full signal
                                min_index = max_indices[0] + min_restricted_index

                                # Thresh in the mapped onto range of values
                                new_threshold = np.min(
                                    active_signal_values
                                ) + min_index / bins * (
                                    np.max(active_signal_values)
                                    - np.min(active_signal_values)
                                )

                                have_new_value = True

                    # Setting values
                    if have_new_value:

                        # Clip the new value to the appropriate interval
                        bounded_threshold_value = np.clip(
                            new_threshold,
                            self.threshold_value_lower_bound[label],
                            self.threshold_value_upper_bound[label],
                        )

                        if self.threshold_safety == "none":
                            # Just accept the new value
                            self.threshold_value[label] = bounded_threshold_value

                        if self.threshold_safety == "peaks passed":
                            # Comes only in combination with threshold_method first
                            # local min enhanced.
                            assert self.threshold_method in ["first local min enhanced"]

                            hypothetically_marked_area = np.logical_and(
                                signal > bounded_threshold_value, label_mask
                            )
                            fraction = 0.1  # NOTE: hardcoded
                            if (
                                np.count_nonzero(hypothetically_marked_area)
                                > fraction * np.count_nonzero(label_mask)
                                and self.pre_peaks_passed[label]
                            ):
                                self.peaks_passed[label] = True

                            # Only allow new value as long a second peak has not arised
                            # or it is still smaller than the first one.
                            if not self.peaks_passed[label]:
                                self.threshold_value[label] = bounded_threshold_value

                        elif self.threshold_safety == "area":
                            # Require that the area identified by the threshold value does not
                            # outgo a certain fraction. Otherwise keep the previous value.
                            hypothetically_marked_area = np.logical_and(
                                signal > bounded_threshold_value, label_mask
                            )
                            fraction = 0.8  # NOTE: hardcoded
                            if np.count_nonzero(
                                hypothetically_marked_area
                            ) < fraction * np.count_nonzero(label_mask):
                                self.threshold_value[label] = bounded_threshold_value

                        elif self.threshold_safety == "ransac":
                            # TODO
                            pass
                            ## Perform RANSAC analysis based on the last 5 available values
                            # last_values = self.threshold_cache[label][-5:]
                            # if len(last_values) > 5:
                            #     all_values = np.array(last_values + [new_threshold])

                            #     ransac = RANSACRegressor()
                            #     ransac.fit(
                            #         np.arange(np.arange(all_values.shape[0])).reshape(
                            #             -1, 1
                            #         ),
                            #         all_values,
                            #     )
                            #     ransac_threshold.predict(
                            #         np.array([all_values.shape[0]])[:, np.newaxis]
                            #     )

                            #     # Apply user-defined boundary values
                            #     updated_threshold_value = np.clip(
                            #         ransac_threshold,
                            #         self.threshold_value_lower_bound[label],
                            #         self.threshold_value_upper_bound[label],
                            #     )

                            #    print("ransac", all_values, updated_threshold)
                            # else:
                            #    # Apply user-defined boundary values
                            #    updated_threshold_value = np.clip(
                            #        new_threshold,
                            #        self.threshold_value_lower_bound[label],
                            #        self.threshold_value_upper_bound[label],
                            #    )
                            #
                            ## Update threshold value
                            # self.threshold_value[label] = updated_threshold_value

                        elif self.threshold_safety == "min":

                            # Initialize tracker for minimal feasible values
                            # (inside the interval)
                            if not hasattr(self, "min_feasible_value"):
                                self.min_feasible_value = {}

                            # Do not allow larger values (inside the interval)
                            if (
                                self.threshold_value_lower_bound[label]
                                + 0.05
                                * (
                                    self.threshold_value_upper_bound[label]
                                    - self.threshold_value_lower_bound[label]
                                )
                                < bounded_threshold_value
                                < self.threshold_value_upper_bound[label]
                                - 0.05
                                * (
                                    self.threshold_value_upper_bound[label]
                                    - self.threshold_value_lower_bound[label]
                                )
                            ):
                                if label in self.min_feasible_value:
                                    self.min_feasible_value[label] = min(
                                        self.min_feasible_value[label],
                                        bounded_threshold_value,
                                    )
                                else:
                                    self.min_feasible_value[
                                        label
                                    ] = bounded_threshold_value

                                # Choose the lowest feasible value so far determined.
                                self.threshold_value[label] = self.min_feasible_value[
                                    label
                                ]

                    else:

                        if self.threshold_safety == "conservative":
                            self.threshold_value[
                                label
                            ] = self.threshold_value_upper_bound[label]

        if self.verbosity >= 1:
            print("Thresholding value", self.threshold_value)

        # Build the mask segment by segment.
        mask = np.zeros(self.labels.shape[:2], dtype=bool)
        for label in range(self.num_labels):
            label_mask = np.logical_and(self.labels == label, self.mask)
            threshold_mask = signal > self.threshold_value[label]
            effective_roi = np.logical_and(label_mask, threshold_mask)
            mask[effective_roi] = True

            # Cache threshold value of trust and all
            if np.count_nonzero(label_mask) > 0:
                mask_ratio = np.count_nonzero(effective_roi) / np.count_nonzero(
                    label_mask
                )
                if 0.1 < mask_ratio < 0.9:
                    self.threshold_cache[label].append(self.threshold_value[label])
                self.threshold_cache_all[label].append(self.threshold_value[label])

        return mask

    def _posterior(
        self, signal: np.ndarray, mask_prior: np.ndarray, img: np.ndarray
    ) -> np.ndarray:
        """
        Posterior analysis of signal for segmented geometry - only consider
        the cases in which it is meaningful.

        Args:
            signal (np.ndarray): (smoothed) signal
            mask_prior (np.ndarray): boolean mask marking prior regions
            img (np.ndarray): original difference of images

        Return:
            np.ndarray: boolean mask of trusted regions.
        """
        if not (
            self.apply_posterior
            and self.posterior_criterion == "value"
            and isinstance(self.posterior_threshold, np.ndarray)
        ):
            return super()._posterior(signal, mask_prior, img)

        # Initialize the output mask
        mask_posterior = np.zeros(signal.shape, dtype=bool)

        # Label the connected regions first
        labels_prior, num_labels_prior = skimage.measure.label(
            mask_prior, return_num=True
        )

        if self.verbosity >= 3:
            plt.figure("Posterior: Labeled regions from prior")
            plt.imshow(labels_prior)
            plt.show()

        # Investigate each labeled region separately; omit label 0, which corresponds
        # to non-marked area.
        for label in range(1, num_labels_prior + 1):

            # Fix one label
            labeled_region = labels_prior == label

            # Initialize acceptance
            # accept = False

            # Check the chosen criterion (is true)
            if self.posterior_criterion == "value":
                # Check whether there exist values in the segment, larger
                # than a provided critical value.

                roi = np.logical_and(labeled_region, self.mask)

                for geometry_label in range(self.num_labels):
                    geometry_label_roi = np.logical_and(
                        roi, self.labels == geometry_label
                    )
                    if (
                        np.count_nonzero(geometry_label_roi) > 0
                        and np.max(signal[geometry_label_roi])
                        > self.posterior_threshold[geometry_label]
                    ):
                        mask_posterior[labeled_region] = True
                        # mask_posterior[geometry_label_roi] = True
                        break

                if self.verbosity >= 3 and np.count_nonzero(geometry_label_roi) > 0:
                    print(
                        f"""Posterior: Label {label},
                        geometry_label {geometry_label},
                        max value: {np.max(signal[geometry_label_roi])}."""
                    )

        return mask_posterior
