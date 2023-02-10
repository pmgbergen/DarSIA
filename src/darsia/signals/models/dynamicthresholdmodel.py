"""
Module containing dynamic thresholding models.
"""

import abc
from typing import Optional, Union

import numpy as np
import scipy.ndimage as ndi
import skimage
from scipy.signal import find_peaks

import darsia


class HistogrammBasedThresholding:
    def __init__(self) -> None:
        # Define tuning parameters for defining histograms,
        # and smooth them. NOTE: They should be in general chosen
        # tailored to the situation. However, these values should
        # also work for most cases.
        self._bins = 200
        self._sigma = 10

    def __call__(
        self, signal: np.ndarray, roi: np.ndarray
    ) -> tuple[Optional[float], bool]:
        """
        Dynamic thresholding of signal in provided roi.

        Args:
            signal (np.ndarray): signal
            roi (np.ndarray): boolean array identifying the considered region

        Returns:
            float: new threshold value.
            bool: identifier for success.
        """
        # Reduce the signal to the effective mask
        active_signal_values = np.ravel(signal)[np.ravel(roi)]

        # Continue by analysing the histogram corresponding to the active signal.

        # Smooth the histogram of the signal, and compute derivates
        hist = ndi.gaussian_filter1d(
            np.histogram(active_signal_values, bins=self._bins)[0],
            sigma=self._sigma,
        )

        # Run provided analysis
        new_threshold, have_new_value = self._analysis(active_signal_values, hist)
        return new_threshold, have_new_value

    @abc.abstractmethod
    def _analysis(
        self, active_signal_values: np.ndarray, hist: np.ndarray
    ) -> tuple[Optional[float], bool]:
        """
        Abstract method for some histogram analysis.

        Args:
            active_signal_values (np.ndarray): 1d array with all signal values
            hist (np.ndarray): 1d array, histogram

        Returns:
            float, optional: determined threshold value
            bool: flag controlling whether the analysis has been successful.

        """
        pass


class StandardOtsu(HistogrammBasedThresholding):
    """
    Wrapper for standard Otsu thresholding.

    """

    def _analysis(
        self, active_signal_values: np.ndarray, hist: np.ndarray
    ) -> tuple[Optional[float], bool]:
        """
        Standard Otsu method for provided histogram.

        Args:
            active_signal_values (np.ndarray): 1d array with all signal values
            hist (np.ndarray): 1d array, histogram

        Returns:
            float, optional: determined threshold value
            bool: flag controlling whether the analysis has been successful.

        """
        otsu_index = skimage.filters.threshold_otsu(hist=hist)
        otsu_threshold = np.min(active_signal_values) + otsu_index / self._bins * (
            np.max(active_signal_values) - np.min(active_signal_values)
        )

        return otsu_threshold, True


class TwoPeakHistogrammAnalysis(HistogrammBasedThresholding):
    """
    Class for histogramm analysis aiming at separating two signal peaks.

    """

    def _analysis(
        self, active_signal_values: np.ndarray, hist: np.ndarray
    ) -> tuple[Optional[float], bool]:
        """
        Histogram analysis.

        Return first local minimum after descending the first
        peak, when moving from left to right. Provide global
        minima if existing and local true minima as well.

        Main goal:
            Find separator of signal peaks.

        Strategy:
            1. Determine the minimum between two significant peaks.
            2. If only one peak present or peaks not significant,
                determine something like a "first local min" based
                on relative criteria.

        Args:
            active_signal_values (np.ndarray): 1d array with all signal values
            hist (np.ndarray): 1d array, histogram

        Returns:
            float, optional: determined threshold value
            bool: flag controlling whether the analysis has been successful.
        """
        # Initialize output
        new_threshold = None
        have_new_value = False

        hist_1st_derivative = np.gradient(hist)
        hist_2nd_derivative = np.gradient(hist_1st_derivative)

        ## For tuning the parameters, plot the histogram and its derivatives
        # if self.verbosity >= 2:
        #    plt.figure("Histogram analysis")
        #    plt.plot(
        #        np.linspace(
        #            np.min(active_signal_values),
        #            np.max(active_signal_values),
        #            hist.shape[0],
        #        ),
        #        hist,
        #        label=f"Label {label}",
        #    )
        #    plt.legend()

        # Prioritize global minima on the interval between the two
        # largest peaks. If only a single peak exists, continue
        # as in 'first local min'.

        # Define tuning parameters for defining histograms,
        # To allow edge values being peaks as well, add low
        # numbers to the sides of the smooth histogram.
        enriched_hist = np.hstack(
            (
                np.array([np.min(hist)]),
                hist,
                np.array([np.min(hist)]),
            )
        )

        # Peak analysis.
        # Find all peaks of the enriched smooth histogram,
        # allowing end values to be identified as peaks.
        peaks, _ = find_peaks(enriched_hist)

        ##################################################################
        # Only continue if at least one peak is present
        if len(peaks) > 0:

            # Relate the indices with the original histogram
            # And continue analysis with the original one.
            peaks_indices = peaks - 1

            # Cache the peak heights
            peaks_heights = hist[peaks_indices]

            #            # Check whether peaks have passed
            #            if self.threshold_safety == "peaks passed":
            #                self.pre_peaks_passed[label] = False
            #                if (
            #                    peaks_heights[0] < np.max(peaks_heights)
            #                    and not self.peaks_passed[label]
            #                ):
            #                    self.pre_peaks_passed[label] = True

            # Fetch the modulus of the second derivative for all peaks
            peaks_2nd_derivative = np.absolute(hist_2nd_derivative[peaks_indices])

            ##################################################################
            # Track the feasibility of peaks. Initialize all peaks as feasible.
            # Feasibility is considered only in the presence of multiple peaks.

            # Determine feasibility. A peak is considered feasible if
            # it is sufficiently far away from the global minimum.
            min_height = np.min(hist)
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

            ##################################################################
            # Determine the two feasible peaks with largest height.
            # For this, first, restrict peaks to feasible ones.
            feasible_peaks_indices = peaks_indices[peaks_are_feasible]
            feasible_peaks_heights = peaks_heights[peaks_are_feasible]

            # Sort the peak values from large to small, and restrict to
            # the two largest
            relative_max_indices = np.flip(np.argsort(feasible_peaks_heights))[
                : min(2, num_feasible_peaks)
            ]
            max_indices = feasible_peaks_indices[relative_max_indices]
            sorted_max_indices = np.sort(max_indices)

            ##################################################################
            # Continue only if there exist two feasible peaks, and the peaks
            # are of similar size.
            if num_feasible_peaks > 1:

                ##################################################################
                # Determine whether the two peaks are significant.

                # Consider the restricted histogram
                restricted_hist = hist[np.arange(*sorted_max_indices)]

                # Identify the global minimum as separator of signals
                restricted_global_min_index = np.argmin(restricted_hist)

                # Map the relative index from the restricted to the full
                # (not-enriched) histogram.
                global_min_index = sorted_max_indices[0] + restricted_global_min_index
                min_value = hist[global_min_index]

                # Check whether the both peaks values actually are sufficiently
                # and relatively different from the min value. Discard the value
                # otherwise.
                peaks_significant = (hist[max_indices[1]] - min_value) > 0.1 * (
                    hist[max_indices[0]] - min_value
                )

                ##################################################################
                # Determine the global minimum (in terms of signal values),
                # determining the candidate for the threshold value
                # Thresh mapped onto range of values

                # Cache some of the general variables needed for a two peak analysis.
                self._hist = hist
                self._restricted_hist = restricted_hist
                self._sorted_max_indices = sorted_max_indices

                # Apply specific analysis.
                threshold_index = self._two_peak_analysis()

                if peaks_significant:
                    new_threshold = np.min(
                        active_signal_values
                    ) + global_min_index / self._bins * (
                        np.max(active_signal_values) - np.min(active_signal_values)
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
                # restricted_hist = hist[max_indices[0] :]
                restricted_hist_1st_derivative = hist_1st_derivative[max_indices[0] :]
                restricted_hist_2nd_derivative = hist_2nd_derivative[max_indices[0] :]

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

                # Only continue if non-empty
                if np.count_nonzero(feasible_restricted_indices) > 0:

                    # Pick the first value in the feasible interval
                    min_restricted_index = np.min(
                        np.argwhere(feasible_restricted_indices)
                    )

                    # Relate to the full signal
                    min_index = max_indices[0] + min_restricted_index

                    # Threshold value in the mapped onto range of values
                    new_threshold = np.min(
                        active_signal_values
                    ) + min_index / self._bins * (
                        np.max(active_signal_values) - np.min(active_signal_values)
                    )

                    # Identify success of the method
                    have_new_value = True

        return new_threshold, have_new_value

    @abc.abstractmethod
    def _two_peak_analysis(self) -> int:
        """
        Abstract method for determining the index corresponding
        to the full histogramm, to define the threshold value.

        Returns:
            int: index in histogram self._hist corresponding
                to considered threshold value.

        """
        pass


class GlobalMinTwoPeakHistogrammAnalysis(TwoPeakHistogrammAnalysis):
    """
    Class defining a two peak analysis for dynamically
    determining a threshold parameter used on a global
    minimum analysis of the signal histogram.

    """

    def _two_peak_analysis(self) -> int:
        """
        Method to determine the threshold parameter based
        on the global minimum attained between two peaks,
        i.e., operating on the restricted histogram.

        Returns:
            int: index in histogram self._hist corresponding
                to considered threshold value.

        """
        # Identify the global minimum as separator of signals
        restricted_global_min_index = np.argmin(self._restricted_hist)

        # Map the relative index from the restricted to the full
        # (not-enriched) histogram.
        global_min_index = self._sorted_max_indices[0] + restricted_global_min_index

        return global_min_index


class OtsuTwoPeakHistogrammAnalysis(TwoPeakHistogrammAnalysis):
    """
    Class defining a two peak analysis for dynamically
    determining a threshold parameter used on a Otsu
    analysis of the signal histogram.

    """

    def _two_peak_analysis(self) -> int:
        """
        Method to determine the threshold parameter based
        on the global minimum attained between two peaks,
        i.e., operating on the restricted histogram.

        Returns:
            int: index in histogram self._hist corresponding
                to considered threshold value.

        """
        otsu_index = skimage.filters.threshold_otsu(hist=self._hist)

        return otsu_index


class DynamicThresholdModel(darsia.StaticThresholdModel):
    """
    Class for dynamic thresholding.
    """

    def __init__(
        self,
        method: Optional[str] = None,
        threshold_lower: Optional[Union[float, list[float]]] = None,
        threshold_upper: Optional[Union[float, list[float]]] = None,
        labels: Optional[np.ndarray] = None,
        key: str = "",
        **kwargs,
    ) -> None:
        """
        Constructor of DynamicThresholdModel.

        Args:
            method (str): method name
            threshold_lower (float or list of float): lower threshold value boundary
            threshold_upper (float or list of float): upper threshold value boundary
            labels (array): labeled domain
            key (str): prefix for options
        """
        # Determine threshold strategy and lower and upper bounds.
        threshold_method = (
            kwargs.get(key + "threshold method", "tailored global min")
            if method is None
            else method
        )
        threshold_value_lower_bound: Union[float, list] = (
            kwargs.get(key + "threshold value min", 0.0)
            if threshold_lower is None
            else threshold_lower
        )
        threshold_value_upper_bound: Optional[Union[float, list]] = (
            kwargs.get(key + "threshold value max", None)
            if threshold_upper is None
            else threshold_upper
        )

        # Call the constructor if the static threshold model.
        super().__init__(threshold_lower, threshold_upper, labels)

        # Identify method and define corresponding thresholding object
        if threshold_method == "tailored global min":
            self.method = GlobalMinTwoPeakHistogrammAnalysis()
        elif threshold_method == "tailored otsu":
            self.method = OtsuTwoPeakHistogrammAnalysis()
        elif threshold_method == "otsu":
            self.method = StandardOtsu()
        else:
            raise ValueError(f"Method {method} not supported for dynamic thresholding")

        # Identify the provided thresholds as fixed bounds, allowing to modify the
        # threshold values.
        self._threshold_lower_bound = (
            self._threshold_lower.copy() if self._threshold_lower is not None else None
        )
        self._threshold_upper_bound = (
            self._threshold_upper.copy() if self._threshold_upper is not None else None
        )

        # Only allow for lower threshold values. Deactive upper thresholding
        self._threshold_upper = None

        # Verbosity
        self.verbosity = kwargs.get("verbosity", 0)

    def __call__(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Main method. Adapt thresholds and apply thresholding.

        Args:
            img (np.ndarray): image
            mask (np.ndarray, optional): boolean mask of interest

        Returns:
            np.ndarray: booelean mask identifying signal according to current threshold values.
        """
        self.calibrate([img], mask)
        return super().__call__(img, mask)

    def calibrate(self, img: list[np.ndarray], mask: Optional[np.ndarray]) -> None:
        """
        Adapt threshold values using a dynamic strategies.
        """
        if self._is_homogeneous:
            self._calibrate_homogeneous(img, mask)
        else:
            self._calibrate_heterogeneous(img, mask)

    def _calibrate_homogeneous(
        self, img: list[np.ndarray], mask: Optional[np.ndarray]
    ) -> None:
        """
        Adapt threshold values globally using a dynamic stratgey.

        Args:
            img (list of np.ndarray): image(s)
        """
        raise NotImplementedError(
            "Currently the dynamic thresholding is only implemented for heterogeneous media."
        )
        pass

    def _calibrate_heterogeneous(
        self, img: list[np.ndarray], mask: Optional[np.ndarray]
    ) -> None:
        """
        Adapt threshold values for each label using a dynamic strategy.

        Args:
            img (list of np.ndarray): image(s)
        """
        # Extract main image for calibration
        assert len(img) == 1
        signal = img[0]

        for label_count, label in enumerate(np.unique(self._labels)):

            # Determine mask of interest, i.e., consider single label,
            # the interval of interest and the provided mask.
            label_mask = self._labels == label
            effective_mask = (
                label_mask if mask is None else np.logical_and(label_mask, mask)
            )

            # Only continue if mask not empty
            if np.count_nonzero(effective_mask) > 0:

                new_threshold, have_new_value = self.method(signal, effective_mask)

                # Setting values
                if have_new_value:

                    # Clip the new value to the appropriate interval
                    bounded_threshold_value = np.clip(
                        new_threshold,
                        self._threshold_lower_bound[label_count],
                        self._threshold_upper_bound[label_count],
                    )

                    # Set self.threshold_lower is used in self.__call__ as effective lower
                    # threshold value
                    self._threshold_lower[label_count] = bounded_threshold_value

        # Display the threshold value
        if self.verbosity >= 1:
            print(f"Threshold value: {self._threshold_lower}")
