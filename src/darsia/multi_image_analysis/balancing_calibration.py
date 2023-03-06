"""
Module collecting several calibration tools,
and in particular objective functions for calibration
in ConcentrationAnalysis.calibrate_balancing().

"""

import abc
from itertools import combinations
from pathlib import Path

from typing import Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

import darsia


class AbstractBalancingCalibration:
    """
    Abstract class for defining an objective function
    to be called in ConcentrationAnalysis.calibrate_balancing().

    """

    @abc.abstractmethod
    def optimize_balancing(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        relative_times: list[float],
        options: dict,
    ):
        """
        Abstract method to define an objective function.

        Returns:
            callable: objective function.

        """
        pass

    def update_balancing_for_calibration(
        self, parameters: np.ndarray, options: dict
    ) -> None:
        """
        Wrapper for updating the balancing (provided as a model),
        depending on whether it is a single model or a combined model.

        Args:
            parameters (np.ndarray): model parameters,
            options (dict): further tuning parameters and extra info.

        """
        # Check whether the balancing is part of a combined model,
        # and possibly determine position of the model
        if isinstance(self.balancing, darsia.CombinedModel):
            pos_balancing = options.get("balancing position")
            self.balancing.update_model_parameters(parameters, pos_balancing)
        else:
            self.balancing.update_model_parameters(parameters)

    def calibrate_balancing(
        self,
        images: Union[list[darsia.Image], list[darsia.GeneralImage]],
        options: dict,
    ) -> bool:
        """
        Utility for calibrating the balancing used in darsia.ConcentrationAnalysis.

        NOTE: Require to combine darsia.ConcentrationAnalysis with a calibration
        model mixin via multiple inheritance.

        Args:
            images (list of darsia.Image): calibration images
            options (dict): container holding tuning information for the numerical
                calibration routine

        Returns:
            bool: success of the calibration study.

        """
        # Apply the same steps as in __call__ to all images, until before balancing is applied.

        # Prepare calibration and determine fixed data
        images_diff = [self._subtract_background(img) for img in images]

        # Extract monochromatic version and take difference wrt the baseline image
        images_signal = [self._extract_scalar_information(diff) for diff in images_diff]

        # Clean signal
        images_clean_signal = [self._clean_signal(signal) for signal in images_signal]

        # The missing steps: balancing, restoration, and the model are not applied.
        # Instead, the balancing and a lightweight restoration will be part of
        # a calibration routine.
        if not hasattr(self, "optimize_balancing"):
            raise NotImplementedError(
                """The concentration analysis is not equipped with a calibration model
                for balancing."""
            )
        opt_result, opt_success = self.optimize_balancing(images_clean_signal, options)

        # Report results
        if opt_success:
            print(
                f"Calibration successful with obtained model parameters {opt_result}."
            )
        else:
            print("Calibration not successful.")

        # Update model
        self.update_balancing_for_calibration(opt_result, options)

        return opt_success


class ContinuityBasedBalancingCalibrationMixin(AbstractBalancingCalibration):
    """
    Calibration balancing based on reducing jumps over interfaces
    for a given labeled image. Has to be combined with
    ConcentrationAnalysis.

    """

    # ! ---- Main  routine

    def optimize_balancing(
        self,
        images: list[np.ndarray],
        options: dict,
    ) -> tuple[np.ndarray, bool]:
        """
        Define objective function such that the root is the min.

        Args:
            input_images (list of np.ndarray): input for _convert_signal
            images_diff (list of np.ndarray): plain differences wrt background image
            relative_times (list of float): times
            options (dict): dictionary with objective value, here the injection rate

        Returns:
            np.ndarray: optimized model parameters
            bool: success flag

        """

        # ! ---- Safety check
        if not isinstance(self.balancing, darsia.HeterogeneousLinearModel):
            raise NotImplementedError(
                """Balancing optimization only
                implemented for darsia.HeterogeneousModel."""
            )

        # ! ---- Perform (expensive) setup and cache

        # Find thick contours
        self._scan_labeled_image(options)

        # Find the balancing aiming by investigating interfaces
        # and populate the answer to the segments.
        optimized_scaling = self._least_squares_minimizer(images, options)

        # Enhance parameters with the same offset as already set
        optimized_offset = self.balancing._offset

        # Collect results
        optimized_parameters = np.hstack((optimized_scaling, optimized_offset))
        success = True

        return optimized_parameters, success

    # ! ---- Auxiliary routines.

    def _scan_labeled_image(self, options: dict) -> None:
        """
        Find thick contours of labeled image.

        Args:
            options (dict): dictionary with possibility to tune
                the definition and detection of a thick contour.

        """

        # Cache label info
        self.unique_labels = np.unique(self.labels)
        self.num_labels = len(self.unique_labels)

        # Check if required information is stored already
        segmentation_contour_path = Path(
            options.get("contour path", "cache/contours.npy")
        )

        if segmentation_contour_path.exists():
            # Read data from cache
            cached_segmentation_contour = np.load(
                segmentation_contour_path, allow_pickle=True
            ).item()
            self.contour_mask = cached_segmentation_contour["contour_mask"]
            self.label_couplings = cached_segmentation_contour["label_couplings"]
            self.coupling_strength = cached_segmentation_contour["coupling_strength"]

        else:
            # Generate data from scratch

            # Setup parameters
            contour_thickness = options.get("contour_thickness", 10)
            contour_overlap_threshold = options.get("contour_overlap_threshold", 1000)

            # Final assignment of contours
            contour_mask = {}
            for label in self.unique_labels:
                contour_mask[label] = self._labeled_mask_to_contour_mask(
                    self.labels == label, contour_thickness
                )

            # User interact. Provide possibility to hardcode label_couplings to be optimized.
            excluded_couplings = options.get("exclude couplings", [])
            only_couplings = options.get("only couplings", None)

            # Determine common intersection of thick contours shared by neighboring segments
            # Find relevant couplings of masked labels.
            coupling_strength = []
            label_couplings = []
            for label1 in self.unique_labels:
                for label2 in self.unique_labels:

                    # Consider only directed pairs
                    if label1 < label2:

                        coupling = (label1, label2)

                        # User interaction. Possibilities:
                        # 1. Restrict to a provided list of label couplings.
                        # 2. Exclude particular couplings.
                        if only_couplings is not None:
                            if (
                                coupling not in only_couplings
                                and tuple(reversed(coupling)) not in only_couplings
                            ):
                                continue
                        elif coupling in excluded_couplings:
                            continue

                        # Check if labeled regions share significant part of contour
                        elif (
                            np.count_nonzero(
                                np.logical_and(
                                    contour_mask[label1], contour_mask[label2]
                                )
                            )
                            > contour_overlap_threshold
                        ):
                            # Track coupling
                            label_couplings.append(coupling)

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
            segmentation_contour_path.parents[0].mkdir(parents=True, exist_ok=True)
            np.save(
                segmentation_contour_path,
                {
                    "contour_mask": self.contour_mask,
                    "label_couplings": self.label_couplings,
                    "coupling_strength": self.coupling_strength,
                },
            )

    # Define thick contours for all labels
    def _labeled_mask_to_contour_mask(
        self, labeled_mask: np.ndarray, thickness: int
    ) -> np.ndarray:
        """
        # Auxiliary method in _scan_labeled_image.

        Starting from a boolean array identifying a region, find
        the contours with a user-defined bandwidth.

        Args:
            labeled_mask (np.ndarray): boolean array identifying a connected region.
            thickness (int): contour thickness obtained through dilation

        Returns:
            np.ndarray: boolean array identifying a band width of the contours

        """
        # Determine the contours of the labeled mask
        contours, _ = cv2.findContours(
            skimage.img_as_ubyte(labeled_mask),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE,
        )

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

    def _least_squares_minimizer(
        self, images: list[np.ndarray], options: dict
    ) -> np.ndarray:
        """

        Args:
            images (list of np.ndarray): signals which in principle went
                through ConcentrationAnalysis.__call__() including
                ConcentrationAnalysis._convert_signal()

        Returns:
            np.ndarray: suggested heterogeneous scaling

        """
        # Strategy: Quantify the discontinuity jump of the signal at
        # all boundaries between different segments. These are stored
        # in interace_ratio_container. These ratios will be used to
        # define a segment-wise scaling factor. To transfer the infos
        # on interfaces to segments a least-squares problem is solved.

        # ! ---- Initialization

        # Initialize collection of interface ratios with empty lists,
        # as well as flag of trustable information by identifying
        # none of the coupling as trustworthy.
        interface_ratio_container: dict = {}
        trustable_summary = {}
        for coupling in self.label_couplings:
            interface_ratio_container[coupling] = []
            trustable_summary[coupling] = False

        # ! ---- Interface analysis

        # Read tuning parameters from options
        mean_thresh = options.get("mean thresh")
        median_disk_radius = options.get("median disk radius")

        # From the labels determine the region properties
        regionprops = skimage.measure.regionprops(self.labels)

        # Find a suitable segmentation_scaling vector for each separate image.
        for signal in images:

            # Define the segment-wise median
            median = np.zeros(signal.shape[:2], dtype=np.uint8)
            for regionprop in regionprops:

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

                # Extend to full image
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
                # TODO Uniformly split contours in N segments and determine
                # jump in an integral sense, instead of throwing away the spatial info.
                mean1 = np.mean(median[roi1])
                mean2 = np.mean(median[roi2])

                # Check whether this result can be trusted - require sufficient signal.
                trustable[coupling] = min(mean1, mean2) >= mean_thresh

                # Define the ratio / later scaling - if value not trustable (for
                # better usability) choose ratio equal to 1
                interface_ratio[coupling] = mean1 / mean2 if trustable[coupling] else 1

            # Keep only trustable information and collect.
            for coupling, ratio in interface_ratio.items():
                if trustable[coupling]:
                    interface_ratio_container[coupling].append(ratio)
                    trustable_summary[coupling] = True

        # ! ---- Summarize analysis

        # Reduce multiple data points per interface to single data point
        # by simply taking the mean.
        summarized_interface_ratio = {}
        for coupling in self.label_couplings:
            if trustable_summary[coupling]:
                summarized_interface_ratio[coupling] = np.mean(
                    np.array(interface_ratio_container[coupling])
                )

            else:
                # Aim at equalizing scaling parameters
                summarized_interface_ratio[coupling] = 1.0

        # Fix the lowest trusted label - requires that the trusted interfaces
        # define a connected graph.
        lowest_trusted_label = np.min(
            [coupling[0] for coupling in trustable_summary.keys()]
        )

        # ! ---- Assembly: Define conditions combining the interfaces

        # Based on the interface ratios, build a linear (overdetermined) system,
        # which characterizes the optimal scaling.
        matrix = np.zeros((0, self.num_labels), dtype=float)
        rhs = np.zeros((0, 1), dtype=float)
        num_constraints = 0

        # Four main contributions:
        # 1. Constraint: Add weight on fixing one label
        # 2. Constraint: Relate label groups
        # 3. Regularization: Weak
        # 4. Interface ratios

        ############################################################################
        # 1. Fix reference labels, here chosen to be the label with lowest label id
        for label in [lowest_trusted_label]:
            basis_vector = np.zeros((1, self.num_labels), dtype=float)
            basis_vector[0, label] = 1
            matrix = np.vstack((matrix, basis_vector))
            rhs = np.vstack((rhs, np.array([[1]])))
            num_constraints += 1

        ############################################################################
        # 2. Add weak constraint and connect similar labels.
        label_groups = options.get("label groups", None)
        similarity_weight = 1  # NOTE: Other values could add more weight
        for group in label_groups:
            if len(group) > 1:
                for coupling in list(combinations(group, 2)):
                    label1, label2 = coupling
                    similarity_balance = np.zeros((1, self.num_labels), dtype=float)
                    similarity_balance[0, label1] = similarity_weight
                    similarity_balance[0, label2] = -similarity_weight
                    matrix = np.vstack((matrix, similarity_balance))
                    rhs = np.vstack((rhs, np.array([0])))
                    num_constraints += 1

        ############################################################################
        # 3. Add a weak constraint on all parameters to stay close to 1
        # removing possible singularity.
        regularization_parameter = 0.0
        for label in self.unique_labels:
            basis_vector = np.zeros((1, self.num_labels), dtype=float)
            basis_vector[0, label] = regularization_parameter
            matrix = np.vstack((matrix, basis_vector))
            rhs = np.vstack((rhs, np.array([[regularization_parameter]])))
            num_constraints += 1

        ############################################################################
        # 4. Add trusted couplings and main information on interface ratios..
        for coupling in self.label_couplings:
            label1, label2 = coupling
            scaling_balance = np.zeros((1, self.num_labels), dtype=float)
            scaling_balance[0, label1] = summarized_interface_ratio[coupling]
            scaling_balance[0, label2] = -1
            matrix = np.vstack((matrix, scaling_balance))
            rhs = np.vstack((rhs, np.array([0])))

        # Scale matrix and rhs with coupling strength to prioritize significant interfaces
        matrix[num_constraints:, :] = np.matmul(
            np.diag(self.coupling_strength), matrix[num_constraints:, :]
        )
        rhs[num_constraints:] = np.matmul(
            np.diag(self.coupling_strength), rhs[num_constraints:]
        )

        # ! ---- Obtain parameters as least squares solution

        # Determine suitable scaling by solving the overdetermined system using
        # a least-squares approach.
        segmentation_scaling = np.linalg.lstsq(matrix, np.ravel(rhs), rcond=None)[0]

        # ! ---- User interaction

        verbosity = options.get("verbosity", 0)
        if verbosity > 0:
            # Print the solution
            print(f"Computed segmentation scaling: {segmentation_scaling}")
            plt.figure()
            plt.imshow(self.labels)
            scaling_image = np.zeros(self.labels.shape[:2], dtype=float)
            for label in range(self.num_labels):
                mask = self.labels == label
                scaling_image[mask] = segmentation_scaling[label]
            plt.figure()
            plt.imshow(scaling_image)
            plt.show()

        return segmentation_scaling
