"""
Module containing a class for interpreting DICOM images.

"""


from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pydicom

import darsia

# TODO extend to 3d


class ImageReader:
    """
    Image reader converting input to darsia.Image format.

    """

    def from_dicom(
        self, path: list[Path], options: dict
    ) -> Union[darsia.Image, list[darsia.Image]]:
        """
        Initialization of DICOMImage by reading the DICOM format and translating it
        into 2d darsia format.

        Args:
            path (Path): path to dicom stacks
            options (dict): options for data reduction

        Returns:
            darsia.Image or list of such: 2d images

        """
        # PYDICOM specific tags for addressing dicom data from dicom files
        tag_position = pydicom.tag.BaseTag(0x00200032)
        tag_rows = pydicom.tag.BaseTag(0x00280010)
        tag_cols = pydicom.tag.BaseTag(0x00280011)
        tag_pixel_size = pydicom.tag.BaseTag(0x00280030)
        tag_slice_thickness = pydicom.tag.BaseTag(0x00180050)
        tag_acquisition_date = pydicom.tag.BaseTag(0x00080022)
        tag_acquisition_time = pydicom.tag.BaseTag(0x00080032)

        # Good to have for future safety checks.
        tag_number_slices = pydicom.tag.BaseTag(0x00540081)
        tag_number_time_slices = pydicom.tag.BaseTag(0x00540101)

        # ! ---- 1. Step: Get overview of number of images

        # Convert srcfile to a list of just a string.
        if not isinstance(path, list):
            path = [path]

        # Initialize arrays for DICOM data
        pixel_data = []  # actual signal
        y_positions = []  # the position of the sample, relevant for sorting
        rows = []  # image size in x-direction
        cols = []  # image size in z-direction
        acq_times = []  # acquisition time of samples
        acq_date_time_tuples = []  # acquision (date, time) of samples
        # voxel_size = []     # dimensions of voxels in meters # TODO rm or allow for different dimensions?

        # Consider each dicom file by itself (each file is a sample and has to
        # be sorted into the larger picture.
        for p in path:

            # Read the dicom sample
            dataset = pydicom.dcmread(p)

            # Extract the actual signal
            pixel_data.append(dataset.pixel_array)

            # Extract y positions for each frame
            # NOTE: Positions are stored as (x,z,y) coordinate
            y_positions.append(np.array(dataset[tag_position].value)[2])

            # Extract number of rows for each frame
            rows.append(dataset[tag_rows].value)

            # Extract number of columns for each frame
            cols.append(dataset[tag_cols].value)

            # Extract acquisition time for each frame
            acq_date_time_tuples.append(
                dataset[tag_acquisition_date].value
                + dataset[tag_acquisition_time].value
            )
            acq_times.append(dataset[tag_acquisition_time].value)
            number_times = dataset[tag_number_time_slices].value

            # Extract pixel size - assume constant for all files - and assume info in mm
            conversion_mm_to_m = 10 ** (-3)
            dimensions = np.zeros(3)
            dimensions[:2] = (
                np.array(dataset[tag_pixel_size].value) * conversion_mm_to_m
            )
            dimensions[2] = dataset[tag_slice_thickness].value * conversion_mm_to_m

        # ! ---- 2. Step: Sort input wrt datetime

        # Convert data to numpy arrays
        pixel_data = np.array(pixel_data)
        y_positions = np.array(y_positions)
        acq_times = np.array(acq_times)
        unique_acq_datetimes = set(acq_date_time_tuples)
        acq_date_time_tuples = np.array(acq_date_time_tuples, dtype=object)

        # Convert date time strings to datetime format. Expect a data format
        # for the input as ('20210222133538.000'), where the first entry
        # represent the date Feb 22, 2021, and the second entry represents
        # the time 13 hours, 35 minutes, and 38 seconds. Sort the list of
        # datetimes, starting with earliest datetime.
        def datetime_conversion(date_time: str) -> datetime:
            return datetime.strptime(date_time[:12], "%Y%m%d%H%M%S")

        acq_datetimes = [
            datetime_conversion(date_time) for date_time in unique_acq_datetimes
        ]
        acq_datetimes = sorted(acq_datetimes)

        # User input. Order of dimensions. Default is (x, y, z)
        # assigned to (depth, width, height).
        order_of_dimensions = options.get("order of dimensions", None)
        if order_of_dimensions is None:
            reordering = [1, 2, 0]
        else:
            reordering = [-1, -1, -1]
            raise NotImplementedError("Other orderings not implemented, yet.")

        # TODO Adapt comment
        # Apply similar operations for global data: reordering, and reduction from 3d to 2d
        dimensions = dimensions[np.array([1, 2, 0])]
        # dimensions = dimensions[:2]

        # Group data into the time frames.
        sorted_pixel_data = dict()
        for i, date_time in enumerate(unique_acq_datetimes):

            # Fetch all frames corresponding to the current time frame
            time_frame = np.argwhere(acq_date_time_tuples == date_time).flatten()

            # For each time frame, sort data in y-direction (stored as third component)
            sorted_time_frame = np.argsort(y_positions[time_frame])

            # Apply sorting to the time frames
            sorted_time_frame = time_frame[sorted_time_frame]

            # Fetch the corresponding, sorted pixel data
            sorted_pixel_data_cache = pixel_data[sorted_time_frame, :, :]

            # Resort the indices: y,z,x -> x,y,z
            # TODO generalize, see above.
            sorted_pixel_data_cache = np.moveaxis(
                sorted_pixel_data_cache, [0, 1, 2], [1, 2, 0]
            )

            # Restrict to a ROI if provided and reduce the dimension - sum over z axis
            # TODO Move this to an external routine - project_3d_to_2d(data, axis=2)
            # TODO darsia.signal.reduction.dimension
            roi = options.get("roi", None)
            if roi is None:
                sorted_2d_pixel_data = np.sum(sorted_pixel_data_cache, axis=2)
            else:
                sorted_2d_pixel_data = np.sum(
                    sorted_pixel_data_cache[:, :, roi], axis=2
                )

            # TODO Rm
            shape = sorted_pixel_data_cache.shape

            # Store
            # TODO allow for general format, also 3d.
            sorted_pixel_data[
                datetime_conversion(date_time)
            ] = sorted_2d_pixel_data.copy()

        # ! ---- 3. Convert to darsia.Image
        # TODO include orientation? assignment of (x,y,z) to (width, height, depth)
        # shape = sorted_2d_pixel_data.shape
        width = dimensions[0] * shape[0]
        height = dimensions[1] * shape[1]
        depth = dimensions[2] * shape[2]
        images = [
            darsia.Image(
                sorted_pixel_data[time], width=width, height=height, timestamp=time
            )
            for time in acq_datetimes
        ]

        return images

    def from_vtu(
        self, path: list[Path], options: dict
    ) -> Union[darsia.Image, list[darsia.Image]]:
        raise NotImplementedError


    def from_optical(
        self, path: list[Path], options: dict
    ) -> Union[darsia.Image, list[darsia.Image]]:
        raise NotImplementedError
