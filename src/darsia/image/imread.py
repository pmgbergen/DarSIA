"""
Module containing routines to read images from file to DarSIA.
Several file types are supported.

"""


from datetime import datetime
from pathlib import Path
from subprocess import check_output
from typing import Optional, Union

import cv2
import numpy as np
import pydicom
from PIL import Image as PIL_Image

import darsia


def imread_from_npy(
    path: Union[Path, list[Path]], **kwargs
) -> Union[darsia.GeneralImage, list[darsia.GeneralImage]]:
    """Converter from npy format to darsia.Image.

    Args:
        path (Path or list of Path): path(s) to npy files.
        keyword arguments:


    """
    raise NotImplementedError


def imread_from_optical(
    path: Union[Path, list[Path]],
    time: Optional[Union[datetime, list[datetime]]] = None,
    transformations: Optional[list] = None,
    **kwargs,
) -> Union[darsia.OpticalImage, list[darsia.OpticalImage]]:
    """Reading functionality from jpg, png, tif format to optical images.

    Args:
        path (Path or list of such): path(s) to image(s).
        time (datetime or list of such): user-specified physical times;
            automatically detected from metadata if 'None'.
        transformations (list of callables): transformations for 2d images.
        keyword arguments:
            series (bool): flag controlling whether a time series of images
                is created.

    Returns:
        OpticalImage (or list of such): converted image, list of such, or
            space-time image, depending on the flag 'series'.

    """
    # TODO check method for grayscale images. shape of array?

    if isinstance(path, Path):
        # Single image

        array, timestamp = _read_single_optical_image(path)
        if time is None:
            time = timestamp

        image = darsia.OpticalImage(
            img=array, time=time, transformations=transformations, **kwargs
        )

        return image

    elif isinstance(path, list):
        # Collection of images

        # Read from file
        data = [_read_single_optical_image(p) for p in path]

        # Split into data arrays and times
        arrays = [d[0] for d in data]
        timestamps = [d[1] for d in data]
        if time is None:
            time = timestamps

        series = kwargs.get("series", False)
        if series:
            # Create a space-time optical image through stacking along the time axis
            space_dim = 2
            space_time_array = np.stack(arrays, axis=space_dim)

            image = darsia.OpticalImage(
                img=space_time_array,
                time=time,
                transformations=transformations,
                **kwargs,
            )
            return image

        else:
            # Create list of optical images
            images = [
                darsia.OpticalImage(
                    img=array, time=timestamp, transformations=transformations, **kwargs
                )
                for array, timestamp in zip(arrays, time)
            ]
            return images
    else:
        raise NotImplementedError


def _read_single_optical_image(path: Path) -> tuple[np.ndarray, Optional[datetime]]:
    """Utility function for setting up a single optical image.

    Args:
        path (Path): path to single optical image.

    Returns:
        np.ndarray: data array in RGB format
        datetime (optional): timestamp

    """
    # Read image and convert to RGB
    array = cv2.cvtColor(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    # Prefered: Read time from exif metafile.
    pil_img = PIL_Image.open(path)
    exif = pil_img.getexif()
    if exif.get(306) is not None:
        timestamp = datetime.strptime(exif.get(306), "%Y:%m:%d %H:%M:%S")
    else:
        timestamp = None

    # Alternatively, use terminal output.
    # Credit to: https://stackoverflow.com/questions/27929025/...
    # ...exif-python-pil-return-empty-dictionary
    if timestamp is None:
        output = check_output(f"identify -verbose {str(path)}".split())
        meta = {}
        for line in output.splitlines()[:-1]:
            spl = line.decode().split(":", 1)
            k, v = spl
            meta[k.lstrip()] = v.strip()

        timestamp_with_info = meta["date"]
        spl = timestamp_with_info.split(":", 1)
        # Hardcoded way of retrieving the datetime of "2022-10-12T11:02:40+02:00"
        timestamp = datetime.strptime(spl[1][1:], "%Y-%m-%dT%H:%M:%S%z")

    return array, timestamp


def imread_from_dicom(
    path: list[Path], options: dict
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
            dataset[tag_acquisition_date].value + dataset[tag_acquisition_time].value
        )
        acq_times.append(dataset[tag_acquisition_time].value)
        number_times = dataset[tag_number_time_slices].value

        # Extract pixel size - assume constant for all files - and assume info in mm
        conversion_mm_to_m = 10 ** (-3)
        dimensions = np.zeros(3)
        dimensions[:2] = np.array(dataset[tag_pixel_size].value) * conversion_mm_to_m
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
            sorted_2d_pixel_data = np.sum(sorted_pixel_data_cache[:, :, roi], axis=2)

        # TODO Rm
        shape = sorted_pixel_data_cache.shape

        # Store
        # TODO allow for general format, also 3d.
        sorted_pixel_data[datetime_conversion(date_time)] = sorted_2d_pixel_data.copy()

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


def imread_from_vtu(
    path: list[Path], options: dict
) -> Union[darsia.Image, list[darsia.Image]]:
    raise NotImplementedError


# ! ---- Main interface


def imread(
    path: Union[str, Path, list[str], list[Path]], **kwargs
) -> Union[darsia.GeneralImage, list[darsia.GeneralImage]]:
    """Determine and call reading routine depending on filetype.
    Provide interface for numpy arrays, standard optical image formats,
    dicom images, and vtu images.

    Args:
        path (str, Path or list of such): path(s) to file(s).
        kwargs: keyword arguments tailored to the file format.

    Returns:
        GeneralImage, or list of such: image of collection of images.

    """

    # Determine file type of images
    if isinstance(path, list):

        # Convert paths to Path format
        path = [Path(p) for p in path]

        # Determine file format
        suffix = path[0].suffix

        # Check consistency
        assert all([p.suffix == suffix for p in path])

    else:

        # Convert path to Path format
        path = Path(path)

        # Determine file format
        suffix = path.suffix

    # Use lowercase for robustness
    suffix = str(suffix).lower()

    # Depending on the ending run the corresponding routine.
    if suffix == ".npy":
        raise NotImplementedError
    elif suffix in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        return imread_from_optical(path, **kwargs)
    elif suffix in [".dcm"]:
        raise NotImplementedError
    elif suffix in [".vtu"]:
        raise NotImplementedError
