"""
Module containing routines to read images from file to DarSIA.
Several file types are supported.

"""

from datetime import datetime
from operator import itemgetter
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
    array = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

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


def imread_from_dicom(path: Union[Path, list[Path]], **kwargs) -> darsia.ScalarImage:
    """
    Initialization of Image by reading from DICOM format.

    Assumptions: Coordinate systems in DICOM format are
    organized in an ijk format, which is different from
    the ijk format used for general images. DICOM images
    come in slices, and conventionally the coordinate system
    is associated relatively to the object to be scanned,
    which is not the coordinate system according to gravity,
    since objects usually lie horizontally in a medical scanner.
    In DarSIA we translate to the standard Cartesian coordinate
    system such that the z-coordinate is aligned with the
    gravitational force. Therefore the ijk indexing of
    standard DICOM images is not consistent with the matrix
    indexing of darsia.GeneralImage. The transformation of
    coordinate axes is performed here.

    Args:
        path (Path, or list of such): path to dicom stacks
        keyword arguments:
            orientation (str): orientation of DICOM
                coordinate system, i.e., row, cols, page
                in DICOM format wrt "xyz"; e.g., if rows,
                cols, page correspond to.

    Returns:
        darsia.GeneralImage: 3d space-time image

    """
    # PYDICOM specific tags for addressing dicom data from dicom files
    tag_position = pydicom.tag.BaseTag(0x00200032)
    tag_rows = pydicom.tag.BaseTag(0x00280010)
    tag_cols = pydicom.tag.BaseTag(0x00280011)
    tag_pixel_size = pydicom.tag.BaseTag(0x00280030)
    tag_slice_thickness = pydicom.tag.BaseTag(0x00180050)
    tag_acquisition_date = pydicom.tag.BaseTag(0x00080022)
    tag_acquisition_time = pydicom.tag.BaseTag(0x00080032)
    tag_number_time_slices = pydicom.tag.BaseTag(0x00540101)

    # Good to have for future safety checks.
    tag_number_slices = pydicom.tag.BaseTag(0x00540081)

    # ! ---- 1. Step: Get overview of number of images

    # Convert srcfile to a list of just a string.
    if not isinstance(path, list):
        path = [path]

    # Initialize arrays for DICOM data
    pixel_data = []  # actual signal
    slice_positions = []  # the position in page direction, i.e., of each slice.
    num_rows = []  # image size in row/z direction.
    num_cols = []  # image size in col/y direction.
    num_slices = []  # image size in page/x direction
    num_times = []  # image size in temporal direction
    acq_times = []  # acquisition time of samples
    acq_date_time_tuples = []  # acquision (date, time) of samples

    # Consider each dicom file by itself (each file is a sample and has to
    # be sorted into the larger picture.
    for p in path:

        # Read the dicom sample
        dataset = pydicom.dcmread(p)

        # The matrix indexing of DICOM images uses a different convention
        # in translating row/col/page (ijk) to xyz. To remain consistent,
        # we employ standard Cartesian xyz. This requires some reorganization.
        # rows corresponds to the positive y axis (j in standard matrix indexing);
        # cols corresponds to the negative z axis (i in standard matrix indexing);
        # page corresponds to the negative x axis (k in standard matrix indexing);
        # In summary, a conversion from one matrix indexing to another has to be
        # performed: DICOM data -> GeneralImage data, [0, 1, 2] -> [1,0,2]

        # Extract the actual signal. And flip the axis to comply with
        # the convention of the row/col/page indexing, also denoted
        # standard matrix indexing, or ijk in DarSIA, or also matlab.
        pixel_data.append(dataset.pixel_array)

        # Extract position for each slice; corresponds to "z_dicom" direction
        # DICOM languag, but assumed to be negative x direction in
        # Cartesian coordinates. After all stored as third component
        # as 'page' corresponds to "z_dicom".
        slice_positions.append(np.array(dataset[tag_position].value)[2])

        # Extract number of datapoints in each direction.
        num_rows.append(dataset[tag_rows].value)
        num_cols.append(dataset[tag_cols].value)
        num_slices.append(dataset[tag_number_slices].value)
        num_times.append(dataset[tag_number_time_slices].value)

        # Extract acquisition time for each slice
        acq_date_time_tuples.append(
            dataset[tag_acquisition_date].value + dataset[tag_acquisition_time].value
        )
        acq_times.append(dataset[tag_acquisition_time].value)

        # Extract pixel size - assume constant for all files - and assume info in mm
        conversion_mm_to_m = 10 ** (-3)
        voxel_size = np.zeros(3)
        # pixels are assumed to have the same dimensions in both directions.
        voxel_size[:2] = np.array(dataset[tag_pixel_size].value) * conversion_mm_to_m
        # k corresponds to the slice thickness
        voxel_size[2] = dataset[tag_slice_thickness].value * conversion_mm_to_m

    # Assume constant image size.
    assert all([num_rows[i] == num_rows[0] for i in range(len(num_rows))])
    assert all([num_cols[i] == num_cols[0] for i in range(len(num_cols))])
    assert all([num_slices[i] == num_slices[0] for i in range(len(num_slices))])
    assert all([num_times[i] == num_times[0] for i in range(len(num_times))])
    shape = (num_rows[0], num_cols[0], num_slices[0], num_times[0])

    # ! ---- 2. Step: Sort input wrt datetime

    # Convert data to numpy arrays.
    voxel_data = np.stack(pixel_data, axis=2)  # comes in standard matrix indexing
    slice_positions = np.array(slice_positions)  # only used to sort slices.
    acq_times = np.array(acq_times)  # used to distinguish between images.

    # Convert date time strings to datetime format. Expect a data format
    # for the input as ('20210222133538.000'), where the first entry
    # represent the date Feb 22, 2021, and the second entry represents
    # the time 13 hours, 35 minutes, and 38 seconds. Sort the list of
    # datetimes, starting with earliest datetime.
    def datetime_conversion(date_time: str) -> datetime:
        return datetime.strptime(date_time[:14], "%Y%m%d%H%M%S")

    # Convert times to different formats, forcing uniqueness and search possibilties.
    acq_date_time_tuples = np.array(acq_date_time_tuples, dtype=object)
    unique_acq_datetimes = list(set(acq_date_time_tuples))
    times_indices = [
        (datetime_conversion(u), i) for i, u in enumerate(unique_acq_datetimes)
    ]
    sorted_times_indices = sorted(times_indices, key=itemgetter(0))
    sorted_times = [t for t, _ in sorted_times_indices]
    sorted_indices = [i for _, i in sorted_times_indices]

    # Group and sort data into the time frames, converting the 3d tensor
    # into a 4d img
    img = np.zeros(shape, dtype=voxel_data.dtype)
    time = sorted_times
    for i, date_time in enumerate(sorted_times):

        # Fetch corresponding datetime in original format
        index = sorted_indices[i]
        unique_date_time = unique_acq_datetimes[index]

        # Fetch all frames corresponding to the current time frame
        time_frame = np.argwhere(acq_date_time_tuples == unique_date_time).flatten()

        # For each time frame, sort data in slice-direction (stored as third component)
        sorted_time_frame = np.argsort(slice_positions[time_frame])

        # Apply sorting to the time frames
        sorted_time_frame = time_frame[sorted_time_frame]

        # Store data at right time
        img[..., i] = voxel_data[..., sorted_time_frame]

    # ! ---- 3. Convert to Image
    meta = {
        "dim": 3,
        "indexing": "ijk",
        "dimensions": [voxel_size[i] * shape[i] for i in range(3)],
        "origin": [0, 0, 0],  # TODO?
        "series": True,
        "datetime": unique_acq_datetimes,  # TODO
    }

    return darsia.ScalarImage(img=img, time=time, **meta)


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
        return imread_from_dicom(path, **kwargs)
    elif suffix in [".vtu"]:
        raise NotImplementedError
