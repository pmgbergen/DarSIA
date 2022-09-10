from daria import Image


def extractROI(img: Image, x: list, y: list, return_roi: bool = False) -> Image:
    """Extracts region of interest based on physical coordinates.

    Arguments:
        x (list): list with two elements and containst the start and end point in x-direction;
            points in metric units.
        y (list): list with two elements and containst the start and end point in y-direction;
            points in metric units.

    Returns:
        Image: image object restricted to the ROI.
    """

    # Assume that x and y are in increasing order.
    assert x[0] < x[1] and y[0] < y[1]

    # Convert metric units to number of pixels, and define top-left and bottom-right
    # corners of the roi, towards addressing the image with conventional ordering
    # of x and y coordinates.
    top_left_corner = img.coordinatesystem.coordinateToPixel((x[0], y[1]))
    bottom_right_corner = img.coordinatesystem.coordinateToPixel((x[1], y[0]))

    # Define the ROI
    roi = (
        slice(top_left_corner[0], bottom_right_corner[0]),
        slice(top_left_corner[1], bottom_right_corner[1]),
    )

    # Define metadata (all quantities in metric units)
    origo = [x[0], y[0]]
    width = x[1] - x[0]
    height = y[1] - y[0]

    # Construct and return image corresponding to ROI
    if return_roi:
        return Image(img=img.img[roi], origo=origo, width=width, height=height), roi
    else:
        return Image(img=img.img[roi], origo=origo, width=width, height=height)


def extractROIPixel(img: Image, roi: tuple) -> Image:
    """Extracts region of interest based on pixel info.

    Arguments:
        roi (tuple of slices): to be used straight away to extract a region of interest;
            using the conventional pixel ordering

    Returns:
        Image: image object restricted to the ROI.
    """
    # Define metadata; Note that img.origo uses a Cartesian ordering, while roi
    # uses the conventional pixel ordering
    origo = [img.origo[0] + roi[1].start * img.dx, img.origo[1] + roi[0].stop * img.dy]
    width = (roi[1].stop - roi[1].start) * img.dx
    height = (roi[0].stop - roi[0].start) * img.dy

    # Construct and return image corresponding to ROI
    return Image(img=img.img[roi], origo=origo, width=width, height=height)
