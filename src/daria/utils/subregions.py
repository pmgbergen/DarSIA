from daria import Image


def extractROI(img: Image, x: list, y: list) -> Image:
    """Extracts region of interest based on physical coordinates.

    Arguments:
        x (list): list with two elements and containst the start and end point in x-direction;
            points in physical coordinates.
        y (list): list with two elements and containst the start and end point in y-direction;
            points in physical coordinates.

    Returns:
        Image: image object restricted to the ROI.
    """

    # TODO conversion of pixels

    # Determine pixel ranges in x- and y-direction, corresponding to the ROI
    x_pix = [
        round((x[0] - img.origo[0]) / img.dx),
        round((x[1] - img.origo[0]) / img.dx),
    ]
    y_pix = [
        round((y[0] - img.origo[1]) / img.dy),
        round((y[1] - img.origo[1]) / img.dy),
    ]

    # Extract ROI
    # TODO do not copy full image - expensive
    # TODO conversion of pixels
    img = img.img
    img = img[
        (img.shape[0] - y_pix[1]) : (img.shape[0] - y_pix[0]),
        x_pix[0] : x_pix[1],
    ]

    # Define metadata
    origo = [x[0], y[0]]
    width = x[1] - x[0]
    height = y[1] - y[0]

    # Construct and return image correpsonding to ROI
    return Image(img=img, origo=origo, width=width, height=height)


def extractROIPixel(img: Image, x: list, y: list) -> Image:
    """Extracts region of interest based on pixels.

    Arguments:
        x (list of int):  list with two elements, containing the start and end pixels in
            x-direction, using physical pixel ordering.
        y (list of int):  list with two elements, containing the start and end pixels in
            y-direction, using physical pixel ordering.

    Returns:
        Image: image object restricted to the ROI.
    """
    # Restrict the image with repect to given pixel ranges
    # TODO do not copy full image - expensive
    # TODO conversion of pixels
    img = img.img
    img = img[
        (img.shape[0] - y[1]) : (img.shape[0] - y[0]),
        x[0] : x[1],
    ]

    # Define metadata
    origo = [img.origo[0] + x[0] * img.dx, img.origo[1] + y[0] * img.dy]
    width = ((x[1] - x[0]) / img.shape[0]) * img.width
    height = ((y[1] - y[0]) / img.shape[1]) * img.height

    # Construct and return image corresponding to ROI
    return Image(img=img, origo=origo, width=width, height=height)
