"""Comparisons of block b closed for dicom and vtu images.

"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import darsia

# ! ---- Constants

cm2m = 1e-2  # conversion cm -> m
h2s = 60**2  # conversion hours -> seconds
ml_per_hour_to_m3_per_s = cm2m**3 / h2s

# ! ---- Model paramaters

porosity_2d = 0.2321
fracture_aperture = 0.1 * cm2m
depth = 1.95 * cm2m
injection_rate = 15 * ml_per_hour_to_m3_per_s

# ! ---- Read DICOM images


def read_dicom_images() -> darsia.Image:
    """Read space-time image from DICOM format for fractip a."""

    # Provide folder with dicom images (note: two folders)
    root = Path(
        "/home/jakub/images/ift/fracflow/01-single-phase-tracer-in-fracture/fractip-b"
    )
    images = [
        root / Path(p) / Path("DICOM/PA1/ST1/SE1")
        for p in [
            "fractip b closed 1 min rekon start 0 15 frames",
        ]
    ]

    if False:
        # Stack images (need to use such approach due to lack of common reference
        dicom_images = []
        for img in images:
            dicom_image = darsia.imread(img, suffix=".dcm", dim=3, series=True)
            dicom_images.append(dicom_image)
        uncorrected_image_3d = darsia.stack(dicom_images)

        # Use an assistant to tune the rotation correction and subregion selection
        test_image = uncorrected_image_3d.time_slice(9)
        test_image.show(
            surpress_3d=True, mode="matplotlib", side_view="voxel", cmap="turbo"
        )
        test_image.show(
            mode="plotly",
            side_view="voxel",
            view="voxel",
            cmap="turbo",
            threshold=0.8,
            relative=True,
        )
        options = {
            "threshold": 0.05,
            "relative": True,
            "verbosity": True,
        }
        rotation_assistant = darsia.RotationCorrectionAssistant(test_image, **options)
        rotation_corrections = rotation_assistant()
        rotated_test_image = test_image.copy()
        for rotation in rotation_corrections:
            rotated_test_image = rotation(rotated_test_image)
        subregion_assistant = darsia.SubregionAssistant(rotated_test_image, **options)
        coordinates = subregion_assistant()

    else:
        # Alternative: Fix tailored input parameters for the correction objects.
        rotation_corrections = [
            darsia.RotationCorrection(
                **{
                    "anchor": np.array([91, 106, 13]),
                    "rotation_from_isometry": True,
                    "pts_src": np.array([[91, 106, 13], [97, 106, 86]]),
                    "pts_dst": np.array([[91, 106, 13], [91, 106, 86]]),
                }
            ),
            darsia.RotationCorrection(
                **{
                    "anchor": np.array([50, 96, 117]),
                    "rotation_from_isometry": True,
                    "pts_src": np.array([[50, 96, 117], [90, 97, 117]]),
                    "pts_dst": np.array([[50, 96, 117], [90, 96, 117]]),
                }
            ),
            darsia.RotationCorrection(
                **{
                    "anchor": np.array([106, 94, 0]),
                    "rotation_from_isometry": True,
                    "pts_src": np.array([[106, 94, 0], [106, 96, 70]]),
                    "pts_dst": np.array([[106, 94, 0], [106, 95, 69]]),
                }
            ),
        ]
        coordinates = np.array(
            [
                [42.21969627956989, 42.29322659139785, -195.13738119462366],
                [42.30791713978495, 42.36046250537635, -195.1190467860215],
            ]
        )

    # Re-read the dicom image.
    # Use the uncorrected images further.
    image_3d = darsia.imread(
        images,
        suffix=".dcm",
        dim=3,
        series=True,
    )
    image_3d = image_3d.time_slice(9)
    for correction in rotation_corrections:
        image_3d = correction(image_3d)
    image_3d = image_3d.subregion(coordinates=coordinates)

    # Plot side view to check the result
    if True:
        image_3d.show(
            mode="matplotlib",
            surpress_3d=True,
            side_view="voxel",
            threshold=0.05,
            relative=True,
        )

    # ! ---- Precondition
    image_3d.img /= np.max(image_3d.img)

    return image_3d


def plot_3d_image(image_3d):
    # Plot side view to check the result
    image_3d.show(
        mode="matplotlib",
        surpress_3d=True,
        side_view="voxel",
        threshold=0.05,
        relative=True,
    )


def plot_sum(image_3d):
    if not isinstance(image_3d, list):
        image_3d = [image_3d]
    max_val = max(
        [
            np.max(
                np.sum(img, axis=0)
                if isinstance(img, np.ndarray)
                else np.sum(img.img, axis=0)
            )
            for img in image_3d
        ]
    )
    for i, img in enumerate(image_3d):
        plt.figure(f"sum {i}")
        if isinstance(img, np.ndarray):
            plt.imshow(np.sum(img, axis=0), cmap="turbo", vmin=0, vmax=max_val)
        else:
            plt.imshow(np.sum(img.img, axis=0), cmap="turbo", vmin=0, vmax=max_val)
        plt.colorbar()
    plt.show()


def plot_slice(image_3d):
    if not isinstance(image_3d, list):
        image_3d = [image_3d]
    max_val = max(
        [np.max(img if isinstance(img, np.ndarray) else img.img) for img in image_3d]
    )
    for i, img in enumerate(image_3d):
        plt.figure(f"slice {i}")
        if isinstance(img, np.ndarray):
            plt.imshow(img[20], cmap="turbo", vmin=0, vmax=max_val)
        else:
            plt.imshow(img.img[20], cmap="turbo", vmin=0, vmax=max_val)
        plt.colorbar()
    plt.show()


# Enrich the signal under the assumption of a monotone displacement experiment.
def accumulate_signal(image: darsia.Image) -> darsia.Image:
    new_image = image.copy()
    for time_id in range(image.time_num):
        new_image.img[..., time_id] = np.max(
            image.img[..., slice(0, time_id + 1)], axis=-1
        )

    return new_image


# ! ---- Apply regularization
def apply_tvd(image: darsia.Image) -> darsia.Image:
    """NOTE: Most time-consuming routine."""

    pass


def read_regularized_images(image: darsia.Image) -> darsia.Image:
    """Read preprocessed regularized images from numpy format."""

    reg_image = image.copy()

    for i in range(0, 13):
        reg_image.img[..., i] = np.load(
            # f"tvd/block-a/heterogeneous_mu_1000_iter/tvd_a_1000_{i}.npy"
            # f"tvd/block-a/mu_0.0001_1000_iter/tvd_{i}.npy" #
            f"tvd/delta/block-a/mu_0.0001_10000_iter/tvd_{i}.npy"  # latest tvd
        )

    return reg_image


# ! ---- Reduce 3d image to two spatial dimensions
def dimensional_reduction(image_3d: darsia.Image) -> darsia.Image:
    """Flatten 3d image."""
    return darsia.reduce_axis(image_3d, axis="z", mode="average")


# ! ---- Rescale images to concentrations


def extract_concentration(
    image_2d: darsia.Image, calibration_interval, injection_rate: float
) -> darsia.Image:
    # Define default concentration analysis, enabling calibration
    class CalibratedConcentrationAnalysis(
        darsia.ConcentrationAnalysis, darsia.InjectionRateModelObjectiveMixin
    ):
        pass

    model = darsia.ScalingModel(scaling=1)
    dicom_concentration_analysis = CalibratedConcentrationAnalysis(
        base=None, model=model
    )

    # Calibrate concentration analysis
    shape_meta = image_2d.shape_metadata()
    geometry = darsia.ExtrudedGeometry(expansion=depth, **shape_meta)

    dicom_concentration_analysis.calibrate_model(
        images=image_2d.time_interval(calibration_interval),
        options={
            "geometry": geometry,
            "injection_rate": injection_rate,
            "initial_guess": [1.0],
            "tol": 1e-5,
            "maxiter": 100,
            "method": "Nelder-Mead",
            "regression_type": "ransac",
        },
        plot_result=True,
    )

    # Interpret the calibration results to effectively determine the injection start.
    reference_time = dicom_concentration_analysis.model_calibration_postanalysis()

    # The results is that the effective injection start occurs at 3.25 minutes. Thus, update
    # the reference time, or simply the relative time. Set reference time with respect to
    # the first active image. Use seconds.
    image_2d.update_reference_time(reference_time)

    # Print some of the specs
    if True:
        print(
            f"The identified reference time / injection start is: {reference_time} [s]"
        )
        print(
            f"The dimensions of the space time dicom image are: {image_2d.dimensions}"
        )
        print(f"Relative times in seconds: {image_2d.time}")

    # Only for plotting reasons - same as above but with updated times and with activated plot:
    if False:
        dicom_concentration_analysis.calibrate_model(
            images=image_2d.time_interval(
                calibration_interval
            ),  # Only use some of the first images due to cut-off in the geometry
            options={
                "geometry": geometry,
                "injection_rate": injection_rate,
                "initial_guess": [1.0],
                "tol": 1e-5,
                "maxiter": 100,
                "method": "Nelder-Mead",
                "regression_type": "ransac",
            },
            plot_result=True,
        )

    # Produce space time image respresentation of concentrations.
    dicom_concentration = dicom_concentration_analysis(image_2d)

    # Debug: Plot the concentration profile. Note that values are larger than 1, which is
    # due to the calibration based on a non-smooth signal. The expected volume has to be
    # distributed among the active pixels. Pixels with strong activity thereby hold
    # unphysical amounts of fluid.
    if False:
        dicom_concentration.show("dicom concentration", 5)

    return dicom_concentration


# Determine total concentrations over time
def plot_concentration_evolution(concentration_2d):
    shape_meta = concentration_2d.shape_metadata()
    geometry = darsia.ExtrudedGeometry(expansion=depth, **shape_meta)
    concentration_values = geometry.integrate(concentration_2d)
    plt.figure("Experiments - injected volume over time")
    plt.plot(concentration_2d.time, concentration_values)
    plt.plot(
        concentration_2d.time,
        [injection_rate * time for time in concentration_2d.time],
        color="black",
        linestyle=(0, (5, 5)),
    )
    plt.xlabel("time [s]")
    plt.ylabel("volume [m**3]")
    plt.show()


# ! ---- VTU images


def read_vtu_images(vtu_ind: int) -> darsia.Image:
    """Read-in all available VTU data."""

    # This corresponds approx. to (only temporary - not used further):
    vtu_time = 0 # not relevant here.

    # Find the corresponding files
    vtu_images_2d = Path(f"data/vtu/block-b/data_2_{str(vtu_ind).zfill(6)}.vtu")
    vtu_images_1d = Path(f"data/vtu/block-b/data_1_{str(vtu_ind).zfill(6)}.vtu")

    # Define vtu images as DarSIA images
    vtu_image_2d = darsia.imread(
        vtu_images_2d,
        time=vtu_time,  # relative time in minutes
        key="temperature",  # key to address concentration data
        shape=(400, 400),  # dim = 2
        vtu_dim=2,  # effective dimension of vtu data
    )

    vtu_image_1d = darsia.imread(
        vtu_images_1d,
        time=vtu_time,  # relative time in minutes
        key="temperature",  # key to address concentration data
        shape=(1001, 51),  # dim = 2
        vtu_dim=1,  # effective dimension of vtu data
        width=fracture_aperture,  # for embedding in 2d
    )

    # PorePy/meshio cuts off coordinate values at some point...
    # Correct manually - width / 2 - aperature / 2.
    vtu_image_1d.origin[0] = (6.98 / 2 - 0.1 / 2) * cm2m

    # Equidimensional reconstructions. Superpose 2d and 1d images.
    # And directly interpret
    porosity_1d = 1.0 - porosity_2d  # for volume conservation
    vtu_image = darsia.superpose(
        [
            darsia.weight(vtu_image_2d, porosity_2d),
            darsia.weight(vtu_image_1d, porosity_1d),
        ]
    )

    # Plot for testing purposes
    if False:
        vtu_image.show("equi-dimensionsional reconstruction")

    # Concentrations - simple since the equi-dimensional vtu images are interpreted as
    # volumetric concentration
    vtu_concentration = vtu_image.copy()

    return vtu_concentration


# ! --- Align DICOM and vtu images


def align_images(dicom_concentration, vtu_concentration):
    # Plot two exemplary images to identify suitable src and dst points which will define
    # the alignment procedure

    if True:
        plt.figure("dicom")
        plt.imshow(np.sum(dicom_concentration.img, axis=0))
        plt.figure("vtu")
        plt.imshow(np.sum(vtu_concentration.img, axis=0))
        plt.show()

    # Pixels of fracture end points
    voxels_src = [[0, 85, 0], [0, 85, 86], [17, 85, 0], [17, 85, 86]]  # in DICOM image
    voxels_dst = [
        [0, 237, 86.5],
        [0, 148, 86.5],
        [17, 237, 86.5],
        [17, 148, 86.5],
    ]  # in VTU image

    # Define coordinate transform and apply it
    coordinatesystem_src = dicom_concentration.coordinatesystem
    coordinatesystem_dst = vtu_concentration.coordinatesystem
    coordinate_transformation = darsia.CoordinateTransformation(
        coordinatesystem_src,
        coordinatesystem_dst,
        voxels_src,
        voxels_dst,
        fit_options={
            "tol": 1e-5,
            "preconditioning": True,
        },
        isometry=False,
    )
    transformed_dicom_concentration = coordinate_transformation(dicom_concentration)

    # Restrict to intersecting active canvas
    intersection = coordinate_transformation.find_intersection()
    aligned_dicom_concentration = transformed_dicom_concentration.subregion(
        voxels=intersection
    )
    aligned_vtu_concentration = vtu_concentration.subregion(voxels=intersection)

    return aligned_dicom_concentration, aligned_vtu_concentration


def single_plot(
    mode: str,
    full_image: darsia.Image,
    add_on: str,
    image_path: Path,
):
    ##############################################################################
    # Plot the reconstructed vtu data, vtu plus Gaussian noise, and the dicom data.

    if mode == "sum":
        image = darsia.reduce_axis(full_image, axis="z", mode="average")
    else:
        image = darsia.reduce_axis(
            full_image, axis="z", mode="slice", slice_idx=20
        )

    # Define some plotting options (font, x and ylabels)
    # matplotlib.rcParams.update({"font.size": 14})
    # Define x and y labels in cm (have to convert from m to cm)
    shape = image.num_voxels
    dimensions = image.dimensions

    x_pixel, y_pixel = np.meshgrid(
        np.linspace(dimensions[0], 0, shape[0]),
        np.linspace(0, dimensions[1], shape[1]),
        indexing="ij",
    )
    vmax = 1.25
    vmin = 0
    contourlevels = [0.045 * vmax, 0.055 * vmax]
    cmap = "turbo"

    # Plot the data with contour line.
    fig_combination, axs_combination = plt.subplots(nrows=1, ncols=1)
    axs_combination.pcolormesh(
        y_pixel, x_pixel, image.img, cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs_combination.text(
        0.005, 0.075, "experiment", color="white", alpha=0.5, rotation=0, fontsize=14
    )
    axs_combination.text(
        0.005, 0.070, add_on, color="white", alpha=0.5, rotation=0, fontsize=14
    )
    axs_combination.set_ylim(top=0.08)
    axs_combination.set_aspect("equal")
    axs_combination.set_xlabel("x [m]", fontsize=14)
    axs_combination.set_ylabel("y [m]", fontsize=14)
    fig_combination.colorbar(
        cm.ScalarMappable(cmap=cmap),
        ax=axs_combination,
        label="volumetric concentration",
    )
    fig_combination.savefig(image_path, dpi=500, transparent=True)

    plt.show()

def qualitative_comparison(
    mode: str,
    full_dicom_image: darsia.Image,
    full_vtu_image: darsia.Image,
    add_on: str,
    image_path: Path,
    colorbar: bool= False
):
    ##############################################################################
    # Plot the reconstructed vtu data, vtu plus Gaussian noise, and the dicom data.

    if mode == "sum":
        dicom_image = darsia.reduce_axis(full_dicom_image, axis="z", mode="average")
        vtu_image = darsia.reduce_axis(full_vtu_image, axis="z", mode="average")
    else:
        dicom_image = darsia.reduce_axis(
            full_dicom_image, axis="z", mode="slice", slice_idx=20
        )
        vtu_image = darsia.reduce_axis(
            full_vtu_image, axis="z", mode="slice", slice_idx=20
        )

    # Define some plotting options (font, x and ylabels)
    # matplotlib.rcParams.update({"font.size": 14})
    # Define x and y labels in cm (have to convert from m to cm)
    shape = dicom_image.num_voxels
    dimensions = dicom_image.dimensions

    x_pixel, y_pixel = np.meshgrid(
        np.linspace(dimensions[0], 0, shape[0]),
        np.linspace(0, dimensions[1], shape[1]),
        indexing="ij",
    )
    vmax = 1.25
    vmin = 0
    contourlevels = [0.045 * vmax, 0.055 * vmax]
    cmap = "turbo"

    # Plot the dicom data with contour line.
    fig_combination, axs_combination = plt.subplots(nrows=1, ncols=1)
    combined_image = dicom_image.img.copy()
    mid = 86
    combined_image[:, mid:] = vtu_image.img[:, mid:]
    axs_combination.pcolormesh(
        y_pixel, x_pixel, combined_image, cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs_combination.contourf(
        y_pixel, x_pixel, vtu_image.img, cmap="Reds", levels=contourlevels, alpha=0.5
    )
    axs_combination.plot(
        [3.45 * cm2m, 3.45 * cm2m],
        [0.0, 0.08],
        color="white",
        alpha=0.3,
        linestyle="dashed",
    )
    axs_combination.text(
        0.005, 0.075, "experiment", color="white", alpha=0.5, rotation=0, fontsize=14
    )
    axs_combination.text(
        0.005, 0.070, add_on, color="white", alpha=0.5, rotation=0, fontsize=14
    )
    axs_combination.text(
        0.04, 0.075, "simulation", color="white", alpha=0.5, rotation=0, fontsize=14
    )
    axs_combination.set_ylim(top=0.08)
    axs_combination.set_aspect("equal")
    axs_combination.set_xlabel("x [m]", fontsize=14)
    axs_combination.set_ylabel("y [m]", fontsize=14)
    if colorbar:
        fig_combination.colorbar(
            cm.ScalarMappable(cmap=cmap),
            ax=axs_combination,
            label="volumetric concentration",
        )
    fig_combination.savefig(image_path, dpi=500, transparent=True)

    plt.show()


###########################################################################
# Main analysis

# Original PET images
if True:
    dicom_image_3d = read_dicom_images()
    dicom_image_3d.save("data/npz/block-b/dicom_raw_3d.npz")
else:
    dicom_image_3d = darsia.imread("data/npz/block-b/dicom_raw_3d.npz")

# Pick corresponding vtu images.
vtu_ind = 439
vtu_2d_concentration = read_vtu_images(vtu_ind)

# Resize to similar shape as dicom image
dicom_voxel_size = dicom_image_3d.voxel_size
vtu_2d_concentration = darsia.equalize_voxel_size(
    vtu_2d_concentration, min(dicom_voxel_size)
)

# Expand vtu image to 3d
dicom_height = dicom_image_3d.dimensions[0]
dicom_shape = dicom_image_3d.img.shape
vtu_concentration_3d = darsia.extrude_along_axis(
    vtu_2d_concentration, dicom_height, dicom_shape[0]
)

# Align dicom and vtu
if True:
    dicom_concentration = dicom_image_3d.copy()
    aligned_dicom_concentration, aligned_vtu_concentration = align_images(
        dicom_concentration, vtu_concentration_3d
    )
    aligned_dicom_concentration.save("data/npz/block-b/aligned_dicom_concentration.npz")
    aligned_vtu_concentration.save(f"data/npz/block-b/aligned_vtu_{vtu_ind}_concentration.npz")
else:
    aligned_dicom_concentration = darsia.imread("data/npz/block-b/aligned_dicom_concentration.npz")
    aligned_vtu_concentration = darsia.imread(
        f"data/npz/block-b/aligned_vtu_{vtu_ind}_concentration.npz"
    )

# Define final vtu concentration, and compute its mass (reference)
vtu_concentration_3d = aligned_vtu_concentration.copy()
vtu_3d_shape = vtu_concentration_3d.shape_metadata()
vtu_3d_geometry = darsia.Geometry(**vtu_3d_shape)
vtu_3d_integral = vtu_3d_geometry.integrate(vtu_concentration_3d)


# DICOM without TVD
def rescale_data(image, ref_integral):
    shape = image.shape_metadata()
    geometry = darsia.Geometry(**shape)
    integral = geometry.integrate(image)
    image.img *= ref_integral / integral
    return image


# Define dicom concentration with same mass
dicom_concentration_3d = aligned_dicom_concentration.copy()
dicom_concentration_3d = rescale_data(dicom_concentration_3d, vtu_3d_integral)
#single_plot("slice", dicom_concentration_3d, "noisy", "plots/block-b/lab_data_slice.png")
#single_plot("sum", dicom_concentration_3d, "noisy", "plots/block-b/lab_data_avg.png")

# Define mask (omega) for trust for regularization
heterogeneous_omega = True
if heterogeneous_omega:
    dicom_rescaled = dicom_concentration_3d.copy()
    dicom_rescaled.img /= np.max(dicom_rescaled.img)
    omega_bound = 0.15
    omega = np.minimum(dicom_rescaled.img, omega_bound)
    mask_zero = dicom_rescaled.img < 1e-4
    omega[mask_zero] = 10
    #plot_slice(omega)

    # Save plot
    reduced_dicom_concentration_2d = darsia.reduce_axis(
        dicom_concentration_3d, axis="z", mode="slice", slice_idx=20
    )
    shape = reduced_dicom_concentration_2d.num_voxels
    dimensions = reduced_dicom_concentration_2d.dimensions

    x_pixel, y_pixel = np.meshgrid(
        np.linspace(dimensions[0], 0, shape[0]),
        np.linspace(0, dimensions[1], shape[1]),
        indexing="ij",
    )
    fig_mask, axs_mask = plt.subplots(nrows=1, ncols=1)
    axs_mask.pcolormesh(y_pixel, x_pixel, np.log(omega[20]), cmap="turbo")
    axs_mask.set_ylim(top=0.08)
    axs_mask.set_aspect("equal")
    axs_mask.set_xlabel("x [m]")  # , fontsize=14)
    axs_mask.set_ylabel("y [m]")  # , fontsize=14)
    fig_mask.colorbar(cm.ScalarMappable(cmap="turbo"), ax=axs_mask, label="log($\omega_f$)")
    fig_mask.savefig("plots/block-b/omega.png", dpi=500, transparent=True)
    plt.close()
else:
    omega = 0.015

# Choose file suffix depeneding on omega
suffix = "heter" if heterogeneous_omega else "hom"

# DICOM concentration with H1 regularization
if True:
    h1_reg_dicom_concentration_3d = darsia.H1_regularization(
        dicom_concentration_3d,
        mu=0.1,
        omega=omega,
        dim=3,
        solver=darsia.CG(maxiter=10000, tol=1e-5),
    )
    h1_reg_dicom_concentration_3d.save(f"data/npz/block-b/h1_reg_dicom_concentration_{suffix}.npz")
else:
    h1_reg_dicom_concentration_3d = darsia.imread(
        f"data/npz/block-b/h1_reg_dicom_concentration_{suffix}.npz"
    )
h1_reg_dicom_concentration_3d = rescale_data(
    h1_reg_dicom_concentration_3d, vtu_3d_integral
)

# DICOM concentration with TVD regularization
isotropic = True
isotropic_suffix = "iso" if isotropic else "aniso"
if True:
    tvd_reg_dicom_concentration_3d = darsia.tvd(
        dicom_concentration_3d,
        method="heterogeneous bregman",
        isotropic=isotropic,
        weight=0.02,
        omega=omega,
        dim=3,
        max_num_iter=100,
        eps=1e-5,
        verbose=True,
        solver=darsia.Jacobi(maxiter=20),
    )
    tvd_reg_dicom_concentration_3d.save(
        f"data/npz/block-b/tvd_{isotropic_suffix}_reg_dicom_concentration_{suffix}.npz"
    )
else:
    tvd_reg_dicom_concentration_3d = darsia.imread(
        f"data/npz/block-b/tvd_{isotropic_suffix}_reg_dicom_concentration_{suffix}.npz"
    )
tvd_reg_dicom_concentration_3d = rescale_data(
    tvd_reg_dicom_concentration_3d, vtu_3d_integral
)

# Make qualitative comparisons
if True:
    #plot_sum(
    #    [
    #        vtu_concentration_3d,
    #        dicom_concentration_3d,
    #        h1_reg_dicom_concentration_3d,
    #        tvd_reg_dicom_concentration_3d,
    #    ]
    #)
    #plot_slice(
    #    [
    #        vtu_concentration_3d,
    #        dicom_concentration_3d,
    #        h1_reg_dicom_concentration_3d,
    #        tvd_reg_dicom_concentration_3d,
    #    ]
    #)
    #qualitative_comparison(
    #    "sum",
    #    dicom_concentration_3d,
    #    vtu_concentration_3d,
    #    "noisy",
    #    "plots/block-b/pure_dicom_avg.png",
    #)
    qualitative_comparison(
        "slice",
        dicom_concentration_3d,
        vtu_concentration_3d,
        "noisy",
        "plots/block-b/pure_dicom_slice.png",
    )
    #qualitative_comparison(
    #    "sum",
    #    h1_reg_dicom_concentration_3d,
    #    vtu_concentration_3d,
    #    "H1",
    #    f"plots/block-b/h1_reg_dicom_avg_{suffix}.png",
    #)
    qualitative_comparison(
        "slice",
        h1_reg_dicom_concentration_3d,
        vtu_concentration_3d,
        "H1",
        f"plots/block-b/h1_reg_dicom_slice_{suffix}.png",
    )
    qualitative_comparison(
        "sum",
        tvd_reg_dicom_concentration_3d,
        vtu_concentration_3d,
        "TVD",
        f"plots/block-b/tvd_{isotropic_suffix}_reg_dicom_avg_{suffix}.png",
        colorbar= True
    )
    qualitative_comparison(
        "slice",
        tvd_reg_dicom_concentration_3d,
        vtu_concentration_3d,
        "TVD",
        f"plots/block-b/tvd_{isotropic_suffix}_reg_dicom_slice_{suffix}.png",
    )

# Quantitative comparison - Wasserstein distance in 2d (only for illustration purposes)

slice_idx = 20
dicom_slice = darsia.reduce_axis(
    dicom_concentration_3d, axis="z", mode="slice", slice_idx=slice_idx
)
vtu_slice = darsia.reduce_axis(
    vtu_concentration_3d, axis="z", mode="slice", slice_idx=slice_idx
)
h1_reg_dicom_slice = darsia.reduce_axis(
    h1_reg_dicom_concentration_3d, axis="z", mode="slice", slice_idx=slice_idx
)
tvd_reg_dicom_slice = darsia.reduce_axis(
    tvd_reg_dicom_concentration_3d, axis="z", mode="slice", slice_idx=slice_idx
)


# Rescale (unphysical but necessary for comparison)
def rescale_slice(image: darsia.Image, ref_integral) -> darsia.Image:
    shape = image.shape_metadata()
    geometry = darsia.Geometry(**shape)
    integral = geometry.integrate(image)
    image.img *= ref_integral / integral
    return image


# Determine reference value
shape = vtu_slice.shape_metadata()
geometry = darsia.Geometry(**shape)
ref_integral = geometry.integrate(vtu_slice)

dicom_slice = rescale_slice(dicom_slice, ref_integral)
h1_reg_dicom_slice = rescale_slice(h1_reg_dicom_slice, ref_integral)
tvd_reg_dicom_slice = rescale_slice(tvd_reg_dicom_slice, ref_integral)

# Wasserstein for 2d images
options = {
    "method": "newton",
    "L": 1e-2,
    "num_iter": 500,
    "tol": 1e-11,
    "tol_distance": 1e-12,
    "regularization": 1e-16,
    "depth": 10,
    "lumping": True,
    "verbose": True,
    "scaling": 10,
    "plot_frequency": 10,
}
kwargs = {
    "method": "newton",
    "options": options,
    "plot_solution": True,
}
distance_dicom_vtu = darsia.wasserstein_distance(
    dicom_slice, vtu_slice, name="plots/block-b/pure dicom vs. vtu", **kwargs
)
distance_h1_dicom_vtu = darsia.wasserstein_distance(
    h1_reg_dicom_slice, vtu_slice, name="plots/block-b/h1 reg dicom vs. vtu", **kwargs
)
distance_tvd_dicom_vtu = darsia.wasserstein_distance(
    tvd_reg_dicom_slice, vtu_slice, name="plots/block-b/tvd reg dicom vs. vtu", **kwargs
)

print("The distances:")
print("Pure DICOM: ", distance_dicom_vtu)
print("H1 reg DICOM: ", distance_h1_dicom_vtu)
print("TVD reg DICOM: ", distance_tvd_dicom_vtu)

assert False, "3d Wasserstein distance computations not sufficiently efficient."

# Quantitative comparison - Wasserstein distance in 3d
options = {
    "method": "newton",
    "L": 1e-2,
    "num_iter": 100,
    "tol": 1e-10,
    "tol_distance": 1e-11,
    "regularization": 1e-16,
    "depth": 10,
    "lumping": True,
    "verbose": True,
}
kwargs = {
    "method": "newton",
    "options": options,
    "plot_solution": False,
}
distance_dicom_vtu_3d = darsia.wasserstein_distance_3d(
    dicom_concentration_3d, vtu_concentration_3d, name="", **kwargs
)
distance_h1_dicom_vtu_3d = darsia.wasserstein_distance_3d(
    h1_reg_dicom_concentration_3d,
    vtu_concentration_3d,
    name="",
    **kwargs,
)
distance_tvd_dicom_vtu_3d = darsia.wasserstein_distance_3d(
    tvd_reg_dicom_concentration_3d,
    vtu_concentration_3d,
    name="",
    **kwargs,
)

print("The distances 3d:")
print("Pure DICOM: ", distance_dicom_vtu_3d)
print("H1 reg DICOM: ", distance_h1_dicom_vtu_3d)
print("TVD reg DICOM: ", distance_tvd_dicom_vtu_3d)
