import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import darsia
from darsia.presets.workflows.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_analysis import (
    HeterogeneousColorAnalysis,
)

logger = logging.getLogger(__name__)


def calibration_color_signal(cls, path: Path, show: bool = False):
    # ! ---- LOAD RUN AND RIG ----

    config = FluidFlowerConfig(path)
    config.check("color_signal", "color_paths", "rig", "data", "protocol")

    # Mypy type checking
    for c in [
        config.color_signal,
        config.color_paths,
        config.rig,
        config.data,
        config.protocol,
    ]:
        assert c is not None

    fluidflower = cls()
    fluidflower.load(config.rig.path)

    # Load experiment
    experiment = darsia.ProtocolledExperiment(
        imaging_protocol=config.protocol.imaging,
        injection_protocol=config.protocol.injection,
        pressure_temperature_protocol=config.protocol.pressure_temperature,
        blacklist_protocol=config.protocol.blacklist,
        pad=config.data.pad,
    )
    fluidflower.load_experiment(experiment)

    # ! ---- COLOR PATH TOOL ----
    color_path_regression = darsia.ColorPathRegression(
        labels=fluidflower.labels,
        ignore_labels=config.color_paths.ignore_labels,
        mask=fluidflower.boolean_porosity,
    )

    # ! ---- DETERMINE BACKGROUND COLORS ----

    baseline = fluidflower.baseline
    background = color_path_regression.base_color_image(baseline)
    if show:
        background.show()

    # Make clusters of the background colors
    background_colors = color_path_regression.get_base_colors(baseline)
    active_background_colors = {
        label: color
        for label, color in background_colors.items()
        if label not in config.color_paths.ignore_labels
    }

    # Apply KMeans clustering to group into 4 clusters
    kmeans = KMeans(n_clusters=config.color_signal.num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(
        np.array(list(active_background_colors.values()))
    )

    # Organize clusters
    clusters = -np.ones(background_colors.keys().__len__(), dtype=int)
    for i, label in enumerate(active_background_colors.keys()):
        clusters[label] = cluster_labels[i]
    if len(config.color_paths.ignore_labels) > 0:
        clusters += 1  # Shift by one to account for ignored label 0
    cluster_set = set(clusters)

    # Define label colors for plotting
    label_colors = plt.cm.get_cmap("tab10", len(cluster_set))

    # Define cluster image and show
    cluster_image = darsia.zeros_like(fluidflower.labels, dtype=int)
    for label in np.unique(fluidflower.labels.img):
        cluster_image.img[fluidflower.labels.img == label] = clusters[label]
    if show:
        cluster_image.show(
            cmap=label_colors, title="Clustered background colors", delay=True
        )
        background.show(title="Background colors", delay=True)
        fluidflower.labels.show(title="Labels", delay=True)
        baseline.show(title="Baseline", delay=True)
        plt.show()

    # Plot each cluster with a different color in 3D
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in cluster_set:
            cluster_points = np.array(list(background_colors.values()))[clusters == i]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                cluster_points[:, 2],
                label=f"Cluster {i + 1}",
                # c=label_colors(i),
            )

        ax.set_title("3D Clustering of Vectors")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    # ! ---- READ COLOR PATHS FROM FILE ----

    color_paths = {
        label: darsia.ColorPath() for label in np.unique(fluidflower.labels.img)
    }
    for label in np.unique(fluidflower.labels.img):
        path = config.color_paths.calibration_file / f"color_path_{label}.json"
        if path.exists():
            color_paths[label].load(path)
            color_paths[label].name = f"Original Color path for label {label}"
        else:
            logger.warning(f"No color path found for label {label}, using empty path.")

    # Pick a reference color path - merely for visualization
    reference_label = config.color_paths.reference_label
    ref_color_path = color_paths[reference_label]
    custom_cmap = ref_color_path.get_color_map()
    if show:
        ref_color_path.show()

    # Plot all relative color paths in 3D
    if show:
        plt.figure(figsize=(8, 4))
        ax = plt.axes(projection="3d")

        for label, color_path in color_paths.items():
            # Sample the absolute color path
            absolute_colors = np.clip(color_path.colors, 0, 1)
            relative_colors = color_path.relative_colors
            if relative_colors is not None and len(relative_colors) > 0:
                ax.plot(
                    *zip(*relative_colors),
                    label=f"Label {label}",
                    c=label_colors(clusters[label]),
                )
                ax.scatter(*zip(*relative_colors), c=absolute_colors, s=20)

        # Mark the origin with a red box
        ax.scatter(0, 0, 0, c="red", s=100, label="Origin (0, 0, 0)", marker="o")

        ax.set_xlabel("R*")
        ax.set_ylabel("G*")
        ax.set_zlabel("B*")
        ax.set_title("Relative color paths in RGB space")
        ax.legend()
        plt.show()

    # Use clustering and define one color path per cluster
    cluster_color_paths = {i: darsia.ColorPath() for i in cluster_set}
    representative_labels = {}
    # TODO deal with empty color paths (and do not start with 1 here)
    # TODO deal with ignore_labels
    for i in cluster_set:
        _color_paths = {
            label: path for label, path in color_paths.items() if clusters[label] == i
        }
        _relative_colors = {
            label: path.relative_colors for label, path in _color_paths.items()
        }
        max_values = {}
        for label, path in _color_paths.items():
            interpolation = darsia.ColorPathInterpolation(
                color_path=path, interpolation="relative"
            )
            interpolation_values = {
                label: [interpolation(c) for c in _relative_colors[label]]
                for label in _relative_colors.keys()
            }
            max_interpolation_value = max(
                [max(v) for v in interpolation_values.values() if len(v) > 0],
                default=1.0,
            )
            max_values[label] = max_interpolation_value

        # Determine the label with the smallest max value
        representative_label = min(max_values, key=max_values.get)
        representative_labels[i] = representative_label

        # Use its color path as representative for the cluster
        cluster_color_paths[i] = color_paths[representative_label]

    # Plot all relative color paths in 3D
    if show:
        plt.figure(figsize=(8, 4))
        ax = plt.axes(projection="3d")

        for label, color_path in color_paths.items():
            # Sample the absolute color path
            absolute_colors = np.clip(color_path.colors, 0, 1)
            relative_colors = color_path.relative_colors
            if relative_colors is not None and len(relative_colors) > 0:
                ax.plot(
                    *zip(*relative_colors),
                    label=f"Label {label}",
                    c=label_colors(clusters[label]),
                    alpha=1 if label == representative_labels[clusters[label]] else 0.3,
                )
                ax.scatter(*zip(*relative_colors), c=absolute_colors, s=20)

        # Mark the origin with a red box
        ax.scatter(0, 0, 0, c="red", s=100, label="Origin (0, 0, 0)", marker="o")

        ax.set_xlabel("R*")
        ax.set_ylabel("G*")
        ax.set_zlabel("B*")
        ax.set_title("Relative color paths in RGB space")
        ax.legend()
        plt.show()

    # Overwrite the color paths with the cluster-based ones
    for label in np.unique(fluidflower.labels.img):
        if label in config.color_paths.ignore_labels:
            continue
        cluster_id = clusters[label]
        color_paths[label] = darsia.ColorPath(
            base_color=color_paths[label].base_color,
            colors=None,
            relative_colors=cluster_color_paths[cluster_id].relative_colors,
            values=cluster_color_paths[cluster_id].values,
            mode="rgb",
            name=f"Cluster {cluster_id} based color path for label {label}",
        )

    # ! ---- CONCENTRATION ANALYSIS ---- ! #

    concentration_analysis = HeterogeneousColorAnalysis(
        baseline=baseline,
        labels=fluidflower.labels,
        # restoration=fluidflower.restoration,
        ignore_labels=config.color_paths.ignore_labels,
    )
    for label in np.unique(fluidflower.labels.img):
        if label in config.color_paths.ignore_labels:
            continue
        concentration_analysis.update_color_path(
            label=label,
            color_path=color_paths[label],
        )

    calibration_images = []
    for path in config.color_signal.calibration_images:
        calibration_image = fluidflower.read_image(path)
        calibration_images.append(calibration_image)

    # ! ---- INTERACTIVE CALIBRATION ---- ! #

    # Start from existing calibration if available
    if config.color_signal.calibration_file.exists():
        concentration_analysis.load(config.color_signal.calibration_file)

    # Perform local calibration
    concentration_analysis.local_calibration_values(
        images=calibration_images,
        mask=fluidflower.boolean_porosity,
    )

    # Store calibration
    concentration_analysis.save(config.color_signal.calibration_file)

    # Test run
    concentration_images = [concentration_analysis(img) for img in calibration_images]

    for i in range(len(calibration_images)):
        calibration_images[i].show(title=f"Calibration image {i}", delay=True)
        concentration_images[i].show(
            title=f"Concentration image {i}", cmap=custom_cmap, delay=True
        )
    plt.show()
