from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from darsia.presets.workflows.setup import setup_depth as setup_depth_module
from darsia.presets.workflows.setup import setup_facies as setup_facies_module
from darsia.presets.workflows.setup import setup_labeling as setup_labeling_module


@dataclass
class DummyImage:
    img: np.ndarray
    saved_path: Path | None = None

    def save(self, path: Path) -> None:
        self.saved_path = path


def test_segment_colored_image_exports_labels_jpg(monkeypatch, tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    config = SimpleNamespace(
        rig=SimpleNamespace(dim=2, width=2.0, height=1.0),
        labeling=SimpleNamespace(
            colored_image=tmp_path / "colored.png",
            rtol=0.001,
            ensure_connectivity=True,
            unite_labels=None,
            labels=labels_path,
        ),
        check=lambda *_args: None,
    )
    labels = DummyImage(np.array([[0, 1], [1, 2]], dtype=np.int32))
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        setup_labeling_module,
        "FluidFlowerConfig",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        setup_labeling_module.darsia, "imread", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        setup_labeling_module.darsia,
        "label_image",
        lambda *_args, **_kwargs: labels,
    )
    monkeypatch.setattr(
        setup_labeling_module,
        "save_discrete_map_illustration",
        lambda array, path, title, colorbar_label: recorded.update(
            {
                "array": array,
                "path": path,
                "title": title,
                "colorbar_label": colorbar_label,
            }
        ),
    )

    setup_labeling_module.segment_colored_image(tmp_path / "config.toml")

    assert labels.saved_path == labels_path
    assert np.array_equal(recorded["array"], labels.img)
    assert recorded["path"] == labels_path.with_suffix(".jpg")
    assert recorded["title"] == "Labels"


def test_setup_facies_exports_facies_jpg(monkeypatch, tmp_path: Path) -> None:
    facies_path = tmp_path / "facies.npz"
    config = SimpleNamespace(
        labeling=SimpleNamespace(labels=tmp_path / "labels.npz"),
        facies=SimpleNamespace(
            path=facies_path,
            props=tmp_path / "facies_props.xlsx",
            label_to_facies_map={0: 10, 1: 20, 2: 30},
        ),
        check=lambda *_args: None,
    )
    labels = DummyImage(np.array([[0, 1], [1, 2]], dtype=np.int32))
    facies = DummyImage(np.array([[10, 20], [20, 30]], dtype=np.int32))
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        setup_facies_module,
        "FluidFlowerConfig",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        setup_facies_module.darsia, "imread", lambda *_args, **_kwargs: labels
    )
    monkeypatch.setattr(
        setup_facies_module.darsia,
        "reassign_labels",
        lambda *_args, **_kwargs: facies,
    )
    monkeypatch.setattr(
        setup_facies_module.pd,
        "read_excel",
        lambda *_args, **_kwargs: pd.DataFrame({"id": [10, 20, 30]}),
    )
    monkeypatch.setattr(
        setup_facies_module,
        "save_discrete_map_illustration",
        lambda array, path, title, colorbar_label: recorded.update(
            {
                "array": array,
                "path": path,
                "title": title,
                "colorbar_label": colorbar_label,
            }
        ),
    )

    setup_facies_module.setup_facies(object, tmp_path / "config.toml")

    assert facies.saved_path == facies_path
    assert np.array_equal(recorded["array"], facies.img)
    assert recorded["path"] == facies_path.with_suffix(".jpg")
    assert recorded["title"] == "Facies"


def test_setup_depth_map_exports_depth_jpg(monkeypatch, tmp_path: Path) -> None:
    depth_map_path = tmp_path / "depth_map"
    config = SimpleNamespace(
        depth=SimpleNamespace(
            measurements=tmp_path / "depth.csv",
            depth_map=depth_map_path,
        ),
        rig=SimpleNamespace(resolution=(3, 4), width=2.0, height=1.0, dim=2),
        check=lambda *_args: None,
    )
    depth_map = DummyImage(np.arange(12, dtype=float).reshape((3, 4)))
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        setup_depth_module,
        "FluidFlowerConfig",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        setup_depth_module.darsia,
        "Image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        setup_depth_module.darsia,
        "interpolate_to_image_from_csv",
        lambda *_args, **_kwargs: depth_map,
    )
    monkeypatch.setattr(
        setup_depth_module,
        "save_scalar_map_illustration",
        lambda array, path, title, colorbar_label: recorded.update(
            {
                "array": array,
                "path": path,
                "title": title,
                "colorbar_label": colorbar_label,
            }
        ),
    )

    setup_depth_module.setup_depth_map(tmp_path / "config.toml")

    assert depth_map.saved_path == depth_map_path.with_suffix(".npz")
    assert np.array_equal(recorded["array"], depth_map.img)
    assert recorded["path"] == depth_map_path.with_suffix(".jpg")
    assert recorded["title"] == "Depth map"
