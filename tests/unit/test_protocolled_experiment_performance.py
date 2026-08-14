from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from darsia.experiment.experiment import ProtocolledExperiment
from darsia.experiment.protocols import ImagingProtocol


def _write_injection_protocol(path: Path, start: datetime) -> None:
    pd.DataFrame(
        [
            {
                "id": 1,
                "location_x": 0.0,
                "location_y": 0.0,
                "start": start.isoformat(),
                "end": (start + timedelta(hours=1)).isoformat(),
                "rate_kg/s": 0.0,
            }
        ]
    ).to_csv(path, index=False)


def _write_pressure_protocol(path: Path, start: datetime) -> None:
    pd.DataFrame(
        [
            {
                "datetime": start.isoformat(),
                "pressure_bar": 1.013,
                "temperature_celsius": 20.0,
                "pressure_gradient_bar": 0.0,
                "temperature_gradient_celsius": 0.0,
            }
        ]
    ).to_csv(path, index=False)


def _write_imaging_protocol(
    path: Path,
    rows: list[dict],
) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _touch_images(folder: Path, count: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(1, count + 1):
        image = folder / f"img_{i:05d}.JPG"
        image.touch()
        paths.append(image)
    return paths


def test_imaging_protocol_prefers_path_match_and_uses_blacklist_index(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    protocol_path = tmp_path / "imaging.csv"
    blacklist_path = tmp_path / "blacklist.csv"
    _write_imaging_protocol(
        protocol_path,
        [
            {
                "path": "sub/img_00999.JPG",
                "image_id": 999,
                "datetime": (start + timedelta(hours=5)).isoformat(),
            },
            {
                "path": "",
                "image_id": 1,
                "datetime": (start + timedelta(hours=1)).isoformat(),
            },
        ],
    )
    pd.DataFrame({"image_id": [999]}).to_csv(blacklist_path, index=False)

    protocol = ImagingProtocol(protocol_path, pad=5, blacklist=blacklist_path)
    dt = protocol.get_datetime(tmp_path / "sub" / "img_00999.JPG")
    assert dt == pd.Timestamp(start + timedelta(hours=5))
    assert protocol.is_blacklisted(tmp_path / "sub" / "img_00999.JPG")


def test_find_images_for_times_uses_deepest_folder_mapping_and_deduplicates(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    injection_path = tmp_path / "injection.csv"
    pressure_path = tmp_path / "pressure.csv"
    _write_injection_protocol(injection_path, start)
    _write_pressure_protocol(pressure_path, start)

    folder_root = tmp_path / "root"
    folder_sub = folder_root / "sub"
    root_images = _touch_images(folder_root, 3)
    sub_images = _touch_images(folder_sub, 3)

    root_protocol = tmp_path / "imaging_root.csv"
    sub_protocol = tmp_path / "imaging_sub.csv"
    _write_imaging_protocol(
        root_protocol,
        [
            {
                "path": f"img_{i:05d}.JPG",
                "image_id": i,
                "datetime": (start + timedelta(hours=i)).isoformat(),
            }
            for i in range(1, 4)
        ],
    )
    _write_imaging_protocol(
        sub_protocol,
        [
            {
                "path": f"sub/img_{i:05d}.JPG",
                "image_id": i,
                "datetime": (start + timedelta(hours=100 + i)).isoformat(),
            }
            for i in range(1, 4)
        ],
    )

    experiment = ProtocolledExperiment(
        data=root_images + sub_images,
        imaging_protocol={folder_root: root_protocol, folder_sub: sub_protocol},
        injection_protocol=injection_path,
        pressure_temperature_protocol=pressure_path,
        blacklist_protocol=None,
        pad=5,
    )

    selected = experiment.find_images_for_times(times=[101.1, 101.2], data=sub_images)
    assert selected == [sub_images[0]]


def test_find_images_for_times_reuses_cached_timeline_for_same_data_pool(
    tmp_path: Path, monkeypatch
) -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    injection_path = tmp_path / "injection.csv"
    pressure_path = tmp_path / "pressure.csv"
    _write_injection_protocol(injection_path, start)
    _write_pressure_protocol(pressure_path, start)

    folder = tmp_path / "images"
    image_paths = _touch_images(folder, 1000)
    protocol_path = tmp_path / "imaging.csv"
    _write_imaging_protocol(
        protocol_path,
        [
            {
                "path": f"img_{i:05d}.JPG",
                "image_id": i,
                "datetime": (start + timedelta(hours=i)).isoformat(),
            }
            for i in range(1, 1001)
        ],
    )

    experiment = ProtocolledExperiment(
        data=image_paths,
        imaging_protocol=protocol_path,
        injection_protocol=injection_path,
        pressure_temperature_protocol=pressure_path,
        blacklist_protocol=None,
        pad=5,
    )

    call_count = 0
    original_iter_available = experiment.iter_available

    def wrapped_iter_available(paths: list[Path]):
        nonlocal call_count
        call_count += 1
        return original_iter_available(paths)

    monkeypatch.setattr(experiment, "iter_available", wrapped_iter_available)

    first = experiment.find_images_for_times(times=[10.0, 250.0], data=image_paths)
    second = experiment.find_images_for_times(times=[500.0, 999.0], data=image_paths)

    assert len(first) == 2
    assert len(second) == 2
    assert call_count == 1


def test_iter_available_resolves_protocol_once_per_path(
    tmp_path: Path, monkeypatch
) -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    injection_path = tmp_path / "injection.csv"
    pressure_path = tmp_path / "pressure.csv"
    _write_injection_protocol(injection_path, start)
    _write_pressure_protocol(pressure_path, start)

    folder_root = tmp_path / "root"
    folder_sub = folder_root / "sub"
    root_images = _touch_images(folder_root, 2)
    sub_images = _touch_images(folder_sub, 2)
    all_images = root_images + sub_images

    root_protocol = tmp_path / "imaging_root.csv"
    sub_protocol = tmp_path / "imaging_sub.csv"
    _write_imaging_protocol(
        root_protocol,
        [
            {
                "path": f"img_{i:05d}.JPG",
                "image_id": i,
                "datetime": (start + timedelta(hours=i)).isoformat(),
            }
            for i in range(1, 3)
        ],
    )
    _write_imaging_protocol(
        sub_protocol,
        [
            {
                "path": f"sub/img_{i:05d}.JPG",
                "image_id": i,
                "datetime": (start + timedelta(hours=100 + i)).isoformat(),
            }
            for i in range(1, 3)
        ],
    )

    experiment = ProtocolledExperiment(
        data=all_images,
        imaging_protocol={folder_root: root_protocol, folder_sub: sub_protocol},
        injection_protocol=injection_path,
        pressure_temperature_protocol=pressure_path,
        blacklist_protocol=None,
        pad=5,
    )

    call_count = 0
    original_protocol_for_path = experiment._protocol_for_path

    def wrapped_protocol_for_path(path: Path):
        nonlocal call_count
        call_count += 1
        return original_protocol_for_path(path)

    monkeypatch.setattr(experiment, "_protocol_for_path", wrapped_protocol_for_path)

    available = experiment.iter_available(all_images)

    assert len(available) == len(all_images)
    assert call_count == len(all_images)
