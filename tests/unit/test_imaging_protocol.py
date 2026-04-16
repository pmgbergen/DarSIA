from datetime import datetime
from pathlib import Path

import darsia


def _create_protocol_csv(path: Path) -> Path:
    protocol_path = path / "imaging_protocol.csv"
    protocol_path.write_text(
        "\n".join(
            [
                "image_id,datetime",
                "1,2024-01-01T00:00:00",
                "2,2024-01-01T01:00:00",
                "3,2024-01-01T02:00:00",
            ]
        )
    )
    return protocol_path


def _create_image_paths(path: Path) -> list[Path]:
    image_paths = [
        path / "img_00001.jpg",
        path / "img_00002.jpg",
        path / "img_00003.jpg",
    ]
    for image_path in image_paths:
        image_path.touch()
    return image_paths


def test_find_images_for_datetimes_nearest_and_unique_order(tmp_path):
    protocol = darsia.ImagingProtocol(_create_protocol_csv(tmp_path), pad=5)
    image_paths = _create_image_paths(tmp_path)
    invalid_path = tmp_path / "invalid.jpg"
    invalid_path.touch()

    selected = protocol.find_images_for_datetimes(
        paths=image_paths + [invalid_path],
        datetimes=[
            datetime.fromisoformat("2024-01-01T00:10:00"),
            datetime.fromisoformat("2024-01-01T00:20:00"),
            datetime.fromisoformat("2024-01-01T01:40:00"),
        ],
    )

    assert selected == [image_paths[0], image_paths[2]]


def test_find_images_for_datetimes_tolerance_is_strict(tmp_path):
    protocol = darsia.ImagingProtocol(_create_protocol_csv(tmp_path), pad=5)
    image_paths = _create_image_paths(tmp_path)
    midpoint = datetime.fromisoformat("2024-01-01T00:30:00")

    strict = protocol.find_images_for_datetimes(
        paths=image_paths,
        datetimes=[midpoint],
        tol=1800.0,
    )
    permissive = protocol.find_images_for_datetimes(
        paths=image_paths,
        datetimes=[midpoint],
        tol=1801.0,
    )

    assert strict == []
    assert permissive == [image_paths[0]]
