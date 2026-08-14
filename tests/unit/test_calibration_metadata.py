from pathlib import Path

import pytest

from darsia.presets.workflows.calibration.metadata import (
    read_calibration_metadata,
    validate_basis_metadata,
    write_calibration_metadata,
)
from darsia.signals.color import ColorEmbeddingBasis


def test_write_and_read_metadata_roundtrip(tmp_path: Path):
    path = tmp_path / "meta.json"
    write_calibration_metadata(
        path,
        basis=ColorEmbeddingBasis.FACIES,
        label_ids=[3, 1, 3, -1],
        extra={"foo": "bar"},
    )
    loaded = read_calibration_metadata(path)
    assert loaded is not None
    assert loaded["basis"] == "facies"
    assert loaded["label_ids"] == [1, 3]
    assert loaded["foo"] == "bar"


def test_validate_metadata_raises_on_basis_mismatch():
    with pytest.raises(ValueError, match="basis mismatch"):
        validate_basis_metadata(
            metadata={"basis": "labels", "label_ids": [0, 1]},
            expected_basis=ColorEmbeddingBasis.FACIES,
            expected_label_ids=[0, 1],
            artifact="color_paths",
        )


def test_validate_metadata_raises_on_label_ids_mismatch():
    with pytest.raises(ValueError, match="label-id mismatch"):
        validate_basis_metadata(
            metadata={"basis": "facies", "label_ids": [0, 2]},
            expected_basis=ColorEmbeddingBasis.FACIES,
            expected_label_ids=[0, 1],
            artifact="color_to_mass",
        )


def test_validate_metadata_legacy_missing_fields_warns():
    with pytest.warns(UserWarning):
        validate_basis_metadata(
            metadata=None,
            expected_basis=ColorEmbeddingBasis.FACIES,
            expected_label_ids=[0, 1],
            artifact="color_paths",
        )
