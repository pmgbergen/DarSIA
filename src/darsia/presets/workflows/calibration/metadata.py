"""Helpers for workflow calibration metadata persistence and checks."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from darsia.signals.color.color_embedding import ColorEmbeddingBasis, parse_color_embedding_basis


def write_calibration_metadata(
    target: Path,
    *,
    basis: ColorEmbeddingBasis,
    label_ids: list[int],
    extra: dict | None = None,
) -> None:
    """Write calibration metadata JSON file to target path."""

    metadata = {
        "basis": basis.value,
        "label_ids": sorted({int(label) for label in label_ids if int(label) >= 0}),
    }
    if extra:
        metadata.update(extra)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2))


def read_calibration_metadata(path: Path) -> dict | None:
    """Read metadata from JSON if present, else return None."""

    if not path.exists():
        return None
    return json.loads(path.read_text())


def validate_basis_metadata(
    *,
    metadata: dict | None,
    expected_basis: ColorEmbeddingBasis,
    expected_label_ids: list[int],
    artifact: str,
    strict: bool = True,
) -> None:
    """Validate loaded metadata against expected basis and label ids."""

    if metadata is None:
        warn(
            f"Legacy {artifact} calibration detected (no metadata); skipping basis "
            "and label-id compatibility check."
        )
        return

    raw_basis = metadata.get("basis")
    if raw_basis is None:
        warn(
            f"{artifact} calibration metadata missing 'basis'; treating as legacy "
            "and skipping strict basis compatibility check."
        )
    else:
        found_basis = parse_color_embedding_basis(raw_basis)
        if found_basis != expected_basis:
            raise ValueError(
                f"{artifact} calibration basis mismatch: expected "
                f"'{expected_basis.value}', found '{found_basis.value}'."
            )

    raw_label_ids = metadata.get("label_ids")
    if raw_label_ids is None:
        warn(
            f"{artifact} calibration metadata missing 'label_ids'; treating as legacy "
            "and skipping strict label-set compatibility check."
        )
        return

    expected = sorted({int(label) for label in expected_label_ids if int(label) >= 0})
    found = sorted({int(label) for label in raw_label_ids if int(label) >= 0})

    if strict and expected != found:
        raise ValueError(
            f"{artifact} calibration label-id mismatch: expected {expected}, "
            f"found {found}."
        )
