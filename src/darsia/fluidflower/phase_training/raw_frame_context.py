"""Per-frame geometry of a RAW run for the existing training feature/model ABI.

``RawFrameGeometryCohort`` offers the interface of
``frame_context.V5FrameGeometryCohort`` for a run whose photographs form a RAW
photographic review (``raw_photographic_review``). Each frame's geometry is
its SHA-bound RAW producer audit, which records the transforms under the same
names as V5 receipts (``H_ac28_to_frame_native``, ``H_v5_B_to_frame_native``).
The reference-context and photo-sand bindings that V5 receipts carried are
explicit configuration. ``frame_context`` itself stays unchanged, because
series configurations bind its exact bytes. Nothing is fitted or retrained.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.native import support as _support
from darsia.fluidflower.native.config import require_parameters
from darsia.fluidflower.phase_saved_assets import VerifiedAssetReader
from darsia.fluidflower.project import sha256_file

CONTRACT = "darsia-raw-frame-geometry-v1"
SUPPORT_CONTRACT = "darsia-raw-frame-feature-source-support-v1"
# Training on RAW reviews uses baseline B and the previous frame on the native
# centre, each with its exact valid mask (no extended ring).
TEMPORAL_CONTRACT = "darsia-raw-native-centre-source-supported-v1"
OBSERVABLE_DOMAIN = "observable_phase_domain"
CONTEXT_BINDING_KEYS = frozenset({
    "reference_context_path", "reference_context_sha256",
    "photo_certified_sand_mask_path", "photo_certified_sand_mask_sha256",
    "photo_sand_qa_pass", "photo_sand_qa_scope", "photo_sand_reference_mask_path",
    "photo_sand_reference_mask_sha256", "photo_sand_reference_receipt_sha256",
    "optical_geometry_qa_pass", "physical_scale_qa_pass", "enclosed_hole_policy",
})


class RawFrameGeometryCohort:
    """One RAW photographic review's per-frame geometry, in the V5 cohort interface.

    ``context_binding`` holds the run's reference-context and photo-sand
    bindings (the keys V5 receipts carried) and is needed only by ``context``.
    It defaults to the ``training_context`` the review root was written with.
    Audits, records and masks are SHA-checked on every access.
    """

    def __init__(self, review_root: str | Path, run: str, *,
                 context_binding: Mapping[str, Any] | None = None):
        from darsia.fluidflower.raw_photographic_review import read_raw_review_manifest

        manifest = read_raw_review_manifest(review_root, run)
        if manifest is None:
            raise ValueError("The review root holds no RAW photographic review for this run")
        path = Path(review_root) / "native_frames" / str(run).casefold() / "manifest.json"
        self._bind(path, manifest, context_binding)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, *,
                      context_binding: Mapping[str, Any] | None = None) -> "RawFrameGeometryCohort":
        """The cohort of a RAW review manifest file, for example the trainer's staged copy."""
        from darsia.fluidflower.raw_photographic_review import validate_raw_review_manifest

        path = Path(manifest_path)
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        manifest = validate_raw_review_manifest(document, str(document.get("run") or ""))
        cohort = cls.__new__(cls)
        cohort._bind(path, manifest, context_binding)
        return cohort

    def _bind(self, path: Path, manifest: Mapping, context_binding: Mapping[str, Any] | None) -> None:
        """Take the run's frames, reference, grid and identities from one checked manifest."""
        if context_binding is None:
            context_binding = manifest.get("training_context") or {}
        unknown = sorted(set(context_binding) - CONTEXT_BINDING_KEYS)
        if unknown:
            raise ValueError(f"Unknown RAW frame context bindings: {unknown}")
        self.run = manifest["run"]
        self.native_manifest_path = path
        self.native_manifest_sha256 = sha256_file(path)
        # A RAW review is its own source revision; no older manifest stands behind it.
        self.source_manifest_path = path
        self.source_manifest_sha256 = self.native_manifest_sha256
        self.rows = manifest["frames"]
        self.reference = manifest["reference"]
        self.shape_hw = tuple(manifest["native_shape_hw"])
        self.identity = dict(native_configuration_sha256=manifest["native_configuration"]["sha256"],
                             raw_binding_sha256=manifest["raw_binding"]["sha256"])
        self.context_binding = dict(context_binding or {})
        self._reader = VerifiedAssetReader()

    def frame(self, index: int) -> dict[str, Any]:
        """The SHA-checked binding of one frame: V5 binding keys plus its RAW identities."""
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError("RAW frame index outside its review timeline")
        row = self.rows[index]
        audit = self._reader.json(row["audit"])
        if (audit.get("jpeg_inputs") is not False
                or (audit.get("source_identity") or {}).get("raw_sha256") != row["acquisition"]["sha256"]):
            raise ValueError("RAW frame audit belongs to another acquisition")
        parameters = require_parameters("support")
        return dict(
            self.context_binding, **self.identity, contract=CONTRACT, run=self.run,
            frame_index=index, case_id=row["case_id"],
            source_manifest_sha256=self.source_manifest_sha256,
            receipt_path=row["audit"]["path"], receipt_sha256=row["audit"]["sha256"],
            native_record_sha256=row["native_record"]["sha256"],
            native_geometry_sha256=row["native_geometry"]["sha256"],
            acquisition_sha256=row["acquisition"]["sha256"], profile_id=row["profile_id"],
            H_ac28_to_frame_native=_support._normal_h(audit["H_ac28_to_frame_native"], parameters=parameters),
            H_v5_B_to_frame_native=_support._normal_h(audit["H_v5_B_to_frame_native"], parameters=parameters),
            validation_only=True, mass_eligible=False)

    def context(self, index: int, *, fields: Iterable[str] | None = None
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Source-bound context of one frame, through the V5 derivation.

        A request for at most the observable phase domain and the two photo-sand
        masks, which is what training on a RAW review asks for, needs no physical
        reference context. Its photo masks come from the unchanged derivation
        alone (the frame's reference H, the bound photo source, guard and fill
        policy), and its observable domain is the whole native grid, that
        derivation's own policy. Any physical or facies field takes the full,
        strict derivation, which refuses a reference context lacking its fields.
        """
        from .frame_context import PHOTO_FIELDS, V5FrameGeometryCohort, derive_frame_context

        requested = None if fields is None else tuple(fields)
        if requested is not None and set(requested) <= {OBSERVABLE_DOMAIN, *PHOTO_FIELDS}:
            binding = self.frame(index)
            photo = set(requested) & set(PHOTO_FIELDS)
            result = (derive_frame_context(binding, fields=photo, expected_shape_hw=self.shape_hw)
                      if photo else {})
            if OBSERVABLE_DOMAIN in requested:
                result[OBSERVABLE_DOMAIN] = np.ones(self.shape_hw, dtype=bool)
            return result, binding
        if requested is None:
            return V5FrameGeometryCohort.context(self, index)
        return V5FrameGeometryCohort.context(self, index, fields=requested)

    def H_v5_B_to_frame(self, index: int) -> np.ndarray:
        """Exact baseline B to current native frame transform, as the V5 cohort gives it."""
        from .frame_context import V5FrameGeometryCohort

        return V5FrameGeometryCohort.H_v5_B_to_frame(self, index)

    def H_previous_to_frame(self, previous_index: int, index: int) -> np.ndarray:
        """Previous to current native frame through the common baseline, as for V5."""
        from .frame_context import V5FrameGeometryCohort

        return V5FrameGeometryCohort.H_previous_to_frame(self, previous_index, index)

    def native_centre(self, index: int | None) -> dict[str, dict]:
        """The ``log_linear_rgb`` and ``valid_mask`` descriptors of a frame, or of baseline B (None).

        Frames read their SHA-bound native record and the baseline its
        acquisition-manifest record, so a caller can verify staged copies.
        """
        if index is None:
            acquisitions = self._reader.json(self.reference["acquisition_manifest"])
            centre = acquisitions["records"][self.reference["record_index"]]["full"]["native_centre"]
        else:
            if type(index) is not int or not 0 <= index < len(self.rows):
                raise ValueError("RAW frame index outside its review timeline")
            centre = self._reader.json(self.rows[index]["native_record"])["full"]["native_centre"]
        return dict(log_linear_rgb=dict(centre["log_linear_rgb"]), valid_mask=dict(centre["valid_mask"]))

    def _valid(self, index: int | None) -> tuple[np.ndarray, str]:
        """The SHA-checked native valid mask of a frame or of baseline B (None), and its SHA."""
        descriptor = self.native_centre(index)["valid_mask"]
        valid = np.asarray(self._reader.array(descriptor))
        if valid.dtype != np.bool_ or valid.shape != self.shape_hw:
            raise ValueError("RAW native valid mask differs from the review's native grid")
        return valid, descriptor["sha256"]

    def feature_support(self, index: int, previous_index: int | None,
                        linear_input: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Source-supported optical pixels of one RAW frame, under the V5 support rule.

        The current, baseline and previous supports are the native valid masks
        of their SHA-bound RAW records, all on the native grid (origin 0, 0);
        ``linear_input`` is accepted for the V5 call signature and not used.
        """
        if previous_index is not None and (type(previous_index) is not int
                                           or not 0 <= previous_index < index):
            raise ValueError("RAW previous frame index is not earlier than current")
        current = self.frame(index)
        previous = None if previous_index is None else self.frame(previous_index)
        current_valid, current_sha = self._valid(index)
        baseline_valid, baseline_sha = self._valid(None)
        previous_valid, previous_sha = (None, None) if previous is None else self._valid(previous_index)
        parameters = require_parameters("support")
        safe, raw_support = _support.compute_feature_support(
            current_valid=current_valid, baseline_valid=baseline_valid, previous_valid=previous_valid,
            h_baseline_to_current=current["H_v5_B_to_frame_native"],
            h_baseline_to_previous=None if previous is None else previous["H_v5_B_to_frame_native"],
            baseline_origin_xy=(0, 0), previous_origin_xy=(0, 0), parameters=parameters)
        receipt = dict(
            contract=SUPPORT_CONTRACT, native_manifest_sha256=self.native_manifest_sha256,
            frame_index=index, previous_frame_index=previous_index,
            frame_geometry_receipt_sha256=current["receipt_sha256"],
            previous_geometry_receipt_sha256=None if previous is None else previous["receipt_sha256"],
            current_valid_sha256=current_sha, baseline_valid_sha256=baseline_sha,
            previous_valid_sha256=previous_sha,
            raw_supported_pixels=int(np.count_nonzero(raw_support)),
            feature_supported_pixels=int(np.count_nonzero(safe)),
            texture_footprint_hw=list(parameters["texture_footprint_hw"]),
            feature_support_mask_sha256=sha256(np.packbits(safe, bitorder="little").tobytes()).hexdigest())
        receipt["receipt_sha256"] = sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                                                      allow_nan=False).encode()).hexdigest()
        return safe, receipt


__all__ = ["CONTEXT_BINDING_KEYS", "CONTRACT", "OBSERVABLE_DOMAIN", "SUPPORT_CONTRACT", "TEMPORAL_CONTRACT",
           "RawFrameGeometryCohort"]
