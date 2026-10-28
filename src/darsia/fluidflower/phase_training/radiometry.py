"""Optional Uniform24 bridge for the standalone studio and headless trainer.

The integration package owns the numerical method. This host adapter only reads
registered review images, samples their chart, and indexes per-run receipts.
Historical review assets remain immutable and are identified as such, not raw.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower.phase_training.review_regions import load_review_rgb
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools.phase_optical_cache_compatibility import canonical_implementation


def core():
    try:
        from darsia_extensions import phase_radiometry
    except ImportError as exc:
        raise RuntimeError(
            "Uniform24 phase preparation requires darsia-extensions >= 0.10. "
            "Legacy review sessions can still be opened without it."
        ) from exc
    return phase_radiometry


def detect_chart(rgb):
    from darsia.fluidflower.image_correction import (
        _reference_colorchecker,
        _checker_orientation_score,
    )

    for corner in ("upper_right", "upper_left", "lower_right", "lower_left"):
        try:
            cv2.setRNGSeed(0)
            swatches, roi = _reference_colorchecker(rgb, corner=corner, roi_rc=None)
            if _checker_orientation_score(swatches) <= 0.20:
                return swatches, roi, corner
        except (AssertionError, ValueError):
            continue
    raise ValueError("No readable, correctly oriented 24-patch ColorChecker was found")


def resolve_profile(value):
    if value in (None, "", "legacy"):
        return None
    if value == "uniform24-after2014":
        import darsia

        return core().make_profile(
            darsia.ColorCheckerAfter2014().swatches_rgb,
            target_provenance={
                "kind": "published",
                "target": "DarSIA ColorCheckerAfter2014",
                "physical_chart_must_match": True,
            },
        )
    if str(value).startswith("image:"):
        from PIL import Image

        path = resolve_review_path(str(value)[6:])
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
        cv2.setRNGSeed(0)
        swatches, roi, corner = detect_chart(rgb)
        return core().make_profile(
            swatches,
            target_provenance={
                "kind": "measured-common-reference-image",
                "path": str(path),
                "file_sha256": sha256_file(path),
                "roi_rc": roi.tolist(),
                "corner": corner,
                "sampling_runtime": runtime_identity(),
            },
        )
    return core().load_profile(resolve_review_path(value))


def runtime_identity():
    from darsia._opencv_runtime import POLICY
    module = core()
    from darsia_extensions import radiometry
    from darsia.fluidflower import image_correction
    import darsia
    import inspect

    return {
        "uniform24_sha256": sha256_file(Path(radiometry.__file__)),
        "preparation_sha256": sha256_file(Path(module.__file__)),
        "adapter_sha256": canonical_implementation(__name__, sha256_file(Path(__file__))),
        "chart_detection_sha256": sha256_file(Path(image_correction.__file__)),
        "chart_sampler_sha256": sha256_file(
            Path(inspect.getfile(darsia.CustomColorChecker))
        ),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "sampling_seed": 0,
        "opencv_ipp_policy": POLICY,
    }


def model_profile(model):
    profiles = [
        head.radiometry_profile for head in (model.co2_presence, model.gas_within_co2)
    ]
    if profiles[0] != profiles[1]:
        raise ValueError("Model heads have different radiometric profiles")
    if profiles[0] is not None:
        core().validate_profile(profiles[0])
    return profiles[0]


def analysis_review_root(review_root, run_id):
    """Follow derived-display lineage without applying normalization twice."""
    root = resolve_review_path(review_root)
    visited = set()
    expected_box = None
    while True:
        if str(root.resolve()).casefold() in visited:
            raise ValueError("Cycle in review-image source lineage")
        visited.add(str(root.resolve()).casefold())
        path = root / "native_frames" / run_id / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if expected_box is not None and manifest["content_box_xywh"] != expected_box:
            raise ValueError(
                "Review geometry differs from the pre-normalization source"
            )
        if not manifest.get("analysis_source_review_root"):
            return root
        expected_box = manifest["content_box_xywh"]
        root = resolve_review_path(manifest["analysis_source_review_root"])


class RunRadiometry:
    def __init__(
        self,
        profile,
        *,
        series_id,
        run_id,
        baseline_path,
        content_box,
        cache_root,
        source_manifest=None,
    ):
        self.profile = core().validate_profile(profile)
        self.series_id, self.run_id = series_id, run_id
        self.content_box = tuple(content_box)
        self.cache_root = Path(cache_root)
        self.identity = runtime_identity()
        self.receipts = {}
        self.roi = None
        self.baseline_path = resolve_review_path(baseline_path)
        self.baseline_sha256 = sha256_file(self.baseline_path)
        self.source_manifest = None
        self.frame_sources = {}
        self.upstream_correction_contract = None
        if source_manifest is not None:
            source_manifest = resolve_review_path(source_manifest)
            manifest = json.loads(Path(source_manifest).read_text(encoding="utf-8-sig"))
            self.source_manifest = {
                "path": str(source_manifest),
                "sha256": sha256_file(Path(source_manifest)),
            }
            self.frame_sources = {
                f"frame_{int(row['frame_index']):03d}": row
                for row in manifest.get("frame_sources", [])
                if "frame_index" in row
            }
            self.upstream_correction_contract = manifest.get("correction_contract")

    def load(self, path, frame_id):
        from darsia._opencv_runtime import scientific_thread, completed_frame
        ipp_before = scientific_thread("RunRadiometry.load:before")
        module = core()
        path = resolve_review_path(path)
        rgb = load_review_rgb(path, self.content_box)
        source = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "kind": "registered-review-image; previous corrections not inferred",
            "content_box_xywh": list(self.content_box),
            "manifest": self.source_manifest,
            "upstream_frame_metadata": self.frame_sources.get(frame_id),
            "upstream_correction_contract": self.upstream_correction_contract,
        }
        key = module.digest(
            {
                "source": source,
                "profile": self.profile["sha256"],
                "runtime": self.identity,
                "series_id": self.series_id,
                "run_id": self.run_id,
                "frame_id": frame_id,
                "baseline_sha256": self.baseline_sha256,
            }
        )
        cache = self.cache_root / f"{key}.json"
        if cache.is_file():
            receipt = json.loads(cache.read_text(encoding="utf-8"))
            p = module.validate_receipt(
                receipt, series_id=self.series_id, run_id=self.run_id
            )
            if (
                p["input_rgb_sha256"] != module.rgb_digest(rgb)
                or p["profile"] != self.profile
                or p["frame_id"] != frame_id
                or p["sampling"]["runtime"] != self.identity
            ):
                raise ValueError(
                    f"Invalid cached correction for {self.run_id}/{frame_id}"
                )
            from darsia_extensions.radiometry import Uniform24

            output = Uniform24(p["correction"]).apply(rgb)
            module.validate_receipt(receipt, output)
            self.roi = p["sampling"]["roi_rc"]
        else:
            from darsia.fluidflower.image_correction import _reference_colorchecker

            if self.roi is None:
                baseline = load_review_rgb(self.baseline_path, self.content_box)
                cv2.setRNGSeed(0)
                try:
                    _, roi, _ = detect_chart(baseline)
                except ValueError as exc:
                    raise ValueError(
                        f"{self.series_id}/{self.run_id}/{frame_id}: {exc}"
                    ) from exc
                self.roi = roi.tolist()
            cv2.setRNGSeed(0)
            observed, _ = _reference_colorchecker(
                rgb, corner="upper_right", roi_rc=self.roi
            )
            output, receipt = module.prepare_image(
                rgb,
                self.profile,
                observed_swatches=observed,
                series_id=self.series_id,
                run_id=self.run_id,
                frame_id=frame_id,
                source=source,
                sampling={"roi_rc": self.roi, "runtime": self.identity},
            )
            completed_frame(ipp_before, "RunRadiometry.load:before-cache-publication")
            cache.parent.mkdir(parents=True, exist_ok=True)
            from darsia_extensions.storage import write_new

            try:
                write_new(
                    cache,
                    lambda stream: stream.write(json.dumps(receipt, indent=2).encode()),
                )
            except FileExistsError:
                if json.loads(cache.read_text(encoding="utf-8")) != receipt:
                    raise ValueError("Concurrent correction cache mismatch")
        # The correction receipt retains its producer identity, also on a hit.
        # Current load observations are separate from those original bytes.
        self.current_opencv_ipp = getattr(self, "current_opencv_ipp", {})
        self.current_opencv_ipp[frame_id] = completed_frame(ipp_before, "RunRadiometry.load:after")
        self.receipts[frame_id] = receipt
        return output

    def report(self):
        receipts = list(self.receipts.values())
        return {
            "profile": self.profile,
            "inventory": core().correction_inventory(receipts),
            "receipts": receipts,
            "current_load_opencv_ipp": dict(getattr(self, "current_opencv_ipp", {})),
        }
