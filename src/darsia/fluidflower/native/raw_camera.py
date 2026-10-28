"""Camera-agnostic RAW decoding under the native sensor contract.

The FF_AC Sony ZV-1 route in :mod:`.registration` is frozen. This module
neither changes nor re-implements it: :func:`select_raw_route` returns the
original renderer for that camera and :func:`decode_sensor` refuses ZV-1 files.

Other registered cameras (Nikon Z 7II for FF_UM, Sony ILCE-7RM3 for FF_BM) are
decoded under the same photometric contract, with every camera constant read
from the file through LibRaw instead of a frozen ZV-1 value:

* one black level is subtracted from the integer mosaic before demosaicing and
  clamped at zero; there is no 0..1 scaling and no automatic exposure;
* the configured OpenCV edge-aware Bayer demosaic runs at full sensor
  resolution, guarded by the same channel-order fixture as the frozen route;
* there is no white balance, no camera-to-XYZ matrix and no gamma or tone
  curve; the camera metadata for these are recorded only;
* sensor saturation is recorded where the integer value reaches the LibRaw
  white level.

Values stay in RAW units (DN). :meth:`DecodedSensor.linear_normalised` gives
the same values divided by ``white_level - black_level`` as an explicit,
recorded unit change; it is never applied implicitly.

Placing RAW values on the companion-JPEG grid is per-camera geometry
calibration. It needs an explicit :class:`GeometryProfile`. Rendering refuses a
missing profile, an unvalidated profile, or a channel the validation does not
cover, instead of guessing.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image

from . import registration
from .io import read_bound_json, sha256_file

RAW_CAMERA_CONTRACT = "darsia-native-raw-camera-v1"
RAW_RENDER_CONTRACT = "darsia-native-raw-camera-jpeg-grid-v1"
PROFILE_FORMAT = "darsia-raw-camera-profile-v1"

FROZEN_ZV1_ROUTE = "frozen-sony-zv1-native"
METADATA_ROUTE = "metadata-bound-native"

SONY_LENS_TABLES = "sony-arw-lens-tables"
NO_LENS_METADATA = "none"

# The configured edge-aware demosaic and its channel-order fixture are bound to
# this CFA layout (LibRaw raw_pattern indices into color_desc "RGBG").
RGGB_PATTERN = ((0, 1), (3, 2))
RGGB_DESCRIPTION = "RGBG"

# Sony ARW lens-table format, read with the ZV-1 rule (registration.raw_coordinates).
SONY_DISTORTION_TAG = 0x7037
SONY_CA_TAG = 0x7035
SONY_DISTORTION_DENOMINATOR = 2.0**14
SONY_CA_DENOMINATOR = 2.0**21
SONY_CROP_TOP_LEFT_TAG = 0x74C7
SONY_CROP_SIZE_TAG = 0x74C8
DEFAULT_CROP_ORIGIN_TAG = 50719
DEFAULT_CROP_SIZE_TAG = 50720
EXIF_IFD_TAG = 34665
SONY_MAKERNOTE_SETTINGS = {
    0x2011: "vignetting_correction",
    0x2012: "lateral_chromatic_aberration",
    0x2013: "distortion_correction",
}
SONY_SETTING_VALUES = {0: "Off", 2: "Auto", 0xFFFFFFFF: "n/a"}

_WHITE_BALANCE_REASON = "Camera-native channel basis, as in the frozen native route"
_XYZ_REASON = "Phase calibration is in camera channel basis"
_ELIGIBILITY_REASON = (
    "Decoding and geometry certificates do not establish photometric or label "
    "parity for this camera; those gates are separate."
)


class UnsupportedRawCamera(ValueError):
    """The RAW file's camera is not registered for any native route."""


class FrozenRouteRequired(ValueError):
    """The camera is bound to the frozen ZV-1 route and must not be re-decoded."""


class GeometryProfileUnavailable(ValueError):
    """No explicit, validated geometry profile covers the requested mapping."""


@dataclass(frozen=True)
class RawCamera:
    """One registered camera body and the RAW route it is bound to."""

    key: str
    make: str
    model: str
    series: tuple[str, ...]
    route: str
    lens_metadata: str


REGISTERED_CAMERAS: tuple[RawCamera, ...] = (
    RawCamera(
        "sony-dsc-zv1",
        "SONY",
        "DSC-ZV1",
        ("FF_AC",),
        FROZEN_ZV1_ROUTE,
        SONY_LENS_TABLES,
    ),
    RawCamera(
        "sony-ilce-7rm3",
        "SONY",
        "ILCE-7RM3",
        ("FF_BM",),
        METADATA_ROUTE,
        SONY_LENS_TABLES,
    ),
    RawCamera(
        "nikon-z-7-2",
        "NIKON CORPORATION",
        "NIKON Z 7_2",
        ("FF_UM",),
        METADATA_ROUTE,
        NO_LENS_METADATA,
    ),
)


@dataclass(frozen=True)
class CameraIdentity:
    """EXIF Make/Model (and Orientation) of a RAW file's first IFD."""

    make: str
    model: str
    orientation: int | None = None


@dataclass(frozen=True)
class RawRoute:
    """Selected RAW route; exactly one of ``renderer`` or ``decoder`` is set."""

    camera: RawCamera
    route: str
    renderer: Callable[..., Any] | None
    decoder: Callable[..., Any] | None


def read_camera_identity(path: str | Path) -> CameraIdentity:
    """Read EXIF Make, Model and Orientation from the TIFF container's IFD0."""
    with tifffile.TiffFile(path) as tiff:
        tags = tiff.pages[0].tags
        make, model, orientation = tags.get(271), tags.get(272), tags.get(274)
        if make is None or model is None:
            raise UnsupportedRawCamera("RAW file has no EXIF Make/Model: " + str(path))
        return CameraIdentity(
            str(make.value).strip(),
            str(model.value).strip(),
            None if orientation is None else int(orientation.value),
        )


def lookup_camera(identity: CameraIdentity) -> RawCamera:
    """Return the registered camera for an exact (case-insensitive) Make/Model."""
    key = (identity.make.strip().casefold(), identity.model.strip().casefold())
    for camera in REGISTERED_CAMERAS:
        if (camera.make.casefold(), camera.model.casefold()) == key:
            return camera
    raise UnsupportedRawCamera(
        f"No native RAW route is registered for {identity.make!r} {identity.model!r}"
    )


def select_raw_route(source: str | Path | CameraIdentity) -> RawRoute:
    """Dispatch by camera: the ZV-1 keeps the original frozen renderer.

    For the ZV-1 the returned renderer is
    :func:`registration.registered_edge_aware_sensor_linear` itself, with its
    frozen parameter document; no code of this module takes part in it.
    """
    identity = (
        source if isinstance(source, CameraIdentity) else read_camera_identity(source)
    )
    camera = lookup_camera(identity)
    if camera.route == FROZEN_ZV1_ROUTE:
        return RawRoute(
            camera,
            camera.route,
            renderer=registration.registered_edge_aware_sensor_linear,
            decoder=None,
        )
    return RawRoute(camera, camera.route, renderer=None, decoder=decode_sensor)


@dataclass(frozen=True)
class RawSensorMetadata:
    """Sensor constants of one RAW file, as LibRaw reports them."""

    make: str
    model: str
    raw_visible_wh: tuple[int, int]
    cfa_pattern: tuple[tuple[int, ...], ...]
    color_description: str
    black_level_per_channel: tuple[int, ...]
    white_level: int
    camera_white_level_per_channel: tuple[int, ...]
    camera_whitebalance: tuple[float, ...]
    rgb_xyz_matrix: tuple[tuple[float, ...], ...]
    libraw_flip: int
    exif_orientation: int | None
    libraw_crop_xywh: tuple[int, int, int, int] | None
    libraw_version: str

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-compatible copy for certificates."""
        return {
            "make": self.make,
            "model": self.model,
            "raw_visible_wh": list(self.raw_visible_wh),
            "cfa_pattern": [list(row) for row in self.cfa_pattern],
            "color_description": self.color_description,
            "black_level_per_channel": list(self.black_level_per_channel),
            "white_level": self.white_level,
            "camera_white_level_per_channel": list(self.camera_white_level_per_channel),
            "camera_whitebalance": list(self.camera_whitebalance),
            "rgb_xyz_matrix": [list(row) for row in self.rgb_xyz_matrix],
            "libraw_flip": self.libraw_flip,
            "exif_orientation": self.exif_orientation,
            "libraw_crop_xywh": (
                None if self.libraw_crop_xywh is None else list(self.libraw_crop_xywh)
            ),
            "libraw_version": self.libraw_version,
        }


def _int_pair(value: Sequence[Any], name: str) -> tuple[int, int]:
    pair = tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must have two integers")
    return pair  # type: ignore[return-value]


def _float_pair(value: Sequence[Any], name: str) -> tuple[float, float]:
    pair = tuple(float(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must have two numbers")
    return pair  # type: ignore[return-value]


def _finite_tuple(value: Sequence[Any], name: str) -> tuple[float, ...]:
    values = tuple(float(v) for v in value)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be a non-empty finite sequence")
    return values


def _canonical_digest(document: Mapping[str, Any]) -> str:
    text = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SensorProfile:
    """Frozen sensor constants for one camera, supplied as calibration data.

    When given to :func:`decode_sensor`, every file must report exactly these
    constants, as the frozen ZV-1 route checks its own. ``saturation_dn`` is an
    optional validated clip level; it is counted and recorded, while the
    contract's saturation mask stays at the LibRaw white level.
    """

    camera_key: str
    raw_visible_wh: tuple[int, int]
    cfa_pattern: tuple[tuple[int, ...], ...]
    color_description: str
    black_level_per_channel: tuple[int, ...]
    white_level: int
    saturation_dn: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def check(self, camera: RawCamera, metadata: RawSensorMetadata) -> None:
        """Refuse a file whose sensor constants differ from this profile."""
        if camera.key != self.camera_key:
            raise ValueError(
                f"Sensor profile is for {self.camera_key}, file is {camera.key}"
            )
        expected = (
            self.raw_visible_wh,
            self.cfa_pattern,
            self.color_description,
            self.black_level_per_channel,
            self.white_level,
        )
        actual = (
            metadata.raw_visible_wh,
            metadata.cfa_pattern,
            metadata.color_description,
            metadata.black_level_per_channel,
            metadata.white_level,
        )
        names = (
            "raw_visible_wh",
            "cfa_pattern",
            "color_description",
            "black_level_per_channel",
            "white_level",
        )
        for name, a, b in zip(names, expected, actual, strict=True):
            if a != b:
                raise ValueError(
                    f"RAW {name} differs from the frozen sensor profile: {b} != {a}"
                )

    def to_document(self) -> dict[str, Any]:
        """Return the JSON section of a camera profile document."""
        return {
            "camera_key": self.camera_key,
            "raw_visible_wh": list(self.raw_visible_wh),
            "cfa_pattern": [list(row) for row in self.cfa_pattern],
            "color_description": self.color_description,
            "black_level_per_channel": list(self.black_level_per_channel),
            "white_level": self.white_level,
            "saturation_dn": self.saturation_dn,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> SensorProfile:
        """Build a sensor profile from its JSON section, checking structure."""
        saturation = document.get("saturation_dn")
        return cls(
            camera_key=str(document["camera_key"]),
            raw_visible_wh=_int_pair(document["raw_visible_wh"], "raw_visible_wh"),
            cfa_pattern=tuple(
                tuple(int(v) for v in row) for row in document["cfa_pattern"]
            ),
            color_description=str(document["color_description"]),
            black_level_per_channel=tuple(
                int(v) for v in document["black_level_per_channel"]
            ),
            white_level=int(document["white_level"]),
            saturation_dn=None if saturation is None else int(saturation),
            provenance=dict(document.get("provenance") or {}),
        )

    @classmethod
    def from_metadata(
        cls,
        camera: RawCamera,
        metadata: RawSensorMetadata,
        *,
        saturation_dn: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> SensorProfile:
        """Freeze the constants one file reports, for review as calibration data."""
        return cls(
            camera.key,
            metadata.raw_visible_wh,
            metadata.cfa_pattern,
            metadata.color_description,
            metadata.black_level_per_channel,
            metadata.white_level,
            saturation_dn,
            dict(provenance or {}),
        )


@dataclass(frozen=True)
class GeometryProfile:
    """Explicit companion-JPEG to RAW map for one camera, in the ZV-1 rule's form.

    ``coordinates`` evaluates :func:`registration.raw_coordinates` with these
    values, i.e. for JPEG ``(u, v)`` and ``r = |(u, v) - centre_uv|``::

        x = cu + raw_offset_x + shift_x + s(r) (u - cu)
        s(r) = (1 + amplitude D(r/R) / distortion_denominator)
               [x (1 + CA_c(r/R) / ca_denominator) for c in ca]

    with linear interpolation of the tables on ``knots``. ``validation`` is
    ``None`` until a measured agreement is attached with :meth:`with_validation`.
    """

    camera_key: str
    jpeg_wh: tuple[int, int]
    raw_visible_wh: tuple[int, int]
    raw_offset_xy: tuple[float, float]
    centre_uv: tuple[float, float]
    radius_px: float
    knots: tuple[float, ...]
    distortion: tuple[float, ...]
    amplitude: float
    distortion_denominator: float
    ca: Mapping[str, tuple[float, ...]] | None
    ca_denominator: float
    shift_xy: tuple[float, float] = (0.0, 0.0)
    orientation: int | None = 1
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # Canonical types keep the digest stable across JSON round trips.
        coerce = {
            "jpeg_wh": _int_pair(self.jpeg_wh, "jpeg_wh"),
            "raw_visible_wh": _int_pair(self.raw_visible_wh, "raw_visible_wh"),
            "raw_offset_xy": _float_pair(self.raw_offset_xy, "raw_offset_xy"),
            "centre_uv": _float_pair(self.centre_uv, "centre_uv"),
            "shift_xy": _float_pair(self.shift_xy, "shift_xy"),
            "radius_px": float(self.radius_px),
            "amplitude": float(self.amplitude),
            "knots": tuple(float(v) for v in self.knots),
            "distortion": tuple(float(v) for v in self.distortion),
            "distortion_denominator": float(self.distortion_denominator),
            "ca_denominator": float(self.ca_denominator),
            "ca": (
                None
                if self.ca is None
                else {str(k): tuple(float(x) for x in v) for k, v in self.ca.items()}
            ),
            "orientation": None if self.orientation is None else int(self.orientation),
        }
        for name, value in coerce.items():
            object.__setattr__(self, name, value)
        if len(self.knots) < 2 or len(self.distortion) != len(self.knots):
            raise ValueError("Geometry profile tables must match its knots")
        if any(b <= a for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError("Geometry profile knots must increase")
        if self.ca is not None:
            if set(self.ca) - {"R", "G", "B"}:
                raise ValueError("Geometry profile CA tables are keyed by R, G or B")
            if any(len(table) != len(self.knots) for table in self.ca.values()):
                raise ValueError("Geometry profile CA tables must match its knots")
        numbers = (
            *self.raw_offset_xy,
            *self.centre_uv,
            *self.shift_xy,
            self.radius_px,
            self.amplitude,
            self.distortion_denominator,
            self.ca_denominator,
        )
        if not all(math.isfinite(float(v)) for v in numbers) or self.radius_px <= 0:
            raise ValueError("Geometry profile parameters must be finite, with R > 0")
        if self.distortion_denominator == 0 or self.ca_denominator == 0:
            raise ValueError("Geometry profile denominators must be non-zero")
        if min(self.jpeg_wh) <= 0 or min(self.raw_visible_wh) <= 0:
            raise ValueError("Geometry profile grids must be non-empty")

    def coefficients(self) -> dict[str, float]:
        """Per-profile coefficients in the frozen route's naming."""
        return dict(
            cu=float(self.centre_uv[0]),
            cv=float(self.centre_uv[1]),
            tx=float(self.shift_xy[0]),
            ty=float(self.shift_xy[1]),
            R=float(self.radius_px),
            amp=float(self.amplitude),
        )

    def lens_parameters(self) -> dict[str, Any]:
        """Lens-model section in the frozen route's naming."""
        return {
            "knots": list(self.knots),
            "distortion_denominator": self.distortion_denominator,
            "ca_denominator": self.ca_denominator,
            "raw_offset_xy": list(self.raw_offset_xy),
        }

    def coordinates(
        self, u: np.ndarray, v: np.ndarray, channel: str, h: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full-resolution RAW coordinates (pixel centres at integers) of a grid."""
        if channel not in ("R", "G", "B"):
            raise ValueError("Channel must be R, G or B")
        ca = None if self.ca is None else self.ca.get(channel)
        homography = np.eye(3) if h is None else np.asarray(h, np.float64)
        return registration.raw_coordinates(
            u,
            v,
            homography,
            self.coefficients(),
            self.distortion,
            ca,
            parameters=self.lens_parameters(),
        )

    def to_document(self, *, include_validation: bool = True) -> dict[str, Any]:
        """Return the JSON section of a camera profile document."""
        document = {
            "camera_key": self.camera_key,
            "jpeg_wh": list(self.jpeg_wh),
            "raw_visible_wh": list(self.raw_visible_wh),
            "raw_offset_xy": list(self.raw_offset_xy),
            "centre_uv": list(self.centre_uv),
            "shift_xy": list(self.shift_xy),
            "radius_px": self.radius_px,
            "knots": list(self.knots),
            "distortion": list(self.distortion),
            "amplitude": self.amplitude,
            "distortion_denominator": self.distortion_denominator,
            "ca": (
                None
                if self.ca is None
                else {k: list(v) for k, v in sorted(self.ca.items())}
            ),
            "ca_denominator": self.ca_denominator,
            "orientation": self.orientation,
            "provenance": dict(self.provenance),
        }
        if include_validation:
            document["validation"] = (
                None if self.validation is None else dict(self.validation)
            )
        return document

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical profile document without its validation."""
        return _canonical_digest(self.to_document(include_validation=False))

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> GeometryProfile:
        """Build a geometry profile from its JSON section, checking structure."""
        ca = document.get("ca")
        orientation = document.get("orientation")
        validation = document.get("validation")
        return cls(
            camera_key=str(document["camera_key"]),
            jpeg_wh=_int_pair(document["jpeg_wh"], "jpeg_wh"),
            raw_visible_wh=_int_pair(document["raw_visible_wh"], "raw_visible_wh"),
            raw_offset_xy=_float_pair(document["raw_offset_xy"], "raw_offset_xy"),
            centre_uv=_float_pair(document["centre_uv"], "centre_uv"),
            shift_xy=_float_pair(document.get("shift_xy", (0.0, 0.0)), "shift_xy"),
            radius_px=float(document["radius_px"]),
            knots=_finite_tuple(document["knots"], "knots"),
            distortion=_finite_tuple(document["distortion"], "distortion"),
            amplitude=float(document["amplitude"]),
            distortion_denominator=float(document["distortion_denominator"]),
            ca=(
                None
                if ca is None
                else {str(k): _finite_tuple(v, "ca") for k, v in ca.items()}
            ),
            ca_denominator=float(document["ca_denominator"]),
            orientation=None if orientation is None else int(orientation),
            provenance=dict(document.get("provenance") or {}),
            validation=None if validation is None else dict(validation),
        )

    def covers(self, channel: str) -> bool:
        """Whether a passed validation includes this channel."""
        record = self.validation
        return (
            bool(record)
            and record.get("passed") is True
            and channel in record.get("channels", ())
        )

    def with_validation(
        self,
        agreement: Mapping[str, Any],
        *,
        channels: Sequence[str],
        tolerance_p95_px: float,
        tolerance_mean_px: float,
    ) -> GeometryProfile:
        """Attach measured agreement evidence after checking it passes.

        The evidence must have been measured with this exact profile. Every
        listed channel must meet both tolerances; otherwise nothing is attached.
        """
        if agreement.get("profile_digest") != self.digest:
            raise ValueError(
                "Agreement evidence was measured with another geometry profile"
            )
        failures = []
        for channel in channels:
            metrics = agreement.get("channels", {}).get(channel)
            if metrics is None:
                failures.append(f"{channel}: not measured")
                continue
            mean = math.hypot(metrics["mean_dx_px"], metrics["mean_dy_px"])
            if (
                metrics["p95_px"] is None
                or metrics["p95_px"] > tolerance_p95_px
                or mean > tolerance_mean_px
            ):
                failures.append(
                    f"{channel}: p95 {metrics['p95_px']} px, |mean| {mean:.3f} px"
                )
        if failures:
            raise GeometryProfileUnavailable(
                "Geometry profile does not meet the stated tolerance: "
                + "; ".join(failures)
            )
        record = {
            "passed": True,
            "channels": list(channels),
            "tolerance_p95_px": float(tolerance_p95_px),
            "tolerance_mean_px": float(tolerance_mean_px),
            "evidence": [dict(agreement)],
        }
        return replace(self, validation=record)


@dataclass(frozen=True)
class CameraProfile:
    """Per-camera calibration document: frozen sensor constants and geometry."""

    camera_key: str
    sensor: SensorProfile | None = None
    geometry: GeometryProfile | None = None

    def to_document(self) -> dict[str, Any]:
        """Return the JSON document."""
        return {
            "format": PROFILE_FORMAT,
            "camera_key": self.camera_key,
            "sensor": None if self.sensor is None else self.sensor.to_document(),
            "geometry": None if self.geometry is None else self.geometry.to_document(),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> CameraProfile:
        """Build a camera profile from its JSON document, checking structure."""
        if document.get("format") != PROFILE_FORMAT:
            raise ValueError(
                f"Unsupported camera profile format {document.get('format')!r}; "
                f"expected {PROFILE_FORMAT}"
            )
        key = str(document["camera_key"])
        if key not in {camera.key for camera in REGISTERED_CAMERAS}:
            raise UnsupportedRawCamera(
                "Camera profile names an unregistered camera: " + key
            )
        sensor = document.get("sensor")
        geometry = document.get("geometry")
        profile = cls(
            key,
            None if sensor is None else SensorProfile.from_document(sensor),
            None if geometry is None else GeometryProfile.from_document(geometry),
        )
        for part in (profile.sensor, profile.geometry):
            if part is not None and part.camera_key != key:
                raise ValueError("Camera profile sections name different cameras")
        return profile


def load_camera_profile(path: str | Path, expected_sha256: str) -> CameraProfile:
    """Load a content-bound camera profile document (JSON)."""
    return CameraProfile.from_document(read_bound_json(path, expected_sha256))


@dataclass(frozen=True, eq=False)
class DecodedSensor:
    """Black-subtracted integer sensor values on the RAW visible grid."""

    camera: RawCamera
    metadata: RawSensorMetadata
    source: Mapping[str, str]
    black_level: int
    mosaic_dn: np.ndarray
    demosaiced_dn: np.ndarray
    saturated: np.ndarray
    demosaic_code: str
    sensor_profile: SensorProfile | None = None

    @property
    def white_range_dn(self) -> int:
        """Denominator of the normalised view: white level minus black level."""
        return int(self.metadata.white_level) - int(self.black_level)

    def linear_dn(self) -> np.ndarray:
        """Camera-native linear RGB in RAW units, CHW float32."""
        return np.moveaxis(self.demosaiced_dn, -1, 0).astype(np.float32)

    def linear_normalised(self) -> np.ndarray:
        """The same values divided by ``white_level - black_level`` (CHW float32)."""
        return self.linear_dn() / np.float32(self.white_range_dn)

    def cfa_plane(self, channel: str) -> tuple[np.ndarray, tuple[float, float]]:
        """Mean of one colour's CFA sites per 2x2 block, and its RAW phase (x, y)."""
        pattern = self.metadata.cfa_pattern
        sites = [
            (row, col)
            for row in range(2)
            for col in range(2)
            if self.metadata.color_description[pattern[row][col]] == channel
        ]
        if not sites:
            raise ValueError("CFA has no sites of channel " + channel)
        plane = self.mosaic_dn[sites[0][0] :: 2, sites[0][1] :: 2].astype(np.float32)
        for row, col in sites[1:]:
            plane = plane + self.mosaic_dn[row::2, col::2].astype(np.float32)
        if len(sites) > 1:
            plane = np.float32(1.0 / len(sites)) * plane
        phase = (
            float(np.mean([c for _, c in sites])),
            float(np.mean([r for r, _ in sites])),
        )
        return plane, phase

    def certificate(self) -> dict[str, Any]:
        """Describe the decoding contract, sources and recorded metadata."""
        meta = self.metadata
        declared = [
            v for v in meta.camera_white_level_per_channel if v > self.black_level
        ]
        camera_white = min(declared) if declared else None

        def count_at_or_above(level: int) -> int:
            # mosaic = max(raw - black, 0), so raw >= level <=> mosaic >= level - black.
            threshold = int(level) - int(self.black_level)
            if threshold <= 0:
                return int(self.mosaic_dn.size)
            return int(np.count_nonzero(self.mosaic_dn >= threshold))

        record = {
            "contract": RAW_CAMERA_CONTRACT,
            "route": METADATA_ROUTE,
            "camera": {
                "key": self.camera.key,
                "make": meta.make,
                "model": meta.model,
                "series": list(self.camera.series),
            },
            "source": dict(self.source),
            "sensor_metadata": meta.as_record(),
            "sensor_constants_source": (
                "file metadata (LibRaw), checked against frozen sensor profile"
                if self.sensor_profile is not None
                else "file metadata (LibRaw)"
            ),
            "sensor_profile_digest": (
                None
                if self.sensor_profile is None
                else _canonical_digest(self.sensor_profile.to_document())
            ),
            "black_level_subtracted": self.black_level,
            "white_level": meta.white_level,
            "demosaic": (
                f"OpenCV {self.demosaic_code}, full sensor resolution, "
                "16-bit integer after black subtraction"
            ),
            "channel_basis": "camera-native black-subtracted linear RGB, RAW units",
            "normalisation": {
                "applied": False,
                "denominator_dn": self.white_range_dn,
                "view": "linear_normalised() = DN / (white_level - black_level)",
            },
            "white_balance": {
                "applied": False,
                "camera_wb_metadata": list(meta.camera_whitebalance),
                "reason": _WHITE_BALANCE_REASON,
            },
            "camera_to_xyz": {
                "applied": False,
                "rgb_xyz_metadata": [list(r) for r in meta.rgb_xyz_matrix],
                "reason": _XYZ_REASON,
            },
            "gamma_tone_auto_bright": "none",
            "sensor_saturated_fraction": float(self.saturated.mean()),
            "sensor_saturated_count": int(self.saturated.sum()),
            "at_or_above_camera_white_level_count": (
                None if camera_white is None else count_at_or_above(camera_white)
            ),
            "demosaiced_shape_hwc": list(self.demosaiced_dn.shape),
            "training_eligible": False,
            "eligibility_reason": _ELIGIBILITY_REASON,
        }
        if (
            self.sensor_profile is not None
            and self.sensor_profile.saturation_dn is not None
        ):
            record["at_or_above_profile_saturation_count"] = count_at_or_above(
                self.sensor_profile.saturation_dn
            )
        return record


def decode_sensor(
    raw_path: str | Path,
    *,
    parameters: Mapping,
    camera_identity: CameraIdentity | None = None,
    sensor_profile: SensorProfile | None = None,
    expected_sha256: str | None = None,
) -> DecodedSensor:
    """Decode a non-ZV-1 RAW file under the native sensor contract.

    ``parameters`` is the same registration section the frozen route receives;
    only ``parameters["edge_aware"]`` (demosaic code and channel-order fixture)
    is read, so the demosaic is the configured one. Camera constants come from
    the file through LibRaw; a ``sensor_profile`` additionally freezes them.
    """
    path = Path(raw_path)
    identity = (
        camera_identity if camera_identity is not None else read_camera_identity(path)
    )
    camera = lookup_camera(identity)
    if camera.route == FROZEN_ZV1_ROUTE:
        raise FrozenRouteRequired(
            f"{camera.make} {camera.model} files use the frozen native route "
            "(registration.registered_edge_aware_sensor_linear) unchanged"
        )
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise ValueError("Missing or mismatched exact source hash: raw_sha256")
    registration.check_demosaic_order(parameters=parameters)
    demosaic_code = str(parameters["edge_aware"]["demosaic_code"])
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("RAW sensor decoding requires rawpy (CPU LibRaw)") from error
    with rawpy.imread(str(path)) as raw:
        original = np.array(raw.raw_image_visible, copy=True)
        sizes = getattr(raw, "sizes", None)
        crop = None
        if (
            sizes is not None
            and getattr(sizes, "crop_width", 0)
            and getattr(sizes, "crop_height", 0)
        ):
            crop = (
                int(sizes.crop_left_margin),
                int(sizes.crop_top_margin),
                int(sizes.crop_width),
                int(sizes.crop_height),
            )
        metadata = RawSensorMetadata(
            make=identity.make,
            model=identity.model,
            raw_visible_wh=(int(original.shape[1]), int(original.shape[0])),
            cfa_pattern=tuple(
                tuple(int(v) for v in row) for row in np.asarray(raw.raw_pattern)
            ),
            color_description=bytes(raw.color_desc).decode("ascii"),
            black_level_per_channel=tuple(int(v) for v in raw.black_level_per_channel),
            white_level=int(raw.white_level),
            camera_white_level_per_channel=tuple(
                int(v)
                for v in (getattr(raw, "camera_white_level_per_channel", None) or ())
            ),
            camera_whitebalance=tuple(float(v) for v in raw.camera_whitebalance),
            rgb_xyz_matrix=tuple(
                tuple(float(v) for v in row) for row in np.asarray(raw.rgb_xyz_matrix)
            ),
            libraw_flip=int(getattr(sizes, "flip", 0) or 0),
            exif_orientation=identity.orientation,
            libraw_crop_xywh=crop,
            libraw_version=".".join(
                str(v) for v in getattr(rawpy, "libraw_version", ())
            ),
        )
    if original.ndim != 2 or original.dtype != np.uint16:
        raise ValueError("RAW visible mosaic must be a 2-D uint16 array")
    if original.shape[0] % 2 or original.shape[1] % 2:
        raise ValueError("Unexpected odd Bayer geometry")
    if (
        metadata.cfa_pattern != RGGB_PATTERN
        or metadata.color_description != RGGB_DESCRIPTION
    ):
        raise ValueError(
            "The configured edge-aware demosaic is verified for RGGB only; "
            f"this file reports {metadata.cfa_pattern} {metadata.color_description!r}"
        )
    levels = metadata.black_level_per_channel
    if len(levels) != 4 or len(set(levels)) != 1:
        raise ValueError(
            f"The native contract subtracts one black level; LibRaw reports {levels}"
        )
    black = int(levels[0])
    if not 0 <= black < metadata.white_level <= np.iinfo(np.uint16).max:
        raise ValueError(
            f"Unexpected RAW black/white levels: {black}/{metadata.white_level}"
        )
    # The frozen route's uint16 demosaic relies on white < 16384 after black
    # subtraction (no overflow); refuse rather than assume headroom.
    if metadata.white_level - black >= 2**14:
        raise ValueError(
            "RAW white level leaves no 14-bit headroom for the uint16 demosaic: "
            f"{metadata.white_level} - {black}"
        )
    if sensor_profile is not None:
        sensor_profile.check(camera, metadata)
    saturated = original >= metadata.white_level
    # Same arithmetic as the frozen edge-aware route: subtract black before the
    # demosaic, clamp at zero, keep RAW units.
    black_subtracted = np.maximum(original.astype(np.int32) - black, 0).astype(
        np.uint16
    )
    del original
    demosaiced = cv2.demosaicing(black_subtracted, getattr(cv2, demosaic_code))
    if demosaiced.shape != (metadata.raw_visible_wh[1], metadata.raw_visible_wh[0], 3):
        raise ValueError("Full-resolution demosaic shape mismatch")
    return DecodedSensor(
        camera=camera,
        metadata=metadata,
        source={"path": str(path), "sha256": digest},
        black_level=black,
        mosaic_dn=black_subtracted,
        demosaiced_dn=demosaiced,
        saturated=saturated,
        demosaic_code=demosaic_code,
        sensor_profile=sensor_profile,
    )


def _parse_makernote_longs(
    note: bytes, byteorder: str, tags: Mapping[int, str]
) -> dict[str, int]:
    """Read inline LONG values of a Sony maker-note IFD (with or without header)."""
    offset = 12 if note[:9] in (b"SONY DSC ", b"SONY CAM ") else 0
    if len(note) < offset + 2:
        return {}
    count = struct.unpack_from(byteorder + "H", note, offset)[0]
    values: dict[str, int] = {}
    for index in range(count):
        start = offset + 2 + 12 * index
        if start + 12 > len(note):
            break
        tag, kind, number, value = struct.unpack_from(byteorder + "HHII", note, start)
        if tag in tags and kind == 4 and number == 1:
            values[tags[tag]] = int(value)
    return values


def read_sony_lens_metadata(arw_path: str | Path) -> dict[str, Any]:
    """Read the Sony lens tables, crop tags and in-camera correction switches.

    Tables: SubIFD ``0x7037`` DistortionCorrParams and ``0x7035``
    ChromaticAberrationCorrParams, each a count followed by the values (one
    distortion table; R then B CA tables). Switches: maker-note ``0x2011``
    vignetting, ``0x2012`` lateral CA and ``0x2013`` distortion correction.
    """
    with tifffile.TiffFile(arw_path) as tiff:
        main = tiff.pages[0]
        pages = [main, *(list(main.pages) if getattr(main, "pages", None) else [])]

        def single(code: int) -> tuple[int, ...] | None:
            found = [
                tuple(int(v) for v in np.atleast_1d(page.tags.get(code).value))
                for page in pages
                if page.tags.get(code) is not None
            ]
            if len(found) > 1 and len(set(found)) > 1:
                raise ValueError(f"Conflicting copies of TIFF tag {code:#x}")
            return found[0] if found else None

        distortion = single(SONY_DISTORTION_TAG)
        ca = single(SONY_CA_TAG)
        origin = single(DEFAULT_CROP_ORIGIN_TAG)
        size = single(DEFAULT_CROP_SIZE_TAG)
        sony_origin = single(SONY_CROP_TOP_LEFT_TAG)
        sony_size = single(SONY_CROP_SIZE_TAG)
        exif = main.tags.get(EXIF_IFD_TAG)
        note = (
            exif.value.get("MakerNote")
            if exif is not None and isinstance(exif.value, dict)
            else None
        )
        raw_settings = (
            _parse_makernote_longs(bytes(note), tiff.byteorder, SONY_MAKERNOTE_SETTINGS)
            if note
            else {}
        )
    if distortion is None or ca is None:
        raise GeometryProfileUnavailable("ARW has no Sony lens tables (0x7037/0x7035)")
    n = distortion[0]
    if n < 2 or len(distortion) < n + 1 or ca[0] != 2 * n or len(ca) < 2 * n + 1:
        raise ValueError("Unexpected Sony lens-table structure")
    settings = {
        name: SONY_SETTING_VALUES.get(value, f"unknown({value})")
        for name, value in raw_settings.items()
    }
    return {
        "distortion": distortion[1 : n + 1],
        "ca_red": ca[1 : n + 1],
        "ca_blue": ca[n + 1 : 2 * n + 1],
        "distortion_header": n,
        "ca_header": ca[0],
        "default_crop_origin": origin,
        "default_crop_size": size,
        "sony_crop_top_left": sony_origin,
        "sony_crop_size": sony_size,
        "settings": settings,
    }


def _switch(settings: Mapping[str, str], name: str) -> bool:
    value = settings.get(name)
    if value not in ("Off", "Auto"):
        raise GeometryProfileUnavailable(
            f"In-camera {name.replace('_', ' ')} setting is {value!r}; "
            "cannot decide whether the JPEG carries that correction"
        )
    return value == "Auto"


def derive_sony_embedded_profile(
    decoded: DecodedSensor, *, jpeg_wh: Sequence[int], raw_offset_xy: Sequence[float]
) -> GeometryProfile:
    """Build an unvalidated profile from the ARW's own lens metadata.

    Read from the file: the distortion and CA tables, which of them the camera
    applied to its JPEG (maker-note switches), and the DefaultCrop, whose
    centre is the table centre and whose half-diagonal is the radius ``R``
    (as for the ZV-1). The fixed crop ``raw_offset_xy`` is per-camera
    calibration, as ``lens.raw_offset_xy`` is for the ZV-1: the ARW crop tags
    do not state it unambiguously, so the candidates they suggest are recorded
    in the provenance and the caller supplies the value.
    """
    if decoded.camera.lens_metadata != SONY_LENS_TABLES:
        raise GeometryProfileUnavailable(
            decoded.camera.key + " has no Sony lens tables"
        )
    lens = read_sony_lens_metadata(decoded.source["path"])
    width, height = _int_pair(jpeg_wh, "jpeg_wh")
    origin, size = lens["default_crop_origin"], lens["default_crop_size"]
    if origin is None or size is None:
        raise GeometryProfileUnavailable(
            "ARW has no DefaultCrop; table centre and radius are unknown"
        )
    declared = lens["sony_crop_size"] or size
    if (width, height) != tuple(declared):
        raise GeometryProfileUnavailable(
            f"Companion JPEG size {(width, height)} differs from the crop "
            f"the ARW declares {tuple(declared)}"
        )
    ox, oy = (float(v) for v in raw_offset_xy)
    raw_w, raw_h = decoded.metadata.raw_visible_wh
    if ox < 0 or oy < 0 or ox + width > raw_w or oy + height > raw_h:
        raise ValueError("JPEG crop lies outside the RAW visible area")
    distortion_on = _switch(lens["settings"], "distortion_correction")
    ca_on = _switch(lens["settings"], "lateral_chromatic_aberration")
    n = lens["distortion_header"]
    sony_origin = lens["sony_crop_top_left"]
    libraw_crop = decoded.metadata.libraw_crop_xywh
    candidates = {"default_crop_origin": list(origin)}
    if sony_origin is not None:
        candidates["sony_crop_top_left"] = list(sony_origin)
        candidates["default_crop_origin_plus_sony_crop_top_left"] = [
            origin[0] + sony_origin[0],
            origin[1] + sony_origin[1],
        ]
        candidates["default_crop_x_sony_crop_y"] = [origin[0], sony_origin[1]]
    if libraw_crop is not None:
        candidates["libraw_crop_origin"] = list(libraw_crop[:2])
    provenance = {
        "derivation": "sony-embedded-lens-tables-v1",
        "raw_sha256": decoded.source["sha256"],
        "distortion_tag": f"{SONY_DISTORTION_TAG:#06x}",
        "ca_tag": f"{SONY_CA_TAG:#06x}",
        "table_knots": n,
        "in_camera_settings": dict(lens["settings"]),
        "distortion_applied": distortion_on,
        "lateral_ca_applied": ca_on,
        "default_crop_origin": list(origin),
        "default_crop_size": list(size),
        "sony_crop_top_left": None if sony_origin is None else list(sony_origin),
        "sony_crop_size": (
            None if lens["sony_crop_size"] is None else list(lens["sony_crop_size"])
        ),
        "crop_origin_source": "explicit calibration (raw_offset_xy)",
        "crop_origin_candidates_from_metadata": candidates,
        "centre": "DefaultCrop centre",
        "radius": "DefaultCrop half-diagonal",
    }
    return GeometryProfile(
        camera_key=decoded.camera.key,
        jpeg_wh=(width, height),
        raw_visible_wh=decoded.metadata.raw_visible_wh,
        raw_offset_xy=(ox, oy),
        centre_uv=(
            origin[0] + (size[0] - 1) / 2 - ox,
            origin[1] + (size[1] - 1) / 2 - oy,
        ),
        radius_px=float(math.hypot(size[0] / 2, size[1] / 2)),
        knots=tuple(float(v) for v in np.linspace(0.0, 1.0, n)),
        distortion=tuple(float(v) for v in lens["distortion"]),
        amplitude=1.0 if distortion_on else 0.0,
        distortion_denominator=SONY_DISTORTION_DENOMINATOR,
        ca=(
            {
                "R": tuple(float(v) for v in lens["ca_red"]),
                "B": tuple(float(v) for v in lens["ca_blue"]),
            }
            if ca_on
            else None
        ),
        ca_denominator=SONY_CA_DENOMINATOR,
        orientation=decoded.metadata.exif_orientation,
        provenance=provenance,
    )


def derive_embedded_profile(
    decoded: DecodedSensor, *, jpeg_wh: Sequence[int], raw_offset_xy: Sequence[float]
) -> GeometryProfile:
    """Derive a profile where the RAW file carries lens metadata, else refuse."""
    if decoded.camera.lens_metadata == SONY_LENS_TABLES:
        return derive_sony_embedded_profile(
            decoded, jpeg_wh=jpeg_wh, raw_offset_xy=raw_offset_xy
        )
    raise GeometryProfileUnavailable(
        f"{decoded.camera.make} {decoded.camera.model}: "
        "the RAW file carries no lens-correction "
        "tables the native rule can read. The RAW-to-JPEG geometry is per-camera calibration "
        "and must be supplied as an explicit, validated geometry profile."
    )


def _check_profile_matches(decoded: DecodedSensor, profile: GeometryProfile) -> None:
    if profile.camera_key != decoded.camera.key:
        raise ValueError(
            f"Geometry profile is for {profile.camera_key}, RAW is {decoded.camera.key}"
        )
    if tuple(profile.raw_visible_wh) != tuple(decoded.metadata.raw_visible_wh):
        raise ValueError("Geometry profile RAW grid differs from the decoded RAW")
    if (
        profile.orientation is not None
        and decoded.metadata.exif_orientation is not None
        and profile.orientation != decoded.metadata.exif_orientation
    ):
        raise ValueError("Geometry profile orientation differs from the RAW file")


def measure_geometry_agreement(
    decoded: DecodedSensor,
    companion_jpeg: str | Path,
    profile: GeometryProfile,
    *,
    parameters: Mapping,
    channels: Sequence[str] = ("G",),
) -> dict[str, Any]:
    """Measure how well a profile maps RAW CFA planes onto the companion JPEG.

    As in the ZV-1 geometry study: each CFA plane (black-subtracted, mean of
    its sites) is warped onto the JPEG half-resolution grid (2x2 box means,
    samples at ``2a + 0.5``) with the profile, both are band-passed in log
    units, and textured templates are matched with the frozen matcher.
    Displacements are full-resolution JPEG pixels; ``radial_median_px`` gives
    the median radial component per ``r/R`` bin, the signature of a missing or
    wrong lens correction. ``parameters``: ``sigma_small``, ``sigma_large``,
    ``minimum_template_std``, ``template_half``, ``search_half``,
    ``grid_step_px``, ``border_px``, ``min_correlation``, ``minimum_matches``,
    ``radial_bin_edges``.
    """
    jpeg = Path(companion_jpeg)
    if jpeg.stem.casefold() != Path(decoded.source["path"]).stem.casefold():
        raise ValueError("RAW and companion JPEG stem mismatch")
    _check_profile_matches(decoded, profile)
    with Image.open(jpeg) as image:
        if image.size != tuple(profile.jpeg_wh):
            raise ValueError(
                f"Companion JPEG is {image.size}, profile expects {tuple(profile.jpeg_wh)}"
            )
        rgb = np.asarray(image.convert("RGB"))
    width, height = profile.jpeg_wh
    hh, ww = height // 2, width // 2
    ay, ax = np.mgrid[0:hh, 0:ww].astype(np.float32)
    u, v = 2 * ax + 0.5, 2 * ay + 0.5
    match = {
        key: parameters[key]
        for key in ("sigma_small", "sigma_large", "minimum_template_std")
    }
    border, step = int(parameters["border_px"]) // 2, max(
        1, int(parameters["grid_step_px"]) // 2
    )
    points = [
        (x, y)
        for y in range(border, hh - border, step)
        for x in range(border, ww - border, step)
    ]
    edges = [float(e) for e in parameters["radial_bin_edges"]]
    cu, cv = profile.centre_uv
    result: dict[str, Any] = {}
    for channel in channels:
        jpeg_half = rgb[: 2 * hh, : 2 * ww, "RGB".index(channel)].astype(np.float32)
        jpeg_half = jpeg_half.reshape(hh, 2, ww, 2).mean((1, 3))
        plane, (phase_x, phase_y) = decoded.cfa_plane(channel)
        x, y = profile.coordinates(u, v, channel)
        mx = ((x - phase_x) / 2).astype(np.float32)
        my = ((y - phase_y) / 2).astype(np.float32)
        warped = cv2.remap(
            np.log(np.maximum(plane, 1.0)),
            mx,
            my,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        a = registration.bandpass(np.log(jpeg_half + 1.0), parameters=match)
        b = registration.bandpass(warped, parameters=match)
        rows = registration.match_rows(
            a,
            b,
            points,
            template_half=int(parameters["template_half"]),
            search_half=int(parameters["search_half"]),
            guess=lambda gx, gy: (gx, gy),
            min_correlation=float(parameters["min_correlation"]),
            parameters=match,
        )
        metrics = registration.match_metrics(rows, scale=2.0)
        if metrics["n"] < int(parameters["minimum_matches"]):
            raise ValueError(
                f"Insufficient RAW-JPEG matches for {channel}: {metrics['n']}"
            )
        displacement = 2.0 * (rows[:, 2:4] - rows[:, :2])
        offset = np.column_stack((2 * rows[:, 0] + 0.5 - cu, 2 * rows[:, 1] + 0.5 - cv))
        radius = np.hypot(offset[:, 0], offset[:, 1])
        radial = (displacement * offset).sum(1) / np.maximum(radius, 1e-9)
        relative = radius / profile.radius_px
        bins = []
        for lo, hi in zip(edges, edges[1:]):
            inside = (relative >= lo) & (relative < hi)
            bins.append(
                {
                    "r_over_R": [lo, hi],
                    "n": int(inside.sum()),
                    "median_px": (
                        float(np.median(radial[inside])) if inside.any() else None
                    ),
                }
            )
        metrics.update(
            mean_dx_px=float(displacement[:, 0].mean()),
            mean_dy_px=float(displacement[:, 1].mean()),
            radial_median_px=bins,
        )
        result[channel] = metrics
    return {
        "profile_digest": profile.digest,
        "camera_key": profile.camera_key,
        "raw_sha256": decoded.source["sha256"],
        "jpeg_sha256": sha256_file(jpeg),
        "parameters": {
            k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in parameters.items()
        },
        "channels": result,
    }


def render_on_jpeg_grid(
    decoded: DecodedSensor,
    profile: GeometryProfile | None,
    *,
    channels: str = "RGB",
    h: np.ndarray | None = None,
    output_wh: Sequence[int] | None = None,
    window_xywh: Sequence[int] | None = None,
    allow_unvalidated: bool = False,
    tile_rows: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remap demosaiced channels onto the JPEG (or a homography-linked) grid.

    Same per-channel bilinear float32 remap with NaN outside the sensor as
    the frozen edge-aware route. A missing profile, an unvalidated profile or
    a channel outside its validation is refused unless ``allow_unvalidated``,
    and the certificate then records that the geometry was not validated.
    """
    if profile is None:
        raise GeometryProfileUnavailable(
            "No geometry profile is supplied for "
            f"{decoded.camera.make} {decoded.camera.model}; "
            "RAW-to-JPEG geometry is per-camera calibration and is never guessed"
        )
    _check_profile_matches(decoded, profile)
    uncovered = [c for c in channels if not profile.covers(c)]
    if uncovered and not allow_unvalidated:
        raise GeometryProfileUnavailable(
            "Geometry profile has no passed validation for channel(s) "
            + ", ".join(uncovered)
        )
    if h is None:
        grid_wh = tuple(profile.jpeg_wh)
    elif output_wh is None:
        raise ValueError("A homography-linked grid needs output_wh")
    else:
        grid_wh = _int_pair(output_wh, "output_wh")
    x0, y0, width, height = (
        (0, 0, *grid_wh) if window_xywh is None else map(int, window_xywh)
    )
    if (
        width <= 0
        or height <= 0
        or x0 < 0
        or y0 < 0
        or x0 + width > grid_wh[0]
        or y0 + height > grid_wh[1]
    ):
        raise ValueError("Render window lies outside the output grid")
    planes = {
        c: decoded.demosaiced_dn[..., "RGB".index(c)].astype(np.float32)
        for c in channels
    }
    output = np.empty((len(channels), height, width), np.float32)
    for first in range(0, height, int(tile_rows)):
        last = min(height, first + int(tile_rows))
        ys, xs = np.mgrid[y0 + first : y0 + last, x0 : x0 + width].astype(np.float32)
        for index, channel in enumerate(channels):
            rx, ry = profile.coordinates(xs, ys, channel, h)
            output[index, first:last] = cv2.remap(
                planes[channel],
                rx.astype(np.float32),
                ry.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )
    finite = np.isfinite(output)
    certificate = {
        "contract": RAW_RENDER_CONTRACT,
        "decoding": decoded.certificate(),
        "geometry_profile_digest": profile.digest,
        "geometry_profile_validated": not uncovered,
        "geometry_validation": (
            None
            if profile.validation is None
            else {
                k: profile.validation[k]
                for k in ("passed", "channels", "tolerance_p95_px", "tolerance_mean_px")
                if k in profile.validation
            }
        ),
        "geometry_provenance": dict(profile.provenance),
        "channels": channels,
        "grid_wh": list(grid_wh),
        "window_xywh": [x0, y0, width, height],
        "grid_to_camera_h": None if h is None else np.asarray(h, np.float64).tolist(),
        "native_warp_interpolation": "per-channel bilinear float32; NaN outside sensor",
        "nonfinite_fraction": float(1 - finite.mean()),
        "negative_fraction": float((output < 0).mean()),
        "above_white_fraction": float((output > decoded.white_range_dn).mean()),
        "output_shape": list(output.shape),
        "training_eligible": False,
        "eligibility_reason": _ELIGIBILITY_REASON,
    }
    return output, certificate


__all__ = [
    "CameraIdentity",
    "CameraProfile",
    "DecodedSensor",
    "FROZEN_ZV1_ROUTE",
    "FrozenRouteRequired",
    "GeometryProfile",
    "GeometryProfileUnavailable",
    "METADATA_ROUTE",
    "PROFILE_FORMAT",
    "REGISTERED_CAMERAS",
    "RawCamera",
    "RawRoute",
    "RawSensorMetadata",
    "SensorProfile",
    "UnsupportedRawCamera",
    "decode_sensor",
    "derive_embedded_profile",
    "derive_sony_embedded_profile",
    "load_camera_profile",
    "lookup_camera",
    "measure_geometry_agreement",
    "read_camera_identity",
    "read_sony_lens_metadata",
    "render_on_jpeg_grid",
    "select_raw_route",
]
