"""Optional sand registration on DarSIA's existing shape-corrected grid.

The sidecar selects runs explicitly. Existing serialized rigs and review images
remain immutable. This adapter must be installed before color corrections are
configured, and changes the shape list also used for clear-water references.
"""
from __future__ import annotations
from darsia.fluidflower.series_paths import series_path
from darsia.fluidflower.tools.phase_optical_cache_compatibility import canonical_implementation

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import threading

import numpy as np

POLICY_FORMAT = "phase-rig-registration-policy-v1"
DEFAULT_POLICY = series_path("ffac", "config") / "rig_registration.json"
HOST_IMPLEMENTATION_SHA256 = canonical_implementation(
    __name__, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array_sha256(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"shape": list(array.shape), "dtype": array.dtype.str},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def registration_profile_for_run(run, policy_path=None):
    """Return an explicitly enabled AC profile; a missing sidecar changes nothing."""
    run = str(run).casefold()
    if not re.fullmatch(r"ac\d+", run):
        return None
    path = DEFAULT_POLICY if policy_path is None else Path(policy_path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    if document.get("format") != POLICY_FORMAT or document.get("series") != "ffac":
        raise ValueError("Unsupported FF-AC rig-registration policy")
    runs = document.get("runs")
    if not isinstance(runs, dict) or any(key != key.casefold() for key in runs):
        raise ValueError("Registration policy requires lower-case run identifiers")
    profile = runs.get(run)
    if profile is None:
        return None
    if not isinstance(profile, dict) or type(profile.get("enabled")) is not bool:
        raise ValueError("A registration run profile requires enabled=true or false")
    if not profile["enabled"]:
        return None
    if "valid_bounds_xyxy" not in profile:
        raise ValueError("Enabled production registration requires frozen valid_bounds_xyxy")
    return {**profile, "series": "ffac", "run": run,
            "policy_source": str(path.resolve()),
            "policy_file_sha256": hashlib.sha256(raw).hexdigest()}


def _core():
    try:
        from darsia_extensions import registration
    except ImportError as exc:
        raise ImportError("Enabled sand registration requires the updated "
                          "darsia-extensions registration module") from exc
    return registration


def _static_domain(rig, shape):
    """Use registered facies, never phase predictions or future plume support."""
    facies = getattr(rig, "facies", None)
    if facies is None:
        raise ValueError("Sand registration requires the rig's registered facies map")
    values = np.asarray(getattr(facies, "img", facies))
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Registration facies and shape-corrected baseline grids differ")
    # In FF-AC, facies 0 is the non-sand/free-water region, including the chart.
    domain = values != 0
    if not domain.any():
        raise ValueError("Registration has no static sand domain")
    return domain


def _bounds(value, shape):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(item) is not int for item in value)):
        raise ValueError("Geometry validity bounds must be four integer pixel coordinates")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("Geometry validity bounds must be nonempty and inside the baseline grid")
    return list(value)


def geometry_validity_contract(rig):
    adapter = getattr(rig, "_phase_rig_registration", None)
    if adapter is None:
        return None
    bounds = adapter.provenance["valid_bounds_xyxy"]
    return {"coordinate_system": "unchanged registered baseline grid",
            "shape_hw": adapter.provenance["shape_hw"],
            "valid_bounds_xyxy": bounds,
            "frozen_analysis_domain": bounds is not None,
            "outside_bounds": "unobserved geometry; excluded from phase evidence and phase mass",
            "per_frame_policy": "reject if interpolation validity does not cover frozen bounds",
            "geometry_revision_sha256": adapter.provenance["geometry_revision_sha256"]}


def registration_valid_mask(rig):
    """Fixed validity from actual replay bounds; never alter facies/depth arrays."""
    contract = geometry_validity_contract(rig)
    if contract is None or contract["valid_bounds_xyxy"] is None:
        return None
    mask = np.zeros(contract["shape_hw"], dtype=bool)
    x0, y0, x1, y1 = contract["valid_bounds_xyxy"]
    mask[y0:y1, x0:x1] = True
    return mask


class RigRegistrationCorrection:
    """Copy-safe correction callable with per-thread, per-image receipts."""
    active = True

    def __init__(self, registration, provenance):
        self.registration = registration
        self.provenance = provenance
        self._thread = threading.local()
        self._receipts = []
        self._lock = threading.Lock()

    @property
    def last_receipt(self):
        return getattr(self._thread, "last_receipt", None)

    @property
    def receipts(self):
        with self._lock:
            return list(self._receipts)

    def _apply(self, array, source_name=None):
        # Clear first so an error cannot accidentally expose the previous frame.
        self._thread.last_receipt = None
        corrected, receipt = self.registration.apply(array)
        native_bounds = _bounds(receipt["valid_bounds_xyxy"], array.shape[:2])
        fixed = self.provenance["valid_bounds_xyxy"]
        if fixed is not None and not (native_bounds[0] <= fixed[0]
                                     and native_bounds[1] <= fixed[1]
                                     and native_bounds[2] >= fixed[2]
                                     and native_bounds[3] >= fixed[3]):
            raise _core().RegistrationError("Frame interpolation validity does not cover "
                                            "the frozen registration analysis domain")
        receipt = dict(receipt, host_geometry_revision_sha256=self.provenance["geometry_revision_sha256"])
        receipt["geometry_validity"] = {"frame_valid_bounds_xyxy": native_bounds,
                                        "frozen_analysis_bounds_xyxy": fixed,
                                        "frozen_domain_covered": fixed is not None}
        if source_name is not None:
            receipt["image_name"] = str(source_name)
        self._thread.last_receipt = receipt
        with self._lock:
            self._receipts.append(receipt)
        return corrected

    def correct_array(self, array):
        return self._apply(np.asarray(array))

    def __call__(self, image, overwrite=False):
        if isinstance(image, np.ndarray):
            return self._apply(image)
        if bool(getattr(image, "series", False)):
            raise ValueError("Register individual acquisition frames before stacking a series")
        output = image if overwrite else image.copy()
        output.img = self._apply(np.asarray(image.img), getattr(image, "name", None))
        return output

    def save(self, path):
        raise ValueError("Runtime sand registration uses a sidecar profile; do not "
                         "overwrite the original serialized rig")


def install_registration(rig, profile, *, run, domain_mask=None):
    """Install on one in-memory rig, before configuring its color pipeline.

    For explicit diagnostic use, profile may be a dict with only ``settings``.
    Production selection is through ``registration_profile_for_run``. No files
    are written and repeat installation is rejected rather than applied twice.
    """
    run = str(run).casefold()
    if not re.fullmatch(r"ac\d+", run):
        raise ValueError("This host adapter currently has a registered FF-AC domain contract only")
    if not isinstance(profile, dict) or profile.get("enabled", True) is not True:
        raise ValueError("An enabled registration profile is required")
    if profile.get("series", "ffac") != "ffac" or profile.get("run", run) != run:
        raise ValueError("Registration profile belongs to another series or run")
    if hasattr(rig, "_phase_rig_registration"):
        raise ValueError("Sand registration is already installed on this rig")
    baseline = getattr(rig, "shape_corrected_baseline", None)
    if baseline is None:
        raise ValueError("Install registration after geometric setup and before color setup")
    reference = np.asarray(baseline.img, dtype=np.float32)
    if reference.ndim != 3 or reference.shape[2] != 3 or not np.isfinite(reference).all():
        raise ValueError("Registration requires a finite shape-corrected RGB baseline")
    domain = _static_domain(rig, reference.shape[:2]) if domain_mask is None else np.asarray(domain_mask)
    if domain.shape != reference.shape[:2] or domain.dtype != np.bool_ or not domain.any():
        raise ValueError("Explicit static registration mask must be a nonempty boolean baseline grid")
    reference_hash, domain_hash = array_sha256(reference), array_sha256(domain)
    fixed_bounds = (_bounds(profile["valid_bounds_xyxy"], reference.shape[:2])
                    if "valid_bounds_xyxy" in profile else None)
    for key, actual in (("reference_sha256", reference_hash), ("domain_sha256", domain_hash)):
        if profile.get(key) is not None and profile[key] != actual:
            raise ValueError(f"Registration {key} does not match this rig")
    core = _core()
    settings = core.RegistrationSettings(**profile.get("settings", {}))
    engine = core.RigRegistration(reference.copy(), domain_mask=domain.copy(), settings=settings)
    core_path = Path(core.__file__)
    provenance = {"active": True, "series": "ffac", "run": run,
        "method": "sand-texture-residual-registration", "stage": "after shape; before color",
        "reference_sha256": reference_hash, "domain_sha256": domain_hash,
        "valid_bounds_xyxy": fixed_bounds,
        "shape_hw": list(reference.shape[:2]), "settings": asdict(settings),
        "implementation_sha256": getattr(core, "IMPLEMENTATION_SHA256",
                                          hashlib.sha256(core_path.read_bytes()).hexdigest()),
        "host_implementation_sha256": HOST_IMPLEMENTATION_SHA256,
        "controls": "deterministic baseline-only tiles in registered static sand domain",
        "baseline_identity": True, "serialized_rig_modified": False,
        "profile": json.loads(json.dumps(profile, allow_nan=False))}
    # A copied, identical policy has the same geometric identity. Its audit
    # location and unrelated run entries in the sidecar do not alter coordinates.
    provenance["geometry_revision_sha256"] = _digest(
        {key: value for key, value in provenance.items() if key != "profile"})
    provenance["geometry_revision_id"] = f"sand-registration-v1:{run}:{provenance['geometry_revision_sha256'][:16]}"
    adapter = RigRegistrationCorrection(engine, provenance)
    # Verify the invariant before mutating the rig. Exact baseline bypass must
    # not interpolate the reference or alter the cached baseline/facies grid.
    identity = adapter.correct_array(reference)
    if not np.array_equal(identity, reference):
        raise ValueError("Registration failed exact baseline identity")
    previous = list(getattr(rig, "shape_corrections", []))
    if any(isinstance(item, RigRegistrationCorrection) for item in previous):
        raise ValueError("A registration correction already exists in the shape chain")
    rig.shape_corrections = [*previous, adapter]
    rig._phase_rig_registration = adapter
    rig._phase_rig_registration_provenance = provenance
    return provenance
