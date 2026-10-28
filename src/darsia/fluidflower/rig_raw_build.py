"""Fresh RAW rig build evidence from actual saved renderer pixels.

This producer does not admit historical archives or update map bindings. Its
certificate joins one new build's RAW/configuration bytes, saved shape chain,
actual corrected-baseline archive and measured renderer invocation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import uuid

import numpy as np

from .rig_raw_input import camera_binding, profile_parameters
from .rig_raw_write import OwnedRawOutput, _canonical


FORMAT = "darsia-raw-rig-baseline-build-v1"
CERTIFICATE_NAME = "baseline_native_renderer.json"
REQUEST_FORMAT = "darsia-raw-rig-build-request-v1"
BASIS_REQUEST_FORMAT = "darsia-raw-shape-basis-request-v1"


def _descriptor(path):
    """Hash one existing physical file and refuse changes observed during that read."""
    from .preset_wasserstein_inputs import current_scope
    scope = current_scope()
    if scope is not None:
        return scope.descriptor(path)
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if any(getattr(before, key) != getattr(after, key) for key in
           ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("RAW rig build file changed while it was hashed: " + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def _verify(rows):
    """Rehash every exact recorded input or output descriptor."""
    for row in rows:
        if _descriptor(row["path"]) != row:
            raise ValueError("RAW rig build input or output changed: " + row["path"])


def _input_descriptor(path):
    """Read a declared source through aliases; keep output identity checks physical."""
    from .preset_wasserstein_inputs import current_scope
    if current_scope() is not None:
        return _descriptor(path)
    from .pipeline.run_io import current_binding
    binding = current_binding()
    reader = getattr(binding, "verified_source_descriptor", None)
    if reader is not None:
        verified = reader(path)
        if verified is not None:
            # The claimed run already hashed this exact physical input. Its
            # reader rechecks the file version; outputs never use this route.
            return verified
    from .storage_paths import resolve_storage_path
    from .setup_source_reads import source_descriptor
    return source_descriptor(resolve_storage_path(path), reader=_descriptor)


def _input_bytes(path):
    """Read source metadata at the same declared input boundary as its pin."""
    from .preset_wasserstein_inputs import current_scope
    scope = current_scope()
    if scope is not None:
        return scope.read(path, kind="raw-build-metadata")
    from .storage_paths import resolve_storage_path
    return resolve_storage_path(path).read_bytes()


def _verify_inputs(rows):
    """Retain exact source descriptors, permitting only configured locator aliases."""
    from .storage_paths import same_storage_descriptor
    for row in rows:
        if not same_storage_descriptor(_input_descriptor(row["path"]), row):
            raise ValueError("RAW rig build input or output changed: " + row["path"])


def _same_input_files(first, second):
    from .storage_paths import same_storage_descriptor
    return len(first) == len(second) and all(same_storage_descriptor(a, b) for a, b in zip(first, second))


def _native_request_dependencies(request, reader=None):
    """Pin optional declared native sources without accepting their scientific use."""
    contract = request.get("native_product", request.get("analysis_exclusions"))
    if contract is None:
        return []
    from .painted_analysis_exclusions import native_analysis_exclusion_dependencies
    return native_analysis_exclusion_dependencies(contract, reader=reader)


def read_build_request_metadata(binding, *, artifact_kind):
    """Verify request bytes and structure without opening its source payloads.

    This supplies output-cap planning before claim. It is not scientific input
    admission; execution still uses read_build_request for all source hashes.
    """
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size_bytes"}:
        raise ValueError("An exact RAW operation request descriptor is required")
    _verify_inputs([binding])
    path = Path(binding["path"])
    if not path.is_absolute() or binding["size_bytes"] > 4 * 1024**2:
        raise ValueError("RAW operation request must be explicit and bounded")
    from .pipeline.job_inputs import _json_bytes
    raw = _input_bytes(path)
    if len(raw) != binding["size_bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("RAW operation request changed during parsing")
    request = _json_bytes(raw)
    fields = {"format", "run_id", "configuration_files", "series_context", "baseline",
              "profile", "operation_overrides", "protocol_inputs", "output",
              "shape_basis_certificate"}
    expected = BASIS_REQUEST_FORMAT if artifact_kind == "raw_shape_basis" else REQUEST_FORMAT
    optional = set() if artifact_kind == "raw_shape_basis" else {"analysis_exclusions", "native_product"}
    if not fields <= set(request) <= fields | optional or request["format"] != expected:
        raise ValueError("RAW operation request fields or artifact role differ")
    for name in optional:
        if name in request and not isinstance(request[name], dict):
            raise ValueError("RAW " + name + " requires its exact original source record")
    if ("analysis_exclusions" in request and "native_product" in request
            and json.dumps(request["analysis_exclusions"], sort_keys=True, allow_nan=False)
                != json.dumps(request["native_product"].get("analysis_exclusions"), sort_keys=True, allow_nan=False)):
        raise ValueError("The RAW native product and analysis exclusions differ")
    if not isinstance(request["configuration_files"], list) or not request["configuration_files"]:
        raise ValueError("Ordered original configuration descriptors are required")
    context = request["series_context"]
    if (not isinstance(context, dict) or set(context) != {"root", "manifest"}
            or Path(context["manifest"]["path"]) != Path(context["root"]) / "series.toml"):
        raise ValueError("RAW operation needs its exact original series context")
    pins = [binding, request["baseline"], request["profile"], context["manifest"],
            *request["configuration_files"]]
    protocols = request["protocol_inputs"]
    if not isinstance(protocols, list):
        raise ValueError("RAW operation protocol input list must be explicit")
    from .protocol_content import require_content
    for row in protocols:
        if (not isinstance(row, dict) or require_content(row) != {"role", "folder", "sheet", "source"}
                or row["role"] not in {"imaging", "injection", "pressure_temperature", "blacklist"}
                or row["folder"] is not None and not isinstance(row["folder"], str)
                or row["sheet"] is not None and not isinstance(row["sheet"], str)):
            raise ValueError("RAW operation has an unsupported protocol binding")
        pins.append(row["source"])
    if artifact_kind == "raw_shape_basis":
        if (set(request["operation_overrides"]) != {"data.raw_camera"}
                or request["shape_basis_certificate"] is not None):
            raise ValueError("A shape basis requires its explicit camera override and no prior certificate")
    else:
        if request["operation_overrides"] != {} or request["shape_basis_certificate"] is None:
            raise ValueError("A full RAW build must bind its basis certificate and use the TOML camera fields")
        pins.append(request["shape_basis_certificate"])
    for pin in pins:
        if (not isinstance(pin, dict) or set(pin) != {"path", "sha256", "size_bytes"}
                or not isinstance(pin["path"], str) or not Path(pin["path"]).is_absolute()
                or not isinstance(pin["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is None
                or type(pin["size_bytes"]) is not int or pin["size_bytes"] < 0):
            raise ValueError("RAW request contains an invalid source descriptor")
    return request, tuple(deepcopy(pins))


def read_build_request(binding, *, artifact_kind):
    """Verify every exact source before configuration or scientific execution.

    An injection protocol row with a schedule digest is verified as its current
    file when only columns Setup does not use changed (protocol_content); the
    returned pins then hold that file's current bytes. Unchanged files keep
    their recorded pins.
    """
    request, pins = read_build_request_metadata(binding, artifact_kind=artifact_kind)
    from .protocol_content import current_source
    current = [(row["source"], current_source(row, check=lambda pin: _verify_inputs([pin])))
               for row in request["protocol_inputs"] if row.get("content") is not None]
    pins = tuple(next((new for old, new in current if old == pin), pin) for pin in pins)
    if artifact_kind != "raw_shape_basis":
        pins = (*pins, *_native_request_dependencies(request))
    _verify_inputs(pins)
    return request, tuple(deepcopy(pins))


def _array_identity(value):
    """Bind finite-size non-object array bytes with their dtype, shape and C order."""
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("RAW rig evidence cannot bind object-array pixels")
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                order="C", sha256=digest, size_bytes=array.nbytes)


def _json_value(value):
    """Project supported evidence scalars, arrays and dates to explicit JSON values."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, type):
        return value.__module__ + "." + value.__qualname__
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("Unsupported RAW rig evidence metadata type")


def _grid(image):
    """Describe the finite ij RGB baseline grid and its pixel-centre conventions."""
    values = image.img
    if (image.space_dim != 2 or image.indexing != "ij" or image.series
            or image.scalar or values.ndim != 3 or values.shape[-1] != 3
            or not np.isfinite(values).all()):
        raise ValueError("RAW rig corrected baseline needs one finite ij RGB image")
    for name in ("dimensions", "origin", "voxel_size"):
        field = np.asarray(getattr(image, name))
        if field.shape != (2,) or not np.isfinite(field).all():
            raise ValueError("RAW rig image has invalid grid metadata: " + name)
    if np.any(np.asarray(image.dimensions) <= 0) or np.any(np.asarray(image.voxel_size) <= 0):
        raise ValueError("RAW rig dimensions and voxel sizes must be positive")
    return dict(metadata=_json_value(image.metadata()),
                voxel_size=_json_value(image.voxel_size),
                num_voxels=_json_value(image.num_voxels),
                array_axes=["row", "column", "camera-native RGB"],
                homography_coordinates=["column", "row", "one"],
                pixel_centres="integer indices", indexing="ij",
                array_row_direction="down", array_column_direction="right")


def _source_snapshot():
    """Bind the full package source supporting the actual renderer invocation."""
    package = Path(__file__).resolve().parents[1]
    return [_descriptor(path) for path in sorted(package.rglob("*.py"))]


def capture_configuration_sources(configuration_paths):
    """Pin ordered operation inputs before the configuration loader reads them."""
    paths = configuration_paths if isinstance(configuration_paths, list) else [configuration_paths]
    if not paths:
        raise ValueError("RAW rig build requires its ordered configuration files")
    return tuple(_input_descriptor(path) for path in paths)


def _require_quality(quality, policies):
    """Check the configured fit, orientation and held-out predicates without inventing residual ceilings."""
    required = {"fit_n", "fit_inlier_fraction", "sampled_jacobian_min", "heldout"}
    if not isinstance(quality, dict) or set(quality) != required:
        raise ValueError("RAW rig renderer lacks its actual quality predicates")
    if (type(quality["fit_n"]) is not int or quality["fit_n"] < policies["minimum_matches"]
            or not np.isfinite(quality["fit_inlier_fraction"])
            or quality["fit_inlier_fraction"] < policies["minimum_inlier_fraction"]
            or not np.isfinite(quality["sampled_jacobian_min"])
            or quality["sampled_jacobian_min"] <= 0):
        raise ValueError("RAW rig renderer did not pass its configured fit/orientation gates")
    expected = [row[0] for row in policies["heldout_zones"]]
    if (not expected or len(expected) != len(set(expected))
            or not isinstance(quality["heldout"], dict)
            or set(quality["heldout"]) != set(expected)):
        raise ValueError("RAW rig renderer lacks its configured held-out zones")
    metrics = {"n", "p50_px", "p95_px", "max_px", "median_match_correlation"}
    for row in quality["heldout"].values():
        if (not isinstance(row, dict) or set(row) != metrics
                or type(row["n"]) is not int or row["n"] < policies["heldout_minimum_matches"]
                or any(not np.isfinite(row[key]) or row[key] < 0
                       for key in ("p50_px", "p95_px", "max_px"))
                or not np.isfinite(row["median_match_correlation"])
                or row["median_match_correlation"] < policies["heldout_min_correlation"]):
            raise ValueError("RAW rig renderer did not pass its configured held-out gates")


def _runtime_identity():
    """Record loaded module wrappers and executable identities without native-binary equivalence."""
    result = dict(python=sys.version, executable=_descriptor(sys.executable),
                  identity_scope="Python module wrappers and executable file identities",
                  native_binary_equivalence_claimed=False,
                  production_qualification_claimed=False)
    libraries = {}
    for name in ("cv2", "rawpy", "numpy"):
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise ValueError("RAW rig renderer runtime was not loaded: " + name)
        libraries[name] = dict(version=str(getattr(module, "__version__", "")),
                               module=_descriptor(module.__file__))
    result["libraries"] = libraries
    return result


def _output_files(folder):
    """List and hash owned output files, excluding the not-yet-published certificate."""
    result = []
    for path in sorted(folder.rglob("*")):
        from .rig_raw_write import _no_links
        _no_links(path)
        if path.is_file() and path.name != CERTIFICATE_NAME:
            result.append(_descriptor(path))
    return result


@dataclass(frozen=True)
class RawRigBuild:
    operation_id: str
    started_utc: str
    destination: Path
    camera: dict
    raw_source: dict
    profile: dict
    configurations: tuple[dict, ...]
    implementation: tuple[dict, ...]
    registration_policies: dict
    artifact_kind: str
    request_binding: dict
    request: dict
    request_inputs: tuple[dict, ...]
    output: OwnedRawOutput

    def bind_rig(self, rig):
        """Bind the original rig to this operation and its input identities."""
        self.output.bind(rig)
        rig._raw_build = self
        rig.raw_camera = deepcopy(self.camera)
        rig.raw_baseline_sha256 = self.raw_source["sha256"]
        rig._raw_build_id = self.operation_id

    def begin_setup(self, rig, log):
        """Start the original setup within the owned output authority."""
        self.bind_rig(rig)
        self.output.begin_setup(rig, log)

    def require_setup(self, rig, log):
        """Require the rig's original, exclusively owned setup context."""
        self.output.require_setup(rig, log)

    def _require_basis_consistency(self, document):
        """Stop a full rebuild when its saved shape basis has changed."""
        pin = self.request["shape_basis_certificate"]
        if pin is None:
            return
        _verify_inputs([pin])
        from .pipeline.job_inputs import _json_bytes
        basis = _json_bytes(_input_bytes(pin["path"]))
        if (basis.get("format") != FORMAT or basis.get("artifact_kind") != "raw_shape_basis"
                or basis.get("complete_rig_model") is not False
                or basis.get("run_id") != self.camera["run_id"]):
            raise ValueError("The full RAW build lacks its exact shape-basis role and run")
        same_bytes = lambda row: (row["sha256"], row["size_bytes"])
        old_chain = basis["renderer"]["shape_corrections"]
        new_chain = document["renderer"]["shape_corrections"]
        chain = lambda rows: [(Path(row["path"]).name, *same_bytes(row)) for row in rows]
        if (basis["corrected_baseline"]["pixels"] != document["corrected_baseline"]["pixels"]
                or basis["corrected_baseline"]["grid"] != document["corrected_baseline"]["grid"]
                or basis["homography_convention"] != document["homography_convention"]
                or chain(old_chain) != chain(new_chain)
                or basis["H_rig_corrected_native_to_baseline_camera_u"]
                    != document["H_rig_corrected_native_to_baseline_camera_u"]
                or same_bytes(basis["inputs"]["raw"]) != same_bytes(self.raw_source)
                or same_bytes(basis["inputs"]["profile"]) != same_bytes(self.profile)
                or basis["inputs"]["camera_binding"]["profile_id"] != self.camera["profile_id"]):
            raise ValueError("RAW full-build/basis consistency STOP: pixels, grid, orientation, shape chain or H differs; geometry must not be replaced")

    def publish(self, rig):
        """Publish only a new build's fully joined saved-pixel certificate."""
        import darsia
        from darsia._opencv_runtime import scientific_thread, completed_frame
        from .native import raw_positioning, registration
        from .storage_paths import same_storage_path

        ipp_before = scientific_thread("raw-rig-certificate:before")
        self.output.check()
        if self.output.rig is not rig or not self.output.save_finished:
            raise ValueError("RAW certificate requires its original completed owned save")

        if (getattr(rig, "_raw_build_id", None) != self.operation_id
                or getattr(rig, "raw_camera", None) != self.camera
                or getattr(rig, "raw_baseline_sha256", None) != self.raw_source["sha256"]
                or not same_storage_path(rig.baseline_path, self.raw_source["path"])):
            raise ValueError("RAW rig certificate belongs to another build or input")
        if not self.destination.is_dir() or self.destination.is_symlink():
            raise ValueError("RAW rig build destination is absent or symbolic")
        certificate_path = self.destination / CERTIFICATE_NAME
        if certificate_path.exists():
            raise FileExistsError("RAW rig build certificate already exists")
        outputs = _output_files(self.destination)
        with (self.destination / "meta_data.json").open("r", encoding="utf-8") as stream:
            saved_operation = json.load(stream)
        if (saved_operation.get("raw_build_id") != self.operation_id
                or saved_operation.get("raw_camera") != self.camera
                or saved_operation.get("raw_baseline_sha256") != self.raw_source["sha256"]
                or not same_storage_path(saved_operation["baseline_path"], self.raw_source["path"])):
            raise ValueError("Saved RAW rig operation identity differs from this fresh build")
        inputs = [self.raw_source, self.profile, *self.configurations,
                  *self.request_inputs, *self.implementation]
        _verify_inputs(inputs)
        if tuple(_source_snapshot()) != self.implementation:
            raise ValueError("RAW rig renderer source membership changed")
        parameters = profile_parameters(self.camera)
        policies = _json_value({key: parameters["common"][key] for key in ("review", "matching")})
        if policies != self.registration_policies:
            raise ValueError("RAW rig renderer quality policies changed after operation admission")
        sensor = raw_positioning.read_sensor(
            self.raw_source["path"], expected_sha256=self.raw_source["sha256"],
            parameters=parameters, profile_id=self.camera["profile_id"],
        )
        camera_u = raw_positioning.camera_u(sensor, parameters=parameters)
        expected_u = list(reversed(parameters["common"]["sensor"]["camera_wh"]))
        if (camera_u.dtype != np.float32 or list(camera_u.shape) != [3, *expected_u]
                or not np.isfinite(camera_u).all()):
            raise ValueError("RAW rig renderer camera-U output differs from its profile")
        scale = np.float32(sensor.metadata["white_level"] - sensor.metadata["black_level"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("RAW rig renderer has no positive sensor white range")
        baseline_path = self.destination / "shape_corrected_baseline.npz"
        archive = _descriptor(baseline_path)
        saved = darsia.imread(baseline_path)
        grid = _grid(saved)
        pixels = _array_identity(saved.img)
        if (pixels != _array_identity(rig.shape_corrected_baseline.img)
                or grid != _grid(rig.shape_corrected_baseline)):
            raise ValueError("Saved RAW rig archive differs from the actual produced pixels or grid")
        expected_assets = [self.destination / f"shape_correction_{index}_{type(step).__name__.lower()}.npz"
                           for index, step in enumerate(rig.shape_corrections)]
        if (not expected_assets
                or set(self.destination.glob("shape_correction_*.npz")) != set(expected_assets)):
            raise ValueError("RAW rig build needs exactly its freshly saved shape chain")
        assets = [_descriptor(path) for path in expected_assets]
        chain = raw_positioning.load_shape_chain(assets)
        initial_metadata = deepcopy(getattr(rig, "_raw_setup_input_metadata", None))
        if not isinstance(initial_metadata, dict):
            raise ValueError("RAW rig build lacks its actual uncorrected input metadata")
        initial_metadata.pop("type", None)
        replay = darsia.OpticalImage(np.moveaxis(camera_u, 0, -1) / scale,
                                     **initial_metadata).img_as(float)
        input_grid = _grid(replay)
        input_pixels = _array_identity(replay.img)
        for step in chain:
            step(replay, overwrite=True)
        # read_raw_image applies the workflow's requested float conversion
        # after OpticalImage has applied its transformations.
        replay = replay.img_as(float)
        if pixels != _array_identity(replay.img) or grid != _grid(replay):
            raise ValueError("Saved shape-chain replay differs from the actual RAW rig archive")
        # The actual saved pixels, rather than dimensions or a separate sensor
        # raster, are the source of this registration invocation.
        review_green = np.asarray(saved.img[..., 1], np.float32) * scale
        camera_green = camera_u[1]
        invocation = dict(
            callable="darsia.fluidflower.native.registration.review_to_camera",
            review_green=_array_identity(review_green),
            camera_green=_array_identity(camera_green),
            review_units="black-subtracted camera DN from saved normalized RAW pixels",
            camera_units="black-subtracted camera DN",
            policies=deepcopy(self.registration_policies),
        )
        h, quality = registration.review_to_camera(review_green, camera_green,
                                                  parameters=parameters["common"])
        _require_quality(quality, parameters["common"]["review"])
        h = np.asarray(h, np.float64)
        if (h.shape != (3, 3) or not np.isfinite(h).all()
                or abs(np.linalg.det(h)) < 1e-12
                or not isinstance(quality, dict)):
            raise ValueError("RAW rig renderer returned an invalid transformation or quality record")
        renderer = dict(
            contract=FORMAT, source_kind="raw", raw_acquisition=dict(self.raw_source),
            profile_id=self.camera["profile_id"], camera_u_wh=list(reversed(expected_u)),
            native_shape_hw=list(saved.img.shape[:2]), native_to_camera_u_h=h.tolist(),
            camera_u_to_native_h=np.linalg.inv(h).tolist(),
            positioning_inputs="actual saved corrected-baseline NPZ green and pinned RAW camera U green",
            source_npz=dict(archive), source_pixels=dict(pixels), source_grid=grid,
            shape_corrections=assets, saved_shape_chain_replay_bit_identical=True,
            saved_shape_chain_replay=dict(input_pixels=input_pixels, input_grid=input_grid,
                normalization=dict(black_subtracted_dn_divisor=float(scale),
                    white_balance_applied=False, gamma_applied=False),
                correction_assets=assets, output_pixels=pixels, output_grid=grid),
            registration_invocation=invocation, registration=_json_value(quality),
            sensor_metadata=_json_value(sensor.metadata),
            white_balance_applied=False, gamma_applied=False,
            direct_sensor_output_equality_claimed=False,
        )
        document = dict(
            format=FORMAT, run_id=self.camera["run_id"], run=self.camera["run_id"],
            artifact_kind=self.artifact_kind,
            producer_role=self.artifact_kind,
            certificate_scope="baseline and ordered shape chain to camera-U registration only",
            complete_rig_model=False, production_qualification_claimed=False,
            output_ownership=dict(
                operation_id=self.operation_id, request=self.request_binding,
                authority=deepcopy(self.request["output"]),
                directory_identity=list(self.output.directory_identity),
                intended_root_identity=list(self.output.root_identity),
                owner=_descriptor(self.destination / "raw_build_owner.json"),
                setup_started=_descriptor(self.destination / "raw_setup_started.json"),
                save_started=_descriptor(self.destination / "raw_save_started.json"),
                save_completed=_descriptor(self.destination / "raw_save_completed.json")),
            producer=dict(operation_id=self.operation_id, pid=os.getpid(),
                          entrypoint=("darsia.fluidflower.rig_raw_basis.build_raw_shape_basis"
                              if self.artifact_kind == "raw_shape_basis" else
                              "darsia.presets.workflows.setup.setup_rig.setup_rig"),
                          started_utc=self.started_utc,
                          finished_utc=datetime.now(timezone.utc).isoformat(),
                          runtime=_runtime_identity()),
            inputs=dict(raw=self.raw_source, profile=self.profile,
                        camera_binding=self.camera, configuration_files=list(self.configurations),
                        operation_request=self.request_binding,
                        operation_overrides=deepcopy(self.request["operation_overrides"]),
                        raw_camera_origin=("bound operation request override" if
                            self.artifact_kind == "raw_shape_basis" else "ordered configuration files"),
                        series_context=deepcopy(self.request["series_context"]),
                        protocol_inputs=deepcopy(self.request["protocol_inputs"]),
                        uncorrected_grid=input_grid),
            renderer_implementation=list(self.implementation),
            setup_scientific_opencv=deepcopy(getattr(rig, "_raw_basis_opencv", None)),
            reading_scientific_opencv=deepcopy(getattr(rig, "_raw_reading_opencv", None)),
            outputs=outputs, corrected_baseline=dict(archive=archive, pixels=pixels, grid=grid),
            renderer=renderer, shape_hw=list(saved.img.shape[:2]),
            H_rig_corrected_native_to_baseline_camera_u=h.tolist(),
            homography_convention=dict(
                source="saved corrected-baseline pixel centres (column,row)",
                target="camera-U pixel centres (column,row)",
                camera_u_axes="x right, y down", raw_orientation=sensor.metadata["orientation"]),
        )
        for name in ("analysis_exclusions", "native_product"):
            if name in self.request:
                document["inputs"][name] = deepcopy(self.request[name])
        photometry_policy = getattr(rig, "_build_photometry_policy", None)
        if photometry_policy is not None:
            if (photometry_policy.get("series_manifest") != self.request["series_context"]["manifest"]
                    or rig.color_corrections
                    or _array_identity(rig.baseline.img) != pixels):
                raise ValueError("The phase-map Build differs from its declared photometry stage")
            document["photometry_policy"] = deepcopy(photometry_policy)
        document["scientific_opencv"] = completed_frame(ipp_before, "raw-rig-certificate:after")
        self._require_basis_consistency(document)
        document["shape_basis_consistency"] = dict(
            required=self.artifact_kind != "raw_shape_basis",
            basis_certificate=deepcopy(self.request["shape_basis_certificate"]),
            byte_exact_verified=self.artifact_kind != "raw_shape_basis",
            mismatch_policy="STOP; no automatic placement or adoption changes")
        payload = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        _verify_inputs(inputs)
        _verify([*outputs, archive, *assets])
        if outputs != _output_files(self.destination) or tuple(_source_snapshot()) != self.implementation:
            raise ValueError("RAW rig build output or source membership changed before publication")
        with self.output.open(certificate_path) as stream:
            stream.write(payload)
        return _descriptor(certificate_path)


def begin_raw_rig_build(configuration_paths, baseline_path, binding, destination,
                        *, configuration_sources, request_binding,
                        artifact_kind="raw_rig_baseline", analysis_exclusions=None, native_product=None):
    """Capture one fresh operation before any rig output or correction is built."""
    camera = camera_binding(binding)
    if camera is None:
        raise ValueError("RAW rig build requires an explicit camera binding")
    if artifact_kind not in {"raw_rig_baseline", "raw_shape_basis"}:
        raise ValueError("Unsupported RAW build artifact role")
    request, request_inputs = read_build_request(request_binding, artifact_kind=artifact_kind)
    for name, value in (("analysis_exclusions", analysis_exclusions), ("native_product", native_product)):
        if json.dumps(request.get(name), sort_keys=True, allow_nan=False) != json.dumps(
                value, sort_keys=True, allow_nan=False):
            raise ValueError("RAW " + name + " differs from the ordered configuration files")
    parameters = profile_parameters(camera)
    configurations = tuple(configuration_sources)
    if (not configurations
            or not _same_input_files(configurations, capture_configuration_sources(configuration_paths))):
        raise ValueError("RAW rig configuration bytes changed while the operation was parsed")
    destination = Path(destination)
    from .image_formats import image_family
    if image_family(baseline_path) != "raw":
        raise ValueError("RAW rig build requires an actual RAW baseline")
    profile = _input_descriptor(camera["configuration"])
    if profile["sha256"] != camera["sha256"]:
        raise ValueError("RAW rig profile changed at operation admission")
    raw = _input_descriptor(baseline_path)
    if (request["run_id"] != camera["run_id"]
            or not _same_input_files(request["configuration_files"], configurations)
            or not _same_input_files([request["baseline"]], [raw])
            or not _same_input_files([request["profile"]], [profile])):
        raise ValueError("RAW operation request differs from its actual ordered input bytes")
    if artifact_kind == "raw_shape_basis" and camera_binding(
            request["operation_overrides"]["data.raw_camera"]) != camera:
        raise ValueError("Shape-basis camera differs from its bound request override")
    from darsia.presets.workflows.config.utils import _series_root
    from .storage_paths import resolve_storage_path, same_storage_path
    series_root = _series_root([Path(row["path"]) for row in configurations])
    if series_root is None or not same_storage_path(series_root, request["series_context"]["root"]):
        raise ValueError("RAW configuration relative-path context differs from its pinned series root")
    protected = request["output"]["forbidden_raw_roots"]
    if not any(_canonical(raw["path"]).is_relative_to(_canonical(resolve_storage_path(root))) for root in protected):
        raise ValueError("The actual RAW baseline is outside all declared protected acquisition roots")
    operation_id = uuid.uuid4().hex
    output = OwnedRawOutput(destination, request["output"], operation_id=operation_id,
                            request=request_binding)
    return RawRigBuild(
        operation_id=operation_id, started_utc=datetime.now(timezone.utc).isoformat(),
        destination=destination, camera=camera, raw_source=raw,
        profile=profile, configurations=configurations,
        implementation=tuple(_source_snapshot()),
        registration_policies=_json_value({key: parameters["common"][key]
                                          for key in ("review", "matching")}),
        artifact_kind=artifact_kind, request_binding=deepcopy(request_binding),
        request=request, request_inputs=request_inputs, output=output,
    )
