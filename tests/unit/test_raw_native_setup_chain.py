"""One tiny original RAW Build from source reading through saved native state.

The sensor decoder and calibrated registration provide deterministic fixture
data; the approved product/Apply receipt are supplied at their external source
boundary. Setup, corrections, properties, saving and certification are real.
"""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest
import toml

import darsia
from darsia.fluidflower import rig_raw_basis, rig_raw_build as raw, storage_paths
from darsia.fluidflower import setup_rig as dispatch, corrected_setup
from darsia.fluidflower.native import raw_positioning, registration
from darsia.presets.workflows.config.protocols import ProtocolsConfig
from darsia.presets.workflows.rig import Rig
from .test_rig_analysis_domain import case, _document  # noqa: F401


@pytest.mark.parametrize("method", [None, "colour_path", "phase_model"])
def test_original_raw_setup_native_admission_save_and_certificate(case, tmp_path, monkeypatch, method):
    from types import SimpleNamespace
    from darsia.fluidflower import photograph_io
    old_folder, current_folder = case.root / "old-raw", case.root / "current-raw"
    old_folder.mkdir()
    source = old_folder / "baseline.ARW"
    source.write_bytes(b"deterministic synthetic sensor fixture")
    source_pin = raw._descriptor(source)
    scale = 32.0
    linear = np.moveaxis(np.arange(18, dtype=np.float32).reshape(2, 3, 3), -1, 0)
    sensor_calls = []
    def sensor(path, *, expected_sha256, **kwargs):
        assert raw._input_descriptor(path)["sha256"] == expected_sha256 == source_pin["sha256"]
        sensor_calls.append(Path(path))
        return SimpleNamespace(metadata=dict(white_level=33, black_level=1, orientation=1))
    monkeypatch.setattr(raw_positioning, "read_sensor", sensor)
    monkeypatch.setattr(raw_positioning, "camera_u", lambda *a, **k: linear.copy())
    instant = datetime(2026, 1, 1, 12)
    monkeypatch.setattr(photograph_io, "raw_capture_time", lambda path: instant)
    quality = dict(fit_n=8, fit_inlier_fraction=1.0, sampled_jacobian_min=1.0,
        heldout=dict(sand=dict(n=3, p50_px=0., p95_px=0., max_px=0., median_match_correlation=1.)))
    monkeypatch.setattr(registration, "review_to_camera", lambda *a, **k: (np.eye(3), deepcopy(quality)))
    monkeypatch.setattr(raw, "_source_snapshot", lambda: [raw._descriptor(Path(raw.__file__))])
    monkeypatch.setattr(raw, "_runtime_identity", lambda: dict(kind="synthetic-runtime-boundary"))
    parameters = dict(common=dict(profiles={"synthetic": {}}, sensor=dict(camera_wh=[3, 2]),
        matching={}, review=dict(minimum_matches=4, minimum_inlier_fraction=.8,
            heldout_zones=[["sand", [0, 0, 2, 1], 1]], heldout_minimum_matches=2, heldout_min_correlation=.7)))
    profile = _document(case.root / "camera-profile.json", dict(registration=parameters,
        source=dict(profiles={"sample": "synthetic"})))
    camera = dict(configuration=profile["path"], sha256=profile["sha256"], profile_id="synthetic", run_id="sample")
    imaging, injection, atmosphere = [case.root / (name + ".csv") for name in ("imaging", "injection", "atmosphere")]
    imaging.write_text("path,datetime\n" + str(source) + ",2026-01-01 12:00:00\n")
    injection.write_text("id,start,end,rate_kg/s,location_x,location_y\n1,2026-01-01 12:00:00,2026-01-01 12:01:00,0.0001,0.5,0.5\n")
    atmosphere.write_text("datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,temperature_gradient_celsius\n2026-01-01 12:00:00,1,23,0,0\n")
    outputs = case.root / "products"
    outputs.mkdir()
    depth = case.root / "depth.npz"
    darsia.ScalarImage(np.full((2, 3), .01), dimensions=[1, 1]).save(depth)
    case.properties.write_text("id,porosity,permeability\n0,1,0\n1,0.4,1e-12\n2,0.3,2e-12\n")
    configuration = case.root / "full-setup.toml"
    document = dict(data=dict(folder=str(old_folder), baseline=str(source), results=str(outputs), use_cache=False),
        rig=dict(width=1., height=1., dim=2, path=str(outputs / "full")),
        protocols=dict(imaging={str(old_folder): str(imaging)}, injection=str(injection), pressure_temperature=str(atmosphere)),
        depth=dict(measurements=str(case.root / "measurements.csv"), depth_map=str(depth)),
        labeling=dict(colored_image=str(case.labels_path), labels=str(case.labels_path), frame="rig-corrected"),
        facies=dict(path=str(case.facies_path), props=str(case.properties)),
        image_porosity=dict(active=True, mode="full"), corrections=dict(type=dict(target_type="float32")))
    phase_method = method == "phase_model"
    if method is not None:
        case.series.write_text(f'[method]\nactive = "{method}"\n')
    if phase_method:
        document["corrections"]["color"] = dict(colorchecker="upper_right")
        document["corrections"]["illumination"] = dict()
        def absent_chart(*args, **kwargs):
            pytest.fail("Phase-map Build must not run legacy photometry on a cropped baseline")
        monkeypatch.setattr(darsia, "find_colorchecker", absent_chart)
    configuration.write_text(toml.dumps(document))
    authority = lambda destination: dict(destination=str(destination), intended_root=str(outputs),
        forbidden_raw_roots=[str(old_folder)], maximum_output_bytes=20_000_000, maximum_file_bytes=8_000_000)
    template = dict(run_id="sample", configuration_files=list(raw.capture_configuration_sources(configuration)),
        series_context=dict(root=str(case.root), manifest=raw._descriptor(case.series)), baseline=source_pin,
        profile=profile, protocol_inputs=rig_raw_basis.protocol_sources(ProtocolsConfig().load(configuration)))
    basis_request = dict(template, format=raw.BASIS_REQUEST_FORMAT,
        operation_overrides={"data.raw_camera": camera}, shape_basis_certificate=None,
        output=authority(outputs / "basis"))
    basis_pin = rig_raw_basis.build_raw_shape_basis(_document(case.root / "basis-request.json", basis_request))
    basis = json.loads(Path(basis_pin["path"]).read_bytes())
    # Preserve the original certificate/request after moving the acquisition.
    # Full Build must consume the same bytes through the declared new location.
    old_folder.rename(current_folder)
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [
        (str(old_folder).replace("\\", "/").casefold(), str(current_folder).replace("\\", "/"))])
    case.contract = None
    case.product["analysis_exclusions"] = None
    case.product["native_grid"] = dict(renderer=basis_pin,
        shape_hw=basis["shape_hw"], legacy_dimensions_m=basis["corrected_baseline"]["grid"]["metadata"]["dimensions"],
        H_rig_corrected_native_to_baseline_camera_u=basis["H_rig_corrected_native_to_baseline_camera_u"],
        H_baseline_camera_u_to_rig_corrected_native=basis["renderer"]["camera_u_to_native_h"], physical_metric_is_separate=True)
    receipt = dict(binding=dict(native_product=case.product), files={
        "native_labels.npz": raw._descriptor(case.labels_path), "native_facies.npz": raw._descriptor(case.facies_path),
        "facies_properties.csv": raw._descriptor(case.properties)})
    case.receipt = receipt
    _document(case.export / "export.json", receipt)
    monkeypatch.setattr(corrected_setup, "require_corrected_export_for_build", lambda paths: deepcopy(receipt))
    document["data"]["raw_camera"] = camera
    document["rig"]["native_product"] = json.dumps(case.product)
    configuration.write_text(toml.dumps(document))
    full_request = dict(template, format=raw.REQUEST_FORMAT, operation_overrides={},
        configuration_files=list(raw.capture_configuration_sources(configuration)),
        shape_basis_certificate=basis_pin, native_product=case.product, output=authority(outputs / "full"))
    full_pin = _document(case.root / "full-request.json", full_request)
    constructed = []
    class CapturedRig(Rig):
        def __init__(self):
            super().__init__()
            constructed.append(self)
    dispatch.setup_rig(CapturedRig, configuration, raw_build_request=full_pin)
    built = constructed[-1]
    assert built.depth.img.shape == (2, 3) and built.facies.img.shape == (2, 3)
    np.testing.assert_array_equal(built.shape_corrected_baseline.img, np.moveaxis(linear, 0, -1) / scale)
    certificate = json.loads((outputs / "full" / raw.CERTIFICATE_NAME).read_bytes())
    if phase_method:
        policy = certificate["photometry_policy"]
        assert policy["method"] == "phase_model"
        assert policy["series_manifest"] == template["series_context"]["manifest"]
        assert policy["build_colour_correction_applied"] is False
        assert policy["build_illumination_correction_applied"] is False
        assert built.color_corrections == []
        assert not list((outputs / "full").glob("color_correction_*.npz"))
        np.testing.assert_array_equal(built.baseline.img, built.shape_corrected_baseline.img)
    else:
        assert "photometry_policy" not in certificate
    assert certificate["shape_basis_consistency"]["byte_exact_verified"] is True
    assert certificate["inputs"]["raw"]["sha256"] == source_pin["sha256"]
    assert Path(basis["inputs"]["raw"]["path"]).parent == old_folder
    assert Path(certificate["inputs"]["raw"]["path"]).parent == current_folder
    assert built._analysis_domain is not None and len(sensor_calls) >= 6
    restored = Rig.load(outputs / "full")
    np.testing.assert_array_equal(restored.labels.img, built.labels.img)
    np.testing.assert_array_equal(restored.baseline.img, built.baseline.img)
    assert restored._analysis_domain is not None
