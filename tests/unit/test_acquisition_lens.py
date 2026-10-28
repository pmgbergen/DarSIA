"""Per-acquisition lens tables: real TIFF tags, configured reader and bridge use.

The RAW decode is a test double; the tag reading and the configured table
reader are real. The camera-U arithmetic itself is unchanged and not tested here.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from darsia.fluidflower.mass import build_native_bridge as bridge
from darsia.fluidflower.native import acquisition_lens, raw_positioning

PROFILE_D = [716, 699, 630, 509, 339, 129, -112, -366, -613, -842, -1058]
OWN_D = [704, 688, 620, 501, 333, 126, -110, -360, -603, -828, -1034]
RED = [0, 512, 384, 384, 384, 512, 512, 640, 768, 896, 896]
BLUE = [0, 512, 640, 768, 768, 640, 512, 384, 256, 128, 256]
OWN_RED = [0, 384, 384, 384, 384, 512, 512, 640, 640, 768, 896]


def _parameters():
    lens = dict(main_page=0, sub_page=0, ca_tag=28725, distortion_tag=28727,
                ca_header=22, ca_min_length=23, red_slice=[1, 12], blue_slice=[12, 23],
                distortion_header=11, distortion_min_length=12, distortion_slice=[1, 12],
                ca_red=list(RED), ca_blue=list(BLUE), knots=[i / 10 for i in range(11)])
    coefficients = dict(cu=10., cv=8., R=12., amp=1., tx=0., ty=0.)
    return dict(common=dict(lens=lens, profiles=dict(
        late=dict(distortion=list(PROFILE_D), coefficients=coefficients),
        early=dict(distortion=[720] + PROFILE_D[1:], coefficients=deepcopy(coefficients)))),
        edge_aware=dict(demosaic_code="COLOR_BayerBG2RGB_EA"))


def _raw(path, distortion=PROFILE_D, red=RED, blue=BLUE, *, ca_header=22):
    with tifffile.TiffWriter(path) as tif:
        tif.write(np.zeros((4, 4), np.uint16), subifds=1)
        ca = [ca_header, *red, *blue]
        d = [11, *distortion]
        tif.write(np.zeros((2, 2), np.uint16), extratags=[
            (28725, "h", len(ca), ca, True), (28727, "h", len(d), d, True)])
    return path


def test_recorded_tables_are_read_through_the_configured_reader(tmp_path):
    path = _raw(tmp_path / "own.ARW", OWN_D, OWN_RED)
    distortion, red, blue = acquisition_lens.recorded_lens_tables(path, _parameters()["common"]["lens"])
    assert (list(distortion), list(red), list(blue)) == (OWN_D, OWN_RED, BLUE)


def test_profile_tables_keep_the_unchanged_configuration(tmp_path):
    parameters = _parameters()
    assert acquisition_lens.acquisition_lens(_raw(tmp_path / "same.ARW"), parameters,
                                             profile_id="late") is None


def test_other_recorded_tables_change_only_the_private_expected_tables(tmp_path):
    parameters = _parameters()
    original = deepcopy(parameters)
    own = acquisition_lens.acquisition_lens(_raw(tmp_path / "own.ARW", OWN_D, OWN_RED),
                                            parameters, profile_id="late")
    assert parameters == original  # The shared project configuration is never modified.
    expected = deepcopy(original)
    expected["common"]["lens"].update(ca_red=OWN_RED, ca_blue=BLUE)
    expected["common"]["profiles"]["late"]["distortion"] = OWN_D
    assert own.parameters == expected
    assert own.provenance == dict(origin="recorded in this acquisition", profile_id="late",
                                  distortion=OWN_D, ca_red=OWN_RED, ca_blue=BLUE)


def test_channel_table_difference_alone_binds_the_acquisition_tables(tmp_path):
    own = acquisition_lens.acquisition_lens(_raw(tmp_path / "ca.ARW", red=OWN_RED),
                                            _parameters(), profile_id="late")
    assert own.parameters["common"]["profiles"]["late"]["distortion"] == PROFILE_D
    assert own.parameters["common"]["lens"]["ca_red"] == OWN_RED


@pytest.mark.parametrize("change", ["ca_header", "short_distortion"])
def test_unexpected_table_structure_is_refused(tmp_path, change):
    if change == "ca_header":
        path = _raw(tmp_path / "bad.ARW", ca_header=21)
    else:
        path = _raw(tmp_path / "bad.ARW", OWN_D[:5])
    with pytest.raises(ValueError, match="lens-table structure"):
        acquisition_lens.acquisition_lens(path, _parameters(), profile_id="late")


class _Original:
    """Minimal stand-in for the original processor's reader and cache."""

    def __init__(self, parameters, profile_id="late"):
        self.parameters = dict(registration=parameters)
        self.binding = dict(profile_id=profile_id)
        self.original_reads, self.asset_checks = [], []

    def _path(self, value):
        return Path(value)

    def _asset(self, descriptor):
        self.asset_checks.append(descriptor["path"])
        return Path(descriptor["path"])

    def _sensor(self, descriptor):
        self.original_reads.append(descriptor["path"])
        key = (str(self._path(descriptor["path"])), descriptor["sha256"])
        if getattr(self, "_current_sensor_key", None) == key:
            return self._current_sensor
        sensor = _sensor(descriptor, distortion=PROFILE_D, red=RED)
        self._current_sensor_key, self._current_sensor = key, sensor
        return sensor


class _Probe(bridge._AcquisitionLensSensor, _Original):
    pass


def _sensor(descriptor, *, distortion, red, model="ZV-1", parameters=None):
    return raw_positioning.Sensor(np.zeros((2, 2, 3), np.uint16),
        dict(path=str(descriptor["path"]), sha256=descriptor["sha256"]),
        dict(make="Camera maker", model=model, white_level=1000, black_level=10),
        tuple(distortion), tuple(red), tuple(BLUE), "late")


@pytest.fixture
def probe(tmp_path, monkeypatch):
    baseline = dict(path=str(_raw(tmp_path / "baseline.ARW")), sha256="b" * 64)
    wash = dict(path=str(_raw(tmp_path / "wash.ARW", OWN_D, OWN_RED)), sha256="w" * 64)
    reads = []

    def read_sensor(path, *, expected_sha256, parameters, profile_id):
        common = parameters["common"]
        reads.append(dict(path=str(path), sha256=expected_sha256, parameters=deepcopy(parameters),
                          profile_id=profile_id))
        return _sensor(dict(path=path, sha256=expected_sha256),
                       distortion=common["profiles"][profile_id]["distortion"],
                       red=common["lens"]["ca_red"], model=probe.model)

    probe = _Probe(_parameters())
    probe.model = "ZV-1"
    probe.reads = reads
    monkeypatch.setattr(bridge.raw_positioning, "read_sensor", read_sensor)
    probe.baseline_sensor = probe._sensor(baseline)
    probe.baseline, probe.wash = baseline, wash
    return probe


def test_profile_acquisition_uses_only_the_original_reader(probe):
    assert probe.original_reads == [probe.baseline["path"]]
    assert probe.reads == []
    assert "lens_tables" not in probe.baseline_sensor.metadata


def test_acquisition_with_own_tables_is_read_with_them_and_cached(probe):
    shared = deepcopy(probe.parameters)
    sensor = probe._sensor(probe.wash)
    assert probe.parameters == shared
    assert len(probe.reads) == 1 and probe.asset_checks == [probe.wash["path"]]
    used = probe.reads[0]["parameters"]["common"]
    assert used["profiles"]["late"]["distortion"] == OWN_D and used["lens"]["ca_red"] == OWN_RED
    assert used["profiles"]["early"] == shared["registration"]["common"]["profiles"]["early"]
    assert (list(sensor.distortion), list(sensor.ca_red)) == (OWN_D, OWN_RED)
    assert sensor.metadata["lens_tables"]["distortion"] == OWN_D
    # The same acquisition is not decoded again; the original cache serves it.
    assert probe._sensor(probe.wash) is sensor
    assert len(probe.reads) == 1
    # Returning to the baseline keeps the original reader and profile tables.
    assert list(probe._sensor(probe.baseline).distortion) == PROFILE_D


def test_own_tables_from_another_camera_model_are_refused(probe):
    probe.model = "Another camera"
    with pytest.raises(ValueError, match="calibrated camera model"):
        probe._sensor(probe.wash)
    assert getattr(probe, "_current_sensor_key")[0] == probe.baseline["path"]


def test_baseline_with_other_tables_than_its_profile_is_refused(probe):
    fresh = _Probe(_parameters())
    with pytest.raises(ValueError, match="baseline acquisition records other lens tables"):
        fresh._sensor(probe.wash)


def test_unreadable_tables_leave_the_error_to_the_original_reader(probe, tmp_path):
    broken = tmp_path / "broken.ARW"
    broken.write_bytes(b"not a TIFF")
    probe._sensor(dict(path=str(broken), sha256="x" * 64))
    assert probe.original_reads[-1] == str(broken)
    assert probe.reads == []


def test_configuration_without_lens_tables_keeps_the_original_reader(tmp_path):
    parameters = _parameters()
    del parameters["common"]["lens"]
    plain = _Probe(parameters)
    descriptor = dict(path=str(_raw(tmp_path / "frame.ARW", OWN_D)), sha256="f" * 64)
    plain._sensor(descriptor)
    assert plain.original_reads == [descriptor["path"]]


def test_bridge_processor_reads_acquisitions_through_the_lens_adapter(monkeypatch, tmp_path):
    class Original(_Original):
        def __init__(self, binding, *, parameters):
            super().__init__(parameters["registration"])
    monkeypatch.setattr(bridge.raw_rig_source, "RawRigSourceProcessor", Original)
    processor = bridge._bridge_processor("binding.json", parameters=dict(registration=_parameters()))
    names = [cls.__name__ for cls in type(processor).__mro__]
    assert (names.index("_MeasuredClearwaterRenderer") < names.index("_AcquisitionLensSensor")
            < names.index("Original"))
    descriptor = dict(path=str(_raw(tmp_path / "frame.ARW")), sha256="f" * 64)
    processor._sensor(descriptor)
    assert processor.original_reads == [descriptor["path"]]
