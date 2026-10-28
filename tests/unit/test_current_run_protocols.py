from datetime import datetime, timedelta

import pytest
import toml

from darsia import InjectionProtocol, PressureTemperatureProtocol
from darsia.fluidflower.mass.n1_pipeline import _atmosphere, _dose
from darsia.fluidflower.pipeline.current_inputs import current_input_reader
from darsia.fluidflower.pipeline.current_run_protocols import freeze_run_protocols, read_frozen_protocols


def protocol_fixture(root, *, constant=False):
    origin = datetime(2024, 2, 7, 12)
    pressure = root / "pressure.csv"
    pressure.write_text("datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,temperature_gradient_celsius\n"
        "2024-02-07 11:00:00,1.01,21,0.01,0.1\n"
        "2024-02-07 12:00:00,0.99,22,0.02,0.2\n"
        "2024-02-07 14:00:00,1.03,24,0.03,0.3\n", encoding="utf-8")
    injection = root / "injection.csv"
    injection.write_text("ID,location_x,location_y,start,end,rate_g/s\n"
        "1,0.2,0.1,2024-02-07 12:00:00,2024-02-07 12:30:00,0.01\n"
        "1,0.2,0.1,2024-02-07 13:00:00,2024-02-07 14:00:00,0.02\n", encoding="utf-8")
    protocols = dict(injection=injection.name, injection_mode="Load from CSV",
                     pressure_temperature_mode="constant" if constant else "Load from CSV",
                     pressure_bar=1.12, temperature_celsius=19.)
    if not constant:
        protocols["pressure_temperature"] = pressure.name
    common = root / "common.toml"
    common.write_text(toml.dumps(dict(protocols=protocols)), encoding="utf-8")
    config = root / "sample.toml"
    config.write_text("", encoding="utf-8")
    manifest = dict(paths=dict(common_config=common.name),
                    runs=dict(ids=["sample"], config_files=dict(sample=config.name)))
    (root / "series.toml").write_text(toml.dumps(manifest), encoding="utf-8")
    def frame(hours):
        return dict(elapsed_h=hours, camera_datetime=(origin+timedelta(hours=hours)).isoformat())
    run = dict(run="sample", baseline=frame(-1), selected_frames=[frame(.25), frame(.75), frame(1.5), frame(2)])
    return manifest, run, pressure, injection


def test_current_parsers_keep_variable_state_disjoint_injections_and_camera_clock(tmp_path):
    manifest, run, pressure_path, injection_path = protocol_fixture(tmp_path)
    value = freeze_run_protocols(tmp_path, manifest, run, reader=current_input_reader())
    state = dict(atmosphere_series=value["atmosphere"], dose=value["dose"])
    pressure, injection = PressureTemperatureProtocol(pressure_path), InjectionProtocol(injection_path)
    for image in [run["baseline"], *run["selected_frames"]]:
        date = datetime.fromisoformat(image["camera_datetime"])
        actual = _atmosphere(state, image)
        expected, gradient = pressure.get_state(date), pressure.get_gradient(date)
        assert actual == pytest.approx(dict(pressure_bar=expected.pressure, temperature_c=expected.temperature,
            pressure_gradient_bar_per_m=gradient.pressure, temperature_gradient_c_per_m=gradient.temperature))
        assert _dose(state, image) == pytest.approx(injection.injected_mass(date=date) * 1000)
    assert len(value["dose"]["injection_stages"]) == 2
    assert value["time_origin"] == "2024-02-07T12:00:00"


def test_explicit_constant_state_remains_available_without_pressure_table(tmp_path):
    manifest, run, _, _ = protocol_fixture(tmp_path, constant=True)
    value = freeze_run_protocols(tmp_path, manifest, run, reader=current_input_reader())
    assert {row["pressure_bar"] for row in value["atmosphere"]["rows"]} == {1.12}
    assert {row["temperature_c"] for row in value["atmosphere"]["rows"]} == {19.}


def test_protocol_edits_require_new_preparation_and_missing_current_pin_is_not_old_fallback(tmp_path):
    manifest, run, pressure, _ = protocol_fixture(tmp_path)
    run["current_protocols"] = freeze_run_protocols(tmp_path, manifest, run, reader=current_input_reader())
    assert read_frozen_protocols(run, reader=current_input_reader())["run"] == "sample"
    pressure.write_text(pressure.read_text().replace("1.01", "1.11"))
    with pytest.raises((ValueError, RuntimeError)):
        read_frozen_protocols(run, reader=current_input_reader())
    with pytest.raises(ValueError, match="Prepare P1 again"):
        read_frozen_protocols(dict(run="sample"), reader=current_input_reader())


def test_inconsistent_image_clock_is_not_reinterpreted(tmp_path):
    manifest, run, _, _ = protocol_fixture(tmp_path)
    run["selected_frames"][0]["elapsed_h"] += 1
    with pytest.raises(ValueError, match="disagree"):
        freeze_run_protocols(tmp_path, manifest, run, reader=current_input_reader())
