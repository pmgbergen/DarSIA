"""Freeze current run observations through the ordinary DarSIA protocol readers."""
from copy import deepcopy
import math
from pathlib import Path

import pandas as pd

from ..mass.n1_pipeline import asset


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Current protocol has nonfinite " + label)
    return value


def _clock(run):
    frames = [run["baseline"], *run["selected_frames"]]
    if not frames:
        raise ValueError("Current protocols require saved image timestamps")
    origins = []
    for frame in frames:
        stamp = pd.Timestamp(frame.get("camera_datetime"))
        if pd.isna(stamp):
            raise ValueError("Current protocols require each saved image's camera timestamp")
        origins.append(stamp - pd.Timedelta(hours=_finite(frame["elapsed_h"], "image time")))
    origin = origins[0]
    if any(abs((other - origin).total_seconds()) > .001 for other in origins):
        raise ValueError("Saved image timestamps and elapsed times disagree on the run clock")
    return origin, frames


def freeze_run_protocols(series_root, manifest, run, *, reader, source_cache=None,
                         config_paths=None):
    """Save current measured values and their exact config/table sources.

    No image is opened. Protocol dates stay in the saved camera clock: the
    existing parsers, not a hardcoded timezone conversion, interpret them.
    """
    from darsia import InjectionProtocol, PressureTemperatureProtocol
    from ..series_paths import series_folder_path
    from darsia.presets.workflows.config.protocols import ProtocolsConfig
    root = Path(series_root)
    origin, frames = _clock(run)
    cache = source_cache if source_cache is not None else {}
    sources = []

    def bind(path):
        physical = reader.resolve_path(path)
        key = str(physical)
        if key not in cache:
            cache[key] = asset(physical)
        bound = deepcopy(cache[key])
        reader.path(bound)
        if bound not in sources:
            sources.append(bound)
        return physical

    if config_paths is None:
        configured = Path(manifest["runs"]["config_files"][run["run"]])
        if not configured.is_absolute():
            configured = root / configured
        config_paths = [series_folder_path(root, "common_config"), configured]
        state = reader.resolve_path(configured).parent / ".color_state" / (run["run"] + ".txt")
        if state.is_file():
            bind(state)
            if state.read_text(encoding="utf-8-sig").strip().casefold() == "on":
                config_paths.append(series_folder_path(root, "config") / "coloron.toml")
    paths = [bind(path) for path in config_paths]
    config = ProtocolsConfig().load(paths)

    def table(value, label):
        if value is None:
            raise ValueError("Complete the current " + label + " protocol in Setup first")
        if isinstance(value, (tuple, list)):
            return bind(value[0]), value[1]
        return bind(value)

    minimum = min(float(frame["elapsed_h"]) for frame in frames)
    maximum = max(float(frame["elapsed_h"]) for frame in frames)
    if config.pressure_temperature is not None:
        pressure = PressureTemperatureProtocol(table(config.pressure_temperature, "pressure/temperature"))
        dates = sorted(set(pressure.df["datetime"]) | {
            origin + pd.Timedelta(hours=minimum), origin + pd.Timedelta(hours=maximum)})
        rows = []
        for stamp in dates:
            state, gradient = pressure.get_state(stamp), pressure.get_gradient(stamp)
            rows.append(dict(time_hours=(stamp - origin).total_seconds() / 3600,
                pressure_bar=_finite(state.pressure, "pressure"),
                temperature_c=_finite(state.temperature, "temperature"),
                pressure_gradient_bar_per_m=_finite(gradient.pressure, "pressure gradient"),
                temperature_gradient_c_per_m=_finite(gradient.temperature, "temperature gradient")))
    elif config.pressure_temperature_mode == "constant":
        rows = [dict(time_hours=value,
            pressure_bar=_finite(config.pressure_bar, "pressure"),
            temperature_c=_finite(config.temperature_celsius, "temperature"),
            pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.)
            for value in sorted({minimum, maximum})]
    else:
        raise ValueError("Complete the current pressure/temperature protocol in Setup first")
    injection = InjectionProtocol(table(config.injection, "injection"))
    stages = []
    for _, item in injection.df.iterrows():
        start = (item["start"] - origin).total_seconds() / 3600
        end = (item["end"] - origin).total_seconds() / 3600
        rate = _finite(item["rate_kg_s"], "injection rate") * 1000
        if end <= start or rate < 0:
            raise ValueError("Current injection intervals need positive duration and nonnegative rate")
        stages.append(dict(start_h=start, end_h=end, mass_rate_g_s=rate,
            location_x=_finite(item["location_x"], "injection x"),
            location_y=_finite(item["location_y"], "injection y"), location_id=str(item["id"])))
    if not stages:
        raise ValueError("Current injection protocol has no intervals")
    for bound in sources:
        reader.path(bound)
    return dict(format="darsia-current-run-protocols-v1", run=run["run"],
        time_origin=origin.isoformat(), sources=sources,
        atmosphere=dict(run=run["run"], rows=rows),
        dose=dict(run=run["run"], dose_kind="nominal_setpoint", injection_stages=stages),
        provenance="Current configured DarSIA protocol tables; source values and camera clock retained")


def read_frozen_protocols(run, *, reader):
    """Consume the prepared observations only while their source bytes match."""
    value = run.get("current_protocols")
    if not isinstance(value, dict) or value.get("format") != "darsia-current-run-protocols-v1" or value.get("run") != run["run"]:
        raise ValueError("Prepare P1 again to bind the current pressure, temperature and injection protocols")
    for source in value["sources"]:
        reader.path(source)
    return deepcopy(value)
