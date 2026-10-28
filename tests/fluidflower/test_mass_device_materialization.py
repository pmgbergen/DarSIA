"""A mass run carries its job's device choice; the CPU route is unchanged.

Tiny synthetic mass inputs on a fake OpenCL runtime; no GPU and no user
profile are touched.
"""

import numpy as np
import pytest

from darsia.fluidflower import mass_materialization as materialization
from darsia.fluidflower.mass_artifacts import (
    MASS_INPUT_DYNAMIC_PRODUCTS,
    write_field_frame,
    write_field_index,
)
from darsia.fluidflower.mass_inputs import MASS_INPUT_PRODUCTS

from test_mass_acceleration_device import opencl, two_gpus  # noqa: F401  (fixture)

SHAPE = (6, 8)
TIMES = [0.0, 1.0]
PARAMETERS = {
    "transfer_shape": "linear",
    "acid_low": 0.1,
    "acid_high": 0.9,
    "co2_onset": 0.1,
    "co2_high": 0.9,
    "gas_onset": 0.5,
    "gas_high": 0.95,
    "max_gas_saturation": 0.6,
}
CHOICE = {
    "format": "darsia-gpu-device-choice-v1",
    "basis": "measured",
    "device_id": "1",
    "device": {"name": "GPU B", "driver": "2.0"},
}


def mass_inputs(root):
    """Write one run's six indexed input fields and return its manifest path and row."""
    generator = np.random.default_rng(1)
    co2 = [generator.random(SHAPE) for _ in TIMES]
    values = {
        "optical_acidification": [generator.random(SHAPE) for _ in TIMES],
        "phase_co2_probability": co2,
        "phase_free_gas_probability": [0.5 * field for field in co2],
        "depth": [np.full(SHAPE, 0.01)],
        "porosity": [np.full(SHAPE, 0.4)],
        "permeability": [np.full(SHAPE, 1.0e-10)],
    }
    products = {}
    for product in MASS_INPUT_PRODUCTS:
        folder = root / "inputs" / product
        frames = [
            write_field_frame(
                folder / f"{index:04d}.npz",
                field,
                run_id="run-a",
                product=product,
                frame_id=f"frame-{index:04d}",
                time_hours=TIMES[index] if product in MASS_INPUT_DYNAMIC_PRODUCTS else 0.0,
            )
            for index, field in enumerate(values[product])
        ]
        reference = write_field_index(
            folder / "index.json",
            run_id="run-a",
            product=product,
            quantity=product,
            unit="1",
            frames=frames,
            grid={"shape_rc": list(SHAPE)},
        )
        products[product] = {"path": reference["path"], "sha256": reference["sha256"]}
    row = {
        "run_id": "run-a",
        "phase_candidate_id": "candidate",
        "products": products,
        "geometry": {"width_m": 0.8, "height_m": 0.6},
        "target_times_hours": TIMES,
        "injected_mass_g": [0.0, 1.0],
        "thermodynamic_state": {
            "pressure_bar": [1.0, 1.0],
            "temperature_c": [20.0, 20.0],
            "pressure_gradient_bar_per_m": [0.1, 0.1],
            "temperature_gradient_c_per_m": [0.0, 0.0],
        },
    }
    return root / "inputs" / "manifest.json", row


def run(root, row, manifest, **options):
    return materialization.materialize_mass_run(
        staging=root / options.pop("stage", "stage"),
        manifest_path=manifest,
        row=row,
        model="simple_flash",
        parameters=PARAMETERS,
        products_requested={"mass_total"},
        **options,
    )


def test_analysis_shape_for_run_reads_the_indexes_only(tmp_path):
    manifest, row = mass_inputs(tmp_path)
    assert materialization.analysis_shape_for_run(manifest, row, {}) == SHAPE
    coarse = {"numerical_resolution": {"policy": "maximum_dimension",
                                       "maximum_dimension_px": 4,
                                       "minimum_dimension_px": 16}}
    assert materialization.analysis_shape_for_run(manifest, row, coarse) == (3, 4)


def test_run_records_the_jobs_device_choice_and_refuses_another(tmp_path, opencl):
    cl = opencl(two_gpus())
    manifest, row = mass_inputs(tmp_path)
    result = run(tmp_path, row, manifest, numerical_dtype="float32",
                 backend_name="opencl-float32", device_id="1", device_choice=CHOICE)
    record = result["backend_execution"]
    assert record["device_choice"] == CHOICE and record["device_name"] == "GPU B"
    assert cl.frames == ["GPU B", "GPU B"]
    assert result["cpu_float64_parity"]["passed"] is True
    with pytest.raises(RuntimeError, match="another device"):
        run(tmp_path, row, manifest, stage="again", numerical_dtype="float32",
            backend_name="opencl-float32", device_id="0", device_choice=CHOICE)
    assert cl.frames == ["GPU B", "GPU B"]


def test_cpu_runs_keep_their_backend_record(tmp_path):
    manifest, row = mass_inputs(tmp_path)
    result = run(tmp_path, row, manifest, numerical_dtype="float64",
                 backend_name="cpu-float64", device_choice=CHOICE)
    assert result["backend_execution"] == {
        "backend": "cpu-float64",
        "runtime": "darsia-numpy",
        "numpy": np.__version__,
        "device_id": "cpu",
        "dtype": "float64",
        "fused_phase_mass_kernel": False,
    }
