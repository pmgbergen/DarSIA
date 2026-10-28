"""The lattice capacity ESTIMATE bounds what the synthetic lattice run writes.

The synthetic two-frame run of test_lattice_runner, storing on its computing
lattice or coarser. Its native preparation is a fake, so only the lattice
products per frame and the run's constant lattice inputs are compared.
"""
import pytest

from darsia.fluidflower.pipeline import lattice_capacity as lc
from test_lattice_inputs import CANVAS
from test_lattice_runner import lattice_world, run  # noqa: F401  (fixture)
from test_phase_rules import parameters as rule_parameters  # noqa: F401  (fixture)
from test_phase_domain import parameters as domain_parameters  # noqa: F401  (fixture)

ARRAY_SUFFIXES = {".npy", ".npz", ".png"}
JSON_ALLOWANCE = 5_000_000


def sizes(paths):
    """Bytes of all files and of the array files among them."""
    files = [path for path in paths if path.is_file()]
    return (sum(path.stat().st_size for path in files),
            sum(path.stat().st_size for path in files if path.suffix in ARRAY_SUFFIXES))


@pytest.mark.parametrize("stored", (False, True))
def test_the_estimate_bounds_the_bytes_of_a_synthetic_lattice_run(lattice_world, stored):
    config = lattice_world["config"]
    if stored:
        config["storage"] = dict(config["storage"], resolution_mm=2.0, extent_m=[0.0, 0.146, 0.0, 0.184])
    for spec in config["frames"]:
        spec["mass_spec"]["photo_top"]["shape_hw"] = list(CANVAS)
    task = dict(lattice_world["task"], ordered_case_ids=[spec["case_id"] for spec in config["frames"]])
    value = lc.lattice_estimate(task, config)
    assert (value.stored_shape_hw != value.lattice_shape_hw) == stored
    _, output = run(lattice_world, "bounded")
    for spec in config["frames"]:
        case = spec["case_id"]
        total, arrays = sizes([*(output / "frames" / case).iterdir(), *(output / "preparation" / case).iterdir()])
        assert total <= value.products_per_frame_estimated_bytes
        assert arrays <= value.products_per_frame_estimated_bytes - JSON_ALLOWANCE
    total, arrays = sizes([*(output / "lattice").iterdir(), *(output / "ops").iterdir(),
                           *(output / name for name in ("manifest.json", "resume.json", "mass_table.csv"))])
    assert total <= value.run_constant_estimated_bytes
    assert arrays <= value.run_constant_estimated_bytes - JSON_ALLOWANCE
