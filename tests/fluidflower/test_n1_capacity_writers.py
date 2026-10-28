"""TESTONLY: the n1 pixel-domain tables are the real route's archives.

A tiny real pixel-domain run (the integration test's frozen saved optics)
writes its products and geometry archives. Every member must be in the n1
capacity tables, at no more than the table's bytes per pixel, and every table
entry must be written.
"""
import importlib.util
from pathlib import Path

import numpy as np

from darsia.fluidflower.pipeline import n1_capacity as n1

_source = Path(__file__).resolve().parent / "test_n1_pixel_domain_integration.py"
_spec = importlib.util.spec_from_file_location("_pixel_domain_integration_n1", _source)
_integration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_integration)
frozen_run = _integration.frozen_run


def test_the_pixel_domain_tables_are_exactly_the_written_members(frozen_run, tmp_path):
    config, _ = frozen_run
    bound = _integration._json(tmp_path / "domain_config.json", config)
    actual = _integration.pipeline.run_n1(dict(run=config["run"], configuration=bound, commit="b" * 40),
                                          tmp_path / "domain", "cpu", "float32")
    assert actual["complete"] and len(actual["records"]) == 2
    for record in actual["records"]:
        root = Path(record["nas_directory"])
        for name, table, scalars in (("products.npz", n1.PIXEL_PRODUCTS, n1.PRODUCT_SCALARS),
                                     ("native_geometry.npz", n1.PIXEL_GEOMETRY, ())):
            with np.load(root / name) as archive:
                members = {member: archive[member] for member in archive.files}
            assert {key for key, value in members.items() if value.ndim == 0} == set(scalars), name
            grid = {key: value for key, value in members.items() if value.ndim}
            assert set(grid) == set(table), name
            for key, value in grid.items():
                assert value.shape == (3, 3) and value.dtype.itemsize <= n1._itemsize(table[key], "float32"), key
