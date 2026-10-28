"""Export the real N1 saved-reader route over the existing tiny synthetic inputs.

These unqualified fixtures exercise the real writer and reader, not campaign
data or a substitute successful result. Their physical admission stays false.
"""
import io
import json
from pathlib import Path

import numpy as np

from darsia.fluidflower import storage_coarsening, storage_extensive_io
from darsia.fluidflower.published_dataset.export_mass_maps import mass_map_members
from tests.unit.test_storage_n1_saved_products import (
    tiny_config, phase1_data, _inputs, _run_with_extensive, _index,
)
from .test_export_options import _job_from_index


def test_n1_maps_keep_exact_saved_order_arrays_and_unqualified_state(
        tiny_config, phase1_data, tmp_path, monkeypatch):
    config = _inputs(tmp_path, tiny_config, phase1_data)
    config["storage"]["resolution_mm"] = 1
    completed = _run_with_extensive(tmp_path, config, monkeypatch)
    job = _job_from_index(_index(tmp_path, completed))
    members, section = mass_map_members(job, 2, "products")
    run = job.runs[0]
    assert list(section["runs"][run]) == completed["ordered_case_ids"]
    manifest = json.loads(Path(completed["manifest"]["path"]).read_bytes())
    for number, case in enumerate(completed["ordered_case_ids"]):
        original = storage_extensive_io.read_committed_payload(completed["records"][number],
            producer_kind="n1-mass-only", resolve_asset=lambda bound: Path(bound["path"]),
            manifest=manifest, manifest_asset=completed["manifest"])
        expected = storage_coarsening.coarsen_verified_extensive(original, target_resolution_mm=2)
        archive, metadata = members[run][2 * number:2 * number + 2]
        with np.load(io.BytesIO(b"".join(archive.chunks())), allow_pickle=False) as arrays:
            assert set(arrays.files) == set(expected["arrays"])
            for name, values in expected["arrays"].items():
                np.testing.assert_array_equal(arrays[name], values)
        receipt = json.loads(metadata.data)
        assert receipt["conservation"] == expected["metadata"]["conservation"]
        assert receipt["physical_binding_admitted"] is False
        assert receipt["production_permission_granted"] is False
        assert section["runs"][run][case]["source_state"] == "synthetic_io_only"
