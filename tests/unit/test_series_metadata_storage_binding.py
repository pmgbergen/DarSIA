"""Explicit output-DATA sealing, not a scientific input identity gate."""
from copy import deepcopy

import pytest

from darsia.fluidflower.series_metadata_assembler import assemble_run_metadata
from darsia.fluidflower.storage_outputs import REQUEST_FORMAT, LA_ENCODING
from .test_series_metadata_assembler import _data


def _storage():
    return dict(format=REQUEST_FORMAT, resolution_mm=1.0,
                gas_score_encoding=LA_ENCODING, extent_m=[-.085, .979, -.012, .600])


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_explicit_storage_is_bound_to_new_data_without_changing_inputs(precision):
    previous, selected, records = _data()
    storage = _storage()
    snapshot = deepcopy((previous, selected, records, storage))
    plain, receipt_plain = assemble_run_metadata(previous, selected, records)
    bound, receipt_bound = assemble_run_metadata(previous, selected, records,
                                                storage_data=storage, precision=precision)
    assert bound.pop("storage") == storage
    assert bound == plain and receipt_bound == receipt_plain
    assert (previous, selected, records, storage) == snapshot
    assert "storage" not in previous


def test_returned_storage_does_not_alias_supplied_series_data():
    previous, selected, records = _data()
    storage = _storage()
    result, _ = assemble_run_metadata(previous, selected, records,
                                      storage_data=storage, precision="float32")
    result["storage"]["extent_m"][0] = -999
    assert storage["extent_m"][0] == -.085


def test_absent_storage_retains_existing_whole_data_exactly():
    previous, selected, records = _data()
    previous["storage"] = _storage()
    result, _ = assemble_run_metadata(previous, selected, records)
    assert result["storage"] == previous["storage"]
    result["storage"]["extent_m"][0] = -999
    assert previous["storage"]["extent_m"][0] == -.085


@pytest.mark.parametrize("precision", [None, "float16", "FLOAT32"])
def test_storage_requires_explicit_supported_future_job_precision(precision):
    with pytest.raises(ValueError, match="Explicit bound storage request"):
        assemble_run_metadata(*_data(), storage_data=_storage(), precision=precision)


@pytest.mark.parametrize("mutation", ["off_lattice", "missing_extent", "extra", "bad_encoding", "unsupported_resolution"])
def test_storage_uses_the_adopted_request_admission(mutation):
    storage = _storage()
    if mutation == "off_lattice":
        storage["extent_m"][0] = -.0852
    elif mutation == "missing_extent":
        del storage["extent_m"]
    elif mutation == "extra":
        storage["inherited_at_worker_start"] = True
    elif mutation == "bad_encoding":
        storage["gas_score_encoding"] = "rescale"
    else:
        storage["resolution_mm"] = 3.0
    with pytest.raises(ValueError):
        assemble_run_metadata(*_data(), storage_data=storage, precision="float32")


@pytest.mark.parametrize("storage", [[], "native", 1])
def test_storage_must_be_explicit_mapping(storage):
    with pytest.raises(ValueError, match="storage DATA must be a mapping"):
        assemble_run_metadata(*_data(), storage_data=storage, precision="float32")
