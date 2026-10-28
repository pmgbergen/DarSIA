"""Named metric/chart catalogue leaves; payload bytes remain unopened here."""
from copy import deepcopy
import json

import pytest

from .test_current_build_catalog import closure_TESTONLY


def test_local_metric_and_adopted_chart_inputs_are_named_and_replayed_without_payload_reads():
    files, config = closure_TESTONLY(local_metric=True)
    before = deepcopy(config)
    catalogue = files.catalog()
    catalogue.current_build(config)
    uses = catalogue.consumers
    metric = json.loads(files.files[config["current_build_binding"]["metric_model"]["path"]])
    for key, bound in metric.items():
        assert uses["current_build.metric." + key]["sha256"] == bound["sha256"]
    assert uses["current_build.metric_model"]["sha256"] == config["current_build_binding"]["metric_model"]["sha256"]
    assert uses["current_build.facies_properties"]["relative_path"] == "facies.csv"
    assert uses["current_build.chart.preparation"]["relative_path"] == "chart/preparation.json"
    assert uses["current_build.chart.arrays"]["relative_path"] == "chart/ac52/chart_labels.npz"
    assert all(path.endswith(".json") for path in files.opened)
    assert any(path.endswith("chart_labels.npz") for path in files.statted)
    replay = files.catalog()
    replay.current_build(config)
    assert replay.entries == catalogue.entries and config == before


@pytest.mark.parametrize("leaf", ["metric.json", "metric/current_edges.npz", "facies.csv",
                                  "chart/preparation.json", "chart/ac52/chart_labels.npz"])
def test_changed_named_metric_or_chart_input_refuses_catalogue(leaf):
    files, config = closure_TESTONLY(local_metric=True)
    files.files[files.root + "/" + leaf] += b"changed"
    with pytest.raises(ValueError, match="changed"):
        files.catalog().current_build(config)
