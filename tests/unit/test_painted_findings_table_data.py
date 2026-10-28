"""Diagnostic table DATA from existing synthetic in-memory table fixtures."""

import csv
import hashlib
import io

import pytest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.painted_adoption_table import table_region_data

from .test_adoption_facies_table import build, parent, session


def test_table_region_data_keeps_csv_identity_sorted_regions_and_json_provenance():
    """Keep exact table metadata while merging sampled and drawn layer IDs."""
    raw = parent()
    table = build(raw=raw)
    before = table.csv_bytes, tuple(dict(row) for row in table.rows)

    spec, regions = table_region_data(table, (9, 2), (7, 2))

    assert regions == [
        dict(label_id=2, facies_id=5, name="F", analysis_role="porous_medium",
             porosity=.41, permeability=8., enabled=True),
        dict(label_id=7, facies_id=3, name="C", analysis_role="porous_medium",
             porosity=.32, permeability=14.5, enabled=True),
        dict(label_id=9, facies_id=5, name="F", analysis_role="porous_medium",
             porosity=.41, permeability=8., enabled=True),
    ]
    assert all(type(row["label_id"]) is int and type(row["facies_id"]) is int
               and type(row["porosity"]) is float and type(row["permeability"]) is float
               and type(row["enabled"]) is bool for row in regions)
    assert set(spec) == {"sha256", "size_bytes", "columns", "layer_to_facies", "name_policy",
                         "source_max_id", "retired_source_ids", "changed_source_ids", "aliases", "provenance"}
    assert spec["sha256"] == hashlib.sha256(table.csv_bytes).hexdigest()
    assert spec["size_bytes"] == len(table.csv_bytes)
    assert spec["columns"] == next(csv.reader(io.StringIO(table.csv_bytes.decode("utf8"))))
    assert spec["layer_to_facies"] == {"2": 5, "7": 3, "9": 5}
    assert spec["name_policy"] == "strip-outer-whitespace; exact-case-and-internal-whitespace"
    assert spec["source_max_id"] == 8
    assert spec["retired_source_ids"] == [4, 7, 8]
    assert spec["changed_source_ids"] == [] and spec["aliases"] == []
    coarse, fine = spec["provenance"]
    assert (coarse["name"], coarse["facies_id"], coarse["layer_ids"], coarse["parent_ids"]) == (
        "C", 3, [7], [3])
    assert (fine["name"], fine["facies_id"], fine["layer_ids"], fine["parent_ids"]) == (
        "F", 5, [2, 9], [5, 7, 8])
    assert all(type(field) is list for row in spec["provenance"] for field in row["fields"])
    assert dict(fine["fields"])["custom"] == "parent:exact-cell"
    assert dict(fine["fields"])["porosity"] == "painting:porosity"
    assert (table.csv_bytes, tuple(dict(row) for row in table.rows)) == before
    assert raw == parent()


def test_table_region_data_preserves_water_alias_and_none_without_a_zero_layer():
    """Facies ID zero remains distinct from the absent outside layer zero."""
    drawing = session()
    drawing.facies = {"Free water": lp.FaciesEntry("free_water"), "C": drawing.facies["C"]}
    drawing.layers = [lp.PaintedLayer(4, "Water", "#aaaaaa", "Free water"),
                      lp.PaintedLayer(7, "Coarse", "#bbbbbb", "C")]
    raw = b"facies_id, Name ,custom\n0,water,old water\n3,C,retained\n9,Retired,unused\n"
    table = build(drawing, raw, used=(7, 4))

    spec, regions = table_region_data(table, {7, 4})

    assert regions == [
        dict(label_id=4, facies_id=0, name="Free water", analysis_role="free_water",
             porosity=1., permeability=None, enabled=True),
        dict(label_id=7, facies_id=3, name="C", analysis_role="porous_medium",
             porosity=.32, permeability=14.5, enabled=True),
    ]
    assert spec["aliases"] == [dict(source_name="water", target_name="Free water",
                                    source_id=0, target_id=0, reason="explicit-free-water-role-alias")]
    assert spec["layer_to_facies"] == {"4": 0, "7": 3}
    assert spec["retired_source_ids"] == [9]
    assert spec["provenance"][0]["parent_ids"] == [0]
    assert spec["provenance"][0]["layer_ids"] == [4]
    assert dict(spec["provenance"][0]["fields"])["facies_id"] == "alias:explicit-free-water-role"
    spec["aliases"][0]["target_name"] = "Changed returned metadata"
    spec["provenance"][0]["parent_ids"].append(99)
    fresh, _ = table_region_data(table, {4, 7})
    assert fresh["aliases"][0]["target_name"] == "Free water"
    assert fresh["provenance"][0]["parent_ids"] == [0]
    assert 0 not in table.layer_to_facies and all(row["label_id"] != 0 for row in regions)


@pytest.mark.parametrize("missing", [0, 255])
def test_table_region_data_does_not_synthesize_missing_or_outside_layers(missing):
    """Keep the original missing-layer failure without creating a frame row."""
    table = build()
    before = table.csv_bytes, dict(table.layer_to_facies)

    with pytest.raises(KeyError) as caught:
        table_region_data(table, {2, missing})

    assert caught.value.args == (missing,)
    assert (table.csv_bytes, dict(table.layer_to_facies)) == before
    assert 0 not in table.layer_to_facies
