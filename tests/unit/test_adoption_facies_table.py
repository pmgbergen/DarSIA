"""Synthetic table derivation only: no Store, files, worker or adoption."""

from copy import deepcopy
import csv
import hashlib
import io

import pytest

from darsia.fluidflower import adoption_facies_table as table
from darsia.fluidflower import layer_painting as lp


def session():
    value = lp.PaintingSession(run_id="synthetic", permeability_unit="mD")
    value.facies = {
        "C": lp.FaciesEntry("porous_medium", .32, 14.5),
        "F": lp.FaciesEntry("porous_medium", .41, 8.),
        "Unused": lp.FaciesEntry("porous_medium", None, None),
    }
    value.layers = [lp.PaintedLayer(9, "Upper fine", "#aaaaaa", "F"),
                    lp.PaintedLayer(2, "Lower fine", "#bbbbbb", "F"),
                    lp.PaintedLayer(7, "Coarse", "#cccccc", "C"),
                    lp.PaintedLayer(8, "Unused", "#dddddd", "Unused")]
    return value


def parent():
    return (b'id,Name,porosity,permeability,swi CO2,krel_gas,1-Sg,krel_water,pc [mbar],custom\r\n'
            b'5,F,0.2,1,0.12,0.8,0.7,0.9,3," keep, exact "\r\n'
            b'7,F,0.3,2,0.12,0.8,0.7,0.9,3," keep, exact "\r\n'
            b'8,F,0.4,3,0.12,0.8,0.7,0.9,3," keep, exact "\r\n'
            b'3,C,0.1,4,0.15,0.6,0.5,0.8,4,coarse\r\n'
            b'4,Old unused,0.1,5,0.16,0.5,0.4,0.7,5,unused\r\n')


def build(value=None, raw=None, used=(9, 7, 2)):
    value = session() if value is None else value
    raw = parent() if raw is None else raw
    return table.build_adoption_facies_table(value, raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(), used_layer_ids=used)


def test_one_row_per_used_painting_facies_with_automatic_shared_layer_ids():
    value, raw = session(), parent()
    before = deepcopy(value), bytes(raw)
    result = build(value, raw)
    assert [dict(row)["Name"] for row in result.rows] == ["C", "F"]
    assert result.layer_to_facies == {2: 5, 7: 3, 9: 5}
    assert result.facies_ids == {"C": 3, "F": 5}
    assert [row["id"] for row in result.rows] == ["3", "5"]
    assert result.source_max_id == 8
    assert result.retired_source_ids == (4, 7, 8)
    assert result.changed_source_ids == ()
    assert result.aliases == ()
    assert result.rows[1]["porosity"] == "0.41"
    assert result.rows[1]["permeability"] == "8.0"
    assert result.rows[1]["permeability_unit"] == "mD"
    assert result.rows[1]["analysis_role"] == "porous_medium"
    assert result.rows[1]["enabled"] == "True"
    assert result.provenance[1].parent_ids == (5, 7, 8)
    assert result.provenance[1].layer_ids == (2, 9)
    assert value == before[0] and raw == before[1]
    assert result.parent_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.sha256 == hashlib.sha256(result.csv_bytes).hexdigest()
    assert build(value, raw, used=(2, 7, 9)).csv_bytes == result.csv_bytes
    with pytest.raises(TypeError):
        result.rows[0]["Name"] = "Changed"
    with pytest.raises(TypeError):
        result.layer_to_facies[2] = 99


def test_full_original_column_order_and_arbitrary_cells_are_carried_exactly():
    raw = parent()
    result = build(raw=raw)
    original = next(csv.reader(io.StringIO(raw.decode("utf8"))))
    assert result.columns[:len(original)] == tuple(original)
    assert result.columns[len(original):] == ("permeability_unit", "analysis_role", "enabled")
    fine = result.rows[1]
    for column, expected in {"swi CO2": "0.12", "krel_gas": "0.8", "1-Sg": "0.7",
                             "krel_water": "0.9", "pc [mbar]": "3", "custom": " keep, exact "}.items():
        assert fine[column] == expected
    assert "swi" not in result.columns
    decoded = list(csv.DictReader(io.StringIO(result.csv_bytes.decode("utf8"))))
    assert decoded == [dict(row) for row in result.rows]
    assert dict(result.provenance[1].fields)["custom"] == "parent:exact-cell"
    assert dict(result.provenance[1].fields)["porosity"] == "painting:porosity"


def test_duplicate_extra_conflict_reports_name_column_and_every_source_value():
    raw = parent().replace(b'7,F,0.3,2,0.12', b'7,F,0.3,2,0.18')
    with pytest.raises(table.FaciesTableError) as caught:
        build(raw=raw)
    error = caught.value
    assert len(error.conflicts) == 1
    conflict = error.conflicts[0]
    assert (conflict.name, conflict.column) == ("F", "swi CO2")
    assert conflict.values == ((5, "0.12"), (7, "0.18"), (8, "0.12"))
    for detail in ("F", "swi CO2", "id 5='0.12'", "id 7='0.18'", "id 8='0.12'"):
        assert detail in str(error)


def test_extra_numeric_text_and_unknown_aliases_are_never_reinterpreted():
    raw = b'id,name,swi CO2,swi\n1,F,0.10,0.2\n2,F,0.1,0.2\n3,C,0.3,0.4\n'
    with pytest.raises(table.FaciesTableError) as caught:
        build(raw=raw)
    assert caught.value.conflicts[0].column == "swi CO2"
    raw = raw.replace(b'2,F,0.1,', b'2,F,0.10,')
    result = build(raw=raw)
    assert result.rows[1]["swi CO2"] == "0.10"
    assert result.rows[1]["swi"] == "0.2"


def test_unused_parent_name_conflicts_do_not_block_or_add_unused_rows():
    raw = parent() + b'6,Old unused,0.1,5,0.99,0.5,0.4,0.7,5,unused\r\n'
    result = build(raw=raw)
    assert [row["Name"] for row in result.rows] == ["C", "F"]
    assert result.retired_source_ids == (4, 6, 7, 8)
    assert all(row.name != "Old unused" for row in result.provenance)


def test_explicit_blank_parent_cells_remain_blank_without_imputation():
    result = build(raw=b'id,name,custom\n1,F,\n2,F,\n3,C,explicit\n')
    assert result.rows[1]["custom"] == ""
    assert result.provenance[1].parent_ids == (1, 2)


def test_unmatched_names_with_extra_columns_are_actionable_without_defaults():
    value = session()
    value.facies["New facies"] = value.facies.pop("F")
    for layer in value.layers:
        if layer.facies == "F":
            layer.facies = "New facies"
    with pytest.raises(table.FaciesTableError) as caught:
        build(value)
    assert caught.value.unmatched_names == ("New facies",)
    assert "swi CO2" in str(caught.value) and "will not be guessed" in str(caught.value)


def test_name_policy_strips_outer_whitespace_but_preserves_case_and_internal_space():
    result = build(raw=parent().replace(b',F,', b', F ,'))
    assert result.rows[1]["Name"] == "F"
    assert result.name_policy == table.NAME_POLICY
    with pytest.raises(table.FaciesTableError, match="NAME match"):
        build(raw=parent().replace(b',F,', b',f,'))
    value = session()
    value.facies["Fine  sand"] = value.facies.pop("F")
    for layer in value.layers:
        if layer.facies == "F":
            layer.facies = "Fine  sand"
    with pytest.raises(table.FaciesTableError, match="NAME match"):
        build(value, parent().replace(b',F,', b',Fine sand,'))


def test_original_header_spelling_order_and_quoted_multiline_cells_survive():
    raw = ('facies_id, Name , Porosity ,custom,permeability,analysis_role,enabled,permeability_unit\n'
           '3,C,0.1,"C line 1\nC line 2",1,excluded,False,old\n'
           '5,F,0.2," text, with comma ",2,excluded,False,old\n').encode("utf8")
    result = build(raw=b'\xef\xbb\xbf' + raw)
    assert result.columns == tuple(next(csv.reader(io.StringIO(raw.decode("utf8")))))
    assert result.rows[0]["custom"] == "C line 1\nC line 2"
    assert result.rows[0][" Porosity "] == "0.32"
    assert result.rows[1]["custom"] == " text, with comma "
    assert result.rows[0]["enabled"] == "True"
    assert result.rows[0]["permeability_unit"] == "mD"


def test_existing_nonporous_role_semantics_are_explicit_not_missing_property_defaults():
    value = lp.PaintingSession(run_id="synthetic", permeability_unit="darcy")
    value.facies = {"Water": lp.FaciesEntry("free_water"),
                    "Excluded": lp.FaciesEntry("excluded"),
                    "Fixture": lp.FaciesEntry("colorchecker")}
    value.layers = [lp.PaintedLayer(index, name, "#aaaaaa", name)
                    for index, name in enumerate(value.facies, 1)]
    result = build(value, b'id,name\n', used=(1, 2, 3))
    assert [row["porosity"] for row in result.rows] == ["1.0", "0.0", "0.0"]
    assert [row["permeability"] for row in result.rows] == ["", "", ""]
    assert [row["enabled"] for row in result.rows] == ["True", "False", "False"]
    assert [row["analysis_role"] for row in result.rows] == ["free_water", "excluded", "colorchecker"]
    assert all(row["permeability_unit"] == "darcy" for row in result.rows)


@pytest.mark.parametrize("field,value", [
    ("porosity", None), ("porosity", 0.), ("porosity", 1.1),
    ("porosity", float("nan")), ("porosity", True),
    ("permeability", None), ("permeability", -1.), ("permeability", float("inf")),
    ("role", "unknown"),
])
def test_missing_or_invalid_painting_properties_refuse_instead_of_using_old_values(field, value):
    drawing = session()
    setattr(drawing.facies["F"], field, value)
    with pytest.raises(table.FaciesTableError):
        build(drawing)


@pytest.mark.parametrize("unit", ["", None, "source_unit", "DARCY"])
def test_missing_or_unsupported_painting_units_are_not_guessed(unit):
    drawing = session()
    drawing.permeability_unit = unit
    with pytest.raises(table.FaciesTableError, match="explicit supported permeability unit"):
        build(drawing)


@pytest.mark.parametrize("raw", [
    b'\xff', b'', b'id,name,Name\n1,F,F\n', b'id,facies_id,name\n1,1,F\n',
    b'id,name\n1,F,extra\n', b'id,name\n1\n', b'id,name\n1,"unterminated\n',
    b'id,name\n1,F\n1.0,C\n', b'id,name\nNaN,F\n', b'id,name\n1e1000000,F\n',
])
def test_malformed_ambiguous_or_unbounded_id_csv_is_refused(raw):
    with pytest.raises(table.FaciesTableError, match="Parent CSV is invalid"):
        build(raw=raw)


def test_expected_source_sha_and_metadata_bounds_are_enforced(monkeypatch):
    raw = parent()
    with pytest.raises(table.FaciesTableError, match="SHA-256"):
        table.build_adoption_facies_table(session(), raw,
            expected_sha256="0" * 64, used_layer_ids=(2, 7, 9))
    with monkeypatch.context() as limited:
        limited.setattr(table, "MAX_PARENT_BYTES", len(raw) - 1)
        with pytest.raises(table.FaciesTableError, match="metadata limit"):
            build(raw=raw)
    monkeypatch.setattr(table, "MAX_PARENT_ROWS", 2)
    with pytest.raises(table.FaciesTableError, match="too many rows"):
        build(raw=raw)


@pytest.mark.parametrize("used", [(), (0,), (True,), (256,), (3,), "279"])
def test_used_layer_identity_is_explicit_and_never_maps_outside_or_missing_layers(used):
    with pytest.raises(table.FaciesTableError):
        build(used=used)


def test_saved_layer_cannot_fall_back_to_an_invented_facies():
    drawing = session()
    drawing.layers[0].facies = "Missing"
    with pytest.raises(table.FaciesTableError, match="missing facies"):
        build(drawing)


def test_ambiguous_saved_painting_names_are_refused_instead_of_collapsed():
    drawing = session()
    drawing.facies[" F "] = lp.FaciesEntry("porous_medium", .4, 10.)
    with pytest.raises(table.FaciesTableError, match="ambiguous"):
        build(drawing)


# Exact current-table bytes, frozen from a metadata-only read. No test accesses
# the project or arrays. Saved facies/layers below are the corresponding closed
# version's literal scalar fields, with unrelated image/drawing metadata omitted.
REAL_PARENT = (
    b'id,name,porosity,permeability,swi CO2,krel_gas,1-Sg,krel_water,pc [mbar]\r\n'
    b'0,water,Nan,Nan,Nan,Nan,Nan,Nan,Nan\r\n'
    b'1,ESF,0.44,4.40E+01,0.32,0.09,0.86,0.71,15\r\n'
    b'2,C,0.44,473,0.14,0.05,0.9,0.93,3.3\r\n'
    b'3,D,0.44,1110,0.12,0.02,0.92,0.95,0.9\r\n'
    b'4,E,0.45,2005,0.12,0.1,0.94,0.93,0.26\r\n'
    b'5,F,0.44,4295,0.12,0.11,0.87,0.72,0.1\r\n'
    b'6,G,0.45,9580,0.12,0.16,0.94,0.75,0.01\r\n'
    b'7,F,0.44,4295,0.12,0.11,0.87,0.72,0.1\r\n'
    b'8,F,0.44,4295,0.12,0.11,0.87,0.72,0.1\r\n')
REAL_FACIES = {
    "Free water": {"role": "free_water", "porosity": None, "permeability": None},
    "ESF": {"role": "porous_medium", "porosity": .44, "permeability": 44.},
    "C": {"role": "porous_medium", "porosity": .44, "permeability": 473.},
    "F": {"role": "porous_medium", "porosity": .44, "permeability": 4295.},
    "Colorchecker": {"role": "excluded", "porosity": None, "permeability": None},
    "Silicone": {"role": "excluded", "porosity": None, "permeability": None},
}
REAL_LAYER_FACIES = ((2, "Free water"), (3, "F"), (4, "ESF"), (5, "C"),
    (6, "F"), (8, "C"), (10, "F"), (11, "C"), (12, "F"), (13, "ESF"),
    (14, "F"), (15, "ESF"), (16, "C"), (17, "Colorchecker"), (18, "Silicone"))


def real_session():
    """Build the closed version's scalar fixture without loading any real asset."""
    value = lp.PaintingSession(run_id="fixture", permeability_unit="darcy")
    value.facies = {name: lp.FaciesEntry(**fields) for name, fields in REAL_FACIES.items()}
    value.layers = [lp.PaintedLayer(identity, name, "#aaaaaa", name)
                    for identity, name in REAL_LAYER_FACIES]
    return value


@pytest.mark.parametrize("checker_role", ["excluded", "colorchecker"])
def test_real_table_and_saved_facies_keep_current_ids_and_blank_nonporous_extras(checker_role):
    drawing = real_session()
    drawing.facies["Colorchecker"].role = checker_role
    raw_before, drawing_before = bytes(REAL_PARENT), deepcopy(drawing)
    assert hashlib.sha256(REAL_PARENT).hexdigest() == "f2ea0976f1d9d8c27d5a3815bec3e4db622bc504f4ce96477aceb69258f359d7"
    result = build(drawing, REAL_PARENT, used=tuple(identity for identity, _ in REAL_LAYER_FACIES))
    assert result.facies_ids == {"Free water": 0, "ESF": 1, "C": 2, "F": 5,
                                 "Colorchecker": 9, "Silicone": 10}
    assert result.layer_to_facies[2] == 0
    assert all(result.layer_to_facies[identity] == result.facies_ids[name]
               for identity, name in REAL_LAYER_FACIES)
    assert result.source_max_id == 8
    assert result.retired_source_ids == (3, 4, 6, 7, 8)
    assert result.changed_source_ids == ()
    assert result.aliases == (table.FaciesAlias("water", "Free water", 0, 0,
                                               "explicit-free-water-role-alias"),)
    by_name = {row["name"]: row for row in result.rows}
    provenance = {row.name: row for row in result.provenance}
    extra = ("swi CO2", "krel_gas", "1-Sg", "krel_water", "pc [mbar]")
    expected = {"ESF": ("0.32", "0.09", "0.86", "0.71", "15"),
                "C": ("0.14", "0.05", "0.9", "0.93", "3.3"),
                "F": ("0.12", "0.11", "0.87", "0.72", "0.1")}
    for name, cells in expected.items():
        assert tuple(by_name[name][column] for column in extra) == cells
        assert all(dict(provenance[name].fields)[column] == "parent:exact-cell" for column in extra)
    for name in ("Free water", "Colorchecker", "Silicone"):
        assert all(by_name[name][column] == "" for column in extra)
        assert all(dict(provenance[name].fields)[column].startswith("not-applicable:analysis-role-")
                   for column in extra)
    assert provenance["Free water"].parent_ids == (0,)
    assert provenance["F"].parent_ids == (5, 7, 8)
    assert by_name["Free water"]["porosity"] == "1.0"
    assert all(by_name[name]["porosity"] == "0.0" for name in ("Colorchecker", "Silicone"))
    assert all(by_name[name]["permeability"] == "" for name in ("Free water", "Colorchecker", "Silicone"))
    assert all(by_name[name]["permeability_unit"] == "darcy" for name in by_name)
    assert REAL_PARENT == raw_before and drawing == drawing_before


def test_parent_without_extra_columns_keeps_existing_new_porous_name_behavior():
    result = build(raw=b'id,name\n3,C\n')
    assert result.facies_ids == {"C": 3, "F": 4}
    assert result.provenance[1].parent_ids == ()


def test_nonporous_extra_conflicts_are_not_carried_into_the_output():
    drawing = lp.PaintingSession(run_id="fixture", permeability_unit="mD")
    drawing.facies = {"Excluded": lp.FaciesEntry("excluded")}
    drawing.layers = [lp.PaintedLayer(1, "excluded", "#aaaaaa", "Excluded")]
    result = build(drawing, b'id,name,custom\n2,Excluded,old\n7,Excluded,conflict\n', used=(1,))
    assert result.facies_ids == {"Excluded": 2}
    assert result.rows[0]["custom"] == ""
    assert dict(result.provenance[0].fields)["custom"] == "not-applicable:analysis-role-excluded"
    assert result.retired_source_ids == (7,)


def test_new_nonporous_ids_are_above_every_parent_id_and_follow_painting_order():
    drawing = real_session()
    drawing.facies = dict(reversed(list(drawing.facies.items())))
    result = build(drawing, REAL_PARENT, used=tuple(identity for identity, _ in REAL_LAYER_FACIES))
    assert result.facies_ids["Silicone"] == 9 and result.facies_ids["Colorchecker"] == 10
    assert {name: result.facies_ids[name] for name in ("Free water", "ESF", "C", "F")} == {
        "Free water": 0, "ESF": 1, "C": 2, "F": 5}


def test_existing_name_ids_are_preserved_without_an_extra_role_or_alias_gate():
    drawing = real_session()
    changed = REAL_PARENT.replace(b'0,water,', b'0,unrelated,')
    result = build(drawing, changed, used=(2, 3, 4, 5))
    assert result.facies_ids["ESF"] == 1 and result.facies_ids["C"] == 2
    assert result.facies_ids["F"] == 5 and result.facies_ids["Free water"] == 9
    assert result.aliases == ()
    drawing.facies["Other water"] = lp.FaciesEntry("free_water")
    drawing.layers.append(lp.PaintedLayer(19, "other water", "#aaaaaa", "Other water"))
    result = build(drawing, REAL_PARENT, used=(2, 19))
    assert result.facies_ids == {"Free water": 9, "Other water": 10}
    assert result.aliases == ()


def test_existing_path_csv_reader_and_opt_in_byte_reader_agree_on_literal_cells(tmp_path):
    from darsia.fluidflower.geometry_raster_sources import _full_csv
    path = tmp_path / "parent.csv"
    path.write_bytes(REAL_PARENT)
    assert _full_csv(path) == _full_csv(None, csv_bytes=REAL_PARENT, strict=True)
    assert path.read_bytes() == REAL_PARENT
