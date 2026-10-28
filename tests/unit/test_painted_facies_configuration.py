"""Pure new-configuration recipes; no Store, Apply, Build or real model proof."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from darsia.fluidflower import adoption_facies_table as tables
from darsia.fluidflower import painted_facies_configuration as configs
from .test_adoption_facies_table import REAL_PARENT, REAL_FACIES, REAL_LAYER_FACIES, real_session


@pytest.fixture
def adopted():
    """Use the real closed parent-name/row scalars with a private table contract."""
    table = tables.build_adoption_facies_table(real_session(), REAL_PARENT,
        expected_sha256=hashlib.sha256(REAL_PARENT).hexdigest(),
        used_layer_ids=tuple(identity for identity, _name in REAL_LAYER_FACIES))
    spec = dict(sha256=table.sha256, size_bytes=len(table.csv_bytes),
        layer_to_facies={str(k): v for k, v in table.layer_to_facies.items()},
        source_max_id=table.source_max_id, retired_source_ids=list(table.retired_source_ids),
        changed_source_ids=list(table.changed_source_ids), aliases=[asdict(x) for x in table.aliases],
        provenance=json.loads(json.dumps([asdict(x) for x in table.provenance])))
    return dict(painted_adoption_table=dict(format="darsia-painted-adoption-facies-table-v1",
        source=dict(csv=dict(path="parent.csv", sha256=hashlib.sha256(REAL_PARENT).hexdigest(),
                             size_bytes=len(REAL_PARENT))),
        preparation=dict(path="closed/index.json", sha256="a" * 64, size_bytes=17), table=spec),
        regions=[dict(label_id=label, facies_id=table.facies_ids[name], name=name,
                      analysis_role=REAL_FACIES[name]["role"]) for label, name in REAL_LAYER_FACIES])


def _old():
    """Keep old segment values distinct from the painting's layer identities."""
    return dict(facies={"path": "old.npz", "props": "old.csv",
        "0": {"labels": [1, 2]}, "1": {"labels": [3, 12, 14]},
        "2": {"labels": [7, 10]}, "3": {"labels": []}, "4": {"labels": []},
        "5": {"labels": [9, 11]}, "6": {"labels": []},
        "7": {"labels": [4, 5, 6, 8]}, "8": {"labels": [0, 13]}},
        shared=dict(porosity_by_facies={"1": .44, "2": .44, "5": .44, "7": .44, "8": .44},
                    swi_by_facies={"1": .32, "2": .14, "5": .12, "7": .12, "8": .12},
                    unquantified_facies=[0, 3, 4, 6]),
        calibration=dict(color_path=dict(ignore_labels=[0, 3, 4, 6])),
        rig=dict(width=.9, height=.49, dim=2),
        models=dict(metadata=dict(label_ids=[0, 1, 2, 5, 7, 8], ignore_labels=[0])),
        provenance=dict(historical=dict(ignore_labels=[7, 8])))


def test_real_parent_names_rebuild_groups_from_layers_and_keep_literal_physics(adopted, monkeypatch):
    """No old segment membership or unrelated numeric value may be reused by id."""
    before = _old()
    snapshot = deepcopy(before)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("No file IO in pure rebuild"))
    result = configs.rebuild_painted_facies_configuration(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert before == snapshot
    assert result.document["facies"]["5"] == {"labels": [3, 6, 10, 12, 14]}
    assert result.document["facies"]["0"] == {"labels": [2]}
    assert result.document["facies"]["9"] == {"labels": [17]}
    assert result.document["facies"]["10"] == {"labels": [18]}
    assert set(result.document["facies"]) == {"path", "props", "0", "1", "2", "5", "9", "10"}
    assert result.document["shared"]["swi_by_facies"] == {"1": .32, "2": .14, "5": .12}
    assert result.document["shared"]["porosity_by_facies"] == {"1": .44, "2": .44, "5": .44}
    assert result.document["shared"]["unquantified_facies"] == [0, 9, 10]
    assert result.document["calibration"]["color_path"]["ignore_labels"] == [0, 9, 10]
    assert result.document["models"] == snapshot["models"]
    assert result.document["provenance"] == snapshot["provenance"]
    assert result.document["rig"] == snapshot["rig"]
    assert result.evidence["model_input_recipe"]["ignore_labels"] == [0, 9, 10]
    assert result.evidence["model_input_recipe_is_validated_model"] is False
    assert result.evidence["adopted_contract"] == adopted["painted_adoption_table"]


@pytest.mark.parametrize("value", [.13, 1])
def test_conflicting_duplicate_name_recipes_are_reported_without_guessing(adopted, value):
    """The equal F rows cannot hide differing numeric configuration recipes."""
    before = _old()
    before["shared"]["swi_by_facies"]["8"] = value
    with pytest.raises(ValueError, match="conflicting numerical recipes for facies F"):
        configs.rebuild_painted_facies_configuration(before, adopted, parent_csv_bytes=REAL_PARENT)


def test_numerically_equal_integer_float_recipes_do_not_silently_retype(adopted):
    """Typed equality prevents a silent recipe conversion during duplicate collapse."""
    before = _old()
    before["shared"]["swi_by_facies"].update({"5": 1, "7": 1.0, "8": 1})
    with pytest.raises(ValueError, match="conflicting numerical recipes"):
        configs.rebuild_painted_facies_configuration(before, adopted, parent_csv_bytes=REAL_PARENT)


def test_missing_recipe_is_reported_without_new_physical_gate_or_fabricated_values(adopted):
    """Existing consumers retain ownership of whether absent physical data is required."""
    before = _old()
    del before["shared"]["swi_by_facies"]["2"]
    missing = configs.rebuild_painted_facies_configuration(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert missing.evidence["missing_physical_recipes"] == [
        {"path": "shared.swi_by_facies", "facies_id": 2, "name": "C"}]
    assert "2" not in missing.document["shared"]["swi_by_facies"]
    result = configs.rebuild_painted_facies_configuration({"facies": {"path": "x", "props": "y"}},
        adopted, parent_csv_bytes=REAL_PARENT)
    assert "shared" not in result.document
    assert result.document["facies"]["10"]["labels"] == [18]


def test_non_group_facies_descriptor_and_fitted_metadata_remain_opaque(adopted):
    """A nested physical asset descriptor is unrelated to native group rows."""
    before = _old()
    before["shared"]["source"] = {"facies": {"path": "physical.npz", "sha256": "c" * 64}}
    result = configs.rebuild_painted_facies_configuration(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert result.document["shared"]["source"] == before["shared"]["source"]
    assert result.document["models"] == before["models"]


def test_declared_new_calibration_basis_follows_table_without_changing_models(adopted):
    """A requested future basis does not manufacture matching fitted artifacts."""
    before = _old()
    before["calibration_label_ids"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    result = configs.rebuild_painted_facies_configuration(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert result.document["calibration_label_ids"] == [0, 1, 2, 5, 9, 10]
    assert result.document["models"] == before["models"]
    assert result.evidence["fitted_model_eligibility_claimed"] is False


def test_rebuilt_configuration_is_idempotent_and_keeps_new_nonporous_ids(adopted):
    """Reusing fresh DATA must not interpret current ids as foreign parent ids."""
    first = configs.rebuild_painted_facies_configuration(_old(), adopted, parent_csv_bytes=REAL_PARENT)
    second = configs.rebuild_painted_facies_configuration(first.document, adopted, parent_csv_bytes=REAL_PARENT)
    assert second.document == first.document and second.changes == ()


def test_n1_data_does_not_gain_a_native_setup_facies_section(adopted):
    """Rebuild declared N1 maps without adding a key its validator does not accept."""
    before = {"format": "darsia-n1-run-v1", "shared": _old()["shared"]}
    result = configs.rebuild_painted_facies_configuration(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert "facies" not in result.document
    assert result.document["format"] == before["format"]
    assert result.document["shared"]["swi_by_facies"] == {"1": .32, "2": .14, "5": .12}


def test_descriptive_retired_changed_metadata_never_adds_a_recipe_gate(adopted):
    """Only the actual name/layer recipe matters; retirement metadata is descriptive."""
    adopted["painted_adoption_table"]["table"].update(retired_source_ids=[5], changed_source_ids=[5])
    result = configs.rebuild_painted_facies_configuration(_old(), adopted, parent_csv_bytes=REAL_PARENT)
    assert result.document["facies"]["5"]["labels"] == [3, 6, 10, 12, 14]


def test_wrong_parent_bytes_or_layer_recipe_cannot_mint_name_authority(adopted):
    """Use exact parent SHA and agreement between both adopted mapping records."""
    with pytest.raises(ValueError, match="SHA-256 differs"):
        configs.rebuild_painted_facies_configuration(_old(), adopted, parent_csv_bytes=REAL_PARENT+b" ")
    bad = deepcopy(adopted)
    bad["painted_adoption_table"]["table"]["layer_to_facies"]["17"] = 5
    with pytest.raises(ValueError, match="mapping differs"):
        configs.rebuild_painted_facies_configuration(_old(), bad, parent_csv_bytes=REAL_PARENT)


@pytest.mark.parametrize("ending,bom", [(b"\n", b""), (b"\r\n", b"\xef\xbb\xbf")])
def test_toml_wrapper_preserves_all_unrelated_bytes_types_and_historical_models(adopted, ending, bom):
    """Patch real table blocks and multiline lists without a whole-file serializer."""
    prefix = b'# unmodified precision and comments\n[rig]\nwidth = 0.9 # exact\ndim = 2\n'
    groups = b'[facies]\npath = "native.npz"\nprops = "table.csv"\n[facies.0]\nlabels=[1,2]\n[facies.7]\nlabels=[4,5,6,8]\n'
    shared = b'[shared.swi_by_facies]\n1=0.32\n2=0.14\n5=0.12\n7=0.12\n8=0.12\n[calibration.color_path]\nignore_labels = [\n 0, 3, 4, 6,\n]\n'
    footer = b'[models.metadata]\nlabel_ids=[0,1,2,5,7,8]\nignore_labels=[0]\n# historical\n[tail]\nvalue = "unchanged"\n'
    before = bom + (prefix+groups+shared+footer).replace(b"\n", ending)
    result = configs.rebuild_painted_facies_toml(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert result.toml_bytes.startswith(bom+prefix.replace(b"\n", ending))
    assert result.toml_bytes.endswith(footer.replace(b"\n", ending))
    assert b'path = "native.npz"'+ending+b'props = "table.csv"'+ending in result.toml_bytes
    assert tomllib.loads(result.toml_bytes.decode("utf-8-sig")) == result.document
    assert result.evidence["unrelated_byte_spans_retained"] is True
    assert result.evidence["model_input_recipe_is_validated_model"] is False


def test_partial_toml_adds_new_groups_without_rewriting_any_original_line(adopted):
    """The 43 partial configs need the same grouping as their merged companions."""
    before = b'# original\n[facies]\npath="old.npz"\nprops="old.csv"\n[rig]\nwidth=0.9\n'
    result = configs.rebuild_painted_facies_toml(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert result.toml_bytes.startswith(before)
    assert result.document["facies"]["5"]["labels"] == [3, 6, 10, 12, 14]


def test_ignore_list_before_groups_and_map_keeps_the_section_renderer_callable(adopted):
    """A list span must not shadow the renderer used by later changed sections."""
    before = (
        b'[calibration.color_path]\nignore_labels=[0,3,4,6]\n'
        b'[facies]\npath="old.npz"\nprops="old.csv"\n'
        b'[facies.7]\nlabels=[4,5,6,8]\n'
        b'[shared.swi_by_facies]\n1=0.32\n2=0.14\n5=0.12\n7=0.12\n8=0.12\n'
        b'[tail]\nvalue = "untouched" # preserve this exact line\n'
    )
    result = configs.rebuild_painted_facies_toml(before, adopted, parent_csv_bytes=REAL_PARENT)
    assert result.document["calibration"]["color_path"]["ignore_labels"] == [0, 9, 10]
    assert result.document["facies"]["5"]["labels"] == [3, 6, 10, 12, 14]
    assert result.document["shared"]["swi_by_facies"] == {"1": .32, "2": .14, "5": .12}
    assert result.toml_bytes.endswith(b'[tail]\nvalue = "untouched" # preserve this exact line\n')
    assert tomllib.loads(result.toml_bytes.decode()) == result.document
