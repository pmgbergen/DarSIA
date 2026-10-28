"""The original proven frame row can enter the new by-name recipe path.

All DATA is synthetic. The original pure table producer supplies the reserved
row; these cases neither prove outside pixels nor approve an interior gap.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from darsia.fluidflower import adoption_facies_table as tables
from darsia.fluidflower import painted_adoption_table as automatic
from darsia.fluidflower import painted_facies_configuration as configs
from darsia.fluidflower.layer_painting import OUTSIDE_FACIES, OUTSIDE_POROSITY
from .test_adoption_facies_table import REAL_PARENT, REAL_FACIES, REAL_LAYER_FACIES, real_session
from .test_painted_facies_configuration import _old


@pytest.fixture
def framed_adoption():
    physical = tables.build_adoption_facies_table(real_session(), REAL_PARENT,
        expected_sha256=hashlib.sha256(REAL_PARENT).hexdigest(),
        used_layer_ids=tuple(identity for identity, _name in REAL_LAYER_FACIES))
    table = automatic._with_frame_row(physical, REAL_PARENT)
    spec = dict(sha256=table.sha256, size_bytes=len(table.csv_bytes), columns=list(table.columns),
        layer_to_facies={str(key): value for key, value in table.layer_to_facies.items()},
        source_max_id=table.source_max_id, retired_source_ids=list(table.retired_source_ids),
        changed_source_ids=list(table.changed_source_ids), aliases=[asdict(item) for item in table.aliases],
        provenance=json.loads(json.dumps([asdict(item) for item in table.provenance])))
    regions = [dict(label_id=label, facies_id=table.facies_ids[name], name=name,
                    analysis_role=REAL_FACIES[name]["role"])
               for label, name in REAL_LAYER_FACIES]
    regions.append(dict(label_id=0, facies_id=table.facies_ids[OUTSIDE_FACIES],
        name=OUTSIDE_FACIES, analysis_role="frame", enabled=False,
        porosity=OUTSIDE_POROSITY, permeability=None))
    return dict(painted_adoption_table=dict(format="darsia-painted-adoption-facies-table-v1",
        source=dict(csv=dict(path="synthetic-parent.csv", sha256=table.parent_sha256,
                             size_bytes=len(REAL_PARENT))),
        preparation=dict(path="synthetic-closed/index.json", sha256="a" * 64, size_bytes=17),
        table=spec), regions=regions)


def test_original_automatic_frame_row_uses_existing_ignore_policy_without_physical_recipes(framed_adoption):
    before = _old()
    original, adoption = deepcopy(before), deepcopy(framed_adoption)
    result = configs.rebuild_painted_facies_configuration(before, framed_adoption, parent_csv_bytes=REAL_PARENT)
    number = framed_adoption["regions"][-1]["facies_id"]
    assert result.document["facies"][str(number)] == dict(labels=[0])
    assert number in result.document["calibration"]["color_path"]["ignore_labels"]
    assert number in result.document["shared"]["unquantified_facies"]
    assert str(number) not in result.document["shared"]["porosity_by_facies"]
    assert str(number) not in result.document["shared"]["swi_by_facies"]
    assert result.document["shared"]["porosity_by_facies"] == {"1": .44, "2": .44, "5": .44}
    assert result.document["shared"]["swi_by_facies"] == {"1": .32, "2": .14, "5": .12}
    assert result.document["models"] == original["models"]
    assert result.document["provenance"] == original["provenance"]
    assert before == original and framed_adoption == adoption
    assert result.evidence["numerical_values_reestimated"] is False
    assert result.evidence["fitted_model_eligibility_claimed"] is False


@pytest.mark.parametrize("fault", ["name", "enabled", "porosity", "permeability",
    "missing_permeability", "layer", "provenance", "parent_ids", "arbitrary_role"])
def test_an_arbitrary_or_changed_frame_claim_cannot_mint_nonphysical_authority(framed_adoption, fault):
    bad = deepcopy(framed_adoption)
    row = bad["regions"][-1]
    table = bad["painted_adoption_table"]["table"]
    provenance = next(item for item in table["provenance"] if item["facies_id"] == row["facies_id"])
    if fault == "name":
        row["name"] = provenance["name"] = "Unapproved interior area"
    elif fault == "enabled":
        row["enabled"] = True
    elif fault == "porosity":
        row["porosity"] = .2
    elif fault == "permeability":
        row["permeability"] = 0.
    elif fault == "missing_permeability":
        row.pop("permeability")
    elif fault == "layer":
        row["label_id"] = 19
        provenance["layer_ids"] = [19]
        table["layer_to_facies"]["19"] = table["layer_to_facies"].pop("0")
    elif fault == "provenance":
        provenance["fields"][0][1] = "unverified:frame-claim"
    elif fault == "parent_ids":
        provenance["parent_ids"].append(999)
    else:
        row["analysis_role"] = "unapproved_physical_role"
    with pytest.raises(ValueError, match="original automatic nonphysical row|no existing analysis role"):
        configs.rebuild_painted_facies_configuration(_old(), bad, parent_csv_bytes=REAL_PARENT)
