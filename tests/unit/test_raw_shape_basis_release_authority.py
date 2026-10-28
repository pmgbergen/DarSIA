"""Source-owner releases replace separate reviewer identity without new gates."""
from copy import deepcopy
from datetime import datetime, timedelta
import json

import pytest

from .test_raw_shape_basis_reader import basis, _json, _read, _changed_byte  # noqa: F401
from .test_raw_shape_basis_consumers import _model
from darsia.fluidflower import painted_adoption_geometry as admission
from darsia.fluidflower import raw_shape_basis_reader as reader


@pytest.mark.parametrize("basis", ["owner-release"], indirect=True)
def test_owner_release_without_any_review_answer_closes_reader_and_preparation_proof(basis, tmp_path):
    assert set(basis.authority) == {"source_closure", "operation_request", "root_release"}
    assert not basis.review_path.exists()
    checked = _read(basis)
    checked.revalidate()
    _, bound, binding = _model(basis, tmp_path)
    proof = admission.make_proof(binding, {"sample": bound})
    assert "full_review" not in proof["raw_shape_basis"]["authorities"]["sample"]
    folder = tmp_path / "preparation"
    folder.mkdir()
    admission.write_configuration_copies(folder, proof)
    admission.verify_proof(binding, proof, {"sample": bound}, folder)
    assert not basis.review_path.exists()


@pytest.mark.parametrize("basis", ["owner-release"], indirect=True)
@pytest.mark.parametrize("part", ["request", "source_closure", "late_release"])
def test_owner_release_still_binds_exact_request_source_and_prebuild_time(basis, part):
    authority = deepcopy(basis.authority)
    release = json.loads(basis.release_path.read_bytes())
    if part == "request":
        release["operation_request"]["sha256"] = "0" * 64
    elif part == "source_closure":
        release["source_closure"]["tree"] = "3" * 40
    else:
        started = datetime.fromisoformat(basis.document["producer"]["started_utc"])
        release["released_utc"] = (started + timedelta(seconds=1)).isoformat()
    authority["root_release"] = _json(basis.release_path, release)
    with pytest.raises(ValueError, match="Root release"):
        reader.read_shape_basis(basis.certificate, run_id="sample", authority=authority,
                                parent_workspace=basis.parent_workspace)


@pytest.mark.parametrize("basis", ["owner-release"], indirect=True)
def test_owner_release_late_drift_is_detected_without_a_review_document(basis):
    checked = _read(basis)
    _changed_byte(basis.release_path)
    with pytest.raises(ValueError, match="changed"):
        checked.revalidate()
