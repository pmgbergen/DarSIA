"""Real bounded file/xy/pose reader; tiny DATA, no current-rig authority."""
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import painting_chart_transfer_reader as reader


def save_json(path, value):
    path.write_bytes(json.dumps(value, allow_nan=False).encode("utf8"))
    return reader._bound(path)


def detection(tmp_path, basis="baseline", *, xy=None):
    xy = (np.asarray([[x, y] for y in range(6) for x in range(4)], np.float64)
          if xy is None else xy)
    baseline = tmp_path / "ac31_baseline.npz"
    np.savez(baseline, xy=xy)
    archive = reader._bound(baseline)
    raw_sha, model_sha = "a"*64, "b"*64
    actual_sha = raw_sha if basis == "baseline" else model_sha
    certificate_path = tmp_path / "certificate.json"
    certificate = save_json(certificate_path, dict(run="ac31", sources=dict(arw_sha256=actual_sha),
        review_to_camera_h=[[1., 0., 10.], [0., 1., 20.], [0., 0., 1.]]))
    sidecar = save_json(baseline.with_suffix(".json"), dict(
        contract="ffac-v5-source-bound-baseline-calibration-v1", run="ac31",
        npz_sha256=archive["sha256"], source=dict(source_arw_sha256=actual_sha,
        geometry_certificate_sha256=certificate["sha256"])))
    source_path = tmp_path / "source.json"
    source = save_json(source_path, dict(
        contract="ffac-ea-native-v5-per-run-array-key-binding-v1", run="ac31",
        fitted_baseline_path=str(baseline), fitted_baseline_sha256=archive["sha256"],
        fitted_baseline_sidecar_sha256=sidecar["sha256"], source_arw_sha256=actual_sha,
        source_certificate_path=str(certificate_path), source_certificate_sha256=certificate["sha256"]))
    j = np.array([[1., 0., .25], [0., 1., -.5], [0., 0., 1.]])
    model_metric = np.diag([2., -3., 1.])
    bridge = save_json(tmp_path / "bridge.json", dict(accepted=True,
        baseline_raw=dict(sha256=raw_sha), model_raw=dict(sha256=model_sha),
        H_baseline_u_to_model_u=j.tolist()))
    model = dict(run_id="ac31", reference_image=dict(sha256=raw_sha),
        model_photograph=dict(sha256=model_sha), H_camera_u_to_metric_mm=(model_metric@j).tolist(),
        H_model_photograph_u_to_metric_mm=model_metric.tolist(), baseline_to_model_registration=bridge)
    row = dict(run="ac31", baseline_path=str(baseline), baseline_sha256=archive["sha256"],
        sidecar_sha256=sidecar["sha256"], ea_source_binding_path=str(source_path),
        ea_source_binding_sha256=source["sha256"])
    return row, model, xy, j


@pytest.mark.parametrize("basis", ["baseline", "model"])
def test_actual_xy_certificate_and_saved_pose_map_to_same_declared_metric(tmp_path, basis):
    row, model, xy, j = detection(tmp_path, basis)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    actual, binding = reader.read_detection(row, model)
    camera = 8.*xy + 3.5 + [10., 20.]
    matrix = np.asarray(model["H_camera_u_to_metric_mm"] if basis == "baseline" else
                        model["H_model_photograph_u_to_metric_mm"])
    expected = np.column_stack((camera, np.ones(len(camera)))) @ matrix.T
    assert np.array_equal(actual, expected[:, :2]/expected[:, 2, None])
    assert binding["basis"].startswith(basis)
    assert len(binding["inputs"]) == (4 if basis == "baseline" else 5)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_unrelated_source_photograph_cannot_inherit_model_geometry(tmp_path):
    row, model, _, _ = detection(tmp_path)
    model["reference_image"]["sha256"] = "c"*64
    with pytest.raises(ValueError, match="neither"):
        reader.read_detection(row, model)


def test_saved_xy_byte_tamper_refuses_before_geometry(tmp_path):
    row, model, xy, _ = detection(tmp_path)
    np.savez(row["baseline_path"], xy=xy+1.)
    with pytest.raises(ValueError, match="input changed"):
        reader.read_detection(row, model)


@pytest.mark.parametrize("fault", ["not_accepted", "wrong_raw", "wrong_metric"])
def test_rebound_saved_pose_must_still_join_both_actual_images_and_current_h(tmp_path, fault):
    row, model, _, _ = detection(tmp_path, "model")
    path = model["baseline_to_model_registration"]["path"]
    bridge = json.loads(Path(path).read_bytes())
    if fault == "not_accepted":
        bridge["accepted"] = False
    elif fault == "wrong_raw":
        bridge["baseline_raw"]["sha256"] = "c"*64
    else:
        model["H_camera_u_to_metric_mm"][0][2] += 1.
    model["baseline_to_model_registration"] = save_json(Path(path), bridge)
    with pytest.raises(ValueError, match="pose bridge"):
        reader.read_detection(row, model)


@pytest.mark.parametrize("fault", ["dtype", "shape", "nonfinite"])
def test_xy_abi_refuses_even_when_original_file_hashes_are_honestly_rebound(tmp_path, fault):
    xy = np.asarray([[x, y] for y in range(6) for x in range(4)], np.float64)
    if fault == "dtype":
        xy = xy.astype(np.float32)
    elif fault == "shape":
        xy = xy[:23]
    else:
        xy[0, 0] = np.nan
    row, model, _, _ = detection(tmp_path, xy=xy)
    with pytest.raises(ValueError, match="24 ColorChecker centres"):
        reader.read_detection(row, model)


def test_certificate_source_cannot_be_rebound_to_a_different_raw(tmp_path):
    row, model, _, _ = detection(tmp_path)
    source_path = Path(row["ea_source_binding_path"])
    source = json.loads(source_path.read_bytes())
    cert_path = Path(source["source_certificate_path"])
    cert = json.loads(cert_path.read_bytes())
    cert["sources"]["arw_sha256"] = "c"*64
    bound = save_json(cert_path, cert)
    source["source_certificate_sha256"] = bound["sha256"]
    side_path = Path(row["baseline_path"]).with_suffix(".json")
    side = json.loads(side_path.read_bytes())
    side["source"]["geometry_certificate_sha256"] = bound["sha256"]
    side_bound = save_json(side_path, side)
    source["fitted_baseline_sidecar_sha256"] = side_bound["sha256"]
    source_bound = save_json(source_path, source)
    row.update(sidecar_sha256=side_bound["sha256"], ea_source_binding_sha256=source_bound["sha256"])
    with pytest.raises(ValueError, match="RAW/certificate"):
        reader.read_detection(row, model)


@pytest.mark.parametrize('certificate_run, accepted', [('AC31', True), ('aC31', True),
                                                       ('ac32', False), (' ac31', False)])
def test_certificate_run_case_preserves_exact_raw_and_rejects_other_runs(tmp_path, certificate_run, accepted):
    row, model, xy, _ = detection(tmp_path)
    source_path = Path(row['ea_source_binding_path'])
    source = json.loads(source_path.read_bytes())
    cert_path = Path(source['source_certificate_path'])
    cert = json.loads(cert_path.read_bytes())
    cert['run'] = certificate_run
    certificate = save_json(cert_path, cert)
    side_path = Path(row['baseline_path']).with_suffix('.json')
    side = json.loads(side_path.read_bytes())
    side['source']['geometry_certificate_sha256'] = certificate['sha256']
    side_bound = save_json(side_path, side)
    source.update(source_certificate_sha256=certificate['sha256'],
                  fitted_baseline_sidecar_sha256=side_bound['sha256'])
    source_bound = save_json(source_path, source)
    row.update(sidecar_sha256=side_bound['sha256'], ea_source_binding_sha256=source_bound['sha256'])
    if accepted:
        actual, binding = reader.read_detection(row, model)
        camera = 8. * xy + 3.5 + [10., 20.]
        expected = np.column_stack((camera, np.ones(len(camera)))) @ np.asarray(model['H_camera_u_to_metric_mm']).T
        assert np.array_equal(actual, expected[:, :2] / expected[:, 2, None])
        assert certificate in binding['inputs']
    else:
        with pytest.raises(ValueError, match='RAW/certificate'):
            reader.read_detection(row, model)


def test_original_int32_seven_field_archive_feeds_lossless_chart_result(tmp_path):
    from darsia.fluidflower.painted_analysis_exclusions import _arrays
    from darsia.fluidflower import painting_chart_transfer as chart
    from tests.unit.test_painting_chart_transfer import inputs

    data = inputs()
    original = data["base_labels"].astype(np.int32)
    mask = np.ones(original.shape, bool)
    path = tmp_path / "labels.npz"
    np.savez_compressed(path, array=original, filled=~mask, support=mask, inside=mask,
        continued=~mask, effective_coverage=mask, gaps=~mask)
    before = path.read_bytes()
    fields = _arrays(reader._bound(path))  # Original strict archive reader, no substitute.
    data["base_labels"] = fields["array"]
    data["inside"] = fields["inside"]
    proposal = chart.propose_chart_transfer("ac39", **data)
    assert fields["array"].dtype == np.int32
    assert proposal.result["material_labels"].dtype == np.int32
    assert proposal.result["effective_labels"].dtype == np.int32
    assert np.all(proposal.result["effective_labels"][8:12, 20:30] == 2)
    untouched = ~(proposal.result["chart_mask"] | data["original_chart_mask"])
    assert np.array_equal(proposal.result["effective_labels"][untouched], original[untouched])
    assert path.read_bytes() == before


def test_public_selected_run_rejects_another_reference_before_any_model_or_array_read(tmp_path):
    # Negative public-method seam, deliberately no fabricated positive Session
    # authority: a mismatched common reference must stop before any I/O.
    actual = save_json(tmp_path / "actual.json", {"run_id": "ac17"})
    foreign = save_json(tmp_path / "foreign.json", {"run_id": "ac17", "other": True})
    session = reader.ChartTransferSession.__new__(reader.ChartTransferSession)
    session.reference_run = 'ac17'
    session.rows = {"ac17": {"target_model": actual}, "ac39": {"reference_model": foreign}}
    with pytest.raises(ValueError, match="exact retained reference"):
        session.proposal("ac39")


def test_new_review_default_does_not_reinterpret_an_explicit_historical_approval():
    from darsia.fluidflower import painting_chart_transfer as chart
    from .test_painting_chart_transfer import inputs

    class Session:
        proposal_format = decision_proposal_format = chart.DETECTED_FORMAT

        def proposal(self, run_id, *, correction_mm=None, proposal_format=None):
            return chart.propose_chart_transfer(run_id, **inputs(), correction_mm=correction_mm,
                proposal_format=self.proposal_format if proposal_format is None else proposal_format)

    old = chart.propose_chart_transfer('ac18', **inputs(), proposal_format=chart.FORMAT)
    approval = chart.confirm_chart_transfer(old, actor='synthetic human', reason='Original historical proposal')
    session = Session()
    assert session.proposal('ac18').metadata()['automatic']
    replay = reader.proposal_for_approval(session, 'ac18', approval)
    assert replay.payload == old.payload
    np.testing.assert_array_equal(chart.apply_chart_transfer(replay, approval)['effective_labels'],
                                  old.result['effective_labels'])
