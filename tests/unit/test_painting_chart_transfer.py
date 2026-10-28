"""Finite result tests; these are not actual 39-run map/Build qualification."""
import hashlib
import numpy as np
import pytest

from darsia.fluidflower import painting_chart_transfer as chart


def inputs():
    labels = np.full((60, 80), 3, np.uint8)
    whole = np.zeros(labels.shape, bool)
    whole[8:16, 20:30] = True
    labels[whole] = 17
    drawn = np.zeros(labels.shape, np.uint8)
    drawn[8:12, 20:30] = 2
    restored = chart.restore_chart_material(labels, drawn, whole,
        chart_id=17, known_ids=[2, 3, 17, 18])
    centres = np.asarray([[x, -y] for y in range(6) for x in range(4)], np.float64)
    return dict(base_labels=labels, inside=np.ones(labels.shape, bool),
        original_chart_mask=whole.copy(), source_material=restored["material_labels"],
        whole_chart_mask=whole, source_to_native_h=np.eye(3),
        native_to_metric_h=np.diag([1., -1., 1.]), reference_centres_mm=centres,
        target_centres_mm=centres + [0., -15.],
        source_binding={"fixture": "finite original-pixel seam, no actual scientific authority"},
        chart_id=17, known_ids=[2, 3, 17, 18])


def test_drawn_first_else_source_row_left_retains_freewater_and_outside_zero():
    labels = np.full((5, 8), 3, np.uint8)
    labels[0] = 0
    labels[1, :2] = 2
    hidden = np.zeros(labels.shape, bool)
    hidden[:, 2:6] = True
    labels[hidden] = 17
    drawn = np.zeros_like(labels)
    drawn[2, 2:6] = 2
    original = labels.copy()
    result = chart.restore_chart_material(labels, drawn, hidden,
        chart_id=17, known_ids=[2, 3, 17])
    assert np.all(result["material_labels"][0, 2:6] == 0)
    assert np.all(result["material_labels"][1:3, 2:6] == 2)
    assert np.all(result["material_labels"][3:, 2:6] == 3)
    assert result["drawn_mask"].sum() == 4
    assert result["left_continued_mask"].sum() == 12
    assert result["unassigned_mask"].sum() == 4
    assert np.array_equal(result["material_labels"][~hidden], original[~hidden])
    assert np.array_equal(labels, original)


def test_missing_left_never_wraps_or_searches_right_or_vertical():
    labels = np.full((3, 5), 3, np.uint8)
    labels[:, :2] = 17
    hidden = labels == 17
    result = chart.restore_chart_material(labels, np.zeros_like(labels), hidden,
        chart_id=17, known_ids=[3, 17])
    assert np.all(result["material_labels"][hidden] == 0)
    assert np.array_equal(result["material_labels"][~hidden], labels[~hidden])


def test_chart_cannot_be_its_own_drawn_material():
    data = inputs()
    with pytest.raises(ValueError, match="non-ColorChecker"):
        chart.restore_chart_material(data["base_labels"], data["base_labels"],
            data["whole_chart_mask"], chart_id=17, known_ids=data["known_ids"])


def test_all_25_unchanged_masks_and_effective_maps_are_bit_exact():
    data = inputs()
    untouched = set(chart.RUNS) - chart.MOVED_RUNS
    assert len(untouched) == 25 and "ac17" in untouched
    for run in untouched:
        proposal = chart.propose_chart_transfer(run, **data)
        assert not proposal.metadata()["moved"]
        assert proposal.metadata()["margin_mm"] == 0.
        assert np.array_equal(proposal.result["chart_mask"], data["original_chart_mask"])
        assert np.array_equal(proposal.result["effective_labels"], data["base_labels"])


def test_only_the_exact_14_selected_runs_move():
    assert chart.MOVED_RUNS == {"ac23", "ac25", "ac26", "ac28", "ac29", "ac30", "ac31",
        "ac32", "ac33", "ac34", "ac35", "ac38", "ac39", "ac40"}
    assert not chart.propose_chart_transfer('sample', **inputs()).metadata()['automatic']


def test_move_replaces_old_exclusion_restores_known_layers_and_changes_no_bystanders():
    data = inputs()
    data["base_labels"][40:45] = 18
    before = data["base_labels"].copy()
    proposal = chart.propose_chart_transfer("ac39", **data)
    result = proposal.result
    assert result["vacated_mask"].any()
    assert np.all(result["effective_labels"][8:12, 20:30] == 2)
    assert np.all(result["effective_labels"][12:16, 20:30] == 3)
    assert not np.any(result["material_labels"] == 17)
    assert np.all(result["effective_labels"][result["chart_mask"]] == 17)
    unrelated = ~(data["original_chart_mask"] | result["chart_mask"])
    assert np.array_equal(result["effective_labels"][unrelated], before[unrelated])
    assert np.array_equal(result["material_labels"][~data["original_chart_mask"]],
                          before[~data["original_chart_mask"]])
    assert np.array_equal(data["base_labels"], before)


def test_exact_five_mm_metric_margin_uses_painted_cells_not_pixel_radius():
    data = inputs()
    data["native_to_metric_h"] = np.diag([2., -3., 1.])
    data["target_centres_mm"] = data["reference_centres_mm"] + [20., -30.]
    result = chart.propose_chart_transfer("ac39", **data).result
    # Whole cells x[19.5,29.5] -> x[29.5,39.5] at two mm/px.
    # Five mm reaches x=27 exactly, whereas constant five PIXELS reaches24.5.
    assert result["chart_mask"][20, 27]
    assert not result["chart_mask"][20, 26]
    assert result["chart_mask"][16, 33]  # 4.5 mm above the moved cell edge.
    assert not result["chart_mask"][15, 33]  # 7.5 mm above it.


def test_manual_fringe_and_holes_are_part_of_whole_footprint():
    data = inputs()
    data["whole_chart_mask"][8, 32] = True  # Painted fringe outside an ideal quad.
    result = chart.propose_chart_transfer("ac39", **data).result
    assert result["chart_mask"][23, 32]
    assert result["chart_mask"][23, 37]  # Exact 4.5 mm from that fringe cell edge.
    assert not result["chart_mask"][23, 38]


def test_own_frame_clips_occlusion_without_inventing_material_outside_it():
    data = inputs()
    data["inside"][30:] = False
    proposal = chart.propose_chart_transfer("ac39", **data)
    assert not proposal.result["chart_mask"][30:].any()
    assert np.array_equal(proposal.result["material_labels"][30:], data["base_labels"][30:])


def test_whole_card_outline_survives_reference_boundary_and_follows_adjustment():
    data = inputs()
    data['inside'][:10] = False
    data['base_labels'][~data['inside']] = 0
    data['original_chart_mask'] = data['base_labels'] == data['chart_id']
    proposal = chart.propose_chart_transfer('sample', **data)
    paths = proposal.display_outlines
    assert np.concatenate(paths['chart_mask'])[:, 1].min() == pytest.approx(7.5)
    assert np.where(proposal.result['chart_mask'])[0].min() == 10
    assert np.array_equal(proposal.result['effective_labels'], data['base_labels'])
    adjusted = chart.propose_chart_transfer('sample', **data, correction_mm=[2., 3.])
    assert np.concatenate(adjusted.display_outlines['chart_mask'])[:, 1].min() == pytest.approx(-.5)
    assert not adjusted.result['chart_mask'][:10].any()
    assert np.array_equal(proposal.display_outlines['original_chart_mask'][0],
                          adjusted.display_outlines['original_chart_mask'][0])
    # Display geometry adds no array or saved approval identity to the product.
    restored = chart.ChartTransferProposal(proposal.payload, proposal.result)
    assert restored.validate() == proposal.validate()


@pytest.mark.parametrize('run', ['ac17', 'ac39'])
def test_original_colorchecker_can_cross_rig_boundary_without_changing_analysis_domain(run):
    data = inputs()
    data['inside'][8:10, 20:30] = False
    before = {key: value.copy() for key, value in data.items() if isinstance(value, np.ndarray)}
    proposal = chart.propose_chart_transfer(run, **data)
    result = chart.apply_chart_transfer(proposal, chart.confirm_chart_transfer(proposal,
        actor='Reviewer', reason='Checked the ColorChecker across the rig edge'))
    assert np.array_equal(result['original_chart_mask'], data['original_chart_mask'])
    assert np.all(result['material_labels'][8:10, 20:30] == 2)
    assert not np.any(result['unassigned_mask'] & ~data['inside'])
    if run == 'ac17':
        assert np.array_equal(result['chart_mask'], data['original_chart_mask'])
        assert np.array_equal(result['effective_labels'], data['base_labels'])
    else:
        assert not np.any(result['chart_mask'] & ~data['inside'])
    assert all(np.array_equal(data[key], value) for key, value in before.items())


@pytest.mark.parametrize('fault, reason', [('exclusion', 'exclusion differs'),
    ('target_layer', 'target labels contain'), ('material', 'continuous material'),
    ('binding', 'source binding'), ('footprint', 'footprint is empty')])
def test_border_crossing_retains_original_exclusion_material_and_source_checks(fault, reason):
    data = inputs()
    data['inside'][8:10, 20:30] = False
    if fault == 'exclusion':
        data['original_chart_mask'][8, 20] = False
    elif fault == 'target_layer':
        data['base_labels'][0, 0] = 99
    elif fault == 'material':
        data['source_material'][0, 0] = data['chart_id']
    elif fault == 'binding':
        data['source_binding'] = {}
    else:
        data['whole_chart_mask'][:] = False
    with pytest.raises(ValueError, match=reason):
        chart.propose_chart_transfer('ac17', **data)


@pytest.mark.parametrize('scale', [1., 1.1])
def test_ordered_centres_fit_rigid_rotation_and_translation_without_scaling(scale):
    data = inputs()
    reference = data["reference_centres_mm"]
    angle = .05
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    target = scale * (reference-reference.mean(0)) @ rotation.T + reference.mean(0) + [4., -9.]
    result = chart.chart_motion(reference, target)
    assert result["dx_mm"] == pytest.approx(4.)
    assert result["dy_up_mm"] == pytest.approx(-9.)
    assert result["rotation_applied"] is True
    assert result['rotation_degrees'] == pytest.approx(np.degrees(angle))
    np.testing.assert_allclose(result['rotation_matrix'], rotation, atol=1e-12)
    np.testing.assert_allclose(np.asarray(result['rotation_matrix']).T @ result['rotation_matrix'], np.eye(2), atol=1e-12)
    assert np.linalg.det(result['rotation_matrix']) == pytest.approx(1.)
    assert (result['fit_rms_mm'] < 1e-12) if scale == 1. else (result['fit_rms_mm'] > 0.)


def test_rotated_analysis_pixels_and_whole_display_outline_use_same_rigid_motion():
    from shapely import dwithin, points
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    data = inputs()
    reference = data['reference_centres_mm'] + [23., -10.]
    centre = reference.mean(0)
    rotation = np.array([[0., -1.], [1., 0.]])
    data['reference_centres_mm'] = reference
    data['target_centres_mm'] = (reference-centre) @ rotation.T + centre + [12., -6.]
    data['inside'][:14] = False
    proposal = chart.propose_chart_transfer('ac35', **data, correction_mm=[2., 3.])
    metadata = proposal.metadata()
    assert metadata['format'] == chart.FORMAT
    assert metadata['applied_motion']['rotation_degrees'] == pytest.approx(90.)
    original = box(19.5, -15.5, 29.5, -7.5)
    rotated = rotate(original, 90., origin=tuple(centre))
    expected = translate(rotated, xoff=14., yoff=-3.)
    assert expected.area == pytest.approx(original.area)
    yy, xx = np.indices(data['inside'].shape)
    expected_mask = dwithin(expected, points(xx, -yy), 5.) & data['inside']
    np.testing.assert_array_equal(proposal.result['chart_mask'], expected_mask)
    rings = proposal.display_outlines['chart_mask']
    displayed = unary_union([Polygon(np.asarray(ring) * [1., -1.]) for ring in rings])
    assert displayed.symmetric_difference(expected.buffer(5.)).area < 1e-9
    assert np.concatenate(rings)[:, 1].min() < 14.  # Whole display remains unclipped.
    old = data['original_chart_mask']
    bystanders = ~(old | expected_mask)
    np.testing.assert_array_equal(proposal.result['effective_labels'][bystanders], data['base_labels'][bystanders])


def test_nonautomatic_run_keeps_detected_rotation_informational_only():
    data = inputs()
    source = data['reference_centres_mm']
    centre = source.mean(0)
    data['target_centres_mm'] = (source-centre) @ np.array([[0., 1.], [-1., 0.]]) + centre
    unchanged = chart.propose_chart_transfer('ac17', **data)
    assert unchanged.metadata()['motion']['rotation_degrees'] == pytest.approx(90.)
    assert unchanged.metadata()['applied_motion']['rotation_degrees'] == 0.
    np.testing.assert_array_equal(unchanged.result['effective_labels'], data['base_labels'])
    corrected = chart.propose_chart_transfer('ac17', **data, correction_mm=[2., 3.])
    assert corrected.metadata()['applied_motion']['rotation_degrees'] == 0.


def test_v1_saved_approval_replays_exact_old_payload_and_is_not_upgraded():
    from darsia.fluidflower.painting_chart_transfer_reader import proposal_for_approval
    data = inputs()
    legacy = chart.propose_chart_transfer('ac39', **data, proposal_format=chart.LEGACY_FORMAT)
    # Captured from the original translation-only implementation before v2.
    assert hashlib.sha256(legacy.payload).hexdigest() == '037b4c84dcb76111924effd96eadbd2e88f00faa20e2894a90f8a5cd4e6696e5'
    approval = chart.confirm_chart_transfer(legacy, actor='Reviewer', reason='Saved v1 pixels')
    assert approval['format'] == chart.LEGACY_APPROVAL_FORMAT
    class Session:
        def proposal(self, run, *, correction_mm=None, proposal_format=chart.FORMAT):
            return chart.propose_chart_transfer(run, **data, correction_mm=correction_mm,
                                                proposal_format=proposal_format)
    replayed = proposal_for_approval(Session(), 'ac39', approval)
    assert replayed.payload == legacy.payload
    chart.apply_chart_transfer(replayed, approval)
    current = Session().proposal('ac39')
    with pytest.raises(ValueError, match='confirmation'):
        chart.apply_chart_transfer(current, approval)
    current_approval = chart.confirm_chart_transfer(current, actor='Reviewer', reason='New rigid pixels')
    assert current_approval['format'] == chart.APPROVAL_FORMAT
    with pytest.raises(ValueError, match='confirmation'):
        chart.apply_chart_transfer(legacy, current_approval)


def test_rigid_fit_refuses_degenerate_ordered_centres():
    reference = np.column_stack((np.arange(24, dtype=np.float64), np.zeros(24)))
    with pytest.raises(ValueError, match='non-collinear'):
        chart.chart_motion(reference, reference)


def test_fresh_policy_uses_detected_motion_for_arbitrary_runs_and_full_photo_domain():
    from darsia.fluidflower.painting_transfer_overlays import full_material_display
    from darsia.fluidflower.painting_chart_transfer_reader import proposal_for_approval
    policy = chart.fresh_transfer_policy()
    fmt = chart.review_proposal_format(dict(transfer_policy=policy))
    assert fmt == chart.DETECTED_FORMAT
    assert policy['material_display'] == 'full-source-to-full-camera-photo-v1'
    data = inputs()
    reference = data['reference_centres_mm']
    rotation = np.array([[0., -1.], [1., 0.]])
    data['target_centres_mm'] = (reference-reference.mean(0)) @ rotation.T + reference.mean(0) + [4., -9.]
    class Session:
        proposal_format = fmt
        def proposal(self, run, *, correction_mm=None, proposal_format=None):
            return chart.propose_chart_transfer(run, **data, correction_mm=correction_mm,
                proposal_format=self.proposal_format if proposal_format is None else proposal_format)
    session = Session()
    for run in ('source', 'next_run', 'experiment_b', 'ac17'):
        proposal = session.proposal(run)
        assert proposal.metadata()['automatic'] and proposal.metadata()['moved']
        assert proposal.metadata()['applied_motion']['rotation_degrees'] == pytest.approx(90.)
        approval = chart.confirm_chart_transfer(proposal, actor='Reviewer', reason='Fresh detected motion')
        assert approval['format'] == chart.DETECTED_APPROVAL_FORMAT
        replay = proposal_for_approval(session, run, approval)
        assert replay.payload == proposal.payload
        chart.apply_chart_transfer(replay, approval)
    # The approved source is carried straight to the full photo, even where
    # its own frame is outside the old 80x60 native raster.
    translation = [[1., 0., 30.], [0., 1., 30.], [0., 0., 1.]]
    own = [[-20., -15.], [100., -15.], [100., 75.], [-20., 75.]]
    full = full_material_display(data['source_material'], np.eye(3), translation, own, (120, 160))
    assert full[40, 15] == 3  # x=-15 native; outside old raster but inside frame.
    assert full[90, 125] == 3  # Beyond the old right and bottom raster edges.
    assert full[0, 0] == 0


def test_absent_fresh_policy_preserves_v2_payload_and_approval_replay():
    from darsia.fluidflower.painting_chart_transfer_reader import proposal_for_approval
    assert chart.review_proposal_format({}) == chart.FORMAT
    data = inputs()
    proposal = chart.propose_chart_transfer('ac39', **data)
    assert hashlib.sha256(proposal.payload).hexdigest() == 'e3ad28cac218cc2c0e040b12278878e759678acc316b3de68840b8468c069c4e'
    approval = chart.confirm_chart_transfer(proposal, actor='Reviewer', reason='Frozen v2 pixels')
    class FreshSession:
        proposal_format = chart.DETECTED_FORMAT
        def proposal(self, run, *, correction_mm=None, proposal_format=None):
            return chart.propose_chart_transfer(run, **data, correction_mm=correction_mm,
                proposal_format=self.proposal_format if proposal_format is None else proposal_format)
    assert proposal_for_approval(FreshSession(), 'ac39', approval).payload == proposal.payload
    assert not chart.propose_chart_transfer('experiment_b', **data).metadata()['moved']


def test_human_correction_is_translation_only_and_bound_by_confirmation():
    data = inputs()
    original = chart.propose_chart_transfer("ac39", **data)
    adjusted = chart.propose_chart_transfer("ac39", **data, correction_mm=[4., 2.])
    assert adjusted.metadata()["motion"] == original.metadata()["motion"]
    assert adjusted.metadata()["applied_motion"]["dx_mm"] == 4.
    assert adjusted.metadata()["applied_motion"]["dy_up_mm"] == -13.
    assert adjusted.metadata()["applied_motion"]["rotation_applied"] is False
    approval = chart.confirm_chart_transfer(adjusted, actor="Reviewer", reason="Adjusted position")
    approval["correction_mm"] = [0., 0.]
    with pytest.raises(ValueError, match="confirmation"):
        chart.apply_chart_transfer(adjusted, approval)
    adjusted_other = chart.propose_chart_transfer("ac17", **data, correction_mm=[4., 2.])
    assert adjusted_other.metadata()['applied_motion']['dx_mm'] == 4.
    assert adjusted_other.metadata()['applied_motion']['dy_up_mm'] == 2.
    assert adjusted_other.metadata()['margin_mm'] == 5.
    assert not adjusted_other.metadata()['automatic']
    assert not np.array_equal(adjusted_other.result['chart_mask'], data['original_chart_mask'])


def test_other_series_run_defaults_unchanged_and_allows_human_adjustment():
    data = inputs()
    original = chart.propose_chart_transfer('experiment_b', **data)
    assert not original.metadata()['moved']
    assert np.array_equal(original.result['effective_labels'], data['base_labels'])
    adjusted = chart.propose_chart_transfer('experiment_b', **data, correction_mm=[2., 3.])
    approval = chart.confirm_chart_transfer(adjusted, actor='Reviewer', reason='Moved chart on this photograph')
    assert chart.apply_chart_transfer(adjusted, approval)['chart_mask'].any()


@pytest.mark.parametrize("fault", ["float32", "nonfinite", "shape"])
def test_detection_arrays_refuse_wrong_abi(fault):
    data = inputs()
    points = data["target_centres_mm"].copy()
    if fault == "float32":
        points = points.astype(np.float32)
    elif fault == "nonfinite":
        points[0, 0] = np.nan
    else:
        points = points[:23]
    with pytest.raises(ValueError, match="24 ColorChecker centres"):
        chart.chart_motion(data["reference_centres_mm"], points)


def test_unknown_outside_material_retains_zero_and_is_not_a_new_human_gate():
    data = inputs()
    data["source_material"][8:10, 20:30] = 0
    proposal = chart.propose_chart_transfer("ac39", **data)
    approval = chart.confirm_chart_transfer(proposal, actor="Reviewer", reason="Checked this outline")
    result = chart.apply_chart_transfer(proposal, approval)
    assert result["unassigned_mask"][8:10, 20:30].all()
    assert np.all(result["effective_labels"][8:10, 20:30] == 0)


@pytest.mark.parametrize("fault", ["pixel", "run", "source", "actor"])
def test_confirmation_binds_exact_result_source_run_and_human(fault):
    proposal = chart.propose_chart_transfer("ac39", **inputs())
    approval = chart.confirm_chart_transfer(proposal, actor="Reviewer", reason="Checked this outline")
    if fault == "pixel":
        proposal.result["effective_labels"][0, 0] = 2
    elif fault == "run":
        approval["run_id"] = "ac40"
    elif fault == "source":
        metadata = proposal.metadata()
        metadata["source_binding"]["fixture"] = "another original source"
        proposal = chart.ChartTransferProposal(chart._json(metadata), proposal.result)
    else:
        approval["actor"] = ""
    with pytest.raises(ValueError):
        chart.apply_chart_transfer(proposal, approval)


def test_approved_results_are_copies_and_chart_remains_excluded_from_mass_domain():
    proposal = chart.propose_chart_transfer("ac39", **inputs())
    approval = chart.confirm_chart_transfer(proposal, actor="Reviewer", reason="Checked this outline")
    result = chart.apply_chart_transfer(proposal, approval)
    # The existing table uses chart17 as excluded, Freewater2/F3 remain
    # separate actual roles. Consumer integration tests must rederive physical
    # arrays from effective_labels; these finite pixels confer no Build proof.
    enabled = np.isin(result["effective_labels"], [2, 3])
    assert not np.any(enabled & result["chart_mask"])
    assert np.all(enabled[result["vacated_mask"]])
    result["effective_labels"][0, 0] = 2
    assert proposal.result["effective_labels"][0, 0] == 3


@pytest.mark.parametrize("bad", [-1, 256])
def test_original_integer_target_is_never_silently_truncated_to_uint8(bad):
    data = inputs()
    data["base_labels"] = data["base_labels"].astype(np.int32)
    data["base_labels"][0, 0] = bad
    with pytest.raises(ValueError, match="IDs 0..255"):
        chart.propose_chart_transfer("ac39", **data)
