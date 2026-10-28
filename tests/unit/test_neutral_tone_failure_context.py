"""Rejected measurements remain rejected and retain their exact fitting context."""
import json
import pickle

import numpy as np
import pytest

from darsia.fluidflower import image_correction as subject


def _swatches(neutrals):
    return subject.linear_to_srgb(np.broadcast_to(
        np.asarray(neutrals)[:, None], (6, 3))[None], dtype=np.float32)


def test_compressed_measured_neutrals_keep_physical_bounds_and_exact_context():
    source = np.repeat(_swatches([.8, .6, .4, .2, .1, .04]), 4, axis=0)
    reference = source.copy()
    source[-1, :, 1] = subject.linear_to_srgb(np.array([.999, .986, .63, .35, .19, .07]))
    reference[-1, :, 1] = subject.linear_to_srgb(np.array([.55, .39, .24, .13, .07, .027]))
    with pytest.raises(subject.NeutralToneFitError, match="channel 1") as caught:
        subject.fit_neutral_tone_model(source, reference)
    info = caught.value.diagnostics
    assert info["slope_bounds"] == [.05, 8.0]
    violations = [item for item in info["intervals"] if item["outside_bounds"]]
    assert any(item["endpoints"] == ["neutral_1", "neutral_0"] for item in violations)
    assert any(item["slope"] > 8 for item in violations)
    np.testing.assert_allclose(info["source_neutrals_linear"], subject.srgb_to_linear(source)[-1])
    json.dumps(info, allow_nan=False)
    restored = pickle.loads(pickle.dumps(caught.value))
    assert str(restored) == str(caught.value)
    assert restored.diagnostics == info


def test_artificial_endpoint_violation_is_distinguishable_from_measured_intervals():
    source = _swatches([.99, .8, .6, .4, .2, .1])
    reference = _swatches([.8, .65, .5, .35, .2, .1])
    with pytest.raises(subject.NeutralToneFitError) as caught:
        subject.fit_neutral_tone_model(source, reference)
    violations = [item for item in caught.value.diagnostics["intervals"] if item["outside_bounds"]]
    assert len(violations) == 1
    assert violations[0]["endpoints"] == ["neutral_0", "white_endpoint"]


def test_valid_neutral_response_is_unchanged():
    swatches = _swatches([.8, .6, .4, .2, .1, .04])
    fit = subject.fit_neutral_tone_model(swatches, swatches)
    for channel in fit["channels"]:
        assert channel["anchors_x"] == channel["anchors_y"]
        assert channel["slope_min"] == channel["slope_max"] == 1.0


def test_explicit_censored_neutrals_use_only_measured_response_for_highlights():
    source = np.repeat(_swatches([.8,.6,.4,.2,.1,.04]),4,axis=0)
    reference = source.copy()
    source[-1,:,1] = subject.linear_to_srgb(np.array([.999,.986,.63,.35,.19,.07]))
    reference[-1,:,1] = subject.linear_to_srgb(np.array([.55,.39,.24,.13,.07,.027]))
    valid = np.ones((4,6,3),bool)
    valid[-1,:2,1] = False
    fit = subject.fit_neutral_tone_model(source,reference,source_validity=valid)
    green = fit['channels'][1]
    assert green['excluded_neutral_indices'] == [0,1]
    assert green['valid_neutral_indices'] == [2,3,4,5]
    assert green['endpoint_policy'] == 'last-valid-measured-slope'
    assert green['anchor_origins'][-1] == 'measured_highlight_extrapolation'
    x,y = np.array(green['anchors_x']),np.array(green['anchors_y'])
    slopes = np.diff(y)/np.diff(x)
    assert .05 <= np.min(slopes) <= np.max(slopes) <= 8.
    assert slopes[-1] == pytest.approx(slopes[-2])
    assert 0 < y[-1] < .5  # A clipped sensor channel does not establish reference unit white.
    assert fit['channels'][0]['excluded_neutral_indices'] == []


@pytest.mark.parametrize('invalid_reference',[False,True])
def test_insufficient_valid_neutrals_are_rejected_on_either_side(invalid_reference):
    swatches=np.repeat(_swatches([.8,.6,.4,.2,.1,.04]),4,axis=0)
    valid=np.ones((4,6,3),bool);valid[-1,:3,1]=False
    options={'reference_validity' if invalid_reference else 'source_validity':valid}
    with pytest.raises(subject.NeutralToneFitError,match='Insufficient valid measured neutrals'):
        subject.fit_neutral_tone_model(swatches,swatches,**options)


def test_censoring_does_not_relax_the_measured_interval_guard():
    source=np.repeat(_swatches([.8,.6,.4,.2,.1,.04]),4,axis=0)
    reference=source.copy()
    source[-1,:,1]=subject.linear_to_srgb(np.array([.999,.986,.63,.62,.19,.07]))
    reference[-1,:,1]=subject.linear_to_srgb(np.array([.55,.39,.24,.13,.07,.027]))
    valid=np.ones((4,6,3),bool);valid[-1,:2,1]=False
    with pytest.raises(subject.NeutralToneFitError,match='physical bounds'):
        subject.fit_neutral_tone_model(source,reference,source_validity=valid)


def test_censored_highlight_is_rejected_instead_of_clipped_to_a_plausible_endpoint():
    source=np.repeat(_swatches([.95,.8,.4,.3,.2,.1]),4,axis=0)
    reference=source.copy()
    reference[-1,:,1]=subject.linear_to_srgb(np.array([.99,.95,.9,.8,.7,.6]))
    valid=np.ones((4,6,3),bool);valid[-1,:2,1]=False
    with pytest.raises(subject.NeutralToneFitError,match='highlight extrapolation'):
        subject.fit_neutral_tone_model(source,reference,source_validity=valid)


def test_affine_excludes_incomplete_patches_and_requires_full_rank():
    import darsia
    reference=darsia.ColorCheckerAfter2014().swatches_rgb
    source=reference.copy()
    valid=np.ones((4,6),bool);valid[0,0]=False
    expected=subject.fit_regularized_affine(source,reference,ridge=.01,huber_delta=.03,valid_swatches=valid)
    source[0,0]=[.999,.001,.99]
    actual=subject.fit_regularized_affine(source,reference,ridge=.01,huber_delta=.03,valid_swatches=valid)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_array_equal(actual[1],expected[1])
    with pytest.raises(ValueError,match='independent complete'):
        subject.fit_regularized_affine(np.ones((4,6,3))*.4,reference,
                                       ridge=.01,huber_delta=.03,valid_swatches=valid)


def test_real_uniform24_censoring_records_evidence_and_preserves_default_behavior():
    import darsia
    reference=darsia.ColorCheckerAfter2014().swatches_rgb
    source=subject.srgb_to_linear(reference)
    source[-1,:,1]=[.999,.986,.63,.35,.19,.07]
    photo=np.repeat(np.repeat(subject.linear_to_srgb(source),80,axis=0),84,axis=1)
    roi=[[0,0],[0,504],[320,504],[320,0]]
    with pytest.raises(subject.NeutralToneFitError):
        subject.fit_uniform24_to_reference_swatches(photo,reference,roi_rc=roi)
    valid=np.ones((4,6,3),bool);valid[-1,:2,1]=False
    reference_valid=np.ones((4,6,3),bool);reference_valid[0,0,2]=False
    result=subject.fit_uniform24_to_reference_swatches(photo,reference,roi_rc=roi,
        source_validity=valid,reference_validity=reference_valid,
        validity_evidence={'source':{'criterion':'exact sensor CFA white level'}})
    measured=result['transform']['measurement_validity']
    assert measured['contract']==subject.UNIFORM24_VALIDITY_CONTRACT
    assert measured['evidence']['source']['criterion']=='exact sensor CFA white level'
    assert 18 not in measured['complete_fit_swatch_indices'] and 19 not in measured['complete_fit_swatch_indices']
    assert result['diagnostics']['after']['excluded_swatch_indices']==[0,18,19]
    for stage in ('before','tone','affine_candidate','after'):
        diagnostics=result['diagnostics'][stage]
        selected=diagnostics['fit_swatch_indices']
        assert selected==measured['complete_fit_swatch_indices']
        assert diagnostics['mean']==pytest.approx(np.mean(np.array(diagnostics['swatch_residuals'])[selected]))
    assert result['transform']['tone_model']['channels'][1]['endpoint_policy']=='last-valid-measured-slope'
    json.dumps(result['transform'],allow_nan=False)
