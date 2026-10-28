"""Current gas-feature aliases retain their original selection and rejection."""
import numpy as np
import pytest
from scipy.special import expit
from darsia.fluidflower.phase.heads import gas_probability

PARAMETERS = {"variants": {"current": ["G0prime", "G0'", "G0\u2032", "G1"],
                           "flatfield_only": "G3", "combined": "G2"}}

def model(variant, width):
    return {"variant": variant, "feature_names": [str(i) for i in range(width)],
            "fit": {"mean": [0.0] * width, "scale": [1.0] * width,
                    "coefficients": [float(i + 1) for i in range(width)],
                    "intercept": 0.125}}

@pytest.mark.parametrize("variant", PARAMETERS["variants"]["current"])
def test_current_aliases_score_only_current_features(variant):
    current = np.asarray([[[0.25, -0.5], [1.25, 0.125]]], np.float32)
    flatfield = np.full((1, 2, 4), 1e6, np.float32)
    actual = gas_probability(model(variant, 2), current, flatfield,
                             parameters=PARAMETERS)
    expected = expit(current.astype(np.float64) @ np.asarray([1.0, 2.0])
                     + 0.125).astype(np.float32)
    assert np.array_equal(actual, expected)

def test_unknown_variant_rejected_before_compatible_concatenation():
    current = np.ones((1, 2, 2), np.float32)
    flatfield = np.ones((1, 2, 4), np.float32)
    with pytest.raises(ValueError, match="^Unknown gas feature variant 'UNSUPPORTED'$"):
        gas_probability(model("UNSUPPORTED", 6), current, flatfield,
                        parameters=PARAMETERS)
