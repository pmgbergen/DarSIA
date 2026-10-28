"""Required method DATA preserves the existing optical and histogram equations."""
import ast
import inspect
from pathlib import Path
import subprocess

import numpy as np
import pytest

from darsia.fluidflower.mass import concentration_joint as joint
from darsia.fluidflower.mass import phase_mass_joint_optics as optics
from darsia.fluidflower.native import registration, support
from darsia.fluidflower.phase import features

ORIGINAL_COMMIT = "f0d338aa2be5e6ee8be6a2db0bd8f6fb70e9bd5a"
METHOD_DATA = {
    "joint_summary": {"bins": 8192, "max_signal": 8.0},
    "joint_fit": {"overflow_weight": 0.25},
    "joint_reference": {"sigma": 2.0, "linear_floor": 0.0001, "min_span": 0.15},
}


def original_function(module, name):
    """Bind one unchanged historical mathematical body to its existing dependencies."""
    repo = Path(__file__).resolve().parents[2]
    rel = Path(module.__file__).relative_to(repo).as_posix()
    text = subprocess.check_output(["git", "show", ORIGINAL_COMMIT + ":" + rel], cwd=repo).decode("utf-8")
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), rel, "exec"), namespace)
    return namespace[name]


def test_histogram_requires_method_data_and_keeps_exact_original_values():
    """The declared grid matches historical fields, including the positive tail bin."""
    signal = np.array([[0., .125, .5, 2., 9.]], np.float32)
    av = np.ones((1, 5), np.float32)
    gv = np.full((1, 5), .25, np.float32)
    solubility = np.full((1, 5), .02, np.float32)
    density = np.full((1, 5), .7, np.float32)
    identity = dict(run="sample", elapsed_h=2., injected_g=1.)
    with pytest.raises(TypeError, match="bins"):
        joint.summarize_frame(signal, av, gv, solubility, density, **identity)
    kwargs = dict(identity, **METHOD_DATA["joint_summary"])
    actual = joint.summarize_frame(signal, av, gv, solubility, density, **kwargs)
    expected = original_function(joint, "summarize_frame")(signal, av, gv, solubility, density, **kwargs)
    assert actual == expected
    alternate = joint.summarize_frame(signal, av, gv, solubility, density,
                                     **identity, bins=16, max_signal=4.)
    assert len(alternate["signal_values"]) == 17
    assert alternate["signal_quantization_half_width"] == 4. / 16 / 2
    assert alternate["signal_tail_capacity_g"] > 0


def test_optical_reference_requires_all_regularization_data_and_matches_original():
    """Explicit thresholds preserve every saved reference field and dtype."""
    baseline = np.full((5, 6, 3), .3, np.float32)
    clear = np.broadcast_to(np.array([.6, .4, .2], np.float32), baseline.shape).copy()
    facies = np.ones((5, 6), np.uint8)
    with pytest.raises(TypeError, match="sigma"):
        optics.build_joint_reference(baseline, clear, facies)
    kwargs = dict(METHOD_DATA["joint_reference"], rgb_encoding="camera_linear")
    actual = optics.build_joint_reference(baseline, clear, facies, **kwargs)
    expected = original_function(optics, "build_joint_reference")(baseline, clear, facies, **kwargs)
    assert vars(actual).keys() == vars(expected).keys()
    for key, left in vars(actual).items():
        right = vars(expected)[key]
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
        elif isinstance(left, dict) and all(isinstance(v, np.ndarray) for v in left.values()):
            for name, value in left.items():
                assert value.dtype == right[name].dtype and value.tobytes() == right[name].tobytes()
        else:
            assert left == right


def test_joint_fit_weight_is_required_and_original_validation_stays_strict():
    """No fit starts with an omitted or invalid declared overflow penalty."""
    with pytest.raises(TypeError, match="overflow_weight"):
        joint.fit_joint_summaries({"sample": [{}]})
    with pytest.raises(ValueError, match="Overflow weight"):
        joint.fit_joint_summaries({"sample": [{}]}, overflow_weight=0.)
    parameter = inspect.signature(joint.fit_joint_summaries).parameters["overflow_weight"]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("function, arguments", [
    (registration.match_metrics, ["scale"]),
    (support.compute_feature_support, ["baseline_origin_xy", "previous_origin_xy"]),
    (features.warp_reference_to_current, ["source_origin_xy"]),
    (features.align_temporal_inputs, ["baseline_origin_xy", "previous_origin_xy"]),
    (features.register_flatfield_reference, ["source_origin_xy", "target_origin_xy"]),
])
def test_grid_origins_and_scale_have_no_hidden_defaults(function, arguments):
    """Callers must supply actual coordinate conventions even for identity grids."""
    signature = inspect.signature(function)
    for argument in arguments:
        assert signature.parameters[argument].default is inspect.Parameter.empty
