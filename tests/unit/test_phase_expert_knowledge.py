"""Phase maps as dynamic expert knowledge, combined with static definitions."""
import numpy as np
import pytest

import darsia
from darsia.fluidflower.mass import phase_expert_knowledge as pek
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter


def _labels():
    return np.array([[0, 1, 2], [255, 2, 1]], np.uint8)


def test_phase_map_regions_equal_the_former_label_tests():
    labels = _labels()
    expert = pek.PhaseExpertKnowledge(labels)
    np.testing.assert_array_equal(expert.dissolved, labels == 1)
    np.testing.assert_array_equal(expert.free_gas, labels == 2)
    np.testing.assert_array_equal(expert.co2, np.isin(labels, (1, 2)))
    np.testing.assert_array_equal(expert.unknown, labels == 255)
    np.testing.assert_array_equal(expert.water, labels == 0)


def test_static_definitions_restrict_their_own_mode_only():
    labels = _labels()
    no_gas_right = np.array([[True, True, False], [True, True, False]])
    expert = pek.PhaseExpertKnowledge(labels, {"saturation_g": no_gas_right})
    np.testing.assert_array_equal(expert.free_gas, (labels == 2) & no_gas_right)
    np.testing.assert_array_equal(expert.dissolved, labels == 1)
    np.testing.assert_array_equal(expert.co2, (labels == 1) | ((labels == 2) & no_gas_right))


@pytest.mark.parametrize("static", [{"other": np.ones((2, 3), bool)},
                                    {"saturation_g": np.ones((3, 3), bool)},
                                    {"saturation_g": np.ones((2, 3), np.uint8)}])
def test_invalid_static_definitions_are_refused(static):
    with pytest.raises(ValueError):
        pek.PhaseExpertKnowledge(_labels(), static)


def test_workflow_adapter_takes_the_phase_map_as_dynamic_state():
    labels = _labels()
    image = darsia.ScalarImage(np.ones((2, 3)), width=3., height=2.)
    adapter = pek.phase_map_adapter(ExpertKnowledgeAdapter())
    gas = adapter.apply(image, "saturation_g", state=labels)
    np.testing.assert_array_equal(gas.img, (labels == 2).astype(float))
    with pytest.raises(ValueError, match="needs its image state"):
        adapter.apply(image, "saturation_g")


def test_adapter_without_dynamic_sources_is_unchanged():
    image = darsia.ScalarImage(np.ones((2, 3)), width=3., height=2.)
    assert ExpertKnowledgeAdapter().apply(image, "saturation_g") is image
