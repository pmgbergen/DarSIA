"""A frame's phase map as dynamic expert knowledge for the mass calculation.

The phase map of one frame says where dissolved CO2 and where free gas may be.
It is dynamic expert knowledge in the sense of
:class:`darsia.presets.workflows.analysis.expert_knowledge.ExpertKnowledgeAdapter`:
the same modes (``concentration_aq``, ``saturation_g``), the same rule that a
dynamic definition is intersected with the static definitions of its mode.
Static definitions here are boolean masks already on the native grid.
"""
from __future__ import annotations

from functools import cached_property
from typing import Mapping

import numpy as np

# Category codes of the saved phase-map format (uint8).
WATER, DISSOLVED, FREE_GAS, UNKNOWN = 0, 1, 2, 255
PHASE_CODES = (WATER, DISSOLVED, FREE_GAS, UNKNOWN)
MODES = ("concentration_aq", "saturation_g")

# Dynamic sources: mode -> allowed region of that mode in one phase map.
PHASE_MAP_SOURCES = {
    "concentration_aq": lambda labels: labels == DISSOLVED,
    "saturation_g": lambda labels: labels == FREE_GAS,
}


def expert_knowledge_of(frame) -> "PhaseExpertKnowledge":
    """The frame's expert knowledge, for full and reduced frame records alike."""
    expert = getattr(frame, "expert_knowledge", None)
    if expert is None:
        expert = PhaseExpertKnowledge(frame.phase_labels, getattr(frame, "expert_static_masks", None))
    return expert


def phase_map_adapter(static_adapter=None):
    """Return a workflow ExpertKnowledgeAdapter whose dynamic state is a phase map.

    ``static_adapter`` keeps its static ROIs; its ``dynamic`` sources become the
    phase-map definitions. The state passed to ``mask_for``/``apply`` is the
    image's uint8 phase-label array.
    """
    from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
    adapter = ExpertKnowledgeAdapter() if static_adapter is None else static_adapter
    adapter.dynamic = dict(PHASE_MAP_SOURCES)
    return adapter


class PhaseExpertKnowledge:
    """Static and dynamic expert knowledge of one frame on its native grid."""

    def __init__(self, phase_labels: np.ndarray, static: Mapping[str, np.ndarray] | None = None):
        labels = np.asarray(phase_labels)
        if labels.ndim != 2 or labels.dtype != np.uint8:
            raise ValueError("A frame phase map must be a two-dimensional uint8 array")
        static = dict(static or {})
        for mode, mask in static.items():
            if mode not in MODES:
                raise ValueError("Unknown expert-knowledge mode: " + str(mode))
            if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != labels.shape:
                raise ValueError("Static expert knowledge must be a boolean native mask")
        self.labels, self.static = labels, static

    def __eq__(self, other):
        """Equal when the phase map and static definitions are equal."""
        if not isinstance(other, PhaseExpertKnowledge):
            return NotImplemented
        return (np.array_equal(self.labels, other.labels) and self.static.keys() == other.static.keys()
                and all(np.array_equal(mask, other.static[mode]) for mode, mask in self.static.items()))

    __hash__ = None

    def allowed(self, mode: str) -> np.ndarray:
        """Region where ``mode`` may be nonzero: phase map and static definitions."""
        if mode not in MODES:
            raise ValueError("Unknown expert-knowledge mode: " + str(mode))
        mask = PHASE_MAP_SOURCES[mode](self.labels)
        static = self.static.get(mode)
        return mask if static is None else mask & static

    @cached_property
    def dissolved(self) -> np.ndarray:
        """Where dissolved CO2 may be."""
        return self.allowed("concentration_aq")

    @cached_property
    def free_gas(self) -> np.ndarray:
        """Where free gas may be."""
        return self.allowed("saturation_g")

    @cached_property
    def co2(self) -> np.ndarray:
        """Where CO2 in either phase may be."""
        return self.dissolved | self.free_gas

    @cached_property
    def unknown(self) -> np.ndarray:
        """Where the phase map records no phase decision."""
        return self.labels == UNKNOWN

    @cached_property
    def water(self) -> np.ndarray:
        """Where the phase map records water without CO2."""
        return self.labels == WATER
