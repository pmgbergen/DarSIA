"""Evaluation modes: how colors are turned into a quantitative signal.

A color path defines an ordered set of key colors. The calibrated values at the key
colors (the signal values) are turned into a signal for arbitrary colors by an
*evaluation mode*. The mode is chosen in the config, is globally constant for all labels
and segments, and is stored with the mass calibration in ``evaluation.json``. A missing
file means the simple piecewise linear evaluation.

"""

import abc
import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

import darsia

logger = logging.getLogger(__name__)


class EvaluationMode(abc.ABC):
    """Base class of evaluation modes, including their (de)serialization.

    Subclasses define a unique ``type`` and are registered automatically. They provide
    their parameters as plain json-serializable data and act as factory for the signal
    function of a label. A signal function has the interface of
    :class:`darsia.PWTransformation`: ``supports``, ``values``, ``update(values=...)``,
    ``__call__``, ``inverse``, ``save`` and ``load``, plus
    ``evaluate_on_path(t)``, which evaluates the signal at the colors on the color path
    at the equidistant parameters ``t``.

    """

    type: ClassVar[str]
    """Unique name of the mode, as used in config and files."""

    registry: ClassVar[dict[str, type["EvaluationMode"]]] = {}
    """All known modes by name."""

    FILE_NAME: ClassVar[str] = "evaluation.json"
    FORMAT_VERSION: ClassVar[int] = 1

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "type" in cls.__dict__:
            EvaluationMode.registry[cls.type] = cls

    # ! ---- Serialization ---- ! #

    @abc.abstractmethod
    def parameters(self) -> dict:
        """Parameters of the mode as json-serializable data."""

    @classmethod
    @abc.abstractmethod
    def from_parameters(cls, parameters: dict) -> "EvaluationMode":
        """Create the mode from its parameters."""

    def to_dict(self) -> dict:
        """Describe the mode (and its parameters) as json-serializable data.

        The empty ``per_label`` entry is reserved for later, label-specific parameters.

        """
        return {
            "format_version": self.FORMAT_VERSION,
            "mode": self.type,
            "parameters": self.parameters(),
            "per_label": {},
        }

    @staticmethod
    def from_dict(data: dict) -> "EvaluationMode":
        """Create a mode from its description, see :meth:`to_dict`."""
        name = data["mode"]
        if name not in EvaluationMode.registry:
            raise ValueError(
                f"Unknown evaluation mode '{name}'. "
                f"Known modes: {sorted(EvaluationMode.registry)}."
            )
        return EvaluationMode.registry[name].from_parameters(data.get("parameters", {}))

    @staticmethod
    def from_name(name: str, parameters: dict | None = None) -> "EvaluationMode":
        """Create a mode by its name, e.g. from the config."""
        return EvaluationMode.from_dict({"mode": name, "parameters": parameters or {}})

    def save(self, folder: Path) -> None:
        """Store the mode in the calibration folder."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / self.FILE_NAME, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(folder: Path) -> "EvaluationMode":
        """Load the mode from the calibration folder.

        Calibrations without file (stored before modes existed) are evaluated
        piecewise linearly.

        """
        path = Path(folder) / EvaluationMode.FILE_NAME
        if not path.exists():
            logger.info(
                "No %s found. Using linear evaluation.", EvaluationMode.FILE_NAME
            )
            return LinearEvaluation()
        with open(path, "r") as f:
            return EvaluationMode.from_dict(json.load(f))

    # ! ---- Factory ---- ! #

    @abc.abstractmethod
    def signal_function(
        self,
        color_path: "darsia.ColorPath",
        color_mode: "darsia.ColorMode",
        supports: np.ndarray | list[float],
        values: np.ndarray | list[float],
    ):
        """Create the signal function of a label.

        Parameters
        ----------
        color_path : ColorPath
            Color path of the label; its key colors are the nodes.
        color_mode : ColorMode
            Color mode of the color path interpretation.
        supports : array-like
            Parameters of the key colors on the color path (equidistant).
        values : array-like
            Signal values at the key colors.

        """


class LinearEvaluation(EvaluationMode):
    """Piecewise linear evaluation between the key colors (default)."""

    type = "linear"

    def parameters(self) -> dict:
        return {}

    @classmethod
    def from_parameters(cls, parameters: dict) -> "LinearEvaluation":
        return cls()

    def signal_function(self, color_path, color_mode, supports, values):
        return darsia.PWTransformation(supports, values)
