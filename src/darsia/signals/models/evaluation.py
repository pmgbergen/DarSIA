"""Evaluation modes: how colors are turned into a quantitative signal.

A color path defines an ordered set of key colors. The calibrated values at the key
colors (the signal values) are turned into a signal for arbitrary colors by an
*evaluation mode*. The mode is chosen in the config, is globally constant for all labels
and segments, and is stored with the mass calibration in ``evaluation.json``. A missing
file means the simple piecewise linear evaluation.

"""

import abc
import itertools
import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

import darsia
from darsia.signals.models.pwtransformation import (
    monotonicity_message,
    monotonicity_violation,
)

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

    projects_onto_path: ClassVar[bool] = True
    """Whether colors are projected onto the color path before the evaluation (the
    signal is then a function of the position along the path), or the evaluation
    uses the (baseline-relative) colors directly."""

    FILE_NAME: ClassVar[str] = "evaluation.json"
    FORMAT_VERSION: ClassVar[int] = 1

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "type" in cls.__dict__:
            EvaluationMode.registry[cls.type] = cls

    @property
    def per_label(self) -> dict[int, dict]:
        """Label-specific data (json-serializable), e.g. derived choices to reproduce."""
        if "_per_label" not in self.__dict__:
            self._per_label: dict[int, dict] = {}
        return self._per_label

    def describe(self) -> str:
        """Short, human-readable description of the mode and its parameters."""
        parameters = ", ".join(f"{k}={v}" for k, v in self.parameters().items())
        return f"{self.type} ({parameters})" if parameters else self.type

    def same_as(self, other: "EvaluationMode") -> bool:
        """Whether the other mode has the same type and parameters (not label data)."""
        return self.type == other.type and self.parameters() == other.parameters()

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

        The ``per_label`` entry holds label-specific data, e.g. choices derived from the
        color paths which must be reproducible.

        """
        return {
            "format_version": self.FORMAT_VERSION,
            "mode": self.type,
            "parameters": self.parameters(),
            "per_label": {str(label): data for label, data in self.per_label.items()},
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
        mode = EvaluationMode.registry[name].from_parameters(data.get("parameters", {}))
        mode.per_label.update(
            {int(label): d for label, d in data.get("per_label", {}).items()}
        )
        return mode

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
        label: int | None = None,
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
        label : int, optional
            Label; allows to store and reuse label-specific data, see ``per_label``.

        """


class LinearEvaluation(EvaluationMode):
    """Piecewise linear evaluation between the key colors (default)."""

    type = "linear"

    def parameters(self) -> dict:
        return {}

    @classmethod
    def from_parameters(cls, parameters: dict) -> "LinearEvaluation":
        return cls()

    def signal_function(self, color_path, color_mode, supports, values, label=None):
        return darsia.PWTransformation(supports, values)


class SplineEvaluation(EvaluationMode):
    """Smooth, monotone cubic (PCHIP) evaluation through the values at the key colors."""

    type = "spline"

    KINDS = ("pchip",)

    def __init__(self, kind: str = "pchip") -> None:
        if kind not in self.KINDS:
            raise ValueError(
                f"Unknown spline kind '{kind}'. Known kinds: {list(self.KINDS)}."
            )
        self.kind = kind

    def parameters(self) -> dict:
        return {"kind": self.kind}

    def describe(self) -> str:
        return f"spline ({self.kind})"

    @classmethod
    def from_parameters(cls, parameters: dict) -> "SplineEvaluation":
        return cls(kind=parameters.get("kind", "pchip"))

    def signal_function(self, color_path, color_mode, supports, values, label=None):
        return darsia.PchipTransformation(supports, values)


def select_affine_supports(key_colors: np.ndarray) -> list[int]:
    """Select at most 4 key colors, which determine an affine function of the color.

    A linear kernel is an affine function of the color (3 slopes and an offset), which
    can interpolate at most 4 key colors. The first and last key colors are always
    kept. Of the interior ones, the pair is chosen which maximizes the volume of the
    tetrahedron with the first and last key color, i.e., the best conditioned choice.
    Degenerate (collinear or coplanar) key colors fall back to an even split by index.

    Parameters
    ----------
    key_colors : np.ndarray
        Key colors of shape (n, 3).

    Returns
    -------
    list[int]
        Indices of the selected key colors (all, if there are at most 4).

    """
    colors = np.asarray(key_colors, dtype=float)
    num = len(colors)
    if num <= 4:
        return list(range(num))
    first, last = colors[0], colors[-1]
    best, best_volume = None, 1e-12
    for i, j in itertools.combinations(range(1, num - 1), 2):
        volume = abs(
            np.linalg.det(
                np.stack((colors[i] - first, colors[j] - first, last - first))
            )
        )
        if volume > best_volume:
            best, best_volume = (i, j), volume
    if best is None:
        i = max(1, min(num - 3, round((num - 1) / 3)))
        j = max(i + 1, min(num - 2, round(2 * (num - 1) / 3)))
        best = (i, j)
        logger.warning(
            "Key colors are (nearly) collinear or coplanar; the linear kernel uses "
            "an even split of the key colors."
        )
    return [0, best[0], best[1], num - 1]


class KernelSignalFunction:
    """Signal function evaluating a kernel interpolation of the key colors.

    The kernel interpolation uses the key colors of the color path as supports and the
    signal values at the key colors as goal values. In contrast to the piecewise
    functions, it is evaluated at the colors themselves and not at their projection onto
    the color path: colors far away from all key colors decay towards the baseline
    signal. The output is clipped to the range of the values.

    """

    def __init__(
        self,
        color_path: "darsia.ColorPath",
        color_mode: "darsia.ColorMode",
        supports: np.ndarray | list[float],
        values: np.ndarray | list[float],
        kernel: "darsia.BaseKernel",
        regularization: float = 1e-8,
        support_indices: list[int] | None = None,
    ) -> None:
        self.color_path = color_path
        self.color_mode = color_mode
        self.kernel = kernel
        self.regularization = regularization
        self.supports = np.asarray(supports, dtype=float)
        """Equidistant parameters of the key colors along the color path."""
        self.key_colors = np.asarray(
            color_path.interpret(self.supports, color_mode, mode="equidistant"),
            dtype=float,
        )
        """Key colors (relative or absolute, depending on the color mode)."""
        self.support_indices = (
            np.arange(len(self.supports))
            if support_indices is None
            else np.asarray(support_indices, dtype=int)
        )
        """Indices of the key colors used as supports of the kernel interpolation. The
        values at the other key colors do not influence the signal."""
        self.values = None
        self._interpolation = darsia.KernelInterpolation(
            kernel, regularization=regularization, deduplicate=False
        )
        """Kernel interpolation; identical key colors (e.g., of a zero path) are kept
        and are solvable by the regularization."""
        self.update(values=values)

    def update(
        self,
        supports: np.ndarray | list[float] | None = None,
        values: np.ndarray | list[float] | None = None,
        dofs: np.ndarray | list[int] | None = None,
    ) -> None:
        """Update the values at the key colors and refit the interpolation weights."""
        if supports is not None:
            raise ValueError("The key colors are defined by the color path.")
        # Candidate values; stored only if they are valid. Only the values at the used
        # key colors matter.
        new_values = self.values
        if values is not None:
            if dofs is not None:
                new_values = np.array(self.values, dtype=float)
                new_values[np.array(dofs)] = values
            else:
                new_values = np.array(values, dtype=float)
        assert len(new_values) == len(
            self.supports
        ), f"wrong size: {len(new_values)} vs. {len(self.supports)}"
        violation = monotonicity_violation(new_values, self.support_indices)
        if violation is not None:
            raise ValueError(monotonicity_message(new_values, violation))
        self.values = new_values

        # Regularized kernel interpolation, i.e., (K + lambda I) w = values. The kernel
        # matrix is only set up once, later updates reuse it.
        used = self.support_indices
        if self._interpolation.supports is None:
            self._interpolation.update(
                supports=self.key_colors[used], values=self.values[used]
            )
            if np.ptp(self.values[used]) > 0:
                condition = np.linalg.cond(self._interpolation.X)
                if condition > 1e8:
                    logger.warning(
                        "Ill-conditioned kernel interpolation (condition number "
                        f"{condition:.1e}): the used key colors are (nearly) "
                        "degenerate or the kernel is too narrow/wide."
                    )
        else:
            self._interpolation.update(values=self.values[used])

    def __call__(self, colors: np.ndarray) -> np.ndarray:
        """Evaluate the signal at colors of shape (..., 3)."""
        signal = self._interpolation(np.asarray(colors))
        used_values = self.values[self.support_indices]
        return np.clip(signal, np.min(used_values), np.max(used_values))

    def evaluate_on_path(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the signal at the colors on the color path at parameters t."""
        t = np.asarray(t, dtype=float)
        colors = self.color_path.interpret(t, self.color_mode, mode="equidistant")
        return self(colors)

    def inverse(self, value: float | np.ndarray) -> float | np.ndarray:
        """Parameter along the color path at which the signal first reaches a value."""
        t, y = signal_curve(self, num_fine_segments=100)
        return curve_inverse(t, y, value)

    def save(self, path: Path) -> None:
        """Store the supports and values (the kernel is part of the mode)."""
        import pandas as pd

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"supports": self.supports, "values": self.values}).to_csv(
            path.with_suffix(".csv"), index=False
        )


class KernelEvaluation(EvaluationMode):
    """Kernel evaluation directly on the colors, with the key colors as supports and the
    signal values at the key colors as goal values.

    The Gaussian kernel uses all key colors. The linear kernel is an affine function of
    the color and can interpolate at most 4 key colors; for more key colors, 4 are
    selected, see :func:`select_affine_supports`, and stored per label in ``per_label``.

    """

    type = "kernel"
    projects_onto_path = False

    KERNELS = ("gaussian", "linear")

    def __init__(
        self,
        kernel: str = "gaussian",
        gamma: float = 10.0,
        shift: float = 1.0,
        regularization: float = 1e-8,
    ) -> None:
        if kernel not in self.KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Known kernels: {list(self.KERNELS)}."
            )
        if gamma <= 0 or shift < 0 or regularization < 0:
            raise ValueError(
                "gamma must be positive, shift and regularization nonnegative."
            )
        self.kernel = kernel
        self.gamma = float(gamma)
        self.shift = float(shift)
        self.regularization = float(regularization)

    def parameters(self) -> dict:
        kernel_parameter = (
            {"gamma": self.gamma}
            if self.kernel == "gaussian"
            else {"shift": self.shift}
        )
        return {
            "kernel": self.kernel,
            **kernel_parameter,
            "regularization": self.regularization,
        }

    @classmethod
    def from_parameters(cls, parameters: dict) -> "KernelEvaluation":
        return cls(
            **{
                key: parameters[key]
                for key in parameters
                if key in {"kernel", "gamma", "shift", "regularization"}
            }
        )

    def describe(self) -> str:
        parameters = self.parameters()
        kernel = parameters.pop("kernel")
        details = ", ".join(f"{k}={v}" for k, v in parameters.items())
        return f"kernel ({kernel}, {details})"

    def _support_indices(self, key_colors: np.ndarray, label: int | None) -> list[int]:
        """Indices of the key colors used by the linear kernel (stored per label)."""
        num = len(key_colors)
        stored = (
            None
            if label is None
            else self.per_label.get(label, {}).get("support_indices")
        )
        if stored is not None:
            valid = (
                len(stored) == len(set(stored)) == min(num, 4)
                and all(0 <= index < num for index in stored)
                and stored[0] == 0
                and stored[-1] == num - 1
                and list(stored) == sorted(stored)
            )
            if valid:
                return [int(index) for index in stored]
            logger.warning(
                "Stored supports of label %s do not match its %d key colors; "
                "selecting them anew.",
                label,
                num,
            )
        indices = [int(index) for index in select_affine_supports(key_colors)]
        if label is not None:
            self.per_label.setdefault(int(label), {})["support_indices"] = indices
        return indices

    def signal_function(self, color_path, color_mode, supports, values, label=None):
        indices = None
        if self.kernel == "linear":
            key_colors = color_path.interpret(
                np.asarray(supports, dtype=float), color_mode, mode="equidistant"
            )
            indices = self._support_indices(np.asarray(key_colors), label)
        kernel = (
            darsia.GaussianKernel(gamma=self.gamma)
            if self.kernel == "gaussian"
            else darsia.LinearKernel(a=self.shift)
        )
        return KernelSignalFunction(
            color_path,
            color_mode,
            supports,
            values,
            kernel=kernel,
            regularization=self.regularization,
            support_indices=indices,
        )


def signal_curve(
    signal_function, num_fine_segments: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the signal along the color path on a fine grid.

    Each segment between two key colors is split into fine segments, and the signal
    function is evaluated at the colors on the path at the fine nodes.

    Parameters
    ----------
    signal_function : PWTransformation or KernelSignalFunction
        Signal function of a label, see :class:`EvaluationMode`.
    num_fine_segments : int
        Number of fine segments per segment between two key colors.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Parameters along the color path and signal at the fine nodes.

    """
    supports = np.asarray(signal_function.supports, dtype=float)
    t = np.concatenate(
        [
            np.linspace(start, stop, num_fine_segments + 1)[:-1]
            for start, stop in zip(supports[:-1], supports[1:])
        ]
        + [supports[-1:]]
    )
    return t, np.asarray(signal_function.evaluate_on_path(t), dtype=float)


def curve_inverse(
    t: np.ndarray, signal: np.ndarray, value: float | np.ndarray
) -> float | np.ndarray:
    """Parameter at which a sampled signal curve first reaches a value.

    The running maximum of the curve is used, so the result is well-defined also for
    curves which are not monotone, e.g. between key colors. Values outside the range of
    the curve are clipped to its end points.

    Parameters
    ----------
    t : np.ndarray
        Parameters along the color path, increasing.
    signal : np.ndarray
        Signal at the parameters.
    value : float or np.ndarray
        Signal value(s) to invert.

    Returns
    -------
    float or np.ndarray
        Parameter(s) along the color path.

    """
    return np.interp(value, np.maximum.accumulate(signal), t)
