from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import numpy as np
from scipy.signal import savgol_filter

Contour = np.ndarray  # OpenCV-style: (N,1,2) or Nx2


# ----------------------------
# Utilities
# ----------------------------


def _as_xy(contour: Contour) -> np.ndarray:
    """Convert contour to Nx2 float64 array."""
    c = np.asarray(contour)
    if c.ndim == 3 and c.shape[1] == 1 and c.shape[2] == 2:
        c = c[:, 0, :]
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"Contour must be (N,1,2) or (N,2); got {c.shape}")
    return c.astype(np.float64, copy=False)


def _as_contour(xy: np.ndarray, dtype=np.int32) -> Contour:
    """Convert Nx2 array to OpenCV contour shape (N,1,2)."""
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be Nx2; got {xy.shape}")
    return xy.astype(dtype).reshape((-1, 1, 2))


def _is_closed(xy: np.ndarray, tol: float = 1e-9) -> bool:
    """Heuristic: contour is closed if first and last points are identical (within tol)."""
    if len(xy) < 3:
        return False
    return np.linalg.norm(xy[0] - xy[-1]) <= tol


def _ensure_closed(xy: np.ndarray) -> np.ndarray:
    """If not closed, append first point to end."""
    if len(xy) == 0:
        return xy
    if not _is_closed(xy):
        return np.vstack([xy, xy[0]])
    return xy


def _wrap_pad_1d(arr: np.ndarray, pad: int) -> np.ndarray:
    """Circular padding for closed contours."""
    if pad <= 0:
        return arr
    return np.concatenate([arr[-pad:], arr, arr[:pad]])


def _clip_window_length(window_length: int, n: int) -> int:
    """Ensure odd window_length and <= n (and at least 3)."""
    wl = int(window_length)
    wl = max(wl, 3)
    if wl % 2 == 0:
        wl += 1
    if wl > n:
        wl = n if n % 2 == 1 else max(3, n - 1)
    return wl


# ----------------------------
# Base class
# ----------------------------


class ContourSmoother(ABC):
    """
    Base interface for all contour smoothers.

    Each smoother must implement:
        __call__(contour) -> contour
    """

    def __call__(self, contour: Contour) -> Contour:
        xy = _as_xy(contour)
        out_xy = self._smooth_xy(xy)
        # preserve integer contour output as default (OpenCV usually expects int32)
        return _as_contour(out_xy, dtype=np.int32)

    @abstractmethod
    def _smooth_xy(self, xy: np.ndarray) -> np.ndarray:
        pass


class Pipeline(ContourSmoother):
    """Apply multiple smoothers in sequence."""

    def __init__(self, steps: Sequence[ContourSmoother]):
        self.steps = list(steps)

    def _smooth_xy(self, xy: np.ndarray) -> np.ndarray:
        c = _as_contour(xy, dtype=np.float64)  # keep shape consistent
        for step in self.steps:
            c = step(c)
        return _as_xy(c)


# ----------------------------
# 1) PolyDP approximation
# ----------------------------


class PolyDPSmoother(ContourSmoother):
    """
    OpenCV polygon approximation:
        cv2.approxPolyDP

    epsilon: either absolute pixels or ratio of arc length (if use_ratio=True).
    """

    def __init__(
        self, epsilon: float = 0.01, closed: bool = True, use_ratio: bool = True
    ):
        self.epsilon = float(epsilon)
        self.closed = bool(closed)
        self.use_ratio = bool(use_ratio)

    def __call__(self, contour: Contour) -> Contour:
        cnt = np.asarray(contour)
        if cnt.ndim == 2:
            cnt = cnt.reshape((-1, 1, 2))
        if self.use_ratio:
            eps = self.epsilon * cv2.arcLength(cnt.astype(np.float32), self.closed)
        else:
            eps = self.epsilon
        out = cv2.approxPolyDP(cnt, eps, self.closed)
        return out

    def _smooth_xy(self, xy: np.ndarray) -> np.ndarray:
        # Not used because __call__ overridden to use cv2 approx directly.
        return xy


# ----------------------------
# 2) Moving average smoothing (pure NumPy)
# ----------------------------


class MovingAverageSmoother(ContourSmoother):
    """
    Simple moving average on x and y independently.

    window: size of averaging window.
    closed: if True, uses circular padding; if False, uses edge padding.
    """

    def __init__(self, window: int = 9, closed: bool | None = None):
        self.window = int(window)
        self.closed = closed

    def _smooth_xy(self, xy: np.ndarray) -> np.ndarray:
        n = len(xy)
        if n < 3:
            return xy

        wl = _clip_window_length(self.window, n)
        pad = wl // 2
        closed = _is_closed(xy) if self.closed is None else self.closed

        x = xy[:, 0]
        y = xy[:, 1]

        if closed:
            xpad = _wrap_pad_1d(x, pad)
            ypad = _wrap_pad_1d(y, pad)
        else:
            xpad = np.pad(x, (pad, pad), mode="edge")
            ypad = np.pad(y, (pad, pad), mode="edge")

        kernel = np.ones(wl, dtype=np.float64) / wl
        xs = np.convolve(xpad, kernel, mode="valid")
        ys = np.convolve(ypad, kernel, mode="valid")

        out = np.stack([xs, ys], axis=1)
        if closed:
            out = _ensure_closed(out)
        return out


# ----------------------------
# 3) Gaussian smoothing (pure NumPy kernel convolution)
# ----------------------------


class GaussianSmoother(ContourSmoother):
    """
    Gaussian smoothing on x and y independently using a Gaussian kernel.

    sigma: Gaussian std dev
    window: kernel size; if None, computed from sigma (approx 6*sigma rounded odd)
    closed: uses circular padding if True
    """

    def __init__(
        self,
        sigma: float = 2.0,
        window: int | None = None,
        closed: bool | None = None,
    ):
        self.sigma = float(sigma)
        self.window = window
        self.closed = closed

    def _gaussian_kernel(self, wl: int, sigma: float) -> np.ndarray:
        half = wl // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        k = np.exp(-(x**2) / (2 * sigma**2))
        return k

    def _smooth_xy(self, xy: np.ndarray) -> np.ndarray:
        n = len(xy)
        if n < 3:
            return xy

        sigma = self.sigma
        wl = self.window
        if wl is None:
            wl = int(np.ceil(6 * sigma)) | 1  # make odd
        else:
            wl = _clip_window_length(wl, n)

        pad = wl // 2
        closed = _is_closed(xy) if self.closed is None else self.closed

        x = xy[:, 0]
        y = xy[:, 1]

        if closed:
            xpad = _wrap_pad_1d(x, pad)
            ypad = _wrap_pad_1d(y, pad)
        else:
            xpad = np.pad(x, (pad, pad), mode="edge")
            ypad = np.pad(y, (pad, pad), mode="edge")

        kernel = self._gaussian_kernel(wl, sigma)
        kernel /= kernel.sum()

        xs = np.convolve(xpad, kernel, mode="valid")
        ys = np.convolve(ypad, kernel, mode="valid")

        out = np.stack([xs, ys], axis=1)
        if closed:
            out = _ensure_closed(out)
        return out


# ----------------------------
# 4) Savitzky-Golay smoothing (requires scipy)
# ----------------------------


class ContourSmoother:
    """
    Base interface: call with an OpenCV contour and return an OpenCV contour.
    """

    def __call__(self, contour: np.ndarray) -> np.ndarray:
        xy = self._as_xy(contour)
        out_xy = self._smooth_xy(xy)
        return self._as_contour(out_xy, dtype=np.int32)

    def _smooth_xy(self, xy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _as_xy(contour: np.ndarray) -> np.ndarray:
        c = np.asarray(contour)
        # OpenCV contour format: (N,1,2) -> convert to (N,2)
        if c.ndim == 3 and c.shape[1] == 1 and c.shape[2] == 2:
            c = c[:, 0, :]
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError(f"Contour must be (N,1,2) or (N,2); got {c.shape}")
        return c.astype(np.float64, copy=False)

    @staticmethod
    def _as_contour(xy: np.ndarray, dtype=np.int32) -> np.ndarray:
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must be Nx2; got {xy.shape}")
        return xy.astype(dtype).reshape((-1, 1, 2))


class SavitzkyGolaySmoother(ContourSmoother):
    """
    Savitzky–Golay contour smoothing (requires SciPy).

    window_length: odd integer, automatically adjusted to be valid for N
    polyorder: must be < window_length, automatically clamped
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3):
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)

    def _smooth_xy(self, xy: np.ndarray) -> np.ndarray:
        n = len(xy)
        if n < 3:
            return xy

        # --- make window_length valid: odd, >=3, <= n ---
        wl = max(3, self.window_length)
        if wl % 2 == 0:
            wl += 1
        if wl > n:
            wl = n if n % 2 == 1 else max(3, n - 1)

        # --- make polyorder valid: < window_length ---
        po = min(self.polyorder, wl - 1)

        # Your original code, adapted to Nx2 xy
        x = savgol_filter(xy[:, 0], window_length=wl, polyorder=po, mode="interp")
        y = savgol_filter(xy[:, 1], window_length=wl, polyorder=po, mode="interp")

        return np.stack((x, y), axis=1)
