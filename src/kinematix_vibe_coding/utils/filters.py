"""Filtres temporels."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def interpolate_invalid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Remplace les images invalides (axe 0) par interpolation linéaire / extrapolation constante."""
    x = np.array(x, float)
    valid = np.asarray(valid, bool)
    if valid.all() or not valid.any():
        return x
    t = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[~valid, c] = np.interp(t[~valid], t[valid], flat[valid, c])
    return flat.reshape(x.shape)


def lowpass(x: np.ndarray, fps: float, cutoff: float, order: int = 2) -> np.ndarray:
    """Butterworth passe-bas à phase nulle le long de l'axe 0 (sans effet si trop peu d'images)."""
    x = np.asarray(x, float)
    if cutoff <= 0 or cutoff >= fps / 2:
        return x.copy()
    b, a = butter(order, cutoff / (fps / 2))
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x.copy()
    return filtfilt(b, a, x, axis=0)


def smooth_bool(mask: np.ndarray, min_len: int = 3) -> np.ndarray:
    """Comble les trous intérieurs < `min_len` puis supprime les îlots vrais < `min_len`."""
    m = np.asarray(mask, bool).copy()
    for a, b in segments(~m):
        if a > 0 and b < len(m) and b - a < min_len:
            m[a:b] = True
    for a, b in segments(m):
        if b - a < min_len:
            m[a:b] = False
    return m


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Intervalles [début, fin) où `mask` est vrai."""
    out, start = [], None
    for i, v in enumerate(np.append(np.asarray(mask, bool), False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out
