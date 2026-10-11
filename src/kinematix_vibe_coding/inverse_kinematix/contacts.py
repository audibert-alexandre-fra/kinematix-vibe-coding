"""Contacts pied-sol et contraintes anti-glissement.

1. Détection : un point du pied (talon, orteils) est en contact s'il est proche du sol et
   quasi immobile horizontalement (par rapport au sol, ou au tapis roulant).
2. Tapis roulant : si les pieds en appui reculent tous à vitesse constante, cette vitesse
   (v_tapis) est estimée et le contact est défini dans le repère du tapis.
3. Ancrage : pendant chaque phase de contact, la cible du point est fixe (ou suit le tapis)
   et posée au sol -> l'IK ne peut plus faire glisser le pied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.filters import lowpass, segments, smooth_bool


@dataclass
class ContactConfig:
    height_threshold: float = 0.05     # m au-dessus de la hauteur "pied à plat"
    speed_threshold: float = 0.5       # m/s (relatif au sol / tapis)
    stance_margin: float = 0.03        # m : pied d'appui = pied le plus bas avec cette marge
    min_frames: int = 3
    treadmill_min_speed: float = 0.2   # m/s : en-dessous, sol fixe
    weight: float = 50.0               # poids des cibles de contact dans l'IK


@dataclass
class ContactResult:
    mask: np.ndarray        # (T, F) contacts
    belt_velocity: np.ndarray  # (2,) vitesse horizontale du sol (0 hors tapis roulant)
    targets: np.ndarray     # (T, F, 3) cibles (NaN hors contact)


def horizontal_velocity(p: np.ndarray, fps: float) -> np.ndarray:
    v = np.gradient(lowpass(p[..., :2], fps, 6.0), axis=0) * fps
    return v


def detect_contacts(foot_pos: np.ndarray, flat_heights: np.ndarray, fps: float,
                    cfg: ContactConfig | None = None, foot_groups: list[list[int]] | None = None) -> ContactResult:
    """`foot_pos` : (T, F, 3) positions monde des sites de pied ; `flat_heights` : (F,) ;
    `foot_groups` : indices des points de chaque pied (ex. [[0, 2, 4], [1, 3, 5]])."""
    cfg = cfg or ContactConfig()
    T, F, _ = foot_pos.shape
    h = foot_pos[..., 2] - flat_heights[None]
    v = horizontal_velocity(foot_pos, fps)
    low = h < cfg.height_threshold
    belt = np.zeros(2)
    if low.sum() > 5:
        cand = v[low]
        med = np.median(cand, axis=0)
        # tapis roulant : vitesse commune des pieds au sol, nettement non nulle
        if np.linalg.norm(med) > cfg.treadmill_min_speed:
            belt = med
    rel_speed = np.linalg.norm(v - belt, axis=-1)
    mask = low & (rel_speed < cfg.speed_threshold)
    if foot_groups is not None:
        # pied d'appui : le pied nettement le plus bas a ses points bas au contact, même si
        # la vitesse estimée (bruitée en profondeur en monoculaire) dépasse le seuil
        foot_h = np.stack([h[:, g].min(1) for g in foot_groups], axis=1)  # (T, nb pieds)
        order = np.sort(foot_h, axis=1)
        lowest = np.argmin(foot_h, axis=1)
        clear = (order[:, 1] - order[:, 0]) > cfg.stance_margin if foot_h.shape[1] > 1 else np.ones(T, bool)
        for k, g in enumerate(foot_groups):
            stance = clear & (lowest == k)
            for f in g:
                near = h[:, f] < foot_h[:, k] + cfg.stance_margin
                mask[:, f] |= stance & low[:, f] & near & (rel_speed[:, f] < 2 * cfg.speed_threshold)
    mask = np.stack([smooth_bool(mask[:, f], cfg.min_frames) for f in range(F)], axis=1)

    targets = np.full((T, F, 3), np.nan)
    t = np.arange(T) / fps
    for f in range(F):
        for a, b in segments(mask[:, f]):
            # position "dans le repère du sol" (compense le tapis) puis ancre médiane
            rel = foot_pos[a:b, f, :2] - np.outer(t[a:b] - t[a], belt)
            anchor = np.median(rel, axis=0)
            targets[a:b, f, :2] = anchor + np.outer(t[a:b] - t[a], belt)
            targets[a:b, f, 2] = flat_heights[f]
    return ContactResult(mask=mask, belt_velocity=belt, targets=targets)


def contact_extras(contacts: ContactResult, site_ids: list[int], weight: float) -> list[list]:
    """Liste par image de contraintes (site, cible, poids, axes) pour `MarkerIK.solve`."""
    T, F = contacts.mask.shape
    extras: list[list] = [[] for _ in range(T)]
    xyz = np.array([0, 1, 2])
    for f in range(F):
        for a, b in segments(contacts.mask[:, f]):
            for t in range(a, b):
                ramp = 0.5 if (t == a or t == b - 1) and b - a > 2 else 1.0
                extras[t].append((site_ids[f], contacts.targets[t, f], weight * ramp, xyz))
    return extras


def foot_sliding(foot_pos: np.ndarray, mask: np.ndarray, belt: np.ndarray, fps: float) -> dict[str, float]:
    """Glissement résiduel pendant les contacts : vitesse horizontale (relative au tapis) et
    dérive totale moyenne par phase de contact."""
    v = np.gradient(foot_pos[..., :2], axis=0) * fps - belt
    speeds, drifts = [], []
    t = np.arange(len(foot_pos)) / fps
    for f in range(foot_pos.shape[1]):
        for a, b in segments(mask[:, f]):
            speeds.append(np.linalg.norm(v[a:b, f], axis=-1))
            rel = foot_pos[a:b, f, :2] - np.outer(t[a:b] - t[a], belt)
            drifts.append(np.linalg.norm(rel - rel[0], axis=-1).max())
    if not speeds:
        return {"sliding_speed_mean_cm_s": 0.0, "sliding_drift_mean_cm": 0.0, "contact_ratio": 0.0}
    return {
        "sliding_speed_mean_cm_s": float(np.mean(np.concatenate(speeds)) * 100),
        "sliding_drift_mean_cm": float(np.mean(drifts) * 100),
        "contact_ratio": float(mask.mean()),
    }
