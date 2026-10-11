"""Évaluation de la précision par cohérence multi-vues.

Les vidéos d'un même sous-dossier de `data/` filment le même mouvement depuis des caméras
différentes : les angles articulaires estimés doivent être identiques. Pour chaque paire de
vues on synchronise les signaux (décalage maximisant la corrélation, les vidéos n'étant pas
synchronisées et pouvant avoir des fps différents), puis on calcule RMSE et corrélation.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from ..inverse_kinematix.angles import load_angles_csv

EVAL_ANGLES = [
    "hip_flexion_l", "hip_flexion_r", "knee_flexion_l", "knee_flexion_r", "ankle_flexion_l", "ankle_flexion_r",
    "hip_adduction_l", "hip_adduction_r", "lumbar_flexion", "elbow_flexion_l", "elbow_flexion_r",
    "shoulder_elevation_l", "shoulder_elevation_r",
]
SYNC_ANGLES = ["knee_flexion_l", "knee_flexion_r", "hip_flexion_l", "hip_flexion_r", "shoulder_elevation_l",
               "shoulder_elevation_r", "elbow_flexion_l", "elbow_flexion_r"]
RATE = 100.0  # Hz, ré-échantillonnage commun


def _resample(t: np.ndarray, x: np.ndarray, rate: float = RATE) -> np.ndarray:
    tt = np.arange(t[0], t[-1], 1.0 / rate)
    return np.interp(tt, t, x)


def _sync_signal(angles: dict[str, np.ndarray], t: np.ndarray) -> np.ndarray:
    sig = []
    for n in SYNC_ANGLES:
        x = _resample(t, angles[n])
        sig.append((x - x.mean()) / (x.std() + 1e-6) * min(1.0, x.std() / 5.0))  # ignore les angles quasi fixes
    return np.sum(sig, axis=0)


def find_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Décalage d (échantillons) tel que a[i] ~ b[i - d] ; renvoie (d, corrélation)."""
    best, best_c = 0, -np.inf
    for d in range(-max_lag, max_lag + 1):
        if d >= 0:
            x, y = a[d:], b[: len(b)]
        else:
            x, y = a, b[-d:]
        n = min(len(x), len(y))
        if n < 0.5 * min(len(a), len(b)):
            continue
        c = np.corrcoef(x[:n], y[:n])[0, 1]
        if c > best_c:
            best, best_c = d, c
    return best, float(best_c)


def compare_views(a: tuple[np.ndarray, dict], b: tuple[np.ndarray, dict], max_lag_s: float = 3.0) -> dict:
    (ta, A), (tb, B) = a, b
    lag, corr = find_lag(_sync_signal(A, ta), _sync_signal(B, tb), int(max_lag_s * RATE))
    out = {"lag_s": lag / RATE, "sync_correlation": corr, "angles": {}}
    for n in EVAL_ANGLES:
        x, y = _resample(ta, A[n]), _resample(tb, B[n])
        if lag >= 0:
            x = x[lag:]
        else:
            y = y[-lag:]
        k = min(len(x), len(y))
        x, y = x[:k], y[:k]
        out["angles"][n] = {
            "rmse_deg": float(np.sqrt(np.mean((x - y) ** 2))),
            "bias_deg": float(np.mean(x - y)),
            "corr": float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 and y.std() > 1e-6 else float("nan"),
            "range_deg": float(max(np.ptp(x), np.ptp(y))),
        }
    rm = [v["rmse_deg"] for v in out["angles"].values()]
    out["mean_rmse_deg"] = float(np.mean(rm))
    return out


def evaluate_output(out_root: Path, verbose: bool = True) -> dict:
    """Évalue chaque sous-dossier de `out_root` contenant >= 2 vues."""
    groups: dict[str, list[Path]] = {}
    for csv in sorted(out_root.rglob("joint_angles.csv")):
        groups.setdefault(str(csv.parent.parent.relative_to(out_root)), []).append(csv)
    report = {}
    for g, csvs in groups.items():
        if len(csvs) < 2:
            continue
        views = {c.parent.name: load_angles_csv(c) for c in csvs}
        pairs = {}
        for (na, va), (nb, vb) in itertools.combinations(views.items(), 2):
            pairs[f"{na} vs {nb}"] = compare_views(va, vb)
        per_angle = {n: float(np.mean([p["angles"][n]["rmse_deg"] for p in pairs.values()])) for n in EVAL_ANGLES}
        report[g] = {"views": list(views), "pairs": pairs, "mean_rmse_per_angle_deg": per_angle,
                     "mean_rmse_deg": float(np.mean(list(per_angle.values())))}
        if verbose:
            print(f"[{g}] {len(views)} vues, RMSE moyenne inter-vues = {report[g]['mean_rmse_deg']:.1f}°")
            for n, v in per_angle.items():
                print(f"    {n:22s} {v:6.1f}°")
        _plot_group(out_root / g, views)
    with open(out_root / "evaluation_multivues.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def _plot_group(folder: Path, views: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["hip_flexion_r", "knee_flexion_r", "ankle_flexion_r", "hip_flexion_l", "knee_flexion_l",
             "shoulder_elevation_r", "lumbar_flexion", "elbow_flexion_r", "shoulder_elevation_l"]
    ref_name, (t0, A0) = next(iter(views.items()))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for name, (t, A) in views.items():
        lag = 0.0
        if name != ref_name:
            lag = compare_views((t0, A0), (t, A))["lag_s"]
        for ax, n in zip(axes.flat, names, strict=True):
            ax.plot(t + lag, A[n], label=name)
            ax.set_title(n)
    for ax in axes[-1]:
        ax.set_xlabel("temps (s, synchronisé)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{folder.name} : angles (°) selon chaque caméra")
    fig.tight_layout()
    fig.savefig(folder / "comparaison_multivues.png", dpi=90)
    plt.close(fig)
