# kinematix-vibe-coding

Capture du mouvement **monoculaire** : une vidéo → pose 3D (deep learning) → marqueurs virtuels →
étalonnage géométrique (taille connue) + calibration automatique MuJoCo → cinématique inverse MuJoCo
sans glissement des pieds → angles articulaires + squelette projeté sur la vidéo.

```bash
uv sync --group dev --extra detector          # Python 3.12
uv run kinematix run data/squat1/raw-video-cam0.mp4 --height 1.80   # -> output/instanthmr/raw-video-cam0/
uv run kinematix batch data                   # toutes les vidéos (tailles : data/heights.json)
uv run kinematix evaluate output              # cohérence des angles entre caméras, par modèle
uv run pytest                                 # tests
```

Modèles de pose (`--backend`) :

| backend | licence | rôle |
|---|---|---|
| `instanthmr` (défaut) | Apache-2.0 + poids licence SAM (commercial OK) | production, ≈ 10 img/s CPU |
| `sam3d` (SAM 3D Body via SAM3DFast) | licence SAM (commercial OK) | plus précis, GPU NVIDIA + poids HF à accès restreint |
| `metrabs` (MonocularBiomechanics) | **non commercial** | référence de comparaison |

Sorties : `output/<backend>/<mouvement>/<vidéo>/` (`overlay_*.mp4`, `joint_angles.csv`, `qpos.npy`,
`trajectories.npz`, `model.xml`, `summary.json`). Rapports : `rapport/`. Détails : `CLAUDE.md`.
