# CLAUDE.md — kinematix-vibe-coding

## Objectif
Système de **capture du mouvement monoculaire** : une seule vidéo en entrée →
cinématique articulaire d'un modèle humain **MuJoCo** (angles articulaires dans le temps),
avec projection du squelette sur la vidéo.

Pipeline :
```
vidéo ──► video/ (lecture, métadonnées, intrinsèques par défaut)
      ──► models/ (détection personne + estimation pose 3D / mesh, backend interchangeable)
      ──► calibration/ (marqueurs virtuels, mise à l'échelle par la taille connue,
                        calibration automatique MuJoCo des segments, plan du sol)
      ──► inverse_kinematix/ (IK MuJoCo frame par frame + contraintes de contact
                              anti-glissement des pieds + lissage)
      ──► utils/ (squelettes, filtres, rendu overlay, métriques)
      ──► output/<backend>/<sous-dossier>/<video>/ (qpos, angles .csv, overlay .mp4, timings)
```

## Contraintes
- Python **3.12**, gestionnaire de paquets **uv** (`uv sync`, `uv run ...`).
- Après chaque élément de la TODO : tests (`uv run pytest`) + vérification de non-régression
  + commit descriptif + push. CI GitHub Actions (`.github/workflows/ci.yml`) : ruff + pytest.
- Le modèle de pose final doit **autoriser l'usage commercial** (MonocularBiomechanics = AGPL-3.0
  + MeTRAbs : uniquement comme référence).
- Taille du sujet connue (une seule personne par vidéo) → sert à l'étalonnage géométrique.
- **Pas de glissement des pieds** sur le sol (contraintes de contact dans l'IK).
- Pipeline **< 5 min par vidéo** (machine de dev : Apple M3 Pro, CPU/CoreML, pas de CUDA).
- Rapports dans `rapport/` (un `rapport_<repo>.txt` par dépôt testé + compte rendu final).

## Arborescence
```
data/                              vidéos d'entrée ; même sous-dossier = même mouvement, caméras différentes
src/kinematix_vibe_coding/         (nom avec "_" : un package Python ne peut pas contenir "-")
    models/                        backends de pose (InstantHMR ONNX, SAM3D Body, ...) + registre
    calibration/                   marqueurs, mise à l'échelle, calibration auto MuJoCo, sol
    inverse_kinematix/             IK MuJoCo, contacts pieds
    video/                         I/O vidéo
    utils/                         squelettes, filtrage, rendu, métriques
    cli.py                         point d'entrée `uv run kinematix ...`
tests/                             tests unitaires (rapides, sans poids) + tests `slow`
rapport/                           rapports texte
output/<backend>/                  résultats sur data/ par modèle de pose (versionnés)
```

## Commandes
```bash
uv sync                                   # installer
uv run pytest                             # tests rapides (CI)
uv run pytest -m slow                     # tests avec poids/vidéos réelles
uv run ruff check .                       # lint
uv run kinematix run data/squat1/raw-video-cam0.mp4 --height 1.80 --out output
uv run kinematix batch data --out output  # toutes les vidéos -> output/<backend>/...
uv run kinematix batch data --backend metrabs   # référence (cache MeTRAbs : tools/run_metrabs.py)
uv run kinematix evaluate output          # cohérence multi-vues + output/comparaison_modeles.md
```

## Conventions
- Monde MuJoCo : Z vers le haut, unités SI (m, rad). Caméra : OpenCV (X droite, Y bas, Z avant).
- Chaque backend de pose renvoie une `PoseSequence` (keypoints 3D caméra, 2D pixels, confiances,
  noms de joints) → indépendant du modèle en aval.
- Les poids des modèles sont téléchargés dans `weights/` (ignoré par git).
- Dépôts tiers (MonocularBiomechanics, InstantHMR, SAM3DFast) clonés dans `third_party/` (ignoré).
- Paramètres par défaut réglés par cohérence multi-vues : reprojection 2D poids 10, filtre 6 Hz,
  poids de contact 50 (compromis anti-glissement / précision), tapis roulant si bassin immobile.

## TODO (mise à jour à chaque étape terminée)
- [x] 0. Exploration : dépôts de référence, licences, vidéos de `data/`, machine (M3 Pro, pas de CUDA).
- [x] 1. Squelette du projet uv (Python 3.12), arborescence, CLAUDE.md, .gitignore.
- [x] 2. CI GitHub Actions (ruff + pytest) + test de fumée.
- [x] 3. MonocularBiomechanics : installation (env séparé), exécution sur une vidéo, mesure
        des temps CPU / estimation GPU → `rapport/rapport_MonocularBiomechanics.txt`.
- [x] 4. InstantHMR : installation, poids ONNX, exécution, temps CPU / CoreML / estimation GPU
        → `rapport/rapport_InstantHMR.txt`.
- [x] 5. SAM3DFast (SAM 3D Body) : installation, poids (HF gated), temps
        → `rapport/rapport_SAM3DFast.txt`.
- [x] 6. `video/` : lecteur vidéo, métadonnées, intrinsèques par défaut + tests.
- [x] 7. `models/` : interface `PoseBackend` + `PoseSequence`, détecteur personne (licence commerciale),
        backend InstantHMR (+ SAM3D si dispo), suivi de la personne principale + tests.
- [x] 8. `utils/` : définitions squelettes (MHR70), filtres temporels, rendu overlay + tests.
- [x] 9. Modèle humain MuJoCo paramétrique (généré par MjSpec, licence libre) avec sites marqueurs.
- [x] 10. `calibration/` : marqueurs virtuels depuis keypoints, mise à l'échelle par taille connue,
        estimation du plan du sol, calibration automatique des segments par optimisation MuJoCo + tests.
- [x] 11. `inverse_kinematix/` : IK MuJoCo (Levenberg-Marquardt, limites articulaires, warm-start),
        détection des contacts pieds, contraintes anti-glissement + tests (glissement ≈ 0).
- [x] 12. CLI `kinematix run/batch` : pipeline complet, export qpos/angles CSV/overlay, timings (< 5 min).
- [x] 13. Évaluation multi-vues : comparaison des angles articulaires entre caméras d'un même
        mouvement (synchronisation temporelle, RMSE, corrélation).
- [ ] 14. `output/` : exécution sur toutes les vidéos de `data/` + `rapport/compte_rendu_performances.txt`.
- [ ] 15. Projection du squelette sur toutes les vidéos (overlays) et livraison.
