"""Exécute NLF-L (Neural Localizer Fields, Sárándi & Pons-Moll, NeurIPS 2024) sur des vidéos —
dans l'environnement du projet (torch installé avec l'extra `detector`) :

    uv run --extra detector python tools/run_nlf.py data --out weights/nlf_cache

Fichiers (usage NON COMMERCIAL) :
  weights/nlf_l_multi_0.3.2.torchscript
      https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.patch4.torchscript
  weights/nlf_data/nlf_data_files/canonical_verts/smpl.npy   (coordonnées canoniques des sommets SMPL)
      extrait de https://github.com/isarandi/nlf/releases/download/v0.2.0/nlf_data_files.zip

Contournements CPU (le modèle TorchScript est prévu pour CUDA) :
  - le réseau est exporté en float16 : sur CPU il produit des NaN sur certaines images (et
    l'ajustement SMPL échoue ensuite, SVD non convergente). On passe tout en float32
    (`model.float()`, `crop_model.backbone_dtype = float32`, images déjà linéarisées en float32) ;
  - le détecteur YOLO intégré ne fonctionne pas sur CPU : boîtes RF-DETR + suivi de la personne
    principale (celles d'output/instanthmr si disponibles, sinon models/detector.py) ;
  - pas d'ajustement SMPL : NLF prédit directement (voie non paramétrique) les 24 articulations
    SMPL et 11 sommets du maillage (talons, orteils, visage) à partir de leurs points canoniques.
Sorties (format commun, models/cached.py) : m, repère caméra.
"""

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np

# sommets SMPL (indices usuels, cf. SPIN / VIBE : vertex_ids['smplh'])
SMPL_VERTEX_POINTS = {"nose": 332, "right_eye": 6260, "left_eye": 2800, "right_ear": 4071, "left_ear": 583,
                      "left_big_toe": 3216, "left_small_toe": 3226, "left_heel": 3387,
                      "right_big_toe": 6617, "right_small_toe": 6624, "right_heel": 6787}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--out", default="weights/nlf_cache")
    p.add_argument("--weights", default="weights/nlf_l_multi_0.3.2.torchscript")
    p.add_argument("--canonical-verts", default="weights/nlf_data/nlf_data_files/canonical_verts/smpl.npy")
    p.add_argument("--boxes", default="output/instanthmr", help="dossier des pose_instanthmr.npz (boîtes)")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    import torchvision  # noqa: F401  (nécessaire au chargement du modèle TorchScript)

    from kinematix_vibe_coding.models.detector import MainPersonTracker, RFDETRDetector
    from kinematix_vibe_coding.video import iter_frames, read_video_info

    t0 = time.time()
    model = torch.jit.load(a.weights, map_location="cpu").float().eval()
    model.crop_model.backbone_dtype = torch.float32
    cano_all = model.cano_all["smpl"]
    cano_verts = torch.tensor(np.load(a.canonical_verts), dtype=torch.float32)
    cano = torch.cat([cano_all[-24:], cano_verts[list(SMPL_VERTEX_POINTS.values())]])
    weights = {k: v.float() for k, v in model.get_weights_for_canonical_points(cano).items()}
    detector = None
    load_s = time.time() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    J = len(cano)
    for video in sorted(Path(a.data).rglob("*.mp4")):
        dst = out / f"{video.parent.name}__{video.stem}.npz"
        if dst.exists():
            continue
        info = read_video_info(video)
        w, h = info.width, info.height
        f = math.hypot(w, h)
        K = torch.tensor([[[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]], dtype=torch.float32)
        t1 = time.time()
        frames = list(iter_frames(video))
        bfile = Path(a.boxes) / video.parent.name / video.stem / "pose_instanthmr.npz"
        if bfile.exists():  # même détection que les autres modèles ; temps de détection compté
            ref = np.load(bfile)
            boxes = list(ref["bbox"])
            found = list(ref["valid"].astype(bool))
            t_det = dict(zip(ref["timings_keys"].tolist(), ref["timings_values"].tolist())).get("detection_s", 0.0)
        else:
            detector = detector or RFDETRDetector("medium")
            tracker = MainPersonTracker((w, h))
            boxes, found = [], []
            for fr in frames:
                box, _ = tracker.update(detector.detect(fr))
                boxes.append(box); found.append(box is not None)
            first = next(b for b in boxes if b is not None)
            boxes = [b if b is not None else first for b in boxes]
            t_det = time.time() - t1
        T = len(frames)
        J3, J2, U = np.full((T, J, 3), np.nan), np.full((T, J, 2), np.nan), np.zeros((T, J))
        t2 = time.time()
        with torch.inference_mode():
            for s in range(0, T, a.batch):
                e = min(T, s + a.batch)
                x = (torch.from_numpy(np.stack(frames[s:e])).permute(0, 3, 1, 2).float() / 255.0) ** 2.2
                bx = [torch.tensor([[b[0], b[1], b[2] - b[0], b[3] - b[1], 1.0]], dtype=torch.float32)
                      for b in boxes[s:e]]
                pred = model._estimate_poses_batched(x, bx, weights, K, None, None, None, 55.0, 64, 1, 1, 25.0,
                                                     False, True, 0.0)
                for i in range(e - s):
                    J3[s + i] = pred["poses3d"][i][0].numpy() / 1000.0
                    J2[s + i] = pred["poses2d"][i][0].numpy()
                    U[s + i] = pred["uncertainties"][i][0].numpy()
                print(f"  {video.name} {e}/{T} ({time.time() - t2:.0f} s)", flush=True)
        dt = time.time() - t2
        ok = np.array(found) & np.isfinite(J3).all(axis=(1, 2))
        np.savez_compressed(dst, joints3d=J3, joints2d=J2, scores=np.ones((T, J)), uncertainties=U,
                            boxes=np.array(boxes, float), frame_valid=ok, coords="camera", fps=info.fps,
                            image_size=np.array([w, h]), focal=f, inference_s=dt + t_det, detection_s=t_det,
                            load_s=load_s)
        print(f"{video}: {T} images, {dt + t_det:.1f} s ({1000 * (dt + t_det) / T:.0f} ms/img), "
              f"{(~ok).sum()} images invalides", flush=True)


if __name__ == "__main__":
    main()
