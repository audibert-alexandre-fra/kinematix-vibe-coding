"""Exécute NLF-L (Neural Localizer Fields, Sárándi & Pons-Moll, NeurIPS 2024) sur des vidéos —
dans l'environnement du projet (torch installé avec l'extra `detector`) :

    uv run --extra detector python tools/run_nlf.py data --out weights/nlf_cache

Poids : https://github.com/isarandi/nlf/releases/download/v0.2.2/nlf_l_multi_0.2.2.torchscript
(la v0.3.2 échoue sur CPU : détecteur YOLO interne et ajustement SMPL). Usage NON COMMERCIAL.
Le détecteur intégré ne fonctionnant pas sur CPU, les boîtes viennent de notre RF-DETR + suivi
de la personne principale (models/detector.py) et sont passées à `estimate_smpl_batched`.
Sorties (format commun, models/cached.py) : 24 articulations SMPL non paramétriques + points
du maillage SMPL ajusté (talons, orteils, visage), en m, repère caméra.
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
    p.add_argument("--weights", default="weights/nlf_l_multi_0.2.2.torchscript")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    import torchvision  # noqa: F401  (nécessaire au chargement du modèle TorchScript)

    from kinematix_vibe_coding.models.detector import MainPersonTracker, RFDETRDetector
    from kinematix_vibe_coding.video import iter_frames, read_video_info

    t0 = time.time()
    model = torch.jit.load(a.weights, map_location="cpu").eval()
    detector = RFDETRDetector("medium")
    load_s = time.time() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vidx = list(SMPL_VERTEX_POINTS.values())
    for video in sorted(Path(a.data).rglob("*.mp4")):
        dst = out / f"{video.parent.name}__{video.stem}.npz"
        if dst.exists():
            continue
        info = read_video_info(video)
        w, h = info.width, info.height
        f = math.hypot(w, h)
        K = torch.tensor([[[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]], dtype=torch.float32)
        t1 = time.time()
        tracker = MainPersonTracker((w, h))
        frames, boxes, found = [], [], []
        for fr in iter_frames(video):
            box, ok = tracker.update(detector.detect(fr))
            frames.append(fr); boxes.append(box); found.append(box is not None)
        t_det = time.time() - t1
        first = next(b for b in boxes if b is not None)
        boxes = [b if b is not None else first for b in boxes]
        T = len(frames)
        J3, J2, U = np.zeros((T, 24 + len(vidx), 3)), np.zeros((T, 24 + len(vidx), 2)), np.zeros((T, 24))
        t2 = time.time()
        with torch.inference_mode():
            for s in range(0, T, a.batch):
                e = min(T, s + a.batch)
                x = torch.from_numpy(np.stack(frames[s:e])).permute(0, 3, 1, 2)
                bx = [torch.tensor([[b[0], b[1], b[2] - b[0], b[3] - b[1]]], dtype=torch.float32)
                      for b in boxes[s:e]]
                pred = model.estimate_smpl_batched(x, bx, intrinsic_matrix=K)
                for i in range(e - s):
                    j3 = pred["joints3d_nonparam"][i][0].numpy()
                    v3 = pred["vertices3d"][i][0].numpy()[vidx]
                    J3[s + i] = np.concatenate([j3, v3]) / 1000.0
                    J2[s + i] = np.concatenate([pred["joints2d_nonparam"][i][0].numpy(),
                                                pred["vertices2d"][i][0].numpy()[vidx]])
                    U[s + i] = pred["joint_uncertainties"][i][0].numpy()
                print(f"  {video.name} {e}/{T} ({time.time() - t2:.0f} s)", flush=True)
        dt = time.time() - t2
        np.savez_compressed(dst, joints3d=J3, joints2d=J2, scores=np.ones(J3.shape[:2]), uncertainties=U,
                            boxes=np.array(boxes, float), frame_valid=np.array(found), coords="camera",
                            fps=info.fps, image_size=np.array([w, h]), focal=f, inference_s=dt + t_det,
                            detection_s=t_det, load_s=load_s)
        print(f"{video}: {T} images, {dt + t_det:.1f} s ({1000 * (dt + t_det) / T:.0f} ms/img)", flush=True)


if __name__ == "__main__":
    main()
