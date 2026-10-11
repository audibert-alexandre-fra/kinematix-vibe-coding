"""Exécute MeTRAbs-ACAE-L (squelette bml_movi_87) sur des vidéos — dans l'environnement de
MonocularBiomechanics (TensorFlow). Usage (depuis third_party/MonocularBiomechanics) :

    uv run python ../../tools/run_metrabs.py ../../data --out ../../weights/metrabs_cache

Intrinsèques passées au modèle : focale = diagonale de l'image, centre optique au centre
(identiques au backend InstantHMR, pour une comparaison équitable). Sorties : un .npz par vidéo
(<sous-dossier>__<nom>.npz) avec poses3d (m, repère caméra), poses2d (px), boxes, fps, temps.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    import tensorflow as tf
    from monocular_demos.utils import load_metrabs

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = load_metrabs()
    load_s = time.time() - t0
    data = Path(a.data)
    for video in sorted(data.rglob("*.mp4")):
        dst = out / f"{video.parent.name}__{video.stem}.npz"
        if dst.exists():
            continue
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        h, w = frames[0].shape[:2]
        f = math.hypot(w, h)
        K = tf.constant([[[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]], tf.float32)
        t1 = time.time()
        P3, P2, B = [], [], []
        for s in range(0, len(frames), a.batch):
            pred = model.detect_poses_batched(tf.constant(np.stack(frames[s:s + a.batch])), intrinsic_matrix=K,
                                              skeleton="bml_movi_87")
            for b, p3, p2 in zip(pred["boxes"], pred["poses3d"], pred["poses2d"]):
                b, p3, p2 = b.numpy(), p3.numpy(), p2.numpy()
                if len(b) == 0:
                    B.append(np.zeros(5)); P3.append(np.full((87, 3), np.nan)); P2.append(np.full((87, 2), np.nan))
                    continue
                k = int(np.argmax((b[:, 2] * b[:, 3]) * b[:, 4]))  # plus grande personne x score
                B.append(b[k]); P3.append(p3[k] / 1000.0); P2.append(p2[k])
        dt = time.time() - t1
        np.savez_compressed(dst, poses3d=np.array(P3), poses2d=np.array(P2), boxes=np.array(B), fps=fps,
                            image_size=np.array([w, h]), focal=f, inference_s=dt, load_s=load_s)
        print(f"{video}: {len(frames)} images, {dt:.1f} s ({1000 * dt / len(frames):.0f} ms/img)", flush=True)


if __name__ == "__main__":
    main()
