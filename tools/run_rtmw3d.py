"""Exécute MMPose RTMW3D-L (3D corps entier, 133 points COCO-WholeBody) sur des vidéos — dans
l'environnement MMPose (Python 3.10, torch 2.1, mmcv 2.1 compilé, mmdet 3.2, mmpose 1.3) :

    third_party/envs/mmpose/.venv/bin/python tools/run_rtmw3d.py data --out weights/rtmw3d_cache

Personne principale : boîtes RF-DETR + suivi de `output/instanthmr/<sous-dossier>/<vidéo>/pose_instanthmr.npz`
si disponibles (même détection pour tous les modèles), sinon RTMDet-m personne (MMDetection)
+ continuité IoU (RTMDet seul choisit parfois une personne coupée au premier plan, gait1/cam0). Pose : projects/rtmpose3d (RTMW3D-L, Cocktail14).
RTMW3D prédit des coordonnées pixels (x, y) et une profondeur RELATIVE au bassin z (m) ; la
profondeur absolue est reconstruite dans models/cached.py (longueurs de segments).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MMPOSE = ROOT / "third_party/envs/mmpose/mmpose"
DET_CFG = MMPOSE / "projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT = ROOT / "weights/mmpose/rtmdet_m_person.pth"    # rtmdet_m_8xb32-100e_coco-obj365-person-235e8209
POSE_CFG = MMPOSE / "projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"
POSE_CKPT = ROOT / "weights/mmpose/rtmw3d-l_cocktail14.pth"  # rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626
# téléchargement (curl) : https://download.openmmlab.com/mmpose/v1/projects/rtmpose/<rtmdet...>.pth
#                         https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/<rtmw3d...>.pth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--out", default="weights/rtmw3d_cache")
    p.add_argument("--device", default="cpu")
    p.add_argument("--boxes", default=str(ROOT / "output/instanthmr"), help="dossier des pose_instanthmr.npz")
    a = p.parse_args()
    sys.path.insert(0, str(MMPOSE / "projects/rtmpose3d"))
    import rtmpose3d  # noqa: F401  (enregistre les modules RTMW3D)
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import adapt_mmdet_pipeline

    t0 = time.time()
    det = init_detector(str(DET_CFG), str(DET_CKPT), device=a.device)
    det.cfg = adapt_mmdet_pipeline(det.cfg)
    pose = init_model(str(POSE_CFG), str(POSE_CKPT), device=a.device)
    load_s = time.time() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for video in sorted(Path(a.data).rglob("*.mp4")):
        dst = out / f"{video.parent.name}__{video.stem}.npz"
        if dst.exists():
            continue
        bfile = Path(a.boxes) / video.parent.name / video.stem / "pose_instanthmr.npz"
        ref_boxes, det_s = None, 0.0
        if bfile.exists():
            ref = np.load(bfile)
            ref_boxes = ref["bbox"]
            det_s = dict(zip(ref["timings_keys"].tolist(), ref["timings_values"].tolist())).get("detection_s", 0.0)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        J2, Z, S, B, V = [], [], [], [], []
        prev = None
        t1, n = time.time(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            h, w = frame.shape[:2]
            if ref_boxes is not None:
                bx = ref_boxes[min(n - 1, len(ref_boxes) - 1)].astype(np.float32)
                res = inference_topdown(pose, frame, bx[None])[0].pred_instances
                kp = np.squeeze(res.keypoints, axis=tuple(range(res.keypoints.ndim - 2)))
                uv = np.squeeze(res.transformed_keypoints, axis=tuple(range(res.transformed_keypoints.ndim - 2)))
                J2.append(uv); Z.append(kp[:, 2] - kp[[11, 12], 2].mean())
                S.append(np.asarray(res.keypoint_scores).reshape(-1)); B.append(bx); V.append(True)
                continue
            inst = inference_detector(det, frame).pred_instances.cpu().numpy()
            keep = (inst.labels == 0) & (inst.scores > 0.3)
            boxes, scores = inst.bboxes[keep], inst.scores[keep]
            if len(boxes) == 0:
                J2.append(np.full((133, 2), np.nan)); Z.append(np.full(133, np.nan)); S.append(np.zeros(133))
                B.append(np.full(4, np.nan)); V.append(False)
                continue
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if prev is not None:  # continuité : IoU avec la boîte précédente
                ix = np.clip(np.minimum(boxes[:, 2], prev[2]) - np.maximum(boxes[:, 0], prev[0]), 0, None)
                iy = np.clip(np.minimum(boxes[:, 3], prev[3]) - np.maximum(boxes[:, 1], prev[1]), 0, None)
                inter = ix * iy
                iou = inter / (area + (prev[2] - prev[0]) * (prev[3] - prev[1]) - inter)
                k = int(np.argmax(iou + 1e-9 * area))
                if iou[k] < 0.2:
                    k = int(np.argmax(area * scores))
            else:
                k = int(np.argmax(area * scores))
            prev = boxes[k]
            res = inference_topdown(pose, frame, boxes[k:k + 1])[0].pred_instances
            kp = np.squeeze(res.keypoints, axis=tuple(range(res.keypoints.ndim - 2)))     # (133, 3) caméra
            uv = np.squeeze(res.transformed_keypoints, axis=tuple(range(res.transformed_keypoints.ndim - 2)))
            sc = np.asarray(res.keypoint_scores).reshape(-1)
            root_z = kp[[11, 12], 2].mean()
            J2.append(uv); Z.append(kp[:, 2] - root_z); S.append(sc); B.append(boxes[k]); V.append(True)
        dt = time.time() - t1 + det_s
        J2a, Za = np.array(J2, float), np.array(Z, float)
        j3 = np.concatenate([np.zeros_like(J2a), Za[..., None]], -1)   # seule la profondeur relative est utile
        np.savez_compressed(dst, joints3d=j3, joints2d=J2a, scores=np.array(S, float), boxes=np.array(B, float),
                            frame_valid=np.array(V), coords="relative_depth", fps=fps, image_size=np.array([w, h]),
                            focal=float(np.hypot(w, h)), inference_s=dt, load_s=load_s)
        print(f"{video}: {n} images, {dt:.1f} s ({1000 * dt / n:.0f} ms/img)", flush=True)


if __name__ == "__main__":
    main()
