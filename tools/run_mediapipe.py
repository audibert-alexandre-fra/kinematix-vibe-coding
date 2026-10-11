"""Exécute MediaPipe Pose Landmarker (heavy, 33 points BlazePose) sur des vidéos — dans un
environnement séparé (mediapipe impose opencv-contrib, incompatible avec opencv-headless) :

    cd third_party/envs/mediapipe && uv venv -p 3.12 && uv pip install mediapipe
    third_party/envs/mediapipe/.venv/bin/python tools/run_mediapipe.py data --out weights/mediapipe_cache

Modèle : weights/mediapipe/pose_landmarker_heavy.task (Apache-2.0, usage commercial autorisé).
Personne principale : si `output/instanthmr/<sous-dossier>/<vidéo>/pose_instanthmr.npz` existe, ses
boîtes (détecteur RF-DETR + suivi, models/detector.py) servent à recadrer l'image (carré 1,3 x la
boîte, 512 px) avant MediaPipe (mode IMAGE) : sinon MediaPipe peut suivre une autre personne
(gait1/cam0 : personne au premier plan) et perd en précision sur les sujets petits dans l'image.
Sorties (format commun des caches, voir models/cached.py) : joints3d = "world landmarks" (m,
origine au milieu des hanches, axes de l'image), joints2d (px), scores = visibilité.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

CROP = 512
MODEL = Path("weights/mediapipe/pose_landmarker_heavy.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/"
             "float16/latest/pose_landmarker_heavy.task")


def box_of(uv: np.ndarray) -> np.ndarray:
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def select_person(poses_uv: list[np.ndarray], prev: np.ndarray | None, w: int, h: int) -> int:
    """Personne principale : continuité (IoU avec la boîte précédente), sinon grande et centrée."""
    boxes = [box_of(p) for p in poses_uv]
    area = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
    if prev is not None:
        def iou(b):
            ix = max(0.0, min(b[2], prev[2]) - max(b[0], prev[0]))
            iy = max(0.0, min(b[3], prev[3]) - max(b[1], prev[1]))
            inter = ix * iy
            return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (prev[2] - prev[0]) * (prev[3] - prev[1]) - inter)
        ious = np.array([iou(b) for b in boxes])
        if ious.max() > 0.2:
            return int(np.argmax(ious))
    centre = np.array([1 - np.hypot((b[0] + b[2]) / 2 / w - 0.5, (b[1] + b[3]) / 2 / h - 0.5) for b in boxes])
    return int(np.argmax(area * centre))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--out", default="weights/mediapipe_cache")
    p.add_argument("--model", default=str(MODEL))
    p.add_argument("--boxes", default="output/instanthmr", help="dossier des pose_instanthmr.npz (boîtes)")
    a = p.parse_args()
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    model = Path(a.model)
    if not model.exists():
        import urllib.request

        model.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for video in sorted(Path(a.data).rglob("*.mp4")):
        dst = out / f"{video.parent.name}__{video.stem}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        bfile = Path(a.boxes) / video.parent.name / video.stem / "pose_instanthmr.npz"
        ref_boxes, det_s = None, 0.0
        if bfile.exists():  # temps de détection RF-DETR compté dans le temps total
            ref = np.load(bfile)
            ref_boxes = ref["bbox"]
            det_s = dict(zip(ref["timings_keys"].tolist(), ref["timings_values"].tolist())).get("detection_s", 0.0)
        mode = vision.RunningMode.IMAGE if ref_boxes is not None else vision.RunningMode.VIDEO
        opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                            running_mode=mode, num_poses=1 if ref_boxes is not None else 4,
                                            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        landmarker = vision.PoseLandmarker.create_from_options(opts)
        load_s = time.time() - t0
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        J3, J2, S, ok_list = [], [], [], []
        t1 = time.time()
        t = 0
        prev = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ref_boxes is not None:
                bx = ref_boxes[min(t, len(ref_boxes) - 1)]
                side = 1.3 * max(bx[2] - bx[0], bx[3] - bx[1])
                x0, y0 = (bx[0] + bx[2]) / 2 - side / 2, (bx[1] + bx[3]) / 2 - side / 2
                sc = CROP / side
                M = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]])
                crop = cv2.warpAffine(rgb, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
                cw, ch, ox, oy = side, side, x0, y0
            else:
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                  int(round(1000 * t / fps)))
                cw, ch, ox, oy = w, h, 0.0, 0.0
            t += 1
            if not res.pose_landmarks:
                J3.append(np.full((33, 3), np.nan)); J2.append(np.full((33, 2), np.nan)); S.append(np.zeros(33))
                ok_list.append(False)
                continue
            uvs = [np.array([[ox + q.x * cw, oy + q.y * ch] for q in lm]) for lm in res.pose_landmarks]
            k = select_person(uvs, prev, w, h)
            lm, wl = res.pose_landmarks[k], res.pose_world_landmarks[k]
            prev = box_of(uvs[k])
            J2.append(uvs[k])
            J3.append([[q.x, q.y, q.z] for q in wl])
            S.append([q.visibility if q.visibility is not None else 1.0 for q in lm])
            ok_list.append(True)
        dt = time.time() - t1 + det_s
        landmarker.close()
        J2a = np.array(J2, float)
        boxes = np.stack([np.nanmin(J2a[..., 0], 1), np.nanmin(J2a[..., 1], 1),
                          np.nanmax(J2a[..., 0], 1), np.nanmax(J2a[..., 1], 1)], 1)
        np.savez_compressed(dst, joints3d=np.array(J3, float), joints2d=J2a, scores=np.array(S, float),
                            boxes=boxes, frame_valid=np.array(ok_list), coords="root_relative", fps=fps,
                            image_size=np.array([w, h]), focal=float(np.hypot(w, h)), inference_s=dt, load_s=load_s)
        print(f"{video}: {t} images, {dt:.1f} s ({1000 * dt / t:.0f} ms/img)", flush=True)


if __name__ == "__main__":
    main()
