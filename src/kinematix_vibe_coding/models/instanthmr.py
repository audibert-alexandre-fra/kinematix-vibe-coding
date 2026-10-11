"""Backend InstantHMR (ONNX) : 70 keypoints MHR70 3D + 2D par image.

Code : réimplémentation du pré-traitement d'InstantHMR (Apache-2.0).
Poids : `momolesang/InstantHMR` (licence SAM, usage commercial autorisé).
Détecteur : RF-DETR (Apache-2.0).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from ..utils.skeleton import MHR70_JOINT_NAMES
from ..video import default_intrinsics, iter_frames, read_video_info
from .base import PoseSequence
from .detector import MainPersonTracker, RFDETRDetector

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224
CROP_EXPAND = 1.2
HF_REPO = "momolesang/InstantHMR"
DEFAULT_WEIGHTS = Path("weights/instanthmr.onnx")


def flip_permutation(names: list[str]) -> np.ndarray:
    """Indice du joint symétrique (gauche <-> droite) pour chaque joint."""
    idx = {n: i for i, n in enumerate(names)}
    perm = []
    for n in names:
        if n.startswith("left_"):
            perm.append(idx["right_" + n[5:]])
        elif n.startswith("right_"):
            perm.append(idx["left_" + n[6:]])
        else:
            perm.append(idx[n])
    return np.array(perm)


def ensure_weights(path: str | Path = DEFAULT_WEIGHTS) -> Path:
    path = Path(path)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(HF_REPO, path.name, local_dir=str(path.parent))
    return path


def preprocess(image_rgb: np.ndarray, bbox: np.ndarray, flip: bool = False):
    """Crop carré 1.2x autour de la bbox, normalisation ImageNet, conditionnement CLIFF (pixels).

    Retourne (crop CHW, sq_x1, sq_y1, sq_size, cliff)."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cliff = np.array([2 * cx / w - 1, 2 * cy / h - 1, max(bw, bh) / max(w, h)], dtype=np.float32)
    size = max(bw, bh) * CROP_EXPAND
    sx1, sy1 = cx - size / 2, cy - size / 2
    # transformation affine image -> crop (gère le padding implicitement)
    s = INPUT_SIZE / size
    M = np.array([[s, 0, -sx1 * s], [0, s, -sy1 * s]], dtype=np.float32)
    crop = cv2.warpAffine(image_rgb, M, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if flip:
        crop = crop[:, ::-1]
        cliff[0] = -cliff[0]
    crop = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(crop.transpose(2, 0, 1)), sx1, sy1, size, cliff


class InstantHMRBackend:
    name = "instanthmr"
    license = "code Apache-2.0 ; poids licence SAM (usage commercial autorisé) ; détecteur RF-DETR Apache-2.0"

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu", detector_variant: str = "medium",
                 flip_tta: bool = True, batch_size: int = 16):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device == "coreml":
            providers.insert(0, "CoreMLExecutionProvider")
        elif device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(ensure_weights(weights)), sess_options=opts, providers=providers)
        self.detector_variant = detector_variant
        self._detector: RFDETRDetector | None = None
        self.flip_tta = flip_tta
        self.batch_size = batch_size
        self.flip_perm = flip_permutation(MHR70_JOINT_NAMES)

    @property
    def detector(self) -> RFDETRDetector:
        if self._detector is None:
            self._detector = RFDETRDetector(self.detector_variant)
        return self._detector

    def run_crops(self, crops: np.ndarray, cliffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inférence brute -> (cam_trans (N,3), joints_2d norm (N,70,2), joints_3d local (N,70,3))."""
        _, _, cam, j2, j3 = self.session.run(None, {"image": crops.astype(np.float32),
                                                    "cliff_cond": cliffs.astype(np.float32)})
        return cam.astype(np.float64), j2.astype(np.float64), j3.astype(np.float64)

    def predict_frames(self, frames: list[np.ndarray], bboxes: list[np.ndarray]):
        """Prédit pour une liste d'images et de bbox. Renvoie joints3d_cam (N,70,3), joints2d (N,70,2)."""
        flips = [False, True] if self.flip_tta else [False]
        crops, cliffs, metas = [], [], []
        for fr, bb in zip(frames, bboxes, strict=True):
            for fl in flips:
                c, sx, sy, size, cl = preprocess(fr, bb, fl)
                crops.append(c)
                cliffs.append(cl)
                metas.append((sx, sy, size, fl))
        cam, j2n, j3 = self.run_crops(np.stack(crops), np.stack(cliffs))
        j2 = (j2n + 1.0) * 0.5 * INPUT_SIZE
        out3, out2 = [], []
        for i, (sx, sy, size, fl) in enumerate(metas):
            p3, p2, t = j3[i].copy(), j2[i].copy(), cam[i].copy()
            if fl:  # remise en miroir : x -> -x et permutation gauche/droite
                p3 = p3[self.flip_perm]
                p3[:, 0] *= -1
                t[0] *= -1
                p2 = p2[self.flip_perm]
                p2[:, 0] = INPUT_SIZE - p2[:, 0]
            p2 = p2 * (size / INPUT_SIZE) + np.array([sx, sy])
            out3.append(p3 + t)
            out2.append(p2)
        out3 = np.array(out3).reshape(len(frames), len(flips), 70, 3).mean(1)
        out2 = np.array(out2).reshape(len(frames), len(flips), 70, 2).mean(1)
        return out3, out2

    def detect_track(self, video_path: str | Path, max_frames: int | None = None):
        info = read_video_info(video_path)
        tracker = MainPersonTracker((info.width, info.height))
        frames, boxes, detected = [], [], []
        for fr in iter_frames(video_path, max_frames=max_frames):
            box, ok = tracker.update(self.detector.detect(fr))
            frames.append(fr)
            boxes.append(box)
            detected.append(ok)
        return info, frames, boxes, np.array(detected)

    def process_video(self, video_path: str | Path, max_frames: int | None = None) -> PoseSequence:
        t0 = time.perf_counter()
        info, frames, boxes, detected = self.detect_track(video_path, max_frames)
        t_det = time.perf_counter() - t0
        T = len(frames)
        # images avant la première détection : on recopie la première bbox valide
        first = next((b for b in boxes if b is not None), None)
        if first is None:
            raise RuntimeError(f"Aucune personne détectée dans {video_path}")
        boxes = [b if b is not None else first for b in boxes]

        t1 = time.perf_counter()
        j3 = np.zeros((T, 70, 3))
        j2 = np.zeros((T, 70, 2))
        for s in range(0, T, self.batch_size):
            e = min(T, s + self.batch_size)
            j3[s:e], j2[s:e] = self.predict_frames(frames[s:e], boxes[s:e])
        t_hmr = time.perf_counter() - t1

        valid = np.array([b is not None for b in boxes]) & (np.cumsum(detected) > 0)
        conf = np.repeat(np.where(detected, 1.0, 0.5)[:, None], 70, axis=1)
        return PoseSequence(
            joints3d_cam=j3, joints2d=j2, confidence=conf, valid=valid, bbox=np.array(boxes),
            fps=info.fps, image_size=(info.width, info.height),
            intrinsics=default_intrinsics(info.width, info.height), joint_names=list(MHR70_JOINT_NAMES),
            backend=self.name,
            timings={"detection_s": t_det, "pose_s": t_hmr, "n_frames": T,
                     "ms_per_frame": 1000 * (t_det + t_hmr) / max(T, 1)},
        )


def focal_from_diag(width: int, height: int) -> float:
    return math.hypot(width, height)
