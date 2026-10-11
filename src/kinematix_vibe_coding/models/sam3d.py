"""Backend SAM 3D Body via SAM3DFast (https://github.com/mohamdev/SAM3DFast).

SAM 3D Body (licence SAM : usage commercial autorisé) produit les mêmes 70 keypoints MHR70
qu'InstantHMR ; il est plus précis mais beaucoup plus lourd (backbone DINOv3-H, GPU NVIDIA
conseillé, poids HuggingFace soumis à acceptation). Ce backend :
  * lance la CLI de SAM3DFast (`python -m sam3dfast.pipeline ... --parquet`) si le dépôt est
    disponible (variable d'environnement SAM3DFAST_DIR, poids dans `models/` du dépôt) ;
  * ou relit directement un export `.parquet` déjà calculé (ex. sur une machine GPU), placé
    dans SAM3D_PARQUET_DIR/<nom_video>.parquet.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..utils.skeleton import MHR70_JOINT_NAMES
from ..video import CameraIntrinsics, default_intrinsics, read_video_info
from .base import PoseSequence


def parquet_to_sequence(parquet: str | Path, video: str | Path, person_id: int | None = None) -> PoseSequence:
    """Convertit un export SAM3DFast (une ligne par (image, personne)) en `PoseSequence`."""
    import pyarrow.parquet as pq

    info = read_video_info(video)
    tab = pq.read_table(parquet, columns=["frame_idx", "person_id", "joints_3d", "joints_2d", "cam_translation",
                                          "cam_focal_length", "cam_principal_point", "bbox", "confidence"])
    cols = {c: tab.column(c).to_pylist() for c in tab.column_names}
    pids = np.array(cols["person_id"])
    if person_id is None:  # personne la plus présente
        vals, counts = np.unique(pids, return_counts=True)
        person_id = int(vals[np.argmax(counts)])
    T = info.n_frames
    j3 = np.zeros((T, 70, 3))
    j2 = np.zeros((T, 70, 2))
    bbox = np.zeros((T, 4))
    conf = np.zeros((T, 70))
    valid = np.zeros(T, bool)
    focal, pp = None, None
    for i, f in enumerate(cols["frame_idx"]):
        if pids[i] != person_id or f >= T:
            continue
        j3[f] = np.asarray(cols["joints_3d"][i], float).reshape(70, 3) + np.asarray(cols["cam_translation"][i])
        j2[f] = np.asarray(cols["joints_2d"][i], float).reshape(70, 2)
        bbox[f] = np.asarray(cols["bbox"][i], float)[:4]
        conf[f] = float(cols["confidence"][i])
        valid[f] = True
        focal = np.asarray(cols["cam_focal_length"][i], float).ravel()
        pp = np.asarray(cols["cam_principal_point"][i], float).ravel()
    if focal is None:
        raise RuntimeError(f"Personne {person_id} absente de {parquet}")
    fx = float(focal[0])
    fy = float(focal[-1])
    intr = CameraIntrinsics(fx, fy, float(pp[0]), float(pp[1])) if pp is not None else default_intrinsics(
        info.width, info.height, fx)
    return PoseSequence(joints3d_cam=j3, joints2d=j2, confidence=conf, valid=valid, bbox=bbox, fps=info.fps,
                        image_size=(info.width, info.height), intrinsics=intr,
                        joint_names=list(MHR70_JOINT_NAMES), backend="sam3d")


class Sam3DBackend:
    name = "sam3d"
    license = "SAM 3D Body : licence SAM (commercial autorisé) ; SAM3DFast : voir dépôt ; YOLO11 (AGPL) à remplacer"

    def __init__(self, repo_dir: str | Path | None = None, parquet_dir: str | Path | None = None,
                 preset: str = "quality", device: str = "cuda:0"):
        self.repo_dir = Path(repo_dir or os.environ.get("SAM3DFAST_DIR", "third_party/SAM3DFast"))
        self.parquet_dir = Path(parquet_dir or os.environ.get("SAM3D_PARQUET_DIR", "weights/sam3d_parquet"))
        self.preset = preset
        self.device = device

    def process_video(self, video_path: str | Path, max_frames: int | None = None) -> PoseSequence:
        video_path = Path(video_path)
        parquet = self.parquet_dir / f"{video_path.stem}.parquet"
        t0 = time.perf_counter()
        if not parquet.exists():
            ckpt = self.repo_dir / "models" / "sam-3d-body-dinov3" / "model.ckpt"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"SAM 3D Body indisponible : ni export {parquet}, ni poids {ckpt}. Les poids "
                    "facebook/sam-3d-body-dinov3 sont soumis à acceptation sur HuggingFace (voir rapport/).")
            parquet.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "sam3dfast.pipeline", "--video", str(video_path.resolve()),
                            "--out", str(parquet.with_suffix(".rrd").resolve()), "--parquet", str(parquet.resolve()),
                            "--no-vertices", "--max-persons", "1", "--no-hands", "--preset", self.preset,
                            "--device", self.device], cwd=self.repo_dir, check=True)
        seq = parquet_to_sequence(parquet, video_path)
        seq.timings = {"pose_s": time.perf_counter() - t0, "n_frames": len(seq)}
        return seq
