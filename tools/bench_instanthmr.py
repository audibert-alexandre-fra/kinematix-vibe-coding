"""Benchmark du dépôt InstantHMR original (RF-DETR + InstantHMR ONNX) sur une vidéo.

    uv run python tools/bench_instanthmr.py VIDEO --device cpu|coreml --detector nano|medium
"""

import argparse
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "third_party/InstantHMR")
from instanthmr import PosePipeline  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--device", default="cpu")
    p.add_argument("--detector", default="medium")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=100)
    a = p.parse_args()
    t0 = time.time()
    pipe = PosePipeline("weights/instanthmr.onnx", device=a.device, detector_variant=a.detector,
                        max_persons=1, detector_stride=a.stride)
    load = time.time() - t0
    cap = cv2.VideoCapture(a.video)
    det, hmr, tot = [], [], []
    n = 0
    while n < a.max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        r = pipe.predict(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        det.append(r.detector_ms); hmr.append(r.hmr_ms); tot.append(r.total_ms)
        n += 1
    print(f"RESULT video={a.video} device={a.device} det={a.detector} stride={a.stride} frames={n} "
          f"load_s={load:.1f} det_ms={np.mean(det[2:]):.1f} hmr_ms={np.mean(hmr[2:]):.1f} "
          f"total_ms={np.mean(tot[2:]):.1f} fps={1000/np.mean(tot[2:]):.1f}")


if __name__ == "__main__":
    main()
