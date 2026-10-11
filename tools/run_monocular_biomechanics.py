"""Exécution headless de MonocularBiomechanics (référence, AGPL) avec mesure des temps.

A lancer depuis third_party/MonocularBiomechanics :
    uv run python ../../tools/run_monocular_biomechanics.py VIDEO [--iters N] [--out DIR]
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--out", default="mb_out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.video))[0]
    timings = {}

    import jax
    import tensorflow as tf
    jax.config.update("jax_enable_x64", True)
    print("TF devices:", tf.config.list_physical_devices(), "JAX:", jax.devices())

    from monocular_demos.utils import joint_names, load_metrabs, video_reader

    kp_file = os.path.join(a.out, f"{name}_keypoints.npz")
    if not os.path.exists(kp_file):
        t0 = time.time()
        model = load_metrabs()
        timings["metrabs_load_s"] = time.time() - t0

        t0 = time.time()
        vid, n_frames = video_reader(a.video, batch_size=2)
        acc = None
        for batch in vid:
            pred = model.detect_poses_batched(batch, skeleton="bml_movi_87")
            if acc is None:
                acc = pred
            else:
                for k in acc:
                    acc[k] = tf.concat([acc[k], pred[k]], axis=0)
        dt = time.time() - t0
        timings["metrabs_inference_s"] = dt
        timings["n_frames"] = int(n_frames)
        timings["metrabs_ms_per_frame"] = 1000 * dt / max(n_frames, 1)
        boxes, p3, p2 = [], [], []
        for b, k3, k2 in zip(acc["boxes"], acc["poses3d"], acc["poses2d"]):
            if len(b) == 0:
                boxes.append(np.zeros(5)); p3.append(np.zeros((87, 3))); p2.append(np.zeros((87, 2)))
            else:
                boxes.append(b[0].numpy()); p3.append(k3[0].numpy()); p2.append(k2[0].numpy())
        np.savez(kp_file, keypoints3d=np.array(p3), keypoints2d=np.array(p2), boxes=np.array(boxes),
                 confs=np.ones((len(p3), 87)))
    d = np.load(kp_file)

    import cv2
    from monocular_demos.biomechanics_mjx.monocular_trajectory import fit_model, get_model
    from monocular_demos.dataset import MonocularDataset, get_samsung_calibration

    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS)
    ts = np.arange(len(d["keypoints2d"])) / fps
    ds = MonocularDataset(
        timestamps=[ts], keypoints_2d=[d["keypoints2d"][None]], keypoints_3d=[d["keypoints3d"][None]],
        keypoint_confidence=[np.ones_like(d["keypoints2d"][..., 0])[None]],
        camera_params=get_samsung_calibration(), phone_attitude=None,
    )
    t0 = time.time()
    model = get_model(ds, xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml",
                      joint_names=joint_names)
    model, metrics = fit_model(model, ds, lr_init_value=1e-3, max_iters=a.iters)
    timings["fit_s"] = time.time() - t0
    timings["fit_iters"] = a.iters
    (state, _, _), (qpos, qvel, _), _ = model(ds.get_all_timestamps(0), trajectory_selection=0, steps=0,
                                               skip_action=True, fast_inference=True, check_constraints=False)
    np.savez(os.path.join(a.out, f"{name}_fitted_model.npz"), timestamps=ts, qpos=np.array(qpos),
             sites=np.array(state.site_xpos), scale=np.array(model.body_scale))
    print("METRICS", {k: float(v) for k, v in metrics.items()})
    print("TIMINGS", timings)


if __name__ == "__main__":
    main()
