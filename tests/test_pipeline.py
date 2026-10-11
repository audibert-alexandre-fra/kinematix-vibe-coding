import json

import numpy as np

from kinematix_vibe_coding.pipeline import PipelineConfig, run_pipeline
from tests.synthetic import synthetic_sequence


def test_pipeline_on_synthetic_walk(tmp_path):
    hum, q_true, seq = synthetic_sequence(height=1.8, T=90, noise=0.01, scale_error=0.92)
    seq.save(tmp_path / "pose_instanthmr.npz")  # cache : pas besoin du modèle de pose
    res = run_pipeline("synthetic.mp4", tmp_path, PipelineConfig(height=1.8, render=False), verbose=False)
    s = res.summary
    assert abs(s["model_standing_height_m"] - 1.8) < 0.01
    assert abs(s["data_scale"] * 0.92 - 1.0) < 0.05          # ambiguïté d'échelle corrigée par la taille
    assert s["ik_marker_error_cm"]["median"] < 2.0
    for side in ("l", "r"):
        est = res.angles[f"knee_flexion_{side}"]
        true = np.degrees(q_true[:, hum.joint_qposadr(f"knee_flexion_{side}")])
        assert np.corrcoef(est, true)[0, 1] > 0.95
        assert np.sqrt(np.mean((est - true) ** 2)) < 6.0
    # anti-glissement
    assert s["foot_sliding_after"]["sliding_drift_mean_cm"] <= s["foot_sliding_before"]["sliding_drift_mean_cm"]
    assert s["foot_sliding_after"]["sliding_drift_mean_cm"] < 1.0
    assert (tmp_path / "joint_angles.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["n_frames"] == 90
