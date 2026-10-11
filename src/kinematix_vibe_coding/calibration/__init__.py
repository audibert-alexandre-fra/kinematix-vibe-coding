"""Étalonnage géométrique : marqueurs, repère monde/sol, mise à l'échelle, calibration auto MuJoCo."""

from .auto_calibration import CalibrationResult, calibrate, select_frames
from .markers import FOOT_MARKERS, MarkerSet
from .scaling import initial_dimensions, measure_segments
from .world_frame import WorldTransform, estimate_world_transform, fit_ground

__all__ = ["FOOT_MARKERS", "CalibrationResult", "MarkerSet", "WorldTransform", "calibrate",
           "estimate_world_transform", "fit_ground", "initial_dimensions", "measure_segments", "select_frames"]
