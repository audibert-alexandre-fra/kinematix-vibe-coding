"""Définitions de squelettes.

MHR70 : ordre des 70 keypoints de SAM 3D Body / InstantHMR (repris de
instanthmr/skeleton.py, Apache-2.0). Corps, puis mains, puis olécrane / fosse
cubitale / acromion / cou.
"""

from __future__ import annotations

MHR70_JOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",          # 0-4
    "left_shoulder", "right_shoulder",                                  # 5-6
    "left_elbow", "right_elbow",                                        # 7-8
    "left_hip", "right_hip",                                            # 9-10
    "left_knee", "right_knee",                                          # 11-12
    "left_ankle", "right_ankle",                                        # 13-14
    "left_big_toe_tip", "left_small_toe_tip", "left_heel",              # 15-17
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",           # 18-20
    "right_thumb_tip", "right_thumb_first_joint",
    "right_thumb_second_joint", "right_thumb_third_joint",              # 21-24
    "right_index_tip", "right_index_first_joint",
    "right_index_second_joint", "right_index_third_joint",              # 25-28
    "right_middle_tip", "right_middle_first_joint",
    "right_middle_second_joint", "right_middle_third_joint",            # 29-32
    "right_ring_tip", "right_ring_first_joint",
    "right_ring_second_joint", "right_ring_third_joint",                # 33-36
    "right_pinky_tip", "right_pinky_first_joint",
    "right_pinky_second_joint", "right_pinky_third_joint",              # 37-40
    "right_wrist",                                                      # 41
    "left_thumb_tip", "left_thumb_first_joint",
    "left_thumb_second_joint", "left_thumb_third_joint",                # 42-45
    "left_index_tip", "left_index_first_joint",
    "left_index_second_joint", "left_index_third_joint",                # 46-49
    "left_middle_tip", "left_middle_first_joint",
    "left_middle_second_joint", "left_middle_third_joint",              # 50-53
    "left_ring_tip", "left_ring_first_joint",
    "left_ring_second_joint", "left_ring_third_joint",                  # 54-57
    "left_pinky_tip", "left_pinky_first_joint",
    "left_pinky_second_joint", "left_pinky_third_joint",                # 58-61
    "left_wrist",                                                       # 62
    "left_olecranon", "right_olecranon",                                # 63-64
    "left_cubital_fossa", "right_cubital_fossa",                        # 65-66
    "left_acromion", "right_acromion",                                  # 67-68
    "neck",                                                             # 69
]


MHR70_EDGES: list[tuple[int, int]] = [
    # Head <-> neck
    (0, 69),
    # Torso
    (69, 5), (69, 6),
    (5, 9), (6, 10), (9, 10),
    # Arms
    (5, 7), (7, 62),
    (6, 8), (8, 41),
    # Legs
    (9, 11), (11, 13), (13, 17),
    (10, 12), (12, 14), (14, 20),
    # Left hand fan
    (62, 42), (62, 46), (62, 50), (62, 54), (62, 58),
    # Right hand fan
    (41, 21), (41, 25), (41, 29), (41, 33), (41, 37),
]


MHR70_INDEX: dict[str, int] = {n: i for i, n in enumerate(MHR70_JOINT_NAMES)}


def mhr70(name: str) -> int:
    return MHR70_INDEX[name]
