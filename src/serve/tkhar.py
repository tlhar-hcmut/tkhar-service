from typing import Dict, List

import model
import numpy as np
from entity.request import HarPoint


def get_img_skeleton(data: List[HarPoint]) -> np.ndarray:
    def _get_joint(idx: int):
        return np.array([data[idx].z, data[idx].x, data[idx].y])

    hip_center = (_get_joint(23) + _get_joint(24)) / 2
    shoulder_center = (_get_joint(9) + _get_joint(10)) / 2
    mouth = (_get_joint(9) + _get_joint(10)) / 2
    return np.array(
        [
            hip_center,  # 1 hip_center
            (hip_center + shoulder_center) / 2,  # 2 spine_center
            (mouth + shoulder_center) / 2,  # 3 spine_center
            _get_joint(0),  # 4 head
            _get_joint(11),  # 5  left_shoulder
            _get_joint(13),  # 6  left_elbow
            _get_joint(15),  # 7  left_wrist
            _get_joint(15),  # 8  left_hand # TODO
            _get_joint(12),  # 9  right_shoulder
            _get_joint(14),  # 10 right_elbow
            _get_joint(16),  # 11 right_wrist
            _get_joint(16),  # 12 right_hand # TODO
            _get_joint(23),  # 13 left_hip
            _get_joint(25),  # 14 left_knee
            _get_joint(27),  # 15 left_ankle
            _get_joint(31),  # 16 left_foot
            _get_joint(24),  # 17 right_hip
            _get_joint(26),  # 18 right_knee
            _get_joint(28),  # 19 right_ankle
            _get_joint(30),  # 20 right_foot
            shoulder_center,  # 21 center_shoulder
            (_get_joint(19) + _get_joint(17)) / 2,  # 22 left_hand_tip
            _get_joint(21),  # 23 left_hand_thump
            (_get_joint(18) + _get_joint(20)) / 2,  # 24 right_hand_tip
            _get_joint(22),  # 25 right_hand_thump
        ]
    )


def get_video_skeleton(data: List[List[HarPoint]]) -> np.ndarray:
    output = [get_img_skeleton(skeleton) for skeleton in data]
    output = np.array(output).transpose(2, 0, 1)
    output = np.expand_dims(output, axis=[-1, 0])
    output = np.squeeze(output, axis=0)
    return np.array(output)


def predict(data: List[List[HarPoint]]) -> Dict:
    input = get_video_skeleton(data)
    print(model.model_se_25(input))
    return {
        "code": 0,
        "message": "OK",
        "data": {"idAction": 0, "nameAction": "Brush teeth", "version": "TK HAR 1.0"},
    }
