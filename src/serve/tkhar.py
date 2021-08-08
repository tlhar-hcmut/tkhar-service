from typing import Dict, List

import model
import numpy as np
import torch
import util
from entity.request import HarPoint
from entity.response import HarPredict, HarResponse, Response


def get_img_skeleton(data: List[HarPoint]) -> np.ndarray:
    def _get_joint(idx: int):
        return np.array([data[idx].z, data[idx].x, data[idx].y])

    hip_center = (_get_joint(23) + _get_joint(24)) / 2
    shoulder_center = (_get_joint(11) + _get_joint(12)) / 2
    mouth = (_get_joint(9) + _get_joint(10)) / 2
    mouth[0] = (mouth[0] + 2*shoulder_center[0])/3
    head = (_get_joint(1) + _get_joint(4))/2
    head[0] = (_get_joint(0)[0] + 2*shoulder_center[0])/3
    return np.array(
        [
            hip_center,  # 1 hip_center
            (hip_center + shoulder_center) / 2,  # 2 spine_center
            (mouth + shoulder_center) / 2,  # 3 spine_center
            head,  # 4 head
            _get_joint(11),  # 5  left_shoulder
            _get_joint(13),  # 6  left_elbow
            (2*_get_joint(15) + _get_joint(13))/3,  # 7  left_wrist
            _get_joint(15),  # 8  left_hand
            _get_joint(12),  # 9  right_shoulder
            _get_joint(14),  # 10 right_elbow
            (2*_get_joint(16) + _get_joint(14))/3,  # 11 right_wrist
            _get_joint(16),  # 12 right_hand
            _get_joint(23),  # 13 left_hip
            _get_joint(25),  # 14 left_knee
            _get_joint(27),  # 15 left_ankle
            _get_joint(31),  # 16 left_foot
            _get_joint(24),  # 17 right_hip
            _get_joint(26),  # 18 right_knee
            _get_joint(28),  # 19 right_ankle
            _get_joint(32),  # 20 right_foot
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


map_action = {
    0: "Brushing teeth",
    1: "Brushing hair",
    2: "Throw",
    3: "Sitting down",
    4: "Standing up from sitting position",
    5: "Clapping",
    6: "Take off a hat/cap",
    7: "Hand waving",
    8: "Jump up",
    9: "Make a phone call/answer phone",
    10: "Shake fist",
    11: "Side kick",
}


def predict(data: List[List[HarPoint]]) -> Response:
    if len(data) == 0:
        return Response(code=-1, message="Empty Data")
    input = get_video_skeleton(data)
    input_zero = np.zeros((1, 3, 300, 26, 2))
    input_zero[0, :, :input.shape[1], :input.shape[2], :input.shape[3]] = input
    input_norm = util.normalize(input_zero)
    output = model.model_se_net(torch.tensor(input_norm).float())[0]
    predict = []
    for id, confidence in enumerate(torch.softmax(output, dim=-1)):
        predict.append(HarPredict(
            id=id,
            action=map_action[id],
            confidence=round(confidence.item() * 100, 2))
        )
    return HarResponse(
        code=0,
        message="OK",
        data=predict,
    )
