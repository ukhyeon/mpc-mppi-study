"""10·11 노트용: 실제 D435i 깊이 기록(TAPE)을 읽고 로봇 좌표 3D 점으로 바꾸는 코드.

10 노트에서 한 줄씩 만든 것과 같은 코드다. 원본은 franka-depth-mppi의
experiment/hand_experiment.py:244 (가장자리 마스크), experiment/target_margin.py:19-31 (역투영·ROI·anchor).
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from common import REPO, TAPE, read_jsonl

META = json.loads((TAPE / 'metadata.json').read_text(encoding='utf-8'))
FRAMES = read_jsonl(TAPE / 'frames.jsonl')
K = np.array(META['intrinsics'])                      # [[fx,0,cx],[0,fy,cy],[0,0,1]]
TARGET = json.loads((REPO / 'experiment' / 'target.json').read_text(encoding='utf-8'))
ROI_LOW = np.array(TARGET['manual_target_world_bounds_m']['min'])
ROI_HIGH = np.array(TARGET['manual_target_world_bounds_m']['max'])

# 카메라 → 로봇 좌표 변환. pose 표기 [x, y, z, qw, qx, qy, qz]; scipy는 [qx, qy, qz, qw] 순서를 받는다.
_T = META['hand_registration']['registered_camera_to_robot_pose_wxyz']
R_CAM = Rotation.from_quat([*_T[4:], _T[3]]).as_matrix()
T_CAM = np.array(_T[:3])


def load_depth(rec, mask_edges=True):
    """프레임 기록 하나 -> 깊이 영상 [480, 640] (m). 0은 측정 실패."""
    d = np.load(TAPE / rec['file'])['depth_m'].copy()
    if mask_edges:                                    # hand_experiment.py:244
        d[:24, :] = 0; d[456:, :] = 0; d[:, :32] = 0; d[:, 608:] = 0
    return d


def backproject(d):
    """깊이 영상 -> 로봇 좌표 3D 점 [N, 3], 그리고 각 점의 픽셀 위치 (행 ys, 열 xs)."""
    ys, xs = np.nonzero(d > 0)
    z = d[ys, xs]
    p_cam = np.column_stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z])
    return p_cam @ R_CAM.T + T_CAM, ys, xs


def roi_points(d):
    """ROI 상자 안의 점만 [M, 3] (float32, 원본과 같은 자료형)."""
    pc, _, _ = backproject(d)
    return pc[np.all((pc >= ROI_LOW) & (pc < ROI_HIGH), axis=1)].astype(np.float32)


def candidate_anchors(points, bin_m=None, min_points=None):
    """yz 평면 bin마다 대표점 하나 (target_margin.py:26-31). nvblox sdf 필터 이전 단계."""
    bin_m = bin_m or TARGET['patch_bin_yz_m']
    min_points = min_points or TARGET['min_points_per_bin']
    bins = np.floor((points[:, 1:] - ROI_LOW[1:]) / bin_m).astype(int)
    groups = bins[:, 0] * 8 + bins[:, 1]
    out = []
    for key in sorted(set(groups)):
        g = points[groups == key]
        if len(g) < min_points:
            continue
        center = np.median(g, axis=0)
        out.append(g[np.argmin(np.linalg.norm(g - center, axis=1))])
    return np.asarray(out, np.float32).reshape(-1, 3)
