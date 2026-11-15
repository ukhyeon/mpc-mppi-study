"""거리·방향 계산. 모든 함수는 점 P[..., 2]를 받아 앞쪽 차원을 그대로 유지한다.

사람 위치가 시점마다 다를 때는 offset[..., 2]를 P와 같은 앞쪽 모양으로 넘긴다
(예: P[K, T, 2]와 offset[T, 2] → 시점 t의 후보 점들은 시점 t의 사람과 비교).
"""
import numpy as np

from .scene import ROBOT_R, segments


def dist_circle(P, c, r):
    """점과 원 표면 사이 거리 (안쪽이면 음수)."""
    return np.linalg.norm(P - c, axis=-1) - r


def dist_capsule(P, a, b, r):
    """점과 캡슐(선분 a-b를 반지름 r로 두른 모양) 표면 사이 거리."""
    ab = b - a
    s = np.clip(np.sum((P - a) * ab, -1) / np.sum(ab * ab, -1), 0.0, 1.0)
    closest = a + s[..., None] * ab
    return np.linalg.norm(P - closest, axis=-1) - r


def segment_gaps(P, offset=(0.0, 0.0)):
    """로봇 원 표면과 각 분절 표면 사이 간격 [..., n_seg] (음수면 겹침). 순서: torso, arm."""
    off = np.asarray(offset, float)
    gaps = []
    for seg in segments():
        if seg['kind'] == 'circle':
            d = dist_circle(P, seg['c'] + off, seg['r'])
        else:
            d = dist_capsule(P, seg['a'] + off, seg['b'] + off, seg['r'])
        gaps.append(d - ROBOT_R)
    return np.stack(gaps, axis=-1)


def nearest(P, offset=(0.0, 0.0)):
    """가장 가까운 분절의 (간격, 분절 번호, 충돌 방향 u).

    u: 로봇 점에서 그 분절 무게중심으로 향하는 단위벡터 (사람 쪽은 u, 로봇 쪽은 -u).
    """
    off = np.asarray(offset, float)
    g = segment_gaps(P, off)
    idx = np.argmin(g, axis=-1)
    coms = np.stack([seg['com'] for seg in segments()])           # [n_seg, 2]
    d = coms[idx] + off - P
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    return np.take_along_axis(g, idx[..., None], -1)[..., 0], idx, u


def com_directions(P, offset=(0.0, 0.0)):
    """모든 분절에 대해 로봇 점 → 분절 무게중심 단위벡터 [..., n_seg, 2]."""
    off = np.asarray(offset, float)
    coms = np.stack([seg['com'] for seg in segments()]) + off
    d = coms - P[..., None, :]
    return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
