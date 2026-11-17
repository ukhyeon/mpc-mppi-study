"""여러 형상의 거리-활성 거리 비용을 하나로 합치기 (18에서 추가). 기존 함수와 장면 수치는 바꾸지 않는다.

형상 하나 = 원 또는 캡슐 + 그 형상의 η. 형상 i의 비용 c_i = eta_cost(간격_i, η_i, w)이고, 합치는 방법은 세 가지다.
  'min' : 간격이 가장 작은(가장 가까운) 형상 하나의 비용 (14~17의 사람 비용과 같은 방식)
  'sum' : 모든 형상의 비용 합 (cuRobo 충돌 비용의 기본)
  'max' : 형상 비용 중 가장 큰 것
"""
import numpy as np

from . import scene as S
from .geometry import dist_circle, dist_capsule
from .mppi import eta_cost


def capsule(a, b, r, eta, name=''):
    """캡슐 형상 (선분 a-b를 반지름 r로 두름)."""
    return dict(kind='capsule', a=np.asarray(a, float), b=np.asarray(b, float), r=float(r), eta=float(eta), name=name)


def circle(c, r, eta, name=''):
    """원 형상."""
    return dict(kind='circle', c=np.asarray(c, float), r=float(r), eta=float(eta), name=name)


def scene_shapes(eta=S.ETA0):
    """14~17의 기준 자세(몸통 원 + 팔 캡슐)를 형상 목록으로. 기본값으로 combine_cost(..., 'min')이 human_cost_uniform과 같다."""
    return [circle(S.TORSO_C, S.TORSO_R, eta, 'torso'), capsule(S.ARM_A, S.ARM_B, S.ARM_R, eta, 'arm')]


def shape_gaps(P, shapes, offset=(0.0, 0.0)):
    """로봇 원 표면과 각 형상 표면 사이 간격 [..., n] (음수면 겹침)."""
    off = np.asarray(offset, float)
    g = [dist_circle(P, s['c'] + off, s['r']) if s['kind'] == 'circle' else dist_capsule(P, s['a'] + off, s['b'] + off, s['r'])
         for s in shapes]
    return np.stack(g, -1) - S.ROBOT_R


def shape_costs(P, shapes, offset=(0.0, 0.0), w=S.W_HUMAN):
    """형상별 비용 [..., n]과 간격 [..., n]."""
    g = shape_gaps(P, shapes, offset)
    eta = np.array([s['eta'] for s in shapes])
    return eta_cost(g, eta, w), g


def combine_cost(Pf, offsets, shapes, mode='min', w=S.W_HUMAN):
    """형상 비용을 mode('min', 'sum', 'max')로 합친 사람 비용 [...]. run_mpc_opts의 human_cost로 쓸 때는 partial로 shapes, mode를 묶는다."""
    c, g = shape_costs(Pf, shapes, offsets, w)
    if mode == 'min':
        return np.take_along_axis(c, np.argmin(g, -1)[..., None], -1)[..., 0]
    if mode == 'sum':
        return c.sum(-1)
    if mode == 'max':
        return c.max(-1)
    raise ValueError(mode)
