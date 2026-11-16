"""기준 자세가 아닌 사람 자세 (16의 7절에서 추가). scene.py의 기본 장면과 수치는 그대로 두고 새 이름으로만 정의한다.

모양(몸통 반지름, 팔 길이·반지름)과 관성값(몸통 diag(40, 32), 팔 길이 1 kg / 수직 3 kg, ROM 잠금 4배)은 scene.py와 같다.
바뀌는 것은 몸통 위치, 팔 방향, ROM 잠금 방향뿐이다. 팔의 관성은 팔과 함께 돌린다(팔 기준 길이·수직 값이 같다).
"""
import numpy as np

from . import scene as S
from .geometry import dist_circle, dist_capsule
from .mass import reduced_mass
from .mppi import eta_cost

ARM_LEN = float(np.linalg.norm(S.ARM_A - S.ARM_B))                    # 0.3 m
M_ARM_LEN = float(S.ARM_AXIS @ S.M_ARM_FREE @ S.ARM_AXIS)              # 1 kg
M_ARM_PERP = float(S.ARM_NORMAL @ S.M_ARM_FREE @ S.ARM_NORMAL)         # 3 kg


def make_pose(torso_c, arm_dir, lock_sign=+1, name=''):
    """몸통 중심 torso_c, 팔이 뻗는 방향 arm_dir(어깨 → 손), ROM 잠금 쪽 부호로 자세를 만든다.

    scene.py와 같은 규칙: 팔 축 = (어깨 − 손) 방향, 팔 수직 n = 축을 +90° 돌린 것, 잠금 방향 = lock_sign × n.
    기준 자세는 make_pose(TORSO_C, (−1, 0), +1)과 같다.
    """
    c = np.asarray(torso_c, float)
    d = np.asarray(arm_dir, float) / np.linalg.norm(arm_dir)
    a = c + S.TORSO_R * d                       # 어깨 쪽 끝 (몸통 가장자리)
    b = a + ARM_LEN * d                         # 손 쪽 끝
    axis = (a - b) / np.linalg.norm(a - b)
    normal = np.array([-axis[1], axis[0]])
    M_arm = M_ARM_LEN * np.outer(axis, axis) + M_ARM_PERP * np.outer(normal, normal)
    return dict(name=name, torso_c=c, arm_a=a, arm_b=b, axis=axis, normal=normal, lock_dir=lock_sign * normal,
                inv={'torso': np.linalg.inv(S.M_TORSO), 'arm': np.linalg.inv(M_arm)},
                com=np.stack([c, 0.5 * (a + b)]))


def pose_gaps(P, pose, offset=(0.0, 0.0)):
    """로봇 원 표면과 각 분절 표면 사이 간격 [..., 2] (torso, arm). geometry.segment_gaps와 같은 정의."""
    off = np.asarray(offset, float)
    g_t = dist_circle(P, pose['torso_c'] + off, S.TORSO_R)
    g_a = dist_capsule(P, pose['arm_a'] + off, pose['arm_b'] + off, S.ARM_R)
    return np.stack([g_t, g_a], -1) - S.ROBOT_R


def pose_nearest(P, pose, offset=(0.0, 0.0)):
    """가장 가까운 분절의 (간격, 분절 번호, u). u = 로봇 점 → 그 분절 무게중심 (geometry.nearest와 같음)."""
    off = np.asarray(offset, float)
    g = pose_gaps(P, pose, off)
    idx = np.argmin(g, axis=-1)
    d = pose['com'][idx] + off - P
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    return np.take_along_axis(g, idx[..., None], -1)[..., 0], idx, u


def pose_mass(u, seg, pose):
    """방향 유효질량 m(u) = 1/(uᵀM⁻¹u). 팔은 u·lock_dir > 0이면 ROM_LOCK_FACTOR배 (mass.directional_mass와 같은 규칙)."""
    u = np.asarray(u, float)
    m = 1.0 / np.einsum('...i,ij,...j->...', u, pose['inv'][seg], u)
    if seg == 'arm':
        m = np.where(u @ pose['lock_dir'] > 1e-9, S.ROM_LOCK_FACTOR * m, m)
    return m


def pose_directional_eta(P, pose, eta_of_mass, reduced=False, offset=(0.0, 0.0)):
    """mppi.directional_eta의 자세 버전. 반환: gap, eta, 분절 번호(0 torso, 1 arm), u"""
    gap, idx, u = pose_nearest(P, pose, offset)
    m = np.where(idx == 0, pose_mass(u, 'torso', pose), pose_mass(u, 'arm', pose))
    if reduced:
        m = reduced_mass(m)
    return gap, eta_of_mass(m), idx, u


def pose_cost_uniform(Pf, offsets, pose, eta=S.ETA0, w=S.W_HUMAN):
    """균일 η 사람 비용 [..., T] (mppi.human_cost_uniform의 자세 버전)."""
    return eta_cost(pose_gaps(Pf, pose, offsets).min(-1), eta, w)


def pose_cost_directional(Pf, offsets, pose, eta_of_mass, reduced=False, w=S.W_HUMAN):
    """방향별 η 사람 비용 [..., T] (mppi.human_cost_directional의 자세 버전)."""
    gap, eta, _, _ = pose_directional_eta(Pf, pose, eta_of_mass, reduced, offsets)
    return eta_cost(gap, eta, w)


# ---- 비교용 u 정의 (16의 8절에서 추가): 로봇 점 → 가장 가까운 분절 표면의 가장 가까운 점. 기본 정의(무게중심 방향)는 위 함수들 그대로다.

def pose_nearest_point(P, pose, offset=(0.0, 0.0)):
    """pose_nearest와 같되 u = 로봇 점 → 그 분절 표면의 가장 가까운 점. 몸통(원)은 중심 방향과 같고, 팔(캡슐)은 축 선분의 가장 가까운 점 방향이다."""
    off = np.asarray(offset, float)
    g = pose_gaps(P, pose, off)
    idx = np.argmin(g, axis=-1)
    a, b = pose['arm_a'] + off, pose['arm_b'] + off
    ab = b - a
    s = np.clip(np.sum((P - a) * ab, -1) / np.sum(ab * ab, -1), 0.0, 1.0)
    target = np.where((idx == 1)[..., None], a + s[..., None] * ab, pose['torso_c'] + off)
    d = target - P
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    return np.take_along_axis(g, idx[..., None], -1)[..., 0], idx, u


def pose_directional_eta_point(P, pose, eta_of_mass, reduced=False, offset=(0.0, 0.0)):
    """pose_directional_eta와 같되 u를 가장 가까운 점 방향으로 쓴다. 반환: gap, eta, 분절 번호, u"""
    gap, idx, u = pose_nearest_point(P, pose, offset)
    m = np.where(idx == 0, pose_mass(u, 'torso', pose), pose_mass(u, 'arm', pose))
    if reduced:
        m = reduced_mass(m)
    return gap, eta_of_mass(m), idx, u


def pose_cost_directional_point(Pf, offsets, pose, eta_of_mass, reduced=False, w=S.W_HUMAN):
    """pose_cost_directional과 같되 u를 가장 가까운 점 방향으로 쓴다."""
    gap, eta, _, _ = pose_directional_eta_point(Pf, pose, eta_of_mass, reduced, offsets)
    return eta_cost(gap, eta, w)


def draw_pose(ax, pose, alpha=0.6, color='k', show_lock=True):
    """자세 그리기 (몸통 원 + 팔 캡슐). show_lock이면 팔 가운데에 잠금 방향 화살표(로봇이 이쪽으로 밀면 잠김)."""
    from matplotlib.patches import Circle, Polygon
    a, b, r = pose['arm_a'], pose['arm_b'], S.ARM_R
    n = pose['normal']
    ax.add_patch(Circle(pose['torso_c'], S.TORSO_R, color=color, alpha=alpha, lw=0))
    ax.add_patch(Polygon([a + r * n, b + r * n, b - r * n, a - r * n], color=color, alpha=alpha, lw=0))
    ax.add_patch(Circle(a, r, color=color, alpha=alpha, lw=0))
    ax.add_patch(Circle(b, r, color=color, alpha=alpha, lw=0))
    if show_lock:
        m = pose['com'][1]
        ax.annotate('', xy=m + 0.12 * pose['lock_dir'], xytext=m, arrowprops=dict(arrowstyle='->', color='C3', lw=1.5))
