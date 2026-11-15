"""08 노트용 2관절 평면 팔 헬퍼: 순기구학, 충돌 구, 그림.

관절 각도 q = (q1, q2) [rad]. q1은 바닥 기준 첫 링크 각도, q2는 첫 링크 기준 둘째 링크 각도.
"""
import numpy as np

from common import eta_cost

L1, L2 = 0.5, 0.4   # 링크 길이 [m]


def fk(q, l1=L1, l2=L2):
    """순기구학. q[..., 2] -> 점 [..., 3, 2]: 바닥(0,0), 팔꿈치, 손끝."""
    q1, q2 = q[..., 0], q[..., 1]
    elbow = np.stack([l1 * np.cos(q1), l1 * np.sin(q1)], -1)
    tip = elbow + np.stack([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)], -1)
    base = np.zeros_like(elbow)
    return np.stack([base, elbow, tip], -2)


def sphere_centers(q, n_per_link=4):
    """링크마다 n개 구 중심을 고르게 배치. q[..., 2] -> [..., 2n, 2]."""
    pts = fk(q)
    s = (np.arange(n_per_link) + 0.5) / n_per_link          # 링크 위 위치 비율
    links = [pts[..., i, None, :] + s[:, None] * (pts[..., i + 1, None, :] - pts[..., i, None, :])
             for i in range(2)]
    return np.concatenate(links, -2)


def sphere_gap(q, obstacles, radius, n_per_link=4):
    """모든 구와 모든 장애물 원 사이 간격 중 최솟값 [...]. 음수면 겹침."""
    c = sphere_centers(q, n_per_link)
    gaps = [np.linalg.norm(c - np.asarray(o), axis=-1) - radius - r for o, r in obstacles]
    return np.min(np.stack(gaps, -1), axis=(-1, -2))


def sphere_eta_cost(q, obstacles, radius, eta, weight=1.0, n_per_link=4):
    """구 하나·장애물 하나마다 η 비용을 계산해 모두 더한다 [...]."""
    c = sphere_centers(q, n_per_link)
    total = 0.0
    for o, r in obstacles:
        gap = np.linalg.norm(c - np.asarray(o), axis=-1) - radius - r
        total = total + eta_cost(gap, eta, weight).sum(-1)
    return total


def draw_arm(ax, q, color='tab:blue', alpha=1.0, spheres=None, lw=4, label=None):
    """팔 그리기. spheres=(반지름, 링크당 개수)를 주면 충돌 구도 그린다."""
    import matplotlib.pyplot as plt
    pts = fk(np.asarray(q, float))
    ax.plot(pts[:, 0], pts[:, 1], '-', color=color, lw=lw, alpha=alpha, solid_capstyle='round', label=label)
    ax.plot(pts[:2, 0], pts[:2, 1], 'o', color='k', ms=5, alpha=alpha)
    ax.plot(pts[2, 0], pts[2, 1], 'o', color=color, ms=7, alpha=alpha)
    if spheres is not None:
        radius, n = spheres
        for c in sphere_centers(np.asarray(q, float), n):
            ax.add_patch(plt.Circle(c, radius, fill=False, color=color, alpha=0.7 * alpha))
    ax.set_aspect('equal')
