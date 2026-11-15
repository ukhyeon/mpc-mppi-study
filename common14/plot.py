"""그림 함수. 01~13과 같은 스타일(setup_plot, 한글 폰트, 격자)."""
import numpy as np

from . import scene as S
from .mass import directional_mass, reduced_mass


def setup_plot():
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic', 'Malgun Gothic']:
        if name in names:
            matplotlib.rcParams['font.family'] = name
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3
    return plt


def draw_person(ax, offset=(0.0, 0.0), eta=None, alpha=0.6, color='k', label=None):
    """평면 사람: 몸통 원 + 팔 캡슐. eta를 주면 비용이 켜지는 범위(주황)를 함께 그린다."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon
    o = np.asarray(offset, float)
    c = S.TORSO_C + o
    a, b = S.ARM_A + o, S.ARM_B + o
    for pad, col, al in ([(S.ROBOT_R + eta, 'orange', 0.15)] if eta else []) + [(0.0, color, alpha)]:
        ax.add_patch(Circle(c, S.TORSO_R + pad, color=col, alpha=al, lw=0))
        r = S.ARM_R + pad
        n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        ax.add_patch(Polygon([a + r * n, b + r * n, b - r * n, a - r * n], color=col, alpha=al, lw=0))
        ax.add_patch(Circle(a, r, color=col, alpha=al, lw=0))
        ax.add_patch(Circle(b, r, color=col, alpha=al, lw=0))
    if label:
        ax.plot([], [], 's', color=color, alpha=alpha, label=label)


def draw_scene(ax, offset=(0.0, 0.0), eta=None, lim=((-0.15, 2.15), (-0.6, 0.6)), show_person=True):
    """시작·목표와 (show_person=True면) offset 위치의 사람을 그린다. 움직이는 사람 그림에서는 False로 두고 따로 그린다."""
    if show_person:
        draw_person(ax, offset, eta=eta)
    ax.plot(*S.START, 'bo', label='시작')
    ax.plot(*S.GOAL, 'g*', ms=15, label='목표')
    ax.set_aspect('equal')
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')


def polar_mass(ax, seg, reduced=False, n=721, **kw):
    """방향각 θ(로봇 → 분절 무게중심 방향 u)에 따른 유효질량(또는 환산질량) 극좌표 그림."""
    th = np.linspace(0, 2 * np.pi, n)
    u = np.stack([np.cos(th), np.sin(th)], -1)
    m = directional_mass(u, seg)
    if reduced:
        m = reduced_mass(m)
    ax.plot(th, m, **kw)
    return th, m
