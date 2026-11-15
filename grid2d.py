"""10 노트용 2D 격자 세계와 2D 깊이 카메라 흉내.

카메라 시뮬레이션과 그림 그리기만 담았다. 지도 만들기(occupancy 갱신, 거리장)는 노트 안에서 직접 쓴다.
격자: 1 cm 칸, x 0~1.0 m (카메라가 보는 방향), y 0~0.8 m (위아래).
"""
import numpy as np

RES = 0.01
NX, NY = 100, 80
CAM = np.array([0.05, 0.40])                      # 카메라 위치
FOV = np.deg2rad(70)                              # 시야각
NRAY = 160
ANGLES = np.linspace(-FOV / 2, FOV / 2, NRAY)     # 광선 방향
DROPOUT = (np.deg2rad(11.5), np.deg2rad(22.0))    # 깊이가 안 돌아오는 각도 구간 (손끝 위 흉내)

PALM = (0.55, 0.57, 0.30, 0.50)                   # 손바닥: x0, x1, y0, y1 [m]
WALL = (0.95, 1.00, 0.00, 0.80)                   # 뒤쪽 벽

FREE, OCC, UNKNOWN = 0, 1, 2


def cell(p):
    """좌표 [m] -> 칸 번호 (ix, iy)."""
    return int(p[0] / RES), int(p[1] / RES)


def make_world(boxes):
    """상자 목록 [(x0, x1, y0, y1), ...] 을 점유(True)로 채운 격자 [NX, NY]."""
    occ = np.zeros((NX, NY), bool)
    for x0, x1, y0, y1 in boxes:
        occ[int(round(x0 / RES)):int(round(x1 / RES)), int(round(y0 / RES)):int(round(y1 / RES))] = True
    return occ


def raycast(occ, dropout=DROPOUT, noise=0.0, rng=None):
    """광선마다 처음 만나는 점유 칸까지 거리 [NRAY]. 아무것도 안 돌아오면 nan (실제 카메라의 0)."""
    depths = np.full(NRAY, np.nan)
    for i, a in enumerate(ANGLES):
        d = np.array([np.cos(a), np.sin(a)])
        for r in np.arange(0.02, 1.3, RES / 4):
            ix, iy = cell(CAM + r * d)
            if not (0 <= ix < NX and 0 <= iy < NY):
                break
            if occ[ix, iy]:
                depths[i] = r
                break
    if dropout is not None:
        depths[(ANGLES > dropout[0]) & (ANGLES < dropout[1])] = np.nan
    if noise:
        depths = depths + rng.normal(0, noise, NRAY)
    return depths


def ray_dir(a):
    return np.array([np.cos(a), np.sin(a)])


def show_rays(ax, occ, depths, every=4):
    """세계와 광선. 초록 = 깊이 반환, 빨강 점선 = 반환 없음."""
    ax.imshow(occ.T, origin='lower', extent=[0, 1, 0, 0.8], cmap='Greys', alpha=0.8)
    for a, dep in zip(ANGLES[::every], depths[::every]):
        if np.isfinite(dep):
            e = CAM + dep * ray_dir(a)
            ax.plot([CAM[0], e[0]], [CAM[1], e[1]], color='tab:green', lw=0.5)
        else:
            e = CAM + 0.9 * ray_dir(a)
            ax.plot([CAM[0], e[0]], [CAM[1], e[1]], color='tab:red', lw=0.5, ls=':')
    ax.plot(*CAM, 'k^', ms=10)
    ax.set_xlabel('x [m] (카메라가 보는 방향)'); ax.set_ylabel('y [m]')


def show_state(ax, state, title, palm=True):
    """지도 상태: 흰 = 빈 칸, 검정 = 점유, 회색 = 모름."""
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    ax.imshow(state.T, origin='lower', extent=[0, 1, 0, 0.8],
              cmap=ListedColormap(['white', 'black', '#b0b0b0']), vmin=0, vmax=2)
    ax.plot(*CAM, 'r^', ms=8)
    if palm:
        x0, x1, y0, y1 = PALM
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='tab:orange', lw=1, ls='--'))
    ax.set_title(title); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
