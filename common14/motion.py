"""사람의 참 움직임 (시각 t[s] → (offset[2], 속도[2])). offset은 scene.py 기준 자세에서의 평행이동량.

모든 움직임은 기준 자세를 지나거나 기준 자세에서 출발하도록 정의해, 장면 수치는 바꾸지 않는다.
"""
import numpy as np

WALK_SPEED = 0.5    # 사람 보행 속도 [m/s] (천천히 걷는 속도)
SIDE_Y0 = 0.6       # 옆에서 들어오는 사람의 시작 위치: 기준 자세에서 +y로 0.6 m


def frontal(v=WALK_SPEED):
    """기준 자세에서 출발해 로봇 쪽(-x)으로 등속으로 다가온다."""
    return lambda t: (np.array([-v * t, 0.0]), np.array([-v, 0.0]))


def crossing(v=WALK_SPEED, y0=SIDE_Y0):
    """경로 옆(+y0)에서 출발해 -y로 등속으로 걸어 경로를 가로지른다. t = y0/v에 기준 자세를 지난다."""
    return lambda t: (np.array([0.0, y0 - v * t]), np.array([0.0, -v]))


def turn_back(t_turn=0.6, v=WALK_SPEED, y0=SIDE_Y0):
    """crossing처럼 경로 쪽으로 오다가 t_turn에 방향을 바꿔(+y) 되돌아간다."""
    def f(t):
        if t < t_turn:
            return np.array([0.0, y0 - v * t]), np.array([0.0, -v])
        return np.array([0.0, y0 - v * t_turn + v * (t - t_turn)]), np.array([0.0, v])
    return f
