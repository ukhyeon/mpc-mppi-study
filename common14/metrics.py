"""지표 함수. traj[N+1, 2]는 실행 궤적, person[N+1, 2]는 같은 시각의 참 사람 offset."""
import numpy as np

from . import scene as S
from .geometry import segment_gaps


def clearance_series(traj, person):
    """시각마다 로봇 원 표면과 가장 가까운 분절 표면 사이 간격 [m] (참 사람 위치 기준)."""
    return np.array([segment_gaps(p, o).min() for p, o in zip(traj, person)])


def min_clearance(traj, person):
    """실행 전체의 최소 이격 [m]. 음수면 겹침."""
    return float(clearance_series(traj, person).min())


def arrival_time(traj, goal=S.GOAL, tol=S.ARRIVE_TOL):
    """목표 tol 안에 처음 들어간 시각 [s]. 못 들어가면 None."""
    idx = np.nonzero(np.linalg.norm(traj - goal, axis=1) < tol)[0]
    return round(float(idx[0] * S.DT), 2) if len(idx) else None


def ess(w):
    """유효 샘플 수 ESS = (Σw)² / Σw². 가중치가 하나에 몰리면 1, 고르게 퍼지면 K."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def command_change(us):
    """실행한 명령의 스텝 간 변화량 평균 ‖u_t − u_{t−1}‖ [m/s²]. 떨림 지표."""
    return float(np.mean(np.linalg.norm(np.diff(us, axis=0), axis=1)))


def path_length(traj):
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def avoid_start_time(traj, thresh=0.05):
    """경로 직선(y=0)에서 thresh 이상 처음 벗어난 시각 [s]. 회피를 시작한 시점의 근사."""
    idx = np.nonzero(np.abs(traj[:, 1]) > thresh)[0]
    return round(float(idx[0] * S.DT), 2) if len(idx) else None
