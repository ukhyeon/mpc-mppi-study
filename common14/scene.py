"""회차 0에서 고정한 장면 수치. 14~21의 모든 노트가 이 값을 그대로 쓴다 (근거는 README.md).

좌표: 평면 위 x, y [m]. 로봇은 2D 점, 사람은 위에서 내려다본 '평면 사람'(몸통 원 + 팔 캡슐).
"""
import numpy as np

# ---- 로봇과 과제 -----------------------------------------------------------
START = np.array([0.0, 0.0])        # 로봇 시작점 [m]
GOAL = np.array([2.0, 0.0])         # 로봇 목표점 [m]
ROBOT_R = 0.05                      # 로봇 점의 반지름 [m] (05·06과 같음)
M_ROBOT = 5.0                       # 로봇 유효질량 [kg], 방향과 무관한 상수

# ---- 평면 사람 (기준 자세: 로봇 경로 위에 서서 팔을 로봇 쪽으로 뻗고 있음) ----
TORSO_C = np.array([1.35, 0.0])     # 몸통 중심 [m]
TORSO_R = 0.15                      # 몸통 반지름 [m]
ARM_A = np.array([1.20, 0.0])       # 어깨 쪽 끝 (몸통 가장자리) [m]
ARM_B = np.array([0.90, 0.0])       # 손 쪽 끝 [m]
ARM_R = 0.045                       # 팔 캡슐 반지름 [m]

# 분절 관성: 방향 u로 밀 때의 유효질량 m(u) = 1 / (u^T M^-1 u) 에 쓰는 2x2 행렬 [kg] (월드 좌표)
M_TORSO = np.diag([40.0, 32.0])                 # 몸통: 거의 등방
ARM_AXIS = (ARM_A - ARM_B) / np.linalg.norm(ARM_A - ARM_B)   # 팔 길이 방향 (+x)
ARM_NORMAL = np.array([-ARM_AXIS[1], ARM_AXIS[0]])            # 팔 수직 방향 (+y)
M_ARM_FREE = 1.0 * np.outer(ARM_AXIS, ARM_AXIS) + 3.0 * np.outer(ARM_NORMAL, ARM_NORMAL)  # 길이 1 kg, 수직 3 kg
ROM_LOCK_DIR = ARM_NORMAL.copy()    # u·n > 0 (팔을 +y 쪽으로 미는 방향)이면 관절이 ROM 끝에 걸려 잠긴다
ROM_LOCK_FACTOR = 4.0               # 잠긴 쪽은 팔 유효질량이 4배 (몸통 일부가 함께 밀리는 것을 흉내)

# ---- 비용과 MPPI (05·06과 같은 기본형; 반복 횟수만 cuRobo와 같은 5) --------------
DT = 0.05            # 한 스텝 [s]
T = 30               # horizon [스텝] → 1.5 s
K = 400              # 후보 수
N_ITERS = 5          # 스텝당 MPPI 갱신 횟수 (cuRobo n_iters와 같음)
SIGMA = 2.0          # 탐색 폭 σ [m/s²], 고정
BETA = 1.0           # 온도 β
U_MAX = 4.0          # 가속도 한계 [m/s²] (clamp)
W_GOAL, W_TERM, W_VTERM, W_U = 1.0, 20.0, 1.0, 1e-3   # 목표 비용 (course.point_cost와 같음)
W_HUMAN = 100.0      # 사람 근접 비용 세기 w (06의 w_obs와 같음)
ETA0 = 0.10          # 균일 활성 거리 η [m] (06과 같음). 14는 이것만 쓴다
STEPS = 100          # 한 실행 길이 [스텝] → 5 s
ARRIVE_TOL = 0.05    # 도달 판정 [m]


def segments(offset=(0.0, 0.0)):
    """사람 분절 목록. offset만큼 평행이동한 자세. 반환: [dict(name, kind, ...)]"""
    o = np.asarray(offset, float)
    return [
        dict(name='torso', kind='circle', c=TORSO_C + o, r=TORSO_R, com=TORSO_C + o),
        dict(name='arm', kind='capsule', a=ARM_A + o, b=ARM_B + o, r=ARM_R, com=0.5 * (ARM_A + ARM_B) + o),
    ]
