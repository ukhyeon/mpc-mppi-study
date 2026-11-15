"""노트 01~07에서 직접 만든 코드의 정리본.

각 함수는 처음 등장하는 노트에서 한 줄씩 설명하며 노트 안에 직접 작성한다.
그 뒤의 노트는 같은 코드를 여기서 import해서 쓴다.

  01  cart_step, cart_rollout          1D 카트 모델과 미래 예측(rollout)
  02  cart_cost                        1D 카트 비용 함수
  03  numerical_grad                   수치 미분 (경사하강용)
  04  run_mpc                          MPC 루프 (계획 → 첫 입력 실행 → 반복)
  05  softmax_weights, mppi_step       MPPI 한 번 갱신
  06  point_rollout, obstacle_gap, point_cost   2D 점과 장애물 비용
  07  smooth_noise, mppi_step(옵션)    매끄러운 노이즈, 특수 후보, 분산 갱신
  plot_scene_2d                        2D 장면 그림
"""
import numpy as np

from common import eta_cost

DT = 0.1   # 01~05의 기본 시간 간격 [s]


# ---------------------------------------------------------------------------
# 01 · 1D 카트: 상태 x = (p, v), 입력 u = 가속도
# ---------------------------------------------------------------------------

def cart_step(x, u, dt=DT):
    """한 스텝 진행. 가속도 u가 dt 동안 일정하다고 본다."""
    p, v = x[..., 0], x[..., 1]
    return np.stack([p + v * dt + 0.5 * u * dt ** 2, v + u * dt], axis=-1)


def cart_rollout(x0, U, dt=DT):
    """입력열 U[..., T]를 차례로 넣어 상태열 X[..., T+1, 2]를 만든다. X[..., 0] = x0.

    U의 앞쪽 차원은 몇 개든 된다: U[T] 하나, U[K, T] 후보 K개.
    """
    U = np.asarray(U, float)
    x = np.broadcast_to(np.asarray(x0, float), U.shape[:-1] + (2,))
    X = [x]
    for t in range(U.shape[-1]):
        x = cart_step(x, U[..., t], dt)
        X.append(x)
    return np.stack(X, axis=-2)


# ---------------------------------------------------------------------------
# 02 · 1D 카트 비용
# ---------------------------------------------------------------------------

def cart_cost(X, U, goal=1.0, w_goal=1.0, w_u=0.01, w_term=10.0, w_vterm=1.0):
    """J = Σ_t [w_goal (p_t - goal)^2 + w_u u_t^2] + w_term (p_T - goal)^2 + w_vterm v_T^2.

    X[..., T+1, 2], U[..., T] -> J[...]
    """
    p, v = X[..., 1:, 0], X[..., -1, 1]
    running = w_goal * (p - goal) ** 2 + w_u * U ** 2
    return running.sum(-1) + w_term * (p[..., -1] - goal) ** 2 + w_vterm * v ** 2


# ---------------------------------------------------------------------------
# 03 · 수치 미분
# ---------------------------------------------------------------------------

def numerical_grad(f, U, eps=1e-5):
    """f(U)의 기울기를 중앙 차분으로 근사. U의 원소마다 f를 두 번 부른다."""
    U = np.asarray(U, float)
    g = np.zeros_like(U)
    for i in np.ndindex(U.shape):
        e = np.zeros_like(U)
        e[i] = eps
        g[i] = (f(U + e) - f(U - e)) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
# 04 · MPC 루프
# ---------------------------------------------------------------------------

def run_mpc(plan, x0, steps, plant_step, U_init, shift=True):
    """plan(x, U_init) -> U* 를 매 스텝 풀고 첫 입력만 실제 시스템(plant_step)에 넣는다.

    shift=True 면 다음 스텝 초기값으로 U*를 한 칸 당긴 것을 쓴다(warm start).
    반환: 실제 상태 기록 xs[steps+1, ...], 실행한 입력 us[steps, ...], 매 스텝 계획 plans
    """
    x, U = np.asarray(x0, float), np.asarray(U_init, float)
    xs, us, plans = [x], [], []
    for _ in range(steps):
        U = plan(x, U)
        plans.append(U.copy())
        x = plant_step(x, U[0])
        xs.append(x)
        us.append(U[0])
        U = np.concatenate([U[1:], U[-1:]]) if shift else np.zeros_like(U)
    return np.array(xs), np.array(us), plans


# ---------------------------------------------------------------------------
# 05 · MPPI 기본형
# ---------------------------------------------------------------------------

def softmax_weights(J, beta):
    """비용이 낮을수록 큰 가중치. ω_k = exp(-(J_k - min J)/β) / Σ exp(...)"""
    z = np.exp(-(J - J.min()) / beta)
    return z / z.sum()


def mppi_step(mu, sigma, cost_fn, K, beta, rng, u_max=None, noise=None,
              n_zero=0, n_neg=0, step_size_mean=1.0, step_size_cov=0.0, kappa=0.0):
    """MPPI 한 번 갱신.

    기본형(05): 평균 mu 주변에 K개 샘플 → 비용 → softmax 가중치 → 가중평균이 새 평균.
    07에서 추가한 옵션:
      noise         (K, *mu.shape) 미리 만든 표준정규 노이즈(매끄러운 노이즈, 고정 샘플용)
      n_zero/n_neg  0 입력 후보, -mu 후보 개수 (K개 중에서 마지막에 배치)
      step_size_mean 새 평균을 얼마나 반영할지 (1이면 그대로 교체)
      step_size_cov  σ를 가중 편차로 얼마나 갱신할지 (0이면 σ 고정), kappa: 분산 바닥값
    반환: (새 mu, 새 sigma, info) info = dict(U, J, w, best)
    """
    eps = rng.standard_normal((K,) + mu.shape) if noise is None else noise[:K]
    U = mu + sigma * eps
    if n_neg:
        U[K - n_zero - n_neg:K - n_zero] = -mu
    if n_zero:
        U[K - n_zero:] = 0.0
    if u_max is not None:
        U = np.clip(U, -u_max, u_max)
    J = cost_fn(U)
    w = softmax_weights(J, beta)
    wb = w.reshape((K,) + (1,) * mu.ndim)
    mu_new = (1 - step_size_mean) * mu + step_size_mean * (wb * U).sum(0)
    sigma_new = sigma
    if step_size_cov > 0:
        var_w = (wb * (U - mu) ** 2).sum(0).mean(0)          # 축별, 시점 평균
        sigma_new = np.sqrt((1 - step_size_cov) * sigma ** 2 + step_size_cov * var_w + kappa)
    return mu_new, sigma_new, dict(U=U, J=J, w=w, best=U[np.argmax(w)].copy())


# ---------------------------------------------------------------------------
# 06 · 2D 점과 장애물
# ---------------------------------------------------------------------------

def point_rollout(p0, v0, U, dt=0.05):
    """2D 점: U[..., T, 2] 가속도열 -> P[..., T+1, 2], V[..., T+1, 2]."""
    U = np.asarray(U, float)
    p = np.broadcast_to(np.asarray(p0, float), U.shape[:-2] + (2,))
    v = np.broadcast_to(np.asarray(v0, float), U.shape[:-2] + (2,))
    P, V = [p], [v]
    for t in range(U.shape[-2]):
        a = U[..., t, :]
        p = p + v * dt + 0.5 * a * dt ** 2
        v = v + a * dt
        P.append(p)
        V.append(v)
    return np.stack(P, axis=-2), np.stack(V, axis=-2)


def obstacle_gap(P, obstacles, robot_r=0.05):
    """각 위치에서 로봇 원 표면과 가장 가까운 장애물 원 표면 사이 간격 [...]. 음수면 겹침."""
    gaps = [np.linalg.norm(P - np.asarray(c), axis=-1) - r - robot_r for c, r in obstacles]
    return np.min(gaps, axis=0)


def point_cost(P, V, U, goal, obstacles, robot_r=0.05, eta=0.1, w_goal=1.0, w_obs=100.0,
               w_u=1e-3, w_term=20.0, w_vterm=1.0, gamma=1.0, per_step=False):
    """2D 점 비용. 시점별 c_t = w_goal|p_t-g|^2 + w_obs C_eta(gap_t) + w_u|u_t|^2, 끝에 terminal 항.

    gamma<1 이면 t번째 비용에 gamma^t를 곱한다(06의 할인). per_step=True면 [..., T] 반환.
    """
    goal = np.asarray(goal, float)
    Pf = P[..., 1:, :]
    d2 = np.sum((Pf - goal) ** 2, -1)
    c = w_goal * d2 + w_u * np.sum(U ** 2, -1)
    if obstacles:
        c = c + w_obs * eta_cost(obstacle_gap(Pf, obstacles, robot_r), eta)
    c[..., -1] += w_term * d2[..., -1] + w_vterm * np.sum(V[..., -1, :] ** 2, -1)
    c = c * gamma ** np.arange(c.shape[-1])
    return c if per_step else c.sum(-1)


# ---------------------------------------------------------------------------
# 07 · 매끄러운 노이즈
# ---------------------------------------------------------------------------

def smooth_noise(K, T, d, n_knots=5, rng=None, halton=False, seed=0):
    """T개 시점을 n_knots개 점만 뽑아 3차 스플라인으로 잇는 노이즈 (K, T, d). 표준편차 약 1.

    halton=True 면 knot 값을 의사난수 대신 Halton 준난수로 뽑는다(07).
    """
    from scipy.interpolate import make_interp_spline
    if halton:
        from scipy.stats import qmc, norm
        u = qmc.Halton(d=n_knots * d, scramble=True, seed=seed).random(K)
        knots = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)).reshape(K, n_knots, d)
    else:
        rng = rng or np.random.default_rng(seed)
        knots = rng.standard_normal((K, n_knots, d))
    z = make_interp_spline(np.linspace(0, 1, n_knots), knots, k=3, axis=1)(np.linspace(0, 1, T))
    return z / z.std()


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------

def plot_scene_2d(ax, goal=None, obstacles=(), robot_r=0.05, eta=None, start=None, lim=None):
    """2D 장면: 장애물(검정), 비용이 켜지는 범위(주황, 반지름+로봇 반지름+η), 시작·목표."""
    import matplotlib.pyplot as plt
    for c, r in obstacles:
        ax.add_patch(plt.Circle(c, r, color='k', alpha=0.6))
        if eta:
            ax.add_patch(plt.Circle(c, r + robot_r + eta, color='orange', alpha=0.15))
    if goal is not None:
        ax.plot(*goal, 'g*', ms=15, label='목표')
    if start is not None:
        ax.plot(*start, 'bo', label='시작')
    ax.set_aspect('equal')
    if lim:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
