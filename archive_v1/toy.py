"""01·02 노트북용 2D 점질량 토이 모델.

cuRobo MPPI 설정(control_space: ACCELERATION)과 같은 구조로, 최적화 변수는 가속도 열이고
상태(위치·속도)는 가속도를 적분해서 얻는다. Franka 7관절 대신 2D 점 하나를 쓴다.
"""
import numpy as np

from common import eta_cost


class PointMass2D:
    """p_{k+1} = p_k + v_k dt + 0.5 a_k dt^2,  v_{k+1} = v_k + a_k dt."""

    def __init__(self, dt=0.05, horizon=30, a_max=4.0):
        self.dt, self.N, self.a_max = dt, horizon, a_max
        k = np.arange(horizon)
        # 위치·속도가 가속도에 대해 선형: P = p0 + t*v0 + Mp @ A,  V = v0 + Mv @ A
        self.t = (k + 1) * dt
        self.Mp = np.where(k[:, None] >= k[None, :], dt ** 2 * (k[:, None] - k[None, :] + 0.5), 0.0)
        self.Mv = np.where(k[:, None] >= k[None, :], dt, 0.0)

    def rollout(self, p0, v0, A):
        """A: [N,2] 또는 [B,N,2] 가속도 열 -> (P, V) 같은 앞쪽 shape의 미래 위치·속도."""
        P = p0 + self.t[:, None] * v0 + np.einsum('kj,...jd->...kd', self.Mp, A)
        V = v0 + np.einsum('kj,...jd->...kd', self.Mv, A)
        return P, V

    def step(self, p, v, a):
        """실제로 한 스텝 진행(첫 행동 실행)."""
        return p + v * self.dt + 0.5 * a * self.dt ** 2, v + a * self.dt


class Scene:
    """원형 장애물 + 목표. 비용 구성은 cuRobo arm_reacher와 같은 '항의 합' 구조."""

    def __init__(self, goal, obstacles, robot_radius=0.05, eta=0.1,
                 w_goal=1.0, w_terminal=20.0, w_acc=1e-3, w_obs=100.0, w_vel_terminal=1.0):
        self.goal = np.asarray(goal, float)
        self.obstacles = [(np.asarray(c, float), float(r)) for c, r in obstacles]
        self.robot_radius, self.eta = robot_radius, eta
        self.w_goal, self.w_terminal, self.w_acc = w_goal, w_terminal, w_acc
        self.w_obs, self.w_vel_terminal = w_obs, w_vel_terminal

    def gaps(self, P):
        """각 위치에서 로봇 원 표면과 각 장애물 표면 사이 간격 [..., n_obs]."""
        return np.stack([np.linalg.norm(P - c, axis=-1) - r - self.robot_radius
                         for c, r in self.obstacles], axis=-1)

    def step_costs(self, P, V, A):
        """시점별 비용 [..., N]. 마지막 시점에 terminal 항을 더한다."""
        goal_err = np.sum((P - self.goal) ** 2, axis=-1)
        c = self.w_goal * goal_err + self.w_acc * np.sum(A ** 2, axis=-1)
        c = c + eta_cost(self.gaps(P), self.eta, self.w_obs).sum(-1)
        c[..., -1] += self.w_terminal * goal_err[..., -1] + self.w_vel_terminal * np.sum(V[..., -1, :] ** 2, -1)
        return c

    def cost_and_grad(self, model, p0, v0, A):
        """단일 가속도 열 A[N,2]의 총비용과 해석적 gradient (gradient 기반 MPC용)."""
        P, V = model.rollout(p0, v0, A)
        cost = self.step_costs(P, V, A).sum()
        gP = 2 * self.w_goal * (P - self.goal)
        gP[-1] += 2 * self.w_terminal * (P[-1] - self.goal)
        for c, r in self.obstacles:
            diff = P - c
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            x = self.eta - (dist - r - self.robot_radius)          # x = eta - gap
            dC_dgap = np.where(x > self.eta, -1.0, np.where(x > 0, -x / self.eta, 0.0))
            gP += self.w_obs * dC_dgap * diff / np.maximum(dist, 1e-9)
        gV = np.zeros_like(V)
        gV[-1] = 2 * self.w_vel_terminal * V[-1]
        gA = model.Mp.T @ gP + model.Mv.T @ gV + 2 * self.w_acc * A
        return cost, gA


def draw_scene(ax, scene, start=None, lim=None):
    import matplotlib.pyplot as plt
    for c, r in scene.obstacles:
        ax.add_patch(plt.Circle(c, r, color='k', alpha=0.6))
        ax.add_patch(plt.Circle(c, r + scene.robot_radius + scene.eta, color='orange', alpha=0.15))
    ax.plot(*scene.goal, 'g*', ms=15, label='goal')
    if start is not None:
        ax.plot(*start, 'bo', label='start')
    ax.set_aspect('equal')
    if lim:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])


# ---------------------------------------------------------------------------
# 02 노트북: cuRobo ParallelMPPI 구조를 따른 numpy MPPI
# ---------------------------------------------------------------------------

def halton_normal(n, dim, seed=0):
    """Halton 준난수 -> 역정규 CDF로 가우시안화 (cuRobo HaltonGenerator.get_gaussian_samples와 같은 발상)."""
    from scipy.stats import qmc, norm
    u = qmc.Halton(d=dim, scramble=True, seed=seed).random(n)
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


def sample_noise(n, horizon, d, ratio_halton=0.3, n_knots=5, seed=0):
    """cuRobo sample_ratio {'halton':0.3, 'halton-knot':0.7} 흉내.

    halton      : 시점마다 독립인 가우시안 -> 들쭉날쭉
    halton-knot : n_knots개 knot만 뽑고 B-spline으로 horizon 전체를 보간 -> 매끄러움
    """
    from scipy.interpolate import make_interp_spline
    n_h = int(round(ratio_halton * n))
    raw = halton_normal(n_h, horizon * d, seed).reshape(n_h, horizon, d)
    knots = halton_normal(n - n_h, n_knots * d, seed + 1).reshape(n - n_h, n_knots, d)
    tk, t = np.linspace(0, 1, n_knots), np.linspace(0, 1, horizon)
    smooth = make_interp_spline(tk, knots, k=3, axis=1)(t)
    return np.concatenate([raw, smooth], axis=0)


class MPPI:
    """cuRobo particle_mpc.yml mppi 블록의 항목 이름을 그대로 쓴 축소판.

    후보 구성(particle_opt_base.py:286-289): null_act_frac*N 개 중 절반은 0 행동, 나머지 절반은 -평균.
    마지막 노이즈 샘플은 0 (parallel_mppi.py:582) -> 현재 평균 자체도 후보에 포함.
    """

    def __init__(self, model, scene, num_particles=400, n_iters=5, init_cov=1.0, gamma=0.98, beta=0.1,
                 step_size_mean=0.9, step_size_cov=0.01, kappa=1e-4, null_act_frac=0.05,
                 sample_mode='BEST', fixed_samples=True, seed=0):
        self.model, self.scene = model, scene
        self.n_iters, self.gamma, self.beta = n_iters, gamma, beta
        self.step_size_mean, self.step_size_cov, self.kappa = step_size_mean, step_size_cov, kappa
        self.sample_mode, self.fixed_samples = sample_mode, fixed_samples
        N, d = model.N, 2
        self.null_n = round(int(null_act_frac * num_particles * 0.5))
        self.neg_n = round(int(null_act_frac * num_particles)) - self.null_n
        self.noise_n = num_particles - self.null_n - self.neg_n
        self.rng = np.random.default_rng(seed)
        self.noise = self._new_noise(seed)
        self.gamma_seq = gamma ** np.arange(N)
        self.init_cov = init_cov
        self.mean = np.zeros((N, d))
        self.cov = np.full((1, d), init_cov)          # DIAG_A: 관절(축)별 분산, horizon 공통
        self.best = self.mean.copy()

    def _new_noise(self, seed):
        z = sample_noise(self.noise_n, self.model.N, 2, seed=seed)
        z[-1] = 0.0
        return z

    def sample(self):
        if not self.fixed_samples:
            self.noise = self._new_noise(int(self.rng.integers(1 << 30)))
        acts = [self.mean + self.noise * np.sqrt(self.cov)]
        acts.append(np.repeat(-self.mean[None], self.neg_n, 0))
        acts.append(np.zeros((self.null_n,) + self.mean.shape))
        return np.clip(np.concatenate(acts), -self.model.a_max, self.model.a_max)   # squash_fn: CLAMP

    def weights(self, costs):
        S = (self.gamma_seq * costs).sum(-1) / self.gamma_seq[0]   # jit_compute_total_cost
        z = -S / self.beta
        w = np.exp(z - z.max())
        return S, w / w.sum()                                       # softmax(-S/beta)

    def iterate(self, p, v):
        acts = self.sample()
        P, V = self.model.rollout(p, v, acts)
        costs = self.scene.step_costs(P, V, acts)                  # [B, N]
        S, w = self.weights(costs)
        self.best = acts[np.argmax(w)].copy()
        wa = w[:, None, None]
        new_mean = (wa * acts).sum(0)
        cov_update = (wa * (acts - self.mean) ** 2).sum(0).mean(0, keepdims=True)  # jit_diag_a_cov_update (이전 평균 기준)
        self.mean = (1 - self.step_size_mean) * self.mean + self.step_size_mean * new_mean
        self.cov = (1 - self.step_size_cov) * self.cov + self.step_size_cov * cov_update + self.kappa
        return dict(acts=acts, P=P, S=S, w=w)

    def optimize(self, p, v):
        for _ in range(self.n_iters):
            info = self.iterate(p, v)
        out = self.best if self.sample_mode == 'BEST' else self.mean
        return out.copy(), info

    def shift(self):
        """한 칸 당기고 마지막 칸은 반복(base_action: REPEAT)."""
        self.mean = np.vstack([self.mean[1:], self.mean[-1:]])
        self.best = np.vstack([self.best[1:], self.best[-1:]])


def run_mppi(scene, steps=100, horizon=30, dt=0.05, start=(0.0, 0.0), **kw):
    """MPPI 기반 MPC 루프. 반환: 실행 궤적, 스텝별 info 일부."""
    model = PointMass2D(dt=dt, horizon=horizon)
    opt = MPPI(model, scene, **kw)
    p, v = np.asarray(start, float), np.zeros(2)
    traj, logs = [p], []
    for k in range(steps):
        A, info = opt.optimize(p, v)
        logs.append(dict(ess=1.0 / np.sum(info['w'] ** 2), cov=opt.cov.copy(), Smin=info['S'].min()))
        p, v = model.step(p, v, A[0])
        traj.append(p)
        opt.shift()
    return np.array(traj), logs, opt
