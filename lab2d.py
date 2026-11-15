"""06·07 노트용: 2D 점 MPPI-MPC 루프. 06 노트에서 직접 작성한 mppi_mpc_2d와 같은 코드다."""
import numpy as np

from course import point_rollout, point_cost, mppi_step

DT2 = 0.05   # 2D 점의 시간 간격 [s]


def mppi_mpc_2d(goal, obstacles, start=(0.0, 0.0), v0=(0.0, 0.0), T=30, K=400, beta=1.0, sigma=2.0,
                steps=80, gamma=1.0, warm=True, use_best=False, iters=1, seed=0, u_max=4.0,
                noise_fn=None, n_zero=0, n_neg=0, step_size_mean=1.0, step_size_cov=0.0, kappa=0.0,
                w_obs=100.0, eta=0.1, snapshot_steps=()):
    """매 스텝: 현재 상태에서 MPPI 갱신 iters번 → 첫 입력 실행 → 평균을 한 칸 shift.

    use_best=False: 가중평균 mu의 첫 입력 실행 / True: 가장 좋은 후보(best)의 첫 입력 실행
    noise_fn(step, it) -> (K, T, 2) 노이즈 (07: 매끄러운 노이즈·고정 샘플). None이면 매번 새 정규 노이즈.
    반환: dict(traj[steps+1,2], us[steps,2], sigma, snaps={step: info})
    """
    rng = np.random.default_rng(seed)
    p, v = np.array(start, float), np.array(v0, float)
    mu = np.zeros((T, 2))
    sig = np.full(2, float(sigma))
    traj, us, snaps = [p], [], {}
    for s in range(steps):
        def cost(U):
            P, V = point_rollout(p, v, U, DT2)
            return point_cost(P, V, U, goal, obstacles, eta=eta, w_obs=w_obs, gamma=gamma)
        for it in range(iters):
            noise = noise_fn(s, it) if noise_fn else None
            mu, sig, info = mppi_step(mu, sig, cost, K, beta, rng, u_max=u_max, noise=noise, n_zero=n_zero,
                                      n_neg=n_neg, step_size_mean=step_size_mean,
                                      step_size_cov=step_size_cov, kappa=kappa)
        if s in snapshot_steps:
            snaps[s] = dict(info, p=p.copy(), v=v.copy(), mu=mu.copy())
        u = info['best'][0] if use_best else mu[0]
        p, v = p + v * DT2 + 0.5 * u * DT2 ** 2, v + u * DT2
        traj.append(p)
        us.append(u)
        mu = np.concatenate([mu[1:], mu[-1:]]) if warm else np.zeros_like(mu)
    return dict(traj=np.array(traj), us=np.array(us), sigma=sig, snaps=snaps)


def arrival_time(traj, goal, tol=0.05):
    """목표 tol 안에 처음 들어간 시각 [s]. 못 들어가면 None."""
    d = np.linalg.norm(traj - np.asarray(goal), axis=1)
    idx = np.nonzero(d < tol)[0]
    return round(idx[0] * DT2, 2) if len(idx) else None
