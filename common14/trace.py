"""후보 단위 기록이 가능한 MPPI-MPC (18에서 추가). run_mpc_opts와 같은 계산·같은 난수 순서이고, 반복마다 on_iter를 부른다.

기존 run_mpc_opts는 그대로 두고, 후보별 비용·가중치가 필요한 분석에서만 이 함수를 쓴다.
"""
import numpy as np

from . import scene as S
from .mppi import rollout, goal_cost, mppi_step, static_person, human_cost_uniform


def run_mpc_trace(human_cost=human_cost_uniform, goal=S.GOAL, steps=S.STEPS, seed=0, sigma=S.SIGMA, beta=S.BETA,
                  K=S.K, n_iters=S.N_ITERS, person=static_person, on_iter=None):
    """run_mpc_opts(line=False)와 같은 실행. 반복마다 on_iter(step, it, p, v, Pf, U, J, w, h)를 부른다.

    p, v: 지금 로봇 위치·속도, Pf[K,T,2]: 후보 경로의 미래 점, U[K,T,2]: 후보 입력, J[K]: 후보 총비용, w[K]: 정규화 가중치, h[K]: 후보의 사람 비용 합.
    반환 dict: traj, us, person, ess (run_mpc_opts와 같은 의미)
    """
    rng = np.random.default_rng(seed)
    goal = np.asarray(goal, float)
    p, v = S.START.copy(), np.zeros(2)
    mu = np.zeros((S.T, 2))
    traj, us, pers, ess = [p], [], [person(0.0)[0]], []
    for s in range(steps):
        off = person(s * S.DT)[0]
        offs = np.repeat(off[None], S.T, 0)
        last = {}

        def cost(U):
            P, V = rollout(p, v, U)
            h = human_cost(P[..., 1:, :], offs).sum(-1)
            last['h'], last['Pf'] = h, P[..., 1:, :]
            return goal_cost(P, V, U, goal).sum(-1) + h

        for it in range(n_iters):
            mu, info = mppi_step(mu, cost, rng, K=K, sigma=sigma, beta=beta)
            if on_iter is not None:
                on_iter(s, it, p, v, last['Pf'], info['U'], info['J'], info['w'], last['h'])
        ess.append(1.0 / np.sum(info['w'] ** 2))
        u = mu[0]
        p, v = p + v * S.DT + 0.5 * u * S.DT ** 2, v + u * S.DT
        traj.append(p)
        us.append(u)
        pers.append(person((s + 1) * S.DT)[0])
        mu = np.concatenate([mu[1:], mu[-1:]])
    return dict(traj=np.array(traj), us=np.array(us), person=np.array(pers), ess=np.array(ess))
