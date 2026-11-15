"""2D 점 로봇 rollout, 비용, MPPI, MPC 루프. 05·06에서 만든 것과 같은 식이다.

- rollout: course.point_rollout과 같은 등가속 적분
- eta_cost: 08의 activation-distance 비용 (겹치기 전 2차, 겹친 뒤 1차)
- mppi_step: course.mppi_step 기본형 (평균 주변 샘플 → 비용 → softmax 가중치 → 가중평균)
"""
import numpy as np

from . import scene as S
from .geometry import segment_gaps


def rollout(p0, v0, U, dt=S.DT):
    """U[..., T, 2] 가속도열 → P[..., T+1, 2], V[..., T+1, 2]."""
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


def eta_cost(gap, eta, weight=1.0):
    """08의 η 비용: gap>=η → 0, 0<=gap<η → (η-gap)²/(2η), gap<0 → (η-gap)-η/2. eta는 gap과 broadcast."""
    eta = np.asarray(eta, float)
    x = eta - np.asarray(gap, float)
    out = np.where(x > eta, x - 0.5 * eta, 0.5 / eta * x ** 2)
    return weight * np.where(x > 0, out, 0.0)


def goal_cost(P, V, U, goal=S.GOAL):
    """목표 비용 [..., T] (course.point_cost의 장애물 제외 부분과 같음)."""
    Pf = P[..., 1:, :]
    d2 = np.sum((Pf - goal) ** 2, -1)
    c = S.W_GOAL * d2 + S.W_U * np.sum(U ** 2, -1)
    c[..., -1] += S.W_TERM * d2[..., -1] + S.W_VTERM * np.sum(V[..., -1, :] ** 2, -1)
    return c


def human_cost_uniform(Pf, offsets, eta=S.ETA0, w=S.W_HUMAN):
    """가장 가까운 분절과의 간격에 균일 η 비용 [..., T]. offsets[T, 2]: 시점별 사람 위치(예측)."""
    g = segment_gaps(Pf, offsets).min(-1)
    return eta_cost(g, eta, w)


def softmax_weights(J, beta):
    z = np.exp(-(J - J.min()) / beta)
    return z / z.sum()


def mppi_step(mu, cost_fn, rng, K=S.K, sigma=S.SIGMA, beta=S.BETA, u_max=S.U_MAX):
    """05의 MPPI 한 번 갱신. 반환: 새 평균, info(U, J, w, best)."""
    U = np.clip(mu + sigma * rng.standard_normal((K,) + mu.shape), -u_max, u_max)
    J = cost_fn(U)
    w = softmax_weights(J, beta)
    mu_new = np.einsum('k,k...->...', w, U)
    return mu_new, dict(U=U, J=J, w=w, best=U[np.argmax(w)].copy())


def run_mpc(person, predictor='static', human_cost=human_cost_uniform, steps=S.STEPS, seed=0,
            n_iters=S.N_ITERS, snapshot_steps=()):
    """사람이 움직이는 장면에서 MPPI-MPC를 돌린다.

    person: 시각 t[s] → (사람 offset[2], 속도[2]) 를 돌려주는 함수 (참 움직임)
    predictor: rollout의 미래 30칸에서 사람을 어디에 둘지
        'static' 지금 위치에 그대로 / 'cv' 지금 속도로 등속 이동 / 'oracle' 참 미래 위치
    human_cost(Pf[K,T,2], offsets[T,2]) → [K,T]
    반환 dict: traj[steps+1,2], us[steps,2], person[steps+1,2], ess[steps], snaps
    """
    rng = np.random.default_rng(seed)
    p, v = S.START.copy(), np.zeros(2)
    mu = np.zeros((S.T, 2))
    tau = (np.arange(S.T) + 1) * S.DT                       # 미래 칸의 시각 (지금부터)
    traj, us, pers, ess, snaps = [p], [], [person(0.0)[0]], [], {}
    for s in range(steps):
        t = s * S.DT
        off, vel = person(t)
        if predictor == 'static':
            offs = np.repeat(off[None], S.T, 0)
        elif predictor == 'cv':
            offs = off + tau[:, None] * vel
        elif predictor == 'oracle':
            offs = np.stack([person(t + x)[0] for x in tau])
        else:
            raise ValueError(predictor)

        def cost(U):
            P, V = rollout(p, v, U)
            return (goal_cost(P, V, U) + human_cost(P[..., 1:, :], offs)).sum(-1)

        for _ in range(n_iters):
            mu, info = mppi_step(mu, cost, rng)
        ess.append(1.0 / np.sum(info['w'] ** 2))
        if s in snapshot_steps:
            snaps[s] = dict(p=p.copy(), offs=offs.copy(), off=off.copy(), mu=mu.copy(),
                            plan=rollout(p, v, mu[None])[0][0])
        u = mu[0]
        p, v = p + v * S.DT + 0.5 * u * S.DT ** 2, v + u * S.DT
        traj.append(p)
        us.append(u)
        pers.append(person(t + S.DT)[0])
        mu = np.concatenate([mu[1:], mu[-1:]])               # warm start (shift)
    return dict(traj=np.array(traj), us=np.array(us), person=np.array(pers), ess=np.array(ess), snaps=snaps)


def static_person(t):
    """움직이지 않는 사람 (15~18의 기본)."""
    return np.zeros(2), np.zeros(2)
