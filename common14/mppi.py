"""2D 점 로봇 rollout, 비용, MPPI, MPC 루프. 05·06에서 만든 것과 같은 식이다.

- rollout: course.point_rollout과 같은 등가속 적분
- eta_cost: 08의 activation-distance 비용 (겹치기 전 2차, 겹친 뒤 1차)
- mppi_step: course.mppi_step 기본형 (평균 주변 샘플 → 비용 → softmax 가중치 → 가중평균)
"""
import numpy as np

from . import scene as S
from .geometry import segment_gaps, nearest
from .mass import directional_mass, reduced_mass


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


def directional_eta(P, offset, eta_of_mass, reduced=False):
    """가장 가까운 분절의 간격과, 그 분절을 충돌 방향 u로 밀 때의 질량으로 정한 η (16에서 추가).

    u는 geometry.nearest의 정의(로봇 점 → 가장 가까운 분절 무게중심) 그대로다.
    eta_of_mass: 질량 [kg] → η [m] 함수. reduced=True면 환산질량 μ(u), False면 사람 쪽 유효질량 m_h(u).
    가장 가까운 분절이 바뀔 때와 팔의 ROM 경계(u·n = 0)에서 η가 뛴다(고치지 않음, 18에서 다룸).
    반환: gap[...], eta[...], 분절 번호[...] (0 torso, 1 arm), u[..., 2]
    """
    gap, idx, u = nearest(P, offset)
    m = np.where(idx == 0, directional_mass(u, 'torso'), directional_mass(u, 'arm'))
    if reduced:
        m = reduced_mass(m)
    return gap, eta_of_mass(m), idx, u


def human_cost_directional(Pf, offsets, eta_of_mass, reduced=False, w=S.W_HUMAN):
    """방향별 η 사람 비용 [..., T] (16에서 추가). 간격과 η는 directional_eta와 같다."""
    gap, eta, _, _ = directional_eta(Pf, offsets, eta_of_mass, reduced)
    return eta_cost(gap, eta, w)


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


def run_mpc_line(human_cost=human_cost_uniform, goal=S.GOAL, steps=S.STEPS, seed=0, sigma=S.SIGMA,
                 n_iters=S.N_ITERS):
    """서 있는 사람(기준 자세) 앞에서 x축 위로만 움직이는 1D MPC (15에서 추가).

    run_mpc와 같은 루프인데 y 방향 탐색 폭을 0으로 둔다. 그러면 모든 후보의 y 입력이 0이라
    로봇은 경로(y=0)를 벗어나지 못하고, 돌아갈 수 없으니 사람 앞에서 멈추거나 사람을 뚫고 지나간다.
    goal: 목표점 [2], sigma: x 방향 탐색 폭 [m/s²]
    반환 dict: traj[steps+1,2], us[steps,2], person[steps+1,2](모두 0), ess[steps]
    """
    rng = np.random.default_rng(seed)
    goal = np.asarray(goal, float)
    sig = np.array([sigma, 0.0])
    p, v = S.START.copy(), np.zeros(2)
    mu = np.zeros((S.T, 2))
    offs = np.zeros((S.T, 2))
    traj, us, ess = [p], [], []
    for _ in range(steps):
        def cost(U):
            P, V = rollout(p, v, U)
            return (goal_cost(P, V, U, goal) + human_cost(P[..., 1:, :], offs)).sum(-1)

        for _ in range(n_iters):
            mu, info = mppi_step(mu, cost, rng, sigma=sig)
        ess.append(1.0 / np.sum(info['w'] ** 2))
        u = mu[0]
        p, v = p + v * S.DT + 0.5 * u * S.DT ** 2, v + u * S.DT
        traj.append(p)
        us.append(u)
        mu = np.concatenate([mu[1:], mu[-1:]])
    traj = np.array(traj)
    return dict(traj=traj, us=np.array(us), person=np.zeros_like(traj), ess=np.array(ess))


def run_mpc_opts(human_cost=human_cost_uniform, goal=S.GOAL, steps=S.STEPS, seed=0, sigma=S.SIGMA, beta=S.BETA,
                 K=S.K, n_iters=S.N_ITERS, line=False, person=static_person):
    """σ, β, K를 바꿀 수 있는 MPPI-MPC (17에서 추가). 나머지는 run_mpc(predictor='static')와 같다.

    line=True면 y 방향 탐색 폭을 0으로 두어 run_mpc_line과 같은 1D 실행이 된다.
    기본값으로 부르면 run_mpc(static) / run_mpc_line과 같은 결과를 낸다(같은 난수 순서).
    반환 dict: traj[steps+1,2], us[steps,2], person, ess[steps](마지막 반복의 ESS),
              ess_iters[steps, n_iters](반복마다 ESS), human_frac[steps](마지막 반복에서 사람 비용 > 0인 후보 비율),
              J_std[steps](마지막 반복의 후보 비용 표준편차)
    """
    rng = np.random.default_rng(seed)
    goal = np.asarray(goal, float)
    sig = np.array([sigma, 0.0]) if line else sigma
    p, v = S.START.copy(), np.zeros(2)
    mu = np.zeros((S.T, 2))
    traj, us, pers, ess, ess_it, hfrac, jstd = [p], [], [person(0.0)[0]], [], [], [], []
    for s in range(steps):
        off = person(s * S.DT)[0]
        offs = np.repeat(off[None], S.T, 0)
        last = {}

        def cost(U):
            P, V = rollout(p, v, U)
            h = human_cost(P[..., 1:, :], offs).sum(-1)
            last['h'] = h
            return goal_cost(P, V, U, goal).sum(-1) + h

        e = []
        for _ in range(n_iters):
            mu, info = mppi_step(mu, cost, rng, K=K, sigma=sig, beta=beta)
            e.append(1.0 / np.sum(info['w'] ** 2))
        ess.append(e[-1]); ess_it.append(e)
        hfrac.append(float(np.mean(last['h'] > 0))); jstd.append(float(info['J'].std()))
        u = mu[0]
        p, v = p + v * S.DT + 0.5 * u * S.DT ** 2, v + u * S.DT
        traj.append(p)
        us.append(u)
        pers.append(person((s + 1) * S.DT)[0])
        mu = np.concatenate([mu[1:], mu[-1:]])
    return dict(traj=np.array(traj), us=np.array(us), person=np.array(pers), ess=np.array(ess), ess_iters=np.array(ess_it),
                human_frac=np.array(hfrac), J_std=np.array(jstd))
