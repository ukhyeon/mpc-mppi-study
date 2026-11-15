"""분절별 방향 유효질량과 환산질량. 관성 행렬은 scene.py에 주어진 값을 그대로 쓴다 (유도는 다루지 않는다)."""
import numpy as np

from .scene import M_TORSO, M_ARM_FREE, ROM_LOCK_DIR, ROM_LOCK_FACTOR, M_ROBOT

INV = {'torso': np.linalg.inv(M_TORSO), 'arm': np.linalg.inv(M_ARM_FREE)}
SEG_NAMES = ['torso', 'arm']


def directional_mass(u, seg):
    """분절 seg를 방향 u로 밀 때의 유효질량 m(u) = 1/(u^T M^-1 u) [kg]. u[..., 2] 단위벡터.

    팔은 u·n > 0 (ROM 잠금 쪽)이면 ROM_LOCK_FACTOR배. 경계 u·n = 0 에서 값이 뛴다(18에서 다룸).
    """
    u = np.asarray(u, float)
    m = 1.0 / np.einsum('...i,ij,...j->...', u, INV[seg], u)
    if seg == 'arm':
        m = np.where(u @ ROM_LOCK_DIR > 1e-9, ROM_LOCK_FACTOR * m, m)   # 1e-9: 경계(길이 방향)는 잠기지 않은 쪽
    return m


def reduced_mass(m_h, m_r=M_ROBOT):
    """환산질량 μ = m_h m_r / (m_h + m_r) [kg]."""
    return m_h * m_r / (m_h + m_r)


def mass_table(angles_deg=(0, 90, 180, 270)):
    """확인용 표: 분절, 방향각, m_h, μ."""
    rows = []
    for seg in SEG_NAMES:
        for a in angles_deg:
            u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
            mh = float(directional_mass(u, seg))
            rows.append((seg, a, mh, float(reduced_mass(mh))))
    return rows
