"""장면 확인 그림을 만든다: python.sh common14/scene_check.py  →  common14/scene_check.png"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
from common14 import *  # noqa: E402,F403
from common14 import scene as S  # noqa: E402

plt = setup_plot()
fig = plt.figure(figsize=(18, 4.8))

ax = fig.add_subplot(1, 4, 1)
draw_scene(ax, eta=S.ETA0, lim=((-0.15, 2.15), (-0.7, 0.7)))
for seg in S.segments():
    ax.plot(*seg['com'], 'rx', ms=8)
ax.annotate('', xy=(1.05, -0.08), xytext=(1.05, -0.32), arrowprops=dict(arrowstyle='->', color='m', lw=2))
ax.text(0.55, -0.45, '아래에서 위로 밀면(u·n>0) 팔 ROM 잠금', color='m', fontsize=8)
ax.plot([], [], 'rx', label='분절 무게중심')
ax.legend(fontsize=8, loc='upper left')
ax.set_title('장면 (주황 = 비용이 켜지는 범위, η=%.2f m)' % S.ETA0, fontsize=10)

ax2 = fig.add_subplot(1, 4, 2, projection='polar')
polar_mass(ax2, 'torso', color='C0')
ax2.set_title('몸통 m_h(u) [kg]: 거의 등방', fontsize=10)

ax3 = fig.add_subplot(1, 4, 3, projection='polar')
polar_mass(ax3, 'arm', color='C1')
ax3.set_title('팔 m_h(u) [kg]\n위쪽 반(u·n>0)이 4배: ROM 잠금 흉내', fontsize=10)

ax4 = fig.add_subplot(1, 4, 4, projection='polar')
polar_mass(ax4, 'torso', reduced=True, label='몸통', color='C0')
polar_mass(ax4, 'arm', reduced=True, label='팔', color='C1')
ax4.set_ylim(0, 5)
ax4.set_title('환산질량 μ(u) [kg], 로봇 m_r = %.0f kg\n(큰 질량일수록 대비가 줄어든다)' % S.M_ROBOT, fontsize=10)
ax4.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.2, -0.15))
fig.text(0.5, 0.01, 'θ = 충돌 방향 u(로봇 → 분절 무게중심)의 각도.  예: 로봇이 팔 바로 아래에 있으면 u는 위쪽(θ=90°)', ha='center', fontsize=9)

fig.tight_layout()
fig.savefig(HERE / 'scene_check.png', dpi=110)
print('saved', HERE / 'scene_check.png')
for seg, a, mh, mu in mass_table():
    print(f'{seg:6s} θ={a:3d}°  m_h={mh:6.2f} kg  μ={mu:5.3f} kg')
