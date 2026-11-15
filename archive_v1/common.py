"""노트북 공용 헬퍼: 경로, 그림 설정, cuRobo와 같은 식으로 다시 쓴 작은 함수들."""
import json
from pathlib import Path

import numpy as np

HOME = Path.home()
REPO = HOME / 'franka-depth-mppi'          # GitHub ukhyeon/franka-depth-mppi 로컬 클론
CUROBO = HOME / 'curobo'                    # cuRobo 0.7.7 (0a50de1)
TAPE = REPO / 'data' / 'hand-tape-20261012-000004-472steps-registered'
EVIDENCE = REPO / 'evidence'
CONDITIONS = ['R', 'LOW', 'HIGH']
COLORS = {'R': '#888888', 'LOW': '#1f77b4', 'HIGH': '#d62728'}


def setup_plot():
    """한글 폰트(Noto Sans CJK KR)가 있으면 사용하고 기본 스타일을 맞춘다."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    # NotoSansCJK-*.ttc 는 matplotlib에서 첫 face 이름(JP)으로 잡히지만 한글 글리프를 포함한다
    for name in ['Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic', 'Malgun Gothic']:
        if name in names:
            matplotlib.rcParams['font.family'] = name
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3
    return plt


def check_paths():
    for name, p in [('REPO', REPO), ('CUROBO', CUROBO), ('TAPE', TAPE), ('EVIDENCE', EVIDENCE)]:
        print(f'{name:9s} {"OK " if p.exists() else "없음"} {p}')


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# cuRobo 식을 numpy로 옮긴 것 (원본 위치는 각 docstring 참고)
# ---------------------------------------------------------------------------

def dt_array(base_dt, horizon=30, base_ratio=0.5, max_dt=0.04):
    """curobo/rollout/dynamics_model/kinematic_model.py:42 TimeTrajConfig.get_dt_array."""
    dts = [base_dt] * int(base_ratio * horizon)
    dts += np.linspace(base_dt, max_dt, int((1 - base_ratio) * horizon)).tolist()
    if len(dts) != horizon:
        dts.insert(0, dts[0])
    return np.array(dts)


def eta_cost(gap, eta, weight=1.0):
    """충돌 구 표면-장애물 간격 gap[m]에 대한 activation-distance 비용.

    curobolib/cpp/sphere_obb_kernel.cu scale_eta_metric 에서 x = eta - gap 으로 둔 형태:
      gap >= eta      -> 0
      0 <= gap < eta  -> (eta-gap)^2 / (2 eta)
      gap < 0         -> (eta-gap) - eta/2
    """
    x = eta - np.asarray(gap, dtype=float)
    out = np.where(x > eta, x - 0.5 * eta, 0.5 / eta * x ** 2)
    return weight * np.where(x > 0, out, 0.0)


def reduced_mass(m_h, m_r=5.0):
    """experiment/policy.py:reduced_mass."""
    return m_r * m_h / (m_r + m_h)


def eta_policy(m_h, p=0.5, eta_ref=0.025, m_ref=0.6):
    """eta = 0.025 * (mu(m_h)/mu(0.6))**p  [m]. p=0.5 는 policy.py, p=1 은 --eta-exponent 1.0."""
    return eta_ref * (reduced_mass(m_h) / reduced_mass(m_ref)) ** p


def show_code(path, start, end, width=150):
    """파일의 start~end 줄(1부터)을 줄 번호와 함께 출력. 긴 줄은 width에서 자른다."""
    lines = Path(path).read_text().splitlines()
    for i in range(start, min(end, len(lines)) + 1):
        line = lines[i - 1]
        print(f'{i:5d}| {line[:width]}{" ..." if len(line) > width else ""}')
