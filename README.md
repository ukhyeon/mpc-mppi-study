# MPC · MPPI 학습 노트

MPC와 MPPI를 처음부터 배워서, 연구 프로젝트 **유효질량 기반 사람 회피 MPC**의 제어·비용을 설계하기 위한 Jupyter 학습 노트입니다. 노트만 순서대로 따라가면 되도록 구성했습니다. 모든 노트에는 실행 결과와 그림이 저장되어 있어 실행하지 않고도 읽을 수 있습니다.

시작은 [00_start_here.ipynb](00_start_here.ipynb)입니다. 지도, 공부 방법, 용어집, 기호표, 실행 환경이 정리되어 있습니다.

## 구성

| 파트 | 노트 | 내용 |
|---|---|---|
| A. 기초 | 01~03 | 상태·입력·모델·rollout, 비용 함수, 최적화 (1D 카트) |
| B. MPC | 04 | receding horizon, 외란, horizon 길이, warm start |
| C. MPPI | 05~07 | 샘플링 최적화와 softmax 가중치, 2D 장애물 회피, 실전 디테일 (cuRobo 설정과 대응) |
| D. 로봇팔 | 08~09 | 2관절 팔 순기구학·충돌 구·η 비용, cuRobo Franka rollout |
| E. 선행 프로젝트 | 10~13 | [franka-depth-mppi](https://github.com/ukhyeon/franka-depth-mppi): 깊이 지도, 손 전용 비용과 질량 → η, 결과 분석, Isaac Sim 실습 |
| F. 새 연구 설계 | 14~22 | 평면 사람 장면에서 요인 하나씩: 움직이는 사람(14), η와 w(15), 방향별 질량(16), MPPI 해상도와 ESS(17), 비용 불연속, 필요 이격 거리, 안전 필터, 지연, cuRobo 확인(18~22, 작성 예정) |

| 파일·폴더 | 내용 |
|---|---|
| `common.py`, `course.py`, `arm2d.py`, `lab2d.py`, `grid2d.py`, `realdata.py` | 01~13 공용 코드 (`course.py`는 01~07에서 직접 만든 코드의 정리본) |
| `common14/` | 14~22 공통 모듈: 고정 장면, 2D 점 MPPI, 분절 질량, 지표. 수치와 근거는 [common14/README.md](common14/README.md) |
| `scripts/` | 13의 Isaac Sim 실습 스크립트 `mpc_lab.py`와 실행 결과 |
| `_design/curriculum.md` | 노트 설계 근거 (학습 방법 조사, 템플릿, 작성 규칙) |
| `archive_v1/` | 첫 버전 노트 (노션 학습 기록과 코드 대조 포함) |
| `kernel_launcher.py` | Jupyter 커널 실행기 |

## 실행 환경

확인한 환경: Ubuntu, Isaac Sim 4.5.0 내장 Python 3.10 (torch 2.5.1+cu118), cuRobo 0.7.7, RTX 4060 Ti.

1. **외부 저장소** (노트가 데이터·코드를 읽는다)
   - `~/franka-depth-mppi`: [franka-depth-mppi](https://github.com/ukhyeon/franka-depth-mppi) 클론. 10~12가 기록 깊이 데이터와 실험 로그를 쓴다.
   - `~/curobo`: cuRobo 0.7.7. 09와 일부 셀이 설정 파일·소스를 읽는다.
2. **Jupyter 커널** — Isaac Sim 내장 파이썬을 Isaac Sim 자체는 건드리지 않고 커널로 쓴다.

   ```bash
   cd ~/isaacsim && ./python.sh -m pip install --target <이 폴더>/.kernel_pkgs ipykernel
   mkdir -p ~/.local/share/jupyter/kernels/isaacsim-curobo
   cat > ~/.local/share/jupyter/kernels/isaacsim-curobo/kernel.json <<'EOF'
   {
     "argv": ["<isaacsim 경로>/python.sh", "<이 폴더>/kernel_launcher.py", "-f", "{connection_file}"],
     "display_name": "Isaac Sim 4.5 + cuRobo",
     "language": "python",
     "env": {"PYTHONUTF8": "1"},
     "metadata": {"debugger": false}
   }
   EOF
   ```

   VSCode에서 노트를 열면 "Isaac Sim 4.5 + cuRobo" 커널이 선택된다. 01~08, 10~12, 14~21은 numpy·scipy·matplotlib만 있으면 다른 커널에서도 돈다.
3. **GPU가 필요한 노트:** 09(cuRobo), 13의 실행 셀(Isaac Sim), 22(cuRobo, 작성 예정).

노트 출력의 경로는 홈 디렉터리 기준(`~/...`)으로 바꿔 두었다.

## 출처와 주의

- 일부 노트(07·09·12·13과 archive_v1)의 출력에는 개념 설명을 위해 [cuRobo](https://github.com/NVlabs/curobo) 소스와 설정 파일(`particle_mpc.yml` 등)의 짧은 발췌가 들어 있다. 해당 코드의 저작권과 라이선스는 NVIDIA에 있다.
- 실험 데이터(깊이 기록, evidence 로그)는 이 저장소에 포함하지 않았다. franka-depth-mppi 저장소에서 읽는다.
