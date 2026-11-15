# 학습 노트 설계 (v2)

대상: **MPC/MPPI를 처음 배우는 사람.** 다른 자료 없이 노트만 따라가도 마지막에 franka-depth-mppi 프로젝트(깊이 카메라 → 지도 → cuRobo MPPI → Isaac Sim Franka 회피)를 이해할 수 있어야 한다.
필요한 배경: 고등학교 수준의 물리(위치·속도·가속도), 파이썬 기초, numpy 배열.

---

## 1. v1 노트의 문제 진단

| 문제 | 예 | 원인 |
|---|---|---|
| 개념이 이유 없이 등장 | 02 "2. 샘플링 — halton vs halton-knot"가 갑자기 나옴 | 필요(문제)보다 이름(설정값)이 먼저 나옴 |
| 이미 아는 사람용 복습 노트 | 노션 대조표, cuRobo 파라미터 이름이 맨 앞에 나옴 | 독자를 "노션 작성자"로 가정 |
| 한 절에 새 개념이 여러 개 | MPPI 한 iteration 절에 샘플·rollout·softmax·분산 갱신이 한꺼번에 나옴 | 작업 기억 과부하 |
| 배경지식 가정 | gradient, softmax, 분산, FK를 설명 없이 사용 | 선수지식 점검 없음 |
| 읽기만 하는 구조 | 예측·연습·확인 문제가 거의 없음 | 수동적 학습 |
| 토이와 실제 코드 사이 간격이 큼 | 점질량 다음에 바로 cuRobo 소스 줄 번호 | 구체 → 추상 사이 단계가 빠짐 |

## 2. 근거로 삼은 학습 방법

| 방법 | 핵심 | 노트에 적용 |
|---|---|---|
| **문제 먼저** (Kapur, productive failure) | 해법을 배우기 전에 문제를 먼저 겪으면, 무엇을 모르는지 깨닫고 이후 설명이 더 잘 들어온다 | 모든 개념은 "문제 상황 → 실패/불편 확인 → 해결 아이디어 → 이름 붙이기" 순서로 다룬다. **이름(용어·설정값)은 필요가 생긴 뒤에만** |
| **구체 → 추상 점진 이행** (Fyfe 등, concreteness fading) | 구체적 사례로 시작해 그림, 기호 순으로 단계적으로 옮겨 가면 전이가 잘 된다 | 1D 카트 → 2D 점 → 2관절 팔 → 7관절 Franka. 한 개념 안에서도 말·그림 → 숫자 예 → 수식 → 코드 → 실제 cuRobo 이름 순서 |
| **풀이 예제와 점진적 제거** (worked example / fading, 인지부하 이론) | 초보자에게는 완성된 풀이 예제가 문제 풀이보다 효과적이다. 단계를 하나씩 빼며 스스로 풀게 한다 | 앞 노트는 완성된 코드를 읽고 실행하는 방식. 뒤로 갈수록 "빈칸 채우기 → 파라미터 바꾸기 → 직접 작성" |
| **예측 → 실행 → 설명** (PRIMM, Predict-Observe-Explain) | 실행 전에 결과를 예측하게 하면 코드 이해와 성취가 좋아진다 | 핵심 셀 앞에 **예측해 보기** 질문, 실행 뒤에 **결과 해석** |
| **인출 연습·간격** (Dunlosky 2013, Donoghue & Hattie 2021에서 가장 효과가 큰 기법) | 다시 읽기보다 스스로 떠올리는 연습이 기억을 오래 남긴다 | 각 노트 시작에 이전 노트 복습 퀴즈 2~3개, 끝에 확인 문제. 답은 `<details>`로 접어 둔다 |
| **자기 설명** | 자기 말로 설명해 보면 이해가 깊어진다 (근거 수준은 중간) | 절 끝에 "한 문장으로 설명해 보기" |
| **인지부하 관리** | 작업 기억은 작다. 불필요한 정보와 주의 분산을 줄인다 | 한 절에 새 개념 하나. 코드 셀은 짧게(대략 25줄 이하). 그림 그리기 같은 보일러플레이트는 `course.py`로 숨기고, **배우는 알고리즘 자체는 노트에 그대로 보인다** |
| **나선형 반복** | 같은 개념을 점점 복잡한 맥락에서 다시 만난다 | 같은 카트·같은 MPPI 함수를 노트마다 확장해서 재사용 |

## 3. 커리큘럼

| # | 노트 | 핵심 질문 | 새 개념 | 토이 |
|---|---|---|---|---|
| 00 | 시작하기 | 무엇을, 어떤 순서로, 어떻게 공부하나 | 지도, 용어집, 기호표, 실행 환경 | — |
| **A. 기초** | | | | |
| 01 | 로봇을 움직인다는 것 | 입력을 주면 미래가 어떻게 되나 | 상태, 입력(가속도), 모델, 시뮬레이션, rollout, horizon | 1D 카트 |
| 02 | 좋은 움직임을 숫자로 | "좋다"를 어떻게 컴퓨터에게 알려주나 | 비용 함수, 항의 합, 가중치 절충, 최종(terminal) 비용 | 1D 카트 |
| 03 | 가장 좋은 행동열 찾기 | 비용이 가장 작은 입력을 어떻게 찾나 | 최적화, 비용 지형(T=2 등고선), 격자 탐색의 한계, 경사하강, 국소 최솟값 | 1D 카트 |
| **B. MPC** | | | | |
| 04 | MPC: 계획하고 한 발만 | 계획대로 움직이면 왜 실패하나 | 개루프 vs 폐루프, receding horizon, 외란·모델 오차, warm start, horizon 길이 | 1D 카트 |
| **C. MPPI** | | | | |
| 05 | 샘플링으로 최적화하기 (MPPI 핵심) | 미분 없이 최적화할 수 있나 | 무작위 슈팅, 평균 주변 샘플링, 상위 k 평균, softmax 가중치와 온도 β, 반복 갱신 → MPPI | 1D 카트 (T=2 등고선 → T=20) |
| 06 | MPPI로 장애물 피하기 | MPPI를 MPC 루프에 넣으면 | 2D, 장애물 비용, 할인 γ, warm start(shift), 가중평균 vs BEST(두 골짜기 문제), soft cost, 벽 local minimum | 2D 점 |
| 07 | 실전 MPPI의 디테일 | cuRobo는 왜 설정이 이렇게 많나 | 문제→해결 형식: 들쭉날쭉 노이즈→매끄러운 노이즈(knot), 고르지 않은 샘플→Halton, 실행마다 다른 결과→fixed samples, 평균이 틀릴 때→0·−평균 후보, 탐색 폭 붕괴→분산 갱신 속도·κ, 한계 초과→clamp. 끝에 `particle_mpc.yml` 대응표 | 2D 점 |
| **D. 로봇팔** | | | | |
| 08 | 점에서 로봇팔로 | 관절이 있는 로봇은 무엇이 다른가 | 관절 공간 vs 작업 공간, 순기구학(FK), 충돌 구 근사, activation distance η 비용 | 2관절 평면 팔 + MPPI |
| 09 | cuRobo로 Franka 계획하기 | 실제 라이브러리는 같은 일을 어떻게 하나 | 7관절, 61개 충돌 구, buffer, 가변 dt horizon, cuRobo rollout shape `[400,30,7]→costs[400,30]`, 비용 항 분해 | cuRobo (GPU) |
| **E. 프로젝트** | | | | |
| 10 | 로봇은 세상을 어떻게 보나 | 깊이 카메라가 비용이 되기까지 | 깊이 영상→점구름, voxel, occupancy, 거리장(ESDF), 미관측≠빈 공간, 잔상(decay) | 2D 격자 |
| 11 | 손 전용 충돌 비용과 유효질량 | 이 프로젝트는 무엇을 추가했나 | ROI, 관측 패치, 질량→η 정책, 비용 합산 훅, feasible의 의미 | 실제 깊이 데이터 |
| 12 | 실험 결과 읽기 | 결과를 어떻게 해석하나 | R/LOW/HIGH 비교, 이격(buffer 포함/제외), 도달 시간, 추종 오차, 한계 | evidence 로그 |
| 13 | Isaac Sim 실습 | 직접 돌려 보면 | SimulationApp, 물리 dt vs MPC dt, PD 추종, η 바꿔 실행 | `scripts/mpc_lab.py` |

## 4. 노트 템플릿 (모든 노트 공통)

```
# NN · 제목
> 한 줄 요약 (노트를 다 읽은 뒤의 한 문장)

[지도] 전체 커리큘럼에서 지금 위치 (짧은 텍스트 표 또는 화살표)
이 노트를 끝내면 할 수 있는 것: 3개
필요한 것: 이전 노트의 어떤 개념 (링크) · 예상 시간

## 복습 퀴즈 (이전 노트)        ← 01은 생략
2~3문항, 답은 <details>

## 1. (첫 번째 개념)
### 문제            구체적 상황. 왜 이게 필요한지 독자가 느끼게.
### 아이디어        쉬운 말 + 그림/숫자 예. 수식은 아직 없음.
### 수식            필요할 때만. 기호 하나하나 말로 풀이.
### 코드            짧은 풀이 예제. 배우는 알고리즘은 노트에 직접 쓴다.
**예측해 보기:** 실행 전 질문 (답은 실행 결과로 확인)
(코드 셀)
**결과 해석:** 무엇을 봐야 하는지
**직접 해보기:** 값 하나 바꿔서 다시 실행 (예상 결과는 <details>)
**한 문장으로:** 자기 설명 프롬프트

## 2. ...

## 핵심 정리          3~5줄
## 확인 문제          3~5문항, 답은 <details>
## 실제 프로젝트에서는   (선택) franka-depth-mppi / cuRobo에서 이 개념이 어디 있는지. 짧게. 파일:줄
## 다음 노트          링크 + 왜 그 다음인지 한 줄
```

## 5. 작성 규칙

- **이름은 필요 뒤에:** 용어·설정값 이름(`halton`, `step_size_cov`, `BEST`…)은 그 장치가 해결하는 문제를 먼저 보여 준 뒤에만 쓴다. 처음 나오는 용어는 **굵게** + 괄호에 영어 + 쉬운 정의 한 줄.
- **한 절에 새 개념 하나.** 새 개념이 둘이면 절을 나눈다.
- **기호 고정** (00의 기호표와 일치):
  - 위치 `p`, 속도 `v`, 상태 `x = (p, v)`, 입력(가속도) `u`, 시간 간격 `Δt`
  - horizon 길이 `T` (스텝 수), 시점 `t = 0..T-1`, 입력열 `U = (u_0, …, u_{T-1})`
  - 비용 `J(U)`, 시점별 비용 `c_t`, 가중치(비용 항) `w_goal` 등
  - 샘플 수 `K`, 샘플 번호 `k`, 샘플 가중치 `ω_k` (비용 항 가중치 `w`와 구분), 온도 `β`, 할인 `γ`
  - 평균 입력열 `μ`, 표준편차 `σ` (분산 `σ²`)
- **수식은 직관 다음.** 수식 아래에 "말로 하면:" 한 줄.
- **코드:** 셀 하나에 하나의 일. 25줄 안팎. 변수 이름은 기호와 맞춘다(`U`, `J`, `beta`). 그림 보일러플레이트는 `course.py`의 plot 함수를 쓴다. 배우는 알고리즘(시뮬레이션, 비용, 경사하강, MPC 루프, MPPI 갱신)은 처음 나오는 노트에서 **노트 안에 직접** 작성하고, 이후 노트에서는 `course.py`의 같은 코드를 import한다("05에서 만든 `mppi_step`과 같은 코드").
- **실행 시간:** 노트 전체 실행 2분 이내 권장(09의 cuRobo, 13의 Isaac Sim 실행 셀 제외).
- **말투:** 담백한 한국어 설명체("~다"). 과장·감탄·이모지 없음. 번역투·접속사 남발 피하기.
- **그림:** 모든 핵심 개념에 그림 하나. 축 이름과 단위를 넣는다. 한글 라벨 사용 가능(`setup_plot()`).
- **실제 프로젝트 연결은 끝에만.** 본문은 개념 학습에 집중하고, cuRobo/레포의 파일:줄은 "실제 프로젝트에서는" 절에 모은다. 로그 수치를 쓸 때는 레포 evidence에서 직접 계산한다.
- 검증된 셀에 `!`/`%` 매직 금지. 출력은 `print`.

## 6. 공용 모듈

- `common.py`: 경로(`REPO`, `CUROBO`, `TAPE`, `EVIDENCE`), `setup_plot`, `show_code`, `read_jsonl`, `load_yaml`, `eta_cost`, `dt_array`, `eta_policy`
- `course.py`: 노트 01~07에서 직접 만든 코드의 "정리본". 이후 노트가 import한다. 함수 목록은 파일 docstring 참고.

## 참고한 자료

- Kapur, productive failure 요약: https://evidencebasedliving.human.cornell.edu/blog/to-better-learn-students-must-first-fail
- Fyfe et al., concreteness fading: https://pc.cogs.indiana.edu/?p=885 , https://learningscientists.org/blog/2018/2/1-1
- Worked-example effect / fading: https://en.wikipedia.org/wiki/Worked-example_effect , https://faculty.engineering.asu.edu/mre/wp-content/uploads/sites/31/2020/02/Stat_Adapt_Fad_JEE06.pdf
- PRIMM: https://www.raspberrypi.org/blog/primm-talk-in-programming-lessons-research-seminar/ , https://textbooks.cs.ksu.edu/tlcs/4-designing-cs-lessons/07-primm
- Dunlosky et al. 2013: https://es.aft.org/sites/default/files/dunlosky_0.pdf ; Donoghue & Hattie 2021: https://www.frontiersin.org/articles/10.3389/feduc.2021.581216/full
- Teaching MPC (IEEE CSM 2024 / arXiv 2106.00944): https://arxiv.org/pdf/2106.00944v1
- MPPI 튜토리얼: Kazim et al. arXiv 2309.12566 https://arxiv.org/abs/2309.12566v2 ; Honda, MPC via probabilistic inference https://arxiv.org/abs/2511.08019v1
- Rule et al., Ten simple rules for Jupyter notebooks: https://scil.ucsd.edu/publications/ten-simple-rules-jupyter-notebooks
