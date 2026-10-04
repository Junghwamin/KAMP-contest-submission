이 폴더는 보고서 「제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석」의 모든 실험을 KAMP 터미널에서 처음부터 다시 계산하고, 그 결과를 보고서 순서대로 보여 주는 소스코드 패키지다.

# KAMP 자원최적화 — 실행 패키지

**결론부터**: 명령 세 줄(`bash check_env.sh` → `bash setup_pod.sh --with-tests` → `python run_all.py`)이면
데이터 적재부터 모델 학습·평가·피크 저감 시뮬레이션·모델 저장·서빙 검증까지 한 번에 끝난다.
끝나면 `python show_results.py` 로 결과를 **보고서 절 순서대로** 넘겨 본다. 노트북은 쓰지 않는다.

---

## 1. 과제와 결과 요약

**과제** — 제6회 K-인공지능 제조데이터 분석 경진대회 과제 ⑤(자원 최적화).
선박엔진용 볼트·너트 제조공장 한 곳의 2021-01-01 ~ 09-14 시간별 데이터(6,168행 × 18열, `data/`)로
① 다음 날 24시간의 시간별 전력(평균 `y_avg`, 15분 최대수요 `y_peak`)을 예측하고
② 최대수요 피크(`y_peak ≥ θ = 187 kW`) 위험 시각을 미리 탐지한다. 테스트는 2021-09-01 ~ 09-14 의 336시간이다.

**수정본 평가 결과** — FULL 62단계가 생략 없이 완료됐고 필수 점검 15개·보조 점검 4개, 번들 검증·리플레이·123개 테스트가 통과했다. 아래 값은 기존 2021년 9월 테스트를 수정한 코드로 재평가한 결과이며, 새로운 외부 독립 데이터로 검증한 것은 아니다. 근거는 `outputs/steps/index.json`과 생성표다.

| 항목 | 확인할 내용 | 원본 표 |
|---|---|---|
| 최종모델 | **2단계 레짐(3분류)**, OOF MAE **11.044kW** | `outputs/tables/ch2_scorecard.csv`·`ch2_regression.csv` |
| 테스트 MAE | **5.310kW**, 보정 RF **6.329kW** 대비 **1.019kW(16.11%) 감소** | `ch2_regression.csv`·`ch2_significance.csv` |
| 실제 피크 시간 오차 | 28시간의 평균전력 Peak-MAE **8.144kW**, 최대전력 Peak15-MAE **10.245kW** | `ch2_regression.csv` |
| 개선의 통계적 유의성 | 기본 95% CI **[0.008, 2.527]kW**, 사후 보조 90% CI **[0.099, 2.257]kW**, 양측 근사 **p=0.046** | `ch2_significance.csv` |
| 피크 탐지 | **정탐 23·오경보 43·미탐 5**, Recall **0.8214**, F1 **0.4894**, τ 약 **173.830kW** | `ch2_peak_detection.csv` |
| 확률 보정 | 시간순 OOF 991시간: Brier **0.08475→0.07745**, ECE **0.07290→0.05747** | `ch5_calibration.csv` |
| 배치 추론 | 학습된 모델의 336시간 예측 **0.0307초**(10회 중앙값, 해당 실행 환경) | `outputs/steps/41_6.4.txt` |
| 현장시험 후보 | 기동 분산 L1c 우선, L2 추가는 일정변경 비용과 추가 절감 비교 후 검토 | `ch4_levers.csv`·`ch4_scenarios.csv` |

### 이번 수정의 평가 기준

- **전처리**: 결측 전력·풍속·공장인원을 채울 때 관측 원점 이후의 값을 사용하지 않는 과거값 전파(`ffill`)를 적용한다. 양방향 보간으로 미래 값을 가져오던 경로를 제거했다. 학습·서빙의 처리 규칙은 함께 유지해야 한다.
- **피크 오차**: 실제 `y_peak ≥ θ`인 시간을 기준으로 평균전력 오차 `Peak-MAE`와 최대전력 오차 `Peak15-MAE`를 계산한다. 테스트에서는 28시간이다. 기존 보고서의 평균전력 `y_avg ≥ θ` 조건(테스트 2시간)은 별도 고부하 지표로 남긴다. 서로 다른 표본의 값을 같은 지표처럼 비교하지 않는다.
- **유의성**: 일 단위 블록 Bootstrap 2,000회, 난수시드 42를 사용한다. MAE 감소량은 `보정 RF − 최종모델`이며 양수가 개선이다. 기존 95% 신뢰구간을 기본 판정으로 유지하고 90% 신뢰구간은 사후 보조분석으로 병기한다. 새 예측이 생성되면 두 구간과 p값도 다시 계산한다.
- **학습·보정**: HPO의 `bagging_freq=1`을 최종 학습에도 보존하고 탐색과 최종 학습의 `n_estimators`를 일치시킨다. 피크 확률 보정은 앞선 OOF 구간으로 학습해 뒤 구간을 평가하는 시간순 교차적합으로 확인한 후, 서빙용 보정기를 전체 OOF에 적합한다.
- **비용**: 전력량요금 절감은 산정하지 않아 0원으로 둔다. 기본요금 절감·인건비 증가·순절감은 동일한 관측기간(2021-01-01~09-14, 월수 환산 `8 + 14/30`)으로 계산하며, 12개월 기본요금 환산액은 별도 참고 열이다. 부하를 옮긴 뒤 관측 최대 222kW를 넘는 새 피크도 허용해 계산하고 초과 여부·시간 수를 표에 표시한다. 이는 실제 청구서 검증 결과가 아니다.

기존 보고서 및 이전 Word 수정본의 MAE 5.220kW·p=0.067·13.56% 개선·26/28건 탐지·오경보 45건·90% CI [0.061, 1.703]kW와 이전 예측 해시는 **수정 전 결과**다. 현재 재계산 값과 섞어 사용하지 않는다.

### 비용 시나리오 결과

아래 값은 FULL 8장 생성표(`ch4_scenarios.csv`·`ch4_levers.csv`)와 독립 계산을 대조해 일치함을 확인했다. 관측기간은 257일이며 월수는 `8 + 14/30 = 8.4667개월`로 환산한다.

| 시나리오 | 최대수요 감소 | 관측기간 기본요금 절감 | 관측기간 인건비 증가 | 관측기간 순절감 | 별도 연간 기본요금 환산액 |
|---|---:|---:|---:|---:|---:|
| S1c: 기동 분산 L1c | 13.0kW | 915,755원 | 0원 | 915,755원 | 1,297,920원 |
| S2: L1c + 생산시간 이동 L2 | 13.1kW | 924,176원 | 0원 | 924,176원 | 1,309,856원 |
| S4: S2 + 야간 이전 | 13.2kW | 929,229원 | 94,720원 | 834,509원 | 1,317,017원 |

순절감 기준 최선은 S2이고 최대수요 감소량 기준 최선은 S4다. S2를 설명하면서 S4의 13.2kW를 사용하지 않는다. L1c에 L2를 추가한 연간 기본요금 환산 이익은 11,936원에 불과하므로 L1c 현장시험을 우선 제안하고 L2는 일정변경 비용을 확인한 뒤 검토한다.

전력량요금 절감은 산정에서 제외(0원)했으며, 표의 인건비 0원은 가정한 교대 추가비용이 없다는 뜻이다. 설비 투자·정지·일정 조정 등 실행비용은 측정하지 않았다. 사후 가정 분석의 추정액이므로 실제 한전 청구 절감액이나 실증된 운영효과로 인용하지 않는다.

**정직하게 밝혀 둘 한계**

1. **현재 계산에서는 RF 대비 MAE 감소량의 95% 구간이 0을 포함하지 않아 기존 5% 기준을 충족한다(p=0.046).** 다만 테스트는 14일(일 단위 블록 14개)로 제한되며 동일 테스트를 재평가한 결과다. 90% 구간은 사후 보조분석으로 유지한다.
   모델 선정에는 테스트 MAE 하나가 아니라 교차검증(OOF) 오차·피크 오차·탐지 성능·fold 편차를 합친 스코어카드를 사용한다. OOF를 모델 선택에도 썼으므로 완전히 독립적인 검증은 아니다.
2. **데이터의 상당 부분이 합성 복제일이다.** 전력 프로파일이 완전히 같은 날이 45개 그룹, 잉여 115일이다(기온은 다르다).
   복제일을 뺀 조건(D2)에서는 레짐 2분류 MAE 12.693kW가 1위, 3분류 12.828kW가 2위였다. D1·D2 전체 순위상관은 0.9429지만 최종모델의 1위가 그대로 유지되지는 않았다. 일반화가 완전히 입증됐다는 뜻이 아니다(`ch2_d1_vs_d2.csv`·`ch6_rank_preservation.csv`).
3. **달력 규칙 한 줄(평일 08~18시)도 비교 기준이다.** 미탐지와 오경보의 교환관계를 Recall·Precision·F1과 함께 확인한다(`ch2_peak_detection.csv`).
4. **과거 222kW 청구 기준이 유지된다는 시나리오 가정에서는 9월 테스트 구간만 낮춰도 기본요금 절감이 0원이다.** 실제 요금제 일반론이 아니며, 저감 시뮬레이션은 전체 기간 실측에 같은 가정을 적용한다.
5. **장기 휴무의 과대예측과 가동일의 과소예측을 구분한다.** `ch3_failures.csv`의 예측오차 부호와 `ch2_gate_diagnosis.csv`의 상태분류를 함께 확인한다. 조건만으로 오차 방향을 단정하지 않는다.
6. **저감 시뮬레이션은 과거 실측 부하를 이용한 사후 시나리오다.** 예측 경보의 미탐지·오경보를 반영한 실증 절감이나 자동 일정 최적화를 완료한 결과가 아니다. 저장 모델은 예측·경보를 출력하고, 조정안은 현장 담당자가 검토한다.
7. **사전 생산·휴무계획에 대한 의존성이 크다.** 생산 관련 직접변수 4개만 제외하면 OOF MAE는 11.044→12.707kW였지만, 여기서 유도한 가동·휴무 달력변수 4개까지 총 8개를 제외하면 18.374kW로 높아졌다. 현재 테스트 성능이 실제 운영에서 미래 계획을 확보할 수 있음을 증명하지는 않는다([계획정보 제거 비교표](outputs/tables/ch2_plan_information_ablation.csv)).

---

## 2. KAMP 에서 실행 — 3단계

아래 명령은 **한 줄씩 복사해 붙여 넣는다.** (`~` 는 홈 폴더)

**0) 업로드한 zip 풀기** — JupyterHub 파일 창에 `kamp_upload.zip` 을 올린 뒤 터미널에서:

```bash
cd ~ && python -m zipfile -e kamp_upload.zip . && cd ~/KAMP_정종묵
```

**1) 진단** — 읽기만 한다. 끝에 '종합 판정'이 나온다.

```bash
bash check_env.sh
```

**2) 준비** — 없는 패키지(보통 optuna·shap)와 pytest 만 설치한다. 이미 있는 numpy·TensorFlow 등은 건드리지 않는다.

```bash
bash setup_pod.sh --with-tests
```

**3) 실행**

(선택) 먼저 **복사본에서** 빠른 동작 확인(FAST). 수치는 최종값이 아니다. 끝나면 원래 폴더로 돌아온다.

```bash
rm -rf ~/kamp_fast && cp -r ~/KAMP_정종묵 ~/kamp_fast && cd ~/kamp_fast && python run_all.py --fast && python show_results.py; cd ~/KAMP_정종묵
```

본 실행(FULL). 터미널·브라우저를 닫아도 계속 돌도록 `nohup`으로 띄운다(JupyterHub 서버 자체가 멈추면 함께 멈춘다 — 9절). 소요시간과 캐시 조건은 5절을 참고한다.
두 줄을 차례로 붙여 넣는다(`cd` 를 `&` 줄에 붙이면 지금 터미널의 폴더가 바뀌지 않는다).

```bash
cd ~/KAMP_정종묵
nohup python -u -X utf8 run_all.py > ~/full_run.out 2>&1 < /dev/null &
```

진행 상황 보기 (`Ctrl+C` 는 **보기만** 멈추고 실행은 계속된다):

```bash
tail -f ~/full_run.out
```

**성공 신호**: 로그의 마지막 줄이 `RUN_ALL_OK mode=FULL` 이다.

**4) 결과 보기** — 성공 신호를 확인한 뒤, 보고서 순서(요약 → 1.1 … 6장)로 한 쪽씩 넘긴다. Enter 다음 쪽, `q` 끝.
명령 앞의 `cd ~/KAMP_정종묵 &&` 는 FAST 복사본이나 다른 폴더에서 열리지 않게 하는 것이다. 5) 도 같다.

```bash
cd ~/KAMP_정종묵 && python show_results.py
```

**5) 결과 묶기** — 예측·표·그림·단계 로그·모델 번들을 zip 하나로(기본 `~/kamp_results_full.zip`).

```bash
cd ~/KAMP_정종묵 && python run_all.py --pack-results
```

**단계마다 멈추며 계산하기** — 터미널 앞에 앉아서 한 장씩 확인하고 싶을 때(nohup 에서는 자동으로 꺼진다):

```bash
python run_all.py --pause chapter
```

> FAST 를 복사본에서 돌리는 이유: FAST 결과가 FULL 결과와 섞이지 않게 하기 위해서다.
> 복사(`cp -r`)는 FULL 을 시작하기 **전에** 한다. FULL 이 도는 중에 복사하면 실행 중 잠금 파일까지 복사돼 FAST 가 시작을 거부한다.
> `rm -rf ~/kamp_fast` 는 지난 FAST 복사본만 지운다(남아 있으면 `cp -r` 이 그 안에 한 겹 더 복사한다).
> 같은 폴더에 FULL 결과가 이미 있으면 `run_all.py --fast` 는 덮어쓰기를 거부한다. FAST 결과를 묶으면 이름이 `~/kamp_results_fast.zip` 이 된다.

### 2-1. KAMP 가 아닌 PC(로컬·새 서버)에서 실행 — git clone

위 1)~2)의 `setup_pod.sh` 는 **패키지가 이미 깔린 KAMP 공유 conda 전용**이다. 없는 것(optuna·shap·pytest)만 채우고
numpy·pandas·lightgbm·TensorFlow 는 설치하지 않는다. 빈 PC 에서 그대로 따라 하면 필수 패키지가 없어
`run_all.py` 가 사전점검에서 멈춘다(종료코드 `2`, "필수 패키지가 없다").
빈 PC 에서는 **새 가상환경을 만들고 `requirements.txt` 로 설치한다.** ("pip install -r 금지"는 공유 conda 에만 해당한다.)

**파이썬 버전**: 3.10~3.12 (권장 3.11). `tensorflow==2.17.0`·`numpy==1.26.4` 는 3.13 을 지원하지 않는다.

**1) 받기**

```bash
git clone https://github.com/Junghwamin/KAMP-contest-submission.git
cd KAMP-contest-submission
```

**2) 가상환경 + 설치** (TensorFlow 포함 전부 설치된다. 수 분 걸린다)

Linux · macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt pytest
```

설치 확인: `bash check_env.sh` (Windows 는 Git Bash 에서) — 끝의 '종합 판정'을 본다.

**3) 실행** — 레포의 `outputs/` 에는 **이미 FULL 결과가 들어 있다.** 목적에 따라 고른다.

| 목적 | 명령 | 시간 |
|---|---|---|
| 들어 있는 결과 보기 | `python show_results.py` | 즉시 |
| 들어 있는 결과·모델 검증(재계산 없음) | `python run_all.py --check-only` | 수 분 |
| 처음부터 다시 계산(FULL) | `python run_all.py` | 30분 이상(5절) |
| 빠른 동작 확인(FAST) | 아래처럼 **복사본에서** | 수 분 |

FAST 는 같은 폴더의 FULL 결과를 덮어쓰지 않도록 거부되므로 복사본에서 돌린다.

```bash
cd .. && rm -rf kamp_fast && cp -r KAMP-contest-submission kamp_fast && cd kamp_fast
python run_all.py --fast && python show_results.py
```

(Windows PowerShell 에서는 `Copy-Item -Recurse KAMP-contest-submission kamp_fast` 로 복사한다.)

- Linux 서버에서 창을 닫아도 계속 돌리려면 2)의 `nohup` 줄을 그대로 쓴다. Windows 는 `python -X utf8 run_all.py` 로 직접 실행한다.
- GPU·CUDA 가 없거나 TensorFlow 가 GPU 에서 오류를 내면 `--cpu` 를 붙인다.
- TensorFlow 설치가 실패해도 `requirements.txt` 에서 `tensorflow` 줄만 빼고 설치하면 된다. 비교 모델(DNN·SimpleRNN)만 건너뛰고 나머지는 그대로 돈다.
- 모델 번들은 numpy·pandas·lightgbm 이 `requirements.txt` 와 **같은 버전**일 때만 열린다(`VERSION_MISMATCH`, 7절).

---

## 3. 폴더 지도

모든 폴더에 그 폴더를 설명하는 `README.md` 가 있다. 예외는 실행 중에 생기는 하위 폴더뿐이다(모델 번들·비교 모델 폴더와 빈 `outputs/baseline_repro/` — 이유는 `outputs/models/README.md`).

```
KAMP_정종묵/
├── README.md            ← 이 문서
├── check_env.sh         ← 1) 환경 진단 (읽기 전용)
├── setup_pod.sh         ← 2) 없는 패키지만 설치 + 한글 폰트
├── run_all.py           ← 3) 전체 계산 (src 를 노트북과 같은 방식으로 실행) + 서빙 검증 + 완료 점검
├── show_results.py      ← 4) 저장된 결과를 보고서 순서로 보기 (계산하지 않는다)
├── requirements.txt     ← 버전 기록(참고용). 공유 conda 에서 'pip install -r' 금지
├── pytest.ini           ← 단위테스트 설정
├── src/        README.md  s00_env.py … s10_package.py — 실험 코드 11개 (장 단위)
├── tools/      README.md  모델 저장·서빙 코어 생성·레짐 CV 프로세스 분리
├── serving/    README.md  모델 번들을 검증하며 읽어 익일 예측하는 추론 전용 코드
│   └── config/ README.md  holidays_kr.json — 운영 공휴일표
├── tests/      README.md  서빙·통계·학습 설정·시뮬레이션·인과성 회귀테스트
├── data/       README.md  okm_augumented_2021.csv — 유일한 입력
└── outputs/    README.md  predictions_test_336h.csv — 제출 예측(336시간)
    ├── figures/ README.md  그림 39장 + figure_index.csv
    ├── tables/  README.md  필수 CSV 110개 (실험표 71개 + 그림 원자료 39개)
    ├── models/  README.md  모델 번들(full/eval·deploy) · 비교 모델
    └── steps/   README.md  단계별 로그 · index.json · 서빙 검증 증거
```

- `outputs/`에 저장된 기존 결과가 있더라도 코드 수정 후에는 FULL 실행으로 갱신해야 한다.
- 실행 후 `run_all.py`가 예측·표·그림·모델·로그를 이번 실행 결과로 다시 쓴다. 최신성은 `index.json`의 실행 상태와 코드 해시로 확인한다.
- 모든 코드는 이 폴더를 기준으로 **상대경로**(`data/…`, `outputs/…`)를 쓴다. 폴더 이름은 바꿔도 되지만 안의 구조는 그대로 둔다.

---

## 4. 보고서 절 → 실행 단계 → 산출물

- **단계 id** 는 `src/` 의 절 번호다(예 `6.6` = `src/s06_eval.py` 의 6.6절, `1.G` = 1장 게이트, `S.1~S.4` = 서빙 검증, `C` = 완료 점검).
  각 단계의 출력은 `outputs/steps/{순번}_{id}.txt` 에 남는다.
- **산출물**: 표는 `outputs/tables/{이름}.csv`, 그림 `F05` 는 `outputs/figures/F05_….png`(그 원자료는 `outputs/tables/F05_src.csv`).
- 한 절만 보려면 `python show_results.py --section 2.6`, 표 하나는 `python show_results.py --table ch2_regression`.

| 보고서 절 | 주 단계 | 참고 단계 | 보고서 표·그림 ← 파일 |
|---|---|---|---|
| 1.1 분석 배경 및 문제 정의 | 1.6 | — | — |
| 1.2 데이터 구성 및 주요 변수 | 1.1 | — | ch1_variable_dict |
| 1.3 제조공정 상태 및 탐색적 분석(EDA) | 1.5 | — | 그림1 ← F05 |
| 1.4 데이터 품질 진단 및 처리 | 1.2, 1.3, 1.G | — | ch1_quality; 그림2 ← F02 |
| 1.5 전처리 및 파생변수 구성 | 1.4, 2.1, 2.4 | — | — |
| 1.6 학습·검증 데이터 구성 및 한계 | 3.1 | 1.6 | 표1-1 ← ch3_folds; 그림3 ← F14 |
| 2.1 실험 설계 | 2.2, 3.2, 3.3, 3.5, 5.0 | 2.4, 3.1 | — |
| 2.2 가이드북 베이스라인 재현 | 4.1, 4.2, 4.3, 4.4, 4.5, 4.G, 5.3 | 0.6 | 표2-1 ← ch4_defects; ch4_rf_original; ch4_corrected; ch4_naive |
| 2.3 입력변수 구성 | 2.3 | 2.1, 2.2, 2.4, 3.5 | ch2_feature_groups |
| 2.4 제안모델 구성 | 5.2, 5.4, 5.5, 5.6 | 5.1, 5.3 | — |
| 2.5 학습 및 최적화 방법 | 5.1 | 5.4 | hpo_trials; F26 |
| 2.6 전력사용량 예측 성능 | 6.1, 6.5 | 3.3, 4.1, 4.5, 5.0, 5.5 | 표2-2 ← ch2_regression; 표2-3 ← ch5_fold_mae_matrix; 그림4 ← F18; 그림5 ← F24 |
| 2.7 피크 위험 탐지 성능 | 3.4, 6.2 | 3.3, 6.5 | 표2-4 ← ch2_peak_detection; 표2-5 ← ch2_peak_detection_oof; 그림6 ← F19 |
| 2.7 임계값 시간순 분리 보조평가 | 6.2 | — | ch2_peak_threshold_temporal (임계값만 시간순 분리, HPO·모델선택은 OOF 재사용) |
| 2.8 변수 제거 실험(Ablation) | 6.3, 6.6 | — | ch2_ablation; 그림7 ← F20; 표2-6 ← ch2_lag_decomposition |
| 2.9 최종모델 선정 | 6.4, 6.G | 6.5, 6.6 | 표2-7 ← ch2_scorecard |
| 2.8~2.9 생산·가동계획 정보 제거 비교 | 6.4 | — | ch2_plan_information_ablation |
| 3.1 주요 영향변수 분석 | 7.0, 7.1 | — | 표3-1 ← ch3_importance; 그림8 ← F27 |
| 3.2 주요 변수의 상호작용 | 7.2 | — | ch3_interaction |
| 3.3 조건별 예측오차 분석 | 7.3 | — | ch3_condition_mae; 그림9 ← F29 |
| 3.4 피크 미탐지와 오경보 분석 | 7.4 | — | ch3_confusion; 그림10 ← F30 |
| 3.5 대표 실패사례 분석 | 7.5 | 6.6 | ch3_failures; 그림11 ← F33 |
| 3.6 최대수요 피크 발생조건 도출 | 7.6 | — | ch3_rules; 그림12 ← F31 |
| 4장 현장 활용방안 | 8.1, 8.2, 8.3, 8.4, S.3 | 10.5 | 표4-1 ← ch4_levers; 그림13 ← F35; 표4-2 ← ch4_scenarios; ch4_protocol |
| 5장 창의성 및 차별성 | 5.7, 5.8 | — | 그림14 ← F21; ch5_calibration_folds; ch5_calibration_forward_predictions (9장은 보고서 초안 생성용이라 제외) |
| 6장 코드 구성 및 재현성 | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10.1, 10.5, S.1, S.2, S.4, C | S.3 | env_versions; 예측 파일; ch6_model_bundles; figure_index |

**빠진 절** — 보고서 문장·초안·발표자료를 만드는 절만 뺐다. 실험 결과(모델·지표·표·그림·예측)에는 영향이 없다:
6.7(보고서 서술 교정), 8.5(4장 초안), 9장 전체(5장 초안), 10.2·10.3·10.4·10.6(채움표·README·발표자료·개인정보 스캔), 최종 게이트.
그래서 실행 단계는 코드 절 57개 + 서빙 검증 4개(S.1~S.4) + 완료 점검(C) = **62단계**다. 목록은 `python run_all.py --list`.

---

## 5. 옵션과 소요시간

**run_all.py** — 계산하고 `outputs/` 에 저장한다.

| 명령 | 하는 일 |
|---|---|
| `python run_all.py` | FULL 실행, 멈춤 없음 (기본) |
| `python run_all.py --fast` | FAST — 탐색·에폭을 줄인 동작 확인용. 수치는 최종값이 아니다 |
| `python run_all.py --pause chapter` | 장이 끝날 때마다 Enter 를 기다린다 (`--pause section` = 단계마다). 터미널 앞에서만 켜진다 |
| `python run_all.py --list` | 실행 계획(62단계)만 보여 준다. 아무것도 쓰지 않는다 |
| `python run_all.py --skip-serving` | 서빙 검증 S.1~S.4 를 건너뛴다 |
| `python run_all.py --no-tests` | 단위테스트(S.4)만 건너뛴다 |
| `python run_all.py --skip-bundle` | 모델 번들 저장(10.5)을 건너뛴다 → S.1~S.3 도 건너뛴다 |
| `python run_all.py --check-only` | 다시 계산하지 않고 이미 있는 outputs 로 서빙 검증·완료 점검만 한다. S.2 는 메모리 비트 비교 대신 제출 파일과 4·6자리 비교로 바뀐다. 지난 실행에서 통과한 서빙 단계를 건너뛰는 옵션과는 함께 쓸 수 없다(증거가 지워진다) |
| `python run_all.py --cpu` | GPU 를 쓰지 않는다 (TF 가 GPU 에서 문제를 일으킬 때) |
| `python run_all.py --force` | FULL 결과 위에 FAST 덮어쓰기·잠금 파일 검사를 무시한다 |
| `python run_all.py --pack-results` | 결과 zip 만들기 (기본 `~/kamp_results_full.zip`, FAST 결과면 `~/kamp_results_fast.zip` · `--out 경로` 로 위치 지정) |

종료코드: `0` 성공 · `1` 단계 실패 · `2` 사전점검 실패(패키지·데이터 없음 등) · `3` 완료 점검 실패 · `130` 중단.

**show_results.py** — 저장된 결과만 읽는다(다시 계산하지 않는다). 아직 실행 전이면 표·그림이 있는지만 보여 준다.

| 명령 | 하는 일 |
|---|---|
| `python show_results.py` | 대화형: Enter/`n` 다음 · `p` 이전 · `2.6` 처럼 절 번호로 이동 · `l` 쪽 목록 · `t 이름` 표 보기 · `f` 그림 목록 · `o` 로그 전체 · `q` 끝 |
| `python show_results.py --all` | 모든 쪽을 한 번에 출력 (파일로 저장할 때: `python show_results.py --all > ~/results.txt`) |
| `python show_results.py --section 2.6` | 한 절만 |
| `python show_results.py --table ch2_regression` | 표 하나 (`--max-rows N` 으로 행 수) |
| `python show_results.py --figures` | 그림 목록 |
| `python show_results.py --list` | 쪽 목록 |
| `--lines N` · `--width W` | 쪽마다 보여 줄 로그 줄 수(기본 40) · 출력 폭 |

**소요시간** — 이번 FULL 본 실행은 26분 06초였으나, 별도로 사전 계산한 6.6 체크포인트를 재사용했다. 사전 계산 시간은 포함하지 않았으므로 처음부터 무캐시로 실행한 시간으로 해석하면 안 된다. 환경·학습 모드·캐시 상태에 따라 달라지며 6.6(변수 제거 결과 검증)이 주요 장시간 단계다.
출력 없이 2분이 지나면 화면에 `… [6.6] 실행 중 · N분 경과`가 뜬다. 중간에 끊기면 `python run_all.py`를 다시 실행한다. 앞선 단계는 다시 계산하며, 6.6의 레짐 변형만 검증된 체크포인트를 재사용할 수 있다.

### 장시간 학습과 작업 캐시

6.6의 2·3분류 레짐 변형은 변형별 새 프로세스에서 4개 fold를 학습하고 OOF 숫자 배열만 부모에 반환한다. 시드·트리 수·4스레드 설정은 그대로 유지한다. 비레짐 CV는 기존 경로를 사용하되 fold별 모델 참조를 해제하고 가비지 수집을 수행한다.

체크포인트는 기본 `.cache/regime_cv`에 저장한다. 여유 공간이 있는 작업 폴더를 지정하려면 실행 전에 선택 환경변수를 설정할 수 있다.

```bash
export KAMP_WORK_DIR="$HOME/kamp_work"
python run_all.py
```

실제 저장 위치는 `$KAMP_WORK_DIR/regime_cv`다. 소스·데이터·입력 프레임·설정·런타임·바이너리의 지문, NPZ 해시, 완료 JSON이 일치하는 경우에만 재사용한다. 동일 요청이 실패하면 한 번 재시도하고, 다시 실패하면 실패로 기록한다. 전체 단계 이어하기 기능은 아니며 작업 캐시는 제출·결과 ZIP에서 제외한다.

---

## 6. 재현성

- **고정한 것**: 시드 42, `PYTHONHASHSEED=42`(`run_all.py` 가 파이썬을 이 값으로 다시 띄운다), TensorFlow 결정성 옵션,
  LightGBM `deterministic` + **스레드 4개 고정**(CPU 코어 수와 무관하게 같은 결과).
- **비트 단위로 같지 않은 것** (정상):
  - TensorFlow 로 학습하는 비교 모델(DNN (MLP)·Simple RNN)의 수치 — TF 버전·GPU 에 따라 달라진다. 최종모델·제출 예측과는 무관하다.
  - 벽시계 시간 — `timing.csv`, 표의 '학습시간(초)' 열, 단계 로그의 소요시간.
  - `ch1_profile_dup.csv` 의 그룹 **번호** — 동점 정렬 순서가 라이브러리 버전마다 다르다(그룹 내용은 같다).
  - 모델 번들 해시(`ch6_model_bundles.csv`) — 번들에 실행 환경(OS·라이브러리 버전)이 기록되기 때문이다.
  - CSV 줄바꿈 — Windows 는 CRLF, 리눅스는 LF 로 저장된다. 파일을 비교할 때는 LF 로 맞춘 뒤 비교한다.
- **같아야 하는 것**: 같은 코드·데이터·라이브러리 버전·FULL 설정에서 재생성한 제출 예측과 평가 번들 리플레이.
  전처리나 학습 설정을 수정하면 예측 해시도 달라질 수 있으므로 수정 전 해시와의 일치로 정확성을 판정하지 않는다. 해당 실행의 비교 결과는 완료 점검 C와 S.2 리플레이 기록에서 확인한다.

```bash
python -c "import hashlib;print(hashlib.sha256(open('outputs/predictions_test_336h.csv','rb').read().replace(b'\r\n',b'\n')).hexdigest())"
```

---

## 7. 서빙 (저장된 모델로 예측)

`run_all.py` 의 10.5 단계가 `outputs/models/full/{eval,deploy}/` 에 모델 번들을 저장하고, 이어서 서빙 단계가 자동으로 검증한다:
S.1 번들 무결성·자가검증 → S.2 9/1~9/14 리플레이와 학습 파이프라인의 메모리 예측 배열 비트 일치 및 제출 경보 라벨 비교 → S.3 이력·계획 CSV로 익일 예측(데모) → S.4 단위·회귀테스트.
증거 파일은 `outputs/steps/serving_*`. 직접 돌려 보려면:

```bash
python -X utf8 -m serving verify --bundle outputs/models/full/deploy
python -X utf8 -m serving replay --bundle outputs/models/full/eval --data data/okm_augumented_2021.csv --start 2021-09-01 --end 2021-09-14 --out replay.csv
python -X utf8 -m serving predict --bundle outputs/models/full/deploy --history outputs/steps/serving_demo_history.csv --plan outputs/steps/serving_demo_plan.csv --out pred.csv
python -X utf8 -B -m pytest tests -q -p no:cacheprovider
```

- 번들은 **만든 환경과 같은 numpy·pandas·lightgbm 버전**(KAMP 기준 1.26.4 · 2.1.4 · 4.5.0, `requirements.txt` 의 핀)에서만 열린다. 다르면 `VERSION_MISMATCH`.
- REST API 는 선택이다: `bash setup_pod.sh --with-api` 로 fastapi·uvicorn·httpx 를 넣은 뒤 `serving/README.md` 5절.
- 입력 형식·오류 코드·배포 절차는 `serving/README.md`.

전처리·피처·번들 예측 함수를 수정할 때는 `python tools/build_serving_core.py`로 서빙 코어를 재생성하고 `python tools/build_serving_core.py --check`로 동기화를 확인한다. 이번 FULL의 S.4에서 **123개 테스트가 17.42초에 통과**했다. S.1은 eval·deploy 번들 모두 통과했고, S.2의 336행 메모리 비교는 불일치가 없었다. S.3은 두 번들의 24행 데모를 생성하고 전력 정보가 섞인 계획 입력을 종료코드 2로 거부했다.

레짐 2·3분류의 분리 프로세스와 기존 방식 예측 배열 일치도 검증했다. 앞서 관측한 네이티브 DLL 종료의 원인을 확정한 것은 아니며, 프로세스 분리 후 이번 FULL 검증에서는 재발하지 않았다. 이 결과는 해당 환경·데이터·실행 범위에 대한 검증이다.

---

## 8. 환경

- **Python 3.10 이상**. KAMP 기준: Python 3.11.9 · numpy 1.26.4 · pandas 2.1.4 · scipy 1.11.4 · scikit-learn 1.4.2 · matplotlib 3.9.2 ·
  lightgbm 4.5.0 · statsmodels 0.14.2 · tensorflow 2.17.0 (+ 설치할 것: optuna 5.0.0 · shap 0.49.1). 필요한 패키지는 `requirements.txt`.
- **공유 conda 에서 `pip install -r requirements.txt` 를 하지 않는다.** 이미 있는 numpy 를 바꾸려다 권한 오류로 막히고 잔해가 남는다. `bash setup_pod.sh` 를 쓴다. **빈 PC·새 가상환경**에서는 반대로 `pip install -r requirements.txt` 로 전부 설치한다(2-1절).
- **TensorFlow 를 업그레이드하지 않는다.** GPU 연동이 깨질 수 있다. TensorFlow 가 없어도 비교 모델(DNN·RNN)만 빠지고 나머지는 돈다.
- 한글 폰트: `Malgun Gothic → NanumGothic → Noto Sans KR → AppleGothic` 순서로 정확히 이 이름을 찾는다. 없으면 그림의 한글만 □ 로 깨지고 코드는 돈다.

---

## 9. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `Permission denied: 'METADATA'` | 공유 환경에서 이미 있는 패키지를 바꾸려 했다. `pip install -r` 대신 `bash setup_pod.sh`. 그래도 막히면 `setup_pod.sh` 가 끝에 보여 주는 '수동 3줄' |
| `Ignoring invalid distribution ~umpy` | 이전 설치 실패의 잔해. `bash check_env.sh` 5절이 위치를 알려 준다. numpy 가 정상 import 되면 그 폴더를 지운다 |
| `ModuleNotFoundError: optuna` (또는 shap) | `bash setup_pod.sh --with-tests` |
| `run_all.py` 가 종료코드 2 로 바로 끝남 | 사전점검 실패. 화면의 이유를 본다: 패키지·데이터 없음, 같은 폴더의 FULL 결과를 FAST 로 덮으려 함(→ 복사본에서 `--fast`), 다른 실행이 돌고 있음(잠금 파일) |
| 세션이 끊겨 실행이 죽음 | 2절의 `nohup … &` 명령으로 다시 돌린다 |
| 상태가 `running` 인데 진행이 없다 (서버가 멈췄다 다시 켜진 뒤 등) | `tail -3 ~/full_run.out` 의 마지막 줄이 `RUN_ALL_…` 이 아니고, `python show_results.py` 머리에 '실행이 중간에 끊긴 기록' 이 보이면 끊긴 것이다. 서버가 다시 켜졌다면 `bash check_env.sh` 로 optuna·shap 이 남아 있는지 보고, 없으면 `bash setup_pod.sh --with-tests` 뒤 2절의 두 줄(`cd`·`nohup`)로 다시 돌린다 |
| `--pause` 가 꺼졌다는 안내 | nohup·파이프처럼 터미널 앞이 아닐 때는 자동으로 꺼진다(정상) |
| S.4 가 skipped | pytest 가 없다 → `bash setup_pod.sh --with-tests` 후 `python -X utf8 -B -m pytest tests -q -p no:cacheprovider` 로 따로 확인. 실제 통과·실패·skip 개수는 실행 출력으로 확인 |
| `VERSION_MISMATCH` (서빙) | 번들을 만든 환경과 numpy·pandas·lightgbm 버전이 다르다. 번들은 만든 환경에서 쓴다 |
| TF/GPU 오류로 4.2·5.5 단계 실패 | `python run_all.py --cpu` |
| 그림의 한글이 □□□ | 폰트 없음. `bash setup_pod.sh` 가 NanumGothic 을 받아 준다. 그다음 다시 실행 |
| 종료코드 3 | 계산은 끝났지만 완료 점검(C)의 필수 항목이 실패. `python show_results.py` 요약 쪽 맨 위와 `outputs/steps/` 의 C 로그를 본다 |
| 실행이 멈춘 것 같다 | 6.6 단계는 오래 걸린다. 2분마다 경과 표시가 나오면 정상 |
| 6.6 학습 프로세스가 네이티브 DLL 오류로 종료됨 | 수정본은 레짐 변형을 새 프로세스로 분리하고 동일 요청을 한 번 재시도한다. 두 번째도 실패하면 로그·종료코드를 확인한다. 재실행 시 이전 단계는 다시 계산하고 검증된 6.6 체크포인트만 재사용한다 |
| 작업 디스크 공간이 부족함 | 실행 전에 여유 공간이 있는 폴더를 `KAMP_WORK_DIR`로 지정한다. 이 설정은 레짐 CV 작업 캐시 위치이며 `outputs/` 위치를 바꾸지 않는다 |
