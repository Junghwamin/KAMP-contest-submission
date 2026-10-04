# KAMP 경진대회 제출 파일

이 저장소에는 최종 제출 파일 `KAMP_정종묵.zip` 하나가 있다.
압축을 풀면 소스코드 · `requirements.txt` · 학습용 데이터 · README · 테스트데이터 예측결과가 나온다.

## 1. zip 받기 — 둘 중 하나

**A. KAMP-NOTE 터미널에서 git clone**

```bash
cd ~ && git clone https://github.com/Junghwamin/KAMP-contest-submission.git
```

zip 위치: `~/KAMP-contest-submission/KAMP_정종묵.zip`

**B. 파일로 받기 (메일 · GitHub 웹 다운로드)**

1. zip 을 PC 에 저장하십시오. (GitHub 웹: `KAMP_정종묵.zip` 클릭 → 오른쪽 위 다운로드 버튼)
2. KAMP-NOTE 의 파일 창에서 zip 을 홈 폴더(`~`)로 올리십시오.

zip 위치: `~/KAMP_정종묵.zip`

## 2. 압축 풀기

KAMP-NOTE 터미널에서 실행하십시오. 압축은 항상 홈 폴더(`~`)에 푼다. 풀린 폴더는 `~/KAMP_정종묵` 이다.

A 로 받았을 때:

```bash
cd ~ && python -m zipfile -e KAMP-contest-submission/KAMP_정종묵.zip . && cd ~/KAMP_정종묵
```

B 로 받았을 때:

```bash
cd ~ && python -m zipfile -e KAMP_정종묵.zip . && cd ~/KAMP_정종묵
```

확인: `ls` 를 실행하십시오. `README.md` · `run_all.py` · `requirements.txt` · `data` · `outputs` · `src` 가 보이면 된다.

> **주의** — `~/KAMP-contest-submission` 안에 풀지 마십시오. 그 폴더에는 `.git` 이 있다. `.git` 이 있는 폴더에서는 `run_all.py` 가 실행을 거부한다.

> **주의** — `~/KAMP_정종묵` 이 이미 있으면 파일이 겹쳐 쓰인다. 이전 결과가 필요 없으면 먼저 지우십시오: `rm -rf ~/KAMP_정종묵`

> **참고** — `unzip` 명령이 없는 팟도 있다. 그래서 파이썬 내장 `zipfile` 을 쓴다. zip 이름이 다르면 명령의 파일 이름만 바꾸십시오.

> **참고** — PC 에서 내용만 보려면 zip 을 마우스 오른쪽 버튼 → '압축 풀기'(Windows) 로 풀면 된다. 실행은 KAMP-NOTE 에서 한다.

## 3. 실행 — KAMP-NOTE TensorFlow 팟

1. <https://note.kamp-ai.kr> 에서 '분석환경 타입'을 **TensorFlow** 로 골라 팟을 띄우십시오.
2. 2절로 압축을 푼 뒤 `~/KAMP_정종묵` 에서 아래 명령을 한 줄씩 붙여 넣으십시오.

```bash
cd ~/KAMP_정종묵
bash check_env.sh
bash setup_pod.sh --with-tests
nohup python -u -X utf8 run_all.py --cpu > ~/full_run.out 2>&1 < /dev/null &
tail -f ~/full_run.out
```

3. 로그의 마지막 줄이 `RUN_ALL_OK mode=FULL` 인지 확인하십시오. `Ctrl+C` 는 로그 보기만 멈춘다.
4. 결과를 보십시오: `cd ~/KAMP_정종묵 && python show_results.py`

자세한 절차는 압축을 푼 뒤 `~/KAMP_정종묵/README.md` 를 보십시오.
