# KAMP 경진대회 제출 파일

이 저장소에는 최종 제출 파일 `KAMP_정종묵.zip` 하나가 있다.
압축을 풀면 소스코드 · `requirements.txt` · 학습용 데이터 · README · 테스트데이터 예측결과가 나온다.

## 사용 방법 — KAMP-NOTE TensorFlow 팟

1. <https://note.kamp-ai.kr> 에 접속하십시오.
2. '분석환경 타입'에서 **TensorFlow** 를 고르십시오. 팟을 띄우십시오.
3. 터미널을 여십시오.
4. 아래 명령을 한 줄씩 붙여 넣으십시오.

```bash
cd ~ && git clone https://github.com/Junghwamin/KAMP-contest-submission.git
cd ~ && python -m zipfile -e KAMP-contest-submission/KAMP_정종묵.zip . && cd ~/KAMP_정종묵
bash check_env.sh
bash setup_pod.sh --with-tests
nohup python -u -X utf8 run_all.py --cpu > ~/full_run.out 2>&1 < /dev/null &
tail -f ~/full_run.out
```

5. 로그의 마지막 줄이 `RUN_ALL_OK mode=FULL` 인지 확인하십시오.
6. 결과를 보십시오: `cd ~/KAMP_정종묵 && python show_results.py`

> **참고** — 압축은 홈 폴더(`~`)에 푸십시오. 그러면 `~/KAMP_정종묵` 에 `.git` 이 없다. `.git` 이 있는 폴더에서는 `run_all.py` 가 실행을 거부한다.

> **주의** — `~/KAMP_정종묵` 이 이미 있으면 파일이 겹쳐 쓰인다. 이전 결과가 필요 없으면 먼저 지우십시오: `rm -rf ~/KAMP_정종묵`

자세한 절차는 압축을 푼 뒤 `~/KAMP_정종묵/README.md` 를 보십시오.
