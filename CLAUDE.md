# CLAUDE.md

## 프로젝트 개요
Brave, Butters & Roberts (2019) "Uncovering the sources of team synergy" 논문을 구현한다 (원문: `docs/papers/`).
- 데이터: MLB 1998–2016 시즌, FanGraphs(fWAR)·Baseball-Reference(bWAR)
- 핵심 개념: 선수 생산성 잔차(팀 승수 vs WAR) → 공간 요인 모형으로 분해
  - WAR− (동료 영향 제거), WAR+ (동료 영향 포함), pcWAR = WAR+ − WAR (선수 보완효과), tcWAR (팀 전체 네트워크 효과)
  - character players(어느 팀에서나 동료를 돕는 선수) / team players(구단 요인), Intangibles(연령·포지션 프로파일 대비 초과분)
- 논문 주장 검증 기준: WAR로 설명되지 않는 팀 성과 변동의 약 40%를 시너지가 설명

## 규칙
- **수식·변수 정의의 근거는 논문 원문**이다. 구현 시 논문 절/수식 번호를 docstring에 명시하고, 논문과 다르게 단순화한 부분은 주석으로 밝힌다.
- 이 폴더는 이전 구현(수식 4·7·8 기반)을 버리고 새로 시작했다. 과거 커밋 코드는 참고만 하고, 논문과 대조해 검증한 뒤 사용한다.
- 문서(README, CLAUDE.md, 노트북 설명)는 한국어, 코드·주석·커밋 메시지는 영어.
- 난수가 들어가는 코드는 시드를 인자로 받는다 (재현성).
- 데이터 가공은 스크립트로 재현 가능해야 한다. `data/raw/`는 git 제외, `data/processed/`는 소용량 CSV만 추적.
- 코딩 작업은 판단에 따라 더 가벼운 모델을 서브에이전트로 실행한다 (사용자 선호).

## 구조
```
src/team_synergy/   모형·데이터 처리 코드
tests/              pytest
notebooks/          실험·시각화 (결과 출력은 가볍게 유지)
data/raw/           원자료 (untracked)
data/processed/     가공 데이터 (소용량만 tracked)
docs/papers/        논문 PDF (git 제외, 로컬에만 보관)
```

## 명령어
```bash
source venv/bin/activate      # Python venv (현재 venv는 3.14로 생성됨)
pip install -e ".[dev]"          # 또는 pip install -r requirements.txt
pytest -q                         # 전체 < 1분
synergy build --war fwar          # 원자료 → data/processed/panel_fwar.parquet (없으면 기대 파일 안내 후 exit 2)
synergy estimate --synthetic --seed 0   # 합성 패널로 3–7절 재귀 추정 end-to-end (~1.5분)
synergy estimate --war fwar [--full-only] [--set key=value]
synergy analyze --synthetic --n-boot 100   # 8절 표·그림 + outputs/synthetic/REPORT.md (~3분)
synergy analyze --war fwar                 # 실데이터: build → estimate 후 실행 (n_boot 500 기본)
jupyter lab
```
구현 기준 문서: `brave2019_synergy_implementation_spec.md` (설정은 `config/default.yaml`, `config/weights.yaml`).

## 오픈 이슈
- 원자료 미배치: README "데이터 배치" 절의 파일을 `data/raw/`에 넣은 뒤 `synergy build` 정합성 체크(570 팀-시즌, 시즌 WAR 합 ≈1000±5%, ID 매핑 실패 <1%) 미검증.
- 요인모형 A안(ALS)의 λ는 선수-시즌당 관측 팀이 2개 이하이면 식별되지 않음(시작값 의존). 조직문화 순위·pcWAR 선수/팀 분해는 B안(`culture_lambda_source: prob`) 기준으로 보고. WAR−/WAR+/pcWAR/tcWAR 합계는 영향 없음.
- ρ̂ 표준오차(집중우도 곡률)는 합성 실험의 시드 간 표준편차의 약 절반 → 실제 추론은 부트스트랩 필요.
- 논문과 다른 해석: 잔차 부호(`consistent`, 합성 검증 ①② 통과), λ 갱신을 V 공간에서 수행(논문 식 20은 Y 공간), 척도 Var(tp)=1(논문 ΛΛ′=I).
- 8절 분석은 구현됨(`analysis/`, `viz/`, `stats/`). 스펙과 다르게 논문을 따른 부분: 표 3 사양(지연항 포함), 표 1 의사 R²(57-fold CV), 그림 8·11 상·하위 25%, 표 5 CPI 실질화·1998 이후 데뷔 표본.
- `config/cpi.yaml`의 CPI-U 값은 BLS 공표치와 대조 필요.
- 그림 8 pcWAR 분해: 팀 성분은 `culture_lambda_source`(B안) 요인, 선수 성분은 나머지(합 = pcWAR 유지).
- 합성 데이터에서 실패하는 점검(표 4 지속성, 표 5 FA2 Cumξ 등)은 합성 생성 과정에 선수 요인의 시즌 간 지속성이 없어서 생긴다. 실데이터로 판정해야 함.
