# 데이터 수집 경과 보고서

작성일: 2026-10-04 · 대상: Brave, Butters & Roberts (2019) 복제(MLB 1998–2016)와 KBO 확장

## 요약

- MLB 원자료 5종 가운데 **Retrosheet 게임로그, FanGraphs WAR(fWAR), Chadwick ID 대응표** 3종을 스크립트로 받았다. 교차 점검(fWAR 시즌 합계, ID 매핑)도 통과했다.
- **Lahman DB와 Baseball-Reference WAR(bWAR)** 은 자동 다운로드가 막혔다. 브라우저로 직접 받아야 한다. Lahman이 없으면 패널을 만들 수 없어서 `synergy build`와 그 뒤 실데이터 분석은 아직 한 번도 돌리지 못했다.
- **KBO**는 주요 출처(KBO 공식 사이트, Statiz)가 자동 수집을 명시적으로 금지하고 있다. KBReport는 접속되지 않는다. 그래서 수집 코드는 만들지 않았고, 정식 데이터 요청이 필요하다.

---

## 1. 지금까지 수집한 데이터와 수집 방법

수집은 `synergy fetch --league mlb` 명령으로 재현할 수 있다(`src/team_synergy/cli.py`).

- 이미 받은 파일은 건너뛴다(`--force`로 다시 받음).
- 같은 호스트에 대한 요청 사이에 1초 이상 간격을 두고, 연구 목적임을 밝힌 User-Agent를 쓴다(`io/download.py`).
- 받은 파일마다 원본 URL, SHA-256, 크기, 받은 시각(UTC)을 `data/raw/MANIFEST.json`에 기록한다.
- 출처가 막히면 우회하지 않는다. 대신 기대 경로와 수동으로 받는 방법을 출력하고 종료 코드 2로 끝낸다.

| 데이터 | 경로 | 규모 | 수집 방법 | 상태 |
|---|---|---|---|---|
| Retrosheet 게임로그 1998–2016 | `data/raw/retrosheet/gl{YYYY}.txt` | 19개 파일, 48MB | `retrosheet.org/gamelogs/gl{YYYY}.zip`을 받아 압축 해제 (`io/retrosheet.py:fetch_gamelogs`) | ✅ |
| FanGraphs 시즌 WAR (타자·투수) | `data/raw/fangraphs/fg_{batting,pitching}.csv` | 23,043행 / 12,222행 | FanGraphs 공개 리더보드 JSON(`/api/leaders/major-league/data`)을 시즌별로 요청. robots.txt가 허용하는 경로다. pybaseball은 FanGraphs 403 문제(pybaseball #479) 때문에 쓰지 않는다 (`io/war_fangraphs.py:fetch_fwar`) | ✅ |
| Chadwick register (선수 ID 대응) | `data/raw/chadwick/people-{0..f}.csv` | 16개 파일, 선수 526,894명 | GitHub `chadwickbureau/register` raw 파일 직접 다운로드 (`io/idmap.py:fetch_register`) | ✅ |
| Lahman DB | `data/raw/lahman/` | — | SABR가 고정 주소 없이 Box 공유 링크로만 배포한다. 대체 미러(`chadwickbureau/baseballdatabank`)는 저장소가 사라졌다 | ❌ 수동 필요 |
| Baseball-Reference WAR | `data/raw/bref/war_daily_{bat,pitch}.txt` | — | `baseball-reference.com/data/`에서 HTTP 403 | ❌ 수동 필요 |

### 교차 점검 결과 (`io/crosscheck.py`, `synergy fetch --check-only`)

| 점검 | 기준 | 결과 |
|---|---|---|
| fWAR 시즌 합계(타자+투수) | 1000 ± 5% | 1998–2016 전 시즌 998.2–1000.9 ✅ |
| FanGraphs ID → Chadwick 매핑 실패율 | < 1% | 5,199명 중 0명 (0.00%) ✅ |
| Lahman `Teams.W` vs Retrosheet 승수 | 팀-시즌 일치 | Lahman이 없어 생략 |
| bWAR 시즌 합계 | 1000 ± 5% | bWAR이 없어 생략 |

테스트: `pytest -q`는 108개 모두 통과했다. 새로 추가된 다운로드·교차 점검 테스트 10개를 포함하며, 네트워크 없이 모의 HTTP로 실행한다.

---

## 2. 원 논문 데이터와 비교해 부족한 것

### 2.1 아직 없는 원자료

| 데이터 | 논문에서의 용도 | 없으면 생기는 일 | 확보 방법 |
|---|---|---|---|
| **Lahman DB** (Teams, Batting, Fielding, Pitching, Appearances, People, Salaries, Managers) | 팀 승수(식 1), 출전량(PA·아웃) 기반 가중치 τ(2절), 포지션·나이 프로파일(Intangibles), 연봉(표 5), 감독 | **패널 자체를 만들 수 없다.** 모든 분석이 멈춘다 | https://sabr.org/lahman-database/ 의 "Comma-delimited version"을 `data/raw/lahman/`에 넣는다(zip 그대로 넣고 `synergy fetch --only lahman`도 가능). 2025판(2026-01 공개)에도 Salaries는 2016까지 있어서 표본과 맞는다 |
| **bWAR** (`war_daily_bat.txt`, `war_daily_pitch.txt`) | 논문의 두 번째 WAR 출처. 모든 결과를 fWAR과 bWAR로 함께 보고한다 | fWAR 결과만 낼 수 있다. 출처 간 강건성 비교를 못 한다 | https://www.baseball-reference.com/data/ 에서 브라우저로 받아 `data/raw/bref/`에 넣는다 |
| **PECOTA** 2008–2016 팀별 시즌 전 예상 승수 (선택) | 표 3 (tcWAR의 승수 예측력) | 표 3만 생략된다 | Baseball Prospectus 유료 자료. `data/raw/pecota/pecota.csv` (franch_id, season, pecota_w) |

### 2.2 받았지만 논문과 같다고 볼 수 없는 부분

- **WAR 버전(vintage) 차이.** 논문은 2018년 무렵의 WAR을 썼고, 지금 받은 fWAR은 2026년 현재 값이다. FanGraphs는 그 사이 WAR 계산을 개정했다. 예를 들어 지금 데이터에는 포수 프레이밍(`CFraming`) 열이 들어 있다. 따라서 선수별 WAR은 논문과 다를 수 있고, 표 1·그림 4 수치를 대조할 때 이 차이를 감안해야 한다. bWAR도 마찬가지다. 2018년판 원자료는 공개 보관본이 확인되지 않았다.
- **fWAR은 시즌 합계라 팀별로 나뉘어 있지 않다.** 시즌 중 이적한 선수의 WAR은 `build/panel.py`가 출전량 비례로 팀별로 나눈다(논문 부록 7.1과 같은 처리).
- **CPI.** 표 5에서 연봉을 실질화할 때 쓰는 `config/cpi.yaml`의 CPI-U 값이 BLS 공표치와 대조되지 않았다(기존 오픈 이슈).

### 2.3 아직 검증하지 못한 패널 정합성 체크 (Lahman이 들어오면 실행)

- 팀-시즌 수 = 570 (30팀 × 19시즌)
- 시즌별 WAR 합 ≈ 1000 ± 5% (패널 기준)
- 패널 수준 ID 매핑 실패율 < 1%
- 팀-시즌마다 가중치 합 Σweight = 1

---

## 3. KBO 데이터 수집 현황

### 3.1 확정된 설계 결정 (2026-10-04, 사용자 결정)

- WAR은 **외부 출처만** 쓴다(Statiz/KBReport). 자체 계산 WAR은 쓰지 않는다.
- 현대 유니콘스(~2007)와 히어로즈(2008~)는 **별개 구단**으로 본다.
- 표본은 **2015–2025를 검토 중**이다(10구단 체제, 110 팀-시즌). 시즌 범위는 설정값으로 두어 2008년까지 넓힐 수 있게 한다. 표본이 MLB(570)의 약 1/5이어서 λ 식별과 지속성 분석(표 4·5)의 검정력이 약해질 수 있다.

### 3.2 출처별 접근성 조사 결과

| 출처 | 제공 내용 | 확인 결과 | 판단 |
|---|---|---|---|
| KBO 공식 (koreabaseball.com) | 팀 순위, 팀별 선수 기록, **경기별 박스스코어·선발 라인업**, 선수 프로필 | robots.txt: "본 사이트의 데이터를 사전 승인 없이 자동 수집·크롤링·복제하는 행위를 금지합니다", 일반 봇 `Disallow: /` | 자동 수집 불가. 사전 승인이 필요하다 |
| Statiz (statiz.co.kr) | KBO WAR (사실상 표준) | robots.txt가 주요 검색엔진을 뺀 모든 봇을 차단하고, Anthropic 봇도 따로 지목해 차단한다 | 자동 수집 불가 |
| KBReport (kbreport.com) | KBO WAR | 80·443 포트 모두 연결 실패. 운영 중단으로 보인다 | 사용 불가 |
| 야구나라 (yagoonara.com) | 시즌별 WAR 상위 선수 | 직접 접근 시 HTTP 403. 전체 선수 WAR이 아니다 | 부적합 |
| Kaggle 공개 데이터셋 | 시즌 타격·투구 기록 (Baseball-Reference 수집본) | WAR 없음, 경기별 라인업 없음 | 보조 자료로만 쓸 수 있다 |

### 3.3 핵심 병목

WAR보다 **경기별 선발 라인업**(MLB의 Retrosheet 게임로그에 해당)이 더 큰 문제다. 2015–2025에 약 7,900경기라서 손으로 옮길 수 없고, 사실상 출처가 KBO 공식 기록뿐이다. 타순 슬롯 비중 S_ijt와 동료 네트워크 α_ijt가 여기서 나오기 때문에 대체할 방법도 없다.

### 3.4 현재 상태

스펙 규칙("다운로드가 막히면 우회하지 않고, 기대 파일 형식을 적고 멈춘다")에 따라 **KBO 수집 코드는 만들지 않았다.**

---

## 4. 분석을 위해 더 수집해야 할 데이터

### 4.1 MLB (우선순위 순)

| 순위 | 데이터 | 담당 | 받은 뒤 할 일 |
|---|---|---|---|
| 1 | Lahman DB CSV | 사용자 수동 다운로드 | `synergy fetch --league mlb --check-only` → `synergy build --war fwar` 정합성 체크 → `estimate` → `analyze` |
| 2 | bWAR `war_daily_bat/pitch.txt` | 사용자 수동 다운로드 | `synergy build --war bwar` 이후 같은 순서 |
| 3 | PECOTA 2008–2016 (선택) | 유료 구매 여부 결정 | 표 3 실행 |
| 4 | CPI-U 1998–2016 BLS 공표치 | 확인 작업 | `config/cpi.yaml` 대조와 수정 |

### 4.2 KBO (모두 정식 요청 필요)

| 필요한 데이터 (MLB 대응) | 논문 변수 | 요청처 후보 |
|---|---|---|
| 팀별 시즌 승·패·무, 경기 수 (Lahman Teams) | W (식 1). 무승부 처리 방식 결정이 필요하다 | KBO 사무국 / 공식 기록 제공업체 |
| 선수-시즌-**팀별** WAR, 타자·투수 구분 (fWAR/bWAR) | WAR, 잔차 y | **Statiz** (연구용 데이터 제공 문의) |
| 선수-시즌-팀별 PA, 투구아웃, G, GS, 수비 이닝, 포지션별 출전 (Batting/Pitching/Fielding/Appearances) | 가중치 τ, 포지션 프로파일 | KBO 사무국 / 공식 기록 제공업체 |
| 경기별 홈·원정 선발 타순 1–9와 수비위치 (Retrosheet 게임로그) | 타순 비중 S_ijt, 동료 네트워크 | KBO 사무국 / 공식 기록 제공업체 |
| 선수 생년·데뷔 연도, 출처 간 선수 ID 대응 (People, Chadwick) | 나이, 경력 | KBO 선수 프로필 + Statiz ID 대응 |
| 연도별 선수 연봉, 외국인 선수 표시 (Salaries) | 표 5 | KBO 연도별 연봉 공시 |
| 팀별 감독, 시즌 중 교체 (Managers) | 감독 통제 | KBO 공식 / 공개 자료 |
| 한국 소비자물가지수 (CPI) | 표 5 실질화 | 통계청 KOSIS (공개) |

**권장 다음 단계**

1. **KBO 원자료 명세서** 작성: 위 표의 파일명, 열 정의, 키, 형식을 정한다. 데이터 요청서에 그대로 첨부할 수 있다.
2. KBO 사무국(또는 공식 기록 제공업체)과 Statiz에 연구용 데이터 제공 문의.
3. 데이터를 기다리는 동안, 명세서에 맞춘 정규화 코드와 리그 파라미터화(시즌별 팀 수·WAR 합계 목표 인자화, `--league kbo`)를 구현하고 합성 KBO 형식 데이터로 테스트해 둔다.
