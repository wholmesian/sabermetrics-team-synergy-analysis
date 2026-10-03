# Team Synergy Sabermetrics

Brave, Butters & Roberts (2019), *Uncovering the sources of team synergy: Player complementarities in the production of wins* (Journal of Sports Analytics 5, 247–279) 구현 프로젝트.
WAR 잔차와 공간 요인 모형(spatial factor model)으로 팀 시너지(pcWAR, tcWAR)를 측정한다.

## 시작하기
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest
```

## 구조
- `src/team_synergy/` 모형 구현 · `tests/` 테스트 · `notebooks/` 실험
- `data/raw/` 원자료(git 제외) · `data/processed/` 가공 데이터(소용량만 추적)
- `docs/papers/` 논문 PDF

자세한 작업 규칙은 `CLAUDE.md` 참고.

## 데이터 배치

아래 원자료를 `data/raw/` 아래 지정된 디렉토리에 배치한 후 분석을 실행한다.
모든 파일이 필요한 것은 아니며, 각 로더는 누락된 파일에 대해 명확한 메시지를 출력한다.

| 소스 | 경로 | 파일 목록 | 입수 방법 |
|------|------|---------|---------|
| **Lahman DB** | `data/raw/lahman/` | Teams.csv, Batting.csv, Fielding.csv, Pitching.csv, Appearances.csv, People.csv, Salaries.csv, Managers.csv | [SABR 배포](https://www.seanlahman.com/baseball-archive/statistics/) 또는 Baseball-Reference에서 CSV 다운로드 |
| **Retrosheet** | `data/raw/retrosheet/` | gl1998.txt, gl1999.txt, …, gl2016.txt (또는 GL{YYYY}.TXT 대문자 형식) | [Retrosheet 게임로그](https://www.retrosheet.org/gamelogs/) 다운로드 및 압축 해제 |
| **Baseball-Reference WAR** | `data/raw/bref/` | war_daily_bat.txt, war_daily_pitch.txt | [Baseball-Reference](https://www.baseball-reference.com/data/) 또는 `pybaseball.bwar_bat()`/`bwar_pitch()` 실행 |
| **FanGraphs WAR** | `data/raw/fangraphs/` | fg_batting.csv, fg_pitching.csv | [FanGraphs 리더보드](https://www.fangraphs.com/leaders.aspx) 내보내기 또는 `pybaseball.batting_stats(qual=0)`/`pitching_stats(qual=0)` 실행 |
| **Chadwick Register** | `data/raw/chadwick/` | people.csv (또는 people-*.csv 분할 파일) | [Chadwick Bureau](https://github.com/chadwickbureau/register/) 또는 `pybaseball.chadwick_register()` 실행 |
| **PECOTA** (선택) | `data/raw/pecota/` | pecota.csv (franch_id, season, pecota_w 열) | [Baseball Prospectus](https://www.baseballprospectus.com/) 유료 데이터; 없으면 표 3 분석만 생략 |
