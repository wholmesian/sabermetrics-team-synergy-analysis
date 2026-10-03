# Brave et al. (2019) 팀 시너지 모형 Python 구현 지시서

작성일: 2026-10-03

## 0. 개요

목표는 Brave, Butters & Roberts (2019, *Journal of Sports Analytics* 5:247–279)의 팀 시너지 측정 파이프라인을 Python 패키지 `synergy`로 재현하는 것이다. 1998–2016 MLB, fWAR·bWAR 두 버전을 동일 코드로 각각 돌린다.

파이프라인은 다섯 단계다.

1. 팀 승수를 팀 WAR 합에 회귀해 팀 생산성 잔차를 얻는다.
2. 출장 가중치로 팀 잔차를 선수(스틴트) 단위 잔차로 쪼갠다.
3. 팀메이트 연결 행렬 A로 공간자기회귀(SAR) 계수 ρ를 최우추정한다.
4. 공간 필터링된 잔차에 2요인 모형(character, team player)을 EM으로 적합한다.
5. 역행렬 Φ = (I − ρA)⁻¹로 in/out-degree를 계산해 WAR−, WAR+, tcWAR, pcWAR를 만들고, 논문 표·그림을 재현한다.

**Claude Code 작업 원칙**

- 이 문서의 수식과 정의가 기준이다. 논문 원문이 잘려 확인 불가한 부분은 10절에 해석과 함께 적어 두었고, 전부 `config/*.yaml` 옵션으로 노출한다.
- 숫자 정확 재현은 목표가 아니다. WAR 값이 2017년 이후 여러 차례 개정되었으므로 정성적 재현(부호, 크기 순서, 순위 상위권)을 목표로 한다.
- 데이터 다운로드가 막히면 우회하지 말고, `data/raw/` 아래 기대 파일명과 형식을 README에 적고 멈춘다.
- 모든 단계는 스틴트 단위 parquet로 중간 산출물을 저장하고, 각 단계는 독립 실행 가능해야 한다.
- Python 3.11+, numpy, scipy, pandas, statsmodels, linearmodels(고정효과), matplotlib, pyyaml, pytest. 무거운 의존성 추가 금지.

## 1. 데이터와 패널 구축

분석 단위는 **스틴트 r = (선수 i, 시즌 t, 팀 n)** 이다. 시즌 중 트레이드된 선수는 팀별로 여러 행을 갖는다. 팀은 Lahman `franchID`로 30개 프랜차이즈에 고정한다(MON→WSN, FLA→MIA, ANA→LAA, TBD→TBR 연결).

| 소스 | 필요한 필드 | 용도 | 접근 |
| --- | --- | --- | --- |
| Lahman DB (2016 이상 버전) | Teams(W, franchID), Batting(AB, BB, HBP, SH, SF), Fielding(POS, InnOuts), Pitching(IPouts, G, GS), Appearances(G_c…G_rf, G_dh, G_p), People(birthYear, debut, retroID, bbrefID), Salaries, Managers(inseason=1) | 승수, PA, 수비아웃, 투구아웃, 포지션, 나이, 연봉, 감독 | SABR 배포 CSV를 `data/raw/lahman/`에 수동 배치 |
| Retrosheet 게임로그 gl1998–gl2016 | 홈·원정 선발 타순 1–9 선수 ID와 수비위치 | 타순 슬롯 비중 S_ijt | retrosheet.org zip, `data/raw/retrosheet/` |
| Baseball-Reference WAR | war_daily_bat, war_daily_pitch (스틴트별 WAR, pitcher 플래그) | bWAR | `pybaseball.bwar_bat/bwar_pitch`, 실패 시 CSV |
| FanGraphs | 시즌 단위 타자 WAR, 투수 WAR, IDfg | fWAR | `pybaseball.batting_stats/pitching_stats(qual=0)`, 실패 시 리더보드 CSV 수동 export |
| Chadwick register | key_fangraphs, key_bbref, key_retro | ID 매핑 | pybaseball 또는 GitHub CSV |
| PECOTA (선택) | 2008–2016 팀별 시즌 전 예상 승수 | 표 3 | 유료. `data/raw/pecota/pecota.csv` (franchID, season, pecota_w) 없으면 해당 분석 skip |

**구축 규칙**

- 선수 구분: 스틴트에서 G_p ≥ 0.5 × G_all 이면 투수, 아니면 야수. 투수의 타격 WAR와 야수의 투구 기록은 버린다(논문 각주 13).
- 투수는 GS/G ≥ 0.5 이면 SP, 아니면 RP.
- PA = AB + BB + HBP + SH + SF. DOuts = Fielding InnOuts의 8개 야수 포지션 합. POuts = Pitching IPouts.
- g_ijt = Appearances의 8개 야수 포지션 출장수(G_c, G_1b, G_2b, G_3b, G_ss, G_lf, G_cf, G_rf) / 그 합.
- S_ijt = 게임로그상 슬롯 j 선발 횟수 / 전체 선발 횟수. 선발이 0인 선수는 9개 슬롯 가중치 평균을 l_it로 쓴다(옵션: Retrosheet 이벤트 파일로 교체 출전까지 집계).
- bWAR는 스틴트별로 그대로 쓴다. fWAR는 시즌 합만 있으므로 κ_it(2절) 비율로 스틴트에 배분한다.
- 주 포지션: 출장 최다 포지션. 최다 포지션 비중 < 0.5 인 야수는 `UT`(utility), G_dh 최다면 `DH`.
- 나이 = 시즌 연도 − birthYear. MLB 경력·팀 경력 = 해당 시즌 직전까지의 누적 출장 경기수.
- 감독 = 개막일 감독(Managers, inseason == 1).

**정합성 체크(빌드 단계에서 assert)**: 팀-시즌 570개, 시즌별 리그 WAR 합이 약 1,000(±5%), 선수 ID 매핑 실패율 < 1%.

## 2. 가중치: l, d, κ, τ, η

선수의 기대 기여는 맥락 없이 출장량만으로 정한다. 타순·수비 가중치 → 플레이 강도 κ → 팀 내 비중 τ → 리그 배분 η 순서로 계산한다.

**타순·수비 위치 가중치** (`config/weights.yaml`)

```latex
l_{it} = \sum_{j=1}^{9} b_j S_{ijt}, \qquad d_{it} = \sum_{j=1}^{8} p_j g_{ijt}
```

- b_j: Tango 외(2007) 타순 가중치(타순이 자주 돌아오는 슬롯에 프리미엄). 논문 Table 2 원값이 없으므로 기본값은 Retrosheet에서 1998–2016 슬롯별 경기당 평균 PA를 계산해 쓴다.
- p_j: FanGraphs 포지션 보정(162경기당 득점) C +12.5, SS +7.5, 2B/3B/CF +2.5, LF/RF −7.5, 1B −12.5, DH −17.5. 양수화를 위해 DH 기준으로 이동(adj − adj_DH)한다. bWAR 버전도 같은 값을 기본으로 쓰되 B-Ref 값 입력 옵션을 둔다.
- 정규화: b(9개)와 p(8개)를 각각 합이 1이 되도록 나눈다. 투수의 d_it = 1 (논문: 투수 수비 가중치 1). 대안 `normalize: max` 옵션도 구현한다.

**플레이 강도 κ** (네트워크와 fWAR 배분에 사용)

```latex
\kappa_{it} = \begin{cases} l_{it}\,\dfrac{PA_{it}}{3\cdot 162} + d_{it}\,\dfrac{DOuts_{it}}{27\cdot 162} & \text{야수} \\[2mm] d_{it}\,\dfrac{POuts_{it}}{27\cdot 162} & \text{투수} \end{cases}
```

**팀 내 비중 τ와 출장 가중치 ητ**: τ는 같은 팀-시즌 안에서 야수끼리, 투수끼리 κ를 나눈 비중이다.

```latex
\tau_{it} = \frac{\kappa_{it}}{\sum_{k \in \text{같은 팀, 같은 그룹}} \kappa_{kt}}, \qquad \text{출장 가중치} = \eta_{g}\,\tau_{it}
```

- η: 리그 WAR 1,000의 야수/투수 배분. fWAR 0.57 / 0.43, bWAR 0.59 / 0.41.
- 검증: 팀-시즌마다 Σ ητ = 1.

## 3. 팀 회귀와 선수 생산성 잔차

팀 잔차를 출장 가중치로 선수에게 나누고, 네트워크 모형의 목표변수는 **WAR 단위의 과대평가분** y_r로 정의한다. 이 부호 규약이 3–6절 전체의 기준이다.

**식 (1) 팀 회귀** (팀-시즌 570개, fWAR·bWAR 각각)

```latex
W_{nt} = \alpha + \beta \sum_{i \in n,t} WAR_{it} + \varepsilon_{nt}
```

기대값: α̂ 약 48–50, β̂ 약 1, ε̂ 표준편차 약 5승, 범위 30–40승.

**식 (4) 선수 분해**: 팀의 (W − α̂)를 출장 가중치대로 배분한 것이 선수의 기대 승수 기여다.

```latex
\hat W_{r} = \eta_{g}\,\tau_{r}\,(W_{nt} - \hat\alpha), \qquad y_r = WAR_r - \hat W_r / \hat\beta
```

- 항등식 검증: 팀-시즌마다 Σ_r y_r = −ε̂_nt / β̂ (오차 1e-10).
- 논문 식 (4)는 ε = Ŵ − WAR (양수 = 초과 성과)로 적혀 있다. 그러나 논문이 보고한 사실(① WAR− 회귀의 잔차 분산이 약 40% 감소, ② tcWAR과 팀 잔차가 양의 상관, ③ tp < 0 & λ > 0이 긍정적 파급)을 문자 그대로의 WAR−·tcWAR 정의와 동시에 만족하는 것은 y = −ε/β̂ 방향뿐이다. 따라서 기본값은 위 y로 하고, `sign_convention: paper_literal | consistent` 옵션을 두어 두 방향 모두 돌린 뒤 검증 지표 ①②로 어느 쪽이 맞는지 로그에 남긴다.
- y의 해석: 양수 = 개인 WAR가 실제 승리 기여보다 크게 잡힌 부분(팀메이트 덕을 본 몫 포함), 음수 = WAR에 잡히지 않은 승리.

## 4. 네트워크 행렬 A와 SAR 추정

A는 팀-시즌별 블록 대각 행렬이다. 블록 크기가 40–65이므로 전체를 희소 행렬로 만들 필요 없이 블록 리스트로 다룬다.

**연결 강도** (같은 팀-시즌 (n,t) 블록 안의 i ≠ j)

```latex
\alpha_{ijt} = \sum_{s \le t} \mathbf{1}[\,i, j \text{ 가 시즌 } s \text{ 에 같은 팀}\,]\,(\kappa_{is} + \kappa_{js})
```

- 과거에 함께 뛴 시즌(다른 팀에서였어도)의 κ 합도 누적한다. 기본은 s ≤ t(미래 정보 차단), 옵션 `history: full`은 표본 전체 합.
- 대각 0, 행 정규화 A_ij = α_ij / Σ_k α_ik. 이렇게 하면 |ρ| < 1에서 I − ρA가 엄격 대각우세라 가역이다.
- 서론의 "타순 인접 슬롯에 더 큰 가중치"는 공식 정의에 없다. `adjacency_kernel: off`(기본)로 두고, 켜면 α에 exp(−|slot_i − slot_j|/h)를 곱하는 확장으로 구현한다.

**1단계: ρ 최우추정** (스틴트 벡터 y에 대한 패널 SAR)

```latex
y = \rho A y + u, \quad u \sim N(0, \sigma^2 I)
```

```latex
\ell_c(\rho) = \sum_{b} \sum_{k} \log(1 - \rho\,\omega_{bk}) - \frac{N}{2} \log \hat\sigma^2(\rho), \qquad \hat\sigma^2(\rho) = \frac{\lVert (I - \rho A) y \rVert^2}{N}
```

- ω_bk = 블록 b의 고유값. A_b = D⁻¹α_b는 대칭행렬 D^(−1/2) α_b D^(−1/2)와 닮음이므로 `eigvalsh`로 실수 고유값을 미리 계산한다.
- `scipy.optimize.minimize_scalar(bounds=(-0.999, 0.999))`. 표준오차는 수치 헤시안.
- 산출: ρ̂, σ̂², 필터링 잔차 v = (I − ρ̂A) y, 블록별 Φ_b = (I − ρ̂A_b)⁻¹ (`np.linalg.solve`).

## 5. 2단계: 2요인 공간 요인모형

필터링 잔차 v를 (선수-시즌 × 30팀) 행렬 V에 놓고, 관측 칸만으로 character 요인 c와 team player 요인 tp를 추정한다. 대부분의 행은 관측 칸이 1개(트레이드된 선수만 2–3개)인 결측 패널이다.

```latex
\hat\varepsilon = \Phi F \Lambda, \quad V = (I - \hat\rho A)\,Y = F\Lambda, \quad \Lambda = \begin{bmatrix} 1 & 1 & \cdots & 1 \\ \lambda_1 & \lambda_2 & \cdots & \lambda_{30} \end{bmatrix}, \quad \sum_n \lambda_n = 0
```

- 행 = 선수-시즌 (i,t), 열 = 프랜차이즈 n, F의 행 f = (c, tp). 스틴트 r의 파급 단위 g_r = c + tp·λ_n(r).
- 1행 = 1 고정: character 효과는 팀과 무관(논문 정의). Σλ = 0: c와 λ의 평행이동 불식별을 막는 위치 제약. 척도: tp 열 분산 1(λ에 역보정). 부호: 절댓값이 가장 큰 λ_n이 양수가 되도록 tp와 λ의 부호를 함께 뒤집는다.
- 하위 지표는 모두 곱 tp·λ에만 의존하므로 척도·부호 제약은 해석에만 영향을 준다. |λ| 순위도 불변이다.

**A안(기본, 논문 재현): Stock–Watson식 EM-ALS** (논문 부록의 Dempster/Shumway–Stoffer/Reis–Watson 절차)

1. 결측 칸을 행의 관측값 평균으로 채워 Ṽ 초기화. λ⁰ = 0.
2. F 갱신: F = Ṽ Λ′(ΛΛ′)⁻¹ (부록의 F̂ = (Φ⁻¹YΛ′)(ΛΛ′)⁻¹).
3. λ 갱신: 열별 가중최소제곱 λ_n = Σ w·tp·(Ṽ_n − c) / Σ w·tp². 가중치 w = 출장 가중치(옵션 `wls_weights: appearance | none`). 이후 평균을 빼서 Σλ = 0 맞추고, 뺀 평균 × tp를 c에 더한다.
4. 결측 칸만 FΛ로 다시 채운다(관측 칸은 원값 유지). tp 표준화.
5. 관측 칸 SSE 상대변화 < 1e-9 또는 5,000회까지 반복.

**주의(식별)**: 관측 칸이 1개인 행은 고정점에서 g_r = v_r로 정확히 맞는다. 즉 WAR−, WAR+, pcWAR, tcWAR는 사실상 ρ̂와 A만으로 정해지고, 요인 분할은 λ(조직문화)와 pcWAR의 선수/팀 분해에만 영향을 준다. λ는 트레이드 행과 EM 경로에 의해 식별되므로 반드시 다음을 함께 산출한다.

- 초기값 20개(무작위 λ⁰)로 재추정한 λ 순위의 Spearman 상관 분포.
- 관측 칸 분산 중 요인이 설명한 비율(논문: 거의 전부).

**B안(강건성): 확률적 요인모형 EM** — f_r ~ N(μ, Σ_f), v_rn = Λ_n′f_r + e_rn, e ~ N(0, σ_e²). E-단계에서 행별 2×2 사후분포, M-단계에서 μ, Σ_f, λ(Σλ = 0 제약 LS), σ_e² 갱신. Var(tp) = 1. F̂ = 사후평균. 축소로 인해 g_r ≠ v_r이며, 이 차이는 자기항(own term)에 흡수된다. `factor_method: als | prob`.

## 6. 네트워크 지표

팀-시즌 블록 b에서 w_ij = (Φ_b)_ij, g_j = c_j + tp_j·λ_n 으로 놓으면 모든 지표가 블록 행렬 연산으로 나온다(WAR 단위; tcWAR만 승수 단위).

```latex
\text{own}_i = w_{ii}\, g_i, \qquad \text{in}_i = \sum_{j \ne i} w_{ij}\, g_j, \qquad \text{out}_i = g_i \sum_{j \ne i} w_{ji}
```

```latex
WAR^-_i = WAR_i - \text{in}_i, \qquad WAR^+_i = WAR^-_i + \text{out}_i, \qquad pcWAR_i = \text{out}_i - \text{in}_i
```

```latex
tcWAR_{nt} = -\hat\beta \sum_{i \in (n,t)} \text{in}_i
```

| 지표 | 의미 | 네트워크 용어 |
| --- | --- | --- |
| own | 자기 항(논문: measurement error) | 대각 |
| in | 팀메이트로부터 받은 파급 | in-degree |
| out | 팀메이트에게 준 파급 | out-degree |
| pcWAR | 순 파급(준 − 받은) | net-degree |
| tcWAR | 팀 시너지로 설명되는 승수 | total-degree × (−β̂) |

**선수/팀 분해** (그림 8): g를 c 부분과 tp·λ 부분으로 나눠 위 식을 각각 계산한다. pcWAR = pcWAR_char + pcWAR_team.

**반드시 통과해야 할 항등식 테스트**

- 팀-시즌마다 Σ pcWAR = 0, Σ WAR+ = Σ WAR.
- y_i = own_i + in_i (A안에서는 관측 칸 1개 행에 대해 1e-8 이내).
- 선수-시즌 지표 = 스틴트 지표의 합(트레이드 선수).

## 7. 재귀 추정

논문은 WAR−, tcWAR 등 회귀에 쓰는 값을 전체표본이 아닌 재귀 추정치로 쓴다. 1998년부터 한 시즌씩 늘린 19개 창 [1998, T], T = 1998…2016마다 3–6절 전체(식 (1)의 α̂, β̂ 포함)를 다시 추정한다.

- 시즌 T의 "재귀 지표" = 창 [1998, T] 추정에서 나온 시즌 T 값. 저장: `metrics_recursive.parquet` (window_end, stint 키, 모든 지표).
- 전체표본 추정(T = 2016)은 `metrics_full.parquet`로 별도 저장. 선수 랭킹(그림 8–12)과 연봉 회귀는 전체표본 값을 기본으로 쓴다.
- 창마다 λ̂를 저장해 조직문화 순위 분포(그림 3)에 쓴다.
- 한 창의 계산은 수 초 수준이어야 한다. `joblib`으로 창 병렬화 가능하나 필수는 아니다.

## 8. 재현 분석 명세

모든 산출물은 fWAR·bWAR 두 버전으로 만든다. 추론은 클러스터 부트스트랩 BCa 500회(`stats/bootstrap.py`)이며, 요인모형은 고정한 채 2단계 회귀만 재표본한다(생성 회귀변수 문제는 README에 명시).

| 산출물 | 정의 | 표본·클러스터 | 논문 기대 결과 |
| --- | --- | --- | --- |
| 표 1 | 식 (1)을 ΣWAR, ΣWAR−(재귀)로 각각. R², 의사 R² = 재귀 추정의 표본외 1 − SSE/SST | 팀-시즌 570, 팀 | β̂ ≈ 1, α̂ < 50, WAR−에서 R² 상승 |
| 그림 2 | 두 회귀 잔차의 커널밀도 + sd(WAR− 잔차)/sd(WAR 잔차) 비율의 BCa CI | 팀 | 미설명 분산 약 40% 감소 |
| 그림 3 | 창별 \|λ_n\| 순위를 0–100으로((rank−1)/29×100), 팀별 박스플롯 + 중앙값 점 | 19개 창 | 상위: STL, SF, ARI |
| 그림 4 | 팀 잔차 vs tcWAR 산점도, 극단값 라벨 | 570 | 2008 LAA·2007 ARI 우상단, 1998 SEA·1999 KC·2015 CIN 좌하단 |
| 시너지 지속성 | tcWAR_nt = a + b·tcWAR_n,t−1 + u. Σ_t u = 팀 시너지 초과승(그림 5, 막대) | 540, 팀 | b 매우 작음(강한 평균회귀) |
| 표 3 | (1) W = a + b·PECOTA + c·tcWAR(재귀) (2) W = a + b·PECOTA + c·tcWAR 예측치(AR(1) 1기 앞). (2)를 재귀로 돌린 표본외 예측 2009–2016 | 270 / 240, 팀 | MAE(PECOTA) − MAE(모형) 약 1.25–1.4승, Diebold–Mariano(절대오차, Newey–West) 유의 |
| 그림 6 | WAR− vs WAR, WAR+ vs WAR (45도선, WAR = 1, 4 세로선) | 선수-시즌 | WAR−는 45도선 밀착, WAR+는 고WAR에서 위 |
| 그림 7 | pcWAR vs WAR | 선수-시즌 | Star 0~+1.5, Scrub 0~−0.5 |
| 그림 8 | 2016 현역, 커리어 평균 출장 가중치 상위 절반, 커리어 평균 pcWAR 상·하위 N명(기본 35), 선수/팀 분해 누적막대 | 선수 | 1위 Mike Trout |
| 표 4 | 아래 식 (16), 사양 (1) 지연항만, (2) 전체 통제 | 선수, 약 20,700 관측 | 지속성 Star ≈ 5 × Scrub, ≈ 1.5 × Role. R² 약 0.21 → 0.8 |
| 그림 9 | 표 4 사양 (1) 잔차의 커리어 합 vs 커리어 평균 지연 WAR, 상·하위 1% 선 | 선수 | Jeter 음수 극단, Beltre 최상위 |
| 그림 10 | 사양 (2)의 포지션별 나이 20–40 평균 한계효과(해당 포지션 관측의 공변량 그대로 두고 나이만 바꿔 예측 평균) + BCa 95% CI | 선수 | 대부분 우상향, SP 평탄 |
| 그림 11 | Intangibles ξ = 사양 (2) 잔차, 커리어 평균, 그림 8과 같은 필터 | 선수 | 1위 Kevin Kiermaier(약 0.1) |
| 그림 12 | David Ross 시즌별 pcWAR·ξ vs WAR, 시즌 라벨 | 단일 선수 | 대부분 pcWAR 양수 |
| 표 5 | 아래 식 (17), 사양 (1) 누적 pcWAR, (2) 누적 (pcWAR − ξ)와 누적 ξ 분리 | 선수, 1998 이후 데뷔 | FA 단계에서 pcWAR 가격 양(+), ξ 음(−) |

**식 (16) pcWAR 지속성** (선수-시즌; 트레이드 선수는 스틴트 합, 팀·감독은 출장 최다 스틴트 기준)

```latex
pcWAR_{it} = \sum_{k \in \{Scrub, Role, Star\}} \rho_k\, pcWAR_{i,t-1}\,\mathbf{1}[\text{tier}_{i,t-1} = k] + \sum_p \mathbf{1}[pos_{it} = p]\left(\gamma_p + \sum_{m=1}^{4} \theta_{pm}\, age_{it}^m\right) + \delta' X_{it} + \phi' Z_{it} + \xi_{it}
```

- tier: WAR_t−1 < 1 Scrub, 1 ≤ · < 4 Role, ≥ 4 Star. 사양 (1)은 tier 더미와 상호작용만.
- X = WAR_t, MLB 경력, 팀 경력(경기수). Z = 리그·팀·감독 고정효과. 포지션 12개: C, 1B, 2B, 3B, SS, LF, CF, RF, DH, UT, SP, RP.

**식 (17) 연봉** (Lahman Salaries, 로그)

```latex
\ln S_{it} = \sum_c FA_{c,it}\left(\theta_c + \pi_c\, \text{CumWAR}_{i,t-1} + \psi_c\, \text{CumpcWAR}_{i,t-1} + \mu_c\, teamExp_{it}\right) + \sum_p pos_{p}\left(\phi_p\, age_{it} + \lambda_p\, mlbExp_{i,t-1} + \tau_p\, mlbExp^2_{i,t-1}\right) + \alpha_i + \varepsilon_{it}
```

- FA_c: 표본 내 이전 출장 시즌 수 0–2 = 사전중재(FA0), 3–5 = 중재(FA1), 6+ = FA(FA2). 선수 고정효과, 선수 클러스터.
- 사양 (2): ψ_c CumpcWAR를 (CumpcWAR − Cumξ)와 Cumξ로 분리, FA0 항은 다중공선성으로 제외.

## 9. 코드 구조, 작업 순서, 검증

합성 데이터로 모형 코드를 먼저 검증한 뒤 실제 데이터를 붙인다. 각 단계는 아래 완료 기준을 통과해야 다음으로 넘어간다.

```text
synergy/
  pyproject.toml   README.md
  config/default.yaml        # 연도, war_source, η, 옵션(sign_convention, history, factor_method, ...)
  config/weights.yaml        # b_j, p_j, 정규화 방식
  data/raw/{lahman,retrosheet,bref,fangraphs,chadwick,pecota}/
  data/processed/            # panel.parquet, metrics_*.parquet
  src/synergy/
    io/        lahman.py retrosheet.py war_bref.py war_fangraphs.py idmap.py pecota.py
    build/     panel.py weights.py
    model/     team_reg.py residuals.py network.py sar.py factor_als.py factor_prob.py
               decompose.py recursive.py
    analysis/  table1.py org_culture.py tc_persistence.py pecota.py player_eval.py
               pc_persistence.py profiles.py intangibles.py salary.py ross.py
    stats/     bootstrap.py dm_test.py
    viz/       figures.py
    cli.py     # synergy build | estimate | analyze | all  --war fwar|bwar
  tests/       test_identities.py test_sar.py test_factor.py test_synthetic.py
  outputs/{tables,figures}/
```

**작업 순서와 완료 기준**

1. 합성 데이터 생성기(`tests/test_synthetic.py`): 30팀 × 10시즌, 팀당 45명, 알려진 ρ = 0.3, λ, c, tp로 y 생성. 완료: ρ̂ 오차 < 0.05, λ̂와 참값 순위상관 > 0.8(B안), 6절 항등식 전부 통과.
2. 데이터 IO와 패널 빌드(1–2절). 완료: 1절 정합성 체크, Σητ = 1.
3. 식 (1)과 y(3절). 완료: Σ_r y_r = −ε̂/β̂, α̂·β̂가 기대 범위.
4. A와 SAR(4절). 완료: 블록 고유값 ∈ [−1, 1], 우도 곡선 그림 저장, ρ̂ 경계해 아님.
5. 요인모형 A안·B안(5절). 완료: 수렴 로그, 다중 초기값 순위상관 리포트, 설명분산 비율.
6. 지표 분해(6절). 완료: 항등식 테스트, 검증 지표 ①②(3절)로 부호 규약 확인.
7. 재귀 추정(7절).
8. 분석·그림(8절), fWAR 먼저 전부 끝낸 뒤 bWAR.
9. `outputs/REPORT.md`: 각 표·그림 옆에 논문 기대 결과와 일치 여부 한 줄씩.

**품질 규칙**: 함수마다 docstring에 대응 논문 식 번호, 모든 난수에 시드, 설정값은 산출물 메타데이터에 기록, `pytest -q` 1분 이내.

## 10. 논문 모호점과 채택한 해석

제공된 논문 텍스트는 일부 열과 Table 2가 잘려 있다. 아래 항목은 모두 설정으로 바꿀 수 있게 구현하고, 원문 확인 시 기본값만 교체한다.

| 항목 | 논문에서 확인되는 것 | 채택한 기본값 | 대안 옵션 |
| --- | --- | --- | --- |
| b_j, p_j 수치 | Tango 외 타순, FanGraphs/B-Ref 포지션 보정을 정규화(Table 2) | Retrosheet 슬롯별 PA, FanGraphs 보정의 DH 기준 이동, 합 = 1 정규화 | max 정규화, 수기 입력 |
| 잔차 부호 | 식 (4) ε = Ŵ − WAR | y = WAR − Ŵ/β̂ (보고된 결과와 정합) | paper_literal |
| β̂ 처리 | ε̂ = ΣŴ − ΣWAR 항등식 | Ŵ/β̂로 정확한 항등식 유지 | β = 1 고정 |
| α_ij 누적 범위 | 함께 뛴 시즌마다 합산 | s ≤ t | 표본 전체 |
| 타순 인접성 | 서론에만 언급 | 끔 | 지수 커널 |
| Λ 제약 | 1행 단위벡터, 나머지 행 합 제약, 척도 정규화 | 1행 = 1, Σλ = 0, Var(tp) = 1 | — |
| 요인 추정 | EM(결측) + WLS/GLS 교대 | A안 EM-ALS, 가중치 = 출장 가중치 | B안 확률모형, 무가중 |
| 트레이드 fWAR 배분 | 출장 비례 | κ 비례 | 경기수 비례 |
| 선발 없는 선수의 l_it | 미기재 | 9개 슬롯 가중치 평균 | 이벤트 파일 집계 |
| 그림 9 기준선 | 분포의 극단 | 상·하위 1% | 설정 |
| 서비스 연수 | 표본 내 시즌 수로 근사 | 동일 | 출장 경기수 기준 |

**식별 한계(README에 명시)**: 선수-시즌 행의 대부분이 한 팀에서만 관측되므로 character/team player 분할과 λ는 트레이드 행과 EM 경로에 의존한다. WAR−, WAR+, pcWAR, tcWAR 합계는 이 분할과 무관하게 ρ̂와 A로 정해진다. 따라서 조직문화 순위와 그림 8의 분해는 B안 및 다중 초기값 결과와 나란히 보고한다.
