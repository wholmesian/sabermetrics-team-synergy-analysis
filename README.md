# Team Synergy Analysis (Brave et al., 2019)

이 프로젝트는 Zotero에 저장된 논문 **"Uncovering the sources of team synergy: Player complementarities in the production of wins" (Scott Brave 외, 2019)**에서 제안된 '팀 시너지' 측정 방법론을 엄밀한 논문 수식 기반으로 파이썬으로 구현한 것입니다.

## 논문의 주요 수식 반영 내역
초기 코드는 일반화된 공간 요인 모형(Spatial Factor Model) 형태만을 띠고 있었으나, 해당 논문의 원문을 직접 파싱하여 논문에서 제시한 특수한 야구 특화 변수와 수식을 그대로 코드에 이식했습니다.

1. **생산성 잔차(Productivity Residuals)의 도출 (수식 4 & 7)**
   논문에서는 WAR 수치의 단순 합과 실제 팀 승수 간의 오차를 기반으로 분석을 시작합니다.
   - 각 선수의 '기대 팀 승수 기여도($\hat{W}_{int}$)'는 팀의 전체 승수에서 대체선수 수준(약 50승)을 뺀 값을 투수/타자의 비율(0.43/0.57, $\eta_{it}$)과 타석/이닝 소화 비중($\tau_{it}$)에 따라 배분하여 계산합니다.
   - 선수의 생산성 잔차($\hat{\epsilon}_{int}$) = $\hat{W}_{int}$ - $WAR_{int}$ 로 정의됩니다.

2. **네트워크 가중치 행렬 구성 (수식 8의 $\alpha_{ijt}$)**
   - 팀원 간의 상호작용 행렬 $A$는 단순히 1과 0으로 이루어진 것이 아니라, 선수의 출전 비중 지표($\kappa_{it}$)들의 합으로 구성됩니다. 출전 시간이 길고 타순/포지션 중요도가 높을수록 동료와의 상호작용(Spillover) 가능성도 높게 산정됩니다.

3. **공간 자귀회귀(SAR)와 시너지 효과(pcWAR, tcWAR)**
   - $\hat{\epsilon} = \rho A \hat{\epsilon} + v$ 라는 SAR 모형을 이용해 $\rho$(공간 계수)를 추정합니다.
   - 기초 충격(Fundamental shocks, $v$)과 $\hat{\epsilon}$의 차이($\rho A \hat{\epsilon}$)가 동료에 의해 발생한 **선수 간 보완효과(Player Complementarity WAR, pcWAR)**로 정의됩니다.
   - 팀 전체의 pcWAR 합산이 바로 팀 시너지(tcWAR)가 됩니다.

## 파일 구성
- `team_synergy_model.py`: 위 논문의 실제 수식을 반영한 파이썬 모형(`BraveEtAlSynergyModel`) 파일.
- `experiment.ipynb`: 작성된 모델을 테스트하고 시뮬레이션 해볼 수 있는 주피터 노트북.
- `requirements.txt`: 실행에 필요한 파이썬 패키지 목록.

## 실행 방법 (환경 설정)
이 프로젝트는 파이썬 패키지 `numpy`, `scipy` 등을 필요로 합니다. 주피터 노트북을 실행하려면 `jupyter`와 `pandas`도 설치해야 합니다.

```bash
# 패키지 설치
pip install -r requirements.txt

# 주피터 노트북 실행
jupyter notebook experiment.ipynb
```
