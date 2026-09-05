# Team Synergy Analysis (Brave et al., 2019)

이 프로젝트는 Zotero에 저장된 논문 **"Uncovering the sources of team synergy: Player complementarities in the production of wins" (Scott Brave 외, 2019)**에서 제안된 '팀 시너지' 측정 방법론을 파이썬으로 구현한 것입니다.

## 방법론 요약
이 연구는 단순히 개인의 기량(WAR)의 합이 팀의 전체 성과와 일치하지 않는다는 점에서 착안하여, 선수들 간의 상호작용(네트워크 효과)을 정량화합니다.
**공간 요인 모형 (Spatial Factor Model)**을 도입하여 다음과 같은 개념을 측정합니다:
- **공간 승수 (Spatial Multiplier):** `(I - ρW)^(-1)` 행렬을 통해 팀 네트워크에서 선수가 발휘하는 직간접적 파급 효과(Spillover)를 계산합니다.

## 필요한 데이터 형태
이 모델을 실제 데이터로 실험하기 위해서는 다음 두 가지 데이터가 필요합니다:

1. **개인 역량 지표 벡터 (Player WAR, 1D Array)**
   - 각 선수의 독립적인 기량 지표입니다. 야구의 경우 대체 선수 대비 승리기여도(WAR)를 사용하며, 선수 수(`N`)만큼의 길이를 가진 1차원 벡터 형태여야 합니다.

2. **상호작용 인접 행렬 (Adjacency Matrix, 2D Array)**
   - 팀원들 간의 상호작용 강도를 나타내는 `N x N` 크기의 가중치 행렬 `W`입니다.
   - 예: 포지션 간의 연관성, 타순의 인접성, 특정 선수들이 함께 경기에 뛴 시간 비율 등.
   - 각 행의 합이 1이 되도록 정규화(Row-normalized)하는 것이 일반적입니다.

## 파일 구성
- `team_synergy_model.py`: 공간 승수 알고리즘을 이용해 시너지를 구하는 파이썬 클래스(`TeamSynergySpatialModel`)가 구현되어 있습니다.
- `experiment.ipynb`: 작성된 모델을 테스트하고 시뮬레이션 해볼 수 있는 주피터 노트북입니다.
- `requirements.txt`: 실행에 필요한 파이썬 패키지 목록입니다.

## 실행 방법 (환경 설정)
이 프로젝트는 파이썬 패키지 `numpy`를 필요로 합니다. 주피터 노트북을 실행하려면 `jupyter`도 설치해야 합니다.

```bash
# 가상환경 생성 및 패키지 설치
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 주피터 노트북 실행
jupyter notebook experiment.ipynb
```
