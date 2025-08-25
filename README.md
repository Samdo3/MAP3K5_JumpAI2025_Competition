# MAP3K5/ASK1 IC50 Activity Prediction - Jump AI 2025 Competition

## 📌 대회 개요

### 대회 배경
**Jump AI 2025: 제3회 AI 신약개발 경진대회**는 AI 신약개발 생태계 활성화와 젊은 연구원들의 인재 유입을 목표로 개최된 대회입니다.

### 주제
MAP3K5 (ASK1) IC50 활성값 예측 모델 개발
- **목표**: 화합물의 구조 정보(SMILES)를 입력으로 ASK1 효소에 대한 IC50 값을 예측
- **중요성**: ASK1은 산화 스트레스 관련 질병(심부전, 신경퇴행성 질환 등)의 중요한 타겟 단백질

### 평가 지표
```
Score = 0.4 × (1 - min(A, 1)) + 0.6 × B
```
- **A**: IC50(nM) 단위의 Normalized RMSE
- **B**: pIC50 변환값 기준 예측값과 실제값 간의 선형 상관관계 제곱 (R²)

## 🎯 핵심 과제 분석

### 데이터셋 특성
- **학습 데이터**: 3,834개 분자 (CAS, ChEMBL, PubChem에서 수집)
- **테스트 데이터**: 127개 분자
- **pIC50 분포**: 3.3 ~ 13.0 (매우 넓은 활성 범위)
- **핵심 도전 과제**: 
  - 고활성 분자(pIC50 > 10)는 전체의 0.8% (27개)로 극히 소수
  - 테스트셋에 학습셋과 다른 scaffold를 가진 분자들 존재 예상

### 도메인 관점의 예측 목표
신약 개발에서 가장 중요한 것은 **"Hit Compound"를 놓치지 않는 것**입니다.
1. **최우선 목표**: 고활성 분자(pIC50 > 9)를 정확히 예측
2. **두 번째 목표**: 저활성 분자(pIC50 < 5)를 확실히 걸러내기
3. **결론**: '양 극단'을 잘 맞추는 전문가 모델 개발

## 🚀 솔루션 전략

### 1. 3단계 파이프라인 아키텍처

#### Phase 1: 데이터 준비 및 특징 공학
```python
# 데이터 로드 및 전처리
- train_dataset_with_3source.csv: 3,834개 분자 (source 정보 포함)
- 분자 특징 추출: MolWt, LogP, TPSA, QED 등 14개 RDKit descriptors
- 고활성 분자 증강: pIC50 > 10인 분자에 대해 SMILES Enumeration 3배 적용
```

#### Phase 2: MPNN 임베딩 추출
```python
# ChemELon 사전학습 모델 기반 분자 임베딩
- ChemELon Foundation Model (2048차원) + Fine-tuning
- Message Passing: Bond-based with ChemELon weights
- Aggregation: Mean pooling
- FFN: 2048 → 1024 → 512 → 256 → 1
- 최종 임베딩: 256차원 (FFN 마지막 은닉층)
```

#### Phase 3: CatBoost 최종 예측
```python
# 임베딩 + 테이블 특징으로 최종 예측
- 입력: MPNN 임베딩(256차원) + RDKit 특징(14차원) = 270차원
- 모델: CatBoost with sample weighting
- CV: 5-Fold Stratified (pIC50 기반 그룹)
```

### 2. 핵심 기법

#### 2.1 CombinedBatchLoss (Weighted MSE + Correlation Loss)
```python
class CombinedBatchLoss:
    def __init__(self, alpha=0.4, weight_configs={10: 5, 12: 10}):
        # alpha: WMSE 비중 (대회 지표 0.4와 일치)
        # weight_configs: 고활성 분자 가중치
        # - pIC50 > 10: 5배 가중치
        # - pIC50 > 12: 10배 가중치
```
- **목적**: 대회 평가지표(0.4×RMSE + 0.6×R²)에 최적화
- **효과**: 고활성 분자 예측 정확도 향상 + 상관관계 개선

#### 2.2 차별화된 학습률 전략
```python
# ChemELon과 FFN에 다른 학습률 적용
- ChemELon 파라미터: base_lr × 0.01 (미세조정)
- FFN 파라미터: base_lr × 1.0 (적극적 학습)
- Warmup: 2 epochs with linear scheduling
```

#### 2.3 SMILES Enumeration (고활성 분자만)
```python
# pIC50 > 10인 분자만 선택적 증강
- 각 분자당 3개 alternative SMILES 생성
- GroupID 유지로 데이터 누출 방지
- 효과: 고활성 분자 학습 강화 without overfitting
```

#### 2.4 샘플 가중치 (CatBoost)
```python
def calculate_sample_weights(y):
    weights = np.ones_like(y)
    weights[y > 8] = 2.0    # 고활성
    weights[y > 10] = 5.0   # 초고활성
    weights[y > 12] = 10.0  # 극초고활성
    return weights
```

### 3. 모델 구성 세부사항

#### 3.1 MPNN Architecture
```python
# Message Passing (ChemELon)
- Input: Bond features + Atom features
- Hidden: 2048차원
- Depth: 5 layers
- Activation: ReLU
- Dropout: 0.2

# Feed-Forward Network
- Architecture: [2048, 1024, 512, 256, 1]
- Dropout: 0.3
- Batch Norm: False (ChemELon과 충돌 방지)
```

#### 3.2 CatBoost Hyperparameters
```python
{
    'iterations': 300,
    'learning_rate': 0.08,
    'depth': 7,
    'l2_leaf_reg': 5,
    'min_data_in_leaf': 20,
    'random_strength': 0.5,
    'bagging_temperature': 0.7,
    'border_count': 128,
    'grow_policy': 'Lossguide',
    'max_leaves': 64
}
```

## 📊 실험 결과

### 최종 성능 (exp7)
```python
# Cross-Validation (5-Fold Stratified)
평균 CV RMSE: 0.4507 ± 0.0226
평균 CV R2: 0.9484 ± 0.0055
Competition Score: 0.9506 (A: 0.9535, B: 0.9488)

# Leaderboard
Public LB Score: 0.5689
```

### 예측 분포 분석
```python
# 훈련 데이터 고활성 분자 예측 성능
- pIC50 > 10인 357개 중 357개 정확히 예측 (100%)
- pIC50 > 11인 357개 중 355개 정확히 예측 (99.4%)

# 테스트 데이터 예측 분포
- 평균: 7.11
- 표준편차: 0.43
- 최소값: 6.07
- 최대값: 8.11
- 핵심 분자(TEST_015~017) 평균 pIC50: 6.74
```

### 주요 실험 히스토리
| 실험 | 방법 | CV RMSE | LB Score | 핵심 개선점 |
|------|------|---------|----------|------------|
| exp1 | Base D-MPNN | 0.288 | 0.418 | 기본 D-MPNN 구현 |
| exp2 | +Source Feature | 0.276 | 0.379 | 데이터 출처 피처 추가 (실패) |
| exp3 | +AutoGluon Meta | 0.279 | 0.300 | AutoGluon 메타 모델 (부적합) |
| exp6 | MPNN Embedding+Tree | 0.473 | 0.535 | 임베딩 + Tree 모델 결합 |
| **exp7** | **+Weighted Loss+SMILES Aug** | **0.451** | **0.569** | **고활성 가중치 + 선택적 증강** |

## 🔍 핵심 인사이트

### 1. "진정한 일반화"를 위한 접근
- **문제**: 모델이 다수를 차지하는 중간 활성 분자(pIC50 6-8)에 과적합
- **해결**: CombinedBatchLoss로 고활성 분자에 5-10배 가중치 부여
- **결과**: 훈련 데이터의 고활성 분자 예측 정확도 100% 달성

### 2. 3단계 파이프라인의 효과
- **MPNN 임베딩**: 분자 구조의 deep representation 학습
- **RDKit Features**: 화학적 특성의 explicit encoding  
- **CatBoost**: 비선형 관계 포착 및 robust prediction
- **시너지**: 임베딩+테이블 특징 결합으로 성능 향상

### 3. ChemELon Transfer Learning
- **사전학습 활용**: 2048차원의 풍부한 분자 표현
- **차별화된 학습률**: ChemELon 0.01x, FFN 1.0x
- **효과**: 과적합 방지하면서도 task-specific 학습 가능

## 📁 디렉토리 구조

```
MAP3K5_JumpAI2025_Competition/
├── README.md
├── data/
│   ├── train_dataset_with_3source.csv # 출처 정보 포함 학습 데이터
│   ├── test.csv
│   └── sample_submission.csv
├── exp7/
│   ├── MPNN_Embedding_DataAG.ipynb    # 최종 솔루션 (Jupyter)
│   ├── MPNN_Embedding_DataAG.py       # 최종 솔루션 (Python)
│   ├── best_catboost_model.cbm        # 학습된 CatBoost 모델
│   └── best_chemprop_model.pt         # 학습된 MPNN 모델
├── data_transform.ipynb               # 데이터 전처리
├── 참고논문/                           # 참고 논문
│   ├── MPNN.pdf
│   ├── Chemprop A Machine Learning Package.pdf
│   └── ...
├── 대회정보.txt                        # 대회 상세 정보
├── 실험노트.txt                        # 실험 기록
├── EDA.txt                            # 탐색적 데이터 분석
└── 평가지표.jpg                        # 평가 지표 설명

## 🛠 환경 설정

### Requirements
```bash
# 주요 라이브러리
- Python 3.8+
- PyTorch 1.13+
- RDKit 2023.03+
- ChemProp 1.5+
- CatBoost 1.2+
- scikit-learn 1.3+
- pandas, numpy, matplotlib
```

### 실행 방법
```bash
# 1. 데이터 전처리 (이미 완료됨)
# data/train_dataset_with_3source.csv 파일 사용

# 2. 모델 학습 및 예측
cd exp7
python MPNN_Embedding_DataAG.py
# 또는 Jupyter로 실행
jupyter notebook MPNN_Embedding_DataAG.ipynb

# 3. 제출 파일 생성
# submission_catboost.csv 자동 생성됨
```

## 📚 참고 문헌

1. **MPNN (Message Passing Neural Networks)**
   - Gilmer et al., "Neural Message Passing for Quantum Chemistry", ICML 2017

2. **ChemProp**
   - Yang et al., "Analyzing Learned Molecular Representations for Property Prediction", JCIM 2019

3. **Scaffold Hopping**
   - Sun et al., "Recent Advances in Scaffold Hopping", Expert Opinion 2017
   - Recent Applications in CNS Drug Discovery, 2022

4. **Transfer Learning in Drug Discovery**
   - ChemELon: Large-scale molecular property prediction

---
*이 프로젝트는 Jump AI 2025 제3회 AI 신약개발 경진대회 참가작입니다.*