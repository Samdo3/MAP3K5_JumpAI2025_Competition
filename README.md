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

### 1. 데이터 전처리 및 증강

#### 1.1 데이터 통합 (`data_transform.ipynb`)
```python
# 3개 데이터 소스에서 SMILES와 pIC50 추출 및 통합
- CAS: 3,430개 → SMILES 정규화 → 중복 제거
- ChEMBL: 714개 → pChEMBL Value 활용
- PubChem: 1,148개 → Activity_Value를 pIC50로 변환
→ 최종: 3,834개 unique 분자
```

#### 1.2 고활성 분자 데이터 증강
```python
# SMILES Enumeration: 분자 표현의 다양성 활용
- 고활성 분자(pIC50 > 10)에 대해 3배 증강
- RDKit을 이용한 동일 분자의 다른 SMILES 표현 생성
- 목적: 모델이 고활성 분자의 다양한 표현을 학습하도록 유도
```

### 2. 모델 아키텍처

#### 2.1 MPNN (Message Passing Neural Network) + Pretrained Embedding
```python
# ChemProp 기반 D-MPNN with Transfer Learning
- 백본: ChemProp의 MPNN 아키텍처
- 사전학습: ChemELon 가중치 활용 (대규모 분자 데이터로 학습된 표현)
- Fine-tuning: 전체 파라미터 학습 가능하도록 설정
- Hidden: 600차원, Depth: 7층, Dropout: 0.3
```

#### 2.2 임베딩 기반 앙상블
```python
# MPNN으로 추출한 분자 임베딩을 Tree 모델들의 입력으로 활용
1. MPNN으로 600차원 분자 임베딩 추출
2. ECFP (Extended Connectivity Fingerprint) 특징 추가
3. CatBoost 앙상블 학습
```

### 3. 핵심 기법

#### 3.1 가중치 기반 손실 함수 (Weighted Loss)
```python
class CombinedBatchLoss:
    """고활성 분자에 높은 가중치를 부여하는 손실 함수"""
    def __init__(self, weight_configs={10: 5, 12: 10}):
        # pIC50 > 10: 5배 가중치
        # pIC50 > 12: 10배 가중치
        self.weight_configs = weight_configs
```
- **목적**: 모델이 소수의 고활성 분자를 무시하지 않도록 강제
- **효과**: 전체 평균 오차보다 고활성 분자 예측에 집중

#### 3.2 Scaffold 편향 제거
```python
# 특정 scaffold에 과적합되지 않도록 처리
- 학습 시 scaffold 정보를 피처로 사용하지 않음
- 대신 부분 구조(ECFP) 특징을 활용하여 일반화 능력 향상
```

#### 3.3 계층적 교차 검증 (Stratified K-Fold)
```python
# pIC50 그룹별 균등 분할
- 초고활성(pIC50 > 10), 고활성(8 < pIC50 ≤ 10), 일반(pIC50 ≤ 8)
- 각 Fold에 모든 그룹이 균등하게 포함되도록 보장
```

### 4. 후처리 기법

#### 4.1 Isotonic Regression Calibration
```python
# 예측값 보정을 통한 성능 향상
- OOF 예측값을 isotonic regression으로 보정
- 단조성을 유지하면서 예측 분포 개선
- Competition Score: 0.96924 → 0.97036
```

## 📊 실험 결과

### 최종 성능
- **Local CV Score**: 0.9472 (RMSE: 0.3313, R²: 0.9353)
- **Public LB Score**: 0.5689
- **예측 범위**: pIC50 5.87 ~ 8.11

### 주요 실험 히스토리
| 실험 | 방법 | CV Score | LB Score | 핵심 개선점 |
|------|------|----------|----------|------------|
| exp1 | Base D-MPNN | 0.968 | 0.418 | 기본 구현 |
| exp2 | +Source Feature | 0.970 | 0.379 | 성능 하락 |
| exp3 | +AutoGluon | 0.972 | 0.300 | 메타 모델 부적합 |
| exp6 | MPNN Embedding+Tree | 0.874 | 0.535 | 임베딩 활용 |
| **exp7** | **+Weighted Loss+증강** | **0.947** | **0.569** | **고활성 집중** |

## 🔍 핵심 인사이트

### 1. "진정한 일반화"를 위한 접근
- **문제**: 모델이 다수를 차지하는 중간 활성 분자에 과적합되어 고활성 분자를 무시
- **해결**: 가중치 손실 함수로 고활성 분자 학습 강제

### 2. Scaffold Hopping 대응
- **문제**: 테스트셋이 학습셋과 다른 scaffold 구조를 가질 가능성
- **해결**: 전체 구조 대신 부분 구조(ECFP) 특징 활용

### 3. 사전학습 모델의 활용
- **ChemELon**: 대규모 분자 데이터로 학습된 표현 활용
- **Fine-tuning**: 전체 파라미터를 task-specific하게 조정

## 📁 디렉토리 구조

```
MAP3K5_JumpAI2025_Competition/
├── README.md
├── data/
│   ├── train_dataset_final_pIC50.csv  # 통합된 학습 데이터
│   ├── test.csv
│   └── sample_submission.csv
├── exp7/
│   ├── MPNN_Embedding_DataAG.ipynb    # 최종 솔루션
│   └── models/                        # 학습된 모델 가중치
├── data_transform.ipynb               # 데이터 전처리
├── papers/                             # 참고 논문
│   ├── MPNN.pdf
│   ├── Chemprop A Machine Learning Package.pdf
│   └── ...
└── docs/
    ├── 대회정보.txt
    ├── 실험노트.txt
    └── 평가지표.jpg
```

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
# 1. 데이터 전처리
python data_transform.ipynb

# 2. 모델 학습 및 예측
python exp7/MPNN_Embedding_DataAG.ipynb

# 3. 제출 파일 생성
# submission_catboost.csv 자동 생성
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

## 🏆 결론 및 향후 개선 방향

### 성과
- 고활성 분자 예측에 특화된 모델 개발
- 가중치 손실 함수를 통한 불균형 데이터 문제 해결
- 사전학습 모델과 도메인 지식의 효과적 결합

### 한계 및 개선 방향
1. **예측 범위 제한**: 테스트셋에서 pIC50 > 9 예측 실패
   - 원인: 학습 데이터의 고활성 분자 부족
   - 개선: 더 많은 외부 고활성 데이터 수집 필요

2. **Generalization Gap**: CV와 LB 점수 차이
   - 원인: 테스트셋의 분포가 학습셋과 상이
   - 개선: Domain Adaptation 기법 적용 검토

3. **앙상블 최적화**
   - 다양한 아키텍처(GNN, Transformer) 결합
   - Stacking 전략 개선

## 👥 Contact

질문이나 피드백이 있으시면 이슈를 남겨주세요!

---
*이 프로젝트는 Jump AI 2025 제3회 AI 신약개발 경진대회 참가작입니다.*