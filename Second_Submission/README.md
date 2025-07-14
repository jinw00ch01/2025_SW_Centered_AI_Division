# 한국어 AI 텍스트 탐지 - 성능 향상 프로젝트

**성능 목표: ROC-AUC 0.757 → 0.93+**

## 📁 프로젝트 구조

```
Second_Submission/
├── 01_data_preparation.ipynb      # 데이터 준비 및 기본 모델 훈련
├── 02_hierarchical_modeling.ipynb # 계층적 문서-문단 모델링  
├── 03_korean_features.ipynb       # 한국어 특화 특성 공학
├── 04_advanced_ensemble.ipynb     # 고급 앙상블 시스템
├── 05_final_inference.ipynb       # 최종 추론 및 후처리
└── README.md                      # 사용법 안내
```

## 🚀 실행 순서

### 1단계: 데이터 준비 및 기본 모델 훈련
```bash
# Colab에서 01_data_preparation.ipynb 실행
```
- A100 GPU 최적화 설정
- 클래스 불균형 해결 (가중치 적용)
- RoBERTa-large 모델 훈련
- 안정적 훈련 (Learning rate 조정, Gradient clipping)

**출력 파일:**
- `base_model/` (훈련된 모델)
- `train_para_df.pkl` (문단 단위 훈련 데이터)
- `test_df.pkl` (테스트 데이터)
- `data_info.pkl` (메타데이터)
- `step1_metadata.json`

### 2단계: 계층적 모델링 
```bash
# 02_hierarchical_modeling.ipynb 실행
```
- 글 단위 컨텍스트 모델링
- 문단 간 어텐션 메커니즘
- 위치 임베딩 활용
- 일관성 체크 레이어

**출력 파일:**
- `hierarchical_model_best.pth`
- `step2_metadata.json`

### 3단계: 한국어 특화 특성 공학
```bash
# 03_korean_features.ipynb 실행  
```
- 한국어 어미 패턴 분석 (습니다, 입니다 등)
- 접속사 및 전이 표현 특성
- 조사 패턴 분석 (은/는, 이/가 등)
- 문체 일관성 측정

**출력 파일:**
- `korean_features_results.pkl`
- `step3_metadata.json`

### 4단계: 고급 앙상블 시스템
```bash
# 04_advanced_ensemble.ipynb 실행
```
- 다중 뷰 앙상블 (기본모델 + 계층적모델 + 한국어특성모델)
- 스태킹 메타 학습
- 동적 가중치 조정
- 성능 기반 모델 융합

**출력 파일:**
- `ensemble_results.pkl`
- `step4_metadata.json`

### 5단계: 최종 추론 및 후처리
```bash
# 05_final_inference.ipynb 실행
```
- 글 단위 일관성 강제
- 순서 정보 기반 가중치
- 분포 맞춤 후처리
- 랭킹 기반 보정

**출력 파일:**
- `final_submission.csv` (메인 제출 파일)
- `submission_*.csv` (다양한 버전들)

## 🎯 핵심 개선사항

### 1. 계층적 문서-문단 모델링 (0.75 → 0.85 예상)
- **문제**: 기존 모델은 각 문단을 독립적으로 처리
- **해결**: 글 전체 컨텍스트와 문단 간 관계를 모델링
- **기법**: Multi-head Attention, Position Encoding, Document-level Transformer

### 2. 한국어 AI 특화 특성 (0.85 → 0.90 예상)  
- **문제**: 언어학적 특성 무시
- **해결**: 한국어 AI 텍스트의 고유 패턴 활용
- **특성**: 어미 획일성, 접속사 과다 사용, 조사 패턴, 어휘 다양성 부족

### 3. 고급 앙상블 시스템 (0.90 → 0.92 예상)
- **문제**: 단일 모델의 한계
- **해결**: 다중 뷰 앙상블 + 메타 학습
- **구성**: RoBERTa + 계층적모델 + 한국어특성모델 + XGBoost

### 4. 일관성 보정 후처리 (0.92 → 0.93+ 예상)
- **문제**: 같은 글 내 문단들의 예측 불일치
- **해결**: 글 단위 일관성 강제 + 위치 가중치
- **기법**: Smoothing, Outlier detection, Distribution matching

## ⚙️ A100 GPU 최적화

- **모델**: RoBERTa-base → RoBERTa-large
- **배치 크기**: 16 → 32 (+ gradient accumulation)
- **시퀀스 길이**: 256 → 512  
- **Mixed Precision**: FP16 활용
- **메모리 활용**: 40GB A100 최대 활용

## 📊 예상 성능 개선

| 단계 | 개선사항 | 예상 AUC |
|------|----------|----------|
| 기존 | first_Submission.ipynb | 0.757 |
| 1단계 | 기본 모델 안정화 | 0.80 |
| 2단계 | + 계층적 모델링 | 0.85 |
| 3단계 | + 한국어 특화 특성 | 0.90 |
| 4단계 | + 고급 앙상블 | 0.92 |
| 5단계 | + 일관성 보정 | **0.93+** |

## 🔧 문제 해결 가이드

### GPU 메모리 부족 시
- `BATCH_SIZE`를 16 또는 8로 줄이기
- `MAX_LENGTH`를 384로 줄이기
- `GRADIENT_ACCUMULATION`을 늘려서 effective batch size 유지

### 모델 로딩 실패 시
- 01단계에서 `base_model_backup` 폴더도 생성됨
- 백업 모델 사용하거나 단계별로 다시 실행

### 특성 추출 오류 시
- 텍스트 전처리 문제일 가능성
- 빈 텍스트나 특수문자 확인

## 📈 성능 모니터링

각 단계별 `step*_metadata.json` 파일에서 성능 확인 가능:
```json
{
  "model_type": "hierarchical",
  "best_auc": 0.85,
  "improvements": ["document_context", "attention", ...]
}
```

## ✅ 검증 방법

1. **교차 검증**: Title 기준 분할로 데이터 누수 방지
2. **일관성 체크**: 같은 글 내 문단들의 예측 분산 모니터링  
3. **분포 검증**: 훈련 데이터와 예측 분포 비교
4. **단계별 성능**: 각 개선사항의 기여도 측정

## 🎁 추가 제출 파일

- `submission_basic_ensemble.csv`: 기본 앙상블만
- `submission_consistency_adjusted.csv`: 일관성 보정 적용
- `submission_distribution_adjusted.csv`: 분포 맞춤 적용
- `submission_final_calibrated.csv`: 모든 후처리 적용 (권장)

**메인 제출**: `final_submission.csv`

---

## 💡 핵심 아이디어

이 프로젝트의 핵심은 **문제 정의에서 허용된 "같은 title의 문단 간 상호 참조"를 최대한 활용**하는 것입니다. 

기존 접근법은 각 문단을 독립적으로 처리했지만, 실제로는:
- AI가 생성한 글이라면 모든 문단이 유사한 AI 특성을 보임
- 인간이 쓴 글이라면 개인적 글쓰기 스타일이 일관적임

이를 활용한 계층적 모델링과 일관성 보정이 성능 향상의 핵심입니다. 