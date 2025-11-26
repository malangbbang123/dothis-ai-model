# 🎥 YouTube Views Prediction — Multi-Modal Model

이 저장소는 비디오의 메타데이터 + 태그 + 기타 특성(feature)을 활용해,  
YouTube 영상의 조회수(views)를 예측하는 **멀티모달 딥러닝 파이프라인**을 구현한 프로젝트입니다.

---

## 📂 Repository Structure

dothis-ai-model/

│
├── load_data.py # raw 데이터 로딩 및 기본 전처리 스크립트
├── preprocessing.py # 피처 엔지니어링 및 데이터 정제 모듈
├── MultiModal.py # 메타데이터 + 태그 + 수치/범주 feature를 활용한 모델 정의 & 학습
├── test.ipynb # 실험 및 결과 확인용 Jupyter Notebook
└── README.md # 이 파일 (프로젝트 설명서)


---

## 🔎 Project Description

- **목표:** YouTube 동영상의 메타데이터(제목, 업로드 날짜, 카테고리, 태그, 설명 등) + 기타 부가 feature를 바탕으로,  
  → 동영상 공개 후 일정 시점의 **조회수(views)** 를 예측  

- **왜 중요한가:**  
  - 영상 크리에이터 및 플랫폼 측면에서 조회수 예측은 콘텐츠 기획, 수익 예측, 추천 최적화 등에 유용  
  - 단순 텍스트/메타데이터뿐 아니라 다중 feature를 조합해 예측 성능을 높이는 게 핵심  

- **접근 방법:**  
  1. raw metadata + tags + 수치/범주 피처를 로드 & 정제 (`load_data.py`, `preprocessing.py`)  
  2. 피처 엔지니어링: 범주형 인코딩, 날짜 처리, 태그 처리, 수치 스케일링 등  
  3. 멀티모달 모델 (`MultiModal.py`): 범주형 + 수치형 + 텍스트/태그 feature를 함께 사용하는 모델  
  4. 실험 및 성능 분석 — `test.ipynb`에서 결과 확인  

---

## 🛠️ How to Run

### 1. 데이터 준비  
- YouTube 메타데이터 CSV (title, tags, upload_date, category, view_count 등) 준비  
- `load_data.py`에서 경로 설정 후 실행 → 기본 정제된 DataFrame 생성  

python load_data.py 

### 2. 피처 전처리 & 정제  
python preprocessing.py

### 3. 모델 학습 / 예측  
python MultiModal.py



---

## 📈 Output & Evaluation

- 모델 출력: 예측된 조회수 (views) 또는 로그 변환된 조회수  
- 평가 지표 예시: MAE, RMSE, R², 또는 필요시 분류형 지표 (예: 조회수 상위 10% 예측 성공 여부)  
- `test.ipynb`에서 실제 예측 결과 및 분석 플롯 제공  

---

## 👤 Author

- **작성자**: 홍지은 (Jieun Hong)  
- **역할**: 데이터 수집 · 전처리 · 모델 설계 및 구현 · 결과 분석  


