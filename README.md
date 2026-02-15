# 👁️ AI 기반 안구 질환 통합 진단 시스템

**Papyrus Project: Jetson Orin Nano & Raspberry Pi 5 통합 솔루션**
Medical-Grade AI Inference on Edge Devices

---

## 🚀 프로젝트 개요

본 프로젝트는 임베디드 환경(Jetson Orin Nano)에서 딥러닝 파이프라인을 구축하여 **5종의 주요 안구 질환을 정밀 진단**하고, **충혈도를 정량적으로 산출**하는 하드웨어-소프트웨어 통합 시스템입니다.

- ✅ 실시간 추론 (Real-time Inference on Edge)
- ✅ 의료용 정확도 (Medical-Grade Accuracy)
- ✅ 프라이버시 보호 (On-Device Processing)

---

## 🌟 핵심 기능

- 📸 **스냅샷 기반 추론**: 가이드라인 UI와 연동하여 버튼 클릭 시 단회 추론을 수행함으로써 젯슨의 연산 자원을 최적화합니다.
- 🎯 **정밀 ROI 추출**: YOLOv8n 모델을 통해 전체 화면에서 안구 영역만 0.8 이상의 신뢰도로 정확히 탐지 및 크롭합니다.
- 🧠 **고성능 질환 분류**: 증강 데이터셋(Augmented Dataset)으로 학습된 EfficientNet V1 B0를 통해 **99.09% 정확도**를 달성했습니다.
- 📊 **충혈 지수 정량화**: OpenCV 기반의 홍채 제거 알고리즘 및 `Lab` 컬러 공간 분석을 통해 충혈도를 수치화합니다.

---

## 🛠️ 기술 스택

| 분류      | 기술 & 라이브러리                        | 비고                |
|-----------|------------------------------------------|---------------------|
| Hardware  | NVIDIA Jetson Orin Nano, Raspberry Pi 5  | Edge AI 추론 장비   |
| Deep Learning | PyTorch, Ultralytics (YOLOv8), EfficientNet, ResNet50 | 모델 학습 & 추론 |
| Vision    | OpenCV 4.x (Sclera Extraction, Redness Analysis) | 이미지 처리 & 분석 |
| Backend   | Flask, FastAPI, SQLite3                  | RESTful API & 데이터 관리 |
| Frontend  | HTML5, CSS3, JavaScript (Vanilla)        | 웹 기반 UI & 실시간 스트림 |
| Security  | SHA256 Hashing, HMAC, PEPPER             | 사용자 데이터 보호  |

---

## 📊 실험 결과 & 기술 명세

### 🏆 모델 성능 지표

EfficientNet V1 B0를 메인 엔진으로 채택했습니다.

| 성능 지표              | 수치         | 해석                |
|------------------------|--------------|---------------------|
| Best Validation Accuracy | 99.09%      | 증강 데이터셋 기준  |
| Inference Latency      | 8.58 ms      | Jetson Orin Nano 기준 |
| Model Size             | ~21 MB       | 메모리 효율적       |
| Supported Classes      | 5종          | Normal, Conjunctivitis, Uveitis, Cataract, Eyelid |

---

## 🔬 핵심 기술 파이프라인

본 시스템은 **5단계의 정밀 기술 파이프라인**을 통해 안구 질환을 진단합니다. 각 단계에서 수학적 엄밀성과 의료용 정확도를 보장합니다.

---

### 1️⃣ YOLOv8 기반 실시간 정렬 및 오토 셔터 (Alignment)

**목적:** 사용자의 안구가 카메라 가이드라인 UI 내에 정확히 위치했을 때 자동으로 캡처를 수행함

**중심점 거리 (ΔD):** 검출된 박스의 중심과 가이드라인 중심 사이의 오차 산출

$$
\Delta D = \sqrt{(x_d - x_g)^2 + (y_d - y_g)^2} \leq 30\text{ px}
$$

변수 설명:
- $(x_d, y_d)$: 검출된 눈 박스의 중심점
- $(x_g, y_g)$: 가이드라인 중심점
- 30px: 자동 촬영 허용 오차 범위 (config.py의 AUTO\_DIST\_THRESHOLD)

**크기 일치율 (S_r):** 가이드라인 대비 안구의 배율 확인

$$
S_r = \frac{W_{\text{detected}}}{W_{\text{guide}}} \in [0.8, 1.1]
$$

변수 설명:
- $W_{\text{detected}}$: 검출된 안구 영역의 너비
- $W_{\text{guide}}$: 가이드라인의 너비 (표준값: 380px)
- [0.8, 1.1]: 크기 비율이 80~110% 범위일 때 촬영 가능

✅ 자동 촬영 조건: 조건 1과 2를 모두 만족 + AUTO\_CAPTURE\_HOLD\_FRAMES(10프레임) 유지 → 자동 캡처 실행

---

### 2️⃣ 공막(Sclera) 정밀 추출 및 전처리 (Segmentation)

**목적:** 진단에 불필요한 홍채(Iris) 영역을 제거하고 순수한 공막(흰자) 픽셀만 추출함

**허프 원 변환 (Hough Circle):** 홍채 경계면 탐지

$$
\left(x - a\right)^2 + \left(y - b\right)^2 = r^2
$$

변수 설명:
- $(a, b)$: 홍채 중심점
- $r$: 홍채 반지름
- 이 원형 마스크를 이용하여 홍채 영역을 식별

**최종 ROI 산출:** 홍채 마스크와 밝기 기반 공막 마스크의 결합

$$
\text{ROI}_{\text{final}} = \left(1 - M_{\text{iris}} \times \alpha\right) \cap M_{\text{sclera}}
$$

변수 설명:
- $M_{\text{iris}}$: 홍채 영역 마스크 (이진 1/0)
- $M_{\text{sclera}}$: 밝기 임계값 기반 공막 마스크
- $\alpha$: 홍채 주변부 노이즈 제거를 위한 가중치 패딩 (보통 0.1~0.2)
- 최종 결과: 순수한 공막 영역만 추출

✅ 처리 단계: BGR→HSV→Hough Circle(홍채 탐지)→마스크 생성→공막 영역만 추출→충혈 분석 준비

---

### 3️⃣ 정량적 충혈도 지수 산출 (Redness Scoring)

**목적:** CIE Lab 컬러 공간을 활용하여 주관적인 충혈 상태를 객관적인 수치로 변환함

**Redness Index (R\_score):** 공막 영역 내 붉은색 채도의 평균값 분석

$$
R_{\text{score}} = \left(\frac{1}{N} \sum_{i=1}^{N} a^*_i\right) - 128
$$

변수 설명:
- $a^*_i$: ROI 영역 내 i번째 픽셀의 Lab 색공간 a* 값
- $N$: ROI 영역 내 총 픽셀 수
- 128: 정상 안구의 기준값 (중립점)
- 음수: 초록 우세 (정상, 건강한 상태)
- 양수: 빨강 우세 (충혈, 염증 상태)

📊 충혈도 해석:
- R\_score = -10 ~ 10: 정상 (Normal)
- R\_score = 10 ~ 20: 경미한 충혈 (Mild Redness)
- R\_score = 20 ~ 30: 중등도 충혈 (Moderate Redness)
- R\_score > 30: 심한 충혈 (Severe Redness)

#### 색공간 변환 파이프라인

$$
\text{BGR (OpenCV)} \xrightarrow{\text{cv2.cvtColor}} \text{HSV} \xrightarrow{\text{Iris Removal}} \text{Sclera ROI}
$$

$$
\text{Sclera ROI} \xrightarrow{\text{cv2.cvtColor}} \text{Lab} \xrightarrow{\text{Extract } a^*} R_{\text{score}}
$$

---

### 4️⃣ 딥러닝 기반 다중 질환 분류 (Classification)

**목적:** 5종 안구 질환(정상, 결막염, 백내장, 포도막염, 다래끼)을 판별함

**EfficientNet V1 B0 모델:**

$$
P(c_i | I) = \text{softmax}\left(\text{EfficientNet}_{\theta}(I)\right)_i
$$

변수 설명:
- $I$: 입력 안구 이미지 (224×224 RGB)
- $\theta$: 학습된 모델 가중치
- $c_i$: i번째 질환 클래스 (0~4)
- $P(c_i|I)$: 이미지 I가 질환 $c_i$일 확률

**5종 분류:**

| 클래스   | 질환                | 특징           |
|----------|---------------------|----------------|
| Class 0  | 정상 (Normal)       | 건강한 안구    |
| Class 1  | 결막염 (Conjunctivitis) | 결막 염증, 충혈 |
| Class 2  | 포도막염 (Uveitis) | 포도막 염증    |
| Class 3  | 백내장 (Cataract)   | 수정체 혼탁    |
| Class 4  | 다래끼 (Eyelid)     | 눈꺼풀 감염    |

📊 성능 지표:
- Target Accuracy: 99.09%
- Inference Time: 8.58ms
- Model Size: ~21MB
- 플랫폼: Jetson Orin Nano (최적화 완료)

---

### 5️⃣ 보안 기반 고유 식별자 생성 (Security Hashing)

**목적:** 개인정보인 전화번호를 직접 저장하지 않고 보안 해시값으로 관리함 (프라이버시 보호)

**Salted SHA-256 해싱:** 전화번호와 서버 비밀키의 결합 해싱

$$
ID_{\text{user}} = \text{SHA256}(P \mid K)
$$

변수 설명:
- $P$: 정규화된 전화번호 (010-1234-5678 → 01012345678)
- $K$: 서버 비밀키 (HASH\_PEPPER 환경변수)
- $\mid$: 문자열 연결 연산자
- $ID_{\text{user}}$: 256비트 고유 식별자 (해시값)

**보안 특징:**
- 단방향 암호화: SHA-256은 복호화 불가능 (일방성 함수)
- Salt 추가: HASH\_PEPPER 키를 섞어 사전 공격 방어
- 원전 번호 미저장: 데이터베이스에는 해시값만 저장
- HMAC 검증: 데이터 무결성 확인용 메시지 인증 코드

✅ 구현 위치:
- 파일: `utils/security_utils.py`
- 함수: `phone_hash_id(phone: str) -> str`
- 사용: `database/db.py`의 `upsert_user_by_phone()`

---

## 📂 프로젝트 구조

```
eye_project/
├── eye_server.py              # 전체 시스템 제어 및 UI 연동
├── eye_server_test.py         # 테스트 모드 (DB/보안 통합)
├── config.py                  # 모델 경로, 임계값, 자동 촬영 설정
├── model_loader.py            # 모델 초기화 및 캐싱
│
├── models/
│   ├── Augmented_EffNet_V1_B0_best.pth    # EfficientNet 분류 모델 (99.09%)
│   └── yolov8n_eye.pt                     # YOLOv8n 눈 탐지 모델
│
├── modules/
│   ├── detector.py                        # YOLO 기반 눈 탐지
│   ├── classifier.py                      # EfficientNet 기반 질환 분류
│   ├── analyzer.py                        # 충혈도 산출 및 픽셀 분석
│   └── __init__.py
│
├── database/
│   ├── db.py                              # SQLite 진단 데이터 관리
│   └── schema.sql                         # DB 스키마 (users, diagnosis_sessions 등)
│
├── utils/
│   ├── image_proc.py                      # 이미지 전처리 & 호남색 제거
│   ├── logger.py                          # 진단 결과 기록
│   └── security_utils.py                  # SHA256 해싱, HMAC 검증
│
├── web/
│   ├── static/
│   │   ├── captures/                      # 촬영 이미지 저장
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── js/
│   │   │   └── main.js                    # 실시간 스트림 & 진단 로직
│   │   └── images/
│   │
│   └── templates/
│       ├── index.html                     # 메인 페이지
│       ├── capture.html                   # 촬영 화면 (가이드라인 UI)
│       ├── result.html                    # 진단 결과 페이지
│       ├── survey.html                    # 사용자 설문조사
│       ├── login.html                     # 로그인 페이지
│       └── layout.html                    # 기본 레이아웃
│
├── README.md                 # 이 파일
├── eye_diagnosis.db          # SQLite 데이터베이스 (런타임)
└── requirements.txt          # Python 패키지 의존성
```

---

## 👥 팀 구성 (Team Papyrus)

**Team Papyrus**는 AI, 하드웨어, 의료 데이터 분야의 전문가 5명으로 구성되어 있으며, 각자의 전문성을 바탕으로 **의료용 임베디드 AI 시스템**을 구축했습니다.

| 팀원   | 담당 영역         | 핵심 역할 |
|--------|-------------------|-----------|
| 박하준 (PM) | AI 모델링 및 최적화 | EfficientNet 정확도(99.09%) 고도화, YOLOv8 오토 셔터 로직 설계, Jetson TensorRT 엔진 최적화 |
| 우세준 | 시스템 통합 및 데이터 검증 | 개별 모듈(YOLO, EffNet, DB) 간 파이프라인 통합, 데이터셋 무결성 검증, 모델 신뢰도 테스트 |
| 김영록 | 하드웨어 설계 및 광학 시스템 | 키오스크 외형 설계, 안구 촬영 최적 조명(LED) 제어, 젯슨-카메라 인터페이스 하드웨어 구성 |
| 김세은 | GUI 개발 및 데이터 시각화 | 사용자 가이드라인 실시간 인터페이스 개발, 진단 결과 차트 및 대시보드 시각화 구현 |
| 나은율 | 데이터베이스 및 의료 연계 | 보안 해싱 기반 DB 설계, 의료 표준(FHIR) 연계 데이터 관리, 리포트 자동 생성 시스템 구축 |

### 팀의 강점
- 🔬 기술 다양성: AI, 임베디드, 하드웨어, 의료 정보를 아우르는 통합 역량
- 🏥 의료 정확도: 의료 데이터 표준(FHIR) 준수 및 보안 규정 준수
- ⚡ 성능 최적화: 엣지 디바이스에서 의료용 정확도 유지
- 🔒 보안 & 프라이버시: 온디바이스 처리 및 암호화된 사용자 데이터 관리

---

## 📚 참고 자료 및 라이선스

- 논문: EfficientNet, YOLOv8 학술 논문
- 데이터셋: 직접 수집 및 증강 (1,500+ 이미지)
- 라이선스: MIT License
- 저장소: [github.com/papyrus-project/eye_project](https://github.com/papyrus-project/eye_project)

---