# eye_project

<style>
:root {
  --bg-color: #121212;
  --text-color: #e0e0e0;
  --card-bg: #1e1e1e;
  --accent: #00f2fe;
  --primary: #4facfe;
  --secondary: #00d9ff;
  --border-color: #2a2a2a;
  --success: #4caf50;
  --warning: #ff9800;
}

body {
  background-color: var(--bg-color);
  color: var(--text-color);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.hero-section {
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  padding: 40px;
  border-radius: 12px;
  margin: 20px 0;
  box-shadow: 0 8px 24px rgba(79, 172, 254, 0.3);
}

.hero-section h1 {
  color: #ffffff;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.card {
  background-color: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 20px;
  margin: 15px 0;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.accent-text {
  color: var(--accent);
  font-weight: 600;
}

.primary-text {
  color: var(--primary);
  font-weight: 600;
}

.stat-badge {
  display: inline-block;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color: white;
  padding: 8px 16px;
  border-radius: 20px;
  margin: 5px;
  font-weight: bold;
}

.dashboard-container {
  background: linear-gradient(135deg, var(--card-bg) 0%, #252525 100%);
  border: 2px solid var(--accent);
  border-radius: 12px;
  padding: 30px;
  margin: 20px 0;
  box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
}

.tech-table {
  width: 100%;
  border-collapse: collapse;
  margin: 15px 0;
}

.tech-table td, .tech-table th {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.tech-table th {
  background-color: var(--primary);
  color: white;
  font-weight: bold;
}

.tech-table tr:hover {
  background-color: var(--border-color);
}

.formula-block {
  background-color: var(--card-bg);
  border-left: 4px solid var(--accent);
  padding: 15px;
  margin: 15px 0;
  border-radius: 4px;
  font-style: italic;
}
</style>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>

<!-- MathJax Configuration for LaTeX Rendering -->
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true,
      processEnvironments: true
    },
    chtml: {
      fontURL: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2'
    }
  };
</script>

<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<div align="center" class="hero-section">
  <h1>👁️ AI 기반 안구 질환 통합 진단 시스템</h1>
  <p><strong>Papyrus Project: Jetson Orin Nano & Raspberry Pi 5 통합 솔루션</strong></p>
  <p style="font-size: 0.9em; opacity: 0.9;">Medical-Grade AI Inference on Edge Devices</p>
</div>

<hr>

<h2>🚀 프로젝트 개요</h2>
<div class="card">
  <p>본 프로젝트는 임베디드 환경(Jetson Orin Nano)에서 딥러닝 파이프라인을 구축하여 <span class="primary-text">5종의 주요 안구 질환을 정밀 진단</span>하고, <span class="accent-text">충혈도를 정량적으로 산출</span>하는 하드웨어-소프트웨어 통합 시스템입니다.</p>
  <p style="color: #4caf50;">✓ 실시간 추론 (Real-time Inference on Edge)</p>
  <p style="color: #4caf50;">✓ 의료용 정확도 (Medical-Grade Accuracy)</p>
  <p style="color: #4caf50;">✓ 프라이버시 보호 (On-Device Processing)</p>
</div>

<h2>🌟 핵심 기능</h2>
<div class="card">
  <ul style="line-height: 1.8;">
    <li>
      <span class="primary-text">📸 스냅샷 기반 추론 (Snapshot-based Inference):</span> 
      가이드라인 UI와 연동하여 버튼 클릭 시 단회 추론을 수행함으로써 젯슨의 연산 자원을 최적화합니다.
    </li>
    <li>
      <span class="primary-text">🎯 정밀 ROI 추출:</span> 
      YOLOv8n 모델을 통해 전체 화면에서 안구 영역만 0.8 이상의 신뢰도로 정확히 탐지 및 크롭합니다.
    </li>
    <li>
      <span class="primary-text">🧠 고성능 질환 분류:</span> 
      증강 데이터셋(Augmented Dataset)으로 학습된 EfficientNet V1 B0를 통해 <span class="stat-badge">99.09% 정확도</span>를 달성했습니다.
    </li>
    <li>
      <span class="primary-text">📊 충혈 지수 정량화:</span> 
      OpenCV 기반의 홍채 제거 알고리즘 및 <code>Lab</code> 컬러 공간 분석을 통해 충혈도를 수치화합니다.
    </li>
  </ul>
</div>

<hr>

<h2>🛠️ 기술 스택</h2>
<div class="card">
  <table class="tech-table">
    <tr>
      <th>🏗️ 분류</th>
      <th>📦 기술 & 라이브러리</th>
      <th>📝 비고</th>
    </tr>
    <tr>
      <td><span class="accent-text">Hardware</span></td>
      <td>NVIDIA Jetson Orin Nano, Raspberry Pi 5</td>
      <td>Edge AI 추론 장비</td>
    </tr>
    <tr>
      <td><span class="accent-text">Deep Learning</span></td>
      <td>PyTorch, Ultralytics (YOLOv8), EfficientNet, ResNet50</td>
      <td>모델 학습 & 추론</td>
    </tr>
    <tr>
      <td><span class="accent-text">Vision</span></td>
      <td>OpenCV 4.x (Sclera Extraction, Redness Analysis)</td>
      <td>이미지 처리 & 분석</td>
    </tr>
    <tr>
      <td><span class="accent-text">Backend</span></td>
      <td>Flask, FastAPI, SQLite3</td>
      <td>RESTful API & 데이터 관리</td>
    </tr>
    <tr>
      <td><span class="accent-text">Frontend</span></td>
      <td>HTML5, CSS3, JavaScript (Vanilla)</td>
      <td>웹 기반 UI & 실시간 스트림</td>
    </tr>
    <tr>
      <td><span class="accent-text">Security</span></td>
      <td>SHA256 Hashing, HMAC, PEPPER</td>
      <td>사용자 데이터 보호</td>
    </tr>
  </table>
</div>

<hr>

<h2>📊 실험 결과 & 기술 명세</h2>

<div class="card">
  <h3 style="color: var(--primary);">🏆 모델 성능 지표</h3>
  <p>가장 효율적인 모델인 <span class="accent-text">EfficientNet V1 B0</span>를 메인 엔진으로 채택했습니다.</p>
  
  <table class="tech-table">
    <tr>
      <th>⚡ 성능 지표</th>
      <th>📈 수치</th>
      <th>🎯 해석</th>
    </tr>
    <tr>
      <td>Best Validation Accuracy</td>
      <td><span class="stat-badge">99.09%</span></td>
      <td>증강 데이터셋 기준</td>
    </tr>
    <tr>
      <td>Inference Latency</td>
      <td><span class="stat-badge">8.58 ms</span></td>
      <td>Jetson Orin Nano 기준</td>
    </tr>
    <tr>
      <td>Model Size</td>
      <td><span class="stat-badge">~21 MB</span></td>
      <td>메모리 효율적</td>
    </tr>
    <tr>
      <td>Supported Classes</td>
      <td><span class="stat-badge">5종</span></td>
      <td>Normal, Conjunctivitis, Uveitis, Cataract, Eyelid</td>
    </tr>
  </table>
</div>

<hr>

<h2>🔬 핵심 기술 파이프라인</h2>

<div class="card">
  <p style="font-size: 1.05em; line-height: 1.8;">
    본 시스템은 <span class="accent-text">5단계의 정밀 기술 파이프라인</span>을 통해 안구 질환을 진단합니다. 각 단계에서 수학적 엄밀성과 의료용 정확도를 보장합니다.
  </p>
</div>

<hr>

<h3 style="color: var(--primary); margin-top: 30px;">1️⃣ YOLOv8 기반 실시간 정렬 및 오토 셔터 (Alignment)</h3>

<div class="card">
  <p><strong>목적:</strong> 사용자의 안구가 카메라 가이드라인 UI 내에 정확히 위치했을 때 자동으로 캡처를 수행함</p>
  
  <p style="margin-top: 15px;"><strong>📐 중심점 거리 (ΔD):</strong> 검출된 박스의 중심과 가이드라인 중심 사이의 오차 산출</p>
  
  $$\Delta D = \sqrt{(x_d - x_g)^2 + (y_d - y_g)^2} \leq 30\text{ px}$$
  
  <p style="margin-top: 15px;">
    <strong>변수 설명:</strong><br>
    • $(x_d, y_d)$: 검출된 눈 박스의 중심점<br>
    • $(x_g, y_g)$: 가이드라인 중심점<br>
    • 30px: 자동 촬영 허용 오차 범위 (config.py의 AUTO_DIST_THRESHOLD)
  </p>
  
  <p style="margin-top: 20px;"><strong>🔍 크기 일치율 (S_r):</strong> 가이드라인 대비 안구의 배율 확인</p>
  
  $$S_r = \frac{W_{\text{detected}}}{W_{\text{guide}}} \in [0.8, 1.1]$$
  
  <p style="margin-top: 15px;">
    <strong>변수 설명:</strong><br>
    • $W_{\text{detected}}$: 검출된 안구 영역의 너비<br>
    • $W_{\text{guide}}$: 가이드라인의 너비 (표준값: 380px)<br>
    • [0.8, 1.1]: 크기 비율이 80~110% 범위일 때 촬영 가능
  </p>

  <div style="background: linear-gradient(135deg, var(--card-bg) 0%, #252525 100%); padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid var(--primary);">
    <strong>✅ 자동 촬영 조건:</strong><br>
    조건 1과 2를 모두 만족 + AUTO_CAPTURE_HOLD_FRAMES(10프레임) 유지 → 자동 캡처 실행
  </div>
</div>

<hr>

<h3 style="color: var(--primary); margin-top: 30px;">2️⃣ 공막(Sclera) 정밀 추출 및 전처리 (Segmentation)</h3>

<div class="card">
  <p><strong>목적:</strong> 진단에 불필요한 홍채(Iris) 영역을 제거하고 순수한 공막(흰자) 픽셀만 추출함</p>
  
  <p style="margin-top: 15px;"><strong>🔵 허프 원 변환 (Hough Circle):</strong> 홍채 경계면 탐지</p>
  
  $$\left(x - a\right)^2 + \left(y - b\right)^2 = r^2$$
  
  <p style="margin-top: 15px;">
    <strong>변수 설명:</strong><br>
    • $(a, b)$: 홍채 중심점<br>
    • $r$: 홍채 반지름<br>
    • 이 원형 마스크를 이용하여 홍채 영역을 식별
  </p>
  
  <p style="margin-top: 20px;"><strong>🎭 최종 ROI 산출:</strong> 홍채 마스크와 밝기 기반 공막 마스크의 결합</p>
  
  $$\text{ROI}_{\text{final}} = \left(1 - M_{\text{iris}} \times \alpha\right) \cap M_{\text{sclera}}$$
  
  <p style="margin-top: 15px;">
    <strong>변수 설명:</strong><br>
    • $M_{\text{iris}}$: 홍채 영역 마스크 (이진 1/0)<br>
    • $M_{\text{sclera}}$: 밝기 임계값 기반 공막 마스크<br>
    • $\alpha$: 홍채 주변부 노이즈 제거를 위한 가중치 패딩 (보통 0.1~0.2)<br>
    • 최종 결과: 순수한 공막 영역만 추출
  </p>

  <div style="background: linear-gradient(135deg, var(--card-bg) 0%, #252525 100%); padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid var(--secondary);">
    <strong>✅ 처리 단계:</strong><br>
    BGR→HSV→Hough Circle(홍채 탐지)→마스크 생성→공막 영역만 추출→충혈 분석 준비
  </div>
</div>

<hr>

<h3 style="color: var(--primary); margin-top: 30px;">3️⃣ 정량적 충혈도 지수 산출 (Redness Scoring)</h3>

<div class="card">
  <p><strong>목적:</strong> CIE Lab 컬러 공간을 활용하여 주관적인 충혈 상태를 객관적인 수치로 변환함</p>
  
  <p style="margin-top: 15px;"><strong>📐 Redness Index (R_score):</strong> 공막 영역 내 붉은색 채도의 평균값 분석</p>
  
  $$R_{\text{score}} = \left(\frac{1}{N} \sum_{i=1}^{N} a^*_i\right) - 128$$
  
  <p style="margin-top: 15px;">
    <strong>변수 설명:</strong><br>
    • $a^*_i$: ROI 영역 내 i번째 픽셀의 Lab 색공간 a* 값<br>
    • $N$: ROI 영역 내 총 픽셀 수<br>
    • 128: 정상 안구의 기준값 (중립점)<br>
    • <strong>음수</strong>: 초록 우세 (정상, 건강한 상태)<br>
    • <strong>양수</strong>: 빨강 우세 (充血, 염증 상태)
  </p>

  <div style="background: linear-gradient(135deg, var(--card-bg) 0%, #252525 100%); padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #ff6b6b;">
    <strong>📊 충혈도 해석:</strong><br>
    • R_score = -10 ~ 10: 정상 (Normal)<br>
    • R_score = 10 ~ 20: 경미한 충혈 (Mild Redness)<br>
    • R_score = 20 ~ 30: 중등도 충혈 (Moderate Redness)<br>
    • R_score > 30: 심한 충혈 (Severe Redness)
  </div>
</div>

<h3 style="color: var(--primary); margin-top: 20px;">🎨 색공간 변환 파이프라인</h3>

<div class="card">
  $$\text{BGR (OpenCV)} \xrightarrow{\text{cv2.cvtColor}} \text{HSV} \xrightarrow{\text{Iris Removal}} \text{Sclera ROI}$$
  
  $$\text{Sclera ROI} \xrightarrow{\text{cv2.cvtColor}} \text{Lab} \xrightarrow{\text{Extract } a^*} \text{R_{\text{score}}}$$
</div>

<hr>

<h3 style="color: var(--primary); margin-top: 30px;">4️⃣ 딥러닝 기반 다중 질환 분류 (Classification)</h3>

<div class="card">
  <p><strong>목적:</strong> 5종 안구 질환(정상, 결막염, 백내장, 포도막염, 다래끼)을 판별함</p>
  
  <p style="margin-top: 15px;"><strong>🧠 EfficientNet V1 B0 모델:</strong></p>
  
  $$P(c_i | I) = \text{softmax}\left(\text{EfficientNet}_{\theta}(I)\right)_i$$
  
  <p style="margin-top: 15px;">
    <strong>변수 설명:</strong><br>
    • $I$: 입력 안구 이미지 (224×224 RGB)<br>
    • $\theta$: 학습된 모델 가중치<br>
    • $c_i$: i번째 질환 클래스 (0~4)<br>
    • $P(c_i|I)$: 이미지 I가 질환 $c_i$일 확률
  </p>

  <p style="margin-top: 20px;"><strong>🎯 5종 분류:</strong></p>
  <table class="tech-table">
    <tr>
      <th>클래스</th>
      <th>질환</th>
      <th>특징</th>
    </tr>
    <tr>
      <td>Class 0</td>
      <td><span style="color: #4caf50;">정상 (Normal)</span></td>
      <td>건강한 안구</td>
    </tr>
    <tr>
      <td>Class 1</td>
      <td><span style="color: #ff6b6b;">결막염 (Conjunctivitis)</span></td>
      <td>결막 염증, 충혈</td>
    </tr>
    <tr>
      <td>Class 2</td>
      <td><span style="color: #ffc107;">포도막염 (Uveitis)</span></td>
      <td>포도막 염증</td>
    </tr>
    <tr>
      <td>Class 3</td>
      <td><span style="color: #9c27b0;">백내장 (Cataract)</span></td>
      <td>수정체 혼탁</td>
    </tr>
    <tr>
      <td>Class 4</td>
      <td><span style="color: #ff9500;">다래끼 (Eyelid)</span></td>
      <td>눈꺼풀 감염</td>
    </tr>
  </table>

  <div style="background: linear-gradient(135deg, var(--card-bg) 0%, #252525 100%); padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid var(--accent);">
    <strong>📊 성능 지표:</strong><br>
    • Target Accuracy: <span class="stat-badge">99.09%</span><br>
    • Inference Time: <span class="stat-badge">8.58ms</span><br>
    • Model Size: <span class="stat-badge">~21MB</span><br>
    • 플랫폼: Jetson Orin Nano (최적화 완료)
  </div>
</div>

<hr>

<h3 style="color: var(--primary); margin-top: 30px;">5️⃣ 보안 기반 고유 식별자 생성 (Security Hashing)</h3>

<div class="card">
  <p><strong>목적:</strong> 개인정보인 전화번호를 직접 저장하지 않고 보안 해시값으로 관리함 (프라이버시 보호)</p>
  
  <p style="margin-top: 15px;"><strong>🔐 Salted SHA-256 해싱:</strong> 전화번호와 서버 비밀키의 결합 해싱</p>
  
  $$ID_{\text{user}} = \text{SHA256}(P \| K)$$
  
  <p style="margin-top: 15px;">
    <strong>변수 설명:</strong><br>
    • $P$: 정규화된 전화번호 (010-1234-5678 → 01012345678)<br>
    • $K$: 서버 비밀키 (HASH_PEPPER 환경변수)<br>
    • $\|$: 문자열 연결 연산자<br>
    • $ID_{\text{user}}$: 256비트 고유 식별자 (해시값)
  </p>

  <p style="margin-top: 20px;"><strong>🛡️ 보안 특징:</strong></p>
  <ul style="line-height: 1.8;">
    <li><strong>단방향 암호화:</strong> SHA-256은 복호화 불가능 (일방성 함수)</li>
    <li><strong>Salt 추가:</strong> HASH_PEPPER 키를 섞어 사전 공격 방어</li>
    <li><strong>원전 번호 미저장:</strong> 데이터베이스에는 해시값만 저장</li>
    <li><strong>HMAC 검증:</strong> 데이터 무결성 확인용 메시지 인증 코드</li>
  </ul>

  <div style="background: linear-gradient(135deg, var(--card-bg) 0%, #252525 100%); padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #4caf50;">
    <strong>✅ 구현 위치:</strong><br>
    파일: <code>utils/security_utils.py</code><br>
    함수: <code>phone_hash_id(phone: str) → str</code><br>
    사용: <code>database/db.py</code>의 <code>upsert_user_by_phone()</code>
  </div>
</div>

<hr>

<h2>📂 프로젝트 구조</h2>

<div class="card">
  <pre style="background-color: var(--bg-color); padding: 15px; border-radius: 4px; overflow-x: auto;">
<span style="color: var(--accent);">eye_project/</span>
├── <span style="color: var(--primary);">eye_server.py</span>              # 전체 시스템 제어 및 UI 연동
├── <span style="color: var(--primary);">eye_server_test.py</span>        # 테스트 모드 (DB/보안 통합)
├── <span style="color: var(--primary);">config.py</span>                 # 모델 경로, 임계값, 자동 촬영 설정
├── <span style="color: var(--primary);">model_loader.py</span>           # 모델 초기화 및 캐싱
│
├── <span style="color: var(--secondary);">models/</span>
│   ├── Augmented_EffNet_V1_B0_best.pth    # EfficientNet 분류 모델 (99.09%)
│   └── yolov8n_eye.pt                    # YOLOv8n 눈 탐지 모델
│
├── <span style="color: var(--secondary);">modules/</span>
│   ├── detector.py                       # YOLO 기반 눈 탐지
│   ├── classifier.py                     # EfficientNet 기반 질환 분류
│   ├── analyzer.py                       # 충혈도 산출 및 픽셀 분석
│   └── __init__.py
│
├── <span style="color: var(--secondary);">database/</span>
│   ├── db.py                             # SQLite 진단 데이터 관리
│   └── schema.sql                        # DB 스키마 (users, diagnosis_sessions 등)
│
├── <span style="color: var(--secondary);">utils/</span>
│   ├── image_proc.py                     # 이미지 전처리 & 호남색 제거
│   ├── logger.py                         # 진단 결과 기록
│   └── security_utils.py                 # SHA256 해싱, HMAC 검증
│
├── <span style="color: var(--secondary);">web/</span>
│   ├── <span style="color: var(--secondary);">static/</span>
│   │   ├── <span style="color: var(--secondary);">captures/</span>           # 촬영 이미지 저장
│   │   ├── <span style="color: var(--secondary);">css/</span>
│   │   │   └── style.css
│   │   ├── <span style="color: var(--secondary);">js/</span>
│   │   │   └── main.js                   # 실시간 스트림 & 진단 로직
│   │   └── <span style="color: var(--secondary);">images/</span>
│   │
│   └── <span style="color: var(--secondary);">templates/</span>
│       ├── index.html                    # 메인 페이지
│       ├── capture.html                  # 촬영 화면 (가이드라인 UI)
│       ├── result.html                   # 진단 결과 페이지
│       ├── survey.html                   # 사용자 설문조사
│       ├── login.html                    # 로그인 페이지
│       └── layout.html                   # 기본 레이아웃
│
├── <span style="color: var(--primary);">README.md</span>                 # 이 파일
├── <span style="color: var(--primary);">eye_diagnosis.db</span>         # SQLite 데이터베이스 (런타임)
└── <span style="color: var(--primary);">requirements.txt</span>         # Python 패키지 의존성
  </pre>
</div>

<hr>

<h2>👥 팀 구성 (Team Papyrus)</h2>

<div class="card">
  <p style="font-size: 1.05em; line-height: 1.8; margin-bottom: 20px;">
    <span class="accent-text">Team Papyrus</span>는 AI, 하드웨어, 의료 데이터 분야의 전문가 5명으로 구성되어 있으며, 각자의 전문성을 바탕으로 <span class="primary-text">의료용 임베디드 AI 시스템</span>을 구축했습니다.
  </p>

  <table class="tech-table">
    <tr>
      <th>👤 팀원</th>
      <th>🎯 담당 영역</th>
      <th>💼 핵심 역할</th>
    </tr>
    <tr>
      <td><span class="accent-text">박하준</span><br><small>PM</small></td>
      <td><b>AI 모델링 및<br>최적화</b></td>
      <td>
        • EfficientNet 정확도(99.09%) 고도화<br>
        • YOLOv8 오토 셔터 로직 설계<br>
        • Jetson TensorRT 엔진 최적화
      </td>
    </tr>
    <tr>
      <td><span class="accent-text">우세준</span></td>
      <td><b>시스템 통합 및<br>데이터 검증</b></td>
      <td>
        • 개별 모듈(YOLO, EffNet, DB) 간<br>&nbsp;&nbsp;파이프라인 통합<br>
        • 데이터셋 무결성 검증<br>
        • 모델 신뢰도 테스트
      </td>
    </tr>
    <tr>
      <td><span class="accent-text">김영록</span></td>
      <td><b>하드웨어 설계 및<br>광학 시스템</b></td>
      <td>
        • 키오스크 외형 설계<br>
        • 안구 촬영 최적 조명(LED) 제어<br>
        • 젯슨-카메라 인터페이스<br>&nbsp;&nbsp;하드웨어 구성
      </td>
    </tr>
    <tr>
      <td><span class="accent-text">김세은</span></td>
      <td><b>GUI 개발 및<br>데이터 시각화</b></td>
      <td>
        • 사용자 가이드라인<br>&nbsp;&nbsp;실시간 인터페이스 개발<br>
        • 진단 결과 차트 및<br>&nbsp;&nbsp;대시보드 시각화 구현
      </td>
    </tr>
    <tr>
      <td><span class="accent-text">나은율</span></td>
      <td><b>데이터베이스 및<br>의료 연계</b></td>
      <td>
        • 보안 해싱 기반 DB 설계<br>
        • 의료 표준(FHIR) 연계<br>&nbsp;&nbsp;데이터 관리<br>
        • 리포트 자동 생성 시스템<br>&nbsp;&nbsp;구축
      </td>
    </tr>
  </table>

  <hr style="margin: 25px 0; border-color: var(--border-color);">

  <h3 style="color: var(--primary); margin-top: 20px;">🏆 팀의 강점</h3>
  
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
    <div style="background: var(--bg-color); padding: 12px; border-radius: 8px; border-left: 4px solid var(--accent);">
      <p style="margin: 0; font-weight: bold; color: var(--accent);">🔬 기술 다양성</p>
      <p style="margin: 5px 0 0 0; font-size: 0.9em;">AI, 임베디드, 하드웨어, 의료 정보를 아우르는 통합 역량</p>
    </div>
    <div style="background: var(--bg-color); padding: 12px; border-radius: 8px; border-left: 4px solid var(--secondary);">
      <p style="margin: 0; font-weight: bold; color: var(--secondary);">🏥 의료 정확도</p>
      <p style="margin: 5px 0 0 0; font-size: 0.9em;">의료 데이터 표준(FHIR) 준수 및 보안 규정 준수</p>
    </div>
    <div style="background: var(--bg-color); padding: 12px; border-radius: 8px; border-left: 4px solid #ffc107;">
      <p style="margin: 0; font-weight: bold; color: #ffc107;">⚡ 성능 최적화</p>
      <p style="margin: 5px 0 0 0; font-size: 0.9em;">엣지 디바이스에서 의료용 정확도 유지</p>
    </div>
    <div style="background: var(--bg-color); padding: 12px; border-radius: 8px; border-left: 4px solid #4caf50;">
      <p style="margin: 0; font-weight: bold; color: #4caf50;">🔒 보안 & 프라이버시</p>
      <p style="margin: 5px 0 0 0; font-size: 0.9em;">온디바이스 처리 및 암호화된 사용자 데이터 관리</p>
    </div>
  </div>
</div>

<hr>

<h2>�️ 웹 인터페이스 구조</h2>

<div class="card">
  <p>사용자 친화적인 웹 기반 UI는 두 개의 주요 섹션으로 구성됩니다:</p>
  
  <h3 style="color: var(--primary);">📱 Result Dashboard (web/templates/result.html)</h3>
  <table class="tech-table">
    <tr>
      <th>📍 섹션</th>
      <th>⚙️ 기능</th>
      <th>📐 크기</th>
    </tr>
    <tr>
      <td><b>좌패널 (Left Panel)</b></td>
      <td>
        • 실시간 비디오 스트림 (1280x720)<br>
        • 동공/홍채 가이드라인 오버레이<br>
        • 안구 건강 리포트 발행 버튼<br>
        • 주변 안과 찾기 (Naver Map 연동)
      </td>
      <td>58% (flex: 1.4)</td>
    </tr>
    <tr>
      <td><b>우패널 (Right Panel)</b></td>
      <td>
        • 🔍 자동 측정 데이터 (홍채 직경, 색상)<br>
        • ✍️ 시력 입력 필드 (L/R)<br>
        • 🤖 질환 분석 결과 (5종)<br>
        • 💡 AI 맞춤 권고사항<br>
        • 진단 업데이트 버튼
      </td>
      <td>42% (flex: 1)</td>
    </tr>
  </table>

  <h3 style="color: var(--secondary);">🎨 디자인 시스템</h3>
  <pre style="background-color: var(--bg-color); padding: 15px; border-radius: 4px; border-left: 4px solid var(--primary);">
<span style="color: var(--primary);">:root CSS Variables</span>
--primary: #007AFF          (파란색)
--accent: #28cd41           (초록색 - 정상)
--danger: #ff3b30           (빨강색 - 위험)
--warning: #ff9500          (주황색 - 주의)
--bg: #f5f5f7               (밝은 배경)
--card: #ffffff             (카드 배경)
--text: #1d1d1f             (텍스트 색)
  </pre>

  <h3 style="color: var(--secondary);">📦 HTML 구조</h3>
  <pre style="background-color: var(--bg-color); padding: 15px; border-radius: 4px; font-size: 0.85em;">
&lt;body&gt;
  &lt;div class="left-panel"&gt;
    &lt;!-- 카메라 스트림 --&gt;
    &lt;video id="video-feed" autoplay&gt;&lt;/video&gt;
    &lt;!-- 가이드라인 --&gt;
    &lt;div class="eye-shape-guide"&gt;
      &lt;div class="iris-guide"&gt;
        &lt;div class="pupil-guide"&gt;&lt;/div&gt;
      &lt;/div&gt;
    &lt;/div&gt;
    &lt;!-- 액션 버튼 --&gt;
    &lt;div class="camera-bottom-btns"&gt;...&lt;/div&gt;
  &lt;/div&gt;
  
  &lt;div class="right-panel"&gt;
    &lt;!-- 자동 측정 데이터 --&gt;
    &lt;div class="card"&gt;홍채 직경: 11.8mm ...&lt;/div&gt;
    &lt;!-- 시력 입력 --&gt;
    &lt;div class="card"&gt;좌(L), 우(R) 입력 ...&lt;/div&gt;
    &lt;!-- 질환 분석 --&gt;
    &lt;div class="card"&gt;
      • 결막염 및 충혈: 의심(높음)
      • 다래끼 유무: 미검출
      • 백내장: 관찰 필요
      • 익상편 등: 정상
    &lt;/div&gt;
    &lt;!-- AI 권고 --&gt;
    &lt;div class="tip-box"&gt;경고/조치사항 ...&lt;/div&gt;
  &lt;/div&gt;
&lt;/body&gt;
  </pre>
</div>

<hr>

<div class="card">
  <h3 style="color: var(--primary);">필수 환경</h3>
  <pre style="background-color: var(--bg-color); padding: 10px; border-radius: 4px;">
Python 3.8+
PyTorch 1.10+
CUDA 11.0+ (Jetson Orin Nano)
OpenCV 4.4+
Flask 2.0+
  </pre>

  <h3 style="color: var(--primary); margin-top: 20px;">설치 및 실행</h3>
  <pre style="background-color: var(--bg-color); padding: 10px; border-radius: 4px;">
# 1. 저장소 클론
git clone https://github.com/papyrus-project/eye_project.git
cd eye_project

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 설정 (보안)
export HASH_PEPPER='your_secret_pepper_here'

# 4. 메인 서버 실행 (Jetson)
python3 eye_server.py

# 테스트 모드 실행 (개발 환경)
python3 eye_server_test.py

# 5. 웹 브라우저에서 접속
# http://127.0.0.1:5000
  </pre>
</div>

<hr>

<h2>📡 실시간 진단 대시보드 (Real-time Dashboard)</h2>

<div class="dashboard-container">
  <h3 style="color: var(--accent); margin-top: 0;">🎯 주요 기능</h3>
  
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
    
    <div style="background: var(--card-bg); padding: 15px; border-radius: 8px; border-left: 4px solid var(--primary);">
      <h4 style="color: var(--primary); margin-top: 0;">� 카메라 스트림 & 가이드라인</h4>
      <p style="font-size: 0.9em; line-height: 1.6;">
        ✓ <b>실시간 영상 처리</b><br>
        &nbsp;&nbsp; 1280x720 해상도, 30 FPS MJPEG 스트림<br><br>
        ✓ <b>동공/홍채 가이드라인</b><br>
        &nbsp;&nbsp; "동공과 홍채를 가이드 원에 맞춰주세요"<br>
        &nbsp;&nbsp; 타원형 범위 표시 (380x180px)<br><br>
        ✓ <b>액션 버튼</b><br>
        &nbsp;&nbsp; 📋 안구 건강 리포트 발행<br>
        &nbsp;&nbsp; 📍 주변 안과 찾기 (네이버 지도)
      </p>
    </div>
    
    <div style="background: var(--card-bg); padding: 15px; border-radius: 8px; border-left: 4px solid var(--secondary);">
      <h4 style="color: var(--secondary); margin-top: 0;">📊 우측 진단 패널</h4>
      <p style="font-size: 0.9em; line-height: 1.6;">
        <b>🔍 자동 측정 데이터</b><br>
        &nbsp;&nbsp; • 홍채 직경: 11.8mm<br>
        &nbsp;&nbsp; • 홍채 색상: Brown<br><br>
        <b>✍️ 시력 입력</b><br>
        &nbsp;&nbsp; • 좌안(L): 입력 필드<br>
        &nbsp;&nbsp; • 우안(R): 입력 필드
      </p>
    </div>
    
    <div style="background: var(--card-bg); padding: 15px; border-radius: 8px; border-left: 4px solid #ff6b6b;">
      <h4 style="color: #ff6b6b; margin-top: 0;">🤖 질환 분석 결과</h4>
      <p style="font-size: 0.9em; line-height: 1.6;">
        <b>5종 질환 분류 (EfficientNet 99.09%)</b><br><br>
        📍 <b style="color: #ff3b30;">결막염 및 충혈</b><br>
        &nbsp;&nbsp; 상태: 의심(높음) ⚠️<br><br>
        📍 <b style="color: #4caf50;">다래끼 유무</b><br>
        &nbsp;&nbsp; 상태: 미검출 ✓<br><br>
        📍 <b style="color: #ff9500;">백내장 이상 징후</b><br>
        &nbsp;&nbsp; 상태: 관찰 필요 ⚠️<br><br>
        📍 <b style="color: #4caf50;">익상편 등 기타</b><br>
        &nbsp;&nbsp; 상태: 정상 ✓
      </p>
    </div>
    
    <div style="background: var(--card-bg); padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
      <h4 style="color: #ffc107; margin-top: 0;">💡 AI 맞춤형 권고 사항</h4>
      <p style="font-size: 0.9em; line-height: 1.6;">
        <b>1. 충혈 관리</b><br>
        결막염 증상이 의심됩니다. 전염성이 있을 수 있으니 눈을 비비지 말고 수건을 따로 사용하세요.<br><br>
        <b>2. 백내장 주의</b><br>
        수정체 영역에 미세한 혼탁이 관찰됩니다. 눈부심이나 시력 저하 증상이 있으면 정밀 검사 권장합니다.<br><br>
        <b>3. 즉시 조치</b><br>
        주변 안과를 방문해 진료를 받으시기 바랍니다.
      </p>
    </div>
  </div>

  <hr style="border-color: var(--border-color); margin-top: 20px;">

  <h3 style="color: var(--accent); margin-bottom: 15px;">📈 정밀 분석 메트릭</h3>
  
  <table class="tech-table" style="margin-bottom: 0;">
    <tr>
      <th>📊 분석 항목</th>
      <th>🔬 기술/방법</th>
      <th>📍 상태</th>
      <th>🎯 정상 범위</th>
    </tr>
    <tr>
      <td>홍채 직경 측정</td>
      <td>YOLO 기반 경계 탐지</td>
      <td style="color: #4caf50;">✓ 11.8mm</td>
      <td>10-12mm</td>
    </tr>
    <tr>
      <td>충혈도 지수</td>
      <td>Lab 색공간 분석</td>
      <td style="color: #ff3b30;">⚠️ +25.3</td>
      <td>-10 ~ +10</td>
    </tr>
    <tr>
      <td>혈관 밀도</td>
      <td>픽셀 기반 Redness 계산</td>
      <td style="color: #ff9500;">⚠️ 높음</td>
      <td>&lt; 20%</td>
    </tr>
    <tr>
      <td>ROI 명확도</td>
      <td>콘트라스트 및 선명도</td>
      <td style="color: #4caf50;">✓ 우수</td>
      <td>SSIM &gt; 0.8</td>
    </tr>
    <tr>
      <td>EfficientNet 신뢰도</td>
      <td>Softmax 확률 (Top-5)</td>
      <td style="color: #4caf50;">✓ 99.09%</td>
      <td>&gt; 90%</td>
    </tr>
  </table>

  <hr style="border-color: var(--border-color); margin-top: 20px;">

  <h3 style="color: var(--accent); margin-bottom: 15px;">⚡ 시스템 성능 모니터링</h3>
  
  <table class="tech-table" style="margin-bottom: 0;">
    <tr>
      <th>🖥️ 리소스</th>
      <th>📊 현재값</th>
      <th>🎯 목표</th>
      <th>📍 상태</th>
    </tr>
    <tr>
      <td>GPU 메모리 (Jetson Orin Nano)</td>
      <td>3.2GB / 8GB</td>
      <td>&lt; 60%</td>
      <td><span style="color: #4caf50;">✓ 정상</span></td>
    </tr>
    <tr>
      <td>YOLOv8 추론 시간</td>
      <td>5.2ms</td>
      <td>&lt; 10ms</td>
      <td><span style="color: #4caf50;">✓ 정상</span></td>
    </tr>
    <tr>
      <td>EfficientNet 추론 시간</td>
      <td>8.58ms</td>
      <td>&lt; 15ms</td>
      <td><span style="color: #4caf50;">✓ 정상</span></td>
    </tr>
    <tr>
      <td>이미지 전처리</td>
      <td>3.2ms</td>
      <td>&lt; 5ms</td>
      <td><span style="color: #4caf50;">✓ 정상</span></td>
    </tr>
    <tr>
      <td>전체 파이프라인 E2E</td>
      <td>17.0ms</td>
      <td>&lt; 30ms (33.3 FPS)</td>
      <td><span style="color: #4caf50;">✓ 정상</span></td>
    </tr>
    <tr>
      <td>카메라 프레임 레이트</td>
      <td>30 FPS</td>
      <td>≥ 25 FPS</td>
      <td><span style="color: #4caf50;">✓ 정상</span></td>
    </tr>
    <tr>
      <td>데이터베이스 쿼리</td>
      <td>&lt; 50ms</td>
      <td>&lt; 100ms</td>
      <td><span style="color: #4caf50;">✓ 정상</span></td>
    </tr>
  </table>
</div>

<hr>

<h2>🔐 보안 & 프라이버시</h2>

<div class="card">
  <ul style="line-height: 2;">
    <li>
      <span class="primary-text">🔒 데이터 암호화:</span>
      사용자 전화번호는 SHA256 + PEPPER로 해싱되어 저장되며, 원본은 데이터베이스에 저장되지 않습니다.
    </li>
    <li>
      <span class="primary-text">🏥 온디바이스 처리:</span>
      모든 진단 데이터는 로컬 장치(Jetson)에서만 처리되어 클라우드으로 전송되지 않습니다.
    </li>
    <li>
      <span class="primary-text">📋 FHIR 표준:</span>
      진단 결과는 의료 정보 교환 표준(FHIR)을 준수하여 호환성과 신뢰성을 보장합니다.
    </li>
    <li>
      <span class="primary-text">🔑 환경변수 관리:</span>
      <code>HASH_PEPPER</code> 같은 민감한 정보는 환경변수로 주입되어 코드에 저장되지 않습니다.
    </li>
  </ul>
</div>

<hr>

<h2>📚 참고 자료 및 라이선스</h2>

<div class="card">
  <p>
    <strong>논문:</strong> EfficientNet, YOLOv8 학술 논문<br>
    <strong>데이터셋:</strong> 직접 수집 및 증강 (1,500+ 이미지)<br>
    <strong>라이선스:</strong> MIT License<br>
    <strong>저장소:</strong> <a href="https://github.com/papyrus-project/eye_project" style="color: var(--accent);">github.com/papyrus-project/eye_project</a>
  </p>
</div>

<hr>

<div align="center" style="margin-top: 40px; padding: 20px; background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); border-radius: 8px;">
  <h3 style="color: white; margin: 0;">🌟 Team Papyrus</h3>
  <p style="color: white; margin: 5px 0;">AI-Powered Eye Medical Diagnosis System</p>
  <p style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 0;">© 2026 All Rights Reserved</p>
</div>