# eye_project

<div align="center">
  <h1>👁️ AI 기반 안구 질환 통합 진단 시스템</h1>
  <p><strong>Papyrus Project: Jetson Orin Nano & Raspberry Pi 5 통합 솔루션</strong></p>
</div>

<hr>

<h2>🚀 프로젝트 개요</h2>
<p>본 프로젝트는 임베디드 환경(Jetson Orin Nano)에서 딥러닝 파이프라인을 구축하여 <b>5종의 주요 안구 질환을 정밀 진단</b>하고, <b>충혈도를 정량적으로 산출</b>하는 하드웨어-소프트웨어 통합 시스템입니다.</p>

<h2>🌟 핵심 기능</h2>
<ul>
  <li><b>스냅샷 기반 추론 (Snapshot-based Inference):</b> 가이드라인 UI와 연동하여 버튼 클릭 시 단회 추론을 수행함으로써 젯슨의 연산 자원을 최적화합니다.</li>
  <li><b>정밀 ROI 추출:</b> YOLOv8n 모델을 통해 전체 화면에서 안구 영역만 0.8 이상의 신뢰도로 정확히 탐지 및 크롭합니다.</li>
  <li><b>고성능 질환 분류:</b> 증강 데이터셋(Augmented Dataset)으로 학습된 EfficientNet V1 B0를 통해 <b>99.09%의 검증 정확도</b>를 달성했습니다.</li>
  <li><b>충혈 지수 정량화:</b> OpenCV 기반의 홍채 제거 알고리즘 및 $Lab$ 컬러 공간 분석을 통해 충혈도를 수치화합니다.</li>
</ul>

<hr>

<h2>🛠️ 기술 스택</h2>
<table border="1">
  <tr>
    <th>분류</th>
    <th>내용</th>
  </tr>
  <tr>
    <td><b>Hardware</b></td>
    <td>NVIDIA Jetson Orin Nano, Raspberry Pi 5</td>
  </tr>
  <tr>
    <td><b>Deep Learning</b></td>
    <td>PyTorch, Ultralytics (YOLOv8), EfficientNet, ResNet50</td>
  </tr>
  <tr>
    <td><b>Vision Library</b></td>
    <td>OpenCV (Sclera Extraction, Redness Analysis)</td>
  </tr>
  <tr>
    <td><b>Deployment</b></td>
    <td>Flask/FastAPI 기반 웹 서버 및 가이드라인 UI</td>
  </tr>
</table>

<hr>

<h2>📊 실험 결과 (Augmented Dataset 기준)</h2>
<p>가장 효율적인 모델인 <b>EfficientNet V1 B0</b>를 메인 엔진으로 채택했습니다.</p>
<ul>
  <li><b>Best Val Acc:</b> 99.09%</li>
  <li><b>Inference Latency:</b> 8.58ms (Jetson Orin Nano 기준)</li>
  <li><b>충혈도 산출 공식:</b> $$Redness\_Score = \frac{\sum_{i=1}^{n} a^*_i}{n} - 128$$</li>
</ul>

<hr>

<h2>📂 프로젝트 구조</h2>
<pre>
<code>
eye_project/
├── eye_server.py          # 전체 시스템 제어 및 UI 연동
├── config.py              # 모델 경로 및 임계값 설정
├── models/                # YOLOv8n, EfficientNet 가중치 파일
├── modules/               # Detector, Classifier, Analyzer 핵심 모듈
├── web/                   # HTML/CSS 기반 사용자 인터페이스
└── utils/                 # 전처리 및 진단 결과 기록(Logger)
</code>
</pre>

<hr>

<h2>👥 팀 구성 (Team Papyrus)</h2>
<ul>
  <li><b>박하준 (전자공학, PM):</b> 시스템 아키텍처 설계 및 충혈도 알고리즘 구현</li>
  <li><b>세준:</b> 데이터셋 구축 및 YOLOv8 눈 탐지 모델 학습</li>
  <li><b>정훈:</b> 추론 엔진 모듈화 및 젯슨/라즈베리파이 환경 최적화</li>
  <li><b>은율:</b> 성능 벤치마크 분석 및 데이터 기반 기술 보고서 작성</li>
</ul>