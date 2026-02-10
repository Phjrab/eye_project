"""
[설정] 눈병 진단 시스템 중앙 관리
- IP주소, 모델 경로, 임계값, 클래스명
"""

import os

# ========================================
# [1] 기본 경로 설정
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ========================================
# [2] 모델 경로
# ========================================
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'yolov8n_eye.pt')  # 추후 추가 예정
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'Augmented_EffNet_V1_B0_best.pth')

# ========================================
# [3] 서버 설정
# ========================================
SERVER_IP = '0.0.0.0'
SERVER_PORT = 5000
DEBUG_MODE = True

# ========================================
# [4] YOLO 검출 임계값
# ========================================
YOLO_CONF_THRESHOLD = 0.5        # 신뢰도 임계값
YOLO_IOU_THRESHOLD = 0.45        # IoU 임계값
YOLO_INPUT_SIZE = 640            # 입력 이미지 크기

# ========================================
# [5] 분류 모델 설정
# ========================================
CLASSIFIER_INPUT_SIZE = (224, 224)       # EfficientNet 입력 크기
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7    # 분류 신뢰도 임계값

# ========================================
# [6] 질환 분류 클래스 (5개)
# ========================================
DISEASE_CLASSES = {
    0: '일반 (Normal)',
    1: '결막염 (Conjunctivitis)',
    2: '포도막염 (Uveitis)',
    3: '백내장 (Cataract)',
    4: '다래끼 (Eyelid)'
}

# ========================================
# [7] 홍채 제거 설정
# ========================================
IRIS_REMOVAL_ENABLED = True      # 홍채 제거 활성화
IRIS_THRESHOLD = 0.3             # 홍채 임계값

# ========================================
# [8] 로깅 설정
# ========================================
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FORMAT = 'csv'               # 'csv' 또는 'db'
os.makedirs(LOG_DIR, exist_ok=True)

# ========================================
# [9] 이미지 처리 설정
# ========================================
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, 'web', 'static', 'captures')
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
