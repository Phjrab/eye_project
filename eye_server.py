"""
Eye Disease Detection Server
웹 인터페이스와 AI 파이프라인의 오케스트레이터
"""

from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
import cv2
import numpy as np
import torch
import time
import os
import json
import base64
import threading
import atexit
import gc
import sqlite3
import re
from datetime import datetime
from PIL import Image

import config
from model_loader import initialize_models, get_models
from utils.image_proc import resize_image, enhance_contrast

# ========================================
# [1] Flask 앱 설정
# ========================================
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.getenv('EYE_APP_SECRET_KEY', 'eye-project-admin-secret')

# 전역 변수
current_frame = None  # 실시간 카메라 프레임
model_manager = None  # 싱글톤 모델 매니저
models_initialized = False  # 모델 초기화 완료 여부
camera_thread = None  # 카메라 스레드
camera_running = False  # 카메라 스레드 실행 상태
debug_boxes_cache = []
debug_frame_counter = 0

HISTORY_DB_PATH = os.path.join(config.BASE_DIR, 'database', 'history.db')

ADMIN_EDITABLE_CONFIG_KEYS = {
    'SERVER_IP': str,
    'SERVER_PORT': int,
    'DEBUG_MODE': bool,
    'CAMERA_DEVICE_INDEX': int,
    'YOLO_CONF_THRESHOLD': float,
    'YOLO_IOU_THRESHOLD': float,
    'YOLO_STATUS_CONF_THRESHOLD': float,
    'YOLO_INPUT_SIZE': int,
    'CLASSIFIER_CONFIDENCE_THRESHOLD': float,
    'IRIS_REMOVAL_ENABLED': bool,
    'IRIS_THRESHOLD': float,
    'AUTO_DIST_THRESHOLD': int,
    'AUTO_SCALE_MIN': float,
    'AUTO_SCALE_MAX': float,
    'AUTO_CAPTURE_HOLD_FRAMES': int
}


def is_admin_session():
    return bool(session.get('is_admin', False))


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if text in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise ValueError('bool 값이 아닙니다')


def cast_config_value(key, value):
    target_type = ADMIN_EDITABLE_CONFIG_KEYS.get(key)
    if target_type is None:
        raise ValueError(f'수정 불가 항목: {key}')

    if target_type is bool:
        return normalize_bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return str(value)


def format_python_literal(value):
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


def update_config_file_values(updates):
    config_file_path = os.path.join(config.BASE_DIR, 'config.py')
    with open(config_file_path, 'r', encoding='utf-8') as file:
        source = file.read()

    updated_source = source
    for key, value in updates.items():
        pattern = rf'^{key}\s*=\s*.*$'
        replacement = f"{key} = {format_python_literal(value)}"
        new_source, count = re.subn(pattern, replacement, updated_source, flags=re.MULTILINE)
        if count == 0:
            raise ValueError(f'config.py에서 항목을 찾지 못했습니다: {key}')
        updated_source = new_source

    with open(config_file_path, 'w', encoding='utf-8') as file:
        file.write(updated_source)


def get_admin_config_snapshot():
    snapshot = {}
    for key in ADMIN_EDITABLE_CONFIG_KEYS:
        snapshot[key] = getattr(config, key, None)
    return snapshot

# ========================================
# [2] 촬영 상태 관리 (왼쪽/오른쪽 눈 시퀀스)
# ========================================
class CaptureState:
    """촬영 상태 관리 클래스"""
    
    # 촬영 상태 조건 체크 프레임 카운터
    ACCEPTED_FRAMES = 0
    
    # 현재 촬영 상태
    current_eye = "LEFT_EYE"  # "LEFT_EYE" 또는 "RIGHT_EYE"
    
    # 촬영 완료된 눈
    captured_eyes = {
        "LEFT_EYE": False,
        "RIGHT_EYE": False
    }
    
    # 자동 촬영 준비 상태
    auto_capture_ready = False
    
    @classmethod
    def reset(cls):
        """촬영 시퀀스 초기화"""
        cls.ACCEPTED_FRAMES = 0
        cls.current_eye = "LEFT_EYE"
        cls.captured_eyes = {"LEFT_EYE": False, "RIGHT_EYE": False}
        cls.auto_capture_ready = False
    
    @classmethod
    def move_to_next_eye(cls):
        """다음 눈으로 이동"""
        if cls.current_eye == "LEFT_EYE":
            cls.current_eye = "RIGHT_EYE"
        else:
            cls.current_eye = "LEFT_EYE"
        cls.ACCEPTED_FRAMES = 0
        cls.auto_capture_ready = False
    
    @classmethod
    def mark_captured(cls, eye_type):
        """촬영 완료 표시"""
        cls.captured_eyes[eye_type] = True


def init_history_db():
    """사용자별 진단 히스토리 DB 초기화"""
    os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)

    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnosis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_url TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_user_time
            ON diagnosis_history(user_id, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS survey_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                survey_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_survey_user_time
            ON survey_history(user_id, created_at DESC)
            """
        )
        conn.commit()
    finally:
        conn.close()


def normalize_user_id(user_id):
    """파일 시스템/DB 저장용 사용자 식별자 정규화"""
    if not user_id:
        return 'anonymous'

    value = str(user_id).strip()
    if not value:
        return 'anonymous'

    normalized = re.sub(r'[^0-9A-Za-z가-힣._-]+', '_', value)
    return normalized[:80] if normalized else 'anonymous'


def save_history_record(user_id, source_image_bgr, analysis):
    """분석 결과와 원본 이미지를 사용자별로 로컬 저장 + DB 기록"""
    safe_user_id = normalize_user_id(user_id)

    user_dir = os.path.join(config.IMAGE_SAVE_DIR, 'users', safe_user_id)
    os.makedirs(user_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'diagnosis_{timestamp}.jpg'
    image_path = os.path.join(user_dir, filename)

    write_ok = cv2.imwrite(image_path, source_image_bgr)
    if not write_ok:
        raise RuntimeError('로컬 이미지 저장 실패')

    image_url = f"/static/captures/users/{safe_user_id}/{filename}"

    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO diagnosis_history (user_id, image_path, image_url, analysis_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                safe_user_id,
                image_path,
                image_url,
                json.dumps(analysis, ensure_ascii=False)
            )
        )
        conn.commit()
        return int(cur.lastrowid), image_url
    finally:
        conn.close()


def list_history_records(user_id, limit=10):
    """사용자별 최근 히스토리 조회"""
    safe_user_id = normalize_user_id(user_id)
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, image_path, image_url, analysis_json, created_at
            FROM diagnosis_history
            WHERE user_id=?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (safe_user_id, int(limit))
        ).fetchall()

        history = []
        for row in rows:
            analysis = {}
            try:
                analysis = json.loads(row['analysis_json']) if row['analysis_json'] else {}
            except Exception:
                analysis = {}

            history.append({
                'id': row['id'],
                'user_id': row['user_id'],
                'image_url': row['image_url'],
                'created_at': row['created_at'],
                'analysis': analysis
            })

        return history
    finally:
        conn.close()


def normalize_survey_payload(payload):
    """설문 payload를 저장 가능한 형태로 정규화"""
    payload = payload or {}

    age_value = payload.get('age')
    age = None
    try:
        if age_value not in (None, ''):
            age = max(1, min(120, int(age_value)))
    except Exception:
        age = None

    symptoms = payload.get('symptoms')
    if not isinstance(symptoms, list):
        symptoms = []

    normalized_symptoms = []
    for symptom in symptoms[:30]:
        text = str(symptom).strip()
        if text:
            normalized_symptoms.append(text[:120])

    def clean_text(field_name, max_len=200):
        value = payload.get(field_name, '')
        return str(value).strip()[:max_len]

    return {
        'age': age,
        'gender': clean_text('gender', 30),
        'wearing': clean_text('wearing', 40),
        'conditions': clean_text('conditions', 500),
        'surgery_history': clean_text('surgery_history', 300),
        'smoking': clean_text('smoking', 20),
        'drinking': clean_text('drinking', 20),
        'symptoms': normalized_symptoms,
        'other_notes': clean_text('other_notes', 500)
    }


def save_survey_record(user_id, survey_payload):
    """사용자 설문 기록 저장"""
    safe_user_id = normalize_user_id(user_id)
    normalized_payload = normalize_survey_payload(survey_payload)

    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO survey_history (user_id, survey_json)
            VALUES (?, ?)
            """,
            (
                safe_user_id,
                json.dumps(normalized_payload, ensure_ascii=False)
            )
        )
        conn.commit()
        return int(cur.lastrowid), normalized_payload
    finally:
        conn.close()


# ========================================
# [3] 자동 촬영 조건 확인 함수
# ========================================
def check_auto_capture(detections, eye_crop_bbox, guideline_bbox):
    """
    자동 촬영 조건 확인
    
    조건 1: 중심점 거리 확인 (AUTO_DIST_THRESHOLD)
    조건 2: 눈 크기 비율 확인 (AUTO_SCALE_MIN ~ AUTO_SCALE_MAX)
    
    Args:
        detections: YOLO 검출 결과 (중심점 좌표 포함)
        eye_crop_bbox: 눈 크로핑 바운딩박스 (x1, y1, x2, y2)
        guideline_bbox: 가이드라인 바운딩박스 (x1, y1, x2, y2)
    
    Returns:
        bool: 자동 촬영 조건 만족 여부
    """
    try:
        # 검출 결과가 없으면 False
        if not detections or len(detections) == 0:
            return False
        
        # 검출된 눈의 중심점 계산
        detection = detections[0]
        x_center = (detection.get('x1', 0) + detection.get('x2', 0)) / 2
        y_center = (detection.get('y1', 0) + detection.get('y2', 0)) / 2
        
        # 가이드라인 중심점 계산
        g_x1, g_y1, g_x2, g_y2 = guideline_bbox
        guideline_x_center = (g_x1 + g_x2) / 2
        guideline_y_center = (g_y1 + g_y2) / 2
        
        # 조건 1: 중심점 거리 확인
        dist = np.sqrt((x_center - guideline_x_center)**2 + (y_center - guideline_y_center)**2)
        if dist > config.AUTO_DIST_THRESHOLD:
            return False
        
        # 조건 2: 눈 크기 비율 확인
        eye_width = eye_crop_bbox[2] - eye_crop_bbox[0]
        eye_height = eye_crop_bbox[3] - eye_crop_bbox[1]
        eye_area = eye_width * eye_height
        
        guideline_width = g_x2 - g_x1
        guideline_height = g_y2 - g_y1
        guideline_area = guideline_width * guideline_height
        
        scale_ratio = eye_area / guideline_area if guideline_area > 0 else 0
        
        if scale_ratio < config.AUTO_SCALE_MIN or scale_ratio > config.AUTO_SCALE_MAX:
            return False
        
        return True
    
    except Exception as e:
        print(f"[WARNING] 자동 촬영 조건 확인 실패: {e}")
        return False


def should_auto_capture():
    """
    자동 촬영 프레임 누적 확인
    
    Returns:
        bool: 자동 촬영 실행 여부
    """
    CaptureState.ACCEPTED_FRAMES += 1
    
    # AUTO_CAPTURE_HOLD_FRAMES만큼 프레임이 조건을 만족하면 True
    if CaptureState.ACCEPTED_FRAMES >= config.AUTO_CAPTURE_HOLD_FRAMES:
        CaptureState.ACCEPTED_FRAMES = 0
        return True
    
    return False

# ========================================
# [2] LED 하드웨어 설정 (Jetson)
# ========================================
try:
    import Jetson.GPIO as GPIO
    LED_PIN = 32
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    pwm_led = GPIO.PWM(LED_PIN, 1000)
    pwm_led.start(0)
    LED_AVAILABLE = True
except:
    print("[WARNING] Jetson GPIO not available. LED control disabled.")
    LED_AVAILABLE = False


# ========================================
# [2-1] YOLO 눈 감지 여부 확인
# ========================================
def check_eye_detection(frame):
    """
    현재 프레임에서 눈이 감지되는지 확인
    
    Args:
        frame: 검사할 이미지 프레임
        
    Returns:
        tuple: (감지 여부, 신뢰도)
    """
    if frame is None:
        return False, 0
    
    try:
        manager = get_models()
        detector = manager.get_detector()
        detections = detector.detect(frame, conf_threshold=config.YOLO_STATUS_CONF_THRESHOLD)

        if detections is None or not hasattr(detections, 'boxes'):
            return False, 0

        boxes = detections.boxes
        if boxes is None or len(boxes) == 0:
            return False, 0

        conf_tensor = boxes.conf
        if conf_tensor is None or conf_tensor.numel() == 0:
            return False, 0

        best_confidence = float(conf_tensor.max().item())
        detected = best_confidence >= config.YOLO_STATUS_CONF_THRESHOLD

        return detected, best_confidence

        return False, 0
    except Exception as e:
        print(f"[WARNING] 눈 감지 확인 실패: {e}")
        return False, 0


# ========================================
# [2-2] EfficientNet을 이용한 질환 분석
# ========================================
def analyze_eye_image(base64_image_data):
    """
    Base64로 인코딩된 이미지를 받아 EfficientNet으로 질환 분류
    
    Args:
        base64_image_data: Base64로 인코딩된 이미지
        
    Returns:
        dict: 분석 결과
    """
    try:
        # 1. Base64 → OpenCV 이미지 변환
        image_data = base64.b64decode(base64_image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return None, None, "이미지 디코딩 실패"
        
        # 2. 모델 매니저에서 분류기 가져오기
        manager = get_models()
        classifier = manager.get_classifier()
        
        # 3. 분류 수행
        classification = classifier.classify(img_bgr)
        
        disease_label = classification['disease']
        confidence_score = classification['confidence'] * 100  # 퍼센티지
        
        return disease_label, confidence_score, None
        
    except Exception as e:
        print(f"[ERROR] 이미지 분석 실패: {e}")
        return None, None, str(e)


def decode_base64_image(base64_image_data):
    """Base64(Data URL 포함) 문자열을 OpenCV BGR 이미지로 디코딩"""
    if not base64_image_data:
        return None, "이미지 데이터가 비어있습니다"

    try:
        encoded = base64_image_data.split(',', 1)[1] if ',' in base64_image_data else base64_image_data
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return None, "이미지 디코딩 실패"

        return img_bgr, None
    except Exception as e:
        return None, str(e)


def upscale_eye_crop_for_classifier(eye_crop):
    """작은 눈 크롭 해상도 방어를 위해 INTER_CUBIC 기반 224x224 리사이즈"""
    if eye_crop is None or eye_crop.size == 0:
        return None

    return cv2.resize(eye_crop, config.CLASSIFIER_INPUT_SIZE, interpolation=cv2.INTER_CUBIC)


def analyze_bilateral_from_image(img_bgr):
    """
    단일 캡처 이미지(양안 포함) 분석
    1) YOLO 검출/크롭
    2) 메모리 정리
    3) EfficientNet 양안 순차 분류
    """
    try:
        manager = get_models()
        detector = manager.get_detector()
        classifier = manager.get_classifier()
        analyzer = manager.get_analyzer()

        source_h, source_w = img_bgr.shape[:2]

        # Step 1) YOLO 검출 + 크롭
        yolo_start = time.time()
        detections = detector.detect(img_bgr, conf_threshold=config.YOLO_CONF_THRESHOLD)
        eye_crops = detector.crop_eyes(img_bgr, detections)
        yolo_elapsed_ms = (time.time() - yolo_start) * 1000.0

        # 검출 신뢰도 기준 상위 2개 유지
        eye_crops = sorted(eye_crops, key=lambda x: x['confidence'], reverse=True)[:2]
        eye_crops = sorted(eye_crops, key=lambda x: ((x['bbox'][0] + x['bbox'][2]) / 2.0))

        # YOLO 단계 메모리 정리 (OOM 방지)
        del detections
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(eye_crops) == 0:
            return {
                'status': 'warning',
                'message': '양안을 검출하지 못했습니다. 얼굴을 조금 더 가까이 맞춰주세요.',
                'left_eye': None,
                'right_eye': None,
                'meta': {
                    'source_resolution': [source_w, source_h],
                    'eyes_detected': 0,
                    'yolo_time_ms': round(yolo_elapsed_ms, 1)
                }
            }

        # 좌/우 레이블링
        labeled_eyes = {}
        if len(eye_crops) >= 2:
            labeled_eyes['left_eye'] = eye_crops[0]
            labeled_eyes['right_eye'] = eye_crops[1]
        else:
            center_x = (eye_crops[0]['bbox'][0] + eye_crops[0]['bbox'][2]) / 2.0
            if center_x < (source_w / 2.0):
                labeled_eyes['left_eye'] = eye_crops[0]
                labeled_eyes['right_eye'] = None
            else:
                labeled_eyes['left_eye'] = None
                labeled_eyes['right_eye'] = eye_crops[0]

        # Step 2) EfficientNet 순차 분류
        result = {
            'status': 'success' if len(eye_crops) >= 2 else 'warning',
            'message': '양안 분석 완료' if len(eye_crops) >= 2 else '한쪽 눈만 검출되었습니다.',
            'left_eye': None,
            'right_eye': None,
            'meta': {
                'source_resolution': [source_w, source_h],
                'eyes_detected': len(eye_crops),
                'yolo_time_ms': round(yolo_elapsed_ms, 1)
            }
        }

        for side in ['left_eye', 'right_eye']:
            eye_item = labeled_eyes.get(side)
            if eye_item is None:
                continue

            cls_start = time.time()
            prepared_eye = upscale_eye_crop_for_classifier(eye_item['image'])
            if prepared_eye is None:
                continue

            classification = classifier.classify(prepared_eye)
            eye_analysis = analyzer.analyze(prepared_eye)
            cls_elapsed_ms = (time.time() - cls_start) * 1000.0

            result[side] = {
                'disease': classification['disease'],
                'class': classification['class'],
                'confidence': round(float(classification['confidence']) * 100.0, 2),
                'redness': round(float(eye_analysis.get('redness', 0.0)), 4),
                'bbox': eye_item['bbox'],
                'detection_confidence': round(float(eye_item['confidence']) * 100.0, 2),
                'process_time_ms': round(cls_elapsed_ms, 1)
            }

            del prepared_eye
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"[ERROR] 양안 분석 실패: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


def analyze_bilateral_from_base64(base64_image_data):
    img_bgr, decode_error = decode_base64_image(base64_image_data)
    if decode_error:
        return {
            'status': 'error',
            'message': decode_error
        }

    return analyze_bilateral_from_image(img_bgr)


def build_ai_guide(analysis):
    """분석 결과 기반 리포트 가이드 생성"""
    left = analysis.get('left_eye') or {}
    right = analysis.get('right_eye') or {}

    candidates = []
    if left.get('disease'):
        candidates.append({'side': 'left', 'disease': left.get('disease', ''), 'confidence': float(left.get('confidence', 0))})
    if right.get('disease'):
        candidates.append({'side': 'right', 'disease': right.get('disease', ''), 'confidence': float(right.get('confidence', 0))})

    if not candidates:
        return {
            'risk_level': 'safe',
            'tag_text': '안내: 유효한 분석 데이터 없음',
            'summary': '눈 영역 분석 데이터가 부족합니다. 조명을 보강하고 다시 촬영해 주세요.',
            'recommended_departments': ['일반 안과'],
            'daily_care': [
                '촬영 시 정면 응시 및 충분한 조명을 확보하세요.',
                '눈 자극(손 비비기, 렌즈 장시간 착용)을 피하세요.'
            ]
        }

    top = max(candidates, key=lambda x: x['confidence'])
    disease = str(top['disease'])
    conf = float(top['confidence'])

    if conf >= 80:
        risk_level = 'danger'
    elif conf >= 60:
        risk_level = 'warning'
    else:
        risk_level = 'safe'

    rule = {
        'tag': f'분석 결과: {disease} ({conf:.1f}%)',
        'summary': f"{top['side']} 눈에서 {disease} 가능성이 관측되었습니다.",
        'dept': ['일반 안과'],
        'care': [
            '건조한 환경을 피하고 실내 습도를 유지하세요.',
            '눈 비비기를 피하고 필요 시 인공눈물을 사용하세요.'
        ]
    }

    if '결막염' in disease or 'Conjunctivitis' in disease:
        rule = {
            'tag': f'주의: 결막염 의심 ({conf:.1f}%)',
            'summary': '충혈/가려움 등 결막염 관련 징후가 관찰되었습니다. 증상이 지속되면 진료를 권장합니다.',
            'dept': ['일반 안과', '안구건조증 클리닉'],
            'care': ['손 위생을 철저히 하고 눈 비비기를 피하세요.', '화면 사용 중 20-20-20 규칙으로 눈 피로를 줄이세요.']
        }
    elif '백내장' in disease or 'Cataract' in disease:
        rule = {
            'tag': f'주의: 백내장 징후 ({conf:.1f}%)',
            'summary': '수정체 혼탁 관련 징후가 감지되었습니다. 시야 흐림이 느껴지면 정밀검사가 필요합니다.',
            'dept': ['백내장/망막 전문 안과', '일반 안과'],
            'care': ['야간 눈부심/시야 흐림 증상을 기록해 진료 시 전달하세요.', '정기적인 시력검사 일정을 권장합니다.']
        }
    elif '포도막염' in disease or 'Uveitis' in disease:
        rule = {
            'tag': f'주의: 포도막염 의심 ({conf:.1f}%)',
            'summary': '염증성 징후가 관측되었습니다. 통증/광과민이 있으면 빠른 진료가 필요합니다.',
            'dept': ['염증성 안질환 클리닉', '일반 안과'],
            'care': ['강한 빛 노출을 줄이고 선글라스를 사용하세요.', '통증/충혈 악화 시 즉시 병원 방문을 권장합니다.']
        }
    elif '다래끼' in disease or 'Eyelid' in disease:
        rule = {
            'tag': f'주의: 다래끼 가능성 ({conf:.1f}%)',
            'summary': '눈꺼풀 주변 염증 징후가 감지되었습니다. 증상 지속 시 진료를 권장합니다.',
            'dept': ['일반 안과'],
            'care': ['눈꺼풀을 청결하게 유지하고 화장품 사용을 줄이세요.', '온찜질을 짧게 하루 1~2회 적용하세요.']
        }
    elif '일반' in disease or 'Normal' in disease:
        rule = {
            'tag': f'정상 소견 우세 ({conf:.1f}%)',
            'summary': '현재 촬영 기준으로 특이 소견이 상대적으로 낮습니다.',
            'dept': ['정기 검진용 일반 안과'],
            'care': ['장시간 화면 사용 시 주기적 휴식을 취하세요.', '충혈/통증이 생기면 조기 검진을 권장합니다.']
        }

    return {
        'risk_level': risk_level,
        'tag_text': rule['tag'],
        'summary': rule['summary'],
        'recommended_departments': rule['dept'],
        'daily_care': rule['care']
    }


# ========================================
# [3] GStreamer 파이프라인
# ========================================
def gstreamer_pipeline(cam_id=0):
    """
    Logitech C920 웹캠용 GStreamer 파이프라인
    
    Args:
        cam_id: 비디오 장치 ID (기본값: 0 = /dev/video0)
        
    Returns:
        GStreamer 파이프라인 문자열
    """
    return (
        f"v4l2src device=/dev/video{cam_id} ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )

# ========================================
# [4] 백그라운드 카메라 스레드 
# ========================================
def start_camera_thread():
    """
    카메라 프레임을 백그라운드에서 계속 읽고 current_frame 업데이트
    """
    global current_frame, camera_running, camera_thread

    if camera_running and camera_thread is not None and camera_thread.is_alive():
        return
    
    def camera_worker():
        global current_frame, camera_running
        print("[카메라 스레드] 백그라운드 프레임 수집 시작...")
        time.sleep(1) # 초기화 대기
        
        cam_id = getattr(config, 'CAMERA_DEVICE_INDEX', 0)
        cap = None
        
        capture_candidates = [
            (f"gstreamer:/dev/video{cam_id}", lambda: cv2.VideoCapture(gstreamer_pipeline(cam_id), cv2.CAP_GSTREAMER)),
            (f"v4l2-index:{cam_id}", lambda: cv2.VideoCapture(cam_id, cv2.CAP_V4L2)),
            (f"v4l2-path:/dev/video{cam_id}", lambda: cv2.VideoCapture(f"/dev/video{cam_id}", cv2.CAP_V4L2)),
            (f"any:/dev/video{cam_id}", lambda: cv2.VideoCapture(f"/dev/video{cam_id}", cv2.CAP_ANY)),
        ]

        for source_name, opener in capture_candidates:
            try:
                candidate = opener()
                if candidate.isOpened():
                    cap = candidate
                    print(f"[카메라 스레드] ✓ 카메라 소스 연결 성공: {source_name}")
                    break
                candidate.release()
            except Exception as e:
                print(f"[카메라 스레드] 카메라 소스 시도 실패 ({source_name}): {e}")

        if cap is None:
            print(f"[카메라 스레드] ✗ 카메라 열기 실패 (/dev/video{cam_id})")
            camera_running = False
            return

        # 로지텍 카메라 권장 포맷/해상도로 고정
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[카메라 스레드] ✓ 카메라 초기화 완료, 프레임 수집 중...")
        frame_count = 0
        
        while camera_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[카메라 스레드] ✗ 프레임 읽기 실패!")
                time.sleep(1)
                continue
            
            current_frame = frame.copy()
            frame_count += 1
            
            # 잘 돌아가는지 확인하기 위한 하트비트 로그 (일단 켜둡니다)
            if frame_count % 30 == 0:
                print(f"[카메라 스레드] {frame_count}개 프레임 수집 완료")
            
            time.sleep(0.033)  # ~30fps
        
        cap.release()
        print(f"[카메라 스레드] 종료 ({frame_count}개 프레임 수집됨)")
    
    camera_running = True
    camera_thread = threading.Thread(target=camera_worker, daemon=True)
    camera_thread.start()
    print("[카메라 스레드] ✓ 시작")


def stop_camera_thread():
    """카메라 스레드 안전 종료"""
    global camera_running, camera_thread
    camera_running = False

    if camera_thread is not None and camera_thread.is_alive():
        camera_thread.join(timeout=2.0)

    camera_thread = None


# ========================================
# [4] 카메라 프레임 스트림
# ========================================
def gen_frames():
    """
    백그라운드 스레드가 수집한 current_frame을 브라우저로 실시간 스트리밍
    """
    global current_frame, debug_boxes_cache, debug_frame_counter
    print("[gen_frames] 브라우저 스트리밍 시작...")
    
    while True:
        if current_frame is None:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, 'Waiting for camera...', (170, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            ret, buffer = cv2.imencode('.jpg', dummy)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        frame_to_send = current_frame.copy()

        # 임시 디버그: YOLO 박스 오버레이
        try:
            debug_frame_counter += 1
            if debug_frame_counter % 5 == 0 or len(debug_boxes_cache) == 0:
                manager = get_models()
                detector = manager.get_detector()
                detections = detector.detect(
                    frame_to_send,
                    conf_threshold=config.YOLO_STATUS_CONF_THRESHOLD
                )

                boxes = []
                if detections is not None and hasattr(detections, 'boxes') and detections.boxes is not None:
                    for box in detections.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf.item()) if box.conf is not None else 0.0
                        boxes.append((x1, y1, x2, y2, conf))

                debug_boxes_cache = boxes

            for (x1, y1, x2, y2, conf) in debug_boxes_cache:
                cv2.rectangle(frame_to_send, (x1, y1), (x2, y2), (40, 205, 65), 2)
                label = f"Eye {conf*100:.1f}%"
                cv2.putText(
                    frame_to_send,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 205, 65),
                    2
                )
        except Exception as e:
            print(f"[WARNING] YOLO 박스 오버레이 실패: {e}")

        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
               
        time.sleep(0.033)


# ========================================
# [5] 핵심 진단 파이프라인 (3단계)
# ========================================
def run_diagnosis_pipeline(snapshot):
    """
    3단계 진단 파이프라인 실행
    
    Stage 1: YOLO 눈 검출 및 크롭
    Stage 2: EfficientNet 질환 분류
    Stage 3: 홍채 제거 및 충혈도 산출
    
    Args:
        snapshot (np.ndarray): 캡처된 프레임
        
    Returns:
        dict: 진단 결과
    """
    try:
        # 모델 매니저 가져오기
        manager = get_models()
        detector = manager.get_detector()
        classifier = manager.get_classifier()
        analyzer = manager.get_analyzer()
        logger = manager.get_logger()
        
        result = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'left_eye': None,
            'right_eye': None,
            'pipeline_steps': []
        }
        
        # ============================================
        # Stage 1: 눈 검출 (YOLO)
        # ============================================
        print("[Stage 1] YOLO 눈 검출 시작...")
        start_time = time.time()
        
        detections = detector.detect(snapshot)
        eye_crops = detector.crop_eyes(snapshot, detections)
        
        elapsed = time.time() - start_time
        result['pipeline_steps'].append({
            'stage': 'Detection (YOLO)',
            'status': 'completed',
            'time_ms': f"{elapsed*1000:.1f}",
            'eyes_detected': len(eye_crops)
        })
        print(f"  ✓ {len(eye_crops)}개 눈 검출 ({elapsed*1000:.1f}ms)")
        
        if len(eye_crops) == 0:
            result['status'] = 'warning'
            result['message'] = '눈을 검출하지 못했습니다. 가이드라인을 확인하세요.'
            return result
        
        # ============================================
        # Stage 2 & 3: 각 눈마다 분류 및 분석
        # ============================================
        eye_sides = ['left_eye', 'right_eye']
        
        for idx, eye_crop in enumerate(eye_crops[:2]):  # 최대 2개 눈만 처리
            eye_side = eye_sides[idx]
            print(f"\n[Stage 2-3] {eye_side} 분류 및 분석...")
            
            crop_image = eye_crop['image']
            
            # Stage 2: 질환 분류
            print(f"  Stage 2: 질환 분류 중...")
            start_time = time.time()
            classification = classifier.classify(crop_image)
            elapsed_classify = time.time() - start_time
            print(f"    ✓ {classification['disease']} (신뢰도: {classification['confidence']*100:.1f}%)")
            
            # Stage 3: 눈 분석 (홍채 제거 + 충혈도)
            print(f"  Stage 3: 충혈도 분석 중...")
            start_time = time.time()
            analysis = analyzer.analyze(crop_image)
            elapsed_analyze = time.time() - start_time
            print(f"    ✓ 충혈도: {analysis['redness']:.3f}")
            
            # 결과 저장
            result[eye_side] = {
                'detected': True,
                'disease': classification['disease'],
                'disease_class': classification['class'],
                'confidence': float(classification['confidence']),
                'probabilities': classification['probabilities'],
                'redness': float(analysis['redness']),
                'bbox': eye_crop['bbox'],
                'detection_confidence': float(eye_crop['confidence']),
                'processing_time_ms': f"{elapsed_classify + elapsed_analyze:.1f}"
            }
            
            result['pipeline_steps'].append({
                'stage': f'Classification & Analysis ({eye_side})',
                'status': 'completed',
                'time_ms': f"{elapsed_classify + elapsed_analyze:.1f}",
                'disease': classification['disease'],
                'redness': f"{analysis['redness']:.3f}"
            })
        
        # ============================================
        # 결과 로깅
        # ============================================
        try:
            logger.log_result({
                'patient_id': request.form.get('patient_id', 'unknown'),
                'left_eye': result.get('left_eye'),
                'right_eye': result.get('right_eye'),
                'notes': request.form.get('notes', '')
            })
        except Exception as e:
            print(f"[WARNING] 로깅 실패: {e}")
        
        print(f"\n[진단 완료] 전체 소요시간: {time.time():.2f}초")
        return result
        
    except Exception as e:
        print(f"[ERROR] 진단 파이프라인 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


# ========================================
# [6] Flask 라우트
# ========================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """실시간 영상 스트림 (MJPEG)"""
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_frame')
def video_frame():
    """현재 카메라 프레임을 JPEG로 반환"""
    global current_frame
    try:
        if current_frame is None:
            # 검정색 더미 프레임 반환
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', dummy)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        
        ret, buffer = cv2.imencode('.jpg', current_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        print(f"[ERROR] /video_frame 실패: {e}")
        return Response(b'', status=500)


@app.route('/detect_status', methods=['GET'])
def detect_status():
    """
    현재 프레임에서 눈의 감지 여부만 반환 (HTML에서 상태 표시용)
    
    Returns:
        - detected: bool - 눈 감지 여부
        - confidence: float - 신뢰도 (0~1)
        - message: str - 상태 메시지
    """
    global current_frame
    try:
        if current_frame is None:
            return jsonify({
                'detected': False,
                'confidence': 0,
                'message': '카메라 연결을 기다리는 중...'
            }), 200
        
        detected, confidence = check_eye_detection(current_frame)
        
        return jsonify({
            'detected': detected,
            'confidence': round(confidence, 3),
            'message': '🟢 눈이 감지되었습니다!' if detected else '⚪ 눈을 맞춰주세요'
        }), 200
    
    except Exception as e:
        print(f"[ERROR] 눈 감지 상태 조회 실패: {e}")
        return jsonify({
            'detected': False,
            'confidence': 0,
            'error': str(e)
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    웹 브라우저에서 캡처한 단일 Base64 이미지를
    YOLO(양안 검출/크롭) -> EfficientNet(양안 순차 분류)로 분석
    
    Request:
        {
            "image": "data:image/jpeg;base64,..."
        }
    
    Response:
        {
            "status": "success|warning|error",
            "message": "...",
            "left_eye": {...},
            "right_eye": {...},
            "meta": {...}
        }
    """
    try:
        data = request.json or {}
        if 'image' not in data:
            return jsonify({'error': '이미지 데이터가 없습니다'}), 400

        user_id = normalize_user_id(data.get('user_id'))
        img_bgr, decode_error = decode_base64_image(data['image'])
        if decode_error:
            return jsonify({'status': 'error', 'message': decode_error}), 400

        analysis = analyze_bilateral_from_image(img_bgr)
        if analysis.get('status') == 'error':
            return jsonify(analysis), 400

        analysis['guide'] = build_ai_guide(analysis)

        try:
            history_id, image_url = save_history_record(user_id, img_bgr, analysis)
            analysis['history_id'] = history_id
            analysis['saved_image_url'] = image_url
            analysis['user_id'] = user_id
        except Exception as save_error:
            analysis['save_warning'] = f'히스토리 저장 실패: {save_error}'

        return jsonify(analysis), 200
        
    except Exception as e:
        print(f"[ERROR] /analyze 라우트 실패: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    """사용자별 진단 히스토리 조회"""
    try:
        user_id = normalize_user_id(request.args.get('user_id'))
        limit = int(request.args.get('limit', 200))
        limit = max(1, min(limit, 1000))

        history = list_history_records(user_id, limit)
        return jsonify({
            'status': 'ok',
            'user_id': user_id,
            'count': len(history),
            'history': history
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/survey', methods=['POST'])
def api_survey():
    """사용자 문진 데이터 저장"""
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        survey = data.get('survey')

        safe_user_id = normalize_user_id(user_id)
        if safe_user_id == 'anonymous':
            return jsonify({
                'status': 'error',
                'message': '사용자 식별자가 필요합니다.'
            }), 400

        row_id, normalized_survey = save_survey_record(safe_user_id, survey)
        return jsonify({
            'status': 'ok',
            'survey_id': row_id,
            'user_id': safe_user_id,
            'survey': normalized_survey
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/diagnose', methods=['POST'])
def diagnose():
    """
    진단 실행 라우트
    클라이언트에서 '진단 시작' 버튼 클릭 시 호출
    
    촬영 상태:
    - 왼쪽 눈 촬영 → 오른쪽 눈으로 이동 → 오른쪽 눈 촬영 → 시퀀스 초기화
    """
    global current_frame
    
    if current_frame is None:
        return jsonify({
            'status': 'error',
            'message': '카메라 연결을 확인하세요.'
        }), 400
    
    try:
        # 1. 조명 제어 (LED 플래시)
        if LED_AVAILABLE:
            pwm_led.ChangeDutyCycle(100)  # 최대 밝기
            time.sleep(0.3)  # 빛 안정화
        
        # 2. 스냅샷 캡처
        snapshot = current_frame.copy()
        
        # 3. 이미지 전처리 (선택사항)
        snapshot = enhance_contrast(snapshot)
        
        # 4. 조명 낮추기
        if LED_AVAILABLE:
            pwm_led.ChangeDutyCycle(10)
        
        # 5. 진단 파이프라인 실행
        diagnosis_result = run_diagnosis_pipeline(snapshot)
        
        # 6. 촬영 상태 관리 및 UI 가이드라인 업데이트
        if diagnosis_result['status'] == 'success':
            current_eye = CaptureState.current_eye
            
            # 현재 촬영 눈 표시
            CaptureState.mark_captured(current_eye)
            diagnosis_result['current_eye'] = current_eye
            diagnosis_result['captured_eyes'] = CaptureState.captured_eyes.copy()
            
            # UI 가이드라인 텍스트 업데이트
            if current_eye == "LEFT_EYE":
                diagnosis_result['next_guide_text'] = "오른쪽 눈을 맞춰주세요. 진단 시작을 누르세요."
                CaptureState.move_to_next_eye()  # 오른쪽 눈으로 이동
            else:
                diagnosis_result['next_guide_text'] = "진단이 완료되었습니다."
                # 양쪽 눈 촬영 완료
                if CaptureState.captured_eyes["LEFT_EYE"] and CaptureState.captured_eyes["RIGHT_EYE"]:
                    diagnosis_result['diagnosis_complete'] = True
                    CaptureState.reset()  # 시퀀스 초기화
        
        # 7. 스냅샷 저장
        if diagnosis_result['status'] == 'success':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            eye_label = CaptureState.captured_eyes
            img_path = os.path.join(
                config.IMAGE_SAVE_DIR, 
                f"diagnosis_{timestamp}_{CaptureState.current_eye}.jpg"
            )
            cv2.imwrite(img_path, snapshot)
            diagnosis_result['snapshot_path'] = img_path
        
        return jsonify(diagnosis_result)
    
    except Exception as e:
        print(f"[ERROR] /diagnose 라우트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        if LED_AVAILABLE:
            pwm_led.ChangeDutyCycle(10)
        
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/capture/state', methods=['GET'])
def get_capture_state():
    """
    현재 촬영 상태 조회
    
    Returns:
        - current_eye: 현재 촬영 중인 눈 ("LEFT_EYE" 또는 "RIGHT_EYE")
        - captured_eyes: {"LEFT_EYE": bool, "RIGHT_EYE": bool}
        - auto_capture_ready: 자동 촬영 준비 상태
        - guide_text: UI에 표시할 가이드 텍스트
    """
    try:
        guide_text = "왼쪽 눈을 맞춰주세요. 진단 시작을 누르세요."
        if CaptureState.current_eye == "RIGHT_EYE":
            guide_text = "오른쪽 눈을 맞춰주세요. 진단 시작을 누르세요."
        
        return jsonify({
            'status': 'ok',
            'current_eye': CaptureState.current_eye,
            'captured_eyes': CaptureState.captured_eyes,
            'auto_capture_ready': CaptureState.auto_capture_ready,
            'guide_text': guide_text
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/capture/reset', methods=['POST'])
def reset_capture_state():
    """
    촬영 시퀀스 초기화
    새로운 진단을 시작할 때 호출
    """
    try:
        CaptureState.reset()
        return jsonify({
            'status': 'ok',
            'message': '촬영 상태가 초기화되었습니다.'
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/status')
def status():
    """서버 상태 확인"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'device': torch.cuda.get_device_name(0),
            'total_memory_gb': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
            'allocated_memory_gb': f"{torch.cuda.memory_allocated(0) / 1e9:.2f}",
            'reserved_memory_gb': f"{torch.cuda.memory_reserved(0) / 1e9:.2f}"
        }
    
    return jsonify({
        'status': 'running',
        'models_loaded': model_manager is not None,
        'camera_connected': current_frame is not None,
        'gpu_info': gpu_info
    })


@app.route('/login')
def login():
    """로그인 페이지"""
    return render_template('login.html')


@app.route('/admin/config')
def admin_config_page():
    """관리자 설정 페이지"""
    if not is_admin_session():
        return redirect(url_for('login'))
    return render_template('admin_config.html')


@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    """관리자 로그인 세션 발급"""
    try:
        data = request.json or {}
        identifier = str(data.get('identifier', '')).strip().lower()

        if identifier == 'admin':
            session['is_admin'] = True
            return jsonify({'status': 'ok', 'role': 'admin'}), 200

        session['is_admin'] = False
        return jsonify({'status': 'ok', 'role': 'user'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/config', methods=['GET'])
def api_admin_config_get():
    """관리자 설정 조회"""
    if not is_admin_session():
        return jsonify({'status': 'error', 'message': '관리자 권한이 필요합니다.'}), 403

    return jsonify({
        'status': 'ok',
        'config': get_admin_config_snapshot(),
        'editable_keys': list(ADMIN_EDITABLE_CONFIG_KEYS.keys())
    }), 200


@app.route('/api/admin/config', methods=['POST'])
def api_admin_config_update():
    """관리자 설정 저장"""
    if not is_admin_session():
        return jsonify({'status': 'error', 'message': '관리자 권한이 필요합니다.'}), 403

    try:
        data = request.json or {}
        updates_raw = data.get('updates') or {}
        if not isinstance(updates_raw, dict) or not updates_raw:
            return jsonify({'status': 'error', 'message': '저장할 설정값이 없습니다.'}), 400

        casted_updates = {}
        for key, raw_value in updates_raw.items():
            if key not in ADMIN_EDITABLE_CONFIG_KEYS:
                continue
            casted_updates[key] = cast_config_value(key, raw_value)

        if not casted_updates:
            return jsonify({'status': 'error', 'message': '유효한 설정 항목이 없습니다.'}), 400

        update_config_file_values(casted_updates)

        for key, value in casted_updates.items():
            setattr(config, key, value)

        return jsonify({
            'status': 'ok',
            'message': '설정이 저장되었습니다. 일부 항목은 재시작 후 완전 적용됩니다.',
            'updated': casted_updates
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/capture')
def capture():
    """촬영 페이지"""
    return render_template('capture.html')


@app.route('/result')
def result():
    """진단 결과 페이지"""
    return render_template('result.html')


@app.route('/report')
def report():
    """리포트 페이지"""
    return render_template('report.html')


@app.route('/survey')
def survey():
    """설문조사 페이지"""
    return render_template('survey.html')


# ========================================
# [7] 서버 시작/종료
# ========================================

@app.before_request
def initialize_on_first_request():
    """서버 시작 시 모델 로드 및 카메라 스레드 시작 (Flask 2.3+ 호환)"""
    global model_manager, models_initialized, camera_running
    if not models_initialized:
        models_initialized = True
        init_history_db()
        print("\n" + "="*50)
        print("[Eye Disease Detection Server]")
        print("="*50)
        model_manager = initialize_models()
        
        # 카메라 백그라운드 스레드 시작
        start_camera_thread()
        
        print("\n✓ 서버 준비 완료! http://0.0.0.0:5000 에서 접속하세요\n")

    # 디버그 리로더/예외 상황에서 카메라 스레드가 내려갔으면 재시작
    if models_initialized and (not camera_running):
        start_camera_thread()


def cleanup_resources():
    """프로세스 종료 시 자원 정리"""
    stop_camera_thread()

    if LED_AVAILABLE:
        pwm_led.stop()
        GPIO.cleanup()

    print("[Shutdown] 서버 종료")


atexit.register(cleanup_resources)


if __name__ == '__main__':
    print("Starting Eye Disease Detection Server...")
    
    # 서버 시작 전에 모델과 카메라 초기화
    if not models_initialized:
        models_initialized = True
        init_history_db()
        print("\n" + "="*50)
        print("[Eye Disease Detection Server]")
        print("="*50)
        model_manager = initialize_models()
        
        # 카메라 백그라운드 스레드 시작
        start_camera_thread()
        
        print("\n✓ 서버 준비 완료! http://0.0.0.0:5000 에서 접속하세요\n")
    
    app.run(
        host=config.SERVER_IP,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE,
        threaded=True,
        use_reloader=False
    )