"""
Eye Disease Detection Server
웹 인터페이스와 AI 파이프라인의 오케스트레이터
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import torch
import time
import os
import json
from datetime import datetime

import config
from model_loader import initialize_models, get_models
from utils.image_proc import resize_image, enhance_contrast

# ========================================
# [1] Flask 앱 설정
# ========================================
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 전역 변수
current_frame = None  # 실시간 카메라 프레임
model_manager = None  # 싱글톤 모델 매니저
models_initialized = False  # 모델 초기화 완료 여부

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
# [3] GStreamer 파이프라인
# ========================================
def gstreamer_pipeline(cam_id=1):
    """
    Logitech C920 웹캠용 GStreamer 파이프라인
    
    Args:
        cam_id: 비디오 장치 ID (기본값: 1 = /dev/video1)
        
    Returns:
        GStreamer 파이프라인 문자열
    """
    return (
        f"v4l2src device=/dev/video{cam_id} ! "
        "image/jpeg, width=1280, height=720, framerate=30/1 ! "
        "jpegdec ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )


# ========================================
# [4] 카메라 프레임 스트림
# ========================================
def gen_frames():
    """
    카메라 프레임을 실시간으로 스트림
    current_frame에도 최신 프레임 저장
    Logitech C920 웹캠 (/dev/video1) 연결
    """
    global current_frame
    cap = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("⚠️  GStreamer 실패, 일반 모드(/dev/video1) 시도...")
        cap = cv2.VideoCapture(1)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 최신 프레임 저장 (스냅샷용)
        current_frame = frame.copy()
        
        # MJPEG 스트림으로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


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
    """실시간 영상 스트림"""
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


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


@app.route('/capture')
def capture():
    """촬영 페이지"""
    return render_template('capture.html')


@app.route('/result')
def result():
    """진단 결과 페이지"""
    return render_template('result.html')


@app.route('/survey')
def survey():
    """설문조사 페이지"""
    return render_template('survey.html')


# ========================================
# [7] 서버 시작/종료
# ========================================

@app.before_request
def initialize_on_first_request():
    """서버 시작 시 모델 로드 (Flask 2.3+ 호환)"""
    global model_manager, models_initialized
    if not models_initialized:
        models_initialized = True
        print("\n" + "="*50)
        print("[Eye Disease Detection Server]")
        print("="*50)
        model_manager = initialize_models()
        print("\n✓ 서버 준비 완료! http://0.0.0.0:5000 에서 접속하세요\n")


@app.teardown_appcontext
def cleanup(exception=None):
    """서버 종료 시 정리"""
    if LED_AVAILABLE:
        pwm_led.stop()
        GPIO.cleanup()
    print("[Shutdown] 서버 종료")


if __name__ == '__main__':
    print("Starting Eye Disease Detection Server...")
    app.run(
        host=config.SERVER_IP,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE,
        threaded=True
    )