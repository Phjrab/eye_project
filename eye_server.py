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
def gstreamer_pipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=2,
):
    """GStreamer 파이프라인 설정"""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


# ========================================
# [4] 카메라 프레임 스트림
# ========================================
def gen_frames():
    """
    카메라 프레임을 실시간으로 스트림
    current_frame에도 최신 프레임 저장
    """
    global current_frame
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
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
        
        # 6. 스냅샷 저장
        if diagnosis_result['status'] == 'success':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_path = os.path.join(config.IMAGE_SAVE_DIR, f'diagnosis_{timestamp}.jpg')
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


# ========================================
# [7] 서버 시작/종료
# ========================================

@app.before_first_request
def before_first_request():
    """서버 시작 시 모델 로드"""
    global model_manager
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

    except Exception as e:
        pwm_led.ChangeDutyCycle(0) # 에러 시 조명 끄기
        return jsonify({"status": "error", "message": str(e)}), 500

# Function to calculate redness ratio in a given region of interest (ROI)
def calculate_redness_ratio(image, roi_coords=(130, 90, 380, 180)):
    x, y, w, h = roi_coords
    roi = image[y:y+h, x:x+w]  # 관심 영역(ROI) 추출

    # HSV 색공간으로 변환 (빨간색 추출에 유리)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 설정 (두 범위를 합쳐야 함)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # 첫 번째 빨간색 범위 마스크 생성
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # 두 번째 빨간색 범위 마스크 생성
    red_mask = mask1 + mask2  # 두 마스크를 합침

    # 빨간색 픽셀 계산
    red_pixels = cv2.countNonZero(red_mask)  # 빨간색 픽셀 개수 계산
    total_pixels = w * h  # 전체 픽셀 개수 계산
    redness_percentage = (red_pixels / total_pixels) * 100  # 빨간색 픽셀 비율 계산

    return round(redness_percentage, 2)  # 소수점 둘째 자리까지 반올림하여 반환

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)