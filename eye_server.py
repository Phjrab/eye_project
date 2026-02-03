from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import Jetson.GPIO as GPIO
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import time

# [1] 설정 및 전역 변수
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
current_frame = None  # 카메라의 최신 프레임을 담는 바구니

# 세은이와 약속한 ROI (640x360 해상도 기준)
ROI_X, ROI_Y = 130, 90
ROI_W, ROI_H = 380, 180

# [2] AI 모델 로드 (EfficientNet-B0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 세준이가 학습시킨 모델 구조와 일치해야 함 (클래스 5개)
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)
model.to(device)
model.eval()

# EfficientNet 표준 전처리 (224x224)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# [3] LED 하드웨어 설정
LED_PIN = 32
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
pwm_led = GPIO.PWM(LED_PIN, 1000)
pwm_led.start(0)

# (GStreamer 및 gen_frames 함수는 기존과 동일하게 유지하되, 
# 아래와 같이 current_frame을 갱신하는 로직이 추가되어야 합니다.)

def gen_frames():
    global current_frame
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    while True:
        success, frame = cap.read()
        if not success: break
        
        current_frame = frame.copy() # 실시간 프레임을 전역 변수에 저장
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==========================================
# [핵심] 진단 실행 라우트 (POST /diagnose)
# ==========================================
@app.route('/diagnose', methods=['POST'])
def diagnose():
    global current_frame
    if current_frame is None:
        return jsonify({"error": "카메라 연결을 확인하세요."}), 400

    try:
        # 1. 조명 최대 밝기 (진단용 플래시)
        pwm_led.ChangeDutyCycle(100)
        time.sleep(0.2) # 빛이 안정될 때까지 찰나의 대기

        # 2. 분석용 스냅샷 캡처 및 ROI 크롭
        # 세은이가 정한 가이드라인 위치만 도려내기
        snap = current_frame.copy()
        roi_img = snap[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        
        # 촬영 직후 조명 낮추기 (눈부심 방지)
        pwm_led.ChangeDutyCycle(10)

        # 3. AI 모델 입력용 전처리
        # OpenCV(BGR) -> PIL(RGB) 변환
        pil_img = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # 4. AI 추론 (Inference)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            prob = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

        # 세준이의 데이터셋 순서 (정상, 백내장, 포도막염, 결막염, 다래끼)
        class_names = ["정상", "백내장", "포도막염", "결막염", "다래끼"]
        result_label = class_names[predicted.item()]

        # 5. 결과 반환
        return jsonify({
            "status": "success",
            "prediction": result_label,
            "confidence": f"{prob*100:.1f}%",
            "advice": "정확한 진단은 안과 전문의와 상담하세요."
        })

    except Exception as e:
        pwm_led.ChangeDutyCycle(0) # 에러 시 조명 끄기
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)