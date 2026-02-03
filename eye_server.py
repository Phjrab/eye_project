from flask import Flask, render_template, Response
import cv2
import numpy as np
import Jetson.GPIO as GPIO

# Flask 설정: 웹 파일들은 'web' 폴더 안에 있다고 알려줍니다.
app = Flask(__name__, 
            template_folder='web/templates', 
            static_folder='web/static')

# 젯슨 오린 나노 CSI 카메라(IMX477 등)를 위한 GStreamer 파이프라인
def gstreamer_pipeline(sensor_id=0, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, flip_method)
    )

def gen_frames():
    # 카메라 연결 (CSI 카메라 전용)
    camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # JPEG으로 인코딩하여 웹으로 전송 가능한 데이터로 변환
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # 이 부분의 이름이 세은 학생이 준 파일명과 정확히 일치해야 합니다.
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 웹 페이지에서 이미지를 실시간으로 갱신해주는 경로
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 0.0.0.0으로 열어야 같은 와이파이에 연결된 노트북에서도 접속 가능합니다.
    app.run(host='0.0.0.0', port=5000, debug=False)

##############################################################

def preprocess_frame(frame):
    # 1. 중앙 크롭 (가이드라인 위치에 맞게)
    h, w, _ = frame.shape
    size = 224
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    cropped = frame[start_y:start_y+size, start_x:start_x+size]
    
    # 2. 모델 입력용 변환 (BGR -> RGB, Normalization)
    img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # 0~1 사이로 정규화
    
    # 3. 차원 맞추기 (Batch size 추가)
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    return img

@app.route('/diagnose', methods=['POST'])
def diagnose():
    # 현재 카메라 프레임 가져오기 (전역 변수 등 활용)
    # 1. preprocess_frame() 실행
    # 2. model.predict() 실행
    # 3. 결과를 JSON으로 반환
    return {"status": "success", "prediction": "데이터 대기 중"}

##############################################################

# GPIO 설정
LED_PIN = 32 # PWM0
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# PWM 인스턴스 생성 (주파수 1000Hz)
pwm_led = GPIO.PWM(LED_PIN, 1000)
pwm_led.start(0) # 처음에는 꺼둠

@app.route('/diagnose', methods=['POST'])
def diagnose():
    # 1. 조명 100% 밝기로 켜기
    pwm_led.ChangeDutyCycle(100)
    
    # 2. 카메라 프레임 한 장 캡처
    # 3. AI 모델(EfficientNet)에 사진 넣기
    # 4. 나온 결과값을 바구니에 담아서 리턴
    
    # 3. 조명 다시 끄기 (또는 10%로 낮추기)
    pwm_led.ChangeDutyCycle(0)
    
    return jsonify({"status": "success"})