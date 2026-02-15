from flask import Flask, render_template, jsonify
import os
import sys
from pathlib import Path

# 경로 설정: utils와 database 모듈 import
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'database'))

from security_utils import phone_hash_id, _require_pepper
from db import init_db, upsert_user_by_phone, create_diagnosis_session

# ============================================================================
# 1. 환경변수 및 상수 설정
# ============================================================================

# HASH_PEPPER 환경변수 확인 및 처리
try:
    pepper = _require_pepper()  # HASH_PEPPER 필수 check
    print(f"✅ HASH_PEPPER 환경변수가 설정되어 있습니다.")
except RuntimeError as e:
    print(f"⚠️  경고: {e}")
    print("   → 환경변수 설정: export HASH_PEPPER='your_secret_pepper'")
    # 테스트 모드에서는 계속 진행하지만, 실제 운영 시에는 실패해야 함
    pass

DB_PATH = os.path.join(Path(__file__).parent, 'eye_diagnosis.db')

# ============================================================================
# 2. Flask 앱 설정 (은율님이 정한 폴더 구조 반영)
# ============================================================================

app = Flask(__name__, 
            template_folder='web/templates', 
            static_folder='web/static')

# ============================================================================
# 3. 데이터베이스 초기화 (앱 시작 시)
# ============================================================================

def init_app_db():
    """서버 시작 시 DB 초기화"""
    try:
        init_db(DB_PATH)
        print(f"✅ 데이터베이스 초기화 완료: {DB_PATH}")
    except Exception as e:
        print(f"❌ 데이터베이스 초기화 실패: {e}")

# ============================================================================
# 4. 라우트 정의
# ============================================================================

@app.route('/')
def index():
    """메인 페이지 (index.html)"""
    return render_template('index.html', status="Server Running")


@app.route('/result')
def result():
    """
    결과 페이지 테스트 (/result)
    - 임시 테스트 데이터를 생성하여 DB에 저장
    - 하준 팀장님의 EfficientNet 정확도(99.09%)와 충혈도 산출 공식 적용
    """
    
    try:
        # ====================================
        # Step 1: 테스트 사용자 생성/조회
        # ====================================
        test_phone = "010-9999-9999"
        test_display_name = "테스트 사용자"
        user_id = upsert_user_by_phone(DB_PATH, test_phone, test_display_name)
        print(f"✅ 테스트 사용자 ID: {user_id}")
        
        # ====================================
        # Step 2: AI 판독 데이터 (하준 팀장님 모델 기반)
        # ====================================
        # EfficientNet 정확도: 99.09%
        ai_reading = {
            "model": "Augmented_EffNet_V1_B0",
            "accuracy": 0.9909,
            "predicted_class": "Normal",
            "confidence": 0.9909,
            "timestamp": "2026-02-15T10:30:00"
        }
        
        # ====================================
        # Step 3: 픽셀 분석 수치 (충혈도 산출 공식 적용)
        # ====================================
        # 충혈도 산출 공식: Redness_Score = mean(a*) - 128
        # a*는 라브 색상공간에서 초록-빨강 축의 값
        # mean(a*) 값 예시: 140 (정상은 128)
        
        pixel_metrics = {
            "redness_score": 12.5,  # mean(a*) - 128 = 140 - 128 = 12.5
            "vessel_density": 0.35,
            "redness_area_ratio": 0.08,
            "total_pixels_analyzed": 1024000,
            "hyperemia_detected": False
        }
        
        # ====================================
        # Step 4: 설문 데이터 (임시)
        # ====================================
        survey = {
            "age": 35,
            "gender": "M",
            "symptoms": ["none"],
            "medical_history": [],
            "previous_diagnoses": []
        }
        
        # ====================================
        # Step 5: FHIR Bundle (선택사항, 나중에 추가 가능)
        # ====================================
        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "document",
            "entry": []
        }
        
        # ====================================
        # Step 6: DB에 진단 세션 저장
        # ====================================
        session_id = create_diagnosis_session(
            db_path=DB_PATH,
            user_id=user_id,
            ai_reading=ai_reading,
            pixel_metrics=pixel_metrics,
            survey=survey,
            impression="정상 소견입니다. EfficientNet 중 99.09% 정확도로 분류되었습니다.",
            fhir_bundle=fhir_bundle
        )
        print(f"✅ 진단 세션 저장됨. Session ID: {session_id}")
        
        # ====================================
        # Step 7: 결과 페이지에 데이터 전달
        # ====================================
        result_data = {
            'class': ai_reading['predicted_class'],
            'confidence': f"{ai_reading['confidence']*100:.2f}%",
            'accuracy': f"{ai_reading['accuracy']*100:.2f}%",
            'redness_score': f"{pixel_metrics['redness_score']:.2f}",
            'vessel_density': f"{pixel_metrics['vessel_density']:.2f}",
            'session_id': session_id,
            'user_id': user_id
        }
        
        return render_template('result.html', data=result_data)
        
    except Exception as e:
        print(f"❌ /result 처리 중 오류: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """기록실 페이지 테스트 (history.html)"""
    try:
        return render_template('history.html')
    except Exception as e:
        print(f"❌ /history 처리 중 오류: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """헬스체크 엔드포인트"""
    return jsonify({'status': 'ok', 'db_path': DB_PATH}), 200


# ============================================================================
# 5. 메인 실행
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("✅ 하준 팀장님, 웹 서버 테스트를 시작합니다.")
    print("=" * 70)
    print(f"👉 접속 주소: http://127.0.0.1:5002")
    print(f"📁 데이터베이스 경로: {DB_PATH}")
    print("=" * 70)
    
    # DB 초기화
    init_app_db()
    
    # Flask 서버 시작 (macOS 환경, 포트 5002)
    app.run(host='0.0.0.0', port=5002, debug=True)