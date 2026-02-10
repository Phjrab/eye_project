"""
[관리자] 모델 로더 - 싱글톤 패턴
GPU 메모리에 미리 로드하여 중복 로드 방지
"""

import torch
from modules import EyeDetector, DiseaseClassifier, EyeAnalyzer
from utils.logger import ResultLogger
import config


class ModelManager:
    """
    싱글톤 패턴으로 모델을 한 번만 로드하여 전역에서 공유
    Jetson의 제한된 자원을 효율적으로 활용
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """
        싱글톤 인스턴스 생성
        """
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        모델 초기화 (한 번만 실행)
        """
        if ModelManager._initialized:
            return
        
        print("[ModelManager] 모델 로딩 시작...")
        
        try:
            # ========================================
            # [1] YOLO 눈 검출 모델 로드
            # ========================================
            print("  - YOLO eye detector 로딩...")
            self.detector = EyeDetector(config.YOLO_MODEL_PATH)
            print("    ✓ YOLO 로드 완료")
            
            # ========================================
            # [2] EfficientNet 질환 분류 모델 로드
            # ========================================
            print("  - EfficientNet classifier 로딩...")
            self.classifier = DiseaseClassifier(config.CLASSIFIER_MODEL_PATH)
            print("    ✓ EfficientNet 로드 완료")
            
            # ========================================
            # [3] 눈 분석기 초기화
            # ========================================
            print("  - Eye analyzer 초기화...")
            self.analyzer = EyeAnalyzer()
            print("    ✓ Analyzer 초기화 완료")
            
            # ========================================
            # [4] 결과 로거 초기화
            # ========================================
            print("  - Result logger 초기화...")
            self.logger = ResultLogger(config.LOG_FORMAT)
            print("    ✓ Logger 초기화 완료")
            
            # ========================================
            # [5] GPU 메모리 정보 출력
            # ========================================
            if torch.cuda.is_available():
                print(f"\n[GPU 정보]")
                print(f"  - 장치: {torch.cuda.get_device_name(0)}")
                print(f"  - 총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                print(f"  - 할당된 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
            print("\n[ModelManager] 모든 모델 로드 완료!\n")
            ModelManager._initialized = True
            
        except Exception as e:
            print(f"[ERROR] 모델 로딩 실패: {e}")
            raise
    
    def get_detector(self):
        """
        YOLO 검출기 반환
        """
        return self.detector
    
    def get_classifier(self):
        """
        분류기 반환
        """
        return self.classifier
    
    def get_analyzer(self):
        """
        분석기 반환
        """
        return self.analyzer
    
    def get_logger(self):
        """
        로거 반환
        """
        return self.logger


# ========================================
# [6] 전역 싱글톤 인스턴스
# ========================================
model_manager = None


def initialize_models():
    """
    모델 매니저 초기화 (서버 시작 시 호출)
    """
    global model_manager
    model_manager = ModelManager()
    return model_manager


def get_models():
    """
    초기화된 모델 매니저 반환
    """
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager
