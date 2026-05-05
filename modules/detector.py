"""
[1단계] YOLO 눈 검출 및 크롭
이미지에서 눈을 검출하고 분류기 입력용 영역을 크롭
"""

import cv2
import numpy as np
from ultralytics import YOLO
import config as config


class EyeDetector:
    """
    YOLOv8을 이용한 눈 검출기
    """
    
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        """
        YOLO 검출기 초기화
        
        Args:
            model_path (str): YOLOv8 모델 가중치 경로
        """
        self.model = YOLO(model_path)
        self.conf_threshold = config.YOLO_CONF_THRESHOLD
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        
    def detect(self, image, conf_threshold=None):
        """
        이미지에서 눈 검출
        
        Args:
            image (np.ndarray): 입력 이미지 (BGR)
            conf_threshold (float, optional): 추론 시 신뢰도 임계값 오버라이드
            
        Returns:
            결과: 검출된 박스와 신뢰도를 포함한 YOLO 결과 객체
        """
        threshold = self.conf_threshold if conf_threshold is None else conf_threshold

        results = self.model.predict(
            image,
            conf=threshold,
            iou=self.iou_threshold,
            imgsz=config.YOLO_INPUT_SIZE,
            verbose=False
        )
        return results[0] if results else None
    
    def crop_eyes(self, image, detections):
        """
        검출된 눈 영역을 이미지에서 크롭
        
        Args:
            image (np.ndarray): 원본 이미지
            detections: YOLO 검출 결과
            
        Returns:
            list: 크롭된 눈 이미지 리스트 (좌표, 신뢰도 포함)
        """
        eye_crops = []
        
        if detections is None or len(detections.boxes) == 0:
            return eye_crops
        
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 패딩 추가 (컨텍스트 정보 확보)
            h, w = image.shape[:2]
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            # 고정 10px 대신 박스 크기 비율 기반 여유 패딩 적용
            pad_x = int(max(12, min(72, box_w * 0.24)))
            pad_y = int(max(10, min(64, box_h * 0.30)))

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            crop = image[y1:y2, x1:x2]
            eye_crops.append({
                'image': crop,
                'bbox': (x1, y1, x2, y2),
                'confidence': float(box.conf)
            })
        
        return eye_crops
