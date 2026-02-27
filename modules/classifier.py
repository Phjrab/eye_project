"""
[2단계] EfficientNet을 이용한 질환 분류
검출된 눈 영역을 질환 카테고리로 분류 (정확도: 99.09%)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import config


class DiseaseClassifier:
    """
    EfficientNet을 이용한 질환 분류기
    """
    
    def __init__(self, model_path=config.CLASSIFIER_MODEL_PATH, num_classes=5):
        """
        분류기 초기화
        
        Args:
            model_path (str): 모델 가중치 경로
            num_classes (int): 질환 클래스 개수 (5개: 정상, 결막염, 포도막염, 백내장, 다래끼)
        """
        self.device = config.DEVICE  # config.py에서 설정된 device 사용
        self.num_classes = num_classes
        self.input_size = config.CLASSIFIER_INPUT_SIZE
        self.confidence_threshold = config.CLASSIFIER_CONFIDENCE_THRESHOLD
        
        # ========================================
        # [1] EfficientNet-B0 모델 로드
        # ========================================
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        # ========================================
        # [2] 학습된 가중치 로드
        # ========================================
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # ========================================
        # [3] 정규화 파라미터
        # ========================================
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
    
    def preprocess(self, image):
        """
        분류 모델 입력을 위한 전처리
        
        Args:
            image (np.ndarray): 입력 이미지 (BGR)
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        # ========================================
        # [1] 이미지 리사이징 (224x224)
        # ========================================
        if isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image
        
        img = img.resize(self.input_size, Image.BILINEAR)
        
        # ========================================
        # [2] 텐서 변환 및 정규화
        # ========================================
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # ImageNet 표준 정규화
        img = (img - self.mean) / self.std
        return img.unsqueeze(0)
    
    def classify(self, image):
        """
        눈 이미지 분류
        
        Args:
            image (np.ndarray): 눈 크롭 이미지
            
        Returns:
            dict: 분류 결과 (질환 클래스, 확률 포함)
        """
        with torch.no_grad():
            input_tensor = self.preprocess(image).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()
        
        return {
            'class': predicted_class,
            'disease': config.DISEASE_CLASSES.get(predicted_class, '미확인'),
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().tolist()
        }
