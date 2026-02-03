import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import cv2  # OpenCV 모듈 추가

class EyeClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # EfficientNet-B0 모델 초기화 (5개 클래스)
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                raise ValueError(f"Failed to load model from {model_path}: {e}")
        else:
            raise ValueError("Model path must be provided.")

        self.model.to(self.device)
        self.model.eval()
        
        # 이미지 전처리 설정 (224x224)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_cv2):
        # OpenCV 이미지를 PIL로 변환 후 전처리
        if not isinstance(image_cv2, (np.ndarray, np.generic)):
            raise ValueError("Input image must be a valid OpenCV image (numpy array).")

        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            # Softmax를 사용하여 확률 계산
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            return predicted_class, probabilities.cpu().numpy()

            