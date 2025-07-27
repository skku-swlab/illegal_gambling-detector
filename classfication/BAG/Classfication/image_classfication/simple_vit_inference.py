import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# VIT 이미지 분류 모델 설정
VIT_MODEL_PATH = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/image_classfication/best_vit_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VIT 모델 로드
print('VIT 이미지 분류 모델 로드 중...')
try:
    vit_model = torch.load(VIT_MODEL_PATH, map_location=device)
    vit_model.eval()
    print('VIT 이미지 분류 모델 로드 완료')
except Exception as e:
    print(f"VIT 이미지 분류 모델 로드 중 오류 발생: {str(e)}")
    vit_model = None

# VIT 모델용 이미지 전처리
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image_vit(image_path):
    """
    VIT 모델용 이미지 전처리 함수
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = vit_transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
        return image_tensor
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {str(e)}")
        return None

def predict_image_vit(image_path):
    """
    VIT 모델을 사용한 이미지 예측 함수
    """
    if vit_model is None:
        return {'class': 'normal', 'confidence': 0.0}
        
    try:
        processed_image = preprocess_image_vit(image_path)
        if processed_image is None:
            return {'class': 'normal', 'confidence': 0.0}
            
        with torch.no_grad():
            processed_image = processed_image.to(device)
            outputs = vit_model(processed_image)
            
            # 소프트맥스 적용
            probabilities = torch.softmax(outputs, dim=1)
            
            # 불법도박 클래스 (index 1)의 확률
            illegal_prob = probabilities[0][1].item()
            
            pred_class = 'illegal' if illegal_prob > 0.5 else 'normal'
            confidence = illegal_prob * 100 if pred_class == 'illegal' else (1 - illegal_prob) * 100
            
            return {
                'class': pred_class,
                'confidence': confidence
            }
    except Exception as e:
        print(f"VIT 이미지 예측 중 오류 발생: {str(e)}")
        return {'class': 'normal', 'confidence': 0.0}

def main():
    """
    메인 함수 - 이미지 분류 수행
    """
    print("\n=== VIT 이미지 분류 시스템 ===")
    print(f"사용 중인 디바이스: {device}")
    
    # 테스트할 이미지 경로 (여기서 직접 설정)
    test_image_path = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/inference/screenshot/www_spochak_com_20250706_175434.png"
    
    # 이미지 파일 존재 확인
    if not os.path.exists(test_image_path):
        print(f"❌ 이미지 파일이 존재하지 않습니다: {test_image_path}")
        print("다른 이미지 경로를 입력하세요:")
        test_image_path = input("이미지 경로: ").strip()
        
        if not os.path.exists(test_image_path):
            print("❌ 입력한 이미지 파일도 존재하지 않습니다. 프로그램을 종료합니다.")
            return
    
    print(f"\n분석할 이미지: {test_image_path}")
    
    # VIT 이미지 분류 수행
    print("\n1. VIT 이미지 분류 수행...")
    image_result = predict_image_vit(test_image_path)
    
    # 결과 출력
    print("\n=== 분석 결과 ===")
    print(f"이미지 경로: {test_image_path}")
    print(f"분류 결과: {image_result['class']}")
    print(f"신뢰도: {image_result['confidence']:.2f}%")
    
    if image_result['class'] == 'illegal':
        print("🚫 불법도박 사이트로 판정되었습니다.")
    else:
        print("✅ 정상 사이트로 판정되었습니다.")
    
    print("\n=== 분석 완료 ===")

if __name__ == "__main__":
    main() 