import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# TensorFlow 버전 확인
print(f"TensorFlow 버전: {tf.__version__}")

def preprocess_image(image_path):
    """
    이미지를 전처리하는 함수
    """
    img = load_img(image_path, target_size=(600, 600))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, image_path):
    """
    이미지를 예측하는 함수
    """
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image, verbose=0)
    pred_class = 'illegal' if prediction[0][0] > 0.5 else 'normal'
    confidence = float(prediction[0][0]) if pred_class == 'illegal' else float(1 - prediction[0][0])
    return {
        'class': pred_class,
        'confidence': confidence * 100
    }

def main():
    # 모델 경로 설정
    MODEL_PATH = 'model.h5'
    
    # 모델 로드
    print("모델을 로드하는 중...")
    try:
        # SavedModel 형식으로 시도
        if os.path.exists(MODEL_PATH.replace('.h5', '')):
            model = tf.keras.models.load_model(MODEL_PATH.replace('.h5', ''), compile=False)
        else:
            # H5 형식으로 시도
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        print("\n다음 방법을 시도해보세요:")
        print("1. 모델을 다시 저장할 때 다음 코드를 사용:")
        print("   model.save('model', save_format='tf')  # SavedModel 형식으로 저장")
        print("2. 또는 이전 버전의 TensorFlow를 설치:")
        print("   pip install tensorflow==2.4.0")
        return
    
    while True:
        # 이미지 경로 입력 받기
        image_path = input("\n분석할 이미지 경로를 입력하세요 (종료하려면 'q' 입력): ")
        
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            print("파일이 존재하지 않습니다. 다시 시도해주세요.")
            continue
            
        try:
            # 이미지 예측 수행
            result = predict_image(model, image_path)
            
            # 결과 출력
            print("\n분석 결과:")
            print(f"분류: {result['class']}")
            print(f"신뢰도: {result['confidence']:.2f}%")
            
        except Exception as e:
            print(f"오류가 발생했습니다: {str(e)}")

if __name__ == '__main__':
    main()
