import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re
import html
import math

class GamblingClassifier:
    def __init__(self, model_path="./gambling_bert_model"):
        # 1. 모델과 토크나이저 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")  # 한국어 BERT 토크나이저
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # 평가 모드로 설정

    def clean_text(self, text):
        """텍스트 정규화 함수"""
        # HTML 태그 제거
        text = html.unescape(text)  # HTML 엔티티 디코딩
        text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
        
        # 특수문자 및 불필요한 공백 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def predict_text(self, text):
        # 2. 텍스트 전처리
        text = self.clean_text(text)
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,  # 학습 시와 동일한 길이
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # 3. 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = predictions.max().item()
            predicted_label = predictions.argmax().item()
            
            # 불법도박 클래스(label 1)에 대한 점수 추출
            gambling_score_orig = predictions[0, 1].item()
            
            # 점수 변환 (매우 작은 값 확대)
            epsilon = 1e-10  # 로그 변환 시 0 방지를 위한 작은 값
            
            # 방법 1: 로그 스케일 변환 (0~1 범위로 정규화)
            if gambling_score_orig > 0:
                # 음수 로그 계산 (값이 작을수록 큰 양수가 됨)
                log_value = -math.log10(gambling_score_orig + epsilon)
                # 로그 값을 0~1 범위로 매핑 (최대 로그값은 10으로 가정)
                # 0에 가까운 값(로그값이 큰)은 1에 가깝게, 1에 가까운 값(로그값이 작은)은 0에 가깝게
                max_log = 10.0  # 1e-10의 -로그값은 10
                gambling_score_log = min(log_value / max_log, 1.0)
            else:
                gambling_score_log = 0.0
                
            # 방법 2: 지수 변환 (이미 0~1 범위)
            gambling_score_exp = math.pow(gambling_score_orig, 0.3) if gambling_score_orig > 0 else 0.0
            
            # 방법 3: 시그모이드 변환 (이미 0~1 범위)
            temperature = 0.01
            if gambling_score_orig > 0:
                gambling_score_sigmoid = 1.0 / (1.0 + math.exp(-gambling_score_orig/temperature))
            else:
                gambling_score_sigmoid = 0.0
                
            # 방법 4: 분위 매핑 (값의 크기 순위에 따라 균등하게 0~1 분포)
            # 미리 정의된 분위에 따라 매핑 (매우 작은 값들을 구분)
            thresholds = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
            for i, threshold in enumerate(thresholds[1:], 1):
                if gambling_score_orig <= threshold:
                    # 해당 구간에 비례하여 점수 할당
                    prev = thresholds[i-1]
                    norm_value = (i-1) / (len(thresholds)-1)  # 기본 구간 점수
                    if prev < gambling_score_orig:  # 구간 내 위치에 따른 추가 점수
                        norm_value += ((gambling_score_orig - prev) / (threshold - prev)) / (len(thresholds)-1)
                    gambling_score_quantile = norm_value
                    break
            else:
                gambling_score_quantile = 1.0

        # 4. 결과 해석
        result = {
            "text": text,
            "prediction": "불법도박" if predicted_label == 1 else "정상",
            "confidence": confidence,
            "label": predicted_label,
            "gambling_score_orig": gambling_score_orig,  # 원본 불법도박 점수
            "gambling_score_log": gambling_score_log,  # 로그 스케일 점수 (0~1)
            "gambling_score_exp": gambling_score_exp,  # 지수 변환 점수 (0~1)
            "gambling_score_sigmoid": gambling_score_sigmoid,  # 시그모이드 변환 점수 (0~1)
            "gambling_score_quantile": gambling_score_quantile,  # 분위 기반 점수 (0~1)
            # 기본 사용 점수 (quantile이 가장 균등한 분포를 보임)
            "gambling_score": gambling_score_quantile
        }
        
        return result

def main():
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
    
    # 1. 분류기 초기화
    classifier = GamblingClassifier()
    print("\n모델 로드 완료! 텍스트를 입력해주세요 (종료하려면 'q' 입력)")
    
    while True:
        # 2. 사용자 입력 받기
        text = input("\n텍스트를 입력하세요: ")
        
        if text.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
            
        # 3. 예측 수행
        result = classifier.predict_text(text)
        
        # 4. 결과 출력
        print("\n=== 예측 결과 ===")
        print(f"입력 텍스트: {result['text']}")
        print(f"분류 결과: {result['prediction']}")
        print(f"신뢰도: {result['confidence']:.2%}")
        
        # 불법도박 점수 출력
        print(f"불법도박 원본 점수: {result['gambling_score_orig']:.8e}")
        
        # 모든 변환 점수는 0~1 범위
        if result['gambling_score_orig'] > 0:
            print("\n=== 0~1 범위로 정규화된 점수 ===")
            print(f"로그 변환 점수: {result['gambling_score_log']:.4f}")
            print(f"지수 변환 점수: {result['gambling_score_exp']:.4f}")
            print(f"시그모이드 점수: {result['gambling_score_sigmoid']:.4f}")
            print(f"분위 매핑 점수: {result['gambling_score_quantile']:.4f} (기본 사용)")
        
        print("================\n")

if __name__ == "__main__":
    main() 