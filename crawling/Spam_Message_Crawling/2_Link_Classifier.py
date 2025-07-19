from config import GAMBLING_MSG_PATH, GAMBLING_CLASSIFIED_PATH
import pandas as pd
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
from tqdm import tqdm  # ✅ 진행률 표시 라이브러리
import os

# 설정
file_path = GAMBLING_MSG_PATH
base_model = "beomi/KcELECTRA-base"
batch_size = 1024  # 배치 크기 설정

# 상위 디렉토리 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "최종_데이터셋_모델", "link_classifier_best")

# 모델이 로컬에 있는지 확인
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path}")

# 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = ElectraForSequenceClassification.from_pretrained(model_path)
model.eval()

# GPU 사용 가능하면 활용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터 불러오기
df = pd.read_csv(file_path)
texts = df["message"].tolist()
results = []

# tqdm으로 진행률 출력
for i in tqdm(range(0, len(texts), batch_size), desc="링크 분류 중..."):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        results.extend(predictions)

# 결과 반영 및 저장
df["link_included"] = results
df.to_csv(GAMBLING_CLASSIFIED_PATH, index=False)
print("✔ 배치 처리로 링크 탐지 완료 및 저장됨.")
