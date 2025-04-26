import pandas as pd
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
from tqdm import tqdm  # ✅ 진행률 표시 라이브러리

# 설정
file_path = "원본데이터/20250402_SKKU_gambling.csv"
model_path = "최종_데이터셋_모델/link_classifier_best"
base_model = "beomi/KcELECTRA-base"
batch_size = 32

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
df.to_csv("20250402_SKKU_gambling.csv", index=False)
print("✔ 배치 처리로 링크 탐지 완료 및 저장됨.")
