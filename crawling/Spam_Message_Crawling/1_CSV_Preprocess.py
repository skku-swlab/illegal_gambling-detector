import os
import re
import pandas as pd

# 1. 경로 설정
file_path = "원본데이터/20250402_SKKU.csv"
base_name, ext = os.path.splitext(file_path)
save_path = f"{base_name}_gambling.csv"

# 2. 원본 데이터 읽기
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 3. '불법 도박' 포함된 전체 블록 추출
blocks = re.findall(r"\[\(KISA:SOL\)].*?\[\(KISA\)]불법 도박.*?\[\(KISA:EOL\)]", text, flags=re.DOTALL)

# 4. 마지막 2개의 [(KISA)] 사이의 메시지를 추출
messages = []
for block in blocks:
    parts = list(re.finditer(r'\[\(KISA\)\]', block))
    if len(parts) >= 2:
        start = parts[-2].end()  # 마지막에서 두 번째 [(KISA)] 뒤
        end = parts[-1].start()  # 마지막 [(KISA)] 앞
        message = block[start:end].strip()
        messages.append(message)

# 5. CSV로 저장
df = pd.DataFrame({'message': messages})
df.to_csv(save_path, index=False)

save_path
