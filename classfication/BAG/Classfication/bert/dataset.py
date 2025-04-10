import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer
from collections import Counter
import re
import html

class GamblingDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=256):
        """
        불법 도박 문장 분류를 위한 데이터셋 클래스
        
        Args:
            csv_file (str): CSV 파일 경로
            tokenizer: BERT 토크나이저
            max_length (int): 최대 시퀀스 길이
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # HTML 태그 제거 및 텍스트 정규화
        self.data['content'] = self.data['content'].apply(self.clean_text)
        
        # 데이터 분포 확인
        label_counts = Counter(self.data['label'])
        print("\n=== 데이터 분포 ===")
        print(f"레이블 0 (정상 사이트) 수: {label_counts[0]}")
        print(f"레이블 1 (불법도박 사이트) 수: {label_counts[1]}")
        
        # 샘플 데이터 확인
        print("\n=== 샘플 데이터 확인 ===")
        print("정상 사이트 샘플:")
        print(self.data[self.data['label'] == 0]['content'].iloc[0][:200])
        print("\n불법도박 사이트 샘플:")
        print(self.data[self.data['label'] == 1]['content'].iloc[0][:200])
        print("================\n")
        
        # 데이터 샘플링 (불균형이 심한 경우)
        if label_counts[0] > label_counts[1] * 2 or label_counts[1] > label_counts[0] * 2:
            print("데이터 불균형이 감지되었습니다. 데이터를 균형있게 샘플링합니다.")
            min_count = min(label_counts[0], label_counts[1])
            self.data = pd.concat([
                self.data[self.data['label'] == 0].sample(min_count),
                self.data[self.data['label'] == 1].sample(min_count)
            ])
            print(f"샘플링 후 데이터 수: {len(self.data)}")

    def clean_text(self, text):
        """텍스트 정규화 함수"""
        # NaN 값 처리
        if pd.isna(text):
            return ""
            
        # HTML 태그 제거
        text = html.unescape(text)  # HTML 엔티티 디코딩
        text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
        
        # 특수문자 및 불필요한 공백 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 각 행에서 텍스트와 라벨 추출
        row = self.data.iloc[idx]
        text = row['content']
        # 텍스트가 NaN인 경우 빈 문자열로 처리
        if pd.isna(text):
            text = ""
        else:
            text = str(text)
        label = row['label']
        
        # 텍스트 토큰화
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        } 