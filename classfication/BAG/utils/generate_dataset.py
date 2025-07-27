import openai
import pandas as pd
import json
from tqdm import tqdm
import time
from datetime import datetime
import random
import os
import sys

# OpenAI API 키 설정
openai.api_key = "YOUR-API-KEY"

def get_random_keywords(keywords, num_keywords=10):
    """랜덤하게 키워드를 선택"""
    return random.sample(keywords, min(num_keywords, len(keywords)))

def generate_gambling_sentences(keywords, start_id, num_sentences=60000, batch_size=20):
    start_time = time.time()
    success_count = 0
    error_count = 0
    current_id = start_id
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터 생성 시작")
    print(f"시작 ID: {start_id}")
    print(f"목표 문장 수: {num_sentences}")
    print(f"배치 크기: {batch_size}")
    print("="*50)
    
    # 기존 데이터셋 읽기
    if os.path.exists('gambling_dataset.csv'):
        existing_df = pd.read_csv('gambling_dataset.csv')
        existing_sentences = existing_df['content'].astype(str).tolist()
        print(f"기존 데이터셋에서 {len(existing_sentences)}개의 문장을 로드했습니다.")
    else:
        existing_sentences = []
        # CSV 파일이 없으면 헤더만 있는 파일 생성
        pd.DataFrame(columns=['id', 'label', 'content']).to_csv('gambling_dataset.csv', index=False, encoding='utf-8')
    
    # 배치 단위로 처리
    for batch_start in tqdm(range(0, num_sentences, batch_size), desc="배치 처리 진행률"):
        batch_end = min(batch_start + batch_size, num_sentences)
        batch_size_actual = batch_end - batch_start
        
        # 이전에 생성된 문장들 중 일부를 샘플링
        previous_samples = []
        if existing_sentences:
            sample_size = min(5, len(existing_sentences))
            previous_samples = random.sample(existing_sentences, sample_size)
        
        # 랜덤하게 키워드 선택
        selected_keywords = get_random_keywords(keywords)
        
        prompt = f"""
        당신은 불법도박사이트를 분류하기 위한 BERT 모델 훈련용 문장을 생성하는 전문가입니다.
        
        다음은 이전에 생성된 문장들의 예시입니다:
        {chr(10).join(previous_samples)}
        
        위 문장들과 다른 새로운 문장 {batch_size_actual}개를 생성해주세요.
        다음 키워드들 중 일부를 참고하여 자연스러운 문장을 만들어주세요:
        {', '.join(selected_keywords)}
        
        콤프,롤링,첫충,페이백,신규,최대,매충 같은 단어에 %를 추가한 문장을 꼭 생성해주세요
        존댓말을 적을 필요는 없으며, 문장을 꼭 완성시킬 필요는 없습니다. 다음 예시를 참고해주세요
        
        예시:
        페이백 최대 50%!
        첫충 15% 매충 7% 최대 100만원
        신규 첫충 이벤트 콤프 최대 7%
        사다리,바카라,스포츠 토토 실시간 경마까지!
        레볼루션 홀덤, 텔레그램, 가상스포츠, 무제한 콤프 지원

        주의사항:
        1. 이전 문장들과 중복되지 않는 새로운 문장을 생성해주세요.
        2. 각 문장은 새로운 줄에 작성해주세요.
        3. 문장은 실제 불법도박사이트에서 사용될 수 있는 자연스러운 문장이어야 합니다.
        4. 문장의 길이는 10-50자 정도로 작성해주세요.
        5. 문장만 생성해주세요
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 불법도박사이트 관련 문장을 생성하는 전문가입니다. 항상 새로운 문장을 생성하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # 다양성 증가
                max_tokens=1000
            )
            
            # GPT 응답 출력
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPT 응답:", flush=True)
            print(response.choices[0].message.content.strip(), flush=True)
            print("-"*50, flush=True)
            sys.stdout.flush()
            
            # 응답에서 문장들을 분리
            batch_sentences = response.choices[0].message.content.strip().split('\n')
            # 빈 문장 제거
            batch_sentences = [s.strip() for s in batch_sentences if s.strip()]
            # 중복 제거
            batch_sentences = list(set(batch_sentences) - set(existing_sentences))
            
            # 생성된 문장들을 바로 CSV 파일에 추가
            if batch_sentences:
                batch_df = pd.DataFrame({
                    'id': range(current_id, current_id + len(batch_sentences)),
                    'label': [1] * len(batch_sentences),
                    'content': batch_sentences
                })
                batch_df.to_csv('gambling_dataset.csv', mode='a', header=False, index=False, encoding='utf-8')
                current_id += len(batch_sentences)
                success_count += len(batch_sentences)
                existing_sentences.extend(batch_sentences)
            
            # 생성된 문장 출력
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 배치 {batch_start//batch_size + 1} 생성된 문장:", flush=True)
            for i, sentence in enumerate(batch_sentences, 1):
                print(f"{i}. {sentence}", flush=True)
            print("-"*50, flush=True)
            sys.stdout.flush()
            
            if (batch_start + batch_size) % 1000 == 0:  # 1000개 문장마다 진행상황 출력
                current_time = time.time()
                elapsed_time = current_time - start_time
                avg_time_per_batch = elapsed_time/(success_count/batch_size) if success_count > 0 else 0
                remaining_batches = (num_sentences - success_count) / batch_size
                estimated_time = avg_time_per_batch * remaining_batches if success_count > 0 else 0
                
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 진행상황:")
                print(f"- 생성된 문장 수: {success_count}")
                print(f"- 실패한 요청 수: {error_count}")
                print(f"- 경과 시간: {elapsed_time:.2f}초")
                print(f"- 예상 남은 시간: {estimated_time:.2f}초")
                print("-"*50)
            
            # API 요청 제한을 피하기 위한 짧은 대기
            time.sleep(1)
            
        except Exception as e:
            error_count += 1
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 오류 발생: {str(e)}")
            time.sleep(5)  # 오류 발생 시 더 긴 대기 시간
            continue
            
    total_time = time.time() - start_time
    avg_time_per_batch = total_time/(success_count/batch_size) if success_count > 0 else 0
    
    print("\n" + "="*50)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터 생성 완료")
    print(f"- 총 소요 시간: {total_time:.2f}초")
    print(f"- 성공한 요청 수: {success_count}")
    print(f"- 실패한 요청 수: {error_count}")
    print(f"- 평균 생성 시간: {avg_time_per_batch:.2f}초/배치")
    print("="*50)

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 프로그램 시작")
    
    # 키워드 파일 읽기
    print("키워드 파일 읽는 중...")
    with open('search_wordlist.txt', 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f.readlines()]
    print(f"총 {len(keywords)}개의 키워드를 로드했습니다.")
    
    # 시작 ID 설정
    start_id = 79055
    
    # 문장 생성 및 실시간 저장
    generate_gambling_sentences(keywords, start_id)

if __name__ == "__main__":
    main()