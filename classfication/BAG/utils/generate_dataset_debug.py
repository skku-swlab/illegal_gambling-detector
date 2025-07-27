import openai
import pandas as pd
import json
from tqdm import tqdm
import time
from datetime import datetime
import random

# OpenAI API 키 설정
openai.api_key = "###"

def get_random_keywords(keywords, num_keywords=10):
    """랜덤하게 키워드를 선택"""
    return random.sample(keywords, min(num_keywords, len(keywords)))

def generate_gambling_sentences(keywords, num_sentences=5, batch_size=5):
    sentences = []
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터 생성 시작")
    print(f"목표 문장 수: {num_sentences}")
    print(f"배치 크기: {batch_size}")
    print("="*50)
    
    # 배치 단위로 처리
    for batch_start in tqdm(range(0, num_sentences, batch_size), desc="배치 처리 진행률"):
        batch_end = min(batch_start + batch_size, num_sentences)
        batch_size_actual = batch_end - batch_start
        
        # 이전에 생성된 문장들 중 일부를 샘플링
        previous_samples = []
        if sentences:
            sample_size = min(5, len(sentences))
            previous_samples = random.sample(sentences, sample_size)
        
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

        주의사항:
        1. 이전 문장들과 중복되지 않는 새로운 문장을 생성해주세요.
        2. 각 문장은 새로운 줄에 작성해주세요.
        3. 문장은 실제 불법도박사이트에서 사용될 수 있는 자연스러운 문장이어야 합니다.
        4. 문장의 길이는 10-50자 정도로 작성해주세요.
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
            
            # 응답에서 문장들을 분리
            batch_sentences = response.choices[0].message.content.strip().split('\n')
            # 중복 제거
            batch_sentences = list(set(batch_sentences) - set(sentences))
            sentences.extend(batch_sentences)
            success_count += len(batch_sentences)
            
            # 생성된 문장 출력
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 배치 {batch_start//batch_size + 1} 생성된 문장:")
            for i, sentence in enumerate(batch_sentences, 1):
                print(f"{i}. {sentence}")
            print("-"*50)
            
            # API 요청 제한을 피하기 위한 짧은 대기
            time.sleep(1)
            
        except Exception as e:
            error_count += 1
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 오류 발생: {str(e)}")
            time.sleep(5)  # 오류 발생 시 더 긴 대기 시간
            continue
            
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터 생성 완료")
    print(f"- 총 소요 시간: {total_time:.2f}초")
    print(f"- 성공한 요청 수: {success_count}")
    print(f"- 실패한 요청 수: {error_count}")
    print(f"- 평균 생성 시간: {total_time/(success_count/batch_size):.2f}초/배치")
    print("="*50)
            
    return sentences

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 프로그램 시작")
    
    # 키워드 파일 읽기
    print("키워드 파일 읽는 중...")
    with open('search_wordlist.txt', 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f.readlines()]
    print(f"총 {len(keywords)}개의 키워드를 로드했습니다.")
    
    # 문장 생성
    sentences = generate_gambling_sentences(keywords)
    
    # DataFrame 생성
    print("\nDataFrame 생성 중...")
    df = pd.DataFrame({
        'id': range(1, len(sentences) + 1),
        'label': [1] * len(sentences),
        'content': sentences
    })
    
    # CSV 파일로 저장
    print("CSV 파일 저장 중...")
    df.to_csv('gambling_dataset_debug.csv', index=False, encoding='utf-8')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터셋 저장 완료")
    print(f"- 파일명: gambling_dataset_debug.csv")
    print(f"- 총 데이터 수: {len(sentences)}개")

if __name__ == "__main__":
    main() 
