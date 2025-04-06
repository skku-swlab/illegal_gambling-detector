import json
import csv
import os
from pathlib import Path

def extract_sentences_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    
    # named_entity 배열의 각 항목에서
    for entity in data['named_entity']:
        # title의 sentence 추출
        for title in entity['title']:
            sentences.append(title['sentence'])
        
        # content의 sentence 추출
        for content in entity['content']:
            sentences.append(content['sentence'])
    
    return sentences

def process_normal_data():
    # 정상 데이터 디렉토리 경로
    normal_data_dir = Path('/home/swlab/Desktop/gambling_crawling/dataset/bert/normal_data')
    
    # 결과를 저장할 리스트
    all_sentences = []
    
    # 모든 JSON 파일 처리
    for json_file in normal_data_dir.glob('*.json'):
        try:
            sentences = extract_sentences_from_json(json_file)
            all_sentences.extend(sentences)
            
            # 80,000개 이상의 문장이 수집되면 중단
            if len(all_sentences) >= 100000:
                all_sentences = all_sentences[:100000]
                break
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # CSV 파일로 저장
    output_file = normal_data_dir / 'normal_sentences.csv'
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label', 'content'])
        
        for idx, sentence in enumerate(all_sentences, 1):
            writer.writerow([idx, 0, sentence])
    
    print(f"처리 완료: 총 {len(all_sentences)}개의 문장이 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    process_normal_data() 