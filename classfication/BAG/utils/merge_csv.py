import pandas as pd
import os

def merge_csv_files():
    # 데이터셋 디렉토리 경로
    dataset_dir = 'dataset'
    
    # CSV 파일 경로
    file1_path = os.path.join(dataset_dir, '/home/swlab/Desktop/gambling_crawling/dataset/bert/gambling_dataset.csv')
    file2_path = os.path.join(dataset_dir, '/home/swlab/Desktop/gambling_crawling/dataset/bert/normal_sentences.csv')
    
    # CSV 파일 읽기
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # 데이터프레임 합치기
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # 중복 제거 (필요한 경우)
    merged_df = merged_df.drop_duplicates()
    
    # ID 컬럼을 1부터 순차적으로 재설정
    merged_df['id'] = range(1, len(merged_df) + 1)
    
    # 결과 저장
    output_path = os.path.join(dataset_dir, 'merged_data.csv')
    merged_df.to_csv(output_path, index=False)
    
    print(f'파일이 성공적으로 합쳐졌습니다.')
    print(f'저장된 파일: {output_path}')
    print(f'총 행 수: {len(merged_df)}')
    print(f'ID 범위: 1 ~ {len(merged_df)}')

if __name__ == '__main__':
    merge_csv_files() 