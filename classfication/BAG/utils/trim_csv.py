import pandas as pd

def remove_illegal_word(csv_path, output_path=None):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # 모든 문자열 컬럼에 대해 "불법" 단어 제거
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype(str).str.replace("불법", "")
    
    # 결과 저장
    if output_path is None:
        # 입력 파일명에 "_cleaned" 접미사 추가
        output_path = csv_path.replace(".csv", "_cleaned.csv")
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"처리 완료: {csv_path} -> {output_path}")

if __name__ == "__main__":
    # 여기에 입력 및 출력 파일 경로를 직접 지정합니다
    input_csv = "/home/swlab/Desktop/gambling_crawling/dataset/merged_data.csv"  # 이 부분을 실제 CSV 파일 경로로 수정하세요
    output_csv = "/home/swlab/Desktop/gambling_crawling/dataset/merged_data_cleaned.csv"  # 출력 파일 경로 (없으면 입력파일명_cleaned.csv로 저장됨)
    
    remove_illegal_word(input_csv, output_csv)
