import os
import csv

def create_label_csv():
    """
    1~70까지의 파일명과 레이블을 가진 CSV 파일을 생성합니다.
    1~35: label=1
    36~70: label=0
    """
    # CSV 파일 저장 경로
    csv_path = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset/test/result.csv"
    
    # CSV 파일 생성
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 헤더 작성
        writer.writerow(['file', 'label'])
        
        # 1부터 35까지의 파일명과 1로 설정된 레이블 작성
        for i in range(1, 36):
            filename = f"{i}.json"
            writer.writerow([filename, 1])
            
        # 36부터 70까지의 파일명과 0으로 설정된 레이블 작성
        for i in range(36, 71):
            filename = f"{i}.json"
            writer.writerow([filename, 0])
    
    print(f"레이블 CSV 파일이 생성되었습니다: {csv_path}")

if __name__ == "__main__":
    create_label_csv() 