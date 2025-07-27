import os
import re
from bs4 import BeautifulSoup
import glob

def convert_txt_to_html(txt_file_path, html_file_path):
    """
    텍스트 파일을 HTML 파일로 변환합니다.
    
    Args:
        txt_file_path (str): 텍스트 파일 경로
        html_file_path (str): 변환된 HTML 파일 저장 경로
    """
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        
        # 텍스트 내용을 HTML 파일로 저장
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
            
        print(f"텍스트 파일이 성공적으로 HTML 파일로 변환되었습니다: {html_file_path}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

def process_all_files():
    """
    redirection_final 폴더의 모든 start.txt, end.txt 파일을 HTML로 변환합니다.
    """
    # 입력 폴더 경로
    input_base_path = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/redirection_final"
    
    # 출력 폴더 경로
    output_base_path = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/crawled_html_files/test"
    
    # 출력 폴더가 없으면 생성
    os.makedirs(output_base_path, exist_ok=True)
    
    # 모든 start.txt, end.txt 파일 찾기
    txt_files = []
    for root, dirs, files in os.walk(input_base_path):
        for file in files:
            if file in ['start.txt', 'end.txt']:
                txt_files.append(os.path.join(root, file))
    
    # 파일 번호 초기화
    file_number = 1
    
    # 각 파일을 HTML로 변환
    for txt_file in txt_files:
        # HTML 파일 경로 생성
        html_file = os.path.join(output_base_path, f"{file_number}.html")
        
        # 파일 변환
        convert_txt_to_html(txt_file, html_file)
        
        # 파일 번호 증가
        file_number += 1

if __name__ == "__main__":
    process_all_files() 