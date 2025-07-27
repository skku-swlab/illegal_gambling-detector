import os
import re
from bs4 import BeautifulSoup

def html_to_txt(html_file_path, txt_file_path):
    """
    HTML 파일을 텍스트 파일로 변환합니다.
    
    Args:
        html_file_path (str): HTML 파일 경로
        txt_file_path (str): 변환된 텍스트 파일 저장 경로
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # HTML 내용을 텍스트 파일로 저장
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"HTML 파일이 성공적으로 텍스트 파일로 변환되었습니다: {txt_file_path}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

def txt_to_html(txt_file_path, html_file_path):
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

def main():
    # 예시 사용법
    html_file = "example.html"
    txt_file = "example.txt"
    new_html_file = "converted_example.html"
    
    # HTML을 텍스트로 변환
    html_to_txt(html_file, txt_file)
    
    # 텍스트를 HTML로 변환
    txt_to_html(txt_file, new_html_file)

if __name__ == "__main__":
    main() 