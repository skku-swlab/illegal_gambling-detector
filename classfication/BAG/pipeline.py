import psycopg2
from psycopg2 import pool
import queue
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import os
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sys
from s3_upload import upload_to_s3

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Classfication'))
sys.path.append(os.path.join(project_root, 'Classfication/gnn_models/gambling_gnn_models'))

from utils.html_to_json import process_html_file
from Classfication.gnn_models.gambling_gnn_models.inference import load_model as load_gat_model, load_data_from_json, inference
from Classfication.gnn_models.gambling_gnn_models.model import GamblingGATModel
import json

# 데이터베이스 연결 설정
DB_CONFIG = {
    "host": "gambling-crawling.cnig8owewqhg.ap-northeast-2.rds.amazonaws.com",
    "user": "postgres",
    "password": "codeep12345!",
    "port": 5432,
    "database": "gambling_db",
    "sslmode": "require"
}

# 커넥션 풀 생성
connection_pool = psycopg2.pool.SimpleConnectionPool(
    1,  # 최소 연결 수
    10, # 최대 연결 수
    **DB_CONFIG
)

# URL 큐 생성
url_queue = queue.Queue()

# 저장 디렉토리 생성
HTML_SAVE_DIR = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/inference/download_html"
SCREENSHOT_SAVE_DIR = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/inference/screenshot"
JSON_SAVE_DIR = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/inference/json_file"
os.makedirs(HTML_SAVE_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_SAVE_DIR, exist_ok=True)
os.makedirs(JSON_SAVE_DIR, exist_ok=True)

# GAT 모델 설정
GAT_MODEL_PATH = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/gnn_models/models/best_model/gambling_gat_binary_20250412_210319_final.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 하이퍼파라미터 설정
input_dim = 2  # 노드 특성 차원 (텍스트 길이, 노드 타입)
hidden_dim = 64
num_heads = 8
dropout = 0.3
gambling_weight = 100.0

# 이미지 분류 모델 설정
IMAGE_MODEL_PATH = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/image_classfication/model.h5'

# GAT 모델 로드
print('GAT 모델 로드 중...')
gat_model = load_gat_model(GAT_MODEL_PATH, input_dim, hidden_dim, num_heads, dropout, gambling_weight, device)
print('GAT 모델 로드 완료')

# 이미지 분류 모델 로드
print('이미지 분류 모델 로드 중...')
try:
    if os.path.exists(IMAGE_MODEL_PATH.replace('.h5', '')):
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH.replace('.h5', ''), compile=False)
    else:
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
    print('이미지 분류 모델 로드 완료')
except Exception as e:
    print(f"이미지 분류 모델 로드 중 오류 발생: {str(e)}")
    image_model = None

def preprocess_image(image_path):
    """
    이미지를 전처리하는 함수
    """
    img = load_img(image_path, target_size=(600, 600))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(image_path):
    """
    이미지를 예측하는 함수
    """
    if image_model is None:
        return {'class': 'normal', 'confidence': 0.0}
        
    try:
        processed_image = preprocess_image(image_path)
        prediction = image_model.predict(processed_image, verbose=0)
        pred_class = 'illegal' if prediction[0][0] > 0.5 else 'normal'
        confidence = float(prediction[0][0]) if pred_class == 'illegal' else float(1 - prediction[0][0])
        return {
            'class': pred_class,
            'confidence': confidence * 100
        }
    except Exception as e:
        print(f"이미지 예측 중 오류 발생: {str(e)}")
        return {'class': 'normal', 'confidence': 0.0}

def check_illegal_gambling(json_filepath, screenshot_path):
    """
    JSON 파일과 스크린샷을 분석하여 불법도박 사이트 여부를 확인합니다.
    BAG 프레임워크와 GAT 모델, 이미지 분류 모델을 사용합니다.
    
    Args:
        json_filepath (str): 분석할 JSON 파일 경로
        screenshot_path (str): 분석할 스크린샷 파일 경로
        
    Returns:
        tuple: (불법도박 사이트 여부, HTML 분석 결과, 이미지 분석 결과)
    """
    try:
        print(f"불법도박 사이트 검사 시작: {json_filepath}")
        
        # HTML 분석 (GAT 모델)
        data = load_data_from_json(json_filepath)
        html_prediction_score = inference(gat_model, data, device)
        is_illegal_html = html_prediction_score > 0.5
        print(f"HTML 분석 예측 점수: {html_prediction_score:.4f}")
        print(f"HTML 분석 결과: {'불법도박' if is_illegal_html else '정상'}")
        
        # 이미지 분석
        image_result = predict_image(screenshot_path)
        is_illegal_image = image_result['class'] == 'illegal'
        print(f"이미지 분석 결과: {image_result['class']} (신뢰도: {image_result['confidence']:.2f}%)")
        
        # OR 연산으로 최종 판단
        is_illegal = is_illegal_html or is_illegal_image
        print(f"최종 판정 결과: {'불법도박' if is_illegal else '정상'}")
        
        return is_illegal, {
            'html_score': html_prediction_score,
            'html_result': 'illegal' if is_illegal_html else 'normal'
        }, {
            'image_class': image_result['class'],
            'image_confidence': image_result['confidence']
        }
        
    except Exception as e:
        print(f"불법도박 사이트 검사 중 오류 발생: {str(e)}")
        return False, None, None

def update_illegal_status(url, is_illegal, html_result=None, image_result=None, s3_key=None):
    """
    URL의 불법도박 사이트 여부를 데이터베이스에 업데이트합니다.
    
    Args:
        url (str): 업데이트할 URL
        is_illegal (bool): 불법도박 사이트 여부
        html_result (dict): HTML 분석 결과
        image_result (dict): 이미지 분석 결과
        s3_key (str): S3에 업로드된 이미지의 객체 키
    """
    try:
        print(f"데이터베이스 업데이트 시작: {url}")
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        # HTML과 이미지 분석 결과를 boolean으로 변환
        html_is_illegal = html_result['html_result'] == 'illegal' if html_result else False
        image_is_illegal = image_result['image_class'] == 'illegal' if image_result else False
        
        update_query = """
            UPDATE crawled_sites 
            SET illegal = %s,
                html_result = %s,
                image_result = %s,
                image_s3 = %s
            WHERE url = %s
        """
        cursor.execute(update_query, (
            is_illegal,  # OR 연산 결과
            html_is_illegal,  # HTML 분석 결과
            image_is_illegal,  # 이미지 분석 결과
            s3_key,  # S3 객체 키
            url
        ))
        conn.commit()
        
        print(f"데이터베이스 업데이트 완료: {url}")
        print(f"- HTML 분석 결과: {'불법도박' if html_is_illegal else '정상'}")
        print(f"- 이미지 분석 결과: {'불법도박' if image_is_illegal else '정상'}")
        print(f"- 최종 판정: {'불법도박' if is_illegal else '정상'}")
        if s3_key:
            print(f"- S3 객체 키: {s3_key}")
        
    except Exception as e:
        print(f"데이터베이스 업데이트 실패: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            connection_pool.putconn(conn)

def setup_chrome_driver():
    """
    Chrome 웹드라이버를 설정합니다.
    """
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 헤드리스 모드
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def take_screenshot(url):
    """
    URL의 스크린샷을 찍습니다.
    
    Args:
        url (str): 스크린샷을 찍을 URL
        
    Returns:
        tuple: (성공 여부, 저장된 스크린샷 파일 경로)
    """
    try:
        print(f"스크린샷 촬영 시작: {url}")
        driver = setup_chrome_driver()
        print("Chrome 드라이버 설정 완료")
        
        print("페이지 로딩 중...")
        driver.get(url)
        time.sleep(3)  # 페이지 로딩 대기
        
        # 알림창 처리
        try:
            alert = driver.switch_to.alert
            print(f"알림창 감지: {alert.text}")
            alert.accept()  # 알림창 확인
            time.sleep(1)  # 알림창 처리 후 잠시 대기
        except:
            pass  # 알림창이 없으면 무시
        
        # 파일명 생성 (URL의 도메인 + 타임스탬프)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = url.split("//")[-1].split("/")[0].replace(".", "_")
        filename = f"{domain}_{timestamp}.png"
        filepath = os.path.join(SCREENSHOT_SAVE_DIR, filename)
        
        print(f"스크린샷 저장 중: {filepath}")
        # 스크린샷 저장
        driver.save_screenshot(filepath)
        print(f"스크린샷 저장 완료: {url} -> {filepath}")
        return True, filepath
        
    except Exception as e:
        print(f"스크린샷 저장 실패 ({url}): {str(e)}")
        return False, None
        
    finally:
        print("Chrome 드라이버 종료")
        driver.quit()

def create_session():
    """
    재시도 로직이 포함된 requests 세션을 생성합니다.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # 최대 3번 재시도
        backoff_factor=1,  # 재시도 간격
        status_forcelist=[500, 502, 503, 504]  # 재시도할 HTTP 상태 코드
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_html(url):
    """
    URL에서 HTML을 다운로드하고 저장합니다.
    
    Args:
        url (str): 다운로드할 URL
        
    Returns:
        tuple: (성공 여부, 저장된 HTML 파일 경로)
    """
    try:
        print(f"HTML 다운로드 시작: {url}")
        session = create_session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print("웹페이지 요청 중...")
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # HTTP 오류 체크
        
        # 파일명 생성 (URL의 도메인 + 타임스탬프)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = url.split("//")[-1].split("/")[0].replace(".", "_")
        filename = f"{domain}_{timestamp}.html"
        filepath = os.path.join(HTML_SAVE_DIR, filename)
        
        print(f"HTML 파일 저장 중: {filepath}")
        # HTML 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        print(f"HTML 다운로드 완료: {url} -> {filepath}")
        return True, filepath
        
    except requests.exceptions.RequestException as e:
        print(f"HTML 다운로드 실패 ({url}): {str(e)}")
        return False, None
    except Exception as e:
        print(f"예상치 못한 오류 발생 ({url}): {str(e)}")
        return False, None
    finally:
        session.close()

def get_unchecked_urls(batch_size=5):
    """
    데이터베이스에서 처리되지 않은 URL을 batch_size만큼 가져오고,
    가져온 URL의 checked 상태를 true로 변경합니다.
    """
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        # 트랜잭션 시작
        cursor.execute("BEGIN")
        
        # 처리되지 않은 URL을 가져오고 바로 checked를 true로 변경
        query = """
            WITH selected_urls AS (
                SELECT url 
                FROM crawled_sites 
                WHERE checked = false 
                LIMIT %s
                FOR UPDATE
            )
            UPDATE crawled_sites 
            SET checked = true
            WHERE url IN (SELECT url FROM selected_urls)
            RETURNING url;
        """
        cursor.execute(query, (batch_size,))
        urls = [row[0] for row in cursor.fetchall()]
        
        # 트랜잭션 커밋
        conn.commit()
        return urls
        
    except Exception as e:
        print(f"데이터베이스 조회 중 오류 발생: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return []
        
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            connection_pool.putconn(conn)

def process_urls():
    """
    큐에서 URL을 가져와서 처리하는 함수
    """
    while True:
        try:
            url = url_queue.get(timeout=5)  # 5초 동안 큐에서 URL을 기다림
            print(f"\n=== URL 처리 시작: {url} ===")
            
            # HTML 다운로드
            print("1. HTML 다운로드 시작...")
            success, html_filepath = download_html(url)
            
            if success and html_filepath:
                print("2. HTML 다운로드 성공, JSON 변환 시작...")
                # HTML을 JSON으로 변환
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                domain = url.split("//")[-1].split("/")[0].replace(".", "_")
                json_filename = f"{domain}_{timestamp}.json"
                json_filepath = os.path.join(JSON_SAVE_DIR, json_filename)
                
                print(f"3. HTML을 JSON으로 변환 중: {html_filepath} -> {json_filepath}")
                process_html_file(html_filepath, json_filepath)
            
            # 스크린샷 촬영
            print("4. 스크린샷 촬영 시작...")
            screenshot_success, screenshot_path = take_screenshot(url)
            
            if success and html_filepath and screenshot_success and screenshot_path:
                # S3에 이미지 업로드
                print("5. S3 이미지 업로드 시작...")
                s3_key = upload_to_s3(screenshot_path)
                if s3_key:
                    print(f"S3 이미지 업로드 성공: {s3_key}")
                else:
                    print("S3 이미지 업로드 실패")
                
                # 불법도박 사이트 검사
                print("6. 불법도박 사이트 검사 시작...")
                is_illegal, html_result, image_result = check_illegal_gambling(json_filepath, screenshot_path)
                print(f"불법도박 사이트 검사 결과: {is_illegal}")
                
                # 데이터베이스 업데이트
                print("7. 데이터베이스 업데이트 시작...")
                update_illegal_status(url, is_illegal, html_result, image_result, s3_key)
            
            print(f"=== URL 처리 완료: {url} ===\n")
            url_queue.task_done()
            time.sleep(1)  # 서버 부하 방지를 위한 딜레이
            
        except queue.Empty:
            print("현재 배치의 모든 URL이 처리되었습니다.")
            break
        except Exception as e:
            print(f"URL 처리 중 오류 발생: {str(e)}")
            url_queue.task_done()

def main():
    print("\n=== 프로그램 시작 ===")
    print(f"HTML 저장 디렉토리: {HTML_SAVE_DIR}")
    print(f"스크린샷 저장 디렉토리: {SCREENSHOT_SAVE_DIR}")
    print(f"JSON 저장 디렉토리: {JSON_SAVE_DIR}")
    
    while True:
        print("\n=== 새로운 배치 시작 ===")
        # 5개씩 URL 가져오기
        urls = get_unchecked_urls(5)
        
        if not urls:
            print("더 이상 처리할 URL이 없습니다.")
            break
            
        print(f"새로운 배치 시작: {len(urls)}개의 URL을 가져왔습니다.")
        print("가져온 URL 목록:")
        for i, url in enumerate(urls, 1):
            print(f"{i}. {url}")
        
        # URL을 큐에 추가
        for url in urls:
            url_queue.put(url)
        
        # URL 처리 시작
        process_urls()
        
        print("=== 현재 배치 처리 완료 ===\n")

if __name__ == "__main__":
    main()
