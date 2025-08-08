from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import requests
import uuid
import json
import time
from threading import Thread, Event
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests
import re
import csv
from datetime import datetime
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import init, Fore, Back, Style
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# 컬러 출력 초기화
init(autoreset=True)

# 전역 변수
should_stop = Event()
total_found = 0
processed_links = 0
total_links = 0

# 실행시마다 새로운 로그 파일 생성
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"gambling_detector_{timestamp}.log"
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 새로운 로그 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            # 콘솔 로깅은 별도로 처리
        ]
    )
    
    logging.info(f"새로운 로그 파일 생성: {log_filename}")
    return log_filename

# 프로그램 시작시 로그 설정
current_log_file = setup_logging()

# 환경 변수 로드
load_dotenv()

# Naver Clova OCR API 설정 (환경 변수)
CLOVA_OCR_URL = os.getenv('CLOVA_OCR_URL', '')
CLOVA_SECRET_KEY = os.getenv('CLOVA_SECRET_KEY', '')

# Google Custom Search API 설정 (환경 변수)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID', '')

# PostgreSQL 데이터베이스 설정 (환경 변수)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "user": os.getenv("DB_USER", ""),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "sslmode": os.getenv("DB_SSLMODE", "require")
}

# 통합 CSV 파일명 설정
MASTER_CSV_FILE = "gambling_detection_results.csv"
CRAWLED_LINKS_CSV_FILE = "crawled_links.csv"

# 도박 관련 키워드 리스트
GAMBLING_KEYWORDS = [
    "카지노", "베팅", "배당률", "슬롯", "게임머니", "라이브 딜러",
    "가입 보너스", "환전", "무료 머니", "배당", "페이백",
    "콤프", "재입플", "입플", "전전", "휴게소", "매충",
    "재충전", "첫충", "첫입금 보너스", "매칭 보너스", "입금 보너스", "출금 보너스", "쿠폰", "추천인 코드",
    "즉시 환전", "빠른 환전", "고배당", "정산", "빠른 출금",
    "룰렛", "블랙잭", "포커", "홀덤", "바카라", "도박", "슬롯 머신",
    "VIP 혜택", "무한 보상", "로열티 프로그램", "토토", "입플", "놀이터"
]

def print_header():
    """프로그램 헤더 출력"""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}Gambling-X Console Version")
    print(f"{Fore.YELLOW}불법 도박 사이트 탐지 프로그램")
    print("="*80)
    print(f"{Fore.WHITE}로그 파일: {current_log_file}")
    print(f"결과 파일: {MASTER_CSV_FILE}")
    print(f"탐지 키워드: {len(GAMBLING_KEYWORDS)}개")
    print("="*80)

def print_status(message, level="INFO"):
    """상태 메시지 출력"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if level == "SUCCESS":
        print(f"{Fore.GREEN}[{timestamp}] ✅ {message}")
    elif level == "WARNING":
        print(f"{Fore.YELLOW}[{timestamp}] ⚠️  {message}")
    elif level == "ERROR":
        print(f"{Fore.RED}[{timestamp}] ❌ {message}")
    elif level == "FOUND":
        print(f"{Fore.MAGENTA}[{timestamp}] 🎯 {message}")
    else:
        print(f"{Fore.CYAN}[{timestamp}] ℹ️  {message}")

def print_progress():
    """진행 상황 출력"""
    global processed_links, total_links, total_found
    if total_links > 0:
        progress = (processed_links / total_links) * 100
        print(f"\r{Fore.BLUE}진행률: {progress:.1f}% ({processed_links}/{total_links}) | 탐지된 사이트: {total_found}개", end="", flush=True)

# 시그널 핸들러 (Ctrl+C 처리)
def signal_handler(sig, frame):
    print_status("\n프로그램을 중단합니다.", "WARNING")
    should_stop.set()
    print_status("안전하게 종료 중입니다. 잠시만 기다려주세요.", "INFO")

signal.signal(signal.SIGINT, signal_handler)

# PostgreSQL 데이터베이스 연결 함수
def get_db_connection():
    """PostgreSQL 데이터베이스 연결을 반환"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        logging.error(f"[DB ERROR] 데이터베이스 연결 실패: {str(e)}")
        print_status(f"데이터베이스 연결 실패: {str(e)}", "ERROR")
        return None

def check_url_exists(url):
    """URL이 이미 데이터베이스에 존재하는지 확인"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM crawled_sites WHERE url = %s", (url,))
            count = cursor.fetchone()[0]
            return count > 0
    except psycopg2.Error as e:
        logging.error(f"[DB ERROR] URL 중복 확인 실패: {str(e)}")
        return False
    finally:
        conn.close()

def save_url_to_db(url, search_keyword):
    """새로운 URL을 데이터베이스에 저장 (중복 확인 후 저장)"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            # 먼저 URL이 이미 존재하는지 확인
            cursor.execute("SELECT COUNT(*) FROM crawled_sites WHERE url = %s", (url,))
            count = cursor.fetchone()[0]
            
            if count > 0:
                logging.info(f"[DB SKIP] 이미 존재하는 URL: {url}")
                return False  # 이미 존재하므로 저장하지 않음
            
            # 존재하지 않으면 새로 저장 (platform='banner', keyword=검색키워드)
            cursor.execute(
                "INSERT INTO crawled_sites (url, platform, keyword, crawled_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)",
                (url, 'banner', search_keyword)
            )
            conn.commit()
            logging.info(f"[DB SUCCESS] 새 URL 저장 완료: {url}, 플랫폼: banner, 키워드: {search_keyword}")
            return True
            
    except psycopg2.Error as e:
        logging.error(f"[DB ERROR] URL 저장 실패: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def save_urls_batch_to_db(urls, search_keyword):
    """여러 URL을 한 번에 데이터베이스에 저장합니다."""
    if not urls:
        return 0
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    saved_count = 0
    try:
        with conn.cursor() as cursor:
            for url in urls:
                # 중복 확인 후 저장
                cursor.execute("SELECT COUNT(*) FROM crawled_sites WHERE url = %s", (url,))
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        "INSERT INTO crawled_sites (url, platform, keyword, crawled_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)",
                        (url, 'banner', search_keyword)
                    )
                    saved_count += 1
                    logging.info(f"[DB SUCCESS] 새 URL 저장: {url}, 플랫폼: banner, 키워드: {search_keyword}")
                else:
                    logging.info(f"[DB SKIP] 이미 존재하는 URL: {url}")
        
        conn.commit()
        print_status(f"데이터베이스에 {saved_count}개의 새 URL 저장 완료", "SUCCESS")
        
    except psycopg2.Error as e:
        logging.error(f"[DB ERROR] 배치 URL 저장 실패: {str(e)}")
        conn.rollback()
        print_status(f"데이터베이스 저장 실패: {str(e)}", "ERROR")
    finally:
        conn.close()
    
    return saved_count

# 도박 사이트 여부 판단 함수
def is_gambling_content(text):
    for keyword in GAMBLING_KEYWORDS:
        if keyword in text:
            logging.info(f"[도박 키워드 감지] '{keyword}'")
            return True, keyword
    return False, None

Image.MAX_IMAGE_PIXELS = None

# 이미지에서 텍스트 추출 함수
def extract_text_clova(image_url):
    logging.info(f"[OCR 처리] {image_url}")

    # 데이터 URL 또는 SVG 이미지는 건너뜁니다.
    if re.match(r"^data:image/", image_url) or image_url.endswith(".svg"):
        logging.info(f"[건너뜀] 데이터 URL 또는 SVG 이미지")
        return None

    headers = {
        'X-OCR-SECRET': CLOVA_SECRET_KEY,
    }

    # 이미지 다운로드
    try:
        image_response = requests.get(image_url, timeout=10)
        if image_response.status_code != 200:
            logging.error(f"[오류] 이미지 다운로드 실패: {image_url}")
            return None
        logging.info(f"[이미지 다운로드] 성공")
    except requests.exceptions.RequestException as e:
        logging.error(f"[ERROR] 이미지 다운로드 중 오류 발생: {str(e)}")
        return None

    # GIF 이미지일 경우 첫 번째 프레임 추출
    try:
        image = Image.open(BytesIO(image_response.content))
    except UnidentifiedImageError:
        logging.error(f"[ERROR] 유효하지 않은 이미지 파일: {image_url}")
        return None
    except Exception as e:
        logging.error(f"[ERROR] 이미지 처리 중 오류 발생: {str(e)}")
        return None

    if image.format == "GIF":
        logging.info(f"[INFO] GIF 이미지 첫 번째 프레임 추출 중: {image_url}")
        image = image.convert("RGB")
        with BytesIO() as output:
            image.save(output, format="JPEG")
            image_content = output.getvalue()
    else:
        image_content = image_response.content

    # Clova OCR 요청
    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    # 요청 데이터 및 파일 설정
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', ('image.jpg', image_content, 'image/jpeg'))
    ]

    # Clova OCR API 호출
    try:
        response = requests.post(CLOVA_OCR_URL, headers=headers, data=payload, files=files)
        if response.status_code != 200:
            logging.error(f"[ERROR] OCR 요청 실패: {response.status_code}, 응답 내용: {response.content}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"[ERROR] Clova OCR API 호출 중 오류 발생: {str(e)}")
        return None

    # 응답 JSON 파싱
    response_json = response.json()
    logging.info(f"[INFO] OCR 응답 수신 완료: {response_json}")

    extracted_text = []
    for image_data in response_json.get('images', []):
        for field in image_data.get('fields', []):
            extracted_text.append(field.get('inferText', '').strip())

    if extracted_text:
        logging.info(f"[텍스트 추출] 성공 ({len(' '.join(extracted_text))}자)")
    else:
        logging.info(f"[텍스트 추출] 실패")

    return " ".join(extracted_text) if extracted_text else "No text found"

# Google Custom Search API를 통해 다중 검색어 처리 함수
def get_google_search_links(keyword, start_result=1, end_result=100):
    """
    Google Custom Search API를 통해 키워드 검색 결과 수집
    
    Args:
        keyword (str): 검색 키워드
        start_result (int): 시작 결과 번호 (1부터 시작)
        end_result (int): 끝 결과 번호 (포함)
    
    Returns:
        list: 필터링된 검색 결과 링크 목록
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    results = []
    
    # 제외할 도메인 리스트 (뉴스 사이트 + 영상 플랫폼 + 기타)
    excluded_domains = [
        # 한국 뉴스 사이트
        "news.naver.com", "news.daum.net", "news.google.com",
        "yna.co.kr", "yonhapnews.co.kr", "newsis.com",
        "chosun.com", "joongang.co.kr", "donga.com", 
        "hani.co.kr", "khan.co.kr", "kyunghyang.com",
        "sbs.co.kr", "kbs.co.kr", "mbc.co.kr", "jtbc.joins.com",
        "ytn.co.kr", "nocutnews.co.kr", "edaily.co.kr",
        "mk.co.kr", "etnews.com", "dt.co.kr", "hankookilbo.com",
        
        # 영상 플랫폼
        "youtube.com", "youtu.be", "tiktok.com",
        "instagram.com", "facebook.com", "twitter.com",
        "vimeo.com", "dailymotion.com", "twitch.tv",
        
        # 기타 제외 사이트
        "korea.kr", "namu.wiki", "pinterest.com", "pinterest.co.kr"
    ]
    
    logging.info(f"[Google 검색] 키워드: {keyword} ({start_result}-{end_result}번 결과)")

    # start_result부터 end_result까지 10개씩 가져오기
    for start in range(start_result, end_result + 1, 10):
        if should_stop.is_set():
            break
        
        # 현재 배치에서 가져올 개수 계산
        current_end = min(start + 9, end_result)
        num_to_fetch = current_end - start + 1
        
        params = {
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": keyword,
            "start": start,
            "num": min(10, num_to_fetch),  # 최대 10개, 남은 개수가 적으면 그만큼만
            "safe": "off"      # 세이프서치 끄기
        }

        try:
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 403:
                logging.error(f"[ERROR] Google API 할당량 초과. 현재까지 {len(results)}개 수집")
                break
            elif response.status_code != 200:
                logging.error(f"[ERROR] 구글 API 요청 실패: {response.status_code}")
                break

            data = response.json()
            items = data.get("items", [])

            if not items:
                logging.info(f"[INFO] 더 이상 검색 결과가 없습니다.")
                break

            # 도메인 필터링 적용
            filtered_links = []
            for item in items:
                link = item["link"]
                # 제외할 도메인이 링크에 포함되어 있지 않은 경우만 추가
                if not any(domain in link for domain in excluded_domains):
                    filtered_links.append(link)
                else:
                    logging.info(f"[필터링됨] 제외된 도메인: {link}")
            
            results.extend(filtered_links)
            total_items = len(items)
            filtered_count = len(filtered_links)
            excluded_count = total_items - filtered_count
            
            logging.info(f"[Google 검색] {start}-{current_end} 페이지: 총 {total_items}개 중 {filtered_count}개 추가, {excluded_count}개 필터링됨 (총 {len(results)}개)")

            if len(items) < 10:
                break
                
            time.sleep(0.5)  # API 호출 간격 추가 (rate limiting 방지)
        except requests.exceptions.RequestException as e:
            logging.error(f"[ERROR] 구글 API 호출 중 오류: {str(e)}")
            break

    logging.info(f"[검색 결과] '{keyword}' 키워드에서 {len(results)}개의 링크 발견 (필터링 후)")
    
    # 크롤링된 링크들을 CSV에 저장
    if results:
        save_crawled_links_to_csv(keyword, results)
    
    return results

# 웹 페이지에서 이미지 URL 추출 함수
def extract_image_urls(url, html_content):
    try:
        # HTML 파싱 시 인코딩 문제 해결
        soup = BeautifulSoup(html_content, 'html.parser')
        image_data = []

        logging.info(f"[이미지 URL 추출] 시작: {url}")

        # 일반 이미지 처리
        for img_tag in soup.find_all("img"):
            try:
                parent_a = img_tag.find_parent("a")
                if parent_a and parent_a.get("href"):
                    image_url = urljoin(url, img_tag.get("src"))
                    link_url = urljoin(url, parent_a.get("href"))
                    image_data.append({
                        "image_url": image_url,
                        "link_url": link_url
                    })
            except Exception as e:
                logging.warning(f"[이미지 처리 오류] {url}에서 이미지 태그 처리 실패: {str(e)}")
                continue

        logging.info(f"[이미지 URL 추출] {len(image_data)}개 발견")
        return image_data
        
    except Exception as e:
        logging.error(f"[HTML 파싱 오류] {url}에서 HTML 파싱 실패: {str(e)}")
        return []

# 통합 CSV 파일에 결과 저장 함수 (수정됨)
def save_to_master_csv(result):
    """
    단일 도박 사이트 결과를 마스터 CSV 파일에 추가
    """
    file_exists = os.path.isfile(MASTER_CSV_FILE)
    
    with open(MASTER_CSV_FILE, 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['detection_time', 'image_url', 'link_url', 'search_keyword', 'search_link', 'extracted_text', 'detected_keyword']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 새로 생성되는 경우에만 헤더 작성
        if not file_exists:
            writer.writeheader()
            logging.info(f"[CSV 초기화] 새로운 마스터 CSV 파일 생성: {MASTER_CSV_FILE}")
        
        # 결과 데이터 추가
        csv_row = {
            'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_url': result.get('image_url', ''),
            'link_url': result.get('link_url', ''),
            'search_keyword': result.get('keyword', ''),
            'search_link': result.get('search_link', ''),
            'extracted_text': result.get('text', ''),
            'detected_keyword': result.get('detected_keyword', '')
        }
        
        writer.writerow(csv_row)
        logging.info(f"[CSV 저장] 도박 사이트 결과 추가: {result.get('link_url', '')}")

# 키워드별 분석 함수
def process_keyword(keyword):
    global processed_links, total_found
    
    links = get_google_search_links(keyword)
    print_status(f"'{keyword}' 키워드 분석 시작 - {len(links)}개 링크", "INFO")
    
    for link in links:
        if should_stop.is_set():
            print_status("사용자 요청으로 분석을 중단합니다.", "WARNING")
            break
            
        processed_links += 1
        print_progress()
        
        logging.info(f"[링크 분석] {link}")
        try:
            response = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if response.status_code == 200:
                image_data = extract_image_urls(link, response.text)
                for item in image_data:
                    if should_stop.is_set():
                        break
                    text_response = extract_text_clova(item["image_url"])
                    if text_response:
                        is_gambling, keyword_detected = is_gambling_content(text_response)
                        if is_gambling:
                            total_found += 1
                            gambling_result = {
                                "image_url": item["image_url"],
                                "link_url": item["link_url"],
                                "keyword": keyword,
                                "search_link": link,
                                "text": text_response,
                                "detected_keyword": keyword_detected
                            }
                            
                            # CSV에 저장
                            save_to_master_csv(gambling_result)
                            
                            # 콘솔에 결과 출력
                            print()  # 진행률 줄바꿈
                            print_status(f"도박 사이트 발견! 키워드: '{keyword_detected}'", "FOUND")
                            print_status(f"URL: {item['link_url']}", "FOUND")
                            print_progress()  # 진행률 다시 출력
                            
                            # 데이터베이스 저장 시도
                            if save_url_to_db(item["link_url"], keyword):
                                print_status(f"💾 새 URL을 데이터베이스에 저장 완료", "SUCCESS")
                            else:
                                print_status(f"🔄 이미 데이터베이스에 존재하는 URL", "WARNING")
            else:
                logging.warning(f"[링크 접속 실패] {link}, 상태 코드: {response.status_code}")
            time.sleep(1)  # API 호출 제한
        except requests.exceptions.RequestException as e:
            logging.error(f"[오류] 링크 분석 실패: {link} - {str(e)}")

    print()  # 진행률 줄바꿈
    print_status(f"'{keyword}' 키워드 분석 완료", "SUCCESS")

# 키워드 파일에서 검색어 읽기 함수
def load_keywords_from_file(filename="search_wordlist.txt"):
    """
    search_wordlist.txt 파일에서 키워드를 읽어옵니다.
    각 라인이 하나의 키워드입니다.
    파일 끝까지 모든 키워드를 로딩합니다.
    """
    keywords = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                keyword = line.strip()
                if keyword and not keyword.startswith('#'):  # 빈 줄과 주석 제외
                    keywords.append(keyword)
        
        logging.info(f"[키워드 로딩] {filename}에서 {len(keywords)}개 키워드 로딩 완료 (제한 없음)")
        return keywords
        
    except FileNotFoundError:
        logging.error(f"[ERROR] {filename} 파일을 찾을 수 없습니다.")
        # 기본 키워드 반환
        default_keywords = ["무료 웹툰", "먹튀 없는", "사다리 사이트 추천", "토토 놀이터", "입플 토토"]
        logging.info(f"[기본 키워드] 기본 키워드 {len(default_keywords)}개 사용")
        return default_keywords
        
    except Exception as e:
        logging.error(f"[ERROR] 키워드 파일 읽기 중 오류: {str(e)}")
        return ["무료 웹툰", "먹튀 없는", "사다리 사이트 추천"]

def test_db_connection():
    """데이터베이스 연결 테스트"""
    print_status("PostgreSQL 데이터베이스 연결 테스트 중...", "INFO")
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                print_status(f"데이터베이스 연결 성공! 버전: {version}", "SUCCESS")
                
                # 테이블 존재 확인
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_tables 
                        WHERE tablename = 'crawled_sites'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                if table_exists:
                    print_status("crawled_sites 테이블 확인됨", "SUCCESS")
                    
                    # 현재 저장된 URL 개수 확인
                    cursor.execute("SELECT COUNT(*) FROM crawled_sites;")
                    count = cursor.fetchone()[0]
                    print_status(f"현재 저장된 URL 개수: {count}개", "INFO")
                else:
                    print_status("crawled_sites 테이블이 존재하지 않습니다!", "ERROR")
                    
        except Exception as e:
            print_status(f"데이터베이스 쿼리 실패: {str(e)}", "ERROR")
        finally:
            conn.close()
        return True
    else:
        print_status("데이터베이스 연결 실패", "ERROR")
        return False

def main():
    """메인 실행 함수"""
    global total_found, processed_links
    
    # 헤더 출력
    print_header()
    
    # 데이터베이스 연결 테스트
    if not test_db_connection():
        print_status("데이터베이스 연결에 실패했습니다. 프로그램을 종료합니다.", "ERROR")
        return
    
    # 키워드 파일에서 검색어 로딩
    print_status("키워드 파일에서 검색어를 로딩합니다...", "INFO")
    keywords = load_keywords_from_file()
    
    if not keywords:
        print_status("키워드가 없습니다. 프로그램을 종료합니다.", "ERROR")
        return
    
    # 명령줄 인수로 시작 지점 및 검색 범위 결정
    print_status(f"총 {len(keywords)}개의 키워드가 로딩되었습니다.", "INFO")
    
    start_index = 0
    search_start = 1
    search_end = 100
    
    if len(sys.argv) > 1:
        try:
            start_num = int(sys.argv[1])
            if start_num < 1 or start_num > len(keywords):
                print_status(f"잘못된 키워드 번호입니다. 1-{len(keywords)} 사이의 숫자를 입력하세요.", "ERROR")
                print_status(f"사용법: python3 app.py [키워드시작번호] [검색시작번호] [검색끝번호]", "INFO")
                print_status(f"예시: python3 app.py 1 1 50  (1번 키워드부터, 1-50번 검색결과)", "INFO")
                print_status(f"예시: python3 app.py 10 51 100  (10번 키워드부터, 51-100번 검색결과)", "INFO")
                return
            start_index = start_num - 1
            print_status(f"명령줄 인수로 {start_num}번 키워드부터 시작합니다.", "INFO")
        except ValueError:
            print_status("잘못된 키워드 시작 번호입니다. 숫자를 입력하세요.", "ERROR")
            print_status(f"사용법: python3 app.py [키워드시작번호] [검색시작번호] [검색끝번호]", "INFO")
            return
    else:
        print_status("시작 번호가 지정되지 않아 1번 키워드부터 시작합니다.", "INFO")
        print_status(f"다음부터는 'python3 app.py [키워드시작번호] [검색시작번호] [검색끝번호]'로 시작할 수 있습니다.", "INFO")
    
    # 검색 결과 범위 설정
    if len(sys.argv) > 2:
        try:
            search_start = int(sys.argv[2])
            if search_start < 1:
                print_status("검색 시작 번호는 1 이상이어야 합니다.", "ERROR")
                return
            print_status(f"검색 시작 번호: {search_start}", "INFO")
        except ValueError:
            print_status("잘못된 검색 시작 번호입니다. 숫자를 입력하세요.", "ERROR")
            return
    
    if len(sys.argv) > 3:
        try:
            search_end = int(sys.argv[3])
            if search_end < search_start:
                print_status("검색 끝 번호는 시작 번호보다 크거나 같아야 합니다.", "ERROR")
                return
            if search_end > 100:
                print_status("검색 끝 번호는 100 이하여야 합니다. (Google API 제한)", "WARNING")
                search_end = 100
            print_status(f"검색 끝 번호: {search_end}", "INFO")
        except ValueError:
            print_status("잘못된 검색 끝 번호입니다. 숫자를 입력하세요.", "ERROR")
            return
    
    # 시작 지점 정보 출력
    if start_index > 0:
        print_status(f"키워드 {start_index + 1}번 '{keywords[start_index]}'부터 시작합니다.", "SUCCESS")
        print_status(f"건너뛴 키워드: {start_index}개", "INFO")
    else:
        print_status("첫 번째 키워드부터 시작합니다.", "SUCCESS")
    
    print_status("Gambling-X 프로그램을 시작합니다!", "SUCCESS")
    print_status(f"처리할 키워드: {len(keywords) - start_index}개 (키워드당 {search_start}-{search_end}번 검색결과)", "INFO")
    print_status("중단하려면 Ctrl+C를 누르세요.", "INFO")
    
    # 시작할 키워드들 미리 보기
    remaining_keywords = keywords[start_index:]
    preview_keywords = remaining_keywords[:5]
    print_status(f"키워드 미리보기: {', '.join(preview_keywords)}...", "INFO")
    print()
    
    start_time = time.time()
    total_found = 0
    processed_links = 0
    total_analyzed_links = 0
    
    # 선택된 지점부터 키워드별로 검색 → 즉시 분석 수행
    for i, keyword in enumerate(keywords[start_index:], start_index + 1):
        try:
            if should_stop.is_set():
                print_status("사용자 요청으로 프로그램을 중단합니다.", "WARNING")
                break
                
            print_status(f"[{i}/{len(keywords)}] '{keyword}' 처리 시작", "INFO")
            
            # 1단계: 키워드로 링크 검색 (새로운 매개변수 사용)
            print_status(f"🔍 Google 검색 중... ({search_start}-{search_end}번 결과)", "INFO")
            links = get_google_search_links(keyword, search_start, search_end)
            
            if not links:
                print_status(f"'{keyword}' 검색 결과 없음", "WARNING")
                continue
                
            print_status(f"{len(links)}개 링크 수집 완료", "SUCCESS")
            
            # 2단계: 수집된 링크들을 즉시 분석
            print_status(f"링크 분석 시작...", "INFO")
            keyword_found = 0
            
            for j, link in enumerate(links, 1):
                if should_stop.is_set():
                    break
                    
                total_analyzed_links += 1
                print(f"\r분석 중: [{j}/{len(links)}] | 전체 탐지: {total_found}개", end="", flush=True)
                
                try:
                    response = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                    if response.status_code == 200:
                        try:
                            # HTML 파싱 및 이미지 추출에 예외 처리 추가
                            image_data = extract_image_urls(link, response.text)
                            for item in image_data:
                                if should_stop.is_set():
                                    break
                                try:
                                    text_response = extract_text_clova(item["image_url"])
                                    if text_response:
                                        is_gambling, keyword_detected = is_gambling_content(text_response)
                                        if is_gambling:
                                            total_found += 1
                                            keyword_found += 1
                                            gambling_result = {
                                                "image_url": item["image_url"],
                                                "link_url": item["link_url"],
                                                "keyword": keyword,
                                                "search_link": link,
                                                "text": text_response,
                                                "detected_keyword": keyword_detected
                                            }
                                            
                                            # CSV에 즉시 저장
                                            save_to_master_csv(gambling_result)
                                            
                                            # 도박 사이트로 탐지된 URL을 데이터베이스에 저장
                                            print()  # 진행률 줄바꿈
                                            print_status(f"도박 사이트 발견! 키워드: '{keyword_detected}'", "FOUND")
                                            print_status(f"URL: {item['link_url']}", "FOUND")
                                            
                                            # 데이터베이스 저장 시도
                                            if save_url_to_db(item["link_url"], keyword):
                                                print_status(f"새 URL을 데이터베이스에 저장 완료", "SUCCESS")
                                            else:
                                                print_status(f"이미 데이터베이스에 존재하는 URL", "WARNING")
                                except Exception as ocr_e:
                                    logging.error(f"[OCR 오류] {item.get('image_url', 'unknown')}에서 OCR 처리 실패: {str(ocr_e)}")
                                    continue
                        except Exception as html_e:
                            logging.error(f"[HTML 파싱 오류] {link}에서 HTML 파싱 실패: {str(html_e)}")
                            continue
                    else:
                        logging.warning(f"[링크 접속 실패] {link}, 상태 코드: {response.status_code}")
                    time.sleep(1)  # API 호출 제한
                except requests.exceptions.RequestException as e:
                    logging.error(f"[요청 오류] 링크 분석 실패: {link} - {str(e)}")
                    continue
                except Exception as e:
                    logging.error(f"[예상치 못한 오류] 링크 처리 중 오류 발생: {link} - {str(e)}")
                    continue
            
            print()  # 진행률 줄바꿈
            
            # 키워드별 결과 요약
            if keyword_found > 0:
                print_status(f"'{keyword}' 완료: {keyword_found}개 도박사이트 탐지", "SUCCESS")
            else:
                print_status(f"'{keyword}' 완료: 도박사이트 탐지 없음", "SUCCESS")
                
            print_status(f"현재까지 총 탐지: {total_found}개 ({i}/{len(keywords)} 키워드 완료)", "INFO")
            
            # API 할당량 관리
            if i % 2 == 0:
                print_status("⏱API 호출 제한을 위해 잠시 대기합니다...", "INFO")
                time.sleep(3)
            
            print("-" * 60)  # 구분선
        except Exception as keyword_e:
            logging.error(f"[키워드 처리 오류] '{keyword}' 처리 중 오류 발생: {str(keyword_e)}")
            print_status(f"'{keyword}' 처리 중 오류 발생, 다음 키워드로 계속...", "ERROR")
            continue
    
    # 최종 결과 요약
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print(f"{Fore.GREEN}{Style.BRIGHT}분석 완료!")
    print(f"{Fore.YELLOW}최종 결과:")
    print(f"   소요 시간: {elapsed_time:.1f}초")
    print(f"   처리된 키워드: {i}/{len(keywords)}개")
    print(f"   분석된 링크: {total_analyzed_links}개")
    print(f"   탐지된 도박 사이트: {total_found}개")
    print(f"   결과 파일: {MASTER_CSV_FILE}")
    print(f"   로그 파일: {current_log_file}")
    print("="*80)
    
    if total_found > 0:
        detection_rate = (total_found / total_analyzed_links * 100) if total_analyzed_links > 0 else 0
        print_status(f"{total_found}개의 불법 도박 사이트가 탐지되었습니다! (탐지율: {detection_rate:.1f}%)", "WARNING")
        print_status(f"상세 결과는 '{MASTER_CSV_FILE}' 파일을 확인하세요.", "INFO")
    else:
        print_status("탐지된 도박 사이트가 없습니다.", "SUCCESS")

# 크롤링된 링크를 CSV에 저장하는 함수
def save_crawled_links_to_csv(keyword, links):
    """
    키워드별로 크롤링된 링크들을 CSV 파일에 저장
    """
    file_exists = os.path.isfile(CRAWLED_LINKS_CSV_FILE)
    
    with open(CRAWLED_LINKS_CSV_FILE, 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['crawl_time', 'keyword', 'link_url', 'link_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 파일이 새로 생성되는 경우에만 헤더 작성
        if not file_exists:
            writer.writeheader()
            logging.info(f"[CSV 초기화] 새로운 크롤링 링크 CSV 파일 생성: {CRAWLED_LINKS_CSV_FILE}")
            print_status(f"크롤링 링크 저장 파일 생성: {CRAWLED_LINKS_CSV_FILE}", "INFO")
        
        # 크롤링된 링크들을 각각 저장
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, link in enumerate(links, 1):
            csv_row = {
                'crawl_time': current_time,
                'keyword': keyword,
                'link_url': link,
                'link_index': i
            }
            writer.writerow(csv_row)
        
        logging.info(f"[CSV 저장] '{keyword}' 키워드로 크롤링된 {len(links)}개 링크 저장 완료")
        print_status(f"'{keyword}' 키워드 링크 {len(links)}개 저장 완료", "SUCCESS")

if __name__ == '__main__':
    try:
        # colorama 모듈이 없으면 설치 안내
        try:
            from colorama import init, Fore, Back, Style
            init(autoreset=True)
        except ImportError:
            print("컬러 출력을 위해 colorama 모듈을 설치해주세요:")
            print("pip install colorama")
            # colorama 없이도 실행 가능하도록 설정
            class DummyColor:
                def __getattr__(self, name):
                    return ""
            Fore = Back = Style = DummyColor()
        
        logging.info(f"Gambling-X Console 프로그램 시작 - 로그 파일: {current_log_file}")
        main()
    except KeyboardInterrupt:
        print_status("\n프로그램이 사용자에 의해 중단되었습니다.", "WARNING")
    except Exception as e:
        print_status(f"예상치 못한 오류가 발생했습니다: {str(e)}", "ERROR")
        logging.error(f"프로그램 실행 중 오류: {str(e)}")
    finally:
        print_status("프로그램을 종료합니다.", "INFO")