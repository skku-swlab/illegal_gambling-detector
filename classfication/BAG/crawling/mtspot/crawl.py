import time
import re
from typing import List, Tuple
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import random
import functools

def retry_on_failure(max_retries=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"시도 {attempt + 1} 실패: {str(e)}")
                    print("드라이버 재초기화 중...")
                    self.setup_driver()
                    time.sleep(2)  # 재시도 전 대기
            return None
        return wrapper
    return decorator

class MtSpotCrawler:
    def __init__(self):
        self.base_url = "https://mt-spot.com"
        self.board_url = f"{self.base_url}/bbs/board.php?bo_table=review&page="
        self.setup_driver()
        
    def setup_driver(self):
        """Chrome 드라이버 설정"""
        chrome_options = Options()
        chrome_options.binary_location = "/usr/bin/google-chrome"
        chrome_options.add_argument('--headless=new')  # 새로운 헤드리스 모드
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920x1080')
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-setuid-sandbox')
        chrome_options.add_argument('--remote-debugging-port=9222')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            print("Chrome 드라이버 초기화 성공")
        except Exception as e:
            print(f"Chrome 드라이버 초기화 실패: {str(e)}")
            raise

    @retry_on_failure(max_retries=3)
    def get_post_links(self, page: int) -> List[str]:
        """페이지에서 게시글 링크들을 수집"""
        try:
            print(f"\n[디버그] 페이지 URL 접근: {self.board_url}{page}")
            self.driver.get(f"{self.board_url}{page}")
            time.sleep(3)  # 페이지 로딩 대기
            
            # 페이지 소스 출력
            print("\n[디버그] 페이지 소스:")
            print(self.driver.page_source[:500])  # 처음 500자만 출력
            
            # 게시글 링크 수집
            links = []
            elements = self.driver.find_elements(By.CSS_SELECTOR, 'div.wr-subject a')
            
            print("\n[디버그] 찾은 요소들:")
            for element in elements:
                href = element.get_attribute('href')
                text = element.text
                if href:
                    links.append(href)
                    print(f"텍스트: {text}")
                    print(f"링크: {href}\n")
            
            print(f"[디버그] 수집된 링크 수: {len(links)}")
            return links
            
        except Exception as e:
            print(f"링크 수집 중 오류 발생: {e}")
            print(f"[디버그] 현재 페이지 URL: {self.driver.current_url}")
            raise  # 재시도 로직을 위해 예외를 다시 발생시킴

    @retry_on_failure(max_retries=3)
    def get_post_content(self, url: str) -> Tuple[str, List[str]]:
        """게시글과 댓글 내용을 수집"""
        try:
            print(f"\n[디버그] 게시글 URL 접근: {url}")
            self.driver.get(url)
            time.sleep(3)  # 페이지 로딩 대기
            
            # 게시글 내용 수집
            content = ""
            try:
                content_element = self.driver.find_element(By.CSS_SELECTOR, 'td.user_say')
                content = content_element.text
                print("\n[디버그] 수집된 게시글 내용:")
                print(content[:200])  # 처음 200자만 출력
            except Exception as e:
                print(f"[디버그] 게시글 내용 수집 실패: {e}")
            
            # 댓글 수집
            comments = []
            try:
                comment_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div.media-content textarea')
                print("\n[디버그] 수집된 댓글:")
                for element in comment_elements:
                    if element.get_attribute('id').startswith('save_comment_'):
                        comment_text = element.get_attribute('value')
                        if comment_text:
                            comments.append(comment_text)
                            print(f"댓글: {comment_text[:100]}")  # 처음 100자만 출력
            except Exception as e:
                print(f"[디버그] 댓글 수집 실패: {e}")
            
            return content, comments
            
        except Exception as e:
            print(f"내용 수집 중 오류 발생: {e}")
            print(f"[디버그] 현재 페이지 URL: {self.driver.current_url}")
            raise  # 재시도 로직을 위해 예외를 다시 발생시킴

    def split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분리"""
        sentences = re.split('[.!?。!\n]\s*', text)
        return [sent.strip() for sent in sentences if sent.strip()]

    def save_to_csv(self, data: List[Tuple[int, int, str]], filename: str, mode: str = 'w'):
        """데이터를 CSV 파일로 저장"""
        df = pd.DataFrame(data, columns=['id', 'label', 'content'])
        # mode가 'a'면 헤더는 처음에만 작성
        df.to_csv(filename, mode=mode, index=False, encoding='utf-8-sig', header=(mode == 'w'))

    def crawl(self):
        """크롤링 실행"""
        try:
            id_counter = 1
            page = 1
            filename = 'mtspot_data.csv'
            
            # 파일 초기화 (헤더만 작성)
            self.save_to_csv([], filename, mode='w')
            
            while True:
                print(f"페이지 {page} 크롤링 중...")
                links = self.get_post_links(page)
                
                if not links:
                    break
                    
                for link in links:
                    current_data = []
                    print(f"게시글 크롤링 중: {link}")
                    content, comments = self.get_post_content(link)
                    
                    if content:
                        sentences = self.split_into_sentences(content)
                        for sentence in sentences:
                            if sentence:
                                current_data.append((id_counter, 1, sentence))
                                id_counter += 1
                    
                    for comment in comments:
                        sentences = self.split_into_sentences(comment)
                        for sentence in sentences:
                            if sentence:
                                current_data.append((id_counter, 1, sentence))
                                id_counter += 1
                    
                    # 현재 게시글의 데이터를 바로 저장
                    if current_data:
                        self.save_to_csv(current_data, filename, mode='a')
                        print(f"게시글 데이터 저장 완료: {len(current_data)}개 문장")
                    
                    time.sleep(random.uniform(2, 4))  # 게시글 간 딜레이
                
                page += 1
                time.sleep(random.uniform(3, 5))  # 페이지 간 딜레이
            
            print("크롤링 완료!")
            
        finally:
            self.driver.quit()  # 브라우저 종료

if __name__ == "__main__":
    crawler = MtSpotCrawler()
    crawler.crawl()
