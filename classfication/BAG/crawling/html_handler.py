import os
from datetime import datetime
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import networkx as nx
import json
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import html as html_lib
import random

# 크롤링할 URL 목록
urls = [
    "https://boss-6594.com/",
    "https://fre-good.com/"
]

# 저장할 파일 시작 번호 (사용자가 이 값을 변경할 수 있음)
START_FILE_NUMBER = 13

# 사용자 에이전트 목록 (다양한 브라우저 시뮬레이션)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
]

class HTMLHandler:
    """
    HTML 파일 크롤링 및 처리를 위한 클래스
    """
    
    def __init__(self, wait_time=10, html_save_dir="/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/crawled_html_files", 
                 graph_save_dir="/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset"):
        """
        HTMLHandler 초기화
        
        Args:
            wait_time (int): 페이지 로딩 기본 대기 시간(초)
            html_save_dir (str): HTML 파일 저장 디렉토리
            graph_save_dir (str): 그래프 JSON 파일 저장 디렉토리
        """
        self.wait_time = wait_time
        self.html_save_dir = html_save_dir
        self.graph_save_dir = graph_save_dir
        
        # 디렉토리 생성
        os.makedirs(self.html_save_dir, exist_ok=True)
        os.makedirs(self.graph_save_dir, exist_ok=True)
        
    def is_valid_content(self, html_content):
        """
        HTML 내용이 Cloudflare 보안 페이지가 아닌 실제 컨텐츠인지 확인합니다.
        
        Args:
            html_content (str): 확인할 HTML 내용
            
        Returns:
            bool: HTML이 유효한 컨텐츠인 경우 True, 아니면 False
        """
        # Cloudflare 등의 보안 페이지 탐지
        invalid_patterns = [
            "Just a moment...",
            "Please wait while we verify",
            "Checking your browser",
            "이 페이지는 로봇이 아닌지 확인",
            "보안 확인 중입니다",
            "DDoS protection by Cloudflare",
            "cloudflare-nginx",
            "Browser check",
            "Bot validation"
        ]
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 타이틀 체크
        title = soup.title.text if soup.title else ""
        for pattern in invalid_patterns:
            if pattern in title or pattern in html_content[:1000]:
                print(f"보안 페이지 감지: '{pattern}'")
                return False
                
        # 최소 컨텐츠 길이와 태그 수 검증은 제외 (사용자 요청)
        
        return True
        
    def get_full_html(self, url, max_retries=3):
        """
        Selenium을 사용하여 자바스크립트가 실행된 후의 완전한 HTML을 가져옵니다.
        보안 페이지 감지 시 여러 번 재시도합니다.
        
        Args:
            url (str): 크롤링할 웹페이지의 URL
            max_retries (int): 최대 재시도 횟수
            
        Returns:
            str: 완전한 HTML 내용 또는 오류 발생 시 None
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 재시도할 때마다 대기 시간 증가 (지수 백오프)
                current_wait_time = self.wait_time * (1.5 ** retry_count)
                
                # 임의의 사용자 에이전트 선택
                user_agent = random.choice(USER_AGENTS)
                
                # Chrome 옵션 설정
                chrome_options = Options()
                chrome_options.add_argument("--headless")  # 브라우저 창 표시 안 함
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")
                chrome_options.add_argument(f"--user-agent={user_agent}")
                
                # 추가 브라우저 설정
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # 자동화 탐지 방지
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                
                # WebDriver 초기화
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
                
                # 자동화 탐지 방지 스크립트 실행
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                
                # 페이지 로드
                print(f"{url} 페이지 로딩 중... {current_wait_time:.1f}초 대기 (시도 {retry_count+1}/{max_retries})")
                driver.get(url)
                
                # 자바스크립트 실행 대기
                time.sleep(current_wait_time)
                
                # 스크롤 다운 시뮬레이션 (더 많은 컨텐츠 로드)
                self.simulate_user_interaction(driver)
                
                # 전체 HTML 가져오기
                html_content = driver.page_source
                
                # WebDriver 종료
                driver.quit()
                
                # HTML 컨텐츠 유효성 검사
                if self.is_valid_content(html_content):
                    print(f"유효한 HTML 컨텐츠 수집 성공 (시도 {retry_count+1})")
                    return html_content
                else:
                    print(f"보안 페이지 감지 - 재시도 중... ({retry_count+1}/{max_retries})")
                    retry_count += 1
                    # 재시도 전 잠시 대기 (보안 시스템 우회 목적)
                    time.sleep(random.uniform(3.0, 7.0))
            
            except Exception as e:
                print(f"URL '{url}' 크롤링 중 오류 발생: {e}")
                retry_count += 1
                
        print(f"최대 재시도 횟수 초과 ({max_retries}회). URL: {url}")
        return None
    
    def simulate_user_interaction(self, driver):
        """
        실제 사용자처럼 페이지와 상호작용합니다.
        
        Args:
            driver: Selenium WebDriver 인스턴스
        """
        try:
            # 페이지 스크롤 다운
            for i in range(3):
                scroll_amount = random.randint(300, 700)  # 스크롤 양 랜덤화
                driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.5))  # 스크롤 간 대기 시간 랜덤화
            
            # 페이지 상단으로 돌아가기
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception as e:
            print(f"사용자 상호작용 시뮬레이션 중 오류: {e}")
    
    def save_html(self, html_content, url):
        """
        HTML 내용을 파일로 저장합니다.
        
        Args:
            html_content (str): 저장할 HTML 내용
            url (str): 크롤링한 URL (파일명 생성에 사용)
            
        Returns:
            str: 저장된 파일 경로 또는 None
        """
        if not html_content:
            return None
            
        # URL에서 파일명 생성 (도메인 추출)
        domain = urlparse(url).netloc.replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_{timestamp}.html"
        
        # HTML 파일 저장
        file_path = os.path.join(self.html_save_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML이 '{file_path}'에 저장되었습니다.")
        return file_path
    
    def crawl_url(self, url):
        """
        URL에서 HTML을 크롤링하고 저장합니다.
        
        Args:
            url (str): 크롤링할 URL
            
        Returns:
            tuple: (저장된 파일 경로, HTML 내용) 또는 (None, None)
        """
        print(f"{url} 크롤링 중...")
        html_content = self.get_full_html(url)
        
        if html_content:
            file_path = self.save_html(html_content, url)
            return file_path, html_content
        
        return None, None
    
    def parse_html(self, html_content):
        """
        HTML 내용을 파싱하는 메소드 (확장 가능)
        
        Args:
            html_content (str): 파싱할 HTML 내용
            
        Returns:
            dict: 파싱 결과
        """
        # 이 메소드는 추후 필요에 따라 확장 가능
        return {"raw_length": len(html_content) if html_content else 0}
    
    def html_to_graph(self, html_file_path=None, html_content=None):
        """
        HTML 파일이나 내용을 그래프 구조로 변환합니다.
        
        Args:
            html_file_path (str, optional): HTML 파일 경로
            html_content (str, optional): HTML 내용 문자열
            
        Returns:
            networkx.Graph: HTML 구조를 표현하는 그래프
        """
        if html_file_path is None and html_content is None:
            raise ValueError("HTML 파일 경로 또는 HTML 내용을 제공해야 합니다.")
        
        # HTML 파일에서 내용 로드
        if html_file_path and not html_content:
            try:
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except Exception as e:
                print(f"HTML 파일을 읽는 중 오류 발생: {e}")
                return None
        
        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 그래프 생성
        G = nx.DiGraph()  # 방향성 그래프 사용 (부모->자식)
        
        # 노드 ID 생성을 위한 카운터
        node_id_counter = 0
        
        # 태그-노드 ID 맵핑을 저장할 사전
        tag_to_node = {}
        
        # HTML을 순회하며 그래프 구성
        def traverse_html(element, parent_id=None):
            nonlocal node_id_counter
            
            # 텍스트 노드는 건너뛰기
            if element.name is None:
                return
            
            current_id = node_id_counter
            node_id_counter += 1
            
            # 태그의 텍스트 가져오기 (직접 자식 텍스트만)
            text = element.string if element.string else ''.join(t for t in element.find_all(text=True, recursive=False))
            text = text.strip() if text else ''
            
            # 노드에 태그 이름과 텍스트 저장
            G.add_node(current_id, tag=element.name, text=text, node_type='tag')
            
            # 태그-노드 ID 맵핑 저장
            tag_to_node[element] = current_id
            
            # 부모-자식 관계 설정
            if parent_id is not None:
                G.add_edge(parent_id, current_id)
            
            # 자식 요소 순회
            for child in element.children:
                if child.name:  # 텍스트 노드 제외
                    traverse_html(child, current_id)
        
        # 루트 요소(html)부터 순회 시작
        traverse_html(soup.html)
        
        print(f"HTML을 그래프로 변환했습니다. 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")
        return G
    
    def graph_to_json_data(self, graph):
        """
        NetworkX 그래프를 JSON 직렬화 가능한 데이터 구조로 변환합니다.
        
        Args:
            graph (networkx.Graph): 변환할 그래프
            
        Returns:
            dict: JSON 직렬화 가능한 그래프 데이터
        """
        # 노드 정보 변환
        nodes = []
        for node_id in graph.nodes():
            # 소수점 2자리까지 반올림
            gambling_score = graph.nodes[node_id].get('gambling_score', 0.0)
            gambling_score = round(gambling_score, 2)
            
            node_data = {
                'id': node_id,
                'tag': graph.nodes[node_id].get('tag', ''),
                'text': graph.nodes[node_id].get('text', ''),
                'node_type': graph.nodes[node_id].get('node_type', ''),
                'gambling_score': gambling_score
            }
            nodes.append(node_data)
        
        # 엣지 정보 변환
        edges = []
        for source, target in graph.edges():
            edge_data = {
                'source': source,
                'target': target
            }
            edges.append(edge_data)
        
        # 전체 그래프 데이터
        graph_data = {
            'nodes': nodes,
            'edges': edges
        }
        
        return graph_data
    
    def save_graph_as_json(self, graph, filename):
        """
        그래프를 JSON 파일로 저장합니다.
        
        Args:
            graph (networkx.Graph): 저장할 그래프
            filename (str): 파일명
            
        Returns:
            str: 저장된 파일 경로 또는 None
        """
        try:
            # 그래프를 JSON 직렬화 가능한 데이터로 변환
            graph_data = self.graph_to_json_data(graph)
            
            # 출력 파일 경로 생성
            output_file = os.path.join(self.graph_save_dir, filename)
            
            # JSON 파일로 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            print(f"그래프가 JSON 파일 '{output_file}'로 저장되었습니다.")
            return output_file
        except Exception as e:
            print(f"그래프 저장 중 오류 발생: {e}")
            return None
    
    def add_gambling_scores_to_graph(self, graph, model_path=None):
        """
        그래프의 각 노드 텍스트에 불법도박 점수를 추가합니다.
        
        Args:
            graph (networkx.Graph): 점수를 추가할 그래프
            model_path (str): BERT 모델 경로
            
        Returns:
            networkx.Graph: 불법도박 점수가 추가된 그래프
        """
        # 기본 모델 경로 설정
        if model_path is None:
            model_path = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/gambling_bert_model"
            if not os.path.exists(model_path):
                print(f"경고: 기본 모델 경로 '{model_path}'가 존재하지 않습니다.")
                # 현재 디렉토리에서 모델 경로 확인
                current_dir_model = "./gambling_bert_model"
                if os.path.exists(current_dir_model):
                    model_path = current_dir_model
                    print(f"대체 모델 경로 '{model_path}'를 사용합니다.")
                else:
                    print(f"오류: 모델을 찾을 수 없습니다. 점수 추가를 건너뜁니다.")
                    return graph
        
        try:
            # 모델 경로 확인
            if not os.path.exists(model_path):
                print(f"오류: 모델 경로 '{model_path}'가 존재하지 않습니다. 점수 추가를 건너뜁니다.")
                return graph
            
            print(f"사용할 모델 경로: {model_path}")
            
            # BERT 모델 및 토크나이저 로드
            print("BERT 모델 로드 중...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"사용 장치: {device}")
            
            try:
                # 토크나이저 로드
                tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
                print("토크나이저 로드 완료")
                
                # 모델 로드
                model = BertForSequenceClassification.from_pretrained(model_path)
                model.to(device)
                model.eval()
                print(f"BERT 모델 로드 완료. 장치: {device}")
            except Exception as e:
                print(f"모델 로드 중 오류 발생: {e}")
                print("모델 로드에 실패했습니다. 점수 추가를 건너뜁니다.")
                return graph
            
            # 불법도박 점수 계산 함수
            def get_gambling_score(text):
                if not text or len(text.strip()) == 0:
                    return 0.0
                    
                # 텍스트 정규화
                text = html_lib.unescape(text)
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'[^\w\s가-힣]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                # 텍스트가 너무 짧으면 처리하지 않음
                if len(text) < 2:
                    return 0.0
                
                try:
                    # 토큰화 및 모델 입력 준비
                    inputs = tokenizer(
                        text,
                        add_special_tokens=True,
                        max_length=256,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    ).to(device)
                    
                    # 추론
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        # 불법도박 클래스(label 1)에 대한 점수 추출
                        gambling_score = predictions[0, 1].item()
                        # 소수점 2자리까지만 표시
                        gambling_score = round(gambling_score, 2)
                    
                    return gambling_score
                except Exception as e:
                    print(f"텍스트 '{text[:30]}...' 분석 중 오류 발생: {e}")
                    return 0.0
            
            # 그래프의 각 노드에 불법도박 점수 추가
            node_count = len(graph.nodes())
            print(f"총 {node_count}개 노드의 텍스트를 분석합니다...")
            
            # 테스트를 위해 최초 5개 노드의 텍스트와 점수 출력
            test_samples = 0
            non_zero_scores = 0
            
            for i, node_id in enumerate(graph.nodes()):
                text = graph.nodes[node_id].get('text', '')
                if text:
                    gambling_score = get_gambling_score(text)
                    graph.nodes[node_id]['gambling_score'] = gambling_score
                    
                    # 0이 아닌 점수 개수 세기
                    if gambling_score > 0.0:
                        non_zero_scores += 1
                    
                    # 처음 5개 샘플 출력
                    if test_samples < 5 and len(text) > 2:
                        print(f"샘플 {test_samples+1}: 텍스트='{text[:50]}...', 점수={gambling_score:.4f}")
                        test_samples += 1
                    
                    # 진행 상황 출력 (10% 단위)
                    if (i+1) % max(1, node_count // 10) == 0 or (i+1) == node_count:
                        print(f"진행 상황: {i+1}/{node_count} 노드 처리 완료 ({(i+1)/node_count:.1%})")
            
            print(f"모든 노드 분석 완료: {non_zero_scores}/{node_count} 노드에 0이 아닌 점수가 할당되었습니다 ({non_zero_scores/node_count:.1%})")
            return graph
            
        except Exception as e:
            import traceback
            print(f"불법도박 점수 추가 중 오류 발생: {e}")
            print("상세 오류:")
            traceback.print_exc()
            return graph  # 오류 발생해도 원본 그래프 반환

    def save_scores_as_json(self, graph, filename):
        """
        그래프의 점수 정보와 엣지 정보를 포함하여 JSON 파일로 저장합니다.
        
        Args:
            graph (networkx.Graph): 저장할 그래프
            filename (str): 파일명
            
        Returns:
            str: 저장된 파일 경로 또는 None
        """
        # 완전한 그래프 구조를 저장하기 위해 기존 save_graph_as_json 메소드 사용
        return self.save_graph_as_json(graph, filename)

def main():
    """
    메인 실행 함수: 모든 URL에서 완전한 HTML을 크롤링하고 저장합니다.
    """
    handler = HTMLHandler()
    
    # 모델 경로 확인 (사용자 지정 또는 기본값)
    model_path = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/gambling_bert_model"
    if not os.path.exists(model_path):
        print(f"지정된 모델 경로 '{model_path}'가 존재하지 않습니다.")
        model_path = None  # add_gambling_scores_to_graph 함수에서 대체 경로를 찾도록 함
    
    # 파일 카운터 초기화 (전역 변수에서 시작 번호 가져옴)
    file_counter = START_FILE_NUMBER
    
    for url in urls:
        print(f"\n=== {url} 크롤링 시작 ===")
        file_path, html_content = handler.crawl_url(url)
        if file_path:
            print(f"HTML 파일이 '{file_path}'에 저장되었습니다.")
            
            # HTML을 그래프로 변환
            graph = handler.html_to_graph(html_content=html_content)
            if graph:
                # 불법도박 점수 추가
                print("불법도박 점수 분석 시작...")
                graph = handler.add_gambling_scores_to_graph(graph, model_path=model_path)
                
                # 파일명 생성 (일련번호 기반)
                filename = f"{file_counter}.json"
                file_counter += 1
                
                # 점수 정보와 엣지 정보를 함께 JSON으로 저장
                handler.save_scores_as_json(graph, filename)
        else:
            print(f"URL '{url}' 처리 실패")
    
    print("\n크롤링 완료!")

if __name__ == "__main__":
    main()

