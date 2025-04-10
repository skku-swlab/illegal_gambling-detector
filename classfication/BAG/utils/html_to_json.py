import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import html as html_lib
from bs4 import BeautifulSoup
import networkx as nx

# 저장할 파일 시작 번호 (사용자가 이 값을 변경할 수 있음)
START_FILE_NUMBER = 1

# BERT 모델 경로
MODEL_PATH = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/bert/gambling_bert_model"

# 그래프 JSON 파일 저장 디렉토리
GRAPH_SAVE_DIR = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset/normal"

def html_to_graph(html_content):
    """
    HTML 내용을 그래프 구조로 변환합니다.
    
    Args:
        html_content (str): HTML 내용 문자열
        
    Returns:
        networkx.Graph: HTML 구조를 표현하는 그래프
    """
    # BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # HTML 태그가 없는 경우 처리
    if soup.html is None:
        print("경고: 유효한 HTML 문서가 아닙니다. 루트 HTML 태그가 없습니다.")
        # 빈 그래프 생성
        G = nx.DiGraph()
        # 루트 노드 추가 (최소한의 노드 필요)
        G.add_node(0, tag="root", text=html_content[:100] + "..." if len(html_content) > 100 else html_content, node_type='tag')
        return G
    
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

def add_gambling_scores_to_graph(graph, model_path=None):
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
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            print(f"오류: 기본 모델 경로 '{model_path}'가 존재하지 않습니다.")
            # 현재 디렉토리에서 모델 경로 확인
            current_dir_model = "./gambling_bert_model"
            if os.path.exists(current_dir_model):
                model_path = current_dir_model
                print(f"대체 모델 경로 '{model_path}'를 사용합니다.")
            else:
                print(f"오류: 모델 경로 '{model_path}'가 존재하지 않습니다.")
                return graph
    
    try:
        # 모델 경로 확인
        if not os.path.exists(model_path):
            print(f"오류: 모델 경로 '{model_path}'가 존재하지 않습니다.")
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
        return graph

def graph_to_json_data(graph):
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

def save_graph_as_json(graph, filename):
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
        graph_data = graph_to_json_data(graph)
        
        # 출력 파일 경로 생성
        output_file = os.path.join(GRAPH_SAVE_DIR, filename)
        
        # JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"그래프가 JSON 파일 '{output_file}'로 저장되었습니다.")
        return output_file
    except Exception as e:
        print(f"그래프 저장 중 오류 발생: {e}")
        return None

def process_html_file(html_file_path, output_json_path):
    """
    HTML 파일을 처리하여 그래프로 변환하고 점수를 추가한 후 JSON으로 저장합니다.
    
    Args:
        html_file_path (str): 처리할 HTML 파일 경로
        output_json_path (str): 저장할 JSON 파일 경로
        
    Returns:
        str: 저장된 JSON 파일 경로 또는 None
    """
    try:
        # 출력 디렉토리 확인
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # HTML 파일 읽기
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # HTML 내용이 비어있는지 확인
        if not html_content.strip():
            print(f"경고: HTML 파일 '{html_file_path}'이 비어 있거나 공백만 포함하고 있습니다.")
            return None
        
        # HTML을 그래프로 변환
        try:
            graph = html_to_graph(html_content)
        except Exception as e:
            print(f"HTML 파싱 중 오류 발생: {e}")
            print("기본 그래프를 생성합니다.")
            graph = nx.DiGraph()
            graph.add_node(0, tag="root", text=html_content[:100] + "..." if len(html_content) > 100 else html_content, node_type='tag')
            
        if graph is None:
            print(f"HTML 파일 '{html_file_path}'을 그래프로 변환하는 데 실패했습니다.")
            return None
            
        # 불법도박 점수 추가
        print("불법도박 점수 분석 시작...")
        graph = add_gambling_scores_to_graph(graph)
        
        # 그래프를 JSON으로 직렬화
        graph_data = graph_to_json_data(graph)
        
        # JSON 파일로 저장
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"그래프가 JSON 파일 '{output_json_path}'로 저장되었습니다.")
        return output_json_path
        
    except Exception as e:
        print(f"HTML 파일 '{html_file_path}' 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    메인 실행 함수: 지정된 디렉토리 아래의 모든 HTML 파일을 처리합니다.
    """
    # HTML 파일이 있는 디렉토리 경로
    html_dir = "/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/crawled_html_files/normal"
    
    # 디렉토리 내의 모든 HTML 파일 찾기
    html_files = [os.path.join(html_dir, f) for f in os.listdir(html_dir) if f.endswith('.html')]
    
    if not html_files:
        print(f"오류: '{html_dir}' 디렉토리에서 HTML 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(html_files)}개의 HTML 파일을 찾았습니다.")
    
    # 파일 번호 초기화
    current_file_number = START_FILE_NUMBER
    
    # 각 HTML 파일 처리
    for html_file_path in html_files:
        print(f"\n파일 처리 시작: {html_file_path}")
        # 파일 번호를 문자열로 변환하여 JSON 파일 경로 생성
        output_json_path = os.path.join(GRAPH_SAVE_DIR, f"{current_file_number}.json")
        process_html_file(html_file_path, output_json_path)
        current_file_number += 1

if __name__ == "__main__":
    main() 