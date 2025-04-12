import os
import json
import torch
from model import GamblingGATModel
from torch_geometric.data import Data

# 모델 로드 함수
def load_model(model_path, input_dim, hidden_dim, num_heads, dropout, gambling_weight, device):
    """
    업데이트된 모델 구조로 모델을 로드합니다.
    3개의 GAT 레이어와 레이어 정규화를 포함합니다.
    """
    model = GamblingGATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_heads=num_heads,
        dropout=dropout,
        gambling_weight=gambling_weight
    ).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드로 설정
    return model

# JSON 파일을 PyTorch Geometric 데이터로 변환하는 함수
def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    edges = data['edges']
    
    node_features = []
    gambling_scores = []
    
    for node in nodes:
        features = [
            len(node['text']) if node['text'] else 0,
            1 if node['node_type'] == 'tag' else 0,
        ]
        node_features.append(features)
        gambling_scores.append(node['gambling_score'])
    
    edge_index = []
    for edge in edges:
        source = edge['source']
        target = edge['target']
        edge_index.append([source, target])
        edge_index.append([target, source])
    
    x = torch.FloatTensor(node_features)
    gambling_scores = torch.FloatTensor(gambling_scores)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, gambling_scores=gambling_scores)

# 추론 함수
def inference(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        return output.item()

# 메인 함수
def main():
    # 설정 값
    model_path = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/gnn_models/models/model_6/gambling_gat_binary_20250412_045321_final.pt'  # 저장된 모델 경로 (binary 모델 사용)
    file_path = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset/illegal/103.json'  # 추론할 파일 경로
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 데이터 로드
    data = load_data_from_json(file_path)

    # 모델 하이퍼파라미터 설정 (train.py와 동일하게 유지)
    input_dim = data.x.shape[1]  # 노드 특성 차원
    hidden_dim = 64
    num_heads = 8
    dropout = 0.3  # 학습 시와 동일한 드롭아웃 비율 사용
    gambling_weight = 100.0  # 학습 시와 동일한 gambling_weight 값 사용
    
    # 모델 로드 - 업데이트된 load_model 함수 사용
    print('모델 로드 중...')
    model = load_model(model_path, input_dim, hidden_dim, num_heads, dropout, gambling_weight, device)
    print('모델 로드 완료')

    # 추론 수행
    print('추론 시작...')
    prediction_score = inference(model, data, device)
    
    # 결과 해석 (이진 분류)
    is_gambling_site = prediction_score > 0.5
    
    # 결과 출력
    print('추론 완료!')
    print(f'예측 점수: {prediction_score:.4f}')
    print(f'해석: {"불법 도박 사이트" if is_gambling_site else "정상 사이트"}')
    print(f'확률: {"%.2f" % (prediction_score * 100 if is_gambling_site else (1 - prediction_score) * 100)}%')

if __name__ == "__main__":
    main()
