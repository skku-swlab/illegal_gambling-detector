import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
from bs4 import BeautifulSoup
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch_geometric.data as geo_data
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
import random
import re

# DOM 트리 파싱을 위한 클래스
class HTMLToGraphConverter:
    def __init__(self, max_nodes=1000):
        self.max_nodes = max_nodes
        self.node_types = ['div', 'a', 'span', 'p', 'img', 'h1', 'h2', 'h3', 'ul', 'li', 'table', 'tr', 'td', 'input', 'button', 'form', 'other']
        self.node_type_to_idx = {node_type: idx for idx, node_type in enumerate(self.node_types)}
    
    def html_to_graph(self, html_content):
        # HTML 파싱
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            print(f"HTML 파싱 오류: {e}")
            return None
        
        # 노드와 엣지 초기화
        nodes = []
        edges = []
        node_map = {}  # HTML 요소와 인덱스 매핑
        
        # DFS로 DOM 트리 순회
        def traverse_dom(element, parent_idx=None):
            if len(nodes) >= self.max_nodes:
                return
            
            # 현재 노드 인덱스
            current_idx = len(nodes)
            node_map[element] = current_idx
            
            # 노드 타입 결정
            tag_name = element.name.lower() if element.name else 'other'
            if tag_name not in self.node_type_to_idx:
                tag_name = 'other'
            
            # 노드 특성 추출
            node_features = {
                'tag': tag_name,
                'text': element.get_text(strip=True)[:100] if element.string else "",
                'attributes': {k: v for k, v in element.attrs.items() if k in ['class', 'id', 'href', 'src']}
            }
            
            # 노드 추가
            nodes.append(node_features)
            
            # 부모-자식 엣지 추가
            if parent_idx is not None:
                edges.append((parent_idx, current_idx))
            
            # 자식 노드들 처리
            for child in element.children:
                if child.name:  # 실제 HTML 태그인 경우만 처리
                    traverse_dom(child, current_idx)
        
        # 루트 요소부터 DOM 트리 순회
        body = soup.body if soup.body else soup
        traverse_dom(body)
        
        # 노드 특성을 원-핫 인코딩으로 변환
        node_features = []
        for node in nodes:
            # 노드 타입 원-핫 인코딩
            tag_idx = self.node_type_to_idx[node['tag']] if node['tag'] in self.node_type_to_idx else self.node_type_to_idx['other']
            tag_one_hot = [0] * len(self.node_types)
            tag_one_hot[tag_idx] = 1
            
            # 텍스트 길이 (정규화)
            text_len = min(len(node['text']), 100) / 100.0
            
            # 속성 수 (정규화)
            attr_count = min(len(node['attributes']), 10) / 10.0
            
            # 특성 벡터 조합
            features = tag_one_hot + [text_len, attr_count]
            node_features.append(features)
        
        # 엣지 인덱스 변환 (PyTorch Geometric 형식)
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        
        return {
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_index': edge_index
        }


# WebSight 데이터셋 로더
class WebSightDataset(Dataset):
    def __init__(self, dataset, split='train', test_size=0.2, seed=42, max_nodes=1000):
        self.dataset = dataset[split]
        self.converter = HTMLToGraphConverter(max_nodes=max_nodes)
        self.data_cache = {}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 캐시에 있으면 캐시된 데이터 반환
        if idx in self.data_cache:
            return self.data_cache[idx]
        
        # HTML 내용 가져오기
        html_content = self.dataset[idx]['html']
        
        # HTML을 그래프로 변환
        graph_data = self.converter.html_to_graph(html_content)
        
        if graph_data is None:
            # 변환 실패 시 더미 데이터 반환
            dummy_features = np.zeros((1, len(self.converter.node_types) + 2), dtype=np.float32)
            dummy_edge_index = np.zeros((2, 0), dtype=np.int64)
            
            graph_data = {
                'node_features': dummy_features,
                'edge_index': dummy_edge_index
            }
        
        # PyTorch Geometric Data 객체 생성
        x = torch.FloatTensor(graph_data['node_features'])
        edge_index = torch.LongTensor(graph_data['edge_index'])
        
        # 자기 자신을 예측하는 자기지도학습을 위한 타겟 설정
        # 노드 특성의 일부를 마스킹하고 그것을 예측
        masked_x = x.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        
        # 20%의 노드 마스킹
        for i in range(len(x)):
            if random.random() < 0.2:
                # 태그 정보만 보존하고 나머지는 마스킹
                mask[i, len(self.converter.node_types):] = True
                masked_x[i, len(self.converter.node_types):] = 0
        
        data = geo_data.Data(
            x=masked_x,
            edge_index=edge_index,
            y=x,  # 원본 특성이 타겟
            mask=mask  # 마스킹된 위치
        )
        
        # 캐시에 저장
        self.data_cache[idx] = data
        
        return data
    
    @staticmethod
    def collate_fn(batch):
        """배치 데이터 처리 함수"""
        return Batch.from_data_list(batch)


# GAT 모델 정의
class PretrainGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0.0):
        super(PretrainGATConv, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # 가중치 행렬
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # 어텐션 메커니즘을 위한 가중치
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
    
    def forward(self, x, edge_index, return_attention_weights=False):
        # 노드 특성을 선형 변환
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # 메시지 전달
        out = self.propagate(edge_index, x=x)
        
        # 결과 처리
        out = out.mean(dim=1)  # 멀티헤드 어텐션 결과의 평균
        
        return out
    
    def message(self, x_j, x_i, edge_index_i, edge_index):
        # 어텐션 계수 계산
        alpha_src = (x_i * self.att_src).sum(dim=-1)  # [num_edges, heads]
        alpha_dst = (x_j * self.att_dst).sum(dim=-1)  # [num_edges, heads]
        alpha = alpha_src + alpha_dst  # [num_edges, heads]
        alpha = nn.functional.leaky_relu(alpha, self.negative_slope)
        
        # 소프트맥스 적용
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))  # [num_edges, heads]
        
        # 드롭아웃 적용
        alpha = nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        
        # 가중치가 적용된 메시지 반환
        return x_j * alpha.unsqueeze(-1)  # [num_edges, heads, out_channels]


class PretrainGATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=8, dropout=0.2):
        super(PretrainGATModel, self).__init__()
        
        # GAT 레이어
        self.gat1 = PretrainGATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = PretrainGATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
        
        # 노드 특성 복원 레이어
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, data):
        x, edge_index, mask = data.x, data.edge_index, data.mask
        
        # GAT 레이어 통과
        x = self.gat1(x, edge_index)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index)
        x = nn.functional.relu(x)
        
        # 노드 특성 복원
        x = self.node_predictor(x)
        
        return x, mask


# 모델 학습 함수
def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
        # 데이터를 디바이스로 이동
        batch = batch.to(device)
        
        # 순전파
        optimizer.zero_grad()
        outputs, mask = model(batch)
        
        # 손실 계산 (마스킹된 부분에 대해서만)
        target = batch.y
        
        # MSE 손실 (마스킹된 부분만)
        loss = nn.MSELoss()(outputs[mask], target[mask])
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch} - Training Loss: {avg_loss:.4f}')
    
    return avg_loss


# 평가 함수
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 데이터를 디바이스로 이동
            batch = batch.to(device)
            
            # 순전파
            outputs, mask = model(batch)
            
            # 손실 계산
            target = batch.y
            loss = nn.MSELoss()(outputs[mask], target[mask])
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')
    
    return avg_loss


# 학습 결과 시각화
def plot_learning_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # 설정 값
    batch_size = 16
    epochs = 50
    lr = 0.001
    weight_decay = 1e-5
    hidden_dim = 64
    num_heads = 8
    dropout = 0.2
    save_dir = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/gnn_models/gnn_logs/pretrain'
    save_every = 5
    seed = 42
    
    # 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 데이터셋 저장 경로 지정 (외부 저장소 사용)
    custom_cache_dir = '/home/swlab/external_datasets/huggingface_cache'
    os.makedirs(custom_cache_dir, exist_ok=True)
    print(f"데이터셋 캐시 경로: {custom_cache_dir}")
    
    # 데이터셋 로드
    print("WebSight 데이터셋 로드 중...")
    websight_dataset = load_dataset("HuggingFaceM4/WebSight", cache_dir=custom_cache_dir)
    
    # 데이터셋 분할
    train_dataset = WebSightDataset(websight_dataset, split='train')
    val_dataset = WebSightDataset(websight_dataset, split='validation')
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=4
    )
    
    # 모델 초기화
    input_dim = len(train_dataset.converter.node_types) + 2  # 노드 특성 차원
    model = PretrainGATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    # 최적화 도구 설정
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 타임스탬프로 실험 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"html_pretrain_gat_{timestamp}"
    
    # 학습 시작
    print("사전학습 시작...")
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 검증
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # 모델 저장 (일정 주기마다)
        if epoch % save_every == 0 or epoch == epochs:
            model_path = os.path.join(save_dir, f"{experiment_name}_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, model_path)
            print(f"모델 저장됨: {model_path}")
    
    # 학습 결과 시각화
    loss_plot_path = os.path.join(save_dir, f"{experiment_name}_loss.png")
    plot_learning_curve(train_losses, val_losses, loss_plot_path)
    
    # 최종 모델 저장
    final_model_path = os.path.join(save_dir, f"{experiment_name}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'node_types': train_dataset.converter.node_types
    }, final_model_path)
    print(f"최종 모델 저장됨: {final_model_path}")
    
    print("사전학습 완료!")


if __name__ == "__main__":
    main()
