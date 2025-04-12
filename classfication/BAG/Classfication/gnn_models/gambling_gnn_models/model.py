import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from torch_geometric.typing import OptTensor, Adj


class GamblingGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0.0, gambling_weight=1.0):
        super(GamblingGATConv, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.gambling_weight = gambling_weight  # gambling_score에 적용할 가중치
        
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
    
    def forward(self, x, edge_index, gambling_scores, return_attention_weights=False):
        # 노드 특성을 선형 변환
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # 메시지 전달
        out = self.propagate(edge_index, x=x, gambling_scores=gambling_scores)
        
        # 결과 처리
        out = out.mean(dim=1)  # 멀티헤드 어텐션 결과의 평균
        
        return out
    
    def message(self, x_j, x_i, edge_index_i, edge_index, gambling_scores):
        # edge_index: [2, num_edges] -> edge_index[0]은 소스 노드, edge_index[1]은 타겟 노드
        
        # 어텐션 계수 계산
        alpha_src = (x_i * self.att_src).sum(dim=-1)  # [num_edges, heads]
        alpha_dst = (x_j * self.att_dst).sum(dim=-1)  # [num_edges, heads]
        alpha = alpha_src + alpha_dst  # [num_edges, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # 소프트맥스 적용 (기본 어텐션 계산)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))  # [num_edges, heads]
        
        # 불법도박 스코어 적용
        # 타겟 노드의 불법도박 스코어를 가져옴
        src_nodes = edge_index[0]  # 소스 노드 인덱스
        src_gambling_scores = gambling_scores[src_nodes]  # 소스 노드의 도박 스코어
        
        # 도박 스코어를 헤드 수에 맞게 확장
        src_gambling_scores = src_gambling_scores.view(-1, 1)  # [num_edges, 1]
        src_gambling_scores = src_gambling_scores.expand(-1, self.heads)  # [num_edges, heads]
        
        # 어텐션에 도박 스코어 적용 (곱셈) - gambling_weight를 가중치로 사용
        # gambling_weight가 높을수록 gambling_score의 영향이 커짐
        weighted_gambling_scores = 1.0 + (self.gambling_weight * src_gambling_scores)  # 기본값 1에 가중치 부여된 도박 스코어 추가
        alpha = alpha * weighted_gambling_scores  # [num_edges, heads]
        
        # 드롭아웃 적용
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 가중치가 적용된 메시지 반환
        return x_j * alpha.unsqueeze(-1)  # [num_edges, heads, out_channels]


class GamblingGATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_heads=8, dropout=0.2, gambling_weight=1.0):
        super(GamblingGATModel, self).__init__()
        
        # GAT 레이어
        self.gat1 = GamblingGATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, gambling_weight=gambling_weight)
        self.gat2 = GamblingGATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, gambling_weight=gambling_weight)
        self.gat3 = GamblingGATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, gambling_weight=gambling_weight)
        
        # 잔차 연결을 위한 투영 레이어
        self.proj1 = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 레이어 정규화만 사용 (배치 정규화 제거)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # 도박 스코어 임베딩 레이어
        self.gambling_embedding = nn.Linear(1, hidden_dim)
        
        # 최종 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        self.dropout_rate = dropout
    
    def forward(self, data):
        x, edge_index, gambling_scores = data.x, data.edge_index, data.gambling_scores
        
        # 도박 스코어 임베딩
        gambling_emb = self.gambling_embedding(gambling_scores.unsqueeze(-1))
        
        # 첫 번째 레이어
        identity = x
        x = self.gat1(x, edge_index, gambling_scores)
        x = self.norm1(x + gambling_emb)  # 도박 스코어 정보 추가
        if self.proj1 is not None:
            identity = self.proj1(identity)
        x = x + identity  # 잔차 연결
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 두 번째 레이어
        identity = x
        x = self.gat2(x, edge_index, gambling_scores)
        x = self.norm2(x + gambling_emb)  # 도박 스코어 정보 추가
        identity = self.proj2(identity)
        x = x + identity  # 잔차 연결
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 세 번째 레이어
        identity = x
        x = self.gat3(x, edge_index, gambling_scores)
        x = self.norm3(x + gambling_emb)  # 도박 스코어 정보 추가
        identity = self.proj3(identity)
        x = x + identity  # 잔차 연결
        x = F.relu(x)
        
        # 그래프 레벨 표현
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # 분류
        x = self.classifier(x)
        
        return x 