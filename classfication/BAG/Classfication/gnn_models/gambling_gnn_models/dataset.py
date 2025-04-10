import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torch_geometric.data as geo_data
from torch_geometric.data import Batch
import networkx as nx
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class GamblingGNNDataset(Dataset):
    def __init__(self, base_dir, test_size=0.2, seed=42):
        """
        데이터셋 클래스: 불법도박사이트와 정상 사이트의 HTML DOM 트리 데이터를 처리
        
        Args:
            base_dir (str): 데이터셋 기본 디렉토리 경로 (illegal과 normal 서브디렉토리 포함)
            test_size (float): 테스트 세트 비율
            seed (int): 무작위 시드
        """
        self.base_dir = base_dir
        self.test_size = test_size
        self.seed = seed
        
        # 불법 도박 사이트와 정상 사이트 폴더 경로
        illegal_dir = os.path.join(base_dir, 'illegal')
        normal_dir = os.path.join(base_dir, 'normal')
        
        # 불법 도박 사이트 파일 경로 (라벨 1)
        illegal_files = [os.path.join(illegal_dir, f) for f in os.listdir(illegal_dir) if f.endswith('.json')]
        
        # 정상 사이트 파일 경로 (라벨 0)
        normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.json')]
        
        # 각 파일 경로에 레이블 정보 추가 (튜플로 저장)
        self.illegal_file_paths = [(path, 1) for path in illegal_files]  # 불법 도박 사이트: 1
        self.normal_file_paths = [(path, 0) for path in normal_files]    # 정상 사이트: 0
        
        # 모든 파일 경로 통합
        self.file_paths = self.illegal_file_paths + self.normal_file_paths
        
        # 데이터 균형 확인
        print(f"도박 사이트 파일 수: {len(self.illegal_file_paths)}")
        print(f"정상 사이트 파일 수: {len(self.normal_file_paths)}")
        
        # 학습 및 테스트 세트 분리
        self.train_files, self.test_files = train_test_split(
            self.file_paths, test_size=test_size, random_state=seed, stratify=[item[1] for item in self.file_paths]
        )
        
        # 데이터 캐시 (메모리 효율성을 위해)
        self.data_cache = {}
        
        print(f"총 파일 수: {len(self.file_paths)}")
        print(f"학습 파일 수: {len(self.train_files)}")
        print(f"테스트 파일 수: {len(self.test_files)}")
    
    def __len__(self):
        return len(self.train_files)
    
    def __getitem__(self, idx):
        # 캐시에 있으면 캐시된 데이터 반환
        if idx in self.data_cache:
            return self.data_cache[idx]
        
        # 파일 경로와 레이블 가져오기
        file_path, label = self.train_files[idx]
        
        # JSON 파일 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # PyTorch Geometric 데이터로 변환 (레이블 정보 추가)
        graph_data = self._convert_to_pytorch_geometric(data, label)
        
        # 캐시에 저장
        self.data_cache[idx] = graph_data
        
        return graph_data
    
    def _convert_to_pytorch_geometric(self, data, label):
        """JSON 데이터를 PyTorch Geometric 형식으로 변환"""
        nodes = data['nodes']
        edges = data['edges']
        
        # 노드 특성 추출
        node_features = []
        gambling_scores = []
        
        for node in nodes:
            # 노드 특성 벡터 생성
            # 1. Tag one-hot encoding (간단화를 위해 여기서는 임베딩으로 처리)
            # 2. gambling_score 값 (이 값은 별도로 추출)
            features = [
                len(node['text']) if node['text'] else 0,  # 텍스트 길이
                1 if node['node_type'] == 'tag' else 0,    # 노드 타입 (tag=1, 기타=0)
            ]
            
            # 특성 벡터에 추가
            node_features.append(features)
            
            # Gambling score 추출 (GAT 어텐션 가중치로 사용)
            gambling_scores.append(node['gambling_score'])
        
        # 간선 리스트 추출
        edge_index = []
        for edge in edges:
            source = edge['source']
            target = edge['target']
            edge_index.append([source, target])
            # 양방향 그래프로 만들기 (선택 사항)
            edge_index.append([target, source])
        
        # NumPy 배열로 변환
        node_features = np.array(node_features, dtype=np.float32)
        gambling_scores = np.array(gambling_scores, dtype=np.float32)
        edge_index = np.array(edge_index, dtype=np.int64).T  # PyG 형식에 맞춰 전치
        
        # PyTorch 텐서로 변환
        x = torch.FloatTensor(node_features)
        gambling_scores = torch.FloatTensor(gambling_scores)
        edge_index = torch.LongTensor(edge_index)
        
        # 레이블 설정 (0: 정상 사이트, 1: 불법 도박 사이트)
        y = torch.tensor([float(label)])
        
        # PyTorch Geometric 데이터 객체 생성
        graph_data = geo_data.Data(
            x=x,
            edge_index=edge_index,
            gambling_scores=gambling_scores,
            y=y
        )
        
        return graph_data
    
    def get_test_dataset(self):
        """테스트 데이터셋 생성"""
        test_data_list = []
        
        for file_path, label in tqdm(self.test_files, desc="Loading test data"):
            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # PyTorch Geometric 데이터로 변환 (레이블 정보 추가)
            graph_data = self._convert_to_pytorch_geometric(data, label)
            test_data_list.append(graph_data)
        
        return test_data_list
    
    @staticmethod
    def collate_fn(batch):
        """배치 데이터 처리 함수"""
        return Batch.from_data_list(batch) 