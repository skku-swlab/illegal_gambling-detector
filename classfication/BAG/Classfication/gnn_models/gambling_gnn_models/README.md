# 불법도박사이트 HTML 구조 분석 GAT 모델

이 모듈은 불법도박사이트의 HTML DOM 트리 구조를 분석하여 불법도박사이트인지 예측하는 GNN(Graph Neural Network) 모델을 제공합니다. 특히 GAT(Graph Attention Network) 모델을 사용하여 도박 관련 키워드가 포함된 노드에 더 많은 가중치를 부여하는 접근 방식을 채택했습니다.

## 주요 특징

- HTML DOM 트리를 그래프로 변환하여 분석
- 도박 관련 키워드에 대한 점수(`gambling_score`)를 어텐션 메커니즘에 활용
- One-class classification 방식으로 학습
- PyTorch Geometric 프레임워크 기반

## 파일 구조

- `dataset.py`: 데이터셋 클래스 (JSON -> PyTorch Geometric)
- `model.py`: GAT 모델 구현 (gambling_score를 어텐션에 적용)
- `train.py`: 모델 학습 코드
- `inference.py`: HTML 파일에 대한 추론 코드

## 설치 요구사항

```bash
pip install torch torch-geometric networkx numpy matplotlib tqdm scikit-learn
```

## 사용 방법

### 1. 모델 학습

```bash
python train.py
```

설정 값은 `train.py` 파일 내 `main()` 함수에 하드코딩되어 있습니다:
- `data_dir`: JSON 데이터셋 경로 (기본값: `/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset`)
- `epochs`: 학습 에폭 수 (기본값: 50)
- `batch_size`: 배치 크기 (기본값: 16)
- `hidden_dim`: 은닉층 차원 (기본값: 64)
- `num_heads`: 어텐션 헤드 수 (기본값: 8)
- `save_dir`: 모델 저장 경로 (기본값: `/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/gnn_models/gnn_logs`)

설정을 변경하려면 `train.py` 파일의 `main()` 함수 내 값을 직접 수정하세요.

### 2. 모델 추론

```bash
python inference.py
```

설정 값은 `inference.py` 파일 내 `main()` 함수에 하드코딩되어 있습니다:
- `input_path`: 입력 HTML 파일 또는 디렉토리 경로 (기본값: `/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/html_files`)
- `model_path`: 학습된 모델 파일 경로 (기본값: `/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/gnn_models/gnn_logs/models/gambling_gat_final.pt`)
- `output_path`: 결과 저장 경로 (기본값: `gnn_inference_results.json`)
- `threshold`: 분류 임계값 (기본값: 0.5)

설정을 변경하려면 `inference.py` 파일의 `main()` 함수 내 값을 직접 수정하세요.

## 모델 상세 구조

GAT(Graph Attention Network) 모델은 각 노드의 특성과 이웃 정보를 활용하여 그래프를 분석합니다. 특히 본 모델은 다음과 같은 구조를 가집니다:

1. **커스텀 GAT 레이어**: `gambling_score`를 어텐션 가중치에 곱하여 도박 관련 키워드가 있는 노드에 더 많은 가중치를 부여
2. **2-계층 구조**: 2개의 GAT 레이어와 최종 분류 레이어
3. **그래프 풀링**: 노드 레벨 특성을 그래프 레벨 특성으로 변환

## 주의 사항

- 학습 데이터는 모두 불법도박사이트이므로 one-class classification을 사용합니다.
- `gambling_score`는 HTML 태그에 포함된 도박 키워드의 빈도 또는 관련성을 나타냅니다.
- 과적합이 문제가 되지 않으며, 오히려 불법도박사이트 패턴을 정확히 학습하기 위해 권장됩니다. 