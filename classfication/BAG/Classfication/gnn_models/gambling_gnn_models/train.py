import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, classification_report

# 커스텀 모듈 임포트
from model import GamblingGATModel
from dataset import GamblingGNNDataset

# 결과와 모델을 저장할 디렉토리 생성
def create_save_dirs():
    # 모델 저장 경로
    model_save_dir = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/gnn_models/models'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 결과 저장 경로
    results_save_dir = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/Classfication/gnn_models/gnn_logs'
    os.makedirs(results_save_dir, exist_ok=True)
    
    return model_save_dir, results_save_dir

# 모델 학습 함수
def train(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
        # 데이터를 디바이스로 이동
        batch = batch.to(device)
        
        # 순전파
        optimizer.zero_grad()
        outputs = model(batch)
        
        # 실제 레이블 (0: 정상, 1: 불법 도박 사이트)
        # 타겟 텐서의 형태를 모델 출력과 일치시키기 위해 reshape
        target = batch.y.view(-1, 1)
        
        # 손실 계산 (이진 분류)
        loss = nn.BCELoss()(outputs, target)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 예측 값과 레이블 저장 (평가 지표 계산용)
        all_predictions.extend(outputs.detach().cpu().numpy())
        all_labels.extend(target.cpu().numpy())
    
    # 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    
    # 성능 지표 계산
    binary_preds = np.array(all_predictions) > 0.5
    binary_labels = np.array(all_labels) > 0.5
    
    accuracy = accuracy_score(binary_labels.flatten(), binary_preds.flatten())
    precision = precision_score(binary_labels.flatten(), binary_preds.flatten())
    recall = recall_score(binary_labels.flatten(), binary_preds.flatten())
    f1 = f1_score(binary_labels.flatten(), binary_preds.flatten())
    
    try:
        auc_score = roc_auc_score(all_labels, all_predictions)
    except:
        auc_score = 0.0
    
    # 특이도(specificity) 계산
    tn, fp, fn, tp = confusion_matrix(binary_labels.flatten(), binary_preds.flatten(), labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f'Epoch {epoch}:')
    print(f'  Training Loss: {avg_loss:.4f}')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall(Sensitivity): {recall:.4f}')
    print(f'  Specificity: {specificity:.4f}')
    print(f'  F1 Score: {f1:.4f}')
    print(f'  AUC: {auc_score:.4f}')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc_score
    }

# 테스트 함수
def test(model, test_data, device):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in tqdm(test_data, desc='Testing'):
            # 데이터를 디바이스로 이동
            data = data.to(device)
            
            # 배치 속성 확인 및 필요시 추가
            if not hasattr(data, 'batch') or data.batch is None:
                # 단일 그래프에 대한 배치 속성 추가 (필요하지 않지만 일관성을 위해)
                data.batch = None
            
            # 예측
            output = model(data)
            
            # 예측값과 실제 레이블 저장
            if hasattr(data, 'y') and data.y is not None:
                predictions.append(output.item())
                labels.append(data.y.item())
            else:
                print("경고: 데이터에 레이블(y)이 없습니다.")
    
    # 이진 분류를 위한 임계값 적용
    binary_preds = np.array(predictions) > 0.5
    binary_labels = np.array(labels) > 0.5
    
    # 성능 지표 계산
    accuracy = accuracy_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds)
    recall = recall_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds)
    confusion = confusion_matrix(binary_labels, binary_preds)
    
    # 특이도(specificity) 계산
    tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 분류 보고서 생성
    class_report = classification_report(binary_labels, binary_preds, target_names=['Normal', 'Gambling'])
    
    try:
        auc_score = roc_auc_score(labels, predictions)
    except:
        auc_score = 0.0
    
    print("\n테스트 결과:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall(Sensitivity): {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print("  혼동 행렬:")
    print(confusion)
    print("\n분류 보고서:")
    print(class_report)
    
    return {
        'predictions': predictions,
        'true_labels': labels,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': confusion,
        'classification_report': class_report
    }

# 학습 결과 시각화
def plot_learning_curve(metrics, save_dir, experiment_name):
    epochs = range(1, len(metrics['loss']) + 1)
    
    # 손실 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_loss.png"))
    plt.close()
    
    # 정확도 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_accuracy.png"))
    plt.close()
    
    # 정밀도 및 재현율 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['precision'], label='Precision')
    plt.plot(epochs, metrics['recall'], label='Recall')
    plt.plot(epochs, metrics['specificity'], label='Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and Specificity Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_prec_recall.png"))
    plt.close()
    
    # F1 스코어 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_f1.png"))
    plt.close()
    
    # AUC 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['auc'], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_auc.png"))
    plt.close()
    
    # 모든 지표를 하나의 그래프에 표시
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics['accuracy'], label='Accuracy')
    plt.plot(epochs, metrics['precision'], label='Precision')
    plt.plot(epochs, metrics['recall'], label='Recall')
    plt.plot(epochs, metrics['specificity'], label='Specificity')
    plt.plot(epochs, metrics['f1'], label='F1 Score')
    plt.plot(epochs, metrics['auc'], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_all_metrics.png"))
    plt.close()

# Early Stopping 클래스 추가
class EarlyStopping:
    """
    Early Stopping 클래스
    검증 손실이 patience 에폭 동안 개선되지 않으면 학습을 조기 중단합니다.
    """
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pt'):
        """
        초기화 함수
        
        Args:
            patience (int): 성능 개선이 없어도 기다릴 에폭 수
            min_delta (float): 개선으로 간주하기 위한 최소 변화량
            path (str): 최적 모델을 저장할 경로
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        """
        Early stopping 검사 수행
        
        Args:
            val_loss (float): 현재 검증 손실
            model (nn.Module): 현재 모델
            
        Returns:
            bool: 학습 중단 여부
        """
        score = -val_loss  # 더 낮은 손실이 더 좋은 점수
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, model):
        """
        모델의 체크포인트를 저장합니다.
        
        Args:
            model (nn.Module): 저장할 모델
        """
        torch.save(model.state_dict(), self.path)
        print(f'검증 손실 개선됨. 모델 저장: {self.path}')

# 검증 함수 추가
def validate(model, dataloader, device):
    """
    모델의 검증을 수행합니다.
    
    Args:
        model (nn.Module): 검증할 모델
        dataloader (DataLoader): 검증 데이터로더
        device (torch.device): 계산을 수행할 디바이스
        
    Returns:
        dict: 검증 지표 (손실, 정확도 등)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 데이터를 디바이스로 이동
            batch = batch.to(device)
            
            # 순전파
            outputs = model(batch)
            
            # 실제 레이블 (0: 정상, 1: 불법 도박 사이트)
            # 타겟 텐서의 형태를 모델 출력과 일치시키기 위해 reshape
            target = batch.y.view(-1, 1)
            
            # 손실 계산 (이진 분류)
            loss = nn.BCELoss()(outputs, target)
            
            total_loss += loss.item()
            
            # 예측 값과 레이블 저장 (평가 지표 계산용)
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    
    # 성능 지표 계산
    binary_preds = np.array(all_predictions) > 0.5
    binary_labels = np.array(all_labels) > 0.5
    
    accuracy = accuracy_score(binary_labels.flatten(), binary_preds.flatten())
    precision = precision_score(binary_labels.flatten(), binary_preds.flatten())
    recall = recall_score(binary_labels.flatten(), binary_preds.flatten())
    f1 = f1_score(binary_labels.flatten(), binary_preds.flatten())
    
    try:
        auc_score = roc_auc_score(all_labels, all_predictions)
    except:
        auc_score = 0.0
    
    # 특이도(specificity) 계산
    tn, fp, fn, tp = confusion_matrix(binary_labels.flatten(), binary_preds.flatten(), labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f'검증 결과:')
    print(f'  검증 손실: {avg_loss:.4f}')
    print(f'  정확도: {accuracy:.4f}')
    print(f'  정밀도: {precision:.4f}')
    print(f'  재현율(민감도): {recall:.4f}')
    print(f'  특이도: {specificity:.4f}')
    print(f'  F1 점수: {f1:.4f}')
    print(f'  AUC: {auc_score:.4f}')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc_score
    }

def main():
    # 설정 값 고정
    base_dir = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset/ablation_study/'
    test_size = 0.1
    val_size = 0.15  # 전체 훈련 데이터의 15%를 검증 세트로 사용
    batch_size = 16
    epochs = 3000
    lr = 0.001
    weight_decay = 1e-5
    hidden_dim = 64
    num_heads = 8
    dropout = 0.3
    gambling_weight = 35.0
    save_every = 1500
    seed = 42
    
    # Early Stopping 설정
    patience = 50  # 50 에폭 동안 개선이 없으면 중단
    min_delta = 0.001  # 최소 개선 기준
    
    # 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 저장 디렉토리 생성
    model_save_dir, results_save_dir = create_save_dirs()
    
    # 타임스탬프로 실험 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gambling_gat_binary_{timestamp}"
    
    # Early Stopping 초기화
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        path=os.path.join(model_save_dir, f"{experiment_name}_best.pt")
    )
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = GamblingGNNDataset(
        base_dir=base_dir,
        test_size=test_size,
        seed=seed
    )
    
    # 테스트 데이터 분리
    test_data = dataset.get_test_dataset()
    
    # 학습 데이터 - 전체 데이터에서 테스트 데이터를 제외한 부분
    # dataset 자체가 학습 데이터이므로 직접 사용
    train_val_dataset = dataset  # 학습 + 검증 데이터셋
    
    # 학습 데이터를 학습 세트와 검증 세트로 나눔
    train_size = int(len(train_val_dataset) * (1 - val_size))
    val_size = len(train_val_dataset) - train_size
    
    # 랜덤 분할
    train_set, val_set = torch.utils.data.random_split(
        train_val_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"전체 데이터셋 크기: {len(train_val_dataset) + len(test_data)}")
    print(f"학습 세트 크기: {len(train_set)}")
    print(f"검증 세트 크기: {len(val_set)}")
    print(f"테스트 세트 크기: {len(test_data)}")
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    # 모델 초기화
    input_dim = train_val_dataset[0].x.shape[1]  # 노드 특성 차원
    model = GamblingGATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_heads=num_heads,
        dropout=dropout,
        gambling_weight=gambling_weight
    ).to(device)
    
    # 최적화 도구 설정
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 학습 시작
    print("학습 시작...")
    train_metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 
        'specificity': [], 'f1': [], 'auc': []
    }
    
    val_metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 
        'specificity': [], 'f1': [], 'auc': []
    }
    
    for epoch in range(1, epochs + 1):
        # 학습
        epoch_train_metrics = train(model, train_loader, optimizer, device, epoch)
        
        # 검증
        epoch_val_metrics = validate(model, val_loader, device)
        
        # 지표 기록
        for key in train_metrics.keys():
            train_metrics[key].append(epoch_train_metrics[key])
            val_metrics[key].append(epoch_val_metrics[key])
        
        # Early Stopping 검사
        if early_stopping(epoch_val_metrics['loss'], model):
            print(f"Early stopping 발동! {patience}에폭 동안 검증 손실이 개선되지 않았습니다.")
            break
        
        # 모델 저장 (일정 주기마다)
        if epoch % save_every == 0 or epoch == epochs:
            model_path = os.path.join(model_save_dir, f"{experiment_name}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"[{epoch}/{epochs}] 에폭 모델 저장됨: {model_path}")
            
            # 현재까지의 학습 상태 저장
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'experiment_name': experiment_name
            }
            checkpoint_path = os.path.join(model_save_dir, f"{experiment_name}_checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"학습 상태(체크포인트) 저장됨: {checkpoint_path}")
    
    # 최적의 모델 로드
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f"{experiment_name}_best.pt")))
    
    # 학습 결과 시각화 (학습 및 검증)
    plot_learning_curves(train_metrics, val_metrics, results_save_dir, experiment_name)
    
    # 테스트
    print("테스트 중...")
    test_results = test(model, test_data, device)
    
    # 결과 저장
    np.savez(
        os.path.join(results_save_dir, f"{experiment_name}_test_results.npz"),
        predictions=test_results['predictions'],
        true_labels=test_results['true_labels'],
        accuracy=test_results['accuracy'],
        precision=test_results['precision'],
        recall=test_results['recall'],
        specificity=test_results['specificity'],
        f1=test_results['f1'],
        auc=test_results['auc'],
        confusion_matrix=test_results['confusion_matrix']
    )
    
    # 결과 요약 파일 생성
    with open(os.path.join(results_save_dir, f"{experiment_name}_summary.txt"), 'w') as f:
        f.write(f"실험 이름: {experiment_name}\n")
        f.write(f"데이터셋 경로: {base_dir}\n")
        f.write(f"하이퍼파라미터:\n")
        f.write(f"  Learning Rate: {lr}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Hidden Dimension: {hidden_dim}\n")
        f.write(f"  Attention Heads: {num_heads}\n")
        f.write(f"  Dropout: {dropout}\n")
        f.write(f"  Weight Decay: {weight_decay}\n")
        f.write(f"  Gambling Weight: {gambling_weight}\n")
        f.write(f"  Early Stopping Patience: {patience}\n\n")
        
        f.write("테스트 결과:\n")
        f.write(f"  Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_results['precision']:.4f}\n")
        f.write(f"  Recall(Sensitivity): {test_results['recall']:.4f}\n")
        f.write(f"  Specificity: {test_results['specificity']:.4f}\n")
        f.write(f"  F1 Score: {test_results['f1']:.4f}\n")
        f.write(f"  AUC: {test_results['auc']:.4f}\n\n")
        
        f.write("혼동 행렬:\n")
        f.write(f"{test_results['confusion_matrix']}\n\n")
        
        f.write("분류 보고서:\n")
        f.write(f"{test_results['classification_report']}\n")
    
    print("학습 및 테스트 완료!")
    print(f"모든 결과는 다음 경로에 저장되었습니다:")
    print(f"  - 모델 파일: {model_save_dir}")
    print(f"  - 결과 및 그래프: {results_save_dir}")

# 학습 결과 시각화 함수 수정 (학습 및 검증 곡선 함께 표시)
def plot_learning_curves(train_metrics, val_metrics, save_dir, experiment_name):
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    # 손실 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['loss'], label='Training Loss')
    plt.plot(epochs, val_metrics['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_loss.png"))
    plt.close()
    
    # 정확도 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['accuracy'], label='Training Accuracy')
    plt.plot(epochs, val_metrics['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_accuracy.png"))
    plt.close()
    
    # 정밀도 및 재현율 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['precision'], label='Training Precision')
    plt.plot(epochs, val_metrics['precision'], label='Validation Precision')
    plt.plot(epochs, train_metrics['recall'], label='Training Recall')
    plt.plot(epochs, val_metrics['recall'], label='Validation Recall')
    plt.plot(epochs, train_metrics['specificity'], label='Training Specificity')
    plt.plot(epochs, val_metrics['specificity'], label='Validation Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and Specificity Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_prec_recall.png"))
    plt.close()
    
    # F1 스코어 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['f1'], label='Training F1 Score')
    plt.plot(epochs, val_metrics['f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_f1.png"))
    plt.close()
    
    # AUC 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['auc'], label='Training AUC')
    plt.plot(epochs, val_metrics['auc'], label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_auc.png"))
    plt.close()
    
    # 모든 지표를 하나의 그래프에 표시
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_metrics['accuracy'], label='Train Accuracy')
    plt.plot(epochs, val_metrics['accuracy'], label='Val Accuracy')
    plt.plot(epochs, train_metrics['f1'], label='Train F1 Score')
    plt.plot(epochs, val_metrics['f1'], label='Val F1 Score')
    plt.plot(epochs, train_metrics['auc'], label='Train AUC')
    plt.plot(epochs, val_metrics['auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_all_metrics.png"))
    plt.close()

if __name__ == "__main__":
    main() 