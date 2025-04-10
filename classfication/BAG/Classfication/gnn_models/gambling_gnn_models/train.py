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
            
            # 예측
            output = model(data)
            
            # 예측값과 실제 레이블 저장
            predictions.append(output.item())
            labels.append(data.y.item())
    
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

def main():
    # 설정 값 고정
    base_dir = '/home/swlab/Desktop/illegal_gambling-detector/classfication/BAG/data/gnn/gnn_datset'
    test_size = 0.1
    batch_size = 16
    epochs = 5000
    lr = 0.001
    weight_decay = 1e-5
    hidden_dim = 64
    num_heads = 8
    dropout = 0.3
    gambling_weight = 20.0  # gambling_score에 적용할 가중치
    save_every = 1000  # 100 에폭마다 모델 저장
    seed = 42
    
    # 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = GamblingGNNDataset(
        base_dir=base_dir,
        test_size=test_size,
        seed=seed
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # 테스트 데이터셋 로드
    test_data = dataset.get_test_dataset()
    
    # 모델 초기화
    input_dim = train_loader.dataset[0].x.shape[1]  # 노드 특성 차원
    model = GamblingGATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,  # 이진 분류를 위한 출력 차원
        num_heads=num_heads,
        dropout=dropout,
        gambling_weight=gambling_weight
    ).to(device)
    
    # 최적화 도구 설정
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 저장 디렉토리 생성
    model_save_dir, results_save_dir = create_save_dirs()
    
    # 타임스탬프로 실험 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gambling_gat_binary_{timestamp}"
    
    # 학습 시작
    print("학습 시작...")
    metrics = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 
        'specificity': [], 'f1': [], 'auc': []
    }
    
    for epoch in range(1, epochs + 1):
        # 학습
        epoch_metrics = train(model, train_loader, optimizer, device, epoch)
        
        # 지표 기록
        for key in metrics.keys():
            metrics[key].append(epoch_metrics[key])
        
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
                'metrics': metrics,
                'experiment_name': experiment_name
            }
            checkpoint_path = os.path.join(model_save_dir, f"{experiment_name}_checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"학습 상태(체크포인트) 저장됨: {checkpoint_path}")
    
    # 학습 결과 시각화
    plot_learning_curve(metrics, results_save_dir, experiment_name)
    
    # 최종 모델 저장
    final_model_path = os.path.join(model_save_dir, f"{experiment_name}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"최종 모델 저장됨: {final_model_path}")
    
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
        f.write(f"  Gambling Weight: {gambling_weight}\n\n")
        
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
    
    # 성능 지표 시각화 (혼동 행렬)
    cm = test_results['confusion_matrix']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Gambling'])
    plt.yticks(tick_marks, ['Normal', 'Gambling'])
    
    # 혼동 행렬에 숫자 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(results_save_dir, f"{experiment_name}_confusion_matrix.png"))
    plt.close()
    
    print("학습 및 테스트 완료!")
    print(f"모든 결과는 다음 경로에 저장되었습니다:")
    print(f"  - 모델 파일: {model_save_dir}")
    print(f"  - 결과 및 그래프: {results_save_dir}")

if __name__ == "__main__":
    main() 