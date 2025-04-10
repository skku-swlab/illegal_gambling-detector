import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from dataset import GamblingDataset
import os
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_metrics = []
        self.evaluation_metrics = []
        self.output_dir = kwargs.get('args').output_dir

    def log(self, logs, start_time=None):
        super().log(logs)
        if 'loss' in logs:
            self.training_metrics.append({
                'epoch': logs.get('epoch', 0),
                'loss': logs.get('loss', 0),
                'learning_rate': logs.get('learning_rate', 0),
                'accuracy': logs.get('accuracy', 0) if 'accuracy' in logs else 0
            })
        if 'eval_loss' in logs:
            self.evaluation_metrics.append({
                'epoch': logs.get('epoch', 0),
                'eval_loss': logs.get('eval_loss', 0),
                'eval_accuracy': logs.get('eval_accuracy', 0),
                'eval_precision': logs.get('eval_precision', 0),
                'eval_recall': logs.get('eval_recall', 0),
                'eval_f1': logs.get('eval_f1', 0)
            })
            # 각 에포크마다 그래프 생성
            self.plot_training_curves(logs.get('epoch', 0))

    def plot_training_curves(self, current_epoch):
        # 최소 2개의 데이터 포인트가 있어야 그래프를 생성
        if len(self.training_metrics) < 2 or len(self.evaluation_metrics) < 2:
            return
            
        plt.figure(figsize=(15, 5))
        
        # 1. 손실 곡선
        plt.subplot(1, 3, 1)
        train_metrics = self.training_metrics
        eval_metrics = self.evaluation_metrics
        
        # 훈련과 검증 메트릭의 길이를 맞춤
        min_length = min(len(train_metrics), len(eval_metrics))
        train_metrics = train_metrics[:min_length]
        eval_metrics = eval_metrics[:min_length]
        
        train_losses = [m['loss'] for m in train_metrics]
        eval_losses = [m['eval_loss'] for m in eval_metrics]
        epochs = [m['epoch'] for m in train_metrics]
        
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, eval_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # 2. 정확도 곡선
        plt.subplot(1, 3, 2)
        train_accuracies = [m.get('accuracy', 0) for m in train_metrics]
        eval_accuracies = [m['eval_accuracy'] for m in eval_metrics]
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.plot(epochs, eval_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # 3. Precision-Recall 곡선
        plt.subplot(1, 3, 3)
        eval_precisions = [m['eval_precision'] for m in eval_metrics]
        eval_recalls = [m['eval_recall'] for m in eval_metrics]
        plt.plot(epochs, eval_precisions, label='Precision')
        plt.plot(epochs, eval_recalls, label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'training_curves_epoch_{current_epoch:.1f}.png'))
        plt.close()

def main():
    # 1. BERT 모델과 토크나이저 불러오기
    model_name = "beomi/kcbert-base"  # 한국어 BERT 모델
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"Using model: {model_name}")

    # 2. 데이터셋 생성
    csv_file_path = "/home/swlab/Desktop/gambling_crawling/dataset/dataset.csv"
    dataset = GamblingDataset(csv_file_path, tokenizer, max_length=256)  # 시퀀스 길이 증가
    print(f"Total dataset size: {len(dataset)}")

    # 3. 데이터셋을 train/validation으로 나누기
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    # 4. 분류용 BERT 모델 불러오기
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )

    # 5. TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir='./gambling_bert_results',
        num_train_epochs=10,  # 에포크 수 더욱 감소
        per_device_train_batch_size=16,  # 배치 크기 감소
        per_device_eval_batch_size=16,
        warmup_steps=200,  # 웜업 스텝 감소
        weight_decay=0.1,  # 가중치 감소 더욱 증가
        learning_rate=1e-5,  # 학습률 더욱 감소
        logging_dir='./gambling_bert_logs',
        logging_steps=len(train_dataset) // (16 * 5),
        evaluation_strategy="epoch",  # 에포크마다 평가
        save_strategy="epoch",  # 에포크마다 저장
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # 최적 모델 선택 기준 변경
        report_to="tensorboard",
        fp16=True,  # 16비트 부동소수점 사용
        gradient_accumulation_steps=4,  # 그래디언트 누적
    )

    # 6. CustomTrainer 객체 생성
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. 학습 시작 시간 기록
    start_time = time.time()
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Training steps per epoch: {len(train_dataset) // training_args.per_device_train_batch_size}")

    # 8. 모델 파인튜닝 시작
    trainer.train()

    # 9. 학습 종료 시간 및 총 소요 시간 계산
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {total_time/3600:.2f} hours")

    # 10. 모델 저장
    save_dir = "./gambling_bert_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    trainer.save_model(save_dir)
    print(f"Model saved to {save_dir}")

    # 11. 학습 과정 분석
    print("\n=== 학습 과정 분석 ===")
    print("\n1. 훈련 메트릭:")
    for epoch_metrics in trainer.training_metrics:
        print(f"\nEpoch {epoch_metrics['epoch']:.1f}:")
        print(f"  Loss: {epoch_metrics['loss']:.4f}")
        print(f"  Learning Rate: {epoch_metrics['learning_rate']:.6f}")
        print(f"  Accuracy: {epoch_metrics['accuracy']:.4f}")

    print("\n2. 검증 메트릭:")
    for epoch_metrics in trainer.evaluation_metrics:
        print(f"\nEpoch {epoch_metrics['epoch']:.1f}:")
        print(f"  Loss: {epoch_metrics['eval_loss']:.4f}")
        print(f"  Accuracy: {epoch_metrics['eval_accuracy']:.4f}")
        print(f"  Precision: {epoch_metrics['eval_precision']:.4f}")
        print(f"  Recall: {epoch_metrics['eval_recall']:.4f}")
        print(f"  F1 Score: {epoch_metrics['eval_f1']:.4f}")

    # 12. 오버피팅 분석
    print("\n=== 오버피팅 분석 ===")
    final_train_loss = trainer.training_metrics[-1]['loss']
    final_eval_loss = trainer.evaluation_metrics[-1]['eval_loss']
    final_train_acc = trainer.training_metrics[-1]['accuracy']
    final_eval_acc = trainer.evaluation_metrics[-1]['eval_accuracy']
    
    print(f"\n최종 훈련 손실: {final_train_loss:.4f}")
    print(f"최종 검증 손실: {final_eval_loss:.4f}")
    print(f"훈련-검증 손실 차이: {abs(final_train_loss - final_eval_loss):.4f}")
    
    if final_train_loss < final_eval_loss * 0.7:
        print("\n경고: 오버피팅이 발생했을 가능성이 높습니다!")
        print("훈련 손실이 검증 손실보다 30% 이상 낮습니다.")
    elif final_train_loss < final_eval_loss * 0.9:
        print("\n주의: 오버피팅이 발생할 가능성이 있습니다.")
        print("훈련 손실이 검증 손실보다 10% 이상 낮습니다.")
    else:
        print("\n오버피팅이 발생하지 않은 것으로 보입니다.")
        print("훈련 손실과 검증 손실의 차이가 적습니다.")

    # 13. 최종 평가 결과 출력
    eval_results = trainer.evaluate()
    print("\n=== 최종 평가 결과 ===")
    print(f"Loss: {eval_results['eval_loss']:.4f}")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall: {eval_results['eval_recall']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()
