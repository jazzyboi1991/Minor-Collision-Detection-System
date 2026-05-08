import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 1, 2단계 코드를 별도의 파일로 분리했을 경우 아래와 같이 import 합니다.
# 만약 하나의 파일로 합쳤다면 이 import 부분은 지우시면 됩니다.
from dataset import HitAndRunDataset
from model import HitAndRun3DCNN

# --- Early Stopping 기법 구현  ---


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# --- 학습 루프 (Training Loop) ---


def train_model(data_dir):
    # CPU에서도 작동하도록 유연하게 디바이스 할당
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    # 1. 데이터 로더 준비
    # 논문 조건: 최적의 주변 마진값 r=1, 프레임 길이 30 적용 [cite: 363]
    train_dataset = HitAndRunDataset(
        data_dir=data_dir, clip_length=30, r_value=1.0)

    # 논문 조건: 배치 크기 15
    train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)

    # 2. 모델 초기화 (Pre-trained 없이 from scratch 학습) [cite: 350]
    model = HitAndRun3DCNN(num_classes=2).to(device)

    # 3. 손실 함수 및 옵티마이저 (논문 조건 반영)
    criterion = nn.CrossEntropyLoss()

    # 논문 조건: Adam 옵티마이저, 학습률 1e-5
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 과적합 방지를 위한 조기 종료 설정
    early_stopping = EarlyStopping(patience=10)
    num_epochs = 100  # 논문 조건: 최대 100 Epochs

    print("학습을 시작합니다...")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 실제 환경에서는 검증 세트(Validation set)의 로스를 계산하여 Early Stopping에 넣습니다.
        # 가상의 val_loss를 넣는 예시:
        # val_loss = validate_model(model, val_loader, criterion, device)
        # early_stopping(val_loss)
        # if early_stopping.early_stop:
        #     print("Early stopping 발동. 모델이 수렴하여 학습을 조기 종료합니다.")
        #     break

    return model

# --- 테스트: 슬라이딩 윈도우 (Sliding Window Inference) ---


def test_sliding_window(model, full_video_tensor):
    """
    논문 조건: 전체 비디오를 학습 때와 동일한 길이(30프레임)로 자르되, 
    stride=1로 1프레임씩 겹치며 이동하는 슬라이딩 윈도우 적용 
    입력 형태: [C(3), T(전체 프레임 수), H(224), W(224)]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_frames = full_video_tensor.size(1)
    clip_length = 30

    if total_frames < clip_length:
        print("비디오 길이가 30프레임 미만이라 윈도우를 구성할 수 없습니다.")
        return

    predictions = []

    with torch.no_grad():
        # Stride 1로 슬라이딩 윈도우 적용
        for i in range(total_frames - clip_length + 1):
            # 30프레임 분량의 클립 추출
            clip = full_video_tensor[:, i:i+clip_length, :, :]

            # 모델 입력을 위해 배치 차원 추가 -> [1, 3, 30, 224, 224]
            clip = clip.unsqueeze(0).to(device)

            # 예측 수행
            outputs = model(clip)

            # Softmax를 통과시켜 0(비충돌)과 1(충돌) 중 확률이 높은 클래스 추출
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            predictions.append(predicted_class)

    # 평가: 전체 클립의 슬라이딩 윈도우 결과 중 충돌(1)이 한 번이라도 일관되게 발생했는지 판단
    if 1 in predictions:
        print("분석 결과: 영상 내에 충돌(A) 상황이 감지되었습니다.")
    else:
        print("분석 결과: 충돌이 없는 정상(S) 영상입니다.")


if __name__ == "__main__":
    # 데이터 경로를 지정하고 전체 파이프라인을 실행하는 진입점입니다.
    print("3D-CNN Hit-and-Run 감지 파이프라인 로드 완료")

    # 실제 학습을 진행하실 때 아래 주석을 해제하고 폴더 경로를 맞춰주시면 됩니다.
    DATA_DIR = 'sample'
    trained_model = train_model(data_dir=DATA_DIR)
