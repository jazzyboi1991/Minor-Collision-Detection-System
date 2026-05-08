import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv3d(nn.Module):
    """ I3D 네트워크를 구성하는 기본 3D Convolution 블록 """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionModule3D(nn.Module):
    """ 논문에 명시된 3D Inception 모듈 (4개의 병렬 브랜치로 구성) """

    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_3x3_2, out_3x3_2, out_pool):
        super(InceptionModule3D, self).__init__()

        # Branch 1: 1x1x1 conv
        self.branch1 = BasicConv3d(in_channels, out_1x1, kernel_size=1)

        # Branch 2: 1x1x1 conv -> 3x3x3 conv
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels, red_3x3, kernel_size=1),
            BasicConv3d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # Branch 3: 1x1x1 conv -> 3x3x3 conv
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channels, red_3x3_2, kernel_size=1),
            BasicConv3d(red_3x3_2, out_3x3_2, kernel_size=3, padding=1)
        )

        # Branch 4: 3x3x3 max pooling -> 1x1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            BasicConv3d(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # 4개의 브랜치를 채널(dim=1) 기준으로 병합
        return torch.cat([b1, b2, b3, b4], dim=1)


class HitAndRun3DCNN(nn.Module):
    """ 물피도주 감지를 위한 최종 1-Stream 3D-CNN 모델 """

    def __init__(self, num_classes=2):  # Class 0: S (비충돌), Class 1: A (충돌)
        super(HitAndRun3DCNN, self).__init__()

        # 1. 초기 Convolution 및 Pooling 레이어
        self.conv1 = BasicConv3d(3, 64, kernel_size=7,
                                 stride=2, padding=3)  # RGB 3채널 입력
        self.maxpool1 = nn.MaxPool3d(kernel_size=(
            1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv2 = BasicConv3d(64, 64, kernel_size=1)
        self.conv3 = BasicConv3d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(
            1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # 2. 3D Inception 블록들 (단순화된 구조 적용)
        self.inception1 = InceptionModule3D(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionModule3D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.inception3 = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)

        # 3. Global Average Pooling
        # 논문에서 CAM(Class Activation Map) 생성을 위해 Flatten 대신 Global Average Pooling을 사용
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 4. 네트워크의 끝단 (Head) - 논문 조건: 1x1x1 Convolution
        self.head_conv = nn.Conv3d(512, num_classes, kernel_size=(1, 1, 1))

        # 모델의 모든 가중치를 무작위로 초기화 (from scratch)
        self._initialize_weights()

    def forward(self, x):
        # x 입력 형태: [Batch, Channels(3), Frames(30), Height(224), Width(224)]
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool3(x)

        x = self.inception3(x)

        x = self.avg_pool(x)  # 출력 형태: [Batch, 512, 1, 1, 1]

        # 1x1x1 Conv 통과
        logits = self.head_conv(x)  # 출력 형태: [Batch, 2, 1, 1, 1]

        # 불필요한 차원 제거 -> [Batch, 2]
        logits = logits.view(logits.size(0), -1)

        # PyTorch의 CrossEntropyLoss 내부에 Softmax가 포함되어 있으므로
        # 학습 시에는 Logits 형태를 그대로 반환합니다. (추론 시에만 Softmax 명시적 적용)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# # --- 테스트 실행 코드 (입력 텐서가 정상적으로 통과하는지 확인용) ---
# if __name__ == "__main__":
#     # 1단계에서 설계한 데이터 규격: [Batch=15, 채널=3(RGB), 프레임=30, 높이=224, 너비=224]
#     dummy_input = torch.randn(15, 3, 30, 224, 224)

#     model = HitAndRun3DCNN(num_classes=2)
#     output = model(dummy_input)

#     # 기대 출력: [15, 2] (15개의 배치 데이터에 대한 2개 클래스의 Logit 값)
#     print("모델 출력 텐서 형태:", output.shape)
