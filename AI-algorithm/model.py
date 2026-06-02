import torch
import torch.nn as nn


class TemporalCollisionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        self.temporal_head = nn.Conv1d(128, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self.features(x)

        # 공간만 평균: (B, C, T, H, W) -> (B, C, T)
        x = x.mean(dim=[3, 4])

        # 시간별 예측: (B, C, T) -> (B, num_classes, T)
        x = self.temporal_head(x)

        return x
