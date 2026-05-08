import torch
import torch.nn as nn


class Simple3DCNN(nn.Module):

    def __init__(self, num_classes=2):
        super(Simple3DCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3,3,3),
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),


            nn.Conv3d(
                32,
                64,
                kernel_size=(3,3,3),
                padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),


            nn.Conv3d(
                64,
                128,
                kernel_size=(3,3,3),
                padding=1
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1,1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # 입력:
        # (B, T, C, H, W)

        x = x.permute(0,2,1,3,4)

        # 변환:
        # (B, C, T, H, W)

        x = self.features(x)

        x = self.classifier(x)

        return x