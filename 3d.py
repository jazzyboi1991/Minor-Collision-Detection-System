import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0,2,1,3,4)

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
