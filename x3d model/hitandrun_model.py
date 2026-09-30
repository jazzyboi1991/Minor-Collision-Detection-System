import os

import torch.nn as nn
from pytorchvideo.models.hub import x3d_m
from torch.utils.checkpoint import checkpoint_sequential


# X3D-M ProjectedPool의 최종 projection 채널 수.
X3D_M_HEAD_CHANNELS = 2048
DROPOUT_P = 0.5
GRAD_CHECKPOINT = os.getenv("GRAD_CHECKPOINT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
GRAD_CHECKPOINT_SEGMENTS = int(os.getenv("GRAD_CHECKPOINT_SEGMENTS", "5"))


class HitAndRun3DCNN(nn.Module):
    """X3D-M 백본 기반 물피도주 감지 모델.

    X3D(Feichtenhofer, CVPR 2020)는 2D 이미지 분류망을 폭·깊이·해상도·프레임
    축으로 점진 확장해 얻은 효율적 3D-CNN이다. 같은 정확도를 S3D보다 훨씬 적은
    연산으로 내는 것이 특징이라, 멀티-아워 영상을 슬라이딩 윈도로 훑는 이
    서비스에서는 '윈도우당 비용'을 줄이는 쪽에 이점이 있다.
    (파라미터: X3D-M 3.8M vs S3D 7.9M)

    pytorchvideo 기본 헤드의 ProjectedPool(192→432→2048)은 유지하되,
    고정 AvgPool3d 커널을 AdaptiveAvgPool3d로 교체한다. 이 구조는 배포용
    X3D 체크포인트의 state_dict와 동일하다.

    서비스 통합 인터페이스는 S3D 버전과 동일하게 유지한다:
      - 입력 (B, 3, T, 224, 224) → 출력 (B, num_classes) logits
      - `head_conv`   : 2048→num_classes 1×1×1 분류 헤드
      - `inception5b` : CAM forward hook 대상(ProjectedPool 출력 별칭)

    Args:
        num_classes: 분류 클래스 수 (기본 2: S/A)
        pretrained : True면 Kinetics-400 사전학습 가중치로 백본 초기화.
                     추론에서는 학습된 state_dict를 로드하므로 False로 생성해
                     불필요한 다운로드를 피한다.
    """

    def __init__(self, num_classes=2, pretrained=False, dropout_p=DROPOUT_P):
        super(HitAndRun3DCNN, self).__init__()
        base = x3d_m(pretrained=pretrained)

        # stem+res 스테이지를 특징 추출기로 사용한다.
        self.features = nn.Sequential(*list(base.blocks)[:-1])

        # 학습 당시 사용한 X3D projection head를 보존한다. 시간축만 먼저
        # 축소하고 공간축은 유지해 CAM이 7×7 특징맵을 사용할 수 있게 한다.
        self.head_pool = base.blocks[-1].pool
        self.head_pool.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.head_conv = nn.Conv3d(X3D_M_HEAD_CHANNELS, num_classes, kernel_size=1)

    @property
    def inception5b(self):
        """CAM hook 호환 별칭 — 분류 헤드 입력과 같은 2048채널 출력.

        property라 모듈이 중복 등록되지 않으며, predict_cam.py 의
        `model.inception5b.register_forward_hook(...)`이 수정 없이 동작한다.
        """
        return self.head_pool

    def forward(self, x):
        if self.training and GRAD_CHECKPOINT:
            x = checkpoint_sequential(
                self.features,
                GRAD_CHECKPOINT_SEGMENTS,
                x,
                use_reentrant=False,
            )
        else:
            x = self.features(x)
        x = self.head_pool(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
