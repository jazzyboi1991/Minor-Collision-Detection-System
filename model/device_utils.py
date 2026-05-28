import torch
import config


def get_device():
    """config.DEVICE_TYPE에 따라 적절한 디바이스를 반환합니다."""
    if config.DEVICE_TYPE == "directml":
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            raise ImportError(
                "torch-directml이 설치되지 않았습니다.\n"
                "pip install torch-directml 을 실행한 후 다시 시도하세요."
            )
    elif config.DEVICE_TYPE == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def is_cuda_like(device) -> bool:
    """AMP·channels_last·cuDNN 등 CUDA 전용 기능을 사용할 수 있는 디바이스인지 여부.

    ROCm PyTorch 빌드도 device 이름이 "cuda"이므로 True를 반환합니다.
    DirectML·CPU는 False를 반환합니다.
    """
    return str(device).startswith("cuda")
