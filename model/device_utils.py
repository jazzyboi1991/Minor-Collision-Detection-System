import torch
import config


def get_device():
    """config.DEVICE_TYPE에 따라 적절한 디바이스를 반환합니다.

    NVIDIA CUDA와 AMD ROCm 모두 DEVICE_TYPE = "cuda" 로 설정합니다.
    ROCm PyTorch 빌드에서는 torch.cuda.is_available() 이 True를 반환하므로
    별도 분기 없이 동일하게 동작합니다.
    """
    if config.DEVICE_TYPE == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def is_rocm() -> bool:
    """현재 PyTorch 빌드가 ROCm(HIP) 기반인지 여부.

    ROCm 빌드에서는 torch.version.hip 이 버전 문자열을 반환하고,
    NVIDIA CUDA 빌드에서는 None을 반환합니다.
    """
    return torch.version.hip is not None


def is_cuda_like(device) -> bool:
    """AMP·cuDNN·pin_memory 등 CUDA 계열 기능을 사용할 수 있는 디바이스인지 여부.

    NVIDIA CUDA 와 ROCm(HIP) PyTorch 빌드 모두 device 이름이 "cuda" 이므로
    True를 반환합니다. CPU는 False를 반환합니다.
    """
    return str(device).startswith("cuda")


def is_channels_last_3d_supported(device) -> bool:
    """channels_last_3d 메모리 포맷을 안전하게 사용할 수 있는지 여부.

    ROCm(HIP)은 channels_last_3d를 완전히 지원하지 않아
    일부 연산에서 오류가 발생할 수 있으므로 비활성화합니다.
    NVIDIA CUDA에서만 활성화합니다.
    """
    if not is_cuda_like(device):
        return False
    if is_rocm():
        return False
    return True
