import argparse
import os
import random
import tempfile

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
except ModuleNotFoundError as error:
    missing_package = error.name
    raise SystemExit(
        f"Missing dependency: {missing_package}\n"
        "Install the CPU test dependencies first:\n"
        "  pip install opencv-python numpy torch torchvision\n"
        "This script does not require a GPU, but it does require these Python packages."
    ) from error


# Conv3D -> BatchNorm3D -> ReLU를 묶은 기본 블록입니다.
# 영상은 (채널, 시간, 높이, 너비) 형태라서 2D CNN이 아니라 3D CNN을 사용합니다.
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


# I3D/Inception 계열 구조입니다.
# 서로 다른 branch에서 특징을 뽑은 뒤 채널 방향으로 합쳐서 더 다양한 움직임 특징을 봅니다.
class InceptionModule3D(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_3x3_2, out_3x3_2, out_pool):
        super().__init__()
        self.branch1 = BasicConv3d(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels, red_3x3, kernel_size=1),
            BasicConv3d(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channels, red_3x3_2, kernel_size=1),
            BasicConv3d(red_3x3_2, out_3x3_2, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            BasicConv3d(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


# 사고 여부를 2개 클래스(Normal/Accident)로 분류하는 3D CNN 모델입니다.
# 입력 텐서 형태는 (batch, 3, 30, 224, 224)를 기대합니다.
class HitAndRun3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = BasicConv3d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = BasicConv3d(64, 64, kernel_size=1)
        self.conv3 = BasicConv3d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.inception1 = InceptionModule3D(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionModule3D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.inception3 = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head_conv = nn.Conv3d(512, num_classes, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))
        x = self.maxpool3(self.inception2(self.inception1(x)))
        x = self.avg_pool(self.inception3(x))
        logits = self.head_conv(x)
        return logits.view(logits.size(0), -1)


def crop_square_and_pad(frame, bbox, r_value, resize):
    """차량 bbox 주변을 정사각형으로 잘라 모델 입력 크기로 변환합니다."""
    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = bbox

    # r_value가 1보다 크면 차량 주변 여백까지 같이 포함합니다.
    vw, vh = (x_max - x_min) * r_value, (y_max - y_min) * r_value
    cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2

    # 가로/세로 중 긴 변을 기준으로 정사각형 crop 영역을 만듭니다.
    side = int(max(vw, vh))
    nx1, ny1 = cx - side // 2, cy - side // 2
    nx2, ny2 = cx + side // 2, cy + side // 2

    # crop 영역이 영상 밖으로 나가면, 실제 영상 안쪽만 자르고 나머지는 검은색으로 padding합니다.
    vx1, vy1 = max(0, nx1), max(0, ny1)
    vx2, vy2 = min(w, nx2), min(h, ny2)
    pl, pt = max(0, -nx1), max(0, -ny1)
    pr, pb = max(0, nx2 - w), max(0, ny2 - h)

    cropped = frame[vy1:vy2, vx1:vx2]
    if pl > 0 or pt > 0 or pr > 0 or pb > 0:
        cropped = np.pad(cropped, ((pt, pb), (pl, pr), (0, 0)), mode="constant")
    return cv2.resize(cropped, resize)


def read_bbox(txt_path, target_id):
    """라벨 txt 파일에서 target_id 차량의 bbox 좌표를 읽습니다."""
    bboxes = {}
    with open(txt_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) >= 6 and parts[0] == "car":
                bboxes[int(parts[1])] = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
    if not bboxes:
        return None

    # target_id가 없으면 파일에 있는 첫 번째 차량을 대신 평가합니다.
    return bboxes.get(target_id, next(iter(bboxes.values())))


def label_from_filename(mp4_file):
    """파일명 규칙에서 정답 라벨을 뽑습니다. 예: LA/NA/RA/SA는 Accident입니다."""
    parts = os.path.splitext(mp4_file)[0].split("_")
    return 1 if len(parts) >= 2 and len(parts[1]) == 2 and parts[1][1] == "A" else 0


def predict_video(model, video_path, txt_path, device, target_id, r_value, resize):
    """영상 하나를 읽고, 한 번이라도 Accident가 나오면 해당 영상을 사고로 판단합니다."""
    bbox = read_bbox(txt_path, target_id)
    if bbox is None:
        return None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 전체 영상을 프레임 단위로 읽어서 차량 영역만 crop한 뒤 텐서로 변환합니다.
    cap = cv2.VideoCapture(video_path)
    tensor_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped = crop_square_and_pad(frame_rgb, bbox, r_value, resize)
        tensor_frames.append(transform(cropped))
    cap.release()

    if not tensor_frames:
        return None

    # 모델은 30프레임 clip을 입력으로 받으므로, 짧은 영상은 마지막 프레임을 복사해 채웁니다.
    while len(tensor_frames) < 30:
        tensor_frames.append(tensor_frames[-1])

    # torch.stack 결과는 (T, C, H, W)이므로 모델 입력에 맞게 (C, T, H, W)로 바꿉니다.
    full_video_tensor = torch.stack(tensor_frames).permute(1, 0, 2, 3)

    # 30프레임 슬라이딩 윈도우로 영상을 훑습니다.
    predicted_label = 0
    for i in range(len(tensor_frames) - 29):
        clip = full_video_tensor[:, i:i + 30, :, :].unsqueeze(0).to(device)
        outputs = model(clip)
        pred_class = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()

        # 충돌은 짧은 순간만 나타날 수 있으므로, 한 번이라도 1이 나오면 사고 영상으로 처리합니다.
        if pred_class == 1:
            predicted_label = 1
            break
    return predicted_label


def evaluate(args):
    """폴더 안의 mp4/txt 쌍을 평가하고 정확도와 오답 목록을 출력합니다."""
    if not args.weights:
        raise ValueError("Model weights are required for full evaluation. Pass --weights path/to/model.pth.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HitAndRun3DCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 같은 이름의 .mp4와 .txt가 모두 있는 파일만 평가 대상으로 삼습니다.
    all_files = set(os.listdir(args.data_dir))
    pairs = sorted(f for f in all_files if f.endswith(".mp4") and f"{os.path.splitext(f)[0]}.txt" in all_files)
    if args.samples is not None:
        random.seed(args.seed)
        pairs = random.sample(pairs, min(args.samples, len(pairs)))

    total = correct = skipped = 0
    wrong = []
    with torch.no_grad():
        for mp4_file in pairs:
            base = os.path.splitext(mp4_file)[0]
            video_path = os.path.join(args.data_dir, mp4_file)
            txt_path = os.path.join(args.data_dir, f"{base}.txt")
            gt = label_from_filename(mp4_file)
            pred = predict_video(model, video_path, txt_path, device, args.target_id, args.r_value, (args.resize, args.resize))
            if pred is None:
                skipped += 1
                continue
            total += 1
            correct += int(pred == gt)
            if pred != gt:
                wrong.append((mp4_file, gt, pred))

    accuracy = (correct / total * 100) if total else 0.0
    print(f"total={total}")
    print(f"correct={correct}")
    print(f"skipped={skipped}")
    print(f"accuracy={accuracy:.2f}%")
    if wrong:
        print("wrong:")
        for name, gt, pred in wrong:
            gt_name = "Accident" if gt == 1 else "Normal"
            pred_name = "Accident" if pred == 1 else "Normal"
            print(f"- {name}: gt={gt_name}, pred={pred_name}")


def run_cpu_smoke_test():
    """GPU, 실제 영상, 가중치 없이 핵심 로직만 빠르게 검증합니다."""
    device = torch.device("cpu")

    # 1. 파일명에서 정답 라벨을 뽑는 규칙이 맞는지 확인합니다.
    assert label_from_filename("220510_LA_0001.mp4") == 1
    assert label_from_filename("220510_LS_0001.mp4") == 0
    assert label_from_filename("220510_NV_0001.mp4") == 0

    # 2. txt 라벨 파일에서 bbox를 읽고, target_id가 없을 때 첫 bbox로 fallback되는지 확인합니다.
    with tempfile.TemporaryDirectory() as temp_dir:
        txt_path = os.path.join(temp_dir, "sample.txt")
        with open(txt_path, "w") as file:
            file.write("car,0,10,20,50,70\n")
            file.write("car,3,30,40,80,90\n")

        assert read_bbox(txt_path, target_id=0) == [10, 20, 50, 70]
        assert read_bbox(txt_path, target_id=99) == [10, 20, 50, 70]

    # 3. crop 영역이 화면 밖으로 나가도 padding 후 원하는 크기로 나오는지 확인합니다.
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    cropped = crop_square_and_pad(frame, bbox=[-5, -5, 40, 30], r_value=1.2, resize=(32, 32))
    assert cropped.shape == (32, 32, 3)

    # 4. 모델이 CPU에서 작은 더미 clip을 받아 2개 클래스 logits를 내는지 확인합니다.
    model = HitAndRun3DCNN(num_classes=2).to(device)
    model.eval()
    dummy_clip = torch.zeros((1, 3, 30, 64, 64), dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(dummy_clip)
    assert output.shape == (1, 2)

    print("CPU smoke test passed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/leezungzoo/Desktop/AI-develop/Sample")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--r-value", type=float, default=1.0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument(
        "--cpu-smoke-test",
        action="store_true",
        help="Run lightweight software checks on CPU without real videos or model weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cpu_smoke_test:
        run_cpu_smoke_test()
    else:
        evaluate(args)
