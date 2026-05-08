import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class HitAndRunDataset(Dataset):
    def __init__(self, data_dir, clip_length=30, r_value=1.0, resize=(224, 224), is_train=True):
        """
        논문 조건:
        - clip_length = 30 (최적 성능 프레임 수)
        - r_value = 1 (마진 최소화, 타겟 차량 타이트 크롭)
        """
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.r_value = r_value
        self.resize = resize
        self.is_train = is_train

        # 데이터 디렉토리 내의 mp4, txt 파일 목록 정리
        self.file_names = [f.split('.')[0] for f in os.listdir(
            data_dir) if f.endswith('.mp4')]

        # 공간적 데이터 증강 (비디오이므로 모든 프레임에 동일하게 적용되어야 함)
        # 1단계에서는 기본 Tensor 변환 및 정규화만 적용 (복잡한 증강은 추후 훈련 루프에서 일괄 적용 권장)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_names)

    def _parse_annotation(self, txt_path):
        bboxes = {}
        action = None
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if not parts or len(parts) < 2:
                    continue

                if parts[0] == 'car':
                    veh_id = int(parts[1])
                    bboxes[veh_id] = [int(parts[2]), int(
                        parts[3]), int(parts[4]), int(parts[5])]
                elif parts[0] in ['A', 'S']:
                    action = parts

        return bboxes, action

    def _crop_and_pad(self, frame, bbox, r):
        """ r 값에 따른 크롭 및 화면 밖으로 나갈 경우의 패딩 처리 """
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox

        veh_w = x_max - x_min
        veh_h = y_max - y_min

        # r 값에 따른 새로운 너비와 높이 계산 (r=1이면 차량 크기 그대로)
        new_w = int(veh_w * r)
        new_h = int(veh_h * r)  # 너비 비율(r)을 높이에도 유사하게 적용 (종횡비 유지 목적)

        # 새로운 중심점은 기존 차량의 중심점
        cx, cy = x_min + veh_w // 2, y_min + veh_h // 2

        new_x_min = cx - new_w // 2
        new_y_min = cy - new_h // 2
        new_x_max = cx + new_w // 2
        new_y_max = cy + new_h // 2

        # 패딩 계산 (화면 밖으로 나가는 영역)
        pad_left = max(0, -new_x_min)
        pad_top = max(0, -new_y_min)
        pad_right = max(0, new_x_max - w)
        pad_bottom = max(0, new_y_max - h)

        # 화면 안쪽의 유효한 크롭 영역
        valid_x_min = max(0, new_x_min)
        valid_y_min = max(0, new_y_min)
        valid_x_max = min(w, new_x_max)
        valid_y_max = min(h, new_y_max)

        cropped_frame = frame[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

        # 패딩이 필요한 경우 numpy pad 이용 (검은색 0 패딩 적용)
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped_frame = np.pad(cropped_frame,
                                   ((pad_top, pad_bottom),
                                    (pad_left, pad_right), (0, 0)),
                                   mode='constant', constant_values=0)

        # 3D-CNN 입력을 위한 고정 크기 리사이즈
        resized_frame = cv2.resize(cropped_frame, self.resize)
        return resized_frame

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        mp4_path = os.path.join(self.data_dir, f"{file_name}.mp4")
        txt_path = os.path.join(self.data_dir, f"{file_name}.txt")

        # 1. 텍스트 라벨 정보 파싱
        bboxes, action = self._parse_annotation(txt_path)
        class_str, target_id, start_f, end_f = action[0], int(
            action[1]), int(action[2]), int(action[3])
        label = 1 if class_str == 'A' else 0  # A(충돌)은 1, S(비충돌)은 0
        target_bbox = bboxes[target_id]

        # 2. 비디오 프레임 추출
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        # 논문 조건: 충돌 시작(start_f)부터 clip_length(30)만큼 추출
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            # BGR을 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # r=1 값에 따른 크롭 및 패딩 수행
            processed_frame = self._crop_and_pad(
                frame, target_bbox, self.r_value)
            frames.append(processed_frame)

        cap.release()

        # 프레임이 30장이 안 되는 경우 (영상이 일찍 끝남), 마지막 프레임으로 복제하여 채움
        while len(frames) < self.clip_length:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros(
                (self.resize[1], self.resize[0], 3), dtype=np.uint8))

        # 3. 텐서 변환 및 차원 변경 (T, H, W, C) -> (T, C, H, W)
        tensor_frames = []
        for f in frames:
            tensor_frames.append(self.transform(f))

        # 3D-CNN 입력 규격인 (C, T, H, W) 형태로 스택 및 Transpose
        video_tensor = torch.stack(tensor_frames)  # Shape: (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Shape: (C, T, H, W)

        return video_tensor, torch.tensor(label, dtype=torch.long)


# # --- 테스트 실행 코드 (실제 폴더 내에 데이터가 있을 경우 동작 확인용) ---
# if __name__ == "__main__":
#     # 데이터셋 경로 지정 (임의의 폴더명 지정, 실제 데이터가 담긴 폴더로 변경)
#     dataset = HitAndRunDataset(data_dir='sample', clip_length=30, r_value=1.0)
#     dataloader = DataLoader(dataset, batch_size=15, shuffle=True)

#     for inputs, labels in dataloader:
#         # 기대 출력: [15, 3, 30, 224, 224] (Batch, C, T, H, W)
#         print("입력 텐서 형태:", inputs.shape)
#         print("라벨 텐서 형태:", labels.shape)  # 기대 출력: [15]
#         break
#     pass
