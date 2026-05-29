import os
import cv2
import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

import config


class HitAndRunDataset(Dataset):
    def __init__(
        self,
        data_dir=config.DATA_DIR,
        clip_length=config.CLIP_LENGTH,
        r_value=config.R_VALUE,
        resize=config.RESIZE,
    ):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.r_value = r_value
        self.resize = resize
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

        self.file_names = sorted(f.rsplit('.', 1)[0] for f in os.listdir(data_dir) if f.endswith('.mp4'))
        self.samples = self._build_index()

    def __len__(self):
        return len(self.samples)

    def _build_index(self):
        samples = []
        for file_name in self.file_names:
            mp4_path = os.path.join(self.data_dir, f"{file_name}.mp4")
            txt_path = os.path.join(self.data_dir, f"{file_name}.txt")
            bboxes, action = self._parse_annotation(txt_path)

            if action is not None:
                class_str = action[0]
                target_id = int(float(action[1]))
                start_f = int(float(action[2]))
            else:
                class_str = 'S'
                target_id = 0
                start_f = 0

            if target_id not in bboxes:
                target_id = next(iter(bboxes), 0)
            target_bbox = bboxes.get(target_id, [0, 0, self.resize[0], self.resize[1]])
            label = 1 if class_str == 'A' else 0

            samples.append({
                'file_name': file_name,
                'mp4_path': mp4_path,
                'label': label,
                'start_f': start_f,
                'bbox': target_bbox,
            })
        return samples

    def _parse_annotation(self, txt_path):
        bboxes = {}
        action = None
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if not parts or len(parts) < 2:
                        continue
                    if parts[0] == 'car' and len(parts) >= 6:
                        bboxes[int(parts[1])] = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
                    elif parts[0] in ['A', 'S']:
                        action = parts
        return bboxes, action

    def _crop_and_pad(self, frame, bbox, r):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox

        veh_w, veh_h = (x_max - x_min) * r, (y_max - y_min) * r
        cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2

        square_size = int(max(veh_w, veh_h))
        new_x_min, new_y_min = cx - square_size // 2, cy - square_size // 2
        new_x_max, new_y_max = cx + square_size // 2, cy + square_size // 2

        pad_left   = max(0, -new_x_min)
        pad_top    = max(0, -new_y_min)
        pad_right  = max(0, new_x_max - w)
        pad_bottom = max(0, new_y_max - h)

        valid_x_min = max(0, new_x_min)
        valid_y_min = max(0, new_y_min)
        valid_x_max = min(w, new_x_max)
        valid_y_max = min(h, new_y_max)
        cropped_frame = frame[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped_frame = np.pad(
                cropped_frame,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0,
            )

        return cv2.resize(cropped_frame, self.resize, interpolation=cv2.INTER_LINEAR)

    def _apply_augmentation(self, frames):
        # 논문 Section 4.3 기반 데이터 증강 (모든 프레임에 동일한 변환 적용)
        do_hflip = random.random() > 0.5

        do_rot = random.random() > 0.5
        angle = random.uniform(-10, 10) if do_rot else 0

        do_jitter = random.random() > 0.5
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)

        # Random Crop: occluded/restricted view 시뮬레이션 (논문 Section 4.3)
        # 비디오 전체에 동일한 crop 파라미터 적용
        do_crop = random.random() > 0.5
        h, w = self.resize[1], self.resize[0]
        if do_crop:
            crop_ratio = random.uniform(0.8, 1.0)
            new_h = int(h * crop_ratio)
            new_w = int(w * crop_ratio)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

        aug_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if do_hflip:
                img = TF.hflip(img)
            if do_rot:
                img = TF.rotate(img, angle)
            if do_jitter:
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
            if do_crop:
                img = TF.crop(img, top, left, new_h, new_w)
                img = TF.resize(img, [h, w])
            aug_frames.append(np.array(img))

        return aug_frames

    def _frames_to_tensor(self, frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - self.mean) / self.std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cap = cv2.VideoCapture(sample['mp4_path'])
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_f'])

        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self._crop_and_pad(frame, sample['bbox'], self.r_value))
        cap.release()

        while len(frames) < self.clip_length:
            frames.append(frames[-1] if frames else np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))

        frames = self._apply_augmentation(frames)

        return self._frames_to_tensor(frames), torch.tensor(sample['label'], dtype=torch.long)
