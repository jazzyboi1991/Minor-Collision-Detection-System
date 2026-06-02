import json
import torch
from torch.utils.data import Dataset


class CollisionDataset(Dataset):

    def __init__(self, json_path):

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        frames = torch.as_tensor(
            item['frames'],
            dtype=torch.float32
        )

        # (T,H,W,C) → (T,C,H,W)
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)

        if frames.ndim != 4:
            raise ValueError(
                f"Expected frames to have 4 dimensions, got shape {tuple(frames.shape)}"
            )

        frames = frames / 255.0

        label = torch.as_tensor(
            item['label'],
            dtype=torch.long
        )

        return frames, label
