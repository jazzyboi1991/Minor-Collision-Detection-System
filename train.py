import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CollisionDataset
from model import Simple3DCNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = CollisionDataset(
    json_path='data.json'
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


model = Simple3DCNN().to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


EPOCHS = 30


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for videos, labels in loader:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {total_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


torch.save(
    model.state_dict(),
    "collision_model.pth"
)

print("Model Saved")