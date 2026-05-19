import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import CollisionDataset
from model import Simple3DCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", default="data.json")
    parser.add_argument("--save-path", default="collision_model_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(dataset, args, device):
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small to create a train/validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    return train_loader, val_loader


def compute_class_weights(dataset, device):
    labels = [int(item["label"]) for item in dataset.data]
    if not labels:
        return None

    counts = torch.bincount(torch.tensor(labels), minlength=2).float()
    if torch.any(counts == 0):
        return None

    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / total if total else 0.0
    accuracy = 100 * correct / total if total else 0.0
    return avg_loss, accuracy


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = CollisionDataset(json_path=args.json_path)
    train_loader, val_loader = make_loaders(dataset, args, device)

    model = Simple3DCNN().to(device)
    class_weights = compute_class_weights(dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model: {args.save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
