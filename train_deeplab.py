# train_deeplab.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FramesMaskDataset
from model import make_deeplab
from metrics import compute_iou_and_dice

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for imgs, targets in tqdm(loader, desc="Training"):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)["out"]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    mean_ious = []

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Validation"):
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)["out"]
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            for p, t in zip(preds, targets_np):
                _, _, mean_iou = compute_iou_and_dice(p, t)
                mean_ious.append(mean_iou)

    return running_loss / len(loader), np.mean(mean_ious)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FramesMaskDataset(args.train_frames, args.train_masks)
    val_dataset   = FramesMaskDataset(args.val_frames, args.val_masks, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = make_deeplab(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_miou = 0.0

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_miou = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val mIoU:   {val_miou:.4f}")

        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pth"))

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_frames", type=str, required=True)
    parser.add_argument("--train_masks", type=str, required=True)
    parser.add_argument("--val_frames", type=str, required=True)
    parser.add_argument("--val_masks", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)