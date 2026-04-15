# evaluate.py

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import FramesMaskDataset
from model import make_deeplab
from metrics import compute_iou_and_dice

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FramesMaskDataset(args.frames_dir, args.masks_dir, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = make_deeplab(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_ious = []
    all_dices = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)["out"]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            targets_np = targets.numpy()[0]

            ious, dices, _ = compute_iou_and_dice(preds, targets_np)

            all_ious.append(ious[1])
            all_dices.append(dices[1])

    print("Mean IoU:", np.mean(all_ious))
    print("Mean Dice:", np.mean(all_dices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    main(args)