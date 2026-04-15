# metrics.py

import numpy as np

def compute_iou_and_dice(pred_mask, true_mask, num_classes=2):
    ious = []
    dices = []

    for c in range(num_classes):
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)

        intersection = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()

        iou = intersection / union if union > 0 else 1.0
        dice = (2 * intersection) / (pred_c.sum() + true_c.sum()) \
               if (pred_c.sum() + true_c.sum()) > 0 else 1.0

        ious.append(iou)
        dices.append(dice)

    return ious, dices, np.mean(ious)