from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from timm.utils.metrics import accuracy, AverageMeter


#### CLASSIFICATION ###


def eval_classifier_top1_and_top5(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label: str = "Test Performance",
    non_blocking: bool = True,
) -> Tuple[float, float, float]:
    # Fmi: Its very important that the model is in eval mode if batchnorms are included...
    model.eval()
    test_loss = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=non_blocking), y.to(
                device, non_blocking=non_blocking
            )
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            acc1, acc5 = accuracy(pred, y, topk=(1, 5))
            top1.update(acc1.item(), X.size(0))
            top5.update(acc5.item(), X.size(0))

        test_loss /= len(dataloader)

    if torch.distributed.is_initialized():
        metrics = torch.tensor(
            [
                top1.sum,
                top1.count,
                top5.sum,
                top5.count,
                test_loss,
            ],
            device=device,
        )
        torch.cuda.synchronize()
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

        top1 = metrics[0].item() / metrics[1].item()
        top5 = metrics[2].item() / metrics[3].item()
        test_loss = metrics[4].item() / torch.distributed.get_world_size()
    else:
        top1, top5 = top1.avg, top5.avg

    print(
        f"{label}: \n Top1 Acc: {top1:.2f}%, Top5 Acc: {top5:.2f}%, Avg loss: {test_loss:.5f} \n"
    )
    return top1, top5, test_loss


def eval_classifier_mAP(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label: str = "Test Performance",
    non_blocking: bool = True,
) -> float:
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=non_blocking), y.to(
                device, non_blocking=non_blocking
            )

            # Forward pass to get logits
            logits = model(X)

            all_logits.append(logits)
            all_targets.append(y)

    # Concatenate all results from all batches (ensure that these are all Tensors)
    all_logits = torch.cat(all_logits)  # Now a single Tensor
    all_targets = torch.cat(all_targets)  # Now a single Tensor

    # Convert Tensors to NumPy arrays for easier processing
    all_logits = all_logits.cpu().numpy()
    all_targets = all_targets.cpu().numpy()

    # Compute mAP for each class
    num_classes = all_logits.shape[1]
    aps = []

    for cls in range(num_classes):
        # Get ground truth and logits for this class
        gt = all_targets[:, cls]
        scores = all_logits[:, cls]

        # Check if there are any positive samples for this class
        if np.sum(gt) == 0:
            # No positive samples in ground truth, skip this class or handle as needed
            print(
                f"Warning: No positive samples for class {cls}. Skipping mAP calculation for this class."
            )
            continue

        # Sort by logits directly (higher logits indicate higher confidence in that class)
        sorted_indices = np.argsort(-scores)
        sorted_gt = gt[sorted_indices]

        # Compute true positives and false positives
        tp = np.cumsum(sorted_gt)
        fp = np.cumsum(1 - sorted_gt)

        # Compute precision and recall
        recalls = tp / np.sum(gt)
        precisions = tp / (tp + fp)

        # 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = precisions[recalls >= t]
            p = np.max(precisions_at_recall) if precisions_at_recall.size > 0 else 0
            ap += p / 11
        aps.append(ap)

    # Compute the mean Average Precision (mAP)
    if len(aps) > 0:
        mAP = 100 * torch.tensor(aps, dtype=torch.float32, device=device).mean()
    else:
        mAP = torch.tensor(0.0, dtype=torch.float32, device=device)

    if torch.distributed.is_initialized():
        torch.cuda.synchronize()
        mAP = mAP.clone()
        torch.distributed.all_reduce(mAP, op=torch.distributed.ReduceOp.SUM)
        mAP /= torch.distributed.get_world_size()

    print(f"{label}: \n mAP: {mAP.item():.4f} \n")

    return float(mAP.item())


### OBJECT DETECTION ###


def compute_ap(tp, fp, n_gt, device):
    """
    Compute Average Precision (AP) from true positives, false positives, and number of ground truths.

    Args:
        tp (Tensor): Cumulative true positives.
        fp (Tensor): Cumulative false positives.
        n_gt (int): Number of ground truth instances.

    Returns:
        ap (float): Average precision.
    """
    if n_gt == 0:  # No ground truth instances for this class
        return 0.0

    # Compute precision and recall
    recall = tp / (n_gt + 1e-16)
    precision = tp / (tp + fp + 1e-16)

    # Add a point at (0, 1) for AP calculation
    recall = torch.cat([torch.tensor([0.0], device=device), recall])
    precision = torch.cat([torch.tensor([1.0], device=device), precision])

    # Compute AP as the area under the precision-recall curve
    ap = torch.trapz(precision, recall).item()
    return ap


def eval_object_detector_ap50(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label: str = "Test Performance",
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.05,
    top_k: int = 100,
    non_blocking: bool = True,
) -> float:
    """
    Optimized evaluation for AP50 metric (IoU = 0.5).

    Args:
        model: Trained object detection model.
        dataloader: DataLoader for the validation dataset.
        device: Device (CPU/GPU) for evaluation.
        iou_thresh: IoU threshold for a match to count as True Positive.
        conf_thresh: Confidence threshold to filter low-confidence predictions.
        top_k: Maximum number of predictions to consider per image.

    Returns:
        ap50 (float): AP at IoU = 0.5.
    """
    model.eval()
    stats = []  # To store TP, FP, confidence, and predicted classes
    n_gt_classes = []  # To store ground truth classes for recall

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device, non_blocking=non_blocking) for img in images]
            targets = [
                {k: v.to(device, non_blocking=non_blocking) for k, v in t.items()}
                for t in targets
            ]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Filter predictions by confidence and keep top_k
                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]

                if conf_thresh > 0:
                    mask = pred_scores > conf_thresh
                    pred_boxes = pred_boxes[mask]
                    pred_scores = pred_scores[mask]
                    pred_labels = pred_labels[mask]

                if len(pred_scores) > top_k:
                    sorted_indices = torch.argsort(-pred_scores)[:top_k]
                    pred_boxes = pred_boxes[sorted_indices]
                    pred_scores = pred_scores[sorted_indices]
                    pred_labels = pred_labels[sorted_indices]

                true_boxes = target["boxes"]
                true_labels = target["labels"]

                n_gt = len(true_boxes)
                n_preds = len(pred_boxes)

                if n_preds == 0:
                    # No predictions: add ground truth stats only
                    n_gt_classes.append(true_labels)
                    stats.append(
                        (
                            torch.zeros(0, device=device),
                            torch.zeros(0, device=device),
                            torch.zeros(0, device=device),
                            torch.zeros(0, device=device),
                        )
                    )
                    continue

                # Compute IoU matrix
                iou_matrix = box_iou(pred_boxes, true_boxes).to(device)

                # Match predictions to ground truth (IoU >= threshold)
                tp = torch.zeros(n_preds, dtype=torch.bool, device=device)
                fp = torch.zeros(n_preds, dtype=torch.bool, device=device)

                if n_gt > 0:
                    matched = iou_matrix >= iou_thresh
                    for pred_idx in range(matched.size(0)):
                        if matched[pred_idx].any():
                            gt_idx = matched[pred_idx].nonzero(as_tuple=True)[0][0]
                            matched[:, gt_idx] = False  # Prevent double matching
                            tp[pred_idx] = True
                        else:
                            fp[pred_idx] = True
                else:
                    # No ground truth: all predictions are false positives
                    fp[:] = True

                # Record stats
                stats.append((tp, fp, pred_scores, pred_labels))
                n_gt_classes.append(true_labels)

    # Aggregate stats
    tp, fp, conf, pred_classes = [torch.cat(x, 0) for x in zip(*stats)]
    gt_classes = torch.cat(n_gt_classes, 0)

    # Compute AP50 for each class

    unique_classes = torch.unique(gt_classes)
    aps = torch.empty(len(unique_classes), dtype=torch.float32, device=device)
    for i, cls in enumerate(unique_classes):
        pred_mask = pred_classes == cls
        cls_tp = tp[pred_mask]
        cls_fp = fp[pred_mask]
        cls_conf = conf[pred_mask]
        n_gt_cls = (gt_classes == cls).sum().item()

        # Sort predictions by confidence
        if len(cls_conf) > 0:
            sort_idx = torch.argsort(-cls_conf)
            cls_tp = cls_tp[sort_idx]
            cls_fp = cls_fp[sort_idx]

        # Compute cumulative TP and FP
        cls_tp = cls_tp.cumsum(0)
        cls_fp = cls_fp.cumsum(0)

        # Compute AP for the class
        ap = compute_ap(cls_tp, cls_fp, n_gt_cls, device=device)
        aps[i] = ap

    # Compute mean AP50 across all classes
    ap50 = 100 * aps.mean()

    if torch.distributed.is_initialized():
        torch.cuda.synchronize()
        torch.distributed.all_reduce(ap50, op=torch.distributed.ReduceOp.SUM)
        ap50 /= torch.distributed.get_world_size()

    print(label + f": AP50: {ap50.item():.4f}")
    return ap50.item()
