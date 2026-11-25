"""
ZitounEye — Evaluation Metrics
================================
Per-class IoU, F1, precision, recall for semantic segmentation.
Optimized confusion matrix computation via vectorized NumPy ops.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class SegmentationMetrics:
    """
    Accumulates a confusion matrix across batches and computes
    per-class and aggregate segmentation metrics.
    
    Usage:
        metrics = SegmentationMetrics(num_classes=7)
        for batch in dataloader:
            preds = model(batch["image"])
            metrics.update(preds.argmax(1), batch["mask"])
        results = metrics.compute()
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -1,
    ):
        """
        Update confusion matrix with a batch of predictions.
        
        Args:
            predictions: (B, H, W) predicted class IDs
            targets: (B, H, W) ground truth class IDs
            ignore_index: class ID to ignore
        """
        preds = predictions.cpu().numpy().flatten()
        trues = targets.cpu().numpy().flatten()
        
        # Filter ignored pixels
        valid = trues != ignore_index
        preds = preds[valid]
        trues = trues[valid]
        
        # Clip to valid range
        preds = np.clip(preds, 0, self.num_classes - 1)
        trues = np.clip(trues, 0, self.num_classes - 1)
        
        # Fast confusion matrix via np.bincount
        indices = trues * self.num_classes + preds
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        cm = cm.reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += cm
    
    def compute(self) -> Dict[str, any]:
        """
        Compute all metrics from accumulated confusion matrix.
        
        Returns dict with:
            - per_class_iou: {class_name: iou}
            - per_class_f1: {class_name: f1}
            - per_class_precision: {class_name: precision}
            - per_class_recall: {class_name: recall}
            - mean_iou: float (excludes background)
            - overall_accuracy: float
            - confusion_matrix: np.ndarray
        """
        cm = self.confusion_matrix
        
        # Per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-8)
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-8)
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-8)
        
        # F1 = 2 * precision * recall / (precision + recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Overall accuracy
        accuracy = tp.sum() / (cm.sum() + 1e-8)
        
        # Mean IoU (exclude background class 0)
        valid_classes = [i for i in range(1, self.num_classes) if cm.sum(axis=1)[i] > 0]
        mean_iou = iou[valid_classes].mean() if valid_classes else 0.0
        
        results = {
            "per_class_iou": {self.class_names[i]: float(iou[i]) for i in range(self.num_classes)},
            "per_class_f1": {self.class_names[i]: float(f1[i]) for i in range(self.num_classes)},
            "per_class_precision": {self.class_names[i]: float(precision[i]) for i in range(self.num_classes)},
            "per_class_recall": {self.class_names[i]: float(recall[i]) for i in range(self.num_classes)},
            "mean_iou": float(mean_iou),
            "overall_accuracy": float(accuracy),
            "confusion_matrix": cm,
        }
        
        return results
    
    def print_report(self, results: Optional[Dict] = None) -> None:
        """Print formatted metrics report."""
        if results is None:
            results = self.compute()
        
        print("\n" + "=" * 70)
        print("SEGMENTATION METRICS REPORT")
        print("=" * 70)
        
        header = f"{'Class':<20} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}"
        print(header)
        print("-" * 70)
        
        for i, name in enumerate(self.class_names):
            iou = results["per_class_iou"][name]
            f1 = results["per_class_f1"][name]
            prec = results["per_class_precision"][name]
            rec = results["per_class_recall"][name]
            
            # Highlight olive class
            marker = "  ★" if name == "olive_grove" else ""
            print(f"{name:<20} {iou:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}{marker}")
        
        print("-" * 70)
        print(f"{'Mean IoU (no bg)':<20} {results['mean_iou']:>8.4f}")
        print(f"{'Overall Accuracy':<20} {results['overall_accuracy']:>8.4f}")
        print("=" * 70)
        
        # Olive-specific analysis
        olive_iou = results["per_class_iou"].get("olive_grove", 0.0)
        vineyard_iou = results["per_class_iou"].get("vineyard", 0.0)
        
        print(f"\n🫒 Olive Grove IoU: {olive_iou:.4f}")
        if olive_iou < 0.70:
            print("   ⚠️  Below 0.70 threshold — review labels or add more training data")
        else:
            print("   ✅ Above 0.70 threshold — good!")
        
        # Check olive↔vineyard confusion
        cm = results["confusion_matrix"]
        if cm.sum(axis=1)[1] > 0:  # if olive has any true pixels
            olive_as_vineyard = cm[1, 2] / cm.sum(axis=1)[1]
            vineyard_as_olive = cm[2, 1] / cm.sum(axis=1)[2] if cm.sum(axis=1)[2] > 0 else 0
            
            print(f"\n🔍 Confusion Analysis:")
            print(f"   Olive mislabeled as vineyard: {olive_as_vineyard:.1%}")
            print(f"   Vineyard mislabeled as olive: {vineyard_as_olive:.1%}")
            
            if olive_as_vineyard > 0.10:
                print("   ⚠️  High olive→vineyard confusion! Consider multi-temporal features.")
