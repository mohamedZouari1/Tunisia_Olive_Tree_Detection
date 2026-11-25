"""
ZitounEye — Loss Functions
===========================
Combined Dice + Focal Loss for multi-class semantic segmentation.
Handles class imbalance via per-class weighting and focal modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    
    Operates on per-class probabilities and computes 1 - (2*intersection / union)
    averaged across classes. Smooth term prevents division by zero.
    """
    
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) integer class labels
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Create valid mask
        valid = (targets != self.ignore_index)  # (B, H, W)
        targets_clean = targets.clone()
        targets_clean[~valid] = 0
        
        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        one_hot = F.one_hot(targets_clean, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Zero out invalid pixels
        valid_mask = valid.unsqueeze(1).float()  # (B, 1, H, W)
        one_hot = one_hot * valid_mask
        probs = probs * valid_mask
        
        # Compute per-class Dice
        dims = (0, 2, 3)  # reduce over batch and spatial dims
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            dice_loss = 1.0 - (dice_per_class * weights).sum() / weights.sum()
        else:
            dice_loss = 1.0 - dice_per_class.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss — downweights easy examples, focuses on hard ones.
    
    Particularly effective for class imbalance (many background pixels,
    few olive pixels).
    
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    """
    
    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) integer class labels
        """
        # Standard cross-entropy per pixel
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)
        
        # Focal modulation
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Gather probability of the true class
        targets_clean = targets.clone()
        targets_clean[targets == self.ignore_index] = 0
        p_t = probs.gather(1, targets_clean.unsqueeze(1)).squeeze(1)  # (B, H, W)
        
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Mask invalid pixels
        valid = (targets != self.ignore_index)
        focal_loss = focal_loss[valid].mean()
        
        return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss.
    
    This is the recommended loss for olive grove segmentation because:
    1. Dice handles region-level overlap (good for varying parcel sizes)
    2. Focal handles pixel-level hard examples (olive/vineyard confusion zones)
    3. Together they address both macro and micro classification errors
    """
    
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: Optional[list] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        
        self.dice = DiceLoss(
            num_classes=num_classes,
            class_weights=weights,
            ignore_index=ignore_index,
        )
        self.focal = FocalLoss(
            num_classes=num_classes,
            gamma=focal_gamma,
            class_weights=weights,
            ignore_index=ignore_index,
        )
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d = self.dice(logits, targets)
        f = self.focal(logits, targets)
        return self.dice_weight * d + self.focal_weight * f


# ═══════════════════════════════════════════════════════════════
# ADVANCED LOSSES — for pushing olive IoU beyond 0.80
# ═══════════════════════════════════════════════════════════════


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss — provably optimal IoU surrogate.
    
    Unlike Dice loss which approximates IoU, Lovász-Softmax directly
    optimizes the IoU metric via a convex surrogate derived from the
    Lovász extension of submodular functions.
    
    Reference: Berman et al., "The Lovász-Softmax loss: A tractable 
    surrogate for the optimization of the intersection-over-union 
    measure in neural networks", CVPR 2018.
    
    This typically gives +2-3% IoU over Dice loss alone.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def _lovasz_softmax_flat(self, probas, labels):
        """Multi-class Lovász-Softmax on flattened predictions."""
        if probas.numel() == 0:
            return probas * 0.0
        
        C = probas.size(1)
        losses = []
        
        for c in range(C):
            fg = (labels == c).float()  # foreground for class c
            if fg.sum() == 0 and c == 0:
                continue  # skip background if empty
            
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = fg[perm]
            
            grad = self._lovasz_grad(fg_sorted)
            loss_c = torch.dot(errors_sorted, grad)
            
            if self.class_weights is not None:
                loss_c = loss_c * self.class_weights[c].to(probas.device)
            
            losses.append(loss_c)
        
        if not losses:
            return torch.tensor(0.0, device=probas.device, requires_grad=True)
        
        return torch.stack(losses).mean()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        B, C, H, W = logits.shape
        probas = F.softmax(logits, dim=1)
        
        # Flatten spatial dims
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        
        # Remove ignored pixels
        valid = targets_flat != self.ignore_index
        probas_flat = probas_flat[valid]
        targets_flat = targets_flat[valid]
        
        return self._lovasz_softmax_flat(probas_flat.unsqueeze(0).permute(1, 0, 2).squeeze(2).T.unsqueeze(0).squeeze(0), 
                                          targets_flat)
    

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that penalizes errors at parcel edges.
    
    Olive groves have distinctive boundaries where they meet roads,
    buildings, or other crop types. Getting these edges right is
    crucial for accurate area estimation.
    
    Uses morphological dilation to find boundary pixels, then
    applies extra penalty to misclassifications in those regions.
    """
    
    def __init__(
        self,
        num_classes: int,
        boundary_width: int = 3,
        boundary_weight: float = 2.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
    
    def _compute_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Find pixels near class boundaries via max-pooling trick."""
        B, H, W = targets.shape
        
        # One-hot encode
        targets_clean = targets.clone()
        targets_clean[targets == self.ignore_index] = 0
        one_hot = F.one_hot(targets_clean, self.num_classes).float()  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Dilate each class mask
        kernel_size = self.boundary_width * 2 + 1
        dilated = F.max_pool2d(
            one_hot,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        
        # Erode each class mask (dilate the complement)
        eroded = 1.0 - F.max_pool2d(
            1.0 - one_hot,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        
        # Boundary = dilated - eroded
        boundary = (dilated - eroded).sum(dim=1)  # (B, H, W)
        boundary = (boundary > 0).float()
        
        return boundary
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        boundary_mask = self._compute_boundary_mask(targets)
        
        # Standard CE loss
        ce_loss = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)
        
        # Upweight boundary pixels
        weight_map = 1.0 + boundary_mask * (self.boundary_weight - 1.0)
        weighted_loss = (ce_loss * weight_map)
        
        valid = (targets != self.ignore_index)
        return weighted_loss[valid].mean()


class UltimateLoss(nn.Module):
    """
    🏆 The Ultimate Loss for olive grove segmentation.
    
    Combines 4 complementary losses:
    1. Dice Loss (0.3) — region-level overlap
    2. Focal Loss (0.3) — hard pixel mining  
    3. Lovász-Softmax (0.2) — direct IoU optimization
    4. Boundary Loss (0.2) — parcel edge accuracy
    
    This combination addresses every failure mode:
    - Class imbalance → Focal + class weights
    - Small parcels → Dice preserves small regions
    - IoU optimization → Lovász directly optimizes the metric
    - Edge blurriness → Boundary loss sharpens predictions
    
    Use this with: loss_type: "ultimate" in config
    """
    
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[list] = None,
        focal_gamma: float = 2.0,
        boundary_width: int = 3,
        ignore_index: int = -1,
    ):
        super().__init__()
        
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        
        self.dice = DiceLoss(num_classes, class_weights=weights, ignore_index=ignore_index)
        self.focal = FocalLoss(num_classes, gamma=focal_gamma, class_weights=weights, ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxLoss(num_classes, class_weights=weights, ignore_index=ignore_index)
        self.boundary = BoundaryLoss(num_classes, boundary_width=boundary_width, ignore_index=ignore_index)
        
        # Loss weights — tuned for olive grove detection
        self.w_dice = 0.3
        self.w_focal = 0.3
        self.w_lovasz = 0.2
        self.w_boundary = 0.2
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = (
            self.w_dice * self.dice(logits, targets) +
            self.w_focal * self.focal(logits, targets) +
            self.w_lovasz * self.lovasz(logits, targets) +
            self.w_boundary * self.boundary(logits, targets)
        )
        return loss
