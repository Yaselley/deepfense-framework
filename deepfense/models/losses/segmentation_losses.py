import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss


class _FramewiseSegmentationLossBase(BaseLoss):
    """Shared plumbing for framewise Dice / IoU / SSIM losses.

    Not registered itself -- subclasses provide the actual loss math.
    """

    def __init__(self, config):
        super().__init__(config)
        self.in_dim = config["embedding_dim"]
        self.num_classes = config["n_classes"]
        self.ignore_index = int(config.get("ignore_index", -100))
        self.eps = float(config.get("eps", 1e-6))
        self.reduction = config.get("reduction", "mean")

        self.fc = nn.Linear(self.in_dim, self.num_classes)

        class_weights = config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            if class_weights.numel() != self.num_classes:
                raise ValueError(
                    f"class_weights has {class_weights.numel()} entries, "
                    f"expected {self.num_classes}"
                )
        # Registered as a buffer so it moves with .to(device) automatically.
        self.register_buffer("class_weights", class_weights, persistent=False)

    # ---- shared helpers -------------------------------------------------

    def _align(self, logits, targets):
        """Reconcile time dimension of logits vs targets (pad/truncate)."""
        t_log = logits.size(1)
        t_tgt = targets.size(1)
        if t_log == t_tgt:
            return logits, targets
        if t_tgt < t_log:
            pad = t_log - t_tgt
            targets = F.pad(targets, (0, pad), value=self.ignore_index)
            return logits, targets
        return logits, targets[:, :t_log]

    def _probs_and_onehot(self, logits, targets):
        """Softmax probs, one-hot targets, and a validity mask, all aligned.

        Positions where targets == ignore_index are zeroed out of both
        tensors via `mask` so they don't contribute to the loss.
        """
        logits, targets = self._align(logits, targets)
        mask = targets != self.ignore_index  # (B, T)

        safe_targets = targets.clone()
        safe_targets[~mask] = 0  # dummy valid index, will be masked out anyway

        probs = F.softmax(logits, dim=-1)  # (B, T, C)
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).float()  # (B, T, C)
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
        return probs, one_hot, mask_f

    def _weighted_class_average(self, per_class_loss):
        """per_class_loss: (B, C) -> per-sample (B,), averaged over classes."""
        if self.class_weights is not None:
            w = self.class_weights / self.class_weights.sum()
            return (per_class_loss * w.unsqueeze(0)).sum(dim=1)
        return per_class_loss.mean(dim=1)

    def _apply_reduction(self, per_sample):
        """per_sample: (B,) -> final scalar/vector per `self.reduction`."""
        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        if self.reduction == "none":
            return per_sample
        raise ValueError(f"Unknown reduction: {self.reduction}")

    def _reduce(self, per_class_loss):
        """per_class_loss: (B, C) -> apply class weights, then `reduction`."""
        return self._apply_reduction(self._weighted_class_average(per_class_loss))

    def _check_input(self, embeddings):
        if embeddings.dim() != 3:
            raise ValueError(
                f"{type(self).__name__} expects (B,T,D), got {tuple(embeddings.shape)}"
            )

    def get_logits(self, embeddings):
        return self.fc(embeddings)

    def get_score(self, embeddings):
        """Identical to FramewiseCrossEntropy: logit structure is unchanged."""
        logits = self.get_logits(embeddings)
        if self.num_classes == 2:
            return logits[..., self.bonafide_label] - logits[..., self.spoof_label]
        return logits


@register_loss("FramewiseDice")
class FramewiseDiceLoss(_FramewiseSegmentationLossBase):
    """Soft Dice loss computed per class, per sample, over the time axis."""

    def _per_sample_loss(self, embeddings, targets, logits=None):
        """Per-sample Dice loss (B,), before the final `reduction` step."""
        if logits is None:
            logits = self.get_logits(embeddings)

        probs, one_hot, mask = self._probs_and_onehot(logits, targets)
        probs = probs * mask
        one_hot = one_hot * mask

        intersection = (probs * one_hot).sum(dim=1)  # (B, C)
        cardinality = probs.sum(dim=1) + one_hot.sum(dim=1)  # (B, C)

        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        loss_per_class = 1 - dice
        return self._weighted_class_average(loss_per_class)

    def forward(self, embeddings, targets, logits=None):
        self._check_input(embeddings)
        return self._apply_reduction(self._per_sample_loss(embeddings, targets, logits))


@register_loss("FramewiseIoU")
class FramewiseIoULoss(_FramewiseSegmentationLossBase):
    """Soft IoU (Jaccard) loss computed per class, per sample, over time."""

    def _per_sample_loss(self, embeddings, targets, logits=None):
        """Per-sample IoU loss (B,), before the final `reduction` step."""
        if logits is None:
            logits = self.get_logits(embeddings)

        probs, one_hot, mask = self._probs_and_onehot(logits, targets)
        probs = probs * mask
        one_hot = one_hot * mask

        intersection = (probs * one_hot).sum(dim=1)  # (B, C)
        total = probs.sum(dim=1) + one_hot.sum(dim=1)  # (B, C)
        union = total - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        loss_per_class = 1 - iou
        return self._weighted_class_average(loss_per_class)

    def forward(self, embeddings, targets, logits=None):
        self._check_input(embeddings)
        return self._apply_reduction(self._per_sample_loss(embeddings, targets, logits))


@register_loss("FramewiseSSIM")
class FramewiseSSIMLoss(_FramewiseSegmentationLossBase):

    def __init__(self, config):
        super().__init__(config)
        self.window_size = int(config.get("ssim_window_size", 10))
        self.sigma = float(config.get("ssim_sigma", 1.5))
        # Standard SSIM stability constants for data_range = 1.
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2

    def _make_kernel(self, t_len, device):
        window = min(self.window_size, t_len)
        if window % 2 == 0:
            window -= 1
        window = max(window, 1)
        coords = torch.arange(window, dtype=torch.float32, device=device) - window // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        # depthwise conv1d kernel: (C, 1, window)
        return g.view(1, 1, -1).repeat(self.num_classes, 1, 1)

    def _per_sample_loss(self, embeddings, targets, logits=None):
        """Per-sample SSIM loss (B,), before the final `reduction` step."""
        if logits is None:
            logits = self.get_logits(embeddings)

        logits, targets = self._align(logits, targets)
        mask = targets != self.ignore_index  # (B, T)
        safe_targets = targets.clone()
        safe_targets[~mask] = 0

        probs = F.softmax(logits, dim=-1)  # (B, T, C)
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).float()  # (B, T, C)

        x = probs.transpose(1, 2)      # (B, C, T)
        y = one_hot.transpose(1, 2)    # (B, C, T)
        m = mask.unsqueeze(1).float()  # (B, 1, T)

        t_len = x.size(-1)
        kernel = self._make_kernel(t_len, x.device)
        pad = kernel.size(-1) // 2

        def local_mean(z):
            return F.conv1d(z, kernel, padding=pad, groups=self.num_classes)

        mu_x, mu_y = local_mean(x), local_mean(y)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x2 = local_mean(x * x) - mu_x2
        sigma_y2 = local_mean(y * y) - mu_y2
        sigma_xy = local_mean(x * y) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2) + self.eps
        )  # (B, C, T)

        # Zero out ignored frames and average only over valid ones.
        ssim_map = ssim_map * m
        valid_counts = m.sum(dim=2).clamp(min=1.0)  # (B, 1)
        ssim_per_class = ssim_map.sum(dim=2) / valid_counts  # (B, C)

        loss_per_class = 1 - ssim_per_class
        return self._weighted_class_average(loss_per_class)

    def forward(self, embeddings, targets, logits=None):
        self._check_input(embeddings)
        return self._apply_reduction(self._per_sample_loss(embeddings, targets, logits))




@register_loss("FramewiseCEDice")
class FramewiseCEDiceLoss(FramewiseDiceLoss):
    """``ce_weight * CrossEntropy + dice_weight * (1 - Dice)``, shared logits."""

    def __init__(self, config):
        super().__init__(config)
        self.ce_weight = float(config.get("ce_weight", 0.5))
        self.dice_weight = float(config.get("dice_weight", 0.5))
        # reduction="none" always: this class applies the final reduction
        # itself, once, to the combined per-sample loss.
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=self.ignore_index, reduction="none"
        )

    def forward(self, embeddings, targets, logits=None):
        self._check_input(embeddings)
        if logits is None:
            logits = self.get_logits(embeddings)
        logits, targets = self._align(logits, targets)

        b, t, c = logits.shape
        flat_ce = self.ce_criterion(logits.reshape(b * t, c), targets.reshape(b * t))
        valid = (targets != self.ignore_index).float()  # (B, T)
        per_sample_ce = flat_ce.reshape(b, t).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        per_sample_dice = self._per_sample_loss(embeddings, targets, logits=logits)

        combined = self.ce_weight * per_sample_ce + self.dice_weight * per_sample_dice
        return self._apply_reduction(combined)


@register_loss("FramewiseCEIoU")
class FramewiseCEIoULoss(FramewiseIoULoss):
    """``ce_weight * CrossEntropy + iou_weight * (1 - IoU)``, shared logits."""

    def __init__(self, config):
        super().__init__(config)
        self.ce_weight = float(config.get("ce_weight", 0.5))
        self.iou_weight = float(config.get("iou_weight", 0.5))
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=self.ignore_index, reduction="none"
        )

    def forward(self, embeddings, targets, logits=None):
        self._check_input(embeddings)
        if logits is None:
            logits = self.get_logits(embeddings)
        logits, targets = self._align(logits, targets)

        b, t, c = logits.shape
        flat_ce = self.ce_criterion(logits.reshape(b * t, c), targets.reshape(b * t))
        valid = (targets != self.ignore_index).float()  # (B, T)
        per_sample_ce = flat_ce.reshape(b, t).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        per_sample_iou = self._per_sample_loss(embeddings, targets, logits=logits)

        combined = self.ce_weight * per_sample_ce + self.iou_weight * per_sample_iou
        return self._apply_reduction(combined)


@register_loss("FramewiseCESSIM")
class FramewiseCESSIMLoss(FramewiseSSIMLoss):
    """``ce_weight * CrossEntropy + ssim_weight * (1 - SSIM)``, shared logits."""

    def __init__(self, config):
        super().__init__(config)
        self.ce_weight = float(config.get("ce_weight", 0.5))
        self.ssim_weight = float(config.get("ssim_weight", 0.5))
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=self.ignore_index, reduction="none"
        )

    def forward(self, embeddings, targets, logits=None):
        self._check_input(embeddings)
        if logits is None:
            logits = self.get_logits(embeddings)
        logits, targets = self._align(logits, targets)

        b, t, c = logits.shape
        flat_ce = self.ce_criterion(logits.reshape(b * t, c), targets.reshape(b * t))
        valid = (targets != self.ignore_index).float()  # (B, T)
        per_sample_ce = flat_ce.reshape(b, t).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        per_sample_ssim = self._per_sample_loss(embeddings, targets, logits=logits)

        combined = self.ce_weight * per_sample_ce + self.ssim_weight * per_sample_ssim
        return self._apply_reduction(combined)