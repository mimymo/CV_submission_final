import torch
import torch.nn.functional as F
from torch import nn

# ──────────────────────────
# 1. Optimizer & Scheduler
# ──────────────────────────
def Make_Optimizer(model):
    # 軽量モデルなら 3e-4 が安定しやすい
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=1e-4
    )

def Make_LR_Scheduler(optimizer):
    # 30 epoch 学習を想定した CosineAnnealing
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6
    )

# ──────────────────────────
# 2. 損失関数
# ──────────────────────────
class DiceBCELoss(nn.Module):
    """
    * 0/255 マスクを自動で 0/1 に補正
    * 動的 pos_weight でクラス不均衡へ対応
    * Dice と BCE を (0.5,0.5) で合成
    * gamma >1 で Tversky 的に FN を追加ペナルティ
    """
    def __init__(self, dice_w: float = 0.5, gamma: float = 1.33):
        super().__init__()
        self.dice_w, self.gamma = dice_w, gamma

    def forward(self, logits, target):
        # ----- マスク正規化 -----
        if target.max() > 1:
            target = target / 255.0
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()

        # ----- BCEWithLogits -----
        pos = target.sum(); neg = target.numel() - pos
        pos_w = neg / (pos + 1e-6)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_w
        )

        # ----- Dice (確率空間) -----
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum((2, 3))
        union = probs.sum((2, 3)) + target.sum((2, 3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        dice = dice.mean()

        # 合成
        return self.dice_w * dice + (1 - self.dice_w) * bce ** self.gamma

# ------------- 追加 Loss -------------
class FocalTverskyLoss(nn.Module):
    """
    α (FP ペナルティ)・β (FN ペナルティ) を調整できる Tversky + γ-focal
    推奨値: α=0.3, β=0.7 (FN を強く罰する)
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1):
        super().__init__()
        self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth

    def forward(self, logits, target):
        if target.max() > 1:  target = target/255.
        if target.ndim == 3:  target = target.unsqueeze(1)
        target = target.float()

        p = torch.sigmoid(logits)
        tp = (p*target).sum((2,3))
        fp = ((1-target)*p).sum((2,3))
        fn = (target*(1-p)).sum((2,3))

        tversky = (tp + self.s) / (tp + self.a*fp + self.b*fn + self.s)
        loss = (1 - tversky) ** self.g
        return loss.mean()
# ------------------------------------


# マルチクラスはシンプルに CE のみ（IoU 評価側で softmax）
def Make_Loss_Function(num_classes: int):
    return DiceBCELoss() if num_classes <= 2 else nn.CrossEntropyLoss()
