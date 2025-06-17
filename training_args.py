import torch
import torch.nn.functional as F
from torch import nn

def Make_Optimizer(model):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4)

def Make_LR_Scheduler(opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=30, eta_min=1e-6)

# Edge Detection Mask
kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                  dtype=torch.float32).view(1, 1, 3, 3)
ky = kx.transpose(2, 3)
def edge_mask(t: torch.Tensor):
    gx = F.conv2d(t, kx.to(t.device), padding=1)
    gy = F.conv2d(t, ky.to(t.device), padding=1)
    g = torch.sqrt(gx**2 + gy**2)
    return (g > 0).float()

# Loss Functions (probability input)
class DiceBCELoss(nn.Module):
    # Dice BCE 確率入力
    def __init__(self, dice_w=0.5):
        super().__init__()
        self.dw = dice_w
    def forward(self, probs, target):
        if target.max() > 1: target = target / 255.
        if target.ndim == 3: target = target.unsqueeze(1)
        target = target.float()

        bce = F.binary_cross_entropy(probs, target)

        dice = 1 - (2 * (probs * target).sum((2, 3)) + 1) / \
                   ((probs + target).sum((2, 3)) + 1)
        return self.dw * dice.mean() + (1 - self.dw) * bce

class FocalTverskyLoss(nn.Module):
    #Focal Tversky Loss 確率入力
    def __init__(self, a=0.3, b=0.7, g=0.75):
        super().__init__()
        self.a, self.b, self.g = a, b, g
    def forward(self, probs, target):
        if target.max() > 1: target = target / 255.
        if target.ndim == 3: target = target.unsqueeze(1)
        target = target.float()
        tp = (probs * target).sum((2, 3))
        fp = ((1 - target) * probs).sum((2, 3))
        fn = (target * (1 - probs)).sum((2, 3))
        tv = (tp + 1) / (tp + self.a * fp + self.b * fn + 1)
        return ((1 - tv) ** self.g).mean()

class ComboLoss(nn.Module):
    # 0.4 DiceBCE + 0.4 FocalTversky + 0.2 Edge-BCE
    def __init__(self):
        super().__init__()
        self.dice = DiceBCELoss()
        self.ft   = FocalTverskyLoss()
        self.bce  = nn.BCELoss()
    def forward(self, probs, target):
        if target.max() > 1: target = target / 255.
        if target.ndim == 3: target = target.unsqueeze(1)
        target = target.float()

        base_loss = 0.4 * self.dice(probs, target) + 0.4 * self.ft(probs, target)

        # エッジ損失
        with torch.no_grad():
            e_gt = edge_mask(target)
        e_pr = edge_mask(probs)
        edge_loss = self.bce(e_pr, e_gt)

        return base_loss + 0.2 * edge_loss

# Factory
def Make_Loss_Function(num_classes: int):
    return ComboLoss() if num_classes <= 2 else nn.CrossEntropyLoss()
