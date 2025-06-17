import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────
# 1. Basic blocks
# ─────────────────────────────
class DownsamplerBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Conv2d(n_in, n_out - n_in, 3, 2, 1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn   = nn.BatchNorm2d(n_out)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return self.act(self.bn(x))


class NonBottleneck1D(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, p_drop: float = 0.1):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(ch, ch, (3, 1), 1, (1, 0), bias=False)
        self.conv1x3_1 = nn.Conv2d(ch, ch, (1, 3), 1, (0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv3x1_2 = nn.Conv2d(ch, ch, (3, 1), 1,
                                   (dilation, 0), dilation=(dilation, 1), bias=False)
        self.conv1x3_2 = nn.Conv2d(ch, ch, (1, 3), 1,
                                   (0, dilation), dilation=(1, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1x3_1(self.conv3x1_1(x))))
        out = self.conv1x3_2(self.conv3x1_2(out))
        out = self.drop(self.bn2(out))
        return self.act(out + x)


class UpsamplerBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(n_in, n_out, 3, 2, 1,
                                         output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

# ─────────────────────────────
# SEBlock for channel attention
# ─────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

# ─────────────────────────────
# 2. ERF-UNet (with SEBlock)
# ─────────────────────────────
class submission_20225757(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # --- Encoder ---
        self.stem  = DownsamplerBlock(in_channels, 16)   # 1/2
        self.down1 = DownsamplerBlock(16, 64)            # 1/4
        self.enc1  = nn.Sequential(*[NonBottleneck1D(64) for _ in range(3)])

        self.down2 = DownsamplerBlock(64, 128)           # 1/8
        enc2_layers = []
        for _ in range(2):
            for d in (2, 4, 8, 16):
                enc2_layers.append(NonBottleneck1D(128, dilation=d))
        self.enc2 = nn.Sequential(*enc2_layers)

        # --- Decoder ---
        self.up1  = UpsamplerBlock(128, 64)              # 1/4
        self.se4  = SEBlock(64)
        self.dec1 = nn.Sequential(NonBottleneck1D(64), NonBottleneck1D(64))

        self.up2  = UpsamplerBlock(64, 16)               # 1/2
        self.se2  = SEBlock(16)
        self.dec2 = NonBottleneck1D(16)

        self.refine = nn.Conv2d(16, 16, 1, bias=False)
        self.head   = nn.ConvTranspose2d(16, num_classes, 2, 2)
        
        

        self._init_weights()

    def forward(self, x):
        x_s = self.stem(x)               # 1/2
        x4  = self.enc1(self.down1(x_s)) # 1/4
        x8  = self.enc2(self.down2(x4))  # 1/8

        y4  = self.dec1(self.se4(self.up1(x8) + x4))
        y2  = self.dec2(self.se2(self.up2(y4) + x_s))
        y2  = F.relu(self.refine(y2))    # refine
        logits = self.head(y2)           # 1/1

        return logits                    # logits → BCEWithLogitsLoss / sigmoid outside

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

__all__ = ['submission_20225757']
