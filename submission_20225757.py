import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic blocks
class Down(nn.Module):
    def __init__(self,n_in,n_out):
        super().__init__()
        self.conv=nn.Conv2d(n_in,n_out-n_in,3,2,1,bias=False)
        self.pool=nn.MaxPool2d(2,2)
        self.bn=nn.BatchNorm2d(n_out); self.act=nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.bn(torch.cat([self.conv(x),self.pool(x)],1)))

class NB1D(nn.Module):
    def __init__(self,ch,d=1):
        super().__init__()
        self.c31=nn.Conv2d(ch,ch,(3,1),1,(1,0),bias=False)
        self.c13=nn.Conv2d(ch,ch,(1,3),1,(0,1),bias=False)
        self.bn1=nn.BatchNorm2d(ch)
        self.c31d=nn.Conv2d(ch,ch,(3,1),1,(d,0),dilation=(d,1),bias=False)
        self.c13d=nn.Conv2d(ch,ch,(1,3),1,(0,d),dilation=(1,d),bias=False)
        self.bn2=nn.BatchNorm2d(ch); self.act=nn.ReLU(inplace=True)
    def forward(self,x):
        y=self.act(self.bn1(self.c13(self.c31(x))))
        y=self.bn2(self.c13d(self.c31d(y)))
        return self.act(y+x)

class Up(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.de=nn.ConvTranspose2d(cin,cout,3,2,1,output_padding=1,bias=False)
        self.bn=nn.BatchNorm2d(cout); self.act=nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.bn(self.de(x)))

# CBAM
class CBAM(nn.Module):
    def __init__(self,ch,r=8):
        super().__init__()
        self.mlp=nn.Sequential(nn.Conv2d(ch,ch//r,1),nn.ReLU(),
                               nn.Conv2d(ch//r,ch,1))
        self.spatial=nn.Conv2d(2,1,7,1,3)
    def forward(self,x):
        # Channel
        maxp=x.amax((2,3),keepdim=True); avgp=x.mean((2,3),keepdim=True)
        c=torch.sigmoid(self.mlp(maxp)+self.mlp(avgp))
        x=x*c
        # Spatial
        s=torch.cat([x.max(1,keepdim=True)[0], x.mean(1,keepdim=True)],1)
        s=torch.sigmoid(self.spatial(s))
        return x*s

# Mini-ASPP (4 枝)
class MiniASPP(nn.Module):
    def __init__(self,in_ch,br=32):  # 32×4=128
        super().__init__()
        self.br=nn.ModuleList([
            nn.Conv2d(in_ch,br,1,bias=False),
            nn.Conv2d(in_ch,br,3,padding=3,dilation=3,bias=False),
            nn.Conv2d(in_ch,br,3,padding=6,dilation=6,bias=False),
            nn.Conv2d(in_ch,br,3,padding=9,dilation=9,bias=False)])
        self.bn=nn.BatchNorm2d(br*4); self.act=nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.bn(torch.cat([b(x) for b in self.br],1)))

# Network
class submission_20225757(nn.Module):
    def __init__(self,in_channels=3,num_classes=1):
        super().__init__()
        # Encoder (24-96-192)
        self.stem=Down(in_channels,24)
        self.d1  =Down(24,96);  self.e1=nn.Sequential(*[NB1D(96) for _ in range(4)])
        self.d2  =Down(96,192); self.e2=nn.Sequential(
            NB1D(192,2),NB1D(192,4),NB1D(192,8),NB1D(192,16))

        # ASPP
        self.aspp=MiniASPP(192,32)   # 128ch
        self.reduce=nn.Conv2d(128,96,1,bias=False)

        # Decoder
        self.up1=Up(96,96);  self.cbam4=CBAM(96)
        self.dec1=NB1D(96)
        self.up2=Up(96,24);  self.cbam2=CBAM(24)
        self.dec2=NB1D(24)
        self.refine=nn.Conv2d(24,24,1,bias=False)
        self.head=nn.ConvTranspose2d(24,num_classes,2,2)

        self.branch4=nn.Conv2d(96,num_classes,1)

        self.alpha=nn.Parameter(torch.tensor(-0.85))  # mix≈0.3
        self.tau  =nn.Parameter(torch.tensor(1.0))

        self._init()

    def forward(self,x):
        x2=self.stem(x)
        x4=self.e1(self.d1(x2))
        x8=self.e2(self.d2(x4))
        x8=self.reduce(self.aspp(x8))

        # DropBlock 10 %
        if self.training:
            mask=(torch.rand_like(x8[:, :1])<0.1).float()
            x8=x8*(1-mask)*1/0.9

        y4=self.cbam4(self.up1(x8)+x4)
        b4=torch.sigmoid(self.branch4(y4))

        y4=self.dec1(y4)
        y2=self.cbam2(self.up2(y4)+x2)
        y2=self.dec2(y2)
        y2=F.relu(self.refine(y2))
        main_p=torch.sigmoid(self.head(y2))

        H,W=main_p.shape[-2:]
        b4u=F.interpolate(b4,(H,W),mode='bilinear',align_corners=False)
        mix=torch.sigmoid(self.alpha)
        p=mix*main_p+(1-mix)*b4u

        tau=F.softplus(self.tau)+1e-3
        p=p.clamp(1e-6,1-1e-6)**(1.0/tau)

        return torch.log(p/(1-p))

    def _init(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

__all__=['submission_20225757']
