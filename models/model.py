import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RobustPDABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
    
    def forward(self, x):
        phase = torch.atan2(x[:, 1, :], x[:, 0, :])
        dphi = phase[:, 1:] - phase[:, :-1]
        dphi = torch.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
        dphi = torch.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
        dphi = F.pad(dphi, (1, 0), "constant", 0).unsqueeze(1)
        return self.smooth(dphi)

class MultiScaleFeatureBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    
    def forward(self, x):
        return self.net(x)

class LKA1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_local = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_dilated = nn.Conv1d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv_fuse = nn.Conv1d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        local_feat = self.conv_local(x)
        dilated_feat = self.conv_dilated(local_feat)
        fused_feat = local_feat + dilated_feat
        attn = self.conv_fuse(fused_feat)
        return x * torch.sigmoid(attn)

class DCS_Net(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.iq_branch = MultiScaleFeatureBlock(in_ch=2, out_ch=64)
        self.mag_branch = MultiScaleFeatureBlock(in_ch=1, out_ch=64)
        self.pda_engine = RobustPDABlock()
        self.pda_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2)
        )
        
        self.feature_mixer = nn.Sequential(
            nn.Conv1d(160, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.Hardswish()
        )
        
        self.feature_gate = LKA1D(dim=128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        mag = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 1e-8)
        
        f_iq = self.iq_branch(x)
        f_mag = self.mag_branch(mag)
        f_pda = self.pda_branch(self.pda_engine(x))
        
        raw_combined = torch.cat([f_iq, f_mag, f_pda], dim=1)
        mixed = self.feature_mixer(raw_combined)
        
        gated = self.feature_gate(mixed)
        
        pooled = self.pool(gated).squeeze(-1)
        return self.classifier(pooled)