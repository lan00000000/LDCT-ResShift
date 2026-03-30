import torch
import torch.nn as nn
from models.unet import UNetModel

class FrequencyAwareAttention(nn.Module):
    """借鉴论文图2的频率感知注意力机制 [cite: 189]"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接最大池化和平均池化 [cite: 189]
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

class FAMEncoder(nn.Module):
    """频率感知多尺度特征编码器 [cite: 185, 191]"""
    def __init__(self, in_ch=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            FrequencyAwareAttention(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

class ResShift_CT(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, model_channels=64, image_size=64):
        super(ResShift_CT, self).__init__()
        self.model = UNetModel(
            image_size=image_size,
            in_channels=in_ch + in_ch,
            out_channels=out_ch,
            model_channels=model_channels,
            num_res_blocks=2,
            attention_resolutions=(16,),
            use_scale_shift_norm=True,
            cond_lq=True 
        )

    def forward(self, x_t, timesteps, lq=None):
        return self.model(x_t, timesteps, lq=lq)