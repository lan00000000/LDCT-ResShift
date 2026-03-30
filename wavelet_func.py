import torch

def dwt_init(x):
    """
    修正后的 Haar 离散小波变换 (DWT)
    确保基底的正交性，避免重构后出现灰色网格
    """
    x00 = x[:, :, 0::2, 0::2] # (0,0)
    x10 = x[:, :, 1::2, 0::2] # (1,0) 垂直
    x01 = x[:, :, 0::2, 1::2] # (0,1) 水平
    x11 = x[:, :, 1::2, 1::2] # (1,1) 对角
    
    # 采用标准 Haar 基底系数
    x_LL = (x00 + x10 + x01 + x11) / 2.0
    x_LH = (x00 + x01 - x10 - x11) / 2.0 # 水平细节
    x_HL = (x00 + x10 - x01 - x11) / 2.0 # 垂直细节
    x_HH = (x00 - x01 - x10 + x11) / 2.0 # 对角细节 (修正符号)
    
    return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)

def dwt_separate(x):
    w = dwt_init(x)
    ll = w[:, 0:1, :, :]
    hf = w[:, 1:4, :, :] 
    return ll, hf

def idwt_init(x):
    """
    修正后的逆离散小波变换 (IDWT)
    实现完美重构 (Perfect Reconstruction)
    """
    x_LL = x[:, 0:1, :, :]
    x_LH = x[:, 1:2, :, :] 
    x_HL = x[:, 2:3, :, :]
    x_HH = x[:, 3:4, :, :]
    
    N, C, H, W = x_LL.shape
    h = torch.zeros([N, 1, H * 2, W * 2], device=x.device)
    
    # 逆变换公式
    h[:, :, 0::2, 0::2] = (x_LL + x_LH + x_HL + x_HH) / 2.0
    h[:, :, 1::2, 0::2] = (x_LL - x_LH + x_HL - x_HH) / 2.0
    h[:, :, 0::2, 1::2] = (x_LL + x_LH - x_HL - x_HH) / 2.0
    h[:, :, 1::2, 1::2] = (x_LL - x_LH - x_HL + x_HH) / 2.0
    return h