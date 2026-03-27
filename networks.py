import torch
import torch.nn as nn
from models.unet import UNetModel


class ResShift_CT(nn.Module):
    """
    当前稳定版：
    - 输入 x_t: 3-channel high-frequency band
    - 条件 lq: 3-channel observed high-frequency band
    - 输出: 3-channel predicted clean high-frequency band
    """
    def __init__(self, hf_ch=3, out_ch=3, model_channels=128, image_size=64):
        super().__init__()

        # cond_lq=True 时，UNet 内部会使用 lq 条件，因此这里总输入通道设为 x_t + lq = 6
        self.model = UNetModel(
            image_size=image_size,
            in_channels=hf_ch * 2,
            out_channels=out_ch,
            model_channels=model_channels,
            num_res_blocks=2,
            attention_resolutions=(16,),
            use_scale_shift_norm=True,
            cond_lq=True,
        )

    def forward(self, x_t, timesteps, lq=None):
        if lq is None:
            raise ValueError("ResShift_CT.forward requires lq.")
        return self.model(x_t, timesteps, lq=lq)