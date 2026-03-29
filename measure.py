import torch
import numpy as np
import torch.nn.functional as F

def compute_measure(x, y, pred, dr):
    def mse(a, b): return ((a - b) ** 2).mean()
    def rmse(a, b): return np.sqrt(mse(a, b))
    def psnr(a, b, d): return 10 * np.log10((d ** 2) / mse(a, b))

    def ssim(img1, img2, d):
        t1, t2 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float(), torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()
        def gauss(ws, sig):
            g = torch.exp(-torch.arange(ws).float()**2/(2*sig**2))
            return (g / g.sum()).unsqueeze(1).mm((g / g.sum()).unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        window = gauss(11, 1.5).type_as(t1)
        mu1, mu2 = F.conv2d(t1, window, padding=5), F.conv2d(t2, window, padding=5)
        sigma1_sq = F.conv2d(t1*t1, window, padding=5) - mu1**2
        sigma2_sq = F.conv2d(t2*t2, window, padding=5) - mu2**2
        sigma12 = F.conv2d(t1*t2, window, padding=5) - mu1*mu2
        c1, c2 = (0.01*d)**2, (0.03*d)**2
        ssim_map = ((2*mu1*mu2+c1)*(2*sigma12+c2))/((mu1**2+mu2**2+c1)*(sigma1_sq+sigma2_sq+c2))
        return ssim_map.mean().item()

    return (psnr(x, y, dr), ssim(x, y, dr), rmse(x, y)), (psnr(pred, y, dr), ssim(pred, y, dr), rmse(pred, y))