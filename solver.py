import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

from networks import ResShift_CT, FAMEncoder
from measure import compute_measure
from models.script_util import create_gaussian_diffusion
from models.gaussian_diffusion import MultiScaleGaussianDiffusion
from torch.optim.lr_scheduler import CosineAnnealingLR

from wavelet_func import (
    apply_wavelet_2level,
    idwt_2level,
)

# ===============================
# SSIM loss
# ===============================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size//2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size//2)
        sigma_x = F.avg_pool2d(x*x, self.window_size, 1, self.window_size//2) - mu_x**2
        sigma_y = F.avg_pool2d(y*y, self.window_size, 1, self.window_size//2) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, self.window_size, 1, self.window_size//2) - mu_x*mu_y
        ssim = ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x+sigma_y+C2))
        return 1 - ssim.mean()


# ===============================
# Solver
# ===============================
class Solver:
    def __init__(self, args, data_loader):
        self.args = args
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- model (HF prediction) ----
        self.model = ResShift_CT(
            in_ch=6, out_ch=3,
            model_channels=args.model_channels,
            image_size=args.patch_size // 2
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.num_epochs, eta_min=1e-7)

        self.ssim_loss = SSIMLoss().to(self.device)

        # ---- multi-scale diffusion ----
        self.diffusion = MultiScaleGaussianDiffusion({
            "HF1": create_gaussian_diffusion(
                normalize_input=False,
                schedule_name="exponential",
                schedule_kwargs={"power": 1.0},
                steps=args.steps,
                kappa=args.kappa,
                predict_type="xstart",
                sf=1,
                latent_flag=False,
            ),
            "HF2": create_gaussian_diffusion(
                normalize_input=False,
                schedule_name="exponential",
                schedule_kwargs={"power": 1.0},
                steps=args.steps,
                kappa=args.kappa,
                predict_type="xstart",
                sf=1,
                latent_flag=False,
            ),
        })

        self.model_dir = os.path.join(args.save_path, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    # ===============================
    # TRAIN
    # ===============================
    def train(self):
        self.model.train()
        log_path = os.path.join(self.args.save_path, "loss_log.txt")
        log_file = open(log_path, "w")
        log_file.write("Epoch\tHF1\tHF2\tSSIM\tTotal\n")

        for epoch in range(1, self.args.num_epochs + 1):
            t_hf1 = t_hf2 = t_ssim = 0.0
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch}")

            for y_0, x_0 in pbar:
                x_0 = x_0.view(-1, 1, self.args.patch_size, self.args.patch_size).to(self.device)
                y_0 = y_0.view(-1, 1, self.args.patch_size, self.args.patch_size).to(self.device)

                # ---- 2-level wavelet ----
                xw = apply_wavelet_2level(x_0)
                yw = apply_wavelet_2level(y_0)

                x_hf1, x_hf2 = xw["HF1"], xw["HF2"]
                y_hf1, y_hf2 = yw["HF1"], yw["HF2"]

                t = torch.randint(0, self.args.steps, (x_0.size(0),), device=self.device)

                # ---- forward diffusion ----
                x_hf1_t = self.diffusion.diffusions["HF1"].q_sample(x_hf1, y_hf1, t)
                x_hf2_t = self.diffusion.diffusions["HF2"].q_sample(x_hf2, y_hf2, t)

                # ---- predict ----
                pred_hf1 = self.model(x_hf1_t, t, lq=y_hf1)
                pred_hf2 = self.model(x_hf2_t, t, lq=y_hf2)

                # ---- losses ----
                loss_hf1 = F.l1_loss(pred_hf1, x_hf1)
                loss_hf2 = F.l1_loss(pred_hf2, x_hf2)

                recon = idwt_2level({
                    "LL2": xw["LL2"],
                    "HF2": pred_hf2,
                    "HF1": pred_hf1
                })

                loss_ssim = 0.0 if epoch < 10 else self.ssim_loss(recon, x_0)
                total_loss = loss_hf1 + loss_hf2 + loss_ssim

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                t_hf1 += loss_hf1.item()
                t_hf2 += loss_hf2.item()
                t_ssim += loss_ssim if isinstance(loss_ssim, float) else loss_ssim.item()
                pbar.set_postfix(loss=f"{total_loss.item():.4f}")

            self.scheduler.step()

            n = len(self.data_loader)
            log_file.write(
                f"{epoch}\t{t_hf1/n:.6f}\t{t_hf2/n:.6f}\t{t_ssim/n:.6f}\t{(t_hf1+t_hf2+t_ssim)/n:.6f}\n"
            )
            log_file.flush()

        log_file.close()

    # ===============================
    # TEST  （你要求的部分：未改）
    # ===============================
    def test(self):
        self.model.eval()
        img_dir = os.path.join(self.args.save_path, 'test_images')
        os.makedirs(img_dir, exist_ok=True)

        all_psnrs, all_ssims = [], []
        patch_size = self.args.patch_size
        stride = patch_size

        print("Testing TRUE Multi-Scale Wavelet Diffusion (LL2 + HF2 + HF1)...")

        with torch.no_grad():
            for i, (y_0, x_0) in enumerate(self.data_loader):
                y_0 = y_0.unsqueeze(1).float().to(self.device)
                x_0 = x_0.unsqueeze(1).float().to(self.device)

                B, _, H, W = y_0.shape
                pred_full = torch.zeros_like(y_0)

                for h in range(0, H, stride):
                    for w in range(0, W, stride):
                        y_patch = y_0[:, :, h:h+patch_size, w:w+patch_size]

                        yw = apply_wavelet_2level(y_patch)

                        # HF2
                        pred_hf2 = self.diffusion.diffusions["HF2"].p_sample_loop(
                            model=self.model,
                            shape=yw["HF2"].shape,
                            y=yw["HF2"],
                            device=self.device,
                            model_kwargs={"lq": yw["HF2"]}
                        )

                        # HF1
                        pred_hf1 = self.diffusion.diffusions["HF1"].p_sample_loop(
                            model=self.model,
                            shape=yw["HF1"].shape,
                            y=yw["HF1"],
                            device=self.device,
                            model_kwargs={"lq": yw["HF1"]}
                        )

                        sample_patch = idwt_2level({
                            "LL2": yw["LL2"],
                            "HF2": pred_hf2,
                            "HF1": pred_hf1
                        })

                        pred_full[:, :, h:h+patch_size, w:w+patch_size] = sample_patch

                def post(x):
                    x = x.squeeze().cpu().numpy()
                    x = x * (self.args.norm_range_max - self.args.norm_range_min) + self.args.norm_range_min
                    return np.clip(x, self.args.trunc_min, self.args.trunc_max)

                pred_img = post(pred_full)
                gt_img = post(x_0)
                noisy_img = post(y_0)

                pred_img -= (pred_img.mean() - noisy_img.mean())
                pred_img = np.clip(pred_img, self.args.trunc_min, self.args.trunc_max)

                _, (p, s, _) = compute_measure(
                    noisy_img, gt_img, pred_img,
                    self.args.trunc_max - self.args.trunc_min
                )

                all_psnrs.append(p)
                all_ssims.append(s)

                print(f"Slice {i:03d} | PSNR: {p:.4f} | SSIM: {s:.4f}")

                plt.figure(figsize=(15,5))
                plt.subplot(1,3,1); plt.imshow(noisy_img, cmap="gray"); plt.axis("off")
                plt.subplot(1,3,2); plt.imshow(pred_img, cmap="gray"); plt.axis("off")
                plt.subplot(1,3,3); plt.imshow(gt_img, cmap="gray"); plt.axis("off")
                plt.savefig(os.path.join(img_dir, f"res_{i:03d}.png"))
                plt.close()

        print("="*40)
        print(f"Average PSNR: {np.mean(all_psnrs):.4f}")
        print(f"Average SSIM: {np.mean(all_ssims):.4f}")
        print("="*40)
