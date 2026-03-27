import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from networks import ResShift_CT
from measure import compute_measure
from models.script_util import create_gaussian_diffusion
from models.gaussian_diffusion import MultiScaleGaussianDiffusion
from wavelet_func import apply_wavelet_2level, idwt_2level


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size // 2)
        sigma_x = F.avg_pool2d(x * x, self.window_size, 1, self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, self.window_size, 1, self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1, self.window_size // 2) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        )
        return 1 - ssim.mean()


class Solver:
    def __init__(self, args, data_loader):
        self.args = args
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResShift_CT(
            hf_ch=3,
            out_ch=3,
            model_channels=args.model_channels,
            image_size=args.patch_size // 2,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.num_epochs, eta_min=1e-7)
        self.ssim_loss = SSIMLoss().to(self.device)

        self.diffusion = MultiScaleGaussianDiffusion({
            "HF1": create_gaussian_diffusion(
                normalize_input=True,
                schedule_name=args.schedule_name,
                sf=1,
                min_noise_level=args.min_noise_level,
                steps=args.steps,
                kappa=args.kappa,
                etas_end=args.etas_end,
                schedule_kwargs={"power": args.schedule_power},
                weighted_mse=False,
                predict_type="xstart",
                timestep_respacing=args.steps,
                scale_factor=None,
                latent_flag=False,
            ),
            "HF2": create_gaussian_diffusion(
                normalize_input=True,
                schedule_name=args.schedule_name,
                sf=1,
                min_noise_level=args.min_noise_level,
                steps=args.steps,
                kappa=args.kappa,
                etas_end=args.etas_end,
                schedule_kwargs={"power": args.schedule_power},
                weighted_mse=False,
                predict_type="xstart",
                timestep_respacing=args.steps,
                scale_factor=None,
                latent_flag=False,
            ),
        })

        self.model_dir = os.path.join(args.save_path, "models")
        self.img_dir = os.path.join(args.save_path, "test_images")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        self.start_epoch = 1
        self.best_loss = float("inf")

        if args.ckpt and os.path.isfile(args.ckpt):
            ckpt = torch.load(args.ckpt, map_location=self.device)
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"], strict=True)
                if args.mode == "train":
                    if "optimizer" in ckpt:
                        self.optimizer.load_state_dict(ckpt["optimizer"])
                    if "scheduler" in ckpt:
                        self.scheduler.load_state_dict(ckpt["scheduler"])
                    self.start_epoch = ckpt.get("epoch", 0) + 1
                    self.best_loss = ckpt.get("best_loss", float("inf"))
            else:
                self.model.load_state_dict(ckpt, strict=True)

    def save_checkpoint(self, epoch, name="model_latest.ckpt", best_loss=None):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_loss": self.best_loss if best_loss is None else best_loss,
            },
            os.path.join(self.model_dir, name),
        )

    def train(self):
        log_path = os.path.join(self.args.save_path, "loss_log.txt")
        log_mode = "a" if self.start_epoch > 1 else "w"

        with open(log_path, log_mode) as log_file:
            if log_mode == "w":
                log_file.write("Epoch\tHF1\tHF2\tSSIM\tTotal\n")

            for epoch in range(self.start_epoch, self.args.num_epochs + 1):
                self.model.train()

                t_hf1 = 0.0
                t_hf2 = 0.0
                t_ssim = 0.0
                t_total = 0.0

                pbar = tqdm(self.data_loader, desc=f"Epoch {epoch:03d}")

                for y_0, x_0 in pbar:
                    x_0 = x_0.view(-1, 1, self.args.patch_size, self.args.patch_size).float().to(self.device)
                    y_0 = y_0.view(-1, 1, self.args.patch_size, self.args.patch_size).float().to(self.device)

                    xw = apply_wavelet_2level(x_0)
                    yw = apply_wavelet_2level(y_0)

                    x_hf1, x_hf2 = xw["HF1"], xw["HF2"]
                    y_hf1, y_hf2 = yw["HF1"], yw["HF2"]

                    t = torch.randint(0, self.args.steps, (x_0.size(0),), device=self.device).long()

                    x_hf1_t = self.diffusion.diffusions["HF1"].q_sample(x_hf1, y_hf1, t)
                    x_hf2_t = self.diffusion.diffusions["HF2"].q_sample(x_hf2, y_hf2, t)

                    pred_hf1 = self.model(x_hf1_t, t, lq=y_hf1)
                    pred_hf2 = self.model(x_hf2_t, t, lq=y_hf2)

                    loss_hf1 = F.l1_loss(pred_hf1, x_hf1)
                    loss_hf2 = F.l1_loss(pred_hf2, x_hf2)

                    recon = idwt_2level({
                        "LL2": xw["LL2"],
                        "HF2": pred_hf2,
                        "HF1": pred_hf1,
                    })

                    if epoch >= self.args.ssim_start_epoch:
                        loss_ssim = self.ssim_loss(recon, x_0)
                    else:
                        loss_ssim = torch.zeros(1, device=self.device, dtype=x_0.dtype).mean()

                    total_loss = (
                        self.args.lambda_hf1 * loss_hf1
                        + self.args.lambda_hf2 * loss_hf2
                        + self.args.lambda_ssim * loss_ssim
                    )

                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    t_hf1 += loss_hf1.item()
                    t_hf2 += loss_hf2.item()
                    t_ssim += loss_ssim.item()
                    t_total += total_loss.item()

                    pbar.set_postfix(
                        total=f"{total_loss.item():.4f}",
                        hf1=f"{loss_hf1.item():.4f}",
                        hf2=f"{loss_hf2.item():.4f}",
                    )

                self.scheduler.step()

                n = max(len(self.data_loader), 1)
                avg_hf1 = t_hf1 / n
                avg_hf2 = t_hf2 / n
                avg_ssim = t_ssim / n
                avg_total = t_total / n

                log_file.write(
                    f"{epoch}\t{avg_hf1:.6f}\t{avg_hf2:.6f}\t{avg_ssim:.6f}\t{avg_total:.6f}\n"
                )
                log_file.flush()

                self.save_checkpoint(epoch, name="model_latest.ckpt", best_loss=avg_total)

                if avg_total < self.best_loss:
                    self.best_loss = avg_total
                    self.save_checkpoint(epoch, name="model_best.ckpt", best_loss=avg_total)

    def test(self):
        self.model.eval()

        all_psnrs = []
        all_ssims = []

        patch_size = self.args.patch_size
        stride = patch_size

        print("Testing 2-level wavelet diffusion denoising...")

        with torch.no_grad():
            for i, (y_0, x_0) in enumerate(self.data_loader):
                y_0 = y_0.unsqueeze(1).float().to(self.device)
                x_0 = x_0.unsqueeze(1).float().to(self.device)

                _, _, h_total, w_total = y_0.shape
                pred_full = torch.zeros_like(y_0)

                for h in range(0, h_total, stride):
                    for w in range(0, w_total, stride):
                        y_patch = y_0[:, :, h:h + patch_size, w:w + patch_size]
                        ph, pw = y_patch.shape[-2:]

                        if ph != patch_size or pw != patch_size:
                            pad_h = patch_size - ph
                            pad_w = patch_size - pw
                            y_patch = F.pad(y_patch, (0, pad_w, 0, pad_h), mode="reflect")
                        else:
                            pad_h = 0
                            pad_w = 0

                        yw = apply_wavelet_2level(y_patch)
                        ll2 = yw["LL2"]

                        pred_hf2 = self.diffusion.diffusions["HF2"].p_sample_loop(
                            y=yw["HF2"],
                            model=self.model,
                            first_stage_model=None,
                            noise=None,
                            clip_denoised=False,
                            model_kwargs={"lq": yw["HF2"]},
                            device=self.device,
                            progress=False,
                        )

                        pred_hf1 = self.diffusion.diffusions["HF1"].p_sample_loop(
                            y=yw["HF1"],
                            model=self.model,
                            first_stage_model=None,
                            noise=None,
                            clip_denoised=False,
                            model_kwargs={"lq": yw["HF1"]},
                            device=self.device,
                            progress=False,
                        )

                        if isinstance(pred_hf2, dict):
                            pred_hf2 = pred_hf2.get("pred_xstart", pred_hf2.get("sample"))
                        if isinstance(pred_hf1, dict):
                            pred_hf1 = pred_hf1.get("pred_xstart", pred_hf1.get("sample"))

                        sample_patch = idwt_2level({
                            "LL2": ll2,
                            "HF2": pred_hf2,
                            "HF1": pred_hf1,
                        })

                        if pad_h > 0 or pad_w > 0:
                            sample_patch = sample_patch[:, :, :ph, :pw]

                        pred_full[:, :, h:h + ph, w:w + pw] = sample_patch

                def post(x):
                    x = x.squeeze().detach().cpu().numpy()
                    x = x * (self.args.norm_range_max - self.args.norm_range_min) + self.args.norm_range_min
                    return np.clip(x, self.args.trunc_min, self.args.trunc_max)

                pred_img = post(pred_full)
                gt_img = post(x_0)
                noisy_img = post(y_0)

                pred_img = np.clip(pred_img, self.args.trunc_min, self.args.trunc_max)

                _, (p, s, _) = compute_measure(
                    noisy_img,
                    gt_img,
                    pred_img,
                    self.args.trunc_max - self.args.trunc_min,
                )

                all_psnrs.append(p)
                all_ssims.append(s)

                print(f"Slice {i:03d} | PSNR: {p:.4f} | SSIM: {s:.4f}")

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(noisy_img, cmap="gray", vmin=self.args.trunc_min, vmax=self.args.trunc_max)
                plt.axis("off")
                plt.title("LDCT")

                plt.subplot(1, 3, 2)
                plt.imshow(pred_img, cmap="gray", vmin=self.args.trunc_min, vmax=self.args.trunc_max)
                plt.axis("off")
                plt.title("Prediction")

                plt.subplot(1, 3, 3)
                plt.imshow(gt_img, cmap="gray", vmin=self.args.trunc_min, vmax=self.args.trunc_max)
                plt.axis("off")
                plt.title("NDCT")

                plt.tight_layout()
                plt.savefig(os.path.join(self.img_dir, f"res_{i:03d}.png"))
                plt.close()

        print("=" * 40)
        print(f"Average PSNR: {np.mean(all_psnrs):.4f}")
        print(f"Average SSIM: {np.mean(all_ssims):.4f}")
        print("=" * 40)