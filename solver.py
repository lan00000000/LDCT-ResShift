import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy 
from networks import ResShift_CT, FAMEncoder
from measure import compute_measure
from models.script_util import create_gaussian_diffusion
from torch.optim.lr_scheduler import CosineAnnealingLR
from wavelet_func import dwt_init, idwt_init, dwt_separate 
from tqdm import tqdm

# --- 可微的 SSIM 损失函数 ---
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.exp(-(torch.arange(window_size).float() - window_size//2)**2 / (2 * sigma**2))
            return gauss / gauss.sum()
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda: window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean() if self.size_average else 1 - ssim_map

class Solver:
    def __init__(self, args, data_loader):
        self.args = args
        self.data_loader = data_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ResShift_CT(
            in_ch=4, out_ch=4,
            model_channels=args.model_channels, 
            image_size=args.patch_size // 2 
        ).to(self.device)
        
        self.online_encoder = FAMEncoder(in_ch=3).to(self.device)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.ema_decay = 0.99 
        self.lambda_fam = self.args.w_perceptual 
        self.l1_loss = nn.L1Loss().to(self.device)
        self.ssim_loss_fn = SSIMLoss().to(self.device)
        
        self.model_dir = os.path.join(self.args.save_path, 'models')
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

        self.start_epoch = 1
        self.best_loss = float('inf')
        
        if args.ckpt and os.path.exists(args.ckpt):
            print(f"--> [RESUME] Loading checkpoint: {args.ckpt}")
            ckpt_data = torch.load(args.ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt_data['model'] if 'model' in ckpt_data else ckpt_data)
            if 'fam' in ckpt_data:
                self.online_encoder.load_state_dict(ckpt_data['fam'])
                self.target_encoder.load_state_dict(ckpt_data['fam'])
            
            if 'epoch' in ckpt_data:
                self.start_epoch = ckpt_data['epoch'] + 1
            else:
                try:
                    base_name = os.path.basename(args.ckpt)
                    if 'model_' in base_name and 'best' not in base_name:
                        self.start_epoch = int(base_name.split('_')[1].split('.')[0]) + 1
                except: pass
            
            if 'best_loss' in ckpt_data:
                self.best_loss = ckpt_data['best_loss']

        # --- 重要：如果是从头训练，重置 best_loss 确保新权重公式能正常保存最优模型 ---
        if self.start_epoch == 1:
            self.best_loss = float('inf')

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.online_encoder.parameters()), 
            lr=args.lr
        )
        
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', args.lr)
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.num_epochs, 
                                           eta_min=1e-7, last_epoch=self.start_epoch - 2)
        
        self.diffusion = create_gaussian_diffusion(
            normalize_input=False, schedule_name='exponential', steps=args.steps,
            kappa=args.kappa, predict_type='xstart', latent_flag=False, sf=1, schedule_kwargs={'power': 1.0}
        )

    def train(self):
        self.model.train()
        self.online_encoder.train()
        self.target_encoder.eval() 
        
        mode = 'a' if self.start_epoch > 1 else 'w'
        log_file = open(os.path.join(self.args.save_path, 'loss_log.txt'), mode)
        if self.start_epoch == 1:
            log_file.write("Epoch\tL1_Weighted\tSSIM_Weighted\tFAM\tTotal\n")
            
        print(f"Starting training from Epoch {self.start_epoch}...")
        
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            # --- 核心：15 轮熔断逻辑 ---
            current_lambda = 0.001 if epoch <= 15 else 0.0
            
            t_pixel, t_ssim, t_fam = 0, 0, 0
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch}")
            
            for y_0, x_0 in pbar:
                x_0 = x_0.view(-1, 1, self.args.patch_size, self.args.patch_size).float().to(self.device)
                y_0 = y_0.view(-1, 1, self.args.patch_size, self.args.patch_size).float().to(self.device)
                
                x0_w, y0_w = dwt_init(x_0), dwt_init(y_0)
                t = torch.randint(0, self.args.steps, (x0_w.shape[0],)).to(self.device)
                x_t_w = self.diffusion.q_sample(x_start=x0_w, y=y0_w, t=t)
                pred_w = self.model(x_t_w, t, lq=y0_w)
                
                pred_spatial = idwt_init(pred_w)
                
                # --- 多尺度损失平衡策略 ---
                loss_pixel = 0.5 * self.l1_loss(pred_spatial, x_0) + 0.5 * F.mse_loss(pred_spatial, x_0)
                loss_ssim = self.ssim_loss_fn(pred_spatial, x_0) * 5.0
                
                _, pred_hf = dwt_separate(pred_spatial); _, target_hf = dwt_separate(x_0)
                feat_pred = self.online_encoder(pred_hf * 50.0)
                with torch.no_grad(): feat_target = self.target_encoder(target_hf * 50.0)
                loss_fam = F.mse_loss(feat_pred, feat_target)
                
                total_loss = loss_pixel + loss_ssim + current_lambda * loss_fam
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # --- 物理防爆：梯度裁剪，防止第 6 轮那样的突发跳变 ---
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.online_encoder.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                with torch.no_grad():
                    for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                        p_t.data.mul_(self.ema_decay).add_(p_o.data, alpha=1 - self.ema_decay)
                
                # --- 变量名统一修复 ---
                t_pixel += loss_pixel.item()
                t_ssim += loss_ssim.item()
                t_fam += loss_fam.item()
                pbar.set_postfix({"Loss": f"{total_loss.item():.5f}"})

            self.scheduler.step()
            
            # --- 日志计算修复 ---
            avg_l1_w = t_pixel / len(self.data_loader)
            avg_ssim_w = t_ssim / len(self.data_loader)
            avg_fam_u = t_fam / len(self.data_loader)
            avg_total = avg_l1_w + avg_ssim_w + (current_lambda * avg_fam_u)
            
            log_file.write(f"{epoch}\t{avg_l1_w:.6f}\t{avg_ssim_w:.6f}\t{avg_fam_u:.6f}\t{avg_total:.6f}\n")
            log_file.flush()
            
            save_dict = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'fam': self.online_encoder.state_dict(),
                'best_loss': self.best_loss
            }

            if avg_total < self.best_loss:
                self.best_loss = avg_total
                save_dict['best_loss'] = self.best_loss
                torch.save(save_dict, os.path.join(self.model_dir, 'model_best.ckpt'))
                print(f"--> Saved Best Model at Epoch {epoch} (Total Loss: {avg_total:.6f})")
            
            if epoch % 10 == 0:
                torch.save(save_dict, os.path.join(self.model_dir, f"model_{epoch}.ckpt"))
                
        log_file.close()

    def test(self):
        self.model.eval()
        img_dir = os.path.join(self.args.save_path, 'test_images')
        if not os.path.exists(img_dir): os.makedirs(img_dir)
        
        all_psnrs, all_ssims = [], []
        patch_size = 128 # 必须与训练时的 patch_size 一致
        stride = 128     # 步长，设为 128 为无重叠拼接

        print("Testing Supervised Wavelet-ResShift with Patch-based Inference...")
        with torch.no_grad():
            for i, (y_0, x_0) in enumerate(self.data_loader):
                # 1. 预处理原图 (512, 512)
                y_0_spatial = y_0.unsqueeze(1).float().to(self.device)
                B, C, H, W = y_0_spatial.shape
                pred_full = torch.zeros_like(y_0_spatial)

                # 2. 分块滑动推理
                # 将 512x512 分成 16 个 128x128 的块
                for h in range(0, H, stride):
                    for w in range(0, W, stride):
                        # 裁剪 Patch
                        y_patch = y_0_spatial[:, :, h:h+patch_size, w:w+patch_size]
                        
                        # 执行小波变换并进入扩散模型
                        y0_w_patch = dwt_init(y_patch)
                        output_dict = self.diffusion.p_sample_loop(
                            y=y0_w_patch, 
                            model=self.model, 
                            device=self.device, 
                            model_kwargs={'lq': y0_w_patch}
                        )
                        sample_patch = idwt_init(output_dict['sample'])
                        
                        # 将结果填回大图
                        pred_full[:, :, h:h+patch_size, w:w+patch_size] = sample_patch

                # 3. 后处理与指标计算
                def post(img_tensor):
                    img = img_tensor.squeeze().cpu().numpy() * (self.args.norm_range_max - self.args.norm_range_min) + self.args.norm_range_min
                    return np.clip(img, self.args.trunc_min, self.args.trunc_max)

                pred_img = post(pred_full)
                gt_img = post(x_0)
                noisy_img = post(y_0)

                # --- 核心修正：全局均值对齐 (针对 alpha 亮度抖动的临时补救) ---
                # 如果你的训练由于 alpha 缩放导致了亮度偏移，这行代码能瞬间提升 PSNR
                pred_img = pred_img - (np.mean(pred_img) - np.mean(noisy_img))
                pred_img = np.clip(pred_img, self.args.trunc_min, self.args.trunc_max)

                _, (p, s, r) = compute_measure(noisy_img, gt_img, pred_img, self.args.trunc_max - self.args.trunc_min)
                
                all_psnrs.append(p); all_ssims.append(s)
                print(f"Slice {i:03d} | PSNR: {p:.4f} | SSIM: {s:.4f}")
                
                # 保存可视化
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1); plt.imshow(noisy_img, cmap='gray'); plt.title('Noisy'); plt.axis('off')
                plt.subplot(1, 3, 2); plt.imshow(pred_img, cmap='gray'); plt.title(f'Denosed (PSNR:{p:.2f})'); plt.axis('off')
                plt.subplot(1, 3, 3); plt.imshow(gt_img, cmap='gray'); plt.title('Ground Truth'); plt.axis('off')
                plt.savefig(os.path.join(img_dir, f'res_{i:03d}.png'), bbox_inches='tight'); plt.close()

        print("\n" + "="*30)
        print(f"Final Test Result ({len(all_psnrs)} slices):")
        print(f"Average PSNR: {np.mean(all_psnrs):.4f}")
        print(f"Average SSIM: {np.mean(all_ssims):.4f}")
        print("="*30 + "\n")