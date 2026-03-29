import argparse
import os
import shutil
import random
import numpy as np
import torch
from loader import get_loader
from solver import Solver

def set_seed(seed):
    """固定随机种子以确保实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--save_path', type=str, default='./save_results/')
    parser.add_argument('--npy_path', type=str, default='./npy_img/')
    parser.add_argument('--ckpt', type=str, default='./save_results/models/model_best.ckpt')
    
    # 冲刺关键：训练与测试步数必须统一为 15 步
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--seed', type=int, default=42)
    # 基础感知权重，用于前 15 轮引导
    parser.add_argument('--w_perceptual', type=float, default=1.0)

    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.exists(args.ckpt):
            print(f"--> [FRESH START] No checkpoint found at {args.ckpt}. Starting from scratch.")
            set_seed(args.seed)
            if os.path.exists(args.save_path):
                shutil.rmtree(args.save_path)
            os.makedirs(args.save_path, exist_ok=True)
            os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
        else:
            print(f"--> [RESUME] Found checkpoint at {args.ckpt}. Resuming training...")

    loader = get_loader(args.mode, args.npy_path, args.test_patient, 10, args.patch_size, args.batch_size)
    
    solver = Solver(args, loader)
    if args.mode == 'train':
        solver.train()
    else:
        solver.test()