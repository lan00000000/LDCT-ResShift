import argparse
import os
import shutil
import random
import numpy as np
import torch

from loader import get_loader
from solver import Solver


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    parser = argparse.ArgumentParser(
        description="LDCT-ResShift: Single-level Wavelet Residual Diffusion for LDCT denoising"
    )

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--save_path', type=str, default='./save_results/')
    parser.add_argument('--npy_path', type=str, default='./npy_img/')
    parser.add_argument('--ckpt', type=str, default='./save_results/models/model_best.ckpt')
    parser.add_argument('--test_patient', type=str, default='L506')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--kappa', type=float, default=2.5)
    parser.add_argument('--schedule_name', type=str, default='exponential')
    parser.add_argument('--schedule_power', type=float, default=1.5)
    parser.add_argument('--min_noise_level', type=float, default=0.01)
    parser.add_argument('--etas_end', type=float, default=0.99)

    parser.add_argument('--model_channels', type=int, default=64)

    # losses
    parser.add_argument('--lambda_hf1', type=float, default=1.0)
    parser.add_argument('--lambda_hf2', type=float, default=1.0)
    parser.add_argument('--lambda_ssim', type=float, default=1.0)
    parser.add_argument('--ssim_start_epoch', type=int, default=10)

    parser.add_argument('--observe_every', type=int, default=10)
    parser.add_argument('--observe_slice_index', type=int, default=85)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    return parser


def prepare_train_folders(args):
    ckpt_exists = os.path.isfile(args.ckpt)

    if ckpt_exists:
        print(f"--> [RESUME] Found checkpoint at: {args.ckpt}")
        print("--> [RESUME] Will restore model / optimizer / scheduler state.")
        print("--> [RESUME] Keep existing save directory.")
        return

    print(f"--> [FRESH START] No checkpoint found at: {args.ckpt}")
    print("--> [FRESH START] Start training from scratch.")

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == 'train':
        prepare_train_folders(args)
    else:
        if args.ckpt and not os.path.isfile(args.ckpt):
            raise FileNotFoundError(
                f"[TEST ERROR] Checkpoint not found: {args.ckpt}\n"
                f"Please train first or pass a valid --ckpt path."
            )

        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)

    loader = get_loader(
        mode=args.mode,
        saved_path=args.npy_path,
        test_patient=args.test_patient,
        patch_n=args.patch_n,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    solver = Solver(args, loader)

    if args.mode == 'train':
        solver.train()
    else:
        solver.test()


if __name__ == "__main__":
    main()