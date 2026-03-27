import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as compare_ssim


def compute_MSE(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return np.mean((img1 - img2) ** 2)


def compute_RMSE(img1, img2):
    mse = compute_MSE(img1, img2)
    return sqrt(mse)


def compute_PSNR(img1, img2, data_range):
    mse = compute_MSE(img1, img2)
    if mse == 0:
        return 100.0
    return 10 * log10((data_range ** 2) / mse)


def compute_SSIM(img1, img2, data_range):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return compare_ssim(img1, img2, data_range=data_range)


def compute_measure(x, y, pred, data_range):
    """
    x: noisy input image
    y: ground-truth image
    pred: predicted image
    data_range: dynamic range, e.g. trunc_max - trunc_min
    """
    original_result = (
        compute_PSNR(x, y, data_range),
        compute_SSIM(x, y, data_range),
        compute_RMSE(x, y),
    )

    pred_result = (
        compute_PSNR(pred, y, data_range),
        compute_SSIM(pred, y, data_range),
        compute_RMSE(pred, y),
    )

    return original_result, pred_result