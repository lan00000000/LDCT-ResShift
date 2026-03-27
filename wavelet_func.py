import torch

# ===============================
# Multi-scale wavelet band weights
# ===============================
WAVELET_BAND_WEIGHT = {
    "LL2": 0.5,
    "HF2": 1.0,
    "HF1": 2.0
}

# ===============================
# Haar DWT (orthogonal, PR)
# ===============================
def dwt_init(x):
    x00 = x[:, :, 0::2, 0::2]
    x10 = x[:, :, 1::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x10 + x01 + x11) / 2.0
    lh = (x00 + x01 - x10 - x11) / 2.0
    hl = (x00 + x10 - x01 - x11) / 2.0
    hh = (x00 - x01 - x10 + x11) / 2.0

    return torch.cat([ll, lh, hl, hh], dim=1)


def idwt_init(x):
    ll, lh, hl, hh = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

    B, _, H, W = ll.shape
    out = torch.zeros((B, 1, H * 2, W * 2), device=x.device, dtype=x.dtype)

    out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    out[:, :, 1::2, 0::2] = (ll - lh + hl - hh) / 2.0
    out[:, :, 0::2, 1::2] = (ll + lh - hl - hh) / 2.0
    out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) / 2.0

    return out

# ===============================
# Two-level wavelet interface
# ===============================
def apply_wavelet_2level(x):
    w1 = dwt_init(x)
    ll1, hf1 = w1[:, 0:1], w1[:, 1:4]

    w2 = dwt_init(ll1)
    ll2, hf2 = w2[:, 0:1], w2[:, 1:4]

    return {
        "LL2": ll2,
        "HF2": hf2,
        "HF1": hf1
    }


def idwt_2level(coeffs):
    ll2, hf2, hf1 = coeffs["LL2"], coeffs["HF2"], coeffs["HF1"]

    assert ll2.shape[1] == 1
    assert hf2.shape[1] == 3
    assert hf1.shape[1] == 3

    ll1 = idwt_init(torch.cat([ll2, hf2], dim=1))
    x = idwt_init(torch.cat([ll1, hf1], dim=1))
    return x
