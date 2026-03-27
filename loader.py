import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader


class ct_dataset(Dataset):
    def __init__(self, mode, saved_path, test_patient, patch_n=None, patch_size=None):
        self.mode = mode
        input_path = sorted(glob(os.path.join(saved_path, "*_input.npy")))
        target_path = sorted(glob(os.path.join(saved_path, "*_target.npy")))

        if mode == "train":
            self.input_ = [f for f in input_path if test_patient not in f]
            self.target_ = [f for f in target_path if test_patient not in f]
        else:
            self.input_ = [f for f in input_path if test_patient in f]
            self.target_ = [f for f in target_path if test_patient in f]

        self.patch_n = patch_n
        self.patch_size = patch_size

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        in_img = np.load(self.input_[idx]).astype(np.float32)
        tg_img = np.load(self.target_[idx]).astype(np.float32)

        if self.mode == "train" and self.patch_size is not None:
            h, w = in_img.shape
            pi, pt = [], []

            for _ in range(self.patch_n):
                y = np.random.randint(0, h - self.patch_size)
                x = np.random.randint(0, w - self.patch_size)

                p_in = in_img[y:y + self.patch_size, x:x + self.patch_size]
                p_tg = tg_img[y:y + self.patch_size, x:x + self.patch_size]

                if np.random.random() > 0.5:
                    p_in = np.flip(p_in, axis=1)
                    p_tg = np.flip(p_tg, axis=1)

                if np.random.random() > 0.5:
                    p_in = np.flip(p_in, axis=0)
                    p_tg = np.flip(p_tg, axis=0)

                k = np.random.randint(0, 4)
                p_in = np.rot90(p_in, k)
                p_tg = np.rot90(p_tg, k)

                pi.append(p_in.copy())
                pt.append(p_tg.copy())

            return np.array(pi, dtype=np.float32), np.array(pt, dtype=np.float32)

        return in_img, tg_img


def get_loader(
    mode,
    saved_path,
    test_patient,
    patch_n=10,
    patch_size=128,
    batch_size=2,
    num_workers=4,
):
    ds = ct_dataset(mode, saved_path, test_patient, patch_n, patch_size)
    return DataLoader(
        ds,
        batch_size=batch_size if mode == "train" else 1,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )