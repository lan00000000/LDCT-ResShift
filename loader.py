import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader

class ct_dataset(Dataset):
    def __init__(self, mode, saved_path, test_patient, patch_n=None, patch_size=None):
        self.mode = mode
        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        if mode == 'train':
            self.input_ = [f for f in input_path if test_patient not in f]
            self.target_ = [f for f in target_path if test_patient not in f]
        else:
            self.input_ = [f for f in input_path if test_patient in f]
            self.target_ = [f for f in target_path if test_patient in f]
        self.patch_n, self.patch_size = patch_n, patch_size

    def __len__(self): return len(self.target_)

    def __getitem__(self, idx):
        in_img, tg_img = np.load(self.input_[idx]), np.load(self.target_[idx])
        
        # --- 物理精度修复：确保使用 float32，并严禁任何亮度缩放 ---
        in_img = in_img.astype(np.float32)
        tg_img = tg_img.astype(np.float32)
        
        if self.mode == 'train' and self.patch_size:
            h, w = in_img.shape
            pi, pt = [], []
            for _ in range(self.patch_n):
                y, x = np.random.randint(0, h - self.patch_size), np.random.randint(0, w - self.patch_size)
                p_in = in_img[y:y+self.patch_size, x:x+self.patch_size]
                p_tg = tg_img[y:y+self.patch_size, x:x+self.patch_size]
                
                # --- 1. 几何增强 (保留：这是提升泛化能力的良药) ---
                if np.random.random() > 0.5:
                    p_in = np.flip(p_in, axis=1); p_tg = np.flip(p_tg, axis=1)
                if np.random.random() > 0.5:
                    p_in = np.flip(p_in, axis=0); p_tg = np.flip(p_tg, axis=0)
                k = np.random.randint(0, 4)
                p_in = np.rot90(p_in, k); p_tg = np.rot90(p_tg, k)
                
                # 注意：此处不再添加任何随机噪声和 alpha 缩放，保持 CT 值的物理真实性
                pi.append(p_in.copy()); pt.append(p_tg.copy())
            return np.array(pi), np.array(pt)
            
        return in_img, tg_img

def get_loader(mode, saved_path, test_patient, patch_n=5, patch_size=128, batch_size=2):
    """
    推荐参数设置：
    - patch_size=256: 冲击 35dB 的黄金尺度，兼顾感受野与显存。
    - patch_n=5: 每张图取 5 个块，配合 batch_size=2，每次迭代模型看 10 个 256 块。
    - batch_size=2: 配合 solver.py 中的 4 步梯度累加，等效 batch_size=8。
    """
    ds = ct_dataset(mode, saved_path, test_patient, patch_n, patch_size)
    return DataLoader(
        ds, 
        batch_size=batch_size if mode=='train' else 1, 
        shuffle=(mode=='train'), 
        num_workers=4,
        pin_memory=True # 开启 pin_memory 加速 GPU 数据读取
    )