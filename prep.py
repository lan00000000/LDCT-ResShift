import os
import argparse
import numpy as np
import pydicom

def load_scan(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path) if s.endswith('.IMA') or s.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        thick = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        thick = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices: s.SliceThickness = thick
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    image[image == -2000] = 0
    for i in range(len(slices)):
        intercept, slope = slices[i].RescaleIntercept, slices[i].RescaleSlope
        image[i] = (slope * image[i].astype(np.float64) + intercept).astype(np.int16)
    return image

def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return np.clip(image, 0.0, 1.0)

def save_dataset(args):
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    patients = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))])
    for p in patients:
        for sub, io in [("quarter_3mm", "input"), ("full_3mm", "target")]:
            path = os.path.join(args.data_path, p, sub)
            if not os.path.exists(path): continue
            pixels = get_pixels_hu(load_scan(path))
            for i in range(len(pixels)):
                f = normalize_(pixels[i], args.norm_range_min, args.norm_range_max)
                np.save(os.path.join(args.save_path, f'{p}_{i}_{io}.npy'), f)
        print(f"Processed patient: {p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/opt/data/private/pdataset')
    parser.add_argument('--save_path', type=str, default='./npy_img/')
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    save_dataset(parser.parse_args())