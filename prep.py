import os
import argparse
import numpy as np
import pydicom


def _safe_dcmread(path):
    try:
        return pydicom.dcmread(path)
    except Exception:
        return None


def _slice_z(s):
    if hasattr(s, "ImagePositionPatient"):
        try:
            return float(s.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(s, "SliceLocation"):
        try:
            return float(s.SliceLocation)
        except Exception:
            pass
    return 0.0


def load_scan(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Scan folder not found: {path}")

    files = []
    for name in os.listdir(path):
        fpath = os.path.join(path, name)
        if not os.path.isfile(fpath):
            continue
        if name.lower().endswith(".ima") or name.lower().endswith(".dcm"):
            files.append(fpath)

    if len(files) == 0:
        raise RuntimeError(f"No DICOM files found in: {path}")

    slices = []
    for f in files:
        ds = _safe_dcmread(f)
        if ds is not None:
            slices.append(ds)

    if len(slices) == 0:
        raise RuntimeError(f"Failed to read any valid DICOM slices in: {path}")

    slices.sort(key=_slice_z)

    if len(slices) >= 2:
        try:
            thick = abs(_slice_z(slices[0]) - _slice_z(slices[1]))
            if thick == 0:
                raise ValueError
        except Exception:
            thick = 1.0
    else:
        thick = 1.0

    for s in slices:
        try:
            s.SliceThickness = thick
        except Exception:
            pass

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    # 某些 CT 外部区域会是 -2000
    image[image == -2000] = 0

    for i, s in enumerate(slices):
        intercept = getattr(s, "RescaleIntercept", 0.0)
        slope = getattr(s, "RescaleSlope", 1.0)

        if slope != 1:
            image[i] = (slope * image[i].astype(np.float32)).astype(np.int16)

        image[i] = image[i] + np.int16(intercept)

    return image.astype(np.int16)


def normalize_(image, min_bound=-1024.0, max_bound=3072.0):
    image = (image - min_bound) / (max_bound - min_bound)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def save_dataset(args):
    os.makedirs(args.save_path, exist_ok=True)

    patients = sorted(
        [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
    )

    if len(patients) == 0:
        raise RuntimeError(f"No patient folders found in: {args.data_path}")

    for p in patients:
        print(f"Processing patient: {p}")

        for subfolder, suffix in [("quarter_3mm", "input"), ("full_3mm", "target")]:
            scan_path = os.path.join(args.data_path, p, subfolder)
            if not os.path.isdir(scan_path):
                print(f"  Skip missing folder: {scan_path}")
                continue

            try:
                slices = load_scan(scan_path)
                pixels = get_pixels_hu(slices)
            except Exception as e:
                print(f"  Failed on {scan_path}: {e}")
                continue

            for i in range(len(pixels)):
                arr = normalize_(
                    pixels[i],
                    min_bound=args.norm_range_min,
                    max_bound=args.norm_range_max,
                )

                # 零填充后文件排序更稳定
                save_name = f"{p}_{i:04d}_{suffix}.npy"
                np.save(os.path.join(args.save_path, save_name), arr)

        print(f"Finished patient: {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/opt/data/private/pdataset')
    parser.add_argument('--save_path', type=str, default='./npy_img/')
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    args = parser.parse_args()

    save_dataset(args)