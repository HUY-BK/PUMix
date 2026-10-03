from pathlib import Path
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def read_list(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ordered_patients(slice_ids):
    patients, seen = [], set()
    for case in slice_ids:
        pid = case.split("_")[0]
        if pid not in seen:
            patients.append(pid)
            seen.add(pid)
    return patients


def resize_image_label(image, label=None, size=256):
    image = np.squeeze(image)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if label is None:
        return image
    label = np.squeeze(label)
    label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, label


class ArraySliceDataset(Dataset):
    """Small wrapper around loader functions; data are read lazily from disk."""

    def __init__(self, case_ids, load_fn, labeled=True, image_size=256):
        self.case_ids = list(case_ids)
        self.load_fn = load_fn
        self.labeled = labeled
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        image, label = self.load_fn(self.case_ids[index])
        image, label = resize_image_label(image, label, self.image_size)
        image = image.astype(np.float32, copy=False)
        if self.labeled:
            return image, label.astype(np.int64, copy=False)
        return image, image
