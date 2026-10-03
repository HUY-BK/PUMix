import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def build_transforms(image_size=256):
    # The released preprocessed datasets are expected to contain normalized arrays.
    # mean=0/std=1 keeps the notebook behavior while ToTensorV2 adds the channel axis.
    identity_norm = A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=1.0)

    weak = A.Compose([
        A.Resize(image_size, image_size),
        identity_norm,
        ToTensorV2(),
    ])

    supervised = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        identity_norm,
        ToTensorV2(),
    ])

    strong = A.Compose([
        A.Resize(image_size, image_size),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.5),
        identity_norm,
        ToTensorV2(),
    ])

    test = A.Compose([
        A.Resize(image_size, image_size),
        identity_norm,
        ToTensorV2(),
    ])
    return weak, supervised, strong, test


def _apply_batch(transform, images, masks=None):
    xs, ys = [], []
    for i in range(images.size(0)):
        image = images[i].detach().cpu().numpy()
        aug_input = {"image": image}
        if masks is not None:
            aug_input["mask"] = masks[i].detach().cpu().numpy().astype(np.int32)
        out = transform(**aug_input)
        xs.append(out["image"])
        if masks is not None:
            mask = out["mask"]
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            ys.append(mask.long())
    return torch.stack(xs), torch.stack(ys) if masks is not None else None


def apply_weak(weak_transform, images, masks=None):
    return _apply_batch(weak_transform, images, masks)


def apply_strong(strong_transform, images, masks=None):
    return _apply_batch(strong_transform, images, masks)


def apply_labeled(weak_transform, supervised_transform, images, masks):
    weak_x, weak_y = _apply_batch(weak_transform, images, masks)
    sup_x, sup_y = _apply_batch(supervised_transform, images, masks)
    return weak_x, weak_y, sup_x, sup_y
