import numpy as np
import torch
import cv2
from medpy.metric.binary import asd, hd95


def _prepare_slice(image, image_size, device):
    image = np.squeeze(image)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    return image


def _resize_label(label, image_size):
    label = np.squeeze(label)
    return cv2.resize(label, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def predict_case(model, image_volume, label_volume, image_size, device):
    model.eval()
    predictions, labels = [], []
    for image, label in zip(image_volume, label_volume):
        x = _prepare_slice(image, image_size, device)
        logits = model(x)[0]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        predictions.append(pred)
        labels.append(_resize_label(label, image_size))
    return np.stack(predictions), np.stack(labels)


def _region_metrics(pred, gt, num_classes, include_bg=False):
    start = 0 if include_bg else 1
    dice, jaccard = [], []
    for c in range(start, num_classes):
        p = pred == c
        g = gt == c
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        dice.append((2.0 * tp) / (2.0 * tp + fp + fn + 1e-6))
        jaccard.append(tp / (tp + fp + fn + 1e-6))
    return np.asarray(dice), np.asarray(jaccard)


def _surface_metrics(pred, gt, num_classes, spacing=(1.0, 1.0, 1.0), include_bg=False):
    start = 0 if include_bg else 1
    asd_values, hd95_values = [], []
    for c in range(start, num_classes):
        p = (pred == c).astype(np.bool_)
        g = (gt == c).astype(np.bool_)
        if p.sum() == 0 or g.sum() == 0:
            asd_values.append(np.nan)
            hd95_values.append(np.nan)
        else:
            asd_values.append(asd(p, g, voxelspacing=spacing))
            hd95_values.append(hd95(p, g, voxelspacing=spacing))
    return np.asarray(asd_values), np.asarray(hd95_values)


@torch.no_grad()
def evaluate(model, data_module, case_ids, num_classes, image_size, device,
             aggregation="global_concat", spacing=(1.0, 1.0, 1.0),compute_surface=True):

    case_predictions, case_labels = [], []
    for case_id in case_ids:
        images, labels = data_module.load_volume(case_id)
        pred, gt = predict_case(model, images, labels, image_size, device)
        case_predictions.append(pred)
        case_labels.append(gt)

    if aggregation == "global_concat":
        pred = np.concatenate(case_predictions, axis=0)
        gt = np.concatenate(case_labels, axis=0)

        dice, jac = _region_metrics(pred, gt, num_classes)

        if compute_surface:
            asd_v, hd_v = _surface_metrics(
                pred, gt, num_classes, spacing=spacing
            )
        else:
            asd_v = None
            hd_v = None

    elif aggregation == "case_mean":
        d_all, j_all = [], []
        a_all, h_all = [], []

        for pred, gt in zip(case_predictions, case_labels):
            d, j = _region_metrics(pred, gt, num_classes)
            d_all.append(d)
            j_all.append(j)

            if compute_surface:
                a, h = _surface_metrics(
                    pred, gt, num_classes, spacing=spacing
                )
                a_all.append(a)
                h_all.append(h)

        dice = np.nanmean(np.stack(d_all), axis=0)
        jac = np.nanmean(np.stack(j_all), axis=0)

        if compute_surface:
            asd_v = np.nanmean(np.stack(a_all), axis=0)
            hd_v = np.nanmean(np.stack(h_all), axis=0)
        else:
            asd_v = None
            hd_v = None


    return {
        "dice_per_class": dice,
        "jaccard_per_class": jac,
        "asd_per_class": asd_v,
        "hd95_per_class": hd_v,
        "dice": float(np.nanmean(dice)),
        "jaccard": float(np.nanmean(jac)),
        "asd": float(np.nanmean(asd_v)) if asd_v is not None else None,
        "hd95": float(np.nanmean(hd_v)) if hd_v is not None else None,
    }