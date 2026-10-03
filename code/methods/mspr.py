import torch
import torch.nn as nn
import torch.nn.functional as F


class MSPR(nn.Module):
    """Multi-Scale Prototype Representation.

    Aggregates three decoder feature maps into a unified coarse feature space and
    maintains class prototypes using an EMA update, matching the mechanism in the
    manuscript and the original notebook.
    """

    def __init__(self, feature_channels=(64, 128, 256), num_classes=4, momentum=0.99):
        super().__init__()
        self.feature_channels = tuple(feature_channels)
        self.total_channels = sum(self.feature_channels)
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)

        self.register_buffer(
            "prototypes", torch.zeros(self.num_classes, self.total_channels)
        )
        self.register_buffer("prototype_initialized", torch.tensor(False))

    def aggregate(self, features):
        if len(features) != 3:
            raise ValueError("MSPR expects three consecutive decoder feature maps.")
        f1, f2, f3 = features
        target_size = f3.shape[-2:]
        f1 = F.interpolate(f1, size=target_size, mode="bilinear", align_corners=True)
        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=True)
        return torch.cat([f1, f2, f3], dim=1)

    @torch.no_grad()
    def _update_flat(self, features, labels, valid_mask=None):
        if valid_mask is not None:
            features = features[valid_mask]
            labels = labels[valid_mask]
        if features.numel() == 0:
            return

        for class_id in range(self.num_classes):
            class_mask = labels == class_id
            if class_mask.any():
                centroid = features[class_mask].mean(dim=0)
                centroid = F.normalize(centroid, dim=0)
                if not self.prototype_initialized.item():
                    self.prototypes[class_id] = centroid
                else:
                    self.prototypes[class_id] = (
                        self.momentum * self.prototypes[class_id]
                        + (1.0 - self.momentum) * centroid
                    )

        self.prototypes.copy_(F.normalize(self.prototypes, dim=1))
        self.prototype_initialized.fill_(True)

    @torch.no_grad()
    def update_labeled(self, student_features, labels):
        raw = self.aggregate(student_features)
        h, w = raw.shape[-2:]
        labels_small = F.interpolate(
            labels.float().unsqueeze(1), size=(h, w), mode="nearest"
        ).squeeze(1).long()
        features_flat = raw.permute(0, 2, 3, 1).reshape(-1, raw.shape[1])
        labels_flat = labels_small.reshape(-1)
        self._update_flat(features_flat, labels_flat)

    @torch.no_grad()
    def update_unlabeled(self, teacher_features, teacher_logits, confidence_threshold=0.9):
        raw = self.aggregate(teacher_features)
        h, w = raw.shape[-2:]
        logits_small = F.interpolate(
            teacher_logits, size=(h, w), mode="bilinear", align_corners=False
        )
        probs = torch.softmax(logits_small, dim=1)
        confidence, pseudo = probs.max(dim=1)

        features_flat = raw.permute(0, 2, 3, 1).reshape(-1, raw.shape[1])
        labels_flat = pseudo.reshape(-1)
        valid = confidence.reshape(-1) > confidence_threshold
        self._update_flat(features_flat, labels_flat, valid)
