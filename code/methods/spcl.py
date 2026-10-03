import torch
import torch.nn as nn
import torch.nn.functional as F


class SPCL(nn.Module):
    """Structure-aware Prototype Contrastive Learning.

    Uses prototype geometry and confidence to select anchors, a prototype of the
    pseudo class as the positive, and two groups of hard/reliable pixel negatives.
    """

    def __init__(self, in_channels, projection_dim=256, temperature=0.07, confidence_threshold=0.9):
        super().__init__()
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.projector = nn.Conv2d(in_channels, projection_dim, kernel_size=1, bias=False)

    def _project_map(self, raw):
        return F.normalize(self.projector(raw), dim=1)

    def _project_prototypes(self, prototypes):
        proto = prototypes.unsqueeze(-1).unsqueeze(-1)
        proto = self.projector(proto).squeeze(-1).squeeze(-1)
        return F.normalize(proto, dim=1)

    def forward(self, mixed_features, mixed_labels, mixed_confidence, prototypes):
        raw = mixed_features
        z = self._project_map(raw)
        proto_z = self._project_prototypes(prototypes).detach()
        h, w = z.shape[-2:]

        labels = F.interpolate(
            mixed_labels.float().unsqueeze(1), size=(h, w), mode="nearest"
        ).squeeze(1).long().reshape(-1)
        conf = F.interpolate(
            mixed_confidence.float().unsqueeze(1), size=(h, w), mode="nearest"
        ).squeeze(1).reshape(-1)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])

        similarity_to_proto = z_flat @ proto_z.t()
        nearest_class = similarity_to_proto.argmax(dim=1)
        farthest_class = similarity_to_proto.argmin(dim=1)

        # Eq. (17): confidence + prototype-geometry verification.
        anchor_mask = (
            (conf > self.confidence_threshold)
            & (farthest_class != labels)
        )
        fail_mask = ~anchor_mask

        if anchor_mask.sum() == 0:
            return z_flat.sum() * 0.0

        anchors = z_flat[anchor_mask]
        anchor_labels = labels[anchor_mask]

        positive_proto = proto_z[anchor_labels]
        pos_logits = (anchors * positive_proto).sum(dim=1, keepdim=True) / self.temperature

        # N_A: valid anchors with a different pseudo-label.
        pass_pixels = z_flat[anchor_mask]
        pass_labels = labels[anchor_mask]
        mask_a = anchor_labels.unsqueeze(1) != pass_labels.unsqueeze(0)
        sim_a = anchors @ pass_pixels.t() / self.temperature
        sim_a = sim_a.masked_fill(~mask_a, float("-inf"))

        # N_B: non-anchor pixels whose nearest prototype agrees with pseudo-label,
        # while that label differs from the current anchor label.
        fail_pixels = z_flat[fail_mask]
        fail_labels = labels[fail_mask]
        fail_nearest = nearest_class[fail_mask]

        negative_logits = [sim_a]
        if fail_pixels.shape[0] > 0:
            valid_b = (fail_nearest == fail_labels)
            mask_b = valid_b.unsqueeze(0) & (
                anchor_labels.unsqueeze(1) != fail_labels.unsqueeze(0)
            )
            sim_b = anchors @ fail_pixels.t() / self.temperature
            sim_b = sim_b.masked_fill(~mask_b, float("-inf"))
            negative_logits.append(sim_b)

        neg_logits = torch.cat(negative_logits, dim=1)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -log_probs[:, 0].mean()
