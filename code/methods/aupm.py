import torch


class AUPM:
    """Asymmetric Uncertainty-Guided PatchMix.

    Implements the paper's bidirectional patch exchange:
      * UDR: labeled -> uncertain unlabeled regions
      * LDH: uncertain unlabeled -> labeled regions

    The mixing mask is obtained from the top-k patch-wise entropy of the
    Teacher prediction on weakly augmented unlabeled images.
    """

    def __init__(self, patch_size=16, low_ratio=0.20, high_ratio=0.30):
        self.patch_size = int(patch_size)
        self.low_ratio = float(low_ratio)
        self.high_ratio = float(high_ratio)

    def _patch_entropy(self, teacher_logits):
        probs = torch.softmax(teacher_logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # [B,H,W]
        b, h, w = entropy.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Image size {(h, w)} must be divisible by patch_size={p}.")
        hp, wp = h // p, w // p
        return entropy.reshape(b, hp, p, wp, p).mean(dim=(2, 4))

    def _build_mask(self, teacher_logits):
        patch_entropy = self._patch_entropy(teacher_logits)
        b, hp, wp = patch_entropy.shape
        num_patches = hp * wp

        # Original notebook samples one ratio for the training iteration/batch.
        ratio = torch.empty(1, device=teacher_logits.device).uniform_(
            self.low_ratio, self.high_ratio
        ).item()
        k = max(1, int(num_patches * ratio))

        flat = patch_entropy.flatten(1)
        topk = flat.topk(k, dim=1).indices
        patch_mask = torch.zeros_like(flat)
        patch_mask.scatter_(1, topk, 1.0)
        patch_mask = patch_mask.view(b, hp, wp)

        p = self.patch_size
        pixel_mask = patch_mask.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)
        return pixel_mask, ratio

    @torch.no_grad()
    def __call__(self, strong_u, teacher_logits_u, weak_l, label_l):
        """
        Args:
            strong_u: strongly augmented unlabeled images [B,C,H,W]
            teacher_logits_u: Teacher logits on weak unlabeled images [B,K,H,W]
            weak_l: weakly augmented labeled images [B,C,H,W]
            label_l: labeled masks aligned with weak_l [B,H,W]

        Returns:
            mixed_u_img, mixed_l_img, mixed_u_label, mixed_l_label,
            mixed_u_conf, mixed_l_conf, mask, sampled_ratio
        """
        if strong_u.shape[0] != weak_l.shape[0]:
            raise ValueError("AUPM requires matched labeled/unlabeled batch sizes.")

        mask, ratio = self._build_mask(teacher_logits_u)  # [B,H,W]
        image_mask = mask.unsqueeze(1)

        probs_u = torch.softmax(teacher_logits_u, dim=1)
        conf_u, pseudo_u = probs_u.max(dim=1)
        conf_l = torch.ones_like(conf_u)

        # Eq. (6)-(7): Unlabeled Data Rectification (L -> U)
        mixed_u_img = strong_u * (1.0 - image_mask) + weak_l * image_mask
        mixed_u_label = pseudo_u * (1.0 - mask) + label_l * mask
        mixed_u_conf = conf_u * (1.0 - mask) + conf_l * mask

        # Eq. (8)-(9): Labeled Data Hardening (U -> L)
        mixed_l_img = weak_l * (1.0 - image_mask) + strong_u * image_mask
        mixed_l_label = label_l * (1.0 - mask) + pseudo_u * mask
        mixed_l_conf = conf_l * (1.0 - mask) + conf_u * mask

        return (
            mixed_u_img,
            mixed_l_img,
            mixed_u_label.long(),
            mixed_l_label.long(),
            mixed_u_conf,
            mixed_l_conf,
            mask,
            ratio,
        )
