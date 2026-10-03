import argparse
import copy
import csv
import json
from pathlib import Path

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import (
    ACDCDataModule,
    SynapseDataModule,
    build_transforms,
    apply_weak,
    apply_strong,
    apply_labeled,
)
from methods import AUPM, MSPR, SPCL
from networks import UNet
from utils.checkpoint import save_checkpoint
from utils.losses import CEDiceLoss, consistency_weight
from utils.metrics import evaluate
from utils.misc import load_config, set_seed, ensure_dir, cycle_next


def build_data_module(cfg):
    name = cfg["dataset"]["name"].lower()
    root = cfg["dataset"]["root"]
    labeled_num = cfg["dataset"]["labeled_num"]
    image_size = cfg["dataset"].get("image_size", 256)
    if name == "acdc":
        return ACDCDataModule(root, labeled_num, image_size)
    if name == "synapse":
        return SynapseDataModule(root, labeled_num, image_size)
    raise ValueError(f"Unsupported dataset: {name}")


@torch.no_grad()
def update_ema(student, teacher, alpha=0.99):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(alpha).add_(s, alpha=1.0 - alpha)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def match_labeled_batch(weak_l, weak_label_l, target_batch_size):
    if weak_l.shape[0] == target_batch_size:
        return weak_l, weak_label_l
    idx = torch.randint(0, weak_l.shape[0], (target_batch_size,), device=weak_l.device)
    return weak_l[idx], weak_label_l[idx]


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_data_module(cfg)
    num_classes = cfg["dataset"]["num_classes"]
    image_size = cfg["dataset"].get("image_size", 256)

    loader_cfg = cfg["loader"]
    common_loader = dict(
        shuffle=True,
        num_workers=loader_cfg.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )
    labeled_loader = DataLoader(
        data.labeled_dataset,
        batch_size=loader_cfg.get("labeled_batch_size", 4),
        **common_loader,
    )
    unlabeled_loader = DataLoader(
        data.unlabeled_dataset,
        batch_size=loader_cfg.get("unlabeled_batch_size", 4),
        **common_loader,
    )

    weak_tf, supervised_tf, strong_tf, _ = build_transforms(image_size)

    student = UNet(num_classes=num_classes).to(device)
    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    feature_channels = tuple(cfg["mspr"].get("feature_channels", [64, 128, 256]))
    mspr = MSPR(
        feature_channels=feature_channels,
        num_classes=num_classes,
        momentum=cfg["mspr"].get("momentum", 0.99),
    ).to(device)
    spcl = SPCL(
        in_channels=sum(feature_channels),
        projection_dim=cfg["spcl"].get("projection_dim", 256),
        temperature=cfg["spcl"].get("temperature", 0.07),
        confidence_threshold=cfg["spcl"].get("confidence_threshold", 0.9),
    ).to(device)
    aupm = AUPM(
        patch_size=cfg["aupm"].get("patch_size", 16),
        low_ratio=cfg["aupm"].get("low_ratio", 0.20),
        high_ratio=cfg["aupm"].get("high_ratio", 0.30),
    )

    criterion = CEDiceLoss(ce_weight=0.5, ignore_bg=False)
    train_cfg = cfg["training"]

    student_lr = train_cfg.get("student_learning_rate", train_cfg.get("learning_rate", 1e-3))
    optimizer = optim.Adam(
        student.parameters(),
        lr=student_lr,
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=train_cfg.get("lr_factor", 0.5),
        patience=train_cfg.get("lr_patience", 20),
    )

    output_dir = Path(cfg["output"]["dir"]) / cfg["experiment_name"]
    ensure_dir(output_dir)
    history_path = output_dir / "history.csv"
    best_path = output_dir / "best.pth"

    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "loss", "loss_sup", "loss_mix_l", "loss_mix_u", "loss_spcl",
            "lambda", "teacher_dice","student_dice", "lr",
        ])

    best_dice = -1.0
    epochs = int(train_cfg["epochs"])
    warmup = int(train_cfg.get("contrastive_warmup_epochs", 20))
    beta = float(train_cfg.get("contrastive_weight", 0.02))
    ema_alpha = float(train_cfg.get("ema_alpha", 0.99))
    aggregation = cfg["evaluation"].get("aggregation", "global_concat")

    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Labeled patients: {sorted(data.labeled_patients)}")
    print(f"Unlabeled patients: {len(data.unlabeled_patients)}")
    print(f"Training on: {device}")

    for epoch in range(epochs):
        student.train()
        spcl.train()
        teacher.eval()

        meters = {k: 0.0 for k in ["loss", "sup", "mix_l", "mix_u", "spcl"]}
        n_steps = max(len(labeled_loader), len(unlabeled_loader))
        l_iter, u_iter = iter(labeled_loader), iter(unlabeled_loader)

        lam = consistency_weight(
            epoch,
            max_weight=train_cfg.get("consistency_max_weight", 0.2),
            rampup_epochs=train_cfg.get("rampup_epochs", 20),
            base=0.0,
        )

        for _ in tqdm(range(n_steps), desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True, leave=True, position=0):
            (images_l, labels_l), l_iter = cycle_next(l_iter, labeled_loader)
            (images_u, _), u_iter = cycle_next(u_iter, unlabeled_loader)
            images_l, labels_l = images_l.to(device), labels_l.to(device)
            images_u = images_u.to(device)

            weak_u, _ = apply_weak(weak_tf, images_u)
            strong_u, _ = apply_strong(strong_tf, images_u)
            weak_l, weak_label_l, sup_l, sup_label_l = apply_labeled(
                weak_tf, supervised_tf, images_l, labels_l
            )
            weak_u, strong_u = weak_u.to(device), strong_u.to(device)
            weak_l, weak_label_l = weak_l.to(device), weak_label_l.to(device)
            sup_l, sup_label_l = sup_l.to(device), sup_label_l.to(device)

            with torch.no_grad():
                teacher_logits_u, t_d1, t_d2, t_d3 = teacher(weak_u)

            weak_l_mix, weak_label_l_mix = match_labeled_batch(
                weak_l, weak_label_l, weak_u.shape[0]
            )

            (
                mixed_u, mixed_l, mixed_u_label, mixed_l_label,
                mixed_u_conf, mixed_l_conf, _, _
            ) = aupm(
                strong_u, teacher_logits_u.detach(), weak_l_mix, weak_label_l_mix
            )

            student_input = torch.cat([sup_l, mixed_l, mixed_u], dim=0)
            logits_all, d1_all, d2_all, d3_all = student(student_input)
            n_sup, n_mix_l = sup_l.shape[0], mixed_l.shape[0]
            logits_sup = logits_all[:n_sup]
            logits_mix_l = logits_all[n_sup:n_sup + n_mix_l]
            logits_mix_u = logits_all[n_sup + n_mix_l:]

            loss_sup = criterion(logits_sup, sup_label_l)
            loss_mix_l = criterion(logits_mix_l, mixed_l_label)
            loss_mix_u = criterion(logits_mix_u, mixed_u_label)
            loss_spcl = torch.zeros((), device=device)

            if epoch >= warmup:
                student_sup_features = [d1_all[:n_sup], d2_all[:n_sup], d3_all[:n_sup]]
                mspr.update_labeled(student_sup_features, sup_label_l)
                mspr.update_unlabeled(
                    [t_d1.detach(), t_d2.detach(), t_d3.detach()],
                    teacher_logits_u.detach(),
                    confidence_threshold=cfg["mspr"].get("confidence_threshold", 0.9),
                )

                mixed_features_raw = mspr.aggregate([
                    d1_all[n_sup:], d2_all[n_sup:], d3_all[n_sup:]
                ])
                mixed_labels = torch.cat([mixed_l_label, mixed_u_label], dim=0)
                mixed_conf = torch.cat([mixed_l_conf, mixed_u_conf], dim=0)
                loss_spcl = spcl(
                    mixed_features_raw,
                    mixed_labels,
                    mixed_conf,
                    mspr.prototypes,
                )

            loss = loss_sup + lam * (loss_mix_l + loss_mix_u) + beta * loss_spcl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(student, teacher, alpha=ema_alpha)

            meters["loss"] += loss.item()
            meters["sup"] += loss_sup.item()
            meters["mix_l"] += loss_mix_l.item()
            meters["mix_u"] += loss_mix_u.item()
            meters["spcl"] += loss_spcl.item()

        teacher_val = evaluate(
            teacher,
            data,
            data.validation_cases(),
            num_classes,
            image_size,
            device,
            aggregation=aggregation,
            compute_surface=False,
        )

        student_val = evaluate(
            student,
            data,
            data.validation_cases(),
            num_classes,
            image_size,
            device,
            aggregation=aggregation,
            compute_surface=False,
        )

        teacher_dice = teacher_val["dice"]
        student_dice = student_val["dice"]

        scheduler.step(teacher_dice)

        if teacher_dice > best_dice:
            best_dice = teacher_dice

            save_checkpoint(
                best_path,
                student,
                teacher,
                optimizer,
                scheduler,
                mspr,
                spcl,
                epoch=epoch,
                best_score=best_dice,
                config=cfg,
            )


        denom = max(n_steps, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch + 1,
                meters["loss"] / denom,
                meters["sup"] / denom,
                meters["mix_l"] / denom,
                meters["mix_u"] / denom,
                meters["spcl"] / denom,
                lam,
                teacher_dice,
                student_dice,
                current_lr,
            ])

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Teacher Dice={teacher_dice:.4f} | "
            f"Student Dice={student_dice:.4f} | "
            f"best={best_dice:.4f} | "
            f"lr={current_lr:.3e}"
        )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    args = parser.parse_args()
    main(args.config)
