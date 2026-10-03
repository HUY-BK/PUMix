import argparse
import json
import numpy as np
import torch

from dataloaders import ACDCDataModule, SynapseDataModule
from networks import UNet
from utils.checkpoint import load_teacher_checkpoint
from utils.metrics import evaluate
from utils.misc import load_config, set_seed


def build_data(cfg):
    d = cfg["dataset"]
    if d["name"].lower() == "acdc":
        return ACDCDataModule(d["root"], d["labeled_num"], d.get("image_size", 256))
    if d["name"].lower() == "synapse":
        return SynapseDataModule(d["root"], d["labeled_num"], d.get("image_size", 256))
    raise ValueError(d["name"])


def main(config_path, checkpoint):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_data(cfg)
    model = UNet(num_classes=cfg["dataset"]["num_classes"]).to(device)
    load_teacher_checkpoint(checkpoint, model, device=device)

    result = evaluate(
        model,
        data,
        data.test_cases(),
        cfg["dataset"]["num_classes"],
        cfg["dataset"].get("image_size", 256),
        device,
        aggregation=cfg["evaluation"].get("aggregation", "global_concat"),
    )

    print("\nPer-class foreground metrics")
    for i, (d, j, a, h) in enumerate(zip(
        result["dice_per_class"], result["jaccard_per_class"],
        result["asd_per_class"], result["hd95_per_class"]
    ), start=1):
        print(f"Class {i}: Dice={d:.4f} Jaccard={j:.4f} ASD={a:.4f} HD95={h:.4f}")

    print("\nMean metrics")
    print(f"Dice:    {result['dice']:.4f}")
    print(f"Jaccard: {result['jaccard']:.4f}")
    print(f"ASD:     {result['asd']:.4f}")
    print(f"HD95:    {result['hd95']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
