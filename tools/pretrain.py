#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for training RF-DETR on a given dataset.

Usage:
    export CUDA_VISIBLE_DEVICES=0; python train.py --model RFDETRLarge   --batch-size 8  --grad-accum-steps 2
    export CUDA_VISIBLE_DEVICES=1; python train.py --model RFDETRXLarge  --batch-size 8  --grad-accum-steps 2
    export CUDA_VISIBLE_DEVICES=2; python train.py --model RFDETR2XLarge --batch-size 8  --grad-accum-steps 2
    export CUDA_VISIBLE_DEVICES=3; python train.py --model RFDETRNano    --batch-size 16 --grad-accum-steps 1
"""

from __future__ import annotations

__all__ = []

import argparse
import sys
from pathlib import Path

import torch
from rfdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
)
from rfdetr.detr import (
    RFDETR,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSmall,
)
from rfdetr_plus.models import (
    RFDETR2XLarge,
    RFDETR2XLargeConfig,
    RFDETRXLarge,
    RFDETRXLargeConfig,
)

current_file = Path(__file__).absolute()
current_dir = current_file.parents[0]
project_root = current_dir.parents[1]
data_dir = project_root / "data" / "aic26_cross_city"

AUG_CONFIGS = {
    "light" : {
        "HorizontalFlip": {"p": 0.5},
    },
    "medium": {
        "HorizontalFlip": {"p": 0.5},
        "Affine": {
            "scale": (0.8, 1.2),
            "translate_percent": (-0.1, 0.1),
            "rotate": (-15, 15),
            "shear": (-5, 5),
            "p": 0.5,
        },
        "ColorJitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
            "p": 0.4,
        },
    },
    "heavy" : {
        # Step 1: Size Normalization
        # Step 2: Basic Geometric Invariance
        "HorizontalFlip": {"p": 0.5},
        # Step 3: Dropout / Occlusion
        "ConstrainedCoarseDropout": {
            "num_holes_range": (1, 3),
            "hole_height_range": (0.3, 0.5),
            "hole_width_range": (0.3, 0.5),
            "bbox_labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "fill": 0,
            "p": 0.3,
        },
        # Step 4: Introduce Affine Transformations (Scale, Rotate, etc.)
        "Rotate": {"limit": 45, "p": 0.5},
        "Affine": {
            "scale": (0.8, 1.2),
            "translate_percent": (-0.1, 0.1),
            "rotate": (-15, 15),
            "shear": (-5, 5),
            "p": 0.5,
        },
        # Step 5: Domain-Specific and Advanced Augmentations
        # Lighting / exposure
        "ColorJitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
            "p": 0.4,
        },
        # Color temperature
        "PlanckianJitter": {"p": 0.4},
        # Noise
        "GaussNoise": {"p": 0.20},
        # Compression
        "ImageCompression": {"quality_range": (40, 80), "p": 0.25},
        # Step 6: Reduce Reliance on Color Features
    },
    "gray"  : {
        "RandomBrightnessContrast": {"p": 0.3},
        "CLAHE": {"p": 0.3},
        "Sharpen": {"p": 0.2},
        "ToGray": {"p": 1.0},
    },
}


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def train(args: argparse.Namespace):
    # 1. Setup
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    # Define model
    model, model_config = config_and_model_from_name(args.model)

    # Define data
    dataset_dir = data_dir / args.data

    # Define output directory
    output_dir = f"{args.model}_{args.data}_e{args.epochs:02d}"
    if args.aug_config != "no_aug":
        output_dir = f"{output_dir}_{args.aug_config}"
    if args.resolution:
        output_dir = f"{output_dir}_{args.resolution}"
    output_dir = project_root / "run" / "train" / output_dir

    # 2. Train
    epochs = args.epochs
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum_steps
    lr = args.lr
    resolution = args.resolution
    device = "cuda" if has_cuda else "cpu"
    aug_config = AUG_CONFIGS.get(args.aug_config, {})

    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        resolution=resolution,
        warmup_epochs=3,
        aug_config=aug_config,
        device=device,
        progress_bar="rich",
    )

    # 3. Convert checkpoints
    # torch.save({"model": model.model.state_dict()}, f"{output_dir}/last.pth")

# endregion


# ==============================================================================
# region UTILS
# ==============================================================================

def config_and_model_from_name(model_name: str) -> tuple[RFDETR, ModelConfig | RFDETRBaseConfig]:
    if model_name == "RFDETRNano":
        model = RFDETRNano()
        model_config = RFDETRNanoConfig()
    elif model_name == "RFDETRSmall":
        model = RFDETRSmall()
        model_config = RFDETRSmallConfig()
    elif model_name == "RFDETRMedium":
        model = RFDETRMedium()
        model_config = RFDETRMediumConfig()
    elif model_name == "RFDETRLarge":
        model = RFDETRLarge()
        model_config = RFDETRLargeConfig()
    elif model_name == "RFDETRXLarge":
        model = RFDETRXLarge(accept_platform_model_license=True)
        model_config = RFDETRXLargeConfig()
    elif model_name == "RFDETR2XLarge":
        model = RFDETR2XLarge(accept_platform_model_license=True)
        model_config = RFDETR2XLargeConfig()
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model, model_config

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    """A hub for running models."""
    train(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--model",            type=str,   default="RFDETR2XLarge")
    parser.add_argument("--data",             type=str,   default="40k")
    parser.add_argument("--epochs",           type=int,   default=5)
    parser.add_argument("--batch-size",       type=int,   default=2)
    parser.add_argument("--grad-accum-steps", type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=0.0001)
    parser.add_argument("--resolution",       type=int,   default=1080)
    parser.add_argument("--aug-config",       type=str,   default="no_aug")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
