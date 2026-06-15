#!/usr/bin/env python3
"""
Recover MRI_Encoder_finetuned.pth.tar for an existing finetune UNet checkpoint.

Trains only MRIEncoder (UNet frozen) to align conditioning with the saved Unet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import TwoDataset
from inference import load_mri_encoder_state_dict
from main import Diffusion, should_validate
from model import AAE, MRIEncoder, UNet
from utils import load_checkpoint, save_checkpoint, seed_torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp_nm_s2_mri_finetune")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_torch(args.seed)
    os.environ["ILCLDM_EXP"] = f"{args.exp.rstrip('/')}/"
    config.exp = os.environ["ILCLDM_EXP"]

    device = config.device
    result_dir = Path("result") / args.exp.rstrip("/")
    unet_path = result_dir / "Unet.pth.tar"
    out_path = result_dir / "MRI_Encoder_finetuned.pth.tar"
    phase1 = Path(
        os.getenv(
            "ILCLDM_MRI_ENCODER_CHECKPOINT",
            "result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar",
        )
    )
    aae_path = Path(os.getenv("ILCLDM_AAE_CHECKPOINT", "result/exp_nm_s1_spect/AAE.pth.tar"))

    if out_path.exists():
        print(f"Already exists: {out_path}")
        return

    aae = AAE().to(device).eval()
    load_checkpoint(str(aae_path), aae, None, config.learning_rate)
    aae.requires_grad_(False)

    unet = UNet().to(device)
    load_checkpoint(str(unet_path), unet, None, config.learning_rate)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    mri_encoder = MRIEncoder().to(device)
    mri_encoder.load_state_dict(load_mri_encoder_state_dict(phase1, device), strict=True)
    mri_encoder.train()

    opt = optim.AdamW(
        mri_encoder.parameters(),
        lr=config.learning_rate * config.MRI_ENCODER_FINETUNE_LR_SCALE,
    )
    diffusion = Diffusion()
    criterion = nn.MSELoss()
    max_epochs = args.epochs or min(100, config.epochs)

    dataset = TwoDataset(
        root_MRI=config.whole_MRI,
        root_Abeta=config.latent_Abeta,
        task=config.train,
        stage="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.numworker,
        pin_memory=(device == "cuda"),
    )

    print(f"Recovering encoder for {args.exp} (up to {max_epochs} epochs)...")
    for epoch in range(max_epochs):
        mri_encoder.train()
        epoch_loss = 0.0
        for MRI, latent_Abeta, _name, label in tqdm(loader, desc=f"recover:{epoch+1}"):
            label = label.to(device)
            MRI = torch.tensor(np.expand_dims(MRI, axis=1), device=device, dtype=torch.float32)
            latent_Abeta = torch.tensor(
                np.expand_dims(latent_Abeta, axis=1), device=device, dtype=torch.float32
            )
            t = diffusion.sample_timesteps(latent_Abeta.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(latent_Abeta, t)
            MRI_cond = mri_encoder(MRI)
            predicted_noise = unet(x_t, MRI_cond, t, label, precomputed_condition=True)
            loss = criterion(predicted_noise, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()

        avg_loss = epoch_loss / len(loader)
        print(f"epoch {epoch+1} loss {avg_loss:.6f}")
        if should_validate(epoch) or epoch == max_epochs - 1:
            save_checkpoint(mri_encoder, opt, filename=str(out_path))
            print(f"Saved {out_path}")

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
