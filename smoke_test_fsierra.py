import numpy as np
import torch

import config
from dataset import OneDataset, TwoDataset
from model import AAE, UNet


def main():
    one = OneDataset(task=config.train, stage="validation")
    spect, name = one[0]
    assert spect.shape == (160, 192, 160), spect.shape

    two = TwoDataset(task=config.train, stage="validation")
    mri, spect, name, label = two[0]
    assert mri.shape == (160, 192, 160), mri.shape
    assert spect.shape == (160, 192, 160), spect.shape
    assert label.shape == (1,), label.shape

    aae = AAE().to(config.device).eval()
    with torch.no_grad():
        x = torch.tensor(np.expand_dims(spect, (0, 1)), device=config.device)
        latent = aae.encoder(x)
        decoded = aae.decoder(latent)
    assert tuple(latent.shape[2:]) == (40, 48, 40), latent.shape
    assert tuple(decoded.shape[2:]) == (160, 192, 160), decoded.shape

    unet = UNet().to(config.device).eval()
    with torch.no_grad():
        y = torch.tensor(np.expand_dims(mri, (0, 1)), device=config.device)
        t = torch.ones(1, dtype=torch.long, device=config.device)
        label_t = torch.tensor(label[None, :], device=config.device)
        noise = unet(latent, y, t, label_t)
    assert noise.shape == latent.shape, noise.shape
    print("Smoke test passed")
    print(f"subject={name}, label={label.tolist()}, device={config.device}")


if __name__ == "__main__":
    main()
