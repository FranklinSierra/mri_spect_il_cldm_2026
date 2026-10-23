import copy
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from code.diffusion_features import (
    batch_latent_to_tensor,
    decode_h64_via_bottleneck,
    encode_mri_condition,
    encode_mri_dual,
    resolve_diffusion_feature_space,
)
from code.roi_mask_utils import (
    try_load_diffusion_roi_latent,
    try_load_roi_mask_numpy,
    try_load_roi_mask_tensor,
)
from code.volume_shapes import get_latent_shape
from dataset import (
    OneDataset,
    TwoDataset,
    dataset_image_to_model,
    get_latent_stats,
    stored_image_from_model,
    write_latent_mu_report,
)
from losses import spect_reconstruction_loss
from losses.spect import roi_weighted_l1
from losses.ssim import ssim_loss_3d
from model import AAE, Discriminator, EMA, MRIEncoder, UNet, LatentClsHead, build_unet, kl_standard_normal
from utils import (
    dump_run_manifest,
    load_checkpoint,
    load_cls_head_checkpoint,
    load_unet_init_checkpoint,
    maybe_clip_grad_norm,
    optimizer_trainable_params,
    resolve_aae_resume,
    save_checkpoint,
    save_cls_head_checkpoint,
    seed_torch,
)

import warnings
warnings.filterwarnings("ignore")


def _decode_sampled_latent(aae_model, sampled_latent):
    """Decode diffusion output; h64 uses optional Conv64→1 bonus path."""
    space = resolve_diffusion_feature_space()
    if space == "h64":
        if not getattr(config, "DIFFUSION_H64_DECODE_BONUS", False):
            return None
        return decode_h64_via_bottleneck(aae_model, sampled_latent)
    return aae_model.decoder(sampled_latent)

def should_validate(epoch):
    return (epoch + 1) % config.validation_interval == 0 or epoch == config.epochs - 1

def to_numpy_float(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.squeeze(data)
    return data.astype(np.float32)

def metric_data_range(target) -> float:
    """Dynamic range for PSNR/SSIM (z-score volumes need percentiles, not [0,1])."""
    target = to_numpy_float(target)
    low, high = np.percentile(target, [1.0, 99.0])
    return float(max(high - low, 1e-6))


def prepare_prediction_for_metrics(prediction):
    """Map decoder/model space to dataset space before PSNR/SSIM.

    Under ``IMAGE_VOLUME_NORM=minmax`` the AAE decoder emits tanh ``[-1, 1]``
    while loaders keep SPECT in ``[0, 1]``. Comparing those ranges directly
    collapses PSNR (~2) and drives SSIM negative. Callers must pass the raw
    model-space prediction; this helper applies ``stored_image_from_model``.
    """
    return to_numpy_float(stored_image_from_model(prediction))


def batch_metric_scores(target, prediction):
    """PSNR/SSIM sums over the batch. ``prediction`` must be model space."""
    target = to_numpy_float(target)
    prediction = prepare_prediction_for_metrics(prediction)
    if target.ndim == 3:
        target = target[None, ...]
        prediction = prediction[None, ...]
    if target.ndim != 4:
        raise ValueError(f"Expected 3D volume or batched 3D volumes, got shape {target.shape}")

    psnr_total = 0
    ssim_total = 0
    for target_i, prediction_i in zip(target, prediction):
        data_range = metric_data_range(target_i)
        psnr_total += round(psnr(target_i, prediction_i, data_range=data_range), 3)
        ssim_total += round(ssim(target_i, prediction_i, data_range=data_range), 3)
    return psnr_total, ssim_total


def _roi_metric_pair(target, prediction, mask):
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    t_crop = target[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    p_crop = prediction[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    m_crop = mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    fill = float(t_crop[m_crop].mean())
    t_crop = np.where(m_crop, t_crop, fill)
    p_crop = np.where(m_crop, p_crop, fill)
    data_range = metric_data_range(t_crop)
    return (
        round(psnr(t_crop, p_crop, data_range=data_range), 3),
        round(ssim(t_crop, p_crop, data_range=data_range), 3),
    )


def batch_metric_scores_roi(target, prediction, roi_mask):
    """Sum of per-subject ROI scores. The caller divides by the subject count.

    ``prediction`` must be model space (same convention as ``batch_metric_scores``).
    """
    target = to_numpy_float(target)
    prediction = prepare_prediction_for_metrics(prediction)
    if target.ndim == 3:
        target = target[None, ...]
        prediction = prediction[None, ...]
    if target.ndim != 4:
        raise ValueError(f"Expected 3D volume or batched 3D volumes, got shape {target.shape}")
    mask = roi_mask > 0.5
    if not mask.any():
        return 0.0, 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    for target_i, prediction_i in zip(target, prediction):
        psnr_i, ssim_i = _roi_metric_pair(target_i, prediction_i, mask)
        psnr_total += psnr_i
        ssim_total += ssim_i
    return psnr_total, ssim_total


def latent_mean_baseline():
    """MSE of copying the train mean latent, in the calibrated diffusion space.

    Absent when the stats JSON has no ``mean_image_mse_calibrated`` field, so
    runs that do not use the minmax+KL stats keep their previous CSV schema.
    """
    stats_path = Path(getattr(config, "LATENT_STATS_PATH", "") or "")
    if not stats_path.is_file():
        return None
    try:
        payload = json.loads(stats_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("mean_image_mse_calibrated")
    if value is None:
        return None
    return float(value)


def batch_latent_metric_scores(z_gen, z_gt):
    """Per-subject sums, so the caller's `/ length` yields the true mean.

    F.mse_loss already averages over the batch; returning that average and then
    dividing by the subject count again under-reported latent MSE/MAE by
    n_batches / n_subjects (e.g. 5/34 = 0.147x on a 34-subject split).
    """
    n = z_gen.shape[0]
    mse = F.mse_loss(z_gen, z_gt).item() * n
    mae = F.l1_loss(z_gen, z_gt).item() * n
    return mse, mae


def mri_film_for_unet(film_tensor):
    return film_tensor if config.FILM_MODE == "spatial_ms" else None


def use_constant_latent_l1(use_latent_loss, objective="epsilon", global_step=0, every_n=10):
    """Add L1(x0_hat, x0) on every step while the network still predicts noise.

    ``global_step`` and ``every_n`` are accepted and ignored. This term is not
    gated by ``AUX_IMAGE_EVERY_N``. When the objective is already ``x0``, the
    main loss is that L1 and this extra term stays off.
    """
    del global_step, every_n
    if objective == "x0":
        return False
    return bool(use_latent_loss)


def reconstruction_training_loss(target, prediction):
    if config.USE_WEIGHTED_RECON_LOSS or config.USE_TAIL_RECON_LOSS:
        loss, _ = spect_reconstruction_loss(
            target,
            prediction,
            use_weighted=config.USE_WEIGHTED_RECON_LOSS,
            use_tail=config.USE_TAIL_RECON_LOSS,
            weighted_alpha=config.WEIGHTED_RECON_ALPHA,
            z_threshold=config.WEIGHTED_Z_THRESHOLD,
            tail_fraction=config.TAIL_RECON_FRACTION,
            tail_beta=config.TAIL_RECON_BETA,
        )
        return loss
    return torch.abs(target - prediction).mean()

class Diffusion:
    def __init__(self, noise_steps=None, beta_start=0.0001, beta_end=0.0200, schedule="cosine",
                 latent_dir=None, cosine_s=0.008):
        if noise_steps is None:
            noise_steps = config.noiseSteps
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        self.cosine_s = cosine_s

        self.beta = self.prepare_noise_schedule().to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Latent calibration stats: diffuse in (z - mean) / std, invert before decode.
        # (0.0, 1.0) when calibration is disabled -> transforms become identity.
        self.latent_mean, self.latent_std = get_latent_stats(latent_dir)
        if self.latent_std < 1e-8:
            self.latent_std = 1.0

    def normalize_latent(self, z):
        """Map a raw AAE latent into the calibrated diffusion space."""
        return (z - self.latent_mean) / self.latent_std

    def denormalize_latent(self, z):
        """Map a calibrated latent back to the raw AAE space (what the decoder expects)."""
        return z * self.latent_std + self.latent_mean

    def prepare_noise_schedule(self):
        if self.schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

        if self.schedule != "cosine":
            raise ValueError(f"Unknown diffusion noise schedule: {self.schedule}")

        s = self.cosine_s
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps)
        alpha_hat = torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_hat = alpha_hat / alpha_hat[0]
        beta = 1 - (alpha_hat[1:] / alpha_hat[:-1])
        return torch.clip(beta, 0.0001, 0.9999)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_timesteps_biased(self, n, high_prob=None, high_threshold=None):
        high_prob = config.HIGH_T_SAMPLE_PROB if high_prob is None else high_prob
        high_threshold = config.HIGH_T_THRESHOLD if high_threshold is None else high_threshold
        t = torch.empty(n, dtype=torch.long)
        for i in range(n):
            if torch.rand(()).item() < high_prob:
                t[i] = torch.randint(high_threshold, self.noise_steps, (1,)).item()
            else:
                t[i] = torch.randint(1, high_threshold, (1,)).item()
        return t

    @staticmethod
    def _clamp_x0(pred):
        lo = getattr(config, "PRED_X0_CLAMP_LO", None)
        hi = getattr(config, "PRED_X0_CLAMP_HI", None)
        if lo is None and hi is None:
            clamp = float(getattr(config, "PRED_X0_CLAMP", 0.0) or 0.0)
            return pred.clamp(-clamp, clamp) if clamp > 0 else pred
        return pred.clamp(lo, hi)

    @staticmethod
    def predicts_x0():
        return getattr(config, "DIFFUSION_OBJECTIVE", "epsilon") == "x0"

    def predict_x0(self, x_t, t, predicted_noise):
        alpha_hat = self.alpha_hat[t][:, None, None, None, None]
        pred = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
        return self._clamp_x0(pred)

    def epsilon_from_x0(self, x_t, t, pred_x0):
        """Noise implied by a clean-latent prediction. Sampler-only; not a loss target."""
        alpha_hat = self.alpha_hat[t][:, None, None, None, None]
        return (x_t - torch.sqrt(alpha_hat) * pred_x0) / torch.sqrt((1.0 - alpha_hat).clamp(min=1e-8))

    def ddim_pred(self, x_t, t, model_output):
        """Return ``(pred_x0, eps)`` for one DDIM step.

        ``epsilon``: the network emits noise; ``x0`` is the clamped algebraic inverse.
        ``x0``: the network emits the clean latent; noise is recovered only so the
        direction term of the DDIM update stays valid.
        """
        if self.predicts_x0():
            pred_x0 = self._clamp_x0(model_output)
            return pred_x0, self.epsilon_from_x0(x_t, t, pred_x0)
        return self.predict_x0(x_t, t, model_output), model_output

    def x0_from_model_output(self, x_t, t, model_output):
        pred_x0, _eps = self.ddim_pred(x_t, t, model_output)
        return pred_x0

    def sample(self, model, y, sampling_steps=None, eta=None):
        if sampling_steps is None:
            sampling_steps = config.sampling_steps
        if eta is None:
            eta = config.ddim_eta

        with torch.no_grad():
            n = y.shape[0]
            latent_shape = get_latent_shape()
            channels = int(getattr(config, "DIFFUSION_CHANNELS", 1))
            x = torch.randn((n, channels, *latent_shape)).to(config.device)
            step_count = min(sampling_steps, self.noise_steps - 1)
            t_max = int(getattr(config, "SAMPLING_T_MAX", 0) or 0)
            if t_max <= 0:
                t_max = self.noise_steps - 1
            t_max = min(t_max, self.noise_steps - 1)
            timesteps = torch.linspace(
                t_max,
                1,
                step_count,
                dtype=torch.long,
                device=config.device,
            )
            timesteps = torch.unique_consecutive(timesteps)

            for idx, timestep in enumerate(tqdm(timesteps, position=0)):
                t = torch.full((n,), timestep.item(), dtype=torch.long, device=config.device)
                model_output = model(x, y, t)
                pred_x0, predicted_noise = self.ddim_pred(x, t, model_output)

                alpha_hat = self.alpha_hat[t][:, None, None, None, None]
                if idx == len(timesteps) - 1:
                    alpha_hat_prev = torch.ones_like(alpha_hat)
                else:
                    prev_timestep = timesteps[idx + 1]
                    prev_t = torch.full((n,), prev_timestep.item(), dtype=torch.long, device=config.device)
                    alpha_hat_prev = self.alpha_hat[prev_t][:, None, None, None, None]

                sigma = eta * torch.sqrt(
                    (1 - alpha_hat_prev) / (1 - alpha_hat) * (1 - alpha_hat / alpha_hat_prev)
                )
                direction = torch.sqrt(torch.clamp(1 - alpha_hat_prev - sigma ** 2, min=0)) * predicted_noise
                noise = sigma * torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_hat_prev) * pred_x0 + direction + noise
        # Diffusion runs in calibrated space (prior ~ N(0,1)); return raw-space latent.
        return self.denormalize_latent(x)

def evaluate_ldm_split(
    diffusion,
    ema_Unet,
    aae_model,
    mri_encoder,
    task,
    stage,
    roi_mask_np=None,
    track_latent=False,
):
    dataset = TwoDataset(
        root_MRI=config.whole_MRI,
        root_Abeta=config.whole_Abeta,
        task=task,
        stage=stage,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.numworker,
        pin_memory=True,
    )
    length = dataset.length_dataset
    psnr_total = 0.0
    ssim_total = 0.0
    psnr_roi_total = 0.0
    ssim_roi_total = 0.0
    latent_mse_total = 0.0
    latent_mae_total = 0.0

    for MRI, Abeta, name, label in tqdm(loader, leave=True):
        MRI = np.expand_dims(MRI, axis=1)
        MRI = torch.tensor(MRI, device=config.device)
        Abeta_tensor = torch.tensor(
            np.expand_dims(Abeta, axis=1), device=config.device, dtype=torch.float32
        )
        feat_space = resolve_diffusion_feature_space()

        with torch.no_grad():
            if mri_encoder is not None:
                MRI_cond, MRI_film = encode_mri_dual(mri_encoder, MRI, feat_space)
                film = mri_film_for_unet(MRI_film)
                sampled_latent = diffusion.sample(
                    lambda x, y, t: ema_Unet(
                        x, y, t, mri_film=film, precomputed_condition=True
                    ),
                    MRI_cond,
                )
            else:
                sampled_latent = diffusion.sample(ema_Unet, MRI)

            if track_latent:
                if feat_space == "h64":
                    _z, z_gt = aae_model.encoder(Abeta_tensor, return_hidden=True)
                elif feat_space == "h64_gap1":
                    from code.diffusion_features import channel_mean_to_one

                    _z, h64 = aae_model.encoder(Abeta_tensor, return_hidden=True)
                    z_gt = channel_mean_to_one(h64)
                else:
                    z_gt = aae_model.encoder(Abeta_tensor)
                latent_mse, latent_mae = batch_latent_metric_scores(sampled_latent, z_gt)
                latent_mse_total += latent_mse
                latent_mae_total += latent_mae

            syn_Abeta = _decode_sampled_latent(aae_model, sampled_latent)

        if syn_Abeta is None:
            continue

        # Raw decoder output (model space); batch_metric_scores restores to dataset range.
        batch_psnr, batch_ssim = batch_metric_scores(Abeta, syn_Abeta)
        psnr_total += batch_psnr
        ssim_total += batch_ssim

        if roi_mask_np is not None:
            psnr_roi, ssim_roi = batch_metric_scores_roi(Abeta, syn_Abeta, roi_mask_np)
            psnr_roi_total += psnr_roi
            ssim_roi_total += ssim_roi

    latent_std = float(getattr(diffusion, "latent_std", 1.0) or 1.0)
    if latent_std < 1e-8:
        latent_std = 1.0
    metrics = {
        "psnr": psnr_total / length,
        "ssim": ssim_total / length,
        "latent_mse": (latent_mse_total / length) / (latent_std ** 2) if track_latent else None,
        "latent_mae": (latent_mae_total / length) / latent_std if track_latent else None,
    }
    if roi_mask_np is not None and length > 0:
        metrics["psnr_roi"] = psnr_roi_total / length
        metrics["ssim_roi"] = ssim_roi_total / length
    return metrics


def build_result_dir():
    result_dir = Path("result") / str(config.exp).strip("/")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def train_autoencoder(root_input, checkpoint_path, run_name):
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.999)) 
    disc = Discriminator().to(config.device)
    opt_disc = optim.Adam(disc.parameters(),lr=config.learning_rate,betas=(0.5, 0.999))

    result_dir = build_result_dir()
    start_epoch, average, should_load = resolve_aae_resume(
        checkpoint_path, result_dir, run_name
    )
    if should_load:
        load_checkpoint(checkpoint_path, model, opt_model, config.learning_rate)
    validations_without_improvement = 0

    for epoch in range(start_epoch, config.epochs):
        print("epoch:", epoch)
        result_dir = build_result_dir()
        lossfile = open(result_dir / f"{run_name}_loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            header = ["Epoch", "recon_loss", "disc_loss_epoch"]
            if config.KL_LOSS:
                header.append("kl_loss")
            writer.writerow(header)
        
        dataset = OneDataset(root_Abeta=root_input, task=config.train, stage="train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        recon_loss_epoch=0
        disc_loss_epoch=0
        kl_loss_epoch=0

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            if config.KL_LOSS:
                z, mu, logvar = model.encoder(Abeta, sample=True)
                decoded_Abeta = model.decoder(z)
                kl_loss = kl_standard_normal(mu, logvar)
                real = dataset_image_to_model(Abeta)
                recon_loss = reconstruction_training_loss(real, decoded_Abeta)
                disc_fake = disc(decoded_Abeta)
                g_loss = -torch.mean(disc_fake)
                loss = recon_loss + config.KL_LAMBDA * kl_loss + g_loss
                opt_model.zero_grad()
                loss.backward()
                disc_real = disc(real)
                disc_fake_det = disc(decoded_Abeta.detach())
            else:
                decoded_Abeta = model(Abeta)
                kl_loss = None
                disc_real = disc(Abeta)
                disc_fake = disc(decoded_Abeta)
                recon_loss = reconstruction_training_loss(Abeta, decoded_Abeta)
                g_loss = -torch.mean(disc_fake)
                loss = recon_loss*config.Lambda + g_loss
                opt_model.zero_grad()
                loss.backward(retain_graph=True)
                disc_fake_det = disc_fake

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake_det))
            disc_loss = (d_loss_real + d_loss_fake)/2

            opt_disc.zero_grad()
            disc_loss.backward()

            maybe_clip_grad_norm(optimizer_trainable_params(opt_model))
            opt_model.step()
            opt_disc.step()

            recon_loss_epoch = recon_loss_epoch + recon_loss
            disc_loss_epoch = disc_loss_epoch + disc_loss
            if kl_loss is not None:
                kl_loss_epoch = kl_loss_epoch + kl_loss.detach()

        row = [epoch+1, recon_loss_epoch.item()/length, disc_loss_epoch.item()/length]
        if config.KL_LOSS:
            row.append(kl_loss_epoch.item()/length)
        writer.writerow(row)
        lossfile.close()

        if not should_validate(epoch):
            continue

        #validation part
        dataset = OneDataset(root_Abeta=root_input, task=config.validation, stage="validation")
        loader = DataLoader(dataset,batch_size= 1,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        psnr_0 = 0
        ssim_0 = 0

        csvfile = open(result_dir / f"{run_name}_validation.csv", 'a+',newline = '')
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(['Epoch','PSNR','SSIM'])

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            
            decoded_Abeta = model(Abeta)
            batch_psnr, batch_ssim = batch_metric_scores(Abeta, decoded_Abeta)
            psnr_0 += batch_psnr
            ssim_0 += batch_ssim
        
        average_epoch = psnr_0/length + ssim_0 * 10/length
        writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
        csvfile.close()
        
        # test part
        delta = float(getattr(config, "EARLY_STOP_MIN_DELTA", 0.0) or 0.0)
        if average_epoch > average + delta:
            average = average_epoch
            validations_without_improvement = 0
            save_checkpoint(model, opt_model, filename=checkpoint_path)

            dataset = OneDataset(root_Abeta=root_input, task=config.test, stage="test")
            loader = DataLoader(dataset,batch_size= 1,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.length_dataset
            psnr_0 = 0
            ssim_0 = 0
            
            csvfile = open(result_dir / f"{run_name}_test.csv", 'a+',newline = '')
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(['Epoch','PSNR','SSIM'])

            for idx, (Abeta, name) in enumerate(loop):
                Abeta = np.expand_dims(Abeta, axis=1)
                Abeta = torch.tensor(Abeta)
                Abeta = Abeta.to(config.device)
                
                decoded_Abeta = model(Abeta)
                batch_psnr, batch_ssim = batch_metric_scores(Abeta, decoded_Abeta)
                psnr_0 += batch_psnr
                ssim_0 += batch_ssim

            writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
            csvfile.close()
        else:
            validations_without_improvement += 1
            print(
                "AAE early stopping counter: "
                f"{validations_without_improvement}/{config.early_stopping_patience}"
            )
            if validations_without_improvement >= config.early_stopping_patience:
                print(f"Stopping AAE early at epoch {epoch + 1}")
                break


# First stage (legacy Abeta)
def train_AAE():
    train_autoencoder(
        root_input=config.whole_Abeta,
        checkpoint_path=config.CHECKPOINT_AAE,
        run_name="stage1_abeta_autoencoder",
    )


def train_mri_encoder():
    train_autoencoder(
        root_input=config.whole_MRI,
        checkpoint_path=config.CHECKPOINT_MRI_ENCODER,
        run_name="stage1_mri_autoencoder",
    )

def encode_dataset(encoder_root, checkpoint_path, output_root, task_file):
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(checkpoint_path, model, opt_model, config.learning_rate)
    image = nib.load(config.path)
    os.makedirs(output_root, exist_ok=True)

    dataset = OneDataset(root_Abeta=encoder_root, task=task_file, stage="encoding")
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config.numworker,pin_memory=True)
    loop = tqdm(loader, leave=True)

    for idx, (Abeta,name) in enumerate(loop):
        Abeta = np.expand_dims(Abeta, axis=1)
        Abeta = torch.tensor(Abeta)
        Abeta = Abeta.to(config.device)
        
        latent_Abeta = model.encoder(Abeta)
        latent_Abeta = latent_Abeta.detach().cpu().numpy()
        latent_Abeta = np.squeeze(latent_Abeta)
        latent_Abeta = latent_Abeta.astype(np.float32)

        latent_Abeta = nib.Nifti1Image(latent_Abeta, image.affine)
        nib.save(latent_Abeta, os.path.join(output_root, str(name[0])))

    report = write_latent_mu_report(output_root)
    if report:
        print(
            "latent mu report: "
            f"mean={report['mean']:.4f} std={report['std']:.4f} "
            f"frac_|z|>4={report['frac_abs_gt_4']:.4f}"
        )


def encoding():
    encode_dataset(config.whole_Abeta, config.CHECKPOINT_AAE, config.latent_Abeta, config.train)


def encode_mri_embeddings():
    for task_file in (config.train, config.validation, config.test):
        encode_dataset(config.whole_MRI, config.MRI_ENCODER_CHECKPOINT, config.latent_MRI, task_file)


def evaluate_cls_balanced_accuracy(diffusion, ema_unet, aae_model, mri_encoder, cls_head, task_file, stage):
    """One-step x0_hat classification accuracy on a split (validation/test).

    Uses whole_Abeta volumes and encodes latents on the fly with the frozen AAE,
    because pre-encoded files in latent_Abeta/ exist only for train subjects.
    """
    cls_head.eval()
    aae_model.eval()
    dataset = TwoDataset(
        root_MRI=config.whole_MRI,
        root_Abeta=config.whole_Abeta,
        task=task_file,
        stage=stage,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.numworker,
        pin_memory=True,
    )
    y_true = []
    y_pred = []
    use_biased_timesteps = config.HIGH_T_SAMPLE_PROB > 0

    with torch.no_grad():
        for MRI, Abeta, _name, label in tqdm(loader, leave=True):
            MRI = torch.tensor(np.expand_dims(MRI, axis=1), device=config.device, dtype=torch.float32)
            Abeta_tensor = torch.tensor(
                np.expand_dims(Abeta, axis=1), device=config.device, dtype=torch.float32
            )
            latent_Abeta = aae_model.encoder(Abeta_tensor)
            latent_Abeta = diffusion.normalize_latent(latent_Abeta)
            label = label.to(config.device).reshape(-1).long()
            if use_biased_timesteps:
                t = diffusion.sample_timesteps_biased(latent_Abeta.shape[0]).to(config.device)
            else:
                t = diffusion.sample_timesteps(latent_Abeta.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(latent_Abeta, t)
            if mri_encoder is not None:
                MRI_cond, MRI_film = encode_mri_dual(
                    mri_encoder, MRI, resolve_diffusion_feature_space()
                )
                predicted_noise = ema_unet(
                    x_t,
                    MRI_cond,
                    t,
                    None,
                    mri_film=mri_film_for_unet(MRI_film),
                    precomputed_condition=True,
                )
            else:
                predicted_noise = ema_unet(x_t, MRI, t, None)
            x0_hat = diffusion.x0_from_model_output(x_t, t, predicted_noise)
            logits = cls_head(x0_hat)
            pred = (torch.sigmoid(logits) >= 0.5).long()
            y_true.extend(label.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())

    if not y_true:
        return 0.0
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


# Second stage
def train_LDM():
    result_dir = build_result_dir()
    dump_run_manifest(result_dir)
    gpus = config.gpus
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    model.eval()
    model.requires_grad_(False)

    mode = config.MRI_ENCODER_MODE
    if mode not in {"raw", "frozen", "finetune"}:
        mode = "frozen" if config.FREEZE_MRI_ENCODER else "raw"
    print(f"LDM MRI encoder mode: {mode}")

    mri_encoder = None
    mri_encoder_trainable = False
    if mode in {"frozen", "finetune"}:
        mri_encoder = MRIEncoder().to(config.device)
        encoder_state = torch.load(config.MRI_ENCODER_CHECKPOINT, map_location=config.device)
        if "state_dict" in encoder_state:
            encoder_state = encoder_state["state_dict"]
        if any(key.startswith("encoder.") for key in encoder_state.keys()):
            encoder_state = {
                key.replace("encoder.", "", 1): value
                for key, value in encoder_state.items()
                if key.startswith("encoder.")
            }
        mri_encoder.load_state_dict(encoder_state, strict=True)
        if mode == "frozen":
            mri_encoder.eval()
            mri_encoder.requires_grad_(False)
        else:
            mri_encoder.train()
            mri_encoder.requires_grad_(True)
            mri_encoder_trainable = True

    Unet = build_unet().to(config.device)
    if config.UNET_INIT_CHECKPOINT:
        load_unet_init_checkpoint(Unet, config.UNET_INIT_CHECKPOINT)
    cls_head = None
    use_cls_loss = config.CLS_LOSS
    if use_cls_loss:
        cls_head = LatentClsHead().to(config.device)
    optimizer_groups = [{"params": [p for p in Unet.parameters() if p.requires_grad], "lr": config.learning_rate}]
    if mri_encoder_trainable:
        optimizer_groups.append(
            {
                "params": [p for p in mri_encoder.parameters() if p.requires_grad],
                "lr": config.learning_rate * config.MRI_ENCODER_FINETUNE_LR_SCALE,
            }
        )
    if cls_head is not None:
        optimizer_groups.append(
            {"params": [p for p in cls_head.parameters() if p.requires_grad], "lr": config.learning_rate}
        )
    opt_Unet = optim.AdamW(optimizer_groups)
    use_data_parallel = config.device == "cuda" and torch.cuda.is_available() and len(gpus) > 1
    if use_data_parallel:
        Unet = nn.DataParallel(Unet,device_ids=gpus,output_device=gpus[0])

    # Estimate train steps for EMA schedule (auto step_start / warmup).
    with open(config.train, encoding="utf-8") as _tf:
        n_train = sum(1 for line in _tf if line.strip())
    steps_per_epoch = max(1, (n_train + config.batch_size - 1) // config.batch_size)
    estimated_total_steps = steps_per_epoch * int(config.epochs)
    ema_step_start = int(config.EMA_STEP_START)
    if ema_step_start <= 0:
        ema_step_start = max(1, int(0.05 * estimated_total_steps))
    ema_warmup = int(config.EMA_WARMUP_STEPS)
    if ema_warmup <= 0:
        ema_warmup = max(200, int(0.05 * estimated_total_steps))
    ema = EMA(
        beta=config.EMA_BETA,
        beta_start=config.EMA_BETA_START,
        warmup_steps=ema_warmup,
        step_start_ema=ema_step_start,
    )
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)
    print(
        f"EMA: beta {config.EMA_BETA_START}→{config.EMA_BETA}, "
        f"step_start={ema_step_start}, warmup_steps={ema_warmup} "
        f"(est. total steps={estimated_total_steps})"
    )
    core_unet = Unet.module if isinstance(Unet, nn.DataParallel) else Unet
    if hasattr(core_unet, "count_attn_blocks"):
        spatial_film_n = (
            core_unet.count_spatial_film_blocks()
            if hasattr(core_unet, "count_spatial_film_blocks")
            else 0
        )
        print(
            f"UNet attn_res={getattr(core_unet, 'attn_res', None)} "
            f"attn_blocks={core_unet.count_attn_blocks()} "
            f"film_mode={getattr(core_unet, 'film_mode', 'off')} "
            f"global_film={getattr(core_unet, 'use_film', False)} "
            f"spatial_film_blocks={spatial_film_n}"
        )

    wandb_run = None
    if getattr(config, "USE_WANDB", False):
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.WANDB_PROJECT,
                name=config.WANDB_RUN_NAME or config.REGISTRY_KEY or str(config.exp).rstrip("/"),
                config={k: getattr(config, k) for k in (
                    "PRED_X0_CLAMP", "LABEL_DROPOUT_PROB", "EMA_BETA", "EMA_BETA_START",
                    "DIFFUSION_OBJECTIVE", "LATENT_LOSS", "LATENT_LOSS_LAMBDA",
                    "ATTN_RES", "USE_FILM", "FILM_MODE", "FILM_COND_CHANNELS",
                    "batch_size", "learning_rate", "noiseSteps",
                )},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"wandb init failed ({exc}); continuing without wandb")
            wandb_run = None

    L2 = nn.MSELoss()
    diffusion = Diffusion()
    best_validation_score = 0.0
    best_latent_mse = float("inf")
    best_bal_acc = 0.0
    validations_without_improvement = 0

    use_aux_image = config.AUX_IMAGE_LOSS
    use_latent_loss = config.LATENT_LOSS
    predicts_x0 = diffusion.predicts_x0()
    diffusion_ssim_lambda = float(getattr(config, "DIFFUSION_SSIM_LAMBDA", 0.0) or 0.0)
    use_ssim_loss = diffusion_ssim_lambda > 0
    if use_ssim_loss and not predicts_x0:
        print(
            f"Warning: DIFFUSION_SSIM_LAMBDA={diffusion_ssim_lambda} requires "
            "objective x0; SSIM image loss disabled for this run."
        )
        use_ssim_loss = False
    elif use_ssim_loss:
        print(f"Diffusion image SSIM loss λ={diffusion_ssim_lambda} (every step)")
    need_spect_volume = use_aux_image or use_ssim_loss
    global_step = 0
    roi_mask_tensor = try_load_roi_mask_tensor(config.device)
    roi_mask_np = try_load_roi_mask_numpy()
    diffusion_roi_weight = float(getattr(config, "DIFFUSION_ROI_WEIGHT", 0.0) or 0.0)
    diffusion_roi_latent = try_load_diffusion_roi_latent(config.device)
    if diffusion_roi_weight > 0 and diffusion_roi_latent is not None:
        print(
            f"Diffusion x0 L1 ROI weight={diffusion_roi_weight} "
            f"(latent mask shape {tuple(diffusion_roi_latent.shape)})"
        )
    use_biased_timesteps = config.HIGH_T_SAMPLE_PROB > 0
    cls_criterion = None
    if use_cls_loss and cls_head is not None:
        cls_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        lossfile = open(result_dir / "loss_curve.csv", 'a+', newline='')
        writer = csv.writer(lossfile)
        if lossfile.tell() == 0:
            header = ["Epoch", "x0_l1" if predicts_x0 else "MSE_loss"]
            if use_aux_image:
                header.append("aux_image_loss")
            if use_ssim_loss:
                header.append("ssim_loss")
            if use_constant_latent_l1(use_latent_loss, objective=config.DIFFUSION_OBJECTIVE):
                header.append("latent_loss")
            if use_cls_loss:
                header.append("cls_loss")
            writer.writerow(header)
        
        dataset = TwoDataset(
            root_MRI=config.whole_MRI,
            root_Abeta=config.latent_Abeta,
            task=config.train,
            stage="train",
            spect_volume_root=config.whole_Abeta,
            return_spect_volume=need_spect_volume,
        )
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset

        MSE_loss_epoch = 0
        seen_subjects = 0
        aux_loss_epoch = 0
        aux_steps = 0
        ssim_loss_epoch = 0
        ssim_steps = 0
        latent_loss_epoch = 0
        latent_steps = 0
        cls_loss_epoch = 0
        cls_steps = 0

        for idx, batch in enumerate(loop):
            if need_spect_volume:
                MRI, latent_Abeta, name, label, spect_volume = batch
            else:
                MRI, latent_Abeta, name, label = batch
                spect_volume = None

            label = label.to(config.device)
            train_label = label if config.USE_TRAIN_LABEL else None
            MRI = torch.tensor(np.expand_dims(MRI, axis=1), device=config.device, dtype=torch.float32)
            latent_Abeta = batch_latent_to_tensor(
                latent_Abeta, config.DIFFUSION_CHANNELS, config.device
            )
            # Calibrate cached raw latent into diffusion space (identity if disabled).
            latent_Abeta = diffusion.normalize_latent(latent_Abeta)

            if use_biased_timesteps:
                t = diffusion.sample_timesteps_biased(latent_Abeta.shape[0]).to(config.device)
            else:
                t = diffusion.sample_timesteps(latent_Abeta.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(latent_Abeta, t)
            feat_space = resolve_diffusion_feature_space()
            if mri_encoder is not None:
                if mri_encoder_trainable:
                    MRI_cond, MRI_film = encode_mri_dual(mri_encoder, MRI, feat_space)
                else:
                    with torch.no_grad():
                        MRI_cond, MRI_film = encode_mri_dual(mri_encoder, MRI, feat_space)
                predicted_noise = Unet(
                    x_t,
                    MRI_cond,
                    t,
                    train_label,
                    mri_film=mri_film_for_unet(MRI_film),
                    precomputed_condition=True,
                )
            else:
                predicted_noise = Unet(x_t, MRI, t, train_label)
            if predicts_x0:
                x0_hat = diffusion.x0_from_model_output(x_t, t, predicted_noise)
                if diffusion_roi_latent is not None and diffusion_roi_weight > 0:
                    primary = roi_weighted_l1(
                        x0_hat, latent_Abeta, diffusion_roi_latent, diffusion_roi_weight
                    )
                else:
                    primary = F.l1_loss(x0_hat, latent_Abeta)
            else:
                x0_hat = None
                primary = L2(predicted_noise, noise)
            loss = primary

            high_t_batch = (t > config.HIGH_T_THRESHOLD).all().item()
            # Aux image decode only for bottleneck spaces (or h64 with bonus flag).
            can_aux_decode = feat_space != "h64" or getattr(config, "DIFFUSION_H64_DECODE_BONUS", False)
            apply_aux = use_aux_image and can_aux_decode and (
                high_t_batch
                or (global_step % config.AUX_IMAGE_EVERY_N == 0)
            )
            add_latent_l1 = use_constant_latent_l1(
                use_latent_loss,
                objective=config.DIFFUSION_OBJECTIVE,
                global_step=global_step,
                every_n=config.AUX_IMAGE_EVERY_N,
            )
            if x0_hat is None and (add_latent_l1 or apply_aux or use_cls_loss):
                x0_hat = diffusion.x0_from_model_output(x_t, t, predicted_noise)

            if add_latent_l1:
                latent_loss = F.l1_loss(x0_hat, latent_Abeta)
                loss = loss + config.LATENT_LOSS_LAMBDA * latent_loss
                latent_loss_epoch += latent_loss.detach()
                latent_steps += 1

            need_decode = (apply_aux or use_ssim_loss) and x0_hat is not None
            spect_pred = None
            spect_gt = None
            if need_decode:
                denorm = diffusion.denormalize_latent(x0_hat)
                spect_pred = _decode_sampled_latent(model, denorm)
                if spect_pred is None:
                    apply_aux = False
                else:
                    spect_gt = dataset_image_to_model(torch.tensor(
                        np.expand_dims(spect_volume, axis=1), device=config.device, dtype=torch.float32
                    ))

            if apply_aux and spect_pred is not None and spect_gt is not None:
                aux_loss, _ = spect_reconstruction_loss(
                    spect_gt,
                    spect_pred,
                    use_weighted=config.USE_WEIGHTED_RECON_LOSS,
                    use_tail=config.USE_TAIL_RECON_LOSS,
                    weighted_alpha=config.WEIGHTED_RECON_ALPHA,
                    z_threshold=config.WEIGHTED_Z_THRESHOLD,
                    tail_fraction=config.TAIL_RECON_FRACTION,
                    tail_beta=config.TAIL_RECON_BETA,
                    roi_mask=roi_mask_tensor,
                    roi_alpha=config.ROI_WEIGHT_ALPHA if roi_mask_tensor is not None else 0.0,
                )
                loss = loss + config.AUX_IMAGE_LAMBDA * aux_loss
                aux_loss_epoch += aux_loss.detach()
                aux_steps += 1

            if use_ssim_loss and spect_pred is not None and spect_gt is not None:
                ssim_term = ssim_loss_3d(spect_pred, spect_gt, data_range=2.0)
                loss = loss + diffusion_ssim_lambda * ssim_term
                ssim_loss_epoch += ssim_term.detach()
                ssim_steps += 1

            if use_cls_loss and cls_head is not None and x0_hat is not None:
                label_flat = label.reshape(-1).float()
                cls_logits = cls_head(x0_hat)
                cls_loss = cls_criterion(cls_logits, label_flat)
                loss = loss + config.CLS_LOSS_LAMBDA * cls_loss
                cls_loss_epoch += cls_loss.detach()
                cls_steps += 1

            opt_Unet.zero_grad()
            loss.backward()
            maybe_clip_grad_norm(optimizer_trainable_params(opt_Unet))
            opt_Unet.step()
            ema.step_ema(ema_Unet, Unet)

            MSE_loss_epoch += primary.detach() * latent_Abeta.shape[0]
            seen_subjects += int(latent_Abeta.shape[0])
            global_step += 1

        loss_mean = MSE_loss_epoch / max(seen_subjects, 1)
        row = [epoch + 1, loss_mean.item() if torch.is_tensor(loss_mean) else float(loss_mean)]
        if use_aux_image:
            row.append((aux_loss_epoch / max(aux_steps, 1)).item())
        if use_ssim_loss:
            row.append((ssim_loss_epoch / max(ssim_steps, 1)).item())
        if use_constant_latent_l1(use_latent_loss, objective=config.DIFFUSION_OBJECTIVE):
            row.append((latent_loss_epoch / max(latent_steps, 1)).item())
        if use_cls_loss:
            row.append((cls_loss_epoch / max(cls_steps, 1)).item())
        writer.writerow(row)
        lossfile.close()
        if wandb_run is not None:
            train_key = "train/x0_l1" if predicts_x0 else "train/mse_loss"
            log = {"epoch": epoch + 1, train_key: row[1]}
            col = 2
            if use_aux_image:
                log["train/aux_image_loss"] = row[col]
                col += 1
            if use_ssim_loss:
                log["train/ssim_loss"] = row[col]
            wandb_run.log(log, step=epoch + 1)

        if not should_validate(epoch):
            continue

        val_metrics = evaluate_ldm_split(
            diffusion,
            ema_Unet,
            model,
            mri_encoder,
            config.validation,
            "validation",
            roi_mask_np=roi_mask_np,
            track_latent=True,
        )

        mean_baseline = latent_mean_baseline()
        csvfile = open(result_dir / "validation.csv", 'a+', newline='')
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            header = ['Epoch', 'PSNR', 'SSIM']
            if roi_mask_np is not None:
                header.extend(['PSNR_roi', 'SSIM_roi'])
            if mean_baseline is not None:
                header.extend(['latent_MSE', 'latent_MSE_mean_baseline'])
            writer.writerow(header)
        val_row = [epoch + 1, val_metrics["psnr"], val_metrics["ssim"]]
        if roi_mask_np is not None:
            val_row.extend([val_metrics.get("psnr_roi", 0.0), val_metrics.get("ssim_roi", 0.0)])
        if mean_baseline is not None:
            val_row.extend([val_metrics["latent_mse"], mean_baseline])
        writer.writerow(val_row)
        csvfile.close()

        latent_csv = open(result_dir / "validation_latent.csv", 'a+', newline='')
        latent_writer = csv.writer(latent_csv)
        if latent_csv.tell() == 0:
            latent_header = ['Epoch', 'latent_MSE', 'latent_MAE', 'PSNR', 'SSIM']
            if mean_baseline is not None:
                latent_header.append('latent_MSE_mean_baseline')
            latent_writer.writerow(latent_header)
        latent_row = [
            epoch + 1,
            val_metrics["latent_mse"],
            val_metrics["latent_mae"],
            val_metrics["psnr"],
            val_metrics["ssim"],
        ]
        if mean_baseline is not None:
            latent_row.append(mean_baseline)
        latent_writer.writerow(latent_row)
        latent_csv.close()

        cls_bal_acc = None
        if cls_head is not None and (config.EARLY_STOP_CLS_METRIC or use_cls_loss):
            cls_bal_acc = evaluate_cls_balanced_accuracy(
                diffusion,
                ema_Unet,
                model,
                mri_encoder,
                cls_head,
                config.validation,
                "validation",
            )
            cls_csv = open(result_dir / "validation_cls.csv", "a+", newline="")
            cls_writer = csv.writer(cls_csv)
            if cls_csv.tell() == 0:
                cls_writer.writerow(["Epoch", "balanced_accuracy"])
            cls_writer.writerow([epoch + 1, cls_bal_acc])
            cls_csv.close()

        min_delta = float(getattr(config, "EARLY_STOP_MIN_DELTA", 0.0) or 0.0)
        if config.EARLY_STOP_CLS_METRIC:
            improved = cls_bal_acc is not None and cls_bal_acc > best_bal_acc + min_delta
            if improved:
                best_bal_acc = cls_bal_acc
        elif config.EARLY_STOP_LATENT_METRIC:
            # Explicit criterion: lower latent MSE is better.
            improved = val_metrics["latent_mse"] < best_latent_mse - min_delta
            if improved:
                best_latent_mse = val_metrics["latent_mse"]
        else:
            validation_score = val_metrics["psnr"] + val_metrics["ssim"] * 10
            improved = validation_score > best_validation_score + min_delta
            if improved:
                best_validation_score = validation_score

        if wandb_run is not None:
            wlog = {
                "epoch": epoch + 1,
                "val/psnr": val_metrics["psnr"],
                "val/ssim": val_metrics["ssim"],
                "val/latent_mse": val_metrics["latent_mse"],
            }
            if cls_bal_acc is not None:
                wlog["val/balanced_accuracy"] = cls_bal_acc
            wandb_run.log(wlog, step=epoch + 1)

        # test part
        if improved:
            validations_without_improvement = 0
            save_checkpoint(ema_Unet, opt_Unet, filename=config.CHECKPOINT_Unet)
            if cls_head is not None:
                save_cls_head_checkpoint(cls_head, result_dir / "ClsHead.pth.tar")
            if mode == "finetune" and mri_encoder is not None:
                finetuned_encoder_path = result_dir / "MRI_Encoder_finetuned.pth.tar"
                save_checkpoint(mri_encoder, None, filename=str(finetuned_encoder_path))
                print(f"=> Saved finetuned MRI encoder to {finetuned_encoder_path}")

            test_metrics = evaluate_ldm_split(
                diffusion,
                ema_Unet,
                model,
                mri_encoder,
                config.test,
                "test",
                roi_mask_np=roi_mask_np,
                track_latent=True,
            )

            csvfile = open(result_dir / "test.csv", 'a+', newline='')
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                header = ['Epoch', 'PSNR', 'SSIM']
                if roi_mask_np is not None:
                    header.extend(['PSNR_roi', 'SSIM_roi'])
                writer.writerow(header)
            test_row = [epoch + 1, test_metrics["psnr"], test_metrics["ssim"]]
            if roi_mask_np is not None:
                test_row.extend([test_metrics.get("psnr_roi", 0.0), test_metrics.get("ssim_roi", 0.0)])
            writer.writerow(test_row)
            csvfile.close()

            test_latent_csv = open(result_dir / "test_latent.csv", 'a+', newline='')
            test_latent_writer = csv.writer(test_latent_csv)
            if test_latent_csv.tell() == 0:
                test_latent_writer.writerow(['Epoch', 'latent_MSE', 'latent_MAE', 'PSNR', 'SSIM'])
            test_latent_writer.writerow([
                epoch + 1,
                test_metrics["latent_mse"],
                test_metrics["latent_mae"],
                test_metrics["psnr"],
                test_metrics["ssim"],
            ])
            test_latent_csv.close()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "test/psnr": test_metrics["psnr"],
                        "test/ssim": test_metrics["ssim"],
                        "test/latent_mse": test_metrics["latent_mse"],
                    },
                    step=epoch + 1,
                )
        else:
            validations_without_improvement += 1
            print(
                "LDM early stopping counter: "
                f"{validations_without_improvement}/{config.early_stopping_patience}"
            )
            if validations_without_improvement >= config.early_stopping_patience:
                print(f"Stopping LDM early at epoch {epoch + 1}")
                break

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    seed_torch()
    if config.RUN_STAGE1_MRI_PRETRAIN:
        train_mri_encoder()
        encode_mri_embeddings()
    else:
        train_AAE()
        encoding()

    if config.RUN_STAGE1_EMBED_EVAL:
        from code.evaluate_mri_embeddings import run_embedding_quality_gate

        run_embedding_quality_gate()

    if config.RUN_STAGE2_DIFFUSION:
        train_LDM()
