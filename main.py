import copy
import csv
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
from code.roi_mask_utils import try_load_roi_mask_numpy, try_load_roi_mask_tensor
from code.volume_shapes import get_latent_shape
from dataset import OneDataset, TwoDataset, get_latent_stats
from losses import spect_reconstruction_loss
from model import AAE, Discriminator, EMA, MRIEncoder, UNet, LatentClsHead, build_unet
from utils import (
    load_checkpoint,
    load_cls_head_checkpoint,
    load_unet_init_checkpoint,
    save_checkpoint,
    save_cls_head_checkpoint,
    seed_torch,
    dump_run_manifest,
)

import warnings
warnings.filterwarnings("ignore")

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
    return to_numpy_float(prediction)


def batch_metric_scores(target, prediction):
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


def batch_metric_scores_roi(target, prediction, roi_mask):
    target = to_numpy_float(target)
    prediction = prepare_prediction_for_metrics(prediction)
    if target.ndim == 4:
        target = target[0]
        prediction = prediction[0]
    mask = roi_mask > 0.5
    if not mask.any():
        return 0.0, 0.0
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


def batch_latent_metric_scores(z_gen, z_gt):
    mse = F.mse_loss(z_gen, z_gt).item()
    mae = F.l1_loss(z_gen, z_gt).item()
    return mse, mae


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

    def predict_x0(self, x_t, t, predicted_noise):
        alpha_hat = self.alpha_hat[t][:, None, None, None, None]
        return (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)

    def sample(self, model, y, sampling_steps=None, eta=None):
        if sampling_steps is None:
            sampling_steps = config.sampling_steps
        if eta is None:
            eta = config.ddim_eta

        with torch.no_grad():
            n = y.shape[0]
            latent_shape = get_latent_shape()
            x = torch.randn((n, 1, *latent_shape)).to(config.device)
            step_count = min(sampling_steps, self.noise_steps - 1)
            timesteps = torch.linspace(
                self.noise_steps - 1,
                1,
                step_count,
                dtype=torch.long,
                device=config.device,
            )
            timesteps = torch.unique_consecutive(timesteps)

            for idx, timestep in enumerate(tqdm(timesteps, position=0)):
                t = torch.full((n,), timestep.item(), dtype=torch.long, device=config.device)
                predicted_noise = model(x, y, t)

                alpha_hat = self.alpha_hat[t][:, None, None, None, None]
                if idx == len(timesteps) - 1:
                    alpha_hat_prev = torch.ones_like(alpha_hat)
                else:
                    prev_timestep = timesteps[idx + 1]
                    prev_t = torch.full((n,), prev_timestep.item(), dtype=torch.long, device=config.device)
                    alpha_hat_prev = self.alpha_hat[prev_t][:, None, None, None, None]

                pred_x0 = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
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
    roi_count = 0
    latent_mse_total = 0.0
    latent_mae_total = 0.0

    for MRI, Abeta, name, label in tqdm(loader, leave=True):
        MRI = np.expand_dims(MRI, axis=1)
        MRI = torch.tensor(MRI, device=config.device)
        Abeta_tensor = torch.tensor(
            np.expand_dims(Abeta, axis=1), device=config.device, dtype=torch.float32
        )

        with torch.no_grad():
            if mri_encoder is not None:
                MRI_cond = mri_encoder(MRI)
                sampled_latent = diffusion.sample(
                    lambda x, y, t: ema_Unet(x, y, t, precomputed_condition=True),
                    MRI_cond,
                )
            else:
                sampled_latent = diffusion.sample(ema_Unet, MRI)
            syn_Abeta = aae_model.decoder(sampled_latent)

            if track_latent:
                z_gt = aae_model.encoder(Abeta_tensor)
                latent_mse, latent_mae = batch_latent_metric_scores(sampled_latent, z_gt)
                latent_mse_total += latent_mse
                latent_mae_total += latent_mae

        batch_psnr, batch_ssim = batch_metric_scores(Abeta, syn_Abeta)
        psnr_total += batch_psnr
        ssim_total += batch_ssim

        if roi_mask_np is not None:
            psnr_roi, ssim_roi = batch_metric_scores_roi(Abeta, syn_Abeta, roi_mask_np)
            psnr_roi_total += psnr_roi
            ssim_roi_total += ssim_roi
            roi_count += 1

    metrics = {
        "psnr": psnr_total / length,
        "ssim": ssim_total / length,
        "latent_mse": latent_mse_total / length if track_latent else None,
        "latent_mae": latent_mae_total / length if track_latent else None,
    }
    if roi_mask_np is not None and roi_count > 0:
        metrics["psnr_roi"] = psnr_roi_total / roi_count
        metrics["ssim_roi"] = ssim_roi_total / roi_count
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

    average = 0
    validations_without_improvement = 0

    for epoch in range(config.epochs):
        print("epoch:", epoch)
        result_dir = build_result_dir()
        lossfile = open(result_dir / f"{run_name}_loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","recon_loss","disc_loss_epoch"])
        
        dataset = OneDataset(root_Abeta=root_input, task=config.train, stage="train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        recon_loss_epoch=0
        disc_loss_epoch=0

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            decoded_Abeta = model(Abeta)
            
            disc_real = disc(Abeta)
            disc_fake = disc(decoded_Abeta)

            recon_loss = reconstruction_training_loss(Abeta, decoded_Abeta)
            g_loss = -torch.mean(disc_fake)
            loss = recon_loss*config.Lambda + g_loss

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            disc_loss = (d_loss_real + d_loss_fake)/2

            opt_model.zero_grad()
            loss.backward(retain_graph=True)

            opt_disc.zero_grad()
            disc_loss.backward()

            opt_model.step()
            opt_disc.step()

            recon_loss_epoch = recon_loss_epoch + recon_loss
            disc_loss_epoch = disc_loss_epoch + disc_loss
        
        writer.writerow([epoch+1, recon_loss_epoch.item()/length, disc_loss_epoch.item()/length])
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
        if average_epoch > average:
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
                MRI_cond = mri_encoder(MRI)
                predicted_noise = ema_unet(x_t, MRI_cond, t, None, precomputed_condition=True)
            else:
                predicted_noise = ema_unet(x_t, MRI, t, None)
            x0_hat = diffusion.predict_x0(x_t, t, predicted_noise)
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
    ema = EMA(0.9999)
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)

    L2 = nn.MSELoss()
    diffusion = Diffusion()
    best_validation_score = 0.0
    best_latent_mse = float("inf")
    best_bal_acc = 0.0
    validations_without_improvement = 0

    use_aux_image = config.AUX_IMAGE_LOSS
    use_latent_loss = config.LATENT_LOSS
    global_step = 0
    roi_mask_tensor = try_load_roi_mask_tensor(config.device)
    roi_mask_np = try_load_roi_mask_numpy()
    use_biased_timesteps = config.HIGH_T_SAMPLE_PROB > 0
    cls_criterion = None
    if use_cls_loss and cls_head is not None:
        cls_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        lossfile = open(result_dir / "loss_curve.csv", 'a+', newline='')
        writer = csv.writer(lossfile)
        if lossfile.tell() == 0:
            header = ["Epoch", "MSE_loss"]
            if use_aux_image:
                header.append("aux_image_loss")
            if use_latent_loss:
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
            return_spect_volume=use_aux_image,
        )
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset

        MSE_loss_epoch = 0
        aux_loss_epoch = 0
        aux_steps = 0
        latent_loss_epoch = 0
        latent_steps = 0
        cls_loss_epoch = 0
        cls_steps = 0

        for idx, batch in enumerate(loop):
            if use_aux_image:
                MRI, latent_Abeta, name, label, spect_volume = batch
            else:
                MRI, latent_Abeta, name, label = batch

            label = label.to(config.device)
            train_label = label if config.USE_TRAIN_LABEL else None
            MRI = torch.tensor(np.expand_dims(MRI, axis=1), device=config.device, dtype=torch.float32)
            latent_Abeta = torch.tensor(
                np.expand_dims(latent_Abeta, axis=1), device=config.device, dtype=torch.float32
            )
            # Calibrate cached raw latent into diffusion space (identity if disabled).
            latent_Abeta = diffusion.normalize_latent(latent_Abeta)

            if use_biased_timesteps:
                t = diffusion.sample_timesteps_biased(latent_Abeta.shape[0]).to(config.device)
            else:
                t = diffusion.sample_timesteps(latent_Abeta.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(latent_Abeta, t)
            if mri_encoder is not None:
                if mri_encoder_trainable:
                    MRI_cond = mri_encoder(MRI)
                else:
                    with torch.no_grad():
                        MRI_cond = mri_encoder(MRI)
                predicted_noise = Unet(x_t, MRI_cond, t, train_label, precomputed_condition=True)
            else:
                predicted_noise = Unet(x_t, MRI, t, train_label)
            loss = L2(predicted_noise, noise)

            high_t_batch = (t > config.HIGH_T_THRESHOLD).all().item()
            apply_aux = use_aux_image and (
                high_t_batch
                or (global_step % config.AUX_IMAGE_EVERY_N == 0)
            )
            need_x0_hat = use_latent_loss or apply_aux or use_cls_loss
            x0_hat = diffusion.predict_x0(x_t, t, predicted_noise) if need_x0_hat else None

            if use_latent_loss:
                latent_loss = F.l1_loss(x0_hat, latent_Abeta)
                loss = loss + config.LATENT_LOSS_LAMBDA * latent_loss
                latent_loss_epoch += latent_loss.detach()
                latent_steps += 1

            if apply_aux:
                spect_pred = model.decoder(diffusion.denormalize_latent(x0_hat))
                spect_gt = torch.tensor(
                    np.expand_dims(spect_volume, axis=1), device=config.device, dtype=torch.float32
                )
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

            if use_cls_loss and cls_head is not None and x0_hat is not None:
                label_flat = label.reshape(-1).float()
                cls_logits = cls_head(x0_hat)
                cls_loss = cls_criterion(cls_logits, label_flat)
                loss = loss + config.CLS_LOSS_LAMBDA * cls_loss
                cls_loss_epoch += cls_loss.detach()
                cls_steps += 1

            opt_Unet.zero_grad()
            loss.backward()
            opt_Unet.step()
            ema.step_ema(ema_Unet, Unet)

            MSE_loss_epoch += L2(predicted_noise, noise).detach()
            global_step += 1

        row = [epoch + 1, MSE_loss_epoch.item() / length]
        if use_aux_image:
            row.append((aux_loss_epoch / max(aux_steps, 1)).item())
        if use_latent_loss:
            row.append((latent_loss_epoch / max(latent_steps, 1)).item())
        if use_cls_loss:
            row.append((cls_loss_epoch / max(cls_steps, 1)).item())
        writer.writerow(row)
        lossfile.close()

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

        csvfile = open(result_dir / "validation.csv", 'a+', newline='')
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            header = ['Epoch', 'PSNR', 'SSIM']
            if roi_mask_np is not None:
                header.extend(['PSNR_roi', 'SSIM_roi'])
            writer.writerow(header)
        val_row = [epoch + 1, val_metrics["psnr"], val_metrics["ssim"]]
        if roi_mask_np is not None:
            val_row.extend([val_metrics.get("psnr_roi", 0.0), val_metrics.get("ssim_roi", 0.0)])
        writer.writerow(val_row)
        csvfile.close()

        latent_csv = open(result_dir / "validation_latent.csv", 'a+', newline='')
        latent_writer = csv.writer(latent_csv)
        if latent_csv.tell() == 0:
            latent_writer.writerow(['Epoch', 'latent_MSE', 'latent_MAE', 'PSNR', 'SSIM'])
        latent_writer.writerow([
            epoch + 1,
            val_metrics["latent_mse"],
            val_metrics["latent_mae"],
            val_metrics["psnr"],
            val_metrics["ssim"],
        ])
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

        if config.EARLY_STOP_CLS_METRIC:
            improved = cls_bal_acc is not None and cls_bal_acc > best_bal_acc
            if improved:
                best_bal_acc = cls_bal_acc
        elif config.EARLY_STOP_LATENT_METRIC:
            # Explicit criterion: lower latent MSE is better.
            improved = val_metrics["latent_mse"] < best_latent_mse
            if improved:
                best_latent_mse = val_metrics["latent_mse"]
        else:
            validation_score = val_metrics["psnr"] + val_metrics["ssim"] * 10
            improved = validation_score > best_validation_score
            if improved:
                best_validation_score = validation_score

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
        else:
            validations_without_improvement += 1
            print(
                "LDM early stopping counter: "
                f"{validations_without_improvement}/{config.early_stopping_patience}"
            )
            if validations_without_improvement >= config.early_stopping_patience:
                print(f"Stopping LDM early at epoch {epoch + 1}")
                break

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
