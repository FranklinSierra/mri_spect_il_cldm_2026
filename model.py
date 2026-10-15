import math
import torch
from torch import nn
import torch.nn.functional as F
import config
from inspect import isfunction

# Basic components

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)

# Adversarial Autoencoder (AAE)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), averaged over every element. Zero when mu=0 and logvar=0."""
    logvar = logvar.clamp(-30.0, 20.0)
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar)


class Encoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim =config.latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = int(latent_dim)
        self.kl = bool(getattr(config, "KL_LOSS", False))
        channels = [16, 32, 64]
        num_res_blocks = 1
        out_channels_latent = self.latent_dim * (2 if self.kl else 1)
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != len(channels)-1:
                layers.append(Downsample(channels[i+1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(channels[-1], out_channels_latent, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def _latent_from_stats(self, stats: torch.Tensor, sample: bool):
        if not self.kl:
            return stats, None, None
        mu, logvar = stats.split(self.latent_dim, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        if not sample:
            return mu, mu, logvar
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def forward(self, x, return_hidden: bool = False, sample: bool = False):
        # Last layer is Conv3d(64 -> latent); preceding map is h64.
        # With KL the conv emits (mu, logvar). Callers still receive mu unless sample=True.
        h64 = self.model[:-1](x)
        latent, mu, logvar = self._latent_from_stats(self.model[-1](h64), sample=sample)
        if sample:
            if return_hidden:
                return latent, h64, mu, logvar
            return latent, mu, logvar
        if return_hidden:
            return latent, h64
        return latent


class MRIEncoder(Encoder):
    """Dedicated encoder alias for MRI pretraining/freezing workflows."""
    pass

class Decoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim = config.latent_dim):
        super(Decoder, self).__init__()
        channels = [64, 32, 16]
        num_res_blocks = 1

        in_channels = channels[0]
        layers = [nn.Conv3d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != 0:
                layers.append(Upsample(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)
        self.bound_output = bool(getattr(config, "KL_LOSS", False))

    def forward(self, x):
        decoded = self.model(x)
        if self.bound_output:
            return torch.tanh(decoded)
        return decoded

class AAE(nn.Module):
    def __init__(self):
        super(AAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data):
        encoded_data = self.encoder(data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data


class MRIAutoencoderHead(nn.Module):
    """Explicit MRI autoencoder container used in staged pipelines."""
    def __init__(self):
        super().__init__()
        self.encoder = MRIEncoder()
        self.decoder = Decoder()

    def forward(self, data):
        return self.decoder(self.encoder(data))

class Discriminator(nn.Module):
    def __init__(self, image_channels = 1, channels = [16, 32, 64, 128]):
        super(Discriminator, self).__init__()

        layers = [nn.Conv3d(image_channels, channels[0], 4, 2, 1), nn.LeakyReLU(0.2)]
        layers += [
            nn.Conv3d(channels[0], channels[1], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[1]),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [
            nn.Conv3d(channels[1], channels[2], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[2]),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [
            nn.Conv3d(channels[2], channels[3], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[3]),
            nn.LeakyReLU(0.2, True)
        ]

        layers.append(nn.Conv3d(channels[3], image_channels, 4, 2, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Unet

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class SpatialFiLM(nn.Module):
    """Spatial scale-shift modulation from an MRI feature map (e.g. encoder h64)."""

    def __init__(self, cond_channels: int, feat_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(cond_channels, 2 * feat_channels, kernel_size=1)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[2:] != h.shape[2:]:
            cond = F.interpolate(
                cond, size=h.shape[2:], mode="trilinear", align_corners=False
            )
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return h * (1 + gamma) + beta


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        time_emb_dim=None,
        cond_emb_dim=None,
        dropout=0,
        norm_groups=32,
        spatial_film=None,
    ):
        super().__init__()
        self.spatial_film = spatial_film
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.cond_mlp = nn.Sequential(
            Swish(),
            nn.Linear(cond_emb_dim, dim_out)
        ) if exists(cond_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, cond_emb=None, cond_spatial=None):
        h = self.block1(x)
        if self.spatial_film is not None and cond_spatial is not None:
            h = self.spatial_film(h, cond_spatial)
        if exists(self.mlp) and time_emb is not None:
            h += self.mlp(time_emb)[:, :, None, None, None]
        if exists(self.cond_mlp) and cond_emb is not None:
            h += self.cond_mlp(cond_emb)[:, :, None, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, depth = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width, depth)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum(
            "bnchwd, bncyxz -> bnhwdyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, depth, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, depth, height, width, depth)

        out = torch.einsum("bnhwdyxz, bncyxz -> bnchwd", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width, depth))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        cond_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False,
        spatial_film=None,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim,
            dim_out,
            time_emb_dim,
            cond_emb_dim=cond_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
            spatial_film=spatial_film,
        )
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb, cond_emb=None, cond_spatial=None):
        x = self.res_block(x, time_emb, cond_emb=cond_emb, cond_spatial=cond_spatial)
        if self.with_attn:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=128,
        norm_groups=32,
        channel_mults=(1, 2, 2, 4),
        attn_res=None,
        res_blocks=2,
        dropout=0,
        with_time_emb=True,
        image_size=40,
        use_film=None,
        film_mode=None,
    ):
        super().__init__()
        if attn_res is None:
            attn_res = tuple(config.ATTN_RES)
        if film_mode is None:
            film_mode = getattr(config, "FILM_MODE", "off")
        if use_film is not None and use_film and film_mode == "off":
            film_mode = "global"
        self.film_mode = film_mode
        use_global_film = film_mode == "global"
        use_spatial_film = film_mode == "spatial_ms"
        self.use_film = use_global_film

        if with_time_emb:
            time_dim = config.time_dim
            self.time_mlp = nn.Sequential(
                TimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                Swish(),
                nn.Linear(time_dim * 4, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        cond_emb_dim = time_dim if use_global_film and time_dim is not None else None
        if use_global_film and time_dim is not None:
            self.cond_proj = nn.Sequential(
                nn.Linear(1, time_dim),
                Swish(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.cond_proj = None

        film_cond_ch = int(getattr(config, "FILM_COND_CHANNELS", 64))

        def _res_block(pre_ch, ch_mult, use_attn):
            spatial_film = None
            if use_spatial_film:
                spatial_film = SpatialFiLM(film_cond_ch, ch_mult)
            return ResnetBlocWithAttn(
                pre_ch,
                ch_mult,
                time_emb_dim=time_dim,
                cond_emb_dim=cond_emb_dim,
                norm_groups=norm_groups,
                dropout=dropout,
                with_attn=use_attn,
                spatial_film=spatial_film,
            )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv3d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(_res_block(pre_channel, channel_mult, use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            _res_block(pre_channel, pre_channel, use_attn=True),
            _res_block(pre_channel, pre_channel, use_attn=False),
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(_res_block(
                    pre_channel + feat_channels.pop(),
                    channel_mult,
                    use_attn,
                ))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        self.cond = DiffusionConditioner()
        n_label_rows = config.num_classes + (1 if config.LABEL_DROPOUT_PROB > 0 else 0)
        self.label_emb = nn.Embedding(n_label_rows, time_dim)
        self.null_label_id = config.num_classes if config.LABEL_DROPOUT_PROB > 0 else None
        self.attn_res = tuple(attn_res)

    def count_attn_blocks(self) -> int:
        return sum(
            1
            for module in list(self.downs) + list(self.mid) + list(self.ups)
            if isinstance(module, ResnetBlocWithAttn) and module.with_attn
        )

    def count_spatial_film_blocks(self) -> int:
        return sum(
            1
            for module in list(self.downs) + list(self.mid) + list(self.ups)
            if isinstance(module, ResnetBlocWithAttn)
            and module.res_block.spatial_film is not None
        )

    def _concat_inputs(self, x, cond):
        if hasattr(self, "roi_latent"):
            roi = self.roi_latent.expand(x.shape[0], -1, -1, -1, -1)
            return torch.cat([x, cond, roi], dim=1)
        return torch.cat([x, cond], dim=1)

    @staticmethod
    def _align_skip(decoder_feat, skip):
        """Match skip-connection spatial size to the decoder feature map."""
        if decoder_feat.shape[2:] == skip.shape[2:]:
            return skip
        return torch.nn.functional.interpolate(
            skip, size=decoder_feat.shape[2:], mode="nearest"
        )

    def _resolve_label(self, label, batch_size, device):
        """Apply train dropout or null-token fallback for inference without label."""
        if label is not None:
            label = label.reshape(-1).long()
            if (
                self.training
                and self.null_label_id is not None
                and config.LABEL_DROPOUT_PROB > 0
            ):
                drop = torch.rand(label.shape[0], device=device) < config.LABEL_DROPOUT_PROB
                label = label.clone()
                label[drop] = int(self.null_label_id)
            return label
        if self.null_label_id is not None:
            return torch.full(
                (batch_size,), int(self.null_label_id), device=device, dtype=torch.long
            )
        return None

    def forward(self, x, y, time, label=None, precomputed_condition=False, mri_film=None):
        target_shape = x.shape[2:]
        cond = y if precomputed_condition else self.cond(y)
        cond_emb = None
        if self.cond_proj is not None:
            pooled = torch.nn.functional.adaptive_avg_pool3d(cond, 1).flatten(1)
            cond_emb = self.cond_proj(pooled)
        cond_spatial = mri_film if self.film_mode == "spatial_ms" else None
        x = self._concat_inputs(x, cond)
        feats = []

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        label = self._resolve_label(label, x.shape[0], x.device)
        if label is not None and t is not None:
            t = t + self.label_emb(label)

        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, cond_emb=cond_emb, cond_spatial=cond_spatial)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, cond_emb=cond_emb, cond_spatial=cond_spatial)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                skip = self._align_skip(x, feats.pop())
                x = layer(
                    torch.cat((x, skip), dim=1),
                    t,
                    cond_emb=cond_emb,
                    cond_spatial=cond_spatial,
                )
            else:
                x = layer(x)

        out = self.final_conv(x)
        if out.shape[2:] != target_shape:
            out = torch.nn.functional.interpolate(
                out, size=target_shape, mode="trilinear", align_corners=False
            )
        return out


class DiffusionConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 16, 4, 2, 1),
            ResidualBlock(16, 16),
            GroupNorm(16),
            Swish(),
            nn.Conv3d(16, 1, 4, 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


from code.volume_shapes import LEGACY_LATENT_SHAPE, get_latent_shape

LATENT_GRID_SHAPE = LEGACY_LATENT_SHAPE


class LatentClsHead(nn.Module):
    """Binary classifier on flattened predicted latent x0_hat."""

    def __init__(self, input_dim: int | None = None):
        super().__init__()
        if input_dim is None:
            ls = get_latent_shape()
            input_dim = int(ls[0] * ls[1] * ls[2])
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, 1),
        )

    def forward(self, x0_hat: torch.Tensor) -> torch.Tensor:
        batch = x0_hat.shape[0]
        flat = x0_hat.reshape(batch, -1)
        return self.net(flat).squeeze(1)


def build_unet(in_channel=None, use_roi_channel=None, out_channel=None):
    """Construct UNet; opt-in ROI input channel via ILCLDM_UNET_ROI_CHANNEL."""
    from code.diffusion_features import diffusion_channels, unet_in_channels

    if use_roi_channel is None:
        use_roi_channel = config.UNET_ROI_CHANNEL
    if in_channel is None:
        in_channel = unet_in_channels(use_roi=use_roi_channel)
    if out_channel is None:
        out_channel = diffusion_channels()
    latent_shape = get_latent_shape()
    net = UNet(
        in_channel=in_channel,
        out_channel=out_channel,
        image_size=latent_shape[0],
        attn_res=tuple(config.ATTN_RES),
        film_mode=getattr(config, "FILM_MODE", "off"),
    )
    if use_roi_channel:
        from code.roi_mask_utils import load_roi_latent_buffer

        net.register_buffer("roi_latent", load_roi_latent_buffer())
    return net


class EMA:
    """Exponential moving average with mandatory beta warmup after step_start."""

    def __init__(
        self,
        beta=None,
        beta_start=None,
        warmup_steps=None,
        step_start_ema=None,
    ):
        super().__init__()
        self.beta_end = float(config.EMA_BETA if beta is None else beta)
        self.beta_start = float(config.EMA_BETA_START if beta_start is None else beta_start)
        self.warmup_steps = int(
            config.EMA_WARMUP_STEPS if warmup_steps is None else warmup_steps
        )
        self.step_start_ema = int(
            config.EMA_STEP_START if step_start_ema is None else step_start_ema
        )
        self.step = 0
        self.beta = self.beta_start

    def current_beta(self) -> float:
        if self.step < self.step_start_ema:
            return self.beta_start
        steps_after = self.step - self.step_start_ema
        if self.warmup_steps <= 0:
            return self.beta_end
        if steps_after < self.warmup_steps:
            t = steps_after / float(self.warmup_steps)
            return self.beta_start + t * (self.beta_end - self.beta_start)
        return self.beta_end

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=None):
        start = self.step_start_ema if step_start_ema is None else int(step_start_ema)
        if self.step < start:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.beta = self.current_beta()
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
