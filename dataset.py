from pathlib import Path

from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import config
import json
import os

REPO_ROOT = Path(__file__).resolve().parent
DATA_INFO_CSV = REPO_ROOT / "data_info" / "data_info.csv"

def read_list(file):
    with open(file, "r") as f:
        S = f.read().split()
    return list(str(i) for i in S)

def resolve_nifti_path(root, image_id):
    for suffix in (".nii", ".nii.gz"):
        path = os.path.join(root, image_id + suffix)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find NIfTI for ID '{image_id}' in {root}")

def is_latent_nifti_root(root) -> bool:
    """Latent caches are not scalar volumes; do not z-score them on load."""
    norm = str(root).replace("\\", "/").rstrip("/")
    return (
        norm.endswith("latent_Abeta")
        or norm.endswith("latent_Abeta_h64")
        or norm.endswith("latent_MRI")
        or "/latent_" in norm
    )


def resolve_latent_path(root, image_id):
    """Resolve cached latent: prefer .npy (h64), else NIfTI."""
    npy = os.path.join(root, f"{image_id}.npy")
    if os.path.exists(npy):
        return npy
    return resolve_nifti_path(root, image_id)


def load_diffusion_latent(root, image_id) -> np.ndarray:
    """
    Load SPECT latent for E2 according to ILCLDM_DIFFUSION_FEATURE_SPACE.

    - z18k: NIfTI (D,H,W)
    - h64: .npy (64,D,H,W)
    - h64_gap1: .npy h64 → channel-mean → (1,D,H,W)
    """
    path = resolve_latent_path(root, image_id)
    space = getattr(config, "DIFFUSION_FEATURE_SPACE", "z18k").strip().lower()
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 4:
            raise ValueError(f"Expected h64 array (C,D,H,W) at {path}, got {arr.shape}")
        if space == "h64_gap1":
            return arr.mean(axis=0, keepdims=True).astype(np.float32)
        if space == "h64":
            return arr
        raise ValueError(
            f"NPy latent at {path} requires DIFFUSION_FEATURE_SPACE h64|h64_gap1, got {space!r}"
        )
    data = nifti_to_numpy(path, zscore=False)
    if space not in ("z18k", "z18", ""):
        # Allow classic NIfTI only for bottleneck space.
        if space != "z18k":
            raise ValueError(f"NIfTI latents only supported for z18k; got space={space!r} path={path}")
    return np.asarray(data, dtype=np.float32)


def zscore_volume(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-volume z-score over all voxels (mean 0, std 1)."""
    data = np.asarray(data, dtype=np.float32)
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std > eps:
        data = (data - mean) / std
    else:
        data = data - mean
    return data.astype(np.float32, copy=False)


def nifti_to_numpy(file, zscore: bool | None = None):
    """
    Load a NIfTI array for the training pipeline.

    Applies per-volume z-score for image roots (whole_MRI / whole_Abeta).
    Latent NIfTI under latent_Abeta or latent_MRI are loaded without z-score.
    """
    data = nib.load(file).get_fdata()
    if zscore is None:
        zscore = not is_latent_nifti_root(os.path.dirname(file))
    if zscore:
        return zscore_volume(data)
    return np.asarray(data, dtype=np.float32)


def _latent_files(latent_dir):
    d = Path(latent_dir)
    return sorted(
        list(d.glob("*.nii")) + list(d.glob("*.nii.gz")) + list(d.glob("*.npy"))
    )


def compute_latent_stats(latent_dir):
    """Global (pooled over all voxels/subjects) mean and std of cached latents."""
    files = _latent_files(latent_dir)
    if not files:
        raise FileNotFoundError(f"No latent files found in {latent_dir}")
    total = 0.0
    total_sq = 0.0
    count = 0
    space = getattr(config, "DIFFUSION_FEATURE_SPACE", "z18k").strip().lower()
    for path in files:
        if path.suffix == ".npy":
            arr = np.load(path).astype(np.float64)
            if space == "h64_gap1" and arr.ndim == 4:
                arr = arr.mean(axis=0, keepdims=True)
            arr = arr.ravel()
        else:
            arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64).ravel()
        total += float(arr.sum())
        total_sq += float(np.square(arr).sum())
        count += arr.size
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    return float(mean), float(np.sqrt(var))


_LATENT_STATS_CACHE = {}


def _latent_stats_path(latent_dir):
    if getattr(config, "LATENT_STATS_PATH", ""):
        return Path(config.LATENT_STATS_PATH)
    base = Path(str(latent_dir).rstrip("/\\")).name or "latent"
    return REPO_ROOT / "data_info" / f"latent_stats_{base}.json"


def get_latent_stats(latent_dir=None):
    """Return (mean, std) for stage-2 latent calibration.

    Returns (0.0, 1.0) when calibration is disabled so callers can apply the
    transform unconditionally. Explicit ILCLDM_LATENT_MEAN/STD take precedence;
    otherwise stats are read from a cached JSON or computed once from the latent
    directory and persisted next to data_info/.
    """
    if not getattr(config, "LATENT_CALIBRATION", False):
        return 0.0, 1.0
    if config.LATENT_MEAN is not None and config.LATENT_STD is not None:
        std = config.LATENT_STD if config.LATENT_STD > 1e-8 else 1.0
        return float(config.LATENT_MEAN), float(std)

    latent_dir = latent_dir or config.latent_Abeta
    key = os.path.abspath(str(latent_dir))
    if key in _LATENT_STATS_CACHE:
        return _LATENT_STATS_CACHE[key]

    stats_path = _latent_stats_path(latent_dir)
    stats = None
    if stats_path.is_file():
        try:
            with open(stats_path) as handle:
                payload = json.load(handle)
            stats = (float(payload["mean"]), float(payload["std"]))
        except Exception:
            stats = None

    if stats is None:
        try:
            mean, std = compute_latent_stats(latent_dir)
        except FileNotFoundError:
            print(
                f"WARNING: latent calibration enabled but no latents in {latent_dir}; "
                "falling back to identity (mean=0, std=1)."
            )
            return 0.0, 1.0
        if std < 1e-8:
            std = 1.0
        stats = (mean, std)
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as handle:
                json.dump(
                    {
                        "mean": stats[0],
                        "std": stats[1],
                        "latent_dir": str(latent_dir),
                        "n_files": len(_latent_files(latent_dir)),
                    },
                    handle,
                    indent=2,
                )
        except Exception:
            pass

    if stats[1] < 1e-8:
        stats = (stats[0], 1.0)
    _LATENT_STATS_CACHE[key] = stats
    return stats

def random_translation(data1):
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    return data1[10+i:170+i,18+j:210+j,10+z:170+z]

def crop(data1):
    if getattr(config, "ROI_VOLUME_SHAPE", None):
        from code.volume_shapes import get_volume_shape, get_working_volume_shape, pad_volume_to_working_shape

        raw_shape = get_volume_shape()
        working_shape = get_working_volume_shape()
        shape = tuple(data1.shape[:3])
        if shape == working_shape:
            return data1
        if shape != raw_shape:
            raise ValueError(f"Expected ROI volume shape {raw_shape}, got {data1.shape}")
        return pad_volume_to_working_shape(data1, raw_shape=raw_shape, working_shape=working_shape)
    if config.SKIP_CROP:
        return data1
    return data1[10:170, 18:210, 10:170]

# This is for the training of the first stage
class OneDataset(Dataset):
    def __init__(self, root_Abeta = config.whole_Abeta, task = config.train, stage = "train", **kwargs):
        self.root_Abeta = root_Abeta
        self.task = task
        self.images = read_list(self.task)
        self.length_dataset = len(self.images)
        self.len = len(self.images)
        self.stage = stage

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_id = self.images[index % self.len]
        path_Abeta = resolve_nifti_path(self.root_Abeta, image_id)
        name = os.path.basename(path_Abeta)
        Abeta = nifti_to_numpy(path_Abeta)
        if self.stage == "train" and not config.SKIP_CROP:
            Abeta = random_translation(Abeta)
        else:
            Abeta = crop(Abeta)
        return Abeta, name

# This is for the training of the second stage
class TwoDataset(Dataset):
    def __init__(
        self,
        root_MRI=config.whole_MRI,
        root_Abeta=config.whole_Abeta,
        task=config.train,
        stage="train",
        spect_volume_root=None,
        return_spect_volume=False,
    ):
        self.root_Abeta = root_Abeta
        self.root_MRI = root_MRI
        self.spect_volume_root = spect_volume_root
        self.return_spect_volume = return_spect_volume
        self.task = task
        self.images = read_list(self.task)
        self.length_dataset = len(self.images)
        self.len = len(self.images)
        self.stage = stage
        data = pd.read_csv(DATA_INFO_CSV, encoding="ISO-8859-1", dtype={"ID": str})
        if data["ID"].duplicated().any():
            duplicates = data.loc[data["ID"].duplicated(), "ID"].tolist()
            raise ValueError(f"Duplicate labels found for IDs: {duplicates}")
        self.labels = data.set_index("ID")["label"].to_dict()

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_id = self.images[index % self.len]
        path_MRI = resolve_nifti_path(self.root_MRI, image_id)
        MRI = nifti_to_numpy(path_MRI)
        MRI = crop(MRI)

        if is_latent_nifti_root(self.root_Abeta):
            Abeta = load_diffusion_latent(self.root_Abeta, image_id)
            try:
                name = os.path.basename(resolve_latent_path(self.root_Abeta, image_id))
            except FileNotFoundError:
                name = f"{image_id}.npy"
        else:
            path_Abeta = resolve_nifti_path(self.root_Abeta, image_id)
            name = os.path.basename(path_Abeta)
            Abeta = nifti_to_numpy(path_Abeta)
            # ROI volumes must share MRI working-shape padding so encoder maps align
            # (e.g. xmodal flatten). Latents keep a different spatial size and stay raw.
            if getattr(config, "ROI_VOLUME_SHAPE", None):
                from code.volume_shapes import get_volume_shape, get_working_volume_shape

                abeta_shape = tuple(Abeta.shape[:3])
                if abeta_shape in (get_volume_shape(), get_working_volume_shape()):
                    Abeta = crop(Abeta)
            elif not self.return_spect_volume and self.stage != "train":
                Abeta = crop(Abeta)

        spect_volume = None
        if self.return_spect_volume:
            spect_root = self.spect_volume_root or config.whole_Abeta
            path_spect = resolve_nifti_path(spect_root, image_id)
            spect_volume = crop(nifti_to_numpy(path_spect))

        if image_id not in self.labels:
            raise ValueError(f"Missing label for ID '{image_id}'")
        label = np.array([self.labels[image_id]], dtype=np.float32)
        if self.return_spect_volume:
            return MRI, Abeta, name, label, spect_volume
        return MRI, Abeta, name, label
