from pathlib import Path

from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import config
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
    return norm.endswith("latent_Abeta") or norm.endswith("latent_MRI") or "/latent_" in norm


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


def minmax_volume(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    return data


def nifti_to_numpy(file, zscore: bool | None = None):
    """
    Load a NIfTI array for the training pipeline.

    By default applies per-volume z-score for image roots (whole_MRI / whole_Abeta).
    Latent NIfTI under latent_Abeta or latent_MRI are loaded without z-score.
    Set ILCLDM_NORMALIZE=1 to use min-max instead of z-score on image volumes.
    """
    data = nib.load(file).get_fdata()
    if zscore is None:
        zscore = not is_latent_nifti_root(os.path.dirname(file))
    if zscore:
        if getattr(config, "normalize_volumes", False):
            return minmax_volume(data)
        return zscore_volume(data)
    return np.asarray(data, dtype=np.float32)

def random_translation(data1):
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    return data1[10+i:170+i,18+j:210+j,10+z:170+z]

def crop(data1):
    return data1[10:170,18:210,10:170]

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
        if self.stage == "train":
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
        path_Abeta = resolve_nifti_path(self.root_Abeta, image_id)
        name = os.path.basename(path_Abeta)
        Abeta = nifti_to_numpy(path_Abeta)
        path_MRI = resolve_nifti_path(self.root_MRI, image_id)
        MRI = nifti_to_numpy(path_MRI)
        MRI = crop(MRI)
        if not self.return_spect_volume and self.stage != "train":
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
