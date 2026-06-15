from pathlib import Path
import csv
import os


REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_SOURCE = REPO_ROOT / "data" / "external" / "fsierra_volumes_norm"
SOURCE_ROOT = Path(
    os.getenv("ILCLDM_SOURCE_ROOT", str(_DEFAULT_SOURCE))
).expanduser().resolve()
DATA_ROOT = REPO_ROOT / "data"
INFO_ROOT = REPO_ROOT / "data_info"


def patient_id(path):
    return path.name.split("_", 1)[0]


def relink(src, dst):
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    os.symlink(src, dst)


def load_existing_labels():
    labels = {}
    old_csv = INFO_ROOT / "data_info.csv"
    if not old_csv.exists():
        return labels
    with old_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ID" in row and "label" in row and row["label"] != "":
                labels[row["ID"]] = int(float(row["label"]))
    return labels


def collect_split(split_name, labels):
    mri_files = {
        patient_id(path): path
        for path in (SOURCE_ROOT / f"{split_name}_mri").glob("*.nii.gz")
    }
    spect_files = {
        patient_id(path): path
        for path in (SOURCE_ROOT / f"{split_name}_spect").glob("*.nii.gz")
    }
    common_ids = sorted(set(mri_files) & set(spect_files))
    missing = sorted(set(mri_files) ^ set(spect_files))
    if missing:
        raise RuntimeError(f"Unpaired {split_name} IDs: {missing}")

    rows = []
    for image_id in common_ids:
        if image_id not in labels:
            raise RuntimeError(
                f"Missing label for ID {image_id}. Keep the previous data_info.csv "
                "or add labels before preparing this dataset."
            )
        mri_dst = DATA_ROOT / "whole_MRI" / f"{image_id}.nii.gz"
        spect_dst = DATA_ROOT / "whole_Abeta" / f"{image_id}.nii.gz"
        relink(mri_files[image_id], mri_dst)
        relink(spect_files[image_id], spect_dst)
        rows.append({
            "ID": image_id,
            "label": labels[image_id],
            "split": split_name,
        })
    return rows


def main():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_ROOT}\n"
            "Set ILCLDM_SOURCE_ROOT to the fsierra volumes_norm directory."
        )
    (DATA_ROOT / "whole_MRI").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "whole_Abeta").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "latent_Abeta").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "result" / "exp_1").mkdir(parents=True, exist_ok=True)
    INFO_ROOT.mkdir(parents=True, exist_ok=True)

    labels = load_existing_labels()
    train_rows = collect_split("train", labels)
    test_rows = collect_split("test", labels)
    rows = sorted(train_rows + test_rows, key=lambda row: row["ID"])

    with (INFO_ROOT / "data_info.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    splits = {
        "train": sorted(row["ID"] for row in train_rows),
        "validation": sorted(row["ID"] for row in test_rows),
        "test": sorted(row["ID"] for row in test_rows),
    }
    for split, ids in splits.items():
        with (INFO_ROOT / f"{split}.txt").open("w") as f:
            f.write("\n".join(ids))
            f.write("\n")

    print(f"Prepared {len(rows)} paired MRI/SPECT subjects")
    for split, ids in splits.items():
        print(f"{split}: {len(ids)}")


if __name__ == "__main__":
    main()
