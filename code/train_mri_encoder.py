from main import encode_mri_embeddings, train_mri_encoder
from utils import seed_torch


def main():
    seed_torch()
    train_mri_encoder()
    encode_mri_embeddings()


if __name__ == "__main__":
    main()
