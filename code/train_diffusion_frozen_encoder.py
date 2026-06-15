from main import train_LDM
from utils import seed_torch


def main():
    seed_torch()
    train_LDM()


if __name__ == "__main__":
    main()
