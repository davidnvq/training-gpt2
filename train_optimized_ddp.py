import torch
from .utils import set_seed


def main():
    set_seed(42)
    print(torch.randn(10))


if __name__ == "__main__":
    main()
