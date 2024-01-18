# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
from pathlib import Path
import argparse

import config
from model import Generator
from utils import get_device, save_image, get_noise
from train import generate_images


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_cells", type=int, required=True)
    parser.add_argument("--n_iters", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def get_max_index(save_dir):
    img_paths = list(Path(save_dir).glob("celeba_*.jpg"))
    if img_paths:
        max_idx = max([int(i.stem.rsplit("_")[1]) for i in img_paths])
    else:
        max_idx = 0
    return max_idx


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()

    gen = Generator().to(DEVICE)
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    gen.load_state_dict(state_dict, strict=True)

    SAVE_DIR = Path(__file__).parent/"samples/grid"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    max_idx = get_max_index(SAVE_DIR)

    for idx in range(max_idx + 1, max_idx + 1 + args.n_iters):
        noise = get_noise(batch_size=args.n_cells, latent_dim=config.LATENT_DIM, device=DEVICE)
        gen_image = generate_images(gen=gen, noise=noise, batch_size=args.n_cells)
        save_image(gen_image, path=SAVE_DIR/"celeba_{idx}.jpg")
