import torch
import argparse
from pathlib import Path

import config
from PIL import Image
from utils import get_device, save_image, generate_images, get_noise
from model import Generator
from generate_images import get_max_index


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_images", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def interpolate(noise1, noise2, batch_size, latent_dim):
    lambs = torch.linspace(start=0, end=1, steps=batch_size).unsqueeze(1).repeat(1, latent_dim)
    noise = (lambs * noise1 + (1 - lambs) * noise2)
    return noise


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()

    gen = Generator().to(DEVICE)
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    gen.load_state_dict(state_dict, strict=True)

    SAVE_DIR = Path(__file__).parent/"generated_images/using_interpolation"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    max_idx = get_max_index(SAVE_DIR)

    for idx in range(max_idx + 1, max_idx + 1 + args.n_images):
        noise1 = get_noise(batch_size=1, latent_dim=config.LATENT_DIM, device=DEVICE)
        noise2 = get_noise(batch_size=1, latent_dim=config.LATENT_DIM, device=DEVICE)
        noise = interpolate(noise1, noise2, batch_size=args.batch_size, latent_dim=config.LATENT_DIM)

        gen_image = gen(noise)
        gen_image = generate_images(
            gen=gen, noise=noise, batch_size=args.batch_size, n_cols=args.batch_size,
        )
        save_image(gen_image, path=SAVE_DIR/f"celeba_{idx}.jpg")
