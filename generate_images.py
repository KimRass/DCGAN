# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
from pathlib import Path
import argparse

import config
from model import Generator
from utils import get_device, save_image
from train import generate_images


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()

    gen = Generator().to(DEVICE)
    state_dict = torch.load(args.ckpt_path, map_location=DEVICE)
    gen.load_state_dict(state_dict, strict=True)

    gen.eval()
    with torch.no_grad():
        for idx in range(1, args.n_images + 1):
            noise = torch.randn(size=(args.batch_size, config.LATENT_DIM), device=DEVICE)
            gen_image = gen(noise)

            gen_image = generate_images(gen=gen, batch_size=64, device=DEVICE)
            save_image(
                gen_image, path=f"{Path(__file__).parent}/generated_images/celeba_{idx}.jpg"
            )
