# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
from pathlib import Path
import argparse

import config
from model import Generator
from utils import get_device, save_image


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, required=True, choices=["grid", "interpolation"],
    )
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_iters", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
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

    gen = Generator(latent_dim=config.LATENT_DIM).to(DEVICE)
    state_dict = torch.load(args.MODEL_PARAMS, map_location=DEVICE)
    gen.load_state_dict(state_dict, strict=True)

    SAVE_DIR = Path(__file__).parent/f"samples/{args.MODE}"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    max_idx = get_max_index(SAVE_DIR)

    for idx in range(max_idx + 1, max_idx + 1 + args.N_ITERS):
        if args.MODE == "grid":
            gen_image = gen.sample(
                batch_size=args.BATCH_SIZE, mean=config.MEAN, std=config.STD, device=DEVICE,
            )
        elif args.MODE == "interpolation":
            gen_image = gen.interpolate_then_sample(
            batch_size=args.BATCH_SIZE, mean=config.MEAN, std=config.STD, device=DEVICE,
        )
        save_image(gen_image, path=SAVE_DIR/f"{args.MODE}_{idx}.jpg")
