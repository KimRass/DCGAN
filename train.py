# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import argparse
import math
from time import time

import config
from model import Generator, Discriminator
from celeba import get_celeba_dataloader
from utils import batched_image_to_grid, save_image, get_device, save_gen, get_elapsed_time


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True) # "30"
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=64)
    # All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128."
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--gen_weight", type=float, required=False, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    disc_optim = Adam(params=disc.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    gen_optim = Adam(params=gen.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    train_dl = get_celeba_dataloader(
        data_dir=args.data_dir,
        img_size=config.IMG_SIZE,
        mean=config.MEAN,
        std=config.STD,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
    )

    crit = nn.BCELoss()

    best_loss = math.inf
    prev_ckpt_path = ".pth"
    for epoch in range(1, args.n_epochs + 1):
        accum_real_disc_loss = 0
        accum_fake_disc_loss = 0
        accum_gen_loss = 0
        start_time = time()
        for step, real_image in enumerate(train_dl, start=1):
            real_image = real_image.to(DEVICE)

            ### Update D.
            real_pred = disc(real_image) # $D(x)$
            real_label = torch.ones_like(real_pred, device=DEVICE)
            real_disc_loss = crit(real_pred, real_label) # $\log(D(x))$ # D 입장에서는 Loss가 낮아져야 함.

            noise = torch.randn(args.batch_size, config.LATENT_DIM, device=DEVICE) # $z$
            fake_image = gen(noise) # $G(z)$
            ### DO NOT update G while updating D!
            fake_pred = disc(fake_image.detach()) # $D(G(z))$
            fake_label = torch.zeros_like(fake_pred, device=DEVICE)
            fake_disc_loss = crit(fake_pred, fake_label) # $\log(1 - D(G(z)))$ # D 입장에서는 Loss가 낮아져야 함.

            disc_loss = real_disc_loss + fake_disc_loss

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            ### Update G.
            fake_pred = disc(fake_image) # $D(G(z))$
            real_label = torch.ones_like(fake_pred, device=DEVICE)
            gen_loss = crit(fake_pred, real_label) # G 입장에서는 Loss가 낮아져야 함.
            gen_loss *= args.gen_weight

            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            accum_real_disc_loss += real_disc_loss.item()
            accum_fake_disc_loss += fake_disc_loss.item()
            accum_gen_loss += gen_loss.item()

        print(f"[ {epoch}/{args.n_epochs} ][ {get_elapsed_time(start_time)} ]", end=" ")
        print(f"[ Real D loss: {accum_real_disc_loss / len(train_dl): .4f} ]", end=" ")
        print(f"[ Fake D loss: {accum_fake_disc_loss / len(train_dl): .4f} ]", end=" ")
        print(f"[ G loss: {accum_gen_loss / len(train_dl): .4f} ]")

        gen.eval()
        with torch.no_grad():
            noise = torch.randn(args.batch_size, config.LATENT_DIM, device=DEVICE)
            fake_image = gen(noise)
            fake_image = fake_image.detach().cpu()
            grid = batched_image_to_grid(
                fake_image[: 64, ...], n_cols=8, mean=config.MEAN, std=config.STD,
            )
            save_image(grid, path=f"{Path(__file__).parent}/generated_images/epoch_{epoch}.jpg")

        tot_loss = accum_real_disc_loss + accum_gen_loss
        if tot_loss < best_loss:
            cur_ckpt_path = f"{Path(__file__).parent}/checkpoints/epoch_{epoch}.pth"
            save_gen(gen=gen, save_path=cur_ckpt_path)
            Path(prev_ckpt_path).unlink(missing_ok=True)
            print(f"Saved checkpoint.")

            best_loss = tot_loss
            prev_ckpt_path = cur_ckpt_path
