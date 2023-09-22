# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from pathlib import Path
import argparse
import math
from time import time
import random

import config
from model import Generator, Discriminator
from celeba import get_celeba_dataloader
from utils import (
    batched_image_to_grid,
    save_image,
    get_device,
    save_checkpoint,
    get_elapsed_time,
    freeze_model,
    unfreeze_model,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=False, default=30) # "30"
    parser.add_argument("--img_size", type=int, required=False, default=64)
    # All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128."
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--disc_lr", type=float, required=False, default=0.00016)
    parser.add_argument("--gen_lr", type=float, required=False, default=0.0002)
    parser.add_argument("--lamb", type=float, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    disc_optim = Adam(params=disc.parameters(), lr=args.disc_lr, betas=(config.BETA1, config.BETA2))
    gen_optim = Adam(params=gen.parameters(), lr=args.gen_lr, betas=(config.BETA1, config.BETA2))

    disc_scaler = GradScaler()
    gen_scaler = GradScaler()

    train_dl = get_celeba_dataloader(
        data_dir=args.data_dir,
        img_size=config.IMG_SIZE,
        mean=config.MEAN,
        std=config.STD,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
    )

    crit = nn.BCEWithLogitsLoss()

    best_value = math.inf
    prev_ckpt_path = ".pth"
    for epoch in range(1, args.n_epochs + 1):
        accum_real_disc_loss = 0
        accum_fake_disc_loss = 0
        accum_gen_loss = 0
        start_time = time()
        for step, real_image in enumerate(train_dl, start=1):
            real_image = real_image.to(DEVICE)

            real_label = torch.ones(size=(args.batch_size, 1), device=DEVICE)
            fake_label = torch.zeros(size=(args.batch_size, 1), device=DEVICE)
            noise = torch.randn(size=(args.batch_size, config.LATENT_DIM), device=DEVICE) # $z$

            ### Update D.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=True):
                real_pred = disc(real_image) # $D(x)$
                real_disc_loss = crit(real_pred, real_label) # $\log(D(x))$ # D 입장에서는 Loss가 낮아져야 함.

                fake_image = gen(noise) # $G(z)$
                ### DO NOT update G while updating D!
                fake_pred1 = disc(fake_image.detach()) # $D(G(z))$
                # $\log(1 - D(G(z)))$ # D 입장에서는 Loss가 낮아져야 함.
                fake_disc_loss = crit(fake_pred1, fake_label)

                disc_loss = real_disc_loss + fake_disc_loss

                real_pred_mean = torch.sigmoid(real_pred).mean()
                fake_pred1_mean = torch.sigmoid(fake_pred1).mean()
                disc_loss += args.lamb * ((real_pred_mean - 0.5) ** 2 + (fake_pred1_mean - 0.5) ** 2)

            disc_optim.zero_grad()
            disc_scaler.scale(disc_loss).backward()
            disc_scaler.step(disc_optim)
            disc_scaler.update()

            ### Update G.
            freeze_model(disc)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=True):
                fake_image = gen(noise) # $G(z)$
                fake_pred2 = disc(fake_image) # $D(G(z))$
                gen_loss = crit(fake_pred2, real_label) # G 입장에서는 Loss가 낮아져야 함.

                fake_pred2_mean = torch.sigmoid(fake_pred2).mean()
                gen_loss += args.lamb * ((fake_pred2_mean - 0.5) ** 2)

            gen_optim.zero_grad()
            gen_scaler.scale(gen_loss).backward()
            gen_scaler.step(gen_optim)
            gen_scaler.update()

            accum_gen_loss += gen_loss.item()

            unfreeze_model(disc)

            accum_real_disc_loss += real_disc_loss.item()
            accum_fake_disc_loss += fake_disc_loss.item()

        value = (real_pred_mean - 0.5) ** 2 + (fake_pred1_mean - 0.5) ** 2 + (fake_pred2_mean - 0.5) ** 2

        print(f"[ {epoch}/{args.n_epochs} ]", end="")
        # print(f"[ Real D loss: {accum_real_disc_loss / len(train_dl):.3f} ]", end="")
        # print(f"[ Fake D loss: {accum_fake_disc_loss / len(train_dl):.3f} ]", end="")
        # print(f"[ G loss: {accum_gen_loss / len(train_dl):.3f} ]", end="")
        print(f"[ D: R as R: {real_pred_mean:.3f} ]", end="")
        print(f"[ D: F as F: {fake_pred1_mean:.3f} ]", end="")
        print(f"[ G: F as R: {fake_pred2_mean:.3f}]", end="")
        print(f"[ Metric: {value:.3f}]")

        gen.eval()
        with torch.no_grad():
            fake_image = gen(noise)
            fake_image = fake_image.detach().cpu()
            grid = batched_image_to_grid(
                fake_image[: 64, ...], n_cols=8, mean=config.MEAN, std=config.STD,
            )
            save_image(grid, path=f"{Path(__file__).parent}/generated_images/epoch_{epoch}.jpg")

        if value < best_value:
            cur_ckpt_path = f"{Path(__file__).parent}/checkpoints/epoch_{epoch}.pth"
            save_checkpoint(
                epoch=epoch,
                disc=disc,
                gen=gen,
                disc_optim=disc_optim,
                gen_optim=gen_optim,
                disc_scaler=disc_scaler,
                gen_scaler=gen_scaler,
                value=value,
                save_path=cur_ckpt_path,
            )
            Path(prev_ckpt_path).unlink(missing_ok=True)
            print(f"Saved checkpoint.")

            best_value = value
            prev_ckpt_path = cur_ckpt_path
