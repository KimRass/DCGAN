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

import config
from model import Generator, Discriminator
from celeba import get_celeba_dataloader
from utils import (
    batched_image_to_grid,
    save_image,
    get_device,
    save_checkpoint,
    save_gen,
    get_elapsed_time,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=False, default=30) # "30"
    parser.add_argument("--img_size", type=int, required=False, default=64)
    # All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch
    # size of 128."
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--lr", type=float, required=False, default=0.0002)
    parser.add_argument("--gen_weight", type=float, required=False, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    disc_optim = Adam(params=disc.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    gen_optim = Adam(params=gen.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))

    scaler = GradScaler()

    train_dl = get_celeba_dataloader(
        data_dir=args.data_dir,
        img_size=config.IMG_SIZE,
        mean=config.MEAN,
        std=config.STD,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
    )

    crit = nn.BCEWithLogitsLoss()

    best_loss = math.inf
    prev_ckpt_path = ".pth"
    for epoch in range(1, args.n_epochs + 1):
        accum_disc_loss = 0
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
                 # $\log(D(x))$ # D 입장에서는 Loss가 낮아져야 함.
                real_disc_loss = crit(real_pred, real_label)

                fake_image = gen(noise) # $G(z)$
                ### DO NOT update G while updating D!
                fake_pred1 = disc(fake_image.detach()) # $D(G(z))$
                # $\log(1 - D(G(z)))$ # D 입장에서는 Loss가 낮아져야 함.
                fake_disc_loss = crit(fake_pred1, fake_label)

                disc_loss = (real_disc_loss + fake_disc_loss) / 2
            disc_optim.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optim)

            accum_disc_loss += disc_loss.item()

            ### Update G.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=True):
                fake_image = gen(noise) # $G(z)$
                fake_pred2 = disc(fake_image) # $D(G(z))$
                gen_loss = crit(fake_pred2, real_label) # G 입장에서는 Loss가 낮아져야 함.
                gen_loss *= args.gen_weight
            gen_optim.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(gen_optim)

            accum_gen_loss += gen_loss.item()

            scaler.update()

        print(f"[ {epoch}/{args.n_epochs} ]", end="")
        print(f"[ D loss: {accum_disc_loss / len(train_dl):.3f} ]", end="")
        print(f"[ G loss: {accum_gen_loss / len(train_dl) / args.gen_weight:.3f} ]")

        gen.eval()
        with torch.no_grad():
            fake_image = gen(noise)
            fake_image = fake_image.detach().cpu()
            grid = batched_image_to_grid(
                fake_image[: 64, ...], n_cols=8, mean=config.MEAN, std=config.STD,
            )
            save_image(
                grid, path=f"{Path(__file__).parent}/generated_images/celeba_epoch_{epoch}.jpg"
            )

        cur_ckpt_path = f"{Path(__file__).parent}/checkpoints/ckpt_celeba_epoch_{epoch}.pth"
        save_checkpoint(
            epoch=epoch,
            disc=disc,
            gen=gen,
            disc_optim=disc_optim,
            gen_optim=gen_optim,
            scaler=scaler,
            save_path=cur_ckpt_path,
        )
        Path(prev_ckpt_path).unlink(missing_ok=True)
        prev_ckpt_path = cur_ckpt_path

        save_gen(
            gen=gen, save_path=f"{Path(__file__).parent}/pretrained/gen_celeba_epoch_{epoch}.pth",
        )
