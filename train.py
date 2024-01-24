# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from pathlib import Path
import argparse
import math
from time import time

import config
from model import Generator, Discriminator
from celeba import get_celeba_dl
from utils import save_image, get_device


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--n_epochs", type=int, required=False, default=30) # "30"
    # All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch
    # size of 128."
    parser.add_argument("--batch_size", type=int, required=False, default=128)
    parser.add_argument("--disc_lr", type=float, required=False, default=0.0002)
    parser.add_argument("--gen_lr", type=float, required=False, default=0.0002)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def save_checkpoint(epoch, disc, gen, disc_optim, gen_optim, scaler, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "G": gen.state_dict(),
        "D": disc.state_dict(),
        "D_optimizer": disc_optim.state_dict(),
        "G_optimizer": gen_optim.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(ckpt, str(save_path))


def save_gen(gen, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), str(save_path))


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    pred = torch.randn(16, 1)
    print(disc.get_loss(pred, real_or_fake="real").shape)

    disc_optim = Adam(params=disc.parameters(), lr=args.DISC_LR, betas=(config.BETA1, config.BETA2))
    gen_optim = Adam(params=gen.parameters(), lr=args.GEN_LR, betas=(config.BETA1, config.BETA2))

    scaler = GradScaler()

    train_dl = get_celeba_dl(
        data_dir=args.DATA_DIR,
        img_size=config.IMG_SIZE,
        mean=config.MEAN,
        std=config.STD,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
    )

    best_loss = math.inf
    prev_ckpt_path = ".pth"
    for epoch in range(1, args.N_EPOCHS + 1):
        accum_disc_loss = 0
        accum_gen_loss = 0
        start_time = time()
        for step, real_image in enumerate(train_dl, start=1):
            real_image = real_image.to(DEVICE)

            ### Update D.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=True):
                real_pred = disc(real_image) # $D(x)$
                 # $\log(D(x))$ # D 입장에서는 Loss가 낮아져야 함.
                real_disc_loss = disc.get_loss(real_pred, real_or_fake="real")

                fake_image = gen.sample(
                    batch_size=args.BATCH_SIZE, mean=config.MEAN, std=config.STD, device=DEVICE,
                ) # $G(z)$
                ### DO NOT update G while updating D!
                fake_pred1 = disc(fake_image.detach()) # $D(G(z))$
                # $\log(1 - D(G(z)))$
                # D 입장에서는 Loss가 낮아져야 함.
                fake_disc_loss = disc.get_loss(fake_pred1, real_or_fake="fake")

                disc_loss = (real_disc_loss + fake_disc_loss) / 2
            disc_optim.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optim)

            accum_disc_loss += disc_loss.item()

            ### Update G.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=True):
                fake_image = gen.sample(
                    batch_size=args.BATCH_SIZE, mean=config.MEAN, std=config.STD, device=DEVICE,
                ) # $G(z)$
                fake_pred2 = disc(fake_image) # $D(G(z))$
                gen_loss = disc.get_loss(fake_pred2, real_or_fake="real") # G 입장에서는 Loss가 낮아져야 함.
            gen_optim.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(gen_optim)

            accum_gen_loss += gen_loss.item()

            scaler.update()

        print(f"[ {epoch}/{args.N_EPOCHS} ]", end="")
        print(f"[ D loss: {accum_disc_loss / len(train_dl):.3f} ]", end="")
        print(f"[ G loss: {accum_gen_loss / len(train_dl):.3f} ]")

        gen_image = gen.sample(
            batch_size=args.BATCH_SIZE, mean=config.MEAN, std=config.STD, device=DEVICE,
        )
        save_image(
            gen_image,
            path=f"{Path(__file__).parent}/generated_images/during_training/celeba_epoch_{epoch}.jpg",
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
