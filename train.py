# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import Generator, Discriminator
from celeba import get_celeba_dataloader
from image_utils import batched_image_to_grid, save_image, get_image_dataset_mean_and_std
from torch_utils import get_device, save_parameters

IMG_SIZE = 64
# All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128."
# BATCH_SIZE = 128
BATCH_SIZE = 256
N_WORKERS = 4

DEVICE = get_device()
gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)
save_parameters(
    model=gen,
    save_path=f"""{Path(__file__).parent}/parameters/test.pth"""
)

# "We used the Adam optimizer with tuned hyperparameters. We used 0.0002 for learning rate. We found
# reducing $\beta_{1}$ to 0.5 helped stabilize training."
LR = 0.0002
beta1 = 0.5
disc_optim = optim.Adam(params=disc.parameters(), lr=LR, betas=(beta1, 0.999))
gen_optim = optim.Adam(params=gen.parameters(), lr=LR, betas=(beta1, 0.999))


root = "/home/ubuntu/project/celeba"
# root = "/Users/jongbeomkim/Documents/datasets/celeba"
# mean, std = get_image_dataset_mean_and_std(root)
mean = (0.506, 0.425, 0.383)
std = (0.311, 0.291, 0.289)
dl = get_celeba_dataloader(root, mean=mean, std=std, batch_size=BATCH_SIZE, n_workers=N_WORKERS)

crit = nn.BCELoss()

N_EPOCHS = 30
disc_losses = list()
gen_losses = list()
for epoch in range(1, N_EPOCHS + 1):
    for batch, (real_image, _) in enumerate(dl, start=1):
        real_image = real_image.to(DEVICE)

        ### Update D
        disc_optim.zero_grad()

        real_pred = disc(real_image) # $D(x)$
        real_label = torch.ones_like(real_pred, device=DEVICE)
        real_label = real_label.detach()
        real_disc_loss = crit(real_pred, real_label) # $\log(D(x))$ # D 입장에서는 Loss가 낮아져야 함.

        noise = torch.randn(BATCH_SIZE, 100, device=DEVICE) # $z$
        # noise = torch.randn(BATCH_SIZE, 100, 1, 1, device=DEVICE) # $z$
        fake_image = gen(noise) # $G(z)$
        # D를 업데이트하는 동안에는 G는 업데이트하지 않으므로 `fake_image`에 대한 미분 계산을 하지 않아도 됩니다!
        fake_pred = disc(fake_image.detach()) # $D(G(z))$
        fake_label = torch.zeros_like(fake_pred, device=DEVICE)
        fake_label = fake_label.detach()
        fake_disc_loss = crit(fake_pred, fake_label) # $\log(1 - D(G(z)))$ # D 입장에서는 Loss가 낮아져야 함.

        disc_loss = real_disc_loss + fake_disc_loss
        disc_loss.backward()

        disc_optim.step()

        ### Update G
        gen_optim.zero_grad()

        fake_pred = disc(fake_image) # $D(G(z))$
        real_label = torch.ones_like(fake_pred, device=DEVICE)
        real_label = real_label.detach()
        gen_loss = crit(fake_pred, real_label) # G 입장에서는 Loss가 낮아져야 함.

        gen_loss.backward()

        gen_optim.step()

        disc_losses.append(disc_loss.item())
        gen_losses.append(gen_loss.item())

        if batch % 50 == 0:
            print(f"""[{epoch}/{str(N_EPOCHS)}][{batch}/{len(dl)}] D loss: {disc_loss.item(): .4f} | G loss: {gen_loss.item(): .4f}""")

        if batch == len(dl):
            fake_image = fake_image.detach().cpu()
            grid = batched_image_to_grid(fake_image[: 64, ...], n_cols=8, mean=mean, std=std)
            save_image(grid, path=f"""./examples/epoch_{epoch}_batch_{batch}.jpg""")

            save_parameters(
                model=gen,
                save_path=f"""{Path(__file__).resolve().parent}/parameters/epoch_{epoch}_batch_{batch}.pth"""
            )
