import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from time import time
from datetime import timedelta

import config


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _to_array(img):
    img = np.array(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(img).save(str(path))


def get_image_dataset_mean_and_std(data_dir, ext="jpg"):
    data_dir = Path(data_dir)

    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for img_path in tqdm(list(data_dir.glob(f"**/*.{ext}"))):
        pil_img = Image.open(img_path)
        tensor = T.ToTensor()(pil_img)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


def batched_image_to_grid(image, n_cols, mean, std):
    b, _, h, w = image.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(2, int(max(h, w) * 0.04))
    grid = make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    grid *= std
    grid += mean
    grid *= 255
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_cols + 1):
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
    return grid


def denorm(tensor, mean, std):
    tensor *= torch.Tensor(std)[None, :, None, None]
    tensor += torch.Tensor(mean)[None, :, None, None]
    return tensor


def image_to_grid(image, n_cols, mean, std):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor, mean=mean, std=std)
    grid = make_grid(tensor, nrow=n_cols, padding=2, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


@torch.no_grad()
def generate_images(gen, noise, batch_size, n_cols=0):
    gen.eval()
    with torch.no_grad():
        fake_image = gen(noise).detach().cpu()
        n_cols = int(batch_size ** 0.5) if n_cols == 0 else n_cols
        # grid = batched_image_to_grid(fake_image, n_cols=n_cols, mean=config.MEAN, std=config.STD)
        grid = image_to_grid(fake_image, n_cols=n_cols, mean=config.MEAN, std=config.STD)
    gen.train()
    return grid


def get_noise(batch_size, latent_dim, device):
    return torch.randn(size=(batch_size, latent_dim), device=device)
