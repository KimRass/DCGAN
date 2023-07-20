# Source: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

IMG_SIZE = 64


def get_celeba_dataloader(root, mean, std, batch_size, n_workers):
    transformer = T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        # "No pre-processing was applied to training images besides scaling to the range of the
        # tanh activation function $[-1, 1]$."
        T.Normalize(mean, std),
    ])
    ds = ImageFolder(root, transform=transformer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)
    return dl
