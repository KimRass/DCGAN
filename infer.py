# References:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
import torch.nn as nn

from model import Generator
from torch_utils import get_device

DEVICE = get_device()

gen = Generator().to(DEVICE)

state_dict = torch.load("/Users/jongbeomkim/Downloads/test.pth", map_location=DEVICE)
gen.load_state_dict(state_dict, strict=True)
