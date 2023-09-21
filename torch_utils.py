import torch
from pathlib import Path


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def save_gen(gen, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), str(save_path))
