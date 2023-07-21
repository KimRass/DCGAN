import torch
from pathlib import Path


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def save_parameters(model, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
