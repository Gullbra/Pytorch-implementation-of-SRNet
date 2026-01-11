"""This module provides utility function for training."""
from argparse import Namespace
import os
import re
from typing import Any, Dict
import torch
from torch import nn
from pathlib import Path

# from ..opts import arguments

# opt = arguments()


def saver(state: Dict[str, float], save_dir: str, epoch: int) -> None:
    save_path = Path(save_dir) / f"net_{epoch}.pt"
    torch.save(state, save_path)


def latest_checkpoint(opt: Namespace) -> int:
    """Returns latest checkpoint."""

    print(f"Searching for checkpoint(s) in {opt.checkpoints_dir}")
    if os.path.exists(opt.checkpoints_dir):
        all_chkpts = "".join(os.listdir(opt.checkpoints_dir))
        if len(all_chkpts) > 0:
            latest = max(map(int, re.findall(r"\d+", all_chkpts)))
        else:
            latest = None
    else:
        latest = None
    return latest


def adjust_learning_rate(optimizer: Any, epoch: int, opt: Namespace) -> None:
    """Sets the learning rate to the initial learning_rate and decays by 10
    every 30 epochs."""
    learning_rate = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


# Weight initialization for conv layers and fc layers
def weights_init(param: Any) -> None:
    """Initializes weights of Conv and fully connected."""

    if isinstance(param, nn.Conv2d):
        torch.nn.init.xavier_uniform_(param.weight.data)
        if param.bias is not None:
            torch.nn.init.constant_(param.bias.data, 0.2)
    elif isinstance(param, nn.Linear):
        torch.nn.init.normal_(param.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(param.bias.data, 0.0)
