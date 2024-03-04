import os
import datetime
import torch
import random
import numpy as np


def set_torch_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_output_dir(
        folder='results',
        time_stamp=True,
        time_format="%Y%m%d-%H%M%S",
        suffix: str = "") -> str:
    """Prepare a directory for outputting training results.
    Returns:
        Path of the output directory created by this function (str).
    """
    if time_stamp:
        suffix = str(datetime.datetime.now().strftime(time_format)) + "_" + suffix

    save_dir = os.path.join(folder or ".", suffix)

    os.makedirs(save_dir, exist_ok=True)

    return save_dir
