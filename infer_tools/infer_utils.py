import os
import json

import torch
import numpy as np

import hifigan
from model import DiffSinger

def get_model(args, configs, device):
    model = DiffSinger(args, configs).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            configs["train_path"]["ckpt_path"],
            "{}.ckpt".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    model.requires_grad_ = False
    return model