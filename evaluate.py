import argparse
import os
import json

import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tools import to_device, log, synth_one_sample
from model import DiffSingerLoss
from data_utils import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None, losses=None):
    # Get dataset
    dataset = Dataset(
        configs["path"]["val_filelist"], configs, configs, sort=False, drop_last=False
    )
    batch_size = configs["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = DiffSingerLoss(configs).to(device)

    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            pitches = batch[8].clone()
            with torch.no_grad():
                # Forward
                output = model(*(batch[1:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Noise Loss: {:.4f}, ".format(
        *([step] + [l for l in loss_means_])
    )

    if logger is not None:
        figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            pitches,
            output,
            vocoder,
            configs,
            configs,
            model.module.diffusion,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = configs["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/{}_reconstructed".format(tag),
            step=step
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/{}_synthesized".format(tag),
            step=step
        )

    return message
