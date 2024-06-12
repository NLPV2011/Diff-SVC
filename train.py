import argparse
import os

import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample
from model import DiffSingerLoss
from data_utils import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")
    
    print(configs)

    # Get dataset
    dataset = Dataset(
        configs["path"]["train_filelist"], configs, configs, sort=True, drop_last=True
    )
    batch_size = configs["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    if args.restore_step != 0 or args.restore_step != None:
        model, optimizer = get_model(args=args, ckpt_path=configs["train_path"]["ckpt_path"]+"/"+str(args.restore_step)+".ckpt", configs=configs, device=device, train=True)
    else:
        model, optimizer = get_model(args=args, ckpt_path=None, configs=configs, device=device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = DiffSingerLoss(configs).to(device)
    print("Number of DiffSinger Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder()

    # Init logger
    for p in configs["train_path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(configs["train_path"]["log_path"], "train")
    val_log_path = os.path.join(configs["train_path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = configs["optimizer"]["grad_acc_step"]
    grad_clip_thresh = configs["optimizer"]["grad_clip_thresh"]
    total_step = configs["step"]["total_step_{}".format("shallow")]
    log_step = configs["step"]["log_step"]
    save_step = configs["step"]["save_step"]
    synth_step = configs["step"]["synth_step"]
    val_step = configs["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                pitches = batch[8].clone()
                assert batch[8][0].shape[0] == batch[5][0].shape[0]
                # Forward
                output = model(*(batch[1:]))
                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    lr = optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses_ = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Noise Loss: {:.4f}".format(
                        *losses_
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, lr=lr)

                if step % synth_step == 0:
                    assert batch[8][0].shape[0] == batch[5][0].shape[0], (batch[8][0].shape,batch[5][0].shape[0])

                    figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        pitches,
                        output,
                        vocoder,
                        configs,
                        configs,
                        model.module.diffusion,
                    )
                    log(
                        train_logger,
                        step,
                        figs=figs,
                        tag="Training",
                    )
                    sampling_rate = configs["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/reconstructed",
                        step=step
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/synthesized",
                        step=step
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder, losses)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    savepath = os.path.join(configs["train_path"]["ckpt_path"], "{}.ckpt".format(step), )
                    rmpath = os.path.join(configs["train_path"]["ckpt_path"], "{}.ckpt".format(step-3*save_step), )
                    os.system(f"rm {rmpath}")
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        savepath,
                    )

                if step >= total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    configs = get_configs_of(args.dataset)
    train_tag = "shallow"

    if configs["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        configs["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(" ---> Type of Modeling:", "shallow")
    print(" ---> Total Batch Size:", int(configs["optimizer"]["batch_size"]))
    print(" ---> Use Pitch Embed:", configs["variance_embedding"]["use_pitch_embed"])
    print(" ---> Use Energy Embed:", configs["variance_embedding"]["use_energy_embed"])
    print(" ---> Path of ckpt:", configs["train_path"]["ckpt_path"])
    print(" ---> Path of log:", configs["train_path"]["log_path"])
    print(" ---> Path of result:", configs["train_path"]["result_path"])
    print("================================================================================================")

    main(args, configs)
