import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.pitch_tools
from .modules import FastspeechEncoder, FastspeechDecoder
from .diffusion import GaussianDiffusionShallow
from utils.tools import get_mask_from_lengths


class DiffSinger(nn.Module):
    """ DiffSinger """

    def __init__(self, configs):
        super(DiffSinger, self).__init__()
        self.model_config = configs

        self.text_encoder = FastspeechEncoder(configs)
        self.decoder = FastspeechDecoder(configs)
        self.mel_linear = nn.Linear(
            configs["transformer"]["decoder_hidden"],
            configs["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.diffusion = GaussianDiffusionShallow(configs)

        self.speaker_emb = None
        if configs["multi_speaker"]:
            n_speakers = configs["n_speakers"]
            self.speaker_emb = nn.Embedding(
                n_speakers,
                configs["transformer"]["encoder_hidden"],
            )
        self.pitch_emb = nn.Embedding(
                256,
                configs["transformer"]["encoder_hidden"],
            )
    def forward(
        self,
        speakers,
        contents,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        pitches=None
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        spk_emb =  self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        output = self.text_encoder(contents, src_masks,spk_emb)

        output += self.pitch_emb(utils.pitch_tools.f0_to_coarse(pitches))

        epsilon_predictions = noise_loss = diffusion_step = None
        cond = output.clone()
        output = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        self.diffusion.aux_mel = output.clone()
        (   output,
            epsilon_predictions,
            noise_loss,
            diffusion_step,
        ) = self.diffusion(
            mels,
            cond,
            mel_masks,
        )
        # else:
        #     raise NotImplementedError

        return (
            output,
            epsilon_predictions,
            noise_loss,
            diffusion_step,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )