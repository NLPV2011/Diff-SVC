import parselmouth
import librosa
import torch
import numpy as np
import utils.tools
from utils.model import get_vocoder
from hubert import hubert_model

from model import DiffSinger

sample_rate = 44100
hop_len = 512

def get_f0(wave, sr, p_len=None, f0_up_key=0):
    x = wave
    assert sr == sample_rate
    if p_len is None:
        p_len = x.shape[0] // hop_len
    else:
        assert abs(p_len - x.shape[0] // hop_len) < 3, "pad length error"
    time_step = hop_len / sample_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size = (p_len - len(f0) + 1) // 2
    if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int16)
    return f0_coarse, f0

class Svc:
    def __init__(self, configs, model_path):
        self.configs = configs
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path=self.model_path)
        self.phoneme_encoder = utils.tools.get_hubert_model(0 if torch.cuda.is_available() else None)
        self.vocoder = get_vocoder()
        
    def get_hubert_model(rank=None):
        hubert_soft = hubert_model.hubert_soft("hubert/hubert-soft-0d54a1f4.pt")
        if rank is not None:
            hubert_soft = hubert_soft.cuda(rank)
        return hubert_soft
        
    def load_model(self, model_path):
        print(f"| loading 'model': {model_path}... |")
        model = DiffSinger(self.configs).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        model.requires_grad_ = False
        return model
    
    def get_phoneme(self, wav, sr):
        devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
        c = utils.tools.get_hubert_content(self.phoneme_encoder, wav).cpu().squeeze(0)
        c = utils.tools.repeat_expand_2d(c, int((wav.shape[1] * sample_rate / 16000) // hop_len)).numpy()
        return c
    
    def infer_segment(self, wav, sr, speaker_id, trans):
        c = self.get_phoneme(wav=wav, sr=sr).T
        c_lens = np.array([c.shape[0]])
        contents = np.array([c])
        speakers = np.array([speaker_id])
        _, f0 = get_f0(wav, sr, c.shape[0], trans)
        
        batch = [wav,
                 torch.from_numpy(speakers).long().to(self.device), 
                 torch.from_numpy(contents).float().to(self.device), 
                 torch.from_numpy(c_lens).to(self.device), 
                 max(c_lens), 
                 None, #torch.from_numpy(None).float().to(self.device), 
                 torch.from_numpy(c_lens).to(self.device), 
                 max(c_lens), 
                 torch.from_numpy(np.array([f0])).float().to(self.device)
                 ]
        
        output = self.model(*(batch[1:]))
        
        mel_len = output[7][0].item()
        pitch = batch[8][0][:mel_len]
        figs = {}
        mel_prediction = output[0][0, :mel_len].detach().transpose(0, 1)

        wav_prediction = self.vocoder.spec2wav(mel_prediction.cpu().numpy().T, f0=pitch.cpu().numpy())
        return wav_prediction, 44100 #because main sample rate is 44100